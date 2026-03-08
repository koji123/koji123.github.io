import { useState } from "react";

// ===== Qwen3.5-397B-A17B Architecture Constants =====
// 60 layers = 15 groups × (3 GDN-MoE + 1 GatedAttn-MoE)
// All layers share same MoE config
const D=4096;        // hidden_size
const VOCAB=248320;
const NLAYERS=60;
const N_GDN=45;      // 15×3 GDN layers
const N_ATT=15;      // 15×1 Full Attention layers

// GDN (Gated DeltaNet / Linear Attention) - 45 layers
const GDN_QK_H=16;    // QK heads
const GDN_V_H=64;     // Value heads
const GDN_HD=128;     // head_dim for GDN

// Gated Attention (Full Attention) - 15 layers
const ATT_Q_H=32;     // Q heads
const ATT_KV_H=2;     // KV heads (GQA-2)
const ATT_HD=256;     // head_dim
const ATT_ROPE=64;    // RoPE dim

// MoE (same for all 60 layers)
const NE=512;          // num_experts
const KE=10;           // num_experts_per_tok (routed)
const MI=1024;         // moe_intermediate_size (per expert)
const N_SHARED=1;      // shared experts
// shared_expert_intermediate_size: likely ~4096 or similar to intermediate_size
// From tiny-random config reference and architecture: assume shared_expert uses intermediate_size
const SI=4096;         // shared_expert_intermediate_size (estimated, typical for D=4096)

const mm=(M,K,N)=>2*M*K*N;
function fmt(n){if(n>=1e15)return(n/1e15).toFixed(2)+"P";if(n>=1e12)return(n/1e12).toFixed(2)+"T";if(n>=1e9)return(n/1e9).toFixed(2)+"G";if(n>=1e6)return(n/1e6).toFixed(2)+"M";if(n>=1e3)return(n/1e3).toFixed(2)+"K";return n.toFixed(0);}
function fB(b){if(b>=1e9)return(b/1e9).toFixed(2)+" GB";if(b>=1e6)return(b/1e6).toFixed(2)+" MB";if(b>=1e3)return(b/1e3).toFixed(2)+" KB";if(b>0)return b.toFixed(0)+" B";return "0";}
function pct(v,t){return t===0?"—":(v/t*100).toFixed(1)+"%";}

function buildGDN(S, Sq, TP, EP, isDec, bpe) {
  // GDN: Linear attention with 16 QK heads, 64 V heads, head_dim=128
  // GDN uses a recurrent state instead of KV cache
  // Projections: Q,K (16 heads×128), V (64 heads×128), output gate, output proj
  const qk_ht=GDN_QK_H/TP, v_ht=GDN_V_H/TP;
  const o=[];

  o.push({s:"GDN RMSNorm",t:"E",
    op:`正規化: (${Sq}×${D})`,
    fg:5*Sq*D, fa:5*Sq*D*TP, par:"replicated",comm:"",cs:0});

  // Q,K projection: 4096 → 16 heads × 128 = 2048
  o.push({s:"GDN QK射影",t:"M",
    op:`Q,K: (${Sq}×${D})×(${D}×${qk_ht*GDN_HD}) ×2`,
    fg:2*mm(Sq,D,qk_ht*GDN_HD), fa:2*mm(Sq,D,GDN_QK_H*GDN_HD),
    par:`head÷${TP}`,comm:"",cs:0});

  // V projection: 4096 → 64 heads × 128 = 8192
  o.push({s:"GDN V射影",t:"M",
    op:`V: (${Sq}×${D})×(${D}×${v_ht*GDN_HD})`,
    fg:mm(Sq,D,v_ht*GDN_HD), fa:mm(Sq,D,GDN_V_H*GDN_HD),
    par:`head÷${TP}`,comm:"",cs:0});

  // Gate projection for output gating: 4096 → V_heads × head_dim = 8192
  o.push({s:"GDN Gate",t:"M",
    op:`出力Gate: (${Sq}×${D})×(${D}×${v_ht*GDN_HD})`,
    fg:mm(Sq,D,v_ht*GDN_HD), fa:mm(Sq,D,GDN_V_H*GDN_HD),
    par:`head÷${TP}`,comm:"",cs:0});

  // DeltaNet recurrence / linear attention computation
  // For each V head: state update + output computation
  // Simplified: O(S × head_dim²) per head (state is head_dim × head_dim)
  // QK heads are grouped: each QK head serves (V_heads/QK_heads) V heads
  if(!isDec) {
    // Prefill: chunk-wise parallel, roughly O(S × QK_H × HD² + S × V_H × HD)
    o.push({s:"GDN DeltaNet",t:"M",
      op:`線形Attn: ${qk_ht}qk_h × (${Sq}×${GDN_HD})×(${GDN_HD}×${GDN_HD}) [state]`,
      fg:qk_ht*Sq*2*GDN_HD*GDN_HD, fa:GDN_QK_H*Sq*2*GDN_HD*GDN_HD,
      par:`head÷${TP}`,comm:"",cs:0});
  } else {
    // Decode: single step state update O(QK_H × HD²)
    o.push({s:"GDN DeltaNet",t:"M",
      op:`状態更新: ${qk_ht}qk_h × (${GDN_HD}×${GDN_HD}) state update`,
      fg:qk_ht*2*GDN_HD*GDN_HD, fa:GDN_QK_H*2*GDN_HD*GDN_HD,
      par:`head÷${TP}`,comm:"",cs:0});
  }

  // Sigmoid gate + element-wise multiply
  o.push({s:"GDN Gate",t:"E",
    op:`sigmoid(gate)⊙out: (${Sq}×${v_ht*GDN_HD})`,
    fg:3*Sq*v_ht*GDN_HD, fa:3*Sq*GDN_V_H*GDN_HD,
    par:`÷${TP}`,comm:"",cs:0});

  // Output projection + AllReduce
  const oRows=v_ht*GDN_HD;
  const arD=Sq*D*bpe;
  const arS=arD*(TP-1)/TP;
  o.push({s:"GDN 出力",t:"M",
    op:`W_o: (${Sq}×${oRows})×(${oRows}×${D}) → AR`,
    fg:mm(Sq,oRows,D), fa:mm(Sq,GDN_V_H*GDN_HD,D),
    par:`行分割÷${TP}`,comm:`★AR(TP=${TP})`,cs:arS});

  return o;
}

function buildAttn(S, Sq, Skv, TP, EP, isDec, bpe) {
  // Gated Attention: 32 Q heads, 2 KV heads (GQA-2), head_dim=256, RoPE=64
  const q_ht=ATT_Q_H/TP, kv_ht=ATT_KV_H; // KV heads replicated if TP>KV_H
  const kv_per_tp = Math.max(1, ATT_KV_H/TP); // min 1
  const o=[];

  o.push({s:"Attn RMSNorm",t:"E",
    op:`正規化: (${Sq}×${D})`,
    fg:5*Sq*D, fa:5*Sq*D*TP, par:"replicated",comm:"",cs:0});

  // Q projection: 4096 → 32 heads × 256 = 8192
  o.push({s:"Attn Q射影",t:"M",
    op:`Q: (${Sq}×${D})×(${D}×${q_ht*ATT_HD})`,
    fg:mm(Sq,D,q_ht*ATT_HD), fa:mm(Sq,D,ATT_Q_H*ATT_HD),
    par:`head÷${TP}`,comm:"",cs:0});

  // KV projection: 4096 → 2 KV heads × 256 × 2(K+V) = 1024
  // GQA-2: very few KV heads, typically replicated across TP
  const kvDim = kv_per_tp*ATT_HD;
  if(!isDec){
    o.push({s:"Attn KV射影",t:"M",
      op:`K,V: (${S}×${D})×(${D}×${kvDim}) ×2 [GQA-2, ${kv_per_tp}kv_h/TP]`,
      fg:2*mm(S,D,kvDim), fa:2*mm(S,D,ATT_KV_H*ATT_HD),
      par:ATT_KV_H>=TP?`kv÷${TP}`:"replicated (KV_H<TP)",comm:"",cs:0});
  } else {
    o.push({s:"Attn KV射影",t:"M",
      op:`K,V新tok: (1×${D})×(${D}×${kvDim}) ×2 → cache追記`,
      fg:2*mm(1,D,kvDim), fa:2*mm(1,D,ATT_KV_H*ATT_HD),
      par:ATT_KV_H>=TP?`kv÷${TP}`:"replicated",comm:"",cs:0});
  }

  // RoPE on Q,K (partial_rotary_factor=0.25, rope_dim=64)
  o.push({s:"Attn RoPE",t:"E",
    op:`RoPE: (${Sq}×${q_ht}h×${ATT_ROPE/2}pairs)`,
    fg:6*Sq*q_ht*(ATT_ROPE/2), fa:6*Sq*ATT_Q_H*(ATT_ROPE/2),
    par:`÷${TP}`,comm:"",cs:0});

  // Gate projection for gated attention
  o.push({s:"Attn Gate",t:"M",
    op:`出力Gate: (${Sq}×${D})×(${D}×${q_ht*ATT_HD})`,
    fg:mm(Sq,D,q_ht*ATT_HD), fa:mm(Sq,D,ATT_Q_H*ATT_HD),
    par:`head÷${TP}`,comm:"",cs:0});

  // Attention computation
  const Skv_ = isDec ? Skv : S;
  // GQA: each Q head group shares KV heads. 32Q / 2KV = 16 Q per KV
  if(!isDec){
    o.push({s:"Attention計算",t:"M",
      op:`Q×K^T: ${q_ht}h × (${S}×${ATT_HD})×(${ATT_HD}×${S})`,
      fg:q_ht*mm(S,ATT_HD,S), fa:ATT_Q_H*mm(S,ATT_HD,S),
      par:`head÷${TP}`,comm:"",cs:0});
    o.push({s:"Attention計算",t:"E",
      op:`Scale+Mask+Softmax: (${q_ht}h×${S}×${S})`,
      fg:7*q_ht*S*S, fa:7*ATT_Q_H*S*S,
      par:`÷${TP}`,comm:"",cs:0});
    o.push({s:"Attention計算",t:"M",
      op:`Weights×V: ${q_ht}h × (${S}×${S})×(${S}×${ATT_HD})`,
      fg:q_ht*mm(S,S,ATT_HD), fa:ATT_Q_H*mm(S,S,ATT_HD),
      par:`head÷${TP}`,comm:"",cs:0});
  } else {
    o.push({s:"Attention計算",t:"M",
      op:`Q×K^T: ${q_ht}h × (1×${ATT_HD})×(${ATT_HD}×${Skv})`,
      fg:q_ht*mm(1,ATT_HD,Skv), fa:ATT_Q_H*mm(1,ATT_HD,Skv),
      par:`head÷${TP}`,comm:"",cs:0});
    o.push({s:"Attention計算",t:"E",
      op:`Softmax: (${q_ht}h×1×${Skv})`,
      fg:5*q_ht*Skv, fa:5*ATT_Q_H*Skv,
      par:`÷${TP}`,comm:"",cs:0});
    o.push({s:"Attention計算",t:"M",
      op:`Weights×V: ${q_ht}h × (1×${Skv})×(${Skv}×${ATT_HD})`,
      fg:q_ht*mm(1,Skv,ATT_HD), fa:ATT_Q_H*mm(1,Skv,ATT_HD),
      par:`head÷${TP}`,comm:"",cs:0});
  }

  // Gate apply
  o.push({s:"Attn Gate",t:"E",
    op:`sigmoid(gate)⊙out: (${Sq}×${q_ht*ATT_HD})`,
    fg:3*Sq*q_ht*ATT_HD, fa:3*Sq*ATT_Q_H*ATT_HD,
    par:`÷${TP}`,comm:"",cs:0});

  // Output projection + AllReduce
  const oR=q_ht*ATT_HD;
  const arD=Sq*D*bpe;
  const arS=arD*(TP-1)/TP;
  o.push({s:"Attn 出力",t:"M",
    op:`W_o: (${Sq}×${oR})×(${oR}×${D}) → AR`,
    fg:mm(Sq,oR,D), fa:mm(Sq,ATT_Q_H*ATT_HD,D),
    par:`行分割÷${TP}`,comm:`★AR(TP=${TP})`,cs:arS});

  return o;
}

function buildMoE(Sq, TP, EP, bpe) {
  const et=NE/EP;
  const tpr=Sq*KE/EP;
  const o=[];

  o.push({s:"残差+RMSNorm",t:"E",
    op:`残差加算 + 正規化: (${Sq}×${D})`,
    fg:6*Sq*D, fa:6*Sq*D*TP, par:"replicated",comm:"",cs:0});

  o.push({s:"Router",t:"M",
    op:`h×W_gate: (${Sq}×${D})×(${D}×${NE}) → scores(${Sq}×${NE})`,
    fg:mm(Sq,D,NE), fa:mm(Sq,D,NE)*TP,
    par:"★replicated",comm:"",cs:0});
  o.push({s:"Router",t:"E",
    op:`sigmoid + topk${KE} + norm: (${Sq}×${NE})`,
    fg:10*Sq*NE, fa:10*Sq*NE*TP, par:"replicated",comm:"",cs:0});

  const a2a=Sq*KE*((EP-1)/EP)*D*bpe;
  o.push({s:"EP Dispatch",t:"E",
    op:`All-to-All: ${Sq}tok×${KE}exp×${((EP-1)/EP).toFixed(2)}×${D}d×${bpe}B`,
    fg:0,fa:0,par:`EP=${EP}(${et}exp/GPU)`,comm:`★A2A#1(EP=${EP})`,cs:a2a});

  o.push({s:"Expert×${KE}",t:"M",
    op:`Gate+Up: ${tpr.toFixed(0)}tok × (1×${D})×(${D}×${2*MI})`,
    fg:tpr*mm(1,D,2*MI), fa:KE*mm(Sq,D,2*MI),
    par:`EP÷${EP}: ${et}exp/GPU`,comm:"",cs:0});
  o.push({s:"Expert×${KE}",t:"E",
    op:`SiLU⊙up: ${tpr.toFixed(0)}tok × (1×${MI})`,
    fg:tpr*5*MI, fa:KE*5*Sq*MI, par:"",comm:"",cs:0});
  o.push({s:"Expert×${KE}",t:"M",
    op:`Down: ${tpr.toFixed(0)}tok × (1×${MI})×(${MI}×${D})`,
    fg:tpr*mm(1,MI,D), fa:KE*mm(Sq,MI,D),
    par:`EP÷${EP}`,comm:"",cs:0});

  o.push({s:"EP Combine",t:"E",
    op:`All-to-All: 対称`,
    fg:0,fa:0,par:"対称",comm:`★A2A#2(EP=${EP})`,cs:a2a});

  // Shared Expert (TP split)
  const si=2*SI/TP;
  o.push({s:"共有Expert",t:"M",
    op:`Gate+Up: (${Sq}×${D})×(${D}×${si})`,
    fg:mm(Sq,D,si), fa:mm(Sq,D,2*SI),
    par:`列分割÷${TP}`,comm:"",cs:0});
  o.push({s:"共有Expert",t:"E",
    op:`SiLU⊙up: (${Sq}×${SI/TP})`,
    fg:5*Sq*SI/TP, fa:5*Sq*SI, par:`÷${TP}`,comm:"",cs:0});
  const arD2=Sq*D*bpe;
  o.push({s:"共有Expert",t:"M",
    op:`Down: (${Sq}×${SI/TP})×(${SI/TP}×${D}) → AR`,
    fg:mm(Sq,SI/TP,D), fa:mm(Sq,SI,D),
    par:`行分割÷${TP}`,comm:`★AR(TP=${TP})`,cs:arD2*(TP-1)/TP});

  o.push({s:"残差②",t:"E",
    op:`加算: routed+shared+h (${Sq}×${D})`,
    fg:2*Sq*D, fa:2*Sq*D*TP, par:"replicated",comm:"",cs:0});

  return o;
}

function grp(ops){const g=[];let c=null;for(const o of ops){if(!c||c.s!==o.s){c={s:o.s,ops:[]};g.push(c);}c.ops.push(o);}return g;}

export default function App(){
  const [mode,setMode]=useState("prefill");
  const [layerType,setLayerType]=useState("gdn");
  const [S,setS]=useState(1000);
  const [Skv,setSkv]=useState(4096);
  const [TP,setTP]=useState(8);
  const [EP,setEP]=useState(8);
  const isDec=mode==="decode";
  const Sq=isDec?1:S;
  const bpe=2;

  // Build ops for selected layer type
  let attnOps;
  if(layerType==="gdn"){
    attnOps=buildGDN(S,Sq,TP,EP,isDec,bpe);
  } else {
    attnOps=buildAttn(S,Sq,Skv,TP,EP,isDec,bpe);
  }
  const moeOps=buildMoE(Sq,TP,EP,bpe);
  const ops=[...attnOps,...moeOps];

  const groups=grp(ops);
  const gpuF=ops.reduce((a,o)=>a+o.fg,0);
  const allF=ops.reduce((a,o)=>a+o.fa,0);
  const gpuSend=ops.reduce((a,o)=>a+o.cs,0);
  const arS_=ops.filter(o=>o.comm.includes("AR")).reduce((a,o)=>a+o.cs,0);
  const a2aS_=ops.filter(o=>o.comm.includes("A2A")).reduce((a,o)=>a+o.cs,0);
  const replF=ops.filter(o=>o.par.includes("replicated")||o.par.includes("★replicated")).reduce((a,o)=>a+o.fg,0);

  const nLayers = layerType==="gdn"?N_GDN:N_ATT;
  const secS=groups.map(g=>({n:g.s,fg:g.ops.reduce((a,o)=>a+o.fg,0),cs:g.ops.reduce((a,o)=>a+o.cs,0)}));
  const mxF=Math.max(...secS.map(x=>x.fg),1);
  const mxC=Math.max(...secS.map(x=>x.cs),1);
  const pc=layerType==="gdn"?"#22c55e":"#4A90D9";

  // Full model estimate
  const gdnLayerGpu = (() => {
    const a=buildGDN(S,Sq,TP,EP,isDec,bpe);
    const m=buildMoE(Sq,TP,EP,bpe);
    return [...a,...m].reduce((s,o)=>s+o.fg,0);
  })();
  const attLayerGpu = (() => {
    const a=buildAttn(S,Sq,Skv,TP,EP,isDec,bpe);
    const m=buildMoE(Sq,TP,EP,bpe);
    return [...a,...m].reduce((s,o)=>s+o.fg,0);
  })();
  const fullModelGpu = N_GDN*gdnLayerGpu + N_ATT*attLayerGpu;

  return(
    <div style={{minHeight:"100vh",background:"linear-gradient(160deg,#08080e,#0e0e1c,#0a0a14)",color:"#e0e0ec",fontFamily:"'IBM Plex Sans','Noto Sans JP',system-ui,sans-serif",padding:"14px 8px"}}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+JP:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
      <div style={{maxWidth:1200,margin:"0 auto"}}>
        <div style={{textAlign:"center",marginBottom:10}}>
          <h1 style={{fontSize:18,fontWeight:700,color:"#f0f0f5",margin:0}}>Qwen3.5-397B-A17B — 1層 FLOPS・通信 (1GPUあたり)</h1>
          <p style={{fontSize:10,color:"#666",margin:"3px 0 0"}}>
            Hybrid: 15×(3×GDN-MoE + 1×GatedAttn-MoE) = 45 GDN層 + 15 Attn層 | 512exp, top-10+1shared, exp_dim=1024
          </p>
        </div>

        {/* Controls */}
        <div style={{display:"flex",justifyContent:"center",alignItems:"center",gap:6,marginBottom:10,flexWrap:"wrap"}}>
          {[{k:"prefill",l:"プリフィル"},{k:"decode",l:"デコード"}].map(m=>(
            <button key={m.k} onClick={()=>setMode(m.k)} style={{padding:"5px 12px",borderRadius:6,fontSize:11,fontWeight:700,cursor:"pointer",
              border:mode===m.k?`2px solid ${m.k==="prefill"?"#4A90D9":"#E8734A"}`:"2px solid rgba(255,255,255,0.08)",
              background:mode===m.k?`${m.k==="prefill"?"#4A90D9":"#E8734A"}18`:"transparent",
              color:mode===m.k?(m.k==="prefill"?"#6BB3FF":"#FFB088"):"#666"}}>{m.l}</button>
          ))}
          <div style={{width:1,height:20,background:"rgba(255,255,255,0.1)"}}/>
          {[{k:"gdn",l:"GDN層 (×45)",c:"#22c55e"},{k:"attn",l:"GatedAttn層 (×15)",c:"#4A90D9"}].map(m=>(
            <button key={m.k} onClick={()=>setLayerType(m.k)} style={{padding:"5px 12px",borderRadius:6,fontSize:11,fontWeight:700,cursor:"pointer",
              border:layerType===m.k?`2px solid ${m.c}`:"2px solid rgba(255,255,255,0.08)",
              background:layerType===m.k?`${m.c}18`:"transparent",
              color:layerType===m.k?m.c:"#666"}}>{m.l}</button>
          ))}
          <div style={{width:1,height:20,background:"rgba(255,255,255,0.1)"}}/>
          {[{l:isDec?"S_kv":"S",v:isDec?Skv:S,fn:isDec?setSkv:setS},{l:"TP",v:TP,fn:setTP},{l:"EP",v:EP,fn:setEP}].map((c,i)=>(
            <div key={i} style={{display:"flex",alignItems:"center",gap:3}}>
              <label style={{fontSize:9,color:"#888"}}>{c.l}=</label>
              <input type="number" value={c.v} onChange={e=>{c.fn(Math.max(1,parseInt(e.target.value)||1));}}
                style={{width:50,padding:"3px 5px",borderRadius:4,border:"1px solid rgba(255,255,255,0.12)",background:"rgba(255,255,255,0.05)",color:"#ddd",fontSize:11,fontFamily:"'IBM Plex Mono',monospace"}}/>
            </div>
          ))}
        </div>

        {/* Summary */}
        <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:5,marginBottom:9}}>
          {[
            {l:`FLOPS/GPU (${layerType==="gdn"?"GDN":"Attn"}1層)`,v:fmt(gpuF),u:`×${nLayers}=${fmt(gpuF*nLayers)}`,c:"#e0e0f0"},
            {l:"全60層/GPU合計",v:fmt(fullModelGpu),u:`45×GDN + 15×Attn`,c:"#ffcc44"},
            {l:"冗長(replicated)",v:fmt(replF),u:pct(replF,gpuF),c:"#ffcc44"},
            {l:"送信/GPU/層",v:fB(gpuSend),u:`×${nLayers}=${fB(gpuSend*nLayers)}`,c:"#ffaa66"},
            {l:"AR送信",v:fB(arS_),u:`Attn出力+共有Exp`,c:"#66bbff"},
            {l:"A2A送信",v:fB(a2aS_),u:`Dispatch+Combine`,c:"#cc66ff"},
          ].map((c,i)=>(
            <div key={i} style={{background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.06)",borderRadius:6,padding:"6px 7px",textAlign:"center"}}>
              <div style={{fontSize:7.5,color:"#666",textTransform:"uppercase",letterSpacing:0.7}}>{c.l}</div>
              <div style={{fontSize:12,fontWeight:700,color:c.c,fontFamily:"'IBM Plex Mono',monospace",margin:"1px 0"}}>{c.v}</div>
              <div style={{fontSize:8.5,color:"#555"}}>{c.u}</div>
            </div>
          ))}
        </div>

        {/* Bars */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:7,marginBottom:9}}>
          {[{ti:"FLOPS/GPU",k:"fg",mx:mxF,u:fmt,co:pc},{ti:"送信/GPU",k:"cs",mx:mxC,u:fB,co:"#cc66ff"}].map((ch,ci)=>(
            <div key={ci} style={{padding:"7px 9px",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.05)",borderRadius:7}}>
              <div style={{fontSize:8.5,fontWeight:600,color:"#777",marginBottom:4,textTransform:"uppercase",letterSpacing:0.8}}>{ch.ti}</div>
              {secS.map((x,i)=>{const v=x[ch.k];return(
                <div key={i} style={{display:"flex",alignItems:"center",marginBottom:2}}>
                  <div style={{width:100,fontSize:9,color:"#999",flexShrink:0,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{x.n}</div>
                  <div style={{flex:1,height:10,background:"rgba(255,255,255,0.03)",borderRadius:2,overflow:"hidden",marginRight:3}}>
                    <div style={{height:"100%",width:`${ch.mx>0?Math.max(v>0?0.4:0,v/ch.mx*100):0}%`,background:`linear-gradient(90deg,${ch.co},${ch.co}88)`,borderRadius:2}}/>
                  </div>
                  <div style={{width:55,fontSize:9,color:"#aaa",fontFamily:"'IBM Plex Mono',monospace",textAlign:"right"}}>{v>0?ch.u(v):"—"}</div>
                </div>
              );})}
            </div>
          ))}
        </div>

        {/* Table */}
        <div style={{background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.05)",borderRadius:7,overflow:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:10,minWidth:950}}>
            <thead>
              <tr style={{background:"rgba(255,255,255,0.04)"}}>
                {["セクション","演算 (名前 + 行列サイズ)","型","FLOPS/GPU","FLOPS/全GPU","分割","通信","送信/GPU"].map((h,i)=>(
                  <th key={i} style={{padding:"5px 3px",textAlign:[3,4,7].includes(i)?"right":"left",color:"#777",fontWeight:600,fontSize:8,textTransform:"uppercase",letterSpacing:0.6,borderBottom:"1px solid rgba(255,255,255,0.06)",whiteSpace:"nowrap"}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.map((g,gi)=>{
                const sgf=g.ops.reduce((a,x)=>a+x.fg,0);
                const sc=g.ops.reduce((a,x)=>a+x.cs,0);
                return g.ops.map((op,oi)=>(
                  <tr key={`${gi}-${oi}`} style={{borderBottom:"1px solid rgba(255,255,255,0.02)",background:oi===0?"rgba(255,255,255,0.008)":"transparent"}}>
                    {oi===0&&<td rowSpan={g.ops.length} style={{padding:"3px",verticalAlign:"top",color:"#aaa",fontWeight:600,fontSize:9,borderRight:"1px solid rgba(255,255,255,0.03)",background:"rgba(255,255,255,0.005)",width:90}}>
                      {g.s}
                      <div style={{fontSize:8,color:"#555",marginTop:1,fontFamily:"'IBM Plex Mono',monospace"}}>
                        {sgf>0&&<div>{fmt(sgf)}</div>}
                        {sc>0&&<div style={{color:"#cc66ff"}}>{fB(sc)}</div>}
                      </div>
                    </td>}
                    <td style={{padding:"3px 4px",color:"#d0d0e0",fontSize:10,fontFamily:"'IBM Plex Mono','Noto Sans JP',monospace",whiteSpace:"pre-wrap",lineHeight:1.35,maxWidth:350}}>{op.op}</td>
                    <td style={{padding:"3px 2px"}}>
                      <span style={{display:"inline-block",padding:"0px 3px",borderRadius:2,fontSize:8.5,fontWeight:700,
                        background:op.t==="M"?"rgba(74,144,217,0.15)":"rgba(255,180,80,0.1)",
                        color:op.t==="M"?"#6BB3FF":"#FFBB66"}}>{op.t==="M"?"MM":"El"}</span>
                    </td>
                    <td style={{padding:"3px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",color:"#e0e0f0",fontWeight:600,fontSize:10,whiteSpace:"nowrap"}}>{op.fg>0?fmt(op.fg):"—"}</td>
                    <td style={{padding:"3px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",color:"#777",fontSize:9,whiteSpace:"nowrap"}}>{op.fa>0?fmt(op.fa):"—"}</td>
                    <td style={{padding:"3px 4px",fontSize:9,color:op.par.includes("★")?"#ffcc44":"#999",fontWeight:op.par.includes("★")?600:400,whiteSpace:"nowrap"}}>{op.par||"—"}</td>
                    <td style={{padding:"3px 4px",fontSize:9,whiteSpace:"nowrap",
                      color:op.comm.includes("AR")?"#66bbff":op.comm.includes("A2A")?"#cc66ff":"#555",
                      fontWeight:op.cs>0?600:400}}>{op.comm||"—"}</td>
                    <td style={{padding:"3px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",fontSize:9,
                      color:op.cs>0?"#ffaa66":"#444",fontWeight:op.cs>0?600:400,whiteSpace:"nowrap"}}>{op.cs>0?fB(op.cs):"—"}</td>
                  </tr>
                ));
              })}
              <tr style={{background:"rgba(255,255,255,0.05)",borderTop:"2px solid rgba(255,255,255,0.1)"}}>
                <td colSpan={3} style={{padding:"5px 3px",fontWeight:700,color:"#f0f0f5",fontSize:10.5}}>
                  合計 ({layerType==="gdn"?"GDN":"Attn"}1層)
                </td>
                <td style={{padding:"5px 3px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",fontWeight:700,color:"#f0f0f5",fontSize:10.5}}>{fmt(gpuF)}</td>
                <td style={{padding:"5px 3px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",color:"#777",fontSize:9.5}}>{fmt(allF)}</td>
                <td/><td style={{fontSize:9,color:"#ddd"}}>送信計</td>
                <td style={{padding:"5px 3px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",fontWeight:700,color:"#ffaa66",fontSize:10.5}}>{fB(gpuSend)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Architecture comparison note */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginTop:10}}>
          <div style={{padding:"9px 11px",background:"rgba(34,197,85,0.06)",border:"1px solid rgba(34,197,85,0.15)",borderRadius:7,fontSize:10,color:"#c8c8d8",lineHeight:1.6}}>
            <strong style={{color:"#22c55e"}}>GDN層 (×45) — Gated DeltaNet</strong><br/>
            線形Attention: KVキャッシュ不要、代わりに(head_dim×head_dim)の<br/>
            再帰状態を保持。デコード時O(head_dim²)で一定。<br/>
            16 QK heads × 128dim, 64 V heads × 128dim<br/>
            出力にsigmoidゲートを適用。<br/>
            長文推論のコスト削減の鍵。
          </div>
          <div style={{padding:"9px 11px",background:"rgba(74,144,217,0.06)",border:"1px solid rgba(74,144,217,0.15)",borderRadius:7,fontSize:10,color:"#c8c8d8",lineHeight:1.6}}>
            <strong style={{color:"#6BB3FF"}}>GatedAttn層 (×15) — Full Attention</strong><br/>
            通常のMulti-Head Attention + GQA-2 + 出力ゲート。<br/>
            32 Q heads, 2 KV heads (16:1 GQA ratio), head_dim=256<br/>
            KVキャッシュ: 2heads×256dim×2(K+V) = 1024要素/tok/層<br/>
            15層のみ → KVキャッシュ量がGLM-5の78層MLAより少ない。<br/>
            RoPEはpartial (64/256 = 25%)。
          </div>
        </div>

        <div style={{marginTop:7,padding:"6px 9px",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)",borderRadius:5,fontSize:9,color:"#666",lineHeight:1.5}}>
          <span style={{color:"#6BB3FF"}}>MM</span>=行列積 <span style={{color:"#FFBB66"}}>El</span>=要素演算 <span style={{color:"#ffcc44"}}>★replicated</span>=冗長計算 <span style={{color:"#66bbff"}}>AR</span>=AllReduce <span style={{color:"#cc66ff"}}>A2A</span>=All-to-All |
          MoE: 512exp, top-10, expert_dim=1024。shared_expert_intermediate_size=4096(推定)。
          GDN DeltaNet FLOPSは簡略化(chunk-wise並列の詳細は省略)。BF16前提。
        </div>
      </div>
    </div>
  );
}