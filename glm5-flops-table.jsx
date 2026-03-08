import { useState } from "react";

const D=6144,H=64,Q_R=2048,KV_R=512;
const D_nope=192,D_rope=64,D_qk=256,D_v=256;
const N_exp=256,K_exp=8,MoE_I=2048,Sh_I=12288;
const mm=(M,K,N)=>2*M*K*N;
function fmt(n){if(n>=1e15)return(n/1e15).toFixed(2)+"P";if(n>=1e12)return(n/1e12).toFixed(2)+"T";if(n>=1e9)return(n/1e9).toFixed(2)+"G";if(n>=1e6)return(n/1e6).toFixed(2)+"M";if(n>=1e3)return(n/1e3).toFixed(2)+"K";return n.toFixed(0);}
function fB(b){if(b>=1e9)return(b/1e9).toFixed(2)+" GB";if(b>=1e6)return(b/1e6).toFixed(2)+" MB";if(b>=1e3)return(b/1e3).toFixed(2)+" KB";if(b>0)return b.toFixed(0)+" B";return "0";}
function pct(v,t){return t===0?"—":(v/t*100).toFixed(1)+"%";}

// All values are PER-GPU
function build(S, Skv, isDec, TP, EP){
  const Sq=isDec?1:S, bpe=2;
  const hpt=H/TP, ept=N_exp/EP;
  const ops=[];
  // comm: per-GPU send bytes for the collective
  // AllReduce ring: each GPU sends+receives ≈ data_size * (N-1)/N
  // We report "send" amount per GPU
  const arSend=(dataBytes,N)=>dataBytes*(N-1)/N;
  // All-to-All: per GPU sends to (EP-1) other ranks

  // ============ RMSNorm 1 ============
  ops.push({sec:"RMSNorm①",op:"正規化",
    fGpu:5*Sq*D, fAll:5*Sq*D*TP, // replicated → same on every GPU, all-GPU = TP×
    par:"replicated",comm:"",cSend:0,cRecv:0,
    note:"各GPUが全6144次元を保持・同一計算"});

  // ============ Q Down (replicated) ============
  ops.push({sec:"Q射影",op:"Down: x×W_dq (→2048)",
    fGpu:mm(Sq,D,Q_R), fAll:mm(Sq,D,Q_R)*TP,
    par:"★replicated",comm:"",cSend:0,cRecv:0,
    note:"各GPUが全W_dq(6144×2048)で同一計算。冗長だが通信回避。"});

  // Q Up nope (head-split → 1/TP of heads)
  ops.push({sec:"Q射影",op:`Up nope: c_q×W_uq (→${hpt}h×192)`,
    fGpu:mm(Sq,Q_R,hpt*D_nope), fAll:mm(Sq,Q_R,H*D_nope),
    par:`head分割 (÷${TP})`,comm:"",cSend:0,cRecv:0,
    note:`各GPUが${hpt}heads分のW_uq_shard(2048×${hpt*D_nope})を保持`});

  ops.push({sec:"Q射影",op:`Up rope: c_q×W_qr (→${hpt}h×64)`,
    fGpu:mm(Sq,Q_R,hpt*D_rope), fAll:mm(Sq,Q_R,H*D_rope),
    par:`head分割 (÷${TP})`,comm:"",cSend:0,cRecv:0,note:""});

  ops.push({sec:"Q射影",op:"RoPE (Q)",
    fGpu:6*Sq*hpt*(D_rope/2), fAll:6*Sq*H*(D_rope/2),
    par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});

  // ============ KV ============
  if(!isDec){
    ops.push({sec:"KV射影",op:"Down: x×W_dkv (→512)",
      fGpu:mm(S,D,KV_R), fAll:mm(S,D,KV_R)*TP,
      par:"★replicated",comm:"",cSend:0,cRecv:0,
      note:"KV_R=512と極小→冗長計算がAllGatherより安い"});
    ops.push({sec:"KV射影",op:`Up K: c_kv×W_uk (→${hpt}h×192)`,
      fGpu:mm(S,KV_R,hpt*D_nope), fAll:mm(S,KV_R,H*D_nope),
      par:`head分割 (÷${TP})`,comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"KV射影",op:`Up V: c_kv×W_uv (→${hpt}h×256)`,
      fGpu:mm(S,KV_R,hpt*D_v), fAll:mm(S,KV_R,H*D_v),
      par:`head分割 (÷${TP})`,comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"KV射影",op:"K rope: x×W_kr (→64)",
      fGpu:mm(S,D,D_rope), fAll:mm(S,D,D_rope)*TP,
      par:"replicated",comm:"",cSend:0,cRecv:0,note:"全head共有"});
    ops.push({sec:"KV射影",op:"RoPE (K)",
      fGpu:6*S*(D_rope/2), fAll:6*S*(D_rope/2)*TP,
      par:"replicated",comm:"",cSend:0,cRecv:0,note:""});
  } else {
    ops.push({sec:"KV射影(新tok)",op:"Down: x×W_dkv",
      fGpu:mm(1,D,KV_R), fAll:mm(1,D,KV_R)*TP,
      par:"replicated",comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"KV射影(新tok)",op:"K rope",
      fGpu:mm(1,D,D_rope), fAll:mm(1,D,D_rope)*TP,
      par:"replicated",comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"KV射影(新tok)",op:"RoPE (K)",
      fGpu:6*(D_rope/2), fAll:6*(D_rope/2)*TP,
      par:"replicated",comm:"",cSend:0,cRecv:0,note:""});
  }

  // ============ Attention ============
  if(!isDec){
    ops.push({sec:"Attention",op:`Q×K^T (${hpt}h)`,
      fGpu:hpt*mm(S,D_qk,S), fAll:H*mm(S,D_qk,S),
      par:`head分割 (÷${TP})`,comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"Attention",op:"Scale+Mask+Softmax",
      fGpu:7*hpt*S*S, fAll:7*H*S*S,
      par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"Attention",op:`Weights×V (${hpt}h)`,
      fGpu:hpt*mm(S,S,D_v), fAll:H*mm(S,S,D_v),
      par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});
  } else {
    ops.push({sec:"Absorbed Attn",op:"Q'=Q_nope×W_uk^T",
      fGpu:hpt*mm(1,D_nope,KV_R), fAll:H*mm(1,D_nope,KV_R),
      par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"Absorbed Attn",op:"nope: Q'×c_kv^T",
      fGpu:hpt*mm(1,KV_R,Skv), fAll:H*mm(1,KV_R,Skv),
      par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"Absorbed Attn",op:"rope: Q_rope×K_rope^T",
      fGpu:hpt*mm(1,D_rope,Skv), fAll:H*mm(1,D_rope,Skv),
      par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"Absorbed Attn",op:"合成+Softmax",
      fGpu:7*hpt*Skv, fAll:7*H*Skv,
      par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"V集約(Abs)",op:"weights×c_kv",
      fGpu:hpt*mm(1,Skv,KV_R), fAll:H*mm(1,Skv,KV_R),
      par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});
    ops.push({sec:"V集約(Abs)",op:"×W_uv",
      fGpu:hpt*mm(1,KV_R,D_v), fAll:H*mm(1,KV_R,D_v),
      par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});
  }

  // ============ W_o + AllReduce ============
  const arData = Sq*D*bpe; // data size for AllReduce
  const arPerGpu = arSend(arData, TP);
  ops.push({sec:"Attn出力",op:"W_o → AllReduce #1",
    fGpu:mm(Sq,hpt*D_v,D), fAll:mm(Sq,H*D_v,D),
    par:`行分割(÷${TP}) → AllReduce`,
    comm:`★ AR#1 (TP=${TP})`,cSend:arPerGpu,cRecv:arPerGpu,
    note:`Ring AR: 各GPUがS×6144×2Bの(${TP}-1)/${TP}を送受信`});

  // ============ Residual + RMSNorm ============
  ops.push({sec:"残差①",op:"x + attn_out",
    fGpu:Sq*D, fAll:Sq*D*TP,
    par:"replicated (AR後)",comm:"",cSend:0,cRecv:0,note:""});
  ops.push({sec:"RMSNorm②",op:"正規化",
    fGpu:5*Sq*D, fAll:5*Sq*D*TP,
    par:"replicated",comm:"",cSend:0,cRecv:0,note:""});

  // ============ Router (replicated) ============
  ops.push({sec:"Router",op:"h×W_gate (→256)",
    fGpu:mm(Sq,D,N_exp), fAll:mm(Sq,D,N_exp)*TP,
    par:"★replicated",comm:"",cSend:0,cRecv:0,
    note:"全GPUが全256 expertスコアを計算 (ルーティング結果を一致させる)"});
  ops.push({sec:"Router",op:"sigmoid+topk+norm",
    fGpu:10*Sq*N_exp, fAll:10*Sq*N_exp*TP,
    par:"replicated",comm:"",cSend:0,cRecv:0,note:""});

  // ============ EP Dispatch ============
  // Per GPU: sends tokens destined for other EP ranks
  // Each token activates 8 experts. This GPU owns ept experts.
  // Prob of an expert being on another rank = (EP-1)/EP
  // Avg tokens sent per GPU = Sq * 8 * (EP-1)/EP  (in units of hidden vecs)
  // But each GPU only processes Sq tokens (replicated input), so it sends
  // its Sq tokens' hidden states to appropriate expert-owning ranks
  const a2aPerGpu = Sq * K_exp * ((EP-1)/EP) * D * bpe;
  ops.push({sec:"EP Dispatch",op:`All-to-All #1 (EP=${EP})`,
    fGpu:0, fAll:0,
    par:`各GPUが自分のSqトークンのh_norm(6144d)を\n選択expertの所在ランクへ送信`,
    comm:`★ A2A#1 (EP=${EP})`,cSend:a2aPerGpu,cRecv:a2aPerGpu,
    note:`各GPU送信: ${Sq}tok×8exp×${((EP-1)/EP).toFixed(2)}×6144×2B\n各GPU受信: 他ランクから届くtoken (対称なら同量)`});

  // ============ Routed Experts ============
  // Per GPU: owns ept experts, processes tokens routed to them
  // Avg tokens per expert ≈ Sq*8/256 * total_GPUs... but from EP perspective:
  // Each EP rank processes tokens for its ept experts
  // Total tokens across all ranks = Sq * 8 (each token activates 8)
  // Per EP rank ≈ Sq * 8 / EP tokens (if balanced)
  const tokPerRank = Sq * K_exp / EP; // avg tokens processed by this rank's experts
  ops.push({sec:"Expert×8",op:"Gate+Up ×8 experts",
    fGpu: tokPerRank * mm(1,D,2*MoE_I) * ept / (N_exp/EP),
    // simplify: each rank has ept experts, processes tokPerRank tokens total
    // = tokPerRank * (2*D*2*MoE_I) per token across all local experts
    // Actually: total per-GPU = (Sq*K_exp/EP) tokens, each does mm(1,D,2*MoE_I)
    fAll: K_exp*mm(Sq,D,2*MoE_I),
    par:`EP分割: 各GPU ${ept}experts保持\n平均${tokPerRank.toFixed(0)}tok受信 (均等時)`,
    comm:"",cSend:0,cRecv:0,note:""});

  // fix fGpu for experts: each token that arrives does one expert forward
  // fGpu = tokPerRank * [mm(1,D,2*MoE_I) per expert call]
  // But tokPerRank = Sq*K_exp/EP, and each such token uses 1 expert on this rank
  ops[ops.length-1].fGpu = (Sq*K_exp/EP)*mm(1,D,2*MoE_I);

  ops.push({sec:"Expert×8",op:"SiLU+gate⊙up",
    fGpu: (Sq*K_exp/EP)*5*MoE_I,
    fAll: K_exp*5*Sq*MoE_I,
    par:"",comm:"",cSend:0,cRecv:0,note:""});

  ops.push({sec:"Expert×8",op:"Down ×8",
    fGpu: (Sq*K_exp/EP)*mm(1,MoE_I,D),
    fAll: K_exp*mm(Sq,MoE_I,D),
    par:"EP分割",comm:"",cSend:0,cRecv:0,note:""});

  ops.push({sec:"Expert×8",op:"重み付き合成",
    fGpu: K_exp*2*Sq*D/EP, // only local portion
    fAll: K_exp*2*Sq*D,
    par:"",comm:"",cSend:0,cRecv:0,note:""});

  // ============ EP Combine ============
  ops.push({sec:"EP Combine",op:`All-to-All #2 (EP=${EP})`,
    fGpu:0, fAll:0,
    par:"expert出力を元のtokenオーナーへ返送",
    comm:`★ A2A#2 (EP=${EP})`,cSend:a2aPerGpu,cRecv:a2aPerGpu,
    note:"Dispatchと対称"});

  // ============ Shared Expert (TP split) ============
  ops.push({sec:"共有Expert",op:`Gate+Up (→${2*Sh_I/TP}/TP)`,
    fGpu:mm(Sq,D,2*Sh_I/TP), fAll:mm(Sq,D,2*Sh_I),
    par:`列分割(÷${TP})`,comm:"",cSend:0,cRecv:0,note:""});
  ops.push({sec:"共有Expert",op:"SiLU+gate⊙up",
    fGpu:5*Sq*Sh_I/TP, fAll:5*Sq*Sh_I,
    par:`÷${TP}`,comm:"",cSend:0,cRecv:0,note:""});

  const arData2 = Sq*D*bpe;
  const arPerGpu2 = arSend(arData2, TP);
  ops.push({sec:"共有Expert",op:"Down → AllReduce #2",
    fGpu:mm(Sq,Sh_I/TP,D), fAll:mm(Sq,Sh_I,D),
    par:`行分割(÷${TP}) → AllReduce`,
    comm:`★ AR#2 (TP=${TP})`,cSend:arPerGpu2,cRecv:arPerGpu2,
    note:""});

  // ============ Residual 2 ============
  ops.push({sec:"残差②",op:"routed+shared+h",
    fGpu:2*Sq*D, fAll:2*Sq*D*TP,
    par:"replicated (AR後)",comm:"",cSend:0,cRecv:0,note:""});

  return ops;
}

function grp(ops){const g=[];let c=null;for(const o of ops){if(!c||c.sec!==o.sec){c={sec:o.sec,ops:[]};g.push(c);}c.ops.push(o);}return g;}

export default function App(){
  const [mode,setMode]=useState("prefill");
  const [S,setS]=useState(1000);
  const [Skv,setSkv]=useState(4096);
  const [TP,setTP]=useState(8);
  const [EP,setEP]=useState(8);
  const isDec=mode==="decode";
  const ops=build(S,Skv,isDec,TP,EP);
  const groups=grp(ops);
  const gpuF=ops.reduce((s,o)=>s+o.fGpu,0);
  const allF=ops.reduce((s,o)=>s+o.fAll,0);
  const gpuSend=ops.reduce((s,o)=>s+o.cSend,0);
  const arSend_=ops.filter(o=>o.comm.includes("AR")).reduce((s,o)=>s+o.cSend,0);
  const a2aSend=ops.filter(o=>o.comm.includes("A2A")).reduce((s,o)=>s+o.cSend,0);
  const replF=ops.filter(o=>o.par.includes("replicated")||o.par.includes("★replicated")).reduce((s,o)=>s+o.fGpu,0);
  const splitF=gpuF-replF;

  const secS=groups.map(g=>({n:g.sec,fg:g.ops.reduce((s,o)=>s+o.fGpu,0),cs:g.ops.reduce((s,o)=>s+o.cSend,0)}));
  const mxF=Math.max(...secS.map(s=>s.fg),1);
  const mxC=Math.max(...secS.map(s=>s.cs),1);
  const pc=isDec?"#E8734A":"#4A90D9";

  return(
    <div style={{minHeight:"100vh",background:"linear-gradient(160deg,#08080e,#0e0e1c,#0a0a14)",color:"#e0e0ec",fontFamily:"'IBM Plex Sans','Noto Sans JP',system-ui,sans-serif",padding:"16px 10px"}}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+JP:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
      <div style={{maxWidth:1200,margin:"0 auto"}}>
        <div style={{textAlign:"center",marginBottom:12}}>
          <h1 style={{fontSize:19,fontWeight:700,color:"#f0f0f5",margin:0}}>GLM-5 — 1層 FLOPS・通信量 (★全て1GPUあたり★)</h1>
          <p style={{fontSize:10,color:"#666",margin:"3px 0 0"}}>FLOPS=各GPUの実演算量 (分割+冗長) | 通信=各GPUの送信量</p>
        </div>

        {/* Controls */}
        <div style={{display:"flex",justifyContent:"center",alignItems:"center",gap:7,marginBottom:12,flexWrap:"wrap"}}>
          {[{k:"prefill",l:"プリフィル"},{k:"decode",l:"デコード"}].map(m=>(
            <button key={m.k} onClick={()=>setMode(m.k)} style={{padding:"6px 14px",borderRadius:6,fontSize:11.5,fontWeight:700,cursor:"pointer",
              border:mode===m.k?`2px solid ${m.k==="prefill"?"#4A90D9":"#E8734A"}`:"2px solid rgba(255,255,255,0.08)",
              background:mode===m.k?`${m.k==="prefill"?"#4A90D9":"#E8734A"}18`:"transparent",
              color:mode===m.k?(m.k==="prefill"?"#6BB3FF":"#FFB088"):"#666"}}>{m.l}</button>
          ))}
          {[{l:isDec?"S_kv":"S",v:isDec?Skv:S,s:isDec?setSkv:setS},{l:"TP",v:TP,s:setTP},{l:"EP",v:EP,s:setEP}].map((c,i)=>(
            <div key={i} style={{display:"flex",alignItems:"center",gap:3}}>
              <label style={{fontSize:9.5,color:"#888"}}>{c.l}=</label>
              <input type="number" value={c.v} onChange={e=>{c.s(Math.max(1,parseInt(e.target.value)||1));}}
                style={{width:50,padding:"3px 5px",borderRadius:4,border:"1px solid rgba(255,255,255,0.12)",background:"rgba(255,255,255,0.05)",color:"#ddd",fontSize:11.5,fontFamily:"'IBM Plex Mono',monospace"}}/>
            </div>
          ))}
        </div>

        {/* Summary - ALL PER GPU */}
        <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:6,marginBottom:10}}>
          {[
            {l:"FLOPS/GPU/層",v:fmt(gpuF),s:`×78=${fmt(gpuF*78)}`,c:"#e0e0f0"},
            {l:"うち冗長計算",v:fmt(replF),s:`${pct(replF,gpuF)} (通信回避)`,c:"#ffcc44"},
            {l:"うち分割計算",v:fmt(splitF),s:pct(splitF,gpuF),c:"#66bbff"},
            {l:"送信量/GPU/層",v:fB(gpuSend),s:`×78=${fB(gpuSend*78)}`,c:"#ffaa66"},
            {l:"AR送信(TP)",v:fB(arSend_),s:`×2回/層`,c:"#66bbff"},
            {l:"A2A送信(EP)",v:fB(a2aSend),s:`×2回/層`,c:"#cc66ff"},
          ].map((c,i)=>(
            <div key={i} style={{background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.06)",borderRadius:7,padding:"7px 8px",textAlign:"center"}}>
              <div style={{fontSize:8,color:"#666",textTransform:"uppercase",letterSpacing:0.8}}>{c.l}</div>
              <div style={{fontSize:13,fontWeight:700,color:c.c,fontFamily:"'IBM Plex Mono',monospace",margin:"1px 0"}}>{c.v}</div>
              <div style={{fontSize:9,color:"#555"}}>{c.s}</div>
            </div>
          ))}
        </div>

        {/* Dual bars */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:10}}>
          {[{ti:"FLOPS/GPU",k:"fg",mx:mxF,u:fmt,co:pc},{ti:"送信量/GPU",k:"cs",mx:mxC,u:fB,co:"#cc66ff"}].map((ch,ci)=>(
            <div key={ci} style={{padding:"8px 10px",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.05)",borderRadius:8}}>
              <div style={{fontSize:9,fontWeight:600,color:"#777",marginBottom:5,textTransform:"uppercase",letterSpacing:1}}>{ch.ti}</div>
              {secS.map((s,i)=>{const v=s[ch.k];return(
                <div key={i} style={{display:"flex",alignItems:"center",marginBottom:2.5}}>
                  <div style={{width:110,fontSize:9.5,color:"#999",flexShrink:0,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{s.n}</div>
                  <div style={{flex:1,height:11,background:"rgba(255,255,255,0.03)",borderRadius:2,overflow:"hidden",marginRight:3}}>
                    <div style={{height:"100%",width:`${ch.mx>0?Math.max(v>0?0.4:0,v/ch.mx*100):0}%`,background:`linear-gradient(90deg,${ch.co},${ch.co}88)`,borderRadius:2}}/>
                  </div>
                  <div style={{width:60,fontSize:9.5,color:"#aaa",fontFamily:"'IBM Plex Mono',monospace",textAlign:"right"}}>{v>0?ch.u(v):"—"}</div>
                </div>
              );})}
            </div>
          ))}
        </div>

        {/* Table */}
        <div style={{background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.05)",borderRadius:8,overflow:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:10.5,minWidth:1000}}>
            <thead>
              <tr style={{background:"rgba(255,255,255,0.04)"}}>
                {["セクション","演算","型",{t:"FLOPS/GPU",w:85},{t:"FLOPS/全GPU",w:85},"分割方式",{t:"通信",w:90},{t:"送信/GPU",w:75}].map((h,i)=>{
                  const txt=typeof h==="string"?h:h.t;
                  return <th key={i} style={{padding:"6px 4px",textAlign:[3,4,7].includes(i)?"right":"left",color:"#777",fontWeight:600,fontSize:8.5,textTransform:"uppercase",letterSpacing:0.7,borderBottom:"1px solid rgba(255,255,255,0.06)",whiteSpace:"nowrap",width:typeof h==="object"?h.w:undefined}}>{txt}</th>;
                })}
              </tr>
            </thead>
            <tbody>
              {groups.map((g,gi)=>{
                const sgf=g.ops.reduce((s,o)=>s+o.fGpu,0);
                const saf=g.ops.reduce((s,o)=>s+o.fAll,0);
                const sc=g.ops.reduce((s,o)=>s+o.cSend,0);
                return g.ops.map((op,oi)=>(
                  <tr key={`${gi}-${oi}`} style={{borderBottom:"1px solid rgba(255,255,255,0.02)",background:oi===0?"rgba(255,255,255,0.008)":"transparent"}}>
                    {oi===0&&<td rowSpan={g.ops.length} style={{padding:"4px",verticalAlign:"top",color:"#aaa",fontWeight:600,fontSize:9.5,borderRight:"1px solid rgba(255,255,255,0.03)",background:"rgba(255,255,255,0.005)",width:90}}>
                      {g.sec}
                      <div style={{fontSize:8,color:"#555",marginTop:1,fontFamily:"'IBM Plex Mono',monospace"}}>
                        {sgf>0&&<div>GPU: {fmt(sgf)}</div>}
                        {sc>0&&<div style={{color:"#cc66ff"}}>送: {fB(sc)}</div>}
                      </div>
                    </td>}
                    <td style={{padding:"3px 4px",color:"#c0c0d0",fontSize:10}}>{op.op}</td>
                    <td style={{padding:"3px 2px"}}>
                      <span style={{display:"inline-block",padding:"0px 3px",borderRadius:2,fontSize:8.5,fontWeight:600,
                        background:op.t==="M"?"rgba(74,144,217,0.12)":"rgba(255,180,80,0.08)",
                        color:op.t==="M"?"#6BB3FF":"#FFBB66"}}>{op.t==="M"?"MM":"El"}</span>
                    </td>
                    <td style={{padding:"3px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",color:"#e0e0f0",fontWeight:600,fontSize:10,whiteSpace:"nowrap"}}>
                      {op.fGpu>0?fmt(op.fGpu):"—"}
                    </td>
                    <td style={{padding:"3px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",color:"#888",fontSize:9.5,whiteSpace:"nowrap"}}>
                      {op.fAll>0?fmt(op.fAll):"—"}
                    </td>
                    <td style={{padding:"3px 4px",fontSize:9,color:op.par.includes("★")?"#ffcc44":"#999",fontWeight:op.par.includes("★")?600:400,whiteSpace:"pre-wrap",lineHeight:1.35,maxWidth:200}}>
                      {op.par||"—"}
                    </td>
                    <td style={{padding:"3px 4px",fontSize:9,whiteSpace:"nowrap",
                      color:op.comm.includes("AR")?"#66bbff":op.comm.includes("A2A")?"#cc66ff":"#555",
                      fontWeight:op.cSend>0?600:400}}>{op.comm||"—"}</td>
                    <td style={{padding:"3px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",fontSize:9.5,
                      color:op.cSend>0?"#ffaa66":"#444",fontWeight:op.cSend>0?600:400,whiteSpace:"nowrap"}}>
                      {op.cSend>0?fB(op.cSend):"—"}
                    </td>
                  </tr>
                ));
              })}
              <tr style={{background:"rgba(255,255,255,0.05)",borderTop:"2px solid rgba(255,255,255,0.1)"}}>
                <td colSpan={3} style={{padding:"6px 4px",fontWeight:700,color:"#f0f0f5",fontSize:11}}>合計/層</td>
                <td style={{padding:"6px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",fontWeight:700,color:"#f0f0f5",fontSize:11}}>{fmt(gpuF)}</td>
                <td style={{padding:"6px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",color:"#888",fontSize:10}}>{fmt(allF)}</td>
                <td/>
                <td style={{padding:"6px 4px",fontWeight:600,color:"#ddd",fontSize:9}}>送信計/GPU</td>
                <td style={{padding:"6px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",fontWeight:700,color:"#ffaa66",fontSize:11}}>{fB(gpuSend)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Legend */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8,marginTop:10}}>
          <div style={{padding:"9px 11px",background:"rgba(255,200,50,0.05)",border:"1px solid rgba(255,200,50,0.15)",borderRadius:7,fontSize:10,color:"#c8c8d8",lineHeight:1.6}}>
            <strong style={{color:"#ffcc44"}}>★replicated = 冗長計算</strong><br/>
            全GPUが同一計算を実行。FLOPS/GPU = FLOPS/全GPU。<br/>
            通信が発生しない代わりにGPU間で計算が重複。<br/><br/>
            対象: W_dq, W_dkv, W_kr, W_gate, RMSNorm<br/>
            →ボトルネック次元が小さい射影＋全GPU同一結果が必要な処理
          </div>
          <div style={{padding:"9px 11px",background:"rgba(74,144,217,0.05)",border:"1px solid rgba(74,144,217,0.15)",borderRadius:7,fontSize:10,color:"#c8c8d8",lineHeight:1.6}}>
            <strong style={{color:"#6BB3FF"}}>head/列/行 分割 + AllReduce</strong><br/>
            FLOPS/GPU = FLOPS/全GPU ÷ TP。<br/>
            W_oと共有Expert W_downで部分和→AllReduceが必要。<br/><br/>
            AllReduce送信/GPU: S×6144×2B×(TP-1)/TP<br/>
            Ring方式: 各GPUが≈data量を送信+受信<br/>
            NVLink (~900GB/s)なら μs〜ms オーダー
          </div>
          <div style={{padding:"9px 11px",background:"rgba(204,102,255,0.05)",border:"1px solid rgba(204,102,255,0.15)",borderRadius:7,fontSize:10,color:"#c8c8d8",lineHeight:1.6}}>
            <strong style={{color:"#cc66ff"}}>EP分割 + All-to-All</strong><br/>
            FLOPS/GPU = FLOPS/全GPU ÷ EP (均等時)。<br/>
            各GPUが{Math.ceil(256/EP)} experts保持。<br/><br/>
            A2A送信/GPU: S×8×{((EP-1)/EP).toFixed(2)}×6144×2B<br/>
            各GPUが他ランクのexpert宛tokenを送信。<br/>
            マルチノード時はIB帯域(~400Gb/s)が律速。
          </div>
        </div>

        <div style={{marginTop:8,padding:"7px 10px",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)",borderRadius:6,fontSize:9.5,color:"#666",lineHeight:1.55}}>
          <strong style={{color:"#888"}}>全値は1GPUあたり。</strong>
          「FLOPS/全GPU」列は参考値(モデル理論値)。Expert FLOPSはtoken均等分配を仮定(実際は偏りあり)。
          FP8推論なら通信量半減。BF16前提。TP,EP,Sは上部で変更可能。
        </div>
      </div>
    </div>
  );
}