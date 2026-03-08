import { useState } from "react";

// ===== Kimi K2.5 Architecture Constants (from config.json) =====
const D=7168,H=64,QR=1536,KR=512;
const Dn=128,Dr=64,Dv=128; // qk_nope=128, qk_rope=64, v_head=128
const NE=384,KE=8,MI=2048,SI=18432; // 384 experts, shared intermediate=18432
const LAYERS=61, DENSE_LAYERS=1; // first_k_dense_replace=1
const RSCALE=2.827; // routed_scaling_factor
const VOCAB=163840;
const mm=(M,K,N)=>2*M*K*N;
function fmt(n){if(n>=1e15)return(n/1e15).toFixed(2)+"P";if(n>=1e12)return(n/1e12).toFixed(2)+"T";if(n>=1e9)return(n/1e9).toFixed(2)+"G";if(n>=1e6)return(n/1e6).toFixed(2)+"M";if(n>=1e3)return(n/1e3).toFixed(2)+"K";return n.toFixed(0);}
function fB(b){if(b>=1e9)return(b/1e9).toFixed(2)+" GB";if(b>=1e6)return(b/1e6).toFixed(2)+" MB";if(b>=1e3)return(b/1e3).toFixed(2)+" KB";if(b>0)return b.toFixed(0)+" B";return "0";}
function pct(v,t){return t===0?"—":(v/t*100).toFixed(1)+"%";}

function build(S,Skv,isDec,TP,EP){
  const Sq=isDec?1:S, bpe=2;
  const ht=H/TP, et=NE/EP;
  const arS=(d,N)=>d*(N-1)/N;
  const o=[];
  const tpr=Sq*KE/EP;
  const Dqk=Dn+Dr; // 192

  // ===== RMSNorm 1 =====
  o.push({s:"RMSNorm①",t:"E",
    op:`正規化: (${Sq}×${D})`,
    fg:5*Sq*D,fa:5*Sq*D*TP,
    par:"replicated",comm:"",cs:0});

  // ===== Q Down (replicated) =====
  o.push({s:"Q射影",t:"M",
    op:`Down x×W_dq: (${Sq}×${D})×(${D}×${QR}) → (${Sq}×${QR})`,
    fg:mm(Sq,D,QR),fa:mm(Sq,D,QR)*TP,
    par:"★replicated",comm:"",cs:0});

  const uq=ht*Dn;
  o.push({s:"Q射影",t:"M",
    op:`Up nope c_q×W_uq: (${Sq}×${QR})×(${QR}×${uq}) → (${Sq}×${ht}h×${Dn})`,
    fg:mm(Sq,QR,uq),fa:mm(Sq,QR,H*Dn),
    par:`head÷${TP}`,comm:"",cs:0});

  const qr=ht*Dr;
  o.push({s:"Q射影",t:"M",
    op:`Up rope c_q×W_qr: (${Sq}×${QR})×(${QR}×${qr}) → (${Sq}×${ht}h×${Dr})`,
    fg:mm(Sq,QR,qr),fa:mm(Sq,QR,H*Dr),
    par:`head÷${TP}`,comm:"",cs:0});

  o.push({s:"Q射影",t:"E",
    op:`RoPE回転: (${Sq}×${ht}h×${Dr/2}pairs)`,
    fg:6*Sq*ht*(Dr/2),fa:6*Sq*H*(Dr/2),
    par:`÷${TP}`,comm:"",cs:0});

  // ===== KV =====
  if(!isDec){
    o.push({s:"KV射影",t:"M",
      op:`Down x×W_dkv: (${S}×${D})×(${D}×${KR}) → c_kv(${S}×${KR}) [cache]`,
      fg:mm(S,D,KR),fa:mm(S,D,KR)*TP,
      par:"★replicated",comm:"",cs:0});

    const uk=ht*Dn;
    o.push({s:"KV射影",t:"M",
      op:`Up K c_kv×W_uk: (${S}×${KR})×(${KR}×${uk}) → K_nope(${S}×${ht}h×${Dn})`,
      fg:mm(S,KR,uk),fa:mm(S,KR,H*Dn),
      par:`head÷${TP}`,comm:"",cs:0});

    const uv=ht*Dv;
    o.push({s:"KV射影",t:"M",
      op:`Up V c_kv×W_uv: (${S}×${KR})×(${KR}×${uv}) → V(${S}×${ht}h×${Dv})`,
      fg:mm(S,KR,uv),fa:mm(S,KR,H*Dv),
      par:`head÷${TP}`,comm:"",cs:0});

    o.push({s:"KV射影",t:"M",
      op:`K rope x×W_kr: (${S}×${D})×(${D}×${Dr}) → K_rope(${S}×${Dr}) [cache]`,
      fg:mm(S,D,Dr),fa:mm(S,D,Dr)*TP,
      par:"replicated",comm:"",cs:0});

    o.push({s:"KV射影",t:"E",
      op:`RoPE回転 K: (${S}×${Dr/2}pairs)`,
      fg:6*S*(Dr/2),fa:6*S*(Dr/2)*TP,
      par:"replicated",comm:"",cs:0});
  } else {
    o.push({s:"KV射影(新tok)",t:"M",
      op:`Down x×W_dkv: (1×${D})×(${D}×${KR}) → c_kv(1×${KR}) [cache追記]`,
      fg:mm(1,D,KR),fa:mm(1,D,KR)*TP,
      par:"replicated",comm:"",cs:0});
    o.push({s:"KV射影(新tok)",t:"M",
      op:`K rope x×W_kr: (1×${D})×(${D}×${Dr}) → K_rope(1×${Dr}) [cache追記]`,
      fg:mm(1,D,Dr),fa:mm(1,D,Dr)*TP,
      par:"replicated",comm:"",cs:0});
    o.push({s:"KV射影(新tok)",t:"E",
      op:`RoPE回転 K: (1×${Dr/2}pairs)`,
      fg:6*(Dr/2),fa:6*(Dr/2)*TP,
      par:"replicated",comm:"",cs:0});
  }

  // ===== Attention =====
  if(!isDec){
    o.push({s:"Attention",t:"M",
      op:`Q×K^T: ${ht}h × (${S}×${Dqk})×(${Dqk}×${S}) → scores(${S}×${S})`,
      fg:ht*mm(S,Dqk,S),fa:H*mm(S,Dqk,S),
      par:`head÷${TP}`,comm:"",cs:0});
    o.push({s:"Attention",t:"E",
      op:`Scale÷√${Dqk} + CausalMask + Softmax: (${ht}h×${S}×${S})`,
      fg:7*ht*S*S,fa:7*H*S*S,
      par:`÷${TP}`,comm:"",cs:0});
    o.push({s:"Attention",t:"M",
      op:`Weights×V: ${ht}h × (${S}×${S})×(${S}×${Dv}) → out(${S}×${Dv})`,
      fg:ht*mm(S,S,Dv),fa:H*mm(S,S,Dv),
      par:`head÷${TP}`,comm:"",cs:0});
  } else {
    o.push({s:"Absorbed",t:"M",
      op:`Q吸収 Q_nope×W_uk^T: ${ht}h × (1×${Dn})×(${Dn}×${KR}) → Q'(1×${KR})`,
      fg:ht*mm(1,Dn,KR),fa:H*mm(1,Dn,KR),
      par:`÷${TP}`,comm:"",cs:0});
    o.push({s:"Absorbed",t:"M",
      op:`nope score Q'×c_kv^T: ${ht}h × (1×${KR})×(${KR}×${Skv}) → (1×${Skv})`,
      fg:ht*mm(1,KR,Skv),fa:H*mm(1,KR,Skv),
      par:`÷${TP}`,comm:"",cs:0});
    o.push({s:"Absorbed",t:"M",
      op:`rope score Q_r×K_r^T: ${ht}h × (1×${Dr})×(${Dr}×${Skv}) → (1×${Skv})`,
      fg:ht*mm(1,Dr,Skv),fa:H*mm(1,Dr,Skv),
      par:`÷${TP}`,comm:"",cs:0});
    o.push({s:"Absorbed",t:"E",
      op:`scores合成 + Softmax: (${ht}h×1×${Skv})`,
      fg:7*ht*Skv,fa:7*H*Skv,
      par:`÷${TP}`,comm:"",cs:0});
    o.push({s:"V集約",t:"M",
      op:`w×c_kv: ${ht}h × (1×${Skv})×(${Skv}×${KR}) → agg(1×${KR})`,
      fg:ht*mm(1,Skv,KR),fa:H*mm(1,Skv,KR),
      par:`÷${TP}`,comm:"",cs:0});
    o.push({s:"V集約",t:"M",
      op:`agg×W_uv: ${ht}h × (1×${KR})×(${KR}×${Dv}) → V_out(1×${Dv})`,
      fg:ht*mm(1,KR,Dv),fa:H*mm(1,KR,Dv),
      par:`÷${TP}`,comm:"",cs:0});
  }

  // ===== W_o + AllReduce =====
  const oR=ht*Dv;
  const arD=Sq*D*bpe;
  o.push({s:"Attn出力",t:"M",
    op:`W_o: (${Sq}×${oR})×(${oR}×${D}) → partial(${Sq}×${D}) → AllReduce`,
    fg:mm(Sq,oR,D),fa:mm(Sq,H*Dv,D),
    par:`行分割÷${TP}`,comm:`★AR#1(TP=${TP})`,cs:arS(arD,TP)});

  o.push({s:"残差①",t:"E",
    op:`加算: (${Sq}×${D}) + (${Sq}×${D})`,
    fg:Sq*D,fa:Sq*D*TP,par:"replicated",comm:"",cs:0});
  o.push({s:"RMSNorm②",t:"E",
    op:`正規化: (${Sq}×${D})`,
    fg:5*Sq*D,fa:5*Sq*D*TP,par:"replicated",comm:"",cs:0});

  // ===== Router =====
  o.push({s:"Router",t:"M",
    op:`h×W_gate: (${Sq}×${D})×(${D}×${NE}) → scores(${Sq}×${NE})`,
    fg:mm(Sq,D,NE),fa:mm(Sq,D,NE)*TP,
    par:"★replicated",comm:"",cs:0});
  o.push({s:"Router",t:"E",
    op:`sigmoid + topk8 + norm: (${Sq}×${NE})`,
    fg:10*Sq*NE,fa:10*Sq*NE*TP,
    par:"replicated",comm:"",cs:0});

  // ===== EP Dispatch =====
  const a2a=Sq*KE*((EP-1)/EP)*D*bpe;
  o.push({s:"EP Dispatch",t:"E",
    op:`All-to-All: ${Sq}tok×8exp×${((EP-1)/EP).toFixed(2)}×${D}dim×${bpe}B`,
    fg:0,fa:0,
    par:`EP=${EP} (${et}exp/GPU)`,comm:`★A2A#1(EP=${EP})`,cs:a2a});

  // ===== Routed Experts =====
  o.push({s:"Expert×8",t:"M",
    op:`Gate+Up: ${tpr.toFixed(0)}tok × (1×${D})×(${D}×${2*MI}) → (1×${2*MI})`,
    fg:tpr*mm(1,D,2*MI),fa:KE*mm(Sq,D,2*MI),
    par:`EP÷${EP}: ${et}exp/GPU`,comm:"",cs:0});
  o.push({s:"Expert×8",t:"E",
    op:`SiLU(gate)⊙up: ${tpr.toFixed(0)}tok × (1×${MI})`,
    fg:tpr*5*MI,fa:KE*5*Sq*MI,
    par:"",comm:"",cs:0});
  o.push({s:"Expert×8",t:"M",
    op:`Down: ${tpr.toFixed(0)}tok × (1×${MI})×(${MI}×${D}) → (1×${D})`,
    fg:tpr*mm(1,MI,D),fa:KE*mm(Sq,MI,D),
    par:`EP÷${EP}`,comm:"",cs:0});
  o.push({s:"Expert×8",t:"E",
    op:`重み付き合成 ×${RSCALE}: ${(Sq/EP).toFixed(0)}tok × Σ(w×out)`,
    fg:KE*2*Sq*D/EP,fa:KE*2*Sq*D,
    par:"",comm:"",cs:0});

  // ===== EP Combine =====
  o.push({s:"EP Combine",t:"E",
    op:`All-to-All: Dispatchと対称`,
    fg:0,fa:0,par:"対称",comm:`★A2A#2(EP=${EP})`,cs:a2a});

  // ===== Shared Expert (TP split) =====
  const si=2*SI/TP;
  o.push({s:"共有Expert",t:"M",
    op:`Gate+Up h×W: (${Sq}×${D})×(${D}×${si}) → (${Sq}×${si})`,
    fg:mm(Sq,D,si),fa:mm(Sq,D,2*SI),
    par:`列分割÷${TP}`,comm:"",cs:0});
  o.push({s:"共有Expert",t:"E",
    op:`SiLU(gate)⊙up: (${Sq}×${SI/TP})`,
    fg:5*Sq*SI/TP,fa:5*Sq*SI,
    par:`÷${TP}`,comm:"",cs:0});
  const arD2=Sq*D*bpe;
  o.push({s:"共有Expert",t:"M",
    op:`Down h×W: (${Sq}×${SI/TP})×(${SI/TP}×${D}) → partial(${Sq}×${D}) → AR`,
    fg:mm(Sq,SI/TP,D),fa:mm(Sq,SI,D),
    par:`行分割÷${TP}`,comm:`★AR#2(TP=${TP})`,cs:arS(arD2,TP)});

  o.push({s:"残差②",t:"E",
    op:`加算: routed + shared + h (${Sq}×${D})`,
    fg:2*Sq*D,fa:2*Sq*D*TP,par:"replicated",comm:"",cs:0});

  return o;
}

function grp(ops){const g=[];let c=null;for(const o of ops){if(!c||c.s!==o.s){c={s:o.s,ops:[]};g.push(c);}c.ops.push(o);}return g;}

export default function App(){
  const [mode,setMode]=useState("prefill");
  const [S,setS]=useState(1000);
  const [Skv,setSkv]=useState(4096);
  const [TP,setTP]=useState(8);
  const [EP,setEP]=useState(8);
  const isDec=mode==="decode";
  const ops=build(S,Skv,isDec,TP,EP);
  const groups=grp(ops);
  const gpuF=ops.reduce((a,o)=>a+o.fg,0);
  const allF=ops.reduce((a,o)=>a+o.fa,0);
  const gpuSend=ops.reduce((a,o)=>a+o.cs,0);
  const arS_=ops.filter(o=>o.comm.includes("AR")).reduce((a,o)=>a+o.cs,0);
  const a2aS_=ops.filter(o=>o.comm.includes("A2A")).reduce((a,o)=>a+o.cs,0);
  const replF=ops.filter(o=>o.par.includes("replicated")||o.par.includes("★replicated")).reduce((a,o)=>a+o.fg,0);

  const secS=groups.map(g=>({n:g.s,fg:g.ops.reduce((a,o)=>a+o.fg,0),cs:g.ops.reduce((a,o)=>a+o.cs,0)}));
  const mxF=Math.max(...secS.map(x=>x.fg),1);
  const mxC=Math.max(...secS.map(x=>x.cs),1);
  const pc=isDec?"#E8734A":"#3A7CA5";

  return(
    <div style={{minHeight:"100vh",background:"linear-gradient(160deg,#080810,#0c0e1e,#080a16)",color:"#e0e0ec",fontFamily:"'IBM Plex Sans','Noto Sans JP',system-ui,sans-serif",padding:"14px 8px"}}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+JP:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
      <div style={{maxWidth:1200,margin:"0 auto"}}>
        <div style={{textAlign:"center",marginBottom:10}}>
          <h1 style={{fontSize:18,fontWeight:700,color:"#f0f0f5",margin:0}}>
            <span style={{color:"#7B68EE"}}>Kimi K2.5</span> — 1層 FLOPS・通信 (全て1GPUあたり)
          </h1>
          <p style={{fontSize:10,color:"#666",margin:"3px 0 0"}}>
            1T params (32B active) · {LAYERS}層 (Dense {DENSE_LAYERS}層 + MoE {LAYERS-DENSE_LAYERS}層) · {NE} experts → 8 active · MLA (kv_lora={KR}) · hidden={D}
          </p>
        </div>

        {/* Config comparison badge */}
        <div style={{display:"flex",justifyContent:"center",gap:4,marginBottom:8,flexWrap:"wrap"}}>
          {[
            {k:"hidden",v:D},{k:"heads",v:H},{k:"layers",v:LAYERS},
            {k:"q_lora",v:QR},{k:"kv_lora",v:KR},
            {k:"qk_nope",v:Dn},{k:"qk_rope",v:Dr},{k:"v_head",v:Dv},
            {k:"experts",v:NE},{k:"top_k",v:KE},{k:"moe_inter",v:MI},{k:"shared_inter",v:SI},
          ].map((c,i)=>(
            <span key={i} style={{fontSize:8.5,padding:"2px 5px",borderRadius:3,background:"rgba(123,104,238,0.1)",border:"1px solid rgba(123,104,238,0.2)",color:"#aaa",fontFamily:"'IBM Plex Mono',monospace"}}>
              {c.k}={c.v}
            </span>
          ))}
        </div>

        {/* Controls */}
        <div style={{display:"flex",justifyContent:"center",alignItems:"center",gap:7,marginBottom:10,flexWrap:"wrap"}}>
          {[{k:"prefill",l:"プリフィル"},{k:"decode",l:"デコード"}].map(m=>(
            <button key={m.k} onClick={()=>setMode(m.k)} style={{padding:"5px 13px",borderRadius:6,fontSize:11,fontWeight:700,cursor:"pointer",
              border:mode===m.k?`2px solid ${m.k==="prefill"?"#3A7CA5":"#E8734A"}`:"2px solid rgba(255,255,255,0.08)",
              background:mode===m.k?`${m.k==="prefill"?"#3A7CA5":"#E8734A"}18`:"transparent",
              color:mode===m.k?(m.k==="prefill"?"#6BB3FF":"#FFB088"):"#666"}}>{m.l}</button>
          ))}
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
            {l:"FLOPS/GPU/層",v:fmt(gpuF),u:`×${LAYERS}=${fmt(gpuF*LAYERS)}`,c:"#e0e0f0"},
            {l:"冗長(replicated)",v:fmt(replF),u:pct(replF,gpuF),c:"#ffcc44"},
            {l:"分割計算",v:fmt(gpuF-replF),u:pct(gpuF-replF,gpuF),c:"#6BB3FF"},
            {l:"送信/GPU/層",v:fB(gpuSend),u:`×${LAYERS}=${fB(gpuSend*LAYERS)}`,c:"#ffaa66"},
            {l:"AR送信(TP)",v:fB(arS_),u:"×2回/層",c:"#6BB3FF"},
            {l:"A2A送信(EP)",v:fB(a2aS_),u:"×2回/層",c:"#cc66ff"},
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
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:10,minWidth:1000}}>
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
                    {oi===0&&<td rowSpan={g.ops.length} style={{padding:"3px",verticalAlign:"top",color:"#aaa",fontWeight:600,fontSize:9,borderRight:"1px solid rgba(255,255,255,0.03)",background:"rgba(255,255,255,0.005)",width:85}}>
                      {g.s}
                      <div style={{fontSize:8,color:"#555",marginTop:1,fontFamily:"'IBM Plex Mono',monospace"}}>
                        {sgf>0&&<div>{fmt(sgf)}</div>}
                        {sc>0&&<div style={{color:"#cc66ff"}}>{fB(sc)}</div>}
                      </div>
                    </td>}
                    <td style={{padding:"3px 4px",color:"#d0d0e0",fontSize:10,fontFamily:"'IBM Plex Mono','Noto Sans JP',monospace",whiteSpace:"pre-wrap",lineHeight:1.35,maxWidth:380}}>{op.op}</td>
                    <td style={{padding:"3px 2px"}}>
                      <span style={{display:"inline-block",padding:"0px 3px",borderRadius:2,fontSize:8.5,fontWeight:700,
                        background:op.t==="M"?"rgba(74,144,217,0.15)":"rgba(255,180,80,0.1)",
                        color:op.t==="M"?"#6BB3FF":"#FFBB66"}}>{op.t==="M"?"MM":"El"}</span>
                    </td>
                    <td style={{padding:"3px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",color:"#e0e0f0",fontWeight:600,fontSize:10,whiteSpace:"nowrap"}}>{op.fg>0?fmt(op.fg):"—"}</td>
                    <td style={{padding:"3px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",color:"#777",fontSize:9,whiteSpace:"nowrap"}}>{op.fa>0?fmt(op.fa):"—"}</td>
                    <td style={{padding:"3px 4px",fontSize:9,color:op.par.includes("★")?"#ffcc44":"#999",fontWeight:op.par.includes("★")?600:400,whiteSpace:"nowrap"}}>{op.par||"—"}</td>
                    <td style={{padding:"3px 4px",fontSize:9,whiteSpace:"nowrap",
                      color:op.comm.includes("AR")?"#6BB3FF":op.comm.includes("A2A")?"#cc66ff":"#555",
                      fontWeight:op.cs>0?600:400}}>{op.comm||"—"}</td>
                    <td style={{padding:"3px 4px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",fontSize:9,
                      color:op.cs>0?"#ffaa66":"#444",fontWeight:op.cs>0?600:400,whiteSpace:"nowrap"}}>{op.cs>0?fB(op.cs):"—"}</td>
                  </tr>
                ));
              })}
              <tr style={{background:"rgba(255,255,255,0.05)",borderTop:"2px solid rgba(255,255,255,0.1)"}}>
                <td colSpan={3} style={{padding:"5px 3px",fontWeight:700,color:"#f0f0f5",fontSize:10.5}}>合計/層</td>
                <td style={{padding:"5px 3px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",fontWeight:700,color:"#f0f0f5",fontSize:10.5}}>{fmt(gpuF)}</td>
                <td style={{padding:"5px 3px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",color:"#777",fontSize:9.5}}>{fmt(allF)}</td>
                <td/><td style={{fontSize:9,color:"#ddd"}}>送信計</td>
                <td style={{padding:"5px 3px",textAlign:"right",fontFamily:"'IBM Plex Mono',monospace",fontWeight:700,color:"#ffaa66",fontSize:10.5}}>{fB(gpuSend)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* GLM-5 vs K2.5 comparison note */}
        <div style={{marginTop:10,padding:"9px 11px",background:"rgba(123,104,238,0.05)",border:"1px solid rgba(123,104,238,0.2)",borderRadius:7,fontSize:10,color:"#c8c8d8",lineHeight:1.6}}>
          <strong style={{color:"#7B68EE"}}>GLM-5 との主な違い:</strong><br/>
          <span style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9.5}}>
          hidden_size: 6144→<strong>7168</strong> | layers: 78→<strong>61</strong> | experts: 256→<strong>384</strong> | q_lora: 2048→<strong>1536</strong><br/>
          qk_nope: 192→<strong>128</strong> | v_head: 256→<strong>128</strong> | shared_inter: 12288→<strong>18432</strong> | dense_layers: 3→<strong>1</strong><br/>
          </span>
          K2.5はDeepSeek-V3と同じMLAパラメータ(nope=128, v=128)。GLM-5はMLA-256変種(nope=192, v=256)でデコード効率を最適化。<br/>
          K2.5は共有Expertが大きく(18432)、expert数も多い(384)が、Q圧縮次元が小さい(1536)。
        </div>

        <div style={{marginTop:7,padding:"6px 9px",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)",borderRadius:5,fontSize:9,color:"#666",lineHeight:1.5}}>
          <span style={{color:"#6BB3FF"}}>MM</span>=行列積(2MKN) <span style={{color:"#FFBB66"}}>El</span>=要素演算 <span style={{color:"#ffcc44"}}>★replicated</span>=冗長計算 <span style={{color:"#6BB3FF"}}>AR</span>=AllReduce <span style={{color:"#cc66ff"}}>A2A</span>=All-to-All |
          Expert FLOPSはtoken均等分配仮定。BF16前提(K2.5はINT4 native対応)。config.jsonはKimi-K2-Instruct準拠。
        </div>
      </div>
    </div>
  );
}