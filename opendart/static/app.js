"use strict";
const el=id=>document.getElementById(id);
const experiments={
NOMINAL:{title:"Normal guided approach",description:"Follow image-based guidance through the standard 200–220 s camera blackout. The tested case reaches target contact."},
BLACKOUT:{title:"Extended camera blackout",description:"Remove usable camera images from 200 to 240 s. Inspect navigation hold and recovery. The tested case still reaches contact."},
NO_MEASUREMENT:{title:"No visual measurements",description:"A negative control. Camera measurements are disabled, so stale navigation holds thrust at zero. The expected outcome is a miss."},
BALLISTIC:{title:"Guidance disabled",description:"A negative control. OpenCV and image navigation remain active, but thrust is disabled. The expected outcome is a miss."}};
const form=el("experiment-form");
if(form){
  form.addEventListener("change",()=>{const preset=new FormData(form).get("preset"),choice=experiments[preset];if(choice){el("experiment-title").textContent=choice.title;el("experiment-description").textContent=choice.description;el("preset-code").textContent=preset+" · Fixed seed 7";}});
  form.addEventListener("submit",async event=>{
    event.preventDefault();const preset=new FormData(form).get("preset");
    if(!Object.hasOwn(experiments,preset))return;
    el("run-experiment").disabled=true;el("message").textContent="Starting the AWS computation…";
    try{
      const response=await fetch("/api/jobs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({preset})});
      const data=await response.json();if(!response.ok)throw new Error(data.error||"The experiment could not start");
      window.location.assign(data.page);
    }catch(error){el("message").textContent=error.message;el("run-experiment").disabled=false;}
  });
}
function metric(label,value){
  const box=document.createElement("div"),name=document.createElement("span"),number=document.createElement("strong");
  name.textContent=label;number.textContent=value;box.append(name,number);el("metrics").append(box);
}
function plot(rows){
  const canvas=el("trajectory"),ctx=canvas.getContext("2d");
  const w=canvas.width=Math.max(320,Math.round(canvas.getBoundingClientRect().width)),h=canvas.height=w<600?350:480,pad=45;
  const points=rows.flatMap(r=>[r.truth,r.estimate]).concat([[0,0]]);
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const xy=p=>[pad+(p[0]-xmin)/(xmax-xmin||1)*(w-2*pad),h-pad-(p[1]-ymin)/(ymax-ymin||1)*(h-2*pad)];
  ctx.fillStyle="#171819";ctx.fillRect(0,0,w,h);ctx.font="12px Arial";ctx.fillStyle="#b4b4b0";
  ctx.fillText("x: "+xmin.toFixed(0)+" to "+xmax.toFixed(0)+" m",pad,h-15);ctx.fillText("y: "+ymin.toFixed(0)+" to "+ymax.toFixed(0)+" m",pad,25);
  for(const [key,color]of [["truth","#a8b6c2"],["estimate","#d6a06b"]]){
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(key==="truth"?[6,5]:[]);ctx.beginPath();
    rows.forEach((r,i)=>{const p=xy(r[key]);if(i===0)ctx.moveTo(...p);else ctx.lineTo(...p);});ctx.stroke();
  }
  ctx.setLineDash([]);const[x,y]=xy([0,0]);ctx.strokeStyle="#ece9e2";ctx.beginPath();ctx.moveTo(x-7,y);ctx.lineTo(x+7,y);ctx.moveTo(x,y-7);ctx.lineTo(x,y+7);ctx.stroke();
}
const modeNames={GUIDANCE:"Guiding",GUIDED:"Guiding",NAV_HOLD:"Navigation hold",COAST:"Coasting",OPENING:"Opening",BALLISTIC:"Ballistic",TERMINAL_COAST:"Terminal coast"};
function render(data,id){
  const r=data.result,s=r.summary,n=v=>Number(v).toFixed(2);
  el("results").hidden=false;
  el("outcome").textContent=({TARGET_IMPACT:"Target contact.",WRONG_BODY_IMPACT:"Wrong-body contact.",MISS:"Missed target.",TIMEOUT:"Time limit reached."})[s.status]||s.status;
  el("status").textContent="Computed with OpenCV "+r.runtime.opencv+" / "+r.runtime.platform;
  el("metrics").replaceChildren();
  [["Estimation error",n(s.final_estimation_error_m)+" m"],["Accepted / missing visual updates",s.accepted_updates+" / "+s.missing_updates],["Navigation hold",n(s.nav_hold_s)+" s"],["Correction Δv",n(s.delta_v_mps)+" m/s"],["Compute time",n(r.runtime.wall_s)+" s"],["Simulated event time",n(s.elapsed_s)+" s"]].forEach(m=>metric(...m));
  plot(r.trajectory);window.addEventListener("resize",()=>plot(r.trajectory));el("frames").replaceChildren();
  for(const frame of r.frames){const figure=document.createElement("figure"),img=document.createElement("img"),caption=document.createElement("figcaption");img.src="/jobs/"+id+"/artifacts/"+frame.file;img.alt="OpenCV overlay at "+frame.elapsed_s+" s";img.loading="lazy";caption.textContent=frame.elapsed_s+" s · "+(modeNames[frame.mode]||frame.mode)+" · "+frame.tracker.toLowerCase().replaceAll("_"," ");figure.append(img,caption);el("frames").append(figure);}
  el("runtime").textContent=JSON.stringify({outcome:s.status,preset:data.preset,generated_utc:r.generated_utc,seed:r.config.seed,accepted_after_blackout:r.recovery.accepted_after_blackout,...r.runtime},null,2);
}
const job=el("job");
if(job){
  const id=job.dataset.job;
  async function poll(){try{
    const response=await fetch("/api/jobs/"+id);if(!response.ok)throw new Error("Result unavailable or expired");const data=await response.json();
    const choice=experiments[data.preset];el("preset-name").textContent=choice?choice.title:data.preset;el("explanation").textContent=choice?choice.description:"";
    el("status").textContent=data.state==="RUNNING"?"Computing on AWS"+(data.progress?" · "+data.progress.elapsed_s+" s simulated · "+(modeNames[data.progress.mode]||data.progress.mode):"")+"…":data.state;
    if(data.state==="COMPLETE")render(data,id);else if(data.state==="RUNNING")setTimeout(poll,1500);else{el("outcome").textContent="Run incomplete.";el("status").textContent=data.message||data.state;}
  }catch(error){el("status").textContent=error.message;}}
  poll();
}
