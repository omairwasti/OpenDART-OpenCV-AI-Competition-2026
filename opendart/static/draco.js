"use strict";
const byId=id=>document.getElementById(id);
const stateNames={SEARCH_UNRESOLVED:"Target not yet resolved",TRACKING:"Target tracking",RESOLVED_BODY:"Resolved target",PARTIAL_FRAME:"Partial transmission",LOST:"Target not selected",REACQUIRING:"Looking for the target again",REACQUIRED:"Target reacquired",COAST:"Coasting"};
let overlayUrl=null;
byId("draco-form").addEventListener("submit",async event=>{
  event.preventDefault();const button=byId("analyze-draco");button.disabled=true;
  byId("draco-status").textContent="Analysing the original image on AWS…";byId("draco-result").hidden=true;
  try{
    const response=await fetch("/api/draco/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image_id:byId("draco-image").value})});
    const data=await response.json();if(!response.ok)throw Error(data.error||"No analysis result is available");
    byId("draco-status").textContent="Analysis complete. A new result was computed on the server.";
    byId("draco-result-title").textContent=stateNames[data.state]||"Perception result";
    byId("draco-metrics").replaceChildren();
    const values=[["Processing time",data.server_processing_ms.toFixed(1)+" ms"],["OpenCV / platform",data.opencv+" / "+data.platform],["Before impact",data.time_to_impact_s.toFixed(3)+" s"]];
    for(const [label,value]of values){const box=document.createElement("div"),name=document.createElement("span"),text=document.createElement("strong");name.textContent=label;text.textContent=value;box.append(name,text);byId("draco-metrics").append(box);}
    byId("draco-original").src="/draco/images/"+data.image_id+".png";
    if(overlayUrl)URL.revokeObjectURL(overlayUrl);
    const bytes=Uint8Array.from(atob(data.overlay_png_base64),c=>c.charCodeAt(0));overlayUrl=URL.createObjectURL(new Blob([bytes],{type:"image/png"}));byId("draco-overlay").src=overlayUrl;
    byId("draco-caption").textContent=data.is_partial?"Received strip only. The missing full-body extent is unknown.":data.resolved_body?"Visible terrain region. This is not a physical centre or a range measurement.":data.selection_available?"Selected target hypothesis. Semantic identity remains unverified.":"No target selected. Candidate components alone do not establish identity.";
    const {overlay_png_base64,...provenance}=data;byId("draco-provenance").textContent=JSON.stringify(provenance,null,2);byId("draco-result").hidden=false;
  }catch(error){byId("draco-status").textContent=error.message;}
  finally{button.disabled=false;}
});
