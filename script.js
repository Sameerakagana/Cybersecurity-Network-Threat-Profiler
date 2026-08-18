async function checkStatus(){
    try{
        const r=await fetch("/api/status");
        const data=await r.json();
        const el=document.getElementById("serverStatus");
        el.textContent=data.models_ready ? "System ready" : "Model training required";
    }catch(e){
        document.getElementById("serverStatus").textContent="Server unavailable";
    }
}

async function analyze(){
    const btn=document.getElementById("analyzeBtn");
    btn.disabled=true;
    btn.textContent="ANALYZING...";

    const data={
        duration:Number(document.getElementById("duration").value),
        protocol_type:document.getElementById("protocol_type").value,
        src_bytes:Number(document.getElementById("src_bytes").value),
        dst_bytes:Number(document.getElementById("dst_bytes").value)
    };

    try{
        const response=await fetch("/api/analyze",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(data)
        });
        const result=await response.json();

        document.getElementById("result").classList.remove("hidden");

        if(result.error){
            document.getElementById("resultTitle").textContent="Analysis could not be completed";
            document.getElementById("classification").textContent="Error";
            document.getElementById("anomaly").textContent=result.error;
            document.getElementById("score").textContent="-";
            document.getElementById("risk").textContent="-";
            document.getElementById("riskBadge").textContent="ERROR";
            return;
        }

        document.getElementById("resultTitle").textContent="Traffic analyzed";
        document.getElementById("classification").textContent=result.classification;
        document.getElementById("anomaly").textContent=result.anomaly_status;
        document.getElementById("score").textContent=result.anomaly_score;
        document.getElementById("risk").textContent=result.risk;

        const badge=document.getElementById("riskBadge");
        badge.textContent=result.risk;
        badge.className="risk-badge "+result.risk.toLowerCase();

        document.getElementById("result").scrollIntoView({behavior:"smooth",block:"center"});
    }catch(e){
        alert("Could not connect to the FastAPI server. Make sure uvicorn is running.");
    }finally{
        btn.disabled=false;
        btn.textContent="RUN THREAT ANALYSIS →";
    }
}

document.getElementById("analyzeBtn").addEventListener("click",analyze);
checkStatus();
