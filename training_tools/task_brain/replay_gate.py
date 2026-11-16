"""Run the trained LoRA + actual temporal gate at 1Hz, without any ROS writes."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from training_tools.task_brain.decision import Gate,Evidence
from training_tools.task_brain.lora_data import RAW,load_image,mosaic
from training_tools.task_brain.lora_model import load,processor
from training_tools.task_brain.scoring import classify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    model=load(adapter=args.adapter);proc=processor();cases=[]
    # Before these reviewed frames the target has not completed opening.
    for ep,not_before in ((20,305),(35,257),(39,243),(47,188),(54,217),(55,224),(56,230),(57,208)):
        state=np.load(RAW/f"episode_{ep:06d}/trajectory.npz");n=len(state["state"])
        frames=list(range(0,n,20))
        if frames[-1]!=n-1:frames.append(n-1)
        observations=[{"episode":ep,"frame":f,"stage":"far" if ep%2==0 else "near",
                       "elapsed_seconds":f/20} for f in frames]
        cases.append((f"original_ep_{ep}",observations,not_before/20,True))
    for directory in sorted(args.evidence.glob("dense_live_*")):
        observations=json.loads((directory/"observations.json").read_text())
        success="failed" not in directory.name
        before=28. if directory.name.endswith("far") else 16. if success else float("inf")
        cases.append((directory.name,observations,before,success))
    reports=[]
    for name,rows,not_before,success in cases:
        context={"phase":"RUNNING","mode":"live","run_id":name,
                 "task_sequence":{"stage":rows[0]["stage"],"id":name,"result":"running","pending":None}}
        gate=Gate(max_gap=2.5);gate.observe_context(context,0.)
        decision=None;latencies=[];raw_completed=0
        for r in rows:
            image=(Image.fromarray(cv2.cvtColor(mosaic([cv2.imread(p) for p in r["images"]]),cv2.COLOR_BGR2RGB))
                   if "images" in r else load_image(r))
            result=classify(model,proc,image,r["stage"]);elapsed=max(0.,r["elapsed_seconds"])
            latency=result["latency_ms"]/1000;latencies.append(latency)
            raw_completed+=result["value"]["status"]=="complete"
            stamp=r.get("head_image_time_ns",int((elapsed+1)*1e9))
            evidence=Evidence(name,r["stage"],stamp,elapsed,result["value"])
            proposed=gate.update(evidence,context,now=elapsed+latency,context_age=0.)
            row={"case":name,"elapsed":elapsed,"value":result["value"],"margin":result.get("margin_nats_per_token"),
                 "latency_seconds":latency,"proposal":proposed,"gate_reason":gate.reason}
            with (args.output/"predictions.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            if proposed:
                gate.mark_dispatched(proposed);decision=elapsed+latency
                # A real transition stops the current stage here. Continue
                # reading to measure observation outputs, but never dispatch twice.
        passed=(decision is not None and decision>=not_before) if success else decision is None
        report={"case":name,"samples":len(rows),"expected_success":success,"not_before_seconds":
                    None if not success else not_before,"first_proposal_seconds":decision,
                "raw_completed_frames":raw_completed,"median_inference_ms":float(np.median(latencies))*1000,
                "passed":passed}
        reports.append(report);print(json.dumps(report),flush=True)
    output={"passed":all(r["passed"] for r in reports),"motion_commands":0,"ROS_service_calls":0,
            "sampling":"1Hz actual distinct images; same Qwen LoRA readout and temporal Gate as deployment",
            "cases":reports,"limits":"Development replay audit after error analysis, not an independent live success rate."}
    (args.output/"report.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output),flush=True)


if __name__=="__main__":main()
