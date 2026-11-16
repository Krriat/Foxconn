"""Episode-disjoint LoRA vs disabled-adapter evaluation; no ROS operations."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from training_tools.task_brain.lora_data import load_image,samples,answer
from training_tools.task_brain.lora_model import load,processor
from training_tools.task_brain.scoring import classify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=("validation","test"),default="test")
    p.add_argument("--per-window",type=int,default=2)
    p.add_argument("--baseline",action="store_true")
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    proc=processor();model=load(adapter=args.adapter)
    rows=samples(args.data,args.split,args.per_window)
    rows += [dict(r,label=3,corruption="dark") for r in rows[::4]]
    results=[]
    for row in rows:
        image=load_image(row);result=classify(model,proc,image,row["stage"])
        output={"episode":row["episode"],"frame":row["frame"],"stage":row["stage"],
            "corruption":row.get("corruption"),"expected":answer(row),"adapter":result}
        if args.baseline:
            with model.disable_adapter():output["base"]=classify(model,proc,image,row["stage"])
        results.append(output)
        with (args.output/"predictions.jsonl").open("a") as f:f.write(json.dumps(output)+"\n")
        print(json.dumps(output),flush=True)
    report={"split":args.split,"episodes":sorted({r["episode"] for r in rows}),"cases":len(rows),
            "no_robot_commands":True,"limits":"Endpoint windows and synthetic unknowns; not a live success rate."}
    for key in ("adapter","base") if args.baseline else ("adapter",):
        correct=sum(r[key]["value"]==r["expected"] for r in results)
        false_done=[r for r in results if r[key]["value"]["status"]=="complete" and r["expected"]["status"]!="complete"]
        misses=[r for r in results if r[key]["value"]["status"]!="complete" and r["expected"]["status"]=="complete"]
        latency=[r[key]["latency_ms"] for r in results]
        report[key]={"exact_accuracy":correct/len(results),"correct":correct,
            "false_completion_count":len(false_done),"missed_completion_count":len(misses),
            "median_ms":float(np.median(latency)),"p95_ms":float(np.percentile(latency,95)),
            "max_ms":max(latency)}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
