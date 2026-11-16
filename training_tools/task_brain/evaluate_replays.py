"""Untouched original trajectories and archived live runs, without any actions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from training_tools.task_brain.lora_data import load_image,answer,mosaic
from training_tools.task_brain.lora_model import load,processor
from training_tools.task_brain.scoring import classify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--baseline",action="store_true")
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    model=load(adapter=args.adapter);proc=processor();rows=[]
    for line in (args.evidence/"reviewed_test_trajectory.jsonl").read_text().splitlines():
        r=json.loads(line);r["source"]="original_heldout_trajectory";rows.append(r)
    for directory in sorted(args.evidence.glob("live_replay_*")):
        if not (directory/"labels.json").exists():continue
        labels=json.loads((directory/"labels.json").read_text())
        for r,label in zip(json.loads((directory/"observations.json").read_text()),labels):
            rows.append(dict(r,label=label,source=directory.name))
    results=[]
    for row in rows:
        image=(Image.fromarray(cv2.cvtColor(mosaic([cv2.imread(p) for p in row["images"]]),cv2.COLOR_BGR2RGB))
               if "images" in row else load_image(row))
        result=classify(model,proc,image,row["stage"])
        output={k:v for k,v in row.items() if k!="images"}
        output.update(expected=answer(row),adapter=result)
        if args.baseline:
            with model.disable_adapter():output["base"]=classify(model,proc,image,row["stage"])
        results.append(output)
        with (args.output/"predictions.jsonl").open("a") as f:f.write(json.dumps(output)+"\n")
        print(json.dumps(output),flush=True)
    report={"no_robot_commands":True,"source_counts":{},"groups":{}}
    for group in sorted({r["source"] for r in results}):
        selected=[r for r in results if r["source"]==group];report["source_counts"][group]=len(selected)
        report["groups"][group]={}
        for method in (("adapter","base") if args.baseline else ("adapter",)):
            false=[r for r in selected if r[method]["value"]["status"]=="complete" and r["expected"]["status"]!="complete"]
            misses=[r for r in selected if r[method]["value"]["status"]!="complete" and r["expected"]["status"]=="complete"]
            report["groups"][group][method]={"exact_correct":sum(r[method]["value"]==r["expected"] for r in selected),
                "cases":len(selected),"false_completion_count":len(false),"missed_completion_count":len(misses),
                "false_cases":[{k:r.get(k) for k in ("episode","frame","index","elapsed_seconds")} for r in false],
                "median_ms":float(np.median([r[method]["latency_ms"] for r in selected]))}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
