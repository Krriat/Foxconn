#!/usr/bin/env python3
"""Compare batch/lr candidates with the same examples and validation samples."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
BASE = WORKSPACE / "AIDE_ACT_dataset_20261111_195100"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-batch", type=int)
    args = parser.parse_args()
    selections_path = WORKSPACE / "training_verification/benchmarks/hardware_selections.json"
    while not selections_path.exists():
        time.sleep(5)
    choices = json.loads(selections_path.read_text())
    env = os.environ.copy()
    env.update(PYTHONPATH=f"{BASE}/act/xcore_act/src:{BASE}/act/xcroe_interACT/src",
               PYTHONNOUSERSITE="1", TORCH_HOME=str(WORKSPACE / ".cache/torch"),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    destination = WORKSPACE / "training_verification/parameter_probes"
    destination.mkdir(exist_ok=True)
    results_path = destination / "all_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    for model in ("act", "interact"):
        hardware = choices[model]
        batches = {(8, 1), (hardware["batch_size"], hardware["grad_accum"])}
        if args.extra_batch:
            batches.add((args.extra_batch, 1))
        rates = (1e-5, 5e-5) if model == "act" else (5e-5, 1e-4)
        project, script = (("xcore_act", "train_act.py") if model == "act"
                           else ("xcroe_interACT", "train_interact3d.py"))
        for batch, accum in sorted(batches):
            for lr in rates:
                if any(r["model"] == model and r["batch_size"] == batch
                       and r["grad_accum"] == accum and r["learning_rate"] == lr
                       and r["returncode"] == 0 for r in results):
                    continue
                steps = 6144 // (batch*accum)
                tag = f"{model}_b{batch}_a{accum}_lr{lr:g}"
                output = destination / tag
                command = [sys.executable, str(BASE / "act" / project / "scripts/training" / script),
                           "--dataset-root", str(WORKSPACE / "prepared_data/suction_20hz_v1/train"),
                           "--dataset-repo-id", "local/dual_arm_suction_train",
                           "--validation-root", str(WORKSPACE / "prepared_data/suction_20hz_v1/val"),
                           "--output-dir", str(output), "--batch-size", str(batch),
                           "--steps", str(steps), "--save-freq", str(steps),
                           "--eval-freq", str(steps), "--eval-batches", str(384//batch),
                           "--log-freq", "25", "--num-workers", str(hardware["num_workers"]),
                           "--mixed-precision", "bf16", "--learning-rate", str(lr),
                           "--backbone-learning-rate", "1e-5", "--weight-decay", "1e-4", "--seed", "1000"]
                if model == "interact":
                    command += ["--grad-accum", str(accum)]
                started = time.monotonic()
                with output.with_suffix(".log").open("w") as log:
                    completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                record = dict(model=model, batch_size=batch, grad_accum=accum, learning_rate=lr,
                              num_workers=hardware["num_workers"], mixed_precision="bf16",
                              steps=steps, training_examples=steps*batch*accum,
                              validation_frames=384, output_dir=str(output),
                              seconds=time.monotonic()-started, returncode=completed.returncode)
                if completed.returncode == 0:
                    validation = [json.loads(line) for line in (output/"validation.jsonl").read_text().splitlines()][-1]
                    metrics = [json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()]
                    record.update(validation=validation, last_metrics=metrics[-1])
                results.append(record)
                (destination/"all_results.json").write_text(json.dumps(results, indent=2)+"\n")
                print(json.dumps(record), flush=True)
                if completed.returncode:
                    raise RuntimeError(f"Probe failed; inspect {output.with_suffix('.log')}")
    print("PARAMETER PROBES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
