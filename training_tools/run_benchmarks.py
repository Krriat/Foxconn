#!/usr/bin/env python3
"""Sequential GPU benchmark matrix; no full training jobs are launched."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
BASE = WORKSPACE / "AIDE_ACT_dataset_20261111_195100"


def main():
    data = WORKSPACE / "prepared_data/suction_20hz_v1"
    manifest = data / "split_manifest.json"
    while not manifest.exists() or not json.loads(manifest.read_text()).get("complete"):
        time.sleep(5)
    env = os.environ.copy()
    env.update(PYTHONPATH=f"{BASE}/act/xcore_act/src:{BASE}/act/xcroe_interACT/src",
               PYTHONNOUSERSITE="1", TORCH_HOME=str(WORKSPACE / ".cache/torch"),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    destination = WORKSPACE / "training_verification/benchmarks"
    destination.mkdir(exist_ok=True)
    cases = [("act", batch, 1, "bf16", 2) for batch in (2, 4, 8, 16, 24)]
    cases += [("interact", b, a, "bf16", 2) for b, a in ((2, 4), (4, 2), (8, 1), (16, 1), (32, 1), (64, 1))]
    cases += [(model, 8, 1, "fp16", 2) for model in ("act", "interact")]
    results = []

    def run_case(model, batch, accum, precision, workers):
        tag = f"{model}_b{batch}_a{accum}_{precision}_w{workers}"
        path = destination / f"{tag}.json"
        command = [sys.executable, str(WORKSPACE / "training_tools/benchmark.py"),
                   "--model", model, "--dataset-root", str(data / "train"),
                   "--batch-size", str(batch), "--grad-accum", str(accum),
                   "--mixed-precision", precision, "--num-workers", str(workers),
                   "--warmup", "5", "--iterations", "8", "--output", str(path)]
        with path.with_suffix(".log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = json.loads(path.read_text())
        results.append(record)
        print(json.dumps({k: record.get(k) for k in (
            "model", "batch_size", "grad_accum", "mixed_precision", "num_workers",
            "status", "samples_per_second", "peak_reserved_gib", "error")}), flush=True)
        (destination / "all_results.json").write_text(json.dumps(results, indent=2)+"\n")
        if record["status"] == "error":
            raise RuntimeError(record.get("traceback", record["error"]))

    for case in cases:
        run_case(*case)
    selections = {}
    for model in ("act", "interact"):
        candidates = [r for r in results if r["model"] == model and r["status"] == "passed"
                      and r["mixed_precision"] == "bf16" and r["peak_reserved_gib"] <= 12.5]
        peak = max(r["samples_per_second"] for r in candidates)
        near_peak = [r for r in candidates if r["samples_per_second"] >= peak*.95]
        best = min(near_peak, key=lambda r: r["batch_size"])
        for workers in (0, 1, 4):
            run_case(model, best["batch_size"], best["grad_accum"], "bf16", workers)
        workers_results = [r for r in results if r["model"] == model and r["status"] == "passed"
                           and r["batch_size"] == best["batch_size"]
                           and r["grad_accum"] == best["grad_accum"] and r["mixed_precision"] == "bf16"]
        best = max(workers_results, key=lambda r: r["samples_per_second"])
        selections[model] = best
    (destination / "hardware_selections.json").write_text(json.dumps(selections, indent=2)+"\n")
    print("BENCHMARK MATRIX COMPLETE", flush=True)


if __name__ == "__main__":
    main()
