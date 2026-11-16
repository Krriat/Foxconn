#!/usr/bin/env python3
"""Batch-one GPU latency on actual checkpoints; never connects to robot control."""
import argparse
from contextlib import nullcontext
import gc
import json
import platform
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from jetson_inference_runtime import Engine


def summary(values):
    a = np.asarray(values)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max()), "over_50ms_count": int((a > 50).sum())}


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return summary(times), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--models", nargs="+", default=["act", "interact"])
    parser.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "bf16"])
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required; CPU fallback is prohibited"
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    sample = np.load(args.root / "sample.npz")
    images = [x.copy() for x in sample["images"]]
    state = sample["state"].copy()
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "platform": platform.platform(), "batch_size": 1, "action_chunk_shape": [1,50,16],
              "warmup": args.warmup, "iterations": args.iterations, "cpu_threads": 4,
              "timing": "wall clock with CUDA synchronization; full includes CPU uint8 RGB preprocessing, H2D, forward, denormalization, D2H; excludes image receipt/JPEG decode/robot control",
              "manifest": json.loads((args.root / "manifest.json").read_text()), "results": {}}
    with torch.inference_mode():
        for name in args.models:
            engine = Engine(args.root, name)
            prepared = engine.prepare(images, state)
            fp32 = engine.postprocess(engine.forward(prepared))
            reference = np.load(args.root / (name + "_reference_fp32.npy"))
            reference_error = float(np.max(np.abs(fp32 - reference)))
            for precision in args.precisions:
                key = name + "_" + precision
                try:
                    if precision == "bf16" and not torch.cuda.is_bf16_supported():
                        report["results"][key] = {"status": "unsupported_bf16"}
                        continue
                    def context():
                        return nullcontext() if precision == "fp32" else torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
                    def forward():
                        with context():
                            return engine.forward(prepared)
                    def full():
                        batch = engine.prepare(images, state)
                        with context():
                            result = engine.forward(batch)
                        return engine.postprocess(result)
                    torch.cuda.reset_peak_memory_stats()
                    forward_stats, out = measure(forward, args.warmup, args.iterations)
                    full_stats, native = measure(full, args.warmup, args.iterations)
                    assert tuple(out.shape) == (1,50,16) and np.isfinite(native).all()
                    report["results"][key] = {"status":"passed", "forward": forward_stats, "full": full_stats,
                         "peak_reserved_gib": torch.cuda.max_memory_reserved()/1024**3,
                         "fp32_max_abs_difference_from_training_machine": reference_error,
                         "native_max_abs_difference_from_fp32": float(np.max(np.abs(native-fp32))),
                         "joint_max_abs_difference_from_fp32_rad": float(np.max(np.abs(native[:,:,:14]-fp32[:,:,:14])))}
                    np.save(args.output.with_name(key+"_prediction.npy"), native)
                except Exception as exc:
                    report["results"][key] = {"status":"error", "error":str(exc),"traceback":traceback.format_exc()}
                args.output.write_text(json.dumps(report, indent=2)+"\n")
                print(key,json.dumps(report["results"].get(key),ensure_ascii=False),flush=True)
            del engine, prepared, fp32
            gc.collect()
            torch.cuda.empty_cache()
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    return 0 if all(x["status"] == "passed" for x in report["results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
