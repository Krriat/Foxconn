#!/usr/bin/env python3
"""AGX Orin batch-one ACT latency, without any robot-control dependencies."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from jetson_inference_runtime import Engine


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def summarize(times):
    a = np.array(times)
    return dict(mean_ms=float(a.mean()), median_ms=float(np.median(a)),
        p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)),
        max_ms=float(a.max()), equivalent_hz=float(1000/a.mean()),
        over_50ms_count=int((a > 50).sum()), over_33_33ms_count=int((a > 1000/30).sum()),
        samples_ms=times)


def measure(fn, warmup, iterations):
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()
    times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(i)
        torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-start))
    return summarize(times), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--precisions', nargs='+', default=['fp32', 'fp16', 'bf16'])
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA required, no CPU fallback'
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((args.root / 'manifest.json').read_text())
    for name, digest in manifest['bundle_files_sha256'].items():
        assert hashlib.sha256((args.root / name).read_bytes()).hexdigest() == digest, name
    for stage, entry in manifest['models'].items():
        for name, digest in entry['files_sha256'].items():
            assert hashlib.sha256((args.root / stage / 'act' / name).read_bytes()).hexdigest() == digest
    samples = np.load(args.root / 'samples.npz')
    images, states = samples['images'], samples['states']
    count = len(images)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    report = dict(device_model=Path('/proc/device-tree/model').read_text().strip('\0'),
        platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
        cuda_capability=list(torch.cuda.get_device_capability(0)),
        power_mode=subprocess.check_output(['nvpmodel', '-q'], text=True).strip(),
        power_or_clocks_changed_by_benchmark_script=False,
        gpu_clock_hz={name: int((Path('/sys/class/devfreq/17000000.gpu') / name).read_text())
                      for name in ('cur_freq', 'min_freq', 'max_freq')},
        gpu_tpc_pg_mask=Path('/sys/devices/platform/gpu.0/tpc_pg_mask').read_text().strip(),
        cpu_threads=args.threads, batch_size=1,
        action_chunk_shape=[1, 50, 16], image_shape=[3, 480, 640, 3],
        warmup=args.warmup, iterations=args.iterations, fixtures=count,
        timing='Synchronized wall clock. forward uses prepared GPU tensors. full includes CPU RGB uint8 preprocessing, H2D, forward, denormalization, D2H; excludes camera/network/JPEG decode and robot control.',
        fp32_uses_tf32=torch.backends.cuda.matmul.allow_tf32,
        manifest=manifest, results={})
    with torch.inference_mode():
        for stage in ('far', 'near'):
            engine = Engine(args.root / stage, 'act')
            prepared = [engine.prepare(views, state) for views, state in zip(images, states)]
            fp32 = np.concatenate([engine.postprocess(engine.forward(p)) for p in prepared])
            reference = np.load(args.root / f'{stage}_reference_fp32.npy')
            reference_error = np.abs(fp32-reference)
            assert float(reference_error[..., :14].max()) < .002, 'FP32 reference disagrees with training machine'
            for precision in args.precisions:
                key = f'{stage}_{precision}'
                try:
                    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
                        report['results'][key] = dict(status='unsupported')
                        continue
                    def context():
                        return (nullcontext() if precision == 'fp32' else
                            torch.autocast('cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16))
                    def forward(i):
                        with context():
                            return engine.forward(prepared[i % count])
                    def full(i):
                        k = i % count
                        batch = engine.prepare(images[k], states[k])
                        with context():
                            output = engine.forward(batch)
                        return engine.postprocess(output)
                    torch.cuda.reset_peak_memory_stats()
                    forward_times, out = measure(forward, args.warmup, args.iterations)
                    full_times, native = measure(full, args.warmup, args.iterations)
                    outputs = np.concatenate([full(i) for i in range(count)])
                    assert tuple(out.shape) == (1, 50, 16) and np.isfinite(outputs).all()
                    joint_error = np.abs(outputs[..., :14]-fp32[..., :14])
                    suction_disagreement = float(((outputs[..., 14:] >= .5) != (fp32[..., 14:] >= .5)).mean())
                    report['results'][key] = dict(status='passed', forward=forward_times, full=full_times,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                        fp32_max_joint_difference_from_training_machine_rad=float(reference_error[..., :14].max()),
                        precision_max_joint_difference_from_fp32_deg=float(np.rad2deg(joint_error.max())),
                        precision_mean_joint_difference_from_fp32_deg=float(np.rad2deg(joint_error.mean())),
                        suction_threshold_disagreement_from_fp32_fraction=suction_disagreement)
                    np.save(args.output.parent / f'{key}_outputs.npy', outputs)
                    print(key, json.dumps({name: report['results'][key][name]['mean_ms'] for name in ('forward', 'full')}), flush=True)
                except Exception as error:
                    report['results'][key] = dict(status='error', error=repr(error), traceback=traceback.format_exc())
                    print(key, repr(error), flush=True)
                write(args.output, report)
            del engine, prepared, fp32
            gc.collect()
            torch.cuda.empty_cache()
    write(args.output, report)
    return 0 if all(r['status'] == 'passed' for r in report['results'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
