#!/usr/bin/env python3
"""Measure complete optimizer updates on real LeRobot batches in fresh processes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
BASE = WORKSPACE / "AIDE_ACT_dataset_20261111_195100"
for project in ("xcore_act", "xcroe_interACT"):
    sys.path.insert(0, str(BASE / "act" / project / "src"))


def load_trainer(model):
    project, name = ("xcore_act", "train_act") if model == "act" else ("xcroe_interACT", "train_interact3d")
    path = BASE / "act" / project / "scripts/training" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("act", "interact"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    try:
        run(args, report)
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status="out_of_memory", error=str(exc))
    except Exception as exc:
        import traceback
        report.update(status="error", error=repr(exc), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


def run(args, report):
    random.seed(1000); np.random.seed(1000); torch.manual_seed(1000); torch.cuda.manual_seed_all(1000)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    trainer = load_trainer(args.model)
    train_argv = ["train", "--dataset-root", str(args.dataset_root),
                  "--dataset-repo-id", "local/dual_arm_suction_train",
                  "--output-dir", str(args.output.parent / "unused"),
                  "--batch-size", str(args.batch_size), "--steps", "10"]
    previous_argv = sys.argv
    sys.argv = train_argv
    config_args = trainer.parse_args()
    sys.argv = previous_argv
    config_args.mixed_precision = args.mixed_precision
    if args.learning_rate is not None:
        config_args.learning_rate = args.learning_rate
    if args.model == "act":
        config = trainer.make_policy_config(config_args, None)
        cfg = trainer.make_train_config(config_args, args.output.parent / "unused", config, None)
        dataset = trainer.make_dataset(cfg)
        model = trainer.make_policy(cfg=config, ds_meta=dataset.meta).cuda()
        preprocessor, _ = trainer.make_pre_post_processors(policy_cfg=config, dataset_stats=dataset.meta.stats)
        optimizer, _ = trainer.make_optimizer_and_scheduler(cfg, model)
    else:
        stats = trainer.compute_normalization_stats(args.dataset_root)
        info = json.loads((args.dataset_root / "meta/info.json").read_text())
        config = trainer.InterACT3DConfig(state_layout=trainer.infer_state_layout(info))
        model = trainer.InterACT3DPolicy(config).cuda()
        dataset = trainer.InterACTWindowDataset(
            args.dataset_root, "local/dual_arm_suction_train", chunk_size=50,
            image_size=224, stats=stats, state_layout=config.state_layout)
        optimizer = trainer.make_optimizer(model, config_args)
        preprocessor = lambda batch: {key: value.cuda(non_blocking=True) for key, value in batch.items()}
    report.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  parameters=sum(p.numel() for p in model.parameters()), frames=len(dataset),
                  optimizer_groups=[dict(lr=g['lr'], weight_decay=g['weight_decay'],
                                         parameters=sum(p.numel() for p in g['params']))
                                    for g in optimizer.param_groups])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers>0,
        prefetch_factor=2 if args.num_workers else None, drop_last=True)
    iterator = iter(loader)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision == "fp16")
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    model.train()
    durations, losses, skips, io_times = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(args.warmup + args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        io_seconds = 0.
        for micro in range(args.grad_accum):
            io_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); batch = next(iterator)
            batch = preprocessor(batch)
            io_seconds += time.perf_counter()-io_start
            with torch.autocast("cuda", enabled=args.mixed_precision != "no", dtype=dtype):
                if args.model == "act":
                    loss, _ = model(batch)
                else:
                    loss = model(images=batch["images"], state=batch["state"],
                                 actions=batch["action"], action_is_pad=batch["action_is_pad"])["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss_value += float(loss.detach()) / args.grad_accum
            scaler.scale(loss/args.grad_accum).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        scale_before = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        skipped = scaler.get_scale() < scale_before
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if iteration >= args.warmup:
            durations.append(elapsed); losses.append(loss_value); skips.append(skipped); io_times.append(io_seconds)
        print(f"{args.model} batch={args.batch_size} accum={args.grad_accum} {args.mixed_precision} "
              f"iteration={iteration} loss={loss_value:.5f} grad={float(grad_norm):.3f} "
              f"seconds={elapsed:.3f} skipped={skipped}", flush=True)
    finite = all(torch.isfinite(p).all() for p in model.parameters())
    if not finite or sum(skips):
        raise FloatingPointError("non-finite parameters or skipped measured optimizer updates")
    report.update(status="passed", mean_update_seconds=statistics.mean(durations),
                  median_update_seconds=statistics.median(durations),
                  samples_per_second=args.batch_size*args.grad_accum/statistics.mean(durations),
                  update_seconds=durations,
                  mean_data_seconds=statistics.mean(io_times), losses=losses,
                  skipped_measured_updates=sum(skips),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)


if __name__ == "__main__":
    raise SystemExit(main())
