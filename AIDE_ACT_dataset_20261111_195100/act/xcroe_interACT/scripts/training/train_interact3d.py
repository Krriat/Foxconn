#!/usr/bin/env python3
"""Standalone RGB InterACT trainer with optional head point-cloud input."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from xcore_interact.data import (
    InterACTWindowDataset,
    NormalizationStats,
    compute_normalization_stats,
)
from xcore_interact.policy import InterACT3DConfig, InterACT3DPolicy
from xcore_interact.schema import infer_state_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="local/xcore_bimanual_interact3d")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--validation-repo-id", default="local/dual_arm_suction_val")
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=0,
                        help="0 evaluates all frames; otherwise sample evenly across validation")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--use-point-cloud", action="store_true",
                        help="Enable the optional head XYZRGB branch; default uses RGB only.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--point-count", type=int, default=1024)
    parser.add_argument("--point-tokens", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--encoder-blocks", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--left-gripper-disabled",
        action="store_true",
        help="Only for legacy datasets recorded before the left gripper was installed.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not (args.dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(args.dataset_root / "meta/info.json")
    if args.validation_root is not None:
        args.validation_root = args.validation_root.expanduser().resolve()
        if args.validation_root == args.dataset_root:
            raise ValueError("training and validation roots must differ")
        if not (args.validation_root / "meta/info.json").is_file():
            raise FileNotFoundError(args.validation_root / "meta/info.json")
    if args.eval_freq < 1 or args.eval_batches < 0:
        raise ValueError("eval-freq must be positive; eval-batches must be nonnegative")
    for name in (
        "steps",
        "save_freq",
        "log_freq",
        "batch_size",
        "grad_accum",
        "chunk_size",
        "point_count",
        "point_tokens",
        "d_model",
        "encoder_blocks",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.resume_from is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")


def setup_logging(output: Path) -> logging.Logger:
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_interact3d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_checkpoint(
    output: Path,
    step: int,
    model: InterACT3DPolicy,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    stats: NormalizationStats,
    args: argparse.Namespace,
) -> Path:
    final = output / "checkpoints" / f"{step:06d}"
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{step:06d}-", dir=final.parent))
    try:
        pretrained = temporary / "pretrained_model"
        model.save_pretrained(pretrained)
        stats.save(pretrained / "normalization.json")
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            temporary / "training_state.pt",
        )
        (temporary / "launch.json").write_text(
            json.dumps(
                {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                indent=2,
            )
            + "\n"
        )
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    latest = output / "checkpoints" / "last"
    temporary_link = output / "checkpoints" / ".last.tmp"
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(final.name, target_is_directory=True)
    os.replace(temporary_link, latest)
    return final


def make_optimizer(model: InterACT3DPolicy, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_ids = {id(parameter) for parameter in model.rgb.features.parameters()}
    backbone = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids and parameter.requires_grad]
    other = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids and parameter.requires_grad]
    groups = [{"params": other, "lr": args.learning_rate}]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


@torch.no_grad()
def evaluate_validation(model, loader, amp_enabled, max_batches=0, amp_dtype=torch.float16):
    model.eval()
    per_dim = torch.zeros(16, device="cuda")
    count = 0
    for index, batch in enumerate(loader):
        batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            prediction = model.predict_action_chunk(
                batch["images"], points=batch.get("points"), state=batch["state"])
        valid = (~batch["action_is_pad"]).unsqueeze(-1)
        per_dim += ((prediction.float() - batch["action"]).abs() * valid).sum(dim=(0, 1))
        count += int(valid.sum())
        if max_batches and index + 1 >= max_batches:
            break
    if not count:
        raise ValueError("validation has no valid action targets")
    per_dim /= count
    if not torch.isfinite(per_dim).all():
        raise FloatingPointError("non-finite validation predictions")
    return {"val_l1": float(per_dim.mean()), "val_l1_per_dimension": per_dim.cpu().tolist(),
            "val_action_targets": count}


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    logger = setup_logging(args.output_dir)

    resume = args.resume_from.expanduser().resolve() if args.resume_from else None
    if resume:
        pretrained = resume / "pretrained_model"
        model = InterACT3DPolicy.from_pretrained(pretrained)
        stats = NormalizationStats.load(pretrained / "normalization.json")
        if model.config.chunk_size != args.chunk_size:
            raise ValueError("--chunk-size must match the resumed model")
    else:
        stats = compute_normalization_stats(args.dataset_root)
        info = json.loads((args.dataset_root / "meta/info.json").read_text())
        config = InterACT3DConfig(
            use_point_cloud=args.use_point_cloud,
            state_layout=infer_state_layout(info),
            chunk_size=args.chunk_size,
            image_size=args.image_size,
            point_count=args.point_count,
            point_tokens=args.point_tokens,
            d_model=args.d_model,
            encoder_blocks=args.encoder_blocks,
            pretrained_backbone=not args.no_pretrained_backbone,
            freeze_backbone=args.freeze_backbone,
            left_gripper_enabled=not args.left_gripper_disabled,
        )
        model = InterACT3DPolicy(config)
    model.to(device)
    dataset = InterACTWindowDataset(
        args.dataset_root,
        args.dataset_repo_id,
        chunk_size=model.config.chunk_size,
        image_size=model.config.image_size,
        stats=stats,
        use_point_cloud=model.config.use_point_cloud,
        state_layout=model.config.state_layout,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )
    validation_loader = None
    if args.validation_root is not None:
        validation_dataset = InterACTWindowDataset(
            args.validation_root, args.validation_repo_id,
            chunk_size=model.config.chunk_size, image_size=model.config.image_size,
            stats=stats, use_point_cloud=model.config.use_point_cloud,
            state_layout=model.config.state_layout)
        if args.eval_batches and args.eval_batches*args.batch_size < len(validation_dataset):
            indices = torch.linspace(0, len(validation_dataset)-1,
                                     args.eval_batches*args.batch_size).long().tolist()
            validation_dataset = torch.utils.data.Subset(validation_dataset, indices)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers, pin_memory=True,
                                       persistent_workers=args.num_workers > 0,
                                       prefetch_factor=2 if args.num_workers > 0 else None)
    optimizer = make_optimizer(model, args)
    if args.no_amp:
        args.mixed_precision = "no"
    amp_enabled = args.mixed_precision != "no"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision == "fp16")
    step = 0
    if resume:
        training_state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        step = int(training_state["step"])
        optimizer.load_state_dict(training_state["optimizer"])
        scaler.load_state_dict(training_state["scaler"])
        torch.set_rng_state(training_state["torch_rng"])
        torch.cuda.set_rng_state_all(training_state["cuda_rng"])
    if step >= args.steps:
        raise ValueError(f"checkpoint step {step} is already >= requested {args.steps}")

    parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "%s layout=%s device=%s gpu=%s params=%d frames=%d batch=%d accumulation=%d",
        model.config.model_type,
        model.config.state_layout,
        device,
        torch.cuda.get_device_name(0),
        parameters,
        len(dataset),
        args.batch_size,
        args.grad_accum,
    )
    iterator = iter(loader)
    metrics_path = args.output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    window_loss = 0.0
    window_count = 0
    last_log = time.monotonic()
    best_record_path = args.output_dir / "best_validation.json"
    validation_signature = dict(root=str(args.validation_root), eval_batches=args.eval_batches,
                                sampled_batch_size=args.batch_size if args.eval_batches else None)
    previous_best = json.loads(best_record_path.read_text()) if best_record_path.exists() else {}
    best_score = (previous_best["val_l1"]
                  if previous_best.get("validation_signature") == validation_signature else float("inf"))
    while step < args.steps:
        is_best = False
        model.train()
        accumulated_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                output = model(
                    images=batch["images"],
                    points=batch.get("points"),
                    state=batch["state"],
                    actions=batch["action"],
                    action_is_pad=batch["action_is_pad"],
                )
                loss = output["loss"] / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss before step {step+1}")
            scaler.scale(loss).backward()
            accumulated_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_skipped = scaler.get_scale() < scale_before
        optimizer.zero_grad(set_to_none=True)
        step += 1
        window_loss += accumulated_loss
        window_count += 1

        if step % args.log_freq == 0:
            elapsed = time.monotonic() - last_log
            record = {
                "step": step,
                "loss": window_loss / window_count,
                "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
                "optimizer_step_skipped": optimizer_step_skipped,
                "steps_per_s": window_count / max(elapsed, 1e-6),
                "cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "backbone_learning_rate": (optimizer.param_groups[1]["lr"] if len(optimizer.param_groups)>1 else None),
                "amp_scale": scaler.get_scale(),
            }
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            logger.info("step=%d loss=%.5f grad=%.3f peak_cuda=%.2fGiB", step, record["loss"], grad_norm, record["cuda_allocated_gib"])
            window_loss = 0.0
            window_count = 0
            last_log = time.monotonic()
        if validation_loader is not None and (step % args.eval_freq == 0 or step == args.steps):
            validation = evaluate_validation(model, validation_loader, amp_enabled, args.eval_batches, amp_dtype)
            with (args.output_dir / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=step, **validation)) + "\n")
            logger.info("validation step=%d normalized_l1=%.6f", step, validation["val_l1"])
            is_best = validation["val_l1"] < best_score
        if step % args.save_freq == 0 or step == args.steps or is_best:
            path = save_checkpoint(args.output_dir, step, model, optimizer, scaler, stats, args)
            logger.info("saved %s", path)
            if is_best:
                best_score = validation["val_l1"]
                best_record_path.write_text(json.dumps(dict(
                    step=step, checkpoint=str(path),
                    validation_signature=validation_signature, **validation), indent=2)+"\n")
                link = path.parent / ".best.tmp"
                link.unlink(missing_ok=True)
                link.symlink_to(path.name, target_is_directory=True)
                os.replace(link, path.parent / "best")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
