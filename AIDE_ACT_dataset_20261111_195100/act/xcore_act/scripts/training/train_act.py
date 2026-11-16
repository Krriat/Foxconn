#!/usr/bin/env python3
"""Standalone ACT trainer for xCore LeRobot datasets, including dual-arm suction.

This script owns the training loop and does not call or import
``lerobot.scripts.lerobot_train``. It reuses LeRobot's dataset, ACT policy,
normalization processors, optimizer configuration, and checkpoint serializer
so saved ``pretrained_model`` directories remain compatible with inference.
"""

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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)


DEFAULT_DATASET_ROOT = Path(
    "/home/niic/aide/assets/dataset/xcroe_ACT/datasets/green-box-819_clean_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/niic/aide/assets/model/xcroe_ACT/model_outputs/act_custom"
)
AMP_SCALER_STATE_FILE = "amp_scaler_state.json"


@dataclass
class WindowMetrics:
    count: int = 0
    loss: float = 0.0
    l1_loss: float = 0.0
    kld_loss: float = 0.0
    grad_norm: float = 0.0
    data_s: float = 0.0
    update_s: float = 0.0

    def update(
        self,
        loss: float,
        output: dict[str, Any],
        grad_norm: float,
        data_s: float,
        update_s: float,
    ) -> None:
        self.count += 1
        self.loss += loss
        self.l1_loss += float(output.get("l1_loss", 0.0))
        self.kld_loss += float(output.get("kld_loss", 0.0))
        self.grad_norm += grad_norm
        self.data_s += data_s
        self.update_s += update_s

    def means(self) -> dict[str, float]:
        denominator = max(self.count, 1)
        return {
            "loss": self.loss / denominator,
            "l1_loss": self.l1_loss / denominator,
            "kld_loss": self.kld_loss / denominator,
            "grad_norm": self.grad_norm / denominator,
            "data_s": self.data_s / denominator,
            "update_s": self.update_s / denominator,
        }

    def reset(self) -> None:
        for name in asdict(self):
            setattr(self, name, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ACT without invoking the lerobot-train entrypoint."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default="local/pickup")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--job-name", default="pickup_act_custom")
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--validation-repo-id", default="local/dual_arm_suction_val")
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=0,
                        help="0 evaluates all frames; otherwise sample evenly across validation")

    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument(
        "--pretrained-backbone-weights",
        default="ResNet18_Weights.IMAGENET1K_V1",
        help="Use 'none' to initialize the ResNet18 backbone randomly.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Checkpoint directory containing pretrained_model/ and training_state/.",
    )
    return parser.parse_args()


def normalized_backbone_weights(value: str) -> str | None:
    return None if value.strip().lower() in {"none", "null"} else value


def validate_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_args(args: argparse.Namespace) -> None:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if not (args.dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot dataset metadata not found: {args.dataset_root / 'meta/info.json'}"
        )
    if args.validation_root is not None:
        args.validation_root = args.validation_root.expanduser().resolve()
        if args.validation_root == args.dataset_root:
            raise ValueError("training and validation roots must differ")
        if not (args.validation_root / "meta/info.json").is_file():
            raise FileNotFoundError(args.validation_root / "meta/info.json")
    if args.eval_freq < 1 or args.eval_batches < 0:
        raise ValueError("eval-freq must be positive; eval-batches must be nonnegative")
    for name in (
        "chunk_size",
        "n_action_steps",
        "batch_size",
        "steps",
        "save_freq",
        "log_freq",
    ):
        validate_positive(name, getattr(args, name))
    for name in (
        "learning_rate",
        "backbone_learning_rate",
        "weight_decay",
        "grad_clip_norm",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.n_action_steps > args.chunk_size:
        raise ValueError("n_action_steps cannot exceed chunk_size")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    resume_from = args.resume_from.expanduser().resolve() if args.resume_from else None
    if resume_from is not None:
        if not (resume_from / "pretrained_model" / "config.json").is_file():
            raise FileNotFoundError(resume_from / "pretrained_model/config.json")
        if not (resume_from / "training_state").is_dir():
            raise FileNotFoundError(resume_from / "training_state")
        inferred_output = resume_from.parent.parent.resolve()
        if args.output_dir is not None and args.output_dir.expanduser().resolve() != inferred_output:
            raise ValueError(
                "--output-dir must match the run containing --resume-from: "
                f"{inferred_output}"
            )
        output_dir = inferred_output
    else:
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else DEFAULT_OUTPUT_DIR.resolve()
        )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose another --output-dir or use --resume-from."
            )
    return output_dir, resume_from


def make_policy_config(args: argparse.Namespace, resume_from: Path | None) -> ACTConfig:
    if resume_from is not None:
        model_dir = resume_from / "pretrained_model"
        policy_cfg = PreTrainedConfig.from_pretrained(model_dir)
        if not isinstance(policy_cfg, ACTConfig):
            raise TypeError(f"Resume checkpoint is not ACT: {type(policy_cfg).__name__}")
        policy_cfg.pretrained_path = model_dir
        policy_cfg.device = "cuda"
        policy_cfg.use_amp = args.mixed_precision != "no"
        policy_cfg.push_to_hub = False
        return policy_cfg

    return ACTConfig(
        device="cuda",
        use_amp=args.mixed_precision != "no",
        push_to_hub=False,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        pretrained_backbone_weights=normalized_backbone_weights(
            args.pretrained_backbone_weights
        ),
        optimizer_lr=args.learning_rate,
        optimizer_lr_backbone=args.backbone_learning_rate,
        optimizer_weight_decay=args.weight_decay,
    )


def make_train_config(
    args: argparse.Namespace,
    output_dir: Path,
    policy_cfg: ACTConfig,
    resume_from: Path | None,
) -> TrainPipelineConfig:
    optimizer_cfg = policy_cfg.get_optimizer_preset()
    optimizer_cfg.grad_clip_norm = args.grad_clip_norm
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=str(args.dataset_root),
            video_backend=args.video_backend,
            use_imagenet_stats=True,
        ),
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=args.job_name,
        resume=resume_from is not None,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_checkpoint=True,
        save_freq=args.save_freq,
        use_policy_training_preset=True,
        optimizer=optimizer_cfg,
        scheduler=None,
        wandb=WandBConfig(enable=False),
    )
    cfg.checkpoint_path = resume_from
    return cfg


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("xcore_act_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def save_launch_args(output_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    target = output_dir / f"launch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def atomic_save_checkpoint(
    cfg: TrainPipelineConfig,
    step: int,
    policy,
    optimizer,
    preprocessor,
    postprocessor,
    accelerator: Accelerator,
) -> Path:
    final_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {final_dir}")
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=final_dir.parent)
    )
    try:
        save_checkpoint(
            checkpoint_dir=temp_dir,
            step=step,
            cfg=cfg,
            policy=policy,
            optimizer=optimizer,
            scheduler=None,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        if accelerator.scaler is not None:
            scaler_path = temp_dir / "training_state" / AMP_SCALER_STATE_FILE
            scaler_path.write_text(
                json.dumps(accelerator.scaler.state_dict(), indent=2) + "\n"
            )
        os.replace(temp_dir, final_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    update_last_checkpoint(final_dir)
    return final_dir


def restore_amp_scaler(
    checkpoint_dir: Path,
    accelerator: Accelerator,
    logger: logging.Logger,
) -> None:
    if accelerator.scaler is None:
        return
    scaler_path = checkpoint_dir / "training_state" / AMP_SCALER_STATE_FILE
    if not scaler_path.is_file():
        logger.warning(
            "checkpoint has no %s; FP16 loss scaling restarts from its initial value",
            AMP_SCALER_STATE_FILE,
        )
        return
    state = json.loads(scaler_path.read_text())
    accelerator.scaler.load_state_dict(state)
    logger.info(
        "restored FP16 scaler scale=%s growth_tracker=%s",
        state.get("scale"),
        state.get("_growth_tracker"),
    )


def finite_or_string(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


@torch.no_grad()
def evaluate_validation(policy, loader, preprocessor, accelerator, max_batches=0):
    policy.eval()
    per_dim = torch.zeros(16, device=accelerator.device)
    count = 0
    for index, batch in enumerate(loader):
        batch = preprocessor(batch)
        with accelerator.autocast():
            prediction = policy.predict_action_chunk(batch)
        valid = (~batch["action_is_pad"]).unsqueeze(-1)
        per_dim += ((prediction.float() - batch["action"]).abs() * valid).sum(dim=(0, 1))
        count += int(valid.sum())
        if max_batches and index + 1 >= max_batches:
            break
    if count == 0:
        raise ValueError("validation has no valid action targets")
    per_dim /= count
    if not torch.isfinite(per_dim).all():
        raise FloatingPointError("non-finite validation predictions")
    return {"val_l1": float(per_dim.mean()), "val_l1_per_dimension": per_dim.cpu().tolist(),
            "val_action_targets": count}


def train(args: argparse.Namespace) -> int:
    validate_args(args)
    output_dir, resume_from = resolve_paths(args)

    # Loading the xCore plugin first applies the compatibility shim for Jetson
    # kernel releases such as ``5.15-niic-tegra`` before Accelerate inspects it.
    register_third_party_plugins()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        step_scheduler_with_optimizer=False,
        cpu=False,
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("This standalone trainer currently supports one process/GPU")
    if accelerator.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ACT training configuration")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    policy_cfg = make_policy_config(args, resume_from)
    policy_cfg.device = accelerator.device.type
    cfg = make_train_config(args, output_dir, policy_cfg, resume_from)
    logger = setup_logging(output_dir)
    save_launch_args(output_dir, args)

    logger.info("Standalone ACT trainer; lerobot-train is not used")
    logger.info("device=%s gpu=%s", accelerator.device, torch.cuda.get_device_name(0))
    logger.info("mixed_precision=%s", args.mixed_precision)
    logger.info("dataset=%s", args.dataset_root)
    logger.info("output_dir=%s", output_dir)

    dataset = make_dataset(cfg)
    validation_loader = None
    if args.validation_root is not None:
        validation_cfg = replace(cfg, dataset=DatasetConfig(
            repo_id=args.validation_repo_id, root=str(args.validation_root),
            video_backend=args.video_backend, use_imagenet_stats=True))
        validation_dataset = make_dataset(validation_cfg)
        if validation_dataset.meta.features != dataset.meta.features:
            raise ValueError("train/validation feature schemas must match")
        if validation_dataset.fps != dataset.fps:
            raise ValueError("train/validation fps must match")
        if args.eval_batches and args.eval_batches * cfg.batch_size < len(validation_dataset):
            indices = torch.linspace(0, len(validation_dataset)-1,
                                     args.eval_batches*cfg.batch_size).long().tolist()
            validation_dataset = torch.utils.data.Subset(validation_dataset, indices)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=2 if cfg.num_workers > 0 else None)
    logger.info(
        "dataset frames=%d episodes=%d fps=%s",
        dataset.num_frames,
        dataset.num_episodes,
        dataset.fps,
    )

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    if resume_from is None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            dataset_stats=dataset.meta.stats,
        )
    else:
        model_dir = resume_from / "pretrained_model"
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=model_dir,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={
                "device_processor": {"device": accelerator.device.type}
            },
        )
    preprocessor.reset()
    postprocessor.reset()

    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    if scheduler is not None:
        raise RuntimeError("ACT preset unexpectedly returned a scheduler")

    step = 0
    if resume_from is not None:
        step, optimizer, scheduler = load_training_state(
            resume_from, optimizer, scheduler
        )
        restore_amp_scaler(resume_from, accelerator, logger)
        if step >= cfg.steps:
            raise ValueError(
                f"Checkpoint step {step} is already >= requested steps {cfg.steps}"
            )
        logger.info("resumed checkpoint=%s step=%d", resume_from, step)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )
    policy, optimizer, dataloader = accelerator.prepare(
        policy, optimizer, dataloader
    )
    data_iterator = cycle(dataloader)

    learnable = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    logger.info(
        "ACT chunk_size=%d n_action_steps=%d parameters=%d batch_size=%d workers=%d",
        policy_cfg.chunk_size,
        policy_cfg.n_action_steps,
        learnable,
        cfg.batch_size,
        cfg.num_workers,
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics = WindowMetrics()
    last_saved_step = step if step % cfg.save_freq == 0 else -1
    samples_seen = step * cfg.batch_size
    best_record_path = output_dir / "best_validation.json"
    validation_signature = dict(root=str(args.validation_root), eval_batches=args.eval_batches,
                                sampled_batch_size=cfg.batch_size if args.eval_batches else None)
    previous_best = json.loads(best_record_path.read_text()) if best_record_path.exists() else {}
    best_score = (previous_best["val_l1"]
                  if previous_best.get("validation_signature") == validation_signature else float("inf"))

    try:
        while step < cfg.steps:
            is_best = False
            data_start = time.perf_counter()
            batch = next(data_iterator)
            batch = preprocessor(batch)
            data_seconds = time.perf_counter() - data_start

            update_start = time.perf_counter()
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss, output = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss before step {step+1}")
            accelerator.backward(loss)
            grad_norm_tensor = accelerator.clip_grad_norm_(
                policy.parameters(), cfg.optimizer.grad_clip_norm
            )
            optimizer.step()
            update_seconds = time.perf_counter() - update_start

            step += 1
            samples_seen += int(batch["action"].shape[0])
            loss_value = float(loss.detach().item())
            grad_norm = float(grad_norm_tensor.detach().item())
            metrics.update(
                loss_value,
                output,
                grad_norm,
                data_seconds,
                update_seconds,
            )

            if step % cfg.log_freq == 0:
                values = metrics.means()
                record = {
                    "step": step,
                    "samples": samples_seen,
                    "epoch": samples_seen / dataset.num_frames,
                    "lr": optimizer.param_groups[0]["lr"],
                    "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                    "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                    "amp_scale": accelerator.scaler.get_scale() if accelerator.scaler is not None else None,
                    "optimizer_step_skipped": accelerator.optimizer_step_was_skipped,
                    **{key: finite_or_string(value) for key, value in values.items()},
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(
                    "step=%d samples=%d epoch=%.3f loss=%.4f l1=%.4f kld=%.4f "
                    "grad=%s lr=%.1e update_s=%.3f data_s=%.3f",
                    step,
                    samples_seen,
                    record["epoch"],
                    values["loss"],
                    values["l1_loss"],
                    values["kld_loss"],
                    finite_or_string(values["grad_norm"]),
                    record["lr"],
                    values["update_s"],
                    values["data_s"],
                )
                metrics.reset()

            if validation_loader is not None and (step % args.eval_freq == 0 or step == cfg.steps):
                validation = evaluate_validation(
                    accelerator.unwrap_model(policy), validation_loader, preprocessor,
                    accelerator, args.eval_batches)
                with (output_dir / "validation.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(step=step, **validation)) + "\n")
                logger.info("validation step=%d normalized_l1=%.6f", step, validation["val_l1"])
                is_best = validation["val_l1"] < best_score

            if step % cfg.save_freq == 0 or step == cfg.steps or is_best:
                logger.info("saving checkpoint at step=%d", step)
                checkpoint_dir = atomic_save_checkpoint(
                    cfg,
                    step,
                    accelerator.unwrap_model(policy),
                    optimizer,
                    preprocessor,
                    postprocessor,
                    accelerator,
                )
                last_saved_step = step
                logger.info("checkpoint=%s", checkpoint_dir)
                if is_best:
                    best_score = validation["val_l1"]
                    best_record_path.write_text(json.dumps(dict(
                        step=step, checkpoint=str(checkpoint_dir),
                        validation_signature=validation_signature, **validation), indent=2)+"\n")
                    link = checkpoint_dir.parent / ".best.tmp"
                    link.unlink(missing_ok=True)
                    link.symlink_to(checkpoint_dir.name, target_is_directory=True)
                    os.replace(link, checkpoint_dir.parent / "best")

    except KeyboardInterrupt:
        logger.warning("training interrupted at step=%d", step)
        if step > 0 and step != last_saved_step:
            checkpoint_dir = atomic_save_checkpoint(
                cfg,
                step,
                accelerator.unwrap_model(policy),
                optimizer,
                preprocessor,
                postprocessor,
                accelerator,
            )
            logger.warning("interrupt checkpoint=%s", checkpoint_dir)
        return 130
    finally:
        accelerator.end_training()

    logger.info("training complete at step=%d", step)
    return 0


def main() -> int:
    return train(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
