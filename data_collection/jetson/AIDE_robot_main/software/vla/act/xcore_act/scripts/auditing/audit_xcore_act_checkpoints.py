#!/usr/bin/env python3
"""Offline safety audit for xCore ACT checkpoints.

The audit never connects to the robot.  It evaluates exact dataset observations
at episode starts, first grasp transitions, and synthetic terminal holds, then
writes a compact JSON result for checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


av.logging.set_level(av.logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def make_raw_observation(
    item: dict, image_keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    observation = {"observation.state": item["observation.state"].numpy()}
    for key in image_keys:
        image = item[key].permute(1, 2, 0).numpy()
        observation[key] = np.rint(image * 255).clip(0, 255).astype(np.uint8)
    return observation


def combine_batches(batches: list[dict]) -> dict:
    keys = set.intersection(*(set(batch) for batch in batches))
    combined = {}
    for key in keys:
        values = [batch[key] for batch in batches]
        if all(isinstance(value, torch.Tensor) for value in values):
            combined[key] = torch.cat(values, dim=0)
    return combined


def load_policy(model_dir: Path, dataset_meta, device_name: str):
    config = PreTrainedConfig.from_pretrained(model_dir)
    config.device = device_name
    config.use_amp = False
    config.pretrained_path = str(model_dir)
    config.pretrained_backbone_weights = None
    policy = make_policy(config, ds_meta=dataset_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=model_dir,
        dataset_stats=dataset_meta.stats,
        preprocessor_overrides={"device_processor": {"device": device_name}},
    )
    image_keys = tuple(
        key for key in config.input_features if key.startswith("observation.images.")
    )
    available = set(dataset_meta.camera_keys)
    missing = set(image_keys).difference(available)
    if missing:
        raise RuntimeError(
            f"Checkpoint {model_dir} requires cameras {sorted(image_keys)}, "
            f"but the dataset provides {sorted(available)}; missing {sorted(missing)}"
        )
    ignored = available.difference(image_keys)
    if ignored:
        print(
            f"Checkpoint {model_dir.name} ignores additional dataset cameras: "
            + ", ".join(sorted(ignored)),
            flush=True,
        )
    return policy, preprocessor, postprocessor, image_keys


def build_samples(
    data: pd.DataFrame, episodes: pd.DataFrame, quick: bool = False, episode_stride: int = 1
) -> list[dict]:
    samples: list[dict] = []
    selected_episodes = episodes.sort_values("episode_index").iloc[::episode_stride]
    for _, episode in selected_episodes.iterrows():
        episode_index = int(episode["episode_index"])
        start = int(episode["dataset_from_index"])
        length = int(episode["length"])
        part = data[data["episode_index"] == episode_index].sort_values("frame_index")
        actions = np.stack(part["action"].to_numpy()).astype(np.float32)
        close_positions = np.flatnonzero((actions[:, -1] >= 1300) & ~np.r_[False, actions[:-1, -1] >= 1300])
        if len(close_positions) == 0:
            raise ValueError(f"Episode {episode_index} has no gripper close transition")
        close = int(close_positions[0])

        local_specs = [("initial", 0, 0)]
        grasp_offsets = (0,) if quick else (-5, 0, 2, 5)
        for offset in grasp_offsets:
            local_specs.append(("grasp", close + offset, offset))
        terminal_specs = (
            (length - 60, 0),
            (length - 30, 30),
            (length - 1, 59),
        )
        if quick:
            terminal_specs = terminal_specs[-1:]
        for local_frame, offset in terminal_specs:
            local_specs.append(("terminal", local_frame, offset))

        for phase, local_frame, offset in local_specs:
            row = part.iloc[local_frame]
            samples.append(
                {
                    "phase": phase,
                    "episode_index": episode_index,
                    "frame_index": int(local_frame),
                    "offset": int(offset),
                    "dataset_index": start + int(local_frame),
                    "state": np.asarray(row["observation.state"], dtype=np.float32),
                    "label": np.asarray(row["action"], dtype=np.float32),
                }
            )
    return samples


def infer_all(dataset, samples, checkpoints, batch_size: int, device_name: str):
    models = {}
    for checkpoint, model_dir in checkpoints.items():
        print(f"Loading checkpoint {checkpoint}", flush=True)
        models[checkpoint] = load_policy(model_dir, dataset.meta, device_name)

    predictions = {checkpoint: np.empty((len(samples), 10, 8), dtype=np.float32) for checkpoint in checkpoints}
    device = torch.device(device_name)
    task = "pick up the green cube and put it into the blue box"
    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start : batch_start + batch_size]
        for checkpoint, (policy, preprocessor, postprocessor, image_keys) in models.items():
            raw_observations = [
                make_raw_observation(dataset[sample["dataset_index"]], image_keys)
                for sample in batch_samples
            ]
            prepared = [
                preprocessor(
                    prepare_observation_for_inference(
                        dict(observation), device, task=task, robot_type="xcore_right_arm"
                    )
                )
                for observation in raw_observations
            ]
            batch = combine_batches(prepared)
            with torch.inference_mode():
                chunk = postprocessor(policy.predict_action_chunk(batch))
            predictions[checkpoint][batch_start : batch_start + len(batch_samples)] = (
                chunk.detach().cpu().numpy()
            )
        print(f"Inference {min(batch_start + batch_size, len(samples))}/{len(samples)}", flush=True)
    return predictions


def summarize_checkpoint(samples: list[dict], prediction: np.ndarray) -> dict:
    state = np.stack([sample["state"] for sample in samples])
    label = np.stack([sample["label"] for sample in samples])
    phase = np.asarray([sample["phase"] for sample in samples])
    offset = np.asarray([sample["offset"] for sample in samples])

    result = {}
    mask = phase == "initial"
    delta = prediction[mask, 0, :7] - state[mask, :7]
    error = prediction[mask, 0, :7] - label[mask, :7]
    max_abs = np.max(np.abs(delta), axis=1)
    result["initial"] = {
        "episodes": int(mask.sum()),
        "joint_delta_mean": delta.mean(axis=0),
        "joint_delta_std": delta.std(axis=0),
        "joint_label_mae": np.abs(error).mean(axis=0),
        "max_abs_delta_median": float(np.median(max_abs)),
        "max_abs_delta_p95": float(np.quantile(max_abs, 0.95)),
        "max_abs_delta_max": float(max_abs.max()),
        "episodes_over_0_015_rad": int((max_abs > 0.015).sum()),
        "gripper_step0_min_mean_max": [
            float(prediction[mask, 0, 7].min()),
            float(prediction[mask, 0, 7].mean()),
            float(prediction[mask, 0, 7].max()),
        ],
    }

    grasp = {}
    for grasp_offset in sorted(set(offset[phase == "grasp"])):
        mask = (phase == "grasp") & (offset == grasp_offset)
        grip = prediction[mask, :, 7]
        arm_error = prediction[mask, 0, :7] - label[mask, :7]
        grasp[str(grasp_offset)] = {
            "episodes": int(mask.sum()),
            "step0_gripper_min_mean_max": [float(grip[:, 0].min()), float(grip[:, 0].mean()), float(grip[:, 0].max())],
            "step0_over_1300": int((grip[:, 0] >= 1300).sum()),
            "first5_any_over_1300": int((grip[:, :5].max(axis=1) >= 1300).sum()),
            "full10_any_over_1300": int((grip.max(axis=1) >= 1300).sum()),
            "step0_arm_label_mae_mean": float(np.abs(arm_error).mean()),
        }
    result["grasp"] = grasp

    terminal = {}
    for terminal_offset in sorted(set(offset[phase == "terminal"])):
        mask = (phase == "terminal") & (offset == terminal_offset)
        arm_delta = prediction[mask, :, :7] - state[mask, None, :7]
        step0 = np.max(np.abs(arm_delta[:, 0]), axis=1)
        first5 = np.max(np.abs(arm_delta[:, :5]), axis=(1, 2))
        full10 = np.max(np.abs(arm_delta), axis=(1, 2))
        grip = prediction[mask, :, 7]
        terminal[str(terminal_offset)] = {
            "episodes": int(mask.sum()),
            "step0_max_abs_median_p95_max": [float(np.median(step0)), float(np.quantile(step0, 0.95)), float(step0.max())],
            "first5_max_abs_median_p95_max": [float(np.median(first5)), float(np.quantile(first5, 0.95)), float(first5.max())],
            "full10_max_abs_median_p95_max": [float(np.median(full10)), float(np.quantile(full10, 0.95)), float(full10.max())],
            "first5_under_0_005_rad": int((first5 <= 0.005).sum()),
            "step0_joint_delta_mean": arm_delta[:, 0].mean(axis=0),
            "gripper_first5_min_mean_max": [float(grip[:, :5].min()), float(grip[:, :5].mean()), float(grip[:, :5].max())],
            "first5_gripper_all_under_1100": int((grip[:, :5].max(axis=1) < 1100).sum()),
        }
    result["terminal"] = terminal
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--checkpoints", default="040000,050000,060000")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested, but CUDA is unavailable")

    dataset_root = Path(args.dataset_root).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    output_path = Path(args.output).resolve()
    checkpoints = {
        step: outputs_root / step / "pretrained_model"
        for step in args.checkpoints.split(",")
    }
    for model_dir in checkpoints.values():
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(model_dir / "model.safetensors")

    dataset = LeRobotDataset(
        repo_id="local/xcore_right_arm_vr_act_v3_hold60",
        root=dataset_root,
        video_backend="pyav",
    )
    data = pd.concat([pd.read_parquet(path) for path in sorted((dataset_root / "data").rglob("*.parquet"))])
    episodes = pd.concat([pd.read_parquet(path) for path in sorted((dataset_root / "meta/episodes").rglob("*.parquet"))])
    samples = build_samples(
        data, episodes, quick=args.quick, episode_stride=args.episode_stride
    )
    predictions = infer_all(
        dataset, samples, checkpoints, args.batch_size, args.device
    )

    report = {
        "dataset_root": str(dataset_root),
        "outputs_root": str(outputs_root),
        "device": args.device,
        "use_amp": False,
        "sample_count": len(samples),
        "sample_definition": {
            "initial": "frame 0 of all 40 episodes",
            "grasp": "first >=1300 gripper target at offset 0" if args.quick else "first >=1300 gripper target at offsets -5, 0, +2, +5",
            "terminal": "synthetic hold offset 59" if args.quick else "synthetic hold offsets 0, 30, 59",
        },
        "checkpoints": {
            checkpoint: summarize_checkpoint(samples, prediction)
            for checkpoint, prediction in predictions.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
