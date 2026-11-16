#!/usr/bin/env python3
"""Run LeRobot ACT training with grasp-window weighted sampling.

This is a thin wrapper around the installed ``lerobot_train``.  It leaves
LeRobot untouched and only replaces the offline training DataLoader sampler.
For every episode, frames around the first gripper-close target are sampled
more often.  This avoids the xCore dataset's dominant gripper-open labels from
overwhelming the short approach/close phase.

All normal ``lerobot-train`` arguments are accepted.  Extra wrapper arguments:

  --xcore.grasp_pre_frames=15
  --xcore.grasp_post_frames=30
  --xcore.grasp_window_weight=5
  --xcore.other_closed_weight=2.5
  --xcore.start_frames=30
  --xcore.start_weight=2
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GraspWeightConfig:
    pre_frames: int = 15
    post_frames: int = 30
    window_weight: float = 5.0
    other_closed_weight: float = 2.5
    start_frames: int = 30
    start_weight: float = 2.0
    close_threshold: float = 1300.0


def _pop_arg(name: str, default: str) -> str:
    prefix = f"--xcore.{name}="
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument.startswith(prefix):
            value = argument[len(prefix) :]
            del sys.argv[index]
            return value
    return default


def parse_wrapper_args() -> GraspWeightConfig:
    config = GraspWeightConfig(
        pre_frames=int(_pop_arg("grasp_pre_frames", "15")),
        post_frames=int(_pop_arg("grasp_post_frames", "30")),
        window_weight=float(_pop_arg("grasp_window_weight", "5.0")),
        other_closed_weight=float(_pop_arg("other_closed_weight", "2.5")),
        start_frames=int(_pop_arg("start_frames", "30")),
        start_weight=float(_pop_arg("start_weight", "2.0")),
        close_threshold=float(_pop_arg("close_threshold", "1300")),
    )
    if config.pre_frames < 0 or config.post_frames < 0 or config.start_frames < 0:
        raise ValueError("Grasp window frame counts must be non-negative")
    if config.window_weight < 1 or config.other_closed_weight < 1 or config.start_weight < 1:
        raise ValueError("Sampling weights must be at least 1")
    return config


def build_grasp_weights(dataset, config: GraspWeightConfig) -> np.ndarray:
    hf_dataset = dataset.hf_dataset
    actions = np.stack([np.asarray(value, dtype=np.float32) for value in hf_dataset["action"]])
    episode_indices = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] < 1:
        raise ValueError(f"Expected 2-D actions, got {actions.shape}")
    if len(actions) != len(dataset):
        raise ValueError(f"Dataset/action length mismatch: {len(dataset)} vs {len(actions)}")

    gripper = actions[:, -1]
    weights = np.ones(len(dataset), dtype=np.float64)
    closed = gripper >= config.close_threshold
    weights[closed] = config.other_closed_weight

    close_frames: list[int] = []
    grasp_window = np.zeros(len(dataset), dtype=bool)
    start_window = frame_indices < config.start_frames
    weights[start_window] = np.maximum(weights[start_window], config.start_weight)
    for episode_index in np.unique(episode_indices):
        mask = episode_indices == episode_index
        local_positions = np.flatnonzero(mask)
        local_closed = closed[local_positions]
        transitions = np.flatnonzero(local_closed & ~np.r_[False, local_closed[:-1]])
        if len(transitions) == 0:
            raise ValueError(f"Episode {int(episode_index)} has no gripper-close target")
        close_local = int(transitions[0])
        close_frame = int(frame_indices[local_positions[close_local]])
        close_frames.append(close_frame)
        lo = max(0, close_local - config.pre_frames)
        hi = min(len(local_positions), close_local + config.post_frames + 1)
        grasp_window[local_positions[lo:hi]] = True

    weights[grasp_window] = np.maximum(weights[grasp_window], config.window_weight)
    probability = weights / weights.sum()
    logging.info(
        "xCore grasp sampler: episodes=%d, first-close frame min/median/max=%d/%.1f/%d, "
        "start frames=%d (sampling %.2f%%), grasp-window frames=%d (sampling %.2f%%), "
        "all closed frames=%d (sampling %.2f%%)",
        len(close_frames),
        min(close_frames),
        float(np.median(close_frames)),
        max(close_frames),
        int(start_window.sum()),
        float(probability[start_window].sum() * 100),
        int(grasp_window.sum()),
        float(probability[grasp_window].sum() * 100),
        int(closed.sum()),
        float(probability[closed].sum() * 100),
    )
    return weights


def main() -> None:
    config = parse_wrapper_args()
    original_dataloader = torch.utils.data.DataLoader

    def weighted_dataloader(dataset, *args, **kwargs):
        if not hasattr(dataset, "hf_dataset") or not hasattr(dataset, "meta"):
            return original_dataloader(dataset, *args, **kwargs)
        weights = build_grasp_weights(dataset, config)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        kwargs["shuffle"] = False
        kwargs["sampler"] = sampler
        return original_dataloader(dataset, *args, **kwargs)

    torch.utils.data.DataLoader = weighted_dataloader
    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
