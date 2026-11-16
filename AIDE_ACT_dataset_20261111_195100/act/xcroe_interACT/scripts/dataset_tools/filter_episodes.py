#!/usr/bin/env python3
"""Copy selected/audited episodes into a clean LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_indices(text: str) -> list[int]:
    return sorted({int(value) for value in text.split(",") if value.strip()})


def image_to_uint8_hwc(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * (255.0 if array.max() <= 1.5 else 1.0), 0, 255)
    return array.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--source-repo-id", default="local/xcore_bimanual_interact3d")
    parser.add_argument("--destination-repo-id", default="local/xcore_bimanual_interact3d_clean")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep", type=parse_indices, help="comma-separated episode indices")
    group.add_argument("--validation-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    destination = args.destination_root.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")
    if args.validation_report:
        report = json.loads(args.validation_report.read_text())
        keep = [int(index) for index in report["valid_episode_indices"]]
    else:
        keep = args.keep
    if not keep:
        raise ValueError("no episodes selected")
    info = json.loads((source_root / "meta" / "info.json").read_text())
    source = LeRobotDataset(
        args.source_repo_id, root=source_root, episodes=keep, video_backend="pyav"
    )
    target = LeRobotDataset.create(
        repo_id=args.destination_repo_id,
        root=destination,
        fps=int(info["fps"]),
        robot_type=info.get("robot_type", "xcore_bimanual_interact3d"),
        features=info["features"],
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=4,
    )
    video_keys = set(source.meta.video_keys)
    current_episode = None
    copied = 0
    for index in range(len(source)):
        item = source[index]
        episode = int(item["episode_index"])
        if current_episode is None:
            current_episode = episode
        elif episode != current_episode:
            target.save_episode()
            print(f"copied source episode {current_episode}")
            current_episode = episode
        frame = {"task": item["task"]}
        for key in info["features"]:
            value = item[key]
            if key in video_keys:
                frame[key] = image_to_uint8_hwc(value)
            elif isinstance(value, torch.Tensor):
                frame[key] = value.detach().cpu().numpy()
            else:
                frame[key] = np.asarray(value)
        target.add_frame(frame)
        copied += 1
    if current_episode is not None:
        target.save_episode()
        print(f"copied source episode {current_episode}")
    target.finalize()
    (destination / "meta" / "filter_provenance.json").write_text(
        json.dumps(
            {"source_root": str(source_root), "source_episode_indices": keep, "copied_frames": copied},
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
