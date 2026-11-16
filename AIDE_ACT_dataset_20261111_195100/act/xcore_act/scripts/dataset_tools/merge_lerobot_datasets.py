#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path


DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".hf_datasets_cache"
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_CACHE))

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 3:
            arr = value.numpy()
            # Dataset video decoder returns CHW float [0, 1]; recorder expects HWC uint8.
            if arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return arr
        return value.numpy()
    return value


def dataset_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not (root / "meta" / "info.json").exists():
        raise argparse.ArgumentTypeError(f"Not a LeRobot dataset root: {root}")
    return root


def feature_signature(dataset: LeRobotDataset) -> dict:
    signature = {}
    for key, value in dataset.features.items():
        if key in DEFAULT_FEATURE_KEYS:
            continue
        signature[key] = {
            "dtype": value.get("dtype"),
            "shape": value.get("shape"),
            "names": value.get("names"),
        }
    return signature


def copy_dataset(src: LeRobotDataset, dst: LeRobotDataset, features: dict, source_name: str, next_episode: int) -> int:
    current_episode = None
    copied_frames = 0

    for idx in range(len(src)):
        item = src[idx]
        episode_index = int(item["episode_index"])
        if current_episode is None:
            current_episode = episode_index
            print(f"[{source_name}] old episode {episode_index} -> merged episode {next_episode}")
        elif episode_index != current_episode:
            dst.save_episode()
            dst.image_writer.wait_until_done()
            next_episode += 1
            current_episode = episode_index
            print(f"[{source_name}] old episode {episode_index} -> merged episode {next_episode}")

        frame = {"task": item["task"]}
        for key in features:
            frame[key] = tensor_to_numpy(item[key])
        dst.add_frame(frame)
        copied_frames += 1

    if current_episode is not None:
        dst.save_episode()
        dst.image_writer.wait_until_done()
        next_episode += 1

    print(f"[{source_name}] copied {copied_frames} frames")
    return next_episode


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple local LeRobot datasets into a new dataset.")
    parser.add_argument("--src-root", action="append", required=True, type=dataset_root, help="Source dataset root. Pass once per dataset, in merge order.")
    parser.add_argument("--dst-root", required=True, type=Path, help="New merged dataset root.")
    parser.add_argument("--repo-id", default="local/right_arm_vr_merged")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-writer-processes", type=int, default=2)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="libsvtav1")
    args = parser.parse_args()

    dst_root = args.dst_root.expanduser().resolve()
    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_root} exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst_root)

    first = LeRobotDataset(args.repo_id, root=args.src_root[0], video_backend=args.video_backend)
    first_signature = feature_signature(first)
    features = {key: value for key, value in first.features.items() if key not in DEFAULT_FEATURE_KEYS}

    total_expected_episodes = 0
    total_expected_frames = 0
    sources: list[tuple[Path, LeRobotDataset]] = [(args.src_root[0], first)]
    for src_root in args.src_root[1:]:
        src = LeRobotDataset(args.repo_id, root=src_root, video_backend=args.video_backend)
        if src.fps != first.fps:
            raise ValueError(f"FPS mismatch: {src_root} has {src.fps}, expected {first.fps}")
        if src.meta.info.get("robot_type") != first.meta.info.get("robot_type"):
            raise ValueError(
                f"robot_type mismatch: {src_root} has {src.meta.info.get('robot_type')}, "
                f"expected {first.meta.info.get('robot_type')}"
            )
        if feature_signature(src) != first_signature:
            raise ValueError(f"Feature mismatch: {src_root} is not compatible with {args.src_root[0]}")
        sources.append((src_root, src))

    for _, src in sources:
        total_expected_episodes += src.meta.total_episodes
        total_expected_frames += src.meta.total_frames

    dst = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dst_root,
        fps=first.fps,
        features=features,
        robot_type=first.meta.info.get("robot_type"),
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        video_backend=args.video_backend,
        vcodec=args.vcodec,
    )

    next_episode = 0
    for src_root, src in sources:
        next_episode = copy_dataset(src, dst, features, src_root.name, next_episode)

    print("Merge complete")
    print(f"Expected episodes: {total_expected_episodes}, merged episodes: {next_episode}")
    print(f"Expected frames: {total_expected_frames}")
    print(f"Merged dataset root: {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
