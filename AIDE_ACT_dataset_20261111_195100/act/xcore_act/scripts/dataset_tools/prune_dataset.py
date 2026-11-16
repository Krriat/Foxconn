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


def parse_episode_list(raw: str) -> list[int]:
    episodes: set[int] = set()
    if not raw:
        return []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.update(range(int(start), int(end) + 1))
        else:
            episodes.add(int(part))
    return sorted(episodes)


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


def copy_selected_episodes(src_root: Path, dst_root: Path, repo_id: str, keep: list[int], args) -> None:
    src = LeRobotDataset(repo_id, root=src_root, episodes=keep, video_backend=args.video_backend)
    features = {k: v for k, v in src.features.items() if k not in DEFAULT_FEATURE_KEYS}
    dst = LeRobotDataset.create(
        repo_id=repo_id,
        root=dst_root,
        fps=src.fps,
        features=features,
        robot_type=src.meta.info.get("robot_type"),
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        video_backend=args.video_backend,
        vcodec=args.vcodec,
    )

    current_episode = None
    copied_frames = 0
    copied_episodes = 0

    for idx in range(len(src)):
        item = src[idx]
        episode_index = int(item["episode_index"])
        if current_episode is None:
            current_episode = episode_index
            print(f"Copying old episode {episode_index} -> new episode {copied_episodes}")
        elif episode_index != current_episode:
            dst.save_episode()
            dst.image_writer.wait_until_done()
            copied_episodes += 1
            current_episode = episode_index
            print(f"Copying old episode {episode_index} -> new episode {copied_episodes}")

        frame = {"task": item["task"]}
        for key in features:
            frame[key] = tensor_to_numpy(item[key])
        dst.add_frame(frame)
        copied_frames += 1

    if current_episode is not None:
        dst.save_episode()
        dst.image_writer.wait_until_done()
        copied_episodes += 1

    print(f"Done. Wrote {copied_episodes} episodes, {copied_frames} frames to {dst_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild a LeRobot dataset with only selected episodes.")
    parser.add_argument("--src-root", default="smolvla/datasets/right_arm_vr")
    parser.add_argument("--dst-root", default="smolvla/datasets/right_arm_vr_clean")
    parser.add_argument("--repo-id", default="YOUR_HF_USER/right_arm_vr_test")
    parser.add_argument("--keep", default="", help="Episodes to keep, e.g. 0,1,4-7")
    parser.add_argument("--drop", default="", help="Episodes to drop, e.g. 3,8,12")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-writer-processes", type=int, default=2)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="libsvtav1")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Not a LeRobot dataset root: {src_root}")

    src_all = LeRobotDataset(args.repo_id, root=src_root, video_backend=args.video_backend)
    all_episodes = list(range(src_all.meta.total_episodes))
    del src_all

    keep = parse_episode_list(args.keep)
    drop = parse_episode_list(args.drop)
    if keep and drop:
        raise ValueError("Use either --keep or --drop, not both.")
    if not keep:
        keep = [ep for ep in all_episodes if ep not in set(drop)]

    missing = sorted(set(keep) - set(all_episodes))
    if missing:
        raise ValueError(f"Episode(s) not found: {missing}; available: {all_episodes}")
    if not keep:
        raise ValueError("No episodes selected.")

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_root} exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst_root)

    print(f"Keeping episodes: {keep}")
    copy_selected_episodes(src_root, dst_root, args.repo_id, keep, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
