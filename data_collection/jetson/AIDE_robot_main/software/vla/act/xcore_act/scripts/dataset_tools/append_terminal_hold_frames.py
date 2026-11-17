#!/usr/bin/env python3
"""Append a synthetic terminal hold to every episode of a LeRobot v3 dataset.

The source dataset is never modified.  For each episode this tool:

* copies the final numeric observation/action ``hold_frames`` times;
* repeats the final frame of every video stream the same number of times;
* rebuilds frame/timestamp/global indices;
* rewrites episode metadata and numeric/image statistics;
* removes historical gaps between episodes in consolidated video files.

The output videos are re-encoded as H.264.  This is substantially faster than
re-encoding AV1 on the xCore deployment computer and is supported by LeRobot's
PyAV video backend.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import av
import numpy as np
import pandas as pd

# PyAV otherwise prints one software-colorspace warning for nearly every frame
# on ARM machines.  Encoding/decoding failures still raise Python exceptions.
av.logging.set_level(av.logging.ERROR)

from lerobot.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.video_utils import get_video_info

from fast_prune_dataset import read_parquet_directory, scalar_stats


STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
NUMERIC_FEATURES = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def write_json(path: Path, value: dict) -> None:
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.write_text(json.dumps(value, indent=4, default=convert) + "\n")


def decode_episode_frames(
    video_path: Path,
    from_timestamp: float,
    frame_count: int,
    fps: float,
):
    """Yield exactly the dataset-visible frames for one episode."""
    tolerance = 0.5 / fps
    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        target_pts = int(round(from_timestamp / float(stream.time_base)))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        yielded = 0
        first_timestamp = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if timestamp < from_timestamp - tolerance:
                continue
            if first_timestamp is None:
                first_timestamp = timestamp
                if abs(timestamp - from_timestamp) > tolerance:
                    raise RuntimeError(
                        f"First decoded frame in {video_path} is at {timestamp:.6f}s; "
                        f"expected {from_timestamp:.6f}s"
                    )
            yield frame.to_ndarray(format="rgb24")
            yielded += 1
            if yielded == frame_count:
                return

    raise RuntimeError(
        f"Decoded only {yielded}/{frame_count} frames from {video_path} "
        f"at {from_timestamp:.6f}s"
    )


def encode_camera(
    src_root: Path,
    dst_root: Path,
    episodes: pd.DataFrame,
    camera_key: str,
    fps: int,
    hold_frames: int,
) -> tuple[list[dict[str, np.ndarray]], dict]:
    """Rebuild one consolidated camera video and return per-episode stats."""
    output_path = dst_root / f"videos/{camera_key}/chunk-000/file-000.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_by_episode: list[dict[str, np.ndarray]] = []
    with av.open(str(output_path), "w") as output:
        stream_out = output.add_stream(
            "h264",
            rate=fps,
            options={"crf": "18", "preset": "veryfast", "g": "2"},
        )
        feature = json.loads((src_root / "meta/info.json").read_text())["features"][camera_key]
        height, width, _ = feature["shape"]
        stream_out.width = int(width)
        stream_out.height = int(height)
        stream_out.pix_fmt = "yuv420p"

        for _, episode in episodes.sort_values("episode_index").iterrows():
            episode_index = int(episode["episode_index"])
            old_length = int(episode["length"])
            new_length = old_length + hold_frames
            chunk_index = int(episode[f"videos/{camera_key}/chunk_index"])
            file_index = int(episode[f"videos/{camera_key}/file_index"])
            from_timestamp = float(episode[f"videos/{camera_key}/from_timestamp"])
            source_path = src_root / (
                f"videos/{camera_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            )

            sampled = sample_indices(new_length)
            sampled_set = set(sampled)
            sampled_images: dict[int, np.ndarray] = {}
            last_rgb = None

            for frame_index, rgb in enumerate(
                decode_episode_frames(source_path, from_timestamp, old_length, fps)
            ):
                if frame_index in sampled_set:
                    chw = rgb.transpose(2, 0, 1)
                    sampled_images[frame_index] = auto_downsample_height_width(chw)
                frame_out = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream_out.encode(frame_out):
                    output.mux(packet)
                last_rgb = rgb

            if last_rgb is None:
                raise RuntimeError(f"Episode {episode_index} camera {camera_key} has no frames")

            for hold_index in range(hold_frames):
                frame_index = old_length + hold_index
                if frame_index in sampled_set:
                    chw = last_rgb.transpose(2, 0, 1)
                    sampled_images[frame_index] = auto_downsample_height_width(chw)
                frame_out = av.VideoFrame.from_ndarray(last_rgb, format="rgb24")
                for packet in stream_out.encode(frame_out):
                    output.mux(packet)

            missing_samples = sorted(sampled_set - sampled_images.keys())
            if missing_samples:
                raise RuntimeError(
                    f"Episode {episode_index} camera {camera_key} is missing sampled frames "
                    f"{missing_samples}"
                )
            image_array = np.stack([sampled_images[index] for index in sampled])
            image_stats = get_feature_stats(
                image_array,
                axis=(0, 2, 3),
                keepdims=True,
            )
            image_stats = {
                name: value if name == "count" else np.squeeze(value / 255.0, axis=0)
                for name, value in image_stats.items()
            }
            stats_by_episode.append(image_stats)
            print(
                f"[{camera_key}] episode {episode_index:02d}: "
                f"{old_length} + {hold_frames} = {new_length}",
                flush=True,
            )

        for packet in stream_out.encode():
            output.mux(packet)

    video_info = get_video_info(output_path)
    return stats_by_episode, video_info


def build_expanded_data(
    data: pd.DataFrame,
    episode_indices: list[int],
    fps: float,
    hold_frames: int,
) -> tuple[pd.DataFrame, list[dict[str, dict[str, np.ndarray]]]]:
    parts: list[pd.DataFrame] = []
    numeric_stats: list[dict[str, dict[str, np.ndarray]]] = []
    cursor = 0

    for episode_index in episode_indices:
        part = data[data["episode_index"] == episode_index].copy()
        part = part.sort_values("frame_index").reset_index(drop=True)
        if part.empty:
            raise ValueError(f"Episode {episode_index} has no data frames")

        final = part.iloc[[-1]].copy()
        extension = pd.concat([final] * hold_frames, ignore_index=True)
        part = pd.concat([part, extension], ignore_index=True)
        length = len(part)
        part["frame_index"] = np.arange(length, dtype=np.int64)
        part["timestamp"] = np.arange(length, dtype=np.float32) / np.float32(fps)
        part["episode_index"] = np.full(length, episode_index, dtype=np.int64)
        part["index"] = np.arange(cursor, cursor + length, dtype=np.int64)
        cursor += length
        parts.append(part)

        episode_stats: dict[str, dict[str, np.ndarray]] = {}
        for feature_key in NUMERIC_FEATURES:
            if feature_key in ("observation.state", "action"):
                values = np.stack(part[feature_key].to_numpy()).astype(np.float32)
            else:
                values = part[feature_key].to_numpy(dtype=np.float32).reshape(-1, 1)
            episode_stats[feature_key] = {
                name: np.asarray(value)
                for name, value in scalar_stats(values).items()
            }
        numeric_stats.append(episode_stats)

    return pd.concat(parts, ignore_index=True), numeric_stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a LeRobot v3 dataset and append terminal hold frames to every episode."
    )
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--hold-frames", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if args.hold_frames <= 0:
        raise ValueError("--hold-frames must be positive")
    if src_root == dst_root:
        raise ValueError("Source and destination must differ; this tool preserves the source dataset")
    if dst_root.is_relative_to(src_root):
        raise ValueError("Destination cannot be inside the source dataset")
    if not (src_root / "meta/info.json").is_file():
        raise FileNotFoundError(src_root / "meta/info.json")
    if dst_root.exists() and not args.overwrite:
        raise FileExistsError(f"{dst_root} exists; pass --overwrite to replace it")

    info = json.loads((src_root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    fps = int(info["fps"])
    camera_keys = [
        key for key, feature in info["features"].items() if feature["dtype"] == "video"
    ]
    if not camera_keys:
        raise ValueError("Dataset has no video features")

    data = read_parquet_directory(src_root, "data")
    episodes = read_parquet_directory(src_root, "meta/episodes")
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    episode_indices = [int(value) for value in episodes["episode_index"]]
    if episode_indices != list(range(len(episodes))):
        raise ValueError(f"Episodes must be contiguous from zero, got {episode_indices}")

    expected_lengths = data.groupby("episode_index").size().to_dict()
    for _, episode in episodes.iterrows():
        episode_index = int(episode["episode_index"])
        if expected_lengths.get(episode_index) != int(episode["length"]):
            raise ValueError(f"Episode {episode_index} data/metadata length mismatch")

    expanded_data, numeric_stats = build_expanded_data(
        data, episode_indices, fps, args.hold_frames
    )

    dst_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{dst_root.name}.tmp-", dir=dst_root.parent)
    )
    try:
        (temp_root / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / "meta/tasks.parquet", temp_root / "meta/tasks.parquet")

        image_stats: dict[str, list[dict[str, np.ndarray]]] = {}
        for camera_key in camera_keys:
            stats, video_info = encode_camera(
                src_root,
                temp_root,
                episodes,
                camera_key,
                fps,
                args.hold_frames,
            )
            image_stats[camera_key] = stats
            info["features"][camera_key]["info"] = video_info

        cursor = 0
        episode_stats_all: list[dict[str, dict[str, np.ndarray]]] = []
        for row_index, episode_index in enumerate(episode_indices):
            old_length = int(episodes.at[row_index, "length"])
            new_length = old_length + args.hold_frames
            episodes.at[row_index, "length"] = new_length
            episodes.at[row_index, "data/chunk_index"] = 0
            episodes.at[row_index, "data/file_index"] = 0
            episodes.at[row_index, "dataset_from_index"] = cursor
            episodes.at[row_index, "dataset_to_index"] = cursor + new_length
            episodes.at[row_index, "meta/episodes/chunk_index"] = 0
            episodes.at[row_index, "meta/episodes/file_index"] = 0

            combined_stats = dict(numeric_stats[row_index])
            for camera_key in camera_keys:
                combined_stats[camera_key] = image_stats[camera_key][row_index]
                episodes.at[row_index, f"videos/{camera_key}/chunk_index"] = 0
                episodes.at[row_index, f"videos/{camera_key}/file_index"] = 0
                episodes.at[row_index, f"videos/{camera_key}/from_timestamp"] = cursor / fps
                episodes.at[row_index, f"videos/{camera_key}/to_timestamp"] = (
                    cursor + new_length
                ) / fps

            for feature_key, feature_stats in combined_stats.items():
                for stat_name in STAT_NAMES:
                    column = f"stats/{feature_key}/{stat_name}"
                    if column in episodes.columns:
                        episodes.at[row_index, column] = feature_stats[stat_name].tolist()
            episode_stats_all.append(combined_stats)
            cursor += new_length

        data_path = temp_root / "data/chunk-000/file-000.parquet"
        episode_path = temp_root / "meta/episodes/chunk-000/file-000.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        expanded_data.to_parquet(data_path, index=False)
        episodes.to_parquet(episode_path, index=False)

        info["total_frames"] = int(len(expanded_data))
        info["total_episodes"] = len(episodes)
        info["splits"] = {"train": f"0:{len(episodes)}"}
        info["xcore_terminal_hold_frames"] = args.hold_frames
        info["xcore_terminal_hold_seconds"] = args.hold_frames / fps
        info["xcore_terminal_hold_source"] = "duplicate_final_state_action_and_video_frame"
        write_json(temp_root / "meta/info.json", info)
        write_json(temp_root / "meta/stats.json", aggregate_stats(episode_stats_all))

        if dst_root.exists():
            shutil.rmtree(dst_root)
        temp_root.rename(dst_root)
    except BaseException:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise

    print(
        f"Wrote {len(episodes)} episodes / {len(expanded_data)} frames to {dst_root}; "
        f"each episode has {args.hold_frames} terminal hold frames.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
