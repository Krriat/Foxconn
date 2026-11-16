#!/usr/bin/env python3
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "fast_prune_dataset.py requires pyarrow. Activate the ACT environment first:\n"
        "  source /home/niic/aide/software/vla/act/xcore_act/environment/activate.sh"
    ) from exc


def parse_episode_list(raw: str) -> list[int]:
    episodes: set[int] = set()
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


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4) + "\n")


def read_parquet_directory(root: Path, relative_dir: str) -> pd.DataFrame:
    paths = sorted((root / relative_dir).rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root / relative_dir}")
    return pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )


def scalar_stats(values: np.ndarray) -> dict:
    values = values.astype(np.float32)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def recompute_basic_stats(data: pd.DataFrame, old_stats: dict) -> dict:
    stats = dict(old_stats)
    for key in ("observation.state", "action"):
        if key in data:
            values = np.stack(data[key].to_numpy())
            stats[key] = scalar_stats(values)
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key in data:
            values = data[key].to_numpy(dtype=np.float32).reshape(-1, 1)
            stats[key] = scalar_stats(values)
    return stats


def recompute_episode_index_stats(episodes: pd.DataFrame, fps: float) -> None:
    """Refresh per-episode stats whose values change when rows are reindexed."""
    for row_idx, row in episodes.iterrows():
        length = int(row["length"])
        episode_index = int(row["episode_index"])
        dataset_from = int(row["dataset_from_index"])
        values_by_key = {
            "timestamp": np.arange(length, dtype=np.float32) / fps,
            "frame_index": np.arange(length, dtype=np.float32),
            "episode_index": np.full(length, episode_index, dtype=np.float32),
            "index": np.arange(dataset_from, dataset_from + length, dtype=np.float32),
        }
        for key, values in values_by_key.items():
            key_stats = scalar_stats(values.reshape(-1, 1))
            for stat_name, stat_value in key_stats.items():
                column = f"stats/{key}/{stat_name}"
                if column in episodes.columns:
                    episodes.at[row_idx, column] = stat_value


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast prune a LeRobot dataset without re-encoding videos.")
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--drop", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    drop = set(parse_episode_list(args.drop))
    if not (src_root / "meta/info.json").exists():
        raise FileNotFoundError(src_root / "meta/info.json")

    src_resolved = src_root.resolve()
    dst_resolved = dst_root.resolve()
    if src_resolved == dst_resolved:
        raise ValueError("--src-root and --dst-root must be different directories")
    if dst_resolved.is_relative_to(src_resolved):
        raise ValueError("--dst-root cannot be inside --src-root")

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_root} exists. Pass --overwrite to replace it.")

    # Validate and compute the pruned metadata before touching the destination.
    # This prevents a missing parquet engine or malformed source dataset from
    # leaving behind a full, unpruned copy that looks successful.
    info = load_json(src_root / "meta/info.json")
    old_stats = load_json(src_root / "meta/stats.json")
    data = read_parquet_directory(src_root, "data")
    episodes = read_parquet_directory(src_root, "meta/episodes")

    if episodes["episode_index"].duplicated().any():
        duplicates = sorted(
            int(ep)
            for ep in episodes.loc[
                episodes["episode_index"].duplicated(keep=False), "episode_index"
            ].unique()
        )
        raise ValueError(f"Duplicate episode metadata found: {duplicates}")
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)

    metadata_episodes = {int(ep) for ep in episodes["episode_index"].tolist()}
    data_episodes = {int(ep) for ep in data["episode_index"].unique().tolist()}
    if metadata_episodes != data_episodes:
        raise ValueError(
            "Episode mismatch between data and metadata: "
            f"metadata_only={sorted(metadata_episodes - data_episodes)}, "
            f"data_only={sorted(data_episodes - metadata_episodes)}"
        )

    available = metadata_episodes
    missing = sorted(drop - available)
    if missing:
        raise ValueError(
            f"Episode(s) not found: {missing}; available: {sorted(available)}"
        )

    keep_old = [int(ep) for ep in episodes["episode_index"].tolist() if int(ep) not in drop]
    if not keep_old:
        raise ValueError("No episodes left after pruning.")

    old_to_new = {old: new for new, old in enumerate(keep_old)}
    data = data[data["episode_index"].isin(keep_old)].copy()
    episodes = episodes[episodes["episode_index"].isin(keep_old)].copy()

    new_data_parts = []
    cursor = 0
    for old_ep in keep_old:
        new_ep = old_to_new[old_ep]
        part = data[data["episode_index"] == old_ep].copy()
        length = len(part)
        part["episode_index"] = new_ep
        part["frame_index"] = np.arange(length, dtype=np.int64)
        part["timestamp"] = part["frame_index"] / float(info["fps"])
        part["index"] = np.arange(cursor, cursor + length, dtype=np.int64)
        new_data_parts.append(part)

        row_idx = episodes.index[episodes["episode_index"] == old_ep][0]
        episodes.loc[row_idx, "episode_index"] = new_ep
        episodes.loc[row_idx, "length"] = length
        episodes.loc[row_idx, "dataset_from_index"] = cursor
        episodes.loc[row_idx, "dataset_to_index"] = cursor + length
        cursor += length

    data = pd.concat(new_data_parts, ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    episodes["data/chunk_index"] = 0
    episodes["data/file_index"] = 0
    episodes["meta/episodes/chunk_index"] = 0
    episodes["meta/episodes/file_index"] = 0
    recompute_episode_index_stats(episodes, float(info["fps"]))

    info["total_episodes"] = len(keep_old)
    info["total_frames"] = int(len(data))
    info["splits"] = {"train": f"0:{len(keep_old)}"}

    dst_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{dst_root.name}.tmp-", dir=dst_root.parent)
    )
    try:
        shutil.copytree(src_root, temp_root, dirs_exist_ok=True)
        # The source may spread data and episode metadata across many parquet
        # files. The pruned output is consolidated, so remove copied parquet
        # shards before writing file-000; otherwise LeRobot would load both the
        # old and new rows and see duplicate episodes.
        shutil.rmtree(temp_root / "data")
        shutil.rmtree(temp_root / "meta/episodes")
        data_path = temp_root / "data/chunk-000/file-000.parquet"
        episodes_path = temp_root / "meta/episodes/chunk-000/file-000.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(data_path, index=False)
        episodes.to_parquet(episodes_path, index=False)
        write_json(temp_root / "meta/info.json", info)
        write_json(
            temp_root / "meta/stats.json", recompute_basic_stats(data, old_stats)
        )

        # Only replace the requested destination after the new dataset is
        # complete. The source dataset is never modified.
        if dst_root.exists():
            shutil.rmtree(dst_root)
        temp_root.rename(dst_root)
    except BaseException:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise

    print(f"Dropped episodes: {sorted(drop)}")
    print(f"Kept old episodes: {keep_old}")
    print(f"Wrote {len(keep_old)} episodes, {len(data)} frames to {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
