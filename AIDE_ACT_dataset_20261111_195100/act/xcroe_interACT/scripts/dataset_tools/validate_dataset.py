#!/usr/bin/env python3
"""Audit bimanual RGB data, optionally checking head point clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from xcore_interact.schema import (
    ACTION_KEY,
    HEAD_POINT_CLOUD_KEY,
    STATE_KEY,
    infer_state_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-valid-point-ratio", type=float, default=0.75)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--use-point-cloud", action="store_true")
    return parser.parse_args()


def validate_schema(info: dict, *, use_point_cloud: bool = False) -> list[str]:
    errors = []
    features = info.get("features", {})
    try:
        infer_state_layout(info)
    except ValueError as exc:
        errors.append(str(exc))
    for key in (STATE_KEY, ACTION_KEY):
        if tuple(features.get(key, {}).get("shape", ())) != (16,):
            errors.append(f"{key} shape must be [16]")
    if use_point_cloud and tuple(features.get(HEAD_POINT_CLOUD_KEY, {}).get("shape", ())) != (1024, 6):
        errors.append("head point cloud shape must be [1024,6]")
    for key in (
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ):
        if features.get(key, {}).get("dtype") != "video":
            errors.append(f"{key} is missing or not video")
    return errors


def main() -> int:
    args = parse_args()
    import pyarrow.parquet as pq

    root = args.dataset_root.expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text())
    schema_errors = validate_schema(info, use_point_cloud=args.use_point_cloud)
    if schema_errors:
        report = {"dataset_root": str(root), "use_point_cloud": args.use_point_cloud,
                  "schema_errors": schema_errors, "episodes": {},
                  "valid_episode_indices": [], "valid": False}
        output = args.output or (root / "meta" / "validation_interact3d.json")
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1
    layout = infer_state_layout(info)
    episodes: dict[int, dict] = {}
    columns = ["episode_index", STATE_KEY, ACTION_KEY]
    if args.use_point_cloud:
        columns.append(HEAD_POINT_CLOUD_KEY)
    for path in sorted((root / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            episode_indices = np.asarray(batch.column(0).to_pylist(), dtype=np.int64)
            states = np.asarray(batch.column(1).to_pylist(), dtype=np.float32)
            actions = np.asarray(batch.column(2).to_pylist(), dtype=np.float32)
            points = (np.asarray(batch.column(3).to_pylist(), dtype=np.float32)
                      if args.use_point_cloud else None)
            for row, episode_index in enumerate(episode_indices):
                record = episodes.setdefault(
                    int(episode_index),
                    {"frames": 0, "invalid_vectors": 0, "low_point_frames": 0, "point_ratio_sum": 0.0},
                )
                record["frames"] += 1
                if not np.isfinite(states[row]).all() or not np.isfinite(actions[row]).all():
                    record["invalid_vectors"] += 1
                for vector in (states[row], actions[row]):
                    joints = vector[:14] if layout == "dual_arm_suction" else vector[[*range(7), *range(8, 15)]]
                    effectors = vector[14:] if layout == "dual_arm_suction" else vector[[7, 15]]
                    invalid_effectors = (not np.isin(effectors, (0., 1.)).all() if layout == "dual_arm_suction"
                                         else np.any((effectors < 0) | (effectors > 4095)))
                    if np.any(np.abs(joints) > 3.2) or invalid_effectors:
                        record["invalid_vectors"] += 1
                if points is not None:
                    valid_points = np.isfinite(points[row]).all(axis=-1) & (points[row, :, 2] > 0)
                    ratio = float(valid_points.mean())
                    record["point_ratio_sum"] += ratio
                    if ratio < args.min_valid_point_ratio:
                        record["low_point_frames"] += 1
    for index, record in episodes.items():
        point_sum = record.pop("point_ratio_sum")
        if args.use_point_cloud:
            record["mean_valid_point_ratio"] = point_sum / max(record["frames"], 1)
        record["valid"] = (
            record["frames"] >= args.min_frames
            and record["invalid_vectors"] == 0
            and record["low_point_frames"] == 0
        )
    report = {
        "dataset_root": str(root),
        "use_point_cloud": args.use_point_cloud,
        "state_layout": layout,
        "schema_errors": schema_errors,
        "episodes": {str(key): value for key, value in sorted(episodes.items())},
        "valid_episode_indices": [key for key, value in sorted(episodes.items()) if value["valid"]],
        "valid": not schema_errors and bool(episodes) and all(value["valid"] for value in episodes.values()),
    }
    output = args.output or (root / "meta" / "validation_interact3d.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

