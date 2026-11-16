#!/usr/bin/env python3
"""Validate a directly recorded xCore same-sample q_m dataset.

This restores the original working acquisition path used before the q_c
experiment. No state[t+N] shift and no external IK conversion are performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fast_prune_dataset import load_json, read_parquet_directory


NAMES = tuple([f"joint_{index}.pos" for index in range(1, 8)] + ["gripper.pos"])
ACTION_SOURCE_FILE = "xcore_action_source.json"
ACTION_SOURCE = "controller_jointPos_m_same_sample"


def validate(root: Path) -> None:
    info = load_json(root / "meta/info.json")
    features = info.get("features", {})
    state_names = tuple(features.get("observation.state", {}).get("names", ()))
    action_names = tuple(features.get("action", {}).get("names", ()))
    if state_names != NAMES:
        raise RuntimeError(f"unexpected observation schema: {list(state_names)}")
    if action_names != NAMES:
        raise RuntimeError(f"unexpected q_m action schema: {list(action_names)}")

    source_path = root / "meta" / ACTION_SOURCE_FILE
    try:
        source = json.loads(source_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot verify q_m action source: {source_path}") from exc
    expected = {
        "observation_source": "controller_jointPos_m",
        "action_source": ACTION_SOURCE,
        "uses_future_measured_state": False,
        "uses_external_ik": False,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise RuntimeError(
                f"invalid {ACTION_SOURCE_FILE}: {key}={source.get(key)!r}, "
                f"expected {value!r}"
            )

    data = read_parquet_directory(root, "data")
    if data.empty:
        raise RuntimeError("dataset contains no frames")
    states = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(data["action"].to_numpy()).astype(np.float64)
    if states.shape[1] != 8 or actions.shape[1] != 8:
        raise RuntimeError(
            f"expected 8-D state/action, got {states.shape} and {actions.shape}"
        )
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise RuntimeError("state/action contains NaN or infinity")
    if np.any(np.abs(actions[:, :7]) > 3.2):
        raise RuntimeError("q_m action contains a joint value outside the 3.2 rad sanity bound")
    if np.any((actions[:, 7] < 0) | (actions[:, 7] > 4095)):
        raise RuntimeError("gripper action is outside [0, 4095]")

    delta = actions[:, :7] - states[:, :7]
    max_abs_delta = float(np.max(np.abs(delta)))
    print(f"dataset={root}")
    print(f"episodes={data['episode_index'].nunique()} frames={len(data)}")
    print("observation_source=q_m action_source=q_m_same_sample")
    print("uses_future_measured_state=false uses_external_ik=false")
    print(
        "action_q_m_minus_observation_q_m: "
        f"max_abs={max_abs_delta:.9f} rad"
    )
    if max_abs_delta > 1e-6:
        raise RuntimeError(
            "arm action is not the same q_m sample as observation.state"
        )
    print("q_m acquisition validation: passed")
    print(
        "WARNING: same-sample q_m is a measured trajectory sample, not a future "
        "ServoJ target; direct ACT training can learn action=state (hold)."
    )


def self_test() -> None:
    assert len(NAMES) == 8
    assert NAMES[0] == "joint_1.pos"
    assert NAMES[-1] == "gripper.pos"
    print("q_m dataset validator self-test: passed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate direct same-sample q_m observation/action recording. "
            "This script does not shift frames or solve IK."
        )
    )
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.root:
        parser.error("--root is required")
    root = Path(args.root).expanduser().resolve()
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(root / "meta/info.json")
    validate(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
