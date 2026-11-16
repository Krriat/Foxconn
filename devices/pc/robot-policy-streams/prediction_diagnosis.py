#!/usr/bin/env python3
"""Subscribe to current observations and inspect ACT targets; publishes no actions."""
import argparse
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml

from inference_node import load_runtime


def describe_targets(chunks, states):
    chunks = np.asarray(chunks)
    prefix = chunks[:, :10, :14]
    within = np.diff(prefix, axis=1).reshape(-1, 14)
    boundary = prefix[1:, 0] - prefix[:-1, -1] if len(prefix) > 1 else np.empty((0, 14))
    stream = prefix.reshape(-1, 14)
    first = chunks[:, 0, :14]
    return {
        "input_joint_ptp_rad": np.ptp(states[:, :14], axis=0).tolist(),
        "target_first_step_ptp_rad": np.ptp(first, axis=0).tolist(),
        "target_prefix_ptp_rad": np.ptp(stream, axis=0).tolist(),
        "first_step_minus_measured_rad_mean": (first - states[:, :14]).mean(axis=0).tolist(),
        "within_prefix_step_abs_rad_mean": np.abs(within).mean(axis=0).tolist(),
        "within_prefix_step_abs_rad_max": np.abs(within).max(axis=0).tolist(),
        "next_chunk_boundary_abs_rad_mean": np.abs(boundary).mean(axis=0).tolist() if len(boundary) else None,
        "next_chunk_boundary_abs_rad_max": np.abs(boundary).max(axis=0).tolist() if len(boundary) else None,
        "raw_suction_min": chunks[:, :10, 14:].min(axis=(0, 1)).tolist(),
        "raw_suction_max": chunks[:, :10, 14:].max(axis=(0, 1)).tolist(),
        "note": "Concatenated first ten targets at nominal 20 Hz are diagnostic only; no target was executed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text())
    assert config["model"] == "act", "Inspect the same ACT model used during the live round"
    runtime, predict, reset, weights_sha = load_runtime(config)
    rclpy.init()
    node = runtime.Inputs()  # Subscriptions only; no hardware client or action publisher.
    def spin():
        try:
            rclpy.spin(node)
        except runtime.ExternalShutdownException:
            pass
    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    def snapshot(timeout=10):
        deadline = time.monotonic() + timeout
        reason = ""
        while time.monotonic() < deadline:
            observation, reason = node.snapshot(-1, config["input_max_skew_seconds"], config["input_max_age_seconds"])
            if observation is not None:
                return observation
            time.sleep(.01)
        raise TimeoutError("Fresh observation unavailable: " + reason)
    try:
        observation = snapshot()
        for _ in range(5):
            predict(observation)
        reset()
        frozen = snapshot()
        state = frozen["state"].copy()
        for camera, rgb in zip(("head", "left_wrist", "right_wrist"), frozen["images"]):
            cv2.imwrite(str(args.output / (camera + ".jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.savez_compressed(args.output / "frozen_inputs.npz", state=state,
                            **{camera: rgb for camera, rgb in zip(("head", "left_wrist", "right_wrist"), frozen["images"])})
        print("FROZEN_STATE", json.dumps(state.tolist()), flush=True)
        repeated = np.stack([predict(frozen) for _ in range(args.repeats)])
        representative = repeated[0]
        increments = np.diff(representative[:10, :14], axis=0)
        reversals = []
        for joint in range(14):
            signs = np.sign(increments[:, joint][np.abs(increments[:, joint]) > .0005])
            reversals.append(int(np.count_nonzero(signs[1:] * signs[:-1] < 0)))
        # Offline counterfactual only: restore the suction input value seen while
        # the previous round was running. The physical outputs remain OFF.
        hypothetical = dict(frozen)
        hypothetical["state"] = state.copy()
        hypothetical["state"][14] = 1.0
        hypothetical["state"][15] = 0.0
        reset()
        left_on = predict(hypothetical)
        reset()
        np.savez_compressed(args.output / "frozen_predictions.npz", repeated=repeated,
                            left_on_counterfactual=left_on)
        rows, chunks, states, timestamps = [], [], [], []
        started, next_tick = time.monotonic(), time.monotonic()
        with (args.output / "live_samples.jsonl").open("w") as log:
            while time.monotonic() - started < args.seconds:
                time.sleep(max(0, next_tick - time.monotonic()))
                current = snapshot(3)
                begin = time.monotonic()
                chunk = predict(current)
                elapsed = time.monotonic() - started
                row = {"elapsed_seconds": elapsed, "state": current["state"].tolist(),
                       "source_stamp": current["stamp"], "source_skew_ms": current["source_skew_ms"],
                       "receive_ages_ms": current["receive_ages_ms"],
                       "inference_ms": (time.monotonic() - begin) * 1000,
                       "first_action": chunk[0].tolist(), "last_used_action": chunk[9].tolist()}
                rows.append(row); chunks.append(chunk); states.append(current["state"].copy());timestamps.append(elapsed)
                log.write(json.dumps(row) + "\n")
                if len(chunks) % 20 == 1:
                    print(f"READONLY chunks={len(chunks)} elapsed={elapsed:.1f}s", flush=True)
                next_tick += .5
                if next_tick < time.monotonic():
                    next_tick = time.monotonic()
        chunks, states = np.stack(chunks), np.stack(states)
        np.savez_compressed(args.output / "live_predictions.npz", predictions=chunks,
                            states=states, elapsed=np.asarray(timestamps))
        report = {"mode": "read_only_current_observation_no_action_publishers",
                  "model": "act", "weights_sha256": weights_sha,
                  "frozen_state": state.tolist(), "frozen_source_skew_ms": frozen["source_skew_ms"],
                  "frozen_repeated_count": args.repeats,
                  "frozen_repeat_max_difference_rad": float(np.max(np.abs(repeated[:, :, :14] - repeated[0, :, :14]))),
                  "frozen_repeat_joint_ptp_rad": np.ptp(repeated[:, :, :14], axis=0).max(axis=0).tolist(),
                  "frozen_first10_ptp_rad": np.ptp(representative[:10, :14], axis=0).tolist(),
                  "frozen_first10_net_delta_rad": (representative[9, :14] - representative[0, :14]).tolist(),
                  "frozen_full50_net_delta_rad": (representative[-1, :14] - representative[0, :14]).tolist(),
                  "frozen_repeat_prefix_boundary_jump_rad": (representative[0, :14] - representative[9, :14]).tolist(),
                  "frozen_first10_direction_reversals_threshold_0_0005_rad": reversals,
                  "frozen_first_action": representative[0].tolist(),
                  "left_on_counterfactual_first_action": left_on[0].tolist(),
                  "left_on_counterfactual_max_joint_change_rad": float(np.abs(left_on[:, :14] - representative[:, :14]).max()),
                  "counterfactual_note": "Same current images/joints but left suction input set to 1; no physical output changed",
                  "live_chunks": len(chunks), "live_duration_seconds": time.monotonic() - started,
                  "live": describe_targets(chunks, states),
                  "input_errors": dict(node.errors),
                  "latency_ms_mean": float(np.mean([r["inference_ms"] for r in rows])),
                  "source_skew_ms_max": max(r["source_skew_ms"] for r in rows)}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("mode", "frozen_repeat_max_difference_rad", "frozen_first10_ptp_rad", "live_chunks", "input_errors")}, indent=2), flush=True)
    finally:
        rclpy.shutdown()
        thread.join(timeout=5)
        node.destroy_node()


if __name__ == "__main__":
    main()
