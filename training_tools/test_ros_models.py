#!/usr/bin/env python3
"""Evaluate both saved policies on ROS observations; publishes no ROS messages."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "AIDE_ACT_dataset_20261111_195100/act"
sys.path[:0] = [str(BASE / "xcore_act/src"), str(BASE / "xcroe_interACT/src"),
                str(BASE / "xcore_act/scripts/auditing")]
from audit_xcore_act_checkpoints import load_policy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.utils import prepare_observation_for_inference
from xcore_interact.data import NormalizationStats, preprocess_observation, denormalize_actions
from xcore_interact.policy import InterACT3DPolicy

CAMERAS = ("head", "left_wrist", "right_wrist")
TOPICS = {key: f"/robot/sensors/cameras/{key}/color/image_raw/compressed" for key in CAMERAS}
TOPICS.update(left_arm="/robot/upperlimb/joint_states/left_arm",
              right_arm="/robot/upperlimb/joint_states/right_arm",
              left_suction="/robot/suction/left/state", right_suction="/robot/suction/right/state")


def summary(values):
    if not len(values):
        return None
    a = np.asarray(values)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def joint_vector(msg, side):
    expected = [f"{side}_joint_{i}" for i in range(1, 8)]
    if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
        raise ValueError("invalid JointState names or positions")
    mapping = dict(zip(msg.name, msg.position))
    result = np.asarray([mapping[name] for name in expected], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite joint state")
    return result


class Inputs(Node):
    def __init__(self):
        super().__init__("act_interact_readonly_test")
        self.lock = threading.Lock()
        self.buffers = {key: deque(maxlen=50) for key in TOPICS}
        self.arrivals = {key: [] for key in TOPICS}
        self.errors = Counter()
        self.decode_ms = []
        self.last_joint_names = {}
        self.subs = []
        self.index = 0
        for key, topic in TOPICS.items():
            msg_type = CompressedImage if key in CAMERAS else JointState if key.endswith("arm") else Bool
            self.subs.append(self.create_subscription(
                msg_type, topic, lambda msg, k=key: self.receive(k, msg), qos_profile_sensor_data))

    def receive(self, key, msg):
        now = time.monotonic()
        try:
            stamp = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                     if hasattr(msg, "header") else None)
            if key in CAMERAS:
                start = time.perf_counter()
                bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None or bgr.shape != (480, 640, 3):
                    raise ValueError("camera decode/shape mismatch; require 640x480 RGB")
                value = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decode_ms = (time.perf_counter() - start) * 1000
            elif key.endswith("arm"):
                value = joint_vector(msg, key.split("_")[0])
            else:
                value = np.float32(bool(msg.data))
            if stamp is not None and stamp <= 0:
                raise ValueError("missing nonzero source timestamp")
        except (ValueError, KeyError) as exc:
            with self.lock:
                self.errors[f"{key}: {exc}"] += 1
            return
        with self.lock:
            self.index += 1
            self.arrivals[key].append(now)
            self.buffers[key].append((stamp, now, self.index, value))
            if key in CAMERAS:
                self.decode_ms.append(decode_ms)
            elif key.endswith("arm"):
                self.last_joint_names[key] = list(msg.name)

    def snapshot(self, last_index, max_skew, max_age):
        now = time.monotonic()
        with self.lock:
            if any(not buffer for buffer in self.buffers.values()):
                return None, "missing_topic"
            # Choose the latest head frame for which all stamped inputs are nearby.
            for head in reversed(self.buffers["head"]):
                if head[2] <= last_index:
                    break
                selected = {"head": head}
                for key in ("left_wrist", "right_wrist", "left_arm", "right_arm"):
                    selected[key] = min(self.buffers[key], key=lambda x: abs(x[0] - head[0]))
                stamps = [x[0] for x in selected.values()]
                if max(stamps) - min(stamps) > max_skew:
                    continue
                selected.update({key: self.buffers[key][-1] for key in ("left_suction", "right_suction")})
                ages = {key: now - value[1] for key, value in selected.items()}
                if max(ages.values()) > max_age:
                    continue
                return {"index": head[2], "stamp": head[0],
                        "images": [selected[key][3] for key in CAMERAS],
                        "state": np.concatenate([selected["left_arm"][3], selected["right_arm"][3],
                                                  [selected["left_suction"][3], selected["right_suction"][3]]]).astype(np.float32),
                        "source_skew_ms": (max(stamps) - min(stamps)) * 1000,
                        "receive_ages_ms": {key: age * 1000 for key, age in ages.items()}}, None
        return None, "no_new_synchronized_frame"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--max-skew-ms", type=float, default=100)
    parser.add_argument("--max-age-ms", type=float, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.max_skew_ms <= 0 or args.max_age_ms <= 0:
        parser.error("duration and freshness limits must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    actpath = ROOT / "model_outputs/act_suction_20261111_213453/checkpoints/best/pretrained_model"
    intpath = ROOT / "model_outputs/interact_suction_20261111_234920/checkpoints/best/pretrained_model"
    meta = LeRobotDatasetMetadata("local/dual_arm_suction_train", root=ROOT / "prepared_data/suction_20hz_v1/train")
    act, pre, post, keys = load_policy(actpath, meta, "cuda")
    interact = InterACT3DPolicy.from_pretrained(intpath).cuda().eval()
    stats = NormalizationStats.load(intpath / "normalization.json")
    assert not interact.config.use_point_cloud

    def predict(name, snapshot):
        if name == "act":
            raw = {"observation.state": snapshot["state"]}
            raw.update({f"observation.images.{key}": rgb for key, rgb in zip(CAMERAS, snapshot["images"])})
            batch = pre(prepare_observation_for_inference(raw, torch.device("cuda"), task="", robot_type=""))
            result = post(act.predict_action_chunk(batch)).float().cpu().numpy()[0]
        else:
            images, points, state = preprocess_observation(
                *snapshot["images"], state=snapshot["state"], stats=stats,
                image_size=interact.config.image_size, device="cuda")
            normalized = interact.predict_action_chunk(images, points, state)
            result = denormalize_actions(normalized[0], stats).float().cpu().numpy()
        if result.shape != (50, 16) or not np.isfinite(result).all():
            raise ValueError(f"{name}: invalid prediction {result.shape}")
        return result

    rclpy.init()
    node = Inputs()
    def spin():
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass
    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    records, predictions = [], {"act": [], "interact": []}
    first = last = None
    error = None
    started = None
    start_iso = datetime.now().astimezone().isoformat()
    try:
        deadline = time.monotonic() + args.timeout
        while first is None and time.monotonic() < deadline:
            first, _ = node.snapshot(-1, args.max_skew_ms / 1000, args.max_age_ms / 1000)
            if first is None:
                time.sleep(0.01)
        if first is None:
            raise TimeoutError("did not receive all seven fresh, synchronized inputs")
        with torch.inference_mode():
            for _ in range(20):
                for name in predictions:
                    predict(name, first)
            torch.cuda.synchronize()
            started = time.monotonic()
            last_index = first["index"]
            next_log = started
            with (args.output_dir / "samples.jsonl").open("w") as log:
                while time.monotonic() - started < args.seconds:
                    snapshot, _ = node.snapshot(last_index, args.max_skew_ms / 1000, args.max_age_ms / 1000)
                    if snapshot is None:
                        time.sleep(0.002)
                        continue
                    last_index = snapshot["index"]
                    last = snapshot
                    row = {key: value for key, value in snapshot.items() if key not in ("images", "state")}
                    row["state"] = snapshot["state"].tolist()
                    for name in predictions:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        prediction = predict(name, snapshot)
                        torch.cuda.synchronize()
                        row[f"{name}_ms"] = (time.perf_counter() - start) * 1000
                        row[f"{name}_first_action"] = prediction[0].tolist()
                        predictions[name].append(prediction)
                    records.append(row)
                    log.write(json.dumps(row) + "\n")
                    if time.monotonic() >= next_log:
                        print(f"samples={len(records)} ACT={row['act_ms']:.2f}ms InterACT={row['interact_ms']:.2f}ms skew={row['source_skew_ms']:.1f}ms", flush=True)
                        next_log = time.monotonic() + 10
    except (Exception, KeyboardInterrupt) as exc:
        error = repr(exc)
        print(error, flush=True)
    finally:
        elapsed = time.monotonic() - started if started is not None else 0
        rclpy.shutdown()
        thread.join(timeout=5)
        node.destroy_node()
    report = {"started_local": start_iso, "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
              "mode": "read_only_live_inference_no_publishers", "precision": "FP32 with TF32 allowed",
              "gpu": torch.cuda.get_device_name(), "topics": TOPICS, "joint_names": node.last_joint_names,
              "state_order": meta.info["features"]["observation.state"]["names"],
              "timing": "CPU image preprocessing + H2D + prediction + denormalization + D2H, CUDA synchronized; excludes ROS receipt and JPEG decode",
              "suction_time_alignment": "Bool has no source stamp; latest received value and local receive age only",
              "source_skew_scope": "five stamped messages: three cameras and two JointState messages; assumes same source clock",
              "max_source_skew_ms_limit": args.max_skew_ms, "max_receive_age_ms_limit": args.max_age_ms,
              "duration_s": elapsed, "inferences_per_model": len(records),
              "paired_observation_hz": len(records) / elapsed if elapsed else 0,
              "errors": dict(node.errors), "fatal_error": error,
              "camera_decode_ms": summary(node.decode_ms), "streams": {}, "models": {}}
    for key, arrivals in node.arrivals.items():
        report["streams"][key] = {"valid_messages": len(arrivals),
            "received_hz": (len(arrivals)-1)/(arrivals[-1]-arrivals[0]) if len(arrivals)>1 else None,
            "max_receive_gap_ms": float(np.diff(arrivals).max()*1000) if len(arrivals)>1 else None}
    if records:
        state_array = np.asarray([row["state"] for row in records], dtype=np.float32)
        lo = np.asarray(meta.stats["observation.state"]["min"])
        hi = np.asarray(meta.stats["observation.state"]["max"])
        report["live_state"] = {"min": state_array.min(axis=0).tolist(), "max": state_array.max(axis=0).tolist(),
            "fraction_outside_training_range_per_dimension": ((state_array < lo-1e-5) | (state_array > hi+1e-5)).mean(axis=0).tolist()}
        report["source_skew_ms"] = summary([row["source_skew_ms"] for row in records])
        report["receive_age_ms"] = {key: summary([row["receive_ages_ms"][key] for row in records]) for key in TOPICS}
        arrays = {"states": state_array, "source_timestamps": np.asarray([row["stamp"] for row in records])}
        for name, path in (("act", actpath), ("interact", intpath)):
            actions = np.stack(predictions[name])
            arrays[name + "_action_chunks"] = actions
            delta = np.abs(actions[:, 0, :14] - state_array[:, :14])
            jump = np.abs(np.diff(actions[:, 0, :14], axis=0)).max(axis=1)
            report["models"][name] = {"checkpoint": str(path),
                "weights_sha256": hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest(),
                "latency_ms": summary([row[name + "_ms"] for row in records]),
                "latency_over_50ms_count": sum(row[name + "_ms"] > 50 for row in records),
                "prediction_shape": list(actions.shape), "all_predictions_finite": bool(np.isfinite(actions).all()),
                "first_action_max_joint_delta_from_feedback_rad": summary(delta.max(axis=1)),
                "consecutive_first_action_max_joint_change_rad": summary(jump),
                "raw_suction_prediction_min": actions[:, :, 14:].min(axis=(0,1)).tolist(),
                "raw_suction_prediction_max": actions[:, :, 14:].max(axis=(0,1)).tolist()}
        np.savez_compressed(args.output_dir / "predictions.npz", **arrays)
        for label, snapshot in (("first", first), ("last", last)):
            for key, rgb in zip(CAMERAS, snapshot["images"]):
                cv2.imwrite(str(args.output_dir / f"{label}_{key}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    report["status"] = "passed_live_input_inference" if records and not error and not node.errors else "needs_review"
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "passed_live_input_inference" else 1


if __name__ == "__main__":
    raise SystemExit(main())
