#!/usr/bin/env python3
"""Subscribe through DDS, validate actual frames and save previews plus metrics."""

import argparse
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", nargs="+", default=["head", "left_wrist", "right_wrist"])
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-fps", type=float)
    parser.add_argument("--fps-tolerance", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node("camera_stream_validation_" + socket.gethostname().replace("-", "_"))
    stats = {name: {"frames": 0, "bytes": 0, "errors": [], "first": None,
                    "last": None, "last_stamp_ns": None, "non_increasing_stamps": 0}
             for name in args.cameras}
    subscriptions = []
    timing = {name: {"stamps": [], "arrival": [], "age_ms": []} for name in args.cameras}
    publishers = {}

    def receive(name, message):
        state = stats[name]
        try:
            if args.raw:
                if message.encoding != "bgr8" or message.step < message.width * 3:
                    raise ValueError(f"Unexpected raw image format: {message.encoding}")
                array = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
                frame = array[:, :message.width * 3].reshape(message.height, message.width, 3)
            else:
                frame = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("JPEG could not be decoded")
            if frame.shape[:2] != (480, 640):
                raise ValueError(f"Unexpected image shape: {frame.shape}")
            now = time.monotonic()
            stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            timing[name]["stamps"].append(stamp_ns / 1e9)
            timing[name]["arrival"].append(now)
            timing[name]["age_ms"].append((node.get_clock().now().nanoseconds - stamp_ns) / 1e6)
            if state["last_stamp_ns"] is not None and stamp_ns <= state["last_stamp_ns"]:
                state["non_increasing_stamps"] += 1
            state["last_stamp_ns"] = stamp_ns
            state["last"] = now
            state["frames"] += 1
            state["bytes"] += len(message.data)
            if state["first"] is None:
                state["first"] = now
                state["frame_id"] = message.header.frame_id
                print(f"RECEIVED {name}: {frame.shape}, frame_id={message.header.frame_id}", flush=True)
            state["shape"] = list(frame.shape)
            state["mean"] = round(float(frame.mean()), 2)
            state["std"] = round(float(frame.std()), 2)
            if state["frames"] in (15, 60):
                cv2.imwrite(str(args.output / f"{name}.jpg"), frame)
        except Exception as error:
            if len(state["errors"]) < 5:
                state["errors"].append(str(error))

    def receive_info(name, message):
        stats[name]["camera_info"] = {
            "width": message.width, "height": message.height,
            "calibrated": bool(message.k[0] != 0), "frame_id": message.header.frame_id,
        }

    for name in args.cameras:
        root = f"/robot/sensors/cameras/{name}/color"
        topic = f"{root}/image_raw" + ("" if args.raw else "/compressed")
        subscriptions.append(node.create_subscription(
            Image if args.raw else CompressedImage, topic,
            lambda message, name=name: receive(name, message), qos_profile_sensor_data))
        subscriptions.append(node.create_subscription(
            CameraInfo, f"{root}/camera_info",
            lambda message, name=name: receive_info(name, message), qos_profile_sensor_data))
    def receive_status(message):
        status = json.loads(message.data)
        publishers[status["node"]] = status
    for publisher in ("robot_head_camera", "robot_wrist_cameras"):
        subscriptions.append(node.create_subscription(
            String, f"/robot/sensors/cameras/status/{publisher}", receive_status, 5))
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        now = time.monotonic()
        def summary(values):
            if not len(values):
                return None
            values = np.asarray(values)
            return {"min": round(float(values.min()), 3),
                    "mean": round(float(values.mean()), 3),
                    "p95": round(float(np.percentile(values, 95)), 3),
                    "max": round(float(values.max()), 3)}
        for name, state in stats.items():
            first, last = state.pop("first"), state.pop("last")
            elapsed = 0.0 if first is None or last is None else last - first
            state["received_fps"] = round((state["frames"] - 1) / elapsed, 2) if elapsed > 0 else 0.0
            state["payload_mbps"] = round(state["bytes"] * 8 / elapsed / 1e6, 3) if elapsed > 0 else 0.0
            state["last_frame_age_sec"] = None if last is None else round(now - last, 3)
            state["arrival_interval_ms"] = summary(np.diff(timing[name]["arrival"]) * 1000)
            state["source_interval_ms"] = summary(np.diff(timing[name]["stamps"]) * 1000)
            state["source_age_ms_requires_aligned_host_clocks"] = summary(timing[name]["age_ms"])
            state["rate_ok"] = (args.expected_fps is None or
                                abs(state["received_fps"] - args.expected_fps) <= args.fps_tolerance)
        # Nearest timestamp pairing is a measurement, not a hardware sync guarantee.
        skews = []
        if len(args.cameras) > 1 and all(timing[name]["stamps"] for name in args.cameras):
            reference = np.asarray(timing[args.cameras[0]]["stamps"])
            start = max(timing[name]["stamps"][0] for name in args.cameras)
            end = min(timing[name]["stamps"][-1] for name in args.cameras)
            for stamp in reference[(reference >= start) & (reference <= end)]:
                selected = [stamp]
                for name in args.cameras[1:]:
                    stamps = np.asarray(timing[name]["stamps"])
                    selected.append(stamps[np.argmin(abs(stamps - stamp))])
                skews.append((max(selected) - min(selected)) * 1000)
        report = {"transport": "raw" if args.raw else "jpeg", "duration_sec": args.seconds,
                  "expected_fps": args.expected_fps, "cameras": stats,
                  "nearest_camera_timestamp_skew_ms": summary(skews),
                  "nearest_camera_triplets": len(skews), "publishers": publishers}
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if all(s["frames"] >= 10 and not s["errors"] and s["rate_ok"]
                    and s["non_increasing_stamps"] == 0
                    and s.get("camera_info", {}).get("width") == 640
                    and s["last_frame_age_sec"] is not None and s["last_frame_age_sec"] < 2
                    for s in stats.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
