#!/usr/bin/env python3
"""Receive the three ROS compressed RGB streams and record diagnostic evidence."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("act_rgb_input_probe")
    records = {name: {"arrivals": [], "stamps": [], "errors": [], "shape": None}
               for name in ("head", "left_wrist", "right_wrist")}
    latest = {}
    subscriptions = []

    def callback(name, msg):
        now = time.monotonic()
        item = records[name]
        item["arrivals"].append(now)
        item["stamps"].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            item["errors"].append("JPEG/image decode failed")
            return
        item["shape"] = list(bgr.shape)
        latest[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    topics = {}
    for name in records:
        topic = f"/robot/sensors/cameras/{name}/color/image_raw/compressed"
        topics[name] = topic
        subscriptions.append(node.create_subscription(
            CompressedImage, topic, lambda msg, key=name: callback(key, msg),
            qos_profile_sensor_data))
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    report = {"duration_s": time.monotonic() - started, "topics": {}}
    for name, item in records.items():
        arrivals = item["arrivals"]
        intervals = np.diff(arrivals)
        stamps = item["stamps"]
        report["topics"][name] = {
            "topic": topics[name], "messages": len(arrivals),
            "received_hz": ((len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                            if len(arrivals) > 1 else None),
            "max_receive_gap_ms": float(intervals.max() * 1000) if len(intervals) else None,
            "decoded_shape_hwc": item["shape"], "decode_errors": len(item["errors"]),
            "last_header_timestamp_s": stamps[-1] if stamps else None,
        }
    if latest:
        np.savez_compressed(args.output_dir / "latest_rgb.npz", **latest)
        for name, rgb in latest.items():
            cv2.imwrite(str(args.output_dir / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    report["all_streams_received"] = len(latest) == 3
    (args.output_dir / "camera_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["all_streams_received"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
