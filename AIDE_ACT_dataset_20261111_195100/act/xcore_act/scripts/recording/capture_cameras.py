#!/usr/bin/env python3
"""Capture diagnostic JPEGs from the shared three-camera ROS node.

This utility is intentionally a ROS subscriber.  Camera hardware ownership
belongs exclusively to ``hardware/robot_vision_ws``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from lerobot_robot_xcore import RosImageCamera, RosImageCameraConfig


CAMERAS = (
    ("head", "/robot/sensors/cameras/head/color/image_raw", 960, 540),
    ("left_wrist", "/robot/sensors/cameras/left_wrist/color/image_raw", 640, 480),
    ("right_wrist", "/robot/sensors/cameras/right_wrist/color/image_raw", 640, 480),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save synchronized-enough diagnostic JPEGs from robot_vision ROS topics."
    )
    parser.add_argument("--output-dir", default="camera_captures")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    if not 5 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [5, 100]")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras: dict[str, RosImageCamera] = {}
    try:
        for name, topic, width, height in CAMERAS:
            camera = RosImageCamera(
                RosImageCameraConfig(
                    topic=topic,
                    width=width,
                    height=height,
                    fps=30,
                    connect_timeout_s=args.connect_timeout,
                )
            )
            camera.connect()
            cameras[name] = camera
            print(f"[OK] {name}: {topic} ({width}x{height})")

        for capture_index in range(args.count):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for name, camera in cameras.items():
                frame_rgb = camera.async_read()
                destination = output_dir / (
                    f"{timestamp}_{capture_index + 1:03d}_{name}.jpg"
                )
                ok = cv2.imwrite(
                    str(destination),
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError(f"failed to write {destination}")
                print(destination)
            if capture_index + 1 < args.count:
                time.sleep(args.interval)
    finally:
        for camera in reversed(list(cameras.values())):
            camera.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
