#!/usr/bin/env python3
"""Read-only preflight for cameras, point cloud, model, and robot state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xcore_interact.cameras import RosRgbdRig
from xcore_interact.pointcloud import valid_point_mask
from xcore_interact.policy import InterACT3DPolicy
from xcore_interact.robot import DualXCoreRobot
from xcore_interact.schema import CameraConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", action="store_true")
    parser.add_argument("--robots", action="store_true")
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--use-point-cloud", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if args.cameras:
        with RosRgbdRig(CameraConfig(use_point_cloud=args.use_point_cloud)) as rig:
            frame = rig.read(point_seed=0)
        report["cameras"] = {
            "head_shape": list(frame.head_rgb.shape),
            "left_wrist_shape": list(frame.left_wrist_rgb.shape),
            "right_wrist_shape": list(frame.right_wrist_rgb.shape),
        }
        if frame.head_points is not None:
            valid_depth = frame.head_points[:, 2][frame.head_points[:, 2] > 0]
            report["cameras"].update(
                point_shape=list(frame.head_points.shape),
                valid_point_ratio=float(valid_point_mask(frame.head_points).mean()),
                depth_range_m=([float(valid_depth.min()), float(valid_depth.max())]
                               if valid_depth.size else None),
            )
    if args.robots:
        robot = DualXCoreRobot()
        try:
            robot.connect(enable_motion=False)
            state = robot.state()
            report["robots"] = {
                "mode": "read_only",
                "state": state.tolist(),
                "finite": bool(np.isfinite(state).all()),
            }
        finally:
            robot.disconnect()
    if args.policy_path:
        model = InterACT3DPolicy.from_pretrained(args.policy_path)
        report["policy"] = {
            "model_type": model.config.model_type,
            "chunk_size": model.config.chunk_size,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "left_gripper_enabled": model.config.left_gripper_enabled,
            "use_point_cloud": model.config.use_point_cloud,
            "state_layout": model.config.state_layout,
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
