#!/usr/bin/env python3
"""Safely deploy InterACT3D to both xCore arms.

Without --enable-motion this performs real camera/robot inference but sends no
commands. Real motion additionally requires an exact confirmation token.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import torch

from xcore_interact.cameras import RosRgbdRig
from xcore_interact.data import (
    NormalizationStats,
    denormalize_actions,
    preprocess_observation,
)
from xcore_interact.policy import InterACT3DPolicy
from xcore_interact.robot import DualXCoreRobot
from xcore_interact.schema import CameraConfig, HardwareConfig

CONFIRMATION = "I_UNDERSTAND_DUAL_ARM_MOTION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm-motion", default="")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument(
        "--left-gripper-disabled",
        action="store_true",
        help="Ignore the installed left gripper (legacy checkpoint/debug only).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.enable_motion and args.confirm_motion != CONFIRMATION:
        raise RuntimeError(f"real motion requires --confirm-motion={CONFIRMATION}")
    if args.duration_s <= 0 or args.fps <= 0 or args.execution_horizon <= 0:
        raise ValueError("duration, fps and execution horizon must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real-time InterACT3D deployment")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy_path = args.policy_path.expanduser().resolve()
    model = InterACT3DPolicy.from_pretrained(policy_path, map_location="cpu").cuda().eval()
    if model.config.state_layout != "bimanual_gripper":
        raise RuntimeError("this hardware adapter controls servo grippers; suction checkpoints require a suction robot adapter")
    stats = NormalizationStats.load(policy_path / "normalization.json")
    camera_config = CameraConfig(
        point_count=model.config.point_count, use_point_cloud=model.config.use_point_cloud
    )
    queue: deque = deque()
    period = 1.0 / args.fps
    hardware = HardwareConfig()
    if args.left_gripper_disabled or not model.config.left_gripper_enabled:
        hardware = replace(
            hardware, left=replace(hardware.left, gripper_enabled=False)
        )
    robot = DualXCoreRobot(hardware)
    cameras = RosRgbdRig(camera_config)
    try:
        robot.connect(enable_motion=args.enable_motion)
        cameras.connect()
        deadline = time.monotonic() + args.duration_s
        next_tick = time.monotonic()
        step = 0
        while time.monotonic() < deadline:
            state = robot.state()
            frame = cameras.read(point_seed=step)
            if not queue:
                images, points, normalized_state = preprocess_observation(
                    frame.head_rgb[..., ::-1].copy(),
                    frame.left_wrist_rgb[..., ::-1].copy(),
                    frame.right_wrist_rgb[..., ::-1].copy(),
                    frame.head_points,
                    state,
                    stats=stats,
                    image_size=model.config.image_size,
                    device="cuda",
                )
                with torch.inference_mode(), torch.cuda.amp.autocast(
                    enabled=args.use_fp16, dtype=torch.float16
                ):
                    normalized = model.predict_action_chunk(images, points, normalized_state)
                actions = denormalize_actions(normalized[0], stats).float().cpu().numpy()
                queue.extend(actions[: min(args.execution_horizon, len(actions))])
            requested = queue.popleft()
            if args.enable_motion:
                sent = robot.send_action(requested)
                logging.info("step=%d sent max_joint_delta=%.4f", step, abs(sent - state).max())
            else:
                logging.info(
                    "DRY-RUN step=%d predicted_left_j1=%.4f right_j1=%.4f (no command sent)",
                    step,
                    requested[0],
                    requested[8],
                )
            step += 1
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                raise RuntimeError(f"deployment loop fell behind by {-delay:.3f}s")
    finally:
        cameras.disconnect()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
