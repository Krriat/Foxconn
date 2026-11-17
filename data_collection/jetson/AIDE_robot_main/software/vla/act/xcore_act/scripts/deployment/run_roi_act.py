#!/usr/bin/env python
"""Run an ROI-conditioned ACT checkpoint on the xCore right arm.

The default mode is read-only: the runner reads the arm, gripper and shared ROS
camera topics, performs inference, validates every prediction and writes an
audit log without starting ServoJ or publishing gripper commands.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from lerobot_robot_xcore import (
    RosImageCameraConfig,
    XCoreRightArm,
    XCoreRightArmConfig,
)
from lerobot_robot_xcore.xcorerightarm import HARD_LOWER_LIMITS, HARD_UPPER_LIMITS
from roi_act_runtime import ChunkFusionController, GreenHSVDetector, load_model_bundle
from roi_act_runtime.preprocessing import prepare_observation, predict_denormalized_chunk


MOTION_CONFIRMATION = "I_UNDERSTAND_XCORE_MOTION"
DEFAULT_MODEL_DIR = Path("/home/niic/aide/models/green_box_v1")
JOINT_KEYS = [f"joint_{index}.pos" for index in range(1, 8)]
STATE_KEYS = JOINT_KEYS + ["gripper.pos"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=float, default=10.0)
    parser.add_argument("--action-stride", type=int, default=3)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--replan-interval", type=int, default=2)
    parser.add_argument("--new-prediction-weight", type=float, default=0.7)
    parser.add_argument(
        "--roi-provider", choices=("green_hsv", "full_frame"), default=None
    )
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm-motion")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--web-preview-dir", type=Path)
    parser.add_argument("--web-preview-width", type=int, default=960)
    parser.add_argument("--web-preview-jpeg-quality", type=int, default=70)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.num_episodes <= 0 or args.episode_time_s <= 0:
        raise ValueError("episode count and duration must be positive")
    if args.enable_motion and args.confirm_motion != MOTION_CONFIRMATION:
        raise PermissionError(
            f"real motion requires --confirm-motion={MOTION_CONFIRMATION}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def make_robot(runtime: dict[str, Any], *, enable_motion: bool) -> XCoreRightArm:
    cameras = {}
    for name, camera in runtime["cameras"].items():
        cameras[name] = RosImageCameraConfig(
            topic=str(camera["topic"]),
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=int(camera["fps"]),
        )
    robot_settings = runtime.get("robot", {})
    config = XCoreRightArmConfig(
        id="xcore_right",
        cameras=cameras,
        enable_motion=enable_motion,
        require_soft_limits=True,
        move_to_initial_pose=enable_motion,
        initial_joint_positions=[float(value) for value in robot_settings["initial_joint_positions"]],
        require_gripper_state=enable_motion,
        gripper_motion_hold_s=0.0,
        max_relative_target_rad=float(robot_settings.get("max_relative_target_rad", 0.035)),
        max_joint_speed_rad_s=float(robot_settings.get("max_joint_speed_rad_s", 0.5)),
        joint_limit_margin_rad=float(robot_settings.get("joint_limit_margin_rad", 0.20943951023931956)),
    )
    return XCoreRightArm(config)


def observation_state(observation: dict[str, Any]) -> np.ndarray:
    values = np.asarray([observation[key] for key in STATE_KEYS], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("measured state contains NaN or infinity")
    return values


def validate_action(
    action: np.ndarray,
    state: np.ndarray,
    runtime: dict[str, Any],
) -> dict[str, float]:
    if action.shape != (8,) or not np.all(np.isfinite(action)):
        raise RuntimeError(f"invalid predicted action: shape={action.shape}")
    margin = float(runtime["robot"].get("joint_limit_margin_rad", 0.20943951023931956))
    lower = np.asarray(HARD_LOWER_LIMITS, dtype=np.float32) + margin
    upper = np.asarray(HARD_UPPER_LIMITS, dtype=np.float32) - margin
    if np.any(action[:7] < lower) or np.any(action[:7] > upper):
        raise RuntimeError(
            "predicted joint target is outside the static joint limits: "
            f"{action[:7].tolist()}"
        )
    if not 0.0 <= float(action[7]) <= 4095.0:
        raise RuntimeError(f"predicted gripper target is outside [0,4095]: {action[7]}")
    maximum_delta = float(runtime["safety"].get("max_policy_target_delta_rad", 0.5))
    delta = np.abs(action[:7] - state[:7])
    if float(delta.max()) > maximum_delta:
        raise RuntimeError(
            "predicted joint target is too far from measured state: "
            f"max_delta={float(delta.max()):.6f} rad > {maximum_delta:.6f} rad"
        )
    return {
        "max_joint_target_delta_rad": float(delta.max()),
        "mean_joint_target_delta_rad": float(delta.mean()),
    }


def action_dict(action: np.ndarray) -> dict[str, float]:
    return {key: float(value) for key, value in zip(STATE_KEYS, action, strict=True)}


def publish_preview(
    directory: Path,
    frames: dict[str, np.ndarray],
    boxes: dict[str, np.ndarray],
    *,
    width: int,
    quality: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, rgb in frames.items():
        frame = rgb.copy()
        box = boxes.get(name)
        height, frame_width = frame.shape[:2]
        if box is not None and box[4] > 0.5:
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(
                frame,
                (round(x1 * frame_width), round(y1 * height)),
                (round(x2 * frame_width), round(y2 * height)),
                (0, 255, 0),
                2,
            )
        if frame_width > width:
            target_height = max(1, round(height * width / frame_width))
            frame = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise RuntimeError(f"failed to encode preview for {name}")
        temporary = directory / f".{name}.jpg.tmp"
        destination = directory / f"{name}.jpg"
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(destination)


def run_smoke_test(bundle, *, roi_provider, device: torch.device, use_amp: bool) -> None:
    height, width, _channels = bundle.metadata["image_shape"]
    observation: dict[str, Any] = {
        key: float(value)
        for key, value in zip(STATE_KEYS, bundle.qpos_mean.cpu().numpy(), strict=True)
    }
    for name in bundle.metadata["camera_names"]:
        observation[name] = np.zeros((height, width, 3), dtype=np.uint8)
        observation[name][height // 3 : height // 2, width // 3 : width // 2, 1] = 255
    prepared = prepare_observation(
        bundle, observation, roi_provider=roi_provider, device=device
    )
    actions = predict_denormalized_chunk(bundle, prepared, use_amp=use_amp)
    if not torch.isfinite(actions).all():
        raise RuntimeError("smoke test produced non-finite actions")
    valid_rois = (
        int(prepared.boxes[..., 4].sum().item())
        if prepared.boxes is not None
        else None
    )
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "action_shape": list(actions.shape),
                "first_action": actions[0, 0].cpu().tolist(),
                "valid_rois": valid_rois,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    validate_args(args)
    device = torch.device(args.device)
    bundle = load_model_bundle(args.model_dir, device=device)
    if bundle.runtime["runtime_type"] == "full_frame_act_v1":
        provider_name = "none"
        roi_provider = None
    else:
        provider_name = args.roi_provider or str(bundle.runtime["roi"]["provider"])
        roi_settings = bundle.runtime["roi"]
        roi_provider = (
            GreenHSVDetector(
                lower_hsv=tuple(int(value) for value in roi_settings["lower_hsv"]),
                upper_hsv=tuple(int(value) for value in roi_settings["upper_hsv"]),
                minimum_area_ratio=float(roi_settings["minimum_area_ratio"]),
            )
            if provider_name == "green_hsv"
            else None
        )
    print(
        json.dumps(
            {
                "model_dir": str(bundle.root),
                "device": str(device),
                "runtime_type": bundle.runtime["runtime_type"],
                "chunk_size": bundle.model.config.chunk_size,
                "camera_names": bundle.metadata["camera_names"],
                "roi_provider": provider_name,
                "motion": args.enable_motion,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.smoke_test:
        run_smoke_test(
            bundle, roi_provider=roi_provider, device=device, use_amp=args.use_amp
        )
    if args.validate_only or args.smoke_test:
        return 0

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    output_dir = args.output_dir
    if output_dir is not None:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    last_prepared = None

    def predict(observation: dict[str, Any]) -> np.ndarray:
        nonlocal last_prepared
        prepared = prepare_observation(
            bundle, observation, roi_provider=roi_provider, device=device
        )
        if bundle.runtime["runtime_type"] == "roi_act_v1":
            minimum_valid = int(bundle.runtime["roi"].get("minimum_valid_cameras", 1))
            if prepared.boxes is None:
                raise RuntimeError("ROI model observation is missing boxes")
            valid_count = int(prepared.boxes[..., 4].sum().item())
            if valid_count < minimum_valid:
                raise RuntimeError(
                    f"ROI provider found {valid_count} valid cameras; need {minimum_valid}"
                )
        last_prepared = prepared
        return (
            predict_denormalized_chunk(bundle, prepared, use_amp=args.use_amp)[0]
            .cpu()
            .numpy()
        )

    controller = ChunkFusionController(
        predict,
        raw_chunk_size=bundle.model.config.chunk_size,
        action_steps=args.execution_horizon,
        action_stride=args.action_stride,
        replan_interval=args.replan_interval,
        new_prediction_weight=args.new_prediction_weight,
    )
    robot = make_robot(bundle.runtime, enable_motion=args.enable_motion)
    audit_stream = None
    try:
        if output_dir is not None:
            audit_stream = (output_dir / "predictions.jsonl").open(
                "a", encoding="utf-8", buffering=1
            )
        robot.connect()
        period = 1.0 / args.fps
        for episode in range(args.num_episodes):
            controller.reset()
            deadline = time.monotonic() + args.episode_time_s
            next_tick = time.monotonic()
            while not stop_requested and time.monotonic() < deadline:
                observation = robot.get_observation()
                state = observation_state(observation)
                action = controller.select_action(observation)
                safety = validate_action(action, state, bundle.runtime)
                sent = None
                if args.enable_motion:
                    sent = robot.send_action(action_dict(action))
                if args.web_preview_dir is not None and last_prepared is not None:
                    publish_preview(
                        args.web_preview_dir,
                        last_prepared.letterboxed_rgb,
                        last_prepared.box_values,
                        width=args.web_preview_width,
                        quality=args.web_preview_jpeg_quality,
                    )
                record = {
                    "time": time.time(),
                    "episode": episode,
                    "step": controller.step - 1,
                    "state": state.tolist(),
                    "predicted_action": action.tolist(),
                    "sent_action": sent,
                    "motion": args.enable_motion,
                    **safety,
                }
                if audit_stream is not None:
                    audit_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                if controller.step % max(1, round(args.fps)) == 0:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -period:
                    print(
                        f"WARNING: policy loop late by {-delay:.3f}s; resynchronizing",
                        flush=True,
                    )
                    next_tick = time.monotonic()
            if stop_requested:
                break
        return 0
    finally:
        controller.close()
        if audit_stream is not None:
            audit_stream.close()
        robot.disconnect()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
