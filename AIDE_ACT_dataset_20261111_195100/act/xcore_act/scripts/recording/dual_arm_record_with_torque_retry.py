#!/usr/bin/env python3
"""Run a trained 12-DoF policy on the calibrated dual SO101 follower arms."""

from __future__ import annotations

import argparse
import builtins
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.robots.bi_so_follower.bi_so_follower import BiSOFollower
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.scripts.lerobot_record import main as lerobot_record_main
from lerobot_robot_xcore import RosImageCameraConfig  # noqa: F401 - registers ros_image


LEFT_DATASET_START_POSE_DEG = {
    "shoulder_pan": 20.13,
    "shoulder_lift": 17.73,
    "elbow_flex": 34.04,
    "wrist_flex": -48.71,
    "wrist_roll": -89.69,
    "gripper": 0.70,
}

RIGHT_DATASET_START_POSE_DEG = {
    "shoulder_pan": -19.99,
    "shoulder_lift": 17.08,
    "elbow_flex": 36.38,
    "wrist_flex": -48.73,
    "wrist_roll": 0.22,
    "gripper": 1.52,
}

DEFAULT_TASK = "Open the white drawer, put the yellow block inside, and then close the white drawer"
DEFAULT_POLICY_PATH = "/home/jetson/cxl/smolvla/output/080000/pretrained_model"

_orig_enable_torque = FeetechMotorsBus.enable_torque
_left_robot_id = "left_so100"
_right_robot_id = "right_so100"
_skip_home = False


def _enable_torque_with_retry(self, motors=None, num_retry: int = 0) -> None:
    _orig_enable_torque(self, motors=motors, num_retry=max(num_retry, 3))


def _patched_bi_init(self: BiSOFollower, config) -> None:
    self.robot_type = self.name
    self.id = config.id
    self.config = config

    left_config = SOFollowerRobotConfig(
        id=_left_robot_id,
        calibration_dir=config.calibration_dir,
        port=config.left_arm_config.port,
        disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
        max_relative_target=config.left_arm_config.max_relative_target,
        use_degrees=config.left_arm_config.use_degrees,
        cameras=config.left_arm_config.cameras,
    )
    right_config = SOFollowerRobotConfig(
        id=_right_robot_id,
        calibration_dir=config.calibration_dir,
        port=config.right_arm_config.port,
        disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
        max_relative_target=config.right_arm_config.max_relative_target,
        use_degrees=config.right_arm_config.use_degrees,
        cameras=config.right_arm_config.cameras,
    )
    self.left_arm = SOFollower(left_config)
    self.right_arm = SOFollower(right_config)

    missing_calibrations = [
        str(arm.calibration_fpath)
        for arm in (self.left_arm, self.right_arm)
        if not arm.calibration_fpath.is_file()
    ]
    if missing_calibrations:
        raise FileNotFoundError(
            "Missing arm calibration file(s): " + ", ".join(missing_calibrations)
        )

    duplicate_cameras = set(self.left_arm.cameras) & set(self.right_arm.cameras)
    if duplicate_cameras:
        raise ValueError(f"Duplicate camera names across arms: {sorted(duplicate_cameras)}")
    self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}


def _patched_observation_features(self: BiSOFollower) -> dict:
    features = {
        **{f"left_{key}": value for key, value in self.left_arm._motors_ft.items()},
        **{f"right_{key}": value for key, value in self.right_arm._motors_ft.items()},
    }
    features.update(self.left_arm._cameras_ft)
    features.update(self.right_arm._cameras_ft)
    return features


def _patched_get_observation(self: BiSOFollower) -> dict:
    left_obs = self.left_arm.get_observation()
    right_obs = self.right_arm.get_observation()
    observation = {
        f"left_{key}": value
        for key, value in left_obs.items()
        if key in self.left_arm._motors_ft
    }
    observation.update(
        {key: value for key, value in left_obs.items() if key not in self.left_arm._motors_ft}
    )
    observation.update(
        {
            f"right_{key}": value
            for key, value in right_obs.items()
            if key in self.right_arm._motors_ft
        }
    )
    observation.update(
        {key: value for key, value in right_obs.items() if key not in self.right_arm._motors_ft}
    )
    return observation


def _patched_is_connected(self: BiSOFollower) -> bool:
    # Returning True for a partial connection lets lerobot-record call our robust
    # disconnect method if the second arm or one of its cameras fails to connect.
    return any(
        arm.bus.is_connected or any(camera.is_connected for camera in arm.cameras.values())
        for arm in (self.left_arm, self.right_arm)
    )


def _patched_disconnect(self: BiSOFollower) -> None:
    for arm in (self.left_arm, self.right_arm):
        if arm.bus.is_connected:
            arm.bus.disconnect(arm.config.disable_torque_on_disconnect)
        for camera in arm.cameras.values():
            if camera.is_connected:
                camera.disconnect()


def _read_positions(arm: SOFollower) -> dict[str, float]:
    return {
        joint: float(value)
        for joint, value in arm.bus.sync_read("Present_Position", num_retry=3).items()
    }


def _move_both_arms_to_dataset_start(robot: BiSOFollower, tolerance: float = 2.5) -> None:
    if not robot.left_arm.config.use_degrees or not robot.right_arm.config.use_degrees:
        raise ValueError("Both arms must use degrees for dataset-start homing.")

    print("Moving both arms to the dual-arm dataset start pose...")
    robot.left_arm.bus.enable_torque(num_retry=3)
    robot.right_arm.bus.enable_torque(num_retry=3)

    left_start = _read_positions(robot.left_arm)
    right_start = _read_positions(robot.right_arm)
    max_delta = max(
        max(abs(left_start[joint] - target) for joint, target in LEFT_DATASET_START_POSE_DEG.items()),
        max(abs(right_start[joint] - target) for joint, target in RIGHT_DATASET_START_POSE_DEG.items()),
    )
    num_steps = min(max(max(1, int(max_delta / 0.75)), 30), 300)

    for step in range(1, num_steps + 1):
        fraction = step / num_steps
        left_goal = {
            joint: left_start[joint] + (target - left_start[joint]) * fraction
            for joint, target in LEFT_DATASET_START_POSE_DEG.items()
        }
        right_goal = {
            joint: right_start[joint] + (target - right_start[joint]) * fraction
            for joint, target in RIGHT_DATASET_START_POSE_DEG.items()
        }
        robot.left_arm.bus.sync_write("Goal_Position", left_goal, num_retry=3)
        robot.right_arm.bus.sync_write("Goal_Position", right_goal, num_retry=3)
        time.sleep(1.0 / 30.0)

    for _ in range(150):
        left_current = _read_positions(robot.left_arm)
        right_current = _read_positions(robot.right_arm)
        left_done = all(
            abs(left_current[joint] - target) <= tolerance
            for joint, target in LEFT_DATASET_START_POSE_DEG.items()
        )
        right_done = all(
            abs(right_current[joint] - target) <= tolerance
            for joint, target in RIGHT_DATASET_START_POSE_DEG.items()
        )
        if left_done and right_done:
            print("Dual-arm dataset start pose reached.")
            time.sleep(0.5)
            return
        robot.left_arm.bus.sync_write("Goal_Position", LEFT_DATASET_START_POSE_DEG, num_retry=3)
        robot.right_arm.bus.sync_write("Goal_Position", RIGHT_DATASET_START_POSE_DEG, num_retry=3)
        time.sleep(1.0 / 30.0)

    print("WARNING: dual-arm start pose move timed out; continuing with the closest reached pose.")


def _patched_connect(self: BiSOFollower, calibrate: bool = True) -> None:
    print(f"Connecting left arm on {self.left_arm.config.port} id={self.left_arm.id}")
    self.left_arm.connect(calibrate)
    print(f"Connecting right arm on {self.right_arm.config.port} id={self.right_arm.id}")
    self.right_arm.connect(calibrate)
    if not _skip_home:
        _move_both_arms_to_dataset_start(self)


def _install_patches() -> None:
    FeetechMotorsBus.enable_torque = _enable_torque_with_retry
    BiSOFollower.__init__ = _patched_bi_init
    BiSOFollower.observation_features = property(_patched_observation_features)
    BiSOFollower.get_observation = _patched_get_observation
    BiSOFollower.is_connected = property(_patched_is_connected)
    BiSOFollower.connect = _patched_connect
    BiSOFollower.disconnect = _patched_disconnect
    builtins.input = lambda *args, **kwargs: ""


def _parse_camera(value: str) -> tuple[str, str, int, int]:
    try:
        name, topic, resolution = value.split(":", 2)
        width_text, height_text = resolution.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Camera must use NAME:/ABSOLUTE/TOPIC:WIDTHxHEIGHT"
        ) from exc
    if not name or not topic.startswith("/") or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Invalid ROS camera specification")
    return name, topic, width, height


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--left-arm-port", default="/dev/ttyACM0")
    parser.add_argument("--right-arm-port", default="/dev/ttyACM1")
    parser.add_argument("--left-robot-id", default="left_so100")
    parser.add_argument("--right-robot-id", default="right_so100")
    parser.add_argument("--max-relative-target", type=float, default=6.0)
    parser.add_argument("--ros-camera", action="append", type=_parse_camera, default=[])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--episode-time-s", type=float, default=30.0)
    parser.add_argument("--reset-time-s", type=float, default=10.0)
    parser.add_argument("--skip-home", action="store_true")
    return parser


def _policy_schema(policy_path: str) -> tuple[int | None, set[str] | None]:
    config_path = Path(policy_path) / "config.json"
    if not config_path.is_file():
        return None, None
    with config_path.open() as config_file:
        config = json.load(config_file)

    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    action_dim = int(action_shape[0]) if action_shape else None
    cameras = {
        key.removeprefix("observation.images.")
        for key, feature in config.get("input_features", {}).items()
        if feature.get("type") == "VISUAL" and key.startswith("observation.images.")
    }
    return action_dim, cameras


def _validate_policy_and_cameras(policy_path: str, camera_names: set[str]) -> None:
    action_dim, policy_cameras = _policy_schema(policy_path)
    if action_dim is not None and action_dim != 12:
        raise ValueError(
            f"This script requires a dual-arm 12-D policy, but checkpoint action shape is {action_dim}."
        )
    if policy_cameras is not None and not policy_cameras.issubset(camera_names):
        raise ValueError(
            "Requested ROS cameras do not provide every checkpoint input: "
            f"checkpoint={sorted(policy_cameras)}, requested={sorted(camera_names)}, "
            f"missing={sorted(policy_cameras.difference(camera_names))}"
        )
    if policy_cameras is not None and camera_names.difference(policy_cameras):
        print(
            "Checkpoint will ignore additional ROS cameras: "
            + ", ".join(sorted(camera_names.difference(policy_cameras)))
        )
    print(f"Policy preflight passed: action_dim={action_dim or 'unknown'}, cameras={sorted(camera_names)}")


def _has_arg(argv: list[str], name: str) -> bool:
    return name in argv or any(arg.startswith(f"{name}=") for arg in argv)


def _camera_config(cameras: list[tuple[str, str, int, int]], fps: int) -> dict:
    return {
        name: {
            "type": "ros_image",
            "topic": topic,
            "width": width,
            "height": height,
            "fps": fps,
        }
        for name, topic, width, height in cameras
    }


def _prepare_lerobot_args(custom, passthrough: list[str]) -> tuple[list[str], str | None]:
    cameras = custom.ros_camera or [
        ("head", "/robot/sensors/cameras/head/color/image_raw", 960, 540),
        ("left_wrist", "/robot/sensors/cameras/left_wrist/color/image_raw", 640, 480),
        ("right_wrist", "/robot/sensors/cameras/right_wrist/color/image_raw", 640, 480),
    ]
    camera_names = [name for name, *_ in cameras]
    if len(camera_names) != len(set(camera_names)):
        raise ValueError(f"Duplicate camera names: {camera_names}")
    _validate_policy_and_cameras(custom.policy_path, set(camera_names))

    left_cameras = [camera for camera in cameras if camera[0].startswith("left_")]
    right_cameras = [camera for camera in cameras if not camera[0].startswith("left_")]
    left_camera_config = _camera_config(left_cameras, custom.fps)
    right_camera_config = _camera_config(right_cameras, custom.fps)

    args = list(passthrough)
    defaults = {
        "--robot.type": "bi_so_follower",
        "--robot.id": "dual_so100",
        "--robot.left_arm_config.port": custom.left_arm_port,
        "--robot.left_arm_config.use_degrees": "true",
        "--robot.left_arm_config.max_relative_target": str(custom.max_relative_target),
        "--robot.left_arm_config.cameras": json.dumps(left_camera_config),
        "--robot.right_arm_config.port": custom.right_arm_port,
        "--robot.right_arm_config.use_degrees": "true",
        "--robot.right_arm_config.max_relative_target": str(custom.max_relative_target),
        "--robot.right_arm_config.cameras": json.dumps(right_camera_config),
        "--dataset.repo_id": "local/eval_dual_arm_policy",
        "--dataset.single_task": custom.task,
        "--dataset.num_episodes": "1",
        "--dataset.episode_time_s": str(custom.episode_time_s),
        "--dataset.reset_time_s": str(custom.reset_time_s),
        "--dataset.fps": str(custom.fps),
        "--dataset.push_to_hub": "false",
        "--dataset.vcodec": "h264",
        "--policy.path": custom.policy_path,
        "--policy.device": custom.policy_device,
        "--play_sounds": "false",
    }
    for name, value in defaults.items():
        if not _has_arg(args, name):
            args.append(f"{name}={value}")

    tmp_dataset = None
    if not _has_arg(args, "--dataset.root"):
        tmp_dataset = tempfile.mkdtemp(prefix="dual_arm_eval_", dir="/tmp")
        shutil.rmtree(tmp_dataset)
        args.append(f"--dataset.root={tmp_dataset}")
        print(f"No-save eval enabled: temporary dataset root is {tmp_dataset}")
    return args, tmp_dataset


def main() -> int:
    global _left_robot_id, _right_robot_id, _skip_home

    custom, passthrough = _custom_parser().parse_known_args()
    _left_robot_id = custom.left_robot_id
    _right_robot_id = custom.right_robot_id
    _skip_home = custom.skip_home

    args, tmp_dataset = _prepare_lerobot_args(custom, passthrough)
    sys.argv[:] = [sys.argv[0], *args]
    _install_patches()
    try:
        lerobot_record_main()
        return 0
    finally:
        if tmp_dataset is not None and os.path.exists(tmp_dataset):
            shutil.rmtree(tmp_dataset, ignore_errors=True)
            print("No-save eval cleanup finished.")


if __name__ == "__main__":
    raise SystemExit(main())
