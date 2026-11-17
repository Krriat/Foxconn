#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def main() -> int:
    parser = argparse.ArgumentParser(description="ACT/xCore environment preflight")
    parser.add_argument("--no-network", action="store_true", help="only test imports and model construction")
    parser.add_argument("--robot-read-only", action="store_true", help="connect and read joints without enabling motion")
    parser.add_argument("--arm-side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--check-cameras",
        action="store_true",
        help="subscribe once to ACT head/right-wrist ROS image topics",
    )
    args = parser.parse_args()
    if args.no_network and args.robot_read_only:
        parser.error("--no-network and --robot-read-only are mutually exclusive")

    import torch
    import torchvision
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot_robot_xcore import (
        RosImageCamera,
        RosImageCameraConfig,
        XCoreLeftArm,
        XCoreLeftArmConfig,
        XCoreRightArm,
        XCoreRightArmConfig,
    )

    print(f"arch={platform.machine()}")
    print(f"lerobot={package_version('lerobot')}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")

    cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.head": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 96, 96)
            ),
            f"observation.images.{args.arm_side}_wrist": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 96, 96)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))
        },
        chunk_size=10,
        n_action_steps=2,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    policy = ACTPolicy(cfg)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    print(f"act_model_parameters={parameter_count}")

    robot_class = XCoreLeftArm if args.arm_side == "left" else XCoreRightArm
    config_class = XCoreLeftArmConfig if args.arm_side == "left" else XCoreRightArmConfig
    schema_robot = robot_class(
        config_class(calibration_dir=Path("/tmp/xcore-act-preflight-schema"))
    )
    expected_actions = tuple(
        [f"joint_{index}.pos" for index in range(1, 8)] + ["gripper.pos"]
    )
    if tuple(schema_robot.action_features) != expected_actions:
        raise RuntimeError(
            f"unexpected xCore ServoJ action schema: {list(schema_robot.action_features)}"
        )
    print("xcore_action_schema=" + ",".join(expected_actions))

    if args.check_cameras:
        specs = (
            ("head", "/robot/sensors/cameras/head/color/image_raw", 960, 540),
            (
                f"{args.arm_side}_wrist",
                f"/{args.arm_side}_wrist/color/image_raw",
                640,
                480,
            ),
        )
        cameras = []
        try:
            for name, topic, width, height in specs:
                camera = RosImageCamera(
                    RosImageCameraConfig(
                        topic=topic, width=width, height=height, fps=30
                    )
                )
                camera.connect()
                cameras.append(camera)
                frame = camera.async_read()
                print(f"camera_{name}=PASS topic={topic} shape={frame.shape}")
        finally:
            for camera in reversed(cameras):
                camera.disconnect()

    if args.robot_read_only:
        robot_config = config_class(
            calibration_dir=Path("/tmp/xcore-act-preflight"),
            enable_motion=False,
        )
        robot = robot_class(robot_config)
        try:
            robot.connect()
            observation = robot.get_observation()
            joints = [observation[f"joint_{index}.pos"] for index in range(1, 8)]
            print("joint_state_rad=" + ",".join(f"{value:.6f}" for value in joints))
            print("robot_read_only=PASS")
        finally:
            robot.disconnect()

    print("preflight=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
