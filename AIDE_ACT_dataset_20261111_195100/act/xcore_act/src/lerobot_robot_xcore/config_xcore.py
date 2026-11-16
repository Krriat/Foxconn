from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("xcore_right_arm")
@dataclass
class XCoreRightArmConfig(RobotConfig):
    """Configuration for the 7-axis right arm at 192.168.2.160.

    Motion is deliberately disabled by default. Set ``enable_motion=true`` only
    after the read-only preflight succeeds and the work cell is safe.
    """

    robot_ip: str = "192.168.2.160"
    local_ip: str = "192.168.2.10"
    sdk_python_path: Path = Path(
        "/home/niic/xcoresdk_python-v0.7.1.ar_6/Release/linux/arm"
    )
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    enable_motion: bool = False
    require_soft_limits: bool = True
    move_to_initial_pose: bool = True
    initial_joint_positions: list[float] = field(
        default_factory=lambda: [
            -1.969919,
            1.365856,
            1.272963,
            1.476893,
            1.735811,
            -0.026802,
            0.165558,
        ]
    )
    initial_move_speed: float = 1000.0
    initial_move_zone: float = 10.0
    initial_move_timeout_s: float = 60.0
    initial_pose_tolerance_rad: float = 0.02
    control_hz: float = 50.0
    command_timeout_s: float = 0.25
    max_relative_target_rad: float = 0.035
    max_joint_speed_rad_s: float = 0.5
    joint_limit_margin_rad: float = 0.20943951023931956  # 12 degrees

    use_gripper: bool = True
    gripper_topic: str = "/robot/gripper/right/command/position"
    gripper_state_topic: str = "/robot/gripper/right/state"
    gripper_min: float = 900.0
    gripper_max: float = 1500.0
    require_gripper_state: bool = True
    gripper_state_timeout_s: float = 3.0
    gripper_switch_threshold: float = 1200.0
    gripper_switch_hysteresis: float = 100.0
    # Match pico_servol demonstrations: arm and gripper continue concurrently.
    gripper_motion_hold_s: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if len(self.initial_joint_positions) != 7:
            raise ValueError("initial_joint_positions must contain 7 joint values")
        if not all(math.isfinite(value) for value in self.initial_joint_positions):
            raise ValueError("initial_joint_positions contains NaN or infinity")
        if self.initial_move_speed <= 0:
            raise ValueError("initial_move_speed must be positive")
        if self.initial_move_zone < 0:
            raise ValueError("initial_move_zone must be non-negative")
        if self.initial_move_timeout_s <= 0:
            raise ValueError("initial_move_timeout_s must be positive")
        if self.initial_pose_tolerance_rad <= 0:
            raise ValueError("initial_pose_tolerance_rad must be positive")
        if self.command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be positive")
        if self.max_relative_target_rad <= 0:
            raise ValueError("max_relative_target_rad must be positive")
        if self.max_joint_speed_rad_s <= 0:
            raise ValueError("max_joint_speed_rad_s must be positive")
        if self.gripper_min >= self.gripper_max:
            raise ValueError("gripper_min must be smaller than gripper_max")
        if not self.gripper_min < self.gripper_switch_threshold < self.gripper_max:
            raise ValueError(
                "gripper_switch_threshold must be between gripper_min and gripper_max"
            )
        if self.gripper_switch_hysteresis < 0:
            raise ValueError("gripper_switch_hysteresis must be non-negative")
        if not (
            self.gripper_min
            < self.gripper_switch_threshold - self.gripper_switch_hysteresis
            < self.gripper_switch_threshold + self.gripper_switch_hysteresis
            < self.gripper_max
        ):
            raise ValueError("gripper switch hysteresis thresholds are outside the configured range")
        if self.gripper_state_timeout_s <= 0:
            raise ValueError("gripper_state_timeout_s must be positive")
        if self.gripper_motion_hold_s < 0:
            raise ValueError("gripper_motion_hold_s must be non-negative")


@RobotConfig.register_subclass("xcore_left_arm")
@dataclass
class XCoreLeftArmConfig(XCoreRightArmConfig):
    """Left-arm counterpart with an independent controller and gripper."""

    robot_ip: str = "192.168.1.160"
    local_ip: str = "192.168.1.10"
    initial_joint_positions: list[float] = field(
        default_factory=lambda: [
            1.66969,
            1.365856,
            -1.272963,
            1.476893,
            -1.735811,
            -0.026802,
            -0.165558,
        ]
    )
    gripper_topic: str = "/robot/gripper/left/command/position"
    gripper_state_topic: str = "/robot/gripper/left/state"
    gripper_min: float = 3600.0
    gripper_max: float = 4000.0
    gripper_switch_threshold: float = 3800.0
    gripper_switch_hysteresis: float = 100.0
