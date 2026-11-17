from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from lerobot_robot_xcore.config_xcore import XCoreLeftArmConfig, XCoreRightArmConfig
from lerobot_robot_xcore.xcorerightarm import (
    XCoreLeftArm,
    XCoreRightArm,
    _finite_vector,
)


def joint_action(value: float = 0.0, gripper: float = 900.0) -> dict[str, float]:
    action = {f"joint_{index}.pos": value for index in range(1, 8)}
    action["gripper.pos"] = gripper
    return action


def test_feature_dimensions() -> None:
    robot = XCoreRightArm(XCoreRightArmConfig(calibration_dir=Path("/tmp/xcore-act-test")))
    assert len(robot.action_features) == 8
    assert len(robot.observation_features) == 8


def test_left_arm_uses_independent_controller_gripper_and_eight_dimensional_schema() -> None:
    config = XCoreLeftArmConfig(calibration_dir=Path("/tmp/xcore-act-left-test"))
    robot = XCoreLeftArm(config)
    assert robot.name == "xcore_left_arm"
    assert config.robot_ip == "192.168.1.160"
    assert config.local_ip == "192.168.1.10"
    assert config.gripper_topic == "/robot/gripper/left/command/position"
    assert config.gripper_state_topic == "/robot/gripper/left/state"
    assert len(robot.action_features) == 8
    assert config.initial_joint_positions[0] == 1.66969


def test_no_gripper_feature_dimensions() -> None:
    robot = XCoreRightArm(
        XCoreRightArmConfig(
            calibration_dir=Path("/tmp/xcore-act-test"),
            use_gripper=False,
        )
    )
    assert len(robot.action_features) == 7
    assert len(robot.observation_features) == 7


def test_reject_non_finite_vector() -> None:
    try:
        _finite_vector([0.0] * 6 + [float("nan")], 7, "test")
    except ValueError as exc:
        assert "NaN" in str(exc)
    else:
        raise AssertionError("NaN should have been rejected")


def test_joint_action_is_clipped_without_motion() -> None:
    robot = XCoreRightArm(XCoreRightArmConfig(calibration_dir=Path("/tmp/xcore-act-test")))
    robot._connected = True
    sent = robot.send_action(joint_action(value=0.1, gripper=900.0))
    for index in range(1, 8):
        assert 0.0 < sent[f"joint_{index}.pos"] <= robot.config.max_relative_target_rad
    assert sent["gripper.pos"] == 900.0


def test_gripper_switch_holds_arm_and_quantizes_target() -> None:
    robot = XCoreRightArm(
        XCoreRightArmConfig(
            calibration_dir=Path("/tmp/xcore-act-test"),
            gripper_motion_hold_s=0.7,
        )
    )
    robot._connected = True
    robot._joint_state = [0.25] * 7
    robot._gripper_position = 900.0
    robot._gripper_target = 900.0
    robot._last_command_joints = [0.25] * 7

    sent = robot.send_action(joint_action(value=0.5, gripper=1301.0))

    for index in range(1, 8):
        assert sent[f"joint_{index}.pos"] == 0.25
    assert sent["gripper.pos"] == 1500.0
    assert robot._gripper_hold_until > time.monotonic()


def test_gripper_observation_uses_measured_state() -> None:
    robot = XCoreRightArm(XCoreRightArmConfig(calibration_dir=Path("/tmp/xcore-act-test")))
    robot._gripper_target = 1500.0
    robot._on_gripper_state(SimpleNamespace(data=1075))
    assert robot._gripper_position == 1075.0
    assert robot._gripper_target == 1500.0
    assert robot._gripper_state_ready is True


def test_gripper_hysteresis_prevents_threshold_chatter() -> None:
    robot = XCoreRightArm(XCoreRightArmConfig(calibration_dir=Path("/tmp/xcore-act-test")))
    assert robot._select_gripper_target(1250.0, 900.0) == 900.0
    assert robot._select_gripper_target(1301.0, 900.0) == 1500.0
    assert robot._select_gripper_target(1150.0, 1500.0) == 1500.0
    assert robot._select_gripper_target(1099.0, 1500.0) == 900.0


def test_initial_pose_configuration_matches_pico_servol() -> None:
    config = XCoreRightArmConfig(calibration_dir=Path("/tmp/xcore-act-test"))
    assert config.move_to_initial_pose is True
    assert config.initial_joint_positions == [
        -1.969919,
        1.365856,
        1.272963,
        1.476893,
        1.735811,
        -0.026802,
        0.165558,
    ]
    assert config.initial_move_speed == 1000.0
    assert config.initial_move_zone == 10.0


class _FakeMoveAbsJCommand:
    def __init__(self, target, speed, zone):
        self.target = list(target)
        self.speed = speed
        self.zone = zone


class _FakeRobot:
    def __init__(self, joints):
        self.joints = list(joints)
        self.calls = []

    def setMotionControlMode(self, mode, ec):
        self.calls.append(("setMotionControlMode", mode))

    def setOperateMode(self, mode, ec):
        self.calls.append(("setOperateMode", mode))

    def setPowerState(self, enabled, ec):
        self.calls.append(("setPowerState", enabled))

    def powerState(self, ec):
        self.calls.append(("powerState",))
        return "on"

    def moveReset(self, ec):
        self.calls.append(("moveReset",))

    def moveAppend(self, command, command_id, ec):
        self.calls.append(("moveAppend", command))
        self.joints = command.target.copy()

    def moveStart(self, ec):
        self.calls.append(("moveStart",))

    def operationState(self, ec):
        self.calls.append(("operationState",))
        return "idle"

    def jointPos(self, ec):
        self.calls.append(("jointPos",))
        return self.joints


def test_startup_uses_nrt_moveabsj_before_servoj() -> None:
    config = XCoreRightArmConfig(calibration_dir=Path("/tmp/xcore-act-test"))
    robot = XCoreRightArm(config)
    robot._sdk = SimpleNamespace(
        MotionControlMode=SimpleNamespace(NrtCommandMode="nrt"),
        OperateMode=SimpleNamespace(automatic="automatic"),
        PowerState=SimpleNamespace(on="on", estop="estop", gstop="gstop"),
        OperationState=SimpleNamespace(idle="idle"),
        MoveAbsJCommand=_FakeMoveAbsJCommand,
        PyString=object,
    )
    fake_robot = _FakeRobot([0.0] * 7)
    robot._robot = fake_robot

    robot._move_to_initial_pose_locked()

    names = [call[0] for call in fake_robot.calls]
    assert names.index("setMotionControlMode") < names.index("moveAppend")
    assert fake_robot.calls[0] == ("setMotionControlMode", "nrt")
    move_command = next(call[1] for call in fake_robot.calls if call[0] == "moveAppend")
    assert move_command.target == config.initial_joint_positions
    assert move_command.speed == config.initial_move_speed
    assert move_command.zone == config.initial_move_zone
