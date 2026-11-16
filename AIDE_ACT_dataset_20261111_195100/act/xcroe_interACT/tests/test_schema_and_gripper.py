import numpy as np
from unittest.mock import patch

from xcore_interact.robot import DualXCoreRobot, GripperIO
from xcore_interact.schema import (
    ACTION_NAMES,
    HardwareConfig,
    LEFT_RESET_JOINTS,
    STATE_NAMES,
    dataset_features,
)
from xcore_interact.telemetry import _TimestampPairBuffer, validate_dual_vector


def test_fixed_16d_schema():
    assert len(STATE_NAMES) == len(ACTION_NAMES) == 16
    features = dataset_features(HardwareConfig().cameras)
    assert features["observation.state"]["shape"] == (16,)
    assert features["action"]["shape"] == (16,)
    assert features["observation.images.head"]["shape"] == (540, 960, 3)
    assert features["observation.images.left_wrist"]["shape"] == (480, 640, 3)
    assert features["observation.images.right_wrist"]["shape"] == (480, 640, 3)
    assert LEFT_RESET_JOINTS == (
        1.66969,
        1.365856,
        -1.272963,
        1.476893,
        -1.735811,
        -0.026802,
        -0.165558,
    )


def test_dual_gripper_topics_and_opposite_directions():
    hardware = HardwareConfig()
    grippers = GripperIO(hardware, enable_motion=False)
    assert hardware.left.gripper_enabled
    assert hardware.left.gripper_command_topic == "/robot/gripper/left/command/position"
    assert hardware.left.gripper_state_topic == "/robot/gripper/left/state"
    assert grippers.command("left", 3590) == 3600
    assert grippers.command("left", 3990) == 4000
    assert grippers.command("right", 910) == 900
    assert grippers.command("right", 1490) == 1500


def test_right_only_gripper_never_commands_left():
    hardware = HardwareConfig()
    grippers = GripperIO(
        hardware, enable_motion=True, motion_sides={"right"}
    )
    grippers.positions["left"] = 3921.0
    assert grippers.motion_sides == frozenset({"right"})
    assert grippers.command("left", 3600) == 3921.0
    assert grippers.command("right", 910) == 900


def test_dual_robot_right_only_enables_only_right_arm():
    class FakeArm:
        def __init__(self):
            self.enabled = None
            self.targets = []

        def connect(self, *, enable_motion):
            self.enabled = enable_motion

        def latest_state(self):
            return np.zeros(7, dtype=np.float32)

        def set_target(self, target):
            self.targets.append(np.asarray(target).copy())
            return np.asarray(target).copy()

        def disconnect(self):
            pass

    class FakeGrippers:
        def __init__(self, _hardware, *, enable_motion, motion_sides):
            self.enable_motion = enable_motion
            self.motion_sides = frozenset(motion_sides)
            self.positions = {"left": 3900.0, "right": 1400.0}

        def connect(self):
            pass

        def command(self, side, value):
            return self.positions[side] if side not in self.motion_sides else float(value)

        def close(self):
            pass

    robot = object.__new__(DualXCoreRobot)
    robot.config = HardwareConfig()
    robot.left = FakeArm()
    robot.right = FakeArm()
    robot.grippers = None
    robot.enable_motion = False
    robot.control_mode = "bimanual"
    robot.motion_sides = frozenset()
    with patch("xcore_interact.robot.GripperIO", FakeGrippers):
        robot.connect(enable_motion=True, control_mode="right_only")
    assert robot.left.enabled is False
    assert robot.right.enabled is True
    assert robot.motion_sides == frozenset({"right"})
    action = np.arange(16, dtype=np.float32)
    sent = robot.send_action(action)
    assert robot.left.targets == []
    assert len(robot.right.targets) == 1
    np.testing.assert_allclose(sent[:7], 0.0)
    np.testing.assert_allclose(sent[8:15], action[8:15])


def test_telemetry_validation():
    values = [0.0] * 16
    values[7] = 3733.0
    values[15] = 904.0
    result = validate_dual_vector(list(STATE_NAMES), values, STATE_NAMES)
    assert result.dtype == np.float32
    assert result.shape == (16,)


def test_telemetry_pairs_only_exact_matching_timestamps():
    pairs = _TimestampPairBuffer(max_pending=4)
    state_100 = np.arange(16, dtype=np.float32)
    action_100 = state_100 + 100
    state_200 = state_100 + 200
    action_200 = state_100 + 300

    assert pairs.add_state(100, state_100, 1.0) is None
    assert pairs.add_action(200, action_200, 1.1) is None
    first = pairs.add_action(100, action_100, 1.2)
    assert first is not None
    np.testing.assert_array_equal(first.state, state_100)
    np.testing.assert_array_equal(first.action, action_100)
    assert first.received_s == 1.2

    second = pairs.add_state(200, state_200, 1.3)
    assert second is not None
    np.testing.assert_array_equal(second.state, state_200)
    np.testing.assert_array_equal(second.action, action_200)
    assert second.received_s == 1.3


def test_telemetry_pair_buffer_never_cross_pairs_unmatched_messages():
    pairs = _TimestampPairBuffer(max_pending=2)
    vector = np.zeros(16, dtype=np.float32)
    pairs.add_state(1, vector + 1, 1.0)
    pairs.add_state(2, vector + 2, 2.0)
    pairs.add_state(3, vector + 3, 3.0)
    assert 1 not in pairs.pending_states
    assert pairs.add_action(1, vector + 10, 4.0) is None
    matched = pairs.add_action(3, vector + 30, 5.0)
    assert matched is not None
    np.testing.assert_array_equal(matched.state, vector + 3)
    np.testing.assert_array_equal(matched.action, vector + 30)
