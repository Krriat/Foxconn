#!/usr/bin/env python3
"""Independent two-controller PICO ServoL teleoperation for both xCore arms.

Each controller follows xcore_act's single-arm state machine: click Grip once
to start its arm and click Grip again to stop it. Down stops and resets only
the matching arm. Right Up/B is forwarded for recording; left Up is ignored.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from xcore_interact.pico_teleop import (
    ButtonEdgeTracker,
    DOWN_BIT,
    DualControllerReceiver,
    GRIP_BIT,
    MAX_RAW_POSITION_JUMP_M,
    MAX_RAW_ROTATION_JUMP_RAD,
    TRIGGER_BIT,
    UP_BIT,
    axis_angle_matrix,
    button_pressed,
    compose_dual_record_vectors,
    limited_position_step,
    limited_rotation_step,
    map_position_offset,
    map_rotation_offset,
    matrix_axis_angle,
    parse_controller_packet,
    quaternion_to_matrix,
    scaled_rotation,
)
from xcore_interact.robot import _load_sdk
from xcore_interact.schema import ArmConfig, HardwareConfig, STATE_NAMES as DUAL_STATE_NAMES

PLAN_PERIOD_S = 0.02
ROTATION_GAIN = 0.25
POSITION_DEADBAND_M = 0.0005
ROTATION_DEADBAND_RAD = np.deg2rad(0.3)
MAX_POSITION_STEP_M = 0.001
MAX_ROTATION_STEP_RAD = np.deg2rad(0.5)
MAX_RAW_POSITION_OFFSET_M = 0.40
MAX_RAW_ROTATION_OFFSET_RAD = np.deg2rad(120.0)
CONTROLLER_TIMEOUT_S = 0.25
CONTROL_TRANSITION_RECOVERY_TIMEOUT_S = 2.0
MAX_CYCLE_LATENESS_S = 0.04
MAX_REJECTED_CYCLES = 100
JOINT_MARGIN_RAD = np.deg2rad(12.0)
MIN_ELBOW_RAD = -np.pi
MAX_ELBOW_RAD = np.pi
MAX_ELBOW_VELOCITY_RAD_S = np.deg2rad(2.0)
MAX_ELBOW_ACCELERATION_RAD_S2 = np.deg2rad(2.0)
MAX_ELBOW_STEP_RAD = np.deg2rad(0.01)

HARD_LOWER = np.array([-3.10669, -2.0944, -3.10669, 0.0, -3.10669, -1.0472, -1.0472])
HARD_UPPER = np.array([3.10669, 2.0944, 3.10669, 2.53073, 3.10669, 1.0472, 1.0472])

# Right is the proven xcore_act/pico_servol A/down pose. Left is the user's
# measured left-controller initial point.
RIGHT_TELEOP_RESET_JOINTS = np.array(
    [-1.969919, 1.365856, 1.272963, 1.476893, 1.735811, -0.026802, 0.165558]
)
LEFT_TELEOP_RESET_JOINTS = np.array(
    [1.66969, 1.365856, -1.272963, 1.476893, -1.735811, -0.026802, -0.165558]
)

CONFIRMATIONS = {
    "bimanual": "I_UNDERSTAND_DUAL_ARM_MOTION",
    "left_only": "I_UNDERSTAND_LEFT_ARM_MOTION",
    "right_only": "I_UNDERSTAND_RIGHT_ARM_MOTION",
}
SINGLE_STATE_NAMES = tuple(
    [f"joint_{index}.pos" for index in range(1, 8)] + ["gripper.pos"]
)
TELEMETRY_TOPICS = {
    "bimanual": {
        "state": "/xcore/dual/record_state",
        "action": "/xcore/dual/record_action",
        "active": "/xcore/dual/teleop_active",
        "buttons": "/xcore/dual/pico_buttons",
    },
    "left_only": {
        "state": "/xcore/left/record_state",
        "action": "/xcore/left/record_action",
        "active": "/xcore/left/teleop_active",
        "buttons": "/xcore/left/pico_buttons",
    },
    "right_only": {
        "state": "/xcore/right/record_state",
        "action": "/xcore/right/record_action",
        "active": "/xcore/right/teleop_active",
        "buttons": "/xcore/right/pico_buttons",
    },
}
CONTROL_IMPLEMENTATION = "isolated-sdk-process-per-arm-v3"
WORKER_UDP_PORTS = {"left": 15050, "right": 15051}


def _check_ec(ec: dict[str, Any], operation: str) -> None:
    if int(ec.get("ec", 0)):
        raise RuntimeError(
            f"{operation} failed: code={ec.get('ec')} message={ec.get('message', '')}"
        )


def _position_gain_default() -> float:
    raw = os.environ.get("XCORE_TELEOP_POSITION_GAIN", "0.618")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"invalid XCORE_TELEOP_POSITION_GAIN={raw!r}") from error
    if not 0.1 <= value <= 1.0:
        raise ValueError("XCORE_TELEOP_POSITION_GAIN must be in [0.1, 1.0]")
    return value


def _validated_reset_joints(values: Any, side: str) -> np.ndarray:
    joints = np.asarray(values, dtype=np.float64)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError(f"--{side}-reset-joints must contain 7 finite values")
    outside = np.flatnonzero((joints < HARD_LOWER) | (joints > HARD_UPPER))
    if outside.size:
        index = int(outside[0])
        raise ValueError(
            f"--{side}-reset-joints J{index + 1}={joints[index]:.6f} is outside "
            f"[{HARD_LOWER[index]:.6f}, {HARD_UPPER[index]:.6f}]"
        )
    return joints


def gripper_target_for_trigger(arm: ArmConfig, trigger_pressed: bool) -> int:
    """Pressed opens the matching gripper; released closes it."""
    return int(
        arm.gripper_open_position
        if trigger_pressed
        else arm.gripper_closed_position
    )


@dataclass
class PoseMapping:
    base_frame: list[float]
    toolset: Any
    hand_anchor_position: np.ndarray
    hand_anchor_rotation: np.ndarray
    robot_anchor_position: np.ndarray
    robot_anchor_rotation: np.ndarray
    commanded_position: np.ndarray
    commanded_rotation: np.ndarray
    commanded_elbow: float
    elbow_velocity: float
    command: Any


def next_elbow_target(
    current: float,
    target: float,
    velocity: float,
    *,
    period_s: float = PLAN_PERIOD_S,
) -> tuple[float, float]:
    target = float(np.clip(target, MIN_ELBOW_RAD, MAX_ELBOW_RAD))
    error = target - current
    target_velocity = float(
        np.clip(error / period_s, -MAX_ELBOW_VELOCITY_RAD_S, MAX_ELBOW_VELOCITY_RAD_S)
    )
    max_velocity_change = MAX_ELBOW_ACCELERATION_RAD_S2 * period_s
    velocity += float(
        np.clip(target_velocity - velocity, -max_velocity_change, max_velocity_change)
    )
    step = float(np.clip(velocity * period_s, -MAX_ELBOW_STEP_RAD, MAX_ELBOW_STEP_RAD))
    velocity = step / period_s
    candidate = current + step
    if (error >= 0.0 and candidate >= target) or (error <= 0.0 and candidate <= target):
        return target, 0.0
    return float(np.clip(candidate, MIN_ELBOW_RAD, MAX_ELBOW_RAD)), velocity


def read_elbow_command(
    path: str | None,
    side: str,
    last_generation: int,
) -> tuple[int, float, bool, bool] | None:
    if not path:
        return None
    try:
        fields = Path(path).read_text(encoding="utf-8").split()
        generation = int(fields[0])
        target = float(fields[1] if side == "left" else fields[2])
        enabled = bool(int(fields[3] if side == "left" else fields[4]))
        control_enabled = bool(int(fields[5]))
    except (FileNotFoundError, IndexError, OSError, TypeError, ValueError):
        return None
    if (
        generation <= last_generation
        or not np.isfinite(target)
        or not MIN_ELBOW_RAD <= target <= MAX_ELBOW_RAD
    ):
        return None
    return generation, target, enabled, control_enabled


def write_elbow_status(path: str | None, worker_states: dict[str, dict[str, Any]]) -> None:
    if not path:
        return
    arms: dict[str, dict[str, Any]] = {}
    generations: list[int] = []
    for side, state in worker_states.items():
        current = state.get("elbow_current")
        target = state.get("elbow_target")
        if current is None or target is None:
            continue
        arms[side] = {
            "current_rad": float(current),
            "target_rad": float(target),
            "active": bool(state.get("servo_active", False)),
        }
        generations.append(int(state.get("elbow_generation", 0)))
    if not arms:
        return
    value = {
        "timestamp": time.time(),
        "generation": min(generations),
        "enabled": all(
            bool(state.get("elbow_control_enabled", True))
            for state in worker_states.values()
        ),
        "arms": arms,
    }
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_name(f".{target_path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target_path)


class RosTeleopBridge:
    """Gripper IO plus xcore_interact dataset telemetry."""

    def __init__(self, hardware: HardwareConfig, control_mode: str = "bimanual"):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Int32, UInt8

        self._rclpy = rclpy
        self._messages = {
            "JointState": JointState,
            "Bool": Bool,
            "Int32": Int32,
            "UInt8": UInt8,
        }
        self.hardware = hardware
        self.control_mode = control_mode
        self.motion_sides = (
            frozenset({"left", "right"})
            if control_mode == "bimanual"
            else frozenset({"left" if control_mode == "left_only" else "right"})
        )
        self.button_side = "left" if control_mode == "left_only" else "right"
        topics = TELEMETRY_TOPICS[control_mode]
        self._lock = threading.Lock()
        self.gripper_state = {
            "left": int(hardware.left.gripper_placeholder),
            "right": int(hardware.right.gripper_placeholder),
        }
        self.gripper_ready = {"left": False, "right": False}
        self.gripper_target = dict(self.gripper_state)
        self._last_gripper_command: dict[str, int | None] = {"left": None, "right": None}
        self._last_active: bool | None = None
        self._last_buttons: int | None = None

        self._context = Context()
        rclpy.init(
            args=None,
            context=self._context,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self._node = rclpy.create_node("xcore_interact_pico_dual_servol", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._state_pub = self._node.create_publisher(
            JointState, topics["state"], 20
        )
        self._action_pub = self._node.create_publisher(
            JointState, topics["action"], 20
        )
        self._active_pub = self._node.create_publisher(
            Bool, topics["active"], 20
        )
        self._buttons_pub = self._node.create_publisher(UInt8, topics["buttons"], 20)
        self._gripper_pubs = {}
        self._subscriptions = []
        for arm in (hardware.left, hardware.right):
            if arm.side not in self.motion_sides:
                continue
            self._gripper_pubs[arm.side] = self._node.create_publisher(
                Int32, arm.gripper_command_topic, 10
            )
            self._subscriptions.append(
                self._node.create_subscription(
                    Int32,
                    arm.gripper_state_topic,
                    lambda message, side=arm.side: self._on_gripper(side, message.data),
                    10,
                )
            )
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name="dual-teleop-ros", daemon=True
        )
        self._spin_thread.start()

    def _on_gripper(self, side: str, value: int) -> None:
        if 0 <= int(value) <= 4095:
            with self._lock:
                self.gripper_state[side] = int(value)
                if side not in self.motion_sides:
                    self.gripper_target[side] = int(value)
                self.gripper_ready[side] = True

    def wait_grippers(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if all(self.gripper_ready[side] for side in self.motion_sides):
                    return
            time.sleep(0.02)
        with self._lock:
            missing = [side for side in self.motion_sides if not self.gripper_ready[side]]
        raise TimeoutError(f"gripper state topics are not ready: {missing}")

    def command_gripper(self, side: str, trigger_pressed: bool) -> None:
        if side not in self.motion_sides:
            return
        arm = self.hardware.left if side == "left" else self.hardware.right
        target = gripper_target_for_trigger(arm, trigger_pressed)
        with self._lock:
            self.gripper_target[side] = target
            if self._last_gripper_command[side] == target:
                return
            self._last_gripper_command[side] = target
        message = self._messages["Int32"]()
        message.data = target
        self._gripper_pubs[side].publish(message)

    def publish_buttons(self, buttons: int, *, force: bool = False) -> None:
        buttons = int(buttons) & 0xFF
        if not force and buttons == self._last_buttons:
            return
        self._last_buttons = buttons
        message = self._messages["UInt8"]()
        message.data = buttons
        self._buttons_pub.publish(message)

    def publish_active(self, active: bool, *, force: bool = False) -> None:
        if not force and self._last_active == bool(active):
            return
        self._last_active = bool(active)
        message = self._messages["Bool"]()
        message.data = bool(active)
        self._active_pub.publish(message)

    def publish_sample(
        self,
        joints_by_side: dict[str, np.ndarray],
        *,
        active_sides: set[str] | frozenset[str],
    ) -> None:
        with self._lock:
            if not all(self.gripper_ready[side] for side in self.motion_sides):
                return
            if self.control_mode == "bimanual":
                state_values, action_values = compose_dual_record_vectors(
                    joints_by_side["left"],
                    joints_by_side["right"],
                    self.gripper_state,
                    self.gripper_target,
                    active_sides,
                )
                state_names = DUAL_STATE_NAMES
            else:
                side = next(iter(self.motion_sides))
                joints = np.asarray(joints_by_side[side], dtype=np.float64)
                state_values = np.concatenate(
                    (joints, [float(self.gripper_state[side])])
                )
                action_values = np.concatenate(
                    (joints, [float(self.gripper_target[side])])
                )
                state_names = SINGLE_STATE_NAMES
        stamp = self._node.get_clock().now().to_msg()
        state = self._messages["JointState"]()
        state.header.stamp = stamp
        state.name = list(state_names)
        state.position = state_values.tolist()
        action = self._messages["JointState"]()
        action.header.stamp = stamp
        action.name = list(state_names)
        action.position = action_values.tolist()
        self._state_pub.publish(state)
        self._action_pub.publish(action)
        self.publish_active(True, force=True)

    def close(self) -> None:
        self.publish_active(False, force=True)
        self._executor.shutdown(timeout_sec=1.0)
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._node.destroy_node()
        try:
            self._context.shutdown()
        except Exception:
            pass


class ServoLArm:
    def __init__(
        self,
        sdk: Any,
        utility: Any,
        config: Any,
        reset_joints: np.ndarray,
        receiver: DualControllerReceiver,
        *,
        position_gain: float,
    ):
        self.sdk = sdk
        self.utility = utility
        self.config = config
        self.reset_joints = np.asarray(reset_joints, dtype=np.float64)
        self.receiver = receiver
        self.position_gain = float(position_gain)
        self.robot: Any | None = None
        self.controller: Any | None = None
        self.lower = HARD_LOWER + JOINT_MARGIN_RAD
        self.upper = HARD_UPPER - JOINT_MARGIN_RAD
        self._latest_joints = np.zeros(7, dtype=np.float64)
        self._elbow_current: float | None = None
        self._elbow_target: float | None = None
        self._elbow_generation = 0
        self._elbow_control_enabled = True
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._elbow_only = False
        self._thread_error: BaseException | None = None
        self._reset_thread: threading.Thread | None = None
        self._resetting = False
        self._reset_error: BaseException | None = None
        self._reset_cancel = threading.Event()
        self._receiving = False
        self._servo_configured = False
        self._move_started = False

    @property
    def side(self) -> str:
        return self.config.side

    @property
    def active(self) -> bool:
        return self._active

    @property
    def elbow_only(self) -> bool:
        return self._active and self._elbow_only

    @property
    def dataset_active(self) -> bool:
        return self._active and not self._elbow_only

    @property
    def resetting(self) -> bool:
        return self._resetting

    def _validate_joints(self, joints: np.ndarray, label: str) -> None:
        joints = np.asarray(joints, dtype=np.float64)
        bad = np.flatnonzero(~np.isfinite(joints) | (joints < self.lower) | (joints > self.upper))
        if bad.size:
            index = int(bad[0])
            raise ValueError(
                f"{self.side} {label} J{index + 1}={joints[index]:.6f} outside "
                f"[{self.lower[index]:.6f},{self.upper[index]:.6f}]"
            )

    def _switch_to_nrt(self) -> None:
        last_ec: dict[str, Any] = {}
        for _ in range(20):
            last_ec = {}
            self.robot.setMotionControlMode(self.sdk.MotionControlMode.NrtCommandMode, last_ec)
            if not int(last_ec.get("ec", 0)):
                return
            time.sleep(0.1)
        _check_ec(last_ec, f"{self.side} switch NRT")

    def _restore_nrt_ready(self) -> None:
        self._switch_to_nrt()
        ec: dict[str, Any] = {}
        self.robot.setOperateMode(self.sdk.OperateMode.automatic, ec)
        _check_ec(ec, f"{self.side} automatic mode")
        ec = {}
        self.robot.setPowerState(True, ec)
        _check_ec(ec, f"{self.side} power on")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            ec = {}
            state = self.robot.powerState(ec)
            _check_ec(ec, f"{self.side} powerState")
            if state == self.sdk.PowerState.on:
                return
            if state in (self.sdk.PowerState.estop, self.sdk.PowerState.gstop):
                raise RuntimeError(f"{self.side} safety stop is active")
            time.sleep(0.1)
        raise TimeoutError(f"{self.side} power-on timeout")

    def _load_limits(self) -> None:
        limits_out = self.sdk.PyTypeVectorArrayDouble2()
        ec: dict[str, Any] = {}
        enabled = bool(self.robot.getSoftLimit(limits_out, ec))
        _check_ec(ec, f"{self.side} getSoftLimit")
        if not enabled:
            raise RuntimeError(f"{self.side} controller soft limits are disabled")
        limits = list(limits_out.content())[:7]
        if len(limits) != 7:
            raise RuntimeError(f"{self.side} returned fewer than seven soft limits")
        for index, pair in enumerate(limits):
            lower, upper = list(pair)
            self.lower[index] = max(self.lower[index], float(lower) + JOINT_MARGIN_RAD)
            self.upper[index] = min(self.upper[index], float(upper) - JOINT_MARGIN_RAD)
        if np.any(self.lower >= self.upper):
            raise RuntimeError(f"{self.side} has no usable range after joint margin")

    def connect(self) -> None:
        print(
            f"Connecting {self.side} arm: robot={self.config.robot_ip} "
            f"local={self.config.local_ip}"
        )
        self.robot = self.sdk.ArRobot(self.config.robot_ip, self.config.local_ip)
        try:
            self._restore_nrt_ready()
            self._load_limits()
            joints = self._read_joints_nrt()
            elbow = self._read_elbow_nrt()
            self._validate_joints(joints, "measured")
            with self._state_lock:
                self._latest_joints = joints
                self._elbow_current = elbow
                if self._elbow_target is None:
                    self._elbow_target = elbow
        except BaseException:
            self.robot = None
            raise

    def _read_joints_nrt(self) -> np.ndarray:
        ec: dict[str, Any] = {}
        joints = np.asarray(list(self.robot.jointPos(ec))[:7], dtype=np.float64)
        _check_ec(ec, f"{self.side} jointPos")
        if joints.shape != (7,):
            raise RuntimeError(f"{self.side} returned invalid joint state shape")
        return joints

    def _read_elbow_nrt(self) -> float:
        ec: dict[str, Any] = {}
        cartesian = self.robot.cartPosture(self.sdk.CoordinateType.endInRef, ec)
        _check_ec(ec, f"{self.side} cartPosture(endInRef)")
        elbow = float(cartesian.elbow)
        if not np.isfinite(elbow) or not MIN_ELBOW_RAD <= elbow <= MAX_ELBOW_RAD:
            raise ValueError(f"{self.side} returned invalid elbow angle {elbow}")
        return elbow

    def _read_joints_rt(self) -> np.ndarray:
        updated = self.robot.updateRobotState(timedelta(milliseconds=1))
        while self.robot.updateRobotState(timedelta(milliseconds=0)):
            updated = True
        if not updated:
            raise RuntimeError(f"{self.side} real-time joint state did not update")
        output = self.sdk.PyTypeVectorDouble()
        status = self.robot.getStateData(self.sdk.RtSupportedFields.jointPos_m, output, 7)
        if status != 0:
            raise RuntimeError(f"{self.side} getStateData(jointPos_m) failed: {status}")
        joints = np.asarray(output.content()[:7], dtype=np.float64)
        self._validate_joints(joints, "measured")
        with self._state_lock:
            self._latest_joints = joints
        return joints

    def latest_joints(self) -> np.ndarray:
        with self._state_lock:
            return self._latest_joints.copy()

    def apply_elbow_command(
        self,
        target: float,
        generation: int,
        *,
        apply_target: bool,
        enabled: bool,
    ) -> None:
        if not np.isfinite(target) or not MIN_ELBOW_RAD <= target <= MAX_ELBOW_RAD:
            raise ValueError(f"{self.side} elbow target is outside [-pi, pi]")
        with self._state_lock:
            if apply_target:
                self._elbow_target = float(target)
            if not enabled and self._elbow_current is not None:
                self._elbow_target = self._elbow_current
            self._elbow_generation = int(generation)
            self._elbow_control_enabled = bool(enabled)

    def elbow_status(self) -> tuple[float | None, float | None, int, bool]:
        with self._state_lock:
            return (
                self._elbow_current,
                self._elbow_target,
                self._elbow_generation,
                self._elbow_control_enabled,
            )

    def refresh_if_inactive(self) -> None:
        if self.active or self.resetting or self.robot is None:
            return
        joints = self._read_joints_nrt()
        elbow = self._read_elbow_nrt()
        self._validate_joints(joints, "measured")
        with self._state_lock:
            self._latest_joints = joints
            self._elbow_current = elbow
            if self._elbow_target is None:
                self._elbow_target = elbow

    def _initialize_mapping(self, hand: Any) -> PoseMapping:
        ec: dict[str, Any] = {}
        end_pose = list(self.robot.posture(self.sdk.CoordinateType.endInRef, ec))
        _check_ec(ec, f"{self.side} posture(endInRef)")
        elbow = self._read_elbow_nrt()
        ec = {}
        base_frame = list(self.robot.baseFrame(ec))
        _check_ec(ec, f"{self.side} baseFrame")
        ec = {}
        toolset = self.robot.toolset(ec)
        _check_ec(ec, f"{self.side} toolset")
        matrix = self.utility.postureToTransArray(end_pose)
        rotation, translation = self.utility.arrayToTransMatrix(matrix)
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(translation, dtype=np.float64).reshape(3)
        flange_pose = self.utility.EndInRefToFlanInBase(base_frame, toolset, end_pose)
        command = self.sdk.CartesianPosition()
        command.pos = list(self.utility.postureToTransArray(flange_pose))
        command.hasElbow = True
        command.elbow = elbow
        with self._state_lock:
            self._elbow_current = elbow
            if self._elbow_target is None:
                self._elbow_target = elbow
        hand_rotation = quaternion_to_matrix(hand.orientation_xyzw)
        return PoseMapping(
            base_frame=base_frame,
            toolset=toolset,
            hand_anchor_position=hand.position_m.copy(),
            hand_anchor_rotation=hand_rotation,
            robot_anchor_position=translation.copy(),
            robot_anchor_rotation=rotation.copy(),
            commanded_position=translation.copy(),
            commanded_rotation=rotation.copy(),
            commanded_elbow=elbow,
            elbow_velocity=0.0,
            command=command,
        )

    def _reanchor(self, mapping: PoseMapping, hand: Any) -> None:
        mapping.hand_anchor_position = hand.position_m.copy()
        mapping.hand_anchor_rotation = quaternion_to_matrix(hand.orientation_xyzw)
        mapping.robot_anchor_position = mapping.commanded_position.copy()
        mapping.robot_anchor_rotation = mapping.commanded_rotation.copy()

    def _candidate(
        self, mapping: PoseMapping, hand: Any
    ) -> tuple[np.ndarray, np.ndarray, float, float, Any]:
        raw_position_offset = hand.position_m - mapping.hand_anchor_position
        raw_rotation_offset = (
            quaternion_to_matrix(hand.orientation_xyzw) @ mapping.hand_anchor_rotation.T
        )
        _, raw_angle = matrix_axis_angle(raw_rotation_offset)
        if (
            np.linalg.norm(raw_position_offset) > MAX_RAW_POSITION_OFFSET_M
            or abs(raw_angle) > MAX_RAW_ROTATION_OFFSET_RAD
        ):
            raise ValueError(f"{self.side} controller offset exceeded raw safety range")
        mapped_position_offset = map_position_offset(self.side, raw_position_offset)
        mapped_rotation_offset = map_rotation_offset(self.side, raw_rotation_offset)
        target_position = (
            mapping.robot_anchor_position + self.position_gain * mapped_position_offset
        )
        target_rotation = (
            scaled_rotation(mapped_rotation_offset, gain=ROTATION_GAIN)
            @ mapping.robot_anchor_rotation
        )
        position = limited_position_step(
            mapping.commanded_position,
            target_position,
            deadband_m=POSITION_DEADBAND_M,
            max_step_m=MAX_POSITION_STEP_M,
        )
        rotation = limited_rotation_step(
            mapping.commanded_rotation,
            target_rotation,
            deadband_rad=ROTATION_DEADBAND_RAD,
            max_step_rad=MAX_ROTATION_STEP_RAD,
        )
        end_matrix = self.utility.transMatrixToArray(rotation, position)
        end_pose = self.utility.transArrayToPosture(end_matrix)
        flange_pose = self.utility.EndInRefToFlanInBase(
            mapping.base_frame, mapping.toolset, end_pose
        )
        command = self.sdk.CartesianPosition()
        command.pos = list(self.utility.postureToTransArray(flange_pose))
        with self._state_lock:
            elbow_target = (
                mapping.commanded_elbow
                if self._elbow_target is None
                else self._elbow_target
            )
        elbow, elbow_velocity = next_elbow_target(
            mapping.commanded_elbow,
            elbow_target,
            mapping.elbow_velocity,
        )
        command.hasElbow = True
        command.elbow = elbow
        return position, rotation, elbow, elbow_velocity, command

    def _elbow_only_candidate(
        self, mapping: PoseMapping
    ) -> tuple[np.ndarray, np.ndarray, float, float, Any]:
        with self._state_lock:
            target = (
                mapping.commanded_elbow
                if self._elbow_target is None
                else self._elbow_target
            )
            elbow_enabled = self._elbow_control_enabled
        if not elbow_enabled:
            mapping.command.hasElbow = False
            return (
                mapping.commanded_position,
                mapping.commanded_rotation,
                mapping.commanded_elbow,
                0.0,
                mapping.command,
            )
        elbow, elbow_velocity = next_elbow_target(
            mapping.commanded_elbow,
            target,
            mapping.elbow_velocity,
        )
        command = mapping.command
        command.hasElbow = True
        command.elbow = elbow
        return (
            mapping.commanded_position,
            mapping.commanded_rotation,
            elbow,
            elbow_velocity,
            command,
        )

    def start_servo(self, *, elbow_only: bool = False) -> None:
        if self.active:
            raise RuntimeError(f"{self.side} ServoL is already active")
        if self.resetting:
            raise RuntimeError(f"{self.side} reset is still running")
        hand = self.receiver.latest(self.side)
        if not hand.pose_fresh(CONTROLLER_TIMEOUT_S):
            raise TimeoutError(f"{self.side} controller pose is stale")
        mapping = self._initialize_mapping(hand)
        self.receiver.begin_control_transition()
        self.receiver.set_teleop_active(self.side, not elbow_only)
        try:
            ec: dict[str, Any] = {}
            self.robot.setRtNetworkTolerance(20, ec)
            _check_ec(ec, f"{self.side} setRtNetworkTolerance")
            ec = {}
            self.robot.setMotionControlMode(self.sdk.MotionControlMode.RtCommandMode, ec)
            _check_ec(ec, f"{self.side} switch RT")
            self.controller = self.robot.getRtMotionController()
            self.robot.startReceiveRobotState(
                timedelta(milliseconds=1), [self.sdk.RtSupportedFields.jointPos_m]
            )
            self._receiving = True
            while self.robot.updateRobotState(timedelta(milliseconds=0)):
                pass
            self._read_joints_rt()
            ec = {}
            self.controller.setServoJoint(PLAN_PERIOD_S, PLAN_PERIOD_S, 0, ec)
            _check_ec(ec, f"{self.side} setServoJoint for ServoL")
            self._servo_configured = True
            if elbow_only:
                self.receiver.wait_for_fresh_pose(
                    self.side,
                    CONTROL_TRANSITION_RECOVERY_TIMEOUT_S,
                    freshness_timeout_s=CONTROLLER_TIMEOUT_S,
                    after_sequence=hand.sequence,
                )
            else:
                activation_hands = self.receiver.wait_for_active_poses_after(
                    self.side,
                    hand.sequence,
                    CONTROL_TRANSITION_RECOVERY_TIMEOUT_S,
                    freshness_timeout_s=CONTROLLER_TIMEOUT_S,
                )
                self._reanchor(mapping, activation_hands[self.side])
            self.controller.startMove(self.sdk.RtControllerMode.cartesianPosition)
            self._move_started = True
            self._stop.clear()
            self._thread_error = None
            self._elbow_only = bool(elbow_only)
            self._active = True
            self._thread = threading.Thread(
                target=self._servo_loop,
                args=(mapping,),
                name=f"pico-{self.side}-servol",
                daemon=True,
            )
            self._thread.start()
            print(
                f"[{self.side.upper()}] ServoL enabled "
                f"({'web elbow hold; Grip enters teleop' if elbow_only else 'Grip teleop'})"
            )
        except BaseException:
            self._cleanup_rt()
            self.receiver.set_teleop_active(self.side, False)
            self._active = False
            self._elbow_only = False
            try:
                self._restore_nrt_ready()
            except Exception as restore_error:
                print(
                    f"[{self.side.upper()}] failed to restore NRT after start error: "
                    f"{restore_error}",
                    file=sys.stderr,
                )
            raise
        finally:
            self.receiver.end_control_transition()

    def _servo_loop(self, mapping: PoseMapping) -> None:
        next_tick = time.monotonic()
        transition_epoch, _ = self.receiver.control_transition_state()
        rejected = 0
        last_reason = ""
        try:
            while not self._stop.is_set():
                current_epoch, transition_active = self.receiver.control_transition_state()
                transition_changed = current_epoch != transition_epoch
                hand = self.receiver.latest(self.side)
                if not hand.pose_fresh(CONTROLLER_TIMEOUT_S):
                    if transition_changed or transition_active:
                        hand = self.receiver.wait_for_fresh_pose(
                            self.side,
                            CONTROL_TRANSITION_RECOVERY_TIMEOUT_S,
                            freshness_timeout_s=CONTROLLER_TIMEOUT_S,
                        )
                        next_tick = time.monotonic()
                    else:
                        raise TimeoutError(
                            f"{self.side} PICO pose exceeded "
                            f"{CONTROLLER_TIMEOUT_S * 1000:.0f} ms"
                        )
                if transition_changed:
                    next_tick = time.monotonic()
                transition_epoch, _ = self.receiver.control_transition_state()
                self._read_joints_rt()
                if self.controller.hasMotionError():
                    raise RuntimeError(f"{self.side} ServoL controller motion error")
                try:
                    if self._elbow_only:
                        position, rotation, elbow, elbow_velocity, command = (
                            self._elbow_only_candidate(mapping)
                        )
                    else:
                        position, rotation, elbow, elbow_velocity, command = self._candidate(
                            mapping, hand
                        )
                except ValueError as error:
                    self.controller.sendCommand(mapping.command)
                    rejected += 1
                    reason = str(error)
                    if reason != last_reason or rejected % 25 == 1:
                        print(f"[{self.side.upper()}] hold last target: {reason}", file=sys.stderr)
                        last_reason = reason
                    if rejected >= MAX_REJECTED_CYCLES:
                        raise RuntimeError(f"{self.side} invalid target persisted for 2 seconds")
                else:
                    self.controller.sendCommand(command)
                    mapping.commanded_position = position
                    mapping.commanded_rotation = rotation
                    mapping.commanded_elbow = elbow
                    mapping.elbow_velocity = elbow_velocity
                    mapping.command = command
                    with self._state_lock:
                        self._elbow_current = elbow
                    rejected = 0
                    last_reason = ""
                latest_epoch, transition_active = self.receiver.control_transition_state()
                if latest_epoch != transition_epoch:
                    transition_epoch = latest_epoch
                    next_tick = time.monotonic()
                    continue
                next_tick += PLAN_PERIOD_S
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                elif delay < -MAX_CYCLE_LATENESS_S:
                    late_epoch, transition_active = self.receiver.control_transition_state()
                    if late_epoch != transition_epoch or transition_active:
                        transition_epoch = late_epoch
                        next_tick = time.monotonic()
                    else:
                        raise RuntimeError(
                            f"{self.side} 50 Hz ServoL loop late by {-delay:.3f}s"
                        )
        except BaseException as error:
            if not self._stop.is_set():
                self._thread_error = error
        finally:
            self.receiver.begin_control_transition()
            try:
                self._cleanup_rt()
                self.receiver.set_teleop_active(self.side, False)
                self._active = False
                self._elbow_only = False
                if self._thread_error is not None:
                    try:
                        self._restore_nrt_ready()
                    except Exception as restore_error:
                        print(
                            f"[{self.side.upper()}] failed to restore NRT after "
                            f"background error: {restore_error}",
                            file=sys.stderr,
                        )
            finally:
                self.receiver.end_control_transition()

    def _cleanup_rt(self) -> None:
        if self.controller is not None and self._move_started:
            try:
                self.controller.stopMove()
            except Exception as error:
                print(f"[{self.side.upper()}] stopMove failed: {error}", file=sys.stderr)
        if self.controller is not None and self._servo_configured:
            try:
                self.controller.stopServoJoint()
            except Exception as error:
                print(f"[{self.side.upper()}] stopServoJoint failed: {error}", file=sys.stderr)
        if self.robot is not None and self._receiving:
            try:
                self.robot.stopReceiveRobotState()
            except Exception as error:
                print(f"[{self.side.upper()}] stopReceiveRobotState failed: {error}", file=sys.stderr)
        self._move_started = False
        self._servo_configured = False
        self._receiving = False
        self.controller = None
        if self.robot is not None:
            try:
                ec: dict[str, Any] = {}
                self.robot.setMotionControlMode(self.sdk.MotionControlMode.Idle, ec)
                _check_ec(ec, f"{self.side} switch Idle")
            except Exception as error:
                print(f"[{self.side.upper()}] switch Idle failed: {error}", file=sys.stderr)

    def stop_servo(self, *, reconnect: bool) -> None:
        self.receiver.begin_control_transition()
        try:
            self._stop.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=3.0)
            if self._thread is not None and self._thread.is_alive():
                raise TimeoutError(f"{self.side} ServoL thread did not stop")
            self._thread = None
            self._active = False
            self._elbow_only = False
            self.receiver.set_teleop_active(self.side, False)
            if reconnect:
                self.reconnect()
            print(f"[{self.side.upper()}] ServoL stopped")
        finally:
            self.receiver.end_control_transition()

    def reconnect(self) -> None:
        self.receiver.begin_control_transition()
        try:
            self.robot = None
            self.controller = None
            time.sleep(0.2)
            self.connect()
        finally:
            self.receiver.end_control_transition()

    def _reset_motion(self) -> None:
        self._validate_joints(self.reset_joints, "reset target")
        self._restore_nrt_ready()
        ec: dict[str, Any] = {}
        self.robot.moveReset(ec)
        _check_ec(ec, f"{self.side} moveReset")
        command = self.sdk.MoveAbsJCommand(self.reset_joints.tolist(), 1000, 10)
        command_id = self.sdk.PyString()
        ec = {}
        self.robot.moveAppend(command, command_id, ec)
        _check_ec(ec, f"{self.side} moveAppend(reset)")
        ec = {}
        self.robot.moveStart(ec)
        _check_ec(ec, f"{self.side} moveStart(reset)")
        time.sleep(0.1)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if self._reset_cancel.is_set():
                ec = {}
                self.robot.stop(ec)
                _check_ec(ec, f"{self.side} stop cancelled reset")
                ec = {}
                self.robot.moveReset(ec)
                _check_ec(ec, f"{self.side} moveReset(cancelled reset)")
                raise InterruptedError(f"{self.side} reset cancelled")
            ec = {}
            state = self.robot.operationState(ec)
            _check_ec(ec, f"{self.side} operationState(reset)")
            if state == self.sdk.OperationState.idle:
                joints = self._read_joints_nrt()
                self._validate_joints(joints, "post-reset")
                with self._state_lock:
                    self._latest_joints = joints
                print(f"[{self.side.upper()}] reset complete")
                return
            time.sleep(0.1)
        ec = {}
        self.robot.stop(ec)
        raise TimeoutError(f"{self.side} reset timed out")

    def request_reset(self) -> None:
        if self.resetting:
            print(f"[{self.side.upper()}] reset already running")
            return
        reconnect_before_reset = self.active
        if self.active:
            # Stop the 50 Hz command stream immediately. Recreating the SDK
            # session happens in the reset worker so the other controller's
            # button state machine remains responsive.
            self.stop_servo(reconnect=False)
        self._reset_cancel.clear()
        self._reset_error = None
        self._resetting = True

        def worker() -> None:
            try:
                if reconnect_before_reset:
                    self.reconnect()
                self._reset_motion()
            except BaseException as error:
                if not self._reset_cancel.is_set():
                    self._reset_error = error
            finally:
                self._resetting = False

        self._reset_thread = threading.Thread(
            target=worker,
            name=f"pico-{self.side}-reset",
            daemon=True,
        )
        self._reset_thread.start()

    def raise_thread_error(self) -> None:
        if self._thread_error is not None:
            error = self._thread_error
            self._thread_error = None
            raise RuntimeError(f"{self.side} ServoL stopped unexpectedly") from error
        if self._reset_error is not None:
            error = self._reset_error
            self._reset_error = None
            raise RuntimeError(f"{self.side} reset failed") from error

    def close(self) -> None:
        if self._thread is not None:
            try:
                self.stop_servo(reconnect=False)
            except Exception as error:
                print(f"[{self.side.upper()}] cleanup failed: {error}", file=sys.stderr)
        if self._reset_thread is not None and self._reset_thread.is_alive():
            self._reset_cancel.set()
            self._reset_thread.join(timeout=3.0)
            if self._reset_thread.is_alive():
                print(
                    f"[{self.side.upper()}] reset thread did not stop during cleanup",
                    file=sys.stderr,
                )
        self._reset_thread = None
        if self.robot is not None:
            try:
                self._switch_to_nrt()
            except Exception as error:
                print(f"[{self.side.upper()}] NRT restore failed: {error}", file=sys.stderr)
        self.robot = None


def _exception_chain(error: BaseException) -> str:
    messages: list[str] = []
    current: BaseException | None = error
    while current is not None:
        text = str(current).strip() or current.__class__.__name__
        messages.append(f"{current.__class__.__name__}: {text}")
        current = current.__cause__
    return " <- ".join(messages)


def _put_worker_status(status_queue: Any, message: dict[str, Any], *, important: bool) -> None:
    try:
        if important:
            status_queue.put(message, timeout=1.0)
        else:
            status_queue.put_nowait(message)
    except queue.Full:
        if important:
            print(
                f"[{message.get('side', 'worker').upper()}] status queue is full",
                file=sys.stderr,
            )


def _arm_worker_main(
    side: str,
    udp_port: int,
    position_gain: float,
    reset_joints_values: Any,
    stop_event: Any,
    status_queue: Any,
    elbow_command_path: str | None,
    allow_motion: bool = True,
) -> None:
    """Own exactly one xCoreSDK instance in an isolated spawned process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    hardware = HardwareConfig()
    arm_config = hardware.left if side == "left" else hardware.right
    reset_joints = _validated_reset_joints(reset_joints_values, side)
    receiver = DualControllerReceiver("127.0.0.1", udp_port)
    arm: ServoLArm | None = None
    try:
        sdk = _load_sdk(hardware.sdk_python_path)
        from xCoreSDK_python import utility

        receiver.start()
        initial_hand = receiver.wait_for_fresh_pose(
            side,
            20.0,
            freshness_timeout_s=20.0,
        )
        arm = ServoLArm(
            sdk,
            utility,
            arm_config,
            reset_joints,
            receiver,
            position_gain=position_gain,
        )
        arm.connect()
        elbow_current, elbow_target, elbow_generation, elbow_control_enabled = (
            arm.elbow_status()
        )
        edges = ButtonEdgeTracker()
        edges.update(side, initial_hand.buttons)
        _put_worker_status(
            status_queue,
            {
                "side": side,
                "kind": "ready",
                "active": False,
                "servo_active": False,
                "resetting": False,
                "joints": arm.latest_joints().tolist(),
                "elbow_current": elbow_current,
                "elbow_target": elbow_target,
                "elbow_generation": elbow_generation,
                "elbow_control_enabled": elbow_control_enabled,
            },
            important=True,
        )
        next_status = time.monotonic()
        next_inactive_refresh = time.monotonic()
        next_elbow_command = time.monotonic()
        while not stop_event.is_set():
            try:
                arm.raise_thread_error()
            except Exception as error:
                detail = _exception_chain(error)
                print(f"[{side.upper()}] background operation failed: {detail}", file=sys.stderr)
                _put_worker_status(
                    status_queue,
                    {"side": side, "kind": "error", "message": detail},
                    important=True,
                )

            hand = receiver.latest(side)
            if allow_motion and hand.packet_fresh(CONTROLLER_TIMEOUT_S):
                edge = edges.update(side, hand.buttons)
                if edge.down_rising:
                    print(f"[{side.upper()}] Down pressed: stop and reset only {side} arm")
                    try:
                        arm.request_reset()
                    except Exception as error:
                        detail = _exception_chain(error)
                        print(f"[{side.upper()}] reset request failed: {detail}", file=sys.stderr)
                        _put_worker_status(
                            status_queue,
                            {"side": side, "kind": "error", "message": detail},
                            important=True,
                        )
                elif edge.grip_rising:
                    try:
                        if arm.resetting:
                            print(f"[{side.upper()}] Grip ignored while reset is running")
                        elif arm.elbow_only:
                            arm.stop_servo(reconnect=True)
                            arm.start_servo()
                        elif arm.active:
                            arm.stop_servo(reconnect=True)
                        else:
                            arm.start_servo()
                    except Exception as error:
                        detail = _exception_chain(error)
                        print(f"[{side.upper()}] ServoL toggle failed: {detail}", file=sys.stderr)
                        _put_worker_status(
                            status_queue,
                            {"side": side, "kind": "error", "message": detail},
                            important=True,
                        )

            now = time.monotonic()
            if now >= next_elbow_command:
                _, _, last_generation, _ = arm.elbow_status()
                request = read_elbow_command(
                    elbow_command_path,
                    side,
                    last_generation,
                )
                if request is not None:
                    generation, target, apply_target, control_enabled = request
                    arm.apply_elbow_command(
                        target,
                        generation,
                        apply_target=apply_target,
                        enabled=control_enabled,
                    )
                    if apply_target and control_enabled:
                        print(
                            f"[{side.upper()}] web elbow target="
                            f"{np.rad2deg(target):.3f} deg"
                        )
                    (
                        elbow_current,
                        elbow_target,
                        elbow_generation,
                        elbow_control_enabled,
                    ) = arm.elbow_status()
                    _put_worker_status(
                        status_queue,
                        {
                            "side": side,
                            "kind": "state",
                            "active": arm.dataset_active,
                            "servo_active": arm.active,
                            "resetting": arm.resetting,
                            "joints": arm.latest_joints().tolist(),
                            "elbow_current": elbow_current,
                            "elbow_target": elbow_target,
                            "elbow_generation": elbow_generation,
                            "elbow_control_enabled": elbow_control_enabled,
                        },
                        important=True,
                    )
                    if not control_enabled and arm.elbow_only:
                        arm.stop_servo(reconnect=True)
                    elif (
                        apply_target
                        and control_enabled
                        and allow_motion
                        and not arm.active
                        and not arm.resetting
                    ):
                        arm.start_servo(elbow_only=True)
                next_elbow_command = now + 0.05

            if now >= next_inactive_refresh:
                # An arm whose Grip is off still contributes its measured
                # hold pose to every sample produced by the other arm.
                arm.refresh_if_inactive()
                next_inactive_refresh = now + 0.1
            if now >= next_status:
                (
                    elbow_current,
                    elbow_target,
                    elbow_generation,
                    elbow_control_enabled,
                ) = arm.elbow_status()
                _put_worker_status(
                    status_queue,
                    {
                        "side": side,
                        "kind": "state",
                        "active": arm.dataset_active,
                        "servo_active": arm.active,
                        "resetting": arm.resetting,
                        "joints": arm.latest_joints().tolist(),
                        "elbow_current": elbow_current,
                        "elbow_target": elbow_target,
                        "elbow_generation": elbow_generation,
                        "elbow_control_enabled": elbow_control_enabled,
                    },
                    important=False,
                )
                next_status = now + PLAN_PERIOD_S
            stop_event.wait(0.005)
    except BaseException as error:
        detail = _exception_chain(error)
        print(f"[{side.upper()}] worker fatal error: {detail}", file=sys.stderr)
        traceback.print_exc()
        _put_worker_status(
            status_queue,
            {"side": side, "kind": "fatal", "message": detail},
            important=True,
        )
    finally:
        if arm is not None:
            arm.close()
        receiver.stop()
        _put_worker_status(
            status_queue,
            {"side": side, "kind": "exit"},
            important=True,
        )


def run_self_test() -> None:
    import struct

    for hand_id, expected in ((0, "left"), (1, "right")):
        packet = bytearray(38)
        packet[0] = hand_id
        struct.pack_into("<9f", packet, 1, 100.0, -200.0, 300.0, 0, 0, 0, 1, 0, 0)
        packet[37] = (1 << TRIGGER_BIT) | (1 << GRIP_BIT) | (1 << DOWN_BIT) | (1 << UP_BIT)
        parsed = parse_controller_packet(bytes(packet))
        assert parsed.side == expected
        np.testing.assert_allclose(parsed.position_m, [0.1, -0.2, 0.3])
        assert parsed.trigger and parsed.grip and parsed.down and parsed.up
    tracker = ButtonEdgeTracker()
    assert not tracker.update("left", 0).grip_rising
    assert tracker.update("left", 1 << GRIP_BIT).grip_rising
    assert not tracker.update("left", 1 << GRIP_BIT).grip_rising
    assert tracker.update("left", 1 << DOWN_BIT).down_rising
    stepped = limited_position_step(np.zeros(3), np.array([0.01, 0, 0]))
    np.testing.assert_allclose(stepped, [MAX_POSITION_STEP_M, 0, 0])
    np.testing.assert_allclose(
        map_position_offset("left", np.array([0.1, 0.2, 0.3])),
        [0.1, -0.3, 0.2],
    )
    mapped_axis, mapped_angle = matrix_axis_angle(
        map_rotation_offset(
            "left", axis_angle_matrix(np.array([0.0, 0.0, -1.0]), 0.2)
        )
    )
    np.testing.assert_allclose(mapped_axis, [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(mapped_angle, 0.2, atol=1e-8)
    axis, angle = matrix_axis_angle(
        scaled_rotation(axis_angle_matrix(np.array([0, 0, 1]), np.deg2rad(90)))
    )
    np.testing.assert_allclose(abs(angle), np.deg2rad(22.5), atol=1e-8)
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-8
    hardware = HardwareConfig()
    for arm in (hardware.left, hardware.right):
        assert gripper_target_for_trigger(arm, True) == int(arm.gripper_open_position)
        assert gripper_target_for_trigger(arm, False) == int(arm.gripper_closed_position)
    elbow = 0.0
    elbow_velocity = 0.0
    for _ in range(200):
        previous = elbow
        elbow, elbow_velocity = next_elbow_target(elbow, 1.0, elbow_velocity)
        assert 0.0 < elbow - previous <= MAX_ELBOW_STEP_RAD + 1e-12
        assert 0.0 < elbow_velocity <= MAX_ELBOW_VELOCITY_RAD_S + 1e-12
    print(
        "PASS: dual PICO parser, button edges, position/rotation filters, "
        "pressed-open/released-closed gripper mapping, elbow smoothing, "
        "and 0.01-degree elbow step limiting"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5050)
    parser.add_argument("--controller-timeout-s", type=float, default=15.0)
    parser.add_argument("--position-gain", type=float, default=_position_gain_default())
    parser.add_argument(
        "--left-reset-joints",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=LEFT_TELEOP_RESET_JOINTS.tolist(),
        help="left-arm Down/reset target in radians",
    )
    parser.add_argument(
        "--right-reset-joints",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=RIGHT_TELEOP_RESET_JOINTS.tolist(),
        help="right-arm Down/reset target in radians",
    )
    parser.add_argument("--confirm-motion", default="")
    parser.add_argument(
        "--control-mode",
        choices=("bimanual", "left_only", "right_only"),
        default="bimanual",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    expected_confirmation = CONFIRMATIONS[args.control_mode]
    if args.confirm_motion != expected_confirmation:
        raise RuntimeError(
            f"real teleoperation in {args.control_mode} requires "
            f"--confirm-motion={expected_confirmation}"
        )
    if not 0.1 <= args.position_gain <= 1.0:
        raise ValueError("--position-gain must be in [0.1, 1.0]")
    if not 1 <= args.udp_port <= 65535:
        raise ValueError("--udp-port must be in [1, 65535]")
    reset_joints_by_side = {
        "left": _validated_reset_joints(args.left_reset_joints, "left"),
        "right": _validated_reset_joints(args.right_reset_joints, "right"),
    }
    control_sides = (
        ("left", "right")
        if args.control_mode == "bimanual"
        else ("left",)
        if args.control_mode == "left_only"
        else ("right",)
    )
    button_side = "left" if args.control_mode == "left_only" else "right"

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    hardware = HardwareConfig()
    elbow_command_path = os.environ.get("XCORE_TELEOP_ELBOW_COMMAND_PATH") or None
    elbow_status_path = os.environ.get("XCORE_TELEOP_ELBOW_STATUS_PATH") or None
    context = mp.get_context("spawn")
    worker_stop = context.Event()
    status_queue = context.Queue(maxsize=512)
    receiver = DualControllerReceiver(
        args.udp_host,
        args.udp_port,
        forward_addresses=tuple(
            ("127.0.0.1", WORKER_UDP_PORTS[side]) for side in control_sides
        ),
    )
    bridge: RosTeleopBridge | None = None
    workers: dict[str, Any] = {}
    worker_states: dict[str, dict[str, Any]] = {
        side: {
            "active": False,
            "servo_active": False,
            "resetting": False,
            "joints": None,
            "ready": False,
            "exited": False,
            "elbow_current": None,
            "elbow_target": None,
            "elbow_generation": 0,
            "elbow_control_enabled": True,
        }
        for side in control_sides
    }

    def handle_worker_status(message: dict[str, Any]) -> None:
        side = str(message.get("side", ""))
        if side not in worker_states:
            return
        kind = message.get("kind")
        state = worker_states[side]
        if kind in ("ready", "state"):
            state["active"] = bool(message.get("active", False))
            state["servo_active"] = bool(message.get("servo_active", False))
            state["resetting"] = bool(message.get("resetting", False))
            joints = message.get("joints")
            if isinstance(joints, list) and len(joints) == 7:
                state["joints"] = np.asarray(joints, dtype=np.float64)
            for key in ("elbow_current", "elbow_target"):
                value = message.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    state[key] = float(value)
            state["elbow_generation"] = int(message.get("elbow_generation", 0))
            state["elbow_control_enabled"] = bool(
                message.get("elbow_control_enabled", True)
            )
            if kind == "ready":
                state["ready"] = True
                print(f"[{side.upper()}] isolated SDK worker ready")
        elif kind in ("error", "fatal"):
            print(
                f"[{side.upper()}] isolated worker {kind}: {message.get('message', '')}",
                file=sys.stderr,
            )
        elif kind == "exit":
            state["active"] = False
            state["servo_active"] = False
            state["exited"] = True

    def drain_worker_status() -> None:
        while True:
            try:
                message = status_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(message, dict):
                handle_worker_status(message)

    try:
        receiver.start()
        print(f"Waiting for {', '.join(control_sides)} PICO controller on UDP {args.udp_port}...")
        if args.control_mode == "bimanual":
            initial_hands = receiver.wait_both(args.controller_timeout_s)
        else:
            initial_hands = {
                control_sides[0]: receiver.wait_for_fresh_pose(
                    control_sides[0],
                    args.controller_timeout_s,
                    freshness_timeout_s=args.controller_timeout_s,
                )
            }
        for side in control_sides:
            worker = context.Process(
                target=_arm_worker_main,
                args=(
                    side,
                    WORKER_UDP_PORTS[side],
                    args.position_gain,
                    reset_joints_by_side[side],
                    worker_stop,
                    status_queue,
                    elbow_command_path,
                    True,
                ),
                name=f"pico-{side}-sdk-worker",
            )
            worker.start()
            workers[side] = worker

        ready_deadline = time.monotonic() + 30.0
        while not all(state["ready"] for state in worker_states.values()):
            remaining = ready_deadline - time.monotonic()
            if remaining <= 0:
                missing = [
                    side for side, state in worker_states.items() if not state["ready"]
                ]
                raise TimeoutError(f"isolated arm workers not ready: {missing}")
            try:
                message = status_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                for side, worker in workers.items():
                    if worker.exitcode is not None and not worker_states[side]["ready"]:
                        raise RuntimeError(
                            f"{side} isolated SDK worker exited with code {worker.exitcode}"
                        )
                continue
            if isinstance(message, dict):
                handle_worker_status(message)

        bridge = RosTeleopBridge(hardware, args.control_mode)
        bridge.wait_grippers()

        edges = ButtonEdgeTracker()
        for side in control_sides:
            edges.update(side, initial_hands[side].buttons)
        bridge.publish_buttons(initial_hands[button_side].buttons)
        for side, hand in initial_hands.items():
            bridge.command_gripper(side, button_pressed(hand.buttons, TRIGGER_BIT))

        print("Ready:")
        print(
            f"  control={CONTROL_IMPLEMENTATION}; control_mode={args.control_mode}; "
            f"motion_sides={sorted(bridge.motion_sides)}"
        )
        for side in control_sides:
            print(f"  {side.title()} Grip: toggle ServoL; Down: reset {side} arm")
        print(
            f"  {button_side.title()} Up/B: forwarded on "
            f"{TELEMETRY_TOPICS[args.control_mode]['buttons']} for recording"
        )
        print(
            f"  gain={args.position_gain:.3f}, rotation_gain={ROTATION_GAIN:.2f}, "
            f"max_step={MAX_POSITION_STEP_M * 1000:.1f}mm/"
            f"{np.rad2deg(MAX_ROTATION_STEP_RAD):.1f}deg @ 50Hz"
        )
        print("  left translation map: robot XYZ = [PICO X, -PICO Z, PICO Y]")
        print("  left rotation map: robot R = M @ PICO R @ M.T")
        print(
            "  reset joints (rad): "
            + " ".join(
                f"{side}={reset_joints_by_side[side].tolist()}"
                for side in control_sides
            )
        )

        next_sample = time.monotonic()
        next_button_publish = time.monotonic()
        next_elbow_status = time.monotonic()
        while not stop.is_set():
            hands = receiver.latest_both()
            drain_worker_status()

            for side in control_sides:
                hand = hands[side]
                if not hand.packet_fresh(CONTROLLER_TIMEOUT_S):
                    continue
                edge = edges.update(side, hand.buttons)
                bridge.command_gripper(side, button_pressed(hand.buttons, TRIGGER_BIT))
                if side == button_side:
                    bridge.publish_buttons(hand.buttons)
                    if edge.up_rising:
                        print(f"[{side.upper()}] Up/B forwarded for recording")
                elif edge.up_rising:
                    print(f"[{side.upper()}] Up ignored")

            active_sides = frozenset(
                side for side, state in worker_states.items() if state["active"]
            )
            any_active = bool(active_sides)
            any_resetting = any(state["resetting"] for state in worker_states.values())
            recording_active = any_active and not any_resetting
            bridge.publish_active(recording_active)
            now = time.monotonic()
            if now >= next_elbow_status:
                write_elbow_status(elbow_status_path, worker_states)
                next_elbow_status = now + 0.1
            if now >= next_button_publish and hands[button_side].packet_fresh(
                CONTROLLER_TIMEOUT_S
            ):
                bridge.publish_buttons(hands[button_side].buttons, force=True)
                next_button_publish = now + 0.1
            joints_ready = all(
                worker_states[side]["joints"] is not None for side in control_sides
            )
            if recording_active and joints_ready and now >= next_sample:
                bridge.publish_sample(
                    {
                        side: worker_states[side]["joints"]
                        for side in control_sides
                    },
                    active_sides=active_sides,
                )
                next_sample = now + PLAN_PERIOD_S
            elif not recording_active:
                next_sample = now
            stop.wait(0.005)
    finally:
        worker_stop.set()
        for worker in workers.values():
            worker.join(timeout=8.0)
        for side, worker in workers.items():
            if worker.is_alive():
                print(
                    f"[{side.upper()}] worker did not stop cleanly; terminating it",
                    file=sys.stderr,
                )
                worker.terminate()
                worker.join(timeout=2.0)
        if bridge is not None:
            bridge.close()
        receiver.stop()
    print("Dual-arm PICO ServoL exited cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
