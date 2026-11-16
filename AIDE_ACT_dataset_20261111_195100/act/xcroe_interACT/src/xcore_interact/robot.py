from __future__ import annotations

import importlib
import logging
import math
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schema import HardwareConfig

logger = logging.getLogger(__name__)

HARD_LOWER = np.array([-3.10669, -2.0944, -3.10669, -1.0472, -3.10669, -1.0472, -1.0472])
HARD_UPPER = np.array([3.10669, 2.0944, 3.10669, 2.53073, 3.10669, 1.0472, 1.0472])
LIMIT_MARGIN = math.radians(12.0)
CONTROL_MODES = frozenset({"bimanual", "right_only"})


def _check_ec(ec: dict[str, Any], operation: str) -> None:
    if int(ec.get("ec", 0)):
        raise RuntimeError(f"{operation} failed: {ec.get('ec')} {ec.get('message', '')}")


def _load_sdk(configured: Path) -> Any:
    candidates = (
        configured,
        Path("/home/niic/xcoresdk_python-v0.7.1.ar_6/Release/linux/arm"),
        Path("/home/niic/aide/hardware/arm/xCoreSDK-v0.7.1.ar_6/Release/linux/arm"),
    )
    for candidate in candidates:
        if candidate.is_dir():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)
            try:
                return importlib.import_module("xCoreSDK_python")
            except ImportError:
                continue
    raise ImportError(f"xCoreSDK_python was not found in: {[str(path) for path in candidates]}")


class XCoreArm:
    """One safe absolute-joint ServoJ session. Motion is off by default."""

    def __init__(self, sdk: Any, config: Any, *, control_hz: float = 50.0):
        self.sdk = sdk
        self.config = config
        self.control_hz = control_hz
        self.robot: Any | None = None
        self.controller: Any | None = None
        self.enable_motion = False
        self._receiving = False
        self._servo = False
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._state = np.zeros(7, dtype=np.float64)
        self._target = np.zeros(7, dtype=np.float64)
        self._last_command = np.zeros(7, dtype=np.float64)
        self._last_target_s = 0.0
        self.lower = HARD_LOWER + LIMIT_MARGIN
        self.upper = HARD_UPPER - LIMIT_MARGIN

    def _read_locked(self) -> np.ndarray:
        if self._receiving:
            updated = self.robot.updateRobotState(timedelta(milliseconds=1))
            while self.robot.updateRobotState(timedelta(milliseconds=0)):
                updated = True
            if not updated:
                raise RuntimeError(f"{self.config.side} robot state did not update")
            values = self.sdk.PyTypeVectorDouble()
            status = self.robot.getStateData(
                self.sdk.RtSupportedFields.jointPos_m, values, 7
            )
            if status != 0:
                raise RuntimeError(f"{self.config.side} getStateData(jointPos_m) failed: {status}")
            joints = np.asarray(values.content()[:7], dtype=np.float64)
        else:
            ec: dict[str, Any] = {}
            joints = np.asarray(list(self.robot.jointPos(ec))[:7], dtype=np.float64)
            _check_ec(ec, f"{self.config.side} jointPos")
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise RuntimeError(f"{self.config.side} returned invalid joint state")
        self._validate(joints, "measured")
        with self._state_lock:
            self._state = joints
        return joints

    def _validate(self, joints: np.ndarray, label: str) -> None:
        bad = np.flatnonzero((joints < self.lower) | (joints > self.upper) | ~np.isfinite(joints))
        if bad.size:
            index = int(bad[0])
            raise ValueError(
                f"{self.config.side} {label} J{index + 1}={joints[index]:.5f} outside "
                f"[{self.lower[index]:.5f},{self.upper[index]:.5f}]"
            )

    def connect(self, *, enable_motion: bool) -> None:
        self.enable_motion = enable_motion
        self.robot = self.sdk.ArRobot(self.config.robot_ip, self.config.local_ip)
        with self._lock:
            initial = self._read_locked()
            self._target = self._last_command = initial.copy()
            if enable_motion:
                ec: dict[str, Any] = {}
                self.robot.setOperateMode(self.sdk.OperateMode.automatic, ec)
                _check_ec(ec, f"{self.config.side} automatic mode")
                ec = {}
                self.robot.setRtNetworkTolerance(20, ec)
                _check_ec(ec, f"{self.config.side} network tolerance")
                ec = {}
                self.robot.setMotionControlMode(self.sdk.MotionControlMode.RtCommandMode, ec)
                _check_ec(ec, f"{self.config.side} RT mode")
                ec = {}
                self.robot.setPowerState(True, ec)
                _check_ec(ec, f"{self.config.side} power on")
                self.controller = self.robot.getRtMotionController()
                self.robot.startReceiveRobotState(
                    timedelta(milliseconds=1), [self.sdk.RtSupportedFields.jointPos_m]
                )
                self._receiving = True
                while self.robot.updateRobotState(timedelta(milliseconds=0)):
                    pass
                initial = self._read_locked()
                self._target = self._last_command = initial.copy()
                period = 1.0 / self.control_hz
                ec = {}
                self.controller.setServoJoint(period, period, 0, ec)
                _check_ec(ec, f"{self.config.side} setServoJoint")
                self.controller.startMove(self.sdk.RtControllerMode.jointPosition)
                self._servo = True
                self._last_target_s = time.monotonic()
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._control_loop,
                    name=f"xcore-{self.config.side}-servoj",
                    daemon=True,
                )
                self._thread.start()

    def _control_loop(self) -> None:
        period = 1.0 / self.control_hz
        max_step = 0.5 * period
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                with self._lock:
                    measured = self._read_locked()
                    if self.controller.hasMotionError():
                        raise RuntimeError(f"{self.config.side} ServoJ motion error")
                    with self._state_lock:
                        stale = time.monotonic() - self._last_target_s > 0.25
                        target = self._last_command.copy() if stale else self._target.copy()
                        previous = self._last_command.copy()
                    command = previous + np.clip(target - previous, -max_step, max_step)
                    self._validate(command, "command")
                    self.controller.sendCommand(self.sdk.JointPosition(command.tolist()))
                    with self._state_lock:
                        self._last_command = command
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                elif delay < -0.08:
                    raise RuntimeError(
                        f"{self.config.side} ServoJ host loop late by {-delay:.3f}s"
                    )
        except BaseException as exc:
            self._error = exc
            self._stop.set()
            logger.exception("%s ServoJ loop failed", self.config.side)

    def state(self) -> np.ndarray:
        if self._error:
            raise RuntimeError(f"{self.config.side} ServoJ failed") from self._error
        if self.enable_motion:
            return self.latest_state()
        with self._lock:
            return self._read_locked()

    def latest_state(self) -> np.ndarray:
        """Return the last validated sample without issuing an SDK request."""
        if self._error:
            raise RuntimeError(f"{self.config.side} ServoJ failed") from self._error
        with self._state_lock:
            return self._state.copy()

    def set_target(self, target: np.ndarray) -> np.ndarray:
        if not self.enable_motion:
            raise RuntimeError(f"{self.config.side} motion is disabled")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("joint target must be seven finite values")
        with self._state_lock:
            measured = self._state.copy()
        safe = np.clip(target, measured - 0.035, measured + 0.035)
        safe = np.clip(safe, self.lower, self.upper)
        with self._state_lock:
            self._target = safe
            self._last_target_s = time.monotonic()
        return safe.copy()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self.robot is not None:
            with self._lock:
                if self.controller is not None and self._servo:
                    try:
                        self.controller.stopMove()
                        self.controller.stopServoJoint()
                    except Exception:
                        logger.exception("failed to stop %s ServoJ", self.config.side)
                if self._receiving:
                    try:
                        self.robot.stopReceiveRobotState()
                    except Exception:
                        logger.exception("failed to stop %s state stream", self.config.side)
        self._receiving = self._servo = False
        self.controller = self.robot = None


class GripperIO:
    """ROS gripper interface. Disabled sides stay constant and publish nothing."""

    def __init__(
        self,
        hardware: HardwareConfig,
        *,
        enable_motion: bool,
        motion_sides: Iterable[str] | None = None,
    ):
        self.hardware = hardware
        self.enable_motion = enable_motion
        requested_sides = (
            {"left", "right"} if enable_motion and motion_sides is None else set(motion_sides or ())
        )
        if not requested_sides <= {"left", "right"}:
            raise ValueError(f"unsupported gripper motion sides: {sorted(requested_sides)}")
        self.motion_sides = frozenset(requested_sides if enable_motion else ())
        self.positions = {
            "left": float(hardware.left.gripper_placeholder),
            "right": float(hardware.right.gripper_placeholder),
        }
        self.ready = {"left": False, "right": False}
        self._rclpy: Any | None = None
        self._node: Any | None = None
        self._publishers: dict[str, Any] = {}

    def connect(self) -> None:
        enabled = [arm for arm in (self.hardware.left, self.hardware.right) if arm.gripper_enabled]
        if not enabled:
            return
        import rclpy
        from std_msgs.msg import Int32

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("xcore_interact_grippers")
        for arm in enabled:
            self._node.create_subscription(
                Int32,
                arm.gripper_state_topic,
                lambda message, side=arm.side: self._on_state(side, message.data),
                10,
            )
            if arm.side in self.motion_sides:
                self._publishers[arm.side] = self._node.create_publisher(
                    Int32, arm.gripper_command_topic, 10
                )
        deadline = time.monotonic() + 3.0
        while not all(self.ready[arm.side] for arm in enabled) and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        missing = [arm.gripper_state_topic for arm in enabled if not self.ready[arm.side]]
        if missing:
            raise RuntimeError(f"gripper state topics not ready: {missing}")

    def _on_state(self, side: str, value: int) -> None:
        if 0 <= int(value) <= 4095:
            self.positions[side] = float(value)
            self.ready[side] = True

    def spin(self) -> None:
        if self._node is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def command(self, side: str, value: float) -> float:
        arm = self.hardware.left if side == "left" else self.hardware.right
        if not arm.gripper_enabled:
            return float(arm.gripper_placeholder)
        if self.enable_motion and side not in self.motion_sides:
            return float(self.positions[side])
        # Quantize to the closest calibrated endpoint. This works for the
        # opposite left/right servo directions without assuming high=closed.
        target = min(
            (arm.gripper_open_position, arm.gripper_closed_position),
            key=lambda endpoint: abs(value - endpoint),
        )
        if side in self._publishers and target != self.positions[side]:
            from std_msgs.msg import Int32

            message = Int32()
            message.data = int(target)
            self._publishers[side].publish(message)
        return target

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        self._node = None


class DualXCoreRobot:
    def __init__(self, config: HardwareConfig | None = None):
        self.config = config or HardwareConfig()
        sdk = _load_sdk(self.config.sdk_python_path)
        self.left = XCoreArm(sdk, self.config.left)
        self.right = XCoreArm(sdk, self.config.right)
        self.grippers: GripperIO | None = None
        self.enable_motion = False
        self.control_mode = "bimanual"
        self.motion_sides: frozenset[str] = frozenset()

    def connect(
        self, *, enable_motion: bool = False, control_mode: str = "bimanual"
    ) -> None:
        control_mode = str(control_mode).strip().lower()
        if control_mode not in CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {sorted(CONTROL_MODES)}, got {control_mode!r}"
            )
        self.enable_motion = enable_motion
        self.control_mode = control_mode
        if not enable_motion:
            self.motion_sides = frozenset()
        elif control_mode == "right_only":
            self.motion_sides = frozenset({"right"})
        else:
            self.motion_sides = frozenset({"left", "right"})
        try:
            # Validate both gripper feedback streams before either arm enters
            # RT ServoJ mode. Missing startup topics therefore cannot leave
            # the arms holding in RT while gripper initialization times out.
            self.grippers = GripperIO(
                self.config,
                enable_motion=enable_motion,
                motion_sides=self.motion_sides,
            )
            self.grippers.connect()
            self.left.connect(enable_motion="left" in self.motion_sides)
            self.right.connect(enable_motion="right" in self.motion_sides)
        except BaseException:
            self.disconnect()
            raise

    def state(self) -> np.ndarray:
        if self.grippers:
            self.grippers.spin()
        left_gripper = (
            self.grippers.positions["left"]
            if self.grippers and self.config.left.gripper_enabled
            else self.config.left.gripper_placeholder
        )
        right_gripper = (
            self.grippers.positions["right"] if self.grippers else self.config.right.gripper_placeholder
        )
        return np.concatenate(
            (self.left.state(), [left_gripper], self.right.state(), [right_gripper])
        ).astype(np.float32)

    def send_action(self, action: np.ndarray) -> np.ndarray:
        if not self.enable_motion:
            raise RuntimeError("dual-arm motion is disabled")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("dual action must contain 16 finite values")
        left = (
            self.left.set_target(action[:7])
            if "left" in self.motion_sides
            else self.left.latest_state()
        )
        right = (
            self.right.set_target(action[8:15])
            if "right" in self.motion_sides
            else self.right.latest_state()
        )
        left_gripper = self.grippers.command("left", action[7])
        right_gripper = self.grippers.command("right", action[15])
        return np.concatenate((left, [left_gripper], right, [right_gripper])).astype(np.float32)

    def disconnect(self) -> None:
        if self.grippers:
            self.grippers.close()
        self.grippers = None
        self.right.disconnect()
        self.left.disconnect()

    def __enter__(self) -> "DualXCoreRobot":
        self.connect(enable_motion=False)
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
