from __future__ import annotations

import importlib
import logging
import math
import sys
import threading
import time
from datetime import timedelta
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot

from .config_xcore import XCoreLeftArmConfig, XCoreRightArmConfig

logger = logging.getLogger(__name__)

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))

# AR-series limits used by pico_servol.cpp, before its 12-degree margin.
HARD_LOWER_LIMITS = (-3.10669, -2.0944, -3.10669, 0.0, -3.10669, -1.0472, -1.0472)
HARD_UPPER_LIMITS = (3.10669, 2.0944, 3.10669, 2.53073, 3.10669, 1.0472, 1.0472)


def _check_ec(ec: dict[str, Any], operation: str) -> None:
    if int(ec.get("ec", 0)) != 0:
        raise RuntimeError(
            f"xCore {operation} failed: code={ec.get('ec')}, "
            f"message={ec.get('message', '')}"
        )


def _finite_vector(values: list[float], expected: int, label: str) -> list[float]:
    if len(values) != expected:
        raise ValueError(f"{label} must contain {expected} values, got {len(values)}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} contains NaN or infinity")
    return result


class XCoreRightArm(Robot):
    """LeRobot adapter that executes absolute joint targets with ServoJ."""

    config_class = XCoreRightArmConfig
    name = "xcore_right_arm"

    def __init__(self, config: XCoreRightArmConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self._sdk: Any | None = None
        self._robot: Any | None = None
        self._rt_controller: Any | None = None
        self._connected = False
        self._receiving_state = False
        self._servo_started = False
        self._sdk_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._control_thread: threading.Thread | None = None
        self._control_error: BaseException | None = None

        self._joint_state = [0.0] * 7
        self._target_joints = [0.0] * 7
        self._last_command_joints = [0.0] * 7
        self._last_action_time = 0.0
        self._command_timeout_active = False
        self._active_lower = list(HARD_LOWER_LIMITS)
        self._active_upper = list(HARD_UPPER_LIMITS)
        self._gripper_position = float(config.gripper_min)
        self._gripper_target = float(config.gripper_min)
        self._gripper_state_ready = False
        self._gripper_state_time = 0.0
        self._gripper_hold_until = 0.0
        self._rclpy: Any | None = None
        self._gripper_node: Any | None = None
        self._gripper_publisher: Any | None = None
        self._gripper_subscription: Any | None = None
        self._owns_rclpy_context = False

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {
            f"{joint}.pos": float for joint in JOINT_NAMES
        }
        if self.config.use_gripper:
            features["gripper.pos"] = float
        features.update(
            {
                name: (cfg.height, cfg.width, 3)
                for name, cfg in self.config.cameras.items()
            }
        )
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {f"{joint}.pos": float for joint in JOINT_NAMES}
        if self.config.use_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._connected and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _load_sdk(self) -> Any:
        sdk_path = self.config.sdk_python_path.expanduser().resolve()
        if not sdk_path.is_dir():
            raise FileNotFoundError(f"xCore Python SDK directory does not exist: {sdk_path}")
        sdk_path_text = str(sdk_path)
        if sdk_path_text not in sys.path:
            sys.path.insert(0, sdk_path_text)
        try:
            return importlib.import_module("xCoreSDK_python")
        except ImportError as exc:
            raise ImportError(
                f"Cannot import the aarch64 xCore Python binding from {sdk_path}. "
                "Use Python 3.10 on this machine."
            ) from exc

    def _read_joints_locked(self) -> list[float]:
        ec: dict[str, Any] = {}
        joints = list(self._robot.jointPos(ec))[:7]
        _check_ec(ec, "jointPos")
        joints = _finite_vector(joints, 7, "joint state")
        with self._state_lock:
            self._joint_state = joints
        return joints

    def _reset_joint_targets_locked(self, joints: list[float]) -> None:
        with self._state_lock:
            self._target_joints = joints.copy()
            self._last_command_joints = joints.copy()
            self._last_action_time = time.monotonic()
            self._command_timeout_active = False

    def _load_active_limits_locked(self) -> None:
        margin = self.config.joint_limit_margin_rad
        lower = [value + margin for value in HARD_LOWER_LIMITS]
        upper = [value - margin for value in HARD_UPPER_LIMITS]

        limits_out = self._sdk.PyTypeVectorArrayDouble2()
        ec: dict[str, Any] = {}
        enabled = bool(self._robot.getSoftLimit(limits_out, ec))
        _check_ec(ec, "getSoftLimit")
        if not enabled and self.config.enable_motion and self.config.require_soft_limits:
            raise RuntimeError("controller soft limits are disabled; refusing to enable ACT motion")
        if not enabled:
            logger.warning("controller soft limits are disabled; using static limits for read-only checks")
        if enabled:
            controller_limits = list(limits_out.content())[:7]
            if len(controller_limits) != 7:
                raise RuntimeError("controller returned fewer than 7 joint soft limits")
            for index, pair in enumerate(controller_limits):
                pair = list(pair)
                lower[index] = max(lower[index], float(pair[0]) + margin)
                upper[index] = min(upper[index], float(pair[1]) - margin)

        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise RuntimeError("joint limits have no valid range after applying the safety margin")
        self._active_lower = lower
        self._active_upper = upper

    def _validate_inside_limits(self, joints: list[float], label: str) -> None:
        for index, (value, lower, upper) in enumerate(
            zip(joints, self._active_lower, self._active_upper, strict=True), start=1
        ):
            if value < lower or value > upper:
                raise ValueError(
                    f"{label} J{index}={value:.6f} rad is outside "
                    f"[{lower:.6f}, {upper:.6f}]"
                )

    def _setup_gripper(self) -> None:
        if not (self.config.use_gripper and self.config.enable_motion):
            return
        try:
            import rclpy
            from std_msgs.msg import Int32
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Python packages are required for the STS3215 gripper. "
                "Source /opt/ros/humble/setup.bash before activating the ACT environment."
            ) from exc

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        self._gripper_node = rclpy.create_node("lerobot_xcore_gripper")
        self._gripper_publisher = self._gripper_node.create_publisher(
            Int32, self.config.gripper_topic, 10
        )
        self._gripper_subscription = self._gripper_node.create_subscription(
            Int32, self.config.gripper_state_topic, self._on_gripper_state, 10
        )

        deadline = time.monotonic() + self.config.gripper_state_timeout_s
        while not self._gripper_state_ready and time.monotonic() < deadline:
            rclpy.spin_once(self._gripper_node, timeout_sec=0.05)
        if not self._gripper_state_ready:
            message = (
                f"No gripper state received on {self.config.gripper_state_topic} "
                f"within {self.config.gripper_state_timeout_s:.1f}s"
            )
            if self.config.require_gripper_state:
                raise RuntimeError(message)
            logger.warning("%s; falling back to commanded gripper state", message)
        else:
            with self._state_lock:
                self._gripper_target = self._quantize_gripper(
                    self._gripper_position
                )

    def _on_gripper_state(self, message: Any) -> None:
        value = float(message.data)
        if not math.isfinite(value):
            return
        value = max(self.config.gripper_min, min(self.config.gripper_max, value))
        with self._state_lock:
            self._gripper_position = value
            self._gripper_state_time = time.monotonic()
            self._gripper_state_ready = True

    def _spin_gripper_once(self) -> None:
        if self._gripper_node is not None and self._rclpy is not None:
            self._rclpy.spin_once(self._gripper_node, timeout_sec=0.0)

    def _quantize_gripper(self, value: float) -> float:
        if value >= self.config.gripper_switch_threshold:
            return float(self.config.gripper_max)
        return float(self.config.gripper_min)

    def _select_gripper_target(self, value: float, previous: float) -> float:
        threshold = self.config.gripper_switch_threshold
        hysteresis = self.config.gripper_switch_hysteresis
        if previous == self.config.gripper_min:
            if value >= threshold + hysteresis:
                return float(self.config.gripper_max)
            return float(self.config.gripper_min)
        if value <= threshold - hysteresis:
            return float(self.config.gripper_min)
        return float(self.config.gripper_max)

    def _publish_gripper(self, value: float) -> None:
        if self._gripper_publisher is None:
            return
        from std_msgs.msg import Int32

        message = Int32()
        message.data = int(round(value))
        self._gripper_publisher.publish(message)
        self._spin_gripper_once()

    def _restore_nrt_ready_locked(self) -> None:
        """Match pico_servol's NRT/automatic/power-on reset preparation."""
        ec: dict[str, Any] = {}
        self._robot.setMotionControlMode(self._sdk.MotionControlMode.NrtCommandMode, ec)
        _check_ec(ec, "setMotionControlMode(NrtCommandMode)")
        ec = {}
        self._robot.setOperateMode(self._sdk.OperateMode.automatic, ec)
        _check_ec(ec, "setOperateMode(automatic)")
        ec = {}
        self._robot.setPowerState(True, ec)
        _check_ec(ec, "setPowerState(true)")

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            ec = {}
            state = self._robot.powerState(ec)
            _check_ec(ec, "powerState")
            if state == self._sdk.PowerState.on:
                return
            if state == self._sdk.PowerState.estop:
                raise RuntimeError(
                    "xCore initial-pose move refused: emergency stop is pressed"
                )
            if state == self._sdk.PowerState.gstop:
                raise RuntimeError(
                    "xCore initial-pose move refused: safety gate is open"
                )
            time.sleep(0.1)
        raise TimeoutError("xCore did not reach PowerState.on within 3 seconds")

    def _stop_nrt_motion_locked(self) -> None:
        """Best-effort stop/reset used when startup MoveAbsJ is interrupted."""
        try:
            ec: dict[str, Any] = {}
            self._robot.stop(ec)
            _check_ec(ec, "stop(initial MoveAbsJ)")
        except Exception:
            logger.exception("failed to stop xCore initial MoveAbsJ")

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                ec = {}
                operation_state = self._robot.operationState(ec)
                _check_ec(ec, "operationState(initial MoveAbsJ stop)")
                if operation_state == self._sdk.OperationState.idle:
                    break
            except Exception:
                logger.exception("failed while waiting for initial MoveAbsJ to stop")
                break
            time.sleep(0.05)

        try:
            ec = {}
            self._robot.moveReset(ec)
            _check_ec(ec, "moveReset(initial MoveAbsJ stop)")
        except Exception:
            logger.exception("failed to reset xCore NRT motion buffer")

    def _move_to_initial_pose_locked(self) -> None:
        """Move to the dataset start pose in NRT mode before enabling ServoJ."""
        target = _finite_vector(
            list(self.config.initial_joint_positions), 7, "initial_joint_positions"
        )
        self._validate_inside_limits(target, "initial pose")
        self._restore_nrt_ready_locked()

        logger.info(
            "Moving xCore to the dataset initial pose with NRT MoveAbsJ "
            "(speed=%.1f, zone=%.1f)",
            self.config.initial_move_speed,
            self.config.initial_move_zone,
        )
        nrt_buffer_prepared = False
        try:
            ec: dict[str, Any] = {}
            self._robot.moveReset(ec)
            _check_ec(ec, "moveReset(initial pose)")
            nrt_buffer_prepared = True

            command = self._sdk.MoveAbsJCommand(
                target,
                self.config.initial_move_speed,
                self.config.initial_move_zone,
            )
            command_id = self._sdk.PyString()
            ec = {}
            self._robot.moveAppend(command, command_id, ec)
            _check_ec(ec, "moveAppend(initial pose)")
            ec = {}
            self._robot.moveStart(ec)
            _check_ec(ec, "moveStart(initial pose)")

            # Avoid observing the transient idle state before the controller
            # starts executing the command, as in pico_servol's waitRobotIdle.
            time.sleep(0.1)
            deadline = time.monotonic() + self.config.initial_move_timeout_s
            while True:
                ec = {}
                operation_state = self._robot.operationState(ec)
                _check_ec(ec, "operationState(initial pose)")
                if operation_state == self._sdk.OperationState.idle:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "xCore initial MoveAbsJ did not finish within "
                        f"{self.config.initial_move_timeout_s:.1f} seconds"
                    )
                time.sleep(0.1)

            measured = self._read_joints_locked()
            errors = [
                abs(actual - goal)
                for actual, goal in zip(measured, target, strict=True)
            ]
            max_error = max(errors)
            if max_error > self.config.initial_pose_tolerance_rad:
                raise RuntimeError(
                    "xCore initial MoveAbsJ finished outside tolerance: "
                    f"max error={max_error:.6f} rad, "
                    f"limit={self.config.initial_pose_tolerance_rad:.6f} rad"
                )
            logger.info(
                "xCore initial pose reached; max joint error=%.6f rad. "
                "Switching to ACT ServoJ.",
                max_error,
            )
        except BaseException:
            if nrt_buffer_prepared:
                self._stop_nrt_motion_locked()
            raise

    def _start_servo_locked(self) -> None:
        ec: dict[str, Any] = {}
        self._robot.setOperateMode(self._sdk.OperateMode.automatic, ec)
        _check_ec(ec, "setOperateMode(automatic)")
        ec = {}
        self._robot.setRtNetworkTolerance(20, ec)
        _check_ec(ec, "setRtNetworkTolerance")
        ec = {}
        self._robot.setMotionControlMode(self._sdk.MotionControlMode.RtCommandMode, ec)
        _check_ec(ec, "setMotionControlMode(RtCommandMode)")
        ec = {}
        self._robot.setPowerState(True, ec)
        _check_ec(ec, "setPowerState(true)")

        self._rt_controller = self._robot.getRtMotionController()
        self._robot.startReceiveRobotState(
            timedelta(milliseconds=1), [self._sdk.RtSupportedFields.jointPos_m]
        )
        self._receiving_state = True
        while self._robot.updateRobotState(timedelta(milliseconds=0)):
            pass
        joints = self._read_joints_locked()
        self._validate_inside_limits(joints, "current state")
        self._reset_joint_targets_locked(joints)

        period = 1.0 / self.config.control_hz
        ec = {}
        self._rt_controller.setServoJoint(period, period, 0, ec)
        _check_ec(ec, "setServoJoint")
        self._rt_controller.startMove(self._sdk.RtControllerMode.jointPosition)
        self._servo_started = True

        self._stop_event.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="xcore-servoj-50hz",
            daemon=True,
        )
        self._control_thread.start()

    def _control_loop(self) -> None:
        period = 1.0 / self.config.control_hz
        max_step = self.config.max_joint_speed_rad_s * period
        next_tick = time.monotonic()
        late_events: list[float] = []
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                with self._sdk_lock:
                    self._robot.updateRobotState(timedelta(milliseconds=1))
                    measured = self._read_joints_locked()
                    if self._rt_controller.hasMotionError():
                        raise RuntimeError("xCore ServoJ reported a motion error")
                    self._validate_inside_limits(measured, "measured state")

                    with self._state_lock:
                        stale = now - self._last_action_time > self.config.command_timeout_s
                        previous = self._last_command_joints.copy()
                        # Do not continuously adopt the latest measured pose as
                        # the timeout target.  Doing so can ratchet gravity sag
                        # or tracking error into a slow drift after the policy
                        # loop stops.  Keep sending the last ServoJ command
                        # until a fresh policy action arrives or ServoJ stops.
                        target = previous.copy() if stale else self._target_joints.copy()
                        if stale != self._command_timeout_active:
                            self._command_timeout_active = stale
                            if stale:
                                logger.warning(
                                    "No policy action for %.3fs; holding the last ServoJ command",
                                    self.config.command_timeout_s,
                                )
                            else:
                                logger.info("Fresh policy action received; leaving timeout hold")

                    command = [
                        current + max(-max_step, min(max_step, goal - current))
                        for current, goal in zip(previous, target, strict=True)
                    ]
                    self._validate_inside_limits(command, "ServoJ command")
                    self._rt_controller.sendCommand(self._sdk.JointPosition(command))
                    with self._state_lock:
                        self._last_command_joints = command

                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._stop_event.wait(delay)
                elif delay < -0.04:
                    late_by = -delay
                    late_now = time.monotonic()
                    late_events = [
                        event for event in late_events if late_now - event <= 1.0
                    ]
                    late_events.append(late_now)

                    # A loaded Python/CUDA process can occasionally miss one
                    # 20 ms wake-up even though the xCore controller itself has
                    # not reported a motion error.  On one isolated miss, stop
                    # advancing the policy target and rebase the host clock.
                    # Preserve a hard failure for a long or repeated loss of
                    # real-time service.
                    with self._state_lock:
                        self._target_joints = self._last_command_joints.copy()
                        self._last_action_time = 0.0
                        self._command_timeout_active = True
                    if late_by >= 0.5 or len(late_events) >= 3:
                        raise RuntimeError(
                            "xCore control loop timing is repeatedly unsafe: "
                            f"late by {late_by:.3f}s, "
                            f"{len(late_events)} late events within 1s"
                        )
                    logger.warning(
                        "xCore control loop was late by %.3fs; holding the last "
                        "ServoJ command and resynchronizing (%d/3 within 1s)",
                        late_by,
                        len(late_events),
                    )
                    self._stop_event.wait(period)
                    next_tick = time.monotonic()
        except BaseException as exc:
            self._control_error = exc
            self._stop_event.set()
            logger.exception("xCore ServoJ loop stopped")

    def _raise_control_error(self) -> None:
        if self._control_error is not None:
            raise RuntimeError("xCore ServoJ background loop failed") from self._control_error

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise RuntimeError("xCore robot is already connected")
        self._control_error = None
        self._sdk = self._load_sdk()
        try:
            self._robot = self._sdk.ArRobot(self.config.robot_ip, self.config.local_ip)
            with self._sdk_lock:
                joints = self._read_joints_locked()
                self._load_active_limits_locked()
                self._validate_inside_limits(joints, "current state")
                self._reset_joint_targets_locked(joints)
            for camera in self.cameras.values():
                camera.connect()
            self._setup_gripper()
            self._connected = True
            if self.config.enable_motion:
                with self._sdk_lock:
                    if self.config.move_to_initial_pose:
                        self._move_to_initial_pose_locked()
                    self._start_servo_locked()
            else:
                logger.warning(
                    "xCore connected in read-only mode; set --robot.enable_motion=true "
                    "only after completing the preflight"
                )
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise RuntimeError("xCore robot is not connected")
        self._raise_control_error()
        if self.config.enable_motion:
            with self._state_lock:
                joints = self._joint_state.copy()
        else:
            with self._sdk_lock:
                joints = self._read_joints_locked()

        self._spin_gripper_once()
        observation: RobotObservation = {
            f"{joint}.pos": value for joint, value in zip(JOINT_NAMES, joints, strict=True)
        }
        if self.config.use_gripper:
            with self._state_lock:
                if (
                    self.config.enable_motion
                    and self.config.require_gripper_state
                    and time.monotonic() - self._gripper_state_time
                    > self.config.gripper_state_timeout_s
                ):
                    raise RuntimeError(
                        f"Gripper state on {self.config.gripper_state_topic} is stale"
                    )
                observation["gripper.pos"] = self._gripper_position
        for name, camera in self.cameras.items():
            observation[name] = camera.async_read()
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise RuntimeError("xCore robot is not connected")
        self._raise_control_error()

        expected = set(self.action_features)
        missing = expected.difference(action)
        if missing:
            raise ValueError(f"action is missing features: {sorted(missing)}")
        joint_action_names = [f"{joint}.pos" for joint in JOINT_NAMES]
        goal_joints = _finite_vector(
            [float(action[name]) for name in joint_action_names],
            7,
            "ServoJ action",
        )
        with self._state_lock:
            measured = self._joint_state.copy()
            previous_gripper_target = self._gripper_target
        safe_joints = [
            max(
                lower,
                min(
                    upper,
                    current
                    + max(
                        -self.config.max_relative_target_rad,
                        min(self.config.max_relative_target_rad, goal - current),
                    ),
                ),
            )
            for goal, current, lower, upper in zip(
                goal_joints,
                measured,
                self._active_lower,
                self._active_upper,
                strict=True,
            )
        ]

        gripper = previous_gripper_target
        publish_gripper = False
        if self.config.use_gripper:
            gripper = float(action["gripper.pos"])
            if not math.isfinite(gripper):
                raise ValueError("gripper action contains NaN or infinity")
            gripper = max(self.config.gripper_min, min(self.config.gripper_max, gripper))
            gripper = self._select_gripper_target(gripper, previous_gripper_target)
            publish_gripper = gripper != previous_gripper_target

        now = time.monotonic()
        if publish_gripper:
            self._gripper_hold_until = now + self.config.gripper_motion_hold_s
            logger.info(
                "Gripper target switched %.0f -> %.0f; holding arm for %.2fs",
                previous_gripper_target,
                gripper,
                self.config.gripper_motion_hold_s,
            )
        holding_for_gripper = now < self._gripper_hold_until
        if holding_for_gripper:
            safe_joints = measured

        with self._state_lock:
            self._target_joints = safe_joints
            self._last_action_time = time.monotonic()
            self._gripper_target = gripper
            if not self._gripper_state_ready:
                self._gripper_position = gripper
        if self.config.enable_motion and self.config.use_gripper and publish_gripper:
            self._publish_gripper(gripper)

        sent: RobotAction = dict(zip(joint_action_names, safe_joints, strict=True))
        if self.config.use_gripper:
            sent["gripper.pos"] = gripper
        return sent

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
        self._control_thread = None

        if self._robot is not None:
            with self._sdk_lock:
                if self._rt_controller is not None and self._servo_started:
                    try:
                        self._rt_controller.stopMove()
                        self._rt_controller.stopServoJoint()
                    except Exception:
                        logger.exception("failed to stop xCore ServoJ cleanly")
                self._servo_started = False
                if self._receiving_state:
                    try:
                        self._robot.stopReceiveRobotState()
                    except Exception:
                        logger.exception("failed to stop xCore state reception cleanly")
                self._receiving_state = False
                if self._sdk is not None and self.config.enable_motion:
                    try:
                        ec: dict[str, Any] = {}
                        self._robot.setMotionControlMode(
                            self._sdk.MotionControlMode.NrtCommandMode, ec
                        )
                        _check_ec(ec, "restore NrtCommandMode")
                    except Exception:
                        logger.exception("failed to restore xCore NRT mode")

        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()

        if self._gripper_node is not None:
            self._gripper_node.destroy_node()
        if self._owns_rclpy_context and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        self._gripper_node = None
        self._gripper_publisher = None
        self._gripper_subscription = None
        self._owns_rclpy_context = False
        self._robot = None
        self._rt_controller = None
        self._connected = False


class XCoreLeftArm(XCoreRightArm):
    """LeRobot adapter for the left xCore controller."""

    config_class = XCoreLeftArmConfig
    name = "xcore_left_arm"
