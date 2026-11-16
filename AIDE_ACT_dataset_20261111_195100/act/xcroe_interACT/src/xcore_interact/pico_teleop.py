from __future__ import annotations

import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

PACKET_SIZE = 38
LEFT_HAND_ID = 0
RIGHT_HAND_ID = 1

TRIGGER_BIT = 0
GRIP_BIT = 1
DOWN_BIT = 2
UP_BIT = 3

POSITION_FILTER_ALPHA = 0.20
ROTATION_FILTER_ALPHA = 0.15
MAX_RAW_POSITION_JUMP_M = 0.10
MAX_RAW_ROTATION_JUMP_RAD = np.deg2rad(45.0)

# Calibrated from this rig's PICO UDP stream and left-arm reference frame:
# PICO left=-Y, up=-Z, forward=-X; robot left=-Z, up=+Y, forward=-X.
LEFT_PICO_TO_ROBOT = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
# Use one rigid-frame calibration for both translation and rotation.  This
# keeps rotations about physical left/up/forward aligned with the translation
# directions verified on the real left arm.
LEFT_ROTATION_TO_ROBOT = LEFT_PICO_TO_ROBOT


def compose_dual_record_vectors(
    left_joints: np.ndarray,
    right_joints: np.ndarray,
    gripper_state: dict[str, int],
    gripper_target: dict[str, int],
    active_sides: set[str] | frozenset[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build one paired dual-arm state/action sample.

    The dataset schema stays 16-D when only one arm is teleoperated. The
    inactive arm remains observable, while its action is held exactly at its
    measured state instead of retaining a stale gripper target.
    """
    unknown_sides = set(active_sides) - {"left", "right"}
    if unknown_sides:
        raise ValueError(f"unknown active sides: {sorted(unknown_sides)}")
    left = np.asarray(left_joints, dtype=np.float64).reshape(-1)
    right = np.asarray(right_joints, dtype=np.float64).reshape(-1)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError("left_joints and right_joints must each contain 7 values")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("joint values must be finite")

    state = np.concatenate(
        (
            left,
            [float(gripper_state["left"])],
            right,
            [float(gripper_state["right"])],
        )
    )
    action = state.copy()
    if "left" in active_sides:
        action[7] = float(gripper_target["left"])
    if "right" in active_sides:
        action[15] = float(gripper_target["right"])
    return state, action


def map_position_offset(side: str, raw_offset: np.ndarray) -> np.ndarray:
    """Map a PICO translation offset into the matching robot reference frame.

    Physical calibration for the left side is PICO left=-Y, up=-Z,
    forward=-X and robot left=-Z, up=+Y, forward=-X.  The right arm keeps the
    proven xcore_act direct mapping.
    """
    offset = np.asarray(raw_offset, dtype=np.float64).reshape(3)
    if side == "left":
        return LEFT_PICO_TO_ROBOT @ offset
    if side == "right":
        return offset.copy()
    raise ValueError(f"unknown controller side: {side!r}")


def map_rotation_offset(side: str, raw_offset: np.ndarray) -> np.ndarray:
    """Express a relative PICO rotation in the matching robot frame."""
    rotation = np.asarray(raw_offset, dtype=np.float64).reshape(3, 3)
    if side == "left":
        return LEFT_ROTATION_TO_ROBOT @ rotation @ LEFT_ROTATION_TO_ROBOT.T
    if side == "right":
        return rotation.copy()
    raise ValueError(f"unknown controller side: {side!r}")


def button_pressed(buttons: int, bit: int) -> bool:
    return bool(int(buttons) & (1 << bit))


def normalize_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or norm < 1e-8:
        raise ValueError("quaternion must be finite and non-zero")
    return quaternion / norm


def quaternion_angular_distance(first_xyzw: np.ndarray, second_xyzw: np.ndarray) -> float:
    first = normalize_quaternion(first_xyzw)
    second = normalize_quaternion(second_xyzw)
    dot = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def quaternion_slerp(first_xyzw: np.ndarray, second_xyzw: np.ndarray, alpha: float) -> np.ndarray:
    first = normalize_quaternion(first_xyzw)
    second = normalize_quaternion(second_xyzw)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(first + alpha * (second - first))
    angle = float(np.arccos(dot))
    denominator = float(np.sin(angle))
    return normalize_quaternion(
        np.sin((1.0 - alpha) * angle) / denominator * first
        + np.sin(alpha * angle) / denominator * second
    )


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    if axis == "x":
        return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=np.float64)
    if axis == "y":
        return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=np.float64)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float64)


def quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def convert_pico_orientation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Apply the same PICO axis convention as xcore_act's pico_servol.cpp."""
    x, y, z, w = normalize_quaternion(np.array([qx, qy, qz, qw]))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    rx = float(pitch)
    ry = float(roll - np.pi / 4.0)
    rz = float(-yaw - np.pi / 2.0)
    matrix = _axis_rotation("z", rz) @ _axis_rotation("y", ry) @ _axis_rotation("x", rx)
    return matrix_to_quaternion(matrix)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [(matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                 (matrix[1, 0] - matrix[0, 1]) / scale]
            )
    return normalize_quaternion(quaternion)


def matrix_axis_angle(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    quaternion = matrix_to_quaternion(matrix)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[:3]))
    if vector_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0]), 0.0
    angle = 2.0 * float(np.arctan2(vector_norm, quaternion[3]))
    return quaternion[:3] / vector_norm, angle


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-10 or abs(angle) < 1e-10:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / axis_norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def limited_position_step(
    current: np.ndarray,
    target: np.ndarray,
    *,
    deadband_m: float = 0.0005,
    max_step_m: float = 0.001,
) -> np.ndarray:
    difference = np.asarray(target) - np.asarray(current)
    distance = float(np.linalg.norm(difference))
    if distance < deadband_m:
        return np.asarray(current, dtype=np.float64).copy()
    if distance <= max_step_m:
        return np.asarray(target, dtype=np.float64).copy()
    return np.asarray(current, dtype=np.float64) + difference * (max_step_m / distance)


def limited_rotation_step(
    current: np.ndarray,
    target: np.ndarray,
    *,
    deadband_rad: float = np.deg2rad(0.3),
    max_step_rad: float = np.deg2rad(0.5),
) -> np.ndarray:
    axis, angle = matrix_axis_angle(np.asarray(target) @ np.asarray(current).T)
    if abs(angle) < deadband_rad:
        return np.asarray(current, dtype=np.float64).copy()
    return axis_angle_matrix(axis, min(abs(angle), max_step_rad)) @ np.asarray(current)


def scaled_rotation(
    raw_offset: np.ndarray,
    *,
    gain: float = 0.25,
    deadband_rad: float = np.deg2rad(0.3),
) -> np.ndarray:
    axis, angle = matrix_axis_angle(raw_offset)
    if abs(angle) < deadband_rad:
        return np.eye(3, dtype=np.float64)
    return axis_angle_matrix(axis, abs(angle) * gain)


@dataclass(frozen=True)
class ControllerPacket:
    side: str
    position_m: np.ndarray
    orientation_xyzw: np.ndarray
    buttons: int

    @property
    def trigger(self) -> bool:
        return button_pressed(self.buttons, TRIGGER_BIT)

    @property
    def grip(self) -> bool:
        return button_pressed(self.buttons, GRIP_BIT)

    @property
    def down(self) -> bool:
        return button_pressed(self.buttons, DOWN_BIT)

    @property
    def up(self) -> bool:
        return button_pressed(self.buttons, UP_BIT)


def parse_controller_packet(data: bytes) -> ControllerPacket:
    if len(data) != PACKET_SIZE or data[0] not in (LEFT_HAND_ID, RIGHT_HAND_ID):
        raise ValueError("PICO packet must be 38 bytes with hand_id 0 or 1")
    values = np.asarray(struct.unpack_from("<9f", data, 1), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("PICO packet contains NaN or infinity")
    raw_quaternion = values[3:7]
    if float(np.linalg.norm(raw_quaternion)) < 1e-8:
        raise ValueError("PICO packet contains an invalid quaternion")
    return ControllerPacket(
        side="left" if data[0] == LEFT_HAND_ID else "right",
        position_m=values[:3] / 1000.0,
        orientation_xyzw=convert_pico_orientation(*raw_quaternion),
        buttons=int(data[37]),
    )


@dataclass
class ControllerState:
    side: str
    position_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    orientation_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )
    raw_position_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    raw_orientation_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )
    buttons: int = 0
    valid: bool = False
    raw_valid: bool = False
    sequence: int = 0
    pose_timestamp_s: float = 0.0
    packet_timestamp_s: float = 0.0

    def copy(self) -> "ControllerState":
        return ControllerState(
            side=self.side,
            position_m=self.position_m.copy(),
            orientation_xyzw=self.orientation_xyzw.copy(),
            raw_position_m=self.raw_position_m.copy(),
            raw_orientation_xyzw=self.raw_orientation_xyzw.copy(),
            buttons=self.buttons,
            valid=self.valid,
            raw_valid=self.raw_valid,
            sequence=self.sequence,
            pose_timestamp_s=self.pose_timestamp_s,
            packet_timestamp_s=self.packet_timestamp_s,
        )

    def pose_fresh(self, timeout_s: float = 0.25) -> bool:
        return self.valid and time.monotonic() - self.pose_timestamp_s <= timeout_s

    def packet_fresh(self, timeout_s: float = 0.25) -> bool:
        return self.packet_timestamp_s > 0 and time.monotonic() - self.packet_timestamp_s <= timeout_s


@dataclass(frozen=True)
class ButtonEdges:
    grip_rising: bool
    down_rising: bool
    up_rising: bool


class ButtonEdgeTracker:
    def __init__(self) -> None:
        self._previous = {"left": 0, "right": 0}

    def update(self, side: str, buttons: int) -> ButtonEdges:
        previous = self._previous[side]
        self._previous[side] = int(buttons)
        return ButtonEdges(
            grip_rising=button_pressed(buttons, GRIP_BIT)
            and not button_pressed(previous, GRIP_BIT),
            down_rising=button_pressed(buttons, DOWN_BIT)
            and not button_pressed(previous, DOWN_BIT),
            up_rising=button_pressed(buttons, UP_BIT)
            and not button_pressed(previous, UP_BIT),
        )


class DualControllerReceiver:
    """Receive and filter both PICO controller streams on one UDP socket."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5050,
        *,
        forward_addresses: tuple[tuple[str, int], ...] = (),
    ):
        self.host = host
        self.port = int(port)
        self.forward_addresses = tuple(forward_addresses)
        self._condition = threading.Condition()
        self._states = {
            "left": ControllerState("left"),
            "right": ControllerState("right"),
        }
        self._teleop_active = {"left": False, "right": False}
        self._control_transition_depth = 0
        self._control_transition_epoch = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("PICO UDP receiver is already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pico-dual-udp", daemon=True)
        self._thread.start()

    def set_teleop_active(self, side: str, active: bool) -> None:
        with self._condition:
            self._teleop_active[side] = bool(active)
            self._condition.notify_all()

    def begin_control_transition(self) -> None:
        """Mark an SDK mode/session transition that may briefly delay threads."""
        with self._condition:
            self._control_transition_depth += 1
            self._control_transition_epoch += 1
            self._condition.notify_all()

    def end_control_transition(self) -> None:
        with self._condition:
            if self._control_transition_depth <= 0:
                raise RuntimeError("unbalanced control transition")
            self._control_transition_depth -= 1
            self._control_transition_epoch += 1
            self._condition.notify_all()

    def control_transition_state(self) -> tuple[int, bool]:
        with self._condition:
            return self._control_transition_epoch, self._control_transition_depth > 0

    def latest(self, side: str) -> ControllerState:
        with self._condition:
            return self._states[side].copy()

    def latest_both(self) -> dict[str, ControllerState]:
        with self._condition:
            return {side: state.copy() for side, state in self._states.items()}

    def wait_both(self, timeout_s: float) -> dict[str, ControllerState]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("PICO UDP receiver failed") from self._error
                if all(state.valid for state in self._states.values()):
                    return {side: state.copy() for side, state in self._states.items()}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [side for side, state in self._states.items() if not state.valid]
                    raise TimeoutError(f"PICO controller streams not received: {missing}")
                self._condition.wait(remaining)

    def wait_for_fresh_pose(
        self,
        side: str,
        timeout_s: float,
        *,
        freshness_timeout_s: float = 0.25,
        after_sequence: int | None = None,
    ) -> ControllerState:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("PICO UDP receiver failed") from self._error
                state = self._states[side]
                is_new = after_sequence is None or state.sequence > after_sequence
                if is_new and state.pose_fresh(freshness_timeout_s):
                    return state.copy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"fresh {side} PICO pose not received")
                self._condition.wait(remaining)

    def wait_for_active_poses_after(
        self,
        side: str,
        after_sequence: int,
        timeout_s: float,
        *,
        freshness_timeout_s: float = 0.25,
    ) -> dict[str, ControllerState]:
        """Wait for a new starting-arm pose and fresh poses for active arms."""
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("PICO UDP receiver failed") from self._error
                active_sides = [
                    active_side
                    for active_side, active in self._teleop_active.items()
                    if active
                ]
                starting_is_new = self._states[side].sequence > after_sequence
                all_fresh = all(
                    self._states[active_side].pose_fresh(freshness_timeout_s)
                    for active_side in active_sides
                )
                if starting_is_new and all_fresh:
                    return {
                        active_side: self._states[active_side].copy()
                        for active_side in active_sides
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stale = [
                        active_side
                        for active_side in active_sides
                        if not self._states[active_side].pose_fresh(freshness_timeout_s)
                    ]
                    raise TimeoutError(
                        f"fresh active PICO poses not received; stale={stale}"
                    )
                self._condition.wait(remaining)

    def _update(self, packet: ControllerPacket) -> None:
        now = time.monotonic()
        with self._condition:
            state = self._states[packet.side]
            state.buttons = packet.buttons
            state.packet_timestamp_s = now
            if state.raw_valid and self._teleop_active[packet.side]:
                position_jump = float(np.linalg.norm(packet.position_m - state.raw_position_m))
                rotation_jump = quaternion_angular_distance(
                    state.raw_orientation_xyzw, packet.orientation_xyzw
                )
                if (
                    position_jump > MAX_RAW_POSITION_JUMP_M
                    or rotation_jump > MAX_RAW_ROTATION_JUMP_RAD
                ):
                    self._condition.notify_all()
                    return
            state.raw_position_m = packet.position_m.copy()
            state.raw_orientation_xyzw = packet.orientation_xyzw.copy()
            state.raw_valid = True
            if not state.valid:
                state.position_m = packet.position_m.copy()
                state.orientation_xyzw = packet.orientation_xyzw.copy()
                state.valid = True
            else:
                state.position_m += POSITION_FILTER_ALPHA * (
                    packet.position_m - state.position_m
                )
                state.orientation_xyzw = quaternion_slerp(
                    state.orientation_xyzw,
                    packet.orientation_xyzw,
                    ROTATION_FILTER_ALPHA,
                )
            state.pose_timestamp_s = now
            state.sequence += 1
            self._condition.notify_all()

    def _run(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.1)
            sock.bind((self.host, self.port))
            printed: set[str] = set()
            while not self._stop.is_set():
                try:
                    data, address = sock.recvfrom(1024)
                except socket.timeout:
                    continue
                try:
                    packet = parse_controller_packet(data)
                except ValueError:
                    continue
                for forward_address in self.forward_addresses:
                    try:
                        sock.sendto(data, forward_address)
                    except OSError as error:
                        print(
                            f"PICO UDP forward to {forward_address} failed: {error}",
                            file=sys.stderr,
                        )
                self._update(packet)
                if packet.side not in printed:
                    print(
                        f"Received {packet.side} controller UDP from "
                        f"{address[0]}:{address[1]}"
                    )
                    printed.add(packet.side)
        except BaseException as error:
            if not self._stop.is_set():
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
        finally:
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def __enter__(self) -> "DualControllerReceiver":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
