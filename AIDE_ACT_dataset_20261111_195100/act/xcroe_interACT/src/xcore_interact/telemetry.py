from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import ACTION_NAMES, STATE_NAMES

STATE_TOPIC = "/xcore/dual/record_state"
ACTION_TOPIC = "/xcore/dual/record_action"
ACTIVE_TOPIC = "/xcore/dual/teleop_active"
BUTTONS_TOPIC = "/xcore/dual/pico_buttons"


@dataclass(frozen=True)
class DualSample:
    state: np.ndarray
    action: np.ndarray
    received_s: float


class _TimestampPairBuffer:
    """Commit only state/action messages carrying the exact same ROS stamp."""

    def __init__(self, max_pending: int = 32):
        self.max_pending = max(2, int(max_pending))
        self.pending_states: dict[int, tuple[np.ndarray, float]] = {}
        self.pending_actions: dict[int, tuple[np.ndarray, float]] = {}
        self.latest: DualSample | None = None
        self.latest_stamp_ns = -1

    def add_state(
        self, stamp_ns: int, vector: np.ndarray, received_s: float
    ) -> DualSample | None:
        return self._add(
            self.pending_states, stamp_ns, vector, received_s
        )

    def add_action(
        self, stamp_ns: int, vector: np.ndarray, received_s: float
    ) -> DualSample | None:
        return self._add(
            self.pending_actions, stamp_ns, vector, received_s
        )

    def _add(
        self,
        pending: dict[int, tuple[np.ndarray, float]],
        stamp_ns: int,
        vector: np.ndarray,
        received_s: float,
    ) -> DualSample | None:
        stamp_ns = int(stamp_ns)
        if stamp_ns <= self.latest_stamp_ns:
            return self.latest
        pending[stamp_ns] = (np.asarray(vector, dtype=np.float32).copy(), received_s)
        while len(pending) > self.max_pending:
            pending.pop(next(iter(pending)))
        return self._commit(stamp_ns)

    def _commit(self, stamp_ns: int) -> DualSample | None:
        if stamp_ns not in self.pending_states or stamp_ns not in self.pending_actions:
            return self.latest
        state, state_received_s = self.pending_states[stamp_ns]
        action, action_received_s = self.pending_actions[stamp_ns]
        self.latest = DualSample(
            state=state.copy(),
            action=action.copy(),
            received_s=max(state_received_s, action_received_s),
        )
        self.latest_stamp_ns = stamp_ns
        # A complete newer pair makes every older unmatched half obsolete.
        for store in (self.pending_states, self.pending_actions):
            for old_stamp in tuple(key for key in store if key <= stamp_ns):
                store.pop(old_stamp, None)
        return self.latest


def validate_dual_vector(names: list[str], values: list[float], expected: tuple[str, ...]) -> np.ndarray:
    if tuple(names) != expected:
        raise ValueError(f"telemetry names differ; expected={list(expected)}, got={names}")
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError("dual-arm telemetry must be 16 finite float values")
    if np.any(np.abs(result[[*range(7), *range(8, 15)]]) > 3.2):
        raise ValueError("joint telemetry exceeds the 3.2 rad sanity bound")
    if np.any((result[[7, 15]] < 0) | (result[[7, 15]] > 4095)):
        raise ValueError("gripper telemetry is outside [0, 4095]")
    return result


class DualTelemetrySubscriber:
    """Read-only subscriber for the patched dual-PICO controller."""

    def __init__(self, max_pair_skew_s: float = 0.04, stale_after_s: float = 0.25):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.signals import SignalHandlerOptions
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Bool, UInt8
        except ImportError as exc:
            raise RuntimeError("ROS 2 Humble Python packages are required") from exc
        self._rclpy = rclpy
        self._context = Context()
        rclpy.init(
            args=None,
            context=self._context,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self._node = rclpy.create_node(
            "xcore_interact_dual_recorder", context=self._context
        )
        self._lock = threading.Lock()
        self._pairs = _TimestampPairBuffer()
        self._active = False
        self._button_events: list[int] = []
        self._last_buttons: int | None = None
        # Kept for constructor compatibility. Pairing is now exact, so a skew
        # tolerance is deliberately no longer used.
        self.max_pair_skew_s = max_pair_skew_s
        self.stale_after_s = stale_after_s
        self._subscriptions = [
            self._node.create_subscription(JointState, STATE_TOPIC, self._on_state, 20),
            self._node.create_subscription(JointState, ACTION_TOPIC, self._on_action, 20),
            self._node.create_subscription(Bool, ACTIVE_TOPIC, self._on_active, 20),
            self._node.create_subscription(UInt8, BUTTONS_TOPIC, self._on_buttons, 20),
        ]
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="interact-dual-telemetry",
            daemon=True,
        )
        self._spin_thread.start()

    @staticmethod
    def _stamp_ns(message: Any) -> int:
        return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)

    def _on_state(self, message: Any) -> None:
        try:
            vector = validate_dual_vector(list(message.name), list(message.position), STATE_NAMES)
        except ValueError:
            return
        stamp_ns = self._stamp_ns(message)
        received_s = time.monotonic()
        with self._lock:
            self._pairs.add_state(stamp_ns, vector, received_s)

    def _on_action(self, message: Any) -> None:
        try:
            vector = validate_dual_vector(list(message.name), list(message.position), ACTION_NAMES)
        except ValueError:
            return
        stamp_ns = self._stamp_ns(message)
        received_s = time.monotonic()
        with self._lock:
            self._pairs.add_action(stamp_ns, vector, received_s)

    def _on_active(self, message: Any) -> None:
        with self._lock:
            self._active = bool(message.data)

    def _on_buttons(self, message: Any) -> None:
        buttons = int(message.data) & 0xFF
        with self._lock:
            if buttons != self._last_buttons:
                self._button_events.append(buttons)
                self._last_buttons = buttons

    def drain_button_events(self) -> list[int]:
        self.spin_once(0.0)
        with self._lock:
            events = self._button_events
            self._button_events = []
        return events

    def spin_once(self, timeout_s: float = 0.0) -> None:
        # ROS callbacks are continuously drained by the executor thread, just
        # like the proven single-arm recorder. Preserve the old method as a
        # cooperative wait used by the recording state machine.
        if timeout_s > 0:
            time.sleep(timeout_s)

    def latest(self, *, require_active: bool = True) -> DualSample:
        self.spin_once(0.0)
        with self._lock:
            sample, active = self._pairs.latest, self._active
        if require_active and not active:
            raise RuntimeError("dual teleoperation is not active")
        if sample is None:
            raise RuntimeError("waiting for dual state/action telemetry")
        now = time.monotonic()
        if now - sample.received_s > self.stale_after_s:
            raise RuntimeError("dual state/action telemetry is stale")
        return DualSample(
            sample.state.copy(), sample.action.copy(), sample.received_s
        )

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        last_error = "no sample"
        while time.monotonic() < deadline:
            self.spin_once(0.05)
            try:
                self.latest(require_active=False)
                return
            except RuntimeError as exc:
                last_error = str(exc)
        raise TimeoutError(f"dual telemetry not ready: {last_error}")

    def close(self) -> None:
        self._executor.shutdown(timeout_sec=1.0)
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._node.destroy_node()
        try:
            self._context.shutdown()
        except Exception:
            pass

    def __enter__(self) -> "DualTelemetrySubscriber":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
