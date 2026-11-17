"""Four-timestamp clock comparison; does not change either system clock.

Offset = PC minus Jetson. See RFC 5905 section 8. The half-RTT bound
assumes nonnegative path delays; oscillator aging is added conservatively.
It does not bound camera exposure/global-time calibration error.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class Estimate:
    offset_ns: int
    uncertainty_ns: int
    rtt_ns: int
    epoch: str
    measured_at: float


class ClockEstimator:
    def __init__(self, max_error_ms=10., stale_seconds=3., window_seconds=5., min_samples=4):
        self.max_error_ns = int(max_error_ms * 1e6)
        self.stale_seconds, self.window_seconds = stale_seconds, window_seconds
        self.min_samples = min_samples
        self.samples = deque(maxlen=64)
        self.epoch = None
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.samples.clear()
            self.epoch = None

    def add(self, t1, t2, t3, t4, elapsed_ns, epoch, at=None):
        at = time.monotonic() if at is None else at
        if any(type(v) is not int or v <= 0 for v in (t1, t2, t3, t4, elapsed_ns)) or not epoch:
            return False
        # A realtime step during an exchange invalidates its four timestamps.
        if t3 < t2 or t4 < t1 or abs((t4 - t1) - elapsed_ns) > 5_000_000:
            self.reset()
            return False
        rtt = (t4 - t1) - (t3 - t2)
        if rtt < 0 or rtt > 2 * self.max_error_ns:
            return False
        offset = ((t2 - t1) + (t3 - t4)) // 2
        with self.lock:
            while self.samples and at - self.samples[0][0] > self.window_seconds:
                self.samples.popleft()
            if epoch != self.epoch:
                self.samples.clear()
                self.epoch = epoch
            if self.samples:
                _, old_rtt, old_offset = min(self.samples, key=lambda v: v[1])
                if abs(offset - old_offset) > 5_000_000 + (old_rtt + rtt) // 2:
                    self.samples.clear()
            self.samples.append((at, rtt, offset))
        return True

    def estimate(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            valid = [v for v in self.samples if now - v[0] <= self.window_seconds]
            if len(valid) < self.min_samples or now - valid[-1][0] > self.stale_seconds:
                raise ValueError('waiting for reliable clock comparison')
            at, rtt, offset = min(valid, key=lambda v: v[1])
            # 200 ppm allowance for relative clock drift since this sample.
            uncertainty = rtt // 2 + int(max(0., now - at) * 200_000) + 100_000
            if uncertainty > self.max_error_ns:
                raise ValueError('clock uncertainty exceeds configured bound')
            return Estimate(offset, uncertainty, rtt, self.epoch, at)


class WallClockGuard:
    def __init__(self, step_ms=5.):
        self.threshold = int(step_ms * 1e6)
        self.previous = None
        self.lock = threading.Lock()

    def update(self, wall_ns, monotonic_ns):
        offset = wall_ns - monotonic_ns
        with self.lock:
            changed = self.previous is not None and abs(offset - self.previous) > self.threshold
            self.previous = offset
        return changed


class CaptureTimestampGuard:
    """Reject SDK clock convergence/steps; never substitute receipt time."""
    def __init__(self, warm_seconds=3.):
        self.warm_seconds = warm_seconds
        self.previous = None
        self.stable_since = None

    def accept(self, stamp_ns, wall_ns, monotonic, epoch):
        prior = self.previous
        self.previous = stamp_ns, wall_ns, epoch
        good = -10_000_000 <= wall_ns - stamp_ns <= 250_000_000
        if prior is not None and epoch == prior[2]:
            good = good and stamp_ns > prior[0]
            good = good and abs((stamp_ns - prior[0]) - (wall_ns - prior[1])) <= 50_000_000
        else:
            self.stable_since = None
        if not good:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = monotonic
        return monotonic - self.stable_since >= self.warm_seconds
