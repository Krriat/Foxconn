"""Time alignment and recording controls; no ROS, hardware or motion APIs."""
from collections import deque
from dataclasses import dataclass
import math
import threading

import numpy as np

CAMERAS = ('head', 'left_wrist', 'right_wrist')
STATE_NAMES = tuple(f'{side}_joint_{i}.pos' for side in ('left', 'right') for i in range(1, 8)) + ('head_pha.pos', 'waist_pitch.pos')
STREAMS = ('left', 'right', 'head_pha', 'waist_pitch') + CAMERAS
ACTION_SOURCE = 'physical_drag_measured_positions_same_sample'
DOUBLE_CLICK_SECONDS = .35


class NotReady(RuntimeError):
    pass


class AwaitFrame(NotReady):
    """A healthy camera's next frame is not here yet; allow bounded jitter."""
    pass


@dataclass(frozen=True)
class Sample:
    stamp: float
    received: float
    value: object


@dataclass(frozen=True)
class Bundle:
    stamp: float
    state: np.ndarray
    images: dict
    source_stamps: np.ndarray


class Inputs:
    def __init__(self, max_skew=.15, max_age=.5, state_max_age=.25, required_fps=None):
        self.max_skew, self.max_age, self.state_max_age = max_skew, max_age, state_max_age
        self.required_fps = required_fps
        self.lock = threading.RLock()
        self.buffers = {key: deque(maxlen=12 if key in CAMERAS else 100) for key in STREAMS}
        # Keep rate history without retaining seconds of full-resolution RGB.
        self.rate_history = {key: deque(maxlen=240) for key in STREAMS}
        self.teaching = {'left': False, 'right': False}
        self.motor_faults = set()
        self.button_at = -math.inf
        self.clock_reset = False

    def push(self, key, stamp, received, value):
        if not math.isfinite(stamp) or stamp <= 0 or not math.isfinite(received):
            raise ValueError('invalid feedback timestamp')
        with self.lock:
            buf = self.buffers[key]
            if buf and stamp < buf[-1].stamp:
                # Drop old history after a ROS clock change; recording must pause.
                for values in self.buffers.values():
                    values.clear()
                for values in self.rate_history.values():
                    values.clear()
                self.clock_reset = True
            self.buffers[key].append(Sample(stamp, received, value))
            history = self.rate_history[key]
            if not history or stamp > history[-1][0]:
                history.append((stamp, received))

    def rates(self, mono_now):
        with self.lock:
            result = {}
            for key, history in self.rate_history.items():
                recent = [(stamp, receipt) for stamp, receipt in history if mono_now - receipt <= 2.]
                if len(recent) < 2 or recent[-1][1] - recent[0][1] < 1.:
                    raise NotReady(f'measuring {key} rate; need at least 1 s of feedback')
                source_rate = (len(recent) - 1) / (recent[-1][0] - recent[0][0])
                receipt_rate = (len(recent) - 1) / (recent[-1][1] - recent[0][1])
                result[key] = min(source_rate, receipt_rate)
            return result

    def snapshot(self, ros_now, mono_now, previous_images=None, previous_sources=None):
        with self.lock:
            if self.clock_reset:
                self.clock_reset = False
                raise NotReady('ROS clock moved backwards; resume only after fresh data')
            if not all(self.teaching.values()):
                raise NotReady('both arms must report teaching mode')
            if self.motor_faults:
                raise NotReady('motor driver fault: ' + ', '.join(sorted(self.motor_faults)))
            if mono_now - self.button_at > self.state_max_age:
                raise NotReady('right controller heartbeat is stale')
            for key in STREAMS:
                if not self.buffers[key]:
                    raise NotReady(f'waiting for {key}')
                age_limit = self.max_age if key in CAMERAS else self.state_max_age
                if mono_now - self.buffers[key][-1].received > age_limit:
                    raise NotReady(f'{key} feedback stopped')
            if self.required_fps is not None:
                for key, rate in self.rates(mono_now).items():
                    if rate < self.required_fps * .95:
                        raise NotReady(f'{key} measured {rate:.1f} Hz; need {self.required_fps:g} Hz '
                                       '(5% rate tolerance); check source/USB/DDS')
            anchor = min(self.buffers[key][-1].stamp for key in CAMERAS)
            samples = {}
            for key, values in self.buffers.items():
                previous = (previous_sources or {}).get(key, -math.inf)
                if key in CAMERAS:
                    previous = max(previous, (previous_images or {}).get(key, -math.inf))
                candidates = [sample for sample in values if sample.stamp > previous]
                if not candidates:
                    raise AwaitFrame(f'waiting for new {key} feedback; refusing duplicate source samples')
                samples[key] = min(candidates, key=lambda s: abs(s.stamp - anchor))
            stamps = np.array([samples[key].stamp for key in STREAMS], dtype=np.float64)
            if stamps.max() - stamps.min() > self.max_skew:
                raise NotReady('image/state timestamps exceed synchronization tolerance')
            if (ros_now - stamps.min() > self.max_age or stamps.max() - ros_now > .05):
                raise NotReady('source timestamps are stale or in the future')
            values = [np.asarray(samples[key].value).reshape(-1) for key in STREAMS[:4]]
            if tuple(len(v) for v in values) != (7, 7, 1, 1):
                raise NotReady('invalid 16-dimensional joint state')
            state = np.concatenate(values).astype(np.float32)
            if not np.isfinite(state).all():
                raise NotReady('joint state contains non-finite values')
            if np.any(np.abs(state[:14]) > 2 * math.pi):
                raise NotReady('arm feedback is outside joint sanity bounds')
            return Bundle(anchor, state, {key: samples[key].value for key in CAMERAS}, stamps)


class ButtonLatch:
    def __init__(self):
        self.released = False
        self.pressed = False

    def update(self, buttons):
        pressed = bool(buttons & (1 << 3))  # right B/up; A remains drag control.
        if not pressed:
            self.released = True
        edge = pressed and not self.pressed and self.released
        self.pressed = pressed
        return edge


class Capture:
    """B starts/stops+saves; double B discards only the current pending episode."""
    def __init__(self, inputs, writer_factory, fps=5., notify=print):
        self.inputs, self.writer_factory, self.fps, self.notify = inputs, writer_factory, fps, notify
        self.writer = None
        self.recording = False
        self.last_tick = None
        self.next_tick = None
        self.previous_images = None
        self.previous_sources = None
        self.last_button_at = None
        self.save_at = None

    def pause(self, reason, notify=True):
        if self.recording:
            self.recording = False
            self.writer.end_segment(reason)
            if notify:
                self.notify(f'Paused: {reason}. B=save, double B=discard, q=quit; s/d also available')

    def start(self, ros_now, mono_now):
        if self.writer is not None:
            raise RuntimeError('finish the pending episode before starting another')
        try:
            self.inputs.snapshot(ros_now, mono_now)
        except NotReady as exc:
            self.notify(f'Cannot record: {exc}')
            return
        if self.writer is None:
            self.writer = self.writer_factory()
        self.writer.check_error()
        self.writer.begin_segment()
        self.previous_images = None
        self.previous_sources = None
        self.last_tick = mono_now - 1 / self.fps
        self.next_tick = mono_now
        self.recording = True
        self.notify('Recording: B=stop/save, double B=discard; right A still controls teaching')

    def button(self, ros_now, mono_now, pressed_at=None):
        # Use receipt time from the ROS callback, not queue-processing time:
        # two delayed events must retain their actual single/double-click gap.
        pressed_at = mono_now if pressed_at is None else pressed_at
        if self.last_button_at is not None and 0 <= pressed_at - self.last_button_at <= DOUBLE_CLICK_SECONDS:
            self.pause('right B double-click', notify=False)
            self._finish('d')
            return
        if self.save_at is not None:
            # A later, separate press can arrive before tick has committed the
            # prior single click. Finish that episode before starting the next.
            if not self._finish('s'):
                return
        self.last_button_at = pressed_at
        if self.writer is None:
            self.start(ros_now, mono_now)
            if not self.recording:
                self.last_button_at = None
        else:
            self.pause('right B stop', notify=False)
            self.save_at = pressed_at + DOUBLE_CLICK_SECONDS
            self.notify(f'Stopped: saving after {DOUBLE_CLICK_SECONDS:.2f} s double-click window; '
                        'press B again within the window to discard')

    def _finish(self, command):
        self.save_at = None
        self.last_button_at = None
        try:
            self.command(command)
        except (RuntimeError, OSError) as exc:
            self.notify(f'Cannot finish episode: {exc}. Pending data retained; B=retry save, double B=discard')
            return False
        return True

    def tick(self, ros_now, mono_now):
        if self.save_at is not None and mono_now > self.save_at:
            self._finish('s')
        if not self.recording or mono_now < self.next_tick - 1e-6:
            return
        if mono_now - self.last_tick > 2 / self.fps + .02:
            self.pause('capture cycle delayed; no fabricated catch-up frames')
            return
        try:
            self.writer.check_error()
            bundle = self.inputs.snapshot(ros_now, mono_now, self.previous_images, self.previous_sources)
            if self.previous_sources is not None:
                gap_limit = 1.5 / self.fps + min(.02, .1 / self.fps)
                if any(bundle.source_stamps[i] - self.previous_sources[key] > gap_limit
                       for i, key in enumerate(STREAMS)):
                    raise NotReady('camera/state source timestamps skipped a capture period')
            self.writer.add(bundle)
        except AwaitFrame:
            # The timer and camera exposure clocks are independent. Retry on
            # the next loop; the cycle/source-age limits still bound the wait.
            return
        except (NotReady, RuntimeError) as exc:
            self.pause(str(exc))
            return
        self.last_tick = mono_now
        # Keep the requested cadence instead of adding loop/arrival delay to
        # every period (a 5 ms polling loop otherwise turns 30 Hz into ~28 Hz).
        self.next_tick = max(self.next_tick + 1 / self.fps, mono_now)
        self.previous_images = dict(zip(CAMERAS, bundle.source_stamps[4:]))
        self.previous_sources = dict(zip(STREAMS, bundle.source_stamps))

    def command(self, command):
        if command not in ('s', 'd'):
            raise ValueError('expected s or d')
        if self.recording:
            self.notify('Press B to stop/save, or double-click B to discard')
            return
        if self.writer is None:
            self.notify('No pending episode')
            return
        if command == 's':
            path = self.writer.save()
            self.notify(f'Saved: {path}')
        else:
            self.writer.discard()
            self.notify('Pending episode discarded')
        self.writer = None
        self.last_button_at = None
        self.save_at = None
