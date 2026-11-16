"""Feedback-anchored PICO position jogging through existing motor ROS nodes."""

import math
from pathlib import Path
import time

from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
import yaml


DEADZONE = 0.15
FEEDBACK_TIMEOUT = 0.25


def load_head_pitch_settings(path=None):
    if path is None:
        path = (Path(__file__).resolve().parents[3] /
                'headmotor/yiyou/yiyou_niic_ros/config/headmotor.yaml')
    saved = yaml.safe_load(Path(path).read_text())['headmotor']['ros__parameters']
    counts = float(saved['pha_counts_per_revolution'])
    direction = float(saved['pha_direction'])
    speed = float(saved.get('pha_csp_velocity_deg_s', saved['profile_velocity_deg_s']))
    lookahead_s = float(saved.get('pha_pico_lookahead_s', 0.1))
    if (not saved['pha_limits_enabled'] or not math.isfinite(counts) or counts <= 0
            or direction not in (-1., 1.) or not math.isfinite(speed) or speed <= 0
            or not math.isfinite(lookahead_s) or lookahead_s <= 0):
        raise ValueError('head PHA requires enabled calibrated limits and positive speed')
    tolerance = float(saved['position_tolerance_counts']) * 360.0 / counts
    bounds = sorted((float(saved[key]) - float(saved['pha_zero_counts'])) *
                    direction * 360.0 / counts for key in
                    ('pha_lower_limit_counts', 'pha_upper_limit_counts'))
    if (not all(math.isfinite(v) for v in (*bounds, speed, tolerance))
            or tolerance < 0 or bounds[0] >= bounds[1]):
        raise ValueError('invalid PHA calibration, speed or limits')
    return dict(speed_deg_s=speed, lower=bounds[0], upper=bounds[1], tolerance=tolerance,
                lookahead_s=lookahead_s)


class PositionJogBridge:
    """Start, stream bounded targets, then hold on every deadman loss.

    Head start is asynchronous inside its driver: service success alone is not
    readiness. Waist resume immediately acknowledges an already enabled drive.
    """
    def __init__(self, node, name, base_topic, start_service, stop_service, *,
                 status_topic, async_start, speed_deg_s, direction, lower, upper,
                 tolerance, lookahead_s=0.1, preserve_direction=False,
                 feedback_qos=None):
        self.node, self.name = node, name
        self.async_start = async_start
        self.speed, self.direction = speed_deg_s, direction
        self.lower, self.upper, self.tolerance = lower, upper, tolerance
        self.lookahead_s = lookahead_s
        self.preserve_direction = preserve_direction
        self.publisher = node.create_publisher(Float64, base_topic + '/target_deg', 1)
        self.position_sub = node.create_subscription(
            Float64, base_topic + '/position_deg', self.on_position,
            1 if feedback_qos is None else feedback_qos)
        self.status_sub = node.create_subscription(String, status_topic, self.on_status, 10)
        self.start_client = node.create_client(Trigger, start_service)
        self.stop_client = node.create_client(Trigger, stop_service)
        self.phase = 'idle'
        self.driver_phase = None
        self.status_at = -math.inf
        self.position = None
        self.position_at = -math.inf
        self.future = None
        self.target = None
        self.last_tick = None
        self.started_at = None
        self.cancelled = False
        self.armed = False
        self.closing = False
        self.last_warning = -math.inf

    def on_position(self, message):
        if math.isfinite(message.data):
            self.position, self.position_at = message.data, time.monotonic()

    def on_status(self, message):
        phase = message.data.partition(':')[0]
        if phase in ('idle', 'configuring', 'enabling', 'running', 'stopping',
                     'holding', 'fault', 'stopped', 'error'):
            self.driver_phase, self.status_at = phase, time.monotonic()

    def warn(self, message, now):
        if now - self.last_warning >= 2.0:
            self.node.get_logger().warning(f'{self.name}: {message}')
            self.last_warning = now

    def fail(self, message, now):
        self.warn(message, now)
        self.armed = False
        self.phase = 'blocked'
        self.future = None

    @property
    def pending(self):
        return self.phase in ('starting', 'running', 'stopping')

    def feedback_ready(self, now):
        return (self.position is not None and now - self.position_at <= FEEDBACK_TIMEOUT
                and self.lower - self.tolerance <= self.position <= self.upper + self.tolerance)

    def stop(self, now):
        if self.phase == 'starting':
            self.cancelled = True  # Do not interrupt the head's enable sequence.
        elif self.phase == 'running':
            self.phase = 'stopping'
            self.started_at = now
            self.future = None
            if self.stop_client.service_is_ready():
                self.future = self.stop_client.call_async(Trigger.Request())
            else:
                self.fail('hold service unavailable; check motor driver', now)

    def shutdown(self):
        self.closing = True
        self.stop(time.monotonic())

    def tick(self, hand, value, now, allowed):
        fresh = hand is not None and now - hand.received <= FEEDBACK_TIMEOUT
        if fresh and not hand.grip:
            self.armed = True
        desired = (not self.closing and fresh and hand.grip
                   and allowed and self.armed and abs(value) > DEADZONE)
        if self.phase == 'blocked':
            if fresh and not hand.grip:
                self.phase = 'idle'
            return
        if self.phase == 'stopping':
            if self.future is not None and self.future.done():
                try:
                    result = self.future.result()
                    if not result.success:
                        raise RuntimeError(result.message)
                except Exception as exc:
                    self.fail(f'hold failed: {exc}', now)
                    return
                if not self.async_start or (self.driver_phase in ('holding', 'idle')
                                             and self.status_at >= self.started_at):
                    self.phase, self.future = 'idle', None
                    return
            if now - self.started_at > 8.0:
                self.fail('hold acknowledgement timed out; check motor state', now)
            return
        if self.phase == 'starting':
            self.cancelled |= not desired
            if now - self.started_at > 8.0:
                self.cancelled = True
                self.armed = False
                self.warn('start timed out; waiting to hold; check motor driver', now)
            if self.future.done():
                try:
                    result = self.future.result()
                    if not result.success:
                        raise RuntimeError(result.message)
                except Exception as exc:
                    self.fail(f'start failed: {exc}', now)
                    return
                if self.async_start:
                    if (self.driver_phase in ('fault', 'stopped', 'error')
                            and self.status_at >= self.started_at):
                        self.fail('driver fault while starting; release Grip and inspect driver', now)
                        return
                    if self.driver_phase != 'running' or self.status_at < self.started_at:
                        return
                if (not self.cancelled and
                        (not self.feedback_ready(now) or self.position_at < self.started_at)):
                    return  # Hold at the driver's anchor until post-start feedback arrives.
                self.phase = 'running'
                if not self.async_start:
                    self.driver_phase = 'running'  # Resume just verified the enabled state.
                self.target = self.position
                self.last_tick = now
                if self.cancelled:
                    self.stop(now)
                else:
                    self.target = min(self.upper, max(self.lower, self.target))
                    self.publisher.publish(Float64(data=float(self.target)))
                    self.node.get_logger().info(f'{self.name}: joystick control ready')
            return
        if self.phase == 'running':
            healthy = self.feedback_ready(now)
            if self.async_start:
                healthy &= (self.driver_phase == 'running'
                            and now - self.status_at <= FEEDBACK_TIMEOUT)
            else:
                healthy &= self.driver_phase not in ('fault', 'stopped', 'error')
            if not desired or not healthy:
                if not healthy:
                    self.armed = False
                    self.warn('feedback stale, outside limits, or driver stopped; holding', now)
                self.stop(now)
                return
            dt = min(0.01, max(0.0, now - self.last_tick))
            self.last_tick = now
            delta = math.copysign(self.speed, value) * self.direction * dt
            target = self.target + delta
            # Bound queued movement to the configured lead from feedback.
            lookahead = self.speed * self.lookahead_s
            bounded = min(self.upper, self.position + lookahead,
                          max(self.lower, self.position - lookahead, target))
            if self.preserve_direction:
                # Feedback limits further progress; it must not pull an issued
                # waist target backwards or jump ahead faster than the jog.
                if delta > 0.0:
                    bounded = min(target, max(self.target, bounded))
                elif delta < 0.0:
                    bounded = max(target, min(self.target, bounded))
                else:
                    bounded = self.target
            self.target = bounded
            self.publisher.publish(Float64(data=float(self.target)))
            return
        if not desired:
            return
        if (not self.feedback_ready(now) or self.publisher.get_subscription_count() == 0
                or not self.start_client.service_is_ready() or not self.stop_client.service_is_ready()):
            self.warn('waiting for CSP services, target subscriber and in-range position feedback', now)
            return
        self.started_at = now
        self.cancelled = False
        self.phase = 'starting'
        self.future = self.start_client.call_async(Trigger.Request())
