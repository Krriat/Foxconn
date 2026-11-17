#!/usr/bin/env python3
"""Bridge PICO UDP controllers to the ROS ServoL jog topics.

The ROS driver owns the SDK connection. Grip on each controller requests power
on for its matching arm before enabling teleoperation; releasing Grip or losing
UDP data stops that arm's ServoL mode.
Horizontal sticks change the matching arm elbow; vertical sticks jog head PHA
(right) and waist pitch (left) through their CSP ROS drivers.
Use --no-arms to jog head and waist without any arm interfaces.
The right controller's A/Down button toggles both arms' Cartesian drag teaching.
Teaching suspends both arms' teleoperation while head/waist remain available.
"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import socket
import struct
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rokae_interfaces.msg import ServoLJog
from rokae_interfaces.srv import GetCartesianState
from std_msgs.msg import Bool, String, UInt8
from std_srvs.srv import Trigger
from pico_auxiliary_jog import PositionJogBridge, load_head_pitch_settings


PREFIX = '/robot_02/upperlimb'
PERIOD = 0.01
UDP_TIMEOUT = 0.25
POSITION_STEP = 0.0005
ROTATION_STEP = math.radians(0.25)
ELBOW_JOYSTICK_HALF_ANGLE_DEG = 30.0


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def directional_joystick(value, cross_axis):
    """Accept an axis within its positive/negative 30 degree cones."""
    angle = math.atan2(abs(cross_axis), abs(value))
    if angle > math.radians(ELBOW_JOYSTICK_HALF_ANGLE_DEG) + 1e-12:
        return 0.0
    return float(value)


def elbow_joystick_x(x, y):
    return directional_joystick(x, y)


def pitch_joystick_y(x, y):
    return directional_joystick(y, x)


def norm3(vector):
    return math.sqrt(sum(value * value for value in vector))


def normalize(q):
    length = math.sqrt(sum(value * value for value in q))
    if not math.isfinite(length) or length < 1e-8:
        raise ValueError('invalid quaternion')
    return tuple(value / length for value in q)


def multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw-ax*bx-ay*by-az*bz,
            aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw)


def inverse(q):
    return (q[0], -q[1], -q[2], -q[3])


def axis_quaternion(axis, angle):
    half = angle / 2.0
    value = math.sin(half)
    return (math.cos(half), axis[0]*value, axis[1]*value, axis[2]*value)


def from_rpy(roll, pitch, yaw):
    return normalize(multiply(
        multiply(axis_quaternion((0, 0, 1), yaw),
                 axis_quaternion((0, 1, 0), pitch)),
        axis_quaternion((1, 0, 0), roll)))


def angle_between(a, b):
    return 2.0 * math.acos(clamp(abs(sum(x*y for x, y in zip(a, b))), 0, 1))


def slerp(a, b, ratio):
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0:
        b = tuple(-value for value in b)
        dot = -dot
    dot = clamp(dot, 0, 1)
    if dot > 0.9995:
        return normalize(tuple(x + ratio*(y-x) for x, y in zip(a, b)))
    theta = math.acos(dot)
    scale = math.sin(theta)
    left = math.sin((1-ratio)*theta) / scale
    right = math.sin(ratio*theta) / scale
    return normalize(tuple(left*x + right*y for x, y in zip(a, b)))


def pico_orientation(qx, qy, qz, qw):
    qw, qx, qy, qz = normalize((qw, qx, qy, qz))
    roll = math.atan2(2*(qw*qx + qy*qz), 1-2*(qx*qx + qy*qy))
    pitch = math.asin(clamp(2*(qw*qy-qz*qx), -1, 1))
    yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return from_rpy(pitch, roll-math.pi/4, -yaw-math.pi/2)


@dataclass(frozen=True)
class Hand:
    position: tuple
    orientation: tuple
    joystick_y: float
    grip: bool
    received: float
    down: bool = False
    joystick_x: float = 0.0
    buttons: int = 0


class PicoReceiver:
    def __init__(self, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', port))
        self.socket.settimeout(0.1)
        self.lock = threading.Lock()
        self.hands = [None, None]
        self.raw = [None, None]
        self.running = True
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def receive(self):
        while self.running:
            try:
                data, _ = self.socket.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != 38 or data[0] not in (0, 1):
                continue
            values = struct.unpack('<9f', data[1:37])
            if not all(math.isfinite(value) for value in values):
                continue
            try:
                orientation = pico_orientation(*values[3:7])
            except ValueError:
                continue
            side = data[0]
            grip = bool(data[37] & (1 << 1))
            down = bool(data[37] & (1 << 2))
            position = tuple(value/1000 for value in values[:3])
            now = time.monotonic()
            with self.lock:
                previous_raw = self.raw[side]
                if previous_raw is not None and grip:
                    jump = norm3(tuple(x-y for x, y in
                                       zip(position, previous_raw[0])))
                    turn = angle_between(orientation, previous_raw[1])
                    if jump > 0.10 or turn > math.radians(45):
                        # Grip release must still pass through to stop motion.
                        continue
                self.raw[side] = (position, orientation)
                previous = self.hands[side]
                if previous is not None:
                    position = tuple(x + 0.20*(y-x) for x, y in
                                     zip(previous.position, position))
                    orientation = slerp(previous.orientation, orientation, 0.15)
                self.hands[side] = Hand(position, orientation,
                                        clamp(values[8], -1.0, 1.0), grip, now, down,
                                        clamp(values[7], -1.0, 1.0), data[37])

    def latest(self):
        with self.lock:
            return tuple(self.hands)

    def close(self):
        self.running = False
        self.socket.close()
        self.thread.join(timeout=0.2)


class ArmBridge:
    def __init__(self, node, side, qos):
        self.node = node
        self.side = side
        self.name = side + '_arm'
        self.publisher = node.create_publisher(
            ServoLJog, f'{PREFIX}/servol/jog/{self.name}', qos)
        self.stop_publisher = node.create_publisher(
            Bool, f'{PREFIX}/servol/stop/{self.name}', 10)
        self.client = node.create_client(
            GetCartesianState, f'{PREFIX}/get_cartesian_state/{self.name}')
        self.power_client = node.create_client(
            Trigger, f'{PREFIX}/power_on/{self.name}')
        self.teach_client = node.create_client(
            Trigger, f'{PREFIX}/cartesian_drag_safe/start/{self.name}')
        self.teach_stop_client = node.create_client(
            Trigger, f'{PREFIX}/cartesian_drag_safe/stop/{self.name}')
        teach_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.teach_status_sub = node.create_subscription(
            String, f'{PREFIX}/cartesian_drag_safe/status/{self.name}',
            self.on_teach_status, teach_qos)
        self.command_pose_sub = node.create_subscription(
            PoseStamped, f'{PREFIX}/servol/command_pose/{self.name}',
            self.on_command_pose, qos)
        self.phase = 'idle'
        self.armed = False
        self.future = None
        self.hand_anchor = None
        self.robot_anchor = None
        self.command_position = None
        self.command_orientation = None
        self.awaiting_driver_anchor = False
        self.driver_started = False
        self.last_log = 0.0
        self.teach_requested = False
        self.teach_future = None
        self.teach_stop_future = None
        self.teach_start_seen = False
        self.teach_stop_ack = False
        self.servol_cleanup_failed = False
        self.down_armed = False
        self.down_was_pressed = False

    @property
    def teaching_pending(self):
        return self.teach_requested or self.phase in (
            'teach_starting', 'teaching', 'teach_stopping', 'teach_failed')

    @property
    def pending(self):
        return (self.phase == 'stopping' or
                self.phase in ('teach_starting', 'teaching', 'teach_stopping') or
                (self.teach_future is not None and not self.teach_future.done()))

    def stop(self, reason):
        if self.phase in ('teach_starting', 'teaching', 'teach_stopping'):
            self.request_teach_stop(reason)
        elif self.phase == 'active':
            self.stop_publisher.publish(Bool(data=True))
            self.phase = 'stopping'
            self.node.get_logger().info(f'{self.name} ServoL stopped: {reason}')
        elif self.phase in ('powering', 'requesting'):
            if self.future is not None:
                self.future.cancel()
            self.phase = 'idle'
            self.future = None

    def suspend_teleop(self):
        self.armed = False
        if self.phase in ('active', 'powering', 'requesting'):
            self.stop('Cartesian drag selected: both arms teleoperation suspended')

    def on_status(self, state):
        if self.phase in ('active', 'stopping') and state.startswith('error:'):
            self.node.get_logger().error(f'{self.name} ServoL {state}')
        if self.phase == 'active' and state in ('ready', 'running'):
            self.driver_started = True
        if self.phase == 'stopping' and (state == 'stopped' or state.startswith('error:')):
            self.phase = 'idle'
            self.servol_cleanup_failed = state != 'stopped'
            if state == 'stopped':
                self.node.get_logger().info(f'{self.name} ServoL stopped; NrtCommand restored')
            else:
                self.teach_requested = False
                self.armed = False
        elif self.phase == 'active' and state.startswith('error:'):
            self.phase = 'idle'
            self.armed = False
            self.servol_cleanup_failed = 'cleanup' in state

    def update_a(self, hand, now):
        if hand is None or now - hand.received > UDP_TIMEOUT:
            return False
        if not hand.down:
            self.down_armed = True
        pressed = hand.down and not self.down_was_pressed and self.down_armed
        self.down_was_pressed = hand.down
        return pressed

    def toggle_teach(self):
        self.armed = False
        if self.teach_requested:
            self.teach_requested = False
            self.node.get_logger().info(f'{self.name} A: queued teaching cancelled')
            return
        if self.phase in ('teach_starting', 'teaching', 'teach_stopping'):
            self.request_teach_stop('A pressed')
            return
        if self.phase == 'teach_failed':
            self.node.get_logger().error(
                f'{self.name} drag cleanup unverified; inspect controller before restarting bridge')
            return
        self.stop('A pressed: Cartesian drag requested')
        if not (self.teach_client.service_is_ready() and self.teach_stop_client.service_is_ready()):
            self.node.get_logger().error(
                f'{self.name} teaching unavailable: enable start_cartesian_drag_safe_service:=true')
            return
        self.teach_requested = True
        self.node.get_logger().info(f'{self.name} A: Cartesian drag requested')

    def on_teach_status(self, message):
        state = message.data
        if self.phase not in ('teach_starting', 'teaching', 'teach_stopping', 'teach_failed'):
            return
        if state in ('starting', 'teaching'):
            self.teach_start_seen = True
            if state == 'teaching' and self.phase == 'teach_starting':
                self.phase = 'teaching'
                self.node.get_logger().info(f'{self.name} teaching enabled; hold physical drag button')
        elif state.startswith('error:') and self.teach_start_seen:
            self.phase = 'teach_failed'
            self.armed = False
            self.node.get_logger().error(f'{self.name} drag {state}')
        # A stop service only acknowledges a stop request. The start future
        # completes after disableDrag and command-lock release, authoritatively.

    def request_teach_stop(self, reason):
        if self.phase != 'teach_stopping':
            self.node.get_logger().info(f'{self.name} teaching stop requested: {reason}')
        self.phase = 'teach_stopping'
        self.armed = False
        if self.teach_stop_future is None and self.teach_stop_client.service_is_ready():
            self.teach_stop_future = self.teach_stop_client.call_async(Trigger.Request())

    def tick_teach(self, fresh, allow_start=True):
        if not fresh:
            self.teach_requested = False
            if self.phase in ('teach_starting', 'teaching', 'teach_stopping'):
                self.request_teach_stop('PICO UDP timed out or bridge exiting')
        if self.teach_requested:
            if self.phase != 'idle' or not allow_start:
                return
            self.teach_requested = False
            self.armed = False
            if not (self.teach_client.service_is_ready() and self.teach_stop_client.service_is_ready()):
                self.node.get_logger().error(f'{self.name} teaching interfaces disappeared; press A again')
                return
            self.teach_start_seen = False
            self.teach_stop_ack = False
            self.teach_stop_future = None
            self.phase = 'teach_starting'
            self.teach_future = self.teach_client.call_async(Trigger.Request())
            return
        if self.teach_future is not None and self.teach_future.done():
            try:
                response = self.teach_future.result()
            except Exception as error:
                # A lost service response does not prove that drag stopped.
                if self.teach_stop_client.service_is_ready():
                    self.teach_stop_client.call_async(Trigger.Request())
                self.phase = 'teach_failed'
                self.node.get_logger().error(f'{self.name} teaching completion unknown: {error}')
            else:
                if response.success:
                    self.node.get_logger().info(f'{self.name} teaching stopped: {response.message}')
                else:
                    self.node.get_logger().error(f'{self.name} teaching failed: {response.message}')
                if response.message.startswith('cleanup unverified:'):
                    self.phase = 'teach_failed'
                elif self.phase != 'teach_failed':
                    self.phase = 'idle'
            self.armed = False
            self.teach_future = None
            self.teach_stop_future = None
            return
        if self.phase == 'teach_stopping' and not self.teach_stop_ack:
            if self.teach_stop_future is None:
                if self.teach_start_seen and self.teach_stop_client.service_is_ready():
                    self.teach_stop_future = self.teach_stop_client.call_async(Trigger.Request())
            elif self.teach_stop_future.done():
                try:
                    response = self.teach_stop_future.result()
                    self.teach_stop_ack = response.success
                except Exception as error:
                    self.node.get_logger().error(f'{self.name} teaching stop request failed: {error}')
                self.teach_stop_future = None
                # If stop arrived before the start callback, retry only once
                # the driver's starting/teaching status confirms it is active.

    def on_command_pose(self, message):
        if not (self.phase == 'active' and self.awaiting_driver_anchor
                and self.driver_started):
            return
        pose = message.pose
        position = (pose.position.x, pose.position.y, pose.position.z)
        try:
            orientation = normalize((pose.orientation.w, pose.orientation.x,
                                     pose.orientation.y, pose.orientation.z))
        except ValueError:
            return
        if not all(math.isfinite(value) for value in position):
            return
        self.robot_anchor = (position, orientation)
        self.command_position, self.command_orientation = self.robot_anchor
        hand = self.node.receiver.latest()[0 if self.side == 'left' else 1]
        if hand is not None:
            self.hand_anchor = (hand.position, hand.orientation)
        self.awaiting_driver_anchor = False
        self.node.get_logger().info(f'{self.name} ServoL driver ready; anchors aligned')

    def tick(self, hand, now, *, teleop_allowed=True,
             teach_start_allowed=True):
        fresh = hand is not None and now-hand.received <= UDP_TIMEOUT
        if self.teaching_pending or self.teach_future is not None:
            self.tick_teach(fresh, teach_start_allowed)
            return
        if not teleop_allowed:
            self.suspend_teleop()
            return
        requested = fresh and hand.grip
        if not requested:
            if fresh and not hand.grip:
                self.armed = True
            reason = 'Grip released' if fresh else 'PICO UDP timed out or no accepted packet'
            self.stop(reason)
            return
        if self.phase == 'stopping':
            return
        if self.phase == 'idle':
            if not self.armed:
                return
            ready = (self.power_client.service_is_ready() and
                     self.publisher.get_subscription_count() > 0 and self.client.service_is_ready())
            if not ready:
                if now-self.last_log > 2.0:
                    self.node.get_logger().warn(
                        f'{self.name}: enable start_servol:=true, '
                        'start_movel_service:=true and start_initializer_service:=true')
                    self.last_log = now
                return
            self.armed = False
            self.future = self.power_client.call_async(Trigger.Request())
            self.phase = 'powering'
            self.node.get_logger().info(f'{self.name} Grip: requesting power on')
            return
        if self.phase == 'powering':
            if not self.future.done():
                return
            try:
                response = self.future.result()
                if not response.success:
                    raise RuntimeError(response.message)
                self.node.get_logger().info(
                    f'{self.name} power on ready: {response.message}')
                self.future = self.client.call_async(GetCartesianState.Request())
                self.phase = 'requesting'
            except Exception as error:
                self.node.get_logger().error(f'{self.name} power on failed: {error}')
                self.phase = 'idle'
                self.future = None
                self.armed = False
            return
        if self.phase == 'requesting':
            if not self.future.done():
                return
            try:
                response = self.future.result()
                if not response.success or not all(math.isfinite(v) for v in response.pose):
                    raise RuntimeError(response.message)
                self.robot_anchor = (
                    tuple(response.pose[:3]), from_rpy(*response.pose[3:6]))
            except Exception as error:
                self.node.get_logger().error(f'{self.name} pose read failed: {error}')
                self.phase = 'idle'
                self.armed = False
                return
            self.hand_anchor = (hand.position, hand.orientation)
            self.command_position, self.command_orientation = self.robot_anchor
            self.armed = False
            self.phase = 'active'
            self.awaiting_driver_anchor = True
            self.driver_started = False
            self.node.get_logger().info(f'{self.name} PICO ServoL requested')

        if self.awaiting_driver_anchor:
            # Keep requesting the current pose while either driver's RT setup
            # is in progress. Hand motion and elbow input begin only after this
            # arm publishes its ready anchor, independently of the other arm.
            self.publish_target(0.0)
            return

        raw_offset = tuple(x-y for x, y in zip(hand.position, self.hand_anchor[0]))
        rotation_offset = multiply(hand.orientation, inverse(self.hand_anchor[1]))
        if norm3(raw_offset) > 0.40 or angle_between(
                rotation_offset, (1, 0, 0, 0)) > math.radians(120):
            self.stop('PICO tracking offset exceeded the allowed range')
            return
        position_goal = tuple(x + 0.30*y for x, y in
                              zip(self.robot_anchor[0], raw_offset))
        rotation_angle = angle_between(rotation_offset, (1, 0, 0, 0))
        rotation_gain = 0.25 if rotation_angle >= math.radians(0.3) else 0.0
        rotation_goal = multiply(
            slerp((1, 0, 0, 0), rotation_offset, rotation_gain),
            self.robot_anchor[1])
        difference = tuple(x-y for x, y in
                           zip(position_goal, self.command_position))
        distance = norm3(difference)
        if distance >= 0.0005:
            ratio = min(1.0, POSITION_STEP/distance)
            self.command_position = tuple(x+ratio*y for x, y in
                                          zip(self.command_position, difference))
        turn = angle_between(self.command_orientation, rotation_goal)
        if turn >= math.radians(0.3):
            ratio = min(1.0, ROTATION_STEP/turn)
            self.command_orientation = slerp(
                self.command_orientation, rotation_goal, ratio)

        # ServoLJog's legacy field name is joystick_y; it now carries gated X.
        self.publish_target(elbow_joystick_x(hand.joystick_x, hand.joystick_y))

    def publish_target(self, joystick_y):
        message = ServoLJog()
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            self.command_position)
        (message.pose.orientation.w, message.pose.orientation.x,
         message.pose.orientation.y, message.pose.orientation.z) = (
             self.command_orientation)
        message.joystick_y = float(joystick_y)
        self.publisher.publish(message)


class PicoServoLBridge(Node):
    def __init__(self, receiver, head_config=None, head_enabled=True, waist_enabled=True,
                 arms_enabled=True):
        super().__init__('pico_servol_bridge')
        self.receiver = receiver
        self.shutting_down = False
        self.teach_mode = False
        self.headwaist_pause = False
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.arms = ([ArmBridge(self, side, qos) for side in ('left', 'right')]
                     if arms_enabled else [])
        self.record_buttons = (self.create_publisher(
            UInt8, f'{PREFIX}/pico/buttons/right_arm', qos) if arms_enabled else None)
        self.auxiliary_axes = []
        if head_enabled:
            settings = load_head_pitch_settings(head_config)
            self.auxiliary_axes.append((1, PositionJogBridge(
                self, 'head_pitch', '/headmotor/csp/pha',
                '/headmotor/csp/start', '/headmotor/csp/stop',
                status_topic='/headmotor/csp/status', async_start=True,
                speed_deg_s=settings['speed_deg_s'], direction=1.0,
                lower=settings['lower'], upper=settings['upper'],
                tolerance=settings['tolerance'], lookahead_s=settings['lookahead_s'])))
        if waist_enabled:
            self.auxiliary_axes.append((0, PositionJogBridge(
                self, 'waist_pitch', '/robot/waist_pitch/csp',
                '/robot/waist_pitch/csp/resume', '/robot/waist_pitch/csp/hold',
                status_topic='/robot/waist_pitch/csp/status', async_start=False,
                speed_deg_s=10.0, direction=-1.0,
                lower=0.0, upper=50.0, tolerance=0.1, preserve_direction=True,
                # Live feedback must not wait for reliable retransmission:
                # delayed positions repeatedly stall the 1-degree lead bound.
                feedback_qos=qos)))
        self.status_sub = (self.create_subscription(
            String, f'{PREFIX}/servol/status', self.on_status, 10) if self.arms else None)
        self.timer = self.create_timer(PERIOD, self.tick)

    def on_status(self, message):
        side, separator, state = message.data.partition(':')
        if self.arms and separator and side in ('left', 'right'):
            self.arms[0 if side == 'left' else 1].on_status(state)

    def stop_dual_teach(self, reason):
        for arm in self.arms:
            arm.teach_requested = False
            if arm.phase in ('teach_starting', 'teaching', 'teach_stopping'):
                arm.request_teach_stop(reason)

    def toggle_dual_teach(self, hands, now):
        if any(arm.teaching_pending for arm in self.arms):
            self.stop_dual_teach('right A pressed: stop both arms')
            return
        if not all(hand is not None and now - hand.received <= UDP_TIMEOUT for hand in hands):
            self.get_logger().warn('Dual-arm teaching requires fresh UDP data from both controllers')
            return
        if any(arm.servol_cleanup_failed for arm in self.arms):
            self.get_logger().error('Teaching cancelled: ServoL cleanup was not verified')
            return
        if not all(arm.teach_client.service_is_ready() and
                   arm.teach_stop_client.service_is_ready() for arm in self.arms):
            self.get_logger().error(
                'Dual-arm teaching unavailable: enable start_cartesian_drag_safe_service:=true')
            return
        for arm in self.arms:
            arm.toggle_teach()

    def tick(self):
        if self.shutting_down:
            return
        now = time.monotonic()
        hands = self.receiver.latest()
        buttons_publisher = getattr(self, 'record_buttons', None)
        if buttons_publisher is not None and hands[1] is not None and now - hands[1].received <= UDP_TIMEOUT:
            buttons_publisher.publish(UInt8(data=hands[1].buttons))
        # Only right A controls the pair; read edges before any arm target.
        edges = [arm.update_a(hand, now) for arm, hand in zip(self.arms, hands)]
        if len(edges) == 2 and edges[1]:
            self.toggle_dual_teach(hands, now)
        if any(arm.teaching_pending for arm in self.arms):
            if (not all(arm.teaching_pending for arm in self.arms)
                    or any(arm.phase == 'teach_failed' for arm in self.arms)
                    or not all(hand is not None and now - hand.received <= UDP_TIMEOUT for hand in hands)):
                self.stop_dual_teach('dual-arm session ended, failed or UDP timed out')
        requested_teach = any(arm.teaching_pending for arm in self.arms)
        if requested_teach:
            self.headwaist_pause = True
            for arm in self.arms:
                arm.suspend_teleop()
        if self.teach_mode and not requested_teach:
            for arm in self.arms:
                arm.armed = False  # Require a fresh release/repress after teaching.
        self.teach_mode = requested_teach
        clean = not any(arm.servol_cleanup_failed for arm in self.arms)
        if requested_teach and not clean:
            for arm in self.arms:
                if arm.teach_requested:
                    arm.teach_requested = False
                    self.get_logger().error('Teaching cancelled: ServoL cleanup was not verified')
        allow_start = clean and all(arm.phase not in (
            'active', 'stopping', 'powering', 'requesting') for arm in self.arms)
        for arm, hand in zip(self.arms, hands):
            arm.tick(hand, now, teleop_allowed=not self.teach_mode,
                     teach_start_allowed=allow_start)
        if not self.teach_mode and any(arm.phase in ('powering', 'requesting', 'active') for arm in self.arms):
            self.headwaist_pause = False
        for side, axis in self.auxiliary_axes:
            allowed = True
            if self.arms and not self.teach_mode and not self.headwaist_pause:
                arm = self.arms[side]
                allowed = arm.phase == 'active' and not arm.awaiting_driver_anchor
            hand = hands[side]
            value = pitch_joystick_y(hand.joystick_x, hand.joystick_y) if hand else 0.0
            axis.tick(hand, value, now, allowed)

    def stop_all(self):
        self.shutting_down = True
        self.timer.cancel()
        for arm in self.arms:
            arm.teach_requested = False
            arm.stop('bridge exiting')
        for _, axis in getattr(self, 'auxiliary_axes', ()):
            axis.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--udp-port', type=int, default=5050)
    parser.add_argument('--head-config', type=Path, default=None,
                        help='same headmotor.yaml used by the head CSP driver')
    parser.add_argument('--no-head', action='store_true', help='disable head joystick control')
    parser.add_argument('--no-waist', action='store_true', help='disable waist joystick control')
    parser.add_argument('--no-arms', action='store_true',
                        help='control head/waist independently; do not create arm interfaces')
    args = parser.parse_args()
    receiver = PicoReceiver(args.udp_port)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PicoServoLBridge(receiver, args.head_config, not args.no_head, not args.no_waist,
                            arms_enabled=not args.no_arms)
    node.get_logger().info(
        f'PICO UDP {args.udp_port} ready; release each Grip once to arm it')
    if args.no_arms:
        node.get_logger().info('Head/waist only: right Grip + Y for head, left Grip + Y for waist')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_all()
        deadline = time.monotonic() + 3.0
        while ((any(arm.pending for arm in node.arms)
                or any(axis.pending for _, axis in node.auxiliary_axes))
               and time.monotonic() < deadline):
            rclpy.spin_once(node, timeout_sec=0.02)
            for arm in node.arms:
                arm.tick_teach(False, allow_start=False)
            for _, axis in node.auxiliary_axes:
                axis.tick(None, 0.0, time.monotonic(), False)
        if (any(arm.pending for arm in node.arms)
                or any(axis.pending for _, axis in node.auxiliary_axes)):
            node.get_logger().error('Stop acknowledgement timed out; check the driver and robot state')
        node.destroy_node()
        receiver.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
