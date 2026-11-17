"""Subscribe to the existing drivers. No SDK, CAN or motor-command interfaces."""
import math
import queue
import time

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, String, UInt8

from .core import ButtonLatch, CAMERAS, Inputs
from .udp_preview import UdpPreview, PORTS

PREFIX = '/robot_02/upperlimb'


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec / 1e9


def joint_values(message, names):
    if len(message.name) != len(message.position) or len(set(message.name)) != len(message.name):
        raise ValueError('invalid JointState names/positions')
    mapping = dict(zip(message.name, message.position))
    values = np.array([mapping[name] for name in names], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('non-finite joint feedback')
    return values


def decode_rgb(message, width, height):
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(message.encoding)
    if not channels or (message.width, message.height) != (width, height):
        raise ValueError('camera encoding or resolution does not match recorder configuration')
    if message.step < width * channels or len(message.data) < height * message.step:
        raise ValueError('invalid image stride/payload')
    raw = np.frombuffer(message.data, dtype=np.uint8, count=height * message.step)
    image = raw.reshape(height, message.step)[:, :width * channels].reshape(height, width, channels)
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = image[:, :, :3]
        if message.encoding.startswith('bgr'):
            image = image[:, :, ::-1]
    return image.copy()


class RecorderNode(Node):
    def __init__(self, args):
        super().__init__('act_teach_recorder', namespace='/robot_02/upperlimb')
        self.inputs = Inputs(args.max_skew, args.max_age, args.state_max_age, required_fps=args.fps)
        self.commands = queue.SimpleQueue()
        self.buttons = ButtonLatch()
        self.last_warning = -math.inf
        self.udp_preview = None
        self.subscriptions_kept = []
        sensor = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        latest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        status = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        image_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        for side in ('left', 'right'):
            self.subscriptions_kept.append(self.create_subscription(
                JointState, f'{PREFIX}/cartesian_drag_safe/joint_states/{side}_arm',
                lambda msg, key=side: self.on_joints(key, msg), sensor))
            self.subscriptions_kept.append(self.create_subscription(
                String, f'{PREFIX}/cartesian_drag_safe/status/{side}_arm',
                lambda msg, key=side: self.on_teach(key, msg), status))
        self.subscriptions_kept.append(self.create_subscription(
            JointState, '/headmotor/csp/joint_states', lambda msg: self.on_joints('head_pha', msg), latest))
        self.subscriptions_kept.append(self.create_subscription(
            Float64, '/robot/waist_pitch/csp/position_deg', self.on_waist, latest))
        self.subscriptions_kept.append(self.create_subscription(
            UInt8, f'{PREFIX}/pico/buttons/right_arm', self.on_buttons, latest))
        for key, topic in (('head_pha', '/headmotor/csp/status'),
                           ('waist_pitch', '/robot/waist_pitch/csp/status')):
            self.subscriptions_kept.append(self.create_subscription(
                String, topic, lambda msg, name=key: self.on_motor_status(name, msg), 10))
        for camera in CAMERAS:
            self.subscriptions_kept.append(self.create_subscription(
                Image, f'/robot/sensors/cameras/{camera}/color/image_raw',
                lambda msg, key=camera: self.on_image(key, msg, args.width, args.height), image_qos))
        if not args.no_udp_video:
            try:
                self.udp_preview = UdpPreview(args.udp_video_ip, args.udp_video_jpeg_quality,
                                              on_error=self.get_logger().warning)
            except OSError as exc:
                self.get_logger().warning(f'UDP preview unavailable: {exc}; recording remains independent')
            else:
                for camera, port in PORTS.items():
                    self.get_logger().info(f'UDP JPEG {camera} -> {args.udp_video_ip}:{port}')

    def destroy_node(self):
        if self.udp_preview is not None:
            self.udp_preview.close()
        return super().destroy_node()

    def ros_now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def warn(self, message):
        now = time.monotonic()
        if now - self.last_warning > 2:
            self.get_logger().warning(str(message))
            self.last_warning = now

    def on_joints(self, key, message):
        try:
            names = [f'{key}_joint_{i}' for i in range(1, 8)] if key in ('left', 'right') else ['head_pha_joint']
            values = joint_values(message, names)
            self.inputs.push(key, stamp_seconds(message.header.stamp), time.monotonic(), values)
        except (KeyError, ValueError) as exc:
            self.warn(f'{key}: {exc}')

    def on_waist(self, message):
        if math.isfinite(message.data):
            self.inputs.push('waist_pitch', self.ros_now(), time.monotonic(), [math.radians(message.data)])

    def on_image(self, key, message, width, height):
        try:
            image = decode_rgb(message, width, height)
            self.inputs.push(key, stamp_seconds(message.header.stamp), time.monotonic(), image)
            if self.udp_preview is not None:
                self.udp_preview.offer(key, image)
        except ValueError as exc:
            self.warn(f'{key}: {exc}')

    def on_teach(self, key, message):
        with self.inputs.lock:
            self.inputs.teaching[key] = message.data == 'teaching'
        if message.data != 'teaching':
            self.commands.put(('pause', f'{key} teaching status: {message.data}'))

    def on_buttons(self, message):
        received = time.monotonic()
        with self.inputs.lock:
            self.inputs.button_at = received
        if self.buttons.update(message.data):
            self.commands.put(('button', received))

    def on_motor_status(self, key, message):
        phase = message.data.partition(':')[0]
        with self.inputs.lock:
            if phase in ('fault', 'error', 'stopped'):
                self.inputs.motor_faults.add(key)
                self.commands.put(('pause', f'{key} driver: {message.data}'))
            elif phase in ('running', 'holding', 'idle'):
                self.inputs.motor_faults.discard(key)
