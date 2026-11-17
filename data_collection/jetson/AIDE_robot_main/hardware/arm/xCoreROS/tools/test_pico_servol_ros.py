"""Exercise the Grip/power/pose sequence without a ROS node or robot connection."""

from concurrent.futures import Future
from types import SimpleNamespace
import math
import struct
import threading
import unittest
from unittest.mock import Mock

import pico_servol_ros as pico
from rclpy.serialization import serialize_message


class FakeClient:
    def __init__(self):
        self.ready = True
        self.requests = []
        self.futures = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        future = Future()
        self.requests.append(request)
        self.futures.append(future)
        return future


class GripPowerTests(unittest.TestCase):
    def make_arm(self, side='left'):
        node = Mock()
        node.create_client.side_effect = lambda *_: FakeClient()
        node.create_publisher.side_effect = lambda *_: Mock()
        node.receiver.latest.return_value = (None, None)
        arm = pico.ArmBridge(node, side, None)
        arm.publisher.get_subscription_count.return_value = 1
        return arm, node

    def hand(self, grip=True, received=0.0, down=False):
        return pico.Hand((0, 0, 0), (1, 0, 0, 0), 0.0, grip, received, down)

    def begin(self, arm):
        arm.tick(self.hand(False), 0.0)
        arm.tick(self.hand(), 0.0)
        self.assertEqual(arm.phase, 'powering')

    def power_ready(self, arm):
        arm.power_client.futures[-1].set_result(
            SimpleNamespace(success=True, message='power state: on'))
        arm.tick(self.hand(), 0.0)
        self.assertEqual(arm.phase, 'requesting')

    def pose_ready(self, arm):
        arm.client.futures[-1].set_result(SimpleNamespace(
            success=True, message='current Cartesian state',
            pose=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0)))
        arm.tick(self.hand(), 0.0)

    def test_each_arm_waits_for_power_then_pose_before_publishing(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                arm, node = self.make_arm(side)
                self.assertEqual(node.create_client.call_args_list[1].args[1],
                                 f'{pico.PREFIX}/power_on/{side}_arm')
                arm.tick(self.hand(), 0.0)
                self.assertEqual(len(arm.power_client.requests), 0)
                self.begin(arm)
                arm.tick(self.hand(), 0.01)
                self.assertEqual(len(arm.power_client.requests), 1)
                self.assertEqual(len(arm.client.requests), 0)
                arm.publisher.publish.assert_not_called()
                self.power_ready(arm)
                arm.publisher.publish.assert_not_called()
                self.pose_ready(arm)
                self.assertEqual(arm.phase, 'active')
                message = arm.publisher.publish.call_args.args[0]
                self.assertEqual(message.pose.position.x, 0.1)
                self.assertEqual(message.pose.position.y, 0.2)
                self.assertEqual(message.pose.position.z, 0.3)
                self.assertEqual(message.pose.orientation.w, 1.0)

    def test_two_arms_hold_start_pose_and_ignore_joystick_until_driver_ready(self):
        arms = [self.make_arm(side)[0] for side in ('left', 'right')]
        for arm in arms:
            self.begin(arm)
            self.power_ready(arm)
            self.pose_ready(arm)
        moving_hand = pico.Hand((0.05, 0.02, 0), (1, 0, 0, 0), 0.0, True, 0.0, False, 1.0)
        for _ in range(20):
            for arm in arms:
                arm.tick(moving_hand, 0.0)
                message = arm.publisher.publish.call_args.args[0]
                self.assertEqual(message.pose.position.x, 0.1)
                self.assertEqual(message.pose.position.y, 0.2)
                self.assertEqual(message.pose.position.z, 0.3)
                self.assertEqual(message.joystick_y, 0.0)

    def test_starting_status_does_not_accept_an_old_anchor(self):
        arm, _ = self.make_arm()
        self.begin(arm)
        self.power_ready(arm)
        self.pose_ready(arm)
        message = pico.PoseStamped()
        message.pose.position.x = 0.5
        message.pose.orientation.w = 1.0
        arm.on_status('starting')
        arm.on_command_pose(message)
        self.assertTrue(arm.awaiting_driver_anchor)
        self.assertEqual(arm.robot_anchor[0], (0.1, 0.2, 0.3))

    def test_one_arm_can_align_and_move_while_the_other_is_starting(self):
        left, node = self.make_arm('left')
        right, _ = self.make_arm('right')
        for arm in (left, right):
            self.begin(arm)
            self.power_ready(arm)
            self.pose_ready(arm)
        hand = self.hand()
        node.receiver.latest.return_value = (hand, hand)
        message = pico.PoseStamped()
        message.pose.position.x = 0.1
        message.pose.position.y = 0.2
        message.pose.position.z = 0.3
        message.pose.orientation.w = 1.0
        left.on_status('ready')
        left.on_command_pose(message)
        self.assertFalse(left.awaiting_driver_anchor)
        self.assertTrue(right.awaiting_driver_anchor)
        moving_hand = pico.Hand((0.02, 0, 0), (1, 0, 0, 0), 0.0, True, 0.0, False, 0.5)
        left.tick(moving_hand, 0.0)
        right.tick(moving_hand, 0.0)
        left_message = left.publisher.publish.call_args.args[0]
        right_message = right.publisher.publish.call_args.args[0]
        self.assertAlmostEqual(left_message.pose.position.x, 0.1 + pico.POSITION_STEP)
        self.assertEqual(left_message.joystick_y, 0.5)
        self.assertEqual(right_message.pose.position.x, 0.1)
        self.assertEqual(right_message.joystick_y, 0.0)

    def test_fault_in_one_arm_does_not_stop_or_repower_the_other(self):
        left, _ = self.make_arm('left')
        right, _ = self.make_arm('right')
        for arm in (left, right):
            self.begin(arm)
            self.power_ready(arm)
            self.pose_ready(arm)
        left.on_status('error:left SDK loop: network error')
        for _ in range(5):
            left.tick(self.hand(), 0.0)
            right.tick(self.hand(), 0.0)
        self.assertEqual(left.phase, 'idle')
        self.assertEqual(right.phase, 'active')
        self.assertEqual(len(left.power_client.requests), 1)
        self.assertEqual(len(right.power_client.requests), 1)
        right.stop_publisher.publish.assert_not_called()

    def test_power_failure_does_not_retry_until_release(self):
        for message in ('power-on rejected: emergency stop',
                        'power-on rejected: safety gate stop'):
            with self.subTest(message=message):
                arm, node = self.make_arm()
                self.begin(arm)
                arm.power_client.futures[-1].set_result(
                    SimpleNamespace(success=False, message=message))
                arm.tick(self.hand(), 0.0)
                for _ in range(5):
                    arm.tick(self.hand(), 0.0)
                self.assertEqual(arm.phase, 'idle')
                self.assertFalse(arm.armed)
                self.assertEqual(len(arm.power_client.requests), 1)
                self.assertEqual(len(arm.client.requests), 0)
                arm.publisher.publish.assert_not_called()
                self.assertIn(message, node.get_logger().error.call_args.args[0])
                self.begin(arm)
                self.assertEqual(len(arm.power_client.requests), 2)

    def test_release_or_udp_timeout_cancels_startup_without_motion(self):
        for stage in ('powering', 'requesting'):
            for release in (True, False):
                with self.subTest(stage=stage, release=release):
                    arm, _ = self.make_arm()
                    self.begin(arm)
                    if stage == 'requesting':
                        self.power_ready(arm)
                    pending = arm.future
                    arm.tick(self.hand(not release),
                             0.0 if release else pico.UDP_TIMEOUT + 0.01)
                    self.assertEqual(arm.phase, 'idle')
                    self.assertTrue(pending.cancelled())
                    self.assertIsNone(arm.future)
                    arm.publisher.publish.assert_not_called()

    def test_missing_power_service_does_not_start(self):
        arm, _ = self.make_arm()
        arm.power_client.ready = False
        arm.tick(self.hand(False), 0.0)
        arm.tick(self.hand(), 0.0)
        self.assertEqual(arm.phase, 'idle')
        self.assertEqual(len(arm.power_client.requests), 0)
        self.assertEqual(len(arm.client.requests), 0)

    def test_driver_fault_requires_new_grip_before_another_power_request(self):
        arm, _ = self.make_arm()
        self.begin(arm)
        self.power_ready(arm)
        self.pose_ready(arm)
        arm.on_status('error:left joint J6 reached the configured safety margin')
        for _ in range(5):
            arm.tick(self.hand(), 0.0)
        self.assertEqual(len(arm.power_client.requests), 1)
        self.assertEqual(arm.phase, 'idle')
        self.begin(arm)
        self.assertEqual(len(arm.power_client.requests), 2)

    def test_release_while_active_still_publishes_stop(self):
        arm, _ = self.make_arm()
        self.begin(arm)
        self.power_ready(arm)
        self.pose_ready(arm)
        arm.tick(self.hand(False), 0.0)
        self.assertEqual(arm.phase, 'stopping')
        self.assertTrue(arm.stop_publisher.publish.call_args.args[0].data)

    def test_pose_failure_does_not_publish_or_retry_while_held(self):
        arm, _ = self.make_arm()
        self.begin(arm)
        self.power_ready(arm)
        arm.client.futures[-1].set_result(SimpleNamespace(
            success=False, message='pose read failed', pose=(0.0,) * 6))
        arm.tick(self.hand(), 0.0)
        arm.tick(self.hand(), 0.0)
        self.assertEqual(arm.phase, 'idle')
        self.assertEqual(len(arm.power_client.requests), 1)
        arm.publisher.publish.assert_not_called()

    def test_udp_down_bit_is_decoded_independently_for_both_sides(self):
        packets = [bytes([side]) + struct.pack('<9f', 0., 0., 0., 0., 0., 0., 1., 0., 0.)
                   + bytes([(1 << 2) | (1 << 3)]) for side in (0, 1)]
        receiver = pico.PicoReceiver.__new__(pico.PicoReceiver)
        receiver.socket = Mock()
        receiver.socket.recvfrom.side_effect = [(p, None) for p in packets] + [OSError()]
        receiver.lock = threading.Lock()
        receiver.hands = [None, None]
        receiver.raw = [None, None]
        receiver.running = True
        receiver.receive()
        for hand in receiver.latest():
            self.assertTrue(hand.down)
            self.assertFalse(hand.grip)
            self.assertEqual(hand.buttons, 12)  # Preserve B for the read-only recorder.

    def test_udp_joystick_boundaries_remain_float_and_serialize(self):
        for side in (0, 1):
            for value in (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0):
                with self.subTest(side=side, value=value):
                    packet = bytes([side]) + struct.pack(
                        '<9f', 0., 0., 0., 0., 0., 0., 1., 0., value) + bytes([0])
                    receiver = pico.PicoReceiver.__new__(pico.PicoReceiver)
                    receiver.socket = Mock()
                    receiver.socket.recvfrom.side_effect = [(packet, None), OSError()]
                    receiver.lock = threading.Lock()
                    receiver.hands = [None, None]
                    receiver.raw = [None, None]
                    receiver.running = True
                    receiver.receive()
                    hand = receiver.latest()[side]
                    self.assertIsInstance(hand.joystick_y, float)
                    self.assertEqual(hand.joystick_y, max(-1.0, min(value, 1.0)))
                    arm, _ = self.make_arm('left' if side == 0 else 'right')
                    arm.command_position = (0.1, 0.2, 0.3)
                    arm.command_orientation = (1.0, 0.0, 0.0, 0.0)
                    arm.publish_target(hand.joystick_y)
                    message = arm.publisher.publish.call_args.args[0]
                    self.assertTrue(serialize_message(message))

    def test_publish_target_converts_integer_joystick_values_to_float(self):
        arm, _ = self.make_arm()
        arm.command_position = (0.1, 0.2, 0.3)
        arm.command_orientation = (1.0, 0.0, 0.0, 0.0)
        for value in (-1, 0, 1, -1.0, 0.0, 0.5, 1.0):
            with self.subTest(value=value):
                arm.publish_target(value)
                message = arm.publisher.publish.call_args.args[0]
                self.assertIsInstance(message.joystick_y, float)
                self.assertEqual(message.joystick_y, float(value))
                self.assertTrue(serialize_message(message))

    def test_elbow_joystick_accepts_left_and_right_thirty_degree_cones(self):
        for direction in (-1., 1.):
            for radius in (0.3, 0.7, 1.0):
                for degrees in (-30., -15., 0., 15., 30.):
                    with self.subTest(direction=direction, radius=radius, degrees=degrees):
                        y = radius * math.sin(math.radians(degrees))
                        x = direction * radius * math.cos(math.radians(degrees))
                        self.assertEqual(pico.elbow_joystick_x(x, y), x)

    def test_elbow_joystick_rejects_directions_outside_each_cone(self):
        for direction in (-1., 1.):
            for radius in (0.3, 0.7, 1.0):
                for degrees in (-90., -45., -30.001, 30.001, 45., 90.):
                    with self.subTest(direction=direction, radius=radius, degrees=degrees):
                        y = radius * math.sin(math.radians(degrees))
                        x = direction * radius * math.cos(math.radians(degrees))
                        self.assertEqual(pico.elbow_joystick_x(x, y), 0.0)
        self.assertEqual(pico.elbow_joystick_x(0., 0.), 0.0)

    def test_direction_gate_applies_to_both_arms_without_stopping_pose_tracking(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                arm, _ = self.make_arm(side)
                self.begin(arm)
                self.power_ready(arm)
                self.pose_ready(arm)
                arm.awaiting_driver_anchor = False
                for x, y, expected in ((.8, 0., .8), (.8, .8, 0.),
                                       (-.8, 0., -.8), (-.8, -.8, 0.), (0., .8, 0.)):
                    hand = pico.Hand((0.02, 0., 0.), (1., 0., 0., 0.),
                                     y, True, 0., False, x)
                    arm.tick(hand, 0.)
                    message = arm.publisher.publish.call_args.args[0]
                    self.assertEqual(message.joystick_y, expected)
                    self.assertGreater(message.pose.position.x, 0.1)
                    self.assertEqual(arm.phase, 'active')
                    arm.stop_publisher.publish.assert_not_called()

    def test_udp_packet_preserves_joystick_x_for_both_sides(self):
        for side in (0, 1):
            packet = bytes([side]) + struct.pack(
                '<9f', 0., 0., 0., 0., 0., 0., 1., 0.5, -0.75) + bytes([0])
            receiver = pico.PicoReceiver.__new__(pico.PicoReceiver)
            receiver.socket = Mock()
            receiver.socket.recvfrom.side_effect = [(packet, None), OSError()]
            receiver.lock = threading.Lock()
            receiver.hands = [None, None]
            receiver.raw = [None, None]
            receiver.running = True
            receiver.receive()
            hand = receiver.latest()[side]
            self.assertEqual(hand.joystick_x, 0.5)
            self.assertEqual(hand.joystick_y, -0.75)


if __name__ == '__main__':
    unittest.main()
