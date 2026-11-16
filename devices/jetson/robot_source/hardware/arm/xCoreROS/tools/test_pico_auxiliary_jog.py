"""Offline deadman, calibration and target-stream tests; no CAN or ROS node."""
from concurrent.futures import Future
import math
from types import SimpleNamespace
from pathlib import Path
import tempfile
import yaml
import unittest
from unittest.mock import Mock, patch

from std_msgs.msg import Float64, String
from rclpy.serialization import serialize_message

import pico_auxiliary_jog as jog
import pico_servol_ros as pico


class Client:
    def __init__(self):
        self.ready = True
        self.futures = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, _request):
        future = Future()
        self.futures.append(future)
        return future


class AuxiliaryTests(unittest.TestCase):
    def make_axis(self, head=False):
        node = Mock()
        node.create_client.side_effect = lambda *_: Client()
        node.create_publisher.return_value.get_subscription_count.return_value = 1
        axis = jog.PositionJogBridge(
            node, 'head' if head else 'waist', '/test', '/start', '/hold',
            status_topic='/status', async_start=head,
            speed_deg_s=5. if head else 10., direction=1. if head else -1.,
            lower=-42. if head else 0., upper=42. if head else 50., tolerance=.1)
        self.feedback(axis, 20., 0.)
        axis.tick(self.hand(False), 0., 0., True)
        return axis

    def hand(self, grip=True, now=0., down=False):
        return SimpleNamespace(grip=grip, received=now, down=down)

    def feedback(self, axis, angle, now):
        with patch.object(jog.time, 'monotonic', return_value=now):
            axis.on_position(Float64(data=float(angle)))

    def status(self, axis, phase, now):
        with patch.object(jog.time, 'monotonic', return_value=now):
            axis.on_status(String(data=phase + ': test'))

    def ready(self, axis):
        axis.tick(self.hand(), 1., 0., True)
        axis.start_client.futures[-1].set_result(SimpleNamespace(success=True, message='ready'))
        if axis.async_start:
            self.status(axis, 'running', .01)
        self.feedback(axis, 20., .01)
        axis.tick(self.hand(now=.01), 1., .01, True)
        self.assertEqual(axis.phase, 'running')

    def test_calibration_uses_existing_pha_limits_and_speed(self):
        settings = jog.load_head_pitch_settings()
        self.assertAlmostEqual(settings['upper'], 390897 * 360. / 3342336.)
        self.assertAlmostEqual(settings['lower'], -settings['upper'])
        self.assertEqual(settings['speed_deg_s'], 15.)
        self.assertEqual(settings['lookahead_s'], .15)

    def test_legacy_head_config_keeps_original_speed_and_lead(self):
        config = (Path(__file__).resolve().parents[3] /
                  'headmotor/yiyou/yiyou_niic_ros/config/headmotor.yaml')
        document = yaml.safe_load(config.read_text())
        saved = document['headmotor']['ros__parameters']
        for key in ('pha_csp_velocity_deg_s', 'pha_csp_acceleration_deg_s2',
                    'pha_csp_deceleration_deg_s2', 'pha_pico_lookahead_s'):
            saved.pop(key)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy_headmotor.yaml'
            path.write_text(yaml.safe_dump(document))
            settings = jog.load_head_pitch_settings(path)
        self.assertEqual(settings['speed_deg_s'], 5.)
        self.assertEqual(settings['lookahead_s'], .1)

    def test_head_profile_applies_15_deg_speed_and_225_deg_target_cap(self):
        settings = jog.load_head_pitch_settings()
        axis = self.make_axis(True)
        axis.speed = settings['speed_deg_s']
        axis.lookahead_s = settings['lookahead_s']
        self.ready(axis)
        axis.tick(self.hand(now=.02), 1., .02, True)
        self.assertAlmostEqual(axis.target, 20.15)
        for step in range(3, 100):
            now = step * .01
            self.feedback(axis, 20., now)
            self.status(axis, 'running', now)
            axis.tick(self.hand(now=now), 1., now, True)
        self.assertAlmostEqual(axis.target, 22.25)

    def test_start_requires_release_grip_fresh_feedback_and_arm_ready(self):
        axis = self.make_axis()
        axis.armed = False
        axis.tick(self.hand(), 1., .01, True)
        self.assertEqual(axis.phase, 'idle')
        axis.tick(self.hand(False), 1., .02, True)
        axis.tick(self.hand(now=.03), 1., .03, False)
        self.assertEqual(axis.phase, 'idle')
        axis.tick(self.hand(now=1.), 1., 1., True)  # stale motor feedback
        self.assertEqual(axis.phase, 'idle')
        self.feedback(axis, 20., 1.)
        axis.tick(self.hand(now=1.), 1., 1., True)
        self.assertEqual(axis.phase, 'starting')
        axis.publisher.publish.assert_not_called()

    def test_missing_services_or_target_subscriber_prevent_start(self):
        for missing in ('start', 'stop', 'subscriber'):
            with self.subTest(missing=missing):
                axis = self.make_axis()
                if missing == 'subscriber':
                    axis.publisher.get_subscription_count.return_value = 0
                else:
                    getattr(axis, missing + '_client').ready = False
                axis.tick(self.hand(), 1., 0., True)
                self.assertEqual(axis.phase, 'idle')

    def test_head_rpc_success_waits_for_running_status(self):
        axis = self.make_axis(True)
        axis.tick(self.hand(), 1., 0., True)
        axis.start_client.futures[-1].set_result(SimpleNamespace(success=True))
        for phase in ('configuring', 'enabling'):
            self.status(axis, phase, .01)
            axis.tick(self.hand(now=.01), 1., .01, True)
            self.assertEqual(axis.phase, 'starting')
            axis.publisher.publish.assert_not_called()
        self.status(axis, 'running', .02)
        self.feedback(axis, 19.5, .02)
        axis.tick(self.hand(now=.02), 1., .02, True)
        self.assertEqual(axis.publisher.publish.call_args.args[0].data, 19.5)

    def test_release_during_head_enable_waits_then_holds_without_targets(self):
        axis = self.make_axis(True)
        axis.tick(self.hand(), 1., 0., True)
        axis.start_client.futures[-1].set_result(SimpleNamespace(success=True))
        self.status(axis, 'enabling', .01)
        axis.tick(self.hand(False, .01), 0., .01, True)
        self.assertEqual(len(axis.stop_client.futures), 0)
        self.status(axis, 'running', .02)
        axis.tick(self.hand(False, .02), 0., .02, True)
        self.assertEqual(axis.phase, 'stopping')
        self.assertEqual(len(axis.stop_client.futures), 1)
        axis.publisher.publish.assert_not_called()

    def test_target_starts_at_feedback_then_fixed_speed_with_opposite_pitch_signs(self):
        for head, expected in ((True, 20.05), (False, 19.9)):
            for value in (.16, .5, 1.):
                with self.subTest(head=head, value=value):
                    axis = self.make_axis(head)
                    self.ready(axis)
                    self.assertEqual(axis.publisher.publish.call_args.args[0].data, 20.)
                    axis.tick(self.hand(now=.02), value, .02, True)
                    message = axis.publisher.publish.call_args.args[0]
                    self.assertAlmostEqual(message.data, expected)
                    self.assertTrue(serialize_message(message))

    def test_backward_reverses_each_pitch_motor(self):
        for head, expected in ((True, 19.95), (False, 20.1)):
            axis = self.make_axis(head)
            self.ready(axis)
            axis.tick(self.hand(now=.02), -1., .02, True)
            self.assertAlmostEqual(axis.target, expected)

    def test_A_keeps_running_head_and_waist_jog_active(self):
        for head in (False, True):
            with self.subTest(head=head):
                axis = self.make_axis(head)
                self.ready(axis)
                before = axis.publisher.publish.call_count
                axis.tick(self.hand(now=.02, down=True), 1., .02, True)
                self.assertEqual(axis.phase, 'running')
                self.assertEqual(axis.publisher.publish.call_count, before + 1)
                self.assertEqual(len(axis.stop_client.futures), 0)

    def test_targets_cannot_run_ahead_or_escape_physical_bounds(self):
        for head in (False, True):
            axis = self.make_axis(head)
            self.ready(axis)
            for step in range(2, 200):
                now = step * .01
                self.feedback(axis, 20., now)
                if head:
                    self.status(axis, 'running', now)
                axis.tick(self.hand(now=now), 1., now, True)
            self.assertAlmostEqual(axis.target, 20. + (0.5 if head else -1.))
            boundary = axis.upper if head else axis.lower
            self.feedback(axis, boundary, 2.)
            axis.target = boundary
            if head:
                self.status(axis, 'running', 2.)
            axis.tick(self.hand(now=2.), 1., 2., True)
            self.assertEqual(axis.target, boundary)

    def test_waist_feedback_recoil_never_reverses_a_held_jog(self):
        for value, target, recoil in ((1., 19., 20.006), (-1., 21., 19.994)):
            with self.subTest(value=value):
                axis = self.make_axis()
                axis.preserve_direction = True
                self.ready(axis)
                axis.target = target
                self.feedback(axis, recoil, .02)
                axis.tick(self.hand(now=.02), value, .02, True)
                self.assertAlmostEqual(axis.target, target)
                self.assertEqual(axis.phase, 'running')
                self.assertEqual(len(axis.stop_client.futures), 0)

    def test_waist_feedback_jumps_cannot_create_a_step_faster_than_jog_speed(self):
        for value, initial, feedback in ((1., 19., 17.), (-1., 21., 23.)):
            with self.subTest(value=value):
                axis = self.make_axis()
                axis.preserve_direction = True
                self.ready(axis)
                axis.target = initial
                self.feedback(axis, feedback, .02)
                axis.tick(self.hand(now=.02), value, .02, True)
                expected = initial - math.copysign(.1, value)
                self.assertAlmostEqual(axis.target, expected)

    def test_waist_accepts_an_explicit_joystick_reversal(self):
        axis = self.make_axis()
        axis.preserve_direction = True
        self.ready(axis)
        self.feedback(axis, 20., .02)
        axis.tick(self.hand(now=.02), 1., .02, True)
        before = axis.target
        self.feedback(axis, 20., .03)
        axis.tick(self.hand(now=.03), -1., .03, True)
        self.assertAlmostEqual(axis.target, before + .1)

    def test_full_and_no_arms_modes_generate_identical_waist_commands_when_ready(self):
        outputs = []
        for arms_enabled in (False, True):
            axis = self.make_axis()
            axis.preserve_direction = True
            self.ready(axis)
            arms = [SimpleNamespace(phase='active', awaiting_driver_anchor=False, tick=Mock(),
                                       update_a=Mock(return_value=False), suspend_teleop=Mock(), teaching_pending=False,
                                       servol_cleanup_failed=False)
                    for _ in range(2)] if arms_enabled else []
            node = SimpleNamespace(shutting_down=False, teach_mode=False, headwaist_pause=False, arms=arms,
                                   auxiliary_axes=[(0, axis)], receiver=Mock())
            for i in range(2, 40):
                now = i * .01
                hand = self.hand(now=now)
                hand.joystick_x, hand.joystick_y = 0., .8
                node.receiver.latest.return_value = (hand, None)
                self.feedback(axis, 20. + .004 * math.sin(i), now)
                with patch.object(pico.time, 'monotonic', return_value=now):
                    pico.PicoServoLBridge.tick(node)
            outputs.append([call.args[0].data for call in axis.publisher.publish.call_args_list])
        self.assertEqual(outputs[0], outputs[1])

    def test_delay_does_not_catch_up_with_a_large_step(self):
        axis = self.make_axis()
        self.ready(axis)
        self.feedback(axis, 20., .2)
        axis.tick(self.hand(now=.2), 1., .2, True)
        self.assertAlmostEqual(axis.target, 19.9)

    def test_every_deadman_loss_holds_and_stops_publishing(self):
        for cause in ('release', 'neutral', 'deadzone', 'udp', 'feedback', 'arm', 'shutdown'):
            with self.subTest(cause=cause):
                axis = self.make_axis()
                self.ready(axis)
                count = axis.publisher.publish.call_count
                now, hand, value, allowed = .02, self.hand(now=.02), 1., True
                if cause == 'release': hand.grip = False
                if cause == 'neutral': value = 0.
                if cause == 'deadzone': value = .15
                if cause == 'udp': hand.received = -1.
                if cause == 'feedback': now, hand.received = 1., 1.
                if cause == 'arm': allowed = False
                if cause == 'shutdown': axis.shutdown()
                axis.tick(hand, value, now, allowed)
                self.assertEqual(axis.phase, 'stopping')
                self.assertEqual(len(axis.stop_client.futures), 1)
                self.assertEqual(axis.publisher.publish.call_count, count)

    def test_head_stop_waits_for_holding_before_rearm(self):
        axis = self.make_axis(True)
        self.ready(axis)
        axis.tick(self.hand(now=.02), 0., .02, True)
        axis.stop_client.futures[-1].set_result(SimpleNamespace(success=True))
        axis.tick(self.hand(now=.03), 1., .03, True)
        self.assertEqual(axis.phase, 'stopping')
        self.status(axis, 'holding', .04)
        axis.tick(self.hand(now=.04), 1., .04, True)
        self.assertEqual(axis.phase, 'idle')
        self.feedback(axis, 20., .05)
        axis.tick(self.hand(now=.05), 1., .05, True)
        self.assertEqual(axis.phase, 'starting')

    def test_driver_fault_blocks_until_grip_release(self):
        axis = self.make_axis(True)
        axis.tick(self.hand(), 1., 0., True)
        axis.start_client.futures[-1].set_result(SimpleNamespace(success=False, message='drive fault'))
        axis.tick(self.hand(), 1., .01, True)
        self.assertEqual(axis.phase, 'blocked')
        axis.tick(self.hand(), 1., .02, True)
        self.assertEqual(len(axis.start_client.futures), 1)
        axis.tick(self.hand(False, .03), 0., .03, True)
        self.assertEqual(axis.phase, 'idle')

    def test_head_status_timeout_requests_hold(self):
        axis = self.make_axis(True)
        self.ready(axis)
        self.feedback(axis, 20., 1.)
        axis.tick(self.hand(now=1.), 1., 1., True)
        self.assertEqual(axis.phase, 'stopping')
        self.assertFalse(axis.armed)

    def test_out_of_range_feedback_is_rejected_and_A_does_not_block_headwaist(self):
        for angle in (-1., 51.):
            axis = self.make_axis()
            self.feedback(axis, angle, .01)
            axis.tick(self.hand(now=.01), 1., .01, True)
            self.assertEqual(axis.phase, 'idle')
            axis.publisher.publish.assert_not_called()
        axis = self.make_axis()
        axis.tick(self.hand(down=True), 1., .01, True)
        self.assertEqual(axis.phase, 'starting')

    def test_resume_waits_for_post_start_feedback_without_restart_chatter(self):
        axis = self.make_axis()
        axis.tick(self.hand(now=.01), 1., .01, True)
        axis.start_client.futures[-1].set_result(SimpleNamespace(success=True))
        axis.tick(self.hand(now=.02), 1., .02, True)
        self.assertEqual(axis.phase, 'starting')
        self.assertEqual(len(axis.start_client.futures), 1)
        axis.publisher.publish.assert_not_called()
        self.feedback(axis, 19.9, .03)
        axis.tick(self.hand(now=.03), 1., .03, True)
        self.assertEqual(axis.phase, 'running')
        self.assertEqual(axis.target, 19.9)

    def test_old_running_status_cannot_enable_new_head_session(self):
        axis = self.make_axis(True)
        self.status(axis, 'running', -1.)
        axis.tick(self.hand(), 1., 0., True)
        axis.start_client.futures[-1].set_result(SimpleNamespace(success=True))
        axis.tick(self.hand(now=.01), 1., .01, True)
        self.assertEqual(axis.phase, 'starting')
        axis.publisher.publish.assert_not_called()

    def test_shutdown_during_start_holds_when_delayed_driver_becomes_ready(self):
        axis = self.make_axis(True)
        axis.tick(self.hand(), 1., 0., True)
        axis.shutdown()
        self.assertTrue(axis.pending)
        axis.start_client.futures[-1].set_result(SimpleNamespace(success=True))
        self.status(axis, 'running', .02)
        axis.tick(None, 0., .02, False)
        axis.publisher.publish.assert_not_called()
        self.assertEqual(axis.phase, 'stopping')
        axis.stop_client.futures[-1].set_result(SimpleNamespace(success=True))
        self.status(axis, 'holding', .03)
        axis.tick(None, 0., .03, False)
        self.assertFalse(axis.pending)

    def test_waist_new_resume_clears_old_session_stopped_status(self):
        axis = self.make_axis()
        self.status(axis, 'stopped', -1.)
        self.ready(axis)
        axis.tick(self.hand(now=.02), 1., .02, True)
        self.assertEqual(axis.phase, 'running')
        self.assertAlmostEqual(axis.target, 19.9)

    def test_pitch_cones_exclude_horizontal_and_diagonal_directions(self):
        for direction in (-1., 1.):
            for degrees in (-30., 0., 30.):
                x = .8 * math.sin(math.radians(degrees))
                y = direction * .8 * math.cos(math.radians(degrees))
                self.assertEqual(pico.pitch_joystick_y(x, y), y)
                self.assertEqual(pico.elbow_joystick_x(x, y), 0.)
        for x, y in ((.8, 0.), (-.8, 0.), (.8, .8), (-.8, -.8)):
            self.assertEqual(pico.pitch_joystick_y(x, y), 0.)

    def test_bridge_routes_left_to_waist_right_to_head_with_own_arm_readiness(self):
        hands = (self.hand(now=0.), self.hand(now=0.))
        for hand in hands:
            hand.joystick_x, hand.joystick_y = 0., .8
        arms = [SimpleNamespace(phase='active', awaiting_driver_anchor=False, tick=Mock(),
                                       update_a=Mock(return_value=False), suspend_teleop=Mock(), teaching_pending=False,
                                       servol_cleanup_failed=False),
                SimpleNamespace(phase='active', awaiting_driver_anchor=True, tick=Mock(),
                                update_a=Mock(return_value=False), suspend_teleop=Mock(), teaching_pending=False,
                                servol_cleanup_failed=False)]
        waist, head = Mock(), Mock()
        node = SimpleNamespace(shutting_down=False, teach_mode=False, headwaist_pause=False, arms=arms,
                               receiver=SimpleNamespace(latest=lambda: hands),
                               auxiliary_axes=[(1, head), (0, waist)])
        with patch.object(pico.time, 'monotonic', return_value=0.):
            pico.PicoServoLBridge.tick(node)
        waist.tick.assert_called_once_with(hands[0], .8, 0., True)
        head.tick.assert_called_once_with(hands[1], .8, 0., False)


if __name__ == '__main__':
    unittest.main()
