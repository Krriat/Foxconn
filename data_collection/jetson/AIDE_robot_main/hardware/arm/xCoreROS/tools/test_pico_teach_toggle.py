"""Right-A dual-arm drag lifecycle and head/waist interlock; no hardware."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pico_servol_ros as pico
import test_pico_servol_ros as base


class TeachToggleTests(unittest.TestCase):
    def setUp(self):
        self.helper = base.GripPowerTests()
        self.arms = [self.helper.make_arm(side)[0] for side in ('left', 'right')]
        self.head, self.waist = Mock(), Mock()
        self.node = SimpleNamespace(
            arms=self.arms, receiver=Mock(), auxiliary_axes=[(1, self.head), (0, self.waist)],
            teach_mode=False, headwaist_pause=False, shutting_down=False,
            get_logger=Mock(), timer=Mock())
        self.node.stop_dual_teach = lambda reason: pico.PicoServoLBridge.stop_dual_teach(self.node, reason)
        self.node.toggle_dual_teach = lambda hands, now: pico.PicoServoLBridge.toggle_dual_teach(self.node, hands, now)

    def hand(self, now=0., grip=False, down=False, buttons=0):
        return pico.Hand((.03, .02, .01), (1., 0., 0., 0.), .8, grip, now, down, 0., buttons)

    def tick(self, left=None, right=None, now=0.):
        self.node.receiver.latest.return_value = (left or self.hand(now), right or self.hand(now))
        with patch.object(pico.time, 'monotonic', return_value=now):
            pico.PicoServoLBridge.tick(self.node)

    def press_right_A(self, grip=False):
        self.tick(self.hand(grip=grip), self.hand(grip=grip))
        self.tick(self.hand(grip=grip), self.hand(grip=grip, down=True))

    def start(self):
        self.press_right_A()
        for arm in self.arms:
            self.assertEqual(arm.phase, 'teach_starting')
            arm.on_teach_status(pico.String(data='starting'))
            arm.on_teach_status(pico.String(data='teaching'))
        return self.arms[0]

    def finish(self, arm, success=True, message='drag disabled'):
        arm.teach_future.set_result(SimpleNamespace(success=success, message=message))

    def active_both(self):
        for arm in self.arms:
            self.helper.begin(arm)
            self.helper.power_ready(arm)
            self.helper.pose_ready(arm)

    def test_right_A_starts_both_and_never_creates_moveabsj(self):
        self.press_right_A()
        for arm in self.arms:
            self.assertEqual(arm.phase, 'teach_starting')
            self.assertEqual(len(arm.teach_client.requests), 1)
            names = [call.args[1] for call in arm.node.create_client.call_args_list]
            self.assertIn(f'{pico.PREFIX}/cartesian_drag_safe/start/{arm.name}', names)
            self.assertIn(f'{pico.PREFIX}/cartesian_drag_safe/stop/{arm.name}', names)
            self.assertFalse(any('move_absj' in name for name in names))

    def test_left_A_does_not_start_or_stop_teaching(self):
        self.tick()
        self.tick(self.hand(down=True))
        for arm in self.arms:
            self.assertEqual(len(arm.teach_client.requests), 0)
        self.start()
        self.tick()
        self.tick(self.hand(down=True))
        for arm in self.arms:
            self.assertEqual(arm.phase, 'teaching')
            self.assertEqual(len(arm.teach_stop_client.requests), 0)

    def test_right_B_is_forwarded_without_toggling_teaching(self):
        self.node.record_buttons = Mock()
        self.tick()
        right = self.hand(buttons=8)
        self.tick(right=right)
        self.assertEqual(self.node.record_buttons.publish.call_args.args[0].data, 8)
        for arm in self.arms:
            self.assertEqual(len(arm.teach_client.requests), 0)
            self.assertEqual(len(arm.power_client.requests), 0)

    def test_start_waits_until_both_servol_sessions_finish_cleanup(self):
        self.active_both()
        self.tick(self.hand(grip=True), self.hand(grip=True))
        counts = [arm.publisher.publish.call_count for arm in self.arms]
        self.tick(self.hand(grip=True), self.hand(grip=True, down=True))
        for arm in self.arms:
            self.assertEqual(arm.phase, 'stopping')
            self.assertTrue(arm.stop_publisher.publish.call_args.args[0].data)
        self.arms[0].on_status('stopped')
        self.tick(self.hand(grip=True), self.hand(grip=True))
        self.assertEqual(sum(len(a.teach_client.requests) for a in self.arms), 0)
        self.arms[1].on_status('stopped')
        self.tick(self.hand(grip=True), self.hand(grip=True))
        for arm in self.arms:
            self.assertEqual(arm.phase, 'teach_starting')
            self.assertEqual(len(arm.teach_client.requests), 1)
        self.assertEqual([a.publisher.publish.call_count for a in self.arms], counts)

    def test_head_and_waist_continue_with_both_arm_streams_blocked(self):
        self.start()
        for _ in range(5):
            self.tick(self.hand(grip=True), self.hand(grip=True))
        self.assertTrue(self.node.teach_mode)
        self.head.tick.assert_called_with(self.node.receiver.latest.return_value[1], .8, 0., True)
        self.waist.tick.assert_called_with(self.node.receiver.latest.return_value[0], .8, 0., True)
        for arm in self.arms:
            self.assertEqual(arm.phase, 'teaching')
            arm.publisher.publish.assert_not_called()
            self.assertEqual(len(arm.power_client.requests), 0)

    def test_second_right_A_stops_both_and_waits_for_both_cleanup_results(self):
        self.start()
        self.press_right_A(grip=True)
        for arm in self.arms:
            self.assertEqual(arm.phase, 'teach_stopping')
            self.assertEqual(len(arm.teach_stop_client.requests), 1)
            arm.teach_stop_future.set_result(SimpleNamespace(success=True, message='stop requested'))
        self.tick(self.hand(grip=True), self.hand(grip=True))
        self.assertTrue(self.node.teach_mode)
        self.finish(self.arms[0])
        self.tick(self.hand(grip=True), self.hand(grip=True))
        self.assertTrue(self.node.teach_mode)
        self.finish(self.arms[1])
        for _ in range(3):
            self.tick(self.hand(grip=True), self.hand(grip=True))
        self.assertFalse(self.node.teach_mode)
        self.assertTrue(self.node.headwaist_pause)
        for arm in self.arms:
            self.assertEqual(len(arm.power_client.requests), 0)
            self.assertFalse(arm.armed)
        self.tick()
        self.tick(self.hand(grip=True), self.hand(grip=True))
        self.assertEqual([len(a.power_client.requests) for a in self.arms], [1, 1])

    def test_early_stop_is_retried_after_each_start_callback_is_observed(self):
        self.press_right_A()
        self.press_right_A()
        for arm in self.arms:
            arm.teach_stop_future.set_result(SimpleNamespace(success=False, message='not active yet'))
        self.tick()
        for arm in self.arms:
            self.assertEqual(len(arm.teach_stop_client.requests), 1)
            arm.on_teach_status(pico.String(data='starting'))
        self.tick()
        for arm in self.arms:
            self.assertEqual(len(arm.teach_stop_client.requests), 2)
            self.assertEqual(arm.phase, 'teach_stopping')

    def test_release_grip_keeps_teaching_but_either_udp_timeout_stops_both(self):
        for side in (0, 1):
            with self.subTest(side=side):
                self.setUp()
                self.start()
                self.tick()
                for arm in self.arms:
                    self.assertEqual(arm.phase, 'teaching')
                    self.assertEqual(len(arm.teach_stop_client.requests), 0)
                hands = [self.hand(now=1.), self.hand(now=1.)]
                hands[side] = self.hand(now=0.)
                self.tick(*hands, now=1.)
                for arm in self.arms:
                    self.assertEqual(arm.phase, 'teach_stopping')
                    self.assertEqual(len(arm.teach_stop_client.requests), 1)

    def test_hold_A_does_not_repeat_and_startup_held_A_is_ignored(self):
        for _ in range(5):
            self.tick(self.hand(down=True), self.hand(down=True))
        self.assertEqual(sum(len(a.teach_client.requests) for a in self.arms), 0)
        self.tick()
        for _ in range(5):
            self.tick(right=self.hand(down=True))
        for arm in self.arms:
            self.assertEqual(len(arm.teach_client.requests), 1)
            self.assertEqual(len(arm.teach_stop_client.requests), 0)

    def test_failed_servol_cleanup_cancels_both_queued_starts(self):
        self.active_both()
        self.press_right_A(grip=True)
        self.arms[0].on_status('stopped')
        self.arms[1].on_status('error:cleanup/power verification failed')
        self.tick()
        for arm in self.arms:
            self.assertFalse(arm.teach_requested)
            self.assertEqual(len(arm.teach_client.requests), 0)

    def test_queued_right_A_can_cancel_both_before_servol_cleanup_finishes(self):
        self.active_both()
        self.press_right_A(grip=True)
        self.press_right_A(grip=True)
        for arm in self.arms:
            self.assertFalse(arm.teach_requested)
            arm.on_status('stopped')
        self.tick()
        self.assertEqual(sum(len(a.teach_client.requests) for a in self.arms), 0)

    def test_one_arm_ending_stops_the_other_without_resuming_servol(self):
        self.start()
        self.finish(self.arms[0])
        for _ in range(3):
            self.tick(self.hand(grip=True), self.hand(grip=True))
        self.assertTrue(self.node.teach_mode)
        self.assertEqual(self.arms[1].phase, 'teach_stopping')
        self.assertEqual(len(self.arms[1].teach_stop_client.requests), 1)
        self.assertEqual(sum(len(a.power_client.requests) for a in self.arms), 0)

    def test_cleanup_failure_blocks_both_and_stops_other_drag(self):
        arm = self.start()
        self.finish(arm, False, 'cleanup unverified:disableDrag failed')
        for _ in range(3):
            self.tick(self.hand(grip=True), self.hand(grip=True))
        self.assertEqual(arm.phase, 'teach_failed')
        self.assertTrue(self.node.teach_mode)
        self.assertEqual(self.arms[1].phase, 'teach_stopping')
        self.assertEqual(sum(len(a.power_client.requests) for a in self.arms), 0)

    def test_lost_start_response_stops_both_and_is_not_treated_as_cleanup(self):
        arm = self.start()
        arm.teach_future.set_exception(RuntimeError('response lost'))
        for _ in range(2):
            self.tick(self.hand(grip=True), self.hand(grip=True))
        self.assertEqual(arm.phase, 'teach_failed')
        self.assertEqual(len(arm.teach_stop_client.requests), 1)
        self.assertEqual(len(self.arms[1].teach_stop_client.requests), 1)
        self.assertTrue(self.node.teach_mode)

    def test_shutdown_stops_both(self):
        self.start()
        pico.PicoServoLBridge.stop_all(self.node)
        for arm in self.arms:
            self.assertFalse(arm.teach_requested)
            self.assertEqual(arm.phase, 'teach_stopping')
            self.assertEqual(len(arm.teach_stop_client.requests), 1)
        self.assertTrue(self.head.shutdown.called)
        self.assertTrue(self.waist.shutdown.called)

    def test_missing_either_start_or_stop_service_does_not_start_one_arm(self):
        for side in (0, 1):
            for attribute in ('teach_client', 'teach_stop_client'):
                with self.subTest(side=side, client=attribute):
                    self.setUp()
                    client = getattr(self.arms[side], attribute)
                    client.ready = False
                    self.press_right_A()
                    self.assertEqual(sum(len(a.teach_client.requests) for a in self.arms), 0)
                    client.ready = True
                    self.tick(right=self.hand(down=True))
                    self.assertEqual(sum(len(a.teach_client.requests) for a in self.arms), 0)

    def test_start_requires_both_controller_streams(self):
        self.tick()
        self.tick(self.hand(now=-1.), self.hand(down=True))
        self.assertEqual(sum(len(a.teach_client.requests) for a in self.arms), 0)

    def test_one_start_rejection_requests_stop_for_the_other(self):
        self.press_right_A()
        self.finish(self.arms[0], False, 'arm occupied')
        for _ in range(2):
            self.tick()
        self.assertEqual(self.arms[1].phase, 'teach_stopping')
        self.assertEqual(len(self.arms[1].teach_stop_client.requests), 1)


if __name__ == '__main__':
    unittest.main()
