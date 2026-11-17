"""Head/waist-only routing and deadman tests without a ROS graph or hardware."""
from concurrent.futures import Future
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from std_msgs.msg import Float64, String

import pico_servol_ros as pico


class HeadWaistOnlyTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(pico.Node, '__init__', return_value=None))
        self.arm_constructor = self.patches.enter_context(patch.object(pico, 'ArmBridge'))
        self.publishers = {}
        self.clients = {}

        def publisher(_type, topic, _qos):
            result = Mock()
            result.get_subscription_count.return_value = 1
            self.publishers[topic] = result
            return result

        def client(_type, service):
            result = Mock()
            result.service_is_ready.return_value = True
            result.futures = []

            def request(_request):
                future = Future()
                result.futures.append(future)
                return future

            result.call_async.side_effect = request
            self.clients[service] = result
            return result

        self.patches.enter_context(patch.object(pico.Node, 'create_publisher', side_effect=publisher))
        self.patches.enter_context(patch.object(pico.Node, 'create_client', side_effect=client))
        self.subscriptions = self.patches.enter_context(patch.object(pico.Node, 'create_subscription'))
        self.patches.enter_context(patch.object(pico.Node, 'create_timer'))
        self.patches.enter_context(patch.object(pico.Node, 'get_logger'))
        self.receiver = Mock()
        self.node = pico.PicoServoLBridge(self.receiver, arms_enabled=False)
        self.axes = dict(self.node.auxiliary_axes)

    def hand(self, now, grip, down=False):
        return pico.Hand((0., 0., 0.), (1., 0., 0., 0.), .8, grip, now, down, 0.)

    def tick(self, now, left_grip=True, right_grip=True, down=False):
        self.receiver.latest.return_value = (
            self.hand(now, left_grip, down), self.hand(now, right_grip, down))
        with patch.object(pico.time, 'monotonic', return_value=now):
            self.node.tick()

    def feedback(self, now, running=False):
        with patch.object(pico.time, 'monotonic', return_value=now):
            for axis in self.axes.values():
                axis.on_position(Float64(data=20.))
            if running:
                self.axes[1].on_status(String(data='running: ready'))

    def test_no_arm_interfaces_are_created(self):
        self.arm_constructor.assert_not_called()
        self.assertEqual(self.node.arms, [])
        self.assertIsNone(self.node.status_sub)
        self.assertTrue(self.axes[0].preserve_direction)
        self.assertFalse(self.axes[1].preserve_direction)
        self.assertEqual(set(self.publishers), {
            '/headmotor/csp/pha/target_deg', '/robot/waist_pitch/csp/target_deg'})
        self.assertEqual(set(self.clients), {
            '/headmotor/csp/start', '/headmotor/csp/stop',
            '/robot/waist_pitch/csp/resume', '/robot/waist_pitch/csp/hold'})
        for call in self.subscriptions.call_args_list:
            self.assertFalse(call.args[1].startswith(pico.PREFIX))
        self.node.on_status(String(data='left:running'))  # Ignore even unsolicited arm status.

    def test_waist_uses_latest_best_effort_feedback_and_head_keeps_existing_qos(self):
        calls = {call.args[1]: call for call in self.subscriptions.call_args_list}
        waist_qos = calls['/robot/waist_pitch/csp/position_deg'].args[3]
        self.assertEqual(waist_qos.depth, 1)
        self.assertEqual(waist_qos.reliability, pico.ReliabilityPolicy.BEST_EFFORT)
        self.assertEqual(waist_qos.durability, pico.DurabilityPolicy.VOLATILE)
        self.assertEqual(calls['/headmotor/csp/pha/position_deg'].args[3], 1)

    def test_both_motors_move_without_arm_services_and_grips_stop_independently(self):
        self.feedback(0.)
        self.tick(0., False, False)
        self.tick(.01)
        for axis in self.axes.values():
            self.assertEqual(axis.phase, 'starting')
            axis.start_client.futures[-1].set_result(SimpleNamespace(success=True))
            axis.publisher.publish.assert_not_called()
        self.feedback(.02, running=True)
        self.tick(.02)
        for axis in self.axes.values():
            self.assertEqual(axis.phase, 'running')
            self.assertEqual(axis.target, 20.)
        self.tick(.03)
        self.assertAlmostEqual(self.axes[1].target, 20.15)
        self.assertAlmostEqual(self.axes[0].target, 19.9)
        head_messages = self.axes[1].publisher.publish.call_count
        self.tick(.04, right_grip=False)
        self.assertEqual(self.axes[1].phase, 'stopping')
        self.assertEqual(self.axes[1].publisher.publish.call_count, head_messages)
        self.assertEqual(self.axes[0].phase, 'running')
        self.assertAlmostEqual(self.axes[0].target, 19.8)
        self.tick(.05, False, False)
        self.assertEqual(self.axes[0].phase, 'stopping')
        self.assertEqual(len(self.clients), 4)
        self.arm_constructor.assert_not_called()

    def test_A_never_requests_arm_teaching_and_shutdown_handles_empty_arms(self):
        self.feedback(0.)
        self.tick(0., False, False)
        self.tick(.01, down=True)
        self.assertEqual(len(self.clients['/headmotor/csp/start'].futures), 1)
        self.assertEqual(len(self.clients['/robot/waist_pitch/csp/resume'].futures), 1)
        self.assertFalse(any('upperlimb' in name for name in self.clients))
        for axis in self.axes.values():
            axis.start_client.futures[-1].set_result(SimpleNamespace(success=False, message='cancelled'))
        self.tick(.02, down=True)
        self.node.stop_all()
        self.node.timer.cancel.assert_called_once()
        self.assertTrue(self.node.shutting_down)
        for axis in self.axes.values():
            self.assertTrue(axis.closing)
            self.assertFalse(axis.pending)
        self.arm_constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
