"""B return recording and command lifecycle; no real ROS/SDK/hardware."""
from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/recording"))
import record_xcore_teach_suction as recorder


class Client:
    def __init__(self):
        self.requests, self.futures = [], []
        self.ready = True

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        future = Future()
        self.futures.append(future)
        return future

    def finish(self, success=True):
        self.futures[-1].set_result(SimpleNamespace(success=success, message="test result"))


class ReturnTest(unittest.TestCase):
    def setUp(self):
        self.starts, self.teach_stops, self.moves, self.motion_stops = (
            {side: Client() for side in recorder.SIDES} for _ in range(4))
        self.session = recorder.DualTeaching(self.starts, self.teach_stops, object, report=lambda _: None)
        for side in recorder.SIDES:
            self.session.on_status(side, "idle")
        self.init = Client()
        self.inputs = Mock()
        self.targets = recorder.load_return_pose()
        self.control = recorder.DualArmReturn(
            self.session, self.init, self.moves, self.motion_stops, object, SimpleNamespace,
            self.inputs, self.targets, report=Mock())

    def moving(self):
        self.assertTrue(self.control.start(0.))
        self.control.tick(.01, 100., recording=True)
        self.init.finish()
        self.control.tick(.02, 100., recording=True)
        self.control.tick(.03, 100., recording=True)
        self.assertEqual(self.control.phase, "moving")

    def test_construction_does_not_initialize_or_move(self):
        self.assertEqual(self.init.requests, [])
        self.assertFalse(self.control.active)
        self.assertTrue(all(not client.requests for client in self.moves.values()))

    def test_wait_for_both_teach_start_futures_and_cleanup_before_initialize(self):
        self.session.start(0.)
        for side in recorder.SIDES:
            self.session.on_status(side, "teaching")
        self.session.tick(.1)
        self.control.start(.2)
        self.session.tick(.3)
        self.control.tick(.3)
        self.assertEqual(self.init.requests, [])
        for side in recorder.SIDES:
            self.session.on_status(side, "stopped")
        self.session.tick(.4)
        self.control.tick(.4)
        self.assertEqual(self.init.requests, [])
        for side in recorder.SIDES:
            self.starts[side].finish()
            self.teach_stops[side].finish()
        self.session.tick(.5)
        self.control.tick(.5)
        self.assertEqual(len(self.init.requests), 1)

    def test_saved_targets_and_low_speed_are_sent_once(self):
        self.moving()
        self.assertFalse(self.control.start(.04))
        for side in recorder.SIDES:
            self.assertEqual(len(self.moves[side].requests), 1)
            request = self.moves[side].requests[0]
            self.assertEqual(request.joint_positions, self.targets[side])
            self.assertEqual((request.joint_speed_scale, request.zone_mm), (.05, 0.))
        self.moves["left"].finish()
        self.control.tick(.1)
        self.assertTrue(self.control.active)
        self.moves["right"].finish()
        self.control.tick(.2)
        self.assertFalse(self.control.active)
        self.inputs.set_return_recording.assert_called_once_with(True)

    def test_no_motion_until_recording_feedback_is_ready(self):
        self.control.start(0.)
        self.control.tick(.01)
        self.init.finish()
        self.control.tick(.02)
        self.inputs.snapshot.side_effect = recorder.NotReady("等待回位关节反馈")
        self.control.tick(.03, 100., True)
        self.assertTrue(all(not client.requests for client in self.moves.values()))
        self.inputs.snapshot.side_effect = None
        self.control.tick(.04, 100., True)
        self.assertTrue(all(len(client.requests) == 1 for client in self.moves.values()))

    def test_missing_service_prevents_even_teach_stop(self):
        self.moves["right"].ready = False
        self.assertFalse(self.control.start(0.))
        self.assertFalse(self.control.active)
        self.assertEqual(self.init.requests, [])
        self.inputs.set_return_recording.assert_not_called()

    def test_initialization_failure_never_issues_a_move(self):
        self.control.start(0.)
        self.control.tick(.01)
        self.init.finish(False)
        self.control.tick(.02)
        self.assertEqual(self.control.phase, "stopping")
        self.assertTrue(all(not client.requests for client in self.moves.values()))

    def test_cancel_while_initialization_pending_never_issues_a_late_move(self):
        self.control.start(0.)
        self.control.tick(.01)
        self.control.stop(.02)
        self.control.tick(.03)
        for client in self.motion_stops.values():
            client.finish()
        self.control.tick(.04)
        self.assertTrue(self.control.active)
        self.init.finish()
        self.control.tick(.05)
        self.assertTrue(all(len(client.requests) == 2 for client in self.motion_stops.values()))
        for client in self.motion_stops.values():
            client.finish()
        self.control.tick(.06)
        self.assertFalse(self.control.active)
        self.assertTrue(all(not client.requests for client in self.moves.values()))

    def test_one_arm_failure_stops_both_and_waits_for_original_requests(self):
        self.moving()
        self.moves["left"].finish(False)
        self.control.tick(.1)
        self.control.tick(.11)
        for client in self.motion_stops.values():
            client.finish()
        self.control.tick(.12)
        self.assertTrue(self.control.active)
        self.moves["right"].finish(False)
        self.control.tick(.13)
        for client in self.motion_stops.values():
            client.finish()
        self.control.tick(.14)
        self.assertFalse(self.control.active)

    def test_unknown_stop_blocks_next_return_and_teach(self):
        self.moving()
        self.control.stop(.1)
        self.control.tick(11.)
        self.assertTrue(self.control.blocked)
        self.assertFalse(self.control.start(12.))
        recorder.handle_key("a", self.session, None, Mock(), 100., 12., self.control)
        self.assertTrue(all(not client.requests for client in self.starts.values()))

    def test_keyboard_b_does_not_save_or_stop_r_and_z_stops_return(self):
        capture = Mock(recording=True)
        capture.writer.rows = [1, 2]
        capture.writer.control_events = []
        recorder.handle_key("b", self.session, capture, Mock(), 100., 0., self.control)
        self.assertTrue(capture.recording)
        capture.finish.assert_not_called()
        self.assertEqual(capture.writer.control_events[0]["joint_targets"], self.targets)
        recorder.handle_key("z", self.session, capture, Mock(), 100., .1, self.control)
        self.assertFalse(self.control.active)
        capture.finish.assert_not_called()

    def test_arrow_down_is_not_b_but_b_and_uppercase_b_are_recognized(self):
        decoder = recorder.KeyDecoder()
        self.assertEqual(decoder.feed("\x1b[B", 0.), [])
        self.assertEqual(decoder.feed("b", 1.), ["b"])
        self.assertEqual(decoder.feed("B", 2.), ["b"])

    def test_invalid_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            path.write_text(json.dumps(dict(units="rad", left=[0.] * 7, right=[0.] * 6)))
            with self.assertRaisesRegex(ValueError, "right"):
                recorder.load_return_pose(path)


class ReturnRecordingTest(unittest.TestCase):
    def test_drag_return_and_post_return_holding_saved_in_one_episode(self):
        args = recorder.parse_args(["--fps", "30", "--width", "4", "--height", "3"])
        outputs = recorder.SuctionOutputs(None, report=lambda _: None)
        inputs = recorder.KeyboardInputs(outputs, args)
        targets = recorder.load_return_pose()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = recorder.dataset_schema(args)
            recorder.prepare_root(root, settings)
            capture = recorder.Capture(inputs, lambda: recorder.EpisodeWriter(root, "return test", settings),
                                       args.fps, report=lambda _: None)
            capture.start(100., 10.)
            captured = []
            previous = None
            for step in range(301):
                elapsed = step / 100
                now, ros_now = 10. + elapsed, 100. + elapsed
                if step == 120:
                    inputs.set_return_recording(True)
                    inputs.teaching = dict.fromkeys(recorder.SIDES, False)
                    capture.mark_gap("B 回位切换反馈", now)
                elif step < 120:
                    inputs.teaching = dict.fromkeys(recorder.SIDES, True)
                if step == 100:
                    outputs.set("left", True, at=now - .003)
                if step == 150:
                    outputs.set("right", True, at=now - .003)
                outputs.record_readback((step >= 100, step >= 150), now - .002)
                for side in recorder.SIDES:
                    # Independent measured trajectory; after 2.5 s one arm
                    # has reached its goal, while both streams keep arriving.
                    progress = min(1., max(0., (elapsed - 2.3) / .4))
                    joints = np.asarray(targets[side]) + (.1 * (1. - progress) if step >= 120 else .15)
                    inputs.push(side, ros_now, now, joints)
                if step % 3 == 0:
                    for camera in recorder.CAMERAS:
                        inputs.push(camera, ros_now, now, np.full((3, 4, 3), step % 255, dtype=np.uint8))
                capture.tick(ros_now, now)
                self.assertTrue(capture.recording)
                if len(capture.writer.rows) != previous:
                    capture.writer.jobs.join()
                    previous = len(capture.writer.rows)
                    captured.append((elapsed, previous))
            capture.finish(True, 13.1)
            meta, arrays = recorder.validate_episode(root / "episode_000000", settings)
            self.assertTrue(meta["complete"])
            self.assertGreater(meta["frames"], 20)
            self.assertTrue(any(t < 1.2 and count > 0 for t, count in captured))
            self.assertTrue(any(t > 2.6 and count > 0 for t, count in captured))
            np.testing.assert_allclose(arrays["state"][-1, :14],
                                       targets["left"] + targets["right"], atol=1e-6)
            np.testing.assert_array_equal(arrays["action"][:, :14], arrays["state"][:, :14])
            self.assertGreater(np.ptp(arrays["state"][-15:, :14], axis=0).max(), .01)
            self.assertEqual(len(list(root.glob("episode_*"))), 1)


if __name__ == "__main__":
    unittest.main()
