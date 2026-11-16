"""Keyboard R-to-S recording and gap recovery; temporary data, no hardware."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

RECORDING = Path(__file__).resolve().parents[1] / "scripts/recording"
sys.path.insert(0, str(RECORDING))
import record_xcore_teach_suction as recorder


def bundle(stamp, target=(0., 0.)):
    state = np.zeros(16, dtype=np.float32)
    action = state.copy()
    action[14:] = target
    return recorder.Bundle(stamp, state, action,
                           {key: np.zeros((3, 4, 3), np.uint8) for key in recorder.CAMERAS},
                           np.full(5, stamp), stamp - .01, np.full(2, stamp - .02))


class ContinuousCaptureTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.args = recorder.parse_args(["--fps", "30", "--width", "4", "--height", "3"])
        self.settings = recorder.dataset_schema(self.args)
        recorder.prepare_root(self.root, self.settings)
        self.inputs = Mock()
        self.report = Mock()
        self.capture = recorder.Capture(
            self.inputs, lambda: recorder.EpisodeWriter(self.root, "continuous keyboard test", self.settings),
            30, report=self.report)
        self.addCleanup(self.cleanup_writer)

    def cleanup_writer(self):
        if self.capture.writer is not None and not self.capture.writer.closed:
            self.capture.finish(False, 100.)

    def sample(self, elapsed, target=(0., 0.)):
        self.inputs.snapshot.side_effect = None
        self.inputs.snapshot.return_value = bundle(100. + elapsed, target)
        self.capture.tick(100. + elapsed, 10. + elapsed)

    def start(self):
        self.capture.start(100., 10.)

    def saved(self):
        return recorder.validate_episode(self.root / "episode_000000", self.settings)

    def test_r_stays_active_through_source_jump_until_s(self):
        self.start()
        self.sample(0.)
        self.sample(1 / 30)
        self.sample(3 / 30, target=(1., 0.))
        self.assertTrue(self.capture.recording)
        self.assertEqual(len(self.capture.writer.rows), 3)
        self.sample(4 / 30, target=(1., 1.))
        self.capture.finish(True, 10.2)
        meta, arrays = self.saved()
        self.assertFalse(self.capture.recording)
        self.assertEqual([(s["start"], s["end"]) for s in meta["segments"]], [(0, 2), (2, 4)])
        np.testing.assert_array_equal(arrays["action"][:, 14:], [[0, 0], [0, 0], [1, 0], [1, 1]])
        self.assertIn("left=66.7ms", meta["capture_gaps"][0]["reasons"][0])
        self.assertAlmostEqual(meta["recording_summary"]["elapsed_seconds"], .2)
        self.assertEqual(meta["recording_summary"]["frames"], 4)

    def test_twenty_hz_keeps_66ms_continuous_and_splits_100ms_gap(self):
        args = recorder.parse_args(["--width", "4", "--height", "3"])
        self.assertEqual(args.fps, 20)
        self.assertEqual(args.root, recorder.DEFAULT_ROOT)
        self.assertEqual(args.root.name, "act_teach_keyboard_suction_30hz")
        settings = recorder.dataset_schema(args)
        root = recorder.prepare_root(self.root / "twenty", settings)
        inputs = Mock()
        capture = recorder.Capture(inputs,
            lambda: recorder.EpisodeWriter(root, "20 Hz gap regression", settings),
            args.fps, report=lambda _: None)
        capture.start(100., 10.)
        try:
            for elapsed in (0., 2/30, 5/30):
                inputs.snapshot.return_value = bundle(100.+elapsed)
                capture.tick(100.+elapsed, 10.+elapsed)
            self.assertTrue(capture.recording)
            capture.finish(True, 10.3)
            meta, arrays = recorder.validate_episode(root / "episode_000000", settings)
            self.assertEqual([(s["start"], s["end"]) for s in meta["segments"]],
                             [(0, 2), (2, 3)])
            self.assertIn("left=100.0ms", meta["capture_gaps"][0]["reasons"][0])
            np.testing.assert_allclose(np.diff(arrays["source_timestamp"][:, 0]),
                                       [2/30, 3/30])
        finally:
            if capture.writer is not None and not capture.writer.closed:
                capture.finish(False, 10.4)

    def test_r_arms_even_before_first_valid_sample_then_recovers(self):
        self.inputs.snapshot.side_effect = recorder.NotReady("左右臂都必须处于 teaching 状态")
        self.start()
        self.capture.tick(100., 10.)
        self.assertTrue(self.capture.recording)
        self.assertEqual(len(self.capture.writer.rows), 0)
        self.sample(1.)
        self.sample(1. + 1 / 30)
        self.capture.finish(True, 11.1)
        meta, _ = self.saved()
        self.assertEqual(meta["frames"], 2)
        self.assertEqual(len(meta["capture_gaps"]), 1)
        self.assertEqual(meta["segments"][0]["start"], 0)

    def test_unknown_io_does_not_create_frames_and_automatically_recovers(self):
        self.start()
        self.sample(0.)
        self.inputs.snapshot.side_effect = recorder.NotReady("M050 状态未知")
        for i in range(1, 8):
            self.capture.tick(100. + i / 30, 10. + i / 30)
        self.assertTrue(self.capture.recording)
        self.assertEqual(len(self.capture.writer.rows), 1)
        self.sample(.4, target=(0., 1.))
        self.capture.finish(True, 10.5)
        meta, arrays = self.saved()
        self.assertEqual(meta["frames"], 2)
        self.assertEqual(len(meta["segments"]), 2)
        self.assertEqual(arrays["action"][-1, 15], 1.)

    def test_long_scheduler_delay_does_not_turn_off_recording(self):
        self.start()
        self.sample(0.)
        self.sample(.5)
        self.assertTrue(self.capture.recording)
        self.capture.finish(True, 10.6)
        meta, _ = self.saved()
        self.assertEqual(len(meta["segments"]), 2)
        self.assertIn("图像/关节跳过了采集周期", meta["capture_gaps"][0]["reasons"][0])
        self.assertEqual(meta["capture_diagnostics"]["scheduler_delay_count"], 1)

    def test_waiting_for_new_camera_never_duplicates_previous_frame(self):
        self.start()
        self.sample(0.)
        self.inputs.snapshot.side_effect = recorder.AwaitFrame("等待新的 head 样本")
        self.capture.tick(100. + 1 / 30, 10. + 1 / 30)
        self.assertEqual(len(self.capture.writer.rows), 1)
        self.capture.tick(100.2, 10.2)
        self.assertTrue(self.capture.recording)
        self.assertEqual(len(self.capture.writer.rows), 1)
        self.sample(.4)
        self.capture.finish(True, 10.5)
        meta, _ = self.saved()
        self.assertEqual(meta["frames"], 2)
        self.assertEqual(len(meta["segments"]), 2)

    def test_z_ends_teaching_but_keeps_r_armed_for_next_a(self):
        self.start()
        self.sample(0.)
        session = Mock()
        recorder.handle_key("z", session, self.capture, Mock(), 100.1, 10.1)
        session.stop.assert_called_once()
        self.assertTrue(self.capture.recording)
        self.sample(.5)
        recorder.handle_key("s", session, self.capture, Mock(), 100.6, 10.6)
        self.assertFalse(self.capture.recording)
        meta, _ = self.saved()
        self.assertEqual(len(meta["segments"]), 2)

    def test_s_stops_further_samples_and_repeated_r_does_not_create_new_writer(self):
        self.start()
        writer = self.capture.writer
        self.capture.start(100.01, 10.01)
        self.assertIs(self.capture.writer, writer)
        self.sample(0.)
        recorder.handle_key("s", Mock(), self.capture, Mock(), 100.1, 10.1)
        self.sample(.2)
        self.assertIsNone(self.capture.writer)
        meta, _ = self.saved()
        self.assertEqual(meta["frames"], 1)

    def test_d_discards_current_staging_and_allows_new_recording(self):
        self.start()
        self.sample(0.)
        pending = self.capture.writer.path
        recorder.handle_key("d", Mock(), self.capture, Mock(), 100.1, 10.1)
        self.assertFalse(pending.exists())
        self.assertFalse(self.capture.recording)
        self.capture.start(101., 11.)
        self.assertTrue(self.capture.recording)

    def test_s_with_no_valid_frames_preserves_diagnostics_without_empty_training_episode(self):
        self.inputs.snapshot.side_effect = recorder.NotReady("等待相机")
        self.start()
        self.capture.tick(100., 10.)
        self.capture.finish(True, 11.)
        self.assertFalse(list(self.root.glob("episode_*")))
        self.assertEqual(len(list(self.root.glob(".episode_*.partial"))), 1)
        self.assertIsNone(self.capture.writer)
        self.start()
        self.assertTrue(self.capture.recording)

    def test_transient_write_queue_error_recovers_without_unwritten_row(self):
        self.start()
        self.sample(0.)
        writer = self.capture.writer
        add = writer.add
        writer.add = Mock(side_effect=RuntimeError("图像写盘队列已满"))
        self.sample(1 / 30)
        self.assertTrue(self.capture.recording)
        self.assertEqual(len(writer.rows), 1)
        writer.add = add
        self.sample(.1)
        self.capture.finish(True, 10.2)
        meta, _ = self.saved()
        self.assertEqual(meta["frames"], 2)
        self.assertEqual(len(meta["segments"]), 2)

    def test_backpressure_retries_same_aligned_sample_without_splitting(self):
        self.start()
        self.sample(0.)
        writer = self.capture.writer
        check = writer.check_capacity
        writer.check_capacity = Mock(side_effect=recorder.WriteQueueFull("queue full"))
        self.sample(1 / 30, target=(1., 0.))
        self.assertEqual(len(writer.rows), 1)
        self.assertTrue(self.capture.recording)
        # The old measured frame must retain its original IO action even when
        # the live inputs have changed while the writer was busy.
        writer.check_capacity = check
        self.inputs.snapshot.side_effect = AssertionError("must retry frozen sample")
        self.capture.tick(100.1, 10.1)
        self.assertEqual(len(writer.rows), 2)
        self.capture.finish(True, 10.2)
        meta, arrays = self.saved()
        self.assertEqual(len(meta['segments']), 1)
        self.assertEqual(meta['capture_gaps'], [])
        np.testing.assert_allclose(arrays['ros_timestamp'], [100., 100.+1/30])
        np.testing.assert_array_equal(arrays['action'][:, 14:], [[0, 0], [1, 0]])
        self.assertGreater(meta['capture_diagnostics']['write_backpressure_seconds'], 0.)

    def test_true_source_loss_after_backpressure_still_splits(self):
        self.start()
        self.sample(0.)
        writer = self.capture.writer
        check = writer.check_capacity
        writer.check_capacity = Mock(side_effect=recorder.WriteQueueFull("queue full"))
        self.sample(1/30)
        writer.check_capacity = check
        self.capture.tick(100.1, 10.1)  # Retain the frozen next frame.
        self.sample(.2)               # Real missing exposures are a new segment.
        self.capture.finish(True, 10.3)
        meta, arrays = self.saved()
        self.assertEqual(meta['frames'], 3)
        self.assertEqual(len(meta['segments']), 2)
        self.assertTrue(np.all(np.diff(arrays['source_timestamp'],axis=0)>0))

    def test_s_saves_accepted_frames_and_reports_pending_sample_not_enqueued(self):
        self.start()
        self.sample(0.)
        self.capture.writer.check_capacity = Mock(side_effect=recorder.WriteQueueFull("queue full"))
        self.sample(1/30)
        self.capture.finish(True, 10.1)
        meta, _ = self.saved()
        self.assertEqual(meta['frames'], 1)
        self.assertEqual(meta['capture_diagnostics']['pending_samples_discarded_on_control_change'], 1)
        self.assertIsNone(self.capture.pending_bundle)

    def test_control_mode_change_drops_pending_sample_from_old_mode(self):
        self.start()
        self.sample(0.)
        check = self.capture.writer.check_capacity
        self.capture.writer.check_capacity = Mock(side_effect=recorder.WriteQueueFull("queue full"))
        self.sample(1/30)
        self.capture.mark_gap('switch feedback mode', 10.1)
        self.assertIsNone(self.capture.pending_bundle)
        self.capture.writer.check_capacity = check
        self.sample(.2)
        self.capture.finish(True, 10.3)
        meta, arrays = self.saved()
        self.assertEqual(meta['frames'], 2)
        self.assertEqual(len(meta['segments']), 2)
        np.testing.assert_allclose(arrays['ros_timestamp'], [100.,100.2])

    def test_clock_reset_starts_new_segment_and_clears_source_high_watermarks(self):
        self.start()
        self.sample(0.)
        self.inputs.snapshot.return_value = bundle(50.)
        self.capture.tick(50., 10.1)
        self.assertTrue(self.capture.recording)
        self.assertEqual(self.inputs.snapshot.call_args.args[2:], (None, None))
        self.capture.finish(True, 10.2)
        meta, arrays = self.saved()
        self.assertEqual(len(meta["segments"]), 2)
        self.assertEqual(arrays["ros_timestamp"].tolist(), [100., 50.])

    def test_progress_reports_armed_waiting_and_frame_count(self):
        self.inputs.snapshot.side_effect = recorder.NotReady("等待相机")
        self.start()
        self.capture.tick(100., 10.)
        self.capture.tick(105.1, 15.1)
        messages = [call.args[0] for call in self.report.call_args_list]
        self.assertTrue(any("[录制中] 0 帧" in message and "等待数据" in message for message in messages))
        self.assertTrue(self.capture.recording)

    def test_thirty_seconds_with_multiple_gaps_remain_one_r_to_s_recording(self):
        self.start()
        for i in range(900):
            if i in (30, 200, 500):
                self.inputs.snapshot.side_effect = recorder.NotReady("短暂缺图")
                self.capture.tick(100. + i / 30, 10. + i / 30)
            else:
                self.sample(i / 30)
            # Logical time advances 30 seconds in this fast unit test. Let the
            # real PNG workers finish so we do not simulate 900 frames in one
            # wall-clock burst (write congestion is covered separately).
            self.capture.writer.jobs.join()
            self.assertTrue(self.capture.recording)
        self.capture.finish(True, 40.)
        meta, _ = self.saved()
        self.assertEqual(meta["frames"], 897)
        self.assertEqual(len(meta["segments"]), 4)
        self.assertAlmostEqual(meta["recording_summary"]["elapsed_seconds"], 30.)

    def test_export_splits_gaps_into_separate_training_episodes(self):
        self.start()
        self.sample(0.)
        self.sample(1 / 30)
        self.sample(.2)
        self.sample(.2 + 1 / 30)
        self.capture.finish(True, 10.3)
        dataset = Mock(image_writer=None)
        dataset.finalize = None
        dataset.stop_image_writer = None
        destination = self.root / "export"
        def factory(**kwargs):
            (kwargs["root"] / "meta").mkdir(parents=True)
            return dataset
        count = recorder.export_dataset(self.root, destination, "local/continuous_test", factory)
        self.assertEqual(count, 2)
        self.assertEqual(dataset.add_frame.call_count, 4)
        self.assertEqual(dataset.save_episode.call_count, 2)


if __name__ == "__main__":
    unittest.main()
