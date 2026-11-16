"""Physical-teaching alignment, controls and real on-disk episode round trips."""
import importlib.util
import queue
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from xcore_teach_recording.core import Bundle, ButtonLatch, CAMERAS, Capture, Inputs, NotReady, STATE_NAMES, STREAMS
from xcore_teach_recording.dataset import EpisodeWriter, prepare_root, schema, validate_episode
from xcore_teach_recording.ros_node import RecorderNode, decode_rgb, joint_values

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('teach_export', ROOT / 'scripts/dataset_tools/export_teach_lerobot.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def populate(inputs, stamp=10., receipt=1.):
    inputs.teaching = dict(left=True, right=True)
    inputs.button_at = receipt
    for key in ('left', 'right'):
        inputs.push(key, stamp, receipt, np.arange(7, dtype=float) / 10)
    inputs.push('head_pha', stamp, receipt, [.1])
    inputs.push('waist_pitch', stamp, receipt, [.2])
    for key in CAMERAS:
        inputs.push(key, stamp, receipt, np.full((3, 4, 3), [10, 20, 30], dtype=np.uint8))


def bundle(i):
    state = np.arange(16, dtype=np.float32) / 10
    return Bundle(10. + i * .2, state, {key: np.full((3, 4, 3), [10, 20, 30], dtype=np.uint8) for key in CAMERAS}, np.full(7, 10. + i * .2))


class AlignmentTests(unittest.TestCase):
    def test_30hz_rejects_20hz_arm_feedback_even_when_last_position_is_fresh(self):
        inputs = Inputs(required_fps=30)
        for i in range(61):
            populate(inputs, 10. + i / 30, 1. + i / 30)
        inputs.rate_history['left'].clear()
        for i in range(41):
            inputs.rate_history['left'].append((10. + i / 20, 1. + i / 20))
        with self.assertRaisesRegex(NotReady, 'left measured 20.0 Hz'):
            inputs.snapshot(12., 3.)

    def test_30hz_accepts_distinct_source_samples_and_reports_rates(self):
        inputs = Inputs(max_skew=1/30, required_fps=30)
        for i in range(61):
            populate(inputs, 10. + i / 30, 1. + i / 30)
        got = inputs.snapshot(12., 3.)
        self.assertEqual(got.state.shape, (16,))
        for rate in inputs.rates(3.).values():
            self.assertAlmostEqual(rate, 30.)

    def test_bursted_old_source_timestamps_do_not_fake_a_high_rate(self):
        inputs = Inputs(required_fps=30)
        for i in range(61):
            populate(inputs, 10. + i / 5, 1. + i / 30)
        with self.assertRaisesRegex(NotReady, 'measured 5.0 Hz'):
            inputs.snapshot(22., 3.)

    def test_new_images_cannot_reuse_a_previous_measured_position(self):
        inputs = Inputs()
        populate(inputs)
        for key in CAMERAS:
            inputs.push(key, 10.03, 1.03, bundle(0).images[key])
        with self.assertRaisesRegex(NotReady, 'duplicate source samples'):
            inputs.snapshot(10.03, 1.03, previous_sources={key: 10. for key in STREAMS})

    def test_state_order_and_units(self):
        inputs = Inputs()
        populate(inputs)
        got = inputs.snapshot(10.01, 1.01)
        self.assertEqual(len(STATE_NAMES), 16)
        self.assertEqual(STATE_NAMES[0], 'left_joint_1.pos')
        self.assertEqual(STATE_NAMES[7], 'right_joint_1.pos')
        np.testing.assert_allclose(got.state[14:], [.1, .2])

    def test_both_teaching_fresh_heartbeat_and_feedback_are_required(self):
        for reason in ('mode', 'buttons', 'motor'):
            with self.subTest(reason=reason):
                inputs = Inputs()
                populate(inputs)
                if reason == 'mode': inputs.teaching['right'] = False
                if reason == 'buttons': inputs.button_at = 0.
                if reason == 'motor': inputs.buffers['waist_pitch'].clear()
                with self.assertRaises(NotReady): inputs.snapshot(10.01, 1.01)
        inputs = Inputs()
        populate(inputs)
        with self.assertRaises(NotReady): inputs.snapshot(10.4, 1.4)

    def test_aligns_states_to_image_time_instead_of_newer_unrelated_state(self):
        inputs = Inputs()
        populate(inputs)
        inputs.push('left', 10.4, 1.01, np.ones(7))
        got = inputs.snapshot(10.4, 1.01)
        self.assertAlmostEqual(got.state[0], 0.)
        self.assertEqual(got.source_stamps[0], 10.)

    def test_camera_skew_and_repeated_images_are_rejected(self):
        inputs = Inputs()
        populate(inputs)
        with self.assertRaisesRegex(NotReady, 'duplicate'):
            inputs.snapshot(10.01, 1.01, {key: 10. for key in CAMERAS})
        inputs.buffers['right_wrist'].clear()
        inputs.push('right_wrist', 10.4, 1.01, bundle(0).images['right_wrist'])
        with self.assertRaisesRegex(NotReady, 'synchronization'):
            inputs.snapshot(10.4, 1.01)

    def test_clock_jump_drops_old_history(self):
        inputs = Inputs()
        populate(inputs)
        inputs.push('left', 9., 1.01, np.zeros(7))
        self.assertEqual(len(inputs.buffers['head']), 0)
        with self.assertRaisesRegex(NotReady, 'clock'):
            inputs.snapshot(10.01, 1.01)

    def test_fresh_cached_positions_do_not_override_a_driver_fault(self):
        inputs = Inputs()
        populate(inputs)
        inputs.motor_faults.add('waist_pitch')
        with self.assertRaisesRegex(NotReady, 'driver fault'):
            inputs.snapshot(10.01, 1.01)

    def test_joint_names_are_reordered_and_malformed_data_is_rejected(self):
        message = SimpleNamespace(name=['b', 'a'], position=[2., 1.])
        np.testing.assert_equal(joint_values(message, ['a', 'b']), [1., 2.])
        message.name = ['a', 'a']
        with self.assertRaises(ValueError): joint_values(message, ['a'])

    def test_padded_bgr_image_is_decoded_to_rgb(self):
        message = SimpleNamespace(width=2, height=1, encoding='bgr8', step=8,
                                  data=bytes([30, 20, 10, 60, 50, 40, 0, 0]))
        np.testing.assert_equal(decode_rgb(message, 2, 1), [[[10, 20, 30], [40, 50, 60]]])
        message.step = 5
        with self.assertRaises(ValueError): decode_rgb(message, 2, 1)


class ControlTests(unittest.TestCase):
    def test_30hz_cadence_does_not_accumulate_polling_delay(self):
        inputs = Inputs(max_skew=1/30)
        writer = Mock()
        capture = Capture(inputs, lambda: writer, fps=30, notify=Mock())
        image_index = -1
        for tick in range(400):  # A 5 ms polling loop, two seconds at 30 Hz.
            elapsed = tick * .005
            current = int(round(elapsed * 30, 8))
            if current != image_index:
                populate(inputs, 10. + current / 30, 1. + elapsed)
                image_index = current
            if tick == 0:
                capture.start(10., 1.)
            capture.tick(10. + elapsed, 1. + elapsed)
        self.assertTrue(capture.recording)
        self.assertEqual(writer.add.call_count, 60)
        stamps = [call.args[0].stamp for call in writer.add.call_args_list]
        np.testing.assert_allclose(np.diff(stamps), 1/30)

    def test_30hz_a_dropped_camera_frame_pauses_before_recording_the_gap(self):
        inputs = Inputs()
        populate(inputs)
        writer = Mock()
        capture = Capture(inputs, lambda: writer, fps=30, notify=Mock())
        capture.start(10., 1.)
        capture.tick(10., 1.)
        populate(inputs, 10. + 2/30, 1. + 2/30)
        capture.tick(10. + 2/30, 1. + 2/30)
        self.assertFalse(capture.recording)
        self.assertEqual(writer.add.call_count, 1)

    def test_B_requires_release_and_A_is_independent(self):
        latch = ButtonLatch()
        for _ in range(5): self.assertFalse(latch.update(8))
        self.assertFalse(latch.update(4))  # A does not start recording.
        self.assertTrue(latch.update(8))
        self.assertFalse(latch.update(8))
        self.assertFalse(latch.update(0))
        self.assertTrue(latch.update(8))

    def test_start_pause_save_and_duplicate_frame_protection(self):
        inputs = Inputs()
        populate(inputs)
        writer = Mock()
        capture = Capture(inputs, lambda: writer, notify=Mock())
        capture.start(10.01, 1.01)
        capture.tick(10.01, 1.01)
        self.assertEqual(writer.add.call_count, 1)
        capture.command('s')
        writer.save.assert_not_called()
        capture.tick(10.21, 1.21)
        self.assertTrue(capture.recording)  # Allow a slightly late exposure.
        capture.tick(10.46, 1.46)
        self.assertFalse(capture.recording)
        writer.end_segment.assert_called_once()
        self.assertEqual(writer.add.call_count, 1)
        capture.command('s')
        writer.save.assert_called_once()
        self.assertIsNone(capture.writer)

    def test_refuses_to_start_outside_teaching_or_without_fresh_images(self):
        factory = Mock()
        inputs = Inputs()
        capture = Capture(inputs, factory, notify=Mock())
        capture.start(10., 1.)
        factory.assert_not_called()
        self.assertFalse(capture.recording)

    def test_waits_for_a_slightly_late_exposure_without_duplicating_it(self):
        inputs = Inputs()
        populate(inputs)
        writer = Mock()
        capture = Capture(inputs, lambda: writer, notify=Mock())
        capture.start(10.01, 1.01)
        capture.tick(10.01, 1.01)
        capture.tick(10.211, 1.211)
        self.assertTrue(capture.recording)
        self.assertEqual(writer.add.call_count, 1)
        populate(inputs, 10.2, 1.22)
        capture.tick(10.22, 1.22)
        self.assertTrue(capture.recording)
        self.assertEqual(writer.add.call_count, 2)

    def test_missing_a_source_frame_pauses_instead_of_compressing_time(self):
        inputs = Inputs()
        populate(inputs)
        writer = Mock()
        capture = Capture(inputs, lambda: writer, notify=Mock())
        capture.start(10.01, 1.01)
        capture.tick(10.01, 1.01)
        populate(inputs, 10.4, 1.40)
        capture.tick(10.4, 1.40)
        self.assertFalse(capture.recording)
        self.assertEqual(writer.add.call_count, 1)

    def test_late_cycle_pauses_without_catchup_frames(self):
        inputs = Inputs()
        populate(inputs)
        writer = Mock()
        capture = Capture(inputs, lambda: writer, notify=Mock())
        capture.start(10.01, 1.01)
        capture.tick(10.01, 1.01)
        capture.tick(11., 2.)
        self.assertFalse(capture.recording)
        self.assertEqual(writer.add.call_count, 1)


class ButtonActionTests(unittest.TestCase):
    def setUp(self):
        self.inputs = Inputs()
        populate(self.inputs)
        self.writer = Mock()
        self.factory = Mock(return_value=self.writer)
        self.capture = Capture(self.inputs, self.factory, notify=Mock())

    def start(self):
        self.capture.button(10., 1.)
        self.capture.tick(10., 1.)
        self.assertTrue(self.capture.recording)

    def test_single_B_starts_immediately_and_does_not_auto_save_the_start(self):
        self.start()
        populate(self.inputs, 10.2, 1.2)
        self.capture.tick(10.2, 1.2)
        populate(self.inputs, 10.4, 1.4)
        self.capture.tick(10.4, 1.4)
        self.assertTrue(self.capture.recording)
        self.writer.save.assert_not_called()

    def test_second_separate_B_stops_immediately_and_saves_after_double_click_window(self):
        self.start()
        frames = self.writer.add.call_count
        self.capture.button(10.5, 1.5)
        self.assertFalse(self.capture.recording)
        self.capture.tick(10.84, 1.84)
        self.writer.save.assert_not_called()
        self.assertEqual(self.writer.add.call_count, frames)
        self.capture.tick(10.86, 1.86)
        self.writer.save.assert_called_once()
        self.writer.discard.assert_not_called()
        self.assertIsNone(self.capture.writer)

    def test_double_B_discards_and_cancels_deferred_save(self):
        self.start()
        self.capture.button(10.5, 1.5)
        self.capture.button(10.7, 1.7)
        self.capture.tick(11., 2.)
        self.writer.discard.assert_called_once()
        self.writer.save.assert_not_called()
        self.assertFalse(self.capture.recording)
        self.assertIsNone(self.capture.writer)

    def test_double_B_immediately_after_start_discards_the_new_episode(self):
        self.start()
        self.capture.button(10.2, 1.2)
        self.writer.discard.assert_called_once()
        self.writer.save.assert_not_called()
        self.assertFalse(self.capture.recording)

    def test_delayed_queue_processing_preserves_real_click_gap(self):
        populate(self.inputs, 11., 2.)
        self.capture.button(11., 2., pressed_at=1.)
        self.capture.tick(11., 2.)
        self.capture.button(11., 2.001, pressed_at=1.6)
        self.capture.tick(11., 2.002)
        self.writer.save.assert_called_once()
        self.writer.discard.assert_not_called()

    def test_next_separate_B_commits_pending_save_then_starts_a_new_episode(self):
        next_writer = Mock()
        self.factory.side_effect = [self.writer, next_writer]
        self.start()
        self.capture.button(10.5, 1.5)
        populate(self.inputs, 10.9, 1.9)
        self.capture.button(10.9, 1.9)
        self.writer.save.assert_called_once()
        self.writer.discard.assert_not_called()
        self.assertIs(self.capture.writer, next_writer)
        self.assertTrue(self.capture.recording)

    def test_fault_paused_data_can_be_saved_without_restarting_capture(self):
        self.start()
        self.capture.pause('camera stopped')
        self.inputs.teaching['left'] = False
        self.capture.button(10.5, 1.5)
        self.capture.tick(10.9, 1.9)
        self.writer.save.assert_called_once()
        self.assertFalse(self.capture.recording)

    def test_failed_save_keeps_pending_data_and_B_can_retry(self):
        self.writer.save.side_effect = [OSError('synthetic disk error'), None]
        self.start()
        self.capture.button(10.5, 1.5)
        self.capture.tick(10.9, 1.9)
        self.assertIs(self.capture.writer, self.writer)
        self.writer.discard.assert_not_called()
        self.capture.tick(11., 2.)
        self.assertEqual(self.writer.save.call_count, 1)
        self.capture.button(11.5, 2.5)
        self.capture.tick(11.9, 2.9)
        self.assertEqual(self.writer.save.call_count, 2)
        self.assertIsNone(self.capture.writer)

    def test_ros_button_edges_keep_receipt_times_and_ignore_held_startup_B(self):
        receiver = SimpleNamespace(inputs=self.inputs, buttons=ButtonLatch(), commands=queue.SimpleQueue())
        masks = [8, 8, 0, 4, 8, 8, 0, 8]
        with patch('xcore_teach_recording.ros_node.time.monotonic', side_effect=range(len(masks))):
            for mask in masks:
                RecorderNode.on_buttons(receiver, SimpleNamespace(data=mask))
        self.assertEqual(receiver.commands.get_nowait(), ('button', 4))
        self.assertEqual(receiver.commands.get_nowait(), ('button', 7))
        self.assertTrue(receiver.commands.empty())

    def test_real_discard_never_removes_an_already_saved_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = schema(5, 4, 3)
            root = prepare_root(Path(directory), settings)
            capture = Capture(self.inputs, lambda: EpisodeWriter(root, 'B workflow', settings), notify=Mock())
            capture.button(10., 1.)
            capture.tick(10., 1.)
            capture.button(10.5, 1.5)
            capture.tick(10.9, 1.9)
            saved = root / 'episode_000000'
            self.assertTrue(saved.is_dir())
            populate(self.inputs, 11., 2.)
            capture.button(11., 2.)
            capture.tick(11., 2.)
            pending = capture.writer.path
            capture.button(11.5, 2.5)
            capture.button(11.7, 2.7)
            self.assertFalse(pending.exists())
            self.assertTrue(saved.is_dir())
            validate_episode(saved)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = schema(5., 4, 3)
        self.root = prepare_root(Path(self.temp.name) / 'raw', self.settings)

    def record(self):
        writer = EpisodeWriter(self.root, 'move object', self.settings)
        self.addCleanup(writer.close)
        for start in (0, 2):
            writer.begin_segment()
            for i in range(start, start + 2): writer.add(bundle(i))
            writer.end_segment('B pause')
        return writer

    def test_raw_roundtrip_preserves_actions_images_and_segments(self):
        path = self.record().save()
        meta, data = validate_episode(path, self.settings)
        self.assertEqual(meta['frames'], 4)
        self.assertEqual([(s['start'], s['end']) for s in meta['segments']], [(0, 2), (2, 4)])
        np.testing.assert_equal(data['state'], data['action'])
        self.assertEqual(data['state'].shape, (4, 16))
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_30hz_parallel_lossless_png_roundtrip(self):
        settings = schema(30, 4, 3)
        root = prepare_root(Path(self.temp.name) / 'raw_30hz', settings)
        writer = EpisodeWriter(root, '30 Hz roundtrip', settings)
        self.addCleanup(writer.close)
        self.assertEqual(len(writer.workers), 3)
        writer.begin_segment()
        for index in range(4):
            frame = bundle(index)
            frame.images['head'][:] = [index, 20, 30]
            writer.add(frame)
        writer.end_segment('B pause')
        path = writer.save()
        meta, data = validate_episode(path, settings)
        self.assertEqual(meta['frames'], 4)
        self.assertEqual(meta['fps'], 30)
        import cv2
        for index in range(4):
            rgb = cv2.imread(str(path / 'images/head' / f'{index:06d}.png'))[:, :, ::-1]
            np.testing.assert_equal(rgb[0, 0], [index, 20, 30])

    def test_existing_schema_cannot_be_silently_replaced(self):
        with self.assertRaises(ValueError): prepare_root(self.root, schema(10., 4, 3))

    def test_discard_only_removes_current_staging_episode(self):
        saved = self.record().save()
        writer = self.record()
        staging = writer.path
        writer.discard()
        self.assertFalse(staging.exists())
        self.assertTrue(saved.exists())
        with self.assertRaises(RuntimeError):
            other = self.record()
            other.close()
            other.path = saved
            other.discard()

    def test_unconfirmed_exit_preserves_a_partial_record(self):
        writer = self.record()
        path = writer.preserve()
        self.assertTrue((path / 'trajectory.npz').exists())
        self.assertFalse(list(self.root.glob('episode_*')))
        with self.assertRaisesRegex(ValueError, 'unconfirmed'): validate_episode(path)

    def test_missing_image_blocks_export_before_dataset_creation(self):
        path = self.record().save()
        (path / 'images/head/000001.png').unlink()
        factory = Mock()
        with self.assertRaises(ValueError):
            exporter.export(self.root, Path(self.temp.name) / 'lerobot', 'local/test', factory)
        factory.assert_not_called()

    def test_disk_error_never_produces_a_completed_episode(self):
        writer = EpisodeWriter(self.root, 'test', self.settings)
        self.addCleanup(writer.close)
        writer.begin_segment()
        with patch('xcore_teach_recording.dataset.cv2.imwrite', return_value=False):
            writer.add(bundle(0))
            writer.jobs.join()
        writer.end_segment('pause')
        with self.assertRaisesRegex(RuntimeError, 'writer failed'): writer.save()
        self.assertFalse(list(self.root.glob('episode_*')))

    def test_export_splits_pauses_and_preserves_rgb_and_16d_features(self):
        self.record().save()
        destination = Path(self.temp.name) / 'lerobot'
        saved, pending = [], []
        dataset = Mock()
        dataset.image_writer = None
        dataset.add_frame.side_effect = lambda frame: pending.append(frame)
        dataset.save_episode.side_effect = lambda: (saved.append(pending.copy()), pending.clear())
        def factory(**kwargs):
            self.assertEqual(kwargs['features']['action']['shape'], (16,))
            self.assertEqual(kwargs['fps'], 5.)
            (destination / 'meta').mkdir(parents=True)
            return dataset
        count = exporter.export(self.root, destination, 'local/test', factory)
        self.assertEqual(count, 2)
        self.assertEqual([len(frames) for frames in saved], [2, 2])
        frame = saved[0][0]
        np.testing.assert_equal(frame['observation.images.head'][0, 0], [10, 20, 30])
        np.testing.assert_equal(frame['action'], frame['observation.state'])
        self.assertTrue((destination / 'meta/physical_teach_source.json').exists())
        with self.assertRaises(FileExistsError): exporter.export(self.root, destination, 'local/test', factory)


if __name__ == '__main__':
    unittest.main()
