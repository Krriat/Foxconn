"""Exposure alignment, intermittent inputs and write metrics; no hardware."""
from pathlib import Path
import sys
import tempfile
import queue
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/recording'))
import record_xcore_teach_suction as r


class ThroughputTest(unittest.TestCase):
    def test_native_encoder_reuses_buffers_across_shapes_without_pixel_change(self):
        encoder = r.PngEncoder()
        rng = np.random.default_rng(19)
        try:
            with tempfile.TemporaryDirectory() as directory:
                for i, shape in enumerate(((1,1,3),(480,640,3),(2,1,3),(11,17,3))):
                    rgb = rng.integers(0,256,shape,dtype=np.uint8)[:,::-1]
                    path = Path(directory)/f'{i}.png'
                    encoder.write(path,rgb)
                    np.testing.assert_array_equal(cv2.imread(str(path))[:,:,::-1],rgb)
        finally:
            encoder.close()

    def test_fast_png_fallback_does_not_require_native_library(self):
        with patch.object(r,'PNG_COMPRESSOR',None):
            encoder = r.PngEncoder()
            self.assertEqual(encoder.backend,'zlib_store')
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'rgb.png'
                rgb = np.full((5,8,3),[254,127,0],np.uint8)
                encoder.write(path,rgb)
                np.testing.assert_array_equal(cv2.imread(str(path))[:,:,::-1],rgb)
            encoder.close()

    def test_twenty_hz_uses_fast_lossless_writer_and_bounded_queue(self):
        settings = r.dataset_schema(r.parse_args(['--fps','20','--width','4','--height','3']))
        with tempfile.TemporaryDirectory() as directory:
            writer = r.EpisodeWriter(Path(directory),'twenty Hz',settings,image_queue_size=2)
            try:
                self.assertEqual(len(writer.workers),3)
                self.assertEqual(writer.jobs.maxsize,2)
            finally:
                writer.discard()

    def test_full_bounded_writer_rejects_row_until_images_can_be_queued(self):
        settings = r.dataset_schema(r.parse_args(['--width','4','--height','3']))
        ready, release = queue.Queue(), threading.Event()
        original = r.PngEncoder.write
        def slow_write(encoder,path,rgb):
            ready.put(True)
            if not release.wait(5.):
                raise RuntimeError('test writer gate timed out')
            return original(encoder,path,rgb)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(r.PngEncoder,'write',slow_write):
                writer = r.EpisodeWriter(Path(directory),'bounded',settings,image_queue_size=1)
                writer.begin_segment()
                def add(i):
                    writer.add(r.Bundle(100.+i/30,np.zeros(16),np.zeros(16),
                        {k:np.zeros((3,4,3),np.uint8) for k in r.CAMERAS},
                        np.full(5,100.+i/30),99.,np.full(2,99.)))
                try:
                    for i in range(3):
                        add(i)
                        ready.get(timeout=3.)
                    add(3)
                    with self.assertRaises(r.WriteQueueFull):
                        add(4)
                    self.assertEqual(len(writer.rows),4)
                    self.assertEqual(writer.jobs.qsize(),1)
                finally:
                    release.set()
                    writer.end_segment('test')
                    writer.jobs.join()
                    writer.discard()

    def test_fast_png_is_pixel_exact_rgb_with_noncontiguous_input(self):
        rng = np.random.default_rng(7)
        for shape in ((3,4,3), (480,640,3)):
            source = rng.integers(0,256,shape,dtype=np.uint8)[:, ::-1]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'rgb.png'
                size = r.write_rgb_png(path, source)
                self.assertEqual(size, path.stat().st_size)
                decoded = cv2.imread(str(path))[:, :, ::-1]
                np.testing.assert_array_equal(decoded, source)

    def exercise(self, source_hz=30., fps=30, lost=None, duration=3., first_frame=0):
        args = r.parse_args(['--fps', str(fps), '--width', '4', '--height', '3'])
        outputs = r.SuctionOutputs(None, report=lambda _: None)
        inputs = r.KeyboardInputs(outputs, args)
        inputs.teaching = dict.fromkeys(r.SIDES, True)
        # A slow diagnostic rate must never veto an otherwise complete frame.
        inputs.rates = Mock(side_effect=r.NotReady('diagnostic rate below threshold'))
        events = []
        for index in range(first_frame, int(duration * source_hz)):
            for camera, phase, delay in [('head', 0., .070), ('left_wrist', .006, .020),
                                          ('right_wrist', .012, .025)]:
                if lost == (camera, index):
                    continue
                stamp = index / source_hz + phase
                events.append((stamp + delay, stamp, camera, index))
        events.sort()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = r.dataset_schema(args)
            r.prepare_root(root, settings)
            capture = r.Capture(inputs, lambda: r.EpisodeWriter(root, 'throughput test', settings),
                                fps, report=lambda _: None)
            capture.start(100., 10.)
            pointer, next_joint, row_count = 0, 0., 0
            for tick in range(int((duration + .3) / .002)):
                elapsed = tick * .002
                now, ros_now = 10. + elapsed, 100. + elapsed
                outputs.record_readback((False, False), now - .001)
                if elapsed >= next_joint:
                    for side in r.SIDES:
                        inputs.push(side, ros_now, now, np.full(7, elapsed))
                    next_joint = elapsed + .01
                while pointer < len(events) and events[pointer][0] <= elapsed:
                    received, stamp, key, index = events[pointer]
                    inputs.push(key, 100. + stamp, 10. + received,
                                np.full((3, 4, 3), index % 255, dtype=np.uint8))
                    pointer += 1
                capture.tick(ros_now, now)
                if len(capture.writer.rows) != row_count:
                    capture.writer.jobs.join()
                    row_count = len(capture.writer.rows)
            capture.finish(True, 10. + duration + .3)
            meta, arrays = r.validate_episode(root / 'episode_000000', settings)
            inputs.rates.assert_not_called()
            self.assertEqual(meta['writer_diagnostics']['written_frames'], meta['frames'])
            self.assertEqual(meta['writer_diagnostics']['queue_full_count'], 0)
            self.assertEqual(meta['capture_diagnostics']['scheduler_delay_count'], 0)
            return meta, arrays

    def test_thirty_hz_with_asymmetric_delivery_lag_retains_all_exposures(self):
        meta, arrays = self.exercise()
        self.assertEqual(meta['frames'], 90)
        self.assertEqual(len(meta['segments']), 1)
        self.assertLess(np.ptp(arrays['source_timestamp'], axis=1).max(), .033334)

    def test_twenty_seven_hz_does_not_stop_whole_recording_on_rate_check(self):
        meta, _ = self.exercise(source_hz=27.)
        self.assertEqual(meta['frames'], 81)

    def test_twenty_hz_request_downsamples_thirty_hz_without_duplicate_sources(self):
        meta, arrays = self.exercise(fps=20)
        self.assertTrue(59 <= meta['frames'] <= 61)
        self.assertEqual(len(meta['segments']), 1)
        self.assertTrue(np.all(np.diff(arrays['source_timestamp'], axis=0) > 0))

    def phase_boundary_sample(self, fps, *, earlier_frame=True, previous_step=None, joint_boundary=False):
        args = r.parse_args(['--fps', str(fps), '--width', '4', '--height', '3'])
        outputs = r.SuctionOutputs(None, report=lambda _: None)
        outputs.record_readback((False, False), 10.05)
        inputs = r.KeyboardInputs(outputs, args)
        inputs.teaching = dict.fromkeys(r.SIDES, True)
        reference, earlier, later = 100.1189342, 100.1022563, 100.1355968
        step = previous_step if previous_step is not None else (2 if fps == 20 else 1) / 30
        previous = dict.fromkeys(r.STREAMS, reference-step)
        previous.update(left_wrist=100.11-step, right_wrist=earlier-step)
        for side in r.SIDES:
            if joint_boundary:
                inputs.push(side, earlier, 10.2, np.full(7, .1))
                inputs.push(side, later, 10.2, np.full(7, .2))
            else:
                inputs.push(side, reference, 10.2, np.full(7, .1))
        for key, stamp in [('head', reference), ('left_wrist', 100.11),
                           *([('right_wrist', earlier)] if earlier_frame else []),
                           ('right_wrist', later)]:
            inputs.push(key, stamp, 10.2, np.zeros((3,4,3), np.uint8))
        return inputs.snapshot(100.35, 10.35, previous_sources=previous), previous

    def test_camera_phase_drift_does_not_create_an_avoidable_source_gap(self):
        # The later exposure is only 15 microseconds nearer to the head. It
        # would introduce a 100ms step at 20Hz (66.7ms at 30Hz), despite a valid
        # continuous exposure already being present in the input buffer.
        for fps in (20, 30):
            with self.subTest(fps=fps):
                bundle, previous = self.phase_boundary_sample(fps)
                index = r.STREAMS.index('right_wrist')
                self.assertAlmostEqual(bundle.source_stamps[index], 100.1022563)
                self.assertLessEqual(bundle.source_stamps[index]-previous['right_wrist'],
                                     1.5/fps+min(.02, .1/fps))
                self.assertLessEqual(np.ptp(bundle.source_stamps), 1/fps)

    def test_source_continuity_preference_neither_hides_loss_nor_replaces_nearest_timing(self):
        bundle, _ = self.phase_boundary_sample(20, earlier_frame=False)
        index = r.STREAMS.index('right_wrist')
        self.assertAlmostEqual(bundle.source_stamps[index], 100.1355968)
        # Both candidates are continuous here: keep the original nearest-time
        # choice instead of systematically preferring older images.
        bundle, _ = self.phase_boundary_sample(20, previous_step=.04)
        self.assertAlmostEqual(bundle.source_stamps[index], 100.1355968)

    def test_joint_pairing_also_avoids_an_unnecessary_continuity_break(self):
        bundle, _ = self.phase_boundary_sample(20, joint_boundary=True)
        for side in r.SIDES:
            self.assertAlmostEqual(bundle.source_stamps[r.STREAMS.index(side)], 100.1022563)
        np.testing.assert_array_equal(bundle.state[:14], np.full(14, .1, np.float32))
        np.testing.assert_array_equal(bundle.action[:14], bundle.state[:14])

    def test_missing_wrist_frame_resumes_without_reusing_old_images(self):
        meta, arrays = self.exercise(lost=('left_wrist', 30))
        self.assertGreaterEqual(meta['frames'], 88)
        self.assertLessEqual(meta['frames'], 89)
        self.assertTrue(np.all(np.diff(arrays['source_timestamp'], axis=0) > 0))
        self.assertGreaterEqual(len(meta['segments']), 2)

    def test_repeated_polling_never_repeats_a_single_source_exposure(self):
        meta, _ = self.exercise(duration=1 / 30)
        self.assertEqual(meta['frames'], 1)

    def test_pre_recording_buffer_does_not_appear_after_r(self):
        meta, arrays = self.exercise(duration=1., first_frame=-5)
        self.assertEqual(meta['frames'], 30)
        self.assertGreaterEqual(arrays['source_timestamp'].min(), 100. - 1e-6)

    def test_long_wait_for_teaching_is_not_reported_as_scheduler_delay(self):
        args = r.parse_args(['--width', '4', '--height', '3'])
        outputs = r.SuctionOutputs(None, report=lambda _: None)
        inputs = r.KeyboardInputs(outputs, args)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = r.dataset_schema(args)
            r.prepare_root(root, settings)
            capture = r.Capture(inputs, lambda: r.EpisodeWriter(root, 'wait test', settings), 30,
                                report=lambda _: None)
            capture.start(100., 10.)
            for i in range(1000):
                outputs.record_readback((False, False), 10. + i*.005)
                capture.tick(100. + i*.005, 10. + i*.005)
            self.assertEqual(capture.scheduler_delay_count, 0)
            self.assertTrue(capture.recording)
            capture.finish(False, 15.)

    def test_receive_diagnostics_separate_source_arrival_and_decode(self):
        args = r.parse_args([])
        inputs = r.KeyboardInputs(None, args)
        for i in range(60):
            inputs.observe('head', 100. + i/30, 10. + i/30,
                           100. + i/30 + .06, decode_ms=.4)
        metrics = inputs.diagnostics(12.)['head']
        self.assertAlmostEqual(metrics['source_hz'], 30.)
        self.assertAlmostEqual(metrics['receive_hz'], 30.)
        self.assertAlmostEqual(metrics['latency_ms_p95'], 60.)
        self.assertAlmostEqual(metrics['decode_ms_p95'], .4)


if __name__ == '__main__':
    unittest.main()
