from __future__ import annotations

import unittest

import numpy as np

from roi_act_runtime.controller import ChunkFusionController
from roi_act_runtime.preprocessing import GreenHSVDetector, letterbox_rgb


class ROIActPreprocessingTest(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio_and_adds_black_bars(self) -> None:
        wrist = np.full((480, 640, 3), 255, dtype=np.uint8)
        output = letterbox_rgb(wrist, 360, 640)
        self.assertEqual(output.shape, (360, 640, 3))
        self.assertTrue(np.all(output[:, :80] == 0))
        self.assertTrue(np.all(output[:, 80:560] == 255))
        self.assertTrue(np.all(output[:, 560:] == 0))

    def test_green_detector_returns_normalized_valid_box(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[20:60, 50:110] = (0, 255, 0)
        box = GreenHSVDetector().detect(image)
        self.assertEqual(box[4], 1.0)
        np.testing.assert_allclose(box[:4], [0.25, 0.2, 0.55, 0.6], atol=0.02)

    def test_green_detector_marks_missing_roi_invalid(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        box = GreenHSVDetector().detect(image)
        np.testing.assert_array_equal(box, [0.0, 0.0, 1.0, 1.0, 0.0])


class ROIActChunkFusionTest(unittest.TestCase):
    def test_stride_and_overlap_match_existing_act_policy(self) -> None:
        calls = 0

        def predict(_observation):
            nonlocal calls
            base = 0.0 if calls == 0 else 100.0
            calls += 1
            return (np.arange(100, dtype=np.float32) + base)[:, None]

        controller = ChunkFusionController(
            predict,
            raw_chunk_size=100,
            action_steps=10,
            action_stride=3,
            replan_interval=2,
            new_prediction_weight=0.7,
        )
        self.assertEqual(float(controller.select_action({})[0]), 0.0)
        self.assertEqual(float(controller.select_action({})[0]), 3.0)
        controller._collect_worker(block=True)
        # New chunk starts at logical step 1. At step 2 it overlaps old raw
        # index 6 with new raw index 3: 0.3*6 + 0.7*103 = 73.9.
        self.assertAlmostEqual(float(controller.select_action({})[0]), 73.9, places=4)
        controller.close()

    def test_invalid_stride_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "need 109 raw actions"):
            ChunkFusionController(
                lambda _observation: np.zeros((100, 8), dtype=np.float32),
                raw_chunk_size=100,
                action_steps=10,
                action_stride=12,
                replan_interval=2,
                new_prediction_weight=0.7,
            )


if __name__ == "__main__":
    unittest.main()
