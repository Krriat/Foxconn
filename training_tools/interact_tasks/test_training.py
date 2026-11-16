"""Check raw-episode task splits and padded physical-unit metrics, offline."""
import unittest
import numpy as np
from train import ErrorTotals, task_rows, validate_provenance


class TrainingTests(unittest.TestCase):
    def test_all_58_demonstrations_in_exactly_one_task_and_split(self):
        seen = set()
        for stage, parity, counts in (('far', 0, (27, 2)), ('near', 1, (25, 4))):
            provenance = validate_provenance(stage)
            for split, count in zip(('train', 'val'), counts):
                raw = set(provenance[split]['raw_episodes'])
                self.assertEqual(len(raw), count)
                self.assertTrue(all(int(x.rsplit('_', 1)[1]) % 2 == parity for x in raw))
                self.assertFalse(raw & seen)
                seen.update(raw)
        self.assertEqual(len(seen), 58)

    def test_continuous_segments_are_never_reclassified_by_exported_id(self):
        # Prepared episode indices differ from raw demonstration numbers.
        for split in ('train', 'val'):
            far, near = task_rows('far', split), task_rows('near', split)
            self.assertFalse({r['episode_index'] for r in far} & {r['episode_index'] for r in near})
            self.assertTrue(any(int(r['raw_episode'].split('_')[-1]) != r['episode_index']
                                for r in far + near))

    def test_padding_does_not_affect_joint_mae_or_suction_errors(self):
        pred, target = np.zeros((1, 2, 16)), np.zeros((1, 2, 16))
        pred[0, 0, :14] = 1
        pred[0, 0, 14] = 1
        pred[0, 1] = 1e6
        metric = ErrorTotals()
        metric.add(pred, target, np.array([[True, False]]), np.ones(16), np.zeros(16))
        result = metric.report()
        self.assertEqual(result['valid_targets'], 1)
        self.assertAlmostEqual(result['joint_mae_deg'], np.rad2deg(1))
        self.assertEqual(result['suction_error_fraction'], [1., 0.])

    def test_physical_suction_threshold_uses_normalization(self):
        pred, target = np.zeros((2, 16)), np.zeros((2, 16))
        pred[0, 14] = 1
        target[0, 14] = 1
        std, mean = np.ones(16), np.zeros(16)
        std[14], mean[14] = .2, .4
        metric = ErrorTotals()
        metric.add(pred, target, np.array([True, True]), std, mean)
        self.assertEqual(metric.report()['suction_error_fraction'], [0., 0.])


if __name__ == '__main__':
    unittest.main()
