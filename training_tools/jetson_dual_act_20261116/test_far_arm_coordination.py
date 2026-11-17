"""Far coordination preserves ACT left/IO and resumes ACT right after lift."""
import copy
import unittest
import numpy as np
from far_arm_coordination import FarArmCoordination


class Checks(unittest.TestCase):
    def setUp(self):
        self.g=FarArmCoordination();self.anchor=[.1]*7+[-.3]*7
        self.actions=np.linspace(-.4,.4,800).reshape(50,16).tolist()

    def call(self,**kwargs):
        return self.g.apply(self.actions,run_id=kwargs.get('run_id','far1'),
            verified_anchor=kwargs.get('anchor',self.anchor),enabled=kwargs.get('enabled',True),
            left_lift_seen=kwargs.get('lift',False))

    def test_right_is_held_and_left_and_suction_are_exact(self):
        original=copy.deepcopy(self.actions);out=self.call()
        self.assertEqual(self.actions,original)
        for src,dst in zip(original,out):
            self.assertEqual(dst[:7],src[:7]);self.assertEqual(dst[14:],src[14:])
            self.assertEqual(dst[7:14],self.anchor[7:])

    def test_lift_releases_without_changing_right_ACT_targets(self):
        self.call();self.assertEqual(self.call(lift=True),self.actions)
        self.assertEqual(self.call(lift=False),self.actions)

    def test_new_run_requires_new_home_and_clears_old_lift(self):
        self.call(lift=True);out=self.call(run_id='far2',anchor=[.2]*14)
        self.assertEqual(out[0][7:14],[.2]*7);self.assertFalse(self.g.released)

    def test_preview_near_manual_and_disabled_paths_are_not_modified(self):
        self.call();self.assertIs(self.call(enabled=False),self.actions)
        self.assertFalse(self.g.status()['right_held'])

    def test_missing_verified_pose_cannot_enable_hold(self):
        with self.assertRaises(ValueError):self.call(anchor=[])


if __name__=='__main__':unittest.main()
