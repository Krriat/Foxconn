"""Joint envelope regression; no ROS initialization or hardware calls."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training_tools/two_stage_policy'))
from protocol import HARD_UPPER, joint_bounds, validate_config, validate_joints, validate_chunk


class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
        self.margin = self.config['joint_limit_margin_rad']
        self.overrides = self.config['joint_upper_limits_rad']
        self.home = self.config['home_left']+self.config['home_right']

    def test_only_two_upper_bounds_change_and_native_margin_remains(self):
        validate_config(self.config)
        lower,upper = joint_bounds(self.margin,self.overrides)
        old_lower,old_upper = joint_bounds(self.margin)
        self.assertEqual(lower,old_lower)
        self.assertEqual([i for i in range(14) if upper[i]!=old_upper[i]],[3,10])
        for i in range(14):
            self.assertLessEqual(upper[i],HARD_UPPER[i%7]-.08)

    def test_reviewed_caps_are_enforced_for_targets_and_measured_joints(self):
        for index,cap in ((3,136.5),(10,140.25)):
            q=self.home.copy();q[index]=math.radians(cap)
            validate_joints(q,self.margin,self.overrides)
            q[index]+=1e-6
            with self.assertRaises(ValueError):validate_joints(q,self.margin,self.overrides)
        q=self.home.copy();q[3]=math.radians(-48)-1e-6
        with self.assertRaises(ValueError):validate_joints(q,self.margin,self.overrides)

    def test_malformed_or_excessive_override_rejected(self):
        for value in (None,float('nan'),float('inf'),True,math.radians(141)):
            c=copy.deepcopy(self.config);c['joint_upper_limits_rad']['right_joint_4']=value
            with self.assertRaises(ValueError):validate_config(c)
        for value in ({'right_joint_1':1.},[],{'left_joint_4':math.radians(140)}):
            c=copy.deepcopy(self.config);c['joint_upper_limits_rad']=value
            with self.assertRaises(ValueError):validate_config(c)

    def test_all_recorded_demonstrations_fit(self):
        base=ROOT/'AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz'
        lower,upper=joint_bounds(self.margin,self.overrides)
        old_lower,old_upper=joint_bounds(self.margin)
        frames,formerly_rejected=0,0
        for path in sorted(base.glob('episode_*/trajectory.npz')):
            action=np.load(path)['action'][:,:14]
            self.assertTrue(np.isfinite(action).all())
            self.assertTrue(((action>=lower)&(action<=upper)).all(),str(path))
            frames+=len(action)
            formerly_rejected+=int(((action<old_lower)|(action>old_upper)).any(axis=1).sum())
        self.assertEqual(frames,27998)
        self.assertEqual(formerly_rejected,720)

    def test_actual_rejected_future_chunk_is_now_valid(self):
        c=json.loads((ROOT/'training_verification/restore_far_20261112/near_limit_fix/rejected_chunk.json').read_text())
        args=(c,c['run_id'],c['publisher_id'],c['sequence']-1)
        # Preserve recorded timestamps; model a fresh 5-ms transport interval.
        now=c['send_time_ns']+5_000_000
        previous=copy.deepcopy(self.config);previous.pop('joint_upper_limits_rad')
        with self.assertRaises(ValueError):validate_chunk(*args,previous,now)
        self.assertEqual(validate_chunk(*args,self.config,now),c['actions'])


if __name__=='__main__':unittest.main()
