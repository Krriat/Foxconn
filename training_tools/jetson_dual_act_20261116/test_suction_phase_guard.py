"""Recorded IO/pose replay; no ROS, SDK, or device connections."""
import json
from pathlib import Path
import unittest
import numpy as np
from suction_phase_guard import FarReleaseGuard

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'AIDE_ACT_dataset_20261111_195100/act_teach_keyboard_suction_30hz'


class SuctionReplayChecks(unittest.TestCase):
    def test_all_original_far_demonstrations_preserve_IO(self):
        count=0
        for ep in sorted(RAW.glob('episode_*')):
            if int(ep.name.split('_')[-1])%2:continue
            record=np.load(ep/'trajectory.npz');gate=FarReleaseGuard()
            with self.subTest(episode=ep.name):
                for i,(state,action) in enumerate(zip(record['state'],record['action'])):
                    target=[bool(x>=.5) for x in action[14:16]]
                    gate.update_measured(state[:14])
                    self.assertEqual(gate.filter(target,now=i*.05,enabled=True),target)
            count+=1
        self.assertEqual(count,29)

    def test_latest_failed_run_blocks_release_then_faults_without_empty_flipping(self):
        p=ROOT/'training_verification/completion_contact_fix_20261117_090845/physical02/live01/ros_events.jsonl'
        gate=FarReleaseGuard();previous=[False,False];started=None;blocked=None;fault=None
        with p.open() as f:
            for line in f:
                row=json.loads(line);value=row['value']
                if row['topic']!='servo_trace' or not value.get('active'):continue
                now=row['received_ns']/1e9
                if started is None:started=now
                measured=value.get('left_measured',[])+value.get('right_measured',[])
                if len(measured)!=14:continue
                target=[True if x>=.6 else False if x<=.4 else old
                        for x,old in zip(value['fused_target'][14:],previous)]
                gate.update_measured(measured)
                try:applied=gate.filter(target,now=now,enabled=True)
                except RuntimeError as e:fault=now;self.assertIn('grasp unconfirmed',str(e));break
                if target[0] is False and applied[0] is True and blocked is None:blocked=now
                previous=applied
        self.assertIsNotNone(blocked);self.assertIsNotNone(fault)
        self.assertGreaterEqual(fault-blocked,1.5);self.assertLess(fault-blocked,1.6)
        self.assertLess(fault-started,17.)

    def test_explicit_stop_and_non_far_paths_bypass_guard(self):
        gate=FarReleaseGuard();gate.filter([True,False],now=0,enabled=True)
        self.assertEqual(gate.filter([False,False],now=10,enabled=False),[False,False])
        self.assertFalse(gate.on_seen)

    def test_reset_clears_previous_stage_lift(self):
        gate=FarReleaseGuard();gate.left_lift_seen=True;gate.reset()
        self.assertFalse(gate.left_lift_seen)


if __name__=='__main__':unittest.main()
