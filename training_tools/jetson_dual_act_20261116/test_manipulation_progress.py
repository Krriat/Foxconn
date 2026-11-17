"""Replay known successful motions and partial failed runs; no ROS/device IO."""
import json
from pathlib import Path
import unittest
import numpy as np
from manipulation_progress import ManipulationProgress

ROOT=Path(__file__).resolve().parents[2]


class RecordedMotionChecks(unittest.TestCase):
    def test_both_known_successful_sweeps_reach_handoff(self):
        for stage,minimum_time in (('far',30.),('near',23.)):
            with self.subTest(stage=stage):
                record=np.load(ROOT/f'training_verification/success_2105_comparison_20261113/success_{stage}_analysis.npz')
                gate=ManipulationProgress();gate.reset(stage);first=None
                for t,q in zip(record['time']-record['time'][0],record['measured']):
                    gate.update(q,float(t))
                    if gate.ready and first is None:first=float(t)
                self.assertIsNotNone(first);self.assertGreater(first,minimum_time)

    def test_both_premature_far_completions_are_blocked(self):
        for directory in ('jetson_dual_act_live_20261117_085515','jetson_dual_act_live_20261117_090304'):
            with self.subTest(directory=directory):
                gate=ManipulationProgress();gate.reset('far');run=None;count=0
                p=ROOT/'training_verification'/directory/'live01/ros_events.jsonl'
                for line in p.open():
                    e=json.loads(line);v=e['value']
                    if e['topic']!='servo_trace' or not v.get('active'):continue
                    if run is None:run=v['run_id']
                    if v['run_id']!=run:continue
                    q=v.get('left_measured',[])+v.get('right_measured',[])
                    if len(q)!=14:continue
                    gate.update(q,e['received_ns']/1e9);count+=1
                    self.assertFalse(gate.ready)
                self.assertGreater(count,1000);self.assertTrue(gate.sweep_seen)

    def test_old_stage_latches_cannot_authorize_new_stage(self):
        gate=ManipulationProgress();gate.ready=True;gate.sweep_seen=True
        gate.reset('near');self.assertFalse(gate.ready);self.assertFalse(gate.sweep_seen)


if __name__=='__main__':unittest.main()
