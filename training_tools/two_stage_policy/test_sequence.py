import unittest
from task_sequence import CompletionGate, SceneDetector
from pathlib import Path
import numpy as np


class Gates(unittest.TestCase):
    def feed(self, gate, *, scene="far_open", suction=(False,False), q=None,
             distance=.01, start=6., target_off=True):
        q = [0.]*14 if q is None else q
        return [gate.update(start+i*.1, int((start+i*.1)*1e6), scene, q, suction,
                            distance, target_off) for i in range(15)]

    def armed(self, stage="far"):
        gate=CompletionGate(stage, 0.)
        gate.update(1., 1, "closed", [0.]*14, [True,False], .5, False)
        return gate

    def test_stationary_is_not_success(self):
        self.assertFalse(any(self.feed(CompletionGate("far",0.))))
        self.assertFalse(any(self.feed(self.armed(),scene="closed")))
        self.assertFalse(any(self.feed(self.armed(),scene="unknown")))

    def test_visual_open_and_release_are_both_required(self):
        self.assertFalse(any(self.feed(self.armed(),suction=(True,False))))
        self.assertFalse(any(self.feed(self.armed(),suction=(False,True))))
        self.assertFalse(any(self.feed(self.armed(),target_off=False)))
        self.assertFalse(any(self.feed(self.armed(),distance=.13)))
        self.assertTrue(any(self.feed(self.armed())))

    def test_near_requires_both_open(self):
        self.assertFalse(any(self.feed(self.armed("near"),scene="far_open")))
        self.assertTrue(any(self.feed(self.armed("near"),scene="both_open")))

    def test_stable_hold_and_fresh_frames(self):
        g=self.armed()
        for i in range(30):
            self.assertFalse(g.update(6.+i*.1,100+i,"far_open",[i*.01]*14,[False,False],.01,True))
        self.assertFalse(g.update(10.,200,"far_open",[0.]*14,[False,False],.01,True))
        for i in range(30):
            self.assertFalse(g.update(10.1+i*.1,200,"far_open",[0.]*14,[False,False],.01,True))

    def test_minimum_stage_duration(self):
        self.assertFalse(any(self.feed(self.armed(),start=2.)))

    def test_unknown_scenes(self):
        d=SceneDetector(Path(__file__).with_name('task_detector.npz'))
        for frame in (np.zeros((480,640,3),np.uint8),np.full((480,640,3),255,np.uint8),
                      np.random.default_rng(0).integers(0,256,(480,640,3),dtype=np.uint8)):
            self.assertEqual(d.classify(frame)['scene'],'unknown')


if __name__=='__main__':unittest.main()
