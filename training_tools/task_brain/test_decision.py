"""Regression checks for asynchronous model results and ACT stage isolation."""
import unittest
from training_tools.task_brain.decision import Gate,Evidence,parse_answer
from training_tools.two_stage_policy.brain_protocol import validate_brain_completion


def context(run="r1",stage="far",phase="RUNNING"):
    return {"phase":phase,"mode":"live","run_id":run,
            "task_sequence":{"stage":stage,"id":"seq1","result":"running","pending":None}}


class Checks(unittest.TestCase):
    def setUp(self):
        self.g=Gate();self.c=context();self.g.observe_context(self.c,0.)

    def send(self,t,*,value=None,run="r1",stage="far",stamp=None,age=0.,context_age=.02,c=None):
        evidence=Evidence(run,stage,int(t*1e9) if stamp is None else stamp,t-age,
                          value or {"scene":"far_open","status":"complete"})
        return self.g.update(evidence,c or self.c,now=t,context_age=context_age)

    def test_minimum_time_then_hold(self):
        for t in (1.,2.,3.,4.,5.,5.5,6.):self.assertIsNone(self.send(t))
        self.assertEqual(self.send(6.6)["service"],"finish_far")

    def test_running_and_unknown_reset(self):
        self.send(5);self.send(6)
        self.send(6.5,value={"scene":"unknown","status":"uncertain"})
        self.assertIsNone(self.send(7));self.assertIsNone(self.send(8))
        self.assertEqual(self.send(9)["service"],"finish_far")

    def test_stale_and_reordered_never_switch(self):
        self.send(5);self.send(6)
        self.assertIsNone(self.send(7,age=3))
        self.assertIsNone(self.send(8,stamp=1))
        self.assertIsNone(self.send(9,context_age=.5))
        self.assertIsNone(self.send(10,run="old"))
        self.assertFalse(self.g.samples)

    def test_old_far_result_cannot_finish_near(self):
        self.send(5);self.send(6)
        new=context("r2","near");self.g.observe_context(new,7)
        self.assertIsNone(self.send(8,c=new));self.assertFalse(self.g.samples)
        for t in (12,13):self.assertIsNone(self.send(t,c=new,run="r2",stage="near",value={"scene":"both_open","status":"complete"}))
        d=self.send(14,c=new,run="r2",stage="near",value={"scene":"both_open","status":"complete"})
        self.assertEqual(d["service"],"finish_near")

    def test_once_and_fault_gates(self):
        self.send(5);self.send(6);d=self.send(7);self.g.mark_dispatched(d)
        self.assertIsNone(self.send(8))
        self.assertIsNone(self.send(9,c=context(phase="FAULT")))
        self.assertIsNone(self.send(10,c=context(phase="HOMING")))

    def test_invalid_and_conflicting_output(self):
        for value in ({"scene":"closed","status":"complete"},
                      {"scene":"far_open","status":"complete","command":"move"},
                      {"scene":"unknown","status":"complete"}):
            with self.assertRaises(ValueError):parse_answer(value,"far")
        with self.assertRaises(ValueError):parse_answer({"scene":"far_open","status":"complete"},"near")

    def test_paused_and_cancelled_sequence(self):
        for mutation in ({"mode":"preview"},{"phase":"IDLE"}):
            c=context();c.update(mutation);self.assertIsNone(self.send(7,c=c))
        c=context();c["task_sequence"]["result"]="cancelled"
        self.assertIsNone(self.send(9,c=c))

    def test_visual_completion_cannot_interrupt_unfinished_manipulation(self):
        self.c['require_manipulation_sweep_for_brain']=True
        self.c['manipulation_progress']={'stage':'far','completion_ready':False}
        for t in range(5,20):self.assertIsNone(self.send(t))
        self.assertFalse(self.g.samples)
        self.c['manipulation_progress']={'stage':'far','completion_ready':True}
        self.assertIsNone(self.send(20));self.assertIsNone(self.send(21))
        self.assertEqual(self.send(22)['service'],'finish_far')

    def test_missing_or_other_stage_progress_never_authorizes_switch(self):
        self.c['require_manipulation_sweep_for_brain']=True
        self.assertIsNone(self.send(5))
        self.c['manipulation_progress']={'stage':'near','completion_ready':True}
        for t in (6,7,8):self.assertIsNone(self.send(t))

    def test_executor_binds_run_stage_sequence_model_and_fresh_image(self):
        now=10_000_000_000
        c=context();c["task_sequence"]["completion_mode"]="qwen_lora"
        cfg={"brain_adapter_sha256":"a"*64}
        value={"run_id":"r1","stage":"far","sequence_id":"seq1","adapter_sha256":"a"*64,
               "image_time_ns":now-1_000_000_000,"decision_time_ns":now-100_000_000,"scene":"far_open"}
        def check(v,ct=c,elapsed=6,home=now-3_000_000_000):
            return validate_brain_completion(v,ct,cfg,now_ns=now,elapsed_seconds=elapsed,home_after_ns=home)
        self.assertEqual(check(value),"far")
        for key,bad in (("run_id","old"),("stage","near"),("sequence_id","old"),
                        ("adapter_sha256","b"*64),("image_time_ns",now-3_000_000_000),
                        ("decision_time_ns",now+1),("scene","closed")):
            with self.assertRaises(ValueError):check(dict(value,**{key:bad}))
        with self.assertRaises(ValueError):check(value,elapsed=4)
        with self.assertRaises(ValueError):check(value,home=now-500_000_000)
        c["task_sequence"]["completion_mode"]="manual"
        with self.assertRaises(ValueError):check(value)


if __name__=="__main__":unittest.main()
