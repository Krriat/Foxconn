"""Real inference loop with synthetic models; never initializes ROS."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import patch
import threading
import time
import unittest
import numpy as np
import yaml
from inference_node import Inference
from stage_policy import checkpoint_map, selected_policy, stage_specs

CONFIG = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())

class StageTests(unittest.TestCase):
    def test_cyclic_gc_runs_only_after_active_phase(self):
        obj=SimpleNamespace(gc_initially_enabled=True,gc_suspended=False,
                            executor_context={'phase':'RUNNING'},mark_progress=lambda *_:None)
        with patch('inference_node.gc') as collector:
            Inference.manage_gc(obj)
            self.assertTrue(obj.gc_suspended)
            collector.disable.assert_called_once()
            collector.collect.assert_not_called()
            Inference.manage_gc(obj)
            collector.disable.assert_called_once()
            obj.executor_context['phase']='IDLE'
            Inference.manage_gc(obj)
            self.assertFalse(obj.gc_suspended)
            collector.enable.assert_called_once()
            collector.collect.assert_called_once()
            Inference.manage_gc(obj)
            collector.collect.assert_called_once()

    def test_external_gc_disable_is_preserved(self):
        obj=SimpleNamespace(gc_initially_enabled=False,gc_suspended=False,
                            executor_context={'phase':'RUNNING'},mark_progress=lambda *_:None)
        with patch('inference_node.gc') as collector:
            Inference.manage_gc(obj)
            obj.executor_context['phase']='IDLE'
            Inference.manage_gc(obj)
            self.assertFalse(obj.gc_suspended)
            collector.enable.assert_not_called()
            collector.collect.assert_not_called()
    def test_bad_stage_config(self):
        for bad in ({'near': CONFIG['act_stage_checkpoints']['near']},
                    dict(CONFIG['act_stage_checkpoints'], far={'path':'relative','sha256':'x'})):
            config=copy.deepcopy(CONFIG);config['act_stage_checkpoints']=bad
            with self.assertRaises(ValueError):stage_specs(config)

    def context(self, stage, run='round', phase='RUNNING'):
        sequence={'stage':stage,'result':'running'} if stage else None
        key,sha=selected_policy(CONFIG,sequence)
        return dict(phase=phase,mode='live',model='act',run_id=run,heartbeat='valid',
                    task_sequence=sequence,policy_key=key,expected_checkpoint_sha256=sha,home_verified_run_id=run,
                    execution_speed_scale=CONFIG.get('execution_speed_scale',1.))

    def test_gate(self):
        obj=SimpleNamespace(lock=threading.Lock(),config=CONFIG,context_received=time.monotonic())
        for stage in ('far','near',None):
            good=self.context(stage)
            for patch,allowed in (({},True),({'phase':'HOMING'},False),({'phase':'VERIFY_HOME'},False),
                    ({'phase':'VERIFY_SCENE'},False),({'home_verified_run_id':'old'},False),
                    ({'expected_checkpoint_sha256':'wrong'},False),({'policy_key':'wrong'},False)):
                obj.executor_context=dict(good,**patch)
                self.assertEqual(Inference.active_context(obj) is not None,allowed)
        obj.context_received=time.monotonic()-1
        self.assertIsNone(Inference.active_context(obj))

    def loop(self, phases, discard_during_predict=False):
        step=[0];sent=[];calls=[];resets=[];context=[None]
        def stop():
            step[0]+=1
            if step[0]>len(phases):return True
            context[0]=phases[step[0]-1]
            return False
        def snap():
            return dict(source_stamp=1000.+step[0]*.05,images=[],state=np.zeros(16),
                        source_stamps_ns={},source_skew_ms=0.,latest_age_ms=0.)
        def predict(key):
            def fn(_):
                calls.append((step[0],key))
                if discard_during_predict and step[0]>1:context[0]=None
                return np.full((50,16),1 if key=='far' else 2)
            return fn
        bank={key:(predict(key),lambda key=key:resets.append((step[0],key)),sha)
              for key,sha in checkpoint_map(CONFIG).items() if key!='default'}
        bank['default']=bank['far']
        obj=SimpleNamespace(stop_event=SimpleNamespace(is_set=stop,wait=lambda _:False),snapshot=snap,
            active_context=lambda:context[0],predict=bank['default'][0],reset=bank['default'][1],
            weights_sha=bank['default'][2],policies=bank,lock=threading.Lock(),config=CONFIG,
            status=dict(actions_published=0,chunks_predicted=0),identity='fake',anchor_joints=False,
            head_event=SimpleNamespace(wait=lambda _:True,clear=lambda:None),last_pause_reason=None,
            snapshot_failure='',executor_context={'phase':'IDLE'},context_received=time.monotonic(),buffers={},
            actions=SimpleNamespace(publish=lambda m:sent.append(json.loads(m.data))))
        obj.last_gate_log = 0.
        obj.manage_gc = lambda: None
        for name in ('mark_progress', 'log_gate'):
            setattr(obj, name, MethodType(getattr(Inference, name), obj))
        Inference.run(obj)
        return sent,calls,resets

    def test_round_switch_and_same_round_gap(self):
        far=self.context('far','far-round');near=self.context('near','near-round')
        sent,calls,resets=self.loop([None,far,None,far,near,near])
        self.assertEqual([m['sequence'] for m in sent],[1,2,1,2])
        self.assertEqual([m['policy_key'] for m in sent],['far','far','near','near'])
        self.assertTrue(all(m['execution_speed_scale']==1.5 and m['step_ns']==50_000_000 for m in sent))
        self.assertEqual([m['actions'][0][0] for m in sent],[1,1,2,2])
        self.assertEqual([(m['run_id'],m['checkpoint_sha256']) for m in sent],
            [('far-round',checkpoint_map(CONFIG)['far'])]*2+[('near-round',checkpoint_map(CONFIG)['near'])]*2)
        self.assertEqual([x for x in resets if x[0]>1],[(2,'far'),(5,'near')])

    def test_same_uuid_cannot_switch_checkpoint(self):
        sent,_,_=self.loop([None,self.context('far'),self.context('near')])
        self.assertEqual([m['policy_key'] for m in sent],['far'])

    def test_inflight_prediction_dropped_after_stop(self):
        sent,_,_=self.loop([None,self.context('near')],True)
        self.assertEqual(sent,[])

    def test_only_images_after_verified_home_can_publish(self):
        context=self.context('far')
        context['policy_observation_after_ns']=int(1000.125*1e9)
        sent,_,_=self.loop([None,context,context])
        self.assertEqual(len(sent),1)
        self.assertGreaterEqual(sent[0]['observation_time_ns'],context['policy_observation_after_ns'])

if __name__=='__main__':unittest.main()
