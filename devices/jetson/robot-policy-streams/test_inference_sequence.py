"""Same-round transient gaps must not reset the chunk anti-replay sequence."""
import json
from types import SimpleNamespace
import threading
import time
import unittest
import numpy as np
from inference_node import Inference


class SequenceTests(unittest.TestCase):
    def check_gap(self, missing_observation):
        loop=[0];published=[]
        def is_set():
            loop[0]+=1
            return loop[0]>4
        stop=SimpleNamespace(is_set=is_set,wait=lambda _:False)
        def snapshot():
            if loop[0]==3 and missing_observation:return None
            return {'source_stamp':1000.+loop[0]*.05,'images':[],
                    'state':np.zeros(16,dtype=np.float32),'source_stamps_ns':{},
                    'source_skew_ms':0.,'latest_age_ms':0.}
        context={'phase':'PREVIEW','run_id':'one-round','heartbeat':'valid','model':'act'}
        def active():
            return None if loop[0]==3 and not missing_observation else context
        obj=SimpleNamespace(stop_event=stop,snapshot=snapshot,active_context=active,
            predict=lambda _:np.zeros((50,16)),reset=lambda:None,lock=threading.Lock(),
            status={'actions_published':0,'chunks_predicted':0},config={'model':'act','action_hz':20},
            identity='test',weights_sha='test',head_event=SimpleNamespace(wait=lambda _:True,clear=lambda:None),
            snapshot_failure='test_temporary_input_gap',executor_context=context,
            context_received=time.monotonic(),last_pause_reason=None,buffers={},
            actions=SimpleNamespace(publish=lambda message:published.append(json.loads(message.data))))
        Inference.run(obj)
        self.assertEqual([m['sequence'] for m in published],[1,2])
        self.assertTrue(all(m['run_id']=='one-round' for m in published))

    def test_temporary_input_gap_preserves_sequence(self):self.check_gap(True)
    def test_temporary_context_gap_preserves_sequence(self):self.check_gap(False)
