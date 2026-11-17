"""Exercise the actual executor chunk callback with recorded valid packets.

AST loading avoids constructing a ROS node/SDK. Protocol validation and the
production callback are unchanged, while publishers and home feedback are fake.
"""
import ast
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
import yaml

from far_arm_coordination import FarArmCoordination
from protocol import ACTIVE_PHASES,encode,validate_chunk
from stage_policy import action_config

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
tree=ast.parse((HERE/'executor_node.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Executor')
callback=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='on_action')
scope=dict(time=time,json=json,ACTIVE_PHASES=ACTIVE_PHASES,action_config=action_config,
           validate_chunk=validate_chunk,encode=encode,String=SimpleNamespace)
exec(compile(ast.fix_missing_locations(ast.Module(body=[callback],type_ignores=[])),str(HERE/'executor_node.py'),'exec'),scope)


class Checks(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((HERE/'config.yaml').read_text());self.messages=[];self.faults=[]
        path=ROOT/'training_verification/guarded_act_retest_20261117_094323_02/live01/ros_events.jsonl'
        with path.open() as f:
            self.packet=next(r['value'] for r in map(json.loads,f) if r['topic']=='action_chunk')
        self.packet=json.loads(json.dumps(self.packet));self.packet['run_id']='fake-coordination-run'
        self.c=SimpleNamespace(phase='RUNNING',mode='live',run_id='fake-coordination-run',rejected=0,
            home_verified_run_id='fake-coordination-run',config=self.cfg,
            sequence_job={'stage':'far','result':'running','completion_mode':'qwen_lora'},
            inference_status={'publisher_id':self.packet['publisher_id']},publisher_id=None,last_sequence=0,
            heartbeats={},accepted=0,progress_run_id='',suction_release_guard=SimpleNamespace(left_lift_seen=False),
            far_arm_coordination=FarArmCoordination(),
            validated_pub=SimpleNamespace(publish=lambda m:self.messages.append(json.loads(m.data))),
            event=lambda *a,**k:None,feedback=lambda:self.cfg['home_left']+self.cfg['home_right'],
            request_stop=lambda reason,**k:self.faults.append(reason))

    def send(self):
        now=time.time_ns();self.packet.update(observation_time_ns=now-170_000_000,
            inference_start_ns=now-160_000_000,inference_end_ns=now-60_000_000,send_time_ns=now-50_000_000)
        self.c.heartbeats[self.packet['executor_heartbeat']]=time.monotonic()
        scope['on_action'](self.c,SimpleNamespace(data=json.dumps(self.packet)))

    def test_auto_far_wire_holds_right_preserves_left_IO_and_raw_packet(self):
        original=json.loads(json.dumps(self.packet));self.send();self.assertFalse(self.faults)
        out=self.messages[-1]
        self.assertTrue(out['far_arm_coordination']['right_held'])
        for src,dst in zip(original['actions'],out['actions']):
            self.assertEqual(src[:7],dst[:7]);self.assertEqual(src[14:],dst[14:])
            self.assertEqual(dst[7:14],self.cfg['home_right'])
        self.assertEqual(self.packet['actions'],original['actions'])

    def test_lift_releases_actual_ACT_right_actions(self):
        self.send();self.c.suction_release_guard.left_lift_seen=True;self.c.progress_run_id=self.c.run_id
        self.packet['sequence']+=1;self.send();self.assertFalse(self.faults)
        self.assertEqual(self.messages[-1]['actions'],self.packet['actions'])

    def test_manual_sequence_wire_is_unchanged(self):
        self.c.sequence_job['completion_mode']='manual';self.send()
        self.assertFalse(self.faults);self.assertEqual(self.messages[-1]['actions'],self.packet['actions'])

    def test_missing_home_cannot_send_even_held_targets(self):
        self.c.home_verified_run_id='';self.send()
        self.assertTrue(self.faults);self.assertFalse(self.messages)

    def test_invalid_raw_right_target_is_rejected_before_coordination(self):
        self.packet['actions'][0][7]=100.;self.send()
        self.assertTrue(self.faults);self.assertFalse(self.messages)


if __name__=='__main__':unittest.main()
