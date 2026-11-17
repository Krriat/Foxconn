"""No ROS initialization: timing validation, inference identity and old defaults."""
import copy
import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml
from inference_node import Inference
from protocol import validate_config, validate_chunk
from stage_policy import selected_policy

CONFIG=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())

class SpeedProtocolTests(unittest.TestCase):
    def packet(self):
        now=1_000_000_000
        return dict(schema='dual_arm_suction_chunk_v2',run_id='r',publisher_id='p',sequence=1,
                    model='act',checkpoint_sha256=CONFIG['checkpoint_sha256']['act'],action_hz=20,
                    max_input_receive_age_ms=60.,step_ns=50_000_000,execution_speed_scale=1.5,
                    execution_speed_profile='contact_slow_v1',
                    observation_time_ns=now-70_000_000,inference_start_ns=now-40_000_000,
                    inference_end_ns=now-25_000_000,send_time_ns=now-20_000_000,
                    actions=[CONFIG['home_left']+CONFIG['home_right']+[.8,0.] for _ in range(50)])
    def validate(self,msg,cfg=None,now=1_000_000_000):
        return validate_chunk(msg,'r','p',0,CONFIG if cfg is None else cfg,now)
    def test_new_config_and_raw_training_step(self):
        validate_config(CONFIG);msg=self.packet();self.assertEqual(len(self.validate(msg)),50)
        self.assertEqual(msg['step_ns'],50_000_000)
        for name,value in [('execution_speed_scale',1.50001),('execution_speed_scale',True),
                           ('execution_speed_scale',float('nan')),('execution_speed_scale',.9),
                           ('max_joint_acceleration_rad_s2',2.25001),('max_joint_speed_rad_s',.50001)]:
            cfg=copy.deepcopy(CONFIG);cfg[name]=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):validate_config(cfg)
    def test_no_mixed_speed_publisher(self):
        for value in (1.,True,float('nan'),2.):
            msg=self.packet();msg['execution_speed_scale']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.validate(msg)
        msg=self.packet();del msg['execution_speed_scale']
        with self.assertRaises(ValueError):self.validate(msg)
        msg=self.packet();msg['execution_speed_profile']='uniform'
        with self.assertRaises(ValueError):self.validate(msg)
    def test_real_time_staleness_not_extended_by_speed(self):
        for key,value in [('observation_time_ns',699_000_000),('send_time_ns',1_011_000_000),
                          ('step_ns',33_333_333),('run_id','old')]:
            msg=self.packet();msg[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(msg)
        with self.assertRaises(ValueError):self.validate(self.packet(),now=1_300_000_000)
    def test_missing_scale_is_legacy_one_only(self):
        cfg=copy.deepcopy(CONFIG);del cfg['execution_speed_scale'];del cfg['execution_speed_profile'];cfg['max_joint_acceleration_rad_s2']=1.
        validate_config(cfg);msg=self.packet();del msg['execution_speed_scale'];del msg['execution_speed_profile']
        self.assertEqual(len(self.validate(msg,cfg)),50)
    def test_inference_waits_for_matching_executor_speed_and_verified_home(self):
        key,sha=selected_policy(CONFIG,None)
        context=dict(phase='RUNNING',mode='live',model='act',run_id='r',home_verified_run_id='r',
                     policy_key=key,expected_checkpoint_sha256=sha,execution_speed_scale=1.5,
                     execution_speed_profile='contact_slow_v1')
        obj=SimpleNamespace(config=CONFIG,lock=threading.Lock(),executor_context=context,context_received=time.monotonic())
        self.assertIsNotNone(Inference.active_context(obj))
        for patch in ({'execution_speed_scale':1.},{'home_verified_run_id':'old'}):
            obj.executor_context=dict(context,**patch);self.assertIsNone(Inference.active_context(obj))

if __name__=='__main__':unittest.main()
