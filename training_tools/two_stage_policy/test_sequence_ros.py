#!/usr/bin/env python3
"""Manual stage sequencing with no box monitor, SDK, or real robot; Domain 77 only."""
import os
assert os.environ.get('ROS_DOMAIN_ID')=='77'
assert os.environ.get('ROS_LOCALHOST_ONLY')=='1'
import json
from pathlib import Path
import tempfile
import threading
import time
import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
import executor_node
from test_executor_ros import FakePlant
from protocol import PREFIX

class FakeIO:
    def __init__(self,config,initial):self.verified=list(initial);self.pending=False
    def request(self,run,target):self.verified=list(target)
    def suspend(self):pass
    def release_left(self,run):self.verified[0]=False
    def close(self):pass
    def state(self):return dict(verified=self.verified,pending_operation=self.pending,settled=not self.pending,error='')

def main():
    executor_node.Suction=FakeIO
    config=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    config.update(enable_motion=True,suction_enabled=True,max_round_seconds=None,driver_node_name='isolated_fake_plant')
    rclpy.init();logs=Path(tempfile.mkdtemp(prefix='manual-sequence-domain77-'))
    controller=executor_node.Executor(config,logs/'events.jsonl');plant=FakePlant(config)
    plant.predicted_suction=[0.,0.]
    client=Node('manual_sequence_test_client');executor=MultiThreadedExecutor(num_threads=6)
    for node in (controller,plant,client):executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    clients={n:client.create_client(Trigger,PREFIX+'/'+n) for n in ('start','start_sequence','start_near','finish_far','finish_near','stop')}
    def wait(pred,seconds=8):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if pred():return
            time.sleep(.01)
        raise AssertionError(f'{controller.phase}: {controller.reason}; sequence={controller.sequence_job}')
    def call(name,success=True):
        assert clients[name].wait_for_service(timeout_sec=2)
        f=clients[name].call_async(Trigger.Request());wait(f.done)
        assert f.result().success==success,f.result().message
    passed=[];home=config['home_left']+config['home_right']
    try:
        wait(lambda:controller.inference_status.get('ready') and len(controller.joints)==2)
        assert not any(name=='/robot/policy/task_scene' for name,types in controller.get_subscriber_names_and_types_by_node(controller.get_name(),'/'))
        call('finish_far',False);call('finish_near',False)
        passed.append('no box monitor subscription or classifier dependency; completion rejected in IDLE')
        plant.apply_home=False;plant.q=[x+.04 for x in home]
        call('start_sequence');wait(lambda:controller.phase=='VERIFY_HOME')
        call('finish_far',False);time.sleep(.3)
        assert controller.phase=='VERIFY_HOME' and controller.accepted==0
        assert not any(k=='servo' for k,t in plant.trace)
        passed.append('successful move replies cannot bypass fresh measured home; completion rejected during home')
        plant.apply_home=True;plant.q=list(home)
        wait(lambda:controller.phase=='RUNNING')
        assert plant.home_calls=={'left':1,'right':1}
        assert controller.home_verified_run_id==controller.run_id and controller.home_verified_time_ns>0
        call('finish_near',False)
        # No task_scene is published at any point. Neither elapsed time nor
        # released suction automatically marks the stage complete.
        time.sleep(1.)
        assert controller.phase=='RUNNING' and controller.sequence_job['stage']=='far'
        passed.append('far policy runs without any box messages and waits for explicit far completion')
        # The fake ServoJ does not simulate a full task trajectory. Move its
        # measured endpoint away from home before testing the second home gate.
        plant.held_endpoint=[x+.04 for x in home]
        wait(lambda:abs(controller.joints['left'][0][0]-home[0])>.03)
        first=controller.run_id;plant.apply_home=False
        call('finish_far');call('finish_far',False)
        wait(lambda:controller.phase=='VERIFY_HOME')
        assert controller.sequence_job['stage']=='near' and controller.run_id!=first
        assert controller.sequence_job['completed_stages']==['far']
        accepted=controller.accepted;time.sleep(.3)
        assert controller.phase=='VERIFY_HOME' and controller.accepted==accepted
        assert controller.action is None and controller.home_verified_run_id==''
        assert plant.home_calls=={'left':2,'right':2}
        passed.append('far completion stops, releases, resets UUID, and requires both arms home again before near')
        plant.apply_home=True;plant.q=list(home)
        wait(lambda:controller.phase=='RUNNING')
        call('finish_far',False)
        assert controller.sequence_job['pending'] is None
        passed.append('duplicate or late far completion cannot finish near')
        plant.stale_run=True;wait(lambda:controller.rejected>0);plant.stale_run=False
        call('finish_near');call('finish_near',False)
        wait(lambda:controller.phase=='IDLE')
        assert controller.sequence_job['result']=='complete'
        assert controller.sequence_job['completed_stages']==['far','near']
        assert controller.suction.state()['verified']==[False,False]
        homes=dict(plant.home_calls);time.sleep(.3)
        assert plant.home_calls==homes and not plant.enabled
        passed.append('manual near completion stops and releases suction with no third stage')
        events=[json.loads(line) for line in (logs/'events.jsonl').read_text().splitlines()]
        chunks=[e for e in events if e['event']=='chunk']
        assert {e['policy_key'] for e in chunks}=={'far','near'}
        for e in chunks:
            assert e['checkpoint_sha256']==config['act_stage_checkpoints'][e['policy_key']]['sha256']
            assert any(v['event']=='home_verified' and v['run_id']==e['run_id'] and v['time_ns']<e['time_ns'] for v in events)
        assert all(e['completion_source']=='operator_service' for e in events if e['event']=='stage_complete')
        assert not any(e.get('phase')=='VERIFY_SCENE' for e in events)
        passed.append('every stage chunk follows its own verified home and uses its correct ACT weights')
        call('start_near');wait(lambda:controller.phase=='RUNNING')
        assert plant.home_calls=={'left':3,'right':3}
        plant.wrong_checkpoint=True;wait(lambda:controller.phase=='FAULT')
        assert 'unexpected model weights' in controller.reason
        call('finish_near',False);plant.wrong_checkpoint=False;call('stop');wait(lambda:controller.phase=='IDLE')
        passed.append('near-only start also homes; wrong weights fault; no completion accepted in fault')
        call('start_sequence');wait(lambda:controller.phase=='RUNNING')
        call('finish_far');wait(lambda:controller.phase=='STAGE_RELEASE')
        controller.suction.pending=True;homes=dict(plant.home_calls);time.sleep(.6)
        assert controller.phase=='STAGE_RELEASE' and plant.home_calls==homes
        call('stop');controller.suction.pending=False;wait(lambda:controller.phase=='IDLE')
        assert controller.sequence_job['result']=='cancelled'
        passed.append('unsettled suction release blocks homing and manual stop cancels next stage')
        call('start_sequence');wait(lambda:controller.phase=='RUNNING')
        controller.round_start=time.monotonic()-91
        wait(lambda:controller.phase=='FAULT')
        assert controller.sequence_job['result']=='fault' and 'manual completion' in controller.reason
        assert controller.sequence_job['pending'] is None
        passed.append('90-second timeout stops/cancels instead of automatically advancing')
        call('stop');wait(lambda:controller.phase=='IDLE')
        call('start');wait(lambda:controller.phase=='RUNNING')
        call('finish_far',False);call('finish_near',False)
        call('stop');wait(lambda:controller.phase=='IDLE')
        passed.append('plain single-round start cannot accidentally enter the sequence')
        print(json.dumps(dict(passed=passed,log=str(logs)),indent=2),flush=True)
    finally:
        if controller.phase not in ('IDLE','FAULT'):
            controller.request_stop('test cleanup');time.sleep(.5)
        executor.shutdown();thread.join(timeout=3)
        for node in (client,controller,plant):node.destroy_node()
        controller.log.close();rclpy.shutdown()

if __name__=='__main__':main()
