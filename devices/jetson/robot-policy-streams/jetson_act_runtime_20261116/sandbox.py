"""Real executor + real dual ACT inference + remote real LoRA; fake plant only.

Private ROS Domain 77 on LAN. Actual robot services stay in Domain 1.
Suction is replaced before Executor construction; no SDK is instantiated.
"""
import os
assert os.environ.get('ROS_DOMAIN_ID') == '77'
assert os.environ.get('ROS_LOCALHOST_ONLY') == '0'
import argparse
import sys
sys.path.append('/usr/lib/python3/dist-packages')
import json
from pathlib import Path
import threading
import time
import subprocess

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from rokae_interfaces.msg import DualArmServoJ
import yaml

import executor_node
from test_executor_ros import FakePlant
from protocol import encode


class FakeIO:
    def __init__(self, config, initial):
        self.verified = list(initial)
    def request(self, run, target): self.verified = list(target)
    def suspend(self): pass
    def release_left(self, run): self.verified[0] = False
    def close(self): pass
    def state(self): return dict(verified=self.verified, pending_operation=False, settled=True, error='')


class Plant(FakePlant):
    def __init__(self, config, fixture_root):
        super().__init__(config)
        self.destroy_publisher(self.inference_pub)
        self.destroy_publisher(self.action_pub)
        self.validated = None
        self.test_status=self.create_publisher(String,'/robot/test/plant_status',1)
        for subscription in self.subs:
            self.destroy_subscription(subscription)
        self.subs = [self.create_subscription(DualArmServoJ, '/robot/upperlimb/servoj/dual_arm',
            self.servo, qos_profile_sensor_data, callback_group=self.group),
            self.create_subscription(String, '/robot/policy/status',
                lambda msg: setattr(self, 'policy_context', json.loads(msg.data)), 1, callback_group=self.group)]
        for timer in list(self.timers):
            self.destroy_timer(timer)
        self.create_timer(.02, self.feedback, callback_group=self.group)
        self.create_timer(.05, self.actions, callback_group=self.group)
        self.subs.append(self.create_subscription(String, '/robot/policy/validated_chunk',
            lambda msg: setattr(self, 'validated', json.loads(msg.data)), qos_profile_sensor_data, callback_group=self.group))
        self.camera_pubs = {c: self.create_publisher(CompressedImage,
            f'/robot/sensors/cameras/{c}/color/image_raw/compressed', 1) for c in ('head','left_wrist','right_wrist')}
        self.fixtures = {s: {c: (fixture_root/s/(c+'.jpg')).read_bytes() for c in self.camera_pubs}
                         for s in ('far','near')}
        self.create_timer(.05, self.cameras)
    def feedback(self):
        self.test_status.publish(String(data=encode(dict(home_calls=self.home_calls,initializes=self.initializes,enabled=self.enabled))))
        self.trajectory_pub.publish(String(data=encode(dict(protocol='timed_chunk_v2', ready=True,
            last_error='', feedback_reader='drain_latest_v1',continuous_action_clock_v1=True,
            contact_slow_1p2_v1=True,execution_time_scaling='per_observation_v1',
            maximum_execution_speed_scale=1.5,maximum_joint_acceleration_rad_s2=2.25))))
        for side, offset in (('left',0),('right',7)):
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = [f'{side}_joint_{i}' for i in range(1,8)]
            message.position = self.q[offset:offset+7]
            self.joint_pubs[side].publish(message)
            self.suction_pubs[side].publish(Bool(data=False))
    def cameras(self):
        stage = (self.policy_context.get('task_sequence') or {}).get('stage','far')
        for camera, publisher in self.camera_pubs.items():
            message = CompressedImage()
            message.header.stamp = self.get_clock().now().to_msg()
            message.format = 'jpeg'
            message.data = self.fixtures[stage][camera]
            publisher.publish(message)
    def actions(self):
        # Only a trace is simulated; action chunks come from the actual ACT GPU.
        if not self.validated or self.validated['run_id'] != self.policy_context.get('run_id'):
            return
        if self.policy_context.get('phase') in ('STARTING_SERVO','RUNNING','PREVIEW'):
            self.trace_pub.publish(String(data=encode(dict(run_id=self.validated['run_id'], active=True,
                # Simulate the native driver's acknowledgment bit. This process
                # has only ROS Domain-77 fakes; it never connects a robot SDK.
                command=self.q, fused_target=self.validated['actions'][0], hardware_commands_sent=True))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--role',choices=('coordinator','plant','controller'),default='coordinator')
    args=parser.parse_args();cfg=yaml.safe_load((args.bundle/'config.yaml').read_text())
    cfg['driver_node_name']='isolated_fake_plant'
    assert cfg['execution_speed_scale']==1.0 and cfg['execution_speed_profile']=='uniform' and cfg['max_joint_acceleration_rad_s2']==1.0
    assert cfg['act_backend']=='tensorrt'
    if args.role in ('plant','controller'):
        rclpy.init()
        if args.role=='plant':node=Plant(cfg,args.bundle/'fixtures')
        else:
            executor_node.Suction=FakeIO
            node=executor_node.Executor(cfg,args.bundle/f'sandbox_events_{time.time_ns()}.jsonl')
        engine=MultiThreadedExecutor(num_threads=4) if args.role=='plant' else None
        try:
            if engine:engine.add_node(node);engine.spin()
            else:rclpy.spin(node)
        except KeyboardInterrupt:pass
        finally:
            if engine:engine.shutdown()
            if args.role=='controller':
                if node.suction:node.suction.close()
                node.log.close()
            node.destroy_node()
            if rclpy.ok():rclpy.shutdown()
        return
    children=[];logs=[];rclpy.init();node=Node('isolated_chain_client')
    latest={};chunks=[];subs=[]
    for key in ('status','inference_status','brain_status'):
        subs.append(node.create_subscription(String,'/robot/policy/'+key,lambda m,k=key:latest.update({k:json.loads(m.data)}),qos_profile_sensor_data))
    subs.append(node.create_subscription(String,'/robot/test/plant_status',lambda m:latest.update(plant=json.loads(m.data)),qos_profile_sensor_data))
    subs.append(node.create_subscription(String,'/robot/policy/action_chunk',lambda m:chunks.append(json.loads(m.data)),qos_profile_sensor_data))
    client=node.create_client(Trigger,'/robot/policy/start_sequence_brain')
    def wait(pred,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.02)
            if pred():return
            if latest.get('status',{}).get('phase')=='FAULT':raise AssertionError(latest['status']['reason'])
            for child in children:
                if child.poll() is not None:raise RuntimeError('isolated child exited: '+str(child.args))
        raise TimeoutError(str({k:latest.get(k) for k in ('status','brain_status','inference_status')}))
    try:
        for role in ('plant','controller','inference'):
            log=(args.bundle/('sandbox_'+role+'.log')).open('w');logs.append(log)
            cmd=([sys.executable,'-u',str(args.bundle/'jetson_inference_bootstrap.py'),'--config',str(args.bundle/'config.yaml')] if role=='inference' else [sys.executable,'-u',str(Path(__file__)),'--bundle',str(args.bundle),'--role',role])
            children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
        wait(lambda:all(latest.get(k,{}).get('ready') for k in ('brain_status','inference_status')) and 'plant' in latest,120)
        assert latest['plant']['home_calls']=={'left':0,'right':0} and latest['plant']['initializes']==0
        assert client.wait_for_service(timeout_sec=5)
        future=client.call_async(Trigger.Request());wait(future.done,10);assert future.result().success,future.result().message
        wait(lambda:(latest.get('status',{}).get('task_sequence') or {}).get('result')=='complete' and latest['status']['phase']=='IDLE',100)
        wait(lambda:latest['plant']['home_calls']=={'left':2,'right':2} and not latest['plant']['enabled'],5)
        events=sorted(args.bundle.glob('sandbox_events_*.jsonl'),key=lambda p:p.stat().st_mtime)[-1]
        rows=[json.loads(line) for line in events.read_text().splitlines()]
        completed=[r for r in rows if r['event']=='stage_complete']
        accepted=[r for r in rows if r['event']=='chunk']
        homes=[r for r in rows if r['event']=='home_verified']
        assert len(homes)==2 and [r['completion_source'] for r in completed]==['qwen_lora','qwen_lora']
        for stage in ('far','near'):
            assert any(r.get('policy_key')==stage and r['checkpoint_sha256']==cfg['act_stage_checkpoints'][stage]['sha256'] for r in accepted)
        result=dict(passed=True,domain=77,separate_processes=True,real_hardware_commands=0,simulated_driver_only=True,backend=latest['inference_status']['backend'],real_jetson_act_weights={s:cfg['act_stage_checkpoints'][s]['sha256'] for s in ('far','near')},home_calls=latest['plant']['home_calls'],verified_home_events=len(homes),stages=latest['status']['task_sequence']['completed_stages'],brain_adapter_sha256=cfg['brain_adapter_sha256'],inference_host=latest['inference_status']['host'],brain_host=latest['brain_status']['host'],execution_speed_scale=1.0,execution_speed_profile='uniform',max_joint_acceleration_rad_s2=1.0,completion_sources=[r['completion_source'] for r in completed],events=str(events))
        (args.bundle/'sandbox_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    finally:
        for child in children:
            if child.poll() is None:child.send_signal(__import__('signal').SIGINT)
        for child in children:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for log in logs:log.close()
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
