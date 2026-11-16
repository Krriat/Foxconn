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
        self.trajectory_pub.publish(String(data=encode(dict(protocol='timed_chunk_v2', ready=True,
            last_error='', feedback_reader='drain_latest_v1'))))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((args.bundle/'config.yaml').read_text())
    config['driver_node_name'] = 'isolated_fake_plant'
    executor_node.Suction = FakeIO
    inference_log = (args.bundle/'sandbox_inference.log').open('w')
    inference_process = subprocess.Popen([sys.executable, '-u', str(args.bundle/'jetson_inference_bootstrap.py'),
        '--config', str(args.bundle/'config.yaml')], stdout=inference_log, stderr=subprocess.STDOUT)
    rclpy.init()
    controller = executor_node.Executor(config, args.bundle/f'sandbox_events_{time.time_ns()}.jsonl')
    plant = Plant(config, args.bundle/'fixtures')
    node = Node('isolated_chain_client')
    client = node.create_client(Trigger,'/robot/policy/start_sequence_brain')
    executor = MultiThreadedExecutor(num_threads=6)
    for n in (controller, plant, node): executor.add_node(n)
    thread = threading.Thread(target=executor.spin,daemon=True);thread.start()
    def wait(predicate, timeout=120):
        end = time.monotonic()+timeout
        while time.monotonic()<end:
            if predicate(): return
            if controller.phase == 'FAULT': raise AssertionError(controller.reason)
            time.sleep(.05)
        raise AssertionError(f'Timeout: {controller.phase}, {controller.reason}, brain={controller.brain_status}')
    try:
        wait(lambda: controller.brain_status.get('ready') and controller.inference_status.get('ready'))
        assert plant.initializes == 0 and plant.home_calls == {'left':0,'right':0}
        assert client.wait_for_service(timeout_sec=5)
        f = client.call_async(Trigger.Request());wait(f.done,10)
        assert f.result().success, f.result().message
        wait(lambda: controller.sequence_job and controller.sequence_job['result']=='complete',100)
        assert controller.phase == 'IDLE' and plant.home_calls == {'left':2,'right':2}
        events = [json.loads(line) for line in controller.log_path.read_text().splitlines()]
        chunks = [e for e in events if e['event']=='chunk']
        for stage in ('far','near'):
            assert any(e['policy_key']==stage and e['checkpoint_sha256']==config['act_stage_checkpoints'][stage]['sha256'] for e in chunks)
        completed = [e for e in events if e['event']=='stage_complete']
        assert [e['completion_source'] for e in completed] == ['qwen_lora','qwen_lora']
        result = dict(passed=True, domain=77, real_hardware_commands=0,
            real_jetson_act_weights={s:config['act_stage_checkpoints'][s]['sha256'] for s in ('far','near')},
            home_calls=plant.home_calls, stages=controller.sequence_job['completed_stages'],
            accepted_chunks=controller.accepted, rejected_chunks=controller.rejected,
            brain_adapter_sha256=config['brain_adapter_sha256'],
            events=str(controller.log_path), inference_host=controller.inference_status.get('host'),
            brain_host=controller.brain_status.get('host'),
            completion_sources=[e['completion_source'] for e in completed])
        (args.bundle/'sandbox_report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    finally:
        inference_process.terminate()
        inference_process.wait(timeout=30)
        inference_log.close()
        executor.shutdown();thread.join(timeout=3)
        if controller.suction:controller.suction.close()
        controller.log.close()
        for n in (controller,plant,node):n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
