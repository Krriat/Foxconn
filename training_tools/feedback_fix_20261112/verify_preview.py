#!/usr/bin/python3
"""Verify live data/model in preview only; never calls the motion start service."""
import json
import argparse
from pathlib import Path
import time
import numpy as np
import rclpy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from rclpy.qos import qos_profile_sensor_data

OUT=Path('/home/jj/code/ACT/training_verification/feedback_fix_20261112')
parser=argparse.ArgumentParser()
parser.add_argument('--seconds',type=float,default=8)
parser.add_argument('--output',type=Path,default=OUT/'preview_verification.json')
args=parser.parse_args()
rclpy.init();node=rclpy.create_node('latest_feedback_preview_verification')
latest={};chunks=[];traces=[]
def status(key,msg):latest[key]=json.loads(msg.data)
subscriptions=[node.create_subscription(String,'/robot/policy/'+key,lambda m,k=key:status(k,m),qos_profile_sensor_data)
               for key in ('status','inference_status','trajectory_status')]
subscriptions.append(node.create_subscription(String,'/robot/policy/action_chunk',lambda m:chunks.append(json.loads(m.data)),qos_profile_sensor_data))
subscriptions.append(node.create_subscription(String,'/robot/policy/servo_trace',lambda m:traces.append(json.loads(m.data)),qos_profile_sensor_data))
clients={key:node.create_client(Trigger,'/robot/policy/'+key) for key in ('preview','stop')}
def wait(predicate,seconds=8):
    end=time.monotonic()+seconds
    while not predicate() and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
    assert predicate(),str(latest)
def call(name):
    assert clients[name].wait_for_service(timeout_sec=5)
    future=clients[name].call_async(Trigger.Request());wait(future.done)
    assert future.result().success,future.result().message
    return future.result().message
started=False
try:
    wait(lambda: all(key in latest for key in ('status','inference_status','trajectory_status')) and latest['inference_status'].get('ready'))
    assert latest['status']['phase']=='IDLE'
    assert latest['trajectory_status']['feedback_reader']=='drain_latest_v1'
    started=True;message=call('preview')
    wait(lambda:latest['status']['phase']=='PREVIEW')
    run=latest['status']['run_id'];begin=time.monotonic()
    while time.monotonic()-begin<args.seconds:rclpy.spin_once(node,timeout_sec=.05)
finally:
    if started:
        call('stop');wait(lambda:latest['status']['phase']=='IDLE')
    node.destroy_node();rclpy.shutdown()
valid_chunks=[c for c in chunks if c.get('run_id')==run]
valid_traces=[t for t in traces if t.get('run_id')==run]
passed=(len(valid_chunks)>=100 and len(valid_traces)>=500 and
        all(len(c.get('observation_state',[]))==16 for c in valid_chunks) and
        all(not t['hardware_commands_sent'] for t in valid_traces) and latest['status']['servo_commands']==0)
result={'passed':passed,'mode':'real_camera_and_model_preview_no_hardware_motion',
        'run_id':run,'chunks':len(valid_chunks),'traces':len(valid_traces),'observation_state_logged':True,
        'hardware_commands_sent':0,'final_status':latest['status'],'latest_inference':latest['inference_status']}
inference_ms=[c['inference_ms'] for c in valid_chunks if 'inference_ms' in c]
if inference_ms:
    result['ACT_inference_ms']={'samples':len(inference_ms),'median':float(np.median(inference_ms)),
                              'p95':float(np.percentile(inference_ms,95)),'maximum':max(inference_ms)}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert passed, 'Preview verification failed; see saved report'
