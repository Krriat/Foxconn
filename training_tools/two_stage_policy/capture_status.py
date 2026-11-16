#!/usr/bin/env python3
"""Only subscribe: record deployed status without starting policy or moving arms."""
import argparse
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=6.)
    args=parser.parse_args()
    rclpy.init();node=Node('two_stage_readonly_observer')
    latest={};counts={};states=set()
    def receive(key,msg):
        latest[key]=json.loads(msg.data);counts[key]=counts.get(key,0)+1
        if key=='status':states.add(latest[key].get('phase'))
    topics=('status','task_scene','inference_status','trajectory_status')
    subs=[node.create_subscription(String,'/robot/policy/'+key,lambda msg,key=key:receive(key,msg),1) for key in topics]
    start=time.monotonic()
    while time.monotonic()-start<args.seconds:rclpy.spin_once(node,timeout_sec=.1)
    report={'mode':'read_only_no_publishers_or_service_calls','latest':latest,'counts':counts,'phases_seen':sorted(states)}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
