#!/usr/bin/env python3
import os
assert os.environ.get('ROS_DOMAIN_ID')=='77' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
import json
from pathlib import Path
import time
import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from task_monitor import Monitor

rclpy.init();monitor=Monitor();source=Node('isolated_task_monitor_source')
executor=SingleThreadedExecutor();executor.add_node(monitor);executor.add_node(source)
pub=source.create_publisher(CompressedImage,'/robot/sensors/cameras/head/color/image_raw/compressed',qos_profile_sensor_data)
context=source.create_publisher(String,'/robot/policy/status',1)
received=[]
sub=source.create_subscription(String,'/robot/policy/task_scene',lambda x:received.append(json.loads(x.data)),1)
root=Path(__file__).resolve().parents[2]
im=cv2.imread(str(root/'training_verification/task_mix_audit_20261112/current_head.jpg'))
data=cv2.imencode('.jpg',im)[1].tobytes()
def send():
    context.publish(String(data=json.dumps(dict(run_id='test',heartbeat='test-heartbeat'))))
    msg=CompressedImage();msg.header.stamp=source.get_clock().now().to_msg();msg.format='jpeg';msg.data=data
    pub.publish(msg)
timer=source.create_timer(.05,send)
try:
    end=time.monotonic()+3.
    while time.monotonic()<end:executor.spin_once(timeout_sec=.05)
    assert len(received)>=10,len(received)
    assert all(x['scene']=='far_open' and x['run_id']=='test' for x in received)
    assert len({x['image_time_ns'] for x in received})==len(received)
    timer.cancel()
    end=time.monotonic()+.7
    while time.monotonic()<end:executor.spin_once(timeout_sec=.05)
    count=len(received)
    end=time.monotonic()+.7
    while time.monotonic()<end:executor.spin_once(timeout_sec=.05)
    assert len(received)==count,'must not republish stale images'
    print(json.dumps({'passed':['actual monitor constructs and classifies compressed images',
                     'round identity and unique timestamps preserved','no repeated publication after stream stops'],
                     'frames':count},indent=2))
finally:
    executor.shutdown();source.destroy_node();monitor.destroy_node();rclpy.shutdown()
