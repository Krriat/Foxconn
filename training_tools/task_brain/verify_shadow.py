"""Read-only live brain/camera/joint audit; creates no publishers or clients."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState,CompressedImage
from std_msgs.msg import String,Bool


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=12);args=p.parse_args()
    args.output.parent.mkdir(exist_ok=True,parents=True)
    rclpy.init();node=Node("lora_brain_readonly_audit")
    status={};brain=[];joints={"left":[],"right":[]};suction={};head=[]
    def message(key,msg):
        value=json.loads(msg.data);status[key]=value
        if key=="brain":brain.append(value)
    subs=[node.create_subscription(String,topic,lambda msg,k=k:message(k,msg),1)
          for k,topic in (("brain","/robot/policy/brain_status"),("executor","/robot/policy/status"))]
    for side in joints:
        subs.append(node.create_subscription(JointState,f"/robot/upperlimb/joint_states/{side}_arm",
            lambda msg,side=side:joints[side].append(list(msg.position)),qos_profile_sensor_data))
        subs.append(node.create_subscription(Bool,f"/robot/suction/{side}/state",
            lambda msg,side=side:suction.update({side:bool(msg.data)}),qos_profile_sensor_data))
    subs.append(node.create_subscription(CompressedImage,"/robot/sensors/cameras/head/color/image_raw/compressed",
        lambda msg:head.append(bytes(msg.data)) if len(head)<1 else None,qos_profile_sensor_data))
    start=time.monotonic()
    while time.monotonic()-start<args.seconds:rclpy.spin_once(node,timeout_sec=.05)
    clients=node.get_client_names_and_types_by_node("robot_policy_lora_brain","/")
    latest=status.get("brain",{})
    latencies=[x["result"]["latency_ms"] for x in brain if x.get("result")]
    report={"mode":"read_only_audit","brain":latest,"executor_phase":status.get("executor",{}).get("phase"),
        "brain_service_clients":clients,"received_brain_status":len(brain),"suction":suction,
        "joint_range_rad":{side:float(np.max(np.ptp(np.asarray(rows),axis=0))) if rows else None
                           for side,rows in joints.items()},
        "observed_latency_median_ms":float(np.median(latencies)) if latencies else None,
        "publishers_or_service_clients_created_by_audit":0}
    visible=latest.get("result",{}).get("visibility_ok",True)
    valid_state=(bool(latest.get("ready")) if visible else
                 latest.get("ready") is False and latest.get("result",{}).get("value")=={"scene":"unknown","status":"uncertain"})
    report["passed"]=(valid_state and latest.get("transitions_enabled") is False
        and latest.get("service_calls")==0 and not clients and report["executor_phase"] in ("IDLE","PREVIEW")
        and suction=={"left":False,"right":False}
        and all(value is not None and value<.005 for value in report["joint_range_rad"].values()))
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    if head:args.output.with_suffix(".jpg").write_bytes(head[0])
    print(json.dumps(report,indent=2),flush=True)
    node.destroy_node();rclpy.shutdown()
    assert report["passed"],"Read-only shadow audit failed"


if __name__=="__main__":main()
