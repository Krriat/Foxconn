"""Observer shadow/transition wiring with camera and executor fakes only."""
import os
assert os.environ.get("ROS_DOMAIN_ID")=="77" and os.environ.get("ROS_LOCALHOST_ONLY")=="1"
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParametersAtomically
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data

from training_tools.task_brain.node import Observer,CAMERAS


class FakeBrain:
    sha="a"*64
    def predict(self,image,stage):
        return {"value":{"scene":"far_open" if stage=="far" else "both_open","status":"complete"},
                "adapter_sha256":self.sha,"latency_ms":0.1,"mode":"fake_no_GPU_no_robot"}


def main():
    rclpy.init();fake=Node("isolated_observer_fixture")
    pub=fake.create_publisher(String,"/robot/policy/status",1)
    camera_pubs={c:fake.create_publisher(CompressedImage,f"/robot/sensors/cameras/{c}/color/image_raw/compressed",
                                      qos_profile_sensor_data) for c in CAMERAS}
    context={"mode":"live","phase":"RUNNING","run_id":"run-far",
             "task_sequence":{"id":"seq","stage":"far","result":"running","pending":None,"completion_mode":"qwen_lora"}}
    requests=[]
    def finish(req,res):
        value={p.name:p.value.string_value if p.value.type==4 else p.value.integer_value for p in req.parameters}
        requests.append(value)
        assert value["run_id"]==context["run_id"] and value["stage"]==context["task_sequence"]["stage"]
        if value["stage"]=="far":
            context["run_id"]="run-near";context["task_sequence"]["stage"]="near"
        else:context["phase"]="IDLE";context["task_sequence"]["result"]="complete"
        res.result.successful=True;res.result.reason="Fake keyed completion only";return res
    service=fake.create_service(SetParametersAtomically,"/robot/policy/brain_finish",finish)
    ok,jpeg=cv2.imencode(".jpg",np.full((480,640,3),100,np.uint8));assert ok
    def publish():
        pub.publish(String(data=json.dumps(context)))
        for p in camera_pubs.values():
            m=CompressedImage();m.header.stamp=fake.get_clock().now().to_msg();m.format="jpeg";m.data=jpeg.tobytes();p.publish(m)
    timer=fake.create_timer(.04,publish)
    directory=Path(tempfile.mkdtemp(prefix="brain-observer-domain77-"))
    shadow=Observer(FakeBrain(),log_path=directory/"shadow.jsonl")
    shadow.gate.min_stage_seconds=0;shadow.gate.hold_seconds=.5
    def spin_until(node,predicate,seconds=8):
        end=time.monotonic()+seconds
        while not predicate() and time.monotonic()<end:
            rclpy.spin_once(fake,timeout_sec=.005);rclpy.spin_once(node,timeout_sec=.005)
        assert predicate(),node.gate.reason
    try:
        spin_until(shadow,lambda:shadow.proposals>0)
        assert shadow.transition_client is None and shadow.service_calls==0 and not requests
        own=fake.get_client_names_and_types_by_node(shadow.get_name(),"/")
        assert not own,own
        shadow.close();shadow.destroy_node()
        active=Observer(FakeBrain(),execute=True,log_path=directory/"transitions.jsonl")
        active.gate.min_stage_seconds=0;active.gate.hold_seconds=.5
        try:
            spin_until(active,lambda:len(requests)==2,seconds=12)
            assert [r["stage"] for r in requests]==["far","near"]
            spin_until(active,lambda:context["phase"]=="IDLE")
            own=fake.get_client_names_and_types_by_node(active.get_name(),"/")
            assert [name for name,types in own]==["/robot/policy/brain_finish"],own
            assert active.service_calls==2
            print(json.dumps({"passed":True,"domain":77,"shadow_service_clients":0,
                "shadow_service_calls":0,"fake_keyed_transition_calls":2,
                "robot_motion_clients":0,"stage_order":[r["stage"] for r in requests],
                "events":str(directory)},indent=2),flush=True)
        finally:active.close();active.destroy_node()
    finally:
        fake.destroy_node();rclpy.shutdown()


if __name__=="__main__":main()
