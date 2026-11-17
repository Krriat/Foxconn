"""Real executor + fake arms, brain-only stage transitions; Domain 77 ONLY."""
import os
assert os.environ.get("ROS_DOMAIN_ID")=="77" and os.environ.get("ROS_LOCALHOST_ONLY")=="1"
import json
from pathlib import Path
import tempfile
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.msg import Parameter,ParameterValue
from rcl_interfaces.srv import SetParametersAtomically
from std_msgs.msg import String
from std_srvs.srv import Trigger
import yaml
import executor_node
from test_executor_ros import FakePlant
from test_sequence_ros import FakeIO


def main():
    executor_node.Suction=FakeIO
    cfg=yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    cfg.update(enable_motion=True,suction_enabled=True,max_round_seconds=None,
               driver_node_name="isolated_fake_plant",brain_adapter_sha256="a"*64)
    rclpy.init();directory=Path(tempfile.mkdtemp(prefix="brain-no-sdk-domain77-"))
    controller=executor_node.Executor(cfg,directory/"events.jsonl");plant=FakePlant(cfg)
    plant.predicted_suction=[0.,0.]
    client=Node("isolated_brain_test_client");executor=MultiThreadedExecutor(num_threads=6)
    for node in (controller,plant,client):executor.add_node(node)
    pub=client.create_publisher(String,"/robot/policy/brain_status",1)
    state={"ready":True,"transitions_enabled":True,"adapter_sha256":"a"*64}
    timer=client.create_timer(.1,lambda:pub.publish(String(data=json.dumps(state))))
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    triggers={name:client.create_client(Trigger,"/robot/policy/"+name)
              for name in ("start_sequence_brain","start_sequence","stop")}
    finish=client.create_client(SetParametersAtomically,"/robot/policy/brain_finish")
    def wait(pred,seconds=8):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if pred():return
            time.sleep(.01)
        raise AssertionError(f"{controller.phase}: {controller.reason}")
    def trigger(name,success=True):
        assert triggers[name].wait_for_service(timeout_sec=3)
        future=triggers[name].call_async(Trigger.Request());wait(future.done)
        assert future.result().success==success,future.result().message
    def payload():
        return {"run_id":controller.run_id,"stage":controller.sequence_job["stage"],
            "sequence_id":controller.sequence_job["id"],"adapter_sha256":"a"*64,
            "image_time_ns":time.time_ns(),"decision_time_ns":time.time_ns(),
            "scene":"far_open" if controller.sequence_job["stage"]=="far" else "both_open"}
    def complete(value,success=True):
        request=SetParametersAtomically.Request()
        for key,value_ in value.items():
            request.parameters.append(Parameter(name=key,value=ParameterValue(type=2,integer_value=value_)
                if type(value_)is int else ParameterValue(type=4,string_value=value_)))
        future=finish.call_async(request);wait(future.done)
        assert future.result().result.successful==success,future.result().result.reason
    checks=[]
    try:
        wait(lambda:controller.inference_status.get("ready") and controller.brain_status.get("ready"))
        state["ready"]=False;wait(lambda:controller.brain_status.get("ready") is False)
        trigger("start_sequence_brain",False)
        assert plant.initializes==0 and plant.home_calls=={"left":0,"right":0}
        state["ready"]=True;wait(lambda:controller.brain_status.get("ready") is True)
        checks.append("unready brain cannot initialize or move either arm")
        state["adapter_sha256"]="b"*64;wait(lambda:controller.brain_status.get("adapter_sha256")=="b"*64)
        trigger("start_sequence_brain",False)
        state["adapter_sha256"]="a"*64;wait(lambda:controller.brain_status.get("adapter_sha256")=="a"*64)
        trigger("start_sequence_brain");wait(lambda:controller.phase=="RUNNING")
        assert plant.home_calls=={"left":1,"right":1}
        checks.append("far ACT cannot run before BOTH fresh measured home confirmations")
        complete(payload(),False)  # minimum 5-second stage duration
        controller.round_start-=6
        complete(dict(payload(),adapter_sha256="b"*64),False)
        complete(dict(payload(),image_time_ns=controller.home_verified_time_ns-1),False)
        checks.append("wrong adapter, early completion and pre-home image rejected")
        old=payload();plant.held_endpoint=[x+.04 for x in cfg["home_left"]+cfg["home_right"]]
        wait(lambda:abs(controller.joints["left"][0][0]-cfg["home_left"][0])>.03)
        plant.apply_home=False
        complete(payload());wait(lambda:controller.phase=="VERIFY_HOME")
        assert plant.home_calls=={"left":2,"right":2} and controller.home_verified_run_id==""
        accepted=controller.accepted;time.sleep(.25)
        assert controller.accepted==accepted and controller.phase=="VERIFY_HOME"
        checks.append("brain far completion stops, releases, resets UUID, and homes BOTH arms again")
        plant.apply_home=True;plant.q=cfg["home_left"]+cfg["home_right"]
        wait(lambda:controller.phase=="RUNNING")
        assert controller.sequence_job["stage"]=="near"
        complete(old,False);controller.round_start-=6
        complete(dict(payload(),scene="far_open"),False)
        checks.append("old far result and incomplete near scene cannot finish near ACT")
        complete(payload());wait(lambda:controller.phase=="IDLE")
        assert controller.sequence_job["completed_stages"]==["far","near"]
        assert controller.sequence_job["result"]=="complete" and controller.suction.state()["verified"]==[False,False]
        assert plant.home_calls=={"left":2,"right":2} and not plant.enabled
        checks.append("brain near completion stops and releases; no third stage")
        trigger("start_sequence");wait(lambda:controller.phase=="RUNNING")
        controller.round_start-=6;complete(payload(),False)
        trigger("stop");wait(lambda:controller.phase=="IDLE")
        checks.append("original manual sequence rejects brain completions and remains manually controlled")
        events=[json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
        completions=[e for e in events if e["event"]=="stage_complete"]
        assert [e["completion_source"] for e in completions]==["qwen_lora","qwen_lora"]
        print(json.dumps({"passed":True,"domain":77,"sdk_or_hardware_clients_loaded":False,
                          "checks":checks,"events":str(directory/"events.jsonl")},indent=2),flush=True)
    finally:
        if controller.phase not in ("IDLE","FAULT"):
            controller.request_stop("isolated test shutdown")
            try:wait(lambda:controller.phase in ("IDLE","FAULT"))
            except Exception:pass
        executor.shutdown();thread.join(timeout=3)
        if controller.suction:controller.suction.close()
        controller.log.close()
        for node in (controller,plant,client):node.destroy_node()
        rclpy.shutdown()


if __name__=="__main__":main()
