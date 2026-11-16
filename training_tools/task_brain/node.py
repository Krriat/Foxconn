"""Qwen LoRA ROS observer. Default shadow mode creates NO robot service client."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import socket
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import Parameter,ParameterValue
from rcl_interfaces.srv import SetParametersAtomically
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from training_tools.task_brain.decision import Gate,Evidence
from training_tools.task_brain.lora_data import mosaic

CAMERAS=("head","left_wrist","right_wrist")


class Observer(Node):
    def __init__(self, brain, *, execute=False, observe_idle=False, idle_stage="near", log_path=None):
        super().__init__("robot_policy_lora_brain")
        self.brain=brain;self.execute=execute;self.observe_idle=observe_idle;self.idle_stage=idle_stage
        self.executor_context={};self.executor_context_received=0.;self.images={};self.future=None;self.pending_service=None
        self.gate=Gate(max_gap=2.5);self.pool=ThreadPoolExecutor(max_workers=1)
        self.last_started=0.;self.last_stamp=0;self.ready=False;self.result=None;self.decision=None
        self.inferences=0;self.proposals=0;self.service_calls=0;self.error="";self.last_valid=0.
        self.transition_error="";self.service_started=0.
        self.log_path=Path(log_path) if log_path else None
        if self.log_path:self.log_path.parent.mkdir(exist_ok=True,parents=True)
        self.transition_client=(self.create_client(SetParametersAtomically,"/robot/policy/brain_finish")
                                if execute else None)
        self.publisher=self.create_publisher(String,"/robot/policy/brain_status",1)
        self.subs=[self.create_subscription(String,"/robot/policy/status",self.status,1)]
        for camera in CAMERAS:
            self.subs.append(self.create_subscription(CompressedImage,
                f"/robot/sensors/cameras/{camera}/color/image_raw/compressed",
                lambda msg,camera=camera:self.images.update({camera:(msg,time.monotonic())}),qos_profile_sensor_data))
        self.create_timer(.05,self.tick);self.create_timer(.1,self.publish)

    def event(self,kind,**values):
        row={"event":kind,"time_ns":time.time_ns(),"execute_transitions":self.execute,**values}
        if self.log_path:
            with self.log_path.open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")

    def status(self,msg):
        try:self.executor_context=json.loads(msg.data);self.executor_context_received=time.monotonic()
        except ValueError:return
        self.gate.observe_context(self.executor_context,self.executor_context_received)

    def publish(self):
        now=time.monotonic()
        fresh=(len(self.images)==3 and all(now-received<=.3 for _,received in self.images.values())
               and now-self.executor_context_received<=.4)
        responsive=self.future is None or now-self.last_started<=3.
        self.publisher.publish(String(data=json.dumps({"schema":"qwen_lora_brain_status_v1",
            "host":socket.gethostname(),
            "ready":self.ready and fresh and responsive and not (self.error or self.transition_error),"transitions_enabled":self.execute,
            "adapter_sha256":self.brain.sha,"run_id":self.executor_context.get("run_id"),
            "stage":(self.executor_context.get("task_sequence") or {}).get("stage"),
            "reason":self.gate.reason,"error":self.error or self.transition_error,"inferences":self.inferences,
            "proposals":self.proposals,"service_calls":self.service_calls,
            "result":self.result,"decision":self.decision,"last_inference_age_seconds":
                None if not self.last_valid else time.monotonic()-self.last_valid},allow_nan=False)))

    def tick(self):
        now=time.monotonic()
        if self.pending_service is not None and self.pending_service.done():
            try:
                reply=self.pending_service.result().result
                self.event("transition_reply",success=reply.successful,reason=reply.reason)
                if not reply.successful:self.transition_error="Transition rejected: "+reply.reason
            except Exception as e:self.transition_error="Transition response: "+str(e)
            self.pending_service=None
        if self.pending_service is not None and now-self.service_started>5.:
            self.transition_error="Brain transition response timeout; automatic retry disabled"
            self.event("transition_timeout",error=self.transition_error)
            self.pending_service=None
        if self.future is not None and self.future.done():
            try:
                result,evidence=self.future.result();self.result=result;self.inferences+=1
                self.ready=bool(result.get("visibility_ok",True));self.error="";self.last_valid=now
                self.event("prediction",run_id=evidence.run_id,stage=evidence.stage,
                           image_stamp_ns=evidence.image_stamp_ns,**result)
                decision=self.gate.update(evidence,self.executor_context,now=now,context_age=now-self.executor_context_received)
                if not self.ready:self.gate.invalidate(result.get("quality",{}).get("reason","input not visible"))
                if decision is not None:
                    self.proposals+=1;self.decision=decision
                    self.event("transition_proposal",decision=decision,shadow=not self.execute)
                    automatic=(self.executor_context.get("task_sequence") or {}).get("completion_mode")=="qwen_lora"
                    if self.execute and not automatic:
                        self.event("manual_sequence_observation_only",decision=decision)
                        self.gate.mark_dispatched(decision)
                    elif self.execute and self.pending_service is None and not self.transition_error:
                        self.dispatch(decision,evidence)
                    elif not self.execute:
                        self.gate.mark_dispatched(decision)  # one shadow proposal per stage
            except Exception as e:
                self.error=str(e);self.gate.invalidate(self.error);self.event("inference_error",error=self.error)
            self.future=None
        active=self.gate.observe_context(self.executor_context,now)
        if self.future is not None or now-self.last_started<1.0:return
        if active is None and self.ready and not self.error and not self.observe_idle:return
        if len(self.images)!=3 or any(now-received>.3 for _,received in self.images.values()):
            self.gate.invalidate("waiting for three fresh cameras");return
        if now-self.executor_context_received>.4:
            self.gate.invalidate("waiting for fresh executor status");return
        stamps=[msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec for msg,_ in self.images.values()]
        if max(stamps)-min(stamps)>150_000_000:return
        msg=self.images["head"][0];stamp=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
        if stamp<=self.last_stamp:return
        frames=[cv2.imdecode(np.frombuffer(self.images[camera][0].data,np.uint8),cv2.IMREAD_COLOR) for camera in CAMERAS]
        image=Image.fromarray(cv2.cvtColor(mosaic(frames),cv2.COLOR_BGR2RGB))
        run=self.executor_context.get("run_id","");stage=active[1] if active else self.idle_stage
        captured=now;self.last_stamp=stamp;self.last_started=now
        def predict():
            result=self.brain.predict(image,stage)
            return result,Evidence(run,stage,stamp,captured,result["value"])
        self.future=self.pool.submit(predict)

    def dispatch(self,decision,evidence):
        # Only one keyed completion endpoint is reachable. This observer never
        # creates clients for home, power, ServoJ, suction, or start services.
        if not self.transition_client.service_is_ready():
            self.gate.invalidate("keyed executor brain_finish service unavailable");return
        payload={k:decision[k] for k in ("run_id","stage","sequence_id")}
        payload.update(adapter_sha256=self.brain.sha,image_time_ns=evidence.image_stamp_ns,
                       decision_time_ns=time.time_ns(),scene=evidence.value["scene"])
        request=SetParametersAtomically.Request()
        for key,value in payload.items():
            request.parameters.append(Parameter(name=key,value=ParameterValue(
                type=2,integer_value=value) if type(value)is int else ParameterValue(type=4,string_value=value)))
        self.gate.mark_dispatched(decision);self.service_calls+=1
        self.service_started=time.monotonic()
        self.pending_service=self.transition_client.call_async(request)
        self.event("transition_sent",**payload)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--execute-transitions",action="store_true")
    p.add_argument("--observe-idle",action="store_true")
    p.add_argument("--idle-stage",choices=("far","near"),default="near")
    p.add_argument("--seconds",type=float)
    p.add_argument("--log",type=Path)
    args=p.parse_args()
    from training_tools.task_brain.inference import Brain
    brain=Brain(args.adapter)
    rclpy.init();node=Observer(brain,execute=args.execute_transitions,observe_idle=args.observe_idle,
                              idle_stage=args.idle_stage,log_path=args.log)
    started=time.monotonic()
    try:
        while rclpy.ok() and (args.seconds is None or time.monotonic()-started<args.seconds):
            rclpy.spin_once(node,timeout_sec=.05)
    finally:
        node.close();node.destroy_node();rclpy.shutdown()


if __name__=="__main__":main()
