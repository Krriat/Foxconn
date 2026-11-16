#!/usr/bin/env python3
"""Verify a preview round over real DDS. Never requests live robot motion."""
import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger
from rokae_interfaces.msg import DualArmServoJ
from protocol import PREFIX
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=35)
    parser.add_argument("--stop-after", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init();node = Node("robot_policy_preview_verification")
    states, messages, arrivals = {}, [], []
    enable_true, traces = [], []
    def status(key, message):
        states[key] = json.loads(message.data)
    def action(message):
        value = json.loads(message.data)
        messages.append(value);arrivals.append(time.monotonic())
    subs = [node.create_subscription(String, PREFIX + "/status", lambda msg:status("executor",msg), 1),
            node.create_subscription(String, PREFIX + "/trajectory_status", lambda msg:status("trajectory",msg), 1),
            node.create_subscription(String, PREFIX + "/inference_status", lambda msg:status("inference",msg), 1),
            node.create_subscription(String, PREFIX + "/action_chunk", action, qos_profile_sensor_data),
            node.create_subscription(String, PREFIX + "/servo_trace", lambda msg: traces.append(json.loads(msg.data)), qos_profile_sensor_data),
            node.create_subscription(DualArmServoJ, "/robot/upperlimb/servoj/dual_arm",
                                     lambda msg:enable_true.append(time.monotonic()) if msg.enable else None,
                                     qos_profile_sensor_data)]
    preview = node.create_client(Trigger, PREFIX + "/preview")
    stop = node.create_client(Trigger, PREFIX + "/stop")
    def spin_until(predicate, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
            if predicate():return True
        return False
    def call(client):
        assert client.wait_for_service(timeout_sec=3), "Service discovery timed out"
        future = client.call_async(Trigger.Request())
        assert spin_until(future.done, 3), "Service call timed out"
        result = future.result()
        assert result.success, result.message
        return result
    run_id, error, stop_sent = None, None, False
    try:
        assert spin_until(lambda: states.get("inference",{}).get("ready") and
                          states.get("trajectory",{}).get("ready") and
                          states.get("executor",{}).get("trajectory_ready") and
                          states.get("executor",{}).get("phase")=="IDLE",20), "Nodes not ready"
        assert not states["executor"]["enable_motion"], "Preview validation requires motion-disabled deployment"
        result = call(preview);print(result.message,flush=True)
        assert spin_until(lambda:states.get("executor",{}).get("phase")=="PREVIEW",3)
        run_id=states["executor"]["run_id"]
        start=time.monotonic();next_log=start
        while time.monotonic()-start<args.seconds:
            rclpy.spin_once(node,timeout_sec=.05)
            state=states.get("executor",{})
            if state.get("phase")=="FAULT":raise RuntimeError(state["reason"])
            if args.stop_after and not stop_sent and time.monotonic()-start>=args.stop_after:
                call(stop);stop_sent=True
            if state.get("phase")=="IDLE":break
            if time.monotonic()>=next_log:
                print(f"phase={state.get('phase')} actions={state.get('accepted_actions')} servo={state.get('servo_commands')}",flush=True)
                next_log=time.monotonic()+5
        assert spin_until(lambda:states.get("executor",{}).get("phase")=="IDLE",2), "Round did not stop"
        selected=[(message,arrival) for message,arrival in zip(messages,arrivals) if message["run_id"]==run_id]
        assert len(selected)>=20,"Too few policy actions"
        assert not enable_true,"Preview emitted enable=true Servo J commands"
        assert states["executor"]["servo_commands"]==0
        sequence=[x[0]["sequence"] for x in selected]
        assert all(a<b for a,b in zip(sequence,sequence[1:])),"Non-increasing action sequence"
        hz=(len(selected)-1)/(selected[-1][1]-selected[0][1])
        assert 18<=hz<=21,f"Unexpected action rate: {hz}"
        active=[trace for trace in traces if trace.get("run_id")==run_id and trace.get("active")]
        assert len(active)>50,"Missing native trajectory previews"
        assert all(not t.get("hardware_commands_sent") for t in active)
        assert max(t["contributors"] for t in active)>1,"No overlapping chunks fused"
        assert all(len(message["actions"])==50 for message,_ in selected)
        speed=max(max(abs(v) for v in t["velocity"]) for t in active)
        accelerations=[];position_residual=[]
        for a,b in zip(active,active[1:]):
            if b["trace_sequence"]!=a["trace_sequence"]+1:continue
            accelerations.extend(abs(x-y)/b["dt_s"] for x,y in zip(b["velocity"],a["velocity"]))
            position_residual.extend(abs(q-p-v*b["dt_s"]) for q,p,v in zip(b["command"],a["command"],b["velocity"]))
        assert speed<=.5+1e-8
        assert max(accelerations)<=1.+1e-6
        assert max(position_residual)<1e-8,"Position reset across a chunk boundary"
        def summary(values):
            return {"median":float(np.median(values)),"p95":float(np.percentile(values,95)),"max":float(max(values))}
        timings={"capture_to_inference_start_ms":summary([(t["inference_start_ns"]-t["observation_time_ns"])/1e6 for t in active]),
                 "inference_ms":summary([(t["inference_end_ns"]-t["inference_start_ns"])/1e6 for t in active]),
                 "send_to_receive_ms":summary([t["network_to_executor_ms"] for t in active]),
                 "observation_to_selection_ms":summary([t["observation_age_ms"] for t in active])}
        (args.output/'servo_trace.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces if t.get("run_id")==run_id))
        report={"native_active_ticks":len(active),"max_speed_rad_s":speed,"max_acceleration_rad_s2":max(accelerations),
                "max_position_integration_residual_rad":max(position_residual),"timings_ms":timings,
                "max_fusion_contributors":max(t["contributors"] for t in active),
                "prediction_index":summary([t["newest_index"] for t in active]),"status":"passed_no_motion_preview","run_id":run_id,"actions_received":len(selected),
                "action_hz":hz,"servo_enable_true_received":len(enable_true),"final":states,
                "stop_method":"manual" if stop_sent else "configured_round_timeout"}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output/'actions.jsonl').write_text(''.join(json.dumps(x[0])+'\n' for x in selected))
        print(json.dumps(report,indent=2),flush=True)
    except Exception as exception:
        error=str(exception)
        if run_id and states.get("executor",{}).get("run_id")==run_id:
            try:call(stop)
            except Exception:pass
        (args.output/'error.json').write_text(json.dumps({"error":error,"states":states},indent=2)+'\n')
        raise
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__=="__main__":main()
