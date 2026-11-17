#!/usr/bin/env python3
"""Exercise the real round controller against ROS fakes in isolated Domain 77."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time

assert os.environ.get("ROS_DOMAIN_ID") == "77", "Fakes must run only in isolated Domain 77"
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from rokae_interfaces.msg import DualArmServoJ
from rokae_interfaces.srv import MoveAbsJ
import yaml
from executor_node import Executor
from protocol import PREFIX, encode
from stage_policy import checkpoint_map, selected_policy


class FakePlant(Node):
    def __init__(self, config):
        super().__init__("isolated_fake_plant")
        self.config = config
        self.feedback_reader = "drain_latest_v1"
        self.scaling_version = "per_observation_v1"
        self.inference_scale = self.config.get("execution_speed_scale",1.)
        self.trajectory_error = ""
        self.suction_states = [False, False]
        self.predicted_suction = [.8, 0.]
        self.q = config["home_left"] + config["home_right"]
        self.trace, self.enabled, self.pause, self.move_delay = [], False, False, 0.0
        self.stop_failures_remaining = 0
        self.apply_home = True
        self.wrong_checkpoint = False
        self.held_endpoint = None
        self.home_calls, self.initializes, self.sequence, self.round_id = {"left": 0, "right": 0}, 0, 0, ""
        self.policy_context, self.corrupt, self.stale_run, self.bad_heartbeat = {}, False, False, False
        self.group = ReentrantCallbackGroup()
        self.services_kept = [self.create_service(Trigger, "/robot/upperlimb/initialize/dual_arm", self.initialize,
                                             callback_group=self.group)]
        self.joint_pubs, self.suction_pubs = {}, {}
        for side in ("left", "right"):
            self.services_kept += [self.create_service(MoveAbsJ, f"/robot/upperlimb/move_absj/{side}_arm",
                              lambda req, res, side=side: self.move(side, req, res), callback_group=self.group),
                              self.create_service(Trigger, f"/robot/upperlimb/stop/{side}_arm",
                              lambda req, res, side=side: self.stop(side, req, res), callback_group=self.group)]
            self.joint_pubs[side] = self.create_publisher(JointState, f"/robot/upperlimb/joint_states/{side}_arm", 5)
            self.suction_pubs[side] = self.create_publisher(Bool, f"/robot/suction/{side}/state", 5)
        self.status_pub = self.create_publisher(String, "/robot/upperlimb/servoj/status", 100)
        self.inference_pub = self.create_publisher(String, PREFIX + "/inference_status", 1)
        self.trajectory_pub = self.create_publisher(String, PREFIX + "/trajectory_status", 1)
        self.action_pub = self.create_publisher(String, PREFIX + "/action_chunk", qos_profile_sensor_data)
        self.trace_pub = self.create_publisher(String, PREFIX + "/servo_trace", qos_profile_sensor_data)
        self.subs = [self.create_subscription(DualArmServoJ, "/robot/upperlimb/servoj/dual_arm", self.servo, qos_profile_sensor_data),
                     self.create_subscription(String, PREFIX + "/status", lambda msg: setattr(self, "policy_context", json.loads(msg.data)), 1)]
        self.create_timer(0.02, self.feedback)
        self.create_timer(0.05, self.actions)

    def initialize(self, _, response):
        self.held_endpoint = None
        self.initializes += 1
        self.trace.append(("initialize", time.monotonic()))
        response.success, response.message = True, "fake initialized"
        return response

    def move(self, side, request, response):
        self.home_calls[side] += 1
        self.trace.append(("move_start_" + side, time.monotonic()))
        assert request.joint_speed_scale == 0.05 and request.zone_mm == 0.0
        time.sleep(self.move_delay)
        offset = 0 if side == "left" else 7
        if self.apply_home:
            self.q[offset:offset+7] = list(request.joint_positions)
        self.trace.append(("move_done_" + side, time.monotonic()))
        response.success, response.message = True, "fake home reached"
        return response

    def stop(self, side, _, response):
        self.trace.append(("stop_" + side, time.monotonic()))
        if self.stop_failures_remaining:
            self.stop_failures_remaining -= 1
            response.success, response.message = False, "fake transient RT-to-NRT transition"
            return response
        response.success, response.message = True, "fake stopped"
        return response

    def servo(self, message):
        if message.enable:
            self.enabled = True
            self.q = list(self.held_endpoint) if self.held_endpoint is not None else list(message.left_positions) + list(message.right_positions)
            self.trace.append(("servo", time.monotonic()))
            for side in ("left", "right"):
                self.status_pub.publish(String(data=side + ":running"))
        elif self.enabled:
            self.enabled = False
            for side in ("left", "right"):
                self.status_pub.publish(String(data=side + ":idle_power_on"))

    def feedback(self):
        self.trajectory_pub.publish(String(data=encode({"protocol":"timed_chunk_v2","ready":True,
                                                       "execution_time_scaling":self.scaling_version,
                                                       "maximum_execution_speed_scale":1.5,
                                                       "maximum_joint_acceleration_rad_s2":2.25,
                                                       "last_error": self.trajectory_error,
                                                       "feedback_reader":self.feedback_reader})))
        for side, offset in (("left", 0), ("right", 7)):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [f"{side}_joint_{i}" for i in range(1, 8)]
            msg.position = self.q[offset:offset+7]
            self.joint_pubs[side].publish(msg)
            self.suction_pubs[side].publish(Bool(data=self.suction_states[0 if side == "left" else 1]))
        self.inference_pub.publish(String(data=encode({"ready": True, "error": "", "model": "act", "publisher_id": "fake",
                                                      "execution_speed_scale":self.inference_scale,
                                                      "loaded_checkpoints": checkpoint_map(self.config),
                                                      "checkpoint_sha256":self.config["checkpoint_sha256"]["act"]})))

    def actions(self):
        if self.pause or self.policy_context.get("phase") not in ("WAITING_ACTION", "STARTING_SERVO", "RUNNING", "PREVIEW"):
            return
        run = self.policy_context["run_id"]
        if self.round_id != run:
            self.round_id, self.sequence = run, 0
        self.sequence += 1
        action = list(self.q) + list(self.predicted_suction)
        action[0] += 0.01
        if self.corrupt:
            action[0] = 100.0
        now = time.time_ns()
        key, sha = selected_policy(self.config, self.policy_context.get("task_sequence"))
        msg = {"schema": "dual_arm_suction_chunk_v2", "run_id": "old" if self.stale_run else run,
               "publisher_id": "fake", "sequence": self.sequence, "model": "act",
               "executor_heartbeat": "expired" if self.bad_heartbeat else self.policy_context["heartbeat"],
               "checkpoint_sha256": self.config["checkpoint_sha256"]["act"] if self.wrong_checkpoint else sha,
               "policy_key": key,
               "action_hz": 20.0, "max_input_receive_age_ms": 20.0, "actions": [action for _ in range(50)],
               "execution_speed_scale": self.config.get("execution_speed_scale",1.),
               "step_ns": 50_000_000, "observation_time_ns": now-20_000_000,
               "inference_start_ns": now-15_000_000, "inference_end_ns": now-5_000_000, "send_time_ns": now}
        self.action_pub.publish(String(data=encode(msg)))
        if self.policy_context["phase"] in ("STARTING_SERVO","RUNNING","PREVIEW"):
            self.trace_pub.publish(String(data=encode({"run_id":run,"active":True,"command":self.q,
                    "fused_target":action,"hardware_commands_sent":self.policy_context["mode"]=="live"})))


def main():
    config = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    config.update(enable_motion=True, suction_enabled=False, max_round_seconds=0.7,
                  driver_node_name="isolated_fake_plant")
    rclpy.init()
    log_dir = Path(tempfile.mkdtemp(prefix="robot-policy-domain77-"))
    controller = Executor(config, log_dir / "events.jsonl")
    plant = FakePlant(config)
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(controller); executor.add_node(plant)
    thread = threading.Thread(target=executor.spin, daemon=True);thread.start()
    client_node = Node("isolated_policy_test_client")
    executor.add_node(client_node)
    clients = {name: client_node.create_client(Trigger, PREFIX + "/" + name) for name in ("start", "preview", "stop")}
    def wait(predicate, seconds=5):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if predicate():return
            time.sleep(0.01)
        raise AssertionError(f"timeout: phase={controller.phase}, reason={controller.reason}")
    def call(name):
        assert clients[name].wait_for_service(timeout_sec=2)
        future = clients[name].call_async(Trigger.Request())
        wait(future.done)
        assert future.result().success, future.result().message
    tests = []
    try:
        wait(lambda: controller.inference_status.get("ready") and len(controller.joints)==2)
        time.sleep(0.3)
        assert not plant.initializes and not any(x[0]=="servo" for x in plant.trace)
        tests.append("startup sends no motion")
        for field, value, message in [('scaling_version','unsupported','speed/acceleration support'),
                                      ('inference_scale',1.,'speed scales differ')]:
            original=getattr(plant,field);setattr(plant,field,value);time.sleep(.15)
            assert clients['start'].wait_for_service(timeout_sec=2)
            failed=clients['start'].call_async(Trigger.Request());wait(failed.done)
            assert not failed.result().success and message in failed.result().message,failed.result().message
            assert not plant.initializes and not any(x[0]=='servo' for x in plant.trace)
            setattr(plant,field,original);time.sleep(.15)
        tests.append('old native time-scaling support or mismatched inference speed blocks hardware initialization')
        plant.trajectory_error = "SDK network connection"
        wait(lambda: controller.trajectory_status.get('last_error'))
        assert clients['start'].wait_for_service(timeout_sec=2)
        failed = clients['start'].call_async(Trigger.Request());wait(failed.done)
        assert not failed.result().success and 'reconnect driver' in failed.result().message
        assert not plant.initializes
        plant.trajectory_error = ''
        wait(lambda: not controller.trajectory_status.get('last_error'))
        tests.append("native fault blocks new round even if driver advertises ready")
        plant.feedback_reader = "legacy_single_read"
        time.sleep(.15)
        future = clients["start"].call_async(Trigger.Request()); wait(future.done)
        assert not future.result().success and "latest-feedback" in future.result().message
        assert not plant.initializes and not any(x[0]=="servo" for x in plant.trace)
        tests.append("live start rejects driver without latest-feedback fix before hardware initialization")
        plant.feedback_reader = "drain_latest_v1"
        time.sleep(.15)
        call("preview"); wait(lambda: controller.accepted >= 3)
        call("stop"); wait(lambda: controller.phase=="IDLE")
        assert not plant.initializes and not any(x[0]=="servo" for x in plant.trace)
        tests.append("preview has no initializer, home, servo or IO writes")
        call("start"); wait(lambda: controller.phase=="RUNNING")
        assert plant.home_calls=={"left":1,"right":1}
        first_servo = next(t for k,t in plant.trace if k=="servo")
        assert first_servo > max(t for k,t in plant.trace if k.startswith("move_done"))
        plant.stop_failures_remaining=2
        call("stop"); wait(lambda: controller.phase=="IDLE")
        tests.append("home once before Servo J, then manual stop")
        assert 'stop_retry' in (log_dir/'events.jsonl').read_text()
        tests.append("transient stop-service rejection is retried before returning idle")
        call("start"); wait(lambda: controller.phase=="RUNNING")
        wait(lambda: controller.phase=="IDLE",3)
        assert plant.home_calls=={"left":2,"right":2}
        tests.append("next round homes again and stops at configured duration")
        class PendingIO:
            pending=True
            def state(self):return {"error":"","pending_operation":self.pending}
            def request(self,*args):pass
            def suspend(self):pass
            def release_left(self,*args):pass
        call("start");wait(lambda:controller.phase=="RUNNING")
        pending_io=PendingIO();controller.suction=pending_io
        call("stop");time.sleep(.3)
        assert controller.phase=="STOPPING", "In-flight suction operation must drain before accepting a new round"
        pending_io.pending=False;wait(lambda:controller.phase=="IDLE")
        controller.suction=None
        tests.append("stop keeps start gate closed until in-flight suction operation drains")
        call("start"); wait(lambda: controller.phase=="RUNNING")
        plant.pause=True
        wait(lambda: controller.phase=="FAULT",3)
        count=len([x for x in plant.trace if x[0]=="servo"])
        plant.pause=False;time.sleep(.2)
        assert len([x for x in plant.trace if x[0]=="servo"])==count
        call("stop");wait(lambda: controller.phase=="IDLE")
        tests.append("command dropout stops and never auto resumes")
        plant.corrupt=True
        call("start");wait(lambda: controller.phase=="FAULT",4)
        plant.corrupt=False;call("stop");wait(lambda: controller.phase=="IDLE")
        tests.append("invalid current-round target is rejected")
        plant.bad_heartbeat=True
        call("start");wait(lambda:controller.phase=="FAULT",4)
        plant.bad_heartbeat=False;call("stop");wait(lambda:controller.phase=="IDLE")
        tests.append("expired Jetson heartbeat is rejected without relying on host clock synchronization")
        plant.move_delay=.3
        call("start");wait(lambda: controller.phase=="HOMING")
        call("stop");wait(lambda: controller.phase=="IDLE",5)
        last_done=max(t for k,t in plant.trace if k.startswith("move_done"))
        for side in ("left","right"):
            assert max(t for k,t in plant.trace if k=="stop_"+side)>last_done
        tests.append("stop during homing sends final stop after pending moves return")
        print(json.dumps({"passed":tests,"log":str(log_dir),"home_calls":plant.home_calls},indent=2),flush=True)
    finally:
        if controller.phase not in ("IDLE","FAULT"):
            controller.request_stop("test cleanup")
            time.sleep(.5)
        executor.shutdown();thread.join(timeout=3)
        for node in (client_node,controller,plant):
            executor.remove_node(node);node.destroy_node()
        controller.log.close();rclpy.shutdown()


if __name__=="__main__":main()
