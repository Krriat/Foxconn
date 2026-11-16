#!/usr/bin/env python3
"""Jetson round controller: home once, consume policy actions, drive local Servo J."""
import argparse
import json
from pathlib import Path
import signal
import threading
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from rokae_interfaces.msg import DualArmServoJ
from rokae_interfaces.srv import MoveAbsJ
import yaml

from protocol import ACTIVE_PHASES, JOINT_NAMES, PREFIX, encode, finite_vector
from protocol import suction_target, validate_chunk, validate_joints, validate_config, round_time_expired
from suction_io import Suction


class Executor(Node):
    def __init__(self, config, log_path):
        super().__init__("robot_policy_executor")
        self.config = validate_config(config)
        self.phase, self.mode, self.run_id = "IDLE", "none", ""
        self.publisher_id, self.last_sequence = None, 0
        self.action, self.last_action = None, None
        self.native_trace, self.native_received = {}, 0.0
        self.joints, self.suction_feedback = {}, {}
        self.driver_status, self.driver_received = {}, {}
        self.inference_status, self.inference_received = {}, 0.0
        self.trajectory_status, self.trajectory_received = {}, 0.0
        self.previous, self.shadow_suction = None, [False, False]
        self.suction = None
        self.pending, self.stop_futures = [], {}
        self.reason, self.stop_destination = "Waiting for an explicit round start", "IDLE"
        self.stop_retry_error = ""
        self.accepted = self.rejected = self.servo_commands = self.limited_ticks = 0
        self.round_start = self.phase_start = time.monotonic()
        self.last_tick = self.phase_start
        self.last_stop_request, self.final_stop_sent, self.stop_started = 0.0, False, 0.0
        self.next_round_index = 0
        self.heartbeats = {}
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", buffering=1)
        self.init_client = self.create_client(Trigger, "/robot/upperlimb/initialize/dual_arm")
        self.move_clients = {side: self.create_client(MoveAbsJ, f"/robot/upperlimb/move_absj/{side}_arm")
                             for side in ("left", "right")}
        self.stop_clients = {side: self.create_client(Trigger, f"/robot/upperlimb/stop/{side}_arm")
                             for side in ("left", "right")}
        self.status_pub = self.create_publisher(String, PREFIX + "/status", 1)
        self.command_pub = self.create_publisher(DualArmServoJ, "/robot/upperlimb/servoj/dual_arm", qos_profile_sensor_data)
        action_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        self.validated_pub = self.create_publisher(String, PREFIX + "/validated_chunk", action_qos)
        self.subs = [self.create_subscription(String, PREFIX + "/action_chunk", self.on_action, action_qos),
                     self.create_subscription(String, PREFIX + "/servo_trace", self.on_trace, QoSProfile(depth=30, reliability=ReliabilityPolicy.BEST_EFFORT)),
                     self.create_subscription(String, PREFIX + "/trajectory_status", self.on_trajectory_status, 1),
                     self.create_subscription(String, PREFIX + "/inference_status", self.on_inference, 1),
                     self.create_subscription(String, "/robot/upperlimb/servoj/status", self.on_driver, 100)]
        for side in ("left", "right"):
            self.subs.append(self.create_subscription(
                JointState, f"/robot/upperlimb/joint_states/{side}_arm",
                lambda message, side=side: self.on_joints(side, message), qos_profile_sensor_data))
            self.subs.append(self.create_subscription(
                Bool, f"/robot/suction/{side}/state",
                lambda message, side=side: self.suction_feedback.update({side: (bool(message.data), time.monotonic())}),
                qos_profile_sensor_data))
        self.services_kept = [self.create_service(Trigger, PREFIX + "/start", self.start_live),
                         self.create_service(Trigger, PREFIX + "/preview", self.start_preview),
                         self.create_service(Trigger, PREFIX + "/stop", self.stop_round),
                         self.create_service(Trigger, PREFIX + "/check", self.check_live)]
        self.create_timer(1.0 / config["servo_hz"], self.tick)
        self.create_timer(0.1, self.publish_status)
        self.event("node_started", enable_motion=config["enable_motion"])

    def event(self, kind, **values):
        self.log.write(encode({"time_ns": time.time_ns(), "event": kind, "phase": self.phase,
                               "run_id": self.run_id, **values}) + "\n")

    def transition(self, phase, reason):
        self.phase, self.reason, self.phase_start = phase, reason, time.monotonic()
        self.event("phase", reason=reason)
        self.publish_status()

    def on_joints(self, side, message):
        try:
            if len(message.name) != 7 or len(message.position) != 7:
                raise ValueError("seven measured joints required")
            mapping = dict(zip(message.name, message.position))
            values = [float(mapping[f"{side}_joint_{i}"]) for i in range(1, 8)]
            finite_vector(values, 7, side)
            stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            previous = self.joints.get(side)
            if previous and stamp <= previous[2]:
                return
            self.joints[side] = (values, time.monotonic(), stamp)
        except (ValueError, KeyError) as error:
            self.request_stop(f"Invalid {side} feedback: {error}", fault=True)

    def feedback(self):
        now = time.monotonic()
        if len(self.joints) != 2 or len(self.suction_feedback) != 2:
            raise ValueError("Waiting for both arms and suction feedback")
        if max(now - self.joints[side][1] for side in ("left", "right")) > self.config["feedback_timeout_seconds"]:
            raise ValueError("Joint feedback is stale")
        if max(now - value[1] for value in self.suction_feedback.values()) > 0.5:
            raise ValueError("Suction readback is stale")
        return validate_joints(self.joints["left"][0] + self.joints["right"][0], self.config["joint_limit_margin_rad"])

    def on_inference(self, message):
        try:
            self.inference_status, self.inference_received = json.loads(message.data), time.monotonic()
        except ValueError:
            pass

    def on_driver(self, message):
        side, _, status = message.data.partition(":")
        if side not in ("left", "right"):
            return
        self.driver_status[side], self.driver_received[side] = status, time.monotonic()
        if self.mode == "live" and self.phase in ("STARTING_SERVO", "RUNNING"):
            if status.startswith("error") or (self.phase == "RUNNING" and status == "stopped"):
                self.request_stop(f"Servo J driver {side}: {status}", fault=True)

    def on_trajectory_status(self, message):
        try:
            self.trajectory_status, self.trajectory_received = json.loads(message.data), time.monotonic()
        except ValueError:
            pass

    def ready(self, live):
        if self.phase != "IDLE":
            raise ValueError(f"A round cannot start in {self.phase}; stop/reset the previous round first")
        if self.suction is not None and (self.suction.state()["pending_operation"] or
                                        not self.suction.state().get("settled", True)):
            raise ValueError("Waiting for manual suction release readback")
        measured = self.feedback()
        if (time.monotonic()-self.trajectory_received > 2.5 or
                self.trajectory_status.get("protocol") != "timed_chunk_v2" or
                not self.trajectory_status.get("ready")):
            raise ValueError("Timed native trajectory driver is not ready")
        if live and self.trajectory_status.get("feedback_reader") != "drain_latest_v1":
            raise ValueError("Native driver lacks latest-feedback FIFO fix; restart the updated driver")
        if (time.monotonic() - self.inference_received > 2.5 or
                not self.inference_status.get("ready") or self.inference_status.get("error")):
            raise ValueError("Inference node is not ready")
        if self.inference_status.get("model") != self.config["model"]:
            raise ValueError("Inference and executor model settings differ")
        if self.inference_status.get("checkpoint_sha256") != self.config["checkpoint_sha256"][self.config["model"]]:
            raise ValueError("Inference weights differ from the reviewed checkpoint")
        if live:
            if not self.config["enable_motion"]:
                raise ValueError("Motion is disabled in config; preview is available")
            if not all(client.service_is_ready() for client in
                       [self.init_client, *self.move_clients.values(), *self.stop_clients.values()]):
                raise ValueError("Initializer, MoveAbsJ and stop services must all be ready")
            drivers = [entry for entry in self.get_subscriptions_info_by_topic("/robot/upperlimb/servoj/dual_arm")
                       if entry.node_name == self.config["driver_node_name"]]
            if len(drivers) != 1:
                raise ValueError("Exactly one local Servo J driver subscriber is required")
            if len(self.get_publishers_info_by_topic("/robot/upperlimb/servoj/dual_arm")) != 1:
                raise ValueError("Another Servo J command publisher is present")
            home = json.loads((Path(self.config["robot_root"]) / self.config["home_source"]).read_text())
            if (home.get("units") != "rad" or home.get("left") != self.config["home_left"] or
                    home.get("right") != self.config["home_right"]):
                raise ValueError("Saved collection home pose changed; review before enabling motion")
        return measured

    def begin(self, live, response):
        try:
            measured = self.ready(live)
            if live and self.config["suction_enabled"]:
                if self.suction is not None:
                    self.suction.close()
                self.suction = Suction(self.config, [self.suction_feedback[side][0]
                                                     for side in ("left", "right")])
            self.mode, self.run_id = ("live" if live else "preview"), str(uuid.uuid4())
            self.publisher_id, self.last_sequence, self.action, self.last_action = None, 0, None, None
            self.previous = list(measured)
            self.native_trace, self.native_received = {}, 0.0
            self.shadow_suction = [self.suction_feedback[side][0] for side in ("left", "right")]
            self.accepted = self.rejected = self.servo_commands = self.limited_ticks = 0
            self.pending, self.stop_futures = [], {}
            self.final_stop_sent = False
            self.round_start = time.monotonic()
            self.next_round_index += 1
            self.event("round_begin", mode=self.mode, measured=measured,
                       home=self.config["home_left"] + self.config["home_right"])
            if live:
                self.pending = [self.init_client.call_async(Trigger.Request())]
                self.transition("INITIALIZING", "Initializing both arms before dataset home move")
            else:
                self.round_start = time.monotonic()
                self.transition("PREVIEW", "No hardware calls or Servo J commands; evaluating live policy targets")
            response.success, response.message = True, f"Accepted {self.mode} round {self.run_id}"
        except Exception as error:
            response.success, response.message = False, str(error)
        return response

    def start_live(self, _, response):
        return self.begin(True, response)

    def start_preview(self, _, response):
        return self.begin(False, response)

    def check_live(self, _, response):
        try:
            self.ready(True)
            response.success, response.message = True, "Ready: start will initialize, home once, then execute policy"
        except Exception as error:
            response.success, response.message = False, str(error)
        return response

    def stop_round(self, _, response):
        if self.phase == "FAULT":
            if self.mode == "live" and self.suction is not None:
                self.suction.release_left(self.run_id)
                self.shadow_suction[0] = False
            self.transition("IDLE", "Fault acknowledged; a new round will home again")
        else:
            self.request_stop("Manual stop", fault=False)
        response.success, response.message = True, self.reason
        return response

    def on_action(self, message):
        if self.phase not in ACTIVE_PHASES:
            return
        receive_ns = time.time_ns()
        try:
            value = json.loads(message.data)
            if value.get("run_id") != self.run_id:
                self.rejected += 1
                return
            expected = self.inference_status.get("publisher_id")
            actions = validate_chunk(value, self.run_id, self.publisher_id or expected,
                                     self.last_sequence, self.config, receive_ns)
            issued = self.heartbeats.get(value.get("executor_heartbeat"))
            if issued is None or time.monotonic()-issued > self.config["heartbeat_timeout_seconds"]:
                raise ValueError("Chunk echoes a missing or expired Jetson heartbeat")
            self.publisher_id, self.last_sequence = value["publisher_id"], value["sequence"]
            self.action, self.last_action = actions, time.monotonic()
            self.accepted += 1
            value["jetson_receive_ns"] = receive_ns
            self.validated_pub.publish(String(data=encode(value)))
            self.event("chunk", **value)
        except Exception as error:
            self.rejected += 1
            self.request_stop(f"Policy chunk rejected: {error}", fault=True)

    def on_trace(self, message):
        if self.phase not in ACTIVE_PHASES:
            return
        try:
            value = json.loads(message.data)
            if value.get("run_id") != self.run_id:
                return
            self.native_trace, self.native_received = value, time.monotonic()
            self.event("servo_trace", trace=value)
            if value.get("error"):
                raise RuntimeError(value["error"])
            if value.get("active"):
                self.previous = validate_joints(value["command"], self.config["joint_limit_margin_rad"])
                self.limited_ticks += int(max(abs(a-b) for a,b in zip(value["command"],value["fused_target"][:14]))>1e-7)
                self.shadow_suction = [suction_target(target, previous, self.config) for target,previous in
                                       zip(value["fused_target"][14:],self.shadow_suction)]
        except Exception as error:
            self.request_stop(f"Native trajectory: {error}", fault=True)

    def send_servo(self, enable):
        message = DualArmServoJ()
        message.enable = enable
        if self.previous is not None:
            message.left_positions, message.right_positions = self.previous[:7], self.previous[7:]
        self.command_pub.publish(message)
        if enable:
            self.servo_commands += 1

    def request_stop(self, reason, fault=False):
        if self.phase == "IDLE":
            return
        if self.phase == "STOPPING":
            if fault:
                self.stop_destination = "FAULT"
            return
        self.action, self.last_action = None, None
        if self.suction is not None:
            self.suction.suspend()
            if reason == "Manual stop" and self.mode == "live":
                self.suction.release_left(self.run_id)
                self.shadow_suction[0] = False
        if self.mode != "live":
            self.transition("FAULT" if fault else "IDLE", reason)
            return
        self.stop_destination = "FAULT" if fault else "IDLE"
        self.stop_started, self.last_stop_request = time.monotonic(), 0.0
        self.stop_futures, self.final_stop_sent = {}, False
        self.stop_retry_error = ""
        self.transition("STOPPING", reason)
        self.send_servo(False)

    def stop_tick(self):
        self.send_servo(False)
        now = time.monotonic()
        original_done = all(future.done() for future in self.pending)
        if original_done and not self.final_stop_sent:
            # A final stop AFTER queued initialization/moves have returned closes
            # the race in which an earlier stop precedes a queued NRT motion.
            self.stop_futures, self.final_stop_sent = {}, True
        if now - self.last_stop_request >= 0.2:
            for side, client in self.stop_clients.items():
                previous = self.stop_futures.get(side)
                if client.service_is_ready() and (previous is None or (previous.done() and not self.final_stop_sent)):
                    self.stop_futures[side] = client.call_async(Trigger.Request())
            self.last_stop_request = now
        if original_done and self.final_stop_sent and len(self.stop_futures) == 2:
            if all(future.done() for future in self.stop_futures.values()):
                try:
                    results = [future.result() for future in self.stop_futures.values()]
                    if not all(result and result.success for result in results):
                        raise RuntimeError("; ".join(result.message if result else "missing stop response"
                                                     for result in results))
                except Exception as error:
                    # A stop sent while the Servo J driver restores NRT can be
                    # rejected transiently. Retry and keep the start gate closed.
                    self.stop_retry_error = str(error)
                    self.event("stop_retry", error=self.stop_retry_error)
                    self.stop_futures = {}
                    return
                # Wait until the Servo J driver's own cleanup has completed.
                if any(status not in ("idle_power_on",) and not status.startswith("error")
                       for status in self.driver_status.values()):
                    return
                if any(status.startswith("error") for status in self.driver_status.values()):
                    self.stop_destination = "FAULT"
                if self.suction is not None:
                    io = self.suction.state()
                    if io["pending_operation"]:
                        return
                    if io["error"]:
                        self.stop_destination = "FAULT"
                        self.reason += "; suction: " + io["error"]
                    elif not io.get("settled", True):
                        return
                self.stop_retry_error = ""
                self.event("round_end", accepted_actions=self.accepted, servo_commands=self.servo_commands,
                           limited_ticks=self.limited_ticks, duration_sec=now - self.round_start,
                           last_command=self.previous)
                self.transition(self.stop_destination, self.reason)

    def tick(self):
        now = time.monotonic()
        dt, self.last_tick = now - self.last_tick, now
        try:
            if self.phase == "STOPPING":
                self.stop_tick()
                return
            if self.phase in ("IDLE", "FAULT"):
                return
            measured = self.feedback()
            if self.phase in ("INITIALIZING", "HOMING", "VERIFY_HOME"):
                if now - self.phase_start > self.config["home_timeout_seconds"]:
                    raise TimeoutError("Homing timed out")
                if self.phase == "INITIALIZING" and all(f.done() for f in self.pending):
                    result = self.pending[0].result()
                    if not result or not result.success:
                        raise RuntimeError("Initialization failed: " + (result.message if result else "no response"))
                    self.pending = []
                    for side, client in self.move_clients.items():
                        request = MoveAbsJ.Request()
                        request.joint_positions = self.config["home_" + side]
                        request.speed_mm_s = self.config["home_speed_mm_s"]
                        request.zone_mm = 0.0
                        request.joint_speed_scale = self.config["home_joint_speed_scale"]
                        request.joint_tolerance_rad = self.config["home_tolerance_rad"]
                        request.stable_samples, request.poll_interval_ms, request.timeout_s = 3, 50, 60.0
                        self.pending.append(client.call_async(request))
                    self.transition("HOMING", "MoveAbsJ to the saved dataset B-key pose")
                elif self.phase == "HOMING" and all(f.done() for f in self.pending):
                    if not all(f.result() and f.result().success for f in self.pending):
                        raise RuntimeError("Home MoveAbsJ failed: " + "; ".join(
                            f.result().message if f.result() else "no response" for f in self.pending))
                    self.transition("VERIFY_HOME", "Checking fresh measured joint feedback after both moves")
                elif self.phase == "VERIFY_HOME":
                    target = self.config["home_left"] + self.config["home_right"]
                    fresh = min(self.joints[side][1] for side in ("left", "right")) > self.phase_start
                    if fresh and max(abs(a - b) for a, b in zip(measured, target)) <= self.config["home_tolerance_rad"]:
                        self.previous = list(measured)
                        self.event("home_verified", measured=measured)
                        self.transition("WAITING_ACTION", "Home verified; waiting for a fresh action from this round")
                return
            if self.phase == "WAITING_ACTION":
                if now - self.phase_start > 5.0:
                    raise TimeoutError("No valid first action after homing")
                if self.action is not None:
                    self.previous = list(measured)
                    self.driver_status, self.driver_received = {}, {}
                    self.transition("STARTING_SERVO", "Starting Servo J at measured joints before moving targets")
                return
            if self.phase == "STARTING_SERVO":
                if now - self.phase_start > 3.0:
                    raise TimeoutError("Servo J driver did not acknowledge both arms")
                if self.last_action is None or now - self.last_action > self.config["command_timeout_seconds"]:
                    raise TimeoutError("Policy action watchdog while starting Servo J")
                self.send_servo(True)
                if (all(self.driver_status.get(side) == "running" for side in ("left", "right"))
                        and self.native_trace.get("active") and self.native_trace.get("hardware_commands_sent")):
                    self.round_start = now
                    limit = self.config["max_round_seconds"]
                    duration = "manual stop; no round time limit" if limit is None else f"{limit:g}-second limit starts now"
                    self.transition("RUNNING", "Both Servo J arms running; " + duration)
                return
            if self.phase in ("RUNNING", "PREVIEW"):
                if round_time_expired(now - self.round_start, self.config):
                    self.request_stop(f"Round reached configured {self.config['max_round_seconds']:g}-second time limit")
                    return
                if self.last_action is None:
                    if now - self.phase_start > 5.0:
                        raise TimeoutError("No valid preview action")
                    return
                if now - self.last_action > self.config["command_timeout_seconds"]:
                    raise TimeoutError("Policy action watchdog: no fresh command")
                if self.phase == "RUNNING":
                    if dt > 0.05:
                        raise TimeoutError("Servo target loop was delayed more than 50 ms")
                    if any(now - self.driver_received.get(side, 0.0) > 0.2 for side in ("left", "right")):
                        raise TimeoutError("Servo J driver status is stale")
                    if self.suction is not None and self.suction.state()["error"]:
                        raise RuntimeError(self.suction.state()["error"])
                if self.native_received and now-self.native_received > .25:
                    raise TimeoutError("Native trajectory trace is stale")
                if self.phase == "RUNNING":
                    self.send_servo(True)
                    if self.suction is not None and self.native_trace.get("active"):
                        self.suction.request(self.run_id, self.shadow_suction)
        except Exception as error:
            self.request_stop(str(error), fault=True)

    def publish_status(self):
        now = time.monotonic()
        heartbeat = str(uuid.uuid4())
        self.heartbeats = {key: issued for key, issued in self.heartbeats.items() if now - issued < 2.0}
        self.heartbeats[heartbeat] = now
        value = {"schema": "policy_executor_status_v1", "phase": self.phase, "mode": self.mode,
                 "heartbeat": heartbeat,
                 "run_id": self.run_id, "model": self.config["model"], "reason": self.reason,
                 "enable_motion": self.config["enable_motion"], "max_round_seconds": self.config["max_round_seconds"],
                 "round_time_limit_enabled": self.config["max_round_seconds"] is not None,
                 "accepted_actions": self.accepted, "rejected_actions": self.rejected,
                 "servo_commands": self.servo_commands, "limited_ticks": self.limited_ticks,
                 "last_action_age_ms": None if self.last_action is None else round((now - self.last_action) * 1000, 2),
                 "round_elapsed_seconds": round(now - self.round_start, 3) if self.phase in ("RUNNING", "PREVIEW") else None,
                 "driver_status": self.driver_status, "last_command": self.previous,
                 "trajectory_ready": (now-self.trajectory_received < 2.5 and
                                      self.trajectory_status.get("protocol") == "timed_chunk_v2" and
                                      self.trajectory_status.get("ready",False)),
                 "native_protocol": "timed_chunk_v2",
                 "max_joint_speed_rad_s": self.config["max_joint_speed_rad_s"],
                 "max_joint_acceleration_rad_s2": self.config["max_joint_acceleration_rad_s2"],
                 "max_relative_target_rad": self.config["max_relative_target_rad"],
                 "max_tracking_error_rad": self.config["max_tracking_error_rad"],
                 "temporal_ensemble_coefficient": self.config["temporal_ensemble_coefficient"],
                 "trajectory_response_seconds": self.config["trajectory_response_seconds"],
                 "execution_lead_ns": int(self.config["execution_lead_seconds"]*1e9),
                 "fusion_contributors": self.native_trace.get("contributors",0),
                 "newest_prediction_index": self.native_trace.get("newest_index"),
                 "native_error": self.native_trace.get("error",""),
                 "feedback_reader": self.trajectory_status.get("feedback_reader","unknown"),
                 "feedback_max_host_age_ms": self.trajectory_status.get("feedback_max_host_age_ms"),
                 "left_feedback": self.native_trace.get("left_feedback"),
                 "right_feedback": self.native_trace.get("right_feedback"),
                 "stop_retry_error": self.stop_retry_error,
                 "suction_target": self.shadow_suction,
                 "suction_io": None if self.suction is None else self.suction.state()}
        self.status_pub.publish(String(data=encode(value)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=Path(__file__).with_name("executor_events.jsonl"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Executor(config, args.log)
    exit_requested = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: exit_requested.set())
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if exit_requested.is_set():
                if node.phase in ("IDLE", "FAULT"):
                    break
                node.request_stop("Executor process shutting down")
                if node.phase in ("IDLE", "FAULT"):
                    break
    finally:
        if node.suction is not None:
            node.suction.close()
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
