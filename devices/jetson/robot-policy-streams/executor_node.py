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
from rcl_interfaces.srv import SetParametersAtomically
from rokae_interfaces.msg import DualArmServoJ
from rokae_interfaces.srv import MoveAbsJ
import yaml

from protocol import ACTIVE_PHASES, JOINT_NAMES, PREFIX, encode, finite_vector
from protocol import suction_target, validate_chunk, validate_joints, validate_config, round_time_expired
from suction_io import Suction
from stage_policy import checkpoint_map, selected_policy, action_config
from brain_protocol import validate_brain_completion
from manipulation_progress import ManipulationProgress
from suction_phase_guard import FarReleaseGuard
from far_arm_coordination import FarArmCoordination


class Executor(Node):
    def __init__(self, config, log_path):
        super().__init__("robot_policy_executor")
        self.config = validate_config(config)
        self.phase, self.mode, self.run_id = "IDLE", "none", ""
        self.home_verified_run_id = ""
        self.home_verified_time_ns = 0
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
        self.sequence_job = None
        self.manipulation_progress = ManipulationProgress()
        self.progress_run_id = None
        self.suction_release_guard = FarReleaseGuard()
        self.far_arm_coordination = FarArmCoordination()
        self.brain_status, self.brain_status_received = {}, 0.
        self.stop_wait_native = False
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
                     self.create_subscription(String, PREFIX + "/brain_status", self.on_brain_status, 1),
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
        self.services_kept += [self.create_service(Trigger, PREFIX + "/start_sequence", self.start_sequence),
                              self.create_service(Trigger, PREFIX + "/start_sequence_brain", self.start_sequence_brain),
                              self.create_service(SetParametersAtomically, PREFIX + "/brain_finish", self.finish_from_brain),
                              self.create_service(Trigger, PREFIX + "/start_near", self.start_near),
                              self.create_service(Trigger, PREFIX + "/finish_far",
                                  lambda _, response: self.finish_stage("far", response)),
                              self.create_service(Trigger, PREFIX + "/finish_near",
                                  lambda _, response: self.finish_stage("near", response))]
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
        return validate_joints(self.joints["left"][0] + self.joints["right"][0], self.config["joint_limit_margin_rad"],
                               self.config.get("joint_upper_limits_rad"))

    def on_inference(self, message):
        try:
            self.inference_status, self.inference_received = json.loads(message.data), time.monotonic()
        except ValueError:
            pass

    def begin_sequence(self, stage, response):
        try:
            self.ready(True)
            if any(value[0] for value in self.suction_feedback.values()):
                raise ValueError("Both suction channels must be released before a sequence starts")
            if not self.config["suction_enabled"]:
                raise ValueError("Sequence requires verified suction control")
            result = self.begin(True, response)
            if result.success:
                self.sequence_job = {"id": str(uuid.uuid4()), "stage": stage, "result": "running",
                                     "pending": None, "completed_stages": [],
                                     "completion_mode": "manual"}
                self.event("sequence_begin", sequence=dict(self.sequence_job))
                self.publish_status()
                result.message = (f"Accepted sequence {self.sequence_job['id']}: returning both arms "
                                  "to dataset home first, then running the stage; completion is manual")
            return result
        except Exception as error:
            response.success, response.message = False, str(error)
            return response

    def start_sequence(self, _, response):
        return self.begin_sequence("far", response)

    def on_brain_status(self, message):
        try:
            self.brain_status, self.brain_status_received = json.loads(message.data), time.monotonic()
        except ValueError:
            pass

    def start_sequence_brain(self, _, response):
        expected = self.config.get("brain_adapter_sha256")
        if (not expected or time.monotonic()-self.brain_status_received > .5 or
                not self.brain_status.get("ready") or not self.brain_status.get("transitions_enabled") or
                self.brain_status.get("adapter_sha256") != expected):
            response.success, response.message = False, "Matching LoRA brain must be ready in transition mode"
            return response
        response = self.begin_sequence("far", response)
        if response.success:
            self.sequence_job["completion_mode"] = "qwen_lora"
            response.message = (f"Accepted brain sequence {self.sequence_job['id']}: home BOTH arms before far ACT; "
                                "LoRA brain will finish far, home BOTH arms again, then finish near")
            self.publish_status()
        return response

    def finish_from_brain(self, request, response):
        try:
            if self.config.get("require_manipulation_sweep_for_brain", False):
                if (self.progress_run_id != self.run_id or not self.manipulation_progress.ready
                        or time.monotonic()-self.native_received > .25):
                    raise ValueError("ACT manipulation sweep is not finished; premature visual completion blocked")
            value = {}
            for parameter in request.parameters:
                if parameter.name in value: raise ValueError("Duplicate brain parameter")
                if parameter.value.type == 4:
                    value[parameter.name] = parameter.value.string_value
                elif parameter.value.type == 2:
                    value[parameter.name] = parameter.value.integer_value
                else: raise ValueError("Unexpected brain parameter type")
            stage = validate_brain_completion(value,
                {"phase": self.phase, "run_id": self.run_id, "task_sequence": self.sequence_job}, self.config,
                now_ns=time.time_ns(), elapsed_seconds=time.monotonic()-self.round_start,
                home_after_ns=self.home_verified_time_ns)
            result = self.finish_stage(stage, Trigger.Response(), source="qwen_lora")
            response.result.successful, response.result.reason = result.success, result.message
        except Exception as error:
            response.result.successful, response.result.reason = False, str(error)
        return response

    def start_near(self, _, response):
        return self.begin_sequence("near", response)

    def finish_stage(self, expected_stage, response, source="operator_service"):
        # Separate endpoints prevent a repeated far request from finishing near.
        if (self.phase != "RUNNING" or not self.sequence_job or
                self.sequence_job.get("result") != "running" or
                self.sequence_job.get("stage") != expected_stage or
                self.sequence_job.get("pending") is not None):
            response.success, response.message = False, (
                f"Cannot finish {expected_stage}: phase={self.phase}, "
                f"stage={(self.sequence_job or {}).get('stage')}")
            return response
        self.event("stage_complete", stage=expected_stage, completion_source=source)
        self.sequence_job["completed_stages"].append(expected_stage)
        self.sequence_job["pending"] = "near" if expected_stage == "far" else "done"
        self.request_stop(("LoRA brain confirmed " if source == "qwen_lora" else "Operator confirmed ")
                          + expected_stage + " stage completion", advance=True)
        response.success = True
        response.message = ("Accepted: stopping and releasing suction, then homing BOTH arms for near ACT"
                            if expected_stage == "far" else
                            "Accepted: stopping and releasing suction to finish the sequence")
        return response

    def sequence_tick(self):
        # Manual stages end only through finish_far/finish_near or an explicit
        # stop. Keep the bounded task timeout for automatic brain sequences.
        if ((self.sequence_job or {}).get("completion_mode") == "qwen_lora" and
                time.monotonic()-self.round_start > 90.):
            raise TimeoutError("Automatic stage exceeded 90 s without brain completion; sequence cancelled")

    def sequence_after_stop(self):
        # This is reached only after both arms acknowledged idle_power_on.
        self.suction.request(self.run_id + ":stage_release", [False, False])
        self.shadow_suction = [False, False]
        self.transition("STAGE_RELEASE", "Verifying both suction channels released before stage transition")

    def release_tick(self):
        io = self.suction.state()
        if io["error"]:
            raise RuntimeError("Stage suction release failed: " + io["error"])
        if time.monotonic()-self.phase_start > 3.:
            raise TimeoutError("Stage suction release was not acknowledged")
        fresh_off = all(not value[0] and value[1] > self.phase_start for value in self.suction_feedback.values())
        if (time.monotonic()-self.phase_start < .5 or not fresh_off or io["pending_operation"]
                or not io.get("settled", False) or any(io["verified"])):
            return
        if self.sequence_job["pending"] == "done":
            self.sequence_job.update(result="complete", pending=None)
            self.event("sequence_complete", sequence=dict(self.sequence_job))
            self.transition("IDLE", "Operator finished sequence; stopped with suction released")
            return
        old_run = self.run_id
        self.run_id = str(uuid.uuid4())
        self.home_verified_run_id = ""
        self.home_verified_time_ns = 0
        self.publisher_id, self.last_sequence, self.action, self.last_action = None, 0, None, None
        self.native_trace, self.native_received = {}, 0.
        self.heartbeats.clear()
        self.previous = list(self.feedback())
        self.sequence_job.update(stage="near", pending=None)
        self.event("stage_reset", old_run_id=old_run, new_run_id=self.run_id, stage="near")
        self.pending = [self.init_client.call_async(Trigger.Request())]
        self.transition("INITIALIZING", "Far stage stopped and suction released; return BOTH arms to dataset home for near ACT")

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
        if self.trajectory_status.get("last_error"):
            raise ValueError("Native trajectory driver fault; reconnect driver before starting: " +
                             str(self.trajectory_status["last_error"]))
        if live and self.trajectory_status.get("feedback_reader") != "drain_latest_v1":
            raise ValueError("Native driver lacks latest-feedback FIFO fix; restart the updated driver")
        scale = self.config.get("execution_speed_scale", 1.)
        profile = self.config.get("execution_speed_profile", "uniform")
        if profile != "uniform" and not self.trajectory_status.get("continuous_action_clock_v1"):
            raise ValueError("Native driver lacks continuous contact-speed profile support")
        if profile == "contact_slow_1p2_v1" and not self.trajectory_status.get("contact_slow_1p2_v1"):
            raise ValueError("Native driver lacks 1.2 contact-speed profile support")
        if (scale != 1. or self.config["max_joint_acceleration_rad_s2"] > 2.) and (
                self.trajectory_status.get("execution_time_scaling") != "per_observation_v1"
                or self.trajectory_status.get("maximum_execution_speed_scale", 1.) < scale
                or self.trajectory_status.get("maximum_joint_acceleration_rad_s2", 2.) < self.config["max_joint_acceleration_rad_s2"]):
            raise ValueError("Native driver lacks requested execution speed/acceleration support; restart the updated driver")
        if (time.monotonic() - self.inference_received > 2.5 or
                not self.inference_status.get("ready") or self.inference_status.get("error")):
            raise ValueError("Inference node is not ready")
        if self.inference_status.get("model") != self.config["model"]:
            raise ValueError("Inference and executor model settings differ")
        if self.inference_status.get("execution_speed_scale", 1.) != scale:
            raise ValueError("Inference and executor execution speed scales differ")
        if self.inference_status.get("execution_speed_profile", "uniform") != profile:
            raise ValueError("Inference and executor execution speed profiles differ")
        if self.inference_status.get("checkpoint_sha256") != self.config["checkpoint_sha256"][self.config["model"]]:
            raise ValueError("Inference weights differ from the reviewed checkpoint")
        if self.config.get("act_stage_checkpoints") and self.inference_status.get("loaded_checkpoints") != checkpoint_map(self.config):
            raise ValueError("Both reviewed stage checkpoints must be loaded and warmed")
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
            self.sequence_job = None
            if live and self.config["suction_enabled"]:
                if self.suction is not None:
                    self.suction.close()
                self.suction = Suction(self.config, [self.suction_feedback[side][0]
                                                     for side in ("left", "right")])
            self.mode, self.run_id = ("live" if live else "preview"), str(uuid.uuid4())
            self.home_verified_run_id = ""
            self.home_verified_time_ns = 0
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
            if self.mode == "live" and self.home_verified_run_id != self.run_id:
                raise ValueError("Both arms must be verified at home before policy actions")
            policy_key, effective_config = action_config(self.config, self.sequence_job)
            if self.config.get("act_stage_checkpoints") and value.get("policy_key") != policy_key:
                raise ValueError("Chunk belongs to the wrong stage checkpoint")
            actions = validate_chunk(value, self.run_id, self.publisher_id or expected,
                                     self.last_sequence, effective_config, receive_ns)
            issued = self.heartbeats.get(value.get("executor_heartbeat"))
            if issued is None or time.monotonic()-issued > self.config["heartbeat_timeout_seconds"]:
                raise ValueError("Chunk echoes a missing or expired Jetson heartbeat")
            coordinate = (self.mode == "live" and self.config.get("coordinate_far_right_arm", False)
                          and (self.sequence_job or {}).get("stage") == "far"
                          and (self.sequence_job or {}).get("completion_mode") == "qwen_lora")
            if coordinate:
                raw_first_right = list(actions[0][7:14])
                anchor = (self.feedback() if self.far_arm_coordination.run_id != self.run_id
                          or self.far_arm_coordination.hold_target is None else [])
                actions = self.far_arm_coordination.apply(actions, run_id=self.run_id,
                    verified_anchor=anchor, enabled=True,
                    left_lift_seen=(self.progress_run_id == self.run_id
                                    and self.suction_release_guard.left_lift_seen))
                value['actions'] = actions
                value['raw_first_right_target'] = raw_first_right
                value['far_arm_coordination'] = self.far_arm_coordination.status()
            else:
                self.far_arm_coordination.apply(actions,run_id=self.run_id,verified_anchor=[],
                                                enabled=False,left_lift_seen=False)
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
                if self.progress_run_id != self.run_id:
                    self.progress_run_id = self.run_id
                    self.manipulation_progress.reset((self.sequence_job or {}).get("stage"))
                    self.suction_release_guard.reset()
                measured = value.get("left_measured", []) + value.get("right_measured", [])
                if len(measured) == 14:
                    self.manipulation_progress.update(measured, time.monotonic())
                self.previous = validate_joints(value["command"], self.config["joint_limit_margin_rad"],
                                                self.config.get("joint_upper_limits_rad"))
                self.limited_ticks += int(max(abs(a-b) for a,b in zip(value["command"],value["fused_target"][:14]))>1e-7)
                predicted_suction = [suction_target(target, previous, self.config) for target,previous in
                                     zip(value["fused_target"][14:],self.shadow_suction)]
                guard_enabled = (self.config.get("guard_far_suction_release", False) and
                                 (self.sequence_job or {}).get("completion_mode") == "qwen_lora" and
                                 (self.sequence_job or {}).get("stage") == "far")
                if len(measured)==14:
                    self.suction_release_guard.update_measured(measured)
                was_blocked=self.suction_release_guard.waiting_since is not None
                self.shadow_suction=self.suction_release_guard.filter(predicted_suction,
                    now=time.monotonic(),enabled=guard_enabled)
                if was_blocked != (self.suction_release_guard.waiting_since is not None):
                    self.event("suction_release_interlock",
                               model_target=predicted_suction,applied_target=self.shadow_suction,
                               guard=self.suction_release_guard.status())
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

    def request_stop(self, reason, fault=False, advance=False):
        if self.sequence_job and self.sequence_job["result"] == "running" and not advance:
            self.sequence_job.update(pending=None, result="fault" if fault else "cancelled")
        if self.phase == "IDLE":
            return
        if self.phase == "STOPPING":
            if reason == "Manual stop" and self.mode == "live" and self.suction is not None:
                self.suction.release_left(self.run_id)
                self.shadow_suction[0] = False
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
        self.stop_wait_native = self.phase in ("RUNNING", "STARTING_SERVO")
        self.stop_started, self.last_stop_request = time.monotonic(), 0.0
        self.stop_futures, self.final_stop_sent = {}, False
        self.stop_retry_error = ""
        self.transition("STOPPING", reason)
        self.send_servo(False)

    def stop_tick(self):
        self.send_servo(False)
        now = time.monotonic()
        # Let the RT owner finish first; concurrent NRT stop/reset used to race
        # ServoJ cleanup. Fall back to stop services after 3 s on missing status.
        if self.stop_wait_native:
            cleaned = all(self.driver_received.get(side, 0.) > self.stop_started and
                          (self.driver_status.get(side) == "idle_power_on" or
                           self.driver_status.get(side, "").startswith("error"))
                          for side in ("left", "right"))
            if not cleaned and now-self.stop_started < 3.:
                return
            self.stop_wait_native = False
            if not cleaned:
                self.stop_destination = "FAULT"
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
                    self.reason += "; " + "; ".join(side + ":" + status for side, status in self.driver_status.items()
                                                     if status.startswith("error"))
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
                if (self.sequence_job and self.sequence_job.get("pending") and
                        self.stop_destination == "IDLE"):
                    self.sequence_after_stop()
                else:
                    if self.sequence_job and self.stop_destination == "FAULT":
                        self.sequence_job.update(result="fault", pending=None)
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
            if self.phase == "STAGE_RELEASE":
                self.release_tick()
                return
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
                        self.home_verified_run_id = self.run_id
                        self.home_verified_time_ns = time.time_ns()
                        self.event("home_verified", measured=measured)
                        self.transition("WAITING_ACTION", "Both arms home verified; waiting for fresh post-home policy input")
                return
            if self.phase == "WAITING_ACTION":
                home = self.config["home_left"] + self.config["home_right"]
                if self.home_verified_run_id != self.run_id or max(abs(a-b) for a,b in zip(measured, home)) > self.config["home_tolerance_rad"]:
                    raise RuntimeError("Both arms must remain at verified home before first policy action")
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
                    if self.sequence_job and self.sequence_job["result"] == "running":
                        self.sequence_tick()
        except Exception as error:
            self.request_stop(str(error), fault=True)

    def publish_status(self):
        now = time.monotonic()
        policy_key, checkpoint_sha = selected_policy(self.config, self.sequence_job)
        heartbeat = str(uuid.uuid4())
        self.heartbeats = {key: issued for key, issued in self.heartbeats.items() if now - issued < 2.0}
        self.heartbeats[heartbeat] = now
        value = {"schema": "policy_executor_status_v1", "phase": self.phase, "mode": self.mode,
                 "heartbeat": heartbeat,
                 "run_id": self.run_id, "model": self.config["model"], "reason": self.reason,
                 "policy_key": policy_key, "expected_checkpoint_sha256": checkpoint_sha,
                 "home_verified_run_id": self.home_verified_run_id,
                 "policy_observation_after_ns": self.home_verified_time_ns,
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
                 "execution_speed_scale": self.config.get("execution_speed_scale", 1.),
                 "execution_speed_profile": self.config.get("execution_speed_profile", "uniform"),
                 "effective_execution_speed_scale": self.native_trace.get("effective_execution_speed_scale",self.config.get("execution_speed_scale",1.)),
                 "motion_speed_phase": self.native_trace.get("motion_speed_phase","waiting"),
                 "effective_action_step_ms": 50. / self.config.get("execution_speed_scale", 1.),
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
                 "suction_release_guard": self.suction_release_guard.status(),
                 "far_arm_coordination": self.far_arm_coordination.status(),
                 "task_sequence": self.sequence_job,
                 "box_scene_check_enabled": False,
                 "completion_mode": (self.sequence_job or {}).get("completion_mode", "manual"),
                 "require_manipulation_sweep_for_brain": self.config.get("require_manipulation_sweep_for_brain", False),
                 "manipulation_progress": (self.manipulation_progress.status()
                                           if self.progress_run_id == self.run_id else None),
                 "brain_adapter_sha256": self.config.get("brain_adapter_sha256"),
                 "stage_timeout_seconds": (90. if (self.sequence_job or {}).get("completion_mode") == "qwen_lora" else None),
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
