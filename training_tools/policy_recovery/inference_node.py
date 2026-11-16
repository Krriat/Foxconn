#!/usr/bin/env python3
"""GPU policy publisher. This process has no hardware or motion-service clients."""
import argparse
from collections import deque
import hashlib
import importlib.util
from pathlib import Path
import signal
import threading
import time
import uuid

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String
import yaml

from protocol import ACTIVE_PHASES, PREFIX, encode, validate_config
from joint_anchor import anchor_joint_chunk


class Inference(Node):
    def __init__(self, config, runtime, predict, reset, weights_sha):
        super().__init__("robot_policy_inference")
        self.config, self.runtime = config, runtime
        self.anchor_joints = config.get("anchor_joint_predictions", False)
        if type(self.anchor_joints) is not bool:
            raise ValueError("anchor_joint_predictions must be boolean")
        self.predict, self.reset, self.weights_sha = predict, reset, weights_sha
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.head_event = threading.Event()
        self.snapshot_failure = "waiting_for_inputs"
        self.last_pause_reason = None
        self.identity = str(uuid.uuid4())
        self.executor_context, self.context_received = {}, 0.0
        self.buffers = {name: deque(maxlen=50) for name in
                        ("head", "left_wrist", "right_wrist", "left_arm", "right_arm",
                         "left_suction", "right_suction")}
        self.status = {"ready": False, "model": config["model"], "publisher_id": self.identity,
                       "checkpoint_sha256": weights_sha, "actions_published": 0,
                       "chunks_predicted": 0, "skipped_action_ticks": 0,
                       "error": "", "last_inference_ms": None,
                       "joint_postprocess": "anchor_observed_joints_v1" if self.anchor_joints else "absolute"}
        self.subs = []
        for key in self.buffers:
            if key in ("head", "left_wrist", "right_wrist"):
                msg, topic = CompressedImage, f"/robot/sensors/cameras/{key}/color/image_raw/compressed"
            elif key.endswith("arm"):
                msg, topic = JointState, f"/robot/upperlimb/joint_states/{key}"
            else:
                msg, topic = Bool, f"/robot/suction/{key.split('_')[0]}/state"
            self.subs.append(self.create_subscription(
                msg, topic, lambda message, key=key: self.receive(key, message), qos_profile_sensor_data))
        self.subs.append(self.create_subscription(String, PREFIX + "/status", self.on_context, 1))
        action_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        self.actions = self.create_publisher(String, PREFIX + "/action_chunk", action_qos)
        self.info = self.create_publisher(String, PREFIX + "/inference_status", 1)
        self.create_timer(1.0, self.publish_status)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def receive(self, key, message):
        received = time.monotonic()
        try:
            stamp = None
            if hasattr(message, "header"):
                stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
                if stamp <= 0:
                    raise ValueError("missing source timestamp")
            if key in ("head", "left_wrist", "right_wrist"):
                bgr = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None or bgr.shape != (480, 640, 3):
                    raise ValueError("expected 640x480 color image")
                value = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif key.endswith("arm"):
                value = self.runtime.joint_vector(message, key.split("_")[0])
            else:
                value = float(message.data)
            with self.lock:
                self.buffers[key].append((stamp, received, value))
            if key == "head":
                self.head_event.set()
        except Exception as error:
            with self.lock:
                self.status["error"] = f"{key}: {error}"

    def on_context(self, message):
        import json
        try:
            value = json.loads(message.data)
            with self.lock:
                self.executor_context, self.context_received = value, time.monotonic()
        except (ValueError, TypeError):
            pass

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            if any(not values for values in self.buffers.values()):
                self.snapshot_failure = "missing_inputs"
                return None
            latest_age = max(now - values[-1][1] for values in self.buffers.values())
            if latest_age > self.config["input_max_age_seconds"]:
                self.snapshot_failure = "input_receive_age_exceeded"
                return None
            for head in reversed(self.buffers["head"]):
                selected = {"head": head}
                for key in ("left_wrist", "right_wrist", "left_arm", "right_arm"):
                    selected[key] = min(self.buffers[key], key=lambda x: abs(x[0] - head[0]))
                stamps = [value[0] for value in selected.values()]
                if max(stamps) - min(stamps) > self.config["input_max_skew_seconds"]:
                    continue
                selected.update({key: self.buffers[key][-1] for key in ("left_suction", "right_suction")})
                if max(now - value[1] for value in selected.values()) > self.config["input_max_age_seconds"]:
                    continue
                self.snapshot_failure = ""
                return {"images": [selected[key][2] for key in ("head", "left_wrist", "right_wrist")],
                        "state": np.concatenate([selected["left_arm"][2], selected["right_arm"][2],
                                                 [selected["left_suction"][2], selected["right_suction"][2]]]).astype(np.float32),
                        "source_stamp": head[0], "source_stamps_ns": {key: int(value[0]*1e9) for key, value in selected.items() if value[0] is not None}, "source_skew_ms": (max(stamps) - min(stamps)) * 1000,
                        "latest_age_ms": latest_age * 1000}
        self.snapshot_failure = "no_fresh_synchronized_observation"
        return None

    def active_context(self):
        with self.lock:
            context = dict(self.executor_context)
            age = time.monotonic() - self.context_received
        if age > 0.4 or context.get("phase") not in ACTIVE_PHASES:
            return None
        if context.get("model") != self.config["model"]:
            return None
        return context

    def run(self):
        run_id, sequence, last_observation = None, 0, 0
        warmed = False
        while not self.stop_event.is_set():
            observation = self.snapshot()
            if not warmed:
                if observation is None:
                    self.stop_event.wait(.05)
                    continue
                try:
                    for _ in range(5):
                        self.predict(observation)
                    self.reset()
                    warmed = True
                    with self.lock:
                        self.status.update(ready=True, error="")
                except Exception as error:
                    with self.lock:
                        self.status.update(ready=False, error=str(error))
                    self.stop_event.wait(1.)
                continue
            context = self.active_context()
            if context is None or observation is None:
                with self.lock:
                    reason = (self.snapshot_failure if observation is None else
                              "executor_phase_" + str(self.executor_context.get("phase", "unknown")))
                    if observation is not None and time.monotonic()-self.context_received > .4:
                        reason = "executor_heartbeat_expired"
                    self.status["ready"] = observation is not None
                    self.status["error"] = "" if observation is not None else "No complete fresh observation"
                    self.status["pause_reason"] = reason
                    if reason != self.last_pause_reason:
                        self.last_pause_reason = reason
                        print(encode({"event":"inference_paused","time_ns":time.time_ns(),"reason":reason,
                            "receive_ages_ms":{k:(time.monotonic()-v[-1][1])*1000 if v else None for k,v in self.buffers.items()}}),flush=True)
                # A brief input/heartbeat gap is still the same executor round.
                # Preserve sequence and observation ordering until a new UUID
                # arrives; resetting here would make valid resumed chunks look
                # like replayed messages to the executor.
                self.stop_event.wait(.01)
                continue
            if context["run_id"] != run_id:
                run_id, sequence, last_observation = context["run_id"], 0, 0
                self.reset()
            with self.lock:
                self.status["pause_reason"] = ""
            self.last_pause_reason = None
            # The camera publishes at 20 Hz. Follow new frames instead of a
            # second free-running 20-Hz timer, which can repeatedly sample the
            # same frame near a phase boundary and lose an entire period.
            if not self.head_event.wait(.05):
                continue
            self.head_event.clear()
            try:
                observation = self.snapshot()
                if observation is None:
                    continue
                origin_ns = int(observation["source_stamp"] * 1e9)
                if origin_ns <= last_observation:
                    continue  # never re-predict the same frame under a new timeline
                start_ns, start_mono = time.time_ns(), time.monotonic_ns()
                chunk = self.predict(observation)
                raw_first_joint_target = chunk[0, :14].copy()
                if self.anchor_joints:
                    chunk = anchor_joint_chunk(chunk, observation["state"])
                end_ns = time.time_ns()
                inference_ms = (time.monotonic_ns()-start_mono)/1e6
                current = self.active_context()
                if current is None or current["run_id"] != run_id:
                    continue
                sequence += 1
                message = {"schema": "dual_arm_suction_chunk_v2", "run_id": run_id,
                           "publisher_id": self.identity, "sequence": sequence,
                           "executor_heartbeat": current["heartbeat"],
                           "model": self.config["model"], "checkpoint_sha256": self.weights_sha,
                           "action_hz": self.config["action_hz"], "step_ns": 50_000_000,
                           "actions": chunk.tolist(), "observation_time_ns": origin_ns,
                           "joint_postprocess": "anchor_observed_joints_v1" if self.anchor_joints else "absolute",
                           "raw_first_joint_target": raw_first_joint_target.tolist(),
                           "source_stamps_ns": observation["source_stamps_ns"],
                           "observation_state": observation["state"].tolist(),
                           "inference_start_ns": start_ns, "inference_end_ns": end_ns,
                           "inference_ms": inference_ms,
                           "observation_source_skew_ms": observation["source_skew_ms"],
                           "max_input_receive_age_ms": observation["latest_age_ms"],
                           "send_time_ns": time.time_ns()}
                self.actions.publish(String(data=encode(message)))
                last_observation = origin_ns
                with self.lock:
                    self.status.update(error="", run_id=run_id, last_inference_ms=round(inference_ms, 3),
                                       protocol="timed_chunk_v2", chunk_steps=50, prediction_hz=20,
                                       observation_state_logged=True)
                    self.status["chunks_predicted"] += 1
                    self.status["actions_published"] += 50
            except Exception as error:
                with self.lock:
                    self.status["error"] = str(error)
                self.stop_event.wait(.05)

    def publish_status(self):
        with self.lock:
            status = dict(self.status)
            status["input_receive_age_ms"] = {
                key: round((time.monotonic() - values[-1][1]) * 1000, 2) if values else None
                for key, values in self.buffers.items()}
        self.info.publish(String(data=encode(status)))


def load_runtime(config):
    root = Path(config["act_root"])
    path = root / "training_tools/test_ros_models.py"
    spec = importlib.util.spec_from_file_location("validated_policy_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    torch = runtime.torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if config["model"] == "act":
        checkpoint = root / "model_outputs/act_suction_20261111_213453/checkpoints/best/pretrained_model"
        meta = runtime.LeRobotDatasetMetadata("local/dual_arm_suction_train", root=root / "prepared_data/suction_20hz_v1/train")
        policy, pre, post, _ = runtime.load_policy(checkpoint, meta, "cuda")
        def predict(observation):
            raw = {"observation.state": observation["state"]}
            raw.update({f"observation.images.{key}": value for key, value in
                        zip(("head", "left_wrist", "right_wrist"), observation["images"])})
            with torch.inference_mode():
                batch = pre(runtime.prepare_observation_for_inference(raw, torch.device("cuda"), task="", robot_type=""))
                result = post(policy.predict_action_chunk(batch)).float().cpu().numpy()[0]
            return checked(result)
        def reset():
            policy.reset()
            pre.reset()
            post.reset()
    elif config["model"] == "interact":
        checkpoint = root / "model_outputs/interact_suction_20261111_234920/checkpoints/best/pretrained_model"
        policy = runtime.InterACT3DPolicy.from_pretrained(checkpoint).cuda().eval()
        stats = runtime.NormalizationStats.load(checkpoint / "normalization.json")
        def predict(observation):
            with torch.inference_mode():
                images, points, state = runtime.preprocess_observation(
                    *observation["images"], state=observation["state"], stats=stats,
                    image_size=policy.config.image_size, device="cuda")
                normalized = policy.predict_action_chunk(images, points, state)
                result = runtime.denormalize_actions(normalized[0], stats).float().cpu().numpy()
            return checked(result)
        def reset():
            if hasattr(policy, "reset"):
                policy.reset()
    else:
        raise ValueError("model must be act or interact")
    weights_sha = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    if weights_sha != config["checkpoint_sha256"][config["model"]]:
        raise ValueError("Checkpoint weights differ from reviewed model")
    return runtime, predict, reset, weights_sha


def checked(result):
    if result.shape != (50, 16) or not np.isfinite(result).all():
        raise ValueError("Expected finite 50x16 action chunk")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = validate_config(yaml.safe_load(args.config.read_text()))
    runtime, predict, reset, weights_sha = load_runtime(config)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Inference(config, runtime, predict, reset, weights_sha)
    exit_requested = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: exit_requested.set())
    try:
        while rclpy.ok() and not exit_requested.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.stop_event.set()
        node.thread.join(timeout=5)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
