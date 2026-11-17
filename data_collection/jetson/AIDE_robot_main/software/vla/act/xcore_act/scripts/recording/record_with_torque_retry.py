#!/usr/bin/env python
"""Run lerobot-record with ACT chunk fusion and optional robot helpers.

When ``--udp-video`` is enabled, frames already received from the shared ROS
camera topics are sent to the Edge API preview endpoints.  This process never
opens a camera device directly.

ACT chunk fusion can resample a 30 Hz prediction chunk for a 10 Hz deployment
loop. With the defaults, the wrapper executes raw chunk indices
``0, 3, 6, ..., 27`` and fuses overlapping predictions with old/new weights
``0.3/0.7``.
"""

from __future__ import annotations

import builtins
import copy
import json
import os
import socket
import shutil
import sys
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.so_follower.so_follower import SOFollower
import lerobot.scripts.lerobot_record as lerobot_record
from lerobot.scripts.lerobot_record import main
from lerobot_robot_xcore import XCoreRightArm


_orig_enable_torque = FeetechMotorsBus.enable_torque
_orig_connect = SOFollower.connect
_orig_get_observation = SOFollower.get_observation
_orig_xcore_get_observation = XCoreRightArm.get_observation
_orig_predict_action = lerobot_record.predict_action
_orig_act_reset = ACTPolicy.reset
_start_from_default_pose = True
_chunk_fusion_enabled = False
_chunk_fusion_threshold = 0.8
_chunk_fusion_old_weight = 0.3
_chunk_fusion_action_stride = 3
_chunk_fusion_replan_interval: int | None = None
_chunk_fusion_controllers = []
_udp_video_forwarder = None
_web_preview_publisher = None

DEFAULT_UDP_VIDEO_IP = "172.16.13.184"
DEFAULT_UDP_VIDEO_PORT = 6000
DEFAULT_LEFT_WRIST_UDP_VIDEO_PORT = 6001
DEFAULT_RIGHT_WRIST_UDP_VIDEO_PORT = 6002
MAX_UDP_PAYLOAD = 65_507
DEFAULT_ROS_CAMERA_CONFIG = (
    '{"head":{"type":"ros_image","topic":"/robot/sensors/cameras/head/color/image_raw",'
    '"width":960,"height":540,"fps":30},'
    '"right_wrist":{"type":"ros_image",'
    '"topic":"/robot/sensors/cameras/right_wrist/color/image_raw",'
    '"width":640,"height":480,"fps":30}}'
)

RIGHT_VR_DEFAULT_POSE_DEG = {
    # Mean first-frame state from smolvla/datasets/right_arm_vr_20260703_105451.
    "shoulder_pan": -10.4,
    "shoulder_lift": 17.22,
    "elbow_flex": 34.66,
    "wrist_flex": -48.79,
    "wrist_roll": -89.54,
    "gripper": 1.73,
}


class UdpFrameSender:
    """Encode/send the latest observed frame without blocking robot control."""

    def __init__(
        self,
        ip: str,
        port: int,
        camera: str,
        jpeg_quality: int,
        send_every: int,
        verbose: bool,
    ) -> None:
        self.addr = (ip, port)
        self.camera = camera
        self.jpeg_quality = jpeg_quality
        self.send_every = max(1, send_every)
        self.verbose = verbose
        self.frame_count = 0
        self.sent_count = 0
        self.t0 = time.monotonic()
        self._last_warning_time = 0.0
        self._queue: Queue[tuple[np.ndarray, bool] | None] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"udp-video-{camera}",
            daemon=True,
        )
        self._thread.start()
        print(f"UDP video: forwarding '{camera}' observations to {ip}:{port}")

    def submit(self, frame: np.ndarray, input_is_rgb: bool) -> None:
        self.frame_count += 1
        if self.frame_count % self.send_every != 0:
            return
        item = (np.ascontiguousarray(frame).copy(), input_is_rgb)
        try:
            self._queue.put_nowait(item)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except Full:
                pass

    def _encode(
        self, frame: np.ndarray, input_is_rgb: bool
    ) -> tuple[bytes, int]:
        frame_bgr = (
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if input_is_rgb
            else frame
        )
        quality = self.jpeg_quality
        while quality >= 5:
            ok, buffer = cv2.imencode(
                ".jpg",
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not ok:
                raise RuntimeError("OpenCV JPEG encoding failed")
            data = buffer.tobytes()
            if len(data) <= MAX_UDP_PAYLOAD:
                return data, quality
            quality -= 5
        raise RuntimeError(
            f"JPEG is larger than one UDP datagram ({len(data)} bytes)"
        )

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        try:
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.2)
                except Empty:
                    continue
                if item is None:
                    break
                frame, input_is_rgb = item
                try:
                    data, quality = self._encode(frame, input_is_rgb)
                    sock.sendto(data, self.addr)
                    self.sent_count += 1
                    if self.verbose and self.sent_count % 30 == 0:
                        elapsed = max(time.monotonic() - self.t0, 1e-6)
                        print(
                            f"UDP video {self.camera}: sent={self.sent_count}, "
                            f"fps={self.sent_count / elapsed:.1f}, "
                            f"jpeg={len(data)} bytes, quality={quality}"
                        )
                except Exception as exc:
                    now = time.monotonic()
                    if now - self._last_warning_time >= 5.0:
                        print(f"WARNING: UDP video {self.camera} failed: {exc}")
                        self._last_warning_time = now
        finally:
            sock.close()

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
        self._thread.join(timeout=2.0)


class ObservationUdpVideoForwarder:
    def __init__(self, senders: list[UdpFrameSender]) -> None:
        self.senders = senders

    def send(self, observation: dict[str, Any], camera_configs: dict) -> None:
        for sender in self.senders:
            frame = observation.get(sender.camera)
            if isinstance(frame, np.ndarray):
                config = camera_configs.get(sender.camera)
                color_mode = getattr(config, "color_mode", "rgb")
                color_mode = getattr(color_mode, "value", color_mode)
                sender.submit(frame, str(color_mode).lower() == "rgb")

    def close(self) -> None:
        for sender in self.senders:
            sender.close()


class WebPreviewPublisher:
    """Publish the latest observed camera frames without blocking robot control."""

    def __init__(
        self,
        directory: Path,
        cameras: list[str],
        width: int,
        jpeg_quality: int,
        publish_every: int,
    ) -> None:
        self.directory = directory
        self.cameras = cameras
        self.width = width
        self.jpeg_quality = jpeg_quality
        self.publish_every = max(1, publish_every)
        self.frame_count = 0
        self._last_warning_time = 0.0
        self._queue: Queue[list[tuple[str, np.ndarray, bool]] | None] = Queue(maxsize=1)
        self._stop = threading.Event()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="web-evaluation-preview",
            daemon=True,
        )
        self._thread.start()
        print(f"Web preview: {', '.join(cameras)} -> {directory}")

    def submit(self, observation: dict[str, Any], camera_configs: dict) -> None:
        self.frame_count += 1
        if self.frame_count % self.publish_every:
            return
        frames: list[tuple[str, np.ndarray, bool]] = []
        for camera in self.cameras:
            frame = observation.get(camera)
            if not isinstance(frame, np.ndarray):
                continue
            config = camera_configs.get(camera)
            color_mode = getattr(config, "color_mode", "rgb")
            color_mode = getattr(color_mode, "value", color_mode)
            frames.append(
                (camera, np.ascontiguousarray(frame).copy(), str(color_mode).lower() == "rgb")
            )
        if not frames:
            return
        try:
            self._queue.put_nowait(frames)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(frames)
            except Full:
                pass

    def _publish(self, camera: str, frame: np.ndarray, input_is_rgb: bool) -> None:
        height, width = frame.shape[:2]
        if width > self.width:
            target_height = max(1, round(height * self.width / width))
            frame = cv2.resize(
                frame,
                (self.width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if input_is_rgb else frame
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"failed to encode web preview for {camera}")
        destination = self.directory / f"{camera}.jpg"
        temporary = self.directory / f".{camera}.jpg.tmp"
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(destination)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if frames is None:
                break
            try:
                for camera, frame, input_is_rgb in frames:
                    self._publish(camera, frame, input_is_rgb)
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_warning_time >= 5.0:
                    print(f"WARNING: web evaluation preview failed: {exc}")
                    self._last_warning_time = now

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
        self._thread.join(timeout=2.0)


class ACTChunkFusionController:
    """Run strided ACT chunk inference ahead of the robot control loop."""

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        device: torch.device,
        use_amp: bool,
        threshold: float,
        old_weight: float,
        action_stride: int,
        replan_interval: int | None = None,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.use_amp = use_amp
        self.threshold = threshold
        self.old_weight = old_weight
        self.new_weight = 1.0 - old_weight
        self.action_stride = action_stride
        self.replan_interval = replan_interval
        self.action_steps = int(policy.config.n_action_steps)
        self.chunk_size = self.action_steps

        if self.action_steps <= 0:
            raise ValueError("ACT n_action_steps must be positive")
        if self.action_stride <= 0:
            raise ValueError("chunk fusion action stride must be positive")
        if self.replan_interval is not None and self.replan_interval <= 0:
            raise ValueError("chunk fusion replan interval must be positive")

        self._actions: dict[int, torch.Tensor] = {}
        self._step = 0
        self._actions_since_replan = 0
        self._worker: threading.Thread | None = None
        self._worker_result: Queue = Queue(maxsize=1)

    @staticmethod
    def _copy_observation(observation: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(observation)

    def _predict_chunk(
        self,
        observation: dict[str, Any],
        task: str | None,
        robot_type: str | None,
    ) -> list[torch.Tensor]:
        observation = prepare_observation_for_inference(
            observation,
            self.device,
            task,
            robot_type,
        )
        observation = self.preprocessor(observation)

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.use_amp
            else nullcontext(),
        ):
            action_chunk = self.policy.predict_action_chunk(observation)

        if action_chunk.ndim != 3:
            action_chunk = action_chunk.unsqueeze(0)
        required_raw_steps = 1 + (self.action_steps - 1) * self.action_stride
        if action_chunk.shape[1] < required_raw_steps:
            raise ValueError(
                "ACT prediction chunk is too short for chunk fusion: "
                f"need {required_raw_steps} raw steps for "
                f"n_action_steps={self.action_steps}, stride={self.action_stride}, "
                f"got {action_chunk.shape[1]}"
            )
        raw_indices = torch.arange(
            0,
            required_raw_steps,
            self.action_stride,
            device=action_chunk.device,
        )
        action_chunk = action_chunk.index_select(1, raw_indices)

        processed_actions = []
        for index in range(action_chunk.shape[1]):
            action = self.postprocessor(action_chunk[:, index, :])
            processed_actions.append(action.detach().cpu())
        return processed_actions

    def _merge_chunk(self, origin: int, action_chunk: list[torch.Tensor]) -> None:
        fused = 0
        added = 0
        for offset, new_action in enumerate(action_chunk):
            timestep = origin + offset
            if timestep < self._step:
                continue
            if timestep in self._actions:
                old_action = self._actions[timestep]
                self._actions[timestep] = (
                    self.old_weight * old_action + self.new_weight * new_action
                )
                fused += 1
            else:
                self._actions[timestep] = new_action
                added += 1
        print(
            f"[chunk-fusion] chunk@{origin} merged: fused={fused}, "
            f"added={added}, queue={len(self._actions)}"
        )

    def _run_worker(
        self,
        origin: int,
        observation: dict[str, Any],
        task: str | None,
        robot_type: str | None,
    ) -> None:
        start = time.perf_counter()
        try:
            chunk = self._predict_chunk(observation, task, robot_type)
            self._worker_result.put((origin, chunk, None, time.perf_counter() - start))
        except BaseException as exc:
            self._worker_result.put((origin, None, exc, time.perf_counter() - start))

    def _start_worker(
        self,
        origin: int,
        observation: dict[str, Any],
        task: str | None,
        robot_type: str | None,
    ) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(origin, self._copy_observation(observation), task, robot_type),
            name="act-chunk-fusion",
            daemon=True,
        )
        self._worker.start()
        print(f"[chunk-fusion] look-ahead started at step {origin}")

    def _collect_worker(self, block: bool) -> bool:
        if self._worker is None:
            return False
        if block:
            self._worker.join()
        elif self._worker.is_alive():
            return False

        self._worker.join()
        origin, chunk, error, elapsed = self._worker_result.get()
        self._worker = None
        if error is not None:
            raise RuntimeError(f"Chunk-fusion inference failed at step {origin}") from error
        print(f"[chunk-fusion] chunk@{origin} ready in {elapsed:.3f}s")
        self._merge_chunk(origin, chunk)
        return True

    def select_action(
        self,
        observation: dict[str, Any],
        task: str | None,
        robot_type: str | None,
    ) -> torch.Tensor:
        self._collect_worker(block=False)

        if self._step not in self._actions:
            if self._worker is not None:
                print(f"[chunk-fusion] queue empty at step {self._step}; waiting for look-ahead")
                self._collect_worker(block=True)
            else:
                start = time.perf_counter()
                chunk = self._predict_chunk(
                    self._copy_observation(observation), task, robot_type
                )
                print(
                    f"[chunk-fusion] synchronous seed ready in "
                    f"{time.perf_counter() - start:.3f}s"
                )
                self._merge_chunk(self._step, chunk)
                self._actions_since_replan = 0

        if self._step not in self._actions:
            raise RuntimeError(f"Chunk fusion produced no action for step {self._step}")

        action = self._actions.pop(self._step)
        current_step = self._step
        self._step += 1
        self._actions_since_replan += 1

        queue_ratio = len(self._actions) / max(self.chunk_size, 1)
        should_replan = (
            self._actions_since_replan >= self.replan_interval
            if self.replan_interval is not None
            else queue_ratio <= self.threshold
        )
        if self._worker is None and should_replan:
            self._start_worker(current_step, observation, task, robot_type)
            self._actions_since_replan = 0
        return action

    def reset(self) -> None:
        if self._worker is not None:
            self._worker.join()
            self._worker = None
        while not self._worker_result.empty():
            self._worker_result.get_nowait()
        self._actions.clear()
        self._step = 0
        self._actions_since_replan = 0

    def close(self) -> None:
        self.reset()


def _enable_torque_with_retry(self, motors=None, num_retry: int = 0) -> None:
    _orig_enable_torque(self, motors=motors, num_retry=max(num_retry, 3))


def _read_joint_positions(robot: SOFollower) -> dict[str, float]:
    return {joint: float(value) for joint, value in robot.bus.sync_read("Present_Position").items()}


def _write_goal_positions(robot: SOFollower, goal_pos: dict[str, float]) -> None:
    for joint, value in goal_pos.items():
        robot.bus.write("Goal_Position", joint, value, num_retry=3)


def _interpolate_pose(
    start: dict[str, float], target: dict[str, float], fraction: float
) -> dict[str, float]:
    return {
        joint: start[joint] + (target[joint] - start[joint]) * fraction
        for joint in target
    }


def _move_to_right_vr_default(robot: SOFollower, tolerance: float = 2.5) -> None:
    if not robot.config.use_degrees:
        print("WARNING: right VR default pose is in degrees; add --robot.use_degrees=true. Skipping homing.")
        return

    print("Moving right arm to the VR-recording default pose...")
    robot.bus.enable_torque(num_retry=3)
    robot.bus.write("P_Coefficient", "shoulder_lift", 32, num_retry=3)

    start = _read_joint_positions(robot)
    max_delta = max(abs(start[joint] - target) for joint, target in RIGHT_VR_DEFAULT_POSE_DEG.items())
    num_steps = max(1, int(max_delta / 0.75))
    num_steps = min(max(num_steps, 30), 300)

    for step in range(1, num_steps + 1):
        fraction = step / num_steps
        _write_goal_positions(robot, _interpolate_pose(start, RIGHT_VR_DEFAULT_POSE_DEG, fraction))
        if step == 1 or step % 30 == 0:
            current = _read_joint_positions(robot)
            print("Default pose current:", {joint: round(current[joint], 1) for joint in RIGHT_VR_DEFAULT_POSE_DEG})
        time.sleep(1.0 / 30.0)

    for step in range(150):
        current = _read_joint_positions(robot)
        done = all(abs(current[joint] - target) <= tolerance for joint, target in RIGHT_VR_DEFAULT_POSE_DEG.items())
        if done:
            print("Default pose reached:", {joint: round(current[joint], 1) for joint in RIGHT_VR_DEFAULT_POSE_DEG})
            time.sleep(0.5)
            return
        if step % 30 == 0:
            print("Default pose current:", {joint: round(current[joint], 1) for joint in RIGHT_VR_DEFAULT_POSE_DEG})
        _write_goal_positions(robot, RIGHT_VR_DEFAULT_POSE_DEG)
        time.sleep(1.0 / 30.0)
    current = _read_joint_positions(robot)
    print("WARNING: default pose move timed out:", {joint: round(current[joint], 1) for joint in RIGHT_VR_DEFAULT_POSE_DEG})


def _connect_with_right_vr_default(self, *args, **kwargs) -> None:
    _orig_connect(self, *args, **kwargs)
    if self.id is None:
        print("WARNING: robot id is None; use --robot.id=right_so100 to match the VR training dataset.")
    if _start_from_default_pose:
        _move_to_right_vr_default(self)
    else:
        print("Skipping the VR-recording default pose.")


def _get_observation_with_udp_video(self, *args, **kwargs):
    observation = _orig_get_observation(self, *args, **kwargs)
    _forward_observation_previews(self, observation)
    return observation


def _get_xcore_observation_with_udp_video(self, *args, **kwargs):
    observation = _orig_xcore_get_observation(self, *args, **kwargs)
    _forward_observation_previews(self, observation)
    return observation


def _forward_observation_previews(self, observation) -> None:
    if _udp_video_forwarder is not None:
        _udp_video_forwarder.send(observation, self.config.cameras)
    if _web_preview_publisher is not None:
        _web_preview_publisher.submit(observation, self.config.cameras)


def _reset_act_with_chunk_fusion(self) -> None:
    controller = getattr(self, "_act_chunk_fusion_controller", None)
    if controller is not None:
        controller.reset()
    _orig_act_reset(self)


def _predict_action_with_optional_chunk_fusion(
    observation,
    policy,
    device,
    preprocessor,
    postprocessor,
    use_amp,
    task=None,
    robot_type=None,
):
    if not _chunk_fusion_enabled:
        return _orig_predict_action(
            observation=observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )

    if not isinstance(policy, ACTPolicy):
        raise TypeError("--chunk-fusion currently supports only ACT policies")
    if policy.config.temporal_ensemble_coeff is not None:
        raise ValueError(
            "Disable ACT temporal ensembling when using --chunk-fusion "
            "(set --policy.temporal_ensemble_coeff=null)"
        )

    controller = getattr(policy, "_act_chunk_fusion_controller", None)
    if controller is None:
        controller = ACTChunkFusionController(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            use_amp=use_amp,
            threshold=_chunk_fusion_threshold,
            old_weight=_chunk_fusion_old_weight,
            action_stride=_chunk_fusion_action_stride,
            replan_interval=_chunk_fusion_replan_interval,
        )
        policy._act_chunk_fusion_controller = controller
        _chunk_fusion_controllers.append(controller)

    return controller.select_action(observation, task, robot_type)


def _install_patches() -> None:
    FeetechMotorsBus.enable_torque = _enable_torque_with_retry
    SOFollower.connect = _connect_with_right_vr_default
    SOFollower.get_observation = _get_observation_with_udp_video
    XCoreRightArm.get_observation = _get_xcore_observation_with_udp_video
    ACTPolicy.reset = _reset_act_with_chunk_fusion
    lerobot_record.predict_action = _predict_action_with_optional_chunk_fusion
    builtins.input = lambda *args, **kwargs: ""


def _shutdown_chunk_fusion() -> None:
    for controller in _chunk_fusion_controllers:
        controller.close()
    _chunk_fusion_controllers.clear()


def _shutdown_udp_video() -> None:
    global _udp_video_forwarder
    if _udp_video_forwarder is not None:
        _udp_video_forwarder.close()
        _udp_video_forwarder = None


def _shutdown_web_preview() -> None:
    global _web_preview_publisher
    if _web_preview_publisher is not None:
        _web_preview_publisher.close()
        _web_preview_publisher = None


def _has_arg(name: str) -> bool:
    return name in sys.argv or any(arg.startswith(f"{name}=") for arg in sys.argv)


def _parse_bool(value: str, option: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{option} expects true or false, got {value!r}")


def _configured_camera_names() -> list[str]:
    raw: str | None = None
    for index, argument in enumerate(sys.argv):
        if argument.startswith("--robot.cameras="):
            raw = argument.split("=", 1)[1]
            break
        if argument == "--robot.cameras" and index + 1 < len(sys.argv):
            raw = sys.argv[index + 1]
            break
    if raw is None:
        raw = DEFAULT_ROS_CAMERA_CONFIG
    try:
        config = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return ["head", "right_wrist"]
    if not isinstance(config, dict):
        return ["head", "right_wrist"]
    return [str(name) for name in config]


def _prepare_start_pose_arg() -> None:
    global _start_from_default_pose

    option = "--start-from-default-pose"
    negative_option = "--no-start-from-default-pose"
    filtered_args = []
    index = 0

    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg == negative_option:
            _start_from_default_pose = False
        elif arg.startswith(f"{option}="):
            _start_from_default_pose = _parse_bool(arg.split("=", 1)[1], option)
        elif arg == option:
            if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--"):
                _start_from_default_pose = _parse_bool(sys.argv[index + 1], option)
                index += 1
            else:
                _start_from_default_pose = True
        else:
            filtered_args.append(arg)
        index += 1

    sys.argv[:] = filtered_args
    print(f"Start from VR-recording default pose: {_start_from_default_pose}")


def _prepare_chunk_fusion_args() -> None:
    global _chunk_fusion_enabled
    global _chunk_fusion_threshold
    global _chunk_fusion_old_weight
    global _chunk_fusion_action_stride
    global _chunk_fusion_replan_interval

    bool_option = "--chunk-fusion"
    threshold_option = "--chunk-fusion-threshold"
    weight_option = "--chunk-fusion-old-weight"
    stride_option = "--chunk-fusion-action-stride"
    replan_option = "--chunk-fusion-replan-interval"
    filtered_args = []
    index = 0

    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg == "--no-chunk-fusion":
            _chunk_fusion_enabled = False
        elif arg.startswith(f"{bool_option}="):
            _chunk_fusion_enabled = _parse_bool(arg.split("=", 1)[1], bool_option)
        elif arg == bool_option:
            if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--"):
                _chunk_fusion_enabled = _parse_bool(sys.argv[index + 1], bool_option)
                index += 1
            else:
                _chunk_fusion_enabled = True
        elif arg.startswith(f"{threshold_option}="):
            _chunk_fusion_threshold = float(arg.split("=", 1)[1])
        elif arg == threshold_option:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{threshold_option} requires a value")
            _chunk_fusion_threshold = float(sys.argv[index + 1])
            index += 1
        elif arg.startswith(f"{weight_option}="):
            _chunk_fusion_old_weight = float(arg.split("=", 1)[1])
        elif arg == weight_option:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{weight_option} requires a value")
            _chunk_fusion_old_weight = float(sys.argv[index + 1])
            index += 1
        elif arg.startswith(f"{stride_option}="):
            _chunk_fusion_action_stride = int(arg.split("=", 1)[1])
        elif arg == stride_option:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{stride_option} requires a value")
            _chunk_fusion_action_stride = int(sys.argv[index + 1])
            index += 1
        elif arg.startswith(f"{replan_option}="):
            _chunk_fusion_replan_interval = int(arg.split("=", 1)[1])
        elif arg == replan_option:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{replan_option} requires a value")
            _chunk_fusion_replan_interval = int(sys.argv[index + 1])
            index += 1
        else:
            filtered_args.append(arg)
        index += 1

    if not 0.0 <= _chunk_fusion_threshold <= 1.0:
        raise ValueError("--chunk-fusion-threshold must be between 0 and 1")
    if not 0.0 <= _chunk_fusion_old_weight <= 1.0:
        raise ValueError("--chunk-fusion-old-weight must be between 0 and 1")
    if _chunk_fusion_action_stride <= 0:
        raise ValueError("--chunk-fusion-action-stride must be positive")
    if (
        _chunk_fusion_replan_interval is not None
        and _chunk_fusion_replan_interval <= 0
    ):
        raise ValueError("--chunk-fusion-replan-interval must be positive")

    sys.argv[:] = filtered_args
    print(
        "Chunk fusion: "
        f"enabled={_chunk_fusion_enabled}, "
        f"threshold={_chunk_fusion_threshold}, "
        f"old/new weights={_chunk_fusion_old_weight}/"
        f"{1.0 - _chunk_fusion_old_weight}, "
        f"action_stride={_chunk_fusion_action_stride}, "
        f"replan_interval={_chunk_fusion_replan_interval or 'queue-threshold'}"
    )


def _prepare_udp_video_args() -> None:
    global _udp_video_forwarder

    enabled = False
    ip = DEFAULT_UDP_VIDEO_IP
    head_port = DEFAULT_UDP_VIDEO_PORT
    head_camera = "head"
    camera_names = _configured_camera_names()
    wrist_camera = next(
        (name for name in camera_names if name in {"left_wrist", "right_wrist"}),
        "right_wrist",
    )
    wrist_port = (
        DEFAULT_LEFT_WRIST_UDP_VIDEO_PORT
        if wrist_camera == "left_wrist"
        else DEFAULT_RIGHT_WRIST_UDP_VIDEO_PORT
    )
    jpeg_quality = 50
    send_every = 1
    verbose = False

    value_options = {
        "--udp-video-ip": ("ip", str),
        "--udp-video-port": ("head_port", int),
        "--udp-video-camera": ("head_camera", str),
        "--udp-video-wrist-port": ("wrist_port", int),
        "--udp-video-right-wrist-port": ("wrist_port", int),
        "--udp-video-jpeg-quality": ("jpeg_quality", int),
        "--udp-video-every": ("send_every", int),
    }
    values = {
        "ip": ip,
        "head_port": head_port,
        "head_camera": head_camera,
        "wrist_port": wrist_port,
        "jpeg_quality": jpeg_quality,
        "send_every": send_every,
    }
    filtered_args = []
    index = 0
    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg == "--no-udp-video":
            enabled = False
        elif arg == "--udp-video":
            if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--"):
                enabled = _parse_bool(sys.argv[index + 1], "--udp-video")
                index += 1
            else:
                enabled = True
        elif arg.startswith("--udp-video="):
            enabled = _parse_bool(arg.split("=", 1)[1], "--udp-video")
        elif arg == "--udp-video-verbose":
            verbose = True
        else:
            matched = False
            for option, (key, converter) in value_options.items():
                if arg == option:
                    if index + 1 >= len(sys.argv):
                        raise ValueError(f"{option} requires a value")
                    values[key] = converter(sys.argv[index + 1])
                    index += 1
                    matched = True
                    break
                if arg.startswith(f"{option}="):
                    values[key] = converter(arg.split("=", 1)[1])
                    matched = True
                    break
            if not matched:
                filtered_args.append(arg)
        index += 1

    jpeg_quality = int(values["jpeg_quality"])
    send_every = int(values["send_every"])
    if not 5 <= jpeg_quality <= 100:
        raise ValueError("--udp-video-jpeg-quality must be in [5, 100]")
    if send_every <= 0:
        raise ValueError("--udp-video-every must be positive")

    sys.argv[:] = filtered_args
    if not enabled:
        print("UDP video forwarding: disabled")
        return

    targets = [
        (str(values["head_camera"]), int(values["head_port"])),
        (wrist_camera, int(values["wrist_port"])),
    ]
    senders = []
    seen = set()
    for camera, port in targets:
        target = (camera, port)
        if target in seen:
            continue
        seen.add(target)
        senders.append(
            UdpFrameSender(
                str(values["ip"]),
                port,
                camera,
                jpeg_quality,
                send_every,
                verbose,
            )
        )
    _udp_video_forwarder = ObservationUdpVideoForwarder(senders)
    print(
        f"UDP video forwarding enabled for head/{wrist_camera} ROS observations."
    )


def _prepare_web_preview_args() -> None:
    global _web_preview_publisher

    directory: str | None = None
    values: dict[str, Any] = {
        "width": 960,
        "jpeg_quality": 70,
        "publish_every": 1,
    }
    value_options = {
        "--web-preview-dir": ("directory", str),
        "--web-preview-width": ("width", int),
        "--web-preview-jpeg-quality": ("jpeg_quality", int),
        "--web-preview-every": ("publish_every", int),
    }
    filtered_args = []
    index = 0
    while index < len(sys.argv):
        arg = sys.argv[index]
        matched = False
        for option, (key, converter) in value_options.items():
            if arg == option:
                if index + 1 >= len(sys.argv):
                    raise ValueError(f"{option} requires a value")
                value = converter(sys.argv[index + 1])
                index += 1
                matched = True
            elif arg.startswith(f"{option}="):
                value = converter(arg.split("=", 1)[1])
                matched = True
            else:
                continue
            if key == "directory":
                directory = str(value)
            else:
                values[key] = value
            break
        if not matched:
            filtered_args.append(arg)
        index += 1

    sys.argv[:] = filtered_args
    if directory is None:
        print("Web evaluation preview: disabled")
        return
    width = int(values["width"])
    jpeg_quality = int(values["jpeg_quality"])
    publish_every = int(values["publish_every"])
    if width <= 0:
        raise ValueError("--web-preview-width must be positive")
    if not 5 <= jpeg_quality <= 100:
        raise ValueError("--web-preview-jpeg-quality must be in [5, 100]")
    if publish_every <= 0:
        raise ValueError("--web-preview-every must be positive")
    _web_preview_publisher = WebPreviewPublisher(
        Path(directory).expanduser().resolve(),
        _configured_camera_names(),
        width,
        jpeg_quality,
        publish_every,
    )


def _prepare_eval_dataset_args() -> str | None:
    no_save_eval = "--no-save-eval" in sys.argv
    has_dataset_root = _has_arg("--dataset.root")
    if not no_save_eval and has_dataset_root:
        return None

    if no_save_eval:
        sys.argv.remove("--no-save-eval")

    tmp_dataset = tempfile.mkdtemp(prefix="smolvla_eval_", dir="/tmp")
    shutil.rmtree(tmp_dataset)
    root_arg = f"--dataset.root={tmp_dataset}"

    filtered_args = []
    skip_next = False
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg in {"--dataset.root", "--dataset.push_to_hub"}:
            skip_next = True
            continue
        if arg.startswith("--dataset.root=") or arg.startswith("--dataset.push_to_hub="):
            continue
        filtered_args.append(arg)

    sys.argv[:] = filtered_args + [root_arg, "--dataset.push_to_hub=false"]
    reason = "--no-save-eval was provided" if no_save_eval else "--dataset.root was omitted"
    print(f"No-save eval enabled ({reason}): temporary dataset root is {tmp_dataset}")
    return tmp_dataset


def _prepare_robot_args() -> None:
    if not _has_arg("--robot.id"):
        sys.argv.append("--robot.id=right_so100")
        print("Using --robot.id=right_so100 by default to match the VR training dataset.")


def _prepare_camera_args() -> None:
    """Use the shared head/right-wrist ROS topics unless the caller overrides them."""
    if _has_arg("--robot.cameras"):
        return
    sys.argv.append(f"--robot.cameras={DEFAULT_ROS_CAMERA_CONFIG}")
    print(
        "Using shared ROS cameras by default: head and right_wrist."
    )


if __name__ == "__main__":
    _prepare_robot_args()
    _prepare_camera_args()
    _prepare_start_pose_arg()
    _prepare_chunk_fusion_args()
    _prepare_udp_video_args()
    _prepare_web_preview_args()
    _install_patches()
    tmp_dataset = _prepare_eval_dataset_args()
    try:
        raise SystemExit(main())
    finally:
        _shutdown_chunk_fusion()
        _shutdown_udp_video()
        _shutdown_web_preview()
        if tmp_dataset is not None:
            if os.path.exists(tmp_dataset):
                shutil.rmtree(tmp_dataset, ignore_errors=True)
            print("No-save eval cleanup finished.")
