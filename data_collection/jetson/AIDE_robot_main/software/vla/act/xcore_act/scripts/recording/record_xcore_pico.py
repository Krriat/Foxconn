#!/usr/bin/env python3
"""Record 1–8 labelled single-arm PICO subtasks per LeRobot Episode.

Motion remains owned by pico_servol; this process only reads ROS
state/action/buttons, captures cameras, and writes the dataset. Each subtask
uses one B press to start and one B press to finish, matching dual-arm capture.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pyarrow.parquet as pq
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, UInt8

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot_robot_xcore import RosImageCameraConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UDP_VIDEO_IP = "172.16.13.184"
DEFAULT_UDP_VIDEO_PORT = 6000
DEFAULT_LEFT_WRIST_UDP_VIDEO_PORT = 6001
DEFAULT_RIGHT_WRIST_UDP_VIDEO_PORT = 6002
DEFAULT_UDP_STATUS_PORT = 6005
MAX_UDP_PAYLOAD = 65_507
DEFAULT_ROS_CAMERAS = {
    "left": (
        ("head", "/robot/sensors/cameras/head/color/image_raw", 960, 540),
        ("left_wrist", "/robot/sensors/cameras/left_wrist/color/image_raw", 640, 480),
    ),
    "right": (
        ("head", "/robot/sensors/cameras/head/color/image_raw", 960, 540),
        ("right_wrist", "/robot/sensors/cameras/right_wrist/color/image_raw", 640, 480),
    ),
}
TELEMETRY_TOPICS = {
    side: {
        "state": f"/xcore/{side}/record_state",
        "action": f"/xcore/{side}/record_action",
        "active": f"/xcore/{side}/teleop_active",
        "buttons": f"/xcore/{side}/pico_buttons",
        "stop": f"/xcore/{side}/record_stop_teleop",
    }
    for side in ("left", "right")
}
ACTION_SOURCE_FILE = "xcore_action_source.json"
ACTION_SOURCE = "controller_jointPos_m_same_sample"
SUBTASK_MANIFEST_FILE = "xcore_act_subtasks.json"
MIN_SUBTASKS = 1
MAX_SUBTASKS = 8

TRIGGER_BIT = 0
GRIP_BIT = 1
ENABLE_BIT = 2  # A/down: return to default (handled by pico_servol).
DISABLE_BIT = 3  # B/up: recording start/stop/double-click discard.

STATE_NAMES = tuple(
    [f"joint_{index}.pos" for index in range(1, 8)] + ["gripper.pos"]
)
JOINT_NAMES = STATE_NAMES[:-1]
ACTION_NAMES = STATE_NAMES
MAX_ABS_JOINT_POSITION_RAD = 3.2
EXTERNAL_RECORDING_COMMANDS = {
    "start",
    "pause",
    "resume",
    "save_episode",
    "discard_episode",
    "stop_session",
    "abort_session",
}


def button_pressed(buttons: int, bit: int) -> bool:
    return bool(buttons & (1 << bit))


@dataclass
class SubtaskProgress:
    task_count: int
    min_frames: int
    start_frame: int = 0
    completed_frames: list[int] = field(default_factory=list)

    @property
    def completed_subtasks(self) -> int:
        return len(self.completed_frames)

    @property
    def complete(self) -> bool:
        return self.completed_subtasks >= self.task_count

    @property
    def subtask_number(self) -> int:
        return min(self.completed_subtasks + 1, self.task_count)

    def current_frames(self, total_frames: int) -> int:
        return max(0, int(total_frames) - self.start_frame)

    def finish(self, total_frames: int) -> int:
        frames = self.current_frames(total_frames)
        if frames < self.min_frames:
            raise ValueError(
                f"subtask has {frames} frames; minimum is {self.min_frames}"
            )
        if self.complete:
            raise RuntimeError("all subtasks are already complete")
        self.completed_frames.append(frames)
        self.start_frame = int(total_frames)
        return frames

    def manifest(self, descriptions: list[str]) -> list[dict[str, int | str]]:
        start = 0
        result: list[dict[str, int | str]] = []
        for index, frames in enumerate(self.completed_frames, start=1):
            result.append(
                {
                    "subtask": index,
                    "description": descriptions[index - 1],
                    "segment": index,
                    "title": descriptions[index - 1],
                    "start_frame": start,
                    "end_frame": start + frames - 1,
                    "frames": frames,
                }
            )
            start += frames
        return result


def normalize_subtask_descriptions(
    task_count: int,
    descriptions: list[str],
    fallback_task: str,
) -> list[str]:
    if not MIN_SUBTASKS <= task_count <= MAX_SUBTASKS:
        raise ValueError(
            f"segments-per-episode must be in [{MIN_SUBTASKS},{MAX_SUBTASKS}]"
        )
    normalized = [str(description).strip() for description in descriptions]
    if not normalized and task_count == 1:
        normalized = [fallback_task.strip()]
    if len(normalized) != task_count:
        raise ValueError(
            "segment-title count must exactly match segments-per-episode"
        )
    if any(not description for description in normalized):
        raise ValueError("subtask descriptions must not be blank")
    if any(len(description) > 500 for description in normalized):
        raise ValueError("subtask descriptions must be at most 500 characters")
    folded = [description.casefold() for description in normalized]
    if len(set(folded)) != len(folded):
        raise ValueError("subtask descriptions must be unique")
    return normalized


def write_subtask_manifest(
    dataset_root: Path,
    descriptions: list[str],
    episodes: list[dict],
) -> None:
    meta_root = dataset_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    destination = meta_root / SUBTASK_MANIFEST_FILE
    temporary = meta_root / f".{SUBTASK_MANIFEST_FILE}.tmp"
    temporary.write_text(
        json.dumps(
            {
                "task_count": len(descriptions),
                "subtask_descriptions": descriptions,
                "episodes": episodes,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def validate_joint_action(action: dict[str, float]) -> tuple[bool, str]:
    values = np.asarray([action[name] for name in ACTION_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return False, "joint action contains NaN or infinity"
    if np.any(np.abs(values[:7]) > MAX_ABS_JOINT_POSITION_RAD):
        return False, "joint action is outside the 3.2 rad sanity bound"
    if not 0.0 <= values[7] <= 4095.0:
        return False, "gripper action is outside [0, 4095]"
    return True, ""


def validate_existing_dataset_actions(root: Path) -> tuple[bool, str]:
    """Reject incompatible schemas and malformed measured-joint actions."""
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
        action_names = tuple(info["features"]["action"]["names"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return False, f"cannot read action schema: {exc}"
    if action_names != ACTION_NAMES:
        return (
            False,
            "action schema is not xCore measured-joint format; expected "
            f"{list(ACTION_NAMES)}, got {list(action_names)}",
        )
    try:
        source_meta = json.loads((root / "meta" / ACTION_SOURCE_FILE).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot verify measured-joint action source: {exc}"
    if source_meta.get("action_source") != ACTION_SOURCE:
        return False, f"action source is not same-sample controller q_m: {source_meta}"

    data_files = sorted((root / "data").rglob("*.parquet"))
    for path in data_files:
        row_offset = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048, columns=["action"]):
            actions = np.asarray(batch.column(0).to_pylist(), dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != len(ACTION_NAMES):
                return False, f"unexpected action shape in {path}"
            for local_index, values in enumerate(actions):
                action = dict(zip(ACTION_NAMES, values, strict=True))
                valid, reason = validate_joint_action(action)
                if not valid:
                    return (
                        False,
                        f"invalid measured joint action at {path}, row "
                        f"{row_offset + local_index}: {reason}",
                    )
            row_offset += len(batch)
    return True, ""


def write_action_source_metadata(root: Path) -> None:
    path = root / "meta" / ACTION_SOURCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_source": "controller_jointPos_m",
                "action_source": ACTION_SOURCE,
                "uses_future_measured_state": False,
                "uses_external_ik": False,
            },
            indent=2,
        )
        + "\n"
    )
    temp_path.replace(path)


def validate_existing_dataset_cameras(
    root: Path, camera_configs: dict
) -> tuple[bool, str]:
    """Prevent appending head/right-wrist frames to an incompatible schema."""
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
        features = info["features"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return False, f"cannot read camera schema: {exc}"

    prefix = "observation.images."
    existing_names = {
        key.removeprefix(prefix) for key in features if key.startswith(prefix)
    }
    expected_names = set(camera_configs)
    if existing_names != expected_names:
        return (
            False,
            f"camera keys differ: existing={sorted(existing_names)}, "
            f"expected={sorted(expected_names)}",
        )
    for name, config in camera_configs.items():
        feature = features.get(f"{prefix}{name}", {})
        shape = tuple(feature.get("shape", ()))
        expected_shape = (int(config.height), int(config.width), 3)
        if shape != expected_shape:
            return (
                False,
                f"camera {name} shape differs: existing={shape}, "
                f"expected={expected_shape}",
            )
    return True, ""


def raise_keyboard_interrupt(_signum, _frame) -> None:
    """Route SIGTERM through the normal recorder cleanup path."""
    raise KeyboardInterrupt


class SessionAborted(Exception):
    """Exit the session after discarding only the current unsaved episode."""


def parse_ros_camera_value(raw: str) -> tuple[str, str, int, int]:
    try:
        name, topic, resolution = raw.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ROS camera must use NAME:/ABSOLUTE/TOPIC:WIDTHxHEIGHT"
        ) from exc
    try:
        width_text, height_text = resolution.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "ROS camera must use NAME:/ABSOLUTE/TOPIC:WIDTHxHEIGHT"
        ) from exc
    if not name or not topic.startswith("/") or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROS camera name/topic/dimensions are invalid")
    return name, topic, width, height


def make_camera_configs(args: argparse.Namespace) -> dict:
    cameras = {}
    for name, topic, camera_width, camera_height in args.ros_camera:
        if name in cameras:
            raise ValueError(f"duplicate ROS camera name: {name}")
        cameras[name] = RosImageCameraConfig(
            topic=topic,
            fps=args.fps,
            width=camera_width,
            height=camera_height,
            connect_timeout_s=args.camera_connect_timeout,
            frame_timeout_ms=args.camera_frame_timeout_ms,
        )
    return cameras


def next_dataset_root(root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = root.with_name(f"{root.name}_{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = root.with_name(f"{root.name}_{timestamp}_{suffix}")
        suffix += 1
    return candidate


class XCoreCameraRobot:
    """Small read-only robot facade matching the recorder's old robot API."""

    def __init__(self, camera_configs: dict):
        self.camera_configs = camera_configs
        self.cameras = make_cameras_from_configs(camera_configs)
        self.is_connected = False

    @property
    def observation_features(self) -> dict:
        features = {name: float for name in STATE_NAMES}
        features.update(
            {
                name: (config.height, config.width, 3)
                for name, config in self.camera_configs.items()
            }
        )
        return features

    @property
    def action_features(self) -> dict:
        return {name: float for name in ACTION_NAMES}

    def connect(self) -> None:
        connected = []
        try:
            for name, camera in self.cameras.items():
                print(f"Connecting camera {name}")
                camera.connect()
                connected.append(camera)
        except Exception as exc:
            for camera in reversed(connected):
                if camera.is_connected:
                    camera.disconnect()
            raise RuntimeError(
                f"Failed to subscribe to camera '{name}'. Verify the robot_vision "
                "node is running and publishing the configured ROS topic."
            ) from exc
        self.is_connected = True

    def get_observation(self, state: dict[str, float] | None) -> dict:
        observation = (
            {name: 0.0 for name in STATE_NAMES}
            if state is None
            else state.copy()
        )
        for name, camera in self.cameras.items():
            observation[name] = camera.async_read()
        return observation

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self.is_connected = False


def make_dataset(robot: XCoreCameraRobot, args: argparse.Namespace) -> LeRobotDataset:
    root = Path(args.root)
    if (root / "meta" / "info.json").exists():
        required_meta = [root / "meta" / "tasks.parquet"]
        missing_meta = [path for path in required_meta if not path.exists()]
        if missing_meta:
            old_root = root
            root = next_dataset_root(root)
            print(
                f"Found incomplete dataset at {old_root}; missing "
                f"{', '.join(str(path.relative_to(old_root)) for path in missing_meta)}. "
                f"Creating a new dataset at {root}."
            )
        else:
            cameras_valid, camera_reason = validate_existing_dataset_cameras(
                root, robot.camera_configs
            )
            if not cameras_valid:
                old_root = root
                root = next_dataset_root(root)
                print(
                    f"Existing dataset {old_root} is not compatible with the "
                    f"head/right-wrist recorder ({camera_reason}). Creating a new "
                    f"dataset at {root}; the old dataset is unchanged."
                )
            else:
                actions_valid, reason = validate_existing_dataset_actions(root)
                if not actions_valid:
                    raise RuntimeError(
                        f"Refusing to append to {root}: {reason}. "
                        "Use a new --root; old invalid episodes must not be mixed "
                        "with corrected data."
                    )
                dataset = LeRobotDataset(
                    repo_id=args.repo_id,
                    root=root,
                    video_backend=args.video_backend,
                    vcodec=args.vcodec,
                )
                if args.image_writer_processes or args.image_writer_threads:
                    dataset.start_image_writer(
                        args.image_writer_processes, args.image_writer_threads
                    )
                print(
                    f"Appending to existing dataset at {root} "
                    f"({dataset.meta.total_episodes} episodes, "
                    f"{dataset.meta.total_frames} frames)"
                )
                return dataset

    if root.exists():
        old_root = root
        root = next_dataset_root(root)
        print(
            f"Dataset root {old_root} already exists but is not complete. "
            f"Creating a new dataset at {root}."
        )

    features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR),
        **hw_to_dataset_features(robot.action_features, ACTION),
    }
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        features=features,
        robot_type=f"xcore_{args.arm_side}_arm",
        use_videos=True,
        video_backend=args.video_backend,
        vcodec=args.vcodec,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )


def wait_for_episode_images(dataset: LeRobotDataset) -> None:
    if dataset.image_writer is not None:
        dataset.image_writer.wait_until_done()

    missing = []
    for key, values in dataset.episode_buffer.items():
        feature = dataset.features.get(key)
        if feature is None or feature["dtype"] not in ["image", "video"]:
            continue
        for value in values:
            path = Path(value)
            if not path.exists():
                missing.append(path)
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break

    if missing:
        sample = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Image writer finished but recorded image files are missing. "
            "Look earlier in the terminal for 'Error writing image ...' messages. "
            "First missing files:\n  "
            f"{sample}"
        )


def save_episode_after_images(dataset: LeRobotDataset) -> None:
    wait_for_episode_images(dataset)
    dataset.save_episode()


class UdpVideoSender:
    def __init__(
        self,
        ip: str,
        port: int,
        camera: str,
        jpeg_quality: int,
        send_every: int,
        verbose: bool,
    ):
        self.addr = (ip, port)
        self.camera = camera
        self.jpeg_quality = jpeg_quality
        self.send_every = max(1, send_every)
        self.verbose = verbose
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.frame_count = 0
        self.t0 = time.time()
        print(f"UDP video: sending '{camera}' frames to {ip}:{port}")

    def send(self, obs: dict) -> None:
        if self.camera not in obs:
            return
        self.frame_count += 1
        if self.frame_count % self.send_every != 0:
            return

        frame_bgr = cv2.cvtColor(obs[self.camera], cv2.COLOR_RGB2BGR)
        quality = self.jpeg_quality
        data = None
        while quality >= 5:
            ok, buffer = cv2.imencode(
                ".jpg",
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
            )
            if not ok:
                raise RuntimeError("OpenCV JPEG encoding failed")
            encoded = buffer.tobytes()
            if len(encoded) <= MAX_UDP_PAYLOAD:
                data = encoded
                break
            quality -= 5
        if data is None:
            raise RuntimeError(
                f"JPEG is larger than one UDP datagram ({len(encoded)} bytes)"
            )
        self.sock.sendto(data, self.addr)

        if self.verbose and self.frame_count % 30 == 0:
            elapsed = max(time.time() - self.t0, 1e-6)
            print(
                f"UDP video sent {self.frame_count} frames "
                f"({self.frame_count / elapsed:.1f} fps)"
            )

    def close(self) -> None:
        self.sock.close()


class UdpStatusSender:
    """Send short, low-rate, plain UTF-8 recording status messages."""

    def __init__(self, ip: str, port: int):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.failed = False
        print(f"UDP status: short English messages -> {ip}:{port}")

    def send(self, message: str) -> None:
        # Keep this channel easy to display directly in PICO/Unity.
        payload = message.strip().encode("ascii", errors="replace")[:96]
        try:
            self.sock.sendto(payload, self.addr)
        except OSError as exc:
            if not self.failed:
                print(f"WARNING: UDP status disabled after send failure: {exc}")
                self.failed = True

    def close(self) -> None:
        self.sock.close()


class ExternalControlServer:
    """Receive non-blocking recording commands from the local edge API."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(str(self.path))
        self.sock.setblocking(False)
        print(f"External recording control: {self.path}")

    def drain(self) -> list[str]:
        commands = []
        while True:
            try:
                payload = self.sock.recv(4096)
            except BlockingIOError:
                break
            try:
                message = json.loads(payload.decode("utf-8"))
                command = message.get("command")
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                print("WARNING: ignoring malformed external recording command")
                continue
            if command not in EXTERNAL_RECORDING_COMMANDS:
                print(f"WARNING: ignoring unknown external command: {command!r}")
                continue
            commands.append(command)
        return commands

    def close(self) -> None:
        self.sock.close()
        self.path.unlink(missing_ok=True)


class AtomicStatusWriter:
    def __init__(self, path: Path, minimum_interval: float = 0.2):
        self.path = path
        self.minimum_interval = minimum_interval
        self.last_write = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, status: dict, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_write < self.minimum_interval:
            return
        payload = {
            **status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
        self.last_write = now


class WebPreviewPublisher:
    """Atomically publish downsized JPEGs read by the API's MJPEG endpoint."""

    def __init__(
        self,
        directory: Path,
        cameras: list[str],
        width: int,
        jpeg_quality: int,
        publish_every: int,
    ):
        self.directory = directory
        self.cameras = cameras
        self.width = width
        self.jpeg_quality = jpeg_quality
        self.publish_every = max(1, publish_every)
        self.frame_count = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        print(f"Web preview: {', '.join(cameras)} -> {directory}")

    def publish(self, observation: dict) -> None:
        self.frame_count += 1
        if self.frame_count % self.publish_every:
            return
        for camera in self.cameras:
            if camera not in observation:
                continue
            frame_rgb = observation[camera]
            height, width = frame_rgb.shape[:2]
            if width > self.width:
                preview_height = max(1, round(height * self.width / width))
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (self.width, preview_height),
                    interpolation=cv2.INTER_AREA,
                )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(
                ".jpg",
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                print(f"WARNING: failed to encode web preview for {camera}")
                continue
            destination = self.directory / f"{camera}.jpg"
            temporary = self.directory / f".{camera}.jpg.tmp"
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(destination)


@dataclass
class BridgeSnapshot:
    active: bool = False
    buttons: int = 0
    state: dict[str, float] | None = None
    action: dict[str, float] | None = None
    state_time: float = 0.0
    action_time: float = 0.0
    state_sequence: int = 0


class PicoDatasetBridge(Node):
    def __init__(self, arm_side: str) -> None:
        super().__init__("xcore_pico_dataset_recorder")
        topics = TELEMETRY_TOPICS[arm_side]
        self._lock = threading.Lock()
        self._snapshot = BridgeSnapshot()
        self._button_events: deque[int] = deque()
        self._last_buttons: int | None = None
        self._pending_states: dict[tuple[int, int], dict[str, float]] = {}
        self._pending_actions: dict[tuple[int, int], dict[str, float]] = {}
        self._invalid_action_samples = 0
        self._stop_publisher = self.create_publisher(Bool, topics["stop"], 10)
        self.create_subscription(JointState, topics["state"], self._on_state, 20)
        self.create_subscription(JointState, topics["action"], self._on_action, 20)
        self.create_subscription(Bool, topics["active"], self._on_active, 10)
        self.create_subscription(UInt8, topics["buttons"], self._on_buttons, 20)

    def _parse_named_vector(
        self, message: JointState, label: str, expected_names: tuple[str, ...]
    ):
        if len(message.name) != len(message.position):
            self.get_logger().warning(f"Ignoring malformed {label} JointState")
            return None
        values = dict(zip(message.name, message.position, strict=True))
        if any(name not in values for name in expected_names):
            self.get_logger().warning(f"Ignoring {label} with missing feature names")
            return None
        sample = {name: float(values[name]) for name in expected_names}
        if not all(np.isfinite(value) for value in sample.values()):
            self.get_logger().warning(f"Ignoring non-finite {label}")
            return None
        return sample

    def _on_state(self, message: JointState) -> None:
        state = self._parse_named_vector(message, "record_state", STATE_NAMES)
        if state is None:
            return
        key = (message.header.stamp.sec, message.header.stamp.nanosec)
        with self._lock:
            self._pending_states[key] = state
            self._commit_sample(key)

    def _on_action(self, message: JointState) -> None:
        action = self._parse_named_vector(message, "record_action", ACTION_NAMES)
        if action is None:
            return
        key = (message.header.stamp.sec, message.header.stamp.nanosec)
        with self._lock:
            self._pending_actions[key] = action
            self._commit_sample(key)

    def _commit_sample(self, key: tuple[int, int]) -> None:
        if key not in self._pending_states or key not in self._pending_actions:
            # Keep a small bounded window if a malformed/lost counterpart occurs.
            while len(self._pending_states) > 10:
                self._pending_states.pop(next(iter(self._pending_states)))
            while len(self._pending_actions) > 10:
                self._pending_actions.pop(next(iter(self._pending_actions)))
            return
        now = time.monotonic()
        state = self._pending_states.pop(key)
        action = self._pending_actions.pop(key)
        valid, reason = validate_joint_action(action)
        if not valid:
            self._invalid_action_samples += 1
            should_report = (
                self._invalid_action_samples == 1
                or self._invalid_action_samples % 50 == 0
            )
            if should_report:
                self.get_logger().error(
                    f"Rejecting invalid q_m record_action sample: {reason}. "
                    "No dataset frame will be written until pico_servol "
                    "publishes a valid measured joint sample."
                )
            return
        self._snapshot.state = state
        self._snapshot.action = action
        self._snapshot.state_time = now
        self._snapshot.action_time = now
        self._snapshot.state_sequence += 1

    def _on_active(self, message: Bool) -> None:
        with self._lock:
            self._snapshot.active = bool(message.data)

    def _on_buttons(self, message: UInt8) -> None:
        with self._lock:
            buttons = int(message.data)
            self._snapshot.buttons = buttons
            if buttons != self._last_buttons:
                self._button_events.append(buttons)
                self._last_buttons = buttons

    def snapshot(self) -> BridgeSnapshot:
        with self._lock:
            return BridgeSnapshot(
                active=self._snapshot.active,
                buttons=self._snapshot.buttons,
                state=None if self._snapshot.state is None else self._snapshot.state.copy(),
                action=None if self._snapshot.action is None else self._snapshot.action.copy(),
                state_time=self._snapshot.state_time,
                action_time=self._snapshot.action_time,
                state_sequence=self._snapshot.state_sequence,
            )

    def drain_button_events(self) -> list[int]:
        with self._lock:
            events = list(self._button_events)
            self._button_events.clear()
            return events

    def request_teleop_stop(self) -> None:
        message = Bool()
        message.data = True
        self._stop_publisher.publish(message)


class XCoreVrRecorder:
    """B/A input state shared by the single-arm recording lifecycle."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.side_label = getattr(args, "arm_side", "right").upper()
        self.recording_enabled = False
        self.record_prev = False
        self.return_prev = False
        self.grip_prev = False
        self.pending_recording_stop_at: float | None = None
        self.recording_save_requested = False
        self.recording_discard_requested = False
        self.paused_by_command = False

    def handle_buttons(self, buttons: int) -> None:
        now = time.monotonic()
        grip_now = button_pressed(buttons, GRIP_BIT)
        record_now = button_pressed(buttons, DISABLE_BIT)
        return_now = button_pressed(buttons, ENABLE_BIT)

        if grip_now and not self.grip_prev:
            print(f"[{self.side_label}] Grip clicked: xCore teleop toggle requested")

        if record_now and not self.record_prev:
            self.paused_by_command = False
            if self.pending_recording_stop_at is not None:
                elapsed = now - self.pending_recording_stop_at
                if elapsed <= self.args.discard_double_click_window:
                    self.pending_recording_stop_at = None
                    self.recording_discard_requested = True
                    print(f"[{self.side_label}] recording discard requested")
                else:
                    self.pending_recording_stop_at = None
                    self.recording_save_requested = True
                    self.recording_enabled = True
                    print(f"[{self.side_label}] recording started")
            elif self.recording_enabled:
                self.recording_enabled = False
                self.pending_recording_stop_at = now
                print(
                    f"[{self.side_label}] recording stopped; "
                    f"press B again within {self.args.discard_double_click_window:.1f}s "
                    "to discard"
                )
            else:
                self.recording_enabled = True
                print(f"[{self.side_label}] recording started")

        if return_now and not self.return_prev:
            print(f"[{self.side_label}] returning to default; reset motion is not recorded")

        self.grip_prev = grip_now
        self.record_prev = record_now
        self.return_prev = return_now

    def finalize_pending_recording_stop(self) -> None:
        if self.pending_recording_stop_at is None:
            return
        elapsed = time.monotonic() - self.pending_recording_stop_at
        if elapsed >= self.args.discard_double_click_window:
            self.pending_recording_stop_at = None
            self.recording_save_requested = True
            print(f"[{self.side_label}] recording save confirmed")

    def consume_recording_stop_request(self):
        if self.recording_discard_requested:
            self.recording_discard_requested = False
            return "discard"
        if self.recording_save_requested:
            self.recording_save_requested = False
            return "save"
        return None

    def handle_external_command(self, command: str) -> None:
        """Map web commands onto the same episode state used by PICO buttons."""
        if command in ("start", "resume"):
            self.recording_enabled = True
            self.paused_by_command = False
            self.pending_recording_stop_at = None
            print(f"[WEB] recording {command} requested")
        elif command == "pause":
            self.recording_enabled = False
            self.paused_by_command = True
            self.pending_recording_stop_at = None
            print("[WEB] recording pause requested")
        elif command == "save_episode":
            self.recording_enabled = False
            self.paused_by_command = False
            self.pending_recording_stop_at = None
            self.recording_discard_requested = False
            self.recording_save_requested = True
            print("[WEB] recording save requested")
        elif command == "discard_episode":
            self.recording_enabled = False
            self.paused_by_command = False
            self.pending_recording_stop_at = None
            self.recording_save_requested = False
            self.recording_discard_requested = True
            print("[WEB] recording discard requested")


def self_test() -> None:
    assert button_pressed(1 << GRIP_BIT, GRIP_BIT)
    assert button_pressed(1 << ENABLE_BIT, ENABLE_BIT)
    assert button_pressed(1 << DISABLE_BIT, DISABLE_BIT)
    assert parse_ros_camera_value("head:/robot/sensors/cameras/head/color/image_raw:960x540") == (
        "head",
        "/robot/sensors/cameras/head/color/image_raw",
        960,
        540,
    )
    state_features = {name: float for name in STATE_NAMES}
    action_features = {name: float for name in ACTION_NAMES}
    features = {
        **hw_to_dataset_features(state_features, OBS_STR),
        **hw_to_dataset_features(action_features, ACTION),
    }
    assert features["observation.state"]["shape"] == (8,)
    assert features["action"]["shape"] == (8,)
    assert tuple(features["observation.state"]["names"]) == STATE_NAMES
    assert tuple(features["action"]["names"]) == ACTION_NAMES
    valid_action = dict(
        zip(ACTION_NAMES, [-1.9, 1.3, 1.2, 1.4, 1.7, -0.02, 0.16, 900.0], strict=True)
    )
    bad_joint = valid_action.copy()
    bad_joint["joint_1.pos"] = MAX_ABS_JOINT_POSITION_RAD + 0.01
    assert validate_joint_action(valid_action)[0]
    assert not validate_joint_action(bad_joint)[0]
    descriptions = normalize_subtask_descriptions(
        3, ["approach", "grasp", "place"], "unused"
    )
    progress = SubtaskProgress(task_count=3, min_frames=2)
    assert progress.finish(2) == 2
    assert progress.subtask_number == 2
    assert progress.finish(5) == 3
    assert progress.finish(7) == 2
    assert progress.complete
    assert [item["description"] for item in progress.manifest(descriptions)] == descriptions
    try:
        normalize_subtask_descriptions(2, ["same", " SAME "], "unused")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate subtask descriptions must be rejected")
    recorder = XCoreVrRecorder(
        argparse.Namespace(discard_double_click_window=0.5)
    )
    b_button = 1 << DISABLE_BIT

    # Physical B semantics must remain identical to right_arm_vr_record.py:
    # first click starts, second click stops and saves after the double-click
    # window, and one more click inside that window discards instead.
    recorder.handle_buttons(b_button)
    assert recorder.recording_enabled
    recorder.handle_buttons(0)
    recorder.handle_buttons(b_button)
    assert not recorder.recording_enabled
    assert recorder.pending_recording_stop_at is not None
    recorder.handle_buttons(0)
    recorder.pending_recording_stop_at -= 0.5
    recorder.finalize_pending_recording_stop()
    assert recorder.consume_recording_stop_request() == "save"

    recorder = XCoreVrRecorder(
        argparse.Namespace(discard_double_click_window=0.5)
    )
    recorder.handle_buttons(b_button)
    recorder.handle_buttons(0)
    recorder.handle_buttons(b_button)
    recorder.handle_buttons(0)
    recorder.handle_buttons(b_button)
    assert recorder.consume_recording_stop_request() == "discard"
    assert not recorder.recording_enabled

    recorder = XCoreVrRecorder(
        argparse.Namespace(discard_double_click_window=0.5)
    )
    recorder.handle_external_command("start")
    assert recorder.recording_enabled
    recorder.handle_external_command("pause")
    assert not recorder.recording_enabled
    recorder.handle_external_command("resume")
    assert recorder.recording_enabled
    recorder.handle_external_command("save_episode")
    assert recorder.consume_recording_stop_request() == "save"
    recorder.handle_external_command("discard_episode")
    assert recorder.consume_recording_stop_request() == "discard"
    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        control = ExternalControlServer(temp_root / "control.sock")
        client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            client.sendto(
                json.dumps({"command": "pause"}).encode("utf-8"),
                str(control.path),
            )
            assert control.drain() == ["pause"]
        finally:
            client.close()
            control.close()
        status_path = temp_root / "status.json"
        AtomicStatusWriter(status_path).publish({"status": "ready"}, force=True)
        assert json.loads(status_path.read_text())["status"] == "ready"
        preview = WebPreviewPublisher(
            temp_root / "preview", ["head"], 32, 70, 1
        )
        preview.publish({"head": np.zeros((16, 64, 3), dtype=np.uint8)})
        image = cv2.imread(str(temp_root / "preview/head.jpg"))
        assert image is not None and image.shape[:2] == (8, 32)
    print("Recorder schema/button self-test: passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record xCore single-arm PICO data with labelled subtask semantics."
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--arm-side", choices=("left", "right"), default="right")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--ros-camera",
        type=parse_ros_camera_value,
        action="append",
        default=[],
        metavar="NAME:/TOPIC:WIDTHxHEIGHT",
        help=(
            "ROS sensor_msgs/Image source; defaults to head and the selected wrist "
            "topics from robot_vision"
        ),
    )
    parser.add_argument("--camera-connect-timeout", type=float, default=5.0)
    parser.add_argument("--camera-frame-timeout-ms", type=float, default=1000.0)
    parser.add_argument("--udp-video", action="store_true", help="send recorded camera streams over UDP")
    parser.add_argument("--udp-video-ip", default=DEFAULT_UDP_VIDEO_IP)
    parser.add_argument("--udp-video-port", type=int, default=DEFAULT_UDP_VIDEO_PORT)
    parser.add_argument("--udp-video-camera", default="head")
    parser.add_argument(
        "--udp-video-wrist-port",
        "--udp-video-right-wrist-port",
        dest="udp_video_wrist_port",
        type=int,
        default=None,
    )
    parser.add_argument("--udp-video-jpeg-quality", type=int, default=50)
    parser.add_argument("--udp-video-every", type=int, default=1)
    parser.add_argument("--udp-video-verbose", action="store_true")
    parser.add_argument("--udp-status-ip", default=DEFAULT_UDP_VIDEO_IP)
    parser.add_argument("--udp-status-port", type=int, default=DEFAULT_UDP_STATUS_PORT)
    parser.add_argument(
        "--no-udp-status",
        action="store_true",
        help="disable the short UDP recording status channel",
    )
    parser.add_argument("--repo-id", required=False)
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT / "data" / "datasets" / "xcore_right_arm_servoj_qm_v1"),
    )
    parser.add_argument("--task", required=False)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--episode-len", type=int, default=450)
    parser.add_argument("--min-episode-frames", type=int, default=90)
    parser.add_argument("--segments-per-episode", type=int, default=1)
    parser.add_argument(
        "--segment-title",
        dest="segment_titles",
        action="append",
        default=[],
        help="description for one ordered subtask; repeat once per subtask",
    )
    parser.add_argument("--discard-double-click-window", type=float, default=0.5)
    parser.add_argument("--state-timeout", type=float, default=0.25)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--control-socket",
        default=None,
        help="optional local Unix datagram socket receiving edge API commands",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        help="optional atomically updated JSON runtime status file",
    )
    parser.add_argument(
        "--web-preview-dir",
        default=None,
        help="optional directory for atomically updated browser preview JPEGs",
    )
    parser.add_argument("--web-preview-width", type=int, default=960)
    parser.add_argument("--web-preview-jpeg-quality", type=int, default=70)
    parser.add_argument("--web-preview-every", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.repo_id:
        raise SystemExit("--repo-id is required")
    if not args.task:
        raise SystemExit("--task is required")
    try:
        subtask_descriptions = normalize_subtask_descriptions(
            args.segments_per_episode,
            args.segment_titles,
            args.task,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.episode_len <= 0 or args.min_episode_frames <= 0:
        raise SystemExit("episode-len and min-episode-frames must be positive")
    if args.min_episode_frames > args.episode_len:
        raise SystemExit("min-episode-frames cannot exceed episode-len")
    if not 5 <= args.udp_video_jpeg_quality <= 100:
        raise SystemExit("--udp-video-jpeg-quality must be in [5, 100]")
    if args.udp_video_every <= 0:
        raise SystemExit("--udp-video-every must be positive")
    if not 1 <= args.udp_status_port <= 65535:
        raise SystemExit("--udp-status-port must be in [1, 65535]")
    if args.web_preview_width <= 0:
        raise SystemExit("--web-preview-width must be positive")
    if not 5 <= args.web_preview_jpeg_quality <= 100:
        raise SystemExit("--web-preview-jpeg-quality must be in [5, 100]")
    if args.web_preview_every <= 0:
        raise SystemExit("--web-preview-every must be positive")
    if args.camera_connect_timeout <= 0:
        raise SystemExit("--camera-connect-timeout must be positive")
    if args.camera_frame_timeout_ms <= 0:
        raise SystemExit("--camera-frame-timeout-ms must be positive")
    if not args.ros_camera:
        args.ros_camera = list(DEFAULT_ROS_CAMERAS[args.arm_side])
        print(
            f"Using head and {args.arm_side}_wrist ROS image topics "
            "from the robot_vision node."
        )
    wrist_camera = f"{args.arm_side}_wrist"
    if args.udp_video_wrist_port is None:
        args.udp_video_wrist_port = (
            DEFAULT_LEFT_WRIST_UDP_VIDEO_PORT
            if args.arm_side == "left"
            else DEFAULT_RIGHT_WRIST_UDP_VIDEO_PORT
        )

    robot = None
    dataset = None
    node = None
    executor = None
    spin_thread = None
    udp_videos = []
    udp_status = None
    control_server = None
    status_writer = None
    web_preview = None
    frame = 0
    episode = 0
    subtask_progress = SubtaskProgress(
        args.segments_per_episode, args.min_episode_frames
    )
    subtask_manifests: list[dict] = []
    last_runtime_status = {
        "status": "starting",
        "phase": "starting",
        "message": "正在初始化录制程序",
        "episode": episode,
        "episode_number": episode + 1,
        "frame": frame,
        "segment_number": 1,
        "segment_title": subtask_descriptions[0],
        "target_segments": args.segments_per_episode,
        "completed_segments": 0,
        "segment_frame": 0,
        "segment_recording": False,
        "button_press_count": 0,
        "next_button_action": "start_segment",
        "subtask_number": 1,
        "subtask_description": subtask_descriptions[0],
        "task_count": args.segments_per_episode,
        "completed_subtasks": 0,
        "subtask_frame": 0,
        "subtask_recording": False,
        "target_episodes": args.num_episodes,
        "episode_len": args.episode_len,
        "recording_enabled": False,
        "teleop_active": False,
        "cameras": [],
        "dataset_root": str(Path(args.root).resolve()),
        "last_event": None,
    }
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    try:
        if args.status_file:
            status_writer = AtomicStatusWriter(Path(args.status_file))
            status_writer.publish(last_runtime_status, force=True)
        if args.control_socket:
            control_server = ExternalControlServer(Path(args.control_socket))
        cameras = make_camera_configs(args)
        robot = XCoreCameraRobot(cameras)
        print("Connecting xCore recorder cameras")
        print(f"Cameras: {', '.join(cameras) if cameras else 'none'}")
        robot.connect()

        dataset = make_dataset(robot, args)
        write_action_source_metadata(Path(dataset.root))
        write_subtask_manifest(Path(dataset.root), subtask_descriptions, [])
        recorder = XCoreVrRecorder(args)
        if not args.no_udp_status:
            udp_status = UdpStatusSender(args.udp_status_ip, args.udp_status_port)
        if args.web_preview_dir:
            web_preview = WebPreviewPublisher(
                Path(args.web_preview_dir),
                list(cameras),
                args.web_preview_width,
                args.web_preview_jpeg_quality,
                args.web_preview_every,
            )
        last_runtime_status.update(
            {
                "status": "ready",
                "phase": "segment_ready",
                "message": (
                    f"头部与{'左' if args.arm_side == 'left' else '右'}腕相机及录制器已就绪，等待 B 开始第 "
                    f"1/{args.segments_per_episode} 个子任务"
                ),
                "cameras": list(cameras),
                "dataset_root": str(dataset.root),
            }
        )
        if status_writer:
            status_writer.publish(last_runtime_status, force=True)

        # Keep Python's normal SIGINT behavior and route SIGTERM above. The
        # default ROS handlers only shut down the context while this program's
        # main loop continues, which can leave subscriber/executor threads alive.
        rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
        node = PicoDatasetBridge(args.arm_side)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(
            target=executor.spin, name="xcore-record-ros", daemon=True
        )
        spin_thread.start()

        if args.udp_video:
            targets = [
                (args.udp_video_camera, args.udp_video_port),
                (wrist_camera, args.udp_video_wrist_port),
            ]
            seen = set()
            for camera_name, port in targets:
                target = (camera_name, port)
                if target in seen:
                    continue
                seen.add(target)
                udp_videos.append(
                    UdpVideoSender(
                        args.udp_video_ip,
                        port,
                        camera_name,
                        args.udp_video_jpeg_quality,
                        args.udp_video_every,
                        args.udp_video_verbose,
                    )
                )

        print(f"ROS state: {TELEMETRY_TOPICS[args.arm_side]['state']}")
        print(f"ROS action: {TELEMETRY_TOPICS[args.arm_side]['action']}")
        print(
            "Grip: click to toggle xCore teleop. Up/B: start/finish each "
            f"subtask ({args.segments_per_episode * 2} presses per Episode); "
            "double B after finishing discards the Episode. A/down: return initial pose."
        )
        if udp_status is not None:
            udp_status.send("Recorder ready.")

        period = 1.0 / args.fps
        last_state_sequence = -1
        last_record_wait_reason = None

        while episode < args.num_episodes:
            loop_start = time.perf_counter()
            if control_server is not None:
                for command in control_server.drain():
                    if command == "abort_session":
                        print("[WEB] recording session abort requested")
                        raise SessionAborted
                    if command == "stop_session":
                        print("[WEB] recording session stop requested")
                        raise KeyboardInterrupt
                    was_recording = recorder.recording_enabled
                    recorder.handle_external_command(command)
                    if (
                        udp_status is not None
                        and not was_recording
                        and recorder.recording_enabled
                    ):
                        udp_status.send(
                            f"Recording started (episode {episode + 1})."
                        )
            for buttons in node.drain_button_events():
                was_recording = recorder.recording_enabled
                recorder.handle_buttons(buttons)
                if (
                    udp_status is not None
                    and not was_recording
                    and recorder.recording_enabled
                ):
                    udp_status.send(
                        f"Recording started (episode {episode + 1})."
                    )
            snapshot = node.snapshot()
            recorder.finalize_pending_recording_stop()

            recording_stop_request = recorder.consume_recording_stop_request()
            if recording_stop_request == "save":
                current_subtask_frames = subtask_progress.current_frames(frame)
                if current_subtask_frames < args.min_episode_frames:
                    recorder.recording_enabled = True
                    recorder.paused_by_command = False
                    last_runtime_status.update(
                        {
                            "status": "recording",
                            "phase": "segment_too_short",
                            "message": (
                                f"第 {subtask_progress.subtask_number} 个子任务只有 "
                                f"{current_subtask_frames} 帧，至少需要 "
                                f"{args.min_episode_frames} 帧；请继续录制"
                            ),
                        }
                    )
                    if status_writer:
                        status_writer.publish(last_runtime_status, force=True)
                else:
                    saved_subtask = subtask_progress.subtask_number
                    saved_subtask_title = subtask_descriptions[saved_subtask - 1]
                    saved_subtask_frames = subtask_progress.finish(frame)
                    last_runtime_status["last_event"] = {
                        "type": "segment_saved",
                        "episode": episode + 1,
                        "segment": saved_subtask,
                        "title": saved_subtask_title,
                        "frames": saved_subtask_frames,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                    if not subtask_progress.complete:
                        last_runtime_status.update(
                            {
                                "status": "ready",
                                "phase": "segment_saved",
                                "message": (
                                    f"第 {saved_subtask}/{args.segments_per_episode} "
                                    f"个子任务已完成；再按 B 开始第 "
                                    f"{subtask_progress.subtask_number} 个子任务"
                                ),
                            }
                        )
                        if status_writer:
                            status_writer.publish(last_runtime_status, force=True)
                        if udp_status is not None:
                            udp_status.send(
                                f"Subtask {saved_subtask}/{args.segments_per_episode} saved."
                            )
                    else:
                        saved_episode = episode + 1
                        saved_frames = frame
                        last_runtime_status.update(
                            {
                                "status": "saving",
                                "phase": "saving",
                                "message": (
                                    f"{args.segments_per_episode} 个子任务已完成，"
                                    f"正在保存 Episode {saved_episode}"
                                ),
                                "episode": episode,
                                "episode_number": saved_episode,
                                "frame": saved_frames,
                                "recording_enabled": False,
                                "saving_episode": saved_episode,
                                "saving_frames": saved_frames,
                            }
                        )
                        if status_writer:
                            status_writer.publish(last_runtime_status, force=True)
                        print(
                            f"\nSaving episode {saved_episode} "
                            f"({saved_frames} frames, {args.segments_per_episode} subtasks)...",
                            flush=True,
                        )
                        save_episode_after_images(dataset)
                        completed_subtasks = subtask_progress.manifest(
                            subtask_descriptions
                        )
                        subtask_manifests.append(
                            {
                                "episode": episode,
                                "frames": saved_frames,
                                "subtasks": completed_subtasks,
                                "segments": completed_subtasks,
                            }
                        )
                        write_subtask_manifest(
                            Path(dataset.root), subtask_descriptions, subtask_manifests
                        )
                        episode += 1
                        frame = 0
                        last_state_sequence = -1
                        subtask_progress = SubtaskProgress(
                            args.segments_per_episode, args.min_episode_frames
                        )
                        last_runtime_status.update(
                            {
                                "status": "ready",
                                "phase": "saved",
                                "message": (
                                    f"Episode {saved_episode} 的 "
                                    f"{args.segments_per_episode} 个子任务已保存"
                                ),
                                "episode": episode,
                                "episode_number": episode + 1,
                                "frame": 0,
                                "saving_episode": None,
                                "saving_frames": None,
                                "last_event": {
                                    "type": "saved",
                                    "episode": saved_episode,
                                    "frames": saved_frames,
                                    "subtasks": args.segments_per_episode,
                                    "segments": args.segments_per_episode,
                                    "timestamp": time.strftime(
                                        "%Y-%m-%dT%H:%M:%S%z"
                                    ),
                                },
                            }
                        )
                        if status_writer:
                            status_writer.publish(last_runtime_status, force=True)
                        if udp_status is not None:
                            udp_status.send(
                                f"Recording saved (episode {saved_episode}, "
                                f"{saved_frames} frames)."
                            )
                        print(
                            "\n"
                            + "=" * 72
                            + "\n"
                            + f"  EPISODE {saved_episode} SAVED SUCCESSFULLY\n"
                            + f"  Subtasks: {args.segments_per_episode}\n"
                            + f"  Frames:   {saved_frames}\n"
                            + f"  Dataset:  {dataset.root}\n"
                            + "=" * 72,
                            flush=True,
                        )
                        if episode >= args.num_episodes:
                            break
                if current_subtask_frames == 0:
                    print("\nNo frames recorded; continue the current subtask.")
            elif recording_stop_request == "discard":
                discarded_frames = frame
                if frame > 0:
                    print(f"\nDiscarding episode {episode} ({frame} frames)")
                    dataset.clear_episode_buffer(delete_images=True)
                    frame = 0
                    last_state_sequence = -1
                else:
                    print("\nNo frames recorded; nothing to discard.")
                subtask_progress = SubtaskProgress(
                    args.segments_per_episode, args.min_episode_frames
                )
                if udp_status is not None:
                    udp_status.send(
                        f"Recording canceled (episode {episode + 1}, "
                        f"{discarded_frames} frames)."
                    )
                last_runtime_status.update(
                    {
                        "phase": "discarded",
                        "message": (
                            f"Episode {episode + 1} 的未保存子任务已丢弃，共 "
                            f"{discarded_frames} 帧；等待重新开始第 1 个子任务"
                        ),
                        "last_event": {
                            "type": "discarded",
                            "episode": episode + 1,
                            "frames": discarded_frames,
                            "timestamp": time.strftime(
                                "%Y-%m-%dT%H:%M:%S%z"
                            ),
                        },
                    }
                )

            now = time.monotonic()
            state_fresh = (
                snapshot.state is not None
                and now - snapshot.state_time <= args.state_timeout
            )
            action_fresh = (
                snapshot.action is not None
                and now - snapshot.action_time <= args.state_timeout
            )
            has_new_state = snapshot.state_sequence != last_state_sequence
            current_subtask_frames = subtask_progress.current_frames(frame)
            subtask_at_limit = current_subtask_frames >= args.episode_len
            should_record = (
                recorder.recording_enabled
                and snapshot.active
                and state_fresh
                and action_fresh
                and has_new_state
                and not subtask_at_limit
            )

            if recorder.recording_enabled and not should_record:
                if subtask_at_limit:
                    wait_reason = (
                        f"当前子任务已达到 {args.episode_len} 帧上限，请按 B 完成"
                    )
                elif not snapshot.active:
                    wait_reason = "摇操尚未开启（请先点击 Grip）"
                elif not state_fresh or not action_fresh:
                    wait_reason = "尚未收到新鲜的 q_m 关节样本"
                else:
                    wait_reason = "正在等待下一组 q_m 关节样本"
                if wait_reason != last_record_wait_reason:
                    print(f"\n[RIGHT] B已开始录制，但{wait_reason}")
                    last_record_wait_reason = wait_reason
            elif should_record:
                if last_record_wait_reason is not None:
                    print("\n[RIGHT] 关节样本已就绪，开始写入录制帧")
                last_record_wait_reason = None

            if recorder.recording_enabled:
                if subtask_at_limit:
                    runtime_phase = "segment_limit"
                    runtime_message = (
                        f"第 {subtask_progress.subtask_number} 个子任务已达到 "
                        f"{args.episode_len} 帧上限，请按 B 完成当前子任务"
                    )
                elif not snapshot.active:
                    runtime_phase = "waiting_teleop"
                    runtime_message = "录制已触发，等待点击 Grip 开启遥操"
                elif not state_fresh or not action_fresh:
                    runtime_phase = "waiting_samples"
                    runtime_message = "遥操已开启，正在等待新鲜的 q_m 关节样本"
                else:
                    runtime_phase = "recording"
                    runtime_message = (
                        f"正在录制第 {subtask_progress.subtask_number}/"
                        f"{args.segments_per_episode} 个子任务："
                        f"{current_subtask_frames}/{args.episode_len} 帧"
                    )
            elif recorder.paused_by_command:
                runtime_phase = "paused"
                runtime_message = f"Episode {episode + 1} 已暂停"
            elif recorder.pending_recording_stop_at is not None:
                runtime_phase = "pending_save"
                runtime_message = "已停止写帧，等待双击 B 判定保存或丢弃"
            elif last_runtime_status.get("phase") in {
                "saved",
                "discarded",
                "segment_saved",
            }:
                runtime_phase = str(last_runtime_status["phase"])
                runtime_message = str(last_runtime_status["message"])
            else:
                runtime_phase = "ready"
                runtime_message = f"等待开始 Episode {episode + 1}"

            obs = None
            if udp_videos or web_preview is not None or should_record:
                obs = robot.get_observation(snapshot.state)
                for udp_video in udp_videos:
                    udp_video.send(obs)
                if web_preview is not None:
                    web_preview.publish(obs)

            if should_record:
                observation_frame = build_dataset_frame(
                    dataset.features, obs, prefix=OBS_STR
                )
                action_frame = build_dataset_frame(
                    dataset.features, snapshot.action, prefix=ACTION
                )
                dataset.add_frame(
                    {
                        **observation_frame,
                        **action_frame,
                        "task": subtask_descriptions[
                            subtask_progress.subtask_number - 1
                        ],
                    }
                )
                last_state_sequence = snapshot.state_sequence
                frame += 1
                print(
                    f"episode={episode} subtask={subtask_progress.subtask_number} "
                    f"frame={subtask_progress.current_frames(frame)}/{args.episode_len}",
                    end="\r",
                )

            last_runtime_status.update(
                {
                    "status": (
                        "recording"
                        if recorder.recording_enabled
                        else "paused"
                        if recorder.paused_by_command
                        else "finalizing"
                        if recorder.pending_recording_stop_at is not None
                        else "ready"
                    ),
                    "phase": runtime_phase,
                    "message": runtime_message,
                    "episode": episode,
                    "episode_number": episode + 1,
                    "frame": frame,
                    "segment_number": subtask_progress.subtask_number,
                    "segment_title": subtask_descriptions[
                        subtask_progress.subtask_number - 1
                    ],
                    "target_segments": args.segments_per_episode,
                    "completed_segments": subtask_progress.completed_subtasks,
                    "segment_frame": subtask_progress.current_frames(frame),
                    "segment_recording": recorder.recording_enabled,
                    "button_press_count": (
                        subtask_progress.completed_subtasks * 2
                        + (2 if recorder.pending_recording_stop_at is not None else 1)
                        if recorder.recording_enabled
                        or recorder.pending_recording_stop_at is not None
                        else subtask_progress.completed_subtasks * 2
                    ),
                    "next_button_action": (
                        "save_segment"
                        if recorder.recording_enabled
                        else "start_segment"
                    ),
                    "subtask_number": subtask_progress.subtask_number,
                    "subtask_description": subtask_descriptions[
                        subtask_progress.subtask_number - 1
                    ],
                    "task_count": args.segments_per_episode,
                    "completed_subtasks": subtask_progress.completed_subtasks,
                    "subtask_frame": subtask_progress.current_frames(frame),
                    "subtask_recording": recorder.recording_enabled,
                    "recording_enabled": recorder.recording_enabled,
                    "teleop_active": snapshot.active,
                    "state_fresh": state_fresh,
                    "action_fresh": action_fresh,
                    "state": (
                        None
                        if snapshot.state is None
                        else [snapshot.state[name] for name in STATE_NAMES]
                    ),
                    "action": (
                        None
                        if snapshot.action is None
                        else [snapshot.action[name] for name in ACTION_NAMES]
                    ),
                }
            )
            if status_writer:
                status_writer.publish(last_runtime_status)

            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

        if args.push_to_hub:
            dataset.push_to_hub()
        print("Done.")
        if udp_status is not None:
            udp_status.send("Recording session completed.")
        last_runtime_status.update(
            {
                "status": "completed",
                "phase": "completed",
                "message": f"采集任务已完成，共保存 {episode} 个 Episodes",
                "episode": episode,
                "episode_number": episode,
                "frame": frame,
            }
        )
        if status_writer:
            status_writer.publish(last_runtime_status, force=True)
        return 0
    except SessionAborted:
        print("\nSession aborted; discarding the current unsaved episode.")
        if dataset is not None and frame > 0:
            dataset.clear_episode_buffer(delete_images=True)
        frame = 0
        last_runtime_status.update(
            {
                "status": "cancelled",
                "phase": "cancelled",
                "message": "采集任务已放弃，当前未保存片段已丢弃",
                "episode": episode,
                "frame": 0,
                "recording_enabled": False,
                "teleop_active": False,
            }
        )
        if status_writer:
            status_writer.publish(last_runtime_status, force=True)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if dataset is not None and frame > 0:
            print(
                f"Discarding incomplete Episode {episode + 1} ({frame} frames, "
                f"{subtask_progress.completed_subtasks}/{args.segments_per_episode} "
                "subtasks complete)."
            )
            dataset.clear_episode_buffer(delete_images=True)
            frame = 0
        last_runtime_status.update(
            {
                "status": "completed",
                "phase": "completed",
                "message": "采集任务已安全结束；未完成全部子任务的 Episode 已丢弃",
                "episode": episode,
                "episode_number": episode + 1,
                "frame": frame,
                "saving_episode": None,
                "saving_frames": None,
            }
        )
        if status_writer:
            status_writer.publish(last_runtime_status, force=True)
        return 0
    except Exception as exc:
        last_runtime_status.update(
            {
                "status": "failed",
                "phase": "failed",
                "message": f"录制程序异常：{exc}",
                "error": str(exc),
            }
        )
        if status_writer:
            status_writer.publish(last_runtime_status, force=True)
        raise
    finally:
        # Stop the independent ROS camera subscribers before the PICO bridge.
        if robot is not None and robot.is_connected:
            robot.disconnect()
        if executor is not None:
            executor.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for udp_video in udp_videos:
            udp_video.close()
        if udp_status is not None:
            udp_status.close()
        if control_server is not None:
            control_server.close()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
    raise SystemExit(main())
