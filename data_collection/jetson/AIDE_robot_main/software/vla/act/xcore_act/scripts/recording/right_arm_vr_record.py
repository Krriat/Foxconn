#!/usr/bin/env python3
import argparse
import builtins
import json
import socket
import sys
import time
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_LEROBOT_SRC = REPO_ROOT / "XLeRobot-orin" / "software" / "src"
if LOCAL_LEROBOT_SRC.exists():
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot_robot_xcore import RosImageCameraConfig

from remote_control.pico_ee_teleop import (
    DISABLE_BIT,
    ENABLE_BIT,
    GRIP_BIT,
    PACKET_SIZE,
    RIGHT_HAND_ID,
    RX_TIMEOUT,
    SO101EETarget,
    SO101Kinematics,
    UDP_RCVBUF,
    button_pressed,
    parse_packet,
    rate_limited_command_action,
    read_arm_state,
)


DEFAULT_UDP_VIDEO_IP = "172.16.13.184"
DEFAULT_UDP_VIDEO_PORT = 6000
DEFAULT_LEFT_WRIST_UDP_VIDEO_PORT = 6001
DEFAULT_RIGHT_WRIST_UDP_VIDEO_PORT = 6002
_ORIG_ENABLE_TORQUE = FeetechMotorsBus.enable_torque


def _enable_torque_with_retry(self, motors=None, num_retry: int = 0) -> None:
    _ORIG_ENABLE_TORQUE(self, motors=motors, num_retry=max(num_retry, 3))


FeetechMotorsBus.enable_torque = _enable_torque_with_retry


def connect_with_default_calibration(robot) -> None:
    old_input = builtins.input
    builtins.input = lambda *args, **kwargs: ""
    try:
        robot.connect()
    finally:
        builtins.input = old_input


DEFAULT_ROS_CAMERAS = (
    ("head", "/robot/sensors/cameras/head/color/image_raw", 960, 540),
    ("left_wrist", "/robot/sensors/cameras/left_wrist/color/image_raw", 640, 480),
    ("right_wrist", "/robot/sensors/cameras/right_wrist/color/image_raw", 640, 480),
)


def parse_ros_camera_value(raw: str) -> tuple[str, str, int, int]:
    try:
        name, topic, resolution = raw.split(":", 2)
        width_text, height_text = resolution.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "ROS camera must use NAME:/ABSOLUTE/TOPIC:WIDTHxHEIGHT"
        ) from exc
    if not name or not topic.startswith("/") or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROS camera name/topic/dimensions are invalid")
    return name, topic, width, height


def make_camera_configs(args) -> dict:
    cameras = {}
    for name, topic, width, height in args.ros_camera:
        if name in cameras:
            raise ValueError(f"duplicate ROS camera name: {name}")
        cameras[name] = RosImageCameraConfig(
            topic=topic,
            fps=args.fps,
            width=width,
            height=height,
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


def validate_existing_dataset_cameras(
    root: Path, camera_configs: dict
) -> tuple[bool, str]:
    """Prevent appending configured camera frames to an incompatible schema."""
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


def make_dataset(robot, args, camera_configs: dict) -> LeRobotDataset:
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
                root, camera_configs
            )
            if not cameras_valid:
                old_root = root
                root = next_dataset_root(root)
                print(
                    f"Existing dataset {old_root} is not compatible with the "
                    f"configured camera recorder ({camera_reason}). Creating a "
                    f"new dataset at {root}; the old dataset is unchanged."
                )
            else:
                dataset = LeRobotDataset(
                    repo_id=args.repo_id,
                    root=root,
                    video_backend=args.video_backend,
                    vcodec=args.vcodec,
                )
                if args.image_writer_processes or args.image_writer_threads:
                    dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
                print(
                    f"Appending to existing dataset at {root} "
                    f"({dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames)"
                )
                return dataset

    if root.exists():
        old_root = root
        root = next_dataset_root(root)
        print(f"Dataset root {old_root} already exists but is not complete. Creating a new dataset at {root}.")

    features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR),
        **hw_to_dataset_features(robot.action_features, ACTION),
    }
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        features=features,
        robot_type="so101_follower",
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
    def __init__(self, ip: str, port: int, camera: str, jpeg_quality: int, send_every: int, verbose: bool):
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

        frame = obs[self.camera]
        # LeRobot OpenCV cameras are configured as RGB; cv2.imencode expects BGR.
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if ok:
            self.sock.sendto(buffer.tobytes(), self.addr)

        if self.verbose and self.frame_count % 30 == 0:
            elapsed = max(time.time() - self.t0, 1e-6)
            print(f"UDP video sent {self.frame_count} frames ({self.frame_count / elapsed:.1f} fps)")

    def close(self) -> None:
        self.sock.close()


class RightArmEETarget(SO101EETarget):
    def __init__(self, *args, default_wrist_roll: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_wrist_roll = default_wrist_roll

    def set_default_target(self) -> None:
        super().set_default_target()
        self.target["wrist_roll"] = self.default_wrist_roll


class RightArmVrRecorder:
    def __init__(self, robot, args):
        self.robot = robot
        self.args = args
        self.targeter = RightArmEETarget(
            read_arm_state(robot),
            SO101Kinematics(),
            default_ee_x=args.default_ee_x,
            default_ee_y=args.default_ee_y,
            default_pan=args.default_pan,
            default_wrist_roll=args.default_wrist_roll,
        )
        self.command_state = self.targeter.target.copy()
        self.teleop_enabled = False
        self.recording_enabled = False
        self.returning_default = False
        self.toggle_prev = False
        self.record_prev = False
        self.pending_recording_stop_at = None
        self.recording_save_requested = False
        self.recording_discard_requested = False
        self.return_prev = False
        self.last_rx = 0.0

    def handle_buttons(self, packet) -> None:
        now = time.monotonic()
        toggle_now = button_pressed(packet.buttons, GRIP_BIT)
        record_now = button_pressed(packet.buttons, DISABLE_BIT)
        return_now = button_pressed(packet.buttons, ENABLE_BIT)

        if toggle_now and not self.toggle_prev:
            self.teleop_enabled = not self.teleop_enabled
            self.returning_default = False
            if self.teleop_enabled:
                self.targeter.reset_origin(packet, read_arm_state(self.robot))
                self.command_state = self.targeter.target.copy()
                print("[RIGHT] episode control enabled")
            else:
                print("[RIGHT] episode control paused")

        if record_now and not self.record_prev:
            if self.pending_recording_stop_at is not None:
                elapsed = now - self.pending_recording_stop_at
                if elapsed <= self.args.discard_double_click_window:
                    self.pending_recording_stop_at = None
                    self.recording_discard_requested = True
                    print("[RIGHT] recording discard requested")
                else:
                    self.pending_recording_stop_at = None
                    self.recording_save_requested = True
                    self.recording_enabled = True
                    print("[RIGHT] recording started")
            elif self.recording_enabled:
                self.recording_enabled = False
                self.pending_recording_stop_at = now
                print(
                    "[RIGHT] recording stopped; "
                    f"press B again within {self.args.discard_double_click_window:.1f}s to discard"
                )
            else:
                self.recording_enabled = True
                print("[RIGHT] recording started")

        if return_now and not self.return_prev:
            self.teleop_enabled = False
            self.returning_default = True
            self.command_state = read_arm_state(self.robot)
            self.targeter.set_default_target()
            print("[RIGHT] returning to default")

        self.toggle_prev = toggle_now
        self.record_prev = record_now
        self.return_prev = return_now

    def make_action(self, packet):
        if self.returning_default:
            return rate_limited_command_action(self.command_state, self.targeter.target), False

        if not self.teleop_enabled:
            return None, False

        if packet is None or time.monotonic() - self.last_rx > self.args.rx_timeout:
            self.teleop_enabled = False
            print("[RIGHT] paused: UDP timeout")
            return None, False

        if not button_pressed(packet.buttons, GRIP_BIT):
            self.targeter.update_from_packet(packet)

        action = rate_limited_command_action(self.command_state, self.targeter.target)
        return action, self.recording_enabled

    def finalize_pending_recording_stop(self) -> None:
        if self.pending_recording_stop_at is None:
            return
        elapsed = time.monotonic() - self.pending_recording_stop_at
        if elapsed >= self.args.discard_double_click_window:
            self.pending_recording_stop_at = None
            self.recording_save_requested = True
            print("[RIGHT] recording save confirmed")

    def consume_recording_stop_request(self):
        if self.recording_discard_requested:
            self.recording_discard_requested = False
            return "discard"
        if self.recording_save_requested:
            self.recording_save_requested = False
            return "save"
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Record right-arm VR teleop data for SmolVLA.")
    parser.add_argument("--right-arm-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="right_so100")
    parser.add_argument("--udp-ip", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5050)
    parser.add_argument("--hand-id", type=int, choices=[0, 1], default=RIGHT_HAND_ID)
    parser.add_argument("--rx-timeout", type=float, default=RX_TIMEOUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--ros-camera",
        type=parse_ros_camera_value,
        action="append",
        default=[],
        metavar="NAME:/TOPIC:WIDTHxHEIGHT",
    )
    parser.add_argument("--camera-connect-timeout", type=float, default=5.0)
    parser.add_argument("--camera-frame-timeout-ms", type=float, default=1000.0)
    parser.add_argument("--udp-video", action="store_true", help="send recorded camera streams over UDP")
    parser.add_argument("--udp-video-ip", default=DEFAULT_UDP_VIDEO_IP)
    parser.add_argument("--udp-video-port", type=int, default=DEFAULT_UDP_VIDEO_PORT)
    parser.add_argument("--udp-video-camera", default="head")
    parser.add_argument(
        "--udp-video-left-wrist-port",
        type=int,
        default=DEFAULT_LEFT_WRIST_UDP_VIDEO_PORT,
    )
    parser.add_argument(
        "--udp-video-right-wrist-port",
        type=int,
        default=DEFAULT_RIGHT_WRIST_UDP_VIDEO_PORT,
    )
    parser.add_argument("--udp-video-jpeg-quality", type=int, default=50)
    parser.add_argument("--udp-video-every", type=int, default=1)
    parser.add_argument("--udp-video-verbose", action="store_true")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=str(REPO_ROOT / "data" / "datasets" / "right_arm_vr"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--episode-len", type=int, default=450)
    parser.add_argument("--min-episode-frames", type=int, default=90)
    parser.add_argument("--default-ee-x", type=float, default=0.150)
    parser.add_argument("--default-ee-y", type=float, default=0.0)
    parser.add_argument("--default-pan", type=float, default=-10.0)
    parser.add_argument("--default-wrist-roll", type=float, default=-90.0)
    parser.add_argument("--discard-double-click-window", type=float, default=0.5)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    if not args.ros_camera:
        args.ros_camera = list(DEFAULT_ROS_CAMERAS)
    if args.camera_connect_timeout <= 0 or args.camera_frame_timeout_ms <= 0:
        parser.error("camera timeouts must be positive")

    robot = None
    sock = None
    udp_videos = []
    dataset = None
    try:
        cameras = make_camera_configs(args)
        robot_config = SO101FollowerConfig(
            port=args.right_arm_port,
            use_degrees=True,
            cameras=cameras,
            id=args.robot_id,
        )
        robot = SO101Follower(robot_config)
        print(f"Connecting right arm on {args.right_arm_port} id={args.robot_id}")
        print(f"Cameras: {', '.join(cameras) if cameras else 'none'}")
        connect_with_default_calibration(robot)

        dataset = make_dataset(robot, args, cameras)
        recorder = RightArmVrRecorder(robot, args)
        if args.udp_video:
            udp_video_targets = [
                (args.udp_video_camera, args.udp_video_port),
                ("left_wrist", args.udp_video_left_wrist_port),
                ("right_wrist", args.udp_video_right_wrist_port),
            ]
            seen_udp_video_targets = set()
            for camera_name, port in udp_video_targets:
                target = (camera_name, port)
                if target in seen_udp_video_targets:
                    continue
                seen_udp_video_targets.add(target)
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

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UDP_RCVBUF)
        sock.bind((args.udp_ip, args.udp_port))
        sock.setblocking(False)

        print(f"UDP listening on {args.udp_ip}:{args.udp_port}")
        print(f"Listening for controller hand_id={args.hand_id}")
        print("Grip: enable/pause teleop. Up/B: start/stop recording; double B discards. A/down: return default.")

        episode = 0
        frame = 0
        period = 1.0 / args.fps
        latest_packet = None

        while episode < args.num_episodes:
            loop_start = time.perf_counter()
            while True:
                try:
                    data, _addr = sock.recvfrom(1024)
                except BlockingIOError:
                    break
                if len(data) != PACKET_SIZE:
                    continue
                packet = parse_packet(data)
                if packet.hand_id != args.hand_id:
                    continue
                latest_packet = packet
                recorder.last_rx = time.monotonic()
                recorder.handle_buttons(packet)

            recorder.finalize_pending_recording_stop()
            recording_stop_request = recorder.consume_recording_stop_request()
            if recording_stop_request == "save":
                if frame > 0:
                    print(f"\nSaving episode {episode} ({frame} frames)")
                    save_episode_after_images(dataset)
                    episode += 1
                    frame = 0
                    print(f"Episode saved. Press right up/B when ready for episode {episode}.")
                    if episode >= args.num_episodes:
                        break
                else:
                    print("\nNo frames recorded; nothing to save.")
            elif recording_stop_request == "discard":
                if frame > 0:
                    print(f"\nDiscarding episode {episode} ({frame} frames)")
                    dataset.clear_episode_buffer(delete_images=True)
                    frame = 0
                else:
                    print("\nNo frames recorded; nothing to discard.")

            action, should_record = recorder.make_action(latest_packet)
            if action is not None:
                action = robot.send_action(action)

            obs = None
            if udp_videos or (should_record and action is not None):
                obs = robot.get_observation()
                for udp_video in udp_videos:
                    udp_video.send(obs)

            if should_record and action is not None:
                observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": args.task})
                frame += 1
                print(f"episode={episode} frame={frame}/{args.episode_len}", end="\r")

                if frame >= args.episode_len:
                    print(f"\nDiscarding episode {episode} ({frame} frames >= {args.episode_len} max frames)")
                    dataset.clear_episode_buffer(delete_images=True)
                    frame = 0
                    recorder.recording_enabled = False
                    recorder.pending_recording_stop_at = None
                    recorder.teleop_enabled = False
                    print(f"Episode discarded. Press right grip when ready for episode {episode}.")

            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

        if args.push_to_hub:
            dataset.push_to_hub()
        print("Done.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if dataset is not None and frame > 0:
            if frame >= args.min_episode_frames:
                print(f"Saving partial episode ({frame} frames).")
                save_episode_after_images(dataset)
            else:
                print(
                    f"Discarding short partial episode ({frame} frames < "
                    f"{args.min_episode_frames} min frames)."
                )
                dataset.clear_episode_buffer(delete_images=True)
        return 0
    finally:
        if sock is not None:
            sock.close()
        for udp_video in udp_videos:
            udp_video.close()
        if robot is not None and robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
