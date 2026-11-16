#!/usr/bin/env python3
"""Record bimanual PICO demonstrations with head XYZRGB and three RGB videos.

Motion remains owned by the Python dual-PICO ServoL controller. This process only reads
its ROS telemetry, captures cameras, and writes a LeRobot dataset. Optional local
control/status paths expose the same lifecycle used by the xCore web workbench.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from xcore_interact.cameras import RosRgbdRig
from xcore_interact.schema import (
    ACTION_KEY,
    HEAD_IMAGE_KEY,
    HEAD_POINT_CLOUD_KEY,
    LEFT_WRIST_IMAGE_KEY,
    RIGHT_WRIST_IMAGE_KEY,
    STATE_KEY,
    CameraConfig,
    dataset_features,
)
from xcore_interact.telemetry import DualTelemetrySubscriber
from xcore_interact.web_runtime import (
    AtomicStatusWriter,
    ExternalControlServer,
    WebPreviewPublisher,
)

RIGHT_UP_BIT = 3
CAMERA_NAMES = ["head", "left_wrist", "right_wrist"]
MIN_SUBTASKS = 1
MAX_SUBTASKS = 8


class RightUpEdgeCounter:
    """Count rising edges while requiring one observed initial button state."""

    def __init__(self) -> None:
        self.previous: bool | None = None

    def feed(self, button_values: list[int]) -> int:
        count = 0
        for buttons in button_values:
            current = bool(int(buttons) & (1 << RIGHT_UP_BIT))
            if self.previous is not None and current and not self.previous:
                count += 1
            self.previous = current
        return count


@dataclass
class SegmentProgress:
    target_segments: int
    min_segment_frames: int
    segment_start_frame: int = 0
    completed_frames: list[int] = field(default_factory=list)

    @property
    def completed_segments(self) -> int:
        return len(self.completed_frames)

    @property
    def complete(self) -> bool:
        return self.completed_segments >= self.target_segments

    @property
    def segment_number(self) -> int:
        return min(self.completed_segments + 1, self.target_segments)

    def current_frames(self, total_frames: int) -> int:
        return max(0, int(total_frames) - self.segment_start_frame)

    def finish_segment(self, total_frames: int) -> int:
        frames = self.current_frames(total_frames)
        if frames < self.min_segment_frames:
            raise ValueError(
                f"segment has {frames} frames; minimum is {self.min_segment_frames}"
            )
        if self.complete:
            raise RuntimeError("all recording segments are already complete")
        self.completed_frames.append(frames)
        self.segment_start_frame = int(total_frames)
        return frames

    def manifest(self, segment_titles: list[str]) -> list[dict[str, int | str]]:
        if len(segment_titles) != self.target_segments:
            raise ValueError("segment title count must match target segments")
        start = 0
        result: list[dict[str, int | str]] = []
        for index, frames in enumerate(self.completed_frames, start=1):
            result.append(
                {
                    "segment": index,
                    "title": segment_titles[index - 1],
                    "subtask": index,
                    "description": segment_titles[index - 1],
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
    overall_task: str,
) -> list[str]:
    """Validate the 1–8 ordered task labels stored in every interACT Episode."""
    overall_task = overall_task.strip()
    if not overall_task:
        raise ValueError("overall task description must not be blank")
    if len(overall_task) > 500:
        raise ValueError("overall task description must be at most 500 characters")
    if not MIN_SUBTASKS <= task_count <= MAX_SUBTASKS:
        raise ValueError(
            f"segments-per-episode must be in [{MIN_SUBTASKS},{MAX_SUBTASKS}]"
    )
    normalized = [str(description).strip() for description in descriptions]
    if not normalized:
        if task_count == 1:
            normalized = [overall_task]
        else:
            raise ValueError(
                "one segment-title is required for every interACT subtask"
            )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/xcore_bimanual_interact3d")
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--control-mode", choices=("bimanual", "right_only"), default="bimanual"
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--episode-time-s", type=float, default=20.0)
    parser.add_argument("--min-episode-frames", type=int, default=20)
    parser.add_argument("--segments-per-episode", type=int, default=4)
    parser.add_argument(
        "--segment-title",
        dest="segment_titles",
        action="append",
        default=[],
        help="title for one segment; repeat exactly once per segment",
    )
    parser.add_argument("--point-count", type=int, default=1024)
    parser.add_argument("--use-point-cloud", action="store_true",
                        help="Also record aligned head depth as XYZRGB; default records RGB only.")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument(
        "--keyboard-control",
        action="store_true",
        help="use Enter and fixed episode time instead of right Up/B",
    )
    parser.add_argument("--discard-double-click-window-s", type=float, default=0.5)
    parser.add_argument("--control-socket", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--web-preview-dir", type=Path)
    parser.add_argument("--web-preview-width", type=int, default=960)
    parser.add_argument("--web-preview-jpeg-quality", type=int, default=70)
    parser.add_argument("--web-preview-every", type=int, default=1)
    return parser.parse_args()


def _raise_keyboard_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _preview_frames(camera: object) -> dict[str, np.ndarray]:
    return {
        "head": camera.head_rgb,
        "left_wrist": camera.left_wrist_rgb,
        "right_wrist": camera.right_wrist_rgb,
    }


def _telemetry_wait_status(error: RuntimeError) -> tuple[str, str]:
    detail = str(error)
    if detail == "dual teleoperation is not active":
        return "waiting_teleop", "双臂均未遥操，等待至少一侧 Grip 开启"
    return "waiting_samples", f"等待新鲜双臂状态/动作数据：{detail}"


def main() -> int:
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.fps <= 0 or args.num_episodes <= 0 or args.episode_time_s <= 0:
        raise ValueError("fps, num-episodes and episode-time-s must be positive")
    if args.min_episode_frames <= 0:
        raise ValueError("min-episode-frames must be positive")
    segment_titles = normalize_subtask_descriptions(
        args.segments_per_episode,
        args.segment_titles,
        args.task,
    )
    if args.discard_double_click_window_s < 0:
        raise ValueError("discard-double-click-window-s must be non-negative")
    if args.web_preview_width <= 0 or args.web_preview_every <= 0:
        raise ValueError("web preview width/every must be positive")
    if not 5 <= args.web_preview_jpeg_quality <= 100:
        raise ValueError("web preview JPEG quality must be in [5,100]")

    root = args.dataset_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"dataset root is not empty: {root}")
    camera_config = CameraConfig(point_count=args.point_count, use_point_cloud=args.use_point_cloud)
    max_frames = max(1, round(args.episode_time_s * args.fps))
    if args.min_episode_frames > max_frames:
        raise ValueError("min-episode-frames cannot exceed episode frame limit")

    control = ExternalControlServer(args.control_socket) if args.control_socket else None
    status_writer = AtomicStatusWriter(args.status_file) if args.status_file else None
    preview = (
        WebPreviewPublisher(
            args.web_preview_dir,
            width=args.web_preview_width,
            jpeg_quality=args.web_preview_jpeg_quality,
            publish_every=args.web_preview_every,
        )
        if args.web_preview_dir
        else None
    )
    dataset: LeRobotDataset | None = None
    episode = 0
    frame_index = 0
    final_status = "completed"
    final_error: str | None = None
    last_event: dict | None = None
    segment_progress: SegmentProgress | None = None
    segment_recording = False
    segment_manifests: list[dict] = []

    def publish(
        status: str,
        phase: str,
        message: str,
        *,
        sample: object | None = None,
        force: bool = False,
    ) -> None:
        if status_writer is None:
            return
        status_writer.publish(
            {
                "status": status,
                "phase": phase,
                "message": message,
                "episode": episode,
                "episode_number": episode + 1,
                "frame": frame_index,
                "segment_number": (
                    1 if segment_progress is None else segment_progress.segment_number
                ),
                "segment_title": segment_titles[
                    (1 if segment_progress is None else segment_progress.segment_number) - 1
                ],
                "target_segments": args.segments_per_episode,
                "completed_segments": (
                    0 if segment_progress is None else segment_progress.completed_segments
                ),
                "segment_frame": (
                    0
                    if segment_progress is None
                    else segment_progress.current_frames(frame_index)
                ),
                "segment_recording": segment_recording,
                "button_press_count": (
                    0
                    if segment_progress is None
                    else segment_progress.completed_segments * 2
                    + (1 if segment_recording else 0)
                ),
                "next_button_action": (
                    "save_segment" if segment_recording else "start_segment"
                ),
                "subtask_number": (
                    1 if segment_progress is None else segment_progress.segment_number
                ),
                "subtask_description": segment_titles[
                    (1 if segment_progress is None else segment_progress.segment_number) - 1
                ],
                "task_count": args.segments_per_episode,
                "completed_subtasks": (
                    0 if segment_progress is None else segment_progress.completed_segments
                ),
                "subtask_frame": (
                    0
                    if segment_progress is None
                    else segment_progress.current_frames(frame_index)
                ),
                "subtask_recording": segment_recording,
                "target_episodes": args.num_episodes,
                "control_mode": args.control_mode,
                "episode_len": max_frames,
                "recording_enabled": status == "recording",
                "teleop_active": sample is not None,
                "state_fresh": sample is not None,
                "action_fresh": sample is not None,
                "cameras": CAMERA_NAMES,
                "dataset_root": str(root),
                "state": None if sample is None else sample.state.tolist(),
                "action": None if sample is None else sample.action.tolist(),
                "last_event": last_event,
                **({"error": final_error} if final_error else {}),
            },
            force=force,
        )

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    publish("starting", "starting", "正在初始化双臂 InterACT 录制器", force=True)
    metadata = {
        "schema_version": 1,
        "model_family": "InterACT+head_PointNet" if args.use_point_cloud else "InterACT_RGB",
        "state_action_layout": "left_7j,left_gripper,right_7j,right_gripper",
        "control_mode": args.control_mode,
        "left_gripper_installed": True,
        "left_gripper_open": 4000.0,
        "left_gripper_closed": 3600.0,
        "right_gripper_open": 1500.0,
        "right_gripper_closed": 900.0,
        "head_depth_only": args.use_point_cloud,
        "camera_transport": "ros2",
        "head_color_topic": camera_config.head_image_topic,
        "head_aligned_depth_topic": camera_config.head_depth_topic if args.use_point_cloud else None,
        "left_wrist_color_topic": camera_config.left_wrist_image_topic,
        "right_wrist_color_topic": camera_config.right_wrist_image_topic,
        "recording_control": (
            "web_and_right_up"
            if control
            else "automatic"
            if args.no_prompt
            else "keyboard"
            if args.keyboard_control
            else "right_up"
        ),
        "overall_task": args.task.strip(),
        "task_count": args.segments_per_episode,
        "subtask_descriptions": segment_titles,
        "segments_per_episode": args.segments_per_episode,
        "segment_titles": segment_titles,
        "segment_min_frames": args.min_episode_frames,
        "segment_max_frames": max_frames,
        "point_fields": ["x_m", "y_m", "z_m", "r", "g", "b"] if args.use_point_cloud else [],
    }

    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=root,
            fps=args.fps,
            robot_type="xcore_bimanual_interact3d" if args.use_point_cloud else "xcore_bimanual_interact_rgb",
            features=dataset_features(camera_config),
            use_videos=True,
            video_backend="pyav",
            image_writer_threads=4,
        )
        with DualTelemetrySubscriber() as telemetry, RosRgbdRig(camera_config) as cameras:
            # Match the proven single-arm recorder: creating a recording task
            # must not require ServoL samples before Grip has been enabled.
            # The recording loop below waits for an active, fresh sample before
            # it writes the first frame, so inactive/reset trajectories remain
            # excluded from the dataset.
            right_up = RightUpEdgeCounter()
            initialization_deadline = time.monotonic() + 2.0
            while right_up.previous is None and time.monotonic() < initialization_deadline:
                telemetry.spin_once(0.05)
                right_up.feed(telemetry.drain_button_events())

            initial_camera = cameras.read(point_seed=0)
            if preview:
                preview.publish(_preview_frames(initial_camera))
            publish(
                "ready",
                "ready",
                "三路相机与采集器已就绪，等待开始录制",
                force=True,
            )

            stop_session = False
            abort_session = False
            while episode < args.num_episodes and not stop_session and not abort_session:
                frame_index = 0
                segment_progress = SegmentProgress(
                    args.segments_per_episode, args.min_episode_frames
                )
                segment_recording = False
                discard_episode = False

                while (
                    not segment_progress.complete
                    and not stop_session
                    and not abort_session
                    and not discard_episode
                ):
                    start_requested = args.no_prompt
                    if (
                        not start_requested
                        and args.keyboard_control
                        and control is None
                    ):
                        input(
                            f"Episode {episode + 1}/{args.num_episodes}, "
                            f"subtask {segment_progress.segment_number}/"
                            f"{args.segments_per_episode}: press Enter to start"
                        )
                        start_requested = True
                    while (
                        not start_requested
                        and not stop_session
                        and not abort_session
                        and not discard_episode
                    ):
                        telemetry.spin_once(0.05)
                        button_start = bool(
                            right_up.feed(telemetry.drain_button_events())
                        )
                        commands = control.drain() if control else []
                        start_requested = button_start or any(
                            command in {"start", "resume"} for command in commands
                        )
                        stop_session = "stop_session" in commands
                        abort_session = "abort_session" in commands
                        discard_episode = "discard_episode" in commands
                        publish(
                            "ready",
                            "segment_ready",
                            f"Episode {episode + 1}：等待 B 开始第 "
                            f"{segment_progress.segment_number}/"
                            f"{args.segments_per_episode} 个子任务",
                        )
                    if stop_session or abort_session or discard_episode:
                        break

                    # The odd-numbered B press has started this segment. Wait
                    # for at least one arm to become active before writing.
                    segment_recording = True
                    while True:
                        commands = control.drain() if control else []
                        if "stop_session" in commands:
                            stop_session = True
                            break
                        if "abort_session" in commands:
                            abort_session = True
                            break
                        if "discard_episode" in commands:
                            discard_episode = True
                            break
                        try:
                            ready_sample = telemetry.latest(require_active=True)
                            break
                        except RuntimeError:
                            telemetry.spin_once(0.05)
                            publish(
                                "recording",
                                "waiting_teleop",
                                f"第 {segment_progress.segment_number}/"
                                f"{args.segments_per_episode} 个子任务已触发，"
                                "等待至少一侧 Grip 开启遥操",
                            )
                    if stop_session or abort_session or discard_episode:
                        break

                    paused = False
                    segment_finished = False
                    next_tick = time.monotonic()
                    publish(
                        "recording",
                        "recording",
                        f"正在采集第 {segment_progress.segment_number}/"
                        f"{args.segments_per_episode} 个子任务",
                        sample=ready_sample,
                        force=True,
                    )
                    while True:
                        telemetry.spin_once(0.0)
                        button_finish = bool(
                            right_up.feed(telemetry.drain_button_events())
                        )
                        commands = control.drain() if control else []
                        if "abort_session" in commands:
                            abort_session = True
                            break
                        if "stop_session" in commands:
                            stop_session = True
                            break
                        if "discard_episode" in commands:
                            discard_episode = True
                            break
                        if "pause" in commands:
                            paused = True
                        if any(command in {"start", "resume"} for command in commands):
                            paused = False
                            next_tick = time.monotonic()

                        finish_requested = button_finish or "save_episode" in commands
                        automatic_finish = (
                            (args.no_prompt or args.keyboard_control)
                            and segment_progress.current_frames(frame_index) >= max_frames
                        )
                        if finish_requested or automatic_finish:
                            current_segment_frames = segment_progress.current_frames(
                                frame_index
                            )
                            if current_segment_frames < args.min_episode_frames:
                                publish(
                                    "recording",
                                    "segment_too_short",
                                    f"第 {segment_progress.segment_number} 个子任务只有 "
                                    f"{current_segment_frames} 帧，至少需要 "
                                    f"{args.min_episode_frames} 帧；请继续录制",
                                    force=True,
                                )
                                time.sleep(0.05)
                                continue
                            saved_segment = segment_progress.segment_number
                            saved_segment_title = segment_titles[saved_segment - 1]
                            saved_segment_frames = segment_progress.finish_segment(
                                frame_index
                            )
                            segment_recording = False
                            last_event = {
                                "type": "segment_saved",
                                "episode": episode + 1,
                                "segment": saved_segment,
                                "title": saved_segment_title,
                                "frames": saved_segment_frames,
                                "timestamp": time.strftime(
                                    "%Y-%m-%dT%H:%M:%S%z"
                                ),
                            }
                            segment_finished = True
                            break

                        if paused:
                            publish(
                                "paused",
                                "paused",
                                f"第 {segment_progress.segment_number}/"
                                f"{args.segments_per_episode} 个子任务已暂停",
                            )
                            time.sleep(0.02)
                            continue

                        if segment_progress.current_frames(frame_index) >= max_frames:
                            publish(
                                "recording",
                                "segment_limit",
                                f"第 {segment_progress.segment_number} 个子任务已达到 "
                                f"{max_frames} 帧上限，请按 B 完成当前子任务",
                            )
                            time.sleep(0.02)
                            continue

                        telemetry.spin_once(0.0)
                        try:
                            sample = telemetry.latest(require_active=True)
                        except RuntimeError as error:
                            wait_phase, wait_message = _telemetry_wait_status(error)
                            publish("recording", wait_phase, wait_message)
                            next_tick = time.monotonic()
                            time.sleep(0.02)
                            continue
                        camera = cameras.read(
                            point_seed=episode * 1_000_000 + frame_index
                        )
                        if preview:
                            preview.publish(_preview_frames(camera))
                        state = sample.state.astype(np.float32)
                        action = sample.action.astype(np.float32)
                        if args.control_mode == "right_only":
                            action = action.copy()
                            action[0:8] = state[0:8]
                        dataset.add_frame(
                            {
                                STATE_KEY: state,
                                ACTION_KEY: action,
                                HEAD_IMAGE_KEY: cv2.cvtColor(
                                    camera.head_rgb, cv2.COLOR_BGR2RGB
                                ),
                                LEFT_WRIST_IMAGE_KEY: cv2.cvtColor(
                                    camera.left_wrist_rgb, cv2.COLOR_BGR2RGB
                                ),
                                RIGHT_WRIST_IMAGE_KEY: cv2.cvtColor(
                                    camera.right_wrist_rgb, cv2.COLOR_BGR2RGB
                                ),
                                **({HEAD_POINT_CLOUD_KEY: camera.head_points} if args.use_point_cloud else {}),
                                "task": segment_titles[
                                    segment_progress.segment_number - 1
                                ],
                            }
                        )
                        frame_index += 1
                        publish(
                            "recording",
                            "recording",
                            f"正在采集第 {segment_progress.segment_number}/"
                            f"{args.segments_per_episode} 个子任务",
                            sample=sample,
                        )
                        next_tick += 1.0 / args.fps
                        delay = next_tick - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                        else:
                            next_tick = time.monotonic()

                    if segment_finished and not (
                        stop_session or abort_session or discard_episode
                    ):
                        double_click_deadline = (
                            time.monotonic() + args.discard_double_click_window_s
                        )
                        publish(
                            "saving",
                            "pending_save",
                            f"第 {segment_progress.completed_segments}/"
                            f"{args.segments_per_episode} 个子任务已完成；"
                            "0.5 秒内再按 B 可丢弃当前 Episode",
                            force=True,
                        )
                        while time.monotonic() < double_click_deadline:
                            telemetry.spin_once(0.02)
                            if right_up.feed(telemetry.drain_button_events()):
                                discard_episode = True
                                break
                            commands = control.drain() if control else []
                            if "discard_episode" in commands:
                                discard_episode = True
                                break
                            if "abort_session" in commands:
                                abort_session = True
                                break
                            if "stop_session" in commands:
                                stop_session = True
                                break

                    if stop_session or abort_session or discard_episode:
                        segment_recording = False
                        break
                    if not segment_progress.complete:
                        publish(
                            "ready",
                            "segment_saved",
                            f"第 {segment_progress.completed_segments}/"
                            f"{args.segments_per_episode} 个子任务已完成；"
                            f"再按 B 开始第 {segment_progress.segment_number} 个子任务",
                            force=True,
                        )

                segment_recording = False
                if discard_episode:
                    dataset.clear_episode_buffer()
                    last_event = {
                        "type": "discarded",
                        "episode": episode + 1,
                        "frames": frame_index,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                    frame_index = 0
                    segment_progress = None
                    publish(
                        "ready",
                        "discarded",
                        f"Episode {episode + 1} 的未保存 "
                        f"{args.segments_per_episode} 个子任务数据已丢弃",
                        force=True,
                    )
                    continue
                if stop_session or abort_session:
                    if frame_index > 0:
                        dataset.clear_episode_buffer()
                        last_event = {
                            "type": "incomplete_discarded",
                            "episode": episode + 1,
                            "frames": frame_index,
                            "segments": segment_progress.completed_segments,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        }
                        frame_index = 0
                    break

                publish(
                    "saving",
                    "saving",
                    f"{args.segments_per_episode} 个子任务已完成，"
                    f"正在保存 Episode {episode + 1}",
                    force=True,
                )
                saved_frames = frame_index
                dataset.save_episode()
                completed_subtasks = segment_progress.manifest(segment_titles)
                segment_manifests.append(
                    {
                        "episode": episode,
                        "frames": saved_frames,
                        "subtasks": completed_subtasks,
                        "segments": completed_subtasks,
                    }
                )
                last_event = {
                    "type": "saved",
                    "episode": episode + 1,
                    "frames": saved_frames,
                    "subtasks": args.segments_per_episode,
                    "segments": args.segments_per_episode,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                episode += 1
                frame_index = 0
                segment_progress = None
                publish(
                    "ready" if episode < args.num_episodes else "completed",
                    "saved",
                    f"Episode {episode} 的 {args.segments_per_episode} 个子任务已保存",
                    force=True,
                )

        final_status = "cancelled" if abort_session else "completed"
        return 0
    except KeyboardInterrupt:
        final_status = "cancelled"
        return 0
    except BaseException as exc:
        final_status = "failed"
        final_error = str(exc)
        raise
    finally:
        if dataset is not None:
            dataset.finalize()
            (root / "meta").mkdir(parents=True, exist_ok=True)
            (root / "meta" / "xcore_interact3d.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            subtask_manifest = {
                "overall_task": args.task.strip(),
                "task_count": args.segments_per_episode,
                "subtask_descriptions": segment_titles,
                # Legacy names remain so existing analysis scripts can still
                # consume newly recorded datasets.
                "segments_per_episode": args.segments_per_episode,
                "segment_titles": segment_titles,
                "episodes": segment_manifests,
            }
            manifest_text = (
                json.dumps(subtask_manifest, indent=2, ensure_ascii=False) + "\n"
            )
            (root / "meta" / "xcore_interact_subtasks.json").write_text(
                manifest_text,
                encoding="utf-8",
            )
            (root / "meta" / "xcore_interact_segments.json").write_text(
                manifest_text,
                encoding="utf-8",
            )
        publish(
            final_status,
            final_status,
            "双臂采集任务已完成"
            if final_status == "completed"
            else "双臂采集任务已放弃"
            if final_status == "cancelled"
            else "双臂采集任务异常",
            force=True,
        )
        if control:
            control.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
