"""Tests for camera health-state transitions."""

from collections import deque

from rclpy.time import Time

from robot_vision_core.camera_manager import (
    STATE_OFFLINE,
    STATE_OK,
    STATE_STALE,
    _CameraMonitor,
)


def make_monitor():
    """Construct the state-only part of a monitor without a ROS node."""
    monitor = _CameraMonitor.__new__(_CameraMonitor)
    monitor.camera_id = "test"
    monitor.camera_name = "test"
    monitor.model = "D435i"
    monitor.semantic_location = "test"
    monitor.expected_scale = 0.001
    monitor.enable_color = True
    monitor.enable_depth = True
    monitor.align_depth = False
    monitor.last_color_time = None
    monitor.last_depth_time = None
    monitor.last_aligned_time = None
    monitor._hz_window_sec = 5.0
    monitor._color_times = deque()
    return monitor


def status_at(monitor, seconds):
    """Compute status with the standard test thresholds."""
    return monitor.compute_status(
        Time(seconds=seconds),
        stale_sec=2.0,
        offline_sec=10.0,
        grace_sec=0.0,
        node_start=Time(seconds=0),
    )


def test_silent_camera_transitions_from_stale_to_offline():
    """A camera that once worked must eventually become OFFLINE."""
    monitor = make_monitor()
    monitor.last_color_time = Time(seconds=1)
    monitor.last_depth_time = Time(seconds=1)

    assert status_at(monitor, 5).state == STATE_STALE
    assert status_at(monitor, 12).state == STATE_OFFLINE


def test_frame_rate_window_expires_without_new_callbacks():
    """Old color samples must be removed by periodic status evaluation."""
    monitor = make_monitor()
    monitor.last_color_time = Time(seconds=1)
    monitor.last_depth_time = Time(seconds=1)
    monitor._color_times.extend((Time(seconds=0), Time(seconds=1)))

    status = status_at(monitor, 12)

    assert status.last_hz == 0.0
    assert not monitor._color_times


def test_last_seen_uses_newest_enabled_stream():
    """The aggregate age must also work for a depth-only camera."""
    monitor = make_monitor()
    monitor.enable_color = False
    monitor.last_depth_time = Time(seconds=4)

    status = status_at(monitor, 5)

    assert status.state == STATE_OK
    assert status.last_seen_age_sec == 1.0
