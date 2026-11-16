import struct
import runpy
import threading
import time
from pathlib import Path

import numpy as np

from xcore_interact.pico_teleop import (
    ButtonEdgeTracker,
    DOWN_BIT,
    DualControllerReceiver,
    GRIP_BIT,
    TRIGGER_BIT,
    UP_BIT,
    axis_angle_matrix,
    compose_dual_record_vectors,
    limited_position_step,
    map_position_offset,
    map_rotation_offset,
    matrix_axis_angle,
    parse_controller_packet,
    scaled_rotation,
)


def _recording_counter_class():
    import runpy
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "recording"
        / "record_dual_rgbd.py"
    )
    return runpy.run_path(str(path))["RightUpEdgeCounter"]


def _recording_wait_status():
    import runpy

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "recording"
        / "record_dual_rgbd.py"
    )
    return runpy.run_path(str(path))["_telemetry_wait_status"]


def _segment_progress_class():
    import runpy

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "recording"
        / "record_dual_rgbd.py"
    )
    return runpy.run_path(str(path))["SegmentProgress"]


def _normalize_subtask_descriptions():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "recording"
        / "record_dual_rgbd.py"
    )
    return runpy.run_path(str(path))["normalize_subtask_descriptions"]


def _dual_servol_symbols():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "teleoperation"
        / "pico_dual_servol.py"
    )
    return runpy.run_path(str(path))


def _packet(hand_id: int, buttons: int) -> bytes:
    message = bytearray(38)
    message[0] = hand_id
    struct.pack_into("<9f", message, 1, 100.0, -200.0, 300.0, 0, 0, 0, 1, 0, 0)
    message[37] = buttons
    return bytes(message)


def test_dual_packet_ids_and_button_bits():
    buttons = (1 << TRIGGER_BIT) | (1 << GRIP_BIT) | (1 << DOWN_BIT) | (1 << UP_BIT)
    left = parse_controller_packet(_packet(0, buttons))
    right = parse_controller_packet(_packet(1, buttons))
    assert left.side == "left"
    assert right.side == "right"
    assert left.trigger and left.grip and left.down and left.up
    np.testing.assert_allclose(right.position_m, [0.1, -0.2, 0.3])


def test_left_and_right_button_edges_are_independent():
    tracker = ButtonEdgeTracker()
    tracker.update("left", 0)
    tracker.update("right", 0)
    assert tracker.update("left", 1 << GRIP_BIT).grip_rising
    assert not tracker.update("right", 0).grip_rising
    assert tracker.update("right", 1 << DOWN_BIT).down_rising
    assert tracker.update("left", 1 << UP_BIT).up_rising


def test_web_elbow_target_is_rate_limited_and_command_is_side_specific(tmp_path: Path):
    symbols = _dual_servol_symbols()
    next_elbow_target = symbols["next_elbow_target"]
    read_elbow_command = symbols["read_elbow_command"]
    first, velocity = next_elbow_target(0.0, 1.0, 0.0)
    assert 0.0 < first < np.deg2rad(5.0) * 0.02
    assert 0.0 < velocity <= np.deg2rad(20.0) * 0.02

    command = tmp_path / "elbow.txt"
    command.write_text("42 -0.25 0.75 1 0 1\n", encoding="utf-8")
    assert read_elbow_command(str(command), "left", 0) == (42, -0.25, True, True)
    assert read_elbow_command(str(command), "right", 0) == (42, 0.75, False, True)
    assert read_elbow_command(str(command), "right", 42) is None


def test_control_transition_waits_for_new_starting_arm_pose():
    receiver = DualControllerReceiver()
    receiver._update(parse_controller_packet(_packet(0, 0)))
    receiver._update(parse_controller_packet(_packet(1, 0)))
    receiver.set_teleop_active("left", True)
    receiver.set_teleop_active("right", True)
    right_sequence = receiver.latest("right").sequence

    initial_epoch, initial_active = receiver.control_transition_state()
    assert not initial_active
    receiver.begin_control_transition()
    active_epoch, transition_active = receiver.control_transition_state()
    assert transition_active and active_epoch > initial_epoch

    def publish_new_right_pose():
        time.sleep(0.01)
        receiver._update(parse_controller_packet(_packet(1, 0)))

    worker = threading.Thread(target=publish_new_right_pose)
    worker.start()
    hands = receiver.wait_for_active_poses_after("right", right_sequence, 0.5)
    worker.join(timeout=0.5)
    assert set(hands) == {"left", "right"}
    assert hands["right"].sequence > right_sequence

    receiver.end_control_transition()
    final_epoch, transition_active = receiver.control_transition_state()
    assert not transition_active and final_epoch > active_epoch


def test_pose_gains_and_rate_limits_match_xcore_act():
    stepped = limited_position_step(np.zeros(3), np.array([0.01, 0, 0]))
    np.testing.assert_allclose(stepped, [0.001, 0, 0])
    raw = axis_angle_matrix(np.array([0, 0, 1]), np.deg2rad(90))
    _, mapped_angle = matrix_axis_angle(scaled_rotation(raw, gain=0.25))
    np.testing.assert_allclose(mapped_angle, np.deg2rad(22.5), atol=1e-8)


def test_left_position_axes_follow_left_arm_reference_frame():
    # Physical calibration: PICO left=-Y, up=-Z, forward=-X;
    # robot left=-Z, up=+Y, forward=-X.
    np.testing.assert_allclose(
        map_position_offset("left", [0.0, -0.1, 0.0]), [0.0, 0.0, -0.1]
    )
    np.testing.assert_allclose(
        map_position_offset("left", [0.0, 0.0, -0.1]), [0.0, 0.1, 0.0]
    )
    np.testing.assert_allclose(
        map_position_offset("left", [-0.1, 0.0, 0.0]), [-0.1, 0.0, 0.0]
    )


def test_right_position_axes_keep_direct_mapping():
    np.testing.assert_allclose(
        map_position_offset("right", [0.1, -0.2, 0.3]), [0.1, -0.2, 0.3]
    )


def test_left_rotation_axes_follow_left_arm_reference_frame():
    # Rotation uses the exact same physical-axis calibration as translation:
    # PICO left=-Y, up=-Z, forward=-X;
    # robot left=-Z, up=+Y, forward=-X.
    cases = (
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([-1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
    )
    for pico_axis, robot_axis in cases:
        mapped = map_rotation_offset("left", axis_angle_matrix(pico_axis, 0.2))
        mapped_axis, mapped_angle = matrix_axis_angle(mapped)
        np.testing.assert_allclose(mapped_axis, robot_axis, atol=1e-8)
        np.testing.assert_allclose(mapped_angle, 0.2, atol=1e-8)


def test_right_rotation_axes_keep_direct_mapping():
    raw = axis_angle_matrix(np.array([0.2, -0.3, 0.4]), 0.2)
    np.testing.assert_allclose(map_rotation_offset("right", raw), raw, atol=1e-8)


def test_dual_record_vectors_hold_only_the_inactive_arm_at_current_state():
    left_joints = np.arange(7, dtype=np.float64) * 0.1
    right_joints = np.arange(7, dtype=np.float64) * -0.1
    gripper_state = {"left": 3700, "right": 1000}
    gripper_target = {"left": 4000, "right": 1500}

    state, action = compose_dual_record_vectors(
        left_joints,
        right_joints,
        gripper_state,
        gripper_target,
        frozenset({"right"}),
    )

    np.testing.assert_array_equal(action[:8], state[:8])
    np.testing.assert_array_equal(action[8:15], state[8:15])
    assert action[15] == 1500
    assert state[15] == 1000


def test_dual_record_vectors_apply_targets_when_both_arms_are_active():
    state, action = compose_dual_record_vectors(
        np.zeros(7),
        np.zeros(7),
        {"left": 3700, "right": 1000},
        {"left": 4000, "right": 1500},
        frozenset({"left", "right"}),
    )
    assert (state[7], state[15]) == (3700, 1000)
    assert (action[7], action[15]) == (4000, 1500)


def test_only_right_up_drives_recording_edges():
    counter = _recording_counter_class()()
    assert counter.feed([0]) == 0
    assert counter.feed([1 << UP_BIT]) == 1
    assert counter.feed([1 << UP_BIT]) == 0
    assert counter.feed([0, 1 << UP_BIT]) == 1


def test_four_segments_form_exactly_one_episode_progression():
    progress = _segment_progress_class()(target_segments=4, min_segment_frames=2)
    total_frames = 0
    for expected_segment in range(1, 5):
        assert progress.segment_number == expected_segment
        total_frames += expected_segment + 1
        assert progress.finish_segment(total_frames) == expected_segment + 1
    assert progress.complete
    assert progress.completed_frames == [2, 3, 4, 5]
    assert progress.manifest(["approach", "grasp", "move", "place"]) == [
        {"segment": 1, "title": "approach", "subtask": 1, "description": "approach", "start_frame": 0, "end_frame": 1, "frames": 2},
        {"segment": 2, "title": "grasp", "subtask": 2, "description": "grasp", "start_frame": 2, "end_frame": 4, "frames": 3},
        {"segment": 3, "title": "move", "subtask": 3, "description": "move", "start_frame": 5, "end_frame": 8, "frames": 4},
        {"segment": 4, "title": "place", "subtask": 4, "description": "place", "start_frame": 9, "end_frame": 13, "frames": 5},
    ]


def test_interact_subtask_count_is_limited_to_one_through_eight():
    normalize = _normalize_subtask_descriptions()
    descriptions = [f"subtask {index}" for index in range(1, 9)]
    assert normalize(8, descriptions, "overall") == descriptions
    assert normalize(1, [], " overall ") == ["overall"]
    for count in (0, 9):
        try:
            normalize(count, [], "overall")
        except ValueError as error:
            assert "[1,8]" in str(error)
        else:
            raise AssertionError(f"invalid subtask count {count} was accepted")


def test_one_and_eight_subtasks_gate_episode_completion():
    progress_class = _segment_progress_class()
    for task_count in (1, 8):
        progress = progress_class(target_segments=task_count, min_segment_frames=1)
        for task_index in range(task_count):
            assert not progress.complete
            progress.finish_segment(task_index + 1)
        assert progress.complete
        assert progress.completed_segments == task_count


def test_interact_subtask_descriptions_are_complete_and_unique():
    normalize = _normalize_subtask_descriptions()
    for descriptions in (["only one"], ["grasp", " GRASP "]):
        try:
            normalize(2, descriptions, "overall")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid descriptions were accepted: {descriptions}")


def test_short_segment_cannot_advance_episode():
    progress = _segment_progress_class()(target_segments=4, min_segment_frames=3)
    try:
        progress.finish_segment(2)
    except ValueError as error:
        assert "minimum is 3" in str(error)
    else:
        raise AssertionError("short segment unexpectedly advanced")
    assert progress.completed_segments == 0
    assert progress.segment_number == 1


def test_dual_recorder_waits_for_grip_after_task_is_ready():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "recording"
        / "record_dual_rgbd.py"
    ).read_text(encoding="utf-8")
    assert "telemetry.wait_ready(" not in source
    assert "telemetry.latest(require_active=True)" in source
    assert '"waiting_teleop"' in source


def test_dual_recorder_pauses_instead_of_failing_on_temporary_telemetry_loss():
    wait_status = _recording_wait_status()
    assert wait_status(RuntimeError("dual teleoperation is not active")) == (
        "waiting_teleop",
        "双臂均未遥操，等待至少一侧 Grip 开启",
    )
    phase, message = wait_status(
        RuntimeError("dual state/action telemetry is stale")
    )
    assert phase == "waiting_samples"
    assert "telemetry is stale" in message


def test_dual_recorder_rebases_slow_frames_instead_of_aborting():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "recording"
        / "record_dual_rgbd.py"
    ).read_text(encoding="utf-8")
    assert "recorder fell behind" not in source
    assert "next_tick = time.monotonic()" in source


def test_segmented_recorder_keeps_double_b_episode_discard_window():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "recording"
        / "record_dual_rgbd.py"
    ).read_text(encoding="utf-8")
    assert "discard_double_click_window_s" in source
    assert "double_click_deadline" in source
    assert "discard_episode = True" in source
