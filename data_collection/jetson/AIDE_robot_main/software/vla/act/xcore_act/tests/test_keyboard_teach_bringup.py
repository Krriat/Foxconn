"""Launcher lifecycle tests. No SDK/USB connections or real ROS graph required."""
from collections import deque
import importlib.util
from itertools import count
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

RECORDING = Path(__file__).resolve().parents[1] / "scripts/recording"
sys.path.insert(0, str(RECORDING))
spec = importlib.util.spec_from_file_location("keyboard_teach_bringup", RECORDING / "launch_xcore_teach_suction.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class BringupTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for relative in ("hardware/arm/xCoreROS/tools/start_upperlimb.sh",
                         "hardware/robot_vision_ws/scripts/start_cameras_30hz.sh"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.args = SimpleNamespace(width=4, height=3, shutdown_timeout=2., m050_timeout_ms=100)
        self.supervisor = launch.Supervisor(self.args, self.root, root=self.root)
        self.monitor = Mock()
        self.monitor.arm_present.return_value = False
        self.monitor.arms_ready.return_value = True
        self.monitor.cameras_ready.return_value = True
        self.monitor.camera_counts.return_value = dict.fromkeys(launch.recorder.CAMERAS, 0)
        self.monitor.clients = [Mock() for _ in range(4)]
        self.monitor.frames = {key: deque([1., 2.]) for key in launch.recorder.CAMERAS}
        self.monitor.shapes = dict(head=(4, 3, "rgb8"))
        self.camera_processes = patch.object(launch, "camera_launches", return_value=[]).start()
        self.camera_lock = patch.object(launch, "camera_lock_busy", return_value=False).start()
        self.addCleanup(patch.stopall)
        # Advance only the startup clock; discovery must not slow unit tests.
        self.clock = patch.object(launch.time, "monotonic", side_effect=lambda: next(self.ticks))
        self.ticks = count(step=.5)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_new_launches_use_existing_entries_without_single_head_camera(self):
        with patch.object(self.supervisor, "start") as start:
            self.supervisor.prepare(self.monitor)
        self.assertEqual([call.args[0] for call in start.call_args_list], ["arms", "cameras"])
        self.assertEqual(start.call_args_list[0].args[1][-1], "start_head_camera:=false")
        self.assertTrue(start.call_args_list[1].args[1][-1].endswith("start_cameras_30hz.sh"))
        for client in self.monitor.clients:
            client.call_async.assert_not_called()

    def test_network_mode_launches_head_only_and_pc_bridge(self):
        self.supervisor.wrist_camera_host = '172.16.12.45'
        (self.root/'hardware/robot_vision_ws/scripts/start_cameras.sh').touch()
        with patch.object(self.supervisor,'start') as start:
            self.supervisor.prepare(self.monitor)
        calls=start.call_args_list
        self.assertEqual([call.args[0] for call in calls],['arms','cameras','wrist_cameras'])
        self.assertTrue(calls[1].args[1][-1].endswith('cameras_head_network.yaml'))
        command=calls[2].args[1]
        self.assertIn('172.16.12.45',command)
        self.assertTrue(any(value.endswith('jetson_camera_bridge.py') for value in command))

    def test_camera_only_mode_never_starts_or_queries_arm_services(self):
        self.supervisor.wrist_camera_host = '172.16.12.45'
        self.supervisor.cameras_only = True
        (self.root/'hardware/robot_vision_ws/scripts/start_cameras.sh').touch()
        with patch.object(self.supervisor,'start') as start:
            self.supervisor.prepare(self.monitor)
        self.assertEqual([call.args[0] for call in start.call_args_list],['cameras','wrist_cameras'])
        self.monitor.arm_present.assert_not_called()
        self.monitor.arms_ready.assert_not_called()

    def test_network_mode_rejects_reusing_local_wrist_publishers(self):
        self.supervisor.wrist_camera_host = '172.16.12.45'
        self.monitor.camera_counts.return_value = dict.fromkeys(launch.recorder.CAMERAS,1)
        with self.assertRaisesRegex(RuntimeError,'--restart-cameras'):
            self.supervisor.prepare_cameras(self.monitor)

    def test_network_readiness_requires_sufficient_received_rate(self):
        ready=launch.Readiness.__new__(launch.Readiness)
        ready.args=SimpleNamespace(width=4,height=3,network_camera_alignment=True,fps=20)
        ready.camera_counts=lambda:dict.fromkeys(launch.recorder.CAMERAS,1)
        ready.shapes=dict.fromkeys(launch.recorder.CAMERAS,(4,3,'rgb8'))
        ready.camera_duplicate_since=ready.camera_ready_since=None
        ready.frames={key:deque([9.,9.5,10.]) for key in launch.recorder.CAMERAS}
        with patch.object(launch.time,'monotonic',return_value=10.):
            self.assertFalse(ready.cameras_ready())
            ready.frames={key:deque(8.5+i/30 for i in range(46)) for key in launch.recorder.CAMERAS}
            ready.camera_ready_since=8.
            self.assertTrue(ready.cameras_ready())

    def test_large_image_transport_environment_is_only_passed_to_cameras(self):
        with patch.dict(launch.os.environ, {"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"}, clear=True):
            with patch.object(launch.subprocess, "Popen") as popen:
                self.supervisor.start("arms", ["fake-arm"])
                self.supervisor.start("cameras", ["fake-camera"])
            self.assertIsNone(popen.call_args_list[0].kwargs['env'])
            camera_env = popen.call_args_list[1].kwargs['env']
            self.assertEqual(camera_env['FASTRTPS_DEFAULT_PROFILES_FILE'], str(launch.CAMERA_DDS_PROFILE))
            self.assertNotIn('FASTRTPS_DEFAULT_PROFILES_FILE', launch.os.environ)

    def test_existing_dds_profiles_and_other_rmw_are_preserved(self):
        for key in ('FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE'):
            with patch.dict(launch.os.environ, {key: '/user/profile.xml'}, clear=True):
                self.assertEqual(launch.camera_environment(), {key: '/user/profile.xml'})
        env = {'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'}
        with patch.dict(launch.os.environ, env, clear=True):
            self.assertEqual(launch.camera_environment(), env)

    def test_local_feedback_port_only_scopes_the_arm_child(self):
        self.args.joint_feedback="local"
        self.args.joint_feedback_port=29709
        with patch.dict(launch.os.environ,{"RMW_IMPLEMENTATION":"rmw_fastrtps_cpp"},clear=True):
            with patch.object(launch.subprocess,"Popen") as popen:
                self.supervisor.start("arms",["fake-arm"])
                self.supervisor.start("cameras",["fake-camera"])
            self.assertEqual(popen.call_args_list[0].kwargs["env"]["AIDE_JOINT_FEEDBACK_PORT"],"29709")
            self.assertNotIn("AIDE_JOINT_FEEDBACK_PORT",popen.call_args_list[1].kwargs["env"])
            self.assertNotIn("AIDE_JOINT_FEEDBACK_PORT",launch.os.environ)

    def test_reuse_does_not_spawn_or_stop_external_processes(self):
        self.monitor.arm_present.return_value = True
        self.monitor.camera_counts.return_value = dict.fromkeys(launch.recorder.CAMERAS, 1)
        with patch.object(self.supervisor, "start") as start, patch.object(launch.os, "killpg") as kill:
            self.supervisor.prepare(self.monitor)
            self.supervisor.close()
        start.assert_not_called()
        kill.assert_not_called()

    def test_partial_cameras_do_not_open_any_more_devices(self):
        self.monitor.arm_present.return_value = True
        self.monitor.camera_counts.return_value = dict(head=1, left_wrist=0, right_wrist=0)
        with patch.object(self.supervisor, "start") as start:
            with self.assertRaisesRegex(RuntimeError, "相机已被"):
                self.supervisor.prepare(self.monitor)
        start.assert_not_called()

    def test_wait_for_missing_services_is_bounded_and_describes_failure(self):
        self.supervisor.startup_timeout = .6
        with self.assertRaisesRegex(TimeoutError, "双臂示教服务"):
            self.supervisor.wait_for(self.monitor, lambda: False, "双臂示教服务")

    def test_startup_child_failure_prevents_recorder_start(self):
        child = Mock()
        child.poll.return_value = 1
        self.supervisor.children = [("arms", child, self.root / "arms.log")]
        with patch.object(launch.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "arms进程退出"):
                self.supervisor.record([])
        popen.assert_not_called()

    def test_legacy_camera_lock_is_reported_before_starting_arms(self):
        self.camera_lock.return_value = True
        self.camera_processes.return_value = [dict(pid=1234, config="/old/head_camera.yaml", started="10")]
        with patch.object(self.supervisor, "start") as start, patch.object(launch.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "PID=1234.*--restart-cameras"):
                self.supervisor.prepare(self.monitor)
        start.assert_not_called()
        kill.assert_not_called()

    def test_explicit_restart_signals_only_known_parent_and_waits_for_release(self):
        self.supervisor.restart_cameras = True
        old = dict(pid=1234, config="/old/head_camera.yaml", started="10")
        self.camera_processes.side_effect = [[old], [old], []]
        self.camera_lock.side_effect = [True, False]
        with patch.object(self.supervisor, "start") as start, patch.object(launch.os, "kill") as kill:
            self.supervisor.prepare(self.monitor)
        kill.assert_called_once_with(1234, signal.SIGINT)
        self.assertEqual([call.args[0] for call in start.call_args_list], ["arms", "cameras"])
        self.assertFalse(self.monitor.shapes)
        self.assertTrue(all(not frames for frames in self.monitor.frames.values()))

    def test_restart_does_not_signal_unknown_lock_owner(self):
        self.supervisor.restart_cameras = True
        self.camera_lock.return_value = True
        with patch.object(launch.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "未找到本仓库"):
                self.supervisor.prepare_cameras(self.monitor)
        kill.assert_not_called()

    def test_reused_pid_is_not_signalled_and_old_camera_timeout_opens_no_devices(self):
        self.supervisor.restart_cameras = True
        self.supervisor.startup_timeout = .6
        old = dict(pid=1234, config="/old/head_camera.yaml", started="10")
        self.camera_processes.side_effect = [[old], []]
        self.camera_lock.return_value = True
        with patch.object(self.supervisor, "start") as start, patch.object(launch.os, "kill") as kill:
            with self.assertRaisesRegex(TimeoutError, "旧相机组尚未完全退出"):
                self.supervisor.prepare(self.monitor)
        start.assert_not_called()
        kill.assert_not_called()

    def test_child_exit_shows_original_error_without_opening_log_separately(self):
        path = self.root / "cameras.log"
        path.write_text("相机已由单头部或三相机入口占用，请先退出原采集程序。\n")
        child = Mock()
        child.poll.return_value = 1
        self.supervisor.children = [("cameras", child, path)]
        with self.assertRaisesRegex(RuntimeError, "单头部或三相机入口占用"):
            self.supervisor.check()

    def test_shutdown_finishes_recorder_before_owned_hardware(self):
        order = []
        rec, arms, cameras = Mock(), Mock(pid=1234), Mock(pid=5678)
        for child in (rec, arms, cameras):
            child.poll.return_value = None
        rec.terminate.side_effect = lambda: order.append("recorder-stop")
        rec.wait.side_effect = lambda **kwargs: order.append("recorder-exited")
        self.supervisor.recording = rec
        self.supervisor.children = [("arms", arms, "arms.log"), ("cameras", cameras, "cameras.log")]
        with patch.object(launch.os, "killpg", side_effect=lambda pid, sig: order.append((pid, sig))):
            self.supervisor.close()
        self.assertEqual(order, ["recorder-stop", "recorder-exited",
                                 (1234, signal.SIGINT), (5678, signal.SIGINT)])

    def test_terminated_child_does_not_hide_an_owned_running_sibling(self):
        arms, cameras = Mock(pid=1234), Mock(pid=5678)
        arms.poll.return_value = 1
        cameras.poll.return_value = None
        self.supervisor.children = [("arms", arms, "arms.log"), ("cameras", cameras, "cameras.log")]
        with patch.object(launch.os, "killpg") as kill:
            self.supervisor.close()
        kill.assert_called_once_with(5678, signal.SIGINT)

    def test_interruption_before_start_does_not_launch(self):
        self.supervisor.interrupt(signal.SIGINT, None)
        with patch.object(self.supervisor, "start") as start:
            with self.assertRaises(InterruptedError):
                self.supervisor.prepare(self.monitor)
        start.assert_not_called()

    def test_offline_modes_never_construct_readiness_or_start_hardware(self):
        for argv in (["--self-test"], ["--validate-only"], ["--export-to", "/tmp/unused-export"]):
            with self.subTest(argv=argv), patch.object(launch.recorder, "main", return_value=0) as main:
                with patch.object(launch, "Readiness") as ready, patch.object(launch.subprocess, "Popen") as popen:
                    self.assertEqual(launch.main(argv), 0)
                ready.assert_not_called()
                popen.assert_not_called()
                main.assert_called_once_with(argv)

    def test_image_readiness_rejects_wrong_size_duplicates_and_truncated_payload(self):
        self.ticks = count(step=.01)
        ready = launch.Readiness.__new__(launch.Readiness)
        ready.args, ready.node = self.args, Mock()
        ready.node.count_publishers.return_value = 1
        ready.frames = {key: deque(maxlen=100) for key in launch.recorder.CAMERAS}
        ready.shapes = {}
        ready.camera_duplicate_since = ready.camera_ready_since = None
        msg = SimpleNamespace(width=4, height=3, step=12, encoding="rgb8", data=bytes(36))
        for key in launch.recorder.CAMERAS:
            ready.image(key, msg)
            ready.image(key, msg)
        self.assertFalse(ready.cameras_ready())
        self.ticks = count(start=1.5, step=.01)
        self.assertTrue(ready.cameras_ready())
        ready.shapes["head"] = (5, 3, "rgb8")
        self.assertFalse(ready.cameras_ready())
        ready.node.count_publishers.return_value = 2
        ready.node.get_publishers_info_by_topic.return_value = [
            SimpleNamespace(node_namespace="/cameras", node_name="one"),
            SimpleNamespace(node_namespace="/cameras", node_name="two")]
        self.assertFalse(ready.cameras_ready())
        self.ticks = count(start=5., step=.01)
        with self.assertRaisesRegex(RuntimeError, "重复发布者"):
            ready.cameras_ready()
        ready.frames["head"].clear()
        msg.data = bytes(5)
        ready.image("head", msg)
        self.assertEqual(len(ready.frames["head"]), 0)

    def test_startup_overlap_clears_without_bypassing_stable_frames(self):
        ready = launch.Readiness.__new__(launch.Readiness)
        ready.args, ready.node = self.args, Mock()
        ready.frames = {key: deque([10., 10.01]) for key in launch.recorder.CAMERAS}
        ready.shapes = dict.fromkeys(launch.recorder.CAMERAS, (4, 3, "rgb8"))
        ready.camera_duplicate_since = ready.camera_ready_since = None
        now = [10.02]
        ready.node.count_publishers.side_effect = [0, 2, 1]
        with patch.object(launch.time, "monotonic", side_effect=lambda: now[0]):
            # Same graph as the reported failure: head=0, left=2, right=1.
            self.assertFalse(ready.cameras_ready())
            now[0] = 10.5
            ready.node.count_publishers.side_effect = None
            ready.node.count_publishers.return_value = 1
            self.assertFalse(ready.cameras_ready())
            self.assertIsNone(ready.camera_duplicate_since)
            now[0] = 11.6
            self.assertTrue(ready.cameras_ready())
            # Losing a camera resets the stable window, even after readiness.
            ready.node.count_publishers.side_effect = [0, 1, 1]
            self.assertFalse(ready.cameras_ready())
            self.assertIsNone(ready.camera_ready_since)

    def test_duplicate_graph_still_allows_waiting_for_camera_shutdown(self):
        ready = launch.Readiness.__new__(launch.Readiness)
        ready.node = Mock()
        ready.node.count_publishers.return_value = 2
        self.assertEqual(ready.camera_counts(), dict.fromkeys(launch.recorder.CAMERAS, 2))

    def test_service_readiness_requires_both_start_stop_pairs_and_clean_status(self):
        ready = launch.Readiness.__new__(launch.Readiness)
        ready.node = Mock()
        ready.node.get_node_names_and_namespaces.return_value = [("driver", launch.recorder.PREFIX)]
        ready.clients = [Mock() for _ in range(4)]
        for client in ready.clients:
            client.service_is_ready.return_value = True
        ready.status = dict(left="idle", right="stopped")
        self.assertTrue(ready.arms_ready())
        ready.clients[-1].service_is_ready.return_value = False
        self.assertFalse(ready.arms_ready())
        ready.status["right"] = "error:cleanup failed"
        with self.assertRaisesRegex(RuntimeError, "清理未确认"):
            ready.arms_ready()


class CameraDiscoveryTest(unittest.TestCase):
    def test_camera_identity_matches_only_this_workspaces_camera_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"
            proc.mkdir()
            config = root / "hardware/robot_vision_ws/src/robot_vision_bringup/config/head_camera.yaml"
            commands = {
                123: ["python3", "/opt/ros/humble/bin/ros2", "launch", "robot_vision_bringup",
                      "multi_realsense.launch.py", f"camera_registry_file:={config}"],
                234: ["ros2", "launch", "rokae_bringup", "dual_arm.launch.py"],
                345: ["ros2", "launch", "robot_vision_bringup", "multi_realsense.launch.py",
                      "camera_registry_file:=/other/workspace/config/cameras.yaml"],
                456: ["python3", "/other/script.py", "launch", "robot_vision_bringup",
                      "multi_realsense.launch.py", f"camera_registry_file:={config}"],
            }
            for pid, argv in commands.items():
                entry = proc / str(pid)
                entry.mkdir()
                (entry / "cmdline").write_bytes(b"\0".join(os.fsencode(arg) for arg in argv) + b"\0")
                (entry / "stat").write_text(f"{pid} (ros2) " + " ".join(["S"] + ["0"] * 18 + ["123456"]))
            self.assertEqual(launch.camera_launches(root, proc),
                             [dict(pid=123, config=str(config), started="123456")])

    def test_existing_empty_lock_file_is_not_a_busy_camera(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, XDG_RUNTIME_DIR=directory):
            path = Path(directory) / f"aide-head-camera-{os.getuid()}.lock"
            path.touch()
            self.assertFalse(launch.camera_lock_busy())
            with path.open("a") as lock:
                launch.fcntl.flock(lock, launch.fcntl.LOCK_EX | launch.fcntl.LOCK_NB)
                self.assertTrue(launch.camera_lock_busy())
            self.assertFalse(launch.camera_lock_busy())
            self.assertTrue(path.exists())


@unittest.skipUnless(os.environ.get("ROS_DOMAIN_ID") == "231" and
                     os.environ.get("ROS_LOCALHOST_ONLY") == "1",
                     "requires isolated ROS domain 231 and localhost-only")
class RosReadinessTest(unittest.TestCase):
    def test_explicit_context_spins_and_reopens_without_global_ros_init(self):
        args = SimpleNamespace(width=4, height=3)
        for _ in range(2):
            ready = launch.Readiness(args)
            try:
                ready.spin(.01)
                self.assertFalse(ready.arms_ready())
                self.assertFalse(ready.cameras_ready())
            finally:
                ready.close()


if __name__ == "__main__":
    unittest.main()
