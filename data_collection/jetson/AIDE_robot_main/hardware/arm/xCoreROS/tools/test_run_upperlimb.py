"""Lifecycle regression tests using temporary child processes, never SDK hardware."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_upperlimb as launcher


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        scripts = self.root / 'scripts'
        scripts.mkdir()
        (self.root / 'install').mkdir()
        (self.root / 'install/local_setup.bash').touch()
        self.camera = scripts / 'start_head_camera.sh'
        self.camera.write_text('exec sleep 30\n')
        self.commands = []
        self.children = []
        original = subprocess.Popen
        def spawn(command, **kwargs):
            self.commands.append(command)
            process = original(command, **kwargs)
            self.children.append(process)
            return process
        self.spawn = patch.object(launcher.subprocess, 'Popen', side_effect=spawn)
        self.spawn.start()

    def tearDown(self):
        self.spawn.stop()
        for child in self.children:
            launcher.stop_process_group(child)
        self.tmp.cleanup()

    def arm(self, code=0):
        return [sys.executable, '-c', f'raise SystemExit({code})']

    def test_cold_start_camera_first_and_arm_exit_cleans_owned_camera(self):
        with patch.object(launcher, 'head_camera_processes', return_value=[]), \
                patch.object(launcher, 'wait_for_camera') as ready:
            self.assertEqual(launcher.run(self.arm(), self.camera), 0)
        self.assertEqual(self.commands[0], ['bash', str(self.camera)])
        self.assertEqual(self.commands[1], self.arm())
        self.assertEqual(ready.call_args.args[1], self.children[0])
        self.assertIsNotNone(self.children[0].poll())

    def test_reuse_camera_does_not_spawn_or_stop_it(self):
        with patch.object(launcher, 'head_camera_processes', return_value=[123]), \
                patch.object(launcher, 'wait_for_camera') as ready:
            self.assertEqual(launcher.run(self.arm(7), self.camera), 7)
        self.assertEqual(self.commands, [self.arm(7)])
        self.assertEqual(len(ready.call_args.args), 1)

    def test_camera_readiness_failure_prevents_arm_start_and_cleans_camera(self):
        with patch.object(launcher, 'head_camera_processes', return_value=[]), \
                patch.object(launcher, 'wait_for_camera', side_effect=TimeoutError('missing frames')):
            self.assertEqual(launcher.run(self.arm(), self.camera), 1)
        self.assertEqual(len(self.commands), 1)
        self.assertIsNotNone(self.children[0].poll())

    def test_duplicate_cameras_are_rejected_before_starting_arm(self):
        with patch.object(launcher, 'head_camera_processes', return_value=[123, 456]):
            self.assertEqual(launcher.run(self.arm(), self.camera), 1)
        self.assertEqual(self.commands, [])

    def test_signal_cancellation_cleans_owned_children(self):
        with patch.object(launcher, 'head_camera_processes', return_value=[]), \
                patch.object(launcher, 'wait_for_camera', side_effect=InterruptedError('cancel')):
            self.assertEqual(launcher.run(self.arm(), self.camera), 130)
        self.assertIsNotNone(self.children[0].poll())
        self.assertEqual(len(self.commands), 1)

    def test_disabling_camera_preserves_arm_only_start(self):
        with patch.object(launcher, 'wait_for_camera', side_effect=AssertionError('camera disabled')):
            self.assertEqual(launcher.run(self.arm()), 0)
        self.assertEqual(self.commands, [self.arm()])

    def test_owned_camera_exit_stops_arm_launch(self):
        self.camera.write_text('exit 0\n')
        with patch.object(launcher, 'head_camera_processes', return_value=[]), \
                patch.object(launcher, 'wait_for_camera'):
            self.assertEqual(launcher.run(['sleep', '30'], self.camera), 1)
        self.assertEqual(len(self.children), 2)
        self.assertTrue(all(child.poll() is not None for child in self.children))

    def test_shell_alias_routes_camera_switch_and_keeps_arm_launch_arguments(self):
        source = Path(__file__).resolve().parent
        workspace = self.root / 'hardware/arm/xCoreROS'
        tools = workspace / 'tools'
        tools.mkdir(parents=True)
        for name in ('start_upperlimb.sh', 'start_state_publisher.sh'):
            (tools / name).write_bytes((source / name).read_bytes())
        (workspace / 'install').mkdir()
        (workspace / 'install/local_setup.bash').touch()
        (tools / 'run_upperlimb.py').write_text('import json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        for camera_enabled in (True, False):
            options = [] if camera_enabled else ['start_head_camera:=false']
            result = subprocess.run(['bash', str(tools / 'start_state_publisher.sh'), *options,
                                     'start_hand_service:=false'], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            arguments = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual('--head-camera-script' in arguments, camera_enabled)
            self.assertEqual(arguments[arguments.index('--')+1:arguments.index('--')+5],
                             ['ros2', 'launch', 'rokae_bringup', 'dual_arm.launch.py'])
            self.assertNotIn('start_head_camera:=false', arguments)
            self.assertEqual(arguments[-1], 'start_hand_service:=false')


if __name__ == '__main__':
    unittest.main()
