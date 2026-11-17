"""Supervisor lifecycle checks with no robot connections or ROS participants."""

import importlib.util
from pathlib import Path
import signal
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    'cartesian_drag_safe_node',
    Path(__file__).resolve().parents[1] / 'scripts/cartesian_drag_safe_node.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DragProcessTest(unittest.TestCase):
    def setUp(self):
        self.popen_patch = patch.object(module.subprocess, 'Popen')
        self.popen = self.popen_patch.start()
        self.addCleanup(self.popen_patch.stop)
        self.child = Mock()
        self.child.poll.return_value = None
        self.popen.return_value = self.child
        self.drag = module.DragProcess('/installed/cartesian_drag_safe')

    def test_invalid_arguments_never_launch_sdk(self):
        for arm, duration in [('both', 60), ('right', 0), ('left', 601),
                              ('right', float('nan')), ('right', float('inf'))]:
            with self.subTest(arm=arm, duration=duration):
                with self.assertRaises(ValueError):
                    self.drag.start(arm, duration)
        self.popen.assert_not_called()

    def test_launch_is_explicit_and_duplicate_is_rejected(self):
        self.popen.assert_not_called()
        self.drag.start('right', 60.0)
        self.popen.assert_called_once_with(
            ['/installed/cartesian_drag_safe', 'right', '60.0'],
            start_new_session=True)
        with self.assertRaises(RuntimeError):
            self.drag.start('left', 30)

    def test_stop_and_shutdown_use_sigint_once_and_wait_for_cleanup(self):
        self.drag.start('left', 600.0)
        self.assertTrue(self.drag.stop())
        self.assertTrue(self.drag.stop())
        self.drag.close()
        self.child.send_signal.assert_called_once_with(signal.SIGINT)
        self.child.wait.assert_called_once_with()
        self.child.kill.assert_not_called()
        self.child.terminate.assert_not_called()
        self.assertFalse(self.drag.stop())

    def test_natural_exit_is_reported_and_allows_restart(self):
        self.drag.start('right', 1.0)
        self.child.poll.return_value = 1
        self.assertEqual(self.drag.reap(), 1)
        self.assertIsNone(self.drag.reap())
        self.assertFalse(self.drag.stop())
        self.child.send_signal.assert_not_called()
        self.drag.start('left', 30.0)
        self.assertEqual(self.popen.call_count, 2)

    def test_launch_failure_leaves_no_active_process(self):
        self.popen.side_effect = FileNotFoundError('missing SDK executable')
        with self.assertRaises(FileNotFoundError):
            self.drag.start('right', 60)
        self.assertIsNone(self.drag.process)
        self.assertFalse(self.drag.stop())


if __name__ == '__main__':
    unittest.main()
