#!/usr/bin/env python3
"""Expose the unmodified SDK Cartesian drag program through ROS services.

No SDK connection or motion occurs until the start service is called. The child
inherits the terminal so all original TCP/wrench and controller output remains
visible. SIGINT uses the original program's disableDrag cleanup path.
"""

import math
from pathlib import Path
import signal
import subprocess

from ament_index_python.packages import get_package_prefix
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger


class DragProcess:
    """Own one SDK process; never forcibly terminate an active drag session."""

    def __init__(self, executable):
        self.executable = str(executable)
        self.process = None
        self.stopping = False

    def reap(self):
        if self.process is None:
            return None
        code = self.process.poll()
        if code is not None:
            self.process = None
            self.stopping = False
        return code

    def start(self, arm, duration):
        self.reap()
        if self.process is not None:
            raise RuntimeError('a drag process is already running; stop it first')
        if arm not in ('left', 'right'):
            raise ValueError("arm must be 'left' or 'right'")
        if not math.isfinite(duration) or not 1.0 <= duration <= 600.0:
            raise ValueError('duration_seconds must be between 1 and 600')
        # A separate session prevents terminal Ctrl+C from also delivering an
        # uncontrolled second signal to the child. close() forwards one SIGINT.
        self.process = subprocess.Popen(
            [self.executable, arm, str(duration)], start_new_session=True)
        self.stopping = False

    def stop(self):
        self.reap()
        if self.process is None:
            return False
        if not self.stopping:
            try:
                self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                self.reap()
                return False
            self.stopping = True
        return True

    def close(self):
        self.stop()
        if self.process is not None:
            # Let the SDK finish disableDrag, including its network calls.
            # Killing the child on a timeout could leave drag enabled.
            self.process.wait()
            self.process = None
            self.stopping = False


class CartesianDragSafeNode(Node):
    def __init__(self):
        super().__init__(
            'cartesian_drag_safe',
            namespace='/robot_02/upperlimb/cartesian_drag_safe')
        self.declare_parameter('arm', 'right')
        self.declare_parameter('duration_seconds', 60.0)
        executable = (
            Path(get_package_prefix('rokae_driver'))
            / 'lib' / 'rokae_driver' / 'cartesian_drag_safe')
        self.drag = DragProcess(executable)
        self.start_service = self.create_service(Trigger, 'start', self.start_drag)
        self.stop_service = self.create_service(Trigger, 'stop', self.stop_drag)
        self.monitor = self.create_timer(0.1, self.check_process)
        self.get_logger().info(
            'Ready: call start to run the original cartesian_drag_safe.cpp; '
            'call stop to request its normal SIGINT cleanup. '
            'Do not run another arm controller concurrently.')

    def start_drag(self, _request, response):
        self.check_process()
        arm = self.get_parameter('arm').value
        duration = self.get_parameter('duration_seconds').value
        try:
            self.drag.start(arm, duration)
        except (OSError, RuntimeError, ValueError) as error:
            response.success = False
            response.message = str(error)
            return response
        response.success = True
        response.message = (
            f'SDK process started for {arm}, {duration:g} s; '
            'this acknowledges process launch, not successful drag enable. '
            'Check terminal output; the original program writes '
            '0 kg / zero CoG and requires holding the physical drag button.')
        self.get_logger().info(response.message)
        return response

    def stop_drag(self, _request, response):
        self.check_process()
        response.success = self.drag.stop()
        response.message = (
            'SIGINT stop requested; wait for the terminal to confirm drag disabled'
            if response.success else 'no SDK drag process is running')
        return response

    def check_process(self):
        code = self.drag.reap()
        if code is not None:
            message = f'cartesian_drag_safe exited with code {code}; see SDK output'
            if code == 0:
                self.get_logger().info(message)
            else:
                self.get_logger().error(message)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CartesianDragSafeNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.drag.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
