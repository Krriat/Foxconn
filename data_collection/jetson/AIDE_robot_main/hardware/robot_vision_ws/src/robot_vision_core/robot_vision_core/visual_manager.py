"""Optional process controls migrated from the exhibition vision scripts."""

import signal
import subprocess
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VisualManager(Node):
    def __init__(self):
        super().__init__('perception_manager', namespace='/robot/vision')
        self.processes = {}
        self.declare_parameter('command_topic', '/robot/vision/control_cmd')
        self.subscription = self.create_subscription(
            String, self.get_parameter('command_topic').value, self.on_command, 10)

    def on_command(self, message):
        # Preserve the actual old implementation: 0/1 small box, 2/3 ArUco ID.
        commands = {'0': ('small_box_detection', True), '1': ('small_box_detection', False),
                    '2': ('aruco_scanner', True), '3': ('aruco_scanner', False)}
        command = commands.get(message.data.strip())
        if command is None:
            self.get_logger().warning('Expected 0/1 for small box or 2/3 for ArUco ID')
            return
        module, start = command
        process = self.processes.get(module)
        if start:
            if process is None or process.poll() is not None:
                self.processes[module] = subprocess.Popen([
                    sys.executable, '-m', 'robot_vision_core.' + module,
                ])
        else:
            self.stop(module)

    def stop(self, module):
        process = self.processes.pop(module, None)
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def destroy_node(self):
        for module in list(self.processes):
            self.stop(module)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
