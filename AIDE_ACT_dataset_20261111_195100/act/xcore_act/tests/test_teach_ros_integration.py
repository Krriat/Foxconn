"""Opt-in ROS/CLI smoke test in isolated domain 231; never use a robot domain."""
import json
from array import array
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from xcore_teach_recording.core import CAMERAS
from xcore_teach_recording.dataset import validate_episode


@unittest.skipUnless(os.environ.get('ROS_DOMAIN_ID') == '231' and os.environ.get('ROS_LOCALHOST_ONLY') == '1',
                     'requires explicitly isolated ROS domain 231 and localhost-only')
class RosCaptureIntegration(unittest.TestCase):
    def test_mock_teaching_and_three_cameras_through_real_recorder_cli(self):
        self.capture(fps=5, width=4, height=3)

    def test_30hz_full_resolution_cameras_through_real_recorder_cli(self):
        self.capture(fps=30, width=640, height=480)

    def test_B_saves_first_episode_and_double_B_discards_second_without_terminal_confirmation(self):
        self.capture(fps=5, width=4, height=3, discard_second=True)

    def capture(self, fps, width, height, discard_second=False):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import JointState, Image
        from std_msgs.msg import String, UInt8, Float64
        rclpy.init()
        node = rclpy.create_node('act_teach_isolated_test_sensors')
        prefix = '/robot_02/upperlimb'
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        joints = {side: node.create_publisher(JointState, f'{prefix}/cartesian_drag_safe/joint_states/{side}_arm', sensor)
                  for side in ('left', 'right')}
        statuses = [node.create_publisher(String, f'{prefix}/cartesian_drag_safe/status/{side}_arm', latched)
                    for side in ('left', 'right')]
        head = node.create_publisher(JointState, '/headmotor/csp/joint_states', sensor)
        waist = node.create_publisher(Float64, '/robot/waist_pitch/csp/position_deg', 10)
        buttons = node.create_publisher(UInt8, f'{prefix}/pico/buttons/right_arm', 10)
        cameras = {name: node.create_publisher(Image, f'/robot/sensors/cameras/{name}/color/image_raw', 5)
                   for name in CAMERAS}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'recording'
            script = Path(__file__).resolve().parents[1] / 'scripts/recording/record_xcore_teach.py'
            process = subprocess.Popen([sys.executable, str(script), '--root', str(root), '--task', 'isolated smoke test',
                                        '--fps', str(fps), '--width', str(width), '--height', str(height),
                                        '--udp-video-ip', '127.0.0.1'], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            started = time.monotonic()
            last_image = -1
            try:
                # All published messages are synthetic; no SDK/CAN imports or motion topics.
                while time.monotonic() - started < (7.5 if discard_second else 6.5):
                    elapsed = time.monotonic() - started
                    stamp = node.get_clock().now().to_msg()
                    for pub in statuses:
                        pub.publish(String(data='teaching'))
                    for side, pub in joints.items():
                        message = JointState()
                        message.header.stamp = stamp
                        message.name = [f'{side}_joint_{i}' for i in range(1, 8)]
                        message.position = [.1] * 7
                        pub.publish(message)
                    message = JointState()
                    message.header.stamp = stamp
                    message.name, message.position = ['head_pha_joint'], [.2]
                    head.publish(message)
                    waist.publish(Float64(data=10.))
                    presses = [(2., 2.1), (4., 4.1)]
                    if discard_second:
                        presses += [(5., 5.1), (6., 6.06), (6.18, 6.24)]
                    buttons.publish(UInt8(data=8 if any(start < elapsed < end for start, end in presses) else 0))
                    image_index = int(elapsed * fps)
                    if image_index != last_image:
                        for pub in cameras.values():
                            image = Image()
                            image.header.stamp = stamp
                            image.width, image.height, image.step = width, height, width * 3
                            image.encoding = 'rgb8'
                            # The generated ROS setter iterates over every
                            # byte for plain bytes. Native array('B') avoids
                            # turning the synthetic publisher into a 2 Hz source.
                            image.data = array('B', bytes([10, 20, 30]) * (width * height))
                            pub.publish(image)
                        last_image = image_index
                    if process.poll() is not None:
                        break
                    rclpy.spin_once(node, timeout_sec=.001)
                    time.sleep(.004)
                process.stdin.write('q\n')
                process.stdin.flush()
                output, _ = process.communicate(timeout=8.)
                self.assertEqual(process.returncode, 0, output)
                episodes = sorted(root.glob('episode_*'))
                self.assertEqual(len(episodes), 1, output)
                self.assertIn('Saved:', output)
                self.assertFalse(list(root.glob('.episode_*.partial')), output)
                if discard_second:
                    self.assertIn('Pending episode discarded', output)
                meta, arrays = validate_episode(episodes[0])
                self.assertGreaterEqual(meta['frames'], 3, output)
                if fps == 30:
                    self.assertGreaterEqual(meta['frames'], 54, output)
                    self.assertLessEqual(meta['frames'], 62, output)
                    self.assertEqual(len(meta['segments']), 1, output)
                    gaps = arrays['source_timestamp'][1:] - arrays['source_timestamp'][:-1]
                    self.assertTrue((gaps > 0).all(), output)
                    self.assertLess(float(gaps[:, 4:].max()), 1.6 / fps, output)
                self.assertEqual(arrays['state'].shape[1], 16)
                self.assertAlmostEqual(float(arrays['state'][0, 14]), .2, places=5)
                self.assertAlmostEqual(float(arrays['state'][0, 15]), .174532925, places=5)
                print('Isolated ROS smoke test: saved and validated', meta['frames'], 'frames')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                node.destroy_node()
                rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
