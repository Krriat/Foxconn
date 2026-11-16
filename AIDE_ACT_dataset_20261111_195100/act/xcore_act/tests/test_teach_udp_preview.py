"""JPEG compatibility and independent, bounded preview of capture inputs."""
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import socket
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from xcore_teach_recording.udp_preview import DEFAULT_IP, MAX_UDP_PAYLOAD, PORTS, UdpPreview, encode_jpeg
from xcore_teach_recording.ros_node import RecorderNode


class JpegTests(unittest.TestCase):
    def test_legacy_single_jpeg_wire_format_preserves_dimensions_and_rgb_order(self):
        rgb = np.full((48, 64, 3), [240, 30, 10], dtype=np.uint8)
        original = rgb.copy()
        payload = encode_jpeg(rgb)
        self.assertTrue(payload.startswith(b'\xff\xd8'))
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, rgb.shape)
        np.testing.assert_allclose(decoded[24, 32], [10, 30, 240], atol=5)
        np.testing.assert_equal(rgb, original)

    def test_noisy_vga_frame_is_reencoded_below_udp_datagram_limit(self):
        rgb = np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        payload = encode_jpeg(rgb, quality=100)
        self.assertLessEqual(len(payload), MAX_UDP_PAYLOAD)
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, rgb.shape)

    def test_default_corrected_ip_and_port_mapping(self):
        self.assertEqual(DEFAULT_IP, '172.16.13.184')
        self.assertEqual(PORTS, dict(head=6000, left_wrist=6001, right_wrist=6002))


class WorkerTests(unittest.TestCase):
    def test_pending_frames_are_bounded_and_do_not_wait_for_slow_encoder(self):
        entered, release = threading.Event(), threading.Event()
        def slow_encode(*args):
            entered.set()
            release.wait(2.)
            return b'jpeg'
        with patch('xcore_teach_recording.udp_preview.socket.socket'), \
                patch('xcore_teach_recording.udp_preview.encode_jpeg', side_effect=slow_encode):
            preview = UdpPreview()
            try:
                preview.offer('head', np.zeros((1, 1, 3), dtype=np.uint8))
                self.assertTrue(entered.wait(1.))
                latest = object()
                for _ in range(100):
                    for camera in PORTS:
                        preview.offer(camera, latest)
                with preview.condition:
                    self.assertEqual(set(preview.pending), set(PORTS))
                    self.assertTrue(all(frame is latest for frame in preview.pending.values()))
            finally:
                release.set()
                preview.close()
            self.assertFalse(preview.worker.is_alive())

    def test_send_error_is_reported_and_worker_continues(self):
        error, sent = threading.Event(), threading.Event()
        sock = Mock()
        def send(payload, address):
            if address[1] == 6000:
                raise OSError('synthetic network outage')
            sent.set()
        sock.sendto.side_effect = send
        with patch('xcore_teach_recording.udp_preview.socket.socket', return_value=sock):
            preview = UdpPreview(on_error=lambda message: error.set())
            try:
                rgb = np.zeros((3, 4, 3), dtype=np.uint8)
                preview.offer('head', rgb)
                self.assertTrue(error.wait(1.))
                preview.offer('left_wrist', rgb)
                self.assertTrue(sent.wait(1.))
            finally:
                preview.close()
            sock.close.assert_called_once()

    def test_ros_receive_callback_shares_received_array_without_mutating_recording(self):
        receiver = SimpleNamespace(inputs=Mock(), udp_preview=Mock(), warn=Mock())
        message = SimpleNamespace(width=2, height=1, encoding='bgr8', step=6,
                                  data=bytes([30, 20, 10] * 2),
                                  header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0)))
        RecorderNode.on_image(receiver, 'head', message, 2, 1)
        image = receiver.inputs.push.call_args.args[3]
        receiver.udp_preview.offer.assert_called_once_with('head', image)
        np.testing.assert_equal(image[0, 0], [10, 20, 30])
        receiver.warn.assert_not_called()

    def test_cli_defaults_and_disable_flag(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/recording/record_xcore_teach.py'
        spec = importlib.util.spec_from_file_location('teach_cli_udp_test', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with patch('sys.argv', [str(path)]):
            args = cli.parse_args()
        self.assertEqual(args.udp_video_ip, '172.16.13.184')
        self.assertFalse(args.no_udp_video)
        with patch('sys.argv', [str(path), '--no-udp-video']):
            self.assertTrue(cli.parse_args().no_udp_video)


@unittest.skipUnless(os.environ.get('ROS_DOMAIN_ID') == '231'
                     and os.environ.get('ROS_LOCALHOST_ONLY') == '1',
                     'loopback socket test requires isolated local test environment')
class LoopbackTests(unittest.TestCase):
    def test_three_real_datagrams_are_jpegs_on_corresponding_camera_sockets(self):
        with ExitStack() as stack:
            receivers = {}
            for camera in PORTS:
                receiver = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
                receiver.bind(('127.0.0.1', 0))
                receiver.settimeout(2.)
                receivers[camera] = receiver
            ports = {key: receiver.getsockname()[1] for key, receiver in receivers.items()}
            preview = UdpPreview('127.0.0.1', ports=ports)
            stack.callback(preview.close)
            for index, camera in enumerate(PORTS):
                preview.offer(camera, np.full((24, 32, 3), [200 - index * 50, 30, 10], dtype=np.uint8))
            for index, receiver in enumerate(receivers.values()):
                payload, _ = receiver.recvfrom(65535)
                decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, (24, 32, 3))
                np.testing.assert_allclose(decoded[12, 16], [10, 30, 200 - index * 50], atol=5)


if __name__ == '__main__':
    unittest.main()
