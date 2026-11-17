"""Best-effort JPEG preview of already received RGB; no camera/robot access.

Wire format matches robot_vision_tools/udp_video_sender.py: one complete JPEG
per UDP datagram, without an application header or custom fragmentation.
"""
import ipaddress
import socket
import threading
import time

import cv2

DEFAULT_IP = '172.16.13.184'
PORTS = dict(head=6000, left_wrist=6001, right_wrist=6002)
MAX_UDP_PAYLOAD = 65_507


def encode_jpeg(rgb, quality=50):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for level in [*range(quality, 5, -5), 5]:
        ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, level])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        if encoded.nbytes <= MAX_UDP_PAYLOAD:
            return encoded.tobytes()
    raise RuntimeError('JPEG exceeds the single-datagram UDP limit even at minimum quality')


class UdpPreview:
    def __init__(self, target_ip=DEFAULT_IP, quality=50, on_error=print, ports=None):
        self.target_ip = str(ipaddress.IPv4Address(target_ip))
        if not 5 <= quality <= 100:
            raise ValueError('UDP JPEG quality must be in [5, 100]')
        self.quality, self.on_error = quality, on_error
        self.ports = dict(PORTS if ports is None else ports)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.condition = threading.Condition()
        self.pending = {}
        self.closed = False
        self.last_error_at = -float('inf')
        self.worker = threading.Thread(target=self._run, name='teach-udp-preview', daemon=True)
        self.worker.start()

    def offer(self, camera, rgb):
        # The receive callback owns this array; encoding never mutates it.
        # Keep one latest pending frame per camera rather than delaying capture.
        with self.condition:
            if not self.closed:
                self.pending[camera] = rgb
                self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending)
                if self.closed:
                    return
                frames, self.pending = self.pending, {}
            for camera, rgb in frames.items():
                if self.closed:
                    return
                try:
                    payload = encode_jpeg(rgb, self.quality)
                    self.socket.sendto(payload, (self.target_ip, self.ports[camera]))
                except Exception as exc:
                    now = time.monotonic()
                    if not self.closed and now - self.last_error_at >= 2.:
                        self.last_error_at = now
                        self.on_error(f'UDP preview {camera}: {exc}; recording remains independent')

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
            self.pending.clear()
            self.condition.notify()
        self.worker.join(timeout=2.)
        self.socket.close()
