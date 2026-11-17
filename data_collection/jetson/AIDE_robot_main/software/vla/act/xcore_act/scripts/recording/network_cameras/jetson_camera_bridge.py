#!/usr/bin/env python3
"""Receive PC wrist images, correct capture UTC to Jetson time, publish ROS RGB."""
from __future__ import annotations

import argparse
from array import array
from collections import deque
import json
import signal
import struct
import threading
import time

import numpy as np

from clock_sync import ClockEstimator, WallClockGuard
from protocol import CAMERAS, DEFAULT_HOST, DEFAULT_PORT, LEFT_SERIAL, RIGHT_SERIAL, connect, receive_packet, send_packet


def decode_image(header, payload, width, height):
    if (header.get('width'), header.get('height'), header.get('encoding')) != (width, height, 'rgb8'):
        raise ValueError('remote image shape/encoding mismatch')
    codec = header.get('codec')
    if codec == 'raw':
        if len(payload) != width * height * 3:
            raise ValueError('invalid raw image length')
        return np.frombuffer(payload, np.uint8).reshape(height, width, 3)
    if codec not in ('png', 'jpeg'):
        raise ValueError('unsupported image codec')
    if codec == 'png':
        if len(payload) < 24 or payload[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('!II', payload[16:24]) != (width, height):
            raise ValueError('invalid PNG dimensions')
    else:
        # Check JPEG SOF dimensions before letting the decoder allocate memory.
        at, shape = 2, None
        if not payload.startswith(b'\xff\xd8'):
            raise ValueError('invalid JPEG')
        while at + 4 <= len(payload):
            if payload[at] != 255:
                break
            marker = payload[at + 1]
            size = int.from_bytes(payload[at+2:at+4], 'big')
            if size < 2 or at + size + 2 > len(payload):
                break
            if marker in (0xc0, 0xc1, 0xc2):
                shape = int.from_bytes(payload[at+7:at+9], 'big'), int.from_bytes(payload[at+5:at+7], 'big')
                break
            at += size + 2
        if shape != (width, height):
            raise ValueError('invalid JPEG dimensions')
    import cv2
    bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (height, width, 3):
        raise ValueError('invalid compressed image')
    return np.ascontiguousarray(bgr[:, :, ::-1])


class RemoteCameras:
    def __init__(self, args, publish):
        self.args, self.publish = args, publish
        self.clock = ClockEstimator(max_error_ms=args.max_clock_error_ms)
        self.wall_guard = WallClockGuard()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.stats = {key: dict(received=0, published=0, rejected=0, skipped_sequences=0,
            last_error='connecting') for key in CAMERAS}
        self.recent = {key: deque(maxlen=600) for key in CAMERAS}
        self.threads = []

    def error(self, camera, exc):
        with self.lock:
            self.stats[camera]['last_error'] = str(exc)
            self.stats[camera]['rejected'] += 1

    def clock_loop(self):
        while not self.stopping.is_set():
            try:
                with connect(self.args.host, self.args.port) as connection:
                    while not self.stopping.is_set():
                        m1, t1 = time.monotonic_ns(), time.time_ns()
                        if self.wall_guard.update(t1, m1):
                            self.clock.reset()
                        send_packet(connection, dict(type='clock', t1=t1))
                        response, payload = receive_packet(connection)
                        t4, m4 = time.time_ns(), time.monotonic_ns()
                        if self.wall_guard.update(t4, m4):
                            self.clock.reset()
                            continue
                        if response.get('type') != 'clock' or response.get('t1') != t1 or payload:
                            raise ValueError('invalid clock response')
                        self.clock.add(t1, response['t2'], response['t3'], t4, m4-m1, response['clock_epoch'])
                        self.stopping.wait(.2)
            except (OSError, EOFError, ValueError, KeyError) as exc:
                for camera in CAMERAS:
                    self.error(camera, 'clock: ' + str(exc))
                self.clock.reset()
                self.stopping.wait(.5)

    def camera_loop(self, camera, serial):
        last_stamp, last_sequence, last_epoch = 0, None, None
        while not self.stopping.is_set():
            try:
                with connect(self.args.host, self.args.port) as connection:
                    send_packet(connection, dict(type='subscribe', camera=camera))
                    while not self.stopping.is_set():
                        header, payload = receive_packet(connection)
                        if header.get('type') == 'waiting':
                            self.error(camera, header.get('reason', 'camera waiting'))
                            continue
                        with self.lock:
                            self.stats[camera]['received'] += 1
                        try:
                            if header.get('type') != 'frame' or header.get('camera') != camera or header.get('serial') != serial:
                                raise ValueError('remote camera identity mismatch')
                            if header.get('synthetic') and not self.args.allow_synthetic:
                                raise ValueError('synthetic images prohibited in normal acquisition')
                            if not header.get('synthetic') and header.get('timestamp_source') != 'camera_global_time':
                                raise ValueError('camera exposure UTC required; receipt timestamps are not accepted')
                            capture_ns, sequence = header.get('capture_utc_ns'), header.get('sequence')
                            if type(capture_ns) is not int or capture_ns <= 0 or type(sequence) is not int or sequence < 1:
                                raise ValueError('invalid capture timestamp or sequence')
                            estimate = self.clock.estimate()
                            if header.get('clock_epoch') != estimate.epoch:
                                raise ValueError('PC clock changed; waiting for fresh clock comparison')
                            stamp = capture_ns - estimate.offset_ns
                            age_ms = (time.time_ns() - stamp) / 1e6
                            if not -self.args.max_clock_error_ms <= age_ms <= self.args.max_frame_age_ms:
                                raise ValueError(f'frame capture age {age_ms:.1f}ms exceeds bounds')
                            if stamp <= last_stamp:
                                raise ValueError('capture timestamp moved backwards; not published')
                            if last_epoch == estimate.epoch and last_sequence is not None and sequence <= last_sequence:
                                raise ValueError('duplicate or reordered frame sequence')
                            rgb = decode_image(header, payload, self.args.width, self.args.height)
                            # Time spent receiving/decompressing must count toward freshness.
                            age_ms = (time.time_ns() - stamp) / 1e6
                            if age_ms > self.args.max_frame_age_ms:
                                raise ValueError('image became stale during decompression')
                            metadata = dict(camera=camera, sequence=sequence, capture_pc_utc_ns=capture_ns,
                                clock_epoch=estimate.epoch,
                                capture_jetson_utc_ns=stamp, pc_minus_jetson_ms=estimate.offset_ns/1e6,
                                clock_error_bound_ms=estimate.uncertainty_ns/1e6, clock_rtt_ms=estimate.rtt_ns/1e6,
                                capture_age_ms=age_ms, codec=header['codec'])
                            self.publish(camera, stamp, rgb, metadata)
                            with self.lock:
                                info = self.stats[camera]
                                info['published'] += 1
                                if last_epoch == estimate.epoch and last_sequence is not None:
                                    info['skipped_sequences'] += max(0, sequence-last_sequence-1)
                                info.update(metadata, last_error='')
                                self.recent[camera].append((time.monotonic(), age_ms))
                            last_stamp, last_sequence, last_epoch = stamp, sequence, estimate.epoch
                        except (ValueError, KeyError, TypeError) as exc:
                            self.error(camera, exc)
            except (OSError, EOFError, ValueError, KeyError) as exc:
                self.error(camera, exc)
                self.stopping.wait(.5)

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            result = {key: dict(info) for key, info in self.stats.items()}
            for key, history in self.recent.items():
                recent = [v for v in history if now-v[0] <= 5.]
                result[key]['fresh'] = bool(recent and now-recent[-1][0] < 1.)
                result[key]['receive_hz'] = ((len(recent)-1)/(recent[-1][0]-recent[0][0])
                    if len(recent)>1 and recent[-1][0]>recent[0][0] else 0.)
                result[key]['capture_age_p95_ms'] = float(np.percentile([v[1] for v in recent],95)) if recent else None
            return result

    def start(self):
        for function, params in [(self.clock_loop, ())] + [
                (self.camera_loop, pair) for pair in zip(CAMERAS, (self.args.left_serial,self.args.right_serial))]:
            thread = threading.Thread(target=function, args=params, daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.stopping.set()
        for thread in self.threads:
            thread.join(4.)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default=DEFAULT_HOST)
    p.add_argument('--port', type=int, default=DEFAULT_PORT)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--left-serial', default=LEFT_SERIAL)
    p.add_argument('--right-serial', default=RIGHT_SERIAL)
    p.add_argument('--max-clock-error-ms', type=float, default=10.)
    p.add_argument('--max-frame-age-ms', type=float, default=350.)
    p.add_argument('--metrics-file')
    p.add_argument('--check-seconds', type=float, default=0., help='no ROS, arms or IO; receive and measure only')
    p.add_argument('--min-check-hz', type=float, default=20., help='minimum received rate for --check-seconds to pass')
    p.add_argument('--allow-synthetic', action='store_true', help='test-only; normal launcher never sets this')
    args = p.parse_args(argv)
    values = (args.max_clock_error_ms, args.max_frame_age_ms, args.check_seconds, args.min_check_hz)
    import math
    if (not 0<args.port<65536 or min(args.width,args.height)<1 or
            any(not math.isfinite(v) for v in values) or not 0<args.max_clock_error_ms<=25 or
            not 0<args.max_frame_age_ms<=500 or args.check_seconds<0 or not 0<=args.min_check_hz<=60):
        p.error('invalid port, dimensions, timing bounds or check duration')
    return args


def main(argv=None):
    args = parse_args(argv)
    node = None
    if not args.check_seconds:
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node('network_wrist_cameras')
        publishers = {key:node.create_publisher(Image, f'/robot/sensors/cameras/{key}/color/image_raw',
            QoSProfile(depth=20,reliability=ReliabilityPolicy.RELIABLE)) for key in CAMERAS}
        diagnostics = node.create_publisher(DiagnosticArray, '/robot/sensors/cameras/diagnostics', 10)
        def publish(camera, stamp, rgb, metadata):
            msg=Image()
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp,1_000_000_000)
            msg.header.frame_id = camera+'_color_optical_frame'
            msg.width=args.width
            msg.height=args.height
            msg.encoding='rgb8';msg.step=args.width*3
            # ROS Humble checks every element when given bytes. Passing a typed
            # byte array avoids ~1.8 million Python checks per VGA image.
            data=array('B')
            data.frombytes(rgb.tobytes())
            msg.data=data
            publishers[camera].publish(msg)
    else:
        publish = lambda *_: None
    client=RemoteCameras(args,publish)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: client.stopping.set())
    metrics=open(args.metrics_file,'a',buffering=1) if args.metrics_file else None
    start=time.monotonic()
    try:
        client.start()
        while not client.stopping.wait(1.):
            state=client.snapshot()
            line=json.dumps(dict(utc_ns=time.time_ns(),cameras=state),ensure_ascii=False)
            print(line,flush=True)
            if metrics:
                metrics.write(line+'\n')
            if node is not None:
                message=DiagnosticArray();message.header.stamp=node.get_clock().now().to_msg()
                for key,info in state.items():
                    status=DiagnosticStatus(name=key+': color',hardware_id='network:'+args.host,
                        level=bytes([0 if info['fresh'] else 1]),message='capture UTC aligned' if info['fresh'] else info['last_error'])
                    status.values=[KeyValue(key='Actual frequency (Hz)',value=str(info['receive_hz']))]
                    status.values += [KeyValue(key=k,value=str(v)) for k,v in info.items()]
                    message.status.append(status)
                diagnostics.publish(message)
                rclpy.spin_once(node,timeout_sec=0.)
            if args.check_seconds and time.monotonic()-start>=args.check_seconds:
                return 0 if all(v['fresh'] and v['published']>0 and
                    v['receive_hz']>=args.min_check_hz for v in state.values()) else 1
    finally:
        client.close()
        if metrics:
            metrics.close()
        if node is not None:
            node.destroy_node();rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
