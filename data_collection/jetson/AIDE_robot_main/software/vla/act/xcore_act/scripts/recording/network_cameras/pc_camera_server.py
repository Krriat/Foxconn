#!/usr/bin/env python3
"""Ubuntu PC: capture two D405s, serve lossless images and UTC clock probes."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import signal
import socket
import socketserver
import threading
import time
import uuid

import numpy as np

from clock_sync import CaptureTimestampGuard, WallClockGuard
from protocol import CAMERAS, DEFAULT_PORT, LEFT_SERIAL, RIGHT_SERIAL, receive_packet, send_packet


class FrameHub:
    def __init__(self):
        self.frames = deque(maxlen=4)
        self.condition = threading.Condition()
        self.sequence = 0
        self.error = 'camera starting / timestamps warming up'

    def clear(self, reason):
        with self.condition:
            self.frames.clear()
            self.error = reason

    def offer(self, header, payload):
        with self.condition:
            self.sequence += 1
            header = dict(header, sequence=self.sequence)
            self.frames.append((header, payload))
            self.error = ''
            self.condition.notify_all()
            return self.sequence

    def after(self, sequence, stopping):
        with self.condition:
            self.condition.wait_for(lambda: stopping.is_set() or
                (self.frames and self.frames[-1][0]['sequence'] > sequence), timeout=.5)
            return next((v for v in self.frames if v[0]['sequence'] > sequence), None)


def encode_rgb(rgb, codec, quality):
    if codec == 'raw':
        return np.ascontiguousarray(rgb).tobytes()
    import cv2
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    extension, options = ('.png', [cv2.IMWRITE_PNG_COMPRESSION, 1]) if codec == 'png' else (
        '.jpg', [cv2.IMWRITE_JPEG_QUALITY, quality])
    ok, encoded = cv2.imencode(extension, bgr, options)
    if not ok:
        raise ValueError('image compression failed')
    return encoded.tobytes()


class CameraServer:
    def __init__(self, args):
        self.args = args
        self.hubs = {key: FrameHub() for key in CAMERAS}
        self.stopping = threading.Event()
        self.epoch_lock = threading.Lock()
        self.epoch = uuid.uuid4().hex
        self.wall_guard = WallClockGuard()
        self.threads = []
        self.trace = None
        if args.diagnostics_dir is not None:
            from capture_trace import CaptureTrace
            self.trace = CaptureTrace(args.diagnostics_dir)

    def wall(self):
        wall, mono = time.time_ns() + int(self.args.test_clock_offset_ms * 1e6), time.monotonic_ns()
        if self.wall_guard.update(wall, mono):
            with self.epoch_lock:
                self.epoch = uuid.uuid4().hex
                for hub in self.hubs.values():
                    hub.clear('PC realtime stepped; waiting for camera timestamps to settle')
            print('PC clock stepped; old images cleared', flush=True)
            if self.trace:
                self.trace.event('pc_clock_step', clock_epoch=self.epoch)
        with self.epoch_lock:
            return wall, self.epoch

    def capture(self, camera, serial):
        args, hub = self.args, self.hubs[camera]
        while not self.stopping.is_set():
            pipeline = None
            session = uuid.uuid4().hex
            previous_number = previous_return = None
            try:
                guard = CaptureTimestampGuard(args.warm_seconds)
                if not args.synthetic:
                    import pyrealsense2 as rs
                    context = rs.context()
                    device = next((d for d in context.query_devices()
                        if d.get_info(rs.camera_info.serial_number) == serial), None)
                    if device is None:
                        raise RuntimeError(f'{camera} serial {serial} not found; use --list')
                    usb = device.get_info(rs.camera_info.usb_type_descriptor) if device.supports(rs.camera_info.usb_type_descriptor) else 'unknown'
                    print(f'[{camera}] serial={serial} USB={usb}', flush=True)
                    for sensor in device.query_sensors():
                        if sensor.supports(rs.option.global_time_enabled):
                            sensor.set_option(rs.option.global_time_enabled, 1.)
                    pipeline = rs.pipeline(context)
                    config = rs.config()
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
                    # D405 color shares the stereo module; retain depth capture,
                    # but no depth alignment, point clouds or ROS on the PC.
                    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
                    pipeline.start(config)
                number, next_at = 0, time.monotonic()
                while not self.stopping.is_set():
                    wait_begin = time.monotonic_ns()
                    if args.synthetic:
                        if self.stopping.wait(max(0., next_at - time.monotonic())):
                            break
                        next_at += 1 / args.fps
                        wall, epoch = self.wall()
                        stamp = wall
                        rgb = np.full((args.height, args.width, 3), number % 256, dtype=np.uint8)
                        number += 1
                    else:
                        frames = pipeline.wait_for_frames(2000)
                        frame = frames.get_color_frame()
                        if not frame:
                            if self.trace:
                                self.trace.event('no_color_frame', camera=camera, session=session)
                            continue
                        wall, epoch = self.wall()
                        number = frame.get_frame_number()
                        if frame.get_frame_timestamp_domain() != rs.timestamp_domain.global_time:
                            if self.trace:
                                self.trace.event('timestamp_domain_rejected', camera=camera,
                                    session=session, camera_frame_number=number,
                                    timestamp_domain=str(frame.get_frame_timestamp_domain()))
                            hub.clear('camera global_time not ready; no arrival-time fallback')
                            continue
                        stamp = round(frame.get_timestamp() * 1_000_000)
                        rgb = np.asanyarray(frame.get_data())
                    returned = time.monotonic_ns()
                    detail = dict(capture_session=session, wait_ms=(returned-wait_begin)/1e6,
                        previous_processing_ms=(wait_begin-previous_return)/1e6
                            if previous_return is not None else None,
                        capture_return_interval_ms=(returned-previous_return)/1e6
                            if previous_return is not None else None,
                        camera_counter_delta=number-previous_number if previous_number is not None else None,
                        capture_return_utc_ns=wall)
                    previous_number, previous_return = number, returned
                    if self.trace:
                        self.trace.event('capture', camera=camera, serial=serial,
                            camera_frame_number=number, capture_utc_ns=stamp,
                            timestamp_source='synthetic' if args.synthetic else 'camera_global_time',
                            clock_epoch=epoch, **detail)
                    if not guard.accept(stamp, wall, time.monotonic(), epoch):
                        if self.trace:
                            self.trace.event('timestamp_guard_rejected', camera=camera,
                                capture_session=session, camera_frame_number=number,
                                capture_utc_ns=stamp, capture_return_utc_ns=wall)
                        hub.clear('camera timestamps warming up or discontinuity detected')
                        continue
                    encode_begin = time.monotonic_ns()
                    payload = encode_rgb(rgb, args.codec, args.jpeg_quality)
                    detail['encode_ms'] = (time.monotonic_ns()-encode_begin)/1e6
                    header = dict(type='frame', camera=camera, serial=serial,
                        clock_epoch=epoch, capture_utc_ns=stamp, camera_frame_number=number,
                        width=args.width, height=args.height, encoding='rgb8', codec=args.codec,
                        synthetic=args.synthetic, timestamp_source='camera_global_time' if not args.synthetic else 'synthetic')
                    if self.trace:
                        header['pc_capture_trace'] = detail
                    sequence = hub.offer(header, payload)
                    if self.trace:
                        self.trace.event('encoded', camera=camera, camera_frame_number=number,
                            capture_utc_ns=stamp, sequence=sequence, payload_bytes=len(payload), **detail)
            except Exception as exc:
                if self.trace:
                    self.trace.event('capture_exception', camera=camera, capture_session=session, error=str(exc))
                hub.clear(str(exc))
                print(f'[{camera}] {exc}; retry in 2s', flush=True)
            finally:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
            self.stopping.wait(2.)

    def handle(self, connection):
        connection.settimeout(2.)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)
        try:
            request, payload = receive_packet(connection)
            if payload:
                raise ValueError('unexpected request payload')
            if request.get('type') == 'clock':
                while not self.stopping.is_set():
                    t2, epoch = self.wall()
                    if request.get('type') != 'clock' or type(request.get('t1')) is not int:
                        raise ValueError('invalid clock request')
                    t3, epoch_after = self.wall()
                    if epoch_after != epoch:
                        continue
                    send_packet(connection, dict(type='clock', t1=request['t1'], t2=t2, t3=t3, clock_epoch=epoch))
                    request, payload = receive_packet(connection)
                    if payload:
                        raise ValueError('unexpected clock payload')
            elif request.get('type') == 'subscribe' and request.get('camera') in self.hubs:
                hub = self.hubs[request['camera']]
                sequence = 0
                while not self.stopping.is_set():
                    frame = hub.after(sequence, self.stopping)
                    if frame is None:
                        wall, epoch = self.wall()
                        send_packet(connection, dict(type='waiting', reason=hub.error,
                            clock_epoch=epoch, pc_utc_ns=wall))
                        continue
                    header, data = frame
                    wall, epoch = self.wall()
                    sequence = header['sequence']
                    if header['clock_epoch'] != epoch or wall - header['capture_utc_ns'] > 300_000_000:
                        if self.trace:
                            self.trace.event('send_rejected', camera=request['camera'],
                                sequence=sequence, camera_frame_number=header['camera_frame_number'],
                                capture_utc_ns=header['capture_utc_ns'], pc_utc_ns=wall)
                        continue
                    send_begin = time.monotonic_ns()
                    send_packet(connection, dict(header, pc_send_utc_ns=wall), data)
                    if self.trace:
                        self.trace.event('sent', camera=request['camera'], sequence=sequence,
                            camera_frame_number=header['camera_frame_number'],
                            send_ms=(time.monotonic_ns()-send_begin)/1e6)
            else:
                raise ValueError('unknown request')
        except (OSError, EOFError, ValueError, KeyError):
            return

    def run(self):
        owner = self
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                owner.handle(self.request)
        class TCPServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
        with TCPServer((self.args.bind, self.args.port), Handler) as server:
            server.timeout = .25
            for camera, serial in zip(CAMERAS, (self.args.left_serial, self.args.right_serial)):
                thread = threading.Thread(target=self.capture, args=(camera, serial), daemon=True)
                thread.start()
                self.threads.append(thread)
            print(f'Listening {self.args.bind}:{self.args.port}; codec={self.args.codec}, source={self.args.fps}Hz', flush=True)
            if self.args.synthetic:
                print('TEST ONLY: synthetic frames, receiver rejects unless --allow-synthetic', flush=True)
            try:
                while not self.stopping.is_set():
                    server.handle_request()
            finally:
                self.stopping.set()
                for thread in self.threads:
                    thread.join(4.)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bind', default='0.0.0.0')
    p.add_argument('--port', type=int, default=DEFAULT_PORT)
    p.add_argument('--left-serial', default=LEFT_SERIAL)
    p.add_argument('--right-serial', default=RIGHT_SERIAL)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, choices=(15, 30), default=30)
    p.add_argument('--codec', choices=('png', 'raw', 'jpeg'), default='png')
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--warm-seconds', type=float, default=3.)
    p.add_argument('--diagnostics-dir', type=Path,
                   help='optional asynchronous per-frame capture/encoding trace; does not save images')
    p.add_argument('--list', action='store_true')
    p.add_argument('--synthetic', action='store_true', help='diagnostic images only; never training data')
    p.add_argument('--test-clock-offset-ms', type=float, default=0., help='synthetic mode only')
    args = p.parse_args(argv)
    if not 0 < args.port < 65536 or min(args.width, args.height) < 1 or args.width * args.height > 1920 * 1080:
        p.error('invalid port or image size')
    if args.left_serial == args.right_serial or not 0 <= args.warm_seconds <= 60 or not 5 <= args.jpeg_quality <= 100:
        p.error('invalid camera serials, warmup or JPEG quality')
    if args.test_clock_offset_ms and not args.synthetic:
        p.error('--test-clock-offset-ms requires --synthetic')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        import pyrealsense2 as rs
        for device in rs.context().query_devices():
            print(json.dumps(dict(name=device.get_info(rs.camera_info.name),
                serial=device.get_info(rs.camera_info.serial_number),
                usb=device.get_info(rs.camera_info.usb_type_descriptor) if device.supports(rs.camera_info.usb_type_descriptor) else 'unknown'), ensure_ascii=False))
        return 0
    server = CameraServer(args)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: server.stopping.set())
    try:
        server.run()
    finally:
        if server.trace:
            server.trace.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
