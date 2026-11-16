#!/usr/bin/env python3
"""Capture physical dual-arm teaching plus PICO head/waist and three ROS cameras."""
import argparse
import fcntl
import json
import ipaddress
from pathlib import Path
import queue
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from xcore_teach_recording.core import Capture, NotReady
from xcore_teach_recording.dataset import EpisodeWriter, prepare_root, schema
from xcore_teach_recording.udp_preview import DEFAULT_IP


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/niic/aide/assets/dataset/act_teach_30hz'))
    parser.add_argument('--task', default='dual-arm physical teaching with head and waist')
    parser.add_argument('--fps', type=int, default=30, help='synchronized capture rate, default: 30 Hz')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--max-skew', type=float, default=None,
                        help='maximum source timestamp spread, default: min(0.15, 1/fps) seconds')
    parser.add_argument('--max-age', type=float, default=.5)
    parser.add_argument('--state-max-age', type=float, default=.25)
    parser.add_argument('--udp-video-ip', default=DEFAULT_IP, help=f'JPEG preview destination, default: {DEFAULT_IP}')
    parser.add_argument('--udp-video-jpeg-quality', type=int, default=50)
    parser.add_argument('--no-udp-video', action='store_true', help='disable UDP camera preview')
    parser.add_argument('--check', action='store_true', help='subscribe and check readiness only; write no data')
    parser.add_argument('--check-seconds', type=float, default=15.)
    args = parser.parse_args()
    if not args.task.strip() or not 0 < args.fps <= 30 or min(args.width, args.height) <= 0:
        parser.error('task must not be empty; require 0<fps<=30 and positive image dimensions')
    if args.max_skew is None:
        args.max_skew = min(.15, 1 / args.fps)
    try:
        args.udp_video_ip = str(ipaddress.IPv4Address(args.udp_video_ip))
    except ipaddress.AddressValueError:
        parser.error('--udp-video-ip must be an IPv4 address')
    if not 5 <= args.udp_video_jpeg_quality <= 100:
        parser.error('--udp-video-jpeg-quality must be in [5, 100]')
    for key in ('max_skew', 'max_age', 'state_max_age', 'check_seconds'):
        if not 0 < getattr(args, key) <= 60:
            parser.error(f'{key} must be positive and at most 60')
    return args


def main():
    args = parse_args()
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from xcore_teach_recording.ros_node import RecorderNode

    # All motor services remain owned by their existing drivers/PICO bridge.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = RecorderNode(args)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_error = []

    def spin():
        try:
            executor.spin()
        except Exception as exc:
            spin_error.append(exc)

    thread = threading.Thread(target=spin, name='teach-ros', daemon=True)
    thread.start()
    lock_file = None
    capture = None
    code = 0
    try:
        settings = schema(args.fps, args.width, args.height)
        if not args.check:
            root = args.root.expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            lock_file = (root / '.recorder.lock').open('a')
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('another recorder owns this dataset root')
            prepare_root(root, settings)
            capture = Capture(node.inputs, lambda: EpisodeWriter(root, args.task.strip(), settings), args.fps,
                              notify=lambda text: print(text, flush=True))
            print('Right A=dual-arm teaching; right B=start/stop+save; double B=discard (350 ms). '
                  'Terminal: s=save stopped data, d=discard stopped data, q=quit.', flush=True)

            def terminal():
                for line in sys.stdin:
                    node.commands.put(('terminal', line.strip().lower()))
            threading.Thread(target=terminal, name='teach-terminal', daemon=True).start()
        print('Waiting for both arms in teaching, head/waist feedback, right-controller heartbeat and all three cameras.', flush=True)
        deadline = time.monotonic() + args.check_seconds
        last_status = None
        while rclpy.ok():
            if spin_error:
                raise RuntimeError(f'ROS receiver failed: {spin_error[0]}')
            mono, ros = time.monotonic(), node.ros_now()
            if args.check:
                try:
                    node.inputs.snapshot(ros, mono)
                except NotReady as exc:
                    if str(exc) != last_status:
                        print(f'Waiting: {exc}', flush=True)
                        last_status = str(exc)
                else:
                    rates = node.inputs.rates(mono)
                    print(f'Ready: {args.fps} Hz, 16-dimensional state and synchronized RGB cameras. '
                          + 'Measured Hz: ' + ', '.join(f'{key}={value:.1f}' for key, value in rates.items()), flush=True)
                    break
                if mono >= deadline:
                    print('Readiness check timed out.', file=sys.stderr)
                    code = 2
                    break
            else:
                quit_requested = False
                # Bound command work so receiver/button storms cannot starve sampling.
                for _ in range(100):
                    try:
                        operation, value = node.commands.get_nowait()
                    except queue.Empty:
                        break
                    if operation == 'pause':
                        capture.pause(value)
                    elif operation == 'button':
                        try:
                            capture.button(node.ros_now(), time.monotonic(), pressed_at=value)
                        except (RuntimeError, OSError) as exc:
                            print(f'Cannot handle B: {exc}', file=sys.stderr, flush=True)
                    elif value == 'q':
                        quit_requested = True
                        break
                    elif value in ('s', 'd'):
                        try:
                            capture.command(value)
                        except (RuntimeError, OSError) as exc:
                            print(f'Cannot finish episode: {exc}', file=sys.stderr, flush=True)
                    else:
                        print('B=start/stop+save, double B=discard; terminal s/d=finish stopped data, q=quit', flush=True)
                if quit_requested:
                    break
                capture.tick(ros, mono)
            time.sleep(.005)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'Recorder failed: {exc}', file=sys.stderr)
        code = 1
    finally:
        if capture is not None and capture.writer is not None:
            capture.pause('recorder exiting')
            try:
                path = capture.writer.preserve()
                print(f'Unconfirmed data retained at {path}; not treated as a saved episode.', flush=True)
            except Exception as exc:
                print(f'Failed to checkpoint pending episode: {exc}', file=sys.stderr)
                code = 1
        executor.shutdown(timeout_sec=2.)
        thread.join(timeout=3.)
        node.destroy_node()
        rclpy.shutdown()
        if lock_file:
            lock_file.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
