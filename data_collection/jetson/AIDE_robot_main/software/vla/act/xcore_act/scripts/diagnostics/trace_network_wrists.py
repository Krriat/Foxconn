#!/usr/bin/env python3
"""Trace PC camera frame counters and transport; no ROS, robot, IO or image writes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np

NETWORK = Path(__file__).resolve().parents[1] / 'recording/network_cameras'
sys.path.insert(0, str(NETWORK))
from protocol import CAMERAS, DEFAULT_HOST, DEFAULT_PORT, connect, receive_packet, send_packet
from clock_sync import ClockEstimator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--seconds', type=float, default=180.)
    parser.add_argument('--source-fps', type=float, default=30.)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if (not 0 < args.port < 65536 or not math.isfinite(args.seconds) or args.seconds <= 0
            or not math.isfinite(args.source_fps) or not 0 < args.source_fps <= 90):
        parser.error('invalid duration, FPS or port')
    root = args.out or Path(tempfile.mkdtemp(prefix='aide-wrist-wire-trace-'))
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / (key + '.jsonl') for key in CAMERAS] + [root / 'report.json']
    if any(path.exists() for path in paths):
        parser.error('trace files already exist; choose a fresh --out directory')
    stop = threading.Event()
    clock = ClockEstimator()
    samples = {key: [] for key in CAMERAS}
    failures = []
    connections = dict.fromkeys(CAMERAS, 0)
    start = time.monotonic()

    def clock_loop():
        while not stop.is_set():
            try:
                with connect(args.host, args.port) as sock:
                    while not stop.is_set():
                        m1, t1 = time.monotonic_ns(), time.time_ns()
                        send_packet(sock, dict(type='clock', t1=t1))
                        response, payload = receive_packet(sock)
                        t4, m4 = time.time_ns(), time.monotonic_ns()
                        if response.get('type') != 'clock' or response.get('t1') != t1 or payload:
                            raise ValueError('invalid clock response')
                        clock.add(t1, response['t2'], response['t3'], t4, m4 - m1,
                                  response['clock_epoch'])
                        stop.wait(.2)
            except (OSError, EOFError, ValueError, KeyError) as exc:
                failures.append(dict(stage='clock', error=str(exc)))
                clock.reset()
                stop.wait(.5)

    def camera_loop(camera):
        with (root / (camera + '.jsonl')).open('x', buffering=1024 * 1024) as output:
            while not stop.is_set():
                try:
                    with connect(args.host, args.port) as sock:
                        connections[camera] += 1
                        send_packet(sock, dict(type='subscribe', camera=camera))
                        while not stop.is_set():
                            header, payload = receive_packet(sock)
                            wall, mono = time.time_ns(), time.monotonic_ns()
                            if header.get('type') == 'waiting':
                                continue
                            if (header.get('type') != 'frame' or header.get('camera') != camera
                                    or header.get('synthetic')):
                                raise ValueError('real camera frame required')
                            stamp, number, sequence = (header.get(key) for key in
                                ('capture_utc_ns', 'camera_frame_number', 'sequence'))
                            if any(type(v) is not int or v < 1 for v in (stamp, number, sequence)):
                                raise ValueError('frame timestamp/counters missing')
                            row = dict(header, receive_utc_ns=wall, receive_monotonic_ns=mono,
                                payload_bytes=len(payload), connection=connections[camera])
                            try:
                                estimate = clock.estimate()
                                if estimate.epoch == header['clock_epoch']:
                                    row.update(pc_minus_jetson_ms=estimate.offset_ns / 1e6,
                                        capture_age_ms=(wall - stamp + estimate.offset_ns) / 1e6,
                                        clock_rtt_ms=estimate.rtt_ns / 1e6)
                            except ValueError:
                                pass
                            output.write(json.dumps(row, separators=(',', ':')) + '\n')
                            samples[camera].append((stamp, number, sequence, mono,
                                header['clock_epoch'], connections[camera]))
                except (OSError, EOFError, ValueError, KeyError) as exc:
                    failures.append(dict(stage=camera, error=str(exc)))
                    stop.wait(.5)

    threads = [threading.Thread(target=clock_loop, daemon=True)] + [
        threading.Thread(target=camera_loop, args=(key,), daemon=True) for key in CAMERAS]
    print('TRACE_DIR=' + str(root), flush=True)
    for thread in threads:
        thread.start()
    try:
        while time.monotonic() - start < args.seconds:
            if stop.wait(min(10., args.seconds - (time.monotonic() - start))):
                break
            print(json.dumps(dict(elapsed=round(time.monotonic() - start, 1),
                frames={key: len(value) for key, value in samples.items()})), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for thread in threads:
            thread.join(5.)
    report = dict(host=args.host, source_fps=args.source_fps, seconds=time.monotonic() - start,
        errors=failures, cameras={}, physical_arms=False, physical_IO=False, image_files_written=False)
    for camera, rows in samples.items():
        pairs = [(a, b) for a, b in zip(rows, rows[1:]) if a[4:] == b[4:]]
        gaps = []
        for a, b in pairs:
            delta_ms = (b[0] - a[0]) / 1e6
            if delta_ms > 1.6 * 1000 / args.source_fps or b[1] - a[1] != 1 or b[2] - a[2] != 1:
                gaps.append(dict(capture_pc_utc_ns=b[0], source_interval_ms=delta_ms,
                    camera_counter_delta=b[1] - a[1], encoded_sequence_delta=b[2] - a[2],
                    receipt_interval_ms=(b[3] - a[3]) / 1e6,
                    receive_elapsed_seconds=(b[3] / 1e9 - start)))
        intervals = [(b[0] - a[0]) / 1e6 for a, b in pairs]
        report['cameras'][camera] = dict(frames=len(rows), connections=connections[camera], gaps=gaps,
            receive_hz=(len(rows) - 1) / ((rows[-1][3] - rows[0][3]) / 1e9) if len(rows) > 1 else 0.,
            source_interval_ms_p50_p95_max=np.percentile(intervals, [50, 95, 100]).tolist()
                if intervals else [])
    (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(len(rows) > 1 for rows in samples.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
