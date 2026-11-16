#!/usr/bin/env python3
"""Measure existing camera topics and temporary PNG writes; no robot/DO commands.

Source ROS and vision workspace setup first. Start the usual camera group once.
The one-command teaching launcher applies camera_capture_fastdds.xml to its
camera children. For a separate camera-only session, use:
  FASTRTPS_DEFAULT_PROFILES_FILE=/absolute/path/to/recording/camera_capture_fastdds.xml bash hardware/robot_vision_ws/scripts/start_cameras_30hz.sh
--mode receive measures lightweight reception; decode also copies RGB;
capture runs the actual recorder with explicitly simulated joints/DO; disk
benchmarks cached camera images on the dataset filesystem. Temporary image
files are removed by this tool; no training episode is added to the real root.
--real-feedback with capture reads existing high-rate arm topics and polls DO
readback without writes. It never starts teaching, initializes or moves arms.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types

import numpy as np

RECORDING = Path(__file__).resolve().parents[1] / 'recording'
sys.path.insert(0, str(RECORDING))
import record_xcore_teach_suction as current


def udp_counters():
    """Whole-host counters; other processes may contribute to these drops."""
    rows = Path('/proc/net/snmp').read_text().splitlines()
    for fields, values in zip(rows, rows[1:]):
        if fields.startswith('Udp:') and values.startswith('Udp:'):
            return dict(zip(fields.split()[1:], map(int, values.split()[1:])))
    return {}


def load_recorder(path):
    if path is None:
        return current
    module = types.ModuleType('capture_profile_baseline')
    module.__file__ = str(RECORDING / 'record_xcore_teach_suction.py')
    sys.modules[module.__name__] = module
    exec(compile(path.read_text(), module.__file__, 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('receive', 'decode', 'capture', 'disk'), required=True)
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recorder-version', type=Path)
    parser.add_argument('--image-qos', choices=('reliable', 'best_effort'), default='reliable')
    parser.add_argument('--real-feedback', action='store_true',
                        help='capture only: existing arm recording topics and read-only M050 feedback')
    parser.add_argument('--disk-parent', type=Path, default=current.DEFAULT_ROOT.parent)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error('--seconds must be 1..300')
    if args.real_feedback and args.mode != 'capture':
        parser.error('--real-feedback requires --mode capture')
    if args.output.exists():
        parser.error('output already exists')
    rec = load_recorder(args.recorder_version)
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, JointState
    from diagnostic_msgs.msg import DiagnosticArray
    from realsense2_camera_msgs.msg import Metadata
    from xcore_teach_recording.ros_node import decode_rgb, joint_values

    class Probe(Node):
        def __init__(self):
            super().__init__('teach_capture_profiler_' + str(__import__('os').getpid()))
            self.samples = {k: [] for k in rec.CAMERAS}
            self.images = {}
            self.metadata, self.camera_diagnostics = {}, {}
            self.metadata_samples = {key: [] for key in rec.CAMERAS}
            self.enabled = False
            self.kept = []
            self.inputs = None
            for key in rec.CAMERAS:
                prefix = f'/robot/sensors/cameras/{key}/color/'
                self.kept.append(self.create_subscription(Image, prefix + 'image_raw',
                    lambda msg, k=key: self.image(k, msg),
                    QoSProfile(depth=20, reliability=(ReliabilityPolicy.RELIABLE
                        if args.image_qos == 'reliable' else ReliabilityPolicy.BEST_EFFORT))))
                self.kept.append(self.create_subscription(Metadata, prefix + 'metadata',
                    lambda msg, k=key: self.meta(k, msg), 10))
            self.kept.append(self.create_subscription(DiagnosticArray,
                '/robot/sensors/cameras/diagnostics', self.diag, 20))
            if args.real_feedback:
                for side in rec.SIDES:
                    self.kept.append(self.create_subscription(JointState,
                        f'{rec.PREFIX}/recording_joint_states/{side}_arm',
                        lambda msg, s=side: self.joints(s, msg),
                        QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)))

        def joints(self, side, message):
            if self.inputs is not None:
                received = time.monotonic()
                stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
                values = joint_values(message, [f'{side}_joint_{i}' for i in range(1,8)])
                self.inputs.push(side, stamp, received, values)
                self.inputs.observe(side, stamp, received, self.get_clock().now().nanoseconds/1e9)

        def meta(self, key, message):
            try:
                self.metadata[key] = json.loads(message.json_data)
                if self.enabled:
                    stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
                    self.metadata_samples[key].append((stamp, time.monotonic()))
            except ValueError:
                pass

        def diag(self, message):
            for status in message.status:
                if any(k in status.name for k in rec.CAMERAS):
                    self.camera_diagnostics[status.name] = {
                        v.key: v.value for v in status.values}

        def image(self, key, message):
            at = time.monotonic()
            stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
            decode_ms = 0.
            if args.mode != 'receive' or key not in self.images:
                begin = time.perf_counter()
                frame = decode_rgb(message, 640, 480)
                decode_ms = (time.perf_counter() - begin) * 1000
                self.images[key] = frame
                if self.inputs is not None:
                    self.inputs.push(key, stamp, at, frame)
            if self.enabled:
                self.samples[key].append((stamp, at,
                    self.get_clock().now().nanoseconds / 1e9 - stamp, decode_ms))

    rclpy.init()
    node = Probe()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    writer = capture = outputs = device = None
    try:
        deadline = time.monotonic() + 20.
        while len(node.images) < 3:
            if time.monotonic() > deadline:
                raise RuntimeError('three existing camera topics not ready; profiler never starts hardware')
            time.sleep(.02)
        info = {k: [dict(node=i.node_namespace + '/' + i.node_name,
                          reliability=str(i.qos_profile.reliability))
                    for i in node.get_publishers_info_by_topic(
                        f'/robot/sensors/cameras/{k}/color/image_raw')]
                for k in rec.CAMERAS}
        if any(len(v) != 1 for v in info.values()):
            raise RuntimeError('camera publisher count must be exactly one per topic: ' + str(info))
        time.sleep(1.)
        with tempfile.TemporaryDirectory(prefix='.capture-benchmark-', dir=args.disk_parent) as directory:
            root = Path(directory)
            config = rec.parse_args(['--root', str(root)])
            settings = rec.dataset_schema(config)
            rec.prepare_root(root, settings)
            device = rec.M050Device(config) if args.real_feedback else None
            outputs = rec.SuctionOutputs(device, report=lambda _: None, read_only=args.real_feedback)
            if args.real_feedback:
                outputs.start()
            if args.mode == 'capture':
                inputs = rec.KeyboardInputs(outputs, config)
                inputs.teaching = dict.fromkeys(rec.SIDES, True)
                if args.real_feedback:
                    inputs.set_return_recording(True)
                node.inputs = inputs
                capture = rec.Capture(inputs,
                    lambda: rec.EpisodeWriter(root, 'BENCHMARK_SYNTHETIC_ARMS_DO', settings),
                    config.fps, report=lambda _: None)
            elif args.mode == 'disk':
                writer = rec.EpisodeWriter(root, 'BENCHMARK_CACHED_IMAGES', settings)
                writer.begin_segment()
            start = time.monotonic()
            cpu_start = time.process_time()
            udp_start = udp_counters()
            node.enabled = True
            if capture is not None:
                capture.start(node.get_clock().now().nanoseconds / 1e9, start)
            accepted, queue_full, write_add_ms = 0, 0, []
            peak_queue = 0
            next_write = start
            next_joints = start
            while time.monotonic() - start < args.seconds:
                now = time.monotonic()
                ros_now = node.get_clock().now().nanoseconds / 1e9
                if capture is not None:
                    if not args.real_feedback:
                        outputs.record_readback((False, False), now - .001)
                    if not args.real_feedback and now >= next_joints:
                        for side in rec.SIDES:
                            inputs.push(side, ros_now, now, np.zeros(7))
                        next_joints = now + .01
                    capture.tick(ros_now, now)
                    writer = capture.writer
                elif writer is not None and now >= next_write:
                    before = time.perf_counter()
                    try:
                        writer.add(rec.Bundle(ros_now, np.zeros(16), np.zeros(16),
                            dict(node.images), np.full(5, ros_now), ros_now, np.full(2, ros_now)))
                        accepted += 1
                    except RuntimeError:
                        queue_full += 1
                    write_add_ms.append((time.perf_counter() - before) * 1000)
                    next_write += 1 / 30
                if writer is not None:
                    peak_queue = max(peak_queue, writer.jobs.qsize())
                time.sleep(.002)
            node.enabled = False
            duration = time.monotonic() - start
            cpu_seconds = time.process_time() - cpu_start
            if capture is not None:
                capture.pause('benchmark end', time.monotonic())
                accepted = len(writer.rows)
            report = dict(mode=args.mode, duration_seconds=duration,
                image_qos=args.image_qos,
                synthetic_arm_and_DO=args.mode == 'capture' and not args.real_feedback,
                real_arm_and_DO_feedback=args.real_feedback, real_robot_or_DO_commands=False,
                environment={key: os.environ.get(key) for key in (
                    'RMW_IMPLEMENTATION', 'ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY',
                    'FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE')},
                disk_writes_use_os_cache=True, fsync_measured=False,
                host_udp_counter_delta={key: value-udp_start.get(key, value)
                                        for key, value in udp_counters().items()},
                host_udp_counter_scope='whole_host_including_other_processes',
                recorder_version=str(args.recorder_version or RECORDING / 'record_xcore_teach_suction.py'),
                publishers=info, cameras={}, publisher_metadata=node.metadata,
                publisher_diagnostics=node.camera_diagnostics, cpu_seconds=cpu_seconds,
                accepted_frames=accepted, saved_fps=accepted / duration,
                queue_full=queue_full, peak_write_queue=peak_queue)
            for key, values in node.samples.items():
                values = np.asarray(values)
                if len(values) < 2:
                    raise RuntimeError('too few image samples: ' + key)
                dt = np.diff(values[:, 0])
                report['cameras'][key] = dict(messages=len(values),
                    receive_hz=(len(values)-1)/(values[-1,1]-values[0,1]),
                    source_timestamp_hz=(len(values)-1)/(values[-1,0]-values[0,0]),
                    source_intervals_over_50ms=int(sum(dt>.05)),
                    source_max_interval_ms=float(dt.max()*1000),
                    receive_lag_ms_p50_p95_max=np.percentile(values[:,2]*1000,[50,95,100]).tolist(),
                    decode_ms_p50_p95_max=np.percentile(values[:,3],[50,95,100]).tolist())
                metadata = np.asarray(node.metadata_samples[key])
                if len(metadata) >= 2:
                    image_stamps = set(np.round(values[:, 0], 6))
                    lo = max(values[0, 0], metadata[0, 0])
                    hi = min(values[-1, 0], metadata[-1, 0])
                    report['cameras'][key]['metadata_receive_hz'] = (
                        (len(metadata)-1)/(metadata[-1,1]-metadata[0,1]))
                    report['cameras'][key]['metadata_without_image'] = int(sum(
                        lo <= stamp <= hi and round(stamp,6) not in image_stamps
                        for stamp in metadata[:,0]))
            if capture is not None:
                report['segments'] = writer.segments
                report['gaps'] = writer.capture_gaps
                report['receive_diagnostics'] = inputs.diagnostics(time.monotonic())
            if writer is not None:
                before = time.monotonic()
                writer.jobs.join()
                report['writer_drain_seconds'] = time.monotonic() - before
                report['image_bytes_written'] = sum(p.stat().st_size for p in writer.path.rglob('*.png'))
                if hasattr(writer, 'write_metrics'):
                    report['writer'] = writer.write_metrics()
                    report['queue_full'] = report['writer']['queue_full_count']
                if capture is not None and accepted:
                    # Verify actual temporary PNGs and timestamp/label arrays
                    # after timing, then remove this benchmark's own staging.
                    writer.flush(complete=True)
                    _, verified = rec.validate_episode(writer.path, settings, check_images=True)
                    report['validation'] = 'all_arrays_and_images_pass'
                    report['max_source_skew_ms'] = float(
                        np.ptp(verified['source_timestamp'],axis=1).max()*1000)
                writer.discard()
                writer = None
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({k:v for k,v in report.items() if k not in
                ('gaps','segments','publisher_metadata','publisher_diagnostics')}, ensure_ascii=False, indent=2))
            print('report:',args.output,flush=True)
    finally:
        node.inputs = None
        if writer is None and capture is not None:
            writer = capture.writer
        if writer is not None and not writer.closed:
            writer.discard()
        if outputs is not None and outputs.started:
            outputs.close()
        elif device is not None:
            device.close()
        executor.shutdown(timeout_sec=2.)
        thread.join(3.)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
