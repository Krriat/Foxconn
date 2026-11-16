#!/usr/bin/env python3
"""Replay saved RGB/joints/DO through the writer in temporary storage.

No ROS, robot, camera or DO commands. All source arrays and decoded pixels are
checked against the temporary output, then that output is removed. Existing
source gaps and mode boundaries remain. --inject-recorded-backpressure retries
once at each recorded full-queue boundary to reproduce the segmentation bug.
"""
import argparse
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'recording'))
import record_xcore_teach_suction as current
from profile_teach_capture import load_recorder


def replay(rec, path, inject):
    settings = json.loads((path.parent / 'schema.json').read_text())
    meta, arrays = rec.validate_episode(path, settings, check_images=False)
    prepared, errors, stopping = queue.Queue(maxsize=16), queue.Queue(), threading.Event()

    def decode():
        try:
            for i in range(meta['frames']):
                images = {}
                for key in rec.CAMERAS:
                    bgr = cv2.imread(str(path / 'images' / key / f'{i:06d}.png'))
                    if bgr is None:
                        raise ValueError(f'missing source image {key}/{i}')
                    images[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bundle = rec.Bundle(arrays['ros_timestamp'][i], arrays['state'][i], arrays['action'][i],
                    images, arrays['source_timestamp'][i], arrays['suction_feedback_timestamp'][i],
                    arrays['suction_command_timestamp'][i])
                while not stopping.is_set():
                    try:
                        prepared.put(bundle, timeout=.1)
                        break
                    except queue.Full:
                        pass
                if stopping.is_set():
                    return
        except Exception as exc:
            errors.put(exc)

    class ReplayInputs:
        current = None

        def snapshot(self, ros_now, _mono, _images=None, previous=None):
            if not errors.empty():
                raise errors.get()
            if self.current is not None and (previous or {}).get('head', -np.inf) >= self.current.source_stamps[2]:
                self.current = None
            if self.current is None:
                try:
                    self.current = prepared.get_nowait()
                except queue.Empty:
                    raise rec.AwaitFrame('waiting for replay prefetch')
            if self.current.stamp > ros_now:
                raise rec.AwaitFrame('waiting for original sample time')
            return self.current

    thread = threading.Thread(target=decode, daemon=True)
    thread.start()
    capture = writer = None
    try:
        with tempfile.TemporaryDirectory(prefix='.capture-replay-', dir=path.parent.parent) as directory:
            root = Path(directory)
            rec.prepare_root(root, settings)
            capture = rec.Capture(ReplayInputs(), lambda: rec.EpisodeWriter(root, 'TEMPORARY_REPLAY', settings),
                                  settings['fps'], report=lambda _: None)
            # Prime the first source image before starting the benchmark clock.
            capture.inputs.current = prepared.get(timeout=30.)
            start = time.monotonic()
            capture.start(float(arrays['ros_timestamp'][0]), start)
            writer = capture.writer
            controls = {s['end']: s['pause_reason'] for s in meta['segments'][:-1]
                        if any(word in s['pause_reason'] for word in ('键盘', '切换', 'teaching:', 'B 回位', 'ROS 时间'))}
            stalls = {s['end'] for s in meta['segments'][:-1]
                      if s['pause_reason'] == '图像写盘队列已满'} if inject else set()
            injected, modes_applied = set(), set()
            add = writer.add

            def enqueue(bundle):
                index = len(writer.rows)
                if index in stalls and index not in injected:
                    injected.add(index)
                    exception = getattr(rec, 'WriteQueueFull', RuntimeError)
                    raise exception('图像写盘队列已满')
                add(bundle)

            writer.add = enqueue
            original_span = float(np.ptp(arrays['ros_timestamp']))
            while len(writer.rows) < meta['frames']:
                now = time.monotonic()
                if now-start > original_span+60.:
                    raise RuntimeError('replay timed out')
                index = len(writer.rows)
                if index in controls and index not in modes_applied:
                    capture.mark_gap(controls[index], now)
                    modes_applied.add(index)
                capture.tick(float(arrays['ros_timestamp'][0])+now-start, now)
                if not errors.empty():
                    raise errors.get()
                time.sleep(.002)
            elapsed = time.monotonic()-start
            capture.pause('replay end', time.monotonic())
            writer.jobs.join()
            writer.flush(complete=True)
            checked, output = rec.validate_episode(writer.path, settings, check_images=False)
            for key, value in arrays.items():
                if not np.array_equal(value, output[key]):
                    raise AssertionError('source array changed: '+key)
            for key in rec.CAMERAS:
                for i in range(meta['frames']):
                    source = cv2.imread(str(path / 'images' / key / f'{i:06d}.png'))
                    target = cv2.imread(str(writer.path / 'images' / key / f'{i:06d}.png'))
                    if source is None or target is None or not np.array_equal(source, target):
                        raise AssertionError(f'RGB pixels changed: {key}/{i}')
            result = dict(source=str(path), frames=meta['frames'], replay_elapsed_seconds=elapsed,
                original_source_span_seconds=original_span, original_segments=len(meta['segments']),
                replay_segments=len(checked['segments']), original_image_bytes=meta['writer_diagnostics']['written_bytes'],
                injected_full_queue_retries=len(injected), arrays='all_exact', pixels='all_exact',
                writer=writer.write_metrics(), real_robot_or_DO_commands=False,
                temporary_output_removed=True)
            writer.discard()
            return result
    finally:
        stopping.set()
        thread.join(3.)
        if writer is not None and not writer.closed:
            writer.discard()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recorder-version', type=Path)
    parser.add_argument('--inject-recorded-backpressure', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    rec = load_recorder(args.recorder_version)
    results = []
    for path in args.episodes:
        result = replay(rec, path.resolve(), args.inject_recorded_backpressure)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
