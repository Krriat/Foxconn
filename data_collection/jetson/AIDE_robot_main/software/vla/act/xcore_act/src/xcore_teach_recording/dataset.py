"""Atomic raw episodes and validation for later LeRobot export."""
import json
from pathlib import Path
import queue
import shutil
import threading
import uuid

import cv2
import numpy as np

from .core import ACTION_SOURCE, CAMERAS, STATE_NAMES, STREAMS


def schema(fps, width, height):
    return dict(version=1, fps=float(fps), width=width, height=height,
                cameras=list(CAMERAS), state_names=list(STATE_NAMES),
                units='radian', action_source=ACTION_SOURCE,
                source_order=list(STREAMS), waist_timestamp='ROS receipt time; source Float64 has no header')


def prepare_root(root, expected):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'schema.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('existing dataset schema differs; choose a new --root')
    else:
        if any(root.glob('episode_*')) or any(root.glob('.episode_*.partial')):
            raise ValueError('episodes exist without schema.json; refusing to mix datasets')
        path.write_text(json.dumps(expected, indent=2) + '\n')
    return root


class EpisodeWriter:
    def __init__(self, root, task, settings, *, image_workers=None, image_queue_size=8):
        self.root, self.task, self.settings = Path(root), task, settings
        self.path = self.root / f'.episode_{uuid.uuid4().hex}.partial'
        self.path.mkdir()
        for key in CAMERAS:
            (self.path / 'images' / key).mkdir(parents=True)
        self.rows, self.segments = [], []
        self.segment_start = None
        self.error = None
        self.closed = False
        self.jobs = queue.Queue(maxsize=image_queue_size)
        # PNG encoding releases the GIL. Full-resolution three-camera 30 Hz
        # needs parallel encoders; retain the bounded queue and fail on overload.
        worker_count = image_workers if image_workers is not None else (3 if settings['fps'] > 20 else 1)
        self.workers = [threading.Thread(target=self._write_images, name=f'teach-images-{i}', daemon=True)
                        for i in range(worker_count)]
        for worker in self.workers:
            worker.start()

    def _write_images(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                if self.error:
                    continue
                index, images = job
                for key, image in images.items():
                    if image.shape != (self.settings['height'], self.settings['width'], 3) or image.dtype != np.uint8:
                        raise ValueError(f'invalid RGB shape/dtype for {key}')
                    path = self.path / 'images' / key / f'{index:06d}.png'
                    # Level 0 keeps lossless RGB while avoiding the PNG deflate
                    # bottleneck at 90 VGA images/s. Disk must sustain ~83 MB/s.
                    compression = 0 if self.settings['fps'] > 20 else 1
                    if not cv2.imwrite(str(path), image[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, compression]):
                        raise OSError(f'image write failed: {path}')
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.jobs.task_done()

    def check_error(self):
        if self.error:
            raise RuntimeError(f'disk/image writer failed: {self.error}')
        if self.closed:
            raise RuntimeError('episode writer is closed')

    def begin_segment(self):
        self.check_error()
        self.segment_start = len(self.rows)

    def end_segment(self, reason):
        if self.segment_start is not None and self.segment_start < len(self.rows):
            self.segments.append(dict(start=self.segment_start, end=len(self.rows), pause_reason=reason))
        self.segment_start = None

    def add(self, bundle):
        self.check_error()
        index = len(self.rows)
        # Arrays are owned by immutable receive samples; no further mutation.
        try:
            self.jobs.put_nowait((index, bundle.images))
        except queue.Full:
            raise RuntimeError('image writer queue full; paused instead of dropping frames')
        self.rows.append((bundle.stamp, bundle.state.copy(), bundle.source_stamps.copy()))

    def flush(self, complete=False):
        self.jobs.join()
        self.check_error()
        count = len(self.rows)
        if count:
            stamp, state, source = zip(*self.rows)
            state = np.stack(state).astype(np.float32)
            np.savez_compressed(self.path / 'trajectory.npz', state=state, action=state.copy(),
                                ros_timestamp=np.asarray(stamp, dtype=np.float64),
                                source_timestamp=np.stack(source),
                                timestamp=np.arange(count, dtype=np.float64) / self.settings['fps'])
        metadata = dict(task=self.task, frames=count, segments=self.segments,
                        complete=complete, **self.settings)
        (self.path / 'episode.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')

    def close(self):
        if not self.closed:
            for _ in self.workers:
                self.jobs.put(None)
            self.jobs.join()
            for worker in self.workers:
                worker.join()
            self.closed = True

    def save(self):
        if not self.rows:
            raise RuntimeError('cannot save an empty episode')
        if self.segment_start is not None:
            raise RuntimeError('pause recording before saving')
        self.flush(complete=True)
        self.close()
        used = [int(p.name[8:]) for p in self.root.glob('episode_*') if p.is_dir() and p.name[8:].isdigit()]
        number = max(used, default=-1) + 1
        destination = self.root / f'episode_{number:06d}'
        if destination.exists():
            raise RuntimeError('episode destination already exists')
        self.path.rename(destination)
        self.path = destination
        return destination

    def discard(self):
        self.close()
        # Only this writer's unique staging directory is eligible for deletion.
        if self.path.parent != self.root or not self.path.name.startswith('.episode_'):
            raise RuntimeError('refusing to discard a completed or foreign episode')
        shutil.rmtree(self.path)

    def preserve(self):
        self.end_segment('recorder exiting; not confirmed for saving')
        try:
            self.flush(complete=False)
        finally:
            self.close()
        return self.path


def validate_episode(path, expected=None, check_images=True):
    path = Path(path)
    meta = json.loads((path / 'episode.json').read_text())
    if not meta.get('complete'):
        raise ValueError('episode is partial/unconfirmed')
    if expected and any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('episode schema does not match dataset')
    if meta.get('state_names') != list(STATE_NAMES) or meta.get('action_source') != ACTION_SOURCE:
        raise ValueError('incompatible state/action schema')
    if not isinstance(meta.get('task'), str) or not meta['task'].strip():
        raise ValueError('missing task description')
    with np.load(path / 'trajectory.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    n = meta['frames']
    if n < 1 or arrays['state'].shape != (n, 16) or arrays['action'].shape != (n, 16):
        raise ValueError('invalid state/action dimensions')
    if not np.isfinite(arrays['state']).all() or not np.array_equal(arrays['state'], arrays['action']):
        raise ValueError('actions must equal finite same-sample measured positions')
    if arrays['source_timestamp'].shape != (n, len(STREAMS)) or arrays['ros_timestamp'].shape != (n,):
        raise ValueError('invalid source timestamp dimensions')
    if not np.isfinite(arrays['source_timestamp']).all():
        raise ValueError('invalid source timestamps')
    cursor = 0
    for segment in meta['segments']:
        start, end = segment['start'], segment['end']
        if start != cursor or not start < end <= n:
            raise ValueError('segments do not partition the recorded frames')
        if np.any(np.diff(arrays['ros_timestamp'][start:end]) <= 0):
            raise ValueError('non-monotonic frame timestamps inside segment')
        cursor = end
    if cursor != n:
        raise ValueError('unaccounted frames outside segments')
    if check_images:
        for camera in CAMERAS:
            for i in range(n):
                filename = path / 'images' / camera / f'{i:06d}.png'
                if not filename.is_file():
                    raise ValueError(f'missing image {camera}/{i:06d}')
                image = cv2.imread(str(filename))
                if image is None or image.shape != (meta['height'], meta['width'], 3):
                    raise ValueError(f'missing/invalid image {camera}/{i:06d}')
    return meta, arrays
