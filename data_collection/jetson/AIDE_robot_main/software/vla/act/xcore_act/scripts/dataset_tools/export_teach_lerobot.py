#!/usr/bin/env python3
"""Validate raw physical-teaching captures and export LeRobot 0.4.3 ACT data."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from xcore_teach_recording.core import CAMERAS, STATE_NAMES
from xcore_teach_recording.dataset import schema, validate_episode


def keyboard_recorder(settings):
    if settings.get('schema_id') != 'keyboard_dual_arm_suction_v1':
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'recording'))
    import record_xcore_teach_suction
    return record_xcore_teach_suction


def export(source, destination, repo_id, factory=None):
    source, destination = Path(source), Path(destination)
    settings = json.loads((source / 'schema.json').read_text())
    keyboard = keyboard_recorder(settings)
    if keyboard is not None:
        return keyboard.export_dataset(source, destination, repo_id, factory)
    expected = schema(settings['fps'], settings['width'], settings['height'])
    if settings != expected:
        raise ValueError('unsupported raw dataset schema')
    episodes = sorted(source.glob('episode_*'))
    if not episodes:
        raise ValueError('no completed episodes; .partial directories are not exported')
    if destination.exists():
        raise FileExistsError('export destination exists; choose a new --destination')
    # Validate all records before creating any output dataset.
    for episode in episodes:
        validate_episode(episode, settings)
    if factory is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError('export requires the existing LeRobot 0.4.3 ACT environment; raw capture does not') from exc
        factory = LeRobotDataset.create
    features = {
        key: {'dtype': 'float32', 'shape': (16,), 'names': list(STATE_NAMES)}
        for key in ('observation.state', 'action')
    }
    features.update({f'observation.images.{key}': {
        'dtype': 'video', 'shape': (settings['height'], settings['width'], 3),
        'names': ['height', 'width', 'channels']} for key in CAMERAS})
    if not float(settings['fps']).is_integer():
        raise ValueError('LeRobot export requires an integer fps')
    dataset = factory(repo_id=repo_id, root=destination, fps=int(settings['fps']), features=features,
                      robot_type='xcore_dual_arm_head_waist_drag', use_videos=True,
                      video_backend='pyav', vcodec='h264', image_writer_processes=0,
                      image_writer_threads=3)
    mapping = []
    exported = 0
    try:
        for episode in episodes:
            meta, arrays = validate_episode(episode, settings, check_images=False)
            # Never train action chunks across pauses or unrecorded repositioning.
            for segment in meta['segments']:
                for index in range(segment['start'], segment['end']):
                    frame = {'observation.state': arrays['state'][index].copy(),
                             'action': arrays['action'][index].copy(), 'task': meta['task']}
                    for key in CAMERAS:
                        bgr = cv2.imread(str(episode / 'images' / key / f'{index:06d}.png'))
                        frame[f'observation.images.{key}'] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    dataset.add_frame(frame)
                if dataset.image_writer is not None:
                    dataset.image_writer.wait_until_done()
                dataset.save_episode()
                mapping.append(dict(episode_index=exported, raw_episode=episode.name, **segment))
                exported += 1
        # Per-version metadata flushing, if supplied by the installed LeRobot.
        finalize = getattr(dataset, 'finalize', None)
        if finalize is not None:
            finalize()
        (destination / 'meta' / 'physical_teach_source.json').write_text(
            json.dumps(dict(schema=settings, episodes=mapping), indent=2) + '\n')
    finally:
        stop = getattr(dataset, 'stop_image_writer', None)
        if stop is not None:
            stop()
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--repo-id', default='local/xcore_physical_teach')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.validate_only:
            expected = json.loads((args.root / 'schema.json').read_text())
            keyboard = keyboard_recorder(expected)
            validator = keyboard.validate_episode if keyboard is not None else validate_episode
            episodes = sorted(args.root.glob('episode_*'))
            if not episodes:
                raise ValueError('no completed episodes')
            for episode in episodes:
                meta, _ = validator(episode, expected)
                print(f'{episode.name}: {meta["frames"]} frames, {len(meta["segments"])} segments; structure/timing OK')
        else:
            if args.destination is None:
                parser.error('--destination is required for export')
            count = export(args.root, args.destination, args.repo_id)
            print(f'Exported {count} continuous LeRobot episodes to {args.destination}')
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'Dataset error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
