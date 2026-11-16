#!/usr/bin/env python3
"""Export recorded camera streams; reads SQLite only, never starts ROS."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    events = [json.loads(s) for s in (root/'run_events.jsonl').read_text().splitlines()]
    traces = [e['trace'] for e in events if e.get('event') == 'servo_trace' and e['trace'].get('active')]
    start, end = traces[0]['selection_time_ns'], traces[-1]['selection_time_ns'] + 1_000_000_000
    database = next((root/'rosbag').glob('*.db3'))
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    cameras = ('head', 'left_wrist', 'right_wrist')
    records, stamps = {}, {}
    for camera in cameras:
        topic = f'/robot/sensors/cameras/{camera}/color/image_raw/compressed'
        rows = connection.execute('SELECT m.timestamp,m.data FROM messages m JOIN topics t ON t.id=m.topic_id '
                                  'WHERE t.name=? AND m.timestamp BETWEEN ? AND ? ORDER BY m.timestamp',
                                  (topic, start-1_000_000_000, end+1_000_000_000)).fetchall()
        if not rows:
            raise ValueError(f'Missing camera {camera}')
        records[camera] = rows
        stamps[camera] = [r[0] for r in rows]
    destination = root/'success_three_cameras.mp4'
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920, 480))
    if not writer.isOpened():
        raise RuntimeError('Video encoder unavailable')
    cache, selected, head_samples = {}, {}, []
    sample_times = list(range(0, 29, 4)) + [30]
    next_sample = 0
    try:
        for i, stamp in enumerate(range(start, end, 50_000_000)):
            frames = []
            for camera in cameras:
                index = bisect_right(stamps[camera], stamp)-1
                if index < 0 or stamp-stamps[camera][index] > 300_000_000:
                    raise ValueError(f'Stale/missing {camera} frame in recording')
                if selected.get(camera) != index:
                    msg = deserialize_message(records[camera][index][1], CompressedImage)
                    decoded = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                    assert decoded.shape == (480, 640, 3)
                    cache[camera], selected[camera] = decoded, index
                frame = cache[camera].copy()
                cv2.putText(frame, f'{camera}  {i/20:.2f}s', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (0, 220, 0), 2)
                frames.append(frame)
            writer.write(np.concatenate(frames, axis=1))
            if next_sample < len(sample_times) and i/20 >= sample_times[next_sample]:
                head_samples.append(frames[0])
                next_sample += 1
    finally:
        writer.release()
        connection.close()
    if len(head_samples) != 9:
        raise ValueError('Expected nine overview frames')
    overview = np.concatenate([np.concatenate(head_samples[k:k+3], axis=1) for k in (0, 3, 6)], axis=0)
    cv2.imwrite(str(root/'success_overview.jpg'), overview)
    cap = cv2.VideoCapture(str(destination))
    frame_count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read(); cap.release()
    assert ok and frame.shape == (480, 1920, 3) and frame_count > 600 and fps == 20
    print(json.dumps(dict(video=str(destination), frames=frame_count, fps=fps, duration_seconds=frame_count/fps,
                         timing='Resampled to 20 Hz from host receive timestamps; original messages retained in rosbag.')))


if __name__ == '__main__':
    main()
