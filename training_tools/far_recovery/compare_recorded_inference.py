"""Replay saved successful observations through the installed inference loader.

No ROS initialization, publishers, services, or hardware calls are made.
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

APP = Path.home() / '.local/share/robot-policy-streams'
sys.path.insert(0, str(APP))
from inference_node import load_runtime

p = argparse.ArgumentParser()
p.add_argument('--recording', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
events = [json.loads(line) for line in (a.recording/'executor_events.jsonl').open()]
chunks = [r for r in events if r['event'] == 'chunk']
start = chunks[0]['observation_time_ns']
chosen = [min(chunks, key=lambda c: abs((c['observation_time_ns']-start)/1e9-t))
          for t in (0, 1, 3, 5, 10, 15, 20, 30, 40, 50)]
cameras = ('head', 'left_wrist', 'right_wrist')
required = {key: {c['source_stamps_ns'][key] for c in chosen} for key in cameras}
images = {key: {} for key in cameras}
db = sqlite3.connect('file:'+str(next((a.recording/'rosbag').glob('*.db3')))+'?mode=ro', uri=True)
for camera in cameras:
    for raw, in db.execute('SELECT m.data FROM messages m JOIN topics t ON t.id=m.topic_id WHERE t.name=?',
                          (f'/robot/sensors/cameras/{camera}/color/image_raw/compressed',)):
        msg = deserialize_message(raw, CompressedImage)
        stamp = int((msg.header.stamp.sec + msg.header.stamp.nanosec/1e9)*1e9)
        if stamp in required[camera]:
            images[camera][stamp] = cv2.cvtColor(cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
db.close()
cfg = yaml.safe_load((APP/'config.yaml').read_text())
sha = hashlib.sha256((a.checkpoint/'model.safetensors').read_bytes()).hexdigest()
assert all(c['checkpoint_sha256'] == sha for c in chosen)
cfg['checkpoint_sha256']['act'] = sha
_, predict, reset, _ = load_runtime(cfg, a.checkpoint)
rows = []
saved_images, saved_states, saved_actions = [], [], []
for i, c in enumerate(chosen):
    if not all(c['source_stamps_ns'][key] in images[key] for key in cameras):
        raise RuntimeError('Exact source images missing; refusing approximate replay')
    rgb = [images[key][c['source_stamps_ns'][key]] for key in cameras]
    state = np.asarray(c['observation_state'], np.float32)
    expected = np.asarray(c['actions'], np.float32)
    reset()
    predicted = predict(dict(images=rgb, state=state))
    rows.append(dict(seconds=(c['observation_time_ns']-start)/1e9,
                     joint_max_diff_deg=float(np.rad2deg(abs(predicted[:,:14]-expected[:,:14])).max()),
                     suction_max_diff=float(abs(predicted[:,14:]-expected[:,14:]).max()),
                     original_suction=expected[2,14:].tolist(), replay_suction=predicted[2,14:].tolist()))
    saved_images.append(rgb); saved_states.append(state); saved_actions.append(expected)
    if i == 0:
        for key, image in zip(cameras, rgb):
            cv2.imwrite(str(a.output/f'success_home_{key}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
np.savez_compressed(a.output/'successful_inputs.npz', images=np.array(saved_images), state=np.array(saved_states),
                    actions=np.array(saved_actions), seconds=np.array([r['seconds'] for r in rows]))
result = dict(checkpoint_sha256=sha, exact_recorded_images=True, samples=len(rows),
              joint_max_diff_deg=max(r['joint_max_diff_deg'] for r in rows),
              suction_max_diff=max(r['suction_max_diff'] for r in rows), rows=rows,
              scope='Numerical reproduction with historical observations, not a new closed-loop success rate')
(a.output/'offline_replay.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
