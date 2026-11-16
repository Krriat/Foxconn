#!/usr/bin/env python3
"""Read-only live state capture, then offline camera-profile input comparison."""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
import yaml

ROOT = Path(__file__).resolve().parents[2]
APP = Path.home()/'.local/share/robot-policy-streams'
sys.path.insert(0, str(APP))
from inference_node import load_runtime

OUT = ROOT/'training_verification/restore_far_20261112/policy-camera-profile-probe-20261112'


def main():
    config = yaml.safe_load((APP/'config.yaml').read_text())
    runtime, predict, reset, sha = load_runtime(config)
    rclpy.init()
    node = runtime.Inputs()
    deadline = time.monotonic()+10
    observation = None
    try:
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.05)
            observation, error = node.snapshot(-1,.10,.30)
            if observation is not None:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if observation is None:
        raise RuntimeError(error)
    # Only the wrist pixels change. Head, joints, suction and model are fixed.
    np.savez_compressed(OUT/'fixed_context.npz',state=observation['state'],head=observation['images'][0])
    rows, results = [], {}
    for label in ('color_before','color_depth','color_after'):
        images = [observation['images'][0]]
        for camera in ('left_wrist','right_wrist'):
            bgr = cv2.imread(str(OUT/f'{label}_{camera}.png'))
            ok,jpeg = cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,80])
            assert ok
            images.append(cv2.cvtColor(cv2.imdecode(jpeg,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB))
        reset()
        prediction = predict(dict(state=observation['state'],images=images))
        results[label] = prediction
        rows.append(dict(label=label,first_joint_offset_deg=np.rad2deg(prediction[0,:14]-observation['state'][:14]).tolist(),
            chunk_joint_motion_deg=np.rad2deg(prediction[-1,:14]-prediction[0,:14]).tolist(),
            left_J4_start_end_deg=np.rad2deg(prediction[[0,-1],3]).tolist(),
            left_suction_start_end=prediction[[0,-1],14].tolist(),
            right_suction_start_end=prediction[[0,-1],15].tolist()))
        print(json.dumps(rows[-1]),flush=True)
    report=dict(weights_sha256=sha,state=observation['state'].tolist(),cases=rows,
        max_joint_difference_deg=float(np.rad2deg(abs(results['color_before'][:,:14]-results['color_depth'][:,:14])).max()),
        color_retest_difference_deg=float(np.rad2deg(abs(results['color_before'][:,:14]-results['color_after'][:,:14])).max()),
        limitation='Conditional predictions at one fixed pose; does not establish closed-loop success.')
    (OUT/'predictions.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'predictions.npz',**results)


if __name__=='__main__':
    main()
