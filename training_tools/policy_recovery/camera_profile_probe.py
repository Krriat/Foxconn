#!/usr/bin/env python3
"""Compare D405 capture profiles while the policy is idle; restore camera service."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


ROOT = Path.home()/'.local/share/robot-camera-streams'
OUT = Path.home()/'policy-camera-profile-probe-20261112'


def capture(camera, label, depth):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(camera['serial']))
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    if depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    samples = []
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        options = {}
        for sensor in profile.get_device().query_sensors():
            options[sensor.get_info(rs.camera_info.name)] = {
                str(o):sensor.get_option(o) for o in sensor.get_supported_options()
                if any(word in str(o).lower() for word in ('exposure', 'gain', 'white_balance'))}
        for index in range(180):
            frame = pipeline.wait_for_frames(3000).get_color_frame()
            if not frame:
                continue
            if index % 30 == 0 or index >= 150:
                bgr = np.asanyarray(frame.get_data())
                samples.append(dict(index=index, mean_rgb=bgr.mean(axis=(0,1))[::-1].tolist(),
                    metadata={str(field):frame.get_frame_metadata(field)
                        for field in (rs.frame_metadata_value.actual_exposure,
                                      rs.frame_metadata_value.gain_level,
                                      rs.frame_metadata_value.auto_exposure,
                                      rs.frame_metadata_value.actual_fps,
                                      rs.frame_metadata_value.white_balance)
                        if frame.supports_frame_metadata(field)}))
            if index == 179:
                cv2.imwrite(str(OUT/f'{label}_{camera["name"]}.png'), np.asanyarray(frame.get_data()))
        return dict(label=label, camera=camera['name'], depth=depth, options=options,
                    samples=samples)
    finally:
        if started:
            pipeline.stop()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    cameras = yaml.safe_load((ROOT/'wrists.yaml').read_text())['cameras']
    rows = []
    subprocess.run(['systemctl','--user','stop','robot-wrist-cameras.service'],check=True)
    try:
        for label, depth in [('color_before',False), ('color_depth',True), ('color_after',False)]:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(capture,camera,label,depth) for camera in cameras]
                for future in futures:
                    row = future.result()
                    rows.append(row)
                    print(json.dumps(dict(label=label,camera=row['camera'],last=row['samples'][-1])),flush=True)
            (OUT/'report.json').write_text(json.dumps(rows,indent=2)+'\n')
    finally:
        subprocess.run([str(ROOT/'start.sh'),'wrists'],check=True)


if __name__ == '__main__':
    main()
