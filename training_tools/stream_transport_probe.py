"""Passive ROS observations for matching source and destination frame timestamps."""
import argparse
import json
import math
import socket
import time
from pathlib import Path

import rclpy
from rclpy.qos import qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=180)
    p.add_argument('--cameras', nargs='*', default=['head', 'left_wrist', 'right_wrist'])
    p.add_argument('--arms', action='store_true')
    p.add_argument('--decode', action='store_true')
    a = p.parse_args()
    if a.decode:
        import cv2
        import numpy as np
    rclpy.init()
    node = rclpy.create_node('passive_stream_transport_probe')
    subscriptions = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', buffering=1) as log:
        def write(topic, **fields):
            log.write(json.dumps(dict(topic=topic, wall_ns=time.time_ns(),
                                      mono_ns=time.monotonic_ns(), **fields), separators=(',', ':'))+'\n')

        def frame(topic, msg):
            wall, mono = time.time_ns(), time.monotonic_ns()
            fields = dict(stamp_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                          frame_id=msg.header.frame_id)
            if isinstance(msg, CompressedImage):
                fields['bytes'] = len(msg.data)
                if a.decode:
                    decoded = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                    fields['decode_ok'] = decoded is not None
                    fields['shape'] = list(decoded.shape) if decoded is not None else None
            else:
                fields['positions'] = list(msg.position)
                fields['valid'] = len(msg.position) == 7 and all(math.isfinite(x) for x in msg.position)
            log.write(json.dumps(dict(topic=topic, wall_ns=wall, mono_ns=mono, **fields), separators=(',', ':'))+'\n')

        def payload(topic, msg):
            try:
                write(topic, data=json.loads(msg.data))
            except ValueError:
                write(topic, data=msg.data)

        def diagnostics(msg):
            write('/robot/hardware_state/diagnostics', data=[dict(
                name=s.name, level=int.from_bytes(s.level, 'little') if isinstance(s.level, bytes) else int(s.level),
                message=s.message, values={v.key:v.value for v in s.values}) for s in msg.status])

        for name in a.cameras:
            topic = f'/robot/sensors/cameras/{name}/color/image_raw/compressed'
            subscriptions.append(node.create_subscription(CompressedImage, topic,
                lambda msg, t=topic: frame(t, msg), qos_profile_sensor_data))
        for name in ('robot_head_camera', 'robot_wrist_cameras'):
            topic = '/robot/sensors/cameras/status/'+name
            subscriptions.append(node.create_subscription(String, topic,
                lambda msg, t=topic: payload(t, msg), qos_profile_sensor_data))
        if a.arms:
            for side in ('left', 'right'):
                topic = f'/robot/upperlimb/joint_states/{side}_arm'
                subscriptions.append(node.create_subscription(JointState, topic,
                    lambda msg, t=topic: frame(t, msg), qos_profile_sensor_data))
                topic = f'/robot/suction/{side}/state'
                subscriptions.append(node.create_subscription(Bool, topic,
                    lambda msg, t=topic: write(t, data=msg.data), qos_profile_sensor_data))
            subscriptions.append(node.create_subscription(DiagnosticArray, '/robot/hardware_state/diagnostics',
                diagnostics, qos_profile_sensor_data))
            subscriptions.append(node.create_subscription(String, '/robot/policy/status',
                lambda msg: payload('/robot/policy/status', msg), qos_profile_sensor_data))
        write('started', host=socket.gethostname(), duration=a.seconds, cameras=a.cameras,
              arms=a.arms, decode=a.decode, passive=True)
        end = time.monotonic()+a.seconds
        try:
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=.1)
        finally:
            write('finished')
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
