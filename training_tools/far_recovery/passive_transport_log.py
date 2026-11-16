"""Bounded, read-only ROS receipt log for comparison on PC and Jetson."""
import argparse
import json
import socket
import time
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=1200)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node('passive_policy_transport_log')
    previous = {}
    with args.output.open('x', buffering=1) as log:
        def write(topic, **data):
            wall, mono = time.time_ns(), time.monotonic_ns()
            before = previous.get(topic)
            previous[topic] = mono
            record = dict(receive_time_ns=wall, monotonic_ns=mono, topic=topic,
                          gap_ms=None if before is None else (mono-before)/1e6,
                          **data)
            log.write(json.dumps(record, separators=(',', ':'))+'\n')

        def stamped(topic, message):
            write(topic, source_time_ns=message.header.stamp.sec*10**9+
                  message.header.stamp.nanosec)

        def policy(topic, message):
            try:
                v = json.loads(message.data)
                keys = ('phase', 'reason', 'run_id', 'sequence', 'send_time_ns',
                        'observation_time_ns', 'inference_ms', 'accepted_actions',
                        'last_action_age_ms', 'task_sequence', 'pause_reason',
                        'input_receive_age_ms', 'ready', 'active_checkpoint_sha256')
                write(topic, **{k: v[k] for k in keys if k in v})
            except (ValueError, TypeError) as error:
                write(topic, decode_error=str(error))

        def diagnostics(message):
            write('/robot/hardware_state/diagnostics', status=[dict(
                name=s.name, level=(int.from_bytes(s.level, 'little')
                                   if isinstance(s.level, bytes) else int(s.level)),
                message=s.message,
                values={v.key:v.value for v in s.values}) for s in message.status])

        subs = []
        for side in ('left', 'right'):
            topic = f'/robot/suction/{side}/state'
            subs.append(node.create_subscription(Bool, topic,
                lambda m, t=topic: write(t, data=m.data), qos_profile_sensor_data))
            topic = f'/robot/upperlimb/joint_states/{side}_arm'
            subs.append(node.create_subscription(JointState, topic,
                lambda m, t=topic: stamped(t, m), qos_profile_sensor_data))
        topic = '/robot/sensors/cameras/head/color/image_raw/compressed'
        subs.append(node.create_subscription(CompressedImage, topic,
            lambda m: stamped(topic, m), qos_profile_sensor_data))
        for name in ('status', 'action_chunk', 'inference_status'):
            t = f'/robot/policy/{name}'
            subs.append(node.create_subscription(String, t,
                lambda m, t=t: policy(t, m), qos_profile_sensor_data))
        subs.append(node.create_subscription(DiagnosticArray,
            '/robot/hardware_state/diagnostics', diagnostics, qos_profile_sensor_data))
        write('logger_started', host=socket.gethostname(), seconds=args.seconds,
              publishes=False, calls_services=False)
        end = time.monotonic()+args.seconds
        try:
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=.2)
        finally:
            write('logger_finished', topics_seen=list(previous))
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
