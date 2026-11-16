#!/usr/bin/env python3
"""Explicitly start one existing policy round, observe it, and always request stop."""
import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', required=True)
    parser.add_argument('--service', choices=('start', 'start_near'), default='start')
    parser.add_argument('--running-seconds', type=float, default=35.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.running_seconds <= 90:
        parser.error('running-seconds must be in (0, 90]')
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('bounded_live_policy_check')
    latest, received, cameras, suction_feedback = {}, {}, {}, {}
    log = (args.output/'supervisor.jsonl').open('w', buffering=1)
    def status(key, msg):
        latest[key], received[key] = json.loads(msg.data), time.monotonic()
        log.write(json.dumps(dict(time_ns=time.time_ns(), topic=key, data=latest[key]))+'\n')
    subs = [node.create_subscription(String, '/robot/policy/'+key,
            lambda msg, key=key: status(key, msg), qos_profile_sensor_data)
            for key in ('status', 'inference_status', 'trajectory_status')]
    # Keep the most recent complete JPEG available for visual supervision.
    saved = {}
    def image_callback(key, msg):
        now = time.monotonic()
        cameras[key] = now
        if now-saved.get(key, 0) > .4:
            temporary = args.output/f'{key}.tmp'
            temporary.write_bytes(bytes(msg.data))
            temporary.replace(args.output/f'{key}.jpg')
            saved[key] = now
    subs += [node.create_subscription(CompressedImage,
             f'/robot/sensors/cameras/{key}/color/image_raw/compressed',
             lambda msg, key=key: image_callback(key, msg), qos_profile_sensor_data)
             for key in ('head', 'left_wrist', 'right_wrist')]
    subs += [node.create_subscription(Bool, f'/robot/suction/{side}/state',
             lambda msg, side=side: suction_feedback.update({side:(msg.data,time.monotonic())}),
             qos_profile_sensor_data) for side in ('left','right')]
    clients = {key: node.create_client(Trigger, '/robot/policy/'+(args.service if key=='start' else key))
               for key in ('check', 'start', 'stop')}
    def spin_until(test, seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            if test(): return True
        return False
    def call(key, require_success=True):
        if not clients[key].wait_for_service(timeout_sec=3):
            raise RuntimeError(key+' service unavailable')
        future = clients[key].call_async(Trigger.Request())
        if not spin_until(future.done, 5): raise RuntimeError(key+' service timeout')
        result = future.result()
        print(json.dumps(dict(service=key, success=result.success, message=result.message)), flush=True)
        if require_success and not result.success: raise RuntimeError(result.message)
        return result
    requested, reason = False, ''
    try:
        if not spin_until(lambda: len(cameras)==3 and all(k in latest for k in
                          ('status','inference_status','trajectory_status')), 10):
            raise RuntimeError('Missing camera or status input')
        if not spin_until(lambda: latest['inference_status'].get('ready') and
                          latest['trajectory_status'].get('ready'), 15):
            raise RuntimeError('Inference warm-up or trajectory readiness timed out')
        if latest['status']['phase'] != 'IDLE': raise RuntimeError('Executor is not idle')
        if (len(suction_feedback)!=2 or any(on or time.monotonic()-stamp>.3
                                           for on,stamp in suction_feedback.values())):
            raise RuntimeError('Suction must be verified off before homing')
        if any(time.monotonic()-v > .3 for v in cameras.values()): raise RuntimeError('Camera stale')
        # The executor on the Jetson may receive the first ready heartbeat a
        # little later than this observer. Retry only this read-only check.
        ready_deadline = time.monotonic() + 10
        while True:
            checked = call('check', require_success=False)
            if checked.success: break
            if checked.message != 'Inference node is not ready' or time.monotonic() >= ready_deadline:
                raise RuntimeError(checked.message)
            retry_at = time.monotonic() + .5
            spin_until(lambda: time.monotonic() >= retry_at, 1)
        requested = True
        call('start')
        began, active, next_print, saw_active = time.monotonic(), None, 0., False
        while True:
            rclpy.spin_once(node, timeout_sec=.02)
            now = time.monotonic()
            state = latest['status']
            phase = state.get('phase')
            if now-received['status'] > 1: raise RuntimeError('Status stream lost')
            if any(now-v > .6 for v in cameras.values()): raise RuntimeError('Camera stream lost')
            if phase == 'FAULT': raise RuntimeError(state.get('reason', 'Executor fault'))
            if phase == 'RUNNING':
                saw_active = True
                if active is None: active = now
            if saw_active and phase in ('STOPPING', 'IDLE'):
                reason = 'Executor stopped: '+state.get('reason',''); break
            if active is not None and now-active >= args.running_seconds:
                reason = 'Bounded diagnostic duration reached'; break
            if now-began > 85+args.running_seconds: raise RuntimeError('Startup/round deadline reached')
            if (args.output/'STOP').exists():
                reason = 'Observer requested stop'; break
            if now >= next_print:
                print(json.dumps(dict(phase=phase, elapsed=None if active is None else round(now-active,2),
                      run_id=state.get('run_id'),
                      suction=(state.get('suction_io') or {}).get('verified'),
                      actions=state.get('accepted_actions'), reason=state.get('reason'))), flush=True)
                next_print = now+2
    except BaseException as exc:
        reason = type(exc).__name__+': '+str(exc)
        raise
    finally:
        if requested:
            try:
                call('stop')
                if not spin_until(lambda: latest['status']['phase'] in ('IDLE','FAULT'), 12):
                    raise RuntimeError('STOP NOT CONFIRMED')
            except Exception as exc:
                reason += '; STOP ERROR: '+str(exc)
        report = dict(reason=reason, latest=latest)
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
