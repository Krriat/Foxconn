#!/usr/bin/env python3
"""One supervised near-only candidate round, with automatic original restoration.

Run under the system ROS Python with ROS_DOMAIN_ID=1. --execute is required for
service/config changes. Preparing artifacts and gate checks makes no changes.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
APP = Path.home() / '.local/share/robot-policy-streams'
REMOTE_CONFIG = '/home/jetson/.local/share/robot-policy-streams/config.yaml'
SSH = ['ssh', '-o', 'ControlPath=/tmp/act_restore_far.sock', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=5', 'jetson@172.16.12.95']
UNIT = 'robot-policy-inference.service'
REMOTE_UNIT = 'robot-policy-executor.service'


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, capture_output=True, timeout=25, **kwargs).stdout


def status():
    import rclpy
    from std_msgs.msg import Bool, String
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init()
    node = rclpy.create_node('candidate_round_status_check')
    received, suction = [], {}
    subscription = node.create_subscription(String, '/robot/policy/status',
        lambda message: received.append(json.loads(message.data)), qos_profile_sensor_data)
    suction_subscriptions = [node.create_subscription(Bool, f'/robot/suction/{side}/state',
        lambda message, side=side: suction.update({side: (bool(message.data), time.monotonic())}),
        qos_profile_sensor_data) for side in ('left', 'right')]
    end = time.monotonic() + 4
    try:
        while time.monotonic() < end and not (received and len(suction) == 2):
            rclpy.spin_once(node, timeout_sec=.05)
        if not received:
            raise RuntimeError('No fresh executor status')
        if len(suction) != 2 or any(time.monotonic()-stamp > .3 for _, stamp in suction.values()):
            raise RuntimeError('No fresh suction readback')
        return dict(received[-1], readback_suction=[suction[side][0] for side in ('left', 'right')])
    finally:
        node.destroy_node()
        rclpy.shutdown()


def request_stop():
    # Use fresh DDS discovery instead of the CLI daemon's cached service graph
    # after restarting the executor.
    import rclpy
    from std_srvs.srv import Trigger
    rclpy.init()
    node = rclpy.create_node('candidate_round_stop')
    try:
        client = node.create_client(Trigger, '/robot/policy/stop')
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError('Stop service unavailable')
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=5)
        if not future.done() or not future.result().success:
            raise RuntimeError('Stop service failed or timed out')
        return future.result().message
    finally:
        node.destroy_node()
        rclpy.shutdown()


def check_gate(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Inference')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'active_context')
    scope = dict(time=time, ACTIVE_PHASES=('WAITING_ACTION', 'STARTING_SERVO', 'RUNNING', 'PREVIEW'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<candidate_gate>', 'exec'), scope)
    checks = []
    for stage, phase, age, result, expected in (
            ('near', 'RUNNING', 0, 'running', True), ('far', 'RUNNING', 0, 'running', False),
            (None, 'RUNNING', 0, 'running', False), ('near', 'IDLE', 0, 'running', False),
            ('near', 'RUNNING', 1, 'running', False), ('near', 'RUNNING', 0, 'cancelled', False)):
        context = dict(phase=phase, model='act', task_sequence=dict(stage=stage, result=result))
        dummy = SimpleNamespace(lock=threading.Lock(), executor_context=context,
                                context_received=time.monotonic()-age, config={'model': 'act'})
        assert (scope['active_context'](dummy) is not None) == expected
        checks.append(dict(stage=stage, phase=phase, age=age, result=result, allowed=expected))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    assert checkpoint.is_relative_to(ROOT / 'model_outputs')
    sha = hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_node = (APP / 'inference_node.py').read_text()
    original_config = (APP / 'config.yaml').read_text()
    original_remote = command(SSH + ['cat ' + shlex.quote(REMOTE_CONFIG)])
    old_path = 'checkpoint = root / "model_outputs/act_suction_20261111_213453/checkpoints/best/pretrained_model"'
    assert original_node.count(old_path) == 1
    candidate_node = original_node.replace(old_path, f'checkpoint = Path({str(checkpoint)!r})')
    needle = '        return context\n'
    assert candidate_node.count(needle) == 1
    candidate_node = candidate_node.replace(needle,
        '        sequence_context = context.get("task_sequence") or {}\n'
        '        if sequence_context.get("stage") != "near" or sequence_context.get("result") != "running":\n'
        '            return None\n' + needle)
    checks = check_gate(candidate_node)
    configs = []
    for source in (original_config, original_remote):
        config = yaml.safe_load(source)
        assert config['model'] == 'act'
        config['checkpoint_sha256']['act'] = sha
        configs.append(yaml.safe_dump(config, sort_keys=False))
    for filename, data in (('inference_before.py', original_node), ('config_before.yaml', original_config),
                           ('remote_config_before.yaml', original_remote), ('inference_candidate.py', candidate_node),
                           ('config_candidate.yaml', configs[0]), ('remote_config_candidate.yaml', configs[1])):
        path = out / filename
        if path.exists() and path.read_text() != data:
            raise RuntimeError(f'Refusing to overwrite a different backup: {path}')
        path.write_text(data)
    (out / 'candidate_manifest.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), sha256=sha,
        gate_tests=checks, original_config_sha256=hashlib.sha256(original_config.encode()).hexdigest()), indent=2)+'\n')
    print(json.dumps(dict(prepared=True, checkpoint=str(checkpoint), sha256=sha, gate_checks=len(checks))), flush=True)
    if not args.execute:
        return
    before = status()
    if before.get('phase') != 'IDLE' or before['readback_suction'] != [False, False]:
        raise RuntimeError('Requires fresh IDLE and both suction channels off')
    mutated = False
    try:
        mutated = True
        command(['systemctl', '--user', 'stop', UNIT])
        command(SSH + [f'systemctl --user stop {REMOTE_UNIT}'])
        (APP / 'inference_node.py').write_text(candidate_node)
        (APP / 'config.yaml').write_text(configs[0])
        command(SSH + ['cat > ' + shlex.quote(REMOTE_CONFIG)], input=configs[1])
        command(SSH + ['~/.local/share/robot-policy-streams/start.sh executor'])
        command([str(APP / 'start.sh'), 'inference'])
        # Service readiness is also checked by the supervisor before it starts motion.
        with (out / 'run.log').open('w') as log:
            result = subprocess.run(['/usr/bin/python3', '-u', str(ROOT / 'training_tools/bounded_live_policy_check.py'),
                '--start', '--service', 'start_near', '--running-seconds', '45', '--output', str(out)],
                stdout=log, stderr=subprocess.STDOUT, timeout=145)
        if result.returncode:
            raise RuntimeError(f'Supervisor returned {result.returncode}')
    finally:
        if mutated:
            try:
                settled = status()
                if settled.get('phase') != 'IDLE' or settled['readback_suction'] != [False, False]:
                    request_stop()
                    # Allow the existing controlled stop to settle before service replacement.
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline and (settled.get('phase') != 'IDLE' or
                                                           settled['readback_suction'] != [False, False]):
                        settled = status()
                    if settled.get('phase') != 'IDLE' or settled['readback_suction'] != [False, False]:
                        raise RuntimeError('Stop did not settle to IDLE; keep executor running for stop handling')
                command(['systemctl', '--user', 'stop', UNIT])
                command(SSH + [f'systemctl --user stop {REMOTE_UNIT}'])
                (APP / 'inference_node.py').write_text(original_node)
                (APP / 'config.yaml').write_text(original_config)
                command(SSH + ['cat > ' + shlex.quote(REMOTE_CONFIG)], input=original_remote)
                command(SSH + ['~/.local/share/robot-policy-streams/start.sh executor'])
                command([str(APP / 'start.sh'), 'inference'])
                (out / 'restored.json').write_text(json.dumps(dict(original_restored=True,
                    original_config_sha256=hashlib.sha256((APP/'config.yaml').read_bytes()).hexdigest(),
                    timestamp_ns=time.time_ns()), indent=2)+'\n')
                print('Original ACT configuration and inference code restored', flush=True)
            except BaseException as exc:
                (out / 'RESTORE_REQUIRED.txt').write_text(repr(exc)+'\n')
                raise


if __name__ == '__main__':
    main()
