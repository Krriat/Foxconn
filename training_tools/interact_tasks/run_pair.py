#!/usr/bin/env python3
"""Persistent two-task training job. Never installs policies or calls ROS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def deployment_snapshot():
    files = [
        ROOT / 'model_outputs/act_suction_20261111_213453/checkpoints/014000/pretrained_model/model.safetensors',
        ROOT / 'model_outputs/act_near_success_20261112/pretrained_model/model.safetensors',
        ROOT / 'model_outputs/task_brain_lora_20261112/adapter/adapter_model.safetensors',
        Path.home() / '.local/share/robot-policy-streams/config.yaml',
    ]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=30000)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = deployment_snapshot()
    write(output / 'deployed_before.json', before)
    started = time.time()
    results = {}
    try:
        for stage in ('far', 'near'):
            command = ['bash', str(ROOT / 'training_tools/interact_tasks/run.sh'),
                       '--stage', stage, '--output', str(output / stage), '--steps', str(args.steps)]
            write(output / 'status.json', dict(state='training', stage=stage, started_unix=started,
                command=command, completed_stages=list(results)))
            with (output / f'{stage}.log').open('w') as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output / stage / 'result.json').read_text())
            if result['state'] != 'complete' or not result['inference']['checkpoint_reload_matches']:
                raise ValueError(f'{stage}: training/evaluation/reload did not complete')
            results[stage] = result
        after = deployment_snapshot()
        write(output / 'deployed_after.json', after)
        if after != before:
            raise ValueError('Deployed weights or local robot config changed during training')
        report = dict(state='complete', elapsed_seconds=time.time()-started,
            deployment_unchanged=True, robot_commands_sent=0, results=results)
        write(output / 'report.json', report)
        write(output / 'status.json', dict(state='complete', completed_stages=list(results),
            report=str(output / 'report.json')))
        print(json.dumps({s: {k: r[k] for k in ('executed_steps', 'best_step', 'pretrained_model',
            'baseline_joint_mae_deg', 'selected_joint_mae_deg', 'reduction_percent')} for s, r in results.items()}), flush=True)
    except BaseException as error:
        write(output / 'status.json', dict(state='failed', error=repr(error), completed_stages=list(results)))
        raise


if __name__ == '__main__':
    main()
