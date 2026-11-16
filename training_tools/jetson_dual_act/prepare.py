"""Package both successful ACTs and the exact original network; offline only."""
import hashlib
import json
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
OUT = ROOT / 'training_verification/jetson_dual_act_20261113/bundle'
REMOTE = '/home/jetson/.local/share/robot-policy-streams/jetson_act_runtime_20261113'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    network = ROOT / 'training_verification/jetson_bundle_20261112'
    manifest = json.loads((network / 'manifest.json').read_text())
    assert hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest() == manifest['source_sha256']
    shutil.copy2(network / 'act_core.py', OUT / 'act_core.py')
    cfg = yaml.safe_load((Path.home() / '.local/share/robot-policy-streams/config.yaml').read_text())
    for stage, spec in cfg['act_stage_checkpoints'].items():
        src = Path(spec['path'])
        assert hashlib.sha256((src/'model.safetensors').read_bytes()).hexdigest() == spec['sha256']
        destination = OUT / 'models' / stage
        shutil.copytree(src, destination)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()}
        (destination/'artifact_checksums.json').write_text(json.dumps(hashes, indent=2)+'\n')
        spec['path'] = REMOTE + '/models/' + stage
    for name in ('inference_node.py', 'executor_node.py', 'protocol.py', 'stage_policy.py', 'brain_protocol.py',
                 'joint_anchor.py', 'suction_io.py', 'test_executor_ros.py'):
        shutil.copy2(ROOT / 'training_tools/two_stage_policy' / name, OUT / name)
    for name in ('jetson_policy_runtime.py', 'jetson_inference_bootstrap.py'):
        shutil.copy2(SOURCE / name, OUT / name)
    cfg.update(inference_runtime='jetson_act', inference_bundle=REMOTE, enable_motion=True)
    (OUT/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    # The saved fixture has real three-camera RGBs and a measured 16D state.
    shutil.copy2(network/'sample.npz', OUT/'sample.npz')
    (OUT/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    main()
