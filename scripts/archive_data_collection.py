#!/usr/bin/env python3
"""Copy teaching sources read-only through already-authenticated SSH masters.

Run from this repository. No authentication material or raw datasets are copied;
no service, recorder, camera, relay or arm operation is started by this exporter.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / 'data_collection'
EXCLUDES = ['.git/', '__pycache__/', '.pytest_cache/', '.venv*/', 'node_modules/',
            'build/', 'build_ros/', 'install/', 'install_ros/', 'log/', 'logs/',
            '_rokae_log_/', 'artifacts/', 'reports/', 'dist/', 'dataset/',
            '*backup*', '*before*', '*.bak', '*.log', '*.jsonl', '*.pyc',
            '*.so', '*.so.*', '*.a', '*.o', '*.whl', '*.deb', '*.engine', '*.onnx',
            '*.safetensors', '*.pt', '*.pth', '*.npz', '*.npy', '*.db3*',
            '*.avi', '*.mp4', '*.zip', '*.tar.gz', '*.tgz', '*.pdf',
            '.env', '.env.*', '*.pem', '*.key', 'CONTINUATION.md', 'REMOTE_EDIT_GUIDE.md']
DEVICES = {
    'jetson': {
        'host': 'jetson@172.16.12.95',
        'socket': '/tmp/foxconn_teach_jetson_20261117.sock',
        'home': '/home/jetson',
        'paths': ['AIDE_robot_main/AGENTS.md', 'AIDE_robot_main/startup/ros_environment.sh',
                  'AIDE_robot_main/startup/ros_environment_debug.sh',
                  'AIDE_robot_main/software/vla/act/xcore_act/',
                  'AIDE_robot_main/hardware/robot_vision_ws/',
                  'AIDE_robot_main/hardware/arm/xCoreROS/src/',
                  'AIDE_robot_main/hardware/arm/xCoreROS/tools/',
                  'AIDE_robot_main/hardware/arm/xCoreROS/README.md',
                  'AIDE_robot_main/hardware/arm/xCoreROS/ARCHITECTURE.md',
                  'AIDE_robot_main/hardware/arm/xCoreROS/SDK_NODES.md',
                  'AIDE_robot_main/hardware/USR-M050/'],
    },
    'umie': {
        'host': 'umie@172.16.12.45',
        'socket': '/tmp/foxconn_teach_umie_20261117.sock',
        'home': '/home/umie',
        'paths': ['aide_pc_wrist_cameras/', '.local/share/robot-camera-streams/',
                  '.local/share/robot-relay-bridge/'],
    },
}


def run(args, **kw):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kw).stdout


def main():
    assert run(['git', '-C', str(REPO), 'branch', '--show-current']).strip() == 'jetson-act-qwen-20261117'
    records = []
    for device, spec in DEVICES.items():
        base = DEST / device
        for relative in spec['paths']:
            remote = spec['home'] + '/' + relative
            target = base / relative.rstrip('/')
            target = target if relative.endswith('/') else target.parent
            target.mkdir(parents=True, exist_ok=True)
            args = ['rsync', '-a', '-e', f'ssh -S {spec["socket"]} -o BatchMode=yes']
            args += ['--exclude=' + pattern for pattern in EXCLUDES]
            args += [spec['host'] + ':' + remote, str(target) + '/']
            run(args)
            print(json.dumps({'copied': device + '/' + relative}), flush=True)
        files = sorted(p for p in base.rglob('*') if p.is_file() and not p.is_symlink())
        relative_files = [str(p.relative_to(base)) for p in files]
        code = """import hashlib,json,sys
from pathlib import Path
base=Path(sys.argv[1]);out={}
for name in json.load(sys.stdin):
 p=base/name
 out[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(out))
"""
        remote_command = '/usr/bin/python3 -c ' + shlex.quote(code) + ' ' + shlex.quote(spec['home'])
        remote_index = json.loads(run(['ssh', '-S', spec['socket'], '-o', 'BatchMode=yes',
                                      spec['host'], remote_command], input=json.dumps(relative_files)))
        for path, name in zip(files, relative_files):
            payload = path.read_bytes()
            assert not payload.startswith(b'\x7fELF'), f'Binary executable in source: {path}'
            assert len(payload) < 50 * 1024 * 1024, f'Large source file: {path}'
            identity = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            assert identity == remote_index[name], f'Source changed during capture: {device}/{name}'
            records.append({'device': device, 'source': spec['home'] + '/' + name,
                            'export': str(path.relative_to(REPO)), **identity})
    manifest = DEST / 'manifests'
    manifest.mkdir(exist_ok=True)
    (manifest / 'source-files.json').write_text(json.dumps(records, indent=2) + '\n')
    report = {'captured_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'parent_inference_commit': '9a4b58123289864564c943dd0ea513d0eabbf6e3',
              'scope': 'teaching/collection sources, their drivers, camera and relay support',
              'source_files': len(records), 'source_bytes': sum(row['bytes'] for row in records),
              'all_source_files_match_device_sha256': True,
              'services_started_or_stopped': False, 'robot_or_suction_commands_sent': False,
              'omitted': ['raw datasets', 'model weights', 'virtual environments', 'build/install output',
                          'old backups and logs', 'SSH/GitHub credentials'],
              'restore_layout': {'jetson': '/home/jetson/', 'umie': '/home/umie/'}}
    (manifest / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
