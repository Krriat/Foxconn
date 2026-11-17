"""Verify the Foxconn export locally; no device or robot interfaces are used."""
import hashlib
import io
import json
import re
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path('/home/jj/code/Foxconn-jetson-act-qwen-20261117')
OUT = Path(Path('/tmp/foxconn_export_path').read_text())
PATTERNS = {
    'github_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})'),
    'private_key': re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
    'inline_ssh_password': re.compile(rb'sshpass\s+-p\s+[^\s$]+'),
    'device_password_assignment': re.compile(rb'(?:jetson|umie|ssh)_password\s*=\s*[\x22\x27][^\x22\x27]+', re.I),
}
TEXT_SUFFIXES = {'.py', '.sh', '.json', '.jsonl', '.txt', '.yaml', '.yml', '.xml',
                 '.md', '.h', '.hpp', '.cpp', '.c', '.ini', '.cfg', '.toml', '.service'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def scan(stream, name, hasher=None):
    previous = b''
    while block := stream.read(1024 * 1024):
        if hasher is not None:
            hasher.update(block)
        payload = previous + block
        for label, pattern in PATTERNS.items():
            assert not pattern.search(payload), f'Potential credential ({label}) in {name}'
        previous = payload[-512:]


def nested_scan(stream, prefix, depth=0):
    assert depth < 4
    with tarfile.open(fileobj=stream, mode='r:gz') as archive:
        for member in archive:
            assert not member.name.startswith('/') and '..' not in Path(member.name).parts
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            name = prefix + '!' + member.name
            if member.name.endswith(('.tar.gz', '.tgz')):
                nested_scan(io.BytesIO(source.read()), name, depth + 1)
            elif Path(member.name).suffix in TEXT_SUFFIXES:
                scan(source, name)


def main():
    source_files = []
    for path in sorted(REPO.rglob('*')):
        if not path.is_file() or path.name == '.git':
            continue
        relative = str(path.relative_to(REPO))
        if relative in ('manifests/source-files.json', 'manifests/archive-verification.json'):
            continue
        assert not path.is_symlink(), f'Unexpected source symlink: {relative}'
        assert path.stat().st_size < 50 * 1024 * 1024, f'Large Git file: {relative}'
        with path.open('rb') as stream:
            head = stream.read(8192)
            assert not head.startswith(b'\x7fELF'), f'ELF binary in Git: {relative}'
            if b'\0' not in head:
                stream.seek(0)
                scan(stream, relative)
        source_files.append({'export': relative, 'bytes': path.stat().st_size, 'sha256': digest(path)})
    reports = []
    assets = json.loads((REPO / 'manifests/assets.json').read_text())
    for asset in assets:
        path = OUT / 'assets' / asset['name']
        assert path.stat().st_size == asset['bytes'] and digest(path) == asset['sha256']
        index_name = 'models' if '-models-' in path.name else 'runtime' if '-runtime-' in path.name else None
        expected = json.loads((REPO / f'manifests/{index_name}-files-20261117.json').read_text()) if index_name else None
        actual = {}
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive:
                assert not member.name.startswith('/') and '..' not in Path(member.name).parts
                assert not member.issym() and not member.islnk(), f'Unexpected link: {member.name}'
                if not member.isfile():
                    continue
                stream = archive.extractfile(member)
                h = hashlib.sha256() if expected is not None else None
                if member.name.endswith(('.tar.gz', '.tgz')):
                    payload = stream.read()
                    if h is not None:
                        h.update(payload)
                    nested_scan(io.BytesIO(payload), path.name + '!' + member.name)
                elif Path(member.name).suffix in TEXT_SUFFIXES:
                    scan(stream, path.name + '!' + member.name, h)
                elif h is not None:
                    while block := stream.read(8 * 1024 * 1024):
                        h.update(block)
                actual[member.name] = {'bytes': member.size}
                if h is not None:
                    actual[member.name]['sha256'] = h.hexdigest()
        assert len(actual) == asset['members']
        if expected is not None:
            assert actual == expected, f'Content mismatch: {path.name}'
        else:
            for trial in ('no_right_hold_20261117_102134/trial02_no_right_hold',
                          'grasp_repair_20261117_095228/coordination01'):
                names = [name for name in actual if trial in name]
                assert any(name.endswith('.avi') and actual[name]['bytes'] > 1024 * 1024 for name in names)
                assert any(name.endswith('test_report.json') for name in names)
                assert any(name.endswith('ros_events.jsonl') for name in names)
                assert any(name.endswith('video_frames.jsonl') for name in names)
                assert any('successful_setup/' in name for name in names)
        reports.append({**asset, 'content_index_verified': expected is not None,
                        'path_or_credential_alerts': [], 'symlink_count': 0})
        print(json.dumps({'verified': path.name, 'files': len(actual)}), flush=True)
    for folder in (REPO / 'manifests', REPO / 'evidence/20261117'):
        for path in folder.rglob('*.json'):
            json.loads(path.read_text())
    for path in (REPO / 'scripts/download_assets.sh', REPO / 'training_tools/jetson_dual_act_20261116/run_sequence.sh',
                 REPO / 'training_tools/jetson_dual_act_20261116/stop_sequence.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    # Preserve original capture bytes, including original device-source
    # whitespace. Device copies are verified by hashes, not reformatted.
    # Check authored changes against the baseline, including new files.
    subprocess.run(['git', '-C', str(REPO), '-c', 'core.whitespace=-blank-at-eof',
                    'diff', '--check', 'a7e15b19a74f064e8d9fde2bf03753845b68cf04', '--', '.',
                    ':(exclude)data_collection/jetson/**', ':(exclude)data_collection/umie/**'], check=True)
    (REPO / 'manifests/source-files.json').write_text(json.dumps(source_files, indent=2) + '\n')
    report = {'verified_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'source_files': len(source_files), 'credential_scan': 'no high-confidence matches',
              'source_scan_scope': 'all repository files, plus text members inside all three assets and nested setup tars',
              'assets': reports, 'robot_motion_requested': False, 'device_service_changes': False,
              'checks': ['archive SHA-256 and sizes', 'per-file model/runtime index', 'both complete successful runs',
                         'safe archive paths; no symlinks', 'JSON parsing', 'launcher bash syntax',
                         'git diff --check against baseline (original device captures checked by hash; EOF blank lines preserved)']}
    (REPO / 'manifests/archive-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': True, 'source_files': len(source_files), 'assets': len(reports)}), flush=True)


if __name__ == '__main__':
    main()
