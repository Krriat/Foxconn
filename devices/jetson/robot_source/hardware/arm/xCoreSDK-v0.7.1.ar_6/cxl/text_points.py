#!/usr/bin/env python3
"""ROS target selection -> named text_dct poses. Never publishes motion commands."""
import argparse
import copy
import difflib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent


def field(data, path):
    for key in path.split('.'):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def vector(value, size):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'需要 {size} 个数')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('坐标必须是有限数值')
    return [float(x) for x in value]


def transform_matrix(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('transform 必须为4×4齐次矩阵或null')
    m = [vector(row, 4) for row in value]
    if any(abs(m[3][i] - [0, 0, 0, 1][i]) > 1e-6 for i in range(4)):
        raise ValueError('transform 最后一行应为 [0,0,0,1]')
    for i in range(3):
        for j in range(3):
            if abs(sum(m[i][k]*m[j][k] for k in range(3)) - (i == j)) > 1e-5:
                raise ValueError('transform 旋转部分必须正交')
    det = (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
           - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
           + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    if abs(det - 1) > 1e-5:
        raise ValueError('transform 旋转行列式必须为+1')
    return m


def load_sources(path):
    config = json.loads(Path(path).read_text())
    sources = config['sources']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('sources 需要至少一个具名来源')
    for name, s in sources.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('来源名仅支持字母、数字、下划线、短横线')
        for key in ('topic', 'type', 'frame_path', 'expected_frame'):
            if not isinstance(s.get(key), str) or not s[key]:
                raise ValueError(f'{name}: 缺少 {key}')
        if not isinstance(s.get('xyz'), list) or len(s['xyz']) != 3 or not all(isinstance(x, str) and x for x in s['xyz']):
            raise ValueError(f'{name}: xyz应为三个字段路径')
        for key, default in [('scale', 1), ('max_age_s', 5)]:
            value = s.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name}: {key}必须为正有限数')
        if s.get('reliability', 'reliable') not in ('reliable', 'best_effort'):
            raise ValueError('reliability只能为reliable或best_effort')
        if s.get('arm') not in ('left', 'right'):
            raise ValueError(f'{name}: arm应为left或right')
        transform_matrix(s.get('transform'))
        if s.get('transform') is not None and not s.get('target_frame'):
            raise ValueError('配置transform时必须填写target_frame说明实际控制器参考系')
    return sources


def decode(s, message):
    data = json.loads(field(message, s['json_field'])) if s.get('json_field') else message
    for path, expected in s.get('require', {}).items():
        if field(data, path) != expected:
            raise ValueError(f'目标无效: {path} != {expected}')
    frame = field(data, s['frame_path'])
    if frame != s['expected_frame']:
        raise ValueError(f'坐标系不匹配: {frame!r}，期望{s["expected_frame"]!r}')
    xyz = vector([field(data, path) for path in s['xyz']], 3)
    xyz = vector([x*s.get('scale', 1) for x in xyz], 3)
    stamp = None
    if s.get('stamp_path'):
        raw = field(data, s['stamp_path'])
        sec, nsec = raw['sec'], raw['nanosec']
        if type(sec) is not int or type(nsec) is not int or not 0 <= nsec < 1000000000:
            raise ValueError('无效消息时间戳')
        stamp = sec + nsec/1e9
    return {'xyz': xyz, 'frame': frame, 'stamp': stamp,
            'metadata': {key: field(data, path) for key, path in s.get('metadata', {}).items()}}


def make_pose(s, sample, rpy, offset):
    m = transform_matrix(s.get('transform'))
    if m is None or not s.get('target_frame'):
        raise ValueError('尚未配置检测坐标系到该臂控制器外部参考系的transform/target_frame，不能写入运动点')
    p = vector(sample['xyz'], 3) + [1]
    xyz = [sum(m[i][j]*p[j] for j in range(4)) + offset[i] for i in range(3)]
    return vector(xyz + vector(rpy, 3), 6)


def poses_in(text):
    section = ''
    poses = {}
    for line in text.splitlines():
        tokens = line.split('#', 1)[0].split()
        if not tokens:
            continue
        if tokens[0].startswith('['):
            section = tokens[0]
        elif section == '[poses]':
            if len(tokens) != 7 or tokens[0] in poses:
                raise ValueError('现有[poses]中有格式错误或重复点名')
            poses[tokens[0]] = vector([float(x) for x in tokens[1:]], 6)
    return poses


def edit_pose(text, name, pose, note, replace=False):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*', name):
        raise ValueError('点名应以字母/下划线开头，只含字母、数字、下划线、短横线')
    poses = poses_in(text)
    if name in poses and not replace:
        raise ValueError('点名已存在；有意覆盖时在命令末尾加 replace')
    eol = '\r\n' if '\r\n' in text else '\n'
    row = name + ' ' + ' '.join(f'{v:.9f}' for v in vector(pose, 6)) + ' # ' + note + eol
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.split('#', 1)[0].strip() == '[poses]']
    if len(starts) != 1:
        raise ValueError('conf必须恰有一个[poses]区域')
    start = starts[0]
    end = next((i for i in range(start+1, len(lines)) if lines[i].lstrip().startswith('[')), len(lines))
    for i in range(start+1, end):
        tokens = lines[i].split('#', 1)[0].split()
        if tokens and tokens[0] == name:
            lines[i] = row
            return ''.join(lines)
    if end and not lines[end-1].endswith('\n'):
        lines[end-1] += eol
    lines.insert(end, row)
    return ''.join(lines)


def save_config(path, expected, updated):
    path = Path(path).resolve()
    if path.read_bytes() != expected:
        raise ValueError('conf在预览后被修改，请重新执行save以免覆盖编辑')
    backup = path.with_name(path.name + f'.bak.{time.time_ns()}')
    shutil.copy2(path, backup)
    fd, temp = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(updated.encode('utf-8'))
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temp, path.stat().st_mode & 0o777)
        if path.read_bytes() != expected:
            raise ValueError('conf发生并发修改，请重新保存')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=HERE/'text_points.sources.json')
    parser.add_argument('--conf', type=Path, default=HERE/'text_dct.conf')
    parser.add_argument('--check', action='store_true', help='仅验证来源配置，不启动ROS')
    args = parser.parse_args()
    sources = load_sources(args.sources)
    if args.check:
        for name, s in sources.items():
            print(name, s['type'], s['topic'], '可转换' if s.get('transform') else '待配置坐标转换')
        return
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    from rosidl_runtime_py.convert import message_to_ordereddict
    rclpy.init()
    node = rclpy.create_node('text_point_selector')
    lock = threading.Lock()
    latest, errors, counters = {}, {}, {}
    selected = None

    def callback(name, msg):
        try:
            sample = decode(sources[name], message_to_ordereddict(msg))
            stamp = sample['stamp']
            age = node.get_clock().now().nanoseconds/1e9 - stamp if stamp is not None else 0
            if age > sources[name].get('max_age_s', 5) or age < -1:
                raise ValueError(f'消息时间戳过期或时钟不同步: age={age:.2f}s')
            with lock:
                counters[name] = counters.get(name, 0) + 1
                sample.update(received=time.monotonic(), sequence=counters[name])
                latest[name] = sample
                errors.pop(name, None)
        except Exception as exc:
            with lock:
                latest.pop(name, None)  # 不再允许选择最后一次有效但已失效的检测点
                errors[name] = str(exc)

    subscriptions = []
    try:
        for name, s in sources.items():
            qos = QoSProfile(depth=1, reliability=(ReliabilityPolicy.BEST_EFFORT
                if s.get('reliability') == 'best_effort' else ReliabilityPolicy.RELIABLE))
            subscriptions.append(node.create_subscription(get_message(s['type']), s['topic'],
                                 lambda msg, name=name: callback(name, msg), qos))
        def spin():
            try:
                rclpy.spin(node)
            except ExternalShutdownException:
                pass
        worker = threading.Thread(target=spin, daemon=True)
        worker.start()
        print('ROS_DOMAIN_ID='+os.environ.get('ROS_DOMAIN_ID', '0'))
        print('命令: list | pick 来源名 | show | save 新点名 姿态模板点名 [dx dy dz] [replace] | quit')
        print('偏移单位m，在目标控制器参考系中相加。只编辑点位，不改steps、不发运动命令。')
        while rclpy.ok():
            try:
                words = shlex.split(input('points> '))
                if not words:
                    continue
                if words == ['quit']:
                    break
                if words == ['list']:
                    with lock:
                        snapshot, issues = copy.deepcopy(latest), dict(errors)
                    for name, s in sources.items():
                        sample = snapshot.get(name)
                        if sample:
                            age = time.monotonic()-sample['received']
                            print(name, f'#{sample["sequence"]} age={age:.1f}s', sample['frame'], sample['xyz'], sample['metadata'],
                                  '已过期' if age > s.get('max_age_s', 5) else '')
                        else:
                            print(name, issues.get(name, '等待数据'), s['topic'])
                elif len(words) == 2 and words[0] == 'pick':
                    name = words[1]
                    with lock:
                        sample = copy.deepcopy(latest[name])
                    if time.monotonic()-sample['received'] > sources[name].get('max_age_s', 5):
                        raise ValueError('数据已过期，请等待新的检测结果')
                    selected = (name, sample)
                    print('已冻结选点（后续话题更新不会替换它）:', name, sample)
                elif words == ['show']:
                    print(selected or '尚未pick')
                elif words[0] == 'save' and len(words) in (3, 4, 6, 7):
                    if not selected:
                        raise ValueError('先pick一个来源')
                    replace = words[-1] == 'replace'
                    values = words[:-1] if replace else words
                    if len(values) not in (3, 6):
                        raise ValueError('save NAME TEMPLATE [dx dy dz] [replace]')
                    offset = vector([float(x) for x in values[3:]], 3) if len(values) == 6 else [0, 0, 0]
                    name, sample = selected
                    s = sources[name]
                    raw = args.conf.read_bytes()
                    original = raw.decode('utf-8')
                    rpy = poses_in(original)[values[2]][3:]
                    pose = make_pose(s, sample, rpy, offset)
                    note = json.dumps({'source': name, 'topic': s['topic'], 'arm': s['arm'],
                        'frame': sample['frame'], 'target_frame': s['target_frame'],
                        'stamp': sample['stamp'], 'sequence': sample['sequence'],
                        'metadata': sample['metadata'], 'rpy_template': values[2], 'offset_m': offset}, ensure_ascii=True)
                    updated = edit_pose(original, values[1], pose, note, replace)
                    print(''.join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True), fromfile='原conf', tofile='待保存conf')))
                    print(f'冻结选点已过去 {time.monotonic()-sample["received"]:.1f}s；目标臂={s["arm"]}。请确认点仍适用于当前场景。')
                    if input('输入 SAVE 写入该点，其他输入取消: ').strip() == 'SAVE':
                        print('已保存；备份:', save_config(args.conf, raw, updated))
                        print(f'可在[steps]自行添加 move {s["arm"]} {values[1]} 30，然后运行 text_dct --check')
                else:
                    print('命令格式不正确；list / pick 来源 / show / save NAME TEMPLATE [dx dy dz] [replace] / quit')
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as exc:
                print('未执行:', exc)
    finally:
        rclpy.try_shutdown()
        if 'worker' in locals():
            worker.join(timeout=3)
        node.destroy_node()


if __name__ == '__main__':
    main()
