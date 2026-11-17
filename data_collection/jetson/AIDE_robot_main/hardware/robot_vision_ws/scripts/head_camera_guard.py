"""Identify local publishers for the configured head camera without opening USB."""
from pathlib import Path
import sys


def running_head_cameras():
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().split(b'\0')
            if (Path(command[0].decode()).name == 'realsense2_camera_node'
                    and b'__node:=head' in command
                    and b'__ns:=/robot_02/sensors/cameras' in command):
                result.append(int(entry.name))
        except (OSError, ValueError, UnicodeError):
            continue
    return sorted(result)


if __name__ == '__main__':
    running = running_head_cameras()
    if running:
        print(f'头部相机已运行 (PID: {running})，拒绝重复打开相机。', file=sys.stderr)
        sys.exit(1)
