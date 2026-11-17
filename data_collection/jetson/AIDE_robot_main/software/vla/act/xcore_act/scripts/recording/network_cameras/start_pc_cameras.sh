#!/usr/bin/env bash
set -eo pipefail
camera_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -x "$camera_dir/.venv/bin/python" ] || \
   ! "$camera_dir/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "虚拟环境未安装完整。请先运行：bash $camera_dir/setup_pc.sh" >&2
  exit 1
fi
if ! "$camera_dir/.venv/bin/python" -c 'import numpy, cv2, pyrealsense2' >/dev/null 2>&1; then
  echo "相机依赖尚未安装完整。请先运行：bash $camera_dir/setup_pc.sh" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
exec "$camera_dir/.venv/bin/python" "$camera_dir/pc_camera_server.py" "$@"
