#!/usr/bin/env bash
set -eo pipefail
camera_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
camera_python="${AIDE_PC_PYTHON:-python3}"
"$camera_python" -c 'import sys; assert sys.version_info >= (3,10), "Use Python 3.10 or newer"'
camera_venv_package="$("$camera_python" -c 'import sys; print("python%d.%d-venv" % sys.version_info[:2])')"
if ! "$camera_python" -c 'import ensurepip' >/dev/null 2>&1; then
  echo "缺少创建虚拟环境所需的 ensurepip。请先在电脑执行：" >&2
  echo "  sudo apt update" >&2
  echo "  sudo apt install $camera_venv_package" >&2
  echo "然后重新运行：bash $camera_dir/setup_pc.sh" >&2
  exit 1
fi
# A failed venv creation can leave an executable Python without pip. Re-running
# venv repairs that partial environment without deleting the user's files.
if [ ! -x "$camera_dir/.venv/bin/python" ] || \
   ! "$camera_dir/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
  "$camera_python" -m venv "$camera_dir/.venv"
fi
"$camera_dir/.venv/bin/python" -m pip install -r "$camera_dir/requirements-pc.txt"
echo "完成。枚举相机：bash $camera_dir/start_pc_cameras.sh --list"
echo "启动发送：bash $camera_dir/start_pc_cameras.sh"
