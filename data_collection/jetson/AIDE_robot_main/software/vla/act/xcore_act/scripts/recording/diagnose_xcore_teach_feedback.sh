#!/usr/bin/env bash
# User-run diagnostic launch; the analysis script itself never controls arms.
set -eo pipefail
recording_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" = --help ]]; then
  echo '用法：bash diagnose_xcore_teach_feedback.sh --fps 20 --task "双臂吸盘抓取"'
  echo '启动现有双臂服务、三相机和采集器，额外保存 SDK/发布/接收耗时。'
  echo '全部就绪后 A 开启示教；先保持静止约60秒，再 R 正常录制；S保存，Q退出后自动分析。'
  exit 0
fi
python3 - <<'PY'
from pathlib import Path
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        command=(p/'cmdline').read_bytes().split(b'\0')[0]
        if Path(command.decode()).name=='ros_dual_arm_driver':
            raise SystemExit('已有双臂驱动运行，请先从原启动终端退出，再运行诊断入口；复用旧进程无法记录SDK耗时。')
    except (OSError,UnicodeError):
        pass
PY
export AIDE_FEEDBACK_TRACE_DIR
AIDE_FEEDBACK_TRACE_DIR="$(mktemp -d /tmp/aide-teach-feedback-XXXXXXXX)"
echo "FEEDBACK_TRACE_DIR=$AIDE_FEEDBACK_TRACE_DIR"
finish_trace() {
  /usr/bin/python3 "$recording_dir/../diagnostics/analyze_teach_feedback.py" \
    --trace-dir "$AIDE_FEEDBACK_TRACE_DIR" || true
}
trap finish_trace EXIT
bash "$recording_dir/start_xcore_teach_suction_network.sh" \
  --bringup-log-dir "$AIDE_FEEDBACK_TRACE_DIR/bringup" "$@"
