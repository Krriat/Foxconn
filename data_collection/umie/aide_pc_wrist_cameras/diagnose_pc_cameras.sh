#!/usr/bin/env bash
# Run on the camera PC. Captures frame timings and read-only USB/kernel logs.
set -eo pipefail
camera_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$camera_dir/diagnostics"
trace_dir="$(mktemp -d "$camera_dir/diagnostics/camera-trace-XXXXXXXX")"
kernel_pid=
cleanup() {
  if [ -n "$kernel_pid" ]; then
    kill "$kernel_pid" 2>/dev/null || true
    wait "$kernel_pid" 2>/dev/null || true
    kernel_pid=
  fi
}
trap cleanup EXIT
printf 'PC_DIAGNOSTICS_DIR=%s\n' "$trace_dir"
{
  date -u --iso-8601=seconds
  uname -a
  ip -br address
  lsusb -t
  lsusb
} > "$trace_dir/system.log" 2>&1 || true
if command -v journalctl >/dev/null 2>&1; then
  journalctl -k -f --since now -o short-monotonic > "$trace_dir/kernel.log" 2>&1 &
  kernel_pid=$!
else
  dmesg --follow --time-format iso > "$trace_dir/kernel.log" 2>&1 &
  kernel_pid=$!
fi
echo '保持运行，Jetson 完成复测后按 Ctrl+C；若 kernel.log 提示权限不足，该次 USB 日志不可用于排除硬件问题。'
bash "$camera_dir/start_pc_cameras.sh" --diagnostics-dir "$trace_dir" "$@" 2>&1 | tee "$trace_dir/server.log"
