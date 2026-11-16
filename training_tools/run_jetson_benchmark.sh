#!/usr/bin/env bash
set -euo pipefail
benchmark_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="$benchmark_root/runtime_libs:${LD_LIBRARY_PATH:-}"
mkdir -p "$benchmark_root/results"
nvpmodel -q > "$benchmark_root/results/power_mode.txt"
tegrastats --interval 1000 > "$benchmark_root/results/tegrastats.log" 2>&1 &
benchmark_monitor_pid=$!
trap 'kill "$benchmark_monitor_pid" 2>/dev/null || true' EXIT
"$benchmark_root/venv/bin/python" "$benchmark_root/bundle/benchmark_jetson.py" \
    --root "$benchmark_root/bundle" --output "$benchmark_root/results/report.json" "$@"
