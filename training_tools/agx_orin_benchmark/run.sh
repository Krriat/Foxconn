#!/usr/bin/env bash
set -euo pipefail
ACT_AGX_BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$ACT_AGX_BENCHMARK_ROOT/runtime_libs:${LD_LIBRARY_PATH:-}"
mkdir -p "$ACT_AGX_BENCHMARK_ROOT/results"
nvpmodel -q > "$ACT_AGX_BENCHMARK_ROOT/results/power_before.txt"
"$ACT_AGX_BENCHMARK_ROOT/venv/bin/python" -m pip freeze > "$ACT_AGX_BENCHMARK_ROOT/results/requirements.lock.txt"
tegrastats --interval 1000 > "$ACT_AGX_BENCHMARK_ROOT/results/tegrastats.log" 2>&1 &
ACT_AGX_MONITOR_PID=$!
trap 'kill "$ACT_AGX_MONITOR_PID" 2>/dev/null || true' EXIT
"$ACT_AGX_BENCHMARK_ROOT/venv/bin/python" -u "$ACT_AGX_BENCHMARK_ROOT/bundle/benchmark.py" \
  --root "$ACT_AGX_BENCHMARK_ROOT/bundle" \
  --output "$ACT_AGX_BENCHMARK_ROOT/results/report.json" "$@"
nvpmodel -q > "$ACT_AGX_BENCHMARK_ROOT/results/power_after.txt"
