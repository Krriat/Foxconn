#!/usr/bin/env bash
set -eo pipefail
policy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/home/jetson/act_model_benchmark_20261112/runtime_libs:${LD_LIBRARY_PATH:-}"
exec /home/jetson/act_model_benchmark_20261112/venv/bin/python -u \
  "$policy_dir/jetson_inference_bootstrap.py" --config "$policy_dir/config.yaml"
