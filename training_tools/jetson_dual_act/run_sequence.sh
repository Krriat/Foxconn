#!/usr/bin/env bash
set -eo pipefail
sequence_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
sequence_tag="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)"
exec /usr/bin/python3 -u "$sequence_root/training_tools/jetson_dual_act/run_live_once.py" \
  --execute --output "$sequence_root/training_verification/jetson_dual_act_20261113/user_runs/$sequence_tag"
