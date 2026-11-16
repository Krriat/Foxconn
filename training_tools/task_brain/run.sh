#!/usr/bin/env bash
set -eo pipefail
brain_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${1:-observe}" in
  observe) brain_options=(--observe-idle) ;;
  auto) brain_options=(--execute-transitions --observe-idle --idle-stage far) ;;
  *) echo "Usage: $0 observe|auto" >&2; exit 2 ;;
esac
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
cd "$brain_root"
exec "$brain_root/.venv-brain/bin/python" -u -m training_tools.task_brain.node \
  --adapter "$brain_root/model_outputs/task_brain_lora_20261112/adapter" \
  --log "$brain_root/training_verification/task_brain_20261112/brain_live.jsonl" \
  "${brain_options[@]}"
