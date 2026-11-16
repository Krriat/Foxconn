#!/usr/bin/env bash
set -eo pipefail
policy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  inference|executor|driver|task-monitor) policy_role="$1" ;;
  *) echo "Usage: $0 inference|executor|driver|task-monitor" >&2; exit 2 ;;
esac
policy_unit="robot-policy-$policy_role"
if [[ "$policy_role" == "inference" && -f "$policy_dir/inference_remote.txt" ]]; then
  cat "$policy_dir/inference_remote.txt" >&2
  exit 2
fi
if systemctl --user is-active --quiet "$policy_unit"; then
  echo "$policy_unit is already running"
  exit 0
fi
exec systemd-run --user --unit="$policy_unit" --collect \
  --property=Restart=no --property=TimeoutStopSec=90 \
  --property="StandardOutput=append:$policy_dir/$policy_role.log" \
  --property=StandardError=inherit \
  --setenv=ROS_DOMAIN_ID=1 --setenv=ROS_LOCALHOST_ONLY=0 \
  --setenv=RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  "$policy_dir/run.sh" "$policy_role"
