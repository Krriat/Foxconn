#!/usr/bin/env bash
set -eo pipefail
state_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if systemctl --user is-active --quiet robot-hardware-state; then
  printf 'robot-hardware-state is already running\n'
  exit 0
fi
exec systemd-run --user --unit=robot-hardware-state --collect \
  --property=Restart=on-failure --property=RestartSec=3 \
  --setenv="ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1}" \
  "$state_dir/run_state.sh"
