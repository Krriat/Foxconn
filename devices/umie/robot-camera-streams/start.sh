#!/usr/bin/env bash
set -eo pipefail
camera_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  wrists) camera_unit=robot-wrist-cameras ;;
  head) camera_unit=robot-head-camera ;;
  *) printf 'Usage: %s wrists|head\n' "$0" >&2; exit 2 ;;
esac
if systemctl --user is-active --quiet "$camera_unit"; then
  printf '%s is already running\n' "$camera_unit"
  exit 0
fi
test -f "$camera_dir/$1.yaml"
exec systemd-run --user --unit="$camera_unit" --collect \
  --property=Restart=on-failure --property=RestartSec=3 \
  --setenv="ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1}" \
  --setenv=ROS_LOCALHOST_ONLY=0 --setenv=RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  "$camera_dir/run_camera.sh" "$1"
