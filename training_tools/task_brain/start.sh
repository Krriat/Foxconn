#!/usr/bin/env bash
set -eo pipefail
brain_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
brain_root="$(cd -- "$brain_dir/../.." && pwd)"
case "${1:-observe}" in observe|auto) brain_mode="${1:-observe}" ;;
  *) echo "Usage: $0 observe|auto" >&2; exit 2 ;;
esac
if systemctl --user is-active --quiet robot-policy-brain.service; then
  echo "robot-policy-brain is already running. Check its mode in /robot/policy/brain_status."
  echo "To change mode, stop only this brain with: systemctl --user stop robot-policy-brain.service"
  exit 0
fi
exec systemd-run --user --unit=robot-policy-brain --collect --property=Restart=no \
  --property=TimeoutStopSec=60 \
  --property="StandardOutput=append:$brain_root/training_verification/task_brain_20261112/brain_service.log" \
  --property=StandardError=inherit /usr/bin/bash "$brain_dir/run.sh" "$brain_mode"
