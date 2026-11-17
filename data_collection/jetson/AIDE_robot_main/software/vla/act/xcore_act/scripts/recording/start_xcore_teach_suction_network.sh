#!/usr/bin/env bash
# Wired link: Jetson 192.168.50.10, PC 192.168.50.20. Forward recorder options.
set -eo pipefail
recording_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$recording_dir/start_xcore_teach_suction.sh" \
  --wrist-camera-host 192.168.50.20 --startup-timeout 90 --restart-cameras "$@"
