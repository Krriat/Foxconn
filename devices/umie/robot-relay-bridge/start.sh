#!/usr/bin/env bash
set -eo pipefail
bridge_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if systemctl --user is-active --quiet robot-relay-bridge; then
  echo 'robot-relay-bridge is already running'
  exit 0
fi
exec systemd-run --user --unit=robot-relay-bridge --collect \
  --property=Restart=on-failure --property=RestartSec=2 \
  --property="StandardOutput=append:$bridge_dir/runtime.log" --property=StandardError=inherit \
  /usr/bin/python3 -u "$bridge_dir/tcp_bridge.py" \
  --listen-host 192.168.50.20 --listen-port 20108 --allowed-peer 192.168.50.10 \
  --upstream-host 192.168.1.7 --upstream-port 20108 --upstream-source 192.168.1.20
