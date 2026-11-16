#!/usr/bin/env bash
set -euo pipefail
bash "$HOME/.local/share/robot-camera-streams/start.sh" head
bash "$HOME/.local/share/robot-state-streams/start.sh"
for policy_role in driver executor inference; do
  bash "$HOME/.local/share/robot-policy-streams/start.sh" "$policy_role"
done
echo "Services started in standby. No robot task has been started."
