#!/usr/bin/env bash
set -euo pipefail
policy_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
test -f "$HOME/.local/share/robot-policy-streams/inference_remote.txt"
if systemctl --user is-active --quiet robot-policy-brain.service; then
  brain_pid="$(systemctl --user show robot-policy-brain.service --property=MainPID --value)"
  if [[ -r "/proc/$brain_pid/cmdline" ]]; then
    brain_command="$(tr '\0' ' ' < "/proc/$brain_pid/cmdline")"
    if [[ "$brain_command" == *--execute-transitions* && "$brain_command" == *--observe-idle* ]]; then
      echo "Qwen LoRA automatic brain is already running."
      exit 0
    fi
  fi
fi
for policy_unit in robot-policy-inference.service robot-policy-brain.service; do
  if systemctl --user is-active --quiet "$policy_unit"; then
    systemctl --user stop "$policy_unit"
  fi
done
exec bash "$policy_root/training_tools/task_brain/start.sh" auto
