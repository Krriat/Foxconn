#!/usr/bin/env bash
set -eo pipefail
vision_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
echo "ACT: requesting three cameras at 640x480@30 Hz; all three need USB 3.x bandwidth."
exec bash "$vision_root/scripts/start_cameras.sh" \
  camera_registry_file:="$vision_root/src/robot_vision_bringup/config/cameras_30hz.yaml" "$@"
