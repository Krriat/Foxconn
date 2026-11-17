#!/usr/bin/env bash
set -eo pipefail
vision_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
aide_root="$(cd -- "$vision_root/../.." && pwd)"
source "$aide_root/startup/ros_environment.sh"
source /opt/ros/humble/setup.bash
if [ ! -f "$vision_root/install/local_setup.bash" ]; then
  echo "请先在 $vision_root 构建 robot_vision_tools。" >&2
  exit 1
fi
source "$vision_root/install/local_setup.bash"
exec ros2 run robot_vision_tools udp_video_sender \
  --camera-registry "$vision_root/src/robot_vision_bringup/config/head_camera.yaml" "$@"
