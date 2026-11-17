#!/usr/bin/env bash
set -eo pipefail
vision_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
aide_root="$(cd -- "$vision_root/../.." && pwd)"
# Share the head-camera lock with start_head_camera.sh: the three-camera
# group also owns the head device and must not run beside the single-camera entry.
exec 8>"${XDG_RUNTIME_DIR:-/tmp}/aide-head-camera-${UID}.lock"
if ! flock -n 8; then
  echo "相机已由单头部或三相机入口占用，请先退出原采集程序。" >&2
  exit 1
fi
python3 "$vision_root/scripts/head_camera_guard.py"
if [ ! -f "$vision_root/install/local_setup.bash" ]; then
  echo "请先在 $vision_root 构建 robot_vision_bringup。" >&2
  exit 1
fi
source "$aide_root/startup/ros_environment.sh"
source /opt/ros/humble/setup.bash
source "$vision_root/install/local_setup.bash"
source "$aide_root/startup/ros_environment.sh"
# Local USB capture does not depend on the old deployment's Wi-Fi interface.
exec ros2 launch robot_vision_bringup multi_realsense.launch.py \
  camera_registry_file:="$vision_root/src/robot_vision_bringup/config/cameras.yaml" "$@"
