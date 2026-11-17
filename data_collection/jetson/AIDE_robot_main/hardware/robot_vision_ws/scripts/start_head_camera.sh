#!/usr/bin/env bash
# Start the current head and right-wrist cameras through the RealSense ROS driver.
set -eo pipefail

vision_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep this lock across exec so simultaneous starts cannot open the same camera.
exec 8>"${XDG_RUNTIME_DIR:-/tmp}/aide-head-camera-${UID}.lock"
if ! flock -n 8; then
  echo "头部和右腕相机组正在启动或运行，请勿重复启动。" >&2
  exit 1
fi
python3 "$vision_root/scripts/head_camera_guard.py"
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
ulimit -Sr 0
unset CYCLONEDDS_URI

if [ ! -f "$vision_root/install/local_setup.bash" ]; then
  echo "Build robot_vision_bringup in $vision_root first." >&2
  exit 1
fi
source "$vision_root/install/local_setup.bash"
exec ros2 launch robot_vision_bringup multi_realsense.launch.py \
  camera_registry_file:="$vision_root/src/robot_vision_bringup/config/head_camera.yaml" \
  start_camera_manager:=false "$@"
