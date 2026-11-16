#!/usr/bin/env bash
# Start the upper-limb state/SDK interfaces and head RGB-D publisher together.
# Starting this script does not initialize, power on, or move either arm.
set -e

workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vision_dir="$workspace_dir/../../robot_vision_ws"
start_head_camera=true
arm_args=()
for arg in "$@"; do
  case "$arg" in
    start_head_camera:=true) start_head_camera=true ;;
    start_head_camera:=false) start_head_camera=false ;;
    start_head_camera:=*) echo "start_head_camera 仅支持 true 或 false" >&2; exit 1 ;;
    *) arm_args+=("$arg") ;;
  esac
done
# Keep this lock in the launch parent until it exits. It also covers simultaneous
# starts before the driver executable appears in /proc.
workspace_key="$(printf '%s' "$workspace_dir" | sha256sum | cut -c1-16)"
exec 9>"${XDG_RUNTIME_DIR:-/tmp}/aide-upperlimb-${UID}-${workspace_key}.lock"
if ! flock -n 9; then
  echo "统一驱动已在启动或运行，请勿重复启动。后台服务状态：systemctl --user status aide-arm-sdk" >&2
  exit 1
fi
# Detect drivers started before this guard, or through ros2 launch directly.
python3 - "$workspace_dir/install/rokae_driver/lib/rokae_driver/ros_dual_arm_driver" <<'PY'
from pathlib import Path
import os
import sys

expected = os.path.realpath(sys.argv[1])
running = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        command = (entry / 'cmdline').read_bytes().split(b'\0')
        if command[0] and os.path.realpath(os.fsdecode(command[0])) == expected:
            running.append(int(entry.name))
    except (OSError, ValueError):
        continue
if running:
    print(f"统一驱动已经运行 (PID: {sorted(running)})，拒绝创建第二套 SDK 连接。", file=sys.stderr)
    print("后台服务已运行时可直接使用行为树。若需前台启动，先执行 systemctl --user stop aide-arm-sdk。", file=sys.stderr)
    sys.exit(1)
PY
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

# Some robot computers have only Fast DDS installed.
if [ -z "${RMW_IMPLEMENTATION:-}" ]; then
  if [ -f /opt/ros/humble/lib/librmw_cyclonedds_cpp.so ]; then
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  else
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  fi
fi

if [ ! -f "$workspace_dir/install/local_setup.bash" ]; then
  echo "Run colcon build --symlink-install --packages-up-to rokae_bringup first." >&2
  exit 1
fi
source "$workspace_dir/install/local_setup.bash"
cd "$workspace_dir"

camera_args=()
if [ "$start_head_camera" = true ]; then
  camera_args=(--head-camera-script "$vision_dir/scripts/start_head_camera.sh")
fi

exec python3 "$workspace_dir/tools/run_upperlimb.py" "${camera_args[@]}" -- \
  ros2 launch rokae_bringup dual_arm.launch.py \
  start_state_publisher:=true \
  start_move_server:=true start_moveabsj_service:=true \
  start_movel_service:=true start_movej_by_path_service:=true \
  start_initializer_service:=true start_kinematics_service:=true \
  start_hand_service:=false start_cartesian_teach_service:=false \
  start_cartesian_drag_safe_service:=true \
  start_go_home_service:=false start_servoj:=false start_servol:=false \
  start_chassis_navigation:=false "${arm_args[@]}"
