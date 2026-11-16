#!/usr/bin/env bash
# One terminal: existing arm teach services, three cameras, keyboard recorder.
set -eo pipefail
recording_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
aide_root="$(cd -- "$recording_dir/../../../../../.." && pwd)"
for setup_file in \
  /opt/ros/humble/setup.bash \
  "$aide_root/hardware/arm/xCoreROS/install/local_setup.bash" \
  "$aide_root/hardware/robot_vision_ws/install/local_setup.bash"; do
  if [ ! -f "$setup_file" ]; then
    echo "缺少 ROS/工作区环境，请先完成构建：$setup_file" >&2
    exit 1
  fi
done
source "$aide_root/startup/ros_environment.sh"
source /opt/ros/humble/setup.bash
source "$aide_root/hardware/arm/xCoreROS/install/local_setup.bash"
source "$aide_root/hardware/robot_vision_ws/install/local_setup.bash"
source "$aide_root/startup/ros_environment.sh"
exec /usr/bin/python3 -u "$recording_dir/launch_xcore_teach_suction.py" "$@"
