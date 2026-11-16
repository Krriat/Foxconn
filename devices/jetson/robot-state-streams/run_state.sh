#!/usr/bin/env bash
set -eo pipefail
state_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
cd "$state_dir"
exec "$state_dir/build/robot_hardware_state" --ros-args -p suction.host:=192.168.50.20 -p suction.port:=20108 >> "$state_dir/runtime.log" 2>&1
