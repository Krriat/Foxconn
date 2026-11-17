#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
exec /usr/bin/python3 "$(dirname -- "$0")/monitor_feedback.py" "$@"
