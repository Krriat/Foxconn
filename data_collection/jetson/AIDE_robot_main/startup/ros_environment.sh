#!/usr/bin/env bash

# User-facing ROS environment. Internal launchers source ros_environment_debug.sh.
# Use Fast DDS with its default network transports for LAN communication.
# Domain 0 contains delivery interfaces and the dual-arm package.
# Other internal drivers and debugging use domain 1.
# This machine's Fast DDS build requests FIFO priority 80 for its threads.
# Keep ROS processes under normal scheduling to avoid starving the shell
# when an asynchronous DDS thread stays runnable. Scope this to this shell
# and its children; leave the system-wide realtime configuration alone.
ulimit -Sr 0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
unset CYCLONEDDS_URI
