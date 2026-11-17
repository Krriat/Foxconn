# Respect an explicit operator choice. Prefer Cyclone DDS when installed,
# otherwise use Fast DDS, matching tools/start_upperlimb.sh.
if [ -z "${RMW_IMPLEMENTATION:-}" ]; then
  if [ -f "/opt/ros/${ROS_DISTRO:-humble}/lib/librmw_cyclonedds_cpp.so" ]; then
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  else
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  fi
fi
