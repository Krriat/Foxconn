#!/usr/bin/env bash
set -eo pipefail
policy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
if [[ "${1:-}" == "executor" || "${1:-}" == "driver" ]]; then
  source /home/jetson/AIDE_robot_main/hardware/arm/xCoreROS/install/setup.bash
fi
# The installed robot overlay exports Domain 0: select our Domain AFTER sourcing.
export ROS_DOMAIN_ID=1 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
case "${1:-}" in
  task-monitor)
    exec /usr/bin/python3 -u "$policy_dir/task_monitor.py"
    ;;
  inference)
    if [[ -f "$policy_dir/inference_remote.txt" ]]; then cat "$policy_dir/inference_remote.txt" >&2; exit 2; fi
    exec /home/jj/code/ACT/.venv-act/bin/python3 -u "$policy_dir/inference_node.py" --config "$policy_dir/config.yaml"
    ;;
  executor)
    exec /usr/bin/python3 -u "$policy_dir/executor_node.py" --config "$policy_dir/config.yaml"
    ;;
  driver)
    exec "$policy_dir/native/build/robot_timed_policy_driver" \
      --ros-args --params-file "$policy_dir/driver.yaml" \
      -r /robot_02/upperlimb/initialize/dual_arm:=/robot/upperlimb/initialize/dual_arm \
      -r /robot_02/upperlimb/stop/left_arm:=/robot/upperlimb/stop/left_arm \
      -r /robot_02/upperlimb/stop/right_arm:=/robot/upperlimb/stop/right_arm \
      -r /robot_02/upperlimb/move_absj/left_arm:=/robot/upperlimb/move_absj/left_arm \
      -r /robot_02/upperlimb/move_absj/right_arm:=/robot/upperlimb/move_absj/right_arm \
      -r /robot_02/upperlimb/servoj/dual_arm:=/robot/upperlimb/servoj/dual_arm \
      -r /robot_02/upperlimb/servoj/status:=/robot/upperlimb/servoj/status
    ;;
  *) echo "Usage: $0 inference|executor|driver|task-monitor" >&2; exit 2 ;;
esac
