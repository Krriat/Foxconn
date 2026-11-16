cmake_minimum_required(VERSION 3.8)
project(rokae_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Joints.msg"
  "msg/DualArmServoJ.msg"
  "msg/DualArmServoL.msg"
  "msg/ServoLJog.msg"
  "msg/ServoJ.msg"
  "msg/TcpState.msg"
  "srv/ControlHand.srv"
  "srv/CartesianTeach.srv"
  "srv/ForwardKinematics.srv"
  "srv/GetCartesianState.srv"
  "srv/GetVisionTarget.srv"
  "srv/InverseKinematics.srv"
  "srv/MoveAbsJ.srv"
  "srv/MoveAbsJPath.srv"
  "srv/MoveL.srv"
  "srv/MoveJByPath.srv"
  "srv/JointTrajectory.srv"
  "srv/MoveLRelative.srv"
  "srv/SetVisionOffset.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
