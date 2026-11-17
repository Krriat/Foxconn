cmake_minimum_required(VERSION 3.8)
project(robot_camera_msgs)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 17)
  set(CMAKE_C_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraStatus.msg"
  "srv/GetCameraInfo.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
