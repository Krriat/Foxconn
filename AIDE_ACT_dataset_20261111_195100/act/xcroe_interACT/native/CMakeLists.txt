cmake_minimum_required(VERSION 3.16)
project(xcore_interact_dual_recording LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(XCORE_SDK_ROOT "/home/niic/aide/hardware/arm/xCoreSDK-v0.7.1.ar_6" CACHE PATH "xCore SDK root")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)

find_library(XCORE_SDK_LIBRARY xCoreSDK
  PATHS "${XCORE_SDK_ROOT}/lib/Linux/aarch64"
  NO_DEFAULT_PATH REQUIRED)
find_library(XMATE_MODEL_LIBRARY xMateModel
  PATHS "${XCORE_SDK_ROOT}/lib/Linux/aarch64"
  NO_DEFAULT_PATH REQUIRED)
find_library(OROCOS_KDL_LIBRARY orocos-kdl
  PATHS "${XCORE_SDK_ROOT}/external/kdl/lib/aarch64"
  NO_DEFAULT_PATH REQUIRED)
find_library(G3LOG_LIBRARY g3log
  PATHS "${XCORE_SDK_ROOT}/external/g3log/lib/aarch64"
  NO_DEFAULT_PATH REQUIRED)

add_executable(pico_dual_recording pico_dual_recording.cpp)
target_include_directories(pico_dual_recording PRIVATE
  "${XCORE_SDK_ROOT}/include"
  "${XCORE_SDK_ROOT}/external"
  "${XCORE_SDK_ROOT}/external/kdl/include"
  "${XCORE_SDK_ROOT}/external/g3log/include")
target_link_libraries(pico_dual_recording
  "${XCORE_SDK_LIBRARY}"
  "${XMATE_MODEL_LIBRARY}"
  "${OROCOS_KDL_LIBRARY}"
  "${G3LOG_LIBRARY}"
  Threads::Threads)
ament_target_dependencies(pico_dual_recording rclcpp sensor_msgs std_msgs)

install(TARGETS pico_dual_recording DESTINATION lib/${PROJECT_NAME})
ament_package()
