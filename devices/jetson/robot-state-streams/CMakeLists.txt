cmake_minimum_required(VERSION 3.16)
project(robot_hardware_state LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(Threads REQUIRED)
set(ROBOT_SOURCE_ROOT "/home/jetson/AIDE_robot_main" CACHE PATH "Existing robot source and SDK")
set(SDK_ROOT "${ROBOT_SOURCE_ROOT}/hardware/arm/xCoreSDK-v0.7.1.ar_6")
set(SDK_LIB_DIR "${SDK_ROOT}/lib/Linux/aarch64")
set(M050_ROOT "${ROBOT_SOURCE_ROOT}/hardware/USR-M050")
add_executable(robot_hardware_state hardware_state.cpp)
target_compile_options(robot_hardware_state PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(robot_hardware_state PRIVATE
  "${SDK_ROOT}/include" "${SDK_ROOT}/external" "${M050_ROOT}/include")
# Use the same intact static archives as the existing robot driver.
target_link_libraries(robot_hardware_state
  "${SDK_LIB_DIR}/libxCoreSDK.a" "${SDK_LIB_DIR}/libxMateModel.a"
  "${M050_ROOT}/build/libm050.so" Threads::Threads)
ament_target_dependencies(robot_hardware_state rclcpp sensor_msgs std_msgs diagnostic_msgs)
set_target_properties(robot_hardware_state PROPERTIES BUILD_RPATH "${M050_ROOT}/build")
