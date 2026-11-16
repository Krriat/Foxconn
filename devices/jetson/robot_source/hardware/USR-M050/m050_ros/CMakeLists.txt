cmake_minimum_required(VERSION 3.16)
project(m050_ros VERSION 0.1.0 LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Threads REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/IOState.msg"
  "srv/GetIO.srv"
  "srv/SetDO.srv"
  DEPENDENCIES builtin_interfaces
)

# Compile the existing SDK into the node without altering its standalone build.
add_library(m050_ros_sdk STATIC ../src/m050.c)
target_include_directories(m050_ros_sdk PUBLIC ../include)
target_link_libraries(m050_ros_sdk PUBLIC Threads::Threads)
set_target_properties(m050_ros_sdk PROPERTIES
  C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)

add_executable(m050_node src/m050_node.cpp)
target_compile_features(m050_node PRIVATE cxx_std_14)
target_link_libraries(m050_node m050_ros_sdk)
ament_target_dependencies(m050_node rclcpp rcl_interfaces builtin_interfaces)
rosidl_get_typesupport_target(typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(m050_node "${typesupport_target}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(m050_node PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS m050_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(ros_integration test/test_ros_integration.py
    TIMEOUT 120
    ENV M050_NODE_EXECUTABLE=$<TARGET_FILE:m050_node>)
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
