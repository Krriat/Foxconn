# Local dual-arm utilities, kept separate from the vendor examples.

set(CXL_TOOLS
    op
    op_single
    pos
    set_rotation
    moveabsj
    recover_moveabsj
    stop
    drag_example
    movel
    servol
    servol_relative
    servol_left
    pico_dual_servol
    pico_dual_servol_elbow
    control_hand
    test
    test1
    fk
    ik
    impedance_drag_teach
    movel_relative
    joint_drag_safe
    cartesian_drag_safe
    dual_cartesian_drag_safe
    movej_ik
    tcp_force_monitor
    controller_reboot
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)



foreach(tool ${CXL_TOOLS})
    add_executable(${tool} ${tool}.cpp)
    target_include_directories(${tool} PUBLIC
        ${SHARED_INCLUDEDIR})
    target_link_libraries(${tool} Rokae::Rokae Threads::Threads eigen)
endforeach()

if(XCORE_BUILD_CXL_ROS)
    # ROS targets require interfaces built in the robot's ROS workspace.
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(rclcpp_action REQUIRED)
    find_package(control_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(tf2 REQUIRED)
    find_package(liftmotor_rs485 REQUIRED)
    find_package(std_srvs REQUIRED)
    find_package(chassis_spin_interfaces REQUIRED)

    add_executable(text_dct text_dct.cpp text_topic_points.cpp text_chassis.cpp)
    target_include_directories(text_dct PUBLIC ${SHARED_INCLUDEDIR})
    target_link_libraries(text_dct Rokae::Rokae Threads::Threads eigen)
    ament_target_dependencies(text_dct
        chassis_spin_interfaces std_srvs geometry_msgs std_msgs rclcpp liftmotor_rs485)
    # Keep generated Fast RTPS type support available to dlopen and resolve indirect dependencies.
    target_link_libraries(text_dct "-Wl,--no-as-needed"
        liftmotor_rs485::liftmotor_rs485__rosidl_typesupport_fastrtps_cpp
        chassis_spin_interfaces::chassis_spin_interfaces__rosidl_typesupport_fastrtps_cpp
        "-Wl,--as-needed")
    target_link_options(text_dct PRIVATE "-Wl,--disable-new-dtags")

    add_executable(text_chassis_test EXCLUDE_FROM_ALL test_text_chassis.cpp text_chassis.cpp text_topic_points.cpp)
    target_link_libraries(text_chassis_test Threads::Threads "-Wl,--no-as-needed"
        chassis_spin_interfaces::chassis_spin_interfaces__rosidl_typesupport_fastrtps_cpp
        "-Wl,--as-needed")
    target_link_options(text_chassis_test PRIVATE "-Wl,--disable-new-dtags")
    ament_target_dependencies(text_chassis_test rclcpp std_msgs std_srvs geometry_msgs chassis_spin_interfaces)
    add_executable(text_topic_points_test EXCLUDE_FROM_ALL test_text_topic_points.cpp text_topic_points.cpp)
    target_link_libraries(text_topic_points_test Threads::Threads)
    ament_target_dependencies(text_topic_points_test rclcpp std_msgs geometry_msgs)

    add_executable(pico_servol pico_servol.cpp)
    target_include_directories(pico_servol PUBLIC ${SHARED_INCLUDEDIR})
    target_link_libraries(pico_servol Rokae::Rokae Threads::Threads eigen)
    ament_target_dependencies(pico_servol rclcpp sensor_msgs std_msgs)

    add_executable(ros_pos_publisher ros_pos_publisher.cpp)
    target_include_directories(ros_pos_publisher PUBLIC ${SHARED_INCLUDEDIR})
    target_link_libraries(ros_pos_publisher Rokae::Rokae Threads::Threads eigen)
    ament_target_dependencies(ros_pos_publisher rclcpp sensor_msgs geometry_msgs tf2)

    add_executable(ros_moveabsj_action_server ros_moveabsj_action_server.cpp)
    target_include_directories(ros_moveabsj_action_server PUBLIC ${SHARED_INCLUDEDIR})
    target_link_libraries(ros_moveabsj_action_server Rokae::Rokae Threads::Threads eigen)
    ament_target_dependencies(ros_moveabsj_action_server rclcpp rclcpp_action control_msgs)

    list(APPEND CXL_TOOLS pico_servol text_dct ros_pos_publisher ros_moveabsj_action_server)
endif()

install(TARGETS ${CXL_TOOLS}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
