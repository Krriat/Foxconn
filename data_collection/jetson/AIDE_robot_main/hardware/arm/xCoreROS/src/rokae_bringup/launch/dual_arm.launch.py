"""Start the dual-arm state publisher and motion servers."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    default_parameters = str(
        Path(get_package_share_directory("rokae_bringup"))
        / "config"
        / "dual_arm.yaml"
    )

    parameters_file = LaunchConfiguration("params_file")
    start_hand_service = LaunchConfiguration("start_hand_service")
    start_cartesian_drag_safe_service = LaunchConfiguration("start_cartesian_drag_safe_service")
    start_cartesian_teach_service = LaunchConfiguration(
        "start_cartesian_teach_service"
    )
    start_go_home_service = LaunchConfiguration("start_go_home_service")
    start_initializer_service = LaunchConfiguration(
        "start_initializer_service"
    )
    start_kinematics_service = LaunchConfiguration(
        "start_kinematics_service"
    )
    start_move_server = LaunchConfiguration("start_move_server")
    start_moveabsj_service = LaunchConfiguration("start_moveabsj_service")
    start_movel_service = LaunchConfiguration("start_movel_service")
    start_movej_by_path_service = LaunchConfiguration(
        "start_movej_by_path_service"
    )
    start_state_publisher = LaunchConfiguration("start_state_publisher")
    start_servoj = LaunchConfiguration("start_servoj")
    start_servol = LaunchConfiguration("start_servol")
    start_chassis_navigation = LaunchConfiguration(
        "start_chassis_navigation"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_cartesian_drag_safe_service", default_value="false",
                description="Enable shared-connection PICO Cartesian drag toggles",
            ),
            DeclareLaunchArgument(
                "start_go_home_service",
                default_value="true",
                description="Start left, right and dual-arm go-home services",
            ),
            DeclareLaunchArgument(
                "start_hand_service",
                default_value="true",
                description="Start the dual-hand end-CAN services",
            ),
            DeclareLaunchArgument(
                "start_cartesian_teach_service",
                default_value="true",
                description="Start left and right Cartesian teach services",
            ),
            DeclareLaunchArgument(
                "start_initializer_service",
                default_value="true",
                description="Start the explicit dual-arm power-on service",
            ),
            DeclareLaunchArgument(
                "start_kinematics_service",
                default_value="true",
                description="Start left and right arm FK/IK services",
            ),
            DeclareLaunchArgument(
                "start_movel_service",
                default_value="true",
                description="Start the dual-arm MoveL services",
            ),
            DeclareLaunchArgument(
                "start_movej_by_path_service",
                default_value="true",
                description="Start left, right and dual-arm MoveJ path services",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_parameters,
                description="ROS 2 parameter YAML file",
            ),
            DeclareLaunchArgument(
                "start_move_server",
                default_value="true",
                description="Start the dual-arm MoveAbsJ action server",
            ),
            DeclareLaunchArgument(
                "start_moveabsj_service",
                default_value="true",
                description="Start the left and right MoveAbsJ services",
            ),
            DeclareLaunchArgument(
                "start_servoj",
                default_value="true",
                description="Start left, right and dual-arm ServoJ topics",
            ),
            DeclareLaunchArgument(
                "start_servol", default_value="true",
                description="Start left, right and dual-arm ServoL topics",
            ),
            DeclareLaunchArgument(
                "start_state_publisher",
                default_value="true",
                description=(
                    "Publish both arms' joints, TCP poses and Jacobians"
                ),
            ),
            DeclareLaunchArgument(
                "start_chassis_navigation",
                default_value="false",
                description=(
                    "Start the scheduler-to-Seer chassis navigation bridge"
                ),
            ),
            Node(
                package="rokae_driver",
                executable="ros_dual_arm_driver",
                output="screen",
                parameters=[
                    parameters_file,
                    {
                        "start_cartesian_drag_safe_service": ParameterValue(
                            start_cartesian_drag_safe_service, value_type=bool
                        ),
                        "start_hand_service": ParameterValue(
                            start_hand_service, value_type=bool
                        ),
                        "start_cartesian_teach_service": ParameterValue(
                            start_cartesian_teach_service, value_type=bool
                        ),
                        "start_go_home_service": ParameterValue(
                            start_go_home_service, value_type=bool
                        ),
                        "start_initializer_service": ParameterValue(
                            start_initializer_service, value_type=bool
                        ),
                        "start_kinematics_service": ParameterValue(
                            start_kinematics_service, value_type=bool
                        ),
                        "start_move_server": ParameterValue(
                            start_move_server, value_type=bool
                        ),
                        "start_moveabsj_service": ParameterValue(
                            start_moveabsj_service, value_type=bool
                        ),
                        "start_movel_service": ParameterValue(
                            start_movel_service, value_type=bool
                        ),
                        "start_movej_by_path_service": ParameterValue(
                            start_movej_by_path_service, value_type=bool
                        ),
                        "start_state_publisher": ParameterValue(
                            start_state_publisher, value_type=bool
                        ),
                        "start_servoj": ParameterValue(
                            start_servoj, value_type=bool
                        ),
                        "start_servol": ParameterValue(start_servol, value_type=bool),
                    },
                ],
            ),
            Node(
                package="rokae_motion",
                executable="chassis_navigation",
                output="screen",
                condition=IfCondition(start_chassis_navigation),
            ),
        ]
    )
