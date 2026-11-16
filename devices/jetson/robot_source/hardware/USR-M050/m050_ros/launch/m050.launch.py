from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {
        "host": ("192.168.1.7", str),
        "port": ("20108", int),
        "unit_id": ("1", int),
        "timeout_ms": ("3000", int),
        "source_ip": ("", str),
        "poll_interval_ms": ("500", int),
    }
    arguments = [
        DeclareLaunchArgument(name, default_value=value)
        for name, (value, _) in defaults.items()
    ]
    arguments.append(DeclareLaunchArgument("namespace", default_value=""))
    parameters = {
        name: ParameterValue(LaunchConfiguration(name), value_type=kind)
        for name, (_, kind) in defaults.items()
    }
    return LaunchDescription(arguments + [
        Node(
            package="m050_ros",
            executable="m050_node",
            namespace=LaunchConfiguration("namespace"),
            parameters=[parameters],
            output="screen",
        )
    ])
