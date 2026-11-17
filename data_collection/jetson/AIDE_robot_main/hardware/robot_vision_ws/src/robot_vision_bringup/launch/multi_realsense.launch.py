#!/usr/bin/env python3

"""从统一 YAML 配置启动多个官方 RealSense 节点和相机管理节点。"""

from __future__ import annotations

import os
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    EmitEvent,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from robot_vision_core.camera_registry import load_camera_registry


def to_launch_bool(value: Any) -> str:
    """将 YAML 值转换为 ROS launch 使用的 true/false 字符串。"""
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return "true" if value != 0 else "false"

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes", "on"}:
            return "true"

        if normalized in {"false", "0", "no", "off", ""}:
            return "false"

    raise ValueError(
        f"Cannot convert value to launch boolean: {value!r}"
    )


def load_camera_config(
    registry_path: str,
) -> dict[str, Any]:
    """读取并检查统一相机注册 YAML。"""
    return load_camera_registry(registry_path)


def validate_camera_config(
    camera_id: str,
    camera_config: dict[str, Any],
) -> None:
    """检查一台已启用相机的必要字段。"""
    camera_name = str(
        camera_config.get("camera_name", "")
    ).strip("/")

    serial_no = str(
        camera_config.get("serial_no", "")
    ).strip()

    if not camera_name:
        raise RuntimeError(
            f"Enabled camera '{camera_id}' has no camera_name"
        )

    if not serial_no:
        raise RuntimeError(
            f"Enabled camera '{camera_id}' has no serial_no"
        )

    streams = camera_config.get("streams")

    if not isinstance(streams, dict):
        raise RuntimeError(
            f"Enabled camera '{camera_id}' has no valid "
            "'streams' dictionary"
        )

    profiles = camera_config.get("profiles", {})

    if profiles is not None and not isinstance(
        profiles,
        dict,
    ):
        raise RuntimeError(
            f"Camera '{camera_id}' field 'profiles' "
            "must be a dictionary"
        )


def build_realsense_launch_arguments(
    camera_namespace: str,
    camera_id: str,
    camera_config: dict[str, Any],
) -> dict[str, str]:
    """把单台相机配置转换为官方 rs_launch.py 参数。"""
    validate_camera_config(
        camera_id,
        camera_config,
    )

    camera_name = str(
        camera_config.get(
            "camera_name",
            camera_id,
        )
    ).strip("/")

    serial_no = str(
        camera_config["serial_no"]
    ).strip()

    # 数字序列号加下划线，避免被解析成整数。
    if not serial_no.startswith("_"):
        serial_no = f"_{serial_no}"

    model = str(
        camera_config.get(
            "model",
            "",
        )
    ).strip().upper()

    streams = camera_config.get(
        "streams",
        {},
    )

    launch_arguments: dict[str, str] = {
        "camera_namespace": camera_namespace,
        "camera_name": camera_name,
        "serial_no": serial_no,
        "enable_color": to_launch_bool(
            streams.get("enable_color", True)
        ),
        "enable_depth": to_launch_bool(
            streams.get("enable_depth", True)
        ),
        "align_depth.enable": to_launch_bool(
            streams.get("align_depth", True)
        ),
        "pointcloud.enable": to_launch_bool(
            streams.get("pointcloud", False)
        ),
        "enable_sync": to_launch_bool(
            streams.get("enable_sync", True)
        ),
        "diagnostics_period": str(
            camera_config.get(
                "diagnostics_period",
                0.0,
            )
        ),
    }

    # 只有 YAML 明确配置 profiles 时，才覆盖官方默认值。
    profiles = camera_config.get(
        "profiles",
        {},
    ) or {}

    color_profile = str(
        profiles.get("color", "")
    ).strip()

    depth_profile = str(
        profiles.get("depth", "")
    ).strip()

    if color_profile:
        if model == "D405":
            launch_arguments[
                "depth_module.color_profile"
            ] = color_profile
        else:
            launch_arguments[
                "rgb_camera.color_profile"
            ] = color_profile

    if depth_profile:
        launch_arguments[
            "depth_module.depth_profile"
        ] = depth_profile

    return launch_arguments


def create_scoped_realsense_include(
    official_launch_path: str,
    launch_arguments: dict[str, str],
) -> GroupAction:
    """在独立作用域中启动官方驱动，直接传递完整的相机参数。"""
    # The installed 4.55.1 driver supports D405 depth_module.color_profile,
    # but its rs_launch.py does not declare that argument. Pass parameters
    # straight to the official executable so no registry setting is lost.
    del official_launch_path
    parameters = {}
    for name, value in launch_arguments.items():
        if name in ("camera_namespace", "camera_name"):
            continue
        if value in ("true", "false"):
            parameters[name] = value == "true"
        elif name == "diagnostics_period":
            parameters[name] = float(value)
        else:
            parameters[name] = ParameterValue(value, value_type=str)
    parameters["camera_name"] = launch_arguments["camera_name"]
    # rs_launch normally supplies these false defaults. The executable's
    # defaults enable infrared streams, which are outside this RGB-D contract.
    parameters.update(enable_infra=False, enable_infra1=False, enable_infra2=False,
                      enable_gyro=False, enable_accel=False)
    camera_root = "/" + launch_arguments["camera_namespace"].strip("/") + "/" + launch_arguments["camera_name"]
    return GroupAction(scoped=True, forwarding=False, actions=[Node(
        package="realsense2_camera", executable="realsense2_camera_node",
        namespace=launch_arguments["camera_namespace"],
        name=launch_arguments["camera_name"],
        parameters=[parameters], output="screen",
        # D405 color is rectified by the SDK; retain one canonical image
        # topic across camera models instead of adding a second image relay.
        remappings=[(camera_root + "/color/image_rect_raw",
                     camera_root + "/color/image_raw"),
                    ("/diagnostics", "/" + launch_arguments["camera_namespace"].strip("/") + "/diagnostics"),
                    ("/tf_static", "/" + launch_arguments["camera_namespace"].strip("/") + "/tf_static")],
    )])


def launch_setup(
    context: LaunchContext,
    *args: Any,
    **kwargs: Any,
) -> list:
    """为 enabled=true 的相机启动官方 RealSense 驱动。"""
    del args, kwargs

    # 注意：这里不再使用 config_file，避免与官方参数重名。
    registry_path = LaunchConfiguration(
        "camera_registry_file"
    ).perform(context)

    config = load_camera_config(
        registry_path
    )

    camera_namespace = str(
        config.get(
            "camera_namespace",
            "robot/sensors/cameras",
        )
    ).strip("/")

    if not camera_namespace:
        raise RuntimeError(
            "camera_namespace cannot be empty"
        )

    official_realsense_launch = os.path.join(
        get_package_share_directory(
            "realsense2_camera"
        ),
        "launch",
        "rs_launch.py",
    )

    if not os.path.isfile(
        official_realsense_launch
    ):
        raise RuntimeError(
            "Official RealSense launch file not found: "
            f"{official_realsense_launch}"
        )

    actions = [
        LogInfo(
            msg=(
                "Loading camera registry: "
                f"{registry_path}"
            )
        ),
        LogInfo(
            msg=(
                "Camera namespace: "
                f"/{camera_namespace}"
            )
        ),
    ]

    enabled_count = 0

    for camera_id, camera_config in config[
        "cameras"
    ].items():
        if not isinstance(
            camera_config,
            dict,
        ):
            raise RuntimeError(
                f"Camera '{camera_id}' configuration "
                "must be a dictionary"
            )

        if not camera_config.get(
            "enabled",
            False,
        ):
            actions.append(
                LogInfo(
                    msg=(
                        f"Camera '{camera_id}' "
                        "is disabled"
                    )
                )
            )
            continue

        enabled_count += 1

        launch_arguments = (
            build_realsense_launch_arguments(
                camera_namespace,
                camera_id,
                camera_config,
            )
        )

        camera_name = launch_arguments[
            "camera_name"
        ]

        serial_no = launch_arguments[
            "serial_no"
        ].lstrip("_")

        actions.append(
            LogInfo(
                msg=(
                    f"Starting camera '{camera_id}': "
                    f"/{camera_namespace}/{camera_name}, "
                    f"serial={serial_no}"
                )
            )
        )

        actions.append(
            create_scoped_realsense_include(
                official_realsense_launch,
                launch_arguments,
            )
        )

    if enabled_count == 0:
        raise RuntimeError(
            "No cameras are enabled in cameras.yaml"
        )

    actions.append(
        LogInfo(
            msg=(
                "Enabled RealSense camera count: "
                f"{enabled_count}"
            )
        )
    )

    return actions


def generate_launch_description() -> LaunchDescription:
    """生成统一视觉设备启动描述。"""
    bringup_share = get_package_share_directory(
        "robot_vision_bringup"
    )

    default_registry_file = os.path.join(
        bringup_share,
        "config",
        "cameras.yaml",
    )

    camera_registry_argument = DeclareLaunchArgument(
        "camera_registry_file",
        default_value=default_registry_file,
        description=(
            "Absolute path to the unified cameras.yaml registry"
        ),
    )

    start_camera_manager_argument = (
        DeclareLaunchArgument(
            "start_camera_manager",
            default_value="true",
            description=(
                "Start robot_vision_core camera_manager"
            ),
        )
    )

    manager_timing_arguments = [
        DeclareLaunchArgument(
            "publish_rate_hz",
            default_value="2.0",
            description="Camera status publication rate",
        ),
        DeclareLaunchArgument(
            "stale_threshold_sec",
            default_value="2.0",
            description="Age at which an enabled stream becomes stale",
        ),
        DeclareLaunchArgument(
            "offline_threshold_sec",
            default_value="10.0",
            description="Age at which a fully silent camera becomes offline",
        ),
        DeclareLaunchArgument(
            "startup_grace_sec",
            default_value="5.0",
            description="Startup grace period before health alarms",
        ),
        DeclareLaunchArgument(
            "hz_window_sec",
            default_value="5.0",
            description="Sliding window used for color frame-rate measurement",
        ),
    ]

    camera_manager_node = Node(
        package="robot_vision_core",
        executable="camera_manager",
        name="camera_manager",
        namespace="robot/vision",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                # camera_manager 内部参数仍叫 config_file，
                # 这里只是把外部 launch 参数传给它。
                "config_file": LaunchConfiguration(
                    "camera_registry_file"
                ),
                "publish_rate_hz": LaunchConfiguration(
                    "publish_rate_hz"
                ),
                "stale_threshold_sec": LaunchConfiguration(
                    "stale_threshold_sec"
                ),
                "offline_threshold_sec": LaunchConfiguration(
                    "offline_threshold_sec"
                ),
                "startup_grace_sec": LaunchConfiguration(
                    "startup_grace_sec"
                ),
                "hz_window_sec": LaunchConfiguration(
                    "hz_window_sec"
                ),
            }
        ],
        condition=IfCondition(
            LaunchConfiguration(
                "start_camera_manager"
            )
        ),
    )

    return LaunchDescription(
        [
            RegisterEventHandler(OnProcessExit(on_exit=lambda event, context:
                [] if context.is_shutdown else [EmitEvent(event=Shutdown(
                    reason=f"Camera process exited ({event.returncode}); restart the camera group"
                ))])),
            camera_registry_argument,
            start_camera_manager_argument,
            *manager_timing_arguments,

            OpaqueFunction(
                function=launch_setup
            ),

            camera_manager_node,
        ]
    )
