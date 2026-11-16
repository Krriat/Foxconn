from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

LEFT_JOINT_NAMES = tuple(f"left_joint_{index}.pos" for index in range(1, 8))
RIGHT_JOINT_NAMES = tuple(f"right_joint_{index}.pos" for index in range(1, 8))
STATE_NAMES = LEFT_JOINT_NAMES + ("left_gripper.pos",) + RIGHT_JOINT_NAMES + (
    "right_gripper.pos",
)
ACTION_NAMES = STATE_NAMES
SUCTION_STATE_NAMES = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES + (
    "left_suction.on", "right_suction.on",
)


def infer_state_layout(info: dict) -> str:
    """Validate names before assigning the 16 values to the two arm heads."""
    features = info.get("features", {})
    state = tuple(features.get("observation.state", {}).get("names") or ())
    action = tuple(features.get("action", {}).get("names") or ())
    if state != action:
        raise ValueError("state/action feature names must match")
    if state == STATE_NAMES:
        return "bimanual_gripper"
    if state == SUCTION_STATE_NAMES:
        return "dual_arm_suction"
    raise ValueError("unsupported 16-D state/action names; expected dual grippers or dual suction")

HEAD_IMAGE_KEY = "observation.images.head"
LEFT_WRIST_IMAGE_KEY = "observation.images.left_wrist"
RIGHT_WRIST_IMAGE_KEY = "observation.images.right_wrist"
HEAD_POINT_CLOUD_KEY = "observation.pointcloud.head"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

LEFT_ROBOT_IP = "192.168.1.160"
LEFT_LOCAL_IP = "192.168.1.10"
RIGHT_ROBOT_IP = "192.168.2.160"
RIGHT_LOCAL_IP = "192.168.2.10"

LEFT_RESET_JOINTS = (
    1.66969,
    1.365856,
    -1.272963,
    1.476893,
    -1.735811,
    -0.026802,
    -0.165558,
)
RIGHT_RESET_JOINTS = (
    -1.652252,
    1.369718,
    1.524893,
    0.984033,
    0.261897,
    0.196173,
    -0.145317,
)

HEAD_IMAGE_TOPIC = "/robot/sensors/cameras/head/color/image_raw"
LEFT_WRIST_IMAGE_TOPIC = "/robot/sensors/cameras/left_wrist/color/image_raw"
RIGHT_WRIST_IMAGE_TOPIC = "/robot/sensors/cameras/right_wrist/color/image_raw"
HEAD_DEPTH_TOPIC = "/robot/sensors/cameras/head/aligned_depth_to_color/image_raw"
HEAD_DEPTH_INFO_TOPIC = "/robot/sensors/cameras/head/aligned_depth_to_color/camera_info"


@dataclass(frozen=True)
class ArmConfig:
    side: str
    robot_ip: str
    local_ip: str
    reset_joints: tuple[float, ...]
    gripper_enabled: bool
    gripper_command_topic: str
    gripper_state_topic: str
    gripper_open_position: float
    gripper_closed_position: float
    gripper_placeholder: float


@dataclass(frozen=True)
class CameraConfig:
    use_point_cloud: bool = False
    head_image_topic: str = HEAD_IMAGE_TOPIC
    left_wrist_image_topic: str = LEFT_WRIST_IMAGE_TOPIC
    right_wrist_image_topic: str = RIGHT_WRIST_IMAGE_TOPIC
    head_depth_topic: str = HEAD_DEPTH_TOPIC
    head_depth_info_topic: str = HEAD_DEPTH_INFO_TOPIC
    head_width: int = 960
    head_height: int = 540
    wrist_width: int = 640
    wrist_height: int = 480
    fps: int = 30
    point_count: int = 1024
    min_depth_m: float = 0.15
    max_depth_m: float = 2.0
    connect_timeout_s: float = 10.0
    frame_timeout_s: float = 1.0
    max_frame_skew_s: float = 0.15


@dataclass(frozen=True)
class HardwareConfig:
    left: ArmConfig = field(
        default_factory=lambda: ArmConfig(
            side="left",
            robot_ip=LEFT_ROBOT_IP,
            local_ip=LEFT_LOCAL_IP,
            reset_joints=LEFT_RESET_JOINTS,
            gripper_enabled=True,
            gripper_command_topic="/robot/gripper/left/command/position",
            gripper_state_topic="/robot/gripper/left/state",
            gripper_open_position=4000.0,
            gripper_closed_position=3600.0,
            gripper_placeholder=3733.0,
        )
    )
    right: ArmConfig = field(
        default_factory=lambda: ArmConfig(
            side="right",
            robot_ip=RIGHT_ROBOT_IP,
            local_ip=RIGHT_LOCAL_IP,
            reset_joints=RIGHT_RESET_JOINTS,
            gripper_enabled=True,
            gripper_command_topic="/robot/gripper/right/command/position",
            gripper_state_topic="/robot/gripper/right/state",
            gripper_open_position=1500.0,
            gripper_closed_position=900.0,
            gripper_placeholder=904.0,
        )
    )
    cameras: CameraConfig = field(default_factory=CameraConfig)
    sdk_python_path: Path = Path(
        "/home/niic/xcoresdk_python-v0.7.1.ar_6/Release/linux/arm"
    )


def dataset_features(camera: CameraConfig) -> dict:
    """LeRobot RGB schema, with optional XYZRGB point-cloud rows."""
    features = {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
        HEAD_IMAGE_KEY: {
            "dtype": "video",
            "shape": (camera.head_height, camera.head_width, 3),
            "names": None,
        },
        LEFT_WRIST_IMAGE_KEY: {
            "dtype": "video",
            "shape": (camera.wrist_height, camera.wrist_width, 3),
            "names": None,
        },
        RIGHT_WRIST_IMAGE_KEY: {
            "dtype": "video",
            "shape": (camera.wrist_height, camera.wrist_width, 3),
            "names": None,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": list(ACTION_NAMES),
        },
    }
    if camera.use_point_cloud:
        features[HEAD_POINT_CLOUD_KEY] = {
            "dtype": "float32", "shape": (camera.point_count, 6), "names": None,
        }
    return features
