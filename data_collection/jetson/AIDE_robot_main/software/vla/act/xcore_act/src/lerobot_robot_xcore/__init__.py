"""LeRobot third-party robot plugin for the ROKAE xCore arms."""

from .compat import (
    patch_accelerate_custom_kernel_check,
    patch_accelerate_without_torch_distributed,
)

patch_accelerate_custom_kernel_check()
patch_accelerate_without_torch_distributed()

from .config_xcore import XCoreLeftArmConfig, XCoreRightArmConfig
from .config_ros_image import RosImageCameraConfig
from .ros_image_camera import RosImageCamera
from .xcorerightarm import XCoreLeftArm, XCoreRightArm

__all__ = [
    "RosImageCamera",
    "RosImageCameraConfig",
    "XCoreLeftArm",
    "XCoreLeftArmConfig",
    "XCoreRightArm",
    "XCoreRightArmConfig",
]
