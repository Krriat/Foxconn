from setuptools import find_packages, setup
from glob import glob

package_name = "robot_vision_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(
        exclude=["test"],
    ),
    data_files=[
        ('share/robot_vision_core/launch', glob('launch/*.launch.py')),
        ('share/robot_vision_core/models', glob('models/*.pt')),
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=[
        "setuptools",
        "PyYAML",
    ],
    zip_safe=True,
    maintainer="zhenguo",
    maintainer_email="zhenguo.jia@auto-jiangjin.com",
    description=(
        "Core camera registry and common vision services "
        "for the robot vision system."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "big_box_detection = robot_vision_core.big_box_detection:main",
            "small_box_detection = robot_vision_core.small_box_detection:main",
            "visual_manager = robot_vision_core.visual_manager:main",
            (
                "camera_manager = "
                "robot_vision_core.camera_manager:main"
            ),
            (
                "aruco_scanner = "
                "robot_vision_core.aruco_scanner:main"
            ),
            (
                "arucode_position_detection = "
                "robot_vision_core.arucode_position_detection:main"
            ),
            (
                "position_detection = "
                "robot_vision_core.position_detection:main"
            ),
        ],
    },
)
