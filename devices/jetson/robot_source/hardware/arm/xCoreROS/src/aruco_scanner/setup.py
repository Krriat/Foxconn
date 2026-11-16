from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'aruco_scanner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=[
        'setuptools',
        'fsspec',
        'numpy>=1.23.5,<2',
        'opencv-python>=4.6,<4.12',
        'ultralytics',
    ],
    zip_safe=True,
    maintainer='zhang',
    maintainer_email='zhang@todo.todo',
    description='Intel D405 ArUco and YOLO target detection for ROS 2.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco_scanner_node = aruco_scanner.aruco_scanner_node:main',
            'arucode_position_node = '
            'aruco_scanner.arucode_position_detection:main',
            'big_box_detection_node = aruco_scanner.big_box_detection:main',
            'small_box_detection_node = '
            'aruco_scanner.small_box_detection:main',
        ],
    },
)
