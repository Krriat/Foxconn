from setuptools import find_packages, setup

package_name = 'robot_vision_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhenguo',
    maintainer_email='zhenguo.jia@auto-jiangjin.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'udp_video_sender = robot_vision_tools.udp_video_sender:main',
            'rgbd_probe = robot_vision_tools.rgbd_probe:main',
        ],
    },
)
