from setuptools import setup
import os
from glob import glob

package_name = 'grasp_vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jet',
    maintainer_email='jet@zoneyung.com',
    description='视觉处理节点：相机、目标检测、抓取姿态生成',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = grasp_vision.camera_node:main',
            'mock_camera_node = grasp_vision.mock_camera_node:main',
            'detection_server = grasp_vision.detection_server:main',
            'grasp_generator = grasp_vision.grasp_generator:main',
        ],
    },
)

