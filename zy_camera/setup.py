from setuptools import setup
import os
from glob import glob

package_name = "zy_camera"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index_packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Launch files installed by CMakeLists.txt
        # Config files installed by CMakeLists.txt
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kai Wang",
    maintainer_email="kai-wang@zju.edu.cn",
    description="RealSense D435 camera configuration package for zy grasping system",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "depth_visualizer = zy_camera.depth_visualizer:main",
        ],
    },
)
