"""
Depth visualizer launch file.

Launches the depth_visualizer node to convert depth images
to pseudo-color images for better visualization.

Usage:
    ros2 launch zy_camera depth_visualizer.launch.py

Published Topics:
    /camera/depth_colored/image_raw (sensor_msgs/Image, bgr8)
    - Pseudo-color visualization of depth data
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directory
    pkg_share = get_package_share_directory("zy_camera")

    # Default config file path
    default_config = os.path.join(pkg_share, "config", "depth_viz_params.yaml")

    # Launch arguments
    declare_config_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to depth visualizer configuration YAML file",
    )

    declare_input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/camera/aligned_depth_to_color/image_raw",
        description="Input depth image topic",
    )

    declare_output_topic_arg = DeclareLaunchArgument(
        "output_topic",
        default_value="/camera/depth_colored/image_raw",
        description="Output colorized depth image topic",
    )

    # Depth visualizer node
    depth_visualizer_node = Node(
        package="zy_camera",
        executable="depth_visualizer",
        name="depth_visualizer",
        parameters=[
            LaunchConfiguration("config_file"),
            {
                "input_topic": LaunchConfiguration("input_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
            },
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            declare_config_arg,
            declare_input_topic_arg,
            declare_output_topic_arg,
            depth_visualizer_node,
        ]
    )
