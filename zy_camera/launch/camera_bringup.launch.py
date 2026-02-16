"""
Camera bringup launch file for RealSense D435.

Launches realsense2_camera_node with production-validated configuration
and remaps topics to match zy_vision expectations.

Topic Remapping:
  /camera/camera/... -> /camera/...

Published Topics (after remap):
  /camera/color/image_raw                    - RGB image
  /camera/color/camera_info                  - Color camera intrinsics
  /camera/aligned_depth_to_color/image_raw   - Depth aligned to color
  /camera/aligned_depth_to_color/camera_info - Aligned depth intrinsics
  /camera/depth/image_rect_raw               - Raw depth image
  /camera/depth/camera_info                  - Depth camera intrinsics
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

    # Path to config file
    config_file = os.path.join(pkg_share, "config", "camera_params.yaml")

    # Launch arguments
    declare_config_arg = DeclareLaunchArgument(
        "config_file",
        default_value=config_file,
        description="Path to camera configuration YAML file",
    )

    # Realsense camera node with topic remapping
    # Remap /camera/camera/... -> /camera/... for cleaner topic names
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="realsense2_camera",
        parameters=[LaunchConfiguration("config_file")],
        remappings=[
            # Color stream
            ("/camera/camera/color/image_raw", "/camera/color/image_raw"),
            ("/camera/camera/color/camera_info", "/camera/color/camera_info"),
            # Depth stream (raw)
            ("/camera/camera/depth/image_rect_raw", "/camera/depth/image_rect_raw"),
            ("/camera/camera/depth/camera_info", "/camera/depth/camera_info"),
            # Aligned depth to color (most important for grasping)
            (
                "/camera/camera/aligned_depth_to_color/image_raw",
                "/camera/aligned_depth_to_color/image_raw",
            ),
            (
                "/camera/camera/aligned_depth_to_color/camera_info",
                "/camera/aligned_depth_to_color/camera_info",
            ),
            # Depth-to-color extrinsics
            (
                "/camera/camera/extrinsics/depth_to_color",
                "/camera/extrinsics/depth_to_color",
            ),
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            declare_config_arg,
            realsense_node,
        ]
    )
