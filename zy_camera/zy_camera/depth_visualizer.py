#!/usr/bin/env python3
"""
Depth image to pseudo-color visualization node.

Converts depth images (16UC1, millimeters) to colorized images (bgr8)
using OpenCV colormap for better visualization.

Subscribes:
    /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image, 16UC1)

Publishes:
    /camera/depth_colored/image_raw (sensor_msgs/Image, bgr8)

Parameters:
    min_depth (float): Minimum depth for normalization in meters (default: 0.0)
    max_depth (float): Maximum depth for normalization in meters (default: 1.0)
    colormap (int): OpenCV colormap ID (default: 5 = COLORMAP_TURBO)
    depth_scale (float): Scale factor to convert raw depth to meters (default: 0.001)

Usage:
    ros2 run zy_camera depth_visualizer
    # or
    ros2 launch zy_camera depth_visualizer.launch.py
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# OpenCV colormap constants for reference
COLORMAP_AUTUMN = 0
COLORMAP_BONE = 1
COLORMAP_JET = 2
COLORMAP_WINTER = 3
COLORMAP_RAINBOW = 4
COLORMAP_OCEAN = 5
COLORMAP_SUMMER = 6
COLORMAP_SPRING = 7
COLORMAP_COOL = 8
COLORMAP_HSV = 9
COLORMAP_PINK = 10
COLORMAP_HOT = 11
COLORMAP_PARULA = 12
COLORMAP_MAGMA = 13
COLORMAP_INFERNO = 14
COLORMAP_PLASMA = 15
COLORMAP_VIRIDIS = 16
COLORMAP_CIVIDIS = 17
COLORMAP_TWILIGHT = 18
COLORMAP_TWILIGHT_SHIFTED = 19
COLORMAP_TURBO = 20
COLORMAP_DEEPGREEN = 21


class DepthVisualizer(Node):
    """ROS2 node that converts depth images to pseudo-color images."""

    def __init__(self):
        super().__init__("depth_visualizer")

        # Declare parameters with defaults
        self.declare_parameter("min_depth", 0.0)
        self.declare_parameter("max_depth", 1.0)
        self.declare_parameter("colormap", cv2.COLORMAP_TURBO)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter(
            "input_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("output_topic", "/camera/depth_colored/image_raw")

        # Get parameters
        self.min_depth = self.get_parameter("min_depth").value
        self.max_depth = self.get_parameter("max_depth").value
        self.colormap = self.get_parameter("colormap").value
        self.depth_scale = self.get_parameter("depth_scale").value
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        # Validate depth range
        if self.max_depth <= self.min_depth:
            self.get_logger().warn(
                f"max_depth ({self.max_depth}) must be greater than min_depth ({self.min_depth}). "
                "Using defaults: [0.0, 1.0]"
            )
            self.min_depth = 0.0
            self.max_depth = 1.0

        # CV Bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()

        # Subscriber for depth image
        self.subscription = self.create_subscription(
            Image, input_topic, self.depth_callback, 10
        )

        # Publisher for colorized depth image
        self.publisher = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(
            f"DepthVisualizer initialized:\n"
            f"  Input:  {input_topic}\n"
            f"  Output: {output_topic}\n"
            f"  Range:  [{self.min_depth}, {self.max_depth}] meters\n"
            f"  Colormap: {self.colormap} (TURBO=20, JET=2, INFERNO=14)\n"
            f"  Scale:  {self.depth_scale} (mm->m)"
        )

    def depth_callback(self, msg: Image) -> None:
        """Process incoming depth image and publish colorized version."""
        try:
            # Convert ROS Image to numpy array
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Handle different encodings
            if msg.encoding == "16UC1":
                # uint16 millimeters -> float32 meters
                depth_meters = depth_raw.astype(np.float32) * self.depth_scale
            elif msg.encoding == "32FC1":
                # Already float32 meters
                depth_meters = depth_raw.astype(np.float32)
            elif msg.encoding == "mono16":
                # uint16, assume millimeters
                depth_meters = depth_raw.astype(np.float32) * self.depth_scale
            else:
                self.get_logger().warn(
                    f"Unsupported encoding: {msg.encoding}. Expected 16UC1 or 32FC1."
                )
                return

            # Normalize depth to [0, 1] with clipping
            # Formula: (depth - min) / (max - min)
            depth_range = self.max_depth - self.min_depth
            if depth_range > 0:
                normalized = np.clip(
                    (depth_meters - self.min_depth) / depth_range, 0.0, 1.0
                )
            else:
                normalized = np.zeros_like(depth_meters)

            # Scale to [0, 255] as uint8
            depth_uint8 = (normalized * 255).astype(np.uint8)

            # Apply colormap (output is BGR)
            colored_depth = cv2.applyColorMap(depth_uint8, self.colormap)

            # Convert back to ROS Image message
            out_msg = self.bridge.cv2_to_imgmsg(colored_depth, encoding="bgr8")
            out_msg.header = msg.header  # Preserve timestamp and frame_id

            # Publish
            self.publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")


def main(args=None):
    """Entry point for the depth visualizer node."""
    rclpy.init(args=args)

    try:
        node = DepthVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
