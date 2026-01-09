#!/usr/bin/env python3
"""
相机节点：发布RGB和深度图像
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

import camera


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # 声明参数
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('publish_rate', 10.0)  # Hz
        
        # 获取参数
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        rate = self.get_parameter('publish_rate').value
        
        # 初始化相机
        self.get_logger().info(f'初始化RealSense相机: {width}x{height}')
        self.camera = camera.RS(width, height)
        
        # 创建发布者
        self.color_pub = self.create_publisher(Image, 'camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, 'camera/depth/image_raw', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 定时器
        self.timer = self.create_timer(1.0 / rate, self.publish_images)
        
        self.get_logger().info('相机节点已启动')
    
    def publish_images(self):
        """发布图像"""
        try:
            # 获取图像
            depth_img, color_img = self.camera.get_img()
            
            # 转换为ROS消息
            color_msg = self.bridge.cv2_to_imgmsg(color_img, encoding='bgr8')
            depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding='passthrough')
            
            # 设置时间戳
            timestamp = self.get_clock().now().to_msg()
            color_msg.header.stamp = timestamp
            color_msg.header.frame_id = 'camera_color_optical_frame'
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = 'camera_depth_optical_frame'
            
            # 发布
            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)
            
        except Exception as e:
            self.get_logger().error(f'发布图像失败: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

