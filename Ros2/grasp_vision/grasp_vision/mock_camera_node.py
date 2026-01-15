#!/usr/bin/env python3
"""
模拟相机节点：用于测试，无需真实硬件
发布测试图像（纯色或随机噪声）
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class MockCameraNode(Node):
    def __init__(self):
        super().__init__('mock_camera_node')
        
        # 声明参数
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('image_type', 'test_pattern')  # test_pattern, noise, checkerboard
        
        # 获取参数
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        rate = self.get_parameter('publish_rate').value
        self.image_type = self.get_parameter('image_type').value
        
        self.get_logger().info(f'初始化模拟相机: {self.width}x{self.height}, 类型: {self.image_type}')
        
        # 创建发布者
        self.color_pub = self.create_publisher(Image, 'camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, 'camera/depth/image_raw', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 计数器用于动画效果
        self.frame_count = 0
        
        # 定时器
        self.timer = self.create_timer(1.0 / rate, self.publish_images)
        
        self.get_logger().info('模拟相机节点已启动 (无需真实硬件)')
    
    def generate_color_image(self):
        """生成彩色测试图像"""
        if self.image_type == 'test_pattern':
            # 彩色渐变测试图案
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for i in range(self.height):
                for j in range(self.width):
                    img[i, j, 0] = int(255 * i / self.height)  # R
                    img[i, j, 1] = int(255 * j / self.width)   # G
                    img[i, j, 2] = int((self.frame_count % 255))  # B (动画)
        
        elif self.image_type == 'noise':
            # 随机噪声
            img = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        
        elif self.image_type == 'checkerboard':
            # 棋盘格
            square_size = 50
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for i in range(0, self.height, square_size):
                for j in range(0, self.width, square_size):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        img[i:i+square_size, j:j+square_size] = [255, 255, 255]
        
        else:
            # 默认：蓝色背景
            img = np.full((self.height, self.width, 3), [200, 150, 100], dtype=np.uint8)
        
        # 添加文本标记
        text = f"Mock Camera Frame {self.frame_count}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img
    
    def generate_depth_image(self):
        """生成深度测试图像（模拟深度数据）"""
        # 生成模拟深度数据（单位：米）
        # 中心近，边缘远
        y, x = np.ogrid[:self.height, :self.width]
        cy, cx = self.height / 2, self.width / 2
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        depth = 0.3 + (distance / np.max(distance)) * 0.5  # 0.3-0.8米
        
        return depth.astype(np.float32)
    
    def publish_images(self):
        """发布图像"""
        try:
            # 生成图像
            color_img = self.generate_color_image()
            depth_img = self.generate_depth_image()
            
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
            
            self.frame_count += 1
            
            if self.frame_count % 100 == 0:
                self.get_logger().info(f'已发布 {self.frame_count} 帧图像')
            
        except Exception as e:
            self.get_logger().error(f'发布图像失败: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = MockCameraNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

