#!/usr/bin/env python3
"""
测试图像话题：订阅并验证相机图像
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys


class ImageTopicTester(Node):
    def __init__(self):
        super().__init__('image_topic_tester')
        
        self.bridge = CvBridge()
        
        # 计数器
        self.color_count = 0
        self.depth_count = 0
        
        # 订阅图像话题
        self.color_sub = self.create_subscription(
            Image,
            'camera/color/image_raw',
            self.color_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            'camera/depth/image_raw',
            self.depth_callback,
            10
        )
        
        self.get_logger().info('图像话题测试节点已启动')
        self.get_logger().info('订阅: /camera/color/image_raw')
        self.get_logger().info('订阅: /camera/depth/image_raw')
    
    def color_callback(self, msg):
        """彩色图像回调"""
        self.color_count += 1
        
        if self.color_count % 10 == 1:
            try:
                # 转换为OpenCV格式
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                self.get_logger().info(
                    f'[彩色图像 #{self.color_count}] '
                    f'尺寸: {cv_image.shape[1]}x{cv_image.shape[0]}, '
                    f'编码: {msg.encoding}'
                )
                
                # 可选：保存图像
                if self.color_count == 1:
                    cv2.imwrite('/tmp/test_color_image.png', cv_image)
                    self.get_logger().info('✓ 保存测试图像: /tmp/test_color_image.png')
                
            except Exception as e:
                self.get_logger().error(f'处理彩色图像失败: {e}')
    
    def depth_callback(self, msg):
        """深度图像回调"""
        self.depth_count += 1
        
        if self.depth_count % 10 == 1:
            try:
                # 转换为OpenCV格式
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                
                self.get_logger().info(
                    f'[深度图像 #{self.depth_count}] '
                    f'尺寸: {cv_image.shape[1]}x{cv_image.shape[0]}, '
                    f'编码: {msg.encoding}, '
                    f'深度范围: {cv_image.min():.3f}-{cv_image.max():.3f}米'
                )
                
            except Exception as e:
                self.get_logger().error(f'处理深度图像失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    print("=" * 60)
    print("图像话题测试")
    print("=" * 60)
    print("\n⚠️  请先启动相机节点:")
    print("  ros2 run grasp_vision mock_camera_node")
    print("\n或使用启动文件:")
    print("  ros2 launch grasp_bringup test_mock_system.launch.py")
    print("\n等待图像消息...\n")
    
    node = ImageTopicTester()
    
    try:
        # 运行30秒
        start_time = node.get_clock().now()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            elapsed = (node.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > 30:
                break
        
        print("\n" + "=" * 60)
        print("测试结果:")
        print("=" * 60)
        print(f"✓ 收到彩色图像: {node.color_count} 帧")
        print(f"✓ 收到深度图像: {node.depth_count} 帧")
        
        if node.color_count > 0 and node.depth_count > 0:
            print("\n🎉 图像话题测试通过！")
            sys.exit(0)
        else:
            print("\n❌ 未收到图像消息，请检查相机节点是否运行")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n测试被中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

