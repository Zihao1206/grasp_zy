#!/usr/bin/env python3
"""
测试真实RealSense相机
验证相机节点是否正常工作
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import time


class RealCameraTester(Node):
    def __init__(self):
        super().__init__('real_camera_tester')
        
        self.bridge = CvBridge()
        
        # 统计信息
        self.color_count = 0
        self.depth_count = 0
        self.start_time = time.time()
        
        # 图像数据
        self.latest_color = None
        self.latest_depth = None
        
        # 订阅话题
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
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('真实相机测试节点已启动')
        self.get_logger().info('=' * 60)
        self.get_logger().info('等待图像数据...')
    
    def color_callback(self, msg):
        """彩色图像回调"""
        self.color_count += 1
        
        try:
            # 转换为OpenCV格式
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if self.color_count == 1:
                self.get_logger().info(
                    f'\n✓ 收到第一帧彩色图像:\n'
                    f'  尺寸: {self.latest_color.shape[1]}x{self.latest_color.shape[0]}\n'
                    f'  编码: {msg.encoding}\n'
                    f'  帧ID: {msg.header.frame_id}'
                )
                # 保存第一帧
                cv2.imwrite('/tmp/ros2_color_first.jpg', self.latest_color)
                self.get_logger().info('  已保存: /tmp/ros2_color_first.jpg')
            
            if self.color_count % 30 == 0:
                elapsed = time.time() - self.start_time
                fps = self.color_count / elapsed
                self.get_logger().info(f'[彩色] 收到 {self.color_count} 帧, FPS: {fps:.2f}')
                
        except Exception as e:
            self.get_logger().error(f'处理彩色图像失败: {e}')
    
    def depth_callback(self, msg):
        """深度图像回调"""
        self.depth_count += 1
        
        try:
            # 转换为OpenCV格式
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            if self.depth_count == 1:
                self.get_logger().info(
                    f'\n✓ 收到第一帧深度图像:\n'
                    f'  尺寸: {self.latest_depth.shape[1]}x{self.latest_depth.shape[0]}\n'
                    f'  编码: {msg.encoding}\n'
                    f'  深度范围: {self.latest_depth.min():.3f} - {self.latest_depth.max():.3f} 米'
                )
                
                # 保存深度图（归一化为可视化）
                depth_normalized = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite('/tmp/ros2_depth_first.jpg', depth_colored)
                self.get_logger().info('  已保存: /tmp/ros2_depth_first.jpg')
            
            if self.depth_count % 30 == 0:
                elapsed = time.time() - self.start_time
                fps = self.depth_count / elapsed
                self.get_logger().info(f'[深度] 收到 {self.depth_count} 帧, FPS: {fps:.2f}')
                
        except Exception as e:
            self.get_logger().error(f'处理深度图像失败: {e}')
    
    def print_summary(self):
        """打印测试摘要"""
        elapsed = time.time() - self.start_time
        
        print('\n' + '=' * 60)
        print('测试摘要')
        print('=' * 60)
        print(f'测试时长: {elapsed:.2f} 秒')
        print(f'彩色图像: {self.color_count} 帧 (平均 {self.color_count/elapsed:.2f} FPS)')
        print(f'深度图像: {self.depth_count} 帧 (平均 {self.depth_count/elapsed:.2f} FPS)')
        
        if self.latest_color is not None:
            print(f'\n彩色图像信息:')
            print(f'  尺寸: {self.latest_color.shape}')
            print(f'  数据类型: {self.latest_color.dtype}')
        
        if self.latest_depth is not None:
            print(f'\n深度图像信息:')
            print(f'  尺寸: {self.latest_depth.shape}')
            print(f'  深度范围: {self.latest_depth.min():.3f} - {self.latest_depth.max():.3f} 米')
            print(f'  数据类型: {self.latest_depth.dtype}')
        
        print('\n保存的文件:')
        print('  /tmp/ros2_color_first.jpg  (彩色图像)')
        print('  /tmp/ros2_depth_first.jpg  (深度图像-可视化)')
        print('=' * 60)
        
        # 判断测试结果
        if self.color_count > 0 and self.depth_count > 0:
            print('\n✓✓✓ 相机测试通过！所有图像正常接收')
            return True
        else:
            print('\n✗✗✗ 相机测试失败！未收到图像数据')
            print('请检查:')
            print('  1. 相机节点是否正在运行')
            print('  2. RealSense相机是否正确连接')
            print('  3. 话题名称是否正确')
            return False


def main(args=None):
    rclpy.init(args=args)
    
    print('=' * 60)
    print('RealSense 相机 ROS2 测试')
    print('=' * 60)
    print('\n⚠️  请确保已启动相机节点:')
    print('  ros2 run grasp_vision camera_node')
    print('\n测试将运行 30 秒...\n')
    
    node = RealCameraTester()
    
    try:
        # 运行30秒
        start = node.get_clock().now()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            elapsed = (node.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > 30:
                break
        
        # 打印摘要
        success = node.print_summary()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print('\n\n测试被中断')
        node.print_summary()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

