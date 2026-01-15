#!/usr/bin/env python3
"""
实时显示相机图像（带窗口）
类似直接运行 camera.py 的效果
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraDisplayNode(Node):
    def __init__(self):
        super().__init__('camera_display_node')
        
        self.bridge = CvBridge()
        
        # 最新图像
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
        self.get_logger().info('相机显示节点已启动')
        self.get_logger().info('=' * 60)
        self.get_logger().info('等待图像数据...')
        self.get_logger().info('')
        self.get_logger().info('窗口操作:')
        self.get_logger().info('  按 "q" 退出')
        self.get_logger().info('  按 "s" 保存当前帧')
        self.get_logger().info('=' * 60)
        
        self.frame_count = 0
        self.save_count = 0
        
        # 创建窗口
        cv2.namedWindow('ROS2 Camera - Color', cv2.WINDOW_NORMAL)
        cv2.namedWindow('ROS2 Camera - Depth', cv2.WINDOW_NORMAL)
        
        # 定时器用于显示图像
        self.timer = self.create_timer(0.03, self.display_images)  # 30Hz
    
    def color_callback(self, msg):
        """彩色图像回调"""
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f'处理彩色图像失败: {e}')
    
    def depth_callback(self, msg):
        """深度图像回调"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'处理深度图像失败: {e}')
    
    def display_images(self):
        """显示图像"""
        if self.latest_color is not None:
            # 显示彩色图像
            display_color = self.latest_color.copy()
            
            # 添加文字信息
            text = f'Frame: {self.frame_count} | Press "q" to quit, "s" to save'
            cv2.putText(display_color, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('ROS2 Camera - Color', display_color)
        
        if self.latest_depth is not None:
            # 深度图像可视化
            depth_normalized = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
            # 添加深度信息
            min_depth = self.latest_depth[self.latest_depth > 0].min() if np.any(self.latest_depth > 0) else 0
            max_depth = self.latest_depth.max()
            text = f'Depth: {min_depth:.2f}m - {max_depth:.2f}m'
            cv2.putText(depth_colored, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('ROS2 Camera - Depth', depth_colored)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('退出...')
            rclpy.shutdown()
        elif key == ord('s'):
            if self.latest_color is not None and self.latest_depth is not None:
                # 保存图像
                color_file = f'/tmp/ros2_saved_color_{self.save_count}.jpg'
                depth_file = f'/tmp/ros2_saved_depth_{self.save_count}.jpg'
                
                cv2.imwrite(color_file, self.latest_color)
                
                depth_normalized = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(depth_file, depth_colored)
                
                self.save_count += 1
                self.get_logger().info(f'✓ 已保存图像: {color_file}, {depth_file}')


def main(args=None):
    rclpy.init(args=args)
    
    print('=' * 60)
    print('ROS2 相机实时显示')
    print('=' * 60)
    print('\n⚠️  请确保相机节点正在运行:')
    print('  ros2 run grasp_vision camera_node\n')
    
    node = CameraDisplayNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

