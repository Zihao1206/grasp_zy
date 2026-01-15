#!/usr/bin/env python3
"""
测试抓取姿态生成服务
"""
import rclpy
from rclpy.node import Node
from grasp_interfaces.srv import DetectObjects, GenerateGrasp
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import math


class GraspGenerationTester(Node):
    def __init__(self):
        super().__init__('grasp_generation_tester')
        
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        
        # 订阅相机图像
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
        
        # 创建服务客户端
        self.detect_client = self.create_client(DetectObjects, 'detect_objects')
        self.grasp_client = self.create_client(GenerateGrasp, 'generate_grasp')
        
        self.get_logger().info('抓取生成测试节点已启动')
    
    def color_callback(self, msg):
        self.latest_color = msg
    
    def depth_callback(self, msg):
        self.latest_depth = msg
    
    def test_grasp_generation(self, target_label='carrot'):
        """测试抓取姿态生成"""
        print('\n' + '=' * 60)
        print('测试抓取姿态生成服务')
        print('=' * 60)
        
        # 1. 等待图像
        print(f'\n目标物体: {target_label}')
        print('\n[1/4] 等待相机图像...')
        timeout = 10
        for i in range(timeout * 10):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_color is not None and self.latest_depth is not None:
                break
        
        if self.latest_color is None or self.latest_depth is None:
            print('✗ 未收到图像')
            return False
        
        print('✓ 已收到图像')
        
        # 2. 目标检测
        print('\n[2/4] 执行目标检测...')
        if not self.detect_client.wait_for_service(timeout_sec=5.0):
            print('✗ 检测服务不可用')
            return False
        
        detect_req = DetectObjects.Request()
        detect_req.color_image = self.latest_color
        
        detect_future = self.detect_client.call_async(detect_req)
        rclpy.spin_until_future_complete(self, detect_future, timeout_sec=30.0)
        
        if not detect_future.done():
            print('✗ 检测超时')
            return False
        
        detect_resp = detect_future.result()
        if not detect_resp.success:
            print(f'✗ 检测失败: {detect_resp.message}')
            return False
        
        print(f'✓ {detect_resp.message}')
        print(f'  检测到: {detect_resp.result.class_names}')
        
        # 检查是否检测到目标物体
        if target_label not in detect_resp.result.class_names:
            print(f'⚠ 未检测到目标物体 "{target_label}"')
            print(f'  可用物体: {detect_resp.result.class_names}')
            return False
        
        # 3. 生成抓取姿态
        print('\n[3/4] 生成抓取姿态...')
        if not self.grasp_client.wait_for_service(timeout_sec=5.0):
            print('✗ 抓取生成服务不可用')
            return False
        
        grasp_req = GenerateGrasp.Request()
        grasp_req.color_image = self.latest_color
        grasp_req.depth_image = self.latest_depth
        grasp_req.detection_result = detect_resp.result
        grasp_req.target_label = target_label
        grasp_req.top_k = 5
        grasp_req.visualize = True
        
        grasp_future = self.grasp_client.call_async(grasp_req)
        rclpy.spin_until_future_complete(self, grasp_future, timeout_sec=30.0)
        
        if not grasp_future.done():
            print('✗ 抓取生成超时')
            return False
        
        grasp_resp = grasp_future.result()
        if not grasp_resp.success:
            print(f'✗ 抓取生成失败: {grasp_resp.message}')
            return False
        
        print(f'✓ {grasp_resp.message}')
        
        # 4. 显示抓取姿态
        print('\n[4/4] 抓取姿态结果:')
        print('-' * 60)
        
        for i, grasp in enumerate(grasp_resp.grasp_poses):
            print(f'\n抓取姿态 #{i+1}:')
            print(f'  图像位置: ({grasp.row}, {grasp.column})')
            print(f'  角度: {math.degrees(grasp.angle):.1f}°')
            print(f'  宽度: {grasp.width:.1f} px')
            print(f'  基座位置: ({grasp.position.x:.3f}, {grasp.position.y:.3f}, {grasp.position.z:.3f}) 米')
            print(f'  姿态: ({math.degrees(grasp.orientation.x):.1f}°, '
                  f'{math.degrees(grasp.orientation.y):.1f}°, '
                  f'{math.degrees(grasp.orientation.z):.1f}°)')
            print(f'  夹爪宽度: {grasp.gripper_width:.3f} 米')
            print(f'  质量分数: {grasp.quality:.3f}')
            print(f'  边缘补偿: {"是" if grasp.slope_flag else "否"}')
        
        print('\n' + '=' * 60)
        print('✓✓✓ 抓取姿态生成测试通过！')
        print('=' * 60)
        
        # 检查可视化结果
        import os
        viz_path = '/home/zh/zh/grasp_zy_zhiyuan/outputs/graspyolo_0.png'
        if os.path.exists(viz_path):
            print(f'\n✓ 可视化结果已保存: {viz_path}')
        
        return True


def main(args=None):
    rclpy.init(args=args)
    
    print('=' * 60)
    print('抓取姿态生成测试')
    print('=' * 60)
    print('\n⚠️  请确保以下节点正在运行:')
    print('  1. ros2 run grasp_vision camera_node')
    print('  2. ros2 run grasp_vision detection_server')
    print('  3. ros2 run grasp_vision grasp_generator')
    
    # 获取目标标签
    if len(sys.argv) > 1:
        target_label = sys.argv[1]
    else:
        target_label = 'carrot'
    
    print(f'\n目标物体: {target_label}')
    print('(可以通过命令行参数指定，例如: python3 test_grasp_generation.py banana)\n')
    
    node = GraspGenerationTester()
    success = node.test_grasp_generation(target_label)
    
    node.destroy_node()
    rclpy.shutdown()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

