#!/usr/bin/env python3
"""
测试各个服务的示例脚本
"""
import rclpy
from rclpy.node import Node
from grasp_interfaces.srv import DetectObjects, GenerateGrasp, GripperControl
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys


class ServiceTester(Node):
    def __init__(self):
        super().__init__('service_tester')
        
        self.bridge = CvBridge()
        
        # 创建服务客户端
        self.detect_client = self.create_client(DetectObjects, 'detect_objects')
        self.grasp_client = self.create_client(GenerateGrasp, 'generate_grasp')
        self.gripper_client = self.create_client(GripperControl, 'gripper_control')
        
        self.get_logger().info('服务测试节点已启动')
    
    def test_gripper(self, position):
        """测试夹爪控制"""
        self.get_logger().info(f'测试夹爪控制: position={position}')
        
        while not self.gripper_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待夹爪服务...')
        
        request = GripperControl.Request()
        request.position = position
        
        future = self.gripper_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response.success:
            self.get_logger().info(f'✓ {response.message}')
        else:
            self.get_logger().error(f'✗ {response.message}')
        
        return response.success
    
    def test_detection(self, image_path):
        """测试目标检测"""
        self.get_logger().info(f'测试目标检测: {image_path}')
        
        # 读取图像
        color_img = cv2.imread(image_path)
        if color_img is None:
            self.get_logger().error(f'无法读取图像: {image_path}')
            return False
        
        while not self.detect_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待检测服务...')
        
        # 转换为 ROS 消息
        color_msg = self.bridge.cv2_to_imgmsg(color_img, encoding='bgr8')
        
        request = DetectObjects.Request()
        request.color_image = color_msg
        
        future = self.detect_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response.success:
            self.get_logger().info(f'✓ {response.message}')
            self.get_logger().info(f'  检测到的物体: {response.result.class_names}')
        else:
            self.get_logger().error(f'✗ {response.message}')
        
        return response.success


def main(args=None):
    rclpy.init(args=args)
    
    tester = ServiceTester()
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  测试夹爪: python3 test_services.py gripper <0|1>")
        print("  测试检测: python3 test_services.py detect <image_path>")
        return
    
    command = sys.argv[1]
    
    if command == 'gripper':
        position = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        tester.test_gripper(position)
    elif command == 'detect':
        image_path = sys.argv[2] if len(sys.argv) > 2 else 'test.jpg'
        tester.test_detection(image_path)
    else:
        print(f"未知命令: {command}")
    
    tester.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

