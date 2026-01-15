#!/usr/bin/env python3
"""
测试模拟服务：测试夹爪控制等服务
需要先启动模拟节点
"""
import rclpy
from rclpy.node import Node
from grasp_interfaces.srv import GripperControl
from std_msgs.msg import String
import sys
import time


class ServiceTester(Node):
    def __init__(self):
        super().__init__('service_tester')
        
        # 创建服务客户端
        self.gripper_client = self.create_client(GripperControl, 'gripper_control')
        
        # 订阅机械臂状态
        self.arm_status = None
        self.arm_sub = self.create_subscription(
            String,
            'arm_status',
            self.arm_status_callback,
            10
        )
        
        self.get_logger().info('服务测试节点已启动')
    
    def arm_status_callback(self, msg):
        """机械臂状态回调"""
        self.arm_status = msg.data
    
    def test_gripper(self):
        """测试夹爪服务"""
        print("\n" + "=" * 60)
        print("测试夹爪控制服务")
        print("=" * 60)
        
        # 等待服务
        print("\n等待夹爪服务...")
        if not self.gripper_client.wait_for_service(timeout_sec=5.0):
            print("✗ 夹爪服务不可用（5秒超时）")
            print("  请先启动: ros2 run grasp_control mock_gripper_server")
            return False
        
        print("✓ 夹爪服务已连接")
        
        # 测试打开夹爪
        print("\n[测试 1/2] 打开夹爪...")
        request = GripperControl.Request()
        request.position = 1
        
        future = self.gripper_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.done():
            response = future.result()
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return False
        else:
            print("✗ 请求超时")
            return False
        
        time.sleep(1)
        
        # 测试闭合夹爪
        print("\n[测试 2/2] 闭合夹爪...")
        request.position = 0
        
        future = self.gripper_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.done():
            response = future.result()
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return False
        else:
            print("✗ 请求超时")
            return False
        
        print("\n✓✓✓ 夹爪服务测试通过！")
        return True
    
    def test_arm_status(self):
        """测试机械臂状态话题"""
        print("\n" + "=" * 60)
        print("测试机械臂状态话题")
        print("=" * 60)
        
        print("\n等待机械臂状态消息...")
        
        # 等待最多5秒
        for i in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.arm_status is not None:
                print(f"✓ 收到机械臂状态: {self.arm_status}")
                print("\n✓✓✓ 机械臂状态话题测试通过！")
                return True
        
        print("✗ 未收到机械臂状态消息（5秒超时）")
        print("  请先启动: ros2 run grasp_control mock_arm_controller")
        return False


def main(args=None):
    rclpy.init(args=args)
    
    tester = ServiceTester()
    
    print("=" * 60)
    print("ROS2 模拟服务测试")
    print("=" * 60)
    print("\n⚠️  请确保已启动模拟节点:")
    print("  ros2 launch grasp_bringup test_mock_system.launch.py")
    print("\n或手动启动:")
    print("  ros2 run grasp_control mock_gripper_server")
    print("  ros2 run grasp_control mock_arm_controller")
    
    input("\n按 Enter 开始测试...")
    
    success = True
    
    # 测试夹爪服务
    success &= tester.test_gripper()
    
    # 测试机械臂状态
    success &= tester.test_arm_status()
    
    tester.destroy_node()
    rclpy.shutdown()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 所有服务测试通过！")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ 部分测试失败")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()

