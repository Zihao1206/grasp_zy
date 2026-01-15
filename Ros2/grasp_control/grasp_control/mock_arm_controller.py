#!/usr/bin/env python3
"""
模拟机械臂控制器：用于测试，无需真实硬件
模拟机械臂运动和状态反馈
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class MockArmController(Node):
    def __init__(self):
        super().__init__('mock_arm_controller')
        
        # 声明参数
        self.declare_parameter('robot_speed', 20)
        
        # 获取参数
        self.robot_speed = self.get_parameter('robot_speed').value
        
        self.get_logger().info(f'初始化模拟机械臂控制器 (速度: {self.robot_speed})')
        
        # 模拟的当前关节角度
        self.current_joint = [0, -129, 127, -0.7, 71, -81]
        
        # 预定义姿态
        self.init_pose = [86, -129, 127, -0.8, 71, -81]
        self.mid_pose = [0, -129, 127, -0.7, 71, -81]
        self.mid_pose1 = [0, -129, 80, -0.7, 100, -81]
        
        # 发布状态
        self.status_pub = self.create_publisher(String, 'arm_status', 10)
        self.timer = self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info('模拟机械臂控制器已启动 (无需真实硬件)')
        self.get_logger().warn('⚠️  这是模拟节点，所有运动指令都会被模拟执行')
    
    def publish_status(self):
        """发布机械臂状态"""
        msg = String()
        msg.data = f'mock_ready [joints: {self.current_joint}]'
        self.status_pub.publish(msg)
    
    def movej(self, target_pose, speed=None):
        """模拟关节运动"""
        if speed is None:
            speed = self.robot_speed
        
        self.get_logger().info(f'模拟运动: {self.current_joint} -> {target_pose} (速度: {speed})')
        
        # 模拟运动时间（根据速度）
        move_time = 2.0 / (speed / 20.0)  # 基础2秒
        time.sleep(0.1)  # 短暂延迟模拟运动
        
        self.current_joint = target_pose
        self.get_logger().info(f'✓ 到达目标位置: {target_pose}')
        
        return 0  # 成功
    
    def inverse_kinematics(self, current_joint, target_pose):
        """模拟逆运动学计算"""
        self.get_logger().info(f'模拟逆运动学: 位姿 {target_pose}')
        
        # 简单模拟：返回一个虚拟的关节角度
        mock_joint = [10, -120, 100, -1.0, 80, -70]
        
        self.get_logger().info(f'✓ 逆解成功: {mock_joint}')
        return 0, mock_joint  # 成功, 关节角度


def main(args=None):
    rclpy.init(args=args)
    node = MockArmController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

