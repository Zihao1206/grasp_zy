#!/usr/bin/env python3
"""
机械臂控制节点
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from robotic_arm_package.robotic_arm import *


class CollisionDetected(Exception):
    """碰撞检测异常"""
    pass


class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')
        
        # 声明参数
        self.declare_parameter('robot_ip', '192.168.127.101')
        self.declare_parameter('robot_port', 8080)
        self.declare_parameter('robot_speed', 20)
        self.declare_parameter('collision_stage', 5)
        
        # 获取参数
        robot_ip = self.get_parameter('robot_ip').value
        robot_port = self.get_parameter('robot_port').value
        self.robot_speed = self.get_parameter('robot_speed').value
        collision_stage = self.get_parameter('collision_stage').value
        
        # 初始化机械臂
        self.get_logger().info(f'连接机械臂: {robot_ip}:{robot_port}')
        self.robot = Arm(RM65, robot_ip, robot_port)
        self.robot.Set_Collision_Stage(collision_stage)
        
        # 预定义姿态
        self.init_pose = [86, -129, 127, -0.8, 71, -81]
        self.mid_pose = [0, -129, 127, -0.7, 71, -81]
        self.mid_pose1 = [0, -129, 80, -0.7, 100, -81]
        self.lift2init_pose = [65, -129, 127, -0.7, 77, 1]
        self.place_mid_pose = [65, -129, 60, 0, 121, 1]
        self.place_mid_pose2 = [69, -129, 60, 0, 9, 1]
        self.place_last_pose = [69, -104, 38, -2, 9, 1]
        
        # 移动到初始位置
        self.get_logger().info('移动到初始位置...')
        self.robot.Movej_Cmd(self.init_pose, self.robot_speed, 0)
        
        # 发布机械臂状态
        self.status_pub = self.create_publisher(String, 'arm_status', 10)
        self.timer = self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info('机械臂控制节点已启动')
    
    def publish_status(self):
        """发布机械臂状态"""
        msg = String()
        msg.data = 'ready'
        self.status_pub.publish(msg)
    
    def movej_safe(self, pose, speed=None):
        """安全移动函数，检测碰撞"""
        if speed is None:
            speed = self.robot_speed
        
        tag = self.robot.Movej_Cmd(pose, speed, 0)
        
        if tag == 0:
            return tag, pose
        
        # 检查碰撞
        if "100D4" in str(tag) or "collision" in str(tag).lower() or "碰撞" in str(tag):
            raise CollisionDetected(f"碰撞检测到: {tag}")
        
        raise RuntimeError(f"MoveJ失败: {tag}")
    
    def recover_from_collision(self):
        """碰撞恢复"""
        self.get_logger().warn('执行碰撞恢复流程...')
        
        try:
            self.robot.Move_Stop_Cmd()
            self.get_logger().info('已发送停止命令')
        except Exception as e:
            self.get_logger().error(f'停止命令失败: {e}')
        
        try:
            self.robot.Clear_System_Err(True)
            self.get_logger().info('已清除系统错误')
            self.robot.Set_Collision_Stage(2)
        except Exception as e:
            self.get_logger().error(f'清除错误失败: {e}')
        
        try:
            self.get_logger().info(f'正在返回安全位置: {self.mid_pose1}')
            self.robot.Movej_Cmd(self.mid_pose1, self.robot_speed, 0)
            self.get_logger().info(f'正在返回初始位置: {self.init_pose}')
            self.robot.Movej_Cmd(self.init_pose, self.robot_speed, 0)
            self.get_logger().info('碰撞恢复完成')
        except Exception as e:
            self.get_logger().error(f'返回安全位失败: {e}')
    
    def inverse_kinematics(self, current_joint, target_pose):
        """计算逆运动学"""
        tag, pose_joint = self.robot.Algo_Inverse_Kinematics(current_joint, target_pose, 1)
        if tag != 0:
            return None
        return pose_joint[0:6]
    
    def get_current_joint(self):
        """获取当前关节角度"""
        # 这里需要实现获取当前关节角度的功能
        # 根据实际API实现
        return self.mid_pose1  # 临时返回


def main(args=None):
    rclpy.init(args=args)
    node = ArmController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

