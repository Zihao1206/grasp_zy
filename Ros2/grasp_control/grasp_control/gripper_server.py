#!/usr/bin/env python3
"""
夹爪控制服务节点
"""
import rclpy
from rclpy.node import Node
from grasp_interfaces.srv import GripperControl
import sys
import os
import time

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

import gripper_zhiyuan as gripper
from robotic_arm_package.robotic_arm import *


class GripperServer(Node):
    def __init__(self):
        super().__init__('gripper_server')
        
        # 声明参数
        self.declare_parameter('robot_ip', '192.168.127.101')
        self.declare_parameter('robot_port', 8080)
        
        # 获取参数
        robot_ip = self.get_parameter('robot_ip').value
        robot_port = self.get_parameter('robot_port').value
        
        # 初始化机械臂（夹爪需要通过机械臂控制）
        self.get_logger().info(f'连接机械臂用于夹爪控制: {robot_ip}:{robot_port}')
        self.robot = Arm(RM65, robot_ip, robot_port)
        
        # 初始化夹爪
        self.get_logger().info('初始化夹爪...')
        self.gripper = gripper.GripperZhiyuan(self.robot)
        self.gripper.gripper_initial()
        self.get_logger().info('夹爪初始化完成')
        
        # 创建服务
        self.srv = self.create_service(
            GripperControl,
            'gripper_control',
            self.control_callback
        )
        
        self.get_logger().info('夹爪控制服务已启动')
    
    def control_callback(self, request, response):
        """夹爪控制回调"""
        try:
            position = request.position
            
            if position == 0:
                self.get_logger().info('闭合夹爪')
            elif position == 1:
                self.get_logger().info('打开夹爪')
            else:
                response.success = False
                response.message = f'无效的夹爪位置: {position} (应为0或1)'
                return response
            
            # 控制夹爪
            self.gripper.gripper_position(position)
            time.sleep(1.0)  # 等待夹爪完成动作
            
            response.success = True
            response.message = f'夹爪{"闭合" if position == 0 else "打开"}成功'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'夹爪控制失败: {str(e)}'
            self.get_logger().error(response.message)
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GripperServer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

