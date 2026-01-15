#!/usr/bin/env python3
"""
模拟夹爪控制服务：用于测试，无需真实硬件
"""
import rclpy
from rclpy.node import Node
from grasp_interfaces.srv import GripperControl
import time


class MockGripperServer(Node):
    def __init__(self):
        super().__init__('mock_gripper_server')
        
        self.get_logger().info('初始化模拟夹爪控制器')
        
        # 当前位置 (0=闭合, 1=打开)
        self.current_position = 0
        
        # 创建服务
        self.srv = self.create_service(
            GripperControl,
            'gripper_control',
            self.control_callback
        )
        
        self.get_logger().info('模拟夹爪控制服务已启动 (无需真实硬件)')
        self.get_logger().warn('⚠️  这是模拟节点，所有夹爪指令都会被模拟执行')
    
    def control_callback(self, request, response):
        """夹爪控制回调"""
        try:
            position = request.position
            
            if position not in [0, 1]:
                response.success = False
                response.message = f'无效的夹爪位置: {position} (应为0或1)'
                return response
            
            action = "闭合" if position == 0 else "打开"
            self.get_logger().info(f'模拟夹爪{action}...')
            
            # 模拟动作延迟
            time.sleep(0.5)
            
            self.current_position = position
            
            response.success = True
            response.message = f'模拟夹爪{action}成功'
            self.get_logger().info(f'✓ {response.message}')
            
        except Exception as e:
            response.success = False
            response.message = f'模拟夹爪控制失败: {str(e)}'
            self.get_logger().error(response.message)
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MockGripperServer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

