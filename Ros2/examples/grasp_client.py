#!/usr/bin/env python3
"""
抓取动作客户端示例
"""
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from grasp_interfaces.action import ExecuteGrasp
import sys


class GraspClient(Node):
    def __init__(self):
        super().__init__('grasp_client')
        self._action_client = ActionClient(self, ExecuteGrasp, 'execute_grasp')
        self.get_logger().info('抓取客户端已启动')
    
    def send_goal(self, target_label='carrot', visualize=True):
        """发送抓取目标"""
        goal_msg = ExecuteGrasp.Goal()
        goal_msg.target_label = target_label
        goal_msg.visualize = visualize
        
        self.get_logger().info(f'等待动作服务器...')
        self._action_client.wait_for_server()
        
        self.get_logger().info(f'发送抓取目标: {target_label}')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """目标响应回调"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('目标被拒绝')
            return
        
        self.get_logger().info('目标已接受，等待结果...')
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        """结果回调"""
        result = future.result().result
        
        if result.success:
            self.get_logger().info(f'抓取成功: {result.message}')
        else:
            self.get_logger().error(f'抓取失败: {result.message}')
        
        rclpy.shutdown()
    
    def feedback_callback(self, feedback_msg):
        """反馈回调"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'[{feedback.progress:.1f}%] {feedback.status}')


def main(args=None):
    rclpy.init(args=args)
    
    # 获取目标物体标签
    if len(sys.argv) > 1:
        target_label = sys.argv[1]
    else:
        target_label = 'carrot'
    
    client = GraspClient()
    client.send_goal(target_label, visualize=True)
    
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()


if __name__ == '__main__':
    main()

