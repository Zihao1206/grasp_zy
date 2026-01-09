#!/usr/bin/env python3
"""
抓取执行器：协调所有节点完成抓取任务
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from grasp_interfaces.action import ExecuteGrasp
from grasp_interfaces.srv import DetectObjects, GenerateGrasp, GripperControl
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
import os
import numpy as np
import time

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from robotic_arm_package.robotic_arm import *
import config as cfg


class CollisionDetected(Exception):
    """碰撞检测异常"""
    pass


class GraspExecutor(Node):
    def __init__(self):
        super().__init__('grasp_executor')
        
        # 回调组
        self.callback_group = ReentrantCallbackGroup()
        
        # 声明参数
        self.declare_parameter('robot_ip', '192.168.127.101')
        self.declare_parameter('robot_port', 8080)
        self.declare_parameter('robot_speed', 20)
        self.declare_parameter('max_attempts', 2)
        
        # 获取参数
        robot_ip = self.get_parameter('robot_ip').value
        robot_port = self.get_parameter('robot_port').value
        self.robot_speed = self.get_parameter('robot_speed').value
        self.max_attempts = self.get_parameter('max_attempts').value
        
        # 初始化机械臂
        self.get_logger().info(f'连接机械臂: {robot_ip}:{robot_port}')
        self.robot = Arm(RM65, robot_ip, robot_port)
        self.robot.Set_Collision_Stage(5)
        
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
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 订阅相机话题
        self.color_image = None
        self.depth_image = None
        self.color_sub = self.create_subscription(
            Image,
            'camera/color/image_raw',
            self.color_callback,
            10,
            callback_group=self.callback_group
        )
        self.depth_sub = self.create_subscription(
            Image,
            'camera/depth/image_raw',
            self.depth_callback,
            10,
            callback_group=self.callback_group
        )
        
        # 服务客户端
        self.detect_client = self.create_client(
            DetectObjects,
            'detect_objects',
            callback_group=self.callback_group
        )
        self.grasp_gen_client = self.create_client(
            GenerateGrasp,
            'generate_grasp',
            callback_group=self.callback_group
        )
        self.gripper_client = self.create_client(
            GripperControl,
            'gripper_control',
            callback_group=self.callback_group
        )
        
        # 动作服务器
        self._action_server = ActionServer(
            self,
            ExecuteGrasp,
            'execute_grasp',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('抓取执行器已启动')
    
    def color_callback(self, msg):
        """颜色图像回调"""
        self.color_image = msg
    
    def depth_callback(self, msg):
        """深度图像回调"""
        self.depth_image = msg
    
    def goal_callback(self, goal_request):
        """处理动作目标请求"""
        self.get_logger().info(f'收到抓取请求: {goal_request.target_label}')
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """处理取消请求"""
        self.get_logger().info('收到取消请求')
        return CancelResponse.ACCEPT
    
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
            # 打开夹爪
            req = GripperControl.Request()
            req.position = 1
            self.gripper_client.call_async(req)
            time.sleep(1)
            
            self.get_logger().info(f'正在返回安全位置: {self.mid_pose1}')
            self.robot.Movej_Cmd(self.mid_pose1, self.robot_speed, 0)
            
            # 闭合夹爪
            req.position = 0
            self.gripper_client.call_async(req)
            
            self.get_logger().info(f'正在返回初始位置: {self.init_pose}')
            self.robot.Movej_Cmd(self.init_pose, self.robot_speed, 0)
            self.get_logger().info('碰撞恢复完成')
        except Exception as e:
            self.get_logger().error(f'返回安全位失败: {e}')
    
    async def execute_callback(self, goal_handle):
        """执行抓取动作"""
        self.get_logger().info('开始执行抓取...')
        
        request = goal_handle.request
        feedback = ExecuteGrasp.Feedback()
        result = ExecuteGrasp.Result()
        
        try:
            # 1. 等待图像
            feedback.status = '等待相机图像...'
            feedback.progress = 10.0
            goal_handle.publish_feedback(feedback)
            
            while self.color_image is None or self.depth_image is None:
                self.get_logger().info('等待相机图像...')
                time.sleep(0.5)
            
            # 2. 目标检测
            feedback.status = '执行目标检测...'
            feedback.progress = 20.0
            goal_handle.publish_feedback(feedback)
            
            self.get_logger().info('调用目标检测服务...')
            detect_req = DetectObjects.Request()
            detect_req.color_image = self.color_image
            
            while not self.detect_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('等待检测服务...')
            
            detect_future = self.detect_client.call_async(detect_req)
            rclpy.spin_until_future_complete(self, detect_future, timeout_sec=10.0)
            
            detect_resp = detect_future.result()
            if not detect_resp.success:
                result.success = False
                result.message = f'目标检测失败: {detect_resp.message}'
                goal_handle.abort()
                return result
            
            self.get_logger().info(detect_resp.message)
            
            # 3. 生成抓取姿态
            feedback.status = '生成抓取姿态...'
            feedback.progress = 40.0
            goal_handle.publish_feedback(feedback)
            
            self.get_logger().info('调用抓取姿态生成服务...')
            grasp_req = GenerateGrasp.Request()
            grasp_req.color_image = self.color_image
            grasp_req.depth_image = self.depth_image
            grasp_req.detection_result = detect_resp.result
            grasp_req.target_label = request.target_label
            grasp_req.top_k = 100
            grasp_req.visualize = request.visualize
            
            while not self.grasp_gen_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('等待抓取姿态生成服务...')
            
            grasp_future = self.grasp_gen_client.call_async(grasp_req)
            rclpy.spin_until_future_complete(self, grasp_future, timeout_sec=10.0)
            
            grasp_resp = grasp_future.result()
            if not grasp_resp.success or len(grasp_resp.grasp_poses) == 0:
                result.success = False
                result.message = f'抓取姿态生成失败: {grasp_resp.message}'
                goal_handle.abort()
                return result
            
            self.get_logger().info(grasp_resp.message)
            
            # 选择第一个抓取姿态
            best_grasp = grasp_resp.grasp_poses[0]
            
            # 4. 执行抓取
            feedback.status = '执行机械臂运动...'
            feedback.progress = 60.0
            goal_handle.publish_feedback(feedback)
            
            # 构建目标位姿
            pose = [
                best_grasp.position.x,
                best_grasp.position.y,
                best_grasp.position.z,
                best_grasp.orientation.x,
                best_grasp.orientation.y,
                best_grasp.orientation.z
            ]
            
            # 根据边缘补偿调整位姿
            if best_grasp.slope_flag:
                pose[2] = cfg.pose2_2
                pose[0] += -0.02
                if pose[2] > 0.534:
                    pose[2] = cfg.pose2_2
                pose_up_to_grasp_position = [p + offset for p, offset in zip(pose, [0, 0, -0.08, 0, 0, 0])]
            else:
                pose[2] = cfg.pose2
                if pose[2] > 0.534:
                    pose[2] = cfg.pose2
                pose_up_to_grasp_position = [p + offset for p, offset in zip(pose, [0, 0, -0.05, 0, 0, 0])]
            
            self.get_logger().info(f'目标抓取位姿: {pose}')
            
            # 多次尝试
            attempt = 0
            success = False
            
            while attempt < self.max_attempts:
                try:
                    self.get_logger().info(f'\n--- 第 {attempt + 1} 次抓取尝试 ---')
                    
                    # 移动到中间位置
                    self.get_logger().info('移动到中间位置...')
                    self.movej_safe(self.mid_pose)
                    self.movej_safe(self.mid_pose1)
                    
                    # 计算逆解
                    self.get_logger().info(f'计算抓取位姿逆解: {pose}')
                    tag1, pose_joint = self.robot.Algo_Inverse_Kinematics(self.mid_pose1, pose, 1)
                    if tag1 != 0:
                        self.get_logger().error(f'✗ 机械臂逆解失败！')
                        self.movej_safe(self.init_pose)
                        result.success = False
                        result.message = '逆解失败，请重新放置物体'
                        goal_handle.abort()
                        return result
                    
                    self.get_logger().info(f'✓ 逆解成功: {pose_joint}')
                    
                    # 计算上方位置逆解
                    tag2, up_to_grasp_joint = self.robot.Algo_Inverse_Kinematics(self.mid_pose1, pose_up_to_grasp_position, 1)
                    if tag2 != 0:
                        self.get_logger().error(f'✗ 上方位置逆解失败！')
                        self.movej_safe(self.init_pose)
                        result.success = False
                        result.message = '上方位置逆解失败'
                        goal_handle.abort()
                        return result
                    
                    # 移动到上方安全位置
                    self.get_logger().info('移动到上方安全位置...')
                    self.movej_safe(up_to_grasp_joint[0:6])
                    
                    # 打开夹爪
                    self.get_logger().info('打开夹爪...')
                    gripper_req = GripperControl.Request()
                    gripper_req.position = 1
                    gripper_future = self.gripper_client.call_async(gripper_req)
                    rclpy.spin_until_future_complete(self, gripper_future, timeout_sec=5.0)
                    time.sleep(1)
                    
                    # 移动到抓取位置
                    self.get_logger().info('移动到抓取位置...')
                    self.movej_safe(pose_joint[0:6])
                    
                    # 闭合夹爪
                    self.get_logger().info('闭合夹爪抓取...')
                    gripper_req.position = 0
                    gripper_future = self.gripper_client.call_async(gripper_req)
                    rclpy.spin_until_future_complete(self, gripper_future, timeout_sec=5.0)
                    time.sleep(1)
                    
                    # 返回上方
                    self.get_logger().info('返回上方安全位置...')
                    self.movej_safe(up_to_grasp_joint[0:6])
                    
                    # 返回中间位置
                    self.get_logger().info('返回中间位置...')
                    self.movej_safe(self.mid_pose1)
                    
                    # 返回初始位置
                    self.get_logger().info('返回提升位置...')
                    self.movej_safe(self.lift2init_pose)
                    
                    # 5. 放置物体
                    feedback.status = '放置物体...'
                    feedback.progress = 80.0
                    goal_handle.publish_feedback(feedback)
                    
                    self.get_logger().info('移动到放置位置...')
                    self.movej_safe(self.place_mid_pose)
                    self.movej_safe(self.place_mid_pose2)
                    self.movej_safe(self.place_last_pose)
                    
                    # 打开夹爪放置
                    self.get_logger().info('打开夹爪放置...')
                    gripper_req.position = 1
                    gripper_future = self.gripper_client.call_async(gripper_req)
                    rclpy.spin_until_future_complete(self, gripper_future, timeout_sec=5.0)
                    time.sleep(1.0)
                    
                    # 闭合夹爪
                    self.get_logger().info('闭合夹爪...')
                    gripper_req.position = 0
                    gripper_future = self.gripper_client.call_async(gripper_req)
                    rclpy.spin_until_future_complete(self, gripper_future, timeout_sec=5.0)
                    
                    # 返回初始位置
                    self.get_logger().info('返回初始位置...')
                    self.movej_safe(self.place_mid_pose2)
                    self.movej_safe(self.place_mid_pose)
                    self.movej_safe(self.init_pose)
                    
                    self.get_logger().info('✓ 抓取成功完成！')
                    success = True
                    break
                    
                except CollisionDetected as exc:
                    self.get_logger().error(f'✗ 第{attempt + 1}次抓取碰撞: {exc}')
                    self.recover_from_collision()
                    attempt += 1
                    if attempt >= self.max_attempts:
                        self.get_logger().error('✗ 连续碰撞，达到最大尝试次数')
                        result.success = False
                        result.message = '连续碰撞，抓取失败'
                        goal_handle.abort()
                        return result
                    self.get_logger().info(f'准备第{attempt + 1}次重试...')
                    
                except Exception as exc:
                    self.get_logger().error(f'✗ MoveJ执行异常: {exc}')
                    self.recover_from_collision()
                    result.success = False
                    result.message = f'执行异常: {str(exc)}'
                    goal_handle.abort()
                    return result
            
            if success:
                feedback.status = '抓取完成'
                feedback.progress = 100.0
                goal_handle.publish_feedback(feedback)
                
                result.success = True
                result.message = '抓取成功'
                goal_handle.succeed()
            else:
                result.success = False
                result.message = '抓取失败'
                goal_handle.abort()
            
        except Exception as e:
            self.get_logger().error(f'抓取执行失败: {str(e)}')
            import traceback
            traceback.print_exc()
            result.success = False
            result.message = f'抓取执行失败: {str(e)}'
            goal_handle.abort()
        
        return result


def main(args=None):
    rclpy.init(args=args)
    
    executor = GraspExecutor()
    multi_executor = MultiThreadedExecutor()
    multi_executor.add_node(executor)
    
    try:
        multi_executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

