#!/usr/bin/env python3
"""
抓取系统完整启动文件
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    # 声明参数
    robot_ip = LaunchConfiguration('robot_ip', default='192.168.127.101')
    robot_port = LaunchConfiguration('robot_port', default='8080')
    robot_speed = LaunchConfiguration('robot_speed', default='20')
    
    return LaunchDescription([
        # 声明启动参数
        DeclareLaunchArgument(
            'robot_ip',
            default_value='192.168.127.101',
            description='机械臂IP地址'
        ),
        DeclareLaunchArgument(
            'robot_port',
            default_value='8080',
            description='机械臂端口'
        ),
        DeclareLaunchArgument(
            'robot_speed',
            default_value='20',
            description='机械臂速度 (0-50)'
        ),
        
        # 1. 相机节点
        Node(
            package='grasp_vision',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[{
                'width': 640,
                'height': 480,
                'publish_rate': 10.0
            }]
        ),
        
        # 2. 目标检测服务（延迟1秒启动）
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='grasp_vision',
                    executable='detection_server',
                    name='detection_server',
                    output='screen',
                    parameters=[{
                        'config_file': 'models/mmdetection/configs/myconfig_zy.py',
                        'checkpoint': 'models/weights/epoch_20_last.pth',
                        'device': 'cuda',
                        'nms_score_threshold': 0.8,
                        'nms_iou_threshold': 0.9
                    }]
                )
            ]
        ),
        
        # 3. 抓取姿态生成服务（延迟2秒启动）
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='grasp_vision',
                    executable='grasp_generator',
                    name='grasp_generator',
                    output='screen',
                    parameters=[{
                        'gene_file': 'doc/single_new.txt',
                        'cornell_data': 'dataset/cornell.data',
                        'model_weights': 'models/test_250927_1644__zoneyung_/epoch_84_accuracy_1.00',
                        'device': 'cuda',
                        'camera_width': 640,
                        'camera_height': 480
                    }]
                )
            ]
        ),
        
        # 4. 夹爪控制服务（延迟3秒启动）
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='grasp_control',
                    executable='gripper_server',
                    name='gripper_server',
                    output='screen',
                    parameters=[{
                        'robot_ip': robot_ip,
                        'robot_port': robot_port
                    }]
                )
            ]
        ),
        
        # 5. 抓取执行器（延迟4秒启动）
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='grasp_main',
                    executable='grasp_executor',
                    name='grasp_executor',
                    output='screen',
                    parameters=[{
                        'robot_ip': robot_ip,
                        'robot_port': robot_port,
                        'robot_speed': robot_speed,
                        'max_attempts': 2
                    }]
                )
            ]
        ),
    ])

