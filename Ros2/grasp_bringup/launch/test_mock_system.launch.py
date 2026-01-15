#!/usr/bin/env python3
"""
测试启动文件：使用模拟节点，无需真实硬件
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # 声明启动参数
        DeclareLaunchArgument(
            'robot_speed',
            default_value='20',
            description='机械臂速度 (0-50)'
        ),
        
        # 1. 模拟相机节点
        Node(
            package='grasp_vision',
            executable='mock_camera_node',
            name='mock_camera_node',
            output='screen',
            parameters=[{
                'width': 640,
                'height': 480,
                'publish_rate': 5.0,
                'image_type': 'test_pattern'  # test_pattern, noise, checkerboard
            }]
        ),
        
        # 2. 模拟夹爪服务
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='grasp_control',
                    executable='mock_gripper_server',
                    name='mock_gripper_server',
                    output='screen'
                )
            ]
        ),
        
        # 3. 模拟机械臂控制器
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='grasp_control',
                    executable='mock_arm_controller',
                    name='mock_arm_controller',
                    output='screen',
                    parameters=[{
                        'robot_speed': LaunchConfiguration('robot_speed')
                    }]
                )
            ]
        ),
    ])

