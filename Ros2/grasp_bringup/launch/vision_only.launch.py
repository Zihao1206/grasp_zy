#!/usr/bin/env python3
"""
仅启动视觉相关节点（用于调试）
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # 相机节点
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
        
        # 目标检测服务
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
        
        # 抓取姿态生成服务
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
    ])

