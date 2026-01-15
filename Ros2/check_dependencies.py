#!/usr/bin/env python3
"""
检查 ROS2 抓取系统的 Python 依赖
"""
import sys
import subprocess

# 必需的依赖列表
REQUIRED_PACKAGES = {
    'pyrealsense2': 'RealSense 相机 SDK',
    'cv2': 'OpenCV (opencv-python)',
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'rclpy': 'ROS2 Python 客户端',
    'sensor_msgs': 'ROS2 传感器消息',
    'cv_bridge': 'ROS2-OpenCV 桥接',
}

# 可选依赖
OPTIONAL_PACKAGES = {
    'mmcv': 'MMDetection 基础库',
    'mmdet': 'MMDetection',
    'transforms3d': '3D 变换库',
    'skimage': 'scikit-image',
}

def check_package(package_name):
    """检查包是否已安装"""
    try:
        if package_name == 'cv2':
            import cv2
        elif package_name == 'skimage':
            import skimage
        else:
            __import__(package_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def get_install_command(package_name):
    """获取安装命令"""
    install_map = {
        'pyrealsense2': 'pip3 install pyrealsense2',
        'cv2': 'pip3 install opencv-python',
        'torch': 'pip3 install torch torchvision',
        'numpy': 'pip3 install numpy',
        'rclpy': 'sudo apt install ros-foxy-rclpy',
        'sensor_msgs': 'sudo apt install ros-foxy-sensor-msgs',
        'cv_bridge': 'sudo apt install ros-foxy-cv-bridge',
        'mmcv': 'pip3 install mmcv-full',
        'mmdet': 'pip3 install mmdet',
        'transforms3d': 'pip3 install transforms3d',
        'skimage': 'pip3 install scikit-image',
    }
    return install_map.get(package_name, f'pip3 install {package_name}')

def main():
    print('=' * 70)
    print('ROS2 抓取系统依赖检查')
    print('=' * 70)
    print(f'\nPython 版本: {sys.version}')
    print(f'Python 路径: {sys.executable}\n')
    
    missing_required = []
    missing_optional = []
    
    # 检查必需依赖
    print('检查必需依赖:')
    print('-' * 70)
    for package, description in REQUIRED_PACKAGES.items():
        installed, error = check_package(package)
        status = '✓' if installed else '✗'
        color = '\033[92m' if installed else '\033[91m'
        reset = '\033[0m'
        
        print(f'{color}{status}{reset} {package:20s} - {description}')
        
        if not installed:
            missing_required.append(package)
    
    # 检查可选依赖
    print('\n检查可选依赖（用于完整功能）:')
    print('-' * 70)
    for package, description in OPTIONAL_PACKAGES.items():
        installed, error = check_package(package)
        status = '✓' if installed else '○'
        color = '\033[92m' if installed else '\033[93m'
        reset = '\033[0m'
        
        print(f'{color}{status}{reset} {package:20s} - {description}')
        
        if not installed:
            missing_optional.append(package)
    
    # 总结
    print('\n' + '=' * 70)
    print('总结')
    print('=' * 70)
    
    if not missing_required:
        print('✓✓✓ 所有必需依赖已安装！')
        if not missing_optional:
            print('✓✓✓ 所有可选依赖也已安装！')
        else:
            print(f'\n提示: {len(missing_optional)} 个可选依赖未安装（不影响基本功能）')
    else:
        print(f'✗✗✗ 缺少 {len(missing_required)} 个必需依赖')
        print('\n请安装以下依赖:')
        print('-' * 70)
        for package in missing_required:
            cmd = get_install_command(package)
            print(f'  {package:20s} → {cmd}')
        
        # 生成一键安装脚本
        print('\n一键安装所有缺失的必需依赖:')
        print('-' * 70)
        print('#!/bin/bash')
        for package in missing_required:
            print(get_install_command(package))
    
    if missing_optional:
        print(f'\n可选依赖（{len(missing_optional)} 个未安装）:')
        print('-' * 70)
        for package in missing_optional:
            cmd = get_install_command(package)
            print(f'  {package:20s} → {cmd}')
    
    print('\n' + '=' * 70)
    
    return 0 if not missing_required else 1

if __name__ == '__main__':
    sys.exit(main())

