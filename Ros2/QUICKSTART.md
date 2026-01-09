# 快速开始指南

## 第一次使用

### 1. 安装 ROS2 依赖

```bash
# 安装 ROS2 Humble（如果还未安装）
# 参考: https://docs.ros.org/en/humble/Installation.html

# 安装必要的 ROS2 包
sudo apt update
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-rclpy \
    ros-humble-action-msgs \
    ros-humble-rqt* \
    python3-colcon-common-extensions
```

### 2. 编译工作空间

```bash
# 进入 Ros2 目录
cd /home/zh/zh/grasp_zy_zhiyuan/Ros2

# 给脚本添加执行权限
chmod +x build.sh setup_env.sh

# 编译
./build.sh
```

### 3. 加载环境

```bash
# 方法1: 使用脚本
source setup_env.sh

# 方法2: 手动加载
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## 运行系统

### 完整系统

```bash
# 确保已加载环境
source install/setup.bash

# 启动完整抓取系统
ros2 launch grasp_bringup grasp_system.launch.py
```

### 仅视觉模块（调试）

```bash
ros2 launch grasp_bringup vision_only.launch.py
```

### 指定参数启动

```bash
ros2 launch grasp_bringup grasp_system.launch.py \
    robot_ip:=192.168.127.101 \
    robot_port:=8080 \
    robot_speed:=20
```

## 执行抓取任务

### 使用命令行

```bash
# 抓取胡萝卜
ros2 action send_goal /execute_grasp grasp_interfaces/action/ExecuteGrasp \
    "{target_label: 'carrot', visualize: true}"

# 抓取肥皂
ros2 action send_goal /execute_grasp grasp_interfaces/action/ExecuteGrasp \
    "{target_label: 'soap', visualize: true}"
```

### 使用 Python 客户端

```bash
cd examples

# 抓取胡萝卜
python3 grasp_client.py carrot

# 抓取白萝卜
python3 grasp_client.py daikon
```

## 测试单个服务

```bash
cd examples

# 测试夹爪（打开）
python3 test_services.py gripper 1

# 测试夹爪（闭合）
python3 test_services.py gripper 0

# 测试目标检测
python3 test_services.py detect /path/to/image.jpg
```

## 查看系统状态

### 查看所有节点

```bash
ros2 node list
```

输出应该包括：
- `/camera_node`
- `/detection_server`
- `/grasp_generator`
- `/gripper_server`
- `/grasp_executor`

### 查看话题

```bash
ros2 topic list
```

### 查看图像

```bash
# 启动图像查看器
ros2 run rqt_image_view rqt_image_view

# 选择话题:
# - /camera/color/image_raw
# - /camera/depth/image_raw
```

### 查看日志

```bash
# 查看特定节点的日志
ros2 node info /grasp_executor

# 查看话题频率
ros2 topic hz /camera/color/image_raw
```

## 常见问题

### Q: 编译失败

```bash
# 清理并重新编译
./build.sh --clean
```

### Q: 相机无法连接

```bash
# 测试 RealSense 相机
rs-enumerate-devices

# 如果没有 rs-enumerate-devices，安装 librealsense
sudo apt install ros-humble-librealsense2*
```

### Q: 机械臂无法连接

```bash
# 检查网络
ping 192.168.127.101

# 检查端口
nc -zv 192.168.127.101 8080
```

### Q: 模型加载失败

```bash
# 检查路径（从项目根目录）
ls models/mmdetection/configs/myconfig_zy.py
ls models/weights/epoch_20_last.pth
ls models/test_250927_1644__zoneyung_/epoch_84_accuracy_1.00

# 检查 CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Q: 节点启动失败

```bash
# 查看详细错误
ros2 run grasp_vision camera_node --ros-args --log-level debug
```

## 停止系统

```bash
# 按 Ctrl+C 停止

# 或者如果使用了 tmux/screen
killall -9 ros2
```

## 下一步

- 查看完整文档: `README.md`
- 修改参数: `grasp_bringup/config/grasp_params.yaml`
- 查看示例代码: `examples/`

## 目标物体类别

系统支持以下物体类别：
- `soap` - 肥皂
- `interrupter` - 空气开关
- `terminal` - 接线端子
- `limit` - 限位开关
- `voltage` - 电压采集模块
- `carrot` - 胡萝卜
- `banana` - 香蕉
- `daikon` - 白萝卜

