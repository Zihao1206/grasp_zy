# 智能抓取系统 ROS2 版本

基于 ROS2 的视觉引导机器人抓取系统，部署在 Jetson Orin NX 上，用于 CTU 设备的智能分拣任务。

## 系统架构

本系统采用模块化设计，分为以下几个 ROS2 包：

### 1. `grasp_interfaces`
自定义消息、服务和动作接口定义：
- **消息**: `GraspPose`, `DetectionResult`
- **服务**: `DetectObjects`, `GenerateGrasp`, `GripperControl`
- **动作**: `ExecuteGrasp`

### 2. `grasp_vision`
视觉处理模块：
- **camera_node**: 发布 RGB 和深度图像
- **detection_server**: 目标检测服务（MMDetection）
- **grasp_generator**: 抓取姿态生成服务（CNN）

### 3. `grasp_control`
机械臂和夹爪控制模块：
- **arm_controller**: 机械臂控制节点
- **gripper_server**: 夹爪控制服务

### 4. `grasp_main`
主控制模块：
- **grasp_executor**: 抓取执行器动作服务器（协调所有模块）

### 5. `grasp_bringup`
启动文件和配置：
- **grasp_system.launch.py**: 启动完整系统
- **vision_only.launch.py**: 仅启动视觉模块（调试用）

## 数据流

```
相机节点 → 发布图像
    ↓
检测服务 → 物体检测
    ↓
抓取生成 → 计算抓取姿态
    ↓
抓取执行器 → 协调机械臂和夹爪 → 完成抓取
```

## 环境要求

- **操作系统**: Ubuntu 20.04
- **ROS 版本**: ROS2 Foxy/Galactic/Humble
- **Python**: 3.8+
- **硬件**: Jetson Orin NX, RealSense D435, RM65 机械臂

### 依赖库

```bash
# ROS2 依赖
sudo apt install ros-${ROS_DISTRO}-cv-bridge
sudo apt install ros-${ROS_DISTRO}-rclpy
sudo apt install ros-${ROS_DISTRO}-action-msgs

# Python 依赖
pip install torch torchvision
pip install opencv-python
pip install pyrealsense2
pip install mmcv mmdet
pip install transforms3d
pip install scikit-image
```

## 安装步骤

### 1. 创建 ROS2 工作空间

```bash
# 进入 Ros2 目录（已创建）
cd /home/zh/zh/grasp_zy_zhiyuan/Ros2

# 编译所有包
colcon build --symlink-install

# 加载环境
source install/setup.bash
```

### 2. 配置环境变量

将以下内容添加到 `~/.bashrc`：

```bash
# ROS2 环境
source /opt/ros/humble/setup.bash  # 根据你的 ROS2 版本修改
source /home/zh/zh/grasp_zy_zhiyuan/Ros2/install/setup.bash

# Conda 环境（可选，如果使用 conda）
conda activate zy_torch
```

### 3. 配置参数

编辑配置文件 `grasp_bringup/config/grasp_params.yaml`，根据实际情况修改：
- 机械臂 IP 和端口
- 模型权重路径
- 相机参数

## 使用方法

### 启动完整系统

```bash
# 确保已加载环境
source install/setup.bash

# 启动整个抓取系统
ros2 launch grasp_bringup grasp_system.launch.py

# 或指定参数
ros2 launch grasp_bringup grasp_system.launch.py robot_ip:=192.168.127.101 robot_speed:=20
```

### 仅测试视觉模块

```bash
# 启动视觉节点
ros2 launch grasp_bringup vision_only.launch.py

# 在另一个终端查看相机图像
ros2 run rqt_image_view rqt_image_view
```

### 调用抓取动作

使用命令行：

```bash
# 发送抓取任务
ros2 action send_goal /execute_grasp grasp_interfaces/action/ExecuteGrasp "{target_label: 'carrot', visualize: true}"
```

使用 Python 客户端：

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from grasp_interfaces.action import ExecuteGrasp

class GraspClient(Node):
    def __init__(self):
        super().__init__('grasp_client')
        self._action_client = ActionClient(self, ExecuteGrasp, 'execute_grasp')
    
    def send_goal(self, target_label='carrot', visualize=True):
        goal_msg = ExecuteGrasp.Goal()
        goal_msg.target_label = target_label
        goal_msg.visualize = visualize
        
        self._action_client.wait_for_server()
        return self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'状态: {feedback.status}, 进度: {feedback.progress}%')

def main():
    rclpy.init()
    client = GraspClient()
    future = client.send_goal('carrot', True)
    rclpy.spin_until_future_complete(client, future)

if __name__ == '__main__':
    main()
```

## 话题和服务

### 话题

- `/camera/color/image_raw` (sensor_msgs/Image): 彩色图像
- `/camera/depth/image_raw` (sensor_msgs/Image): 深度图像
- `/arm_status` (std_msgs/String): 机械臂状态

### 服务

- `/detect_objects` (grasp_interfaces/srv/DetectObjects): 目标检测
- `/generate_grasp` (grasp_interfaces/srv/GenerateGrasp): 抓取姿态生成
- `/gripper_control` (grasp_interfaces/srv/GripperControl): 夹爪控制

### 动作

- `/execute_grasp` (grasp_interfaces/action/ExecuteGrasp): 执行完整抓取流程

## 调试工具

### 查看话题列表

```bash
ros2 topic list
```

### 查看话题数据

```bash
ros2 topic echo /camera/color/image_raw
```

### 查看服务列表

```bash
ros2 service list
```

### 调用服务

```bash
ros2 service call /gripper_control grasp_interfaces/srv/GripperControl "{position: 1}"
```

### 查看节点图

```bash
# 安装 rqt
sudo apt install ros-${ROS_DISTRO}-rqt*

# 查看节点图
rqt_graph
```

### 查看 TF 树

```bash
ros2 run tf2_tools view_frames
```

## 故障排查

### 1. 相机无法连接

```bash
# 检查 RealSense 设备
rs-enumerate-devices

# 测试相机
python3 camera.py
```

### 2. 机械臂无法连接

- 检查网络连接：`ping 192.168.127.101`
- 检查机械臂电源
- 查看机械臂控制器状态

### 3. 模型加载失败

- 确认模型权重路径正确
- 检查 CUDA 是否可用：`python3 -c "import torch; print(torch.cuda.is_available())"`
- 查看日志：`ros2 node info /detection_server`

### 4. 碰撞检测触发

- 调整碰撞灵敏度参数
- 检查机械臂工作空间是否有障碍物
- 查看抓取姿态是否合理

### 5. 编译错误

```bash
# 清理编译文件
rm -rf build install log

# 重新编译
colcon build --symlink-install

# 如果出现依赖问题
rosdep install --from-paths src --ignore-src -r -y
```

## 性能优化

1. **降低图像发布频率**: 修改 `publish_rate` 参数（默认 10Hz）
2. **使用 TensorRT**: 加速模型推理
3. **调整碰撞检测级别**: 平衡安全性和速度
4. **并行处理**: 使用 MultiThreadedExecutor

## 系统配置为服务

创建 systemd 服务自动启动：

```bash
sudo nano /etc/systemd/system/ros2_grasp.service
```

内容如下：

```ini
[Unit]
Description=ROS2 Grasp System
After=network.target

[Service]
Type=simple
User=jet
WorkingDirectory=/home/jet/zoneyung/grasp_static/Ros2
ExecStart=/bin/bash -c "source /opt/ros/humble/setup.bash && source /home/jet/zoneyung/grasp_static/Ros2/install/setup.bash && ros2 launch grasp_bringup grasp_system.launch.py"
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable ros2_grasp.service
sudo systemctl start ros2_grasp.service
sudo systemctl status ros2_grasp.service
```

## 相比原系统的优势

1. **模块化设计**: 各模块独立，易于维护和扩展
2. **标准化接口**: 使用 ROS2 标准消息和服务
3. **分布式部署**: 可以在不同设备上运行不同节点
4. **更好的调试**: 使用 ROS2 工具链进行调试
5. **可扩展性**: 易于添加新功能（如多机械臂协同）
6. **参数化配置**: 通过参数文件灵活配置
7. **状态监控**: 实时反馈系统状态

## 联系方式

- **维护者**: jet
- **邮箱**: jet@zoneyung.com
- **项目路径**: `/home/jet/zoneyung/grasp_static`

## 许可证

MIT License

