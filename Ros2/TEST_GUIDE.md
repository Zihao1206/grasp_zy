# 无硬件测试指南

在没有连接机械臂和相机的情况下，可以使用模拟节点测试 ROS2 系统架构。

## 📋 测试清单

- ✅ 接口定义测试（消息/服务/动作）
- ✅ 模拟相机节点测试
- ✅ 模拟机械臂控制器测试
- ✅ 模拟夹爪服务测试
- ✅ 话题通信测试
- ✅ 服务调用测试

## 🚀 快速开始

### 1. 编译工作空间

```bash
cd /home/zh/zh/grasp_zy_zhiyuan/Ros2
./build.sh
source install/setup.bash
```

### 2. 测试接口定义

无需启动任何节点，直接测试接口是否正确编译：

```bash
python3 tests/test_interface.py
```

**预期输出**：
```
============================================================
测试 ROS2 接口导入
============================================================

[1/4] 测试消息导入...
✓ GraspPose 导入成功
✓ DetectionResult 导入成功

[2/4] 测试服务导入...
✓ DetectObjects 导入成功
✓ GenerateGrasp 导入成功
✓ GripperControl 导入成功

[3/4] 测试动作导入...
✓ ExecuteGrasp 导入成功

[4/4] 测试标准消息导入...
✓ sensor_msgs 导入成功
✓ geometry_msgs 导入成功
✓ std_msgs 导入成功

============================================================
✓✓✓ 所有接口测试通过！
============================================================
```

### 3. 启动模拟系统

在一个终端启动所有模拟节点：

```bash
# 终端 1
source install/setup.bash
ros2 launch grasp_bringup test_mock_system.launch.py
```

你会看到三个模拟节点启动：
- `mock_camera_node` - 模拟相机，发布测试图像
- `mock_gripper_server` - 模拟夹爪服务
- `mock_arm_controller` - 模拟机械臂控制器

### 4. 测试图像话题

在另一个终端测试图像发布：

```bash
# 终端 2
source install/setup.bash
python3 tests/test_image_topics.py
```

**预期输出**：
```
============================================================
图像话题测试
============================================================

⚠️  请先启动相机节点:
  ros2 run grasp_vision mock_camera_node

等待图像消息...

[彩色图像 #1] 尺寸: 640x480, 编码: bgr8
✓ 保存测试图像: /tmp/test_color_image.png
[深度图像 #1] 尺寸: 640x480, 编码: passthrough, 深度范围: 0.300-0.800米
...
```

### 5. 测试服务调用

在另一个终端测试服务：

```bash
# 终端 3
source install/setup.bash
python3 tests/test_mock_services.py
```

**预期输出**：
```
============================================================
ROS2 模拟服务测试
============================================================

============================================================
测试夹爪控制服务
============================================================

等待夹爪服务...
✓ 夹爪服务已连接

[测试 1/2] 打开夹爪...
✓ 模拟夹爪打开成功

[测试 2/2] 闭合夹爪...
✓ 模拟夹爪闭合成功

✓✓✓ 夹爪服务测试通过！

============================================================
测试机械臂状态话题
============================================================

等待机械臂状态消息...
✓ 收到机械臂状态: mock_ready [joints: [0, -129, 127, -0.7, 71, -81]]

✓✓✓ 机械臂状态话题测试通过！
```

## 🔍 手动测试

### 查看话题列表

```bash
ros2 topic list
```

应该看到：
```
/camera/color/image_raw
/camera/depth/image_raw
/arm_status
/parameter_events
/rosout
```

### 查看话题信息

```bash
# 查看图像话题信息
ros2 topic info /camera/color/image_raw

# 查看话题频率
ros2 topic hz /camera/color/image_raw

# 查看话题数据（前10条）
ros2 topic echo /camera/color/image_raw --once
```

### 查看图像（可视化）

```bash
# 安装图像查看器（如果未安装）
sudo apt install ros-${ROS_DISTRO}-rqt-image-view

# 启动图像查看器
ros2 run rqt_image_view rqt_image_view
```

在 GUI 中选择话题 `/camera/color/image_raw`，你会看到模拟的测试图像。

### 查看服务列表

```bash
ros2 service list
```

应该看到：
```
/gripper_control
/mock_arm_controller/...
/mock_gripper_server/...
...
```

### 手动调用服务

```bash
# 打开夹爪
ros2 service call /gripper_control grasp_interfaces/srv/GripperControl "{position: 1}"

# 闭合夹爪
ros2 service call /gripper_control grasp_interfaces/srv/GripperControl "{position: 0}"
```

### 查看节点信息

```bash
# 查看所有节点
ros2 node list

# 查看节点详细信息
ros2 node info /mock_camera_node
```

### 查看节点关系图

```bash
# 安装 rqt_graph（如果未安装）
sudo apt install ros-${ROS_DISTRO}-rqt-graph

# 查看节点图
rqt_graph
```

## 📊 测试不同场景

### 场景 1: 不同的图像类型

```bash
# 启动带噪声图像的相机
ros2 run grasp_vision mock_camera_node --ros-args -p image_type:=noise

# 或棋盘格
ros2 run grasp_vision mock_camera_node --ros-args -p image_type:=checkerboard
```

### 场景 2: 修改发布频率

```bash
# 高频发布（30Hz）
ros2 run grasp_vision mock_camera_node --ros-args -p publish_rate:=30.0

# 低频发布（1Hz）
ros2 run grasp_vision mock_camera_node --ros-args -p publish_rate:=1.0
```

### 场景 3: 修改图像分辨率

```bash
ros2 run grasp_vision mock_camera_node --ros-args -p width:=1280 -p height:=720
```

## 🎯 测试检查点

### ✅ 基础功能测试

- [ ] 接口编译成功
- [ ] 接口可以正常导入
- [ ] 模拟节点可以启动
- [ ] 话题正常发布
- [ ] 服务可以调用
- [ ] 节点间通信正常

### ✅ 性能测试

- [ ] 图像发布频率符合预期
- [ ] 服务响应时间正常（< 1秒）
- [ ] 内存使用合理
- [ ] CPU 使用合理

### ✅ 稳定性测试

- [ ] 长时间运行（30分钟）无崩溃
- [ ] 节点可以正常重启
- [ ] 断开重连正常

## 🐛 常见问题

### Q1: 编译失败

```bash
# 清理后重新编译
rm -rf build install log
./build.sh
```

### Q2: 找不到接口

```bash
# 确保已 source 环境
source install/setup.bash

# 检查接口是否编译
ros2 interface list | grep grasp_interfaces
```

应该看到：
```
grasp_interfaces/action/ExecuteGrasp
grasp_interfaces/msg/DetectionResult
grasp_interfaces/msg/GraspPose
grasp_interfaces/srv/DetectObjects
grasp_interfaces/srv/GenerateGrasp
grasp_interfaces/srv/GripperControl
```

### Q3: 节点启动失败

```bash
# 查看详细错误
ros2 run grasp_vision mock_camera_node --ros-args --log-level debug
```

### Q4: 话题没有数据

```bash
# 检查发布者
ros2 topic info /camera/color/image_raw

# 应该显示 Publisher count: 1
```

## 📝 下一步

测试通过后，可以：

1. **添加真实硬件节点**：
   - 替换 `mock_camera_node` 为 `camera_node`（需要 RealSense）
   - 替换模拟控制器为真实机械臂控制器

2. **测试检测和抓取生成**：
   - 这两个需要模型文件，但不需要硬件
   - 可以用测试图像进行测试

3. **集成测试**：
   - 启动完整系统（使用 `grasp_system.launch.py`）
   - 测试端到端流程

## 🔧 自定义测试

你可以编写自己的测试脚本：

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from grasp_interfaces.srv import GripperControl

class MyTest(Node):
    def __init__(self):
        super().__init__('my_test')
        # 你的测试代码
        
def main():
    rclpy.init()
    node = MyTest()
    rclpy.spin(node)
    
if __name__ == '__main__':
    main()
```

## 📞 获取帮助

如果遇到问题：

1. 查看日志：`ros2 node info /节点名`
2. 查看话题：`ros2 topic echo /话题名`
3. 查看服务：`ros2 service type /服务名`
4. 查看文档：`README.md`, `ARCHITECTURE.md`

---

**提示**：所有模拟节点都会在日志中显示 "⚠️  这是模拟节点" 标记，以便区分。

