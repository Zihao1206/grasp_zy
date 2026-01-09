# 视觉机器人抓取系统 (Robotic Grasping System)

基于视觉的机器人抓取系统，部署在 Jetson Orin NX 开发板上，用于 CTU 设备的智能分拣任务。系统通过 Intel RealSense 相机感知环境，使用深度学习模型检测物体并规划抓取姿态，控制 RM65 机械臂完成抓取和放置操作。

## 📋 项目概述

本项目是一个完整的机器人视觉抓取解决方案，包含：
- 物体检测与识别（MMDetection）
- 抓取点生成（自定义 CNN）
- 机械臂运动控制（RM65）
- CTU 设备通信
- 实时图像处理与坐标变换

## 🛠️ 技术栈

| 技术 | 说明 |
|------|------|
| **编程语言** | Python 3.x |
| **深度学习框架** | PyTorch + MMDetection |
| **相机接口** | pyrealsense2 (Intel RealSense) |
| **机械臂控制** | 自定义 SDK (`robotic_arm_package`) |
| **通信协议** | 自定义二进制协议 + CRC16 校验 |
| **操作系统** | Ubuntu 20.04 |
| **部署环境** | Jetson Orin NX 8GB |

## 📦 安装

### 环境要求

- Python 3.8+
- CUDA 11.x
- Intel RealSense SDK
- RM65 机械臂 SDK

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repository-url>
cd grasp_zy_zhiyuan

# 2. 创建并激活 conda 环境
conda create -n zy_torch python=3.8
conda activate zy_torch

# 3. 安装依赖
pip install torch torchvision
pip install mmdet
pip install pyrealsense2
pip install opencv-python numpy

# 4. 安装机械臂 SDK
cd robotic_arm_package
# 按照机械臂 SDK 文档安装
```

## 🚀 快速开始

### 基本使用

```bash
# 激活 conda 环境
conda activate zy_torch

# 启动主抓取程序
python grasp_zy_test.py

# 启动 CTU 通信服务
python ctu_conn.py

# 启动 CTU 控制程序
python RunCtu.py
```

### 调试工具

```bash
# 机械臂关节调试 GUI
python RoboticArm.py

# 夹爪调试 GUI
python RoboticGripper.py

# 相机测试
python camera.py
```

### 系统服务

```bash
# 启用 systemd 服务
sudo systemctl enable RunGraspd.service

# 启动服务
sudo systemctl start RunGraspd.service

# 查看状态
sudo systemctl status RunGraspd.service

# 查看日志
sudo journalctl -u RunGraspd.service
```

## 📁 项目结构

```
grasp_zy_zhiyuan/
├── config.py                 # 全局配置文件
├── grasp_zy_zhiyuan*.py      # 主抓取程序
├── ctu_conn.py              # CTU 通信服务
├── gripper_zhiyuan.py       # 夹爪控制
├── camera.py                # 相机接口
├── dataset/                 # 数据集
├── models/                  # 模型文件
│   ├── weights/            # 模型权重
│   ├── gqcnn_server/       # GQ-CNN 服务器
│   └── mmdetection/        # MMDetection 配置
├── robotic_arm_package/    # 机械臂 SDK
├── outputs/                # 输出目录
├── others/                 # 其他工具脚本
└── doc/                    # 文档
```

## 🔧 配置

关键配置参数位于 [`config.py`](config.py:1)：

| 参数 | 说明 |
|------|------|
| `Tcam2base` | 相机到基座的变换矩阵 |
| `Rbase2cam` | 基座到相机的旋转矩阵 |
| `angle = 1/7` | 边缘倾斜补偿角度（π/7 弧度） |
| `tcp_compensate` | TCP 补偿量（0.018 米） |
| `place_last_pose` | 放置位置 |
| `robot_speed` | 机械臂速度（0-50 级） |

## 🎯 支持的物体类别

| 类别 ID | 名称 | 说明 |
|---------|------|------|
| 1 | soap | 肥皂 |
| 2 | interrupter | 空气开关 |
| 3 | terminal | 接线端子 |
| 4 | limit | 限位开关 |
| 5 | voltage | 电压采集模块 |

## 🔄 抓取流程

```mermaid
graph LR
    A[相机采集图像] --> B[MMDetection 检测]
    B --> C[生成抓取点]
    C --> D[边缘补偿]
    D --> E[深度补偿]
    E --> F[逆运动学求解]
    F --> G[轨迹规划]
    G --> H[机械臂执行]
```

1. **检测阶段**：MMDetection 检测物体，实例分割
2. **抓取生成**：自定义 CNN 生成抓取点 (row, column, angle, width)
3. **边缘补偿**：根据位置应用 π/7 弧度倾斜
4. **深度补偿**：固定 -0.18 米偏移
5. **逆运动学**：求解关节角度
6. **轨迹规划**：先移动到上方，再下降抓取

## 📡 CTU 通信协议

- **协议类型**：自定义二进制协议
- **校验方式**：CRC16
- **帧结构**：SOF(2) + LEN(2) + DATA + CRC16(2)
- **心跳机制**：10 秒间隔自动发送

### 命令字

| 命令字 | 说明 |
|--------|------|
| 0x70 | CTU_GRASP_START（开始分拣） |
| 0x71 | CTU_GRASP_SPEED（调速） |
| 0x80 | GRASP_COUNT（物品数量） |
| 0x81 | GRASP_START（开始抓取） |
| 0x82 | GRASP_OVER（抓取完成） |

## 🌐 网络配置

| 设备 | IP 地址 | 端口 |
|------|---------|------|
| CTU | 192.168.127.253 | 8899 |
| 机械臂 | 192.168.127.101 | 8080 |
| 开发板（有线） | 192.168.127.102 | - |
| 开发板（无线） | 192.168.2.51 | - |

## 📊 性能指标

| 指标 | 要求 |
|------|------|
| 单次抓取时间 | < 5 秒 |
| 模型推理时间 | < 100ms |
| 通信延迟 | < 100ms |
| 系统启动时间 | < 30 秒 |

## ⚠️ 安全限制

- 机械臂速度：0-50 级（不是百分比）
- Z 轴高度限制：0.538 米（防止碰撞）
- 碰撞检测：`Set_Collision_Stage(4)`
- 最大尝试次数：物品数 + 5 次

## 🐛 调试技巧

- 设置 `vis=True` 可视化抓取点
- 使用 `RoboticArm.py` 测试关节 6
- 使用 `RoboticGripper.py` 测试夹爪
- 运行 `camera.py` 检查相机
- 查看 `outputs/` 目录的调试图像

## 📝 注意事项

1. **路径问题**：所有模型和数据文件使用绝对路径，修改代码时务必检查
2. **坐标系**：注意 OpenCV (row, column) 和 NumPy (array) 的区别
3. **速度限制**：机械臂速度 0-50 级，不是百分比
4. **补偿参数**：边缘倾斜 π/7 弧度，深度 -0.18 米是经验值
5. **网络通信**：CTU 使用二进制协议，不是文本协议
6. **错误恢复**：系统有自动重连和重试机制，但次数有限
7. **硬件依赖**：RealSense 相机和 RM65 机械臂的 SDK 必须正确安装

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- **开发板**：jet/空格（支持 sudo）
- **SSH 端口**：33322
- **项目目录**：/home/jet/zoneyung/grasp_static
- **Conda 环境**：zy_torch
