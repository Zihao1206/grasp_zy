# AGENTS.md

**注意：请用中文回答或者编写文档**

---

## 项目概述

基于视觉的机器人抓取系统，Jetson Orin NX部署，CTU分拣任务。Intel RealSense相机 + PyTorch/MMDetection检测 + RM65机械臂控制。

**架构状态**：两套并行实现
- **原始Python系统**（根目录）：单体式，直接运行
- **ROS2系统**（Ros2/）：模块化，推荐生产环境

---

## 技术栈

- **语言**：Python 3.x
- **深度学习**：PyTorch + MMDetection
- **相机**：pyrealsense2 (RealSense D435)
- **机械臂**：robotic_arm_package SDK
- **通信**：自定义二进制 + CRC16
- **系统**：Ubuntu 20.04, Jetson Orin NX 8GB

---

## 核心命令

```bash
# 启动
conda activate zy_torch
python grasp_zy_zhiyuan1215.py  # 主抓取程序（注意：不是README说的test.py）
python ctu_conn.py              # CTU通信服务

# 调试
python RoboticArm.py             # 机械臂关节调试
python RoboticGripper.py         # 夹爪调试
python camera.py                 # 相机测试
```

---

## 代码风格和约定

### 1. 硬编码路径
模型权重：`weights/epoch_20.pth`，数据：`single_zy.txt`，配置：`mmdetection/configs/myconfig_zy.py`
**注意**：修改代码时务必检查路径

### 2. 混合编程风格
- OpenCV：BGR格式，(row, column)坐标
- PyTorch：CUDA张量操作
- NumPy：数组/矩阵变换

### 3. 全局配置 (config.py)
- `Tcam2base`：相机到基座变换矩阵
- `Rbase2cam`：基座到相机旋转矩阵
- `angle = 1/7`：边缘倾斜补偿（π/7弧度）
- `tcp_compensate = 0.018`：TCP补偿（米）

### 4. 模型加载
- 基因文件：`single_zy.txt`（NAS架构）
- 权重路径：绝对路径加载
- 默认设备：CUDA:0

---

## 关键非显而易见模式

### 1. CTU通信协议
- **格式**：SOF(2) + LEN(2) + DATA + CRC16(2)
- **心跳**：10秒间隔
- **命令字**：0x70(开始分拣), 0x71(调速), 0x80(物品数量), 0x81(开始抓取), 0x82(抓取完成)

### 2. 抓取流程
检测(MMDetection) → 抓取生成(AugmentCNN) → 边缘补偿(π/7) → 深度补偿(-0.18m) → 逆运动学 → 轨迹规划

### 3. 坐标变换链
```
图像(row, col) → 相机(x,y,z,内参) → 基座(Tcam2base矩阵) → 末端(机械臂控制)
```

### 4. 夹爪控制
- **协议**：Modbus RTU，地址0x01
- **功能码**：0x2a(控制), 0x2b(位置), 0x26(速度)
- **单位**：51200=1圈，最大5圈

### 5. 物体类别
```python
GoogsMapping = {
    "1": "soap",         # 肥皂
    "2": "interrupter",  # 空气开关
    "3": "terminal",     # 接线端子
    "4": "limit",        # 限位开关
    "5": "voltage"       # 电压采集模块
}
```

---

## ROS2重构指南

### 系统对比

| 维度 | 原系统 | ROS2系统 |
|------|--------|----------|
| 架构 | 单一脚本(600+行) | 5个功能包，模块化 |
| 接口 | 函数调用 | ROS2话题/服务/动作 |
| 配置 | 硬编码 | YAML参数文件 |
| 扩展性 | 需改源码 | 添加新节点 |
| 调试 | print/logging | ros2工具链 |

### ROS2包结构
```
Ros2/
├── grasp_interfaces/          # msg/srv/action定义
├── grasp_vision/              # camera_node, detection_server, grasp_generator
├── grasp_control/             # arm_controller, gripper_server
├── grasp_main/                # grasp_executor(动作服务器)
└── grasp_bringup/             # launch文件, config/grasp_params.yaml
```

### 核心逻辑映射
| 原系统方法 | ROS2节点/服务 |
|-----------|-------------|
| RS() | camera_node |
| MMDetection推理 | /detect_objects服务 |
| AugmentCNN推理 | /generate_grasp服务 |
| grasp_img2real_yolo | grasp_generator内部 |
| Algo_Inverse_Kinematics | arm_controller内部 |
| Movej_Cmd | arm_controller发布 |
| gripper_position | /gripper_control服务 |
| CTU循环 | grasp_executor动作 |

### 重构顺序
1. 确认ROS2版本功能完整性
2. 迁移CTU通信节点（缺失）
3. 集成坐标变换到TF2
4. 增强夹爪服务（速度/位置反馈）
5. 测试并切换到ROS2

---

## 重要提醒

1. **路径**：所有模型/数据使用绝对路径，修改时务必检查
2. **坐标系**：注意OpenCV(row, column)和NumPy(array)区别
3. **速度**：机械臂速度0-50级，**不是百分比**
4. **补偿**：边缘倾斜π/7，深度-0.18米是经验值
5. **协议**：CTU是**二进制**协议，非文本
6. **恢复**：有自动重连/重试，但次数有限
7. **硬件**：RealSense + RM65 SDK必须正确安装

---

## 网络配置

- **CTU**：192.168.127.253:8899
- **机械臂**：192.168.127.101:8080
- **开发板（有线）**：192.168.127.102
- **开发板（无线）**：192.168.2.51

---

## 联系方式

- **开发板**：jet/空格（支持sudo）
- **SSH端口**：33322
- **项目目录**：/home/jet/zoneyung/grasp_static
- **Conda环境**：zy_torch
