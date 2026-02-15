# 视觉机器人抓取系统

基于视觉的机器人抓取系统，Jetson Orin NX 部署，CTU 分拣任务。Intel RealSense 相机 + PyTorch/MMDetection 检测 + RM65 机械臂控制。

## 技术栈

Python 3.x | PyTorch + MMDetection | pyrealsense2 | RM65 SDK | Ubuntu 20.04 | Jetson Orin NX 8GB

## 安装

```bash
conda create -n zy_torch python=3.8
conda activate zy_torch
pip install torch torchvision mmdet pyrealsense2 opencv-python numpy
```

## 快速开始

```bash
conda activate zy_torch

# 主抓取程序
python grasp_zy_zhiyuan1215.py

# CTU 通信服务
python ctu_conn.py

# 调试工具
python RoboticArm.py      # 机械臂调试
python RoboticGripper.py  # 夹爪调试
python camera.py          # 相机测试
```

## 项目结构

```
grasp_zy/
├── grasp_zy_zhiyuan1215.py  # 主抓取程序
├── ctu_conn.py              # CTU 通信服务
├── config.py                # 全局配置
├── camera.py                # 相机接口
├── gripper_zhiyuan.py       # 夹爪控制
├── utils/                   # 工具库
├── robotic_arm_package/     # 机械臂 SDK
├── models/
│   ├── weights/             # 模型权重
│   ├── gqcnn_server/        # 抓取生成
│   └── mmdetection/         # 检测模型
└── graspnet-baseline/       # GraspNet 基线
```

## 配置 (config.py)

| 参数 | 说明 |
|------|------|
| `Tcam2base` | 相机到基座变换矩阵 |
| `angle = 1/7` | 边缘倾斜补偿 (π/7 弧度) |
| `tcp_compensate = 0.018` | TCP 补偿 (米) |
| `robot_speed` | 机械臂速度 (0-50 级) |

## 支持物体

soap (肥皂) | interrupter (空气开关) | terminal (接线端子) | limit (限位开关) | voltage (电压采集模块)

## 网络配置

| 设备 | IP | 端口 |
|------|-----|------|
| CTU | 192.168.127.253 | 8899 |
| 机械臂 | 192.168.127.101 | 8080 |
| 开发板 | 192.168.127.102 | - |

## 注意事项

- 机械臂速度 0-50 级（非百分比）
- Z 轴高度限制 0.538m
- 模型/数据使用绝对路径
- 详细文档见 [AGENTS.md](AGENTS.md)
