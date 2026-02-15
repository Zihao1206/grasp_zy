# 视觉机器人抓取系统

> **项目正在向 ROS2 架构迁移中**

基于视觉的机器人抓取系统，Jetson Orin NX 部署，CTU 分拣任务。Intel RealSense 相机 + PyTorch/MMDetection 检测 + RM65 机械臂控制。

## 项目状态

| 分支 | 状态 | 说明 |
|------|------|------|
| `legacy/` | 维护模式 | 原始单体 Python 实现 |
| ROS2 | 开发中 | 模块化架构，推荐生产环境 |

## 项目结构

```
grasp_zy/
├── legacy/                    # 原始实现（已归档）
│   ├── grasp_zy_zhiyuan1215.py
│   ├── ctu_conn.py
│   ├── config.py
│   ├── models/
│   ├── utils/
│   └── ...
├── README.md
└── AGENTS.md                  # 详细文档
```

## 运行旧版代码

```bash
conda activate zy_torch

# 主抓取程序
python legacy/grasp_zy_zhiyuan1215.py

# CTU 通信服务
python legacy/ctu_conn.py

# 调试工具
python legacy/robotic_arm_package/RoboticArm.py
python legacy/camera.py
```

## 技术栈

Python 3.x | PyTorch + MMDetection | pyrealsense2 | RM65 SDK | Ubuntu 20.04 | Jetson Orin NX 8GB

## 详细文档

见 [AGENTS.md](AGENTS.md)
