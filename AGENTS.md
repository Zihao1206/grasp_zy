# AGENTS.md

**Generated:** 2026-02-16 | **Commit:** 55a5dfd | **Branch:** dev

**注意：请用中文回答或者编写文档**

---

## 项目概述

基于视觉的机器人抓取系统，Jetson Orin NX部署，CTU分拣任务。Intel RealSense D435相机 + PyTorch/MMDetection检测 + RM65机械臂控制。

**架构状态**：
- **legacy/**（原始Python系统）：已归档，维护模式，生产使用
- **ROS2系统**：设计阶段（见 `legacy/design.md`），代码尚未实现

> **ROS2仅为设计文档，当前生产系统使用 legacy/ 目录**

---

## 核心入口点

| 入口 | 文件 | 用途 |
|------|------|------|
| **主抓取** | `legacy/grasp_zy_zhiyuan1215.py` | Grasp类，完整抓取流水线 |
| **CTU服务** | `legacy/ctu_conn.py` | TCP通信服务，10秒心跳 |
| **相机测试** | `legacy/camera.py` | RS类，RealSense封装 |
| **夹爪调试** | `legacy/gripper_zhiyuan.py` | Modbus RTU控制 |

```bash
conda activate zy_torch
python legacy/grasp_zy_zhiyuan1215.py  # 主抓取程序
python legacy/ctu_conn.py              # CTU通信服务
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.x |
| 深度学习 | PyTorch + MMDetection 3.x |
| 相机 | pyrealsense2 (RealSense D435) |
| 机械臂 | robotic_arm_package SDK (ctypes) |
| 通信 | 自定义二进制 + CRC16-Modbus |

---

## 关键非显而易见模式

### 1. 坐标变换链 (grasp_img2real_yolo)
```
图像像素(row, col)
    ↓ intr^-1 @ [col+70, row, 1] * depth
相机坐标(x, y, z)
    ↓ Tcam2base @ [x, y, z, 1]
基座坐标
    ↓ 边缘补偿(9区域) + TCP补偿
末端位姿 [x, y, z, rx, ry, rz]
    ↓ Algo_Inverse_Kinematics
关节角度
```

### 2. 补偿值（硬编码）
| 参数 | 值 | 用途 |
|------|-----|------|
| `angle` | 1/7 (≈8.2°) | 边缘倾斜基础角度 |
| `tcp_compensate` | [0, 0, 0.018]米 | TCP Z轴补偿 |
| `t_tcp_flange` | [0, 0, 0.2]米 | TCP到法兰偏移 |
| `col_offset` | +70像素 | 图像裁剪补偿 |

### 3. CTU协议
- **帧格式**: `SOF(0x55AA) + LEN(2B) + DATA + CRC16(2B)`
- **数据段**: `SOD(0xA5) + CMD + DATA + EOD(0x5A)`
- **CRC**: 已禁用校验（`validate_crc16`始终返回True）
- **发送延迟**: 200ms强制延迟（`safe_send`）

### 4. 物体类别
```python
GoodsMapping = {"1":"soap", "2":"interrupter", "3":"terminal", "4":"limit", "5":"voltage"}
```

### 5. 模型加载流程
```python
# 1. 加载NAS基因
gene = open('doc/single_new.txt').readline()
genotype = gt.from_str(gene)  # genotypes.py

# 2. AugmentCNN抓取模型
model = AugmentCNN('dataset/cornell.data', 100, 4, 8, 5, False, genotype)

# 3. MMDetection检测模型
det_model = init_detector('configs/myconfig_zy.py', 'weights/epoch_20.pth')
```

---

## 项目结构

```
grasp_zy/
├── legacy/                           # 主代码（生产使用）
│   ├── grasp_zy_zhiyuan1215.py     # 主抓取程序 (622行)
│   ├── ctu_conn.py                 # CTU通信 (173行)
│   ├── ctu_protocol.py             # 协议编解码 (255行)
│   ├── camera.py                   # RealSense封装 (199行)
│   ├── config.py                   # 全局配置 (38行)
│   ├── gripper_zhiyuan.py          # 夹爪控制 (65行)
│   ├── utils/                      # [见 legacy/utils/AGENTS.md]
│   ├── robotic_arm_package/        # [见 legacy/robotic_arm_package/AGENTS.md]
│   ├── models/
│   │   ├── gqcnn_server/           # AugmentCNN [见 legacy/models/gqcnn_server/AGENTS.md]
│   │   ├── genotypes.py            # NAS架构定义
│   │   └── mmdetection/            # MMDetection (第三方)
│   ├── graspnet-baseline/          # [见 legacy/graspnet-baseline/AGENTS.md]
│   ├── design.md                   # ROS2设计文档（英文）
│   └── design.zh.md                # ROS2设计文档（中文）
└── AGENTS.md
```

### 文件命名问题
| 问题 | 文件 | 影响 |
|------|------|------|
| 日期后缀 | `grasp_zy_zhiyuan1215.py` | 版本控制反模式 |
| 无扩展名 | `RM_control` | 无语法高亮 |
| 括号 | `camera(2).py` | Shell需转义 |

---

## ROS2迁移状态

**当前状态**: 仅有设计文档，无实际代码

设计的7个包（见 `legacy/design.md`）：
1. zy_interfaces - 消息/服务/动作定义
2. zy_camera - RealSense驱动封装
3. zy_vision - 检测+抓取生成
4. zy_robot - 机械臂+夹爪控制
5. zy_comm - CTU通信
6. zy_executor - 任务编排
7. zy_bringup - 启动配置

---

## 反模式 (Anti-Patterns)

### 禁止
- ❌ 更改矩阵乘法顺序 `M = S @ T @ R` (utils/datasets.py:563)
- ❌ BGR图像直接用Matplotlib显示
- ❌ `from X import *` 污染命名空间（7处）
- ❌ `time.sleep()` 替代事件驱动（23处）

### 警告
- ⚠️ SDK非线程安全，ROS2需加互斥锁
- ⚠️ 逆运动学可能失败，必须检查返回tag
- ⚠️ 两套Tcam2base值（config.py vs 主程序）
- ⚠️ CRC校验已禁用（validate_crc16始终True）

### 技术债务
- 7处通配符导入（核心代码）
- 23处阻塞式`time.sleep()`
- 无项目级CI/CD
- 无测试框架

---

## 网络配置

| 设备 | 地址 |
|------|------|
| CTU | 192.168.127.253:8899 |
| 机械臂 | 192.168.127.101:8080 |
| 开发板(有线) | 192.168.127.102 |
| 开发板(无线) | 192.168.2.51 |

---

## 部署信息

- **开发板**: jet/空格 (支持sudo)
- **SSH端口**: 33322
- **项目目录**: /home/jet/zoneyung/grasp_static
- **Conda环境**: zy_torch
- **系统**: Ubuntu 20.04, Jetson Orin NX 8GB
