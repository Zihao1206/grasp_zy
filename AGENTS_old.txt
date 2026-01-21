# AGENTS.md

This file provides guidance to agents when working with code in this repository.

**注意：请用中文回答或者编写文档**

---

## 项目概述

这是一个基于视觉的机器人抓取系统，部署在Jetson Orin NX开发板上，用于CTU设备的智能分拣任务。系统通过Intel RealSense相机感知环境，使用深度学习模型检测物体并规划抓取姿态，控制RM65机械臂完成抓取和放置操作。

## 技术栈

- **编程语言**：Python 3.x
- **深度学习框架**：PyTorch + MMDetection
- **相机接口**：pyrealsense2 (Intel RealSense)
- **机械臂控制**：自定义SDK (`robotic_arm_package`)
- **通信协议**：自定义二进制协议 + CRC16校验
- **操作系统**：Ubuntu 20.04
- **部署环境**：Jetson Orin NX 8GB

## 核心命令

### 启动命令
```bash
# 激活conda环境
conda activate zy_torch

# 启动抓取程序
python grasp_zy_test.py          # 主抓取程序
python ctu_conn.py               # CTU通信服务
python RunCtu.py                 # CTU控制程序

# 调试工具
python RoboticArm.py             # 机械臂关节调试GUI
python RoboticGripper.py         # 夹爪调试GUI
python camera.py                 # 相机测试
```

### 系统服务
```bash
# systemd服务管理
sudo systemctl enable RunGraspd.service
sudo systemctl start RunGraspd.service
sudo systemctl status RunGraspd.service
sudo journalctl -u RunGraspd.service  # 查看日志
```

### 文件传输
```bash
# 从开发板拷贝文件
scp -rC -P 33322 jet@127.0.0.1:/home/jet/zoneyung/grasp_static/ ./
```

## 代码风格和约定

### 1. 硬编码路径
项目中大量使用硬编码的绝对路径，主要分布在：
- 模型权重路径：`weights/tune_epoch_20_loss_0.0478_accuracy_1.000`
- 数据文件路径：`single_zy.txt`, `cornell.data`
- 配置文件：`mmdetection/configs/myconfig_zy.py`
- 输出目录：`outputs/`, `zy/`

**注意**：修改代码时务必检查路径是否正确

### 2. 混合编程风格
- **OpenCV**：使用BGR格式，(row, column)坐标
- **PyTorch**：使用CUDA加速，张量操作
- **NumPy**：数组操作和矩阵变换
- **混合使用**：图像处理用OpenCV，深度学习用PyTorch

### 3. 全局配置
关键参数集中在`config.py`：
- `Tcam2base`：相机到基座的变换矩阵
- `Rbase2cam`：基座到相机的旋转矩阵
- `angle = 1/7`：边缘倾斜补偿角度（π/7弧度）
- `tcp_compensate`：TCP补偿量（0.018米）
- `place_last_pose`：放置位置

### 4. 模型加载
- 使用基因文件（`single_zy.txt`）存储NAS搜索的网络架构
- 模型权重使用绝对路径加载
- 默认使用CUDA:0设备

### 5. 错误处理模式
```python
# 典型的错误处理模式
try:
    # 操作代码
    pass
except Exception as e:
    logging.error(f"错误信息: {str(e)}")
    traceback.print_exc()
    # 恢复标志位
    self.grasp_running_flag = False
```

## 关键非显而易见模式

### 1. CTU通信协议
- **二进制协议**：自定义帧格式，CRC16校验
- **帧结构**：SOF(2) + LEN(2) + DATA + CRC16(2)
- **心跳机制**：10秒间隔自动发送
- **命令字**：
  - 0x70：CTU_GRASP_START（开始分拣）
  - 0x71：CTU_GRASP_SPEED（调速）
  - 0x80：GRASP_COUNT（物品数量）
  - 0x81：GRASP_START（开始抓取）
  - 0x82：GRASP_OVER（抓取完成）

### 2. 抓取流程
1. **检测阶段**：MMDetection检测物体，实例分割
2. **抓取生成**：自定义CNN生成抓取点(row, column, angle, width)
3. **边缘补偿**：根据位置应用π/7弧度倾斜
4. **深度补偿**：固定-0.18米偏移
5. **逆运动学**：求解关节角度
6. **轨迹规划**：先移动到上方，再下降抓取

### 3. 坐标变换链
```
图像坐标 (row, column)
    ↓
相机坐标 (x, y, z) - 使用RealSense内参
    ↓
基座坐标 - 使用Tcam2base变换矩阵
    ↓
末端坐标 - 机械臂运动控制
```

### 4. 夹爪控制
- **协议**：Modbus RTU
- **地址**：0x01（设备地址）
- **功能码**：0x2a（控制），0x2b（位置），0x26（速度）
- **位置单位**：51200 = 1圈，最大5圈
- **速度单位**：51200 = 1圈/秒

### 5. 物体类别映射
```python
GoogsMapping = {
    "1": "soap",         # 肥皂
    "2": "interrupter",  # 空气开关
    "3": "terminal",     # 接线端子
    "4": "limit",        # 限位开关
    "5": "voltage"       # 电压采集模块
}
```

### 6. 性能要求
- 单次抓取时间：< 5秒
- 模型推理时间：< 100ms
- 通信延迟：< 100ms
- 系统启动时间：< 30秒

### 7. 安全限制
- 机械臂速度：0-50级（不是百分比）
- Z轴高度限制：0.538米（防止碰撞）
- 碰撞检测：Set_Collision_Stage(4)
- 最大尝试次数：物品数 + 5次

### 8. 网络配置
- **CTU**：192.168.127.253:8899
- **机械臂**：192.168.127.101:8080
- **开发板**：192.168.127.102（有线）
- **开发板**：192.168.2.51（无线，CMCC-92FZ）

### 9. 模型文件
- **检测模型**：`weights/epoch_20.pth`
- **抓取模型**：`weights/tune_epoch_20_loss_0.0478_accuracy_1.000`
- **基因文件**：`single_zy.txt`（网络架构）
- **配置文件**：`cornell.data`

### 10. 调试技巧
- 设置`vis=True`可视化抓取点
- 使用`RoboticArm.py`测试关节6
- 使用`RoboticGripper.py`测试夹爪
- 运行`camera.py`检查相机
- 查看`outputs/`目录的调试图像

## 模式特定指南

### 调试模式（Debug Mode）
参见：`.roo/rules-debug/AGENTS.md`
- 日志调试方法
- 可视化调试技巧
- 网络调试工具
- 硬件调试GUI
- 常见错误处理

### 架构模式（Architect Mode）
参见：`.roo/rules-architect/AGENTS.md`
- 系统架构概览
- 数据流架构图
- 通信协议设计
- 抓取算法流程
- 坐标系统说明

### 问答模式（Ask Mode）
参见：`.roo/rules-ask/AGENTS.md`
- 项目背景说明
- 关键概念解释
- 常见问题解答
- 性能指标说明
- 扩展性指导

## 重要提醒

1. **路径问题**：所有模型和数据文件使用绝对路径，修改代码时务必检查
2. **坐标系**：注意OpenCV(row, column)和NumPy(array)的区别
3. **速度限制**：机械臂速度0-50级，不是百分比
4. **补偿参数**：边缘倾斜π/7弧度，深度-0.18米是经验值
5. **网络通信**：CTU使用二进制协议，不是文本协议
6. **错误恢复**：系统有自动重连和重试机制，但次数有限
7. **硬件依赖**：RealSense相机和RM65机械臂的SDK必须正确安装

## 联系方式

- **开发板**：jet/空格（支持sudo）
- **SSH端口**：33322
- **项目目录**：/home/jet/zoneyung/grasp_static
- **Conda环境**：zy_torch