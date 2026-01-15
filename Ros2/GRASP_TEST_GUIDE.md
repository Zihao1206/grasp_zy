# 抓取系统测试指南

完整的测试流程，从视觉到控制再到完整抓取。

## 🎯 测试流程图

```
阶段1: 视觉测试 (不需要机械臂)
  ├─ 1.1 相机测试 ✅ (已完成)
  ├─ 1.2 目标检测测试
  └─ 1.3 抓取姿态生成测试

阶段2: 控制测试 (需要机械臂)
  ├─ 2.1 夹爪测试
  └─ 2.2 机械臂运动测试

阶段3: 完整流程测试
  └─ 3.1 端到端抓取测试
```

## 阶段 1：视觉系统测试（不需要机械臂）

### 1.1 相机测试 ✅

**已完成！** 参考：`TEST_REAL_CAMERA.md`

### 1.2 目标检测测试

测试 MMDetection 模型是否正常工作。

#### 启动节点

**终端 1 - 相机**：
```bash
cd /home/zh/zh/grasp_zy_zhiyuan/Ros2
source install/setup.zsh
ros2 run grasp_vision camera_node
```

**终端 2 - 检测服务**：
```bash
source install/setup.zsh
ros2 run grasp_vision detection_server
```

#### 运行测试

**终端 3**：
```bash
source install/setup.zsh
python3 tests/test_detection.py
```

**预期输出**：
```
============================================================
目标检测服务测试
============================================================

✓ 已收到图像
✓ 检测服务已连接
✓ 检测成功: 检测到3个物体

检测结果:
------------------------------------------------------------
检测到 3 个物体:
  [1] carrot          - 置信度: 0.956
  [2] banana          - 置信度: 0.887
  [3] soap            - 置信度: 0.823

✓ 可视化结果已保存: /tmp/ros2_detection_result.jpg

✓✓✓ 目标检测测试通过！
```

**查看结果**：
```bash
eog /tmp/ros2_detection_result.jpg
```

### 1.3 抓取姿态生成测试

测试抓取 CNN 模型是否正常工作。

#### 启动节点

**终端 1 - 相机**（如已启动则跳过）：
```bash
ros2 run grasp_vision camera_node
```

**终端 2 - 检测服务**（如已启动则跳过）：
```bash
ros2 run grasp_vision detection_server
```

**终端 3 - 抓取生成服务**：
```bash
source install/setup.zsh
ros2 run grasp_vision grasp_generator
```

#### 运行测试

**终端 4**：
```bash
source install/setup.zsh

# 测试抓取胡萝卜
python3 tests/test_grasp_generation.py carrot

# 或测试其他物体
python3 tests/test_grasp_generation.py banana
python3 tests/test_grasp_generation.py soap
```

**预期输出**：
```
============================================================
抓取姿态生成测试
============================================================

目标物体: carrot

[1/4] 等待相机图像...
✓ 已收到图像

[2/4] 执行目标检测...
✓ 检测到3个物体
  检测到: ['carrot', 'banana', 'soap']

[3/4] 生成抓取姿态...
✓ 成功生成5个抓取姿态

[4/4] 抓取姿态结果:
------------------------------------------------------------

抓取姿态 #1:
  图像位置: (240, 320)
  角度: 45.2°
  宽度: 68.5 px
  基座位置: (0.123, -0.045, 0.540) 米
  姿态: (0.0°, 0.0°, 45.2°)
  夹爪宽度: 0.042 米
  质量分数: 0.956
  边缘补偿: 否

✓✓✓ 抓取姿态生成测试通过！
✓ 可视化结果已保存: outputs/graspyolo_0.png
```

**查看结果**：
```bash
eog /home/zh/zh/grasp_zy_zhiyuan/outputs/graspyolo_0.png
```

---

## 阶段 2：控制系统测试（需要机械臂）

### ⚠️ 前提条件

- 机械臂已连接并通电
- 机械臂 IP: 192.168.127.101
- 网络连接正常

### 2.1 夹爪测试

#### 使用模拟节点测试（无硬件）

```bash
# 启动模拟夹爪
source install/setup.zsh
ros2 run grasp_control mock_gripper_server

# 在另一个终端测试
python3 tests/test_mock_services.py
```

#### 使用真实夹爪测试（有硬件）

```bash
# 启动真实夹爪服务
source install/setup.zsh
ros2 run grasp_control gripper_server

# 测试打开
ros2 service call /gripper_control grasp_interfaces/srv/GripperControl "{position: 1}"

# 测试闭合
ros2 service call /gripper_control grasp_interfaces/srv/GripperControl "{position: 0}"
```

### 2.2 机械臂运动测试

#### 测试基本运动

```bash
# 使用原项目的测试脚本
cd /home/zh/zh/grasp_zy_zhiyuan/Test/Code
python3 RoboticArm.py  # GUI 测试工具
```

或者使用 ROS2 启动文件测试：

```bash
# 启动完整系统（包括控制）
cd /home/zh/zh/grasp_zy_zhiyuan/Ros2
source install/setup.zsh
ros2 launch grasp_bringup grasp_system.launch.py
```

---

## 阶段 3：完整抓取测试

### 3.1 端到端抓取测试

#### 方法 1：使用启动文件

```bash
# 启动完整系统
cd /home/zh/zh/grasp_zy_zhiyuan/Ros2
source install/setup.zsh
ros2 launch grasp_bringup grasp_system.launch.py
```

等待所有节点启动后（约 5 秒），在另一个终端：

```bash
# 发送抓取命令
source install/setup.zsh

# 抓取胡萝卜
ros2 action send_goal /execute_grasp grasp_interfaces/action/ExecuteGrasp \
    "{target_label: 'carrot', visualize: true}"
```

#### 方法 2：使用 Python 客户端

```bash
source install/setup.zsh
cd examples

# 抓取胡萝卜
python3 grasp_client.py carrot

# 抓取香蕉
python3 grasp_client.py banana
```

**预期输出**：
```
抓取客户端已启动
发送抓取目标: carrot

目标已接受，等待结果...
[10.0%] 等待相机图像...
[20.0%] 执行目标检测...
[40.0%] 生成抓取姿态...
[60.0%] 执行机械臂运动...
[80.0%] 放置物体...
[100.0%] 抓取完成

抓取成功: 成功完成抓取任务
```

---

## 🔧 故障排查

### 问题 1：检测服务启动失败

**错误**：模型加载失败

**解决**：
```bash
# 检查模型文件是否存在
ls models/mmdetection/configs/myconfig_zy.py
ls models/weights/epoch_20_last.pth

# 检查路径是否正确
cd /home/zh/zh/grasp_zy_zhiyuan
```

### 问题 2：抓取生成服务启动失败

**错误**：找不到模型文件

**解决**：
```bash
# 检查基因文件和权重
ls doc/single_new.txt
ls models/test_250927_1644__zoneyung_/epoch_84_accuracy_1.00
ls dataset/cornell.data
```

### 问题 3：机械臂无法连接

**检查**：
```bash
# 测试网络连接
ping 192.168.127.101

# 检查端口
nc -zv 192.168.127.101 8080
```

### 问题 4：内存不足

如果 Jetson 内存不足：

```bash
# 关闭不需要的服务
# 一次只启动需要的节点

# 或降低图像分辨率
ros2 run grasp_vision camera_node --ros-args -p width:=320 -p height:=240
```

---

## 📊 测试检查清单

### 阶段 1：视觉测试
- [ ] 相机能发布图像
- [ ] 检测服务能识别物体
- [ ] 抓取生成能输出姿态
- [ ] 可视化结果正确

### 阶段 2：控制测试
- [ ] 夹爪能打开/闭合
- [ ] 机械臂能移动到指定位置
- [ ] 碰撞检测正常工作
- [ ] 急停和恢复正常

### 阶段 3：完整测试
- [ ] 能检测到目标物体
- [ ] 能生成有效抓取姿态
- [ ] 逆运动学求解成功
- [ ] 机械臂能准确到达抓取位置
- [ ] 夹爪能成功抓取物体
- [ ] 能将物体放置到目标位置
- [ ] 整个流程无碰撞

---

## 🎯 推荐测试顺序

### 如果**没有机械臂**：

1. ✅ 测试相机（已完成）
2. ⬜ 测试检测服务
3. ⬜ 测试抓取生成
4. ⬜ 使用模拟节点测试完整流程

### 如果**有机械臂**：

1. ✅ 测试相机（已完成）
2. ⬜ 测试检测服务
3. ⬜ 测试抓取生成
4. ⬜ 单独测试夹爪
5. ⬜ 单独测试机械臂运动
6. ⬜ 测试完整抓取流程

---

## 📝 测试记录模板

```
测试日期: ____________________
测试人员: ____________________

[ ] 相机测试           通过/失败  备注: __________
[ ] 检测服务测试        通过/失败  备注: __________
[ ] 抓取生成测试        通过/失败  备注: __________
[ ] 夹爪测试           通过/失败  备注: __________
[ ] 机械臂运动测试      通过/失败  备注: __________
[ ] 完整抓取测试        通过/失败  备注: __________

成功抓取次数: ____ / ____ (成功率: ___%)

问题记录:
_________________________________________________
_________________________________________________
```

---

## 下一步

测试完成后：
1. 调整参数优化性能
2. 记录成功率和失败案例
3. 根据需要调整算法
4. 部署为系统服务

