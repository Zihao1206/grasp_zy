# 真实相机测试指南

## 前提条件

- ✅ RealSense D435 相机已连接
- ✅ ROS2 Foxy 已安装
- ✅ 工作空间已编译

## 快速测试

### 方法 1：使用测试脚本（推荐）

```bash
# 终端 1：启动相机节点
cd /home/zh/zh/grasp_zy_zhiyuan/Ros2
source install/setup.bash
ros2 run grasp_vision camera_node

# 终端 2：运行测试
cd /home/zh/zh/grasp_zy_zhiyuan/Ros2
source install/setup.bash
python3 tests/test_real_camera.py
```

**预期输出**：
```
============================================================
RealSense 相机 ROS2 测试
============================================================

✓ 收到第一帧彩色图像:
  尺寸: 640x480
  编码: bgr8
  已保存: /tmp/ros2_color_first.jpg

✓ 收到第一帧深度图像:
  尺寸: 640x480
  深度范围: 0.300 - 2.500 米
  已保存: /tmp/ros2_depth_first.jpg

[彩色] 收到 30 帧, FPS: 10.02
[深度] 收到 30 帧, FPS: 10.01
...

============================================================
测试摘要
============================================================
测试时长: 30.02 秒
彩色图像: 300 帧 (平均 10.00 FPS)
深度图像: 300 帧 (平均 10.00 FPS)

✓✓✓ 相机测试通过！所有图像正常接收
```

### 方法 2：使用 ROS2 命令行工具

```bash
# 启动相机节点（终端 1）
ros2 run grasp_vision camera_node

# 查看话题（终端 2）
ros2 topic list
# 应该看到:
# /camera/color/image_raw
# /camera/depth/image_raw

# 查看话题信息
ros2 topic info /camera/color/image_raw

# 查看发布频率
ros2 topic hz /camera/color/image_raw

# 查看消息内容（只看一帧）
ros2 topic echo /camera/color/image_raw --once
```

### 方法 3：可视化查看

```bash
# 安装 rqt_image_view（如果未安装）
sudo apt install ros-foxy-rqt-image-view

# 启动相机节点（终端 1）
ros2 run grasp_vision camera_node

# 启动图像查看器（终端 2）
ros2 run rqt_image_view rqt_image_view
```

在 GUI 中：
1. 点击话题下拉菜单
2. 选择 `/camera/color/image_raw` 或 `/camera/depth/image_raw`
3. 应该能实时看到相机画面

## 调整相机参数

### 修改分辨率

```bash
# 启动时指定参数
ros2 run grasp_vision camera_node --ros-args \
    -p width:=1280 \
    -p height:=720
```

### 修改发布频率

```bash
# 高频发布（30Hz）
ros2 run grasp_vision camera_node --ros-args \
    -p publish_rate:=30.0

# 低频发布（5Hz）
ros2 run grasp_vision camera_node --ros-args \
    -p publish_rate:=5.0
```

### 使用启动文件

编辑 `grasp_bringup/launch/vision_only.launch.py`：

```python
Node(
    package='grasp_vision',
    executable='camera_node',
    name='camera_node',
    output='screen',
    parameters=[{
        'width': 1280,        # 修改分辨率
        'height': 720,
        'publish_rate': 15.0  # 修改频率
    }]
)
```

## 录制和回放数据

### 录制数据包

```bash
# 录制所有相机话题（30秒）
ros2 bag record -o camera_test \
    /camera/color/image_raw \
    /camera/depth/image_raw \
    --duration 30

# 或录制所有话题
ros2 bag record -a --duration 30
```

### 回放数据包

```bash
# 回放
ros2 bag play camera_test

# 查看数据包信息
ros2 bag info camera_test
```

## 常见问题排查

### Q1: 相机节点启动失败

**错误**：
```
RuntimeError: No device connected
```

**解决**：
```bash
# 检查 RealSense 设备
rs-enumerate-devices

# 如果看不到设备，尝试重新插拔USB
# 确保USB是3.0接口（蓝色）

# 检查权限
sudo usermod -aG video $USER
# 需要重新登录
```

### Q2: 图像频率太低

**检查**：
```bash
ros2 topic hz /camera/color/image_raw
```

**解决**：
1. 检查CPU使用率是否过高
2. 降低图像分辨率
3. 增加 `publish_rate` 参数

### Q3: 深度图像全是0

**原因**：可能距离太近或太远

**检查**：
```bash
# 查看深度值
ros2 topic echo /camera/depth/image_raw --once

# RealSense D435 有效范围: 0.3m - 3m
```

### Q4: 话题名称不对

**检查**：
```bash
ros2 topic list
```

确保看到：
- `/camera/color/image_raw`
- `/camera/depth/image_raw`

如果名称不同，需要修改其他节点的订阅。

### Q5: 权限问题

```bash
# 添加USB权限规则
sudo cp /opt/ros/foxy/lib/librealsense2/60-librealsense2-udev-rules.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## 性能基准

| 参数 | 值 |
|------|-----|
| 默认分辨率 | 640x480 |
| 默认频率 | 10 Hz |
| 彩色图像大小 | ~900 KB/帧 |
| 深度图像大小 | ~600 KB/帧 |
| CPU使用率 | ~15-20% |

## 下一步

相机测试通过后，可以继续测试：

1. **目标检测**（需要模型文件）
   ```bash
   ros2 run grasp_vision detection_server
   ```

2. **抓取姿态生成**（需要模型文件）
   ```bash
   ros2 run grasp_vision grasp_generator
   ```

3. **完整视觉流程**
   ```bash
   ros2 launch grasp_bringup vision_only.launch.py
   ```

## 保存的测试文件

测试脚本会自动保存：
- `/tmp/ros2_color_first.jpg` - 第一帧彩色图像
- `/tmp/ros2_depth_first.jpg` - 第一帧深度图像（可视化）

可以用图像查看器打开：
```bash
eog /tmp/ros2_color_first.jpg
eog /tmp/ros2_depth_first.jpg
```

## 集成到系统

相机测试通过后，可以：
1. 修改 `camera_node.py` 的参数
2. 更新启动文件
3. 与其他节点集成测试

## 参考

- RealSense 文档: https://dev.intelrealsense.com/
- ROS2 图像传输: https://docs.ros.org/en/foxy/Tutorials/Advanced/Image-Pipeline.html

