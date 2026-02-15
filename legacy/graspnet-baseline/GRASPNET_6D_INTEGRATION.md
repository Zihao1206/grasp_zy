# GraspNet 6D 位姿集成指南

本文档描述如何使用 `graspnet_wrapper_6d.py` 将 GraspNet 直接集成到 `grasp_zy_zhiyuan0828.py` 系统中，输出可直接使用的 6D 机械臂位姿。

## 集成优势

- **一步到位**：直接从 RGB-D 图像到机械臂 6D 位姿 (x, y, z, rx, ry, rz)
- **无需像素转换**：跳过像素坐标 ↔ 3D 坐标的来回转换，减少误差
- **接口简洁**：直接替换原有 GQCNN，保持系统架构不变

## 集成步骤

### 1. 导入 6D 包装器

在 `grasp_zy_zhiyuan0828.py` 开头添加：

```python
# 添加 GraspNet 路径
GRASPNET_ROOT = '/home/zh/zh/graspnet-baseline'  # 修改为您的路径
sys.path.append(GRASPNET_ROOT)

# 导入 6D 包装器
from graspnet_wrapper_6d import GraspNet6DWrapper
```

### 2. 初始化 6D 包装器

在 `Grasp.__init__` 方法中添加：

```python
def __init__(self, hardware=False):
    # ... 原有代码 ...
    
    # 初始化 GraspNet 6D 包装器
    self.graspnet_6d = GraspNet6DWrapper(
        checkpoint_path=os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar'),
        num_point=20000,
        collision_thresh=0.01
    )
    
    # 设置相机参数（RealSense）
    intrinsic = np.array([
        [392.25048828, 0, 320.16729736],
        [0, 392.25048828, 242.32826233],
        [0, 0, 1]
    ])
    self.graspnet_6d.set_camera_params(intrinsic, factor_depth=1000.0)
    
    # 设置变换矩阵（相机到机械臂）
    # 这个矩阵需要标定，这里使用示例值
    self.graspnet_6d.set_transform_matrix(self.Tcam2base)
```

### 3. 修改 obj_grasp 方法

替换原有的抓取检测和转换逻辑：

```python
def obj_grasp(self, label, vis=False):
    # ... 原有代码直到获取 mask ...
    
    # 替换原有的 GQCNN 抓取检测和转换：
    # 原代码：
    # img_in = self.to_tensor(depth_img, color_img, rgb_include=1, depth_include=1)
    # k_max = 100
    # topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, black_image, pixel_wise_stride=1)
    # coordinate, ori, width_gripper, angle, z_compensate, slope_flag = self.grasp_img2real_yolo(
    #     color_img_raw, depth_img_raw, best_grasp, np.pi/7, 0, vis=vis, color=(0, 255, 0), note='', collision_check=True
    # )
    # gesture = mat2euler(ori, axes='sxyz')
    # pose = np.hstack((coordinate, gesture))
    
    # 新代码（使用 GraspNet 6D）：
    grasp_poses = self.graspnet_6d.detect_grasps_6d(
        depth_img_raw, color_img_raw, black_image, num_grasps=10
    )
    
    if len(grasp_poses) == 0:
        print("No grasp detected!")
        return False
    
    # 选择最佳抓取
    best_pose = grasp_poses[0]  # (x, y, z, rx, ry, rz, width)
    
    # 提取位姿和宽度
    position = best_pose[:3]  # (x, y, z)
    gesture = best_pose[3:6]  # (rx, ry, rz)
    width_gripper = best_pose[6]  # 抓取宽度（米）
    
    # 组合成机械臂位姿
    pose = np.hstack((position, gesture))
    
    # 后续代码保持不变
    if self.hardware:
        self.robot.Movej_Cmd(self.mid_pose, self.robot_speed, 0)
        self.robot.Movej_Cmd(self.mid_pose1, self.robot_speed, 0)
        
        # 直接使用 pose，无需额外转换
        print(f"Grasp pose: {pose}")
        print(f"Gripper width: {width_gripper}")
        
        # 调整高度（根据物体高度）
        pose[2] = 0.540  # 根据实际情况调整
        if pose[2] > 0.534:
            pose[2] = 0.550
        
        # 计算接近位置
        pose_up_to_grasp_position = pose + [0, 0, -0.08, 0, 0, 0]
        
        # 执行抓取
        tag, up_to_grasp_joint = self.robot.Algo_Inverse_Kinematics(self.mid_pose1, pose_up_to_grasp_position, 1)
        # ... 剩余执行代码 ...
```

### 4. 修改 action_yolo 方法

同样替换抓取检测逻辑：

```python
def action_yolo(self, label, vis=False):
    # ... 原有代码 ...
    
    # 替换原有的 GQCNN 抓取检测：
    # 原代码：
    # img_in = self.to_tensor(depth_img, color_img, rgb_include=1, depth_include=0)
    # k_max = 100
    # topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, black_image, pixel_wise_stride=1)
    # coordinate, width_gripper, angle = self.grasp_img2real_yolo(color_img_raw, depth_img_raw, best_grasp, 0, vis=vis, color=(0, 255, 0), note='', collision_check=True)
    
    # 新代码（使用 GraspNet 6D）：
    grasp_poses = self.graspnet_6d.detect_grasps_6d(
        depth_img_raw, color_img_raw, black_image, num_grasps=10
    )
    
    if len(grasp_poses) == 0:
        print("No grasp detected!")
        return False
    
    # 选择最佳抓取
    best_pose = grasp_poses[0]
    position = best_pose[:3]
    gesture = best_pose[3:6]
    width_gripper = best_pose[6]
    
    # 组合成机械臂位姿
    pose = np.hstack((position, gesture))
    
    # 后续代码保持不变
    if self.hardware:
        self.robot.Movej_Cmd(self.mid_pose, self.robot_speed, 0)
        self.robot.Movej_Cmd(self.mid_pose1, self.robot_speed, 0)
        
        # 调整高度
        pose[2] = 0.546
        if pose[2] > 0.534:
            pose[2] = 0.546
        
        # 计算接近位置
        pose_up_to_grasp_position = pose + [0, 0, -0.05, 0, 0, 0]
        
        # 执行抓取
        tag, up_to_grasp_joint = self.robot.Algo_Inverse_Kinematics(self.mid_pose1, pose_up_to_grasp_position, 1)
        # ... 剩余执行代码 ...
```

## 关键参数说明

### 1. 相机内参
```python
intrinsic = np.array([
    [392.25048828, 0, 320.16729736],
    [0, 392.25048828, 242.32826233],
    [0, 0, 1]
])
```
- **fx, fy**: 焦距（像素单位）
- **cx, cy**: 主点坐标（像素单位）

### 2. 变换矩阵
```python
Tcam2base = np.array([
    [-0.01537554, -0.99988175, -0.00028888, 0.2070103],
    [ 0.9998815  ,-0.01537576 , 0.00076007,-0.03249003],
    [-0.00076442 ,-0.00027716,  0.99999967, 0.02642268],
    [0., 0., 0., 1.]
])
```
- **需要标定**：这个矩阵需要根据实际情况测量或标定
- **格式**：4x4 齐次变换矩阵 [R | t; 0 0 0 1]

### 3. 返回的 6D 位姿
```python
best_pose = (x, y, z, rx, ry, rz, width)
```
- **x, y, z**: 抓取位置（米，机械臂基坐标系）
- **rx, ry, rz**: 抓取方向（弧度，欧拉角，机械臂基坐标系）
- **width**: 抓取宽度（米）

## 测试步骤

### 1. 测试 6D 包装器
```bash
python graspnet_wrapper_6d.py
```

### 2. 测试集成（无硬件）
```python
grasp = Grasp(hardware=False)
grasp.obj_grasp(label='test', vis=True)
```

### 3. 完整测试（有硬件）
```python
grasp = Grasp(hardware=True)
grasp.obj_grasp(label='carrot', vis=False)
```

## 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 抓取位置错误 | Tcam2base 矩阵不正确 | 重新标定相机和机械臂的相对位置 |
| 抓取方向错误 | 欧拉角顺序不匹配 | 检查机械臂使用的欧拉角顺序（XYZ/ZYX） |
| 没有抓取输出 | mask 不包含物体 | 检查 mask 生成逻辑，确保物体在 mask 内 |
| 机械臂逆解失败 | 目标位姿不可达 | 调整抓取高度或位置，确保在机械臂工作空间内 |

## 标定指南

### 相机-机械臂标定

1. **眼在手外（Eye-to-Hand）**：
   - 固定标定板在机械臂末端
   - 移动机械臂到多个位置，拍摄标定板
   - 使用 AX=XB 方程求解 Tcam2base

2. **眼在手上（Eye-in-Hand）**：
   - 固定标定板在某个位置
   - 移动机械臂，从多个角度拍摄标定板
   - 使用 AX=XB 方程求解 Tcam2base

3. **简化方法**：
   ```python
   # 手动测量相机和机械臂的相对位置
   # 测量相机在机械臂基坐标系下的位置和姿态
   position_cam_in_base = np.array([x, y, z])  # 单位：米
   rotation_cam_in_base = np.array([rx, ry, rz])  # 单位：弧度
   
   # 构建变换矩阵
   Tcam2base = np.eye(4)
   Tcam2base[0:3, 0:3] = R.from_euler('xyz', rotation_cam_in_base).as_matrix()
   Tcam2base[0:3, 3] = position_cam_in_base
   ```

完成集成后，您的系统将直接从 RGB-D 图像输出机械臂可执行的 6D 位姿，无需中间转换步骤。