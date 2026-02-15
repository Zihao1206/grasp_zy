# GraspNet 集成到现有抓取系统

本文档描述如何使用 `graspnet_wrapper.py` 将 GraspNet 集成到 `grasp_zy_zhiyuan0828.py` 系统中，替换现有的 GQCNN 抓取检测模型。

## 集成方案概述

使用 `GraspNetWrapper` 类作为新的抓取检测模块，替换原有的 `AugmentCNN`，保持机械臂控制、相机接口和目标检测模块不变。

## 快速集成（推荐）

### 1. 导入包装器

在 `grasp_zy_zhiyuan0828.py` 开头添加：

```python
# 添加 GraspNet 路径
GRASPNET_ROOT = '/home/zh/zh/graspnet-baseline'  # 修改为您的路径
sys.path.append(GRASPNET_ROOT)

# 导入包装器
from graspnet_wrapper import GraspNetWrapper
```

### 2. 初始化 GraspNetWrapper

在 `Grasp.__init__` 方法中添加：

```python
def __init__(self, hardware=False):
    # ... 原有代码 ...
    
    # 初始化 GraspNet 包装器
    self.graspnet_wrapper = GraspNetWrapper(
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
    self.graspnet_wrapper.set_camera_params(intrinsic, factor_depth=1000.0)
```

### 3. 替换抓取检测调用

#### 修改 `obj_grasp` 方法：

```python
def obj_grasp(self, label, vis=False):
    # ... 原有代码直到获取 mask ...
    
    # 替换抓取检测（原 GQCNN 代码）：
    # img_in = self.to_tensor(depth_img, color_img, rgb_include=1, depth_include=1)
    # k_max = 100
    # topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, black_image, pixel_wise_stride=1)
    
    # 新代码（使用 GraspNetWrapper）：
    topk_grasps, best_grasp = self.graspnet_wrapper.detect_grasps(
        depth_img_raw, color_img_raw, black_image, num_grasps=10
    )
    
    # 后续代码保持不变
    coordinate, ori, width_gripper, angle, z_compensate, slope_flag = self.grasp_img2real_yolo(
        color_img_raw, depth_img_raw, best_grasp, np.pi/7, 0,
        vis=vis, color=(0, 255, 0), note='', collision_check=True
    )
    
    # ... 剩余代码 ...
```

#### 修改 `action_yolo` 方法：

```python
def action_yolo(self, label, vis=False):
    # ... 原有代码 ...
    
    # 替换抓取检测（原 GQCNN 代码）：
    # img_in = self.to_tensor(depth_img, color_img, rgb_include=1, depth_include=0)
    # k_max = 100
    # topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, black_image, pixel_wise_stride=1)
    
    # 新代码（使用 GraspNetWrapper）：
    topk_grasps, best_grasp = self.graspnet_wrapper.detect_grasps(
        depth_img_raw, color_img_raw, black_image, num_grasps=10
    )
    
    # ... 剩余代码 ...
```

## 手动集成（备选）

如果不想使用包装器，可以参考 `graspnet_wrapper.py` 中的实现，手动将 GraspNet 集成到您的系统中。主要步骤包括：

1. **初始化模型**：加载 checkpoint 和创建 GraspNet 实例
2. **数据预处理**：深度图 → 点云 → 采样
3. **网络推理**：前向传播和抓取解码
4. **后处理**：碰撞检测、排序和坐标转换

详细信息请参考 `graspnet_wrapper.py` 源码。

## 注意事项

### 1. 深度单位
确保深度图单位与 `factor_depth` 匹配：
- 如果深度图是毫米：`factor_depth=1000.0`
- 如果深度图是米：`factor_depth=1.0`

### 2. 坐标系转换
`detect_grasps` 返回的是像素坐标，需要转换为机械臂基坐标系：

```python
# 在 grasp_img2real_yolo 中完成转换
# 需要 Tcam2base 变换矩阵
```

### 3. 性能优化
如果推理速度太慢：
- 减少 `num_point`（如 20000 → 10000）
- 禁用碰撞检测（`collision_thresh=-1`）
- 使用 TensorRT 加速

## 测试步骤

### 1. 测试包装器
```bash
python graspnet_wrapper.py
```

### 2. 测试集成系统（无硬件）
```python
grasp = Grasp(hardware=False)
grasp.obj_grasp(label='test', vis=True)
```

### 3. 完整测试（有硬件）
```python
grasp = Grasp(hardware=True)
grasp.obj_grasp(label='your_object', vis=False)
```

## 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 无抓取输出 | mask 不包含物体 | 检查 mask 生成逻辑 |
| 抓取在边框上 | 未使用 object_mask | 生成精确的 object_mask.png |
| 深度值异常 | factor_depth 设置错误 | 检查深度图单位 |
| 机械臂运动错误 | 逆解失败 | 检查目标位姿是否可达 |

## 文件说明

- `graspnet_wrapper.py`: GraspNet 包装器，提供简洁接口
- `GRASPNET_INTEGRATION.md`: 集成文档（本文档）
- `generate_object_mask.py`: 生成物体 mask 的工具

完成集成后，您的系统将使用 GraspNet 进行抓取检测，同时保持原有的机械臂控制、目标检测和用户界面功能。