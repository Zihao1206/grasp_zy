# ROS2 包架构设计文档

## 项目概述

**项目**：基于视觉的机器人抓取系统 (grasp_zy)
**目标平台**：Jetson Orin NX 上的 ROS2
**目标**：将单体式 Python 系统重构为模块化 ROS2 包

---

## 设计理由

### 关键设计决策

#### 1. 分离相机和推理包

**决策**：将 `zy_vision` 拆分为 `zy_camera` 和 `zy_vision`

**理由**：

- **依赖隔离**：相机包只需要 pyrealsense2；视觉包需要 PyTorch
- **灵活部署**：可以将仅相机部署到没有 GPU 的边缘设备
- **资源优化**：没有推理的设备不会加载重型库
- **遵循 ROS2 标准**：匹配 real sense 驱动 (real sense2_camera)、yolo_ros、fetch_robotics

**包数量**：7 个包（编排分离为独立包）

#### 2. 无单独校准包

**决策**：**不**创建 `zy_calib` 包

**理由**：

- **内参参数**：通过 camera_node 的 `sensor_msgs/CameraInfo` 发布
- **外参参数**：通过 TF2 static_transform_publisher 发布
- **补偿参数**：内联嵌入节点中或通过 YAML 参数
- **避免混淆**：`zy_calib` 暗示校准工具包功能，这不是我们的意图

**考虑并拒绝的替代方案**：

- `zy_config`（太通用）
- `zy_common`（太宽泛，范围不清）
- `zy_calib`（暗示校准工具包功能）

#### 3. 内联工具函数

**决策**：将常用的工具函数直接嵌入到使用它们的节点中

**理由**：

- **使用分析**：utils/utils.py（770 行）中只有 2 个函数实际被使用：
    - `in_paint` - 仅在 grasp_generator 中使用
    - `letterbox` / `letterbox1` - 仅在 grasp_generator 中使用
- **代码质量**：10+ 个函数从未被调用（死代码）
- **依赖关系**：避免复杂的包间 Python 模块依赖

**迁移的函数**：

- `in_paint()` → zy_vision/grasp_generator.py
- `letterbox()` → zy_vision/grasp_generator.py
- `scale_coords()` → zy_vision/grasp_generator.py

**未迁移的函数**（死代码）：

- `float3`, `init_seeds`, `model_info`, `weights_init_normal`, `coco_class_weights` 等

#### 4. 集中式 YAML 配置

**决策**：所有 YAML 配置位于 `zy_bringup/config/`

**理由**：

- **统一管理**：所有运行时参数的单一位置
- **ROS2 标准**：遵循 ros2 参数加载模式
- **易于更新**：无需更改代码即可修改参数
- **类型安全**：校准数据使用 Python（如果需要）或静态 TF

#### 5. 无单独 TF 包

**决策**：使用 launch 文件发布静态 TF，而不是 `zy_tf` 包

**理由**：

- **更简单**：`tf2_ros` 包提供节点，无需包装
- **标准**：所有 ROS2 项目直接在 launch 文件中使用 static_transform_publisher
- **更少代码**：发布 TF 不需要自定义 Python 代码

#### 6. 独立编排包

**决策**：创建独立的 `zy_executor` 包用于任务编排，**不**合并到 `zy_comm`

**理由**：

- **单一职责**：`zy_comm` 只负责 TCP 通信；`zy_executor` 只负责编排逻辑
- **独立部署**：CTU 通信可部署在靠近 CTU 的边缘设备；编排运行在主机器人上
- **清晰的测试边界**：每个包可以独立测试（模拟 CTU 命令，模拟服务响应）
- **灵活升级**：通信协议更改不影响编排逻辑；编排算法更改不影响通信
- **可扩展性**：未来可轻松添加多个任务源（不仅限于 CTU）或多个执行器
- **标准 ROS2 模式**：匹配实际架构（Nav2 的独立 BT 执行器，工业系统的独立通信/编排层）

**包分离优势**：

- **zy_comm**：纯 TCP Socket + 二进制协议处理 → 轻量级，最小依赖
- **zy_executor**：ROS2 编排逻辑 → 服务客户端、话题发布者、状态管理
- **松耦合**：通过 ROS2 主题（`/ctu/command`）通信，而非函数调用

**数据流**：

  ```
  CTU 设备
      ↓ (TCP 二进制协议)
  zy_comm/ctu_communication
      ↓ (发布 ROS2 主题)
  /ctu/command (CTUCommand.msg)
      ↓
  zy_executor/ctu_orchestrator
      ↓ (订阅 + 编排服务)
  ├─→ /detect_objects (服务调用)
  ├─→ /generate_grasp (服务调用)
  ├─→ /gripper_control (服务调用)
  └─→ /arm_grasp_command (主题发布)
  ```

---

## 包结构

```
Ros2/
├── zy_interfaces/              # 接口定义
│   ├── msg/
│   │   ├── DetectionResult.msg
│   │   └── GraspPose.msg
│   ├── srv/
│   │   ├── DetectObjects.srv
│   │   ├── GenerateGrasp.srv
│   │   └── GripperControl.srv
│   └── action/
│       └── ExecuteGrasp.action
│
├── zy_camera/                   # 相机驱动（使用官方 realsense-ros）
│   ├── config/
│   │   └── camera_params.yaml         # 相机配置（ROS2 标准位置）
│   └── launch/
│       └── camera_bringup.launch.py    # 启动 realsense2_camera_node
│
├── zy_vision/                 # 视觉推理（修改）
│   ├── nodes/
│   │   ├── detection_server.py
│   │   └── grasp_generator.py
│   ├── launch/
│   │   └── inference_bringup.launch.py
│   └── config/
│       └── inference_params.yaml
│
├── zy_robot/                  # 机器人控制
│   ├── nodes/
│   │   ├── arm_controller.py
│   │   └── gripper_server.py
│   ├── launch/
│   │   └── robot_bringup.launch.py
│   └── config/
│       └── robot_params.yaml
│
  ├── zy_comm/                   # CTU 通信（新）
  │   ├── nodes/
  │   │   └── ctu_communication.py
  │   ├── protocols/
  │   │   └── ctu_protocol.py
  │   ├── launch/
  │   │   └── comm_bringup.launch.py
  │   └── config/
  │       └── comm_params.yaml
  │
  ├── zy_executor/               # 任务编排（新）
  │   ├── nodes/
  │   │   └── ctu_orchestrator.py
  │   ├── launch/
  │   │   └── executor_bringup.launch.py
  │   └── config/
  │       └── executor_params.yaml
  │
 └── zy_bringup/               # 系统启动器
     ├── launch/
     │   ├── grasp_system.launch.py      # 完整系统
     │   ├── static_tfs.launch.py       # TF 发布
     │   └── camera_calib.launch.py     # CameraInfo 参数
     └── config/
         ├── camera_params.yaml           # 相机内参
         ├── robot_params.yaml            # 机器人配置
         └── comm_params.yaml            # 通信配置
 ```

---

## 包职责

### zy_interfaces

**角色**：定义通信契约
**包含**：msg, srv, action 定义
**依赖**：无（纯定义）

---

### zy_camera

**角色**：RealSense D435i 相机驱动包装（使用官方 realsense-ros）
**无自定义节点**：使用 Intel 官方的 `realsense2-camera` 包

**功能**：

- 配置和启动官方 realsense-ros 驱动
- 设置分辨率、帧率、对齐等参数
- 配置命名空间和话题名称
- 静态 TF 发布（camera 到 base_link 的变换）

**发布的主题**（由 realsense-ros 发布）：

- `/camera/camera/color/image_raw` (sensor_msgs/Image)
- `/camera/camera/color/camera_info` (sensor_msgs/CameraInfo)
- `/camera/camera/depth/image_rect_raw` (sensor_msgs/Image)
- `/camera/camera/depth/camera_info` (sensor_msgs/CameraInfo)
- `/camera/camera/aligned_depth_to_color/image_raw` (sensor_msgs/Image) - 对齐后的深度
- `/camera/camera/aligned_depth_to_color/camera_info` (sensor_msgs/CameraInfo)

**配置** (`launch/camera_bringup.launch.py`)：

```python
# camera_bringup.launch.py - 使用自定义参数启动 realsense-ros
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense2_camera_node',
            parameters=[{
                # 相机配置
                'camera_name': 'camera',
                'camera_namespace': 'camera',

                # 启用流
                'enable_color': True,
                'enable_depth': True,
                'enable_sync': True,

                # D435i 特有：启用 IMU
                'enable_gyro': True,
                'enable_accel': True,
                'unite_imu_method': '2',  # 线性插值

                # 分辨率和帧率（为抓取优化）
                'rgb_camera.color_profile': '640x480x30',
                'depth_module.depth_profile': '640x480x90',  # 高帧率用于抓取

                # 对齐深度到彩色
                'align_depth.enable': True,

                # 滤波器（减少噪声）
                'pointcloud.enable': False,  # 抓取不需要
                'spatial_filter.enable': True,
                'temporal_filter.enable': True,

                # 坐标系
                'base_frame_id': 'camera_link',
                'publish_tf': True,
            }],
            output='screen'
        )
    ])
```

**配置** (`launch/camera_params.yaml`)：

```yaml
# 备选方案：使用 YAML 文件配置相机参数
camera_name: camera
camera_namespace: camera

# 流配置
enable_color: true
enable_depth: true
enable_sync: true

# D435i IMU
enable_gyro: true
enable_accel: true
unite_imu_method: 2  # 线性插值

# 分辨率和 FPS
rgb_camera:
  color_profile: 640x480x30
depth_module:
  depth_profile: 640x480x90  # 高帧率用于动态抓取场景

# 后处理
align_depth:
  enable: true
spatial_filter:
  enable: true
temporal_filter:
  enable: true

# 坐标系
base_frame_id: camera_link
publish_tf: true
```

**依赖**：

- `realsense2-camera` (Intel 官方 ROS2 驱动 - 通过 apt 安装)
- **无 Python 依赖**（纯配置包）

**安装**：

```bash
# 在 Ubuntu/Jetson 上安装 realsense-ros
sudo apt install ros-${ROS_DISTRO}-realsense2-camera

# 或从源码编译（推荐用于 Jetson）
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master
cd realsense-ros
colcon build
source install/setup.bash
```

**关键设计**：
- **无自定义代码**：使用 Intel 经过实战测试的驱动
- **仅配置**：zy_camera 提供启动文件和参数
- **自动 CameraInfo**：realsense-ros 自动发布 CameraInfo
- **D435i 优化**：高深度帧率（90Hz）用于动态抓取场景
- **标准主题**：使用 realsense-ros 主题命名约定
- **静态 TF**：在单独的启动文件中发布 camera_link 变换

---

### zy_vision

**角色**：视觉推理（检测 + 抓取生成）
**主要节点**：`detection_server.py`, `grasp_generator.py`

**功能**：

#### detection_server.py

- 订阅：`/camera/color/image_raw`
- 提供服务：`/detect_objects` (DetectObjects.srv)
- 加载 MMDetection 模型
- 执行 NMS 过滤

#### grasp_generator.py

- 订阅：`/camera/color/image_raw`, `/camera/depth/image_raw`
- 订阅：`/camera/color/camera_info`, `/camera/depth/camera_info`
- 订阅服务：`/detect_objects`
- 提供服务：`/generate_grasp` (GenerateGrasp.srv)
- 加载 AugmentCNN 模型
- 嵌入工具函数：`in_paint()`, `letterbox()`, `scale_coords()`

**配置** (`inference_params.yaml`)：

```yaml
detection:
  config_file: 'models/mmdetection/configs/myconfig_zy.py'
  checkpoint: 'models/weights/epoch_20_last.pth'
  device: 'cuda'
  nms_score_threshold: 0.8
  nms_iou_threshold: 0.9

grasp_generation:
  gene_file: 'doc/single_new.txt'
  cornell_data: 'dataset/cornell.data'
  model_weights: 'models/test_250927_1644__zoneyung_/epoch_84_accuracy_1.00'
  device: 'cuda'
  top_k: 100
```

**依赖**：

- `sensor_msgs` (标准 ROS2)
- `cv_bridge` (标准 ROS2)
- `PyTorch`
- `MMDetection`
- 自定义模型 (AugmentCNN)

**关键设计**：

- 与相机驱动分离以便独立部署
- 工具函数内联以避免复杂的 Python 模块依赖
- 使用来自 camera_node 的 CameraInfo 进行精确的坐标转换

---

### zy_robot

**角色**：机械臂和夹爪控制
**主要节点**：`arm_controller.py`, `gripper_server.py`

**功能**：

#### arm_controller.py

- 订阅：`/arm_grasp_command`（抓取位姿命令）
- 发布：`/arm_status`（状态更新）
- 提供逆运动学
- 碰撞检测和恢复
- 预定义位姿（init, mid, place 等）
- 执行完整抓取序列（grasp → place），当接收抓取命令时

#### gripper_server.py

- 提供服务：`/gripper_control` (GripperControl.srv)
- Modbus RTU 通信
- 位置控制（0-1 范围）

**配置** (`robot_params.yaml`)：

```yaml
arm:
  ip: '192.168.127.101'
  port: 8080
  speed: 20  # 0-50
  collision_stage: 5

  # Predefined poses
  poses:
    init: [ 86, -129, 127, -0.8, 71, -81 ]
    mid: [ 0, -129, 127, -0.7, 71, -81 ]
    mid1: [ 0, -129, 80, -0.7, 100, -81 ]
    lift2init: [ 65, -129, 127, -0.7, 77, 1 ]
    place_mid: [ 65, -129, 60, 0, 121, 1 ]
    place_mid2: [ 69, -129, 60, 0, 9, 1 ]
    place_last: [ 69, -104, 38, -2, 9, 1 ]

gripper:
  max_position: 256000  # 5 turns * 51200
  voltage: 3
  modbus_baudrate: 115200
```

**依赖**：

- `sensor_msgs` (标准 ROS2)
- `std_msgs` (标准 ROS2)
- `robotic_arm_package` (RM65 SDK)

**关键设计**：通过 ROS2 服务进行硬件抽象 - 其他节点无需了解 SDK 细节。

---

### zy_comm

**角色**：CTU 设备通信（仅 TCP Socket + 协议解析）
**主要节点**：`ctu_communication.py`

**功能**：

- 与 CTU 的 TCP 连接（192.168.127.253:8899）
- 心跳机制（10秒间隔）
- 协议解析（二进制 + CRC16）
- 发布 CTU 命令到 ROS2 主题：`/ctu/command`
- 订阅执行器状态：`/executor/status`（向 CTU 反馈）

**配置** (`comm_params.yaml`)：

  ```yaml
  ctu:
    ip: '192.168.127.253'
    port: 8899
    heartbeat_interval: 10  # seconds
    reconnect_interval: 1
    max_reconnect_interval: 30

  protocol:
    sof: b'\x55\xAA'
    crc_polynomial: 0xA001
  ```

**依赖**：

- `std_msgs` (标准 ROS2)
- `zy_interfaces`（消息定义：CTUCommand）

**关键设计**：

- **纯通信**：无编排逻辑，仅 TCP + 协议处理
- **ROS2 桥接**：将 CTU 二进制协议转换为 ROS2 消息
- **无状态**：无编排状态；执行器维护所有状态
- **独立**：可与编排分开部署

  ---

### zy_executor

**角色**：任务编排（协调视觉服务和机械臂控制）
**主要节点**：`ctu_orchestrator.py`

**功能**：

- 订阅 CTU 命令：`/ctu/command`（来自 zy_comm）
- 协调视觉服务和机械臂控制
- 状态管理：grasp_running 标志、尝试计数器、逆解失败跟踪
- 服务客户端：`/detect_objects`, `/generate_grasp`, `/gripper_control`
- 话题发布：`/arm_grasp_command`（向 arm_controller 发送抓取位姿）
- 话题发布：`/executor/status`（向 CTU 报告编排状态）
- 订阅机械臂状态：`/arm_status`（等待机械臂执行完成）

**嵌入式映射**（无需单独配置文件）：

  ```python
  GOODS_MAPPING = {
    "1": "soap",
    "2": "interrupter",
    "3": "terminal",
    "4": "limit",
    "5": "voltage"
}
  ```

**嵌入式状态变量**（简单标志位，无复杂状态机）：

  ```python
  # 编排状态
self.grasp_running = False  # 防止并发命令
self.grasp_count = 0  # 已抓取次数
self.inverse_failures = 0  # 逆解失败次数
self.max_attempts = 2  # 单次抓取重试上限
self.max_inverse_failures = 3  # 逆解失败总数上限
  ```

**配置** (`executor_params.yaml`)：

  ```yaml
  grasp:
    max_attempts: 2
    max_inverse_failures: 3

  orchestration:
    service_timeout: 10.0  # seconds
    arm_timeout: 30.0      # seconds
  ```

**依赖**：

- `std_msgs` (标准 ROS2)
- `zy_interfaces`（消息定义）

**关键设计**：

- **纯编排**：无 TCP 通信，仅 ROS2 主题/服务
- **决策驱动循环**：基于检测结果的 while 循环，而非固定迭代
- **状态机**：简单标志位状态（IDLE、DETECTING、PLANNING、EXECUTING）
- **向 CTU 反馈**：发布状态到 `/executor/status` 供 CTU 通信节点使用
- **服务协调**：顺序调用检测 → 生成 → 抓取服务

 ---

### zy_bringup

**角色**：系统启动器和配置中心
**Launch 文件**：

#### static_tfs.launch.py

使用 `tf2_ros/static_transform_publisher` 发布静态坐标变换：

```python
from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_launch_description():
    # Tcam2base matrix (from config.py migration)
    T_cam2base = np.array([
        [0.02742095, -0.99940903, -0.0207286, 0.20841901],
        [0.9995487, 0.02766746, -0.01170045, -0.02848768],
        [0.01226705, -0.02039841, 0.99971667, 0.03739014],
        [0., 0., 0., 1., ]
    ])

    # Extract translation
    translation = T_cam2base[:3, 3]  # [0.20841901, -0.02848768, 0.03739014]

    # Convert rotation matrix to quaternion
    rotation_matrix = T_cam2base[:3, :3]
    r = R.from_matrix(rotation_matrix)
    qx, qy, qz, qw = r.as_quat()

    return LaunchDescription([
        # Camera to base transform (quaternion format)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_to_base_tf',
            arguments=[
                str(translation[0]), str(translation[1]), str(translation[2]),  # x, y, z
                str(qx), str(qy), str(qz), str(qw),  # qx, qy, qz, qw
                'base_link',  # parent frame
                'camera_color_optical_frame'  # child frame
            ],
            output='screen'
        ),

        # TCP to end-effector compensation transform
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tcp_tf',
            arguments=[
                '0.0', '0.0', '0.018',  # TCP compensation offset
                '0.0', '0.0', '0.0', '1.0',
                'ee_link',
                'tcp_link'
            ],
            output='screen'
        )
    ])
```

**关键设计**：

- 使用标准 `tf2_ros` 包
- 无需自定义 Python 代码
- 外参校准 (Tcam2base) 矩阵转换为四元数
- 相机基座和 TCP 补偿的单独变换

#### camera_calib.launch.py

使用 YAML 中的内参启动 camera_node：

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare camera parameters
    return LaunchDescription([
        Node(
            package='zy_camera',
            executable='camera_node',
            name='camera_node',
            parameters=[{
                'width': 640,
                'height': 480,
                'fx': 604.335,
                'fy': 604.404,
                'cx': 316.187,
                'cy': 248.611,
                'publish_rate': 10.0
            }],
            output='screen'
        )
    ])
```

**关键设计**：

- 内参从 launch 文件传递到 camera_node
- CameraInfo 由 camera_node 使用这些参数发布
- 其他节点订阅 `/camera/color/camera_info` 获取内参

#### grasp_system.launch.py

主系统启动文件 - 编排所有包：

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([
        # 1. Static TFs (from static_tfs.launch.py)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(['static_tfs.launch.py'])
        ),

        # 2. Camera node (with intrinsics)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(['../zy_camera/launch/camera_bringup.launch.py'])
        ),

        # 3. Detection service (delayed)
        TimerAction(
            period=1.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(['../zy_vision/launch/inference_bringup.launch.py'])
                )
            ]
        ),

        # 4. Arm controller (delayed)
        TimerAction(
            period=1.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(['../zy_robot/launch/robot_bringup.launch.py'])
                )
            ]
        ),

        # 5. Gripper service (delayed)
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='zy_robot',
                    executable='gripper_server',
                    name='gripper_server',
                    parameters=['config/robot_params.yaml'],
                    output='screen'
                )
            ]
        ),

        # 6. CTU communication (delayed)
        TimerAction(
            period=3.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(['../zy_comm/launch/comm_bringup.launch.py'])
                )
            ]
        ),
    ])
```

**关键设计**：

- 使用 IncludeLaunchDescription 引用包特定的启动文件
- 包独立性：每个包都有自己的启动文件
- 定时启动：确保依赖项已就绪
- 中央编排：单个命令启动整个系统

---

## 通信架构

### ROS2 主题和服务

| 主题/服务                       | 类型                               | 发布者                          | 订阅者                                                     | 用途             |
 |-----------------------------|----------------------------------|------------------------------|---------------------------------------------------------|----------------|
| `/camera/color/image_raw`   | sensor_msgs/Image                | zy_camera/camera_node        | zy_vision/detection_server, zy_vision/grasp_generator   | RGB 图像数据       |
| `/camera/color/camera_info` | sensor_msgs/CameraInfo           | zy_camera/camera_node        | zy_vision/detection_server, zy_vision/grasp_generator   | 内参             |
| `/camera/depth/image_raw`   | sensor_msgs/Image                | zy_camera/camera_node        | zy_vision/grasp_generator                               | 深度图像数据         |
| `/camera/depth/camera_info` | sensor_msgs/CameraInfo           | zy_camera/camera_node        | zy_vision/grasp_generator                               | 内参             |
| `/detect_objects`           | zy_interfaces/srv/DetectObjects  | zy_vision/detection_server   | zy_vision/grasp_generator, zy_executor/ctu_orchestrator | 物体检测服务         |
| `/generate_grasp`           | zy_interfaces/srv/GenerateGrasp  | zy_vision/grasp_generator    | zy_executor/ctu_orchestrator                            | 抓取位姿生成服务       |
| `/arm_grasp_command`        | zy_interfaces/msg/GraspCommand   | zy_executor/ctu_orchestrator | zy_robot/arm_controller                                 | 机械臂抓取位姿命令      |
| `/gripper_control`          | zy_interfaces/srv/GripperControl | zy_robot/gripper_server      | zy_robot/arm_controller, zy_executor/ctu_orchestrator   | 夹爪控制服务         |
| `/arm_status`               | std_msgs/String                  | zy_robot/arm_controller      | zy_executor/ctu_orchestrator                            | 机械臂状态更新        |
| `/ctu/command`              | zy_interfaces/msg/CTUCommand     | zy_comm/ctu_communication    | zy_executor/ctu_orchestrator                            | CTU 命令（来自 TCP） |
| `/executor/status`          | zy_interfaces/msg/ExecutorStatus | zy_executor/ctu_orchestrator | zy_comm/ctu_communication                               | 执行器状态反馈给 CTU   |

### ROS2 服务

| 服务                 | 服务器                        | 客户端                                                     | 用途     |
  |--------------------|----------------------------|---------------------------------------------------------|--------|
| `/detect_objects`  | zy_vision/detection_server | zy_vision/grasp_generator, zy_executor/ctu_orchestrator | 物体检测   |
| `/generate_grasp`  | zy_vision/grasp_generator  | zy_executor/ctu_orchestrator                            | 抓取位姿生成 |
| `/gripper_control` | zy_robot/gripper_server    | zy_robot/arm_controller, zy_executor/ctu_orchestrator   | 夹爪位置控制 |

### TF2 变换

| 父坐标系        | 子坐标系                         | 类型 | 发布者                  |
|-------------|------------------------------|----|----------------------|
| `base_link` | `camera_color_optical_frame` | 静态 | static_tfs.launch.py |
| `base_link` | `camera_depth_optical_frame` | 静态 | static_tfs.launch.py |
| `ee_link`   | `tcp_link`                   | 静态 | static_tfs.launch.py |

---

## 数据流

  ```
  CTU 设备 (192.168.127.253:8899)
      ↓ (TCP 二进制协议：SOF + LEN + DATA + CRC16)
  zy_comm/ctu_communication
      ├─→ 心跳线程 (10秒间隔)
      ├─→ 监听线程 (接收命令)
      └─→ /ctu/command (CTUCommand.msg)
          ↓
      zy_executor/ctu_orchestrator
          ↓ (订阅 CTU 命令)
          ├─→ /detect_objects (服务调用)
          │   zy_vision/detection_server ← /camera/color/image_raw
          │                           ← /camera/color/camera_info
          │
          ├─→ /generate_grasp (服务调用)
          │   zy_vision/grasp_generator ← /camera/color/image_raw
          │                           ← /camera/depth/image_raw
          │                           ← /camera/color/camera_info
          │                           ← /camera/depth/camera_info
          │                           ← [调用服务] /detect_objects
          │
          ├─→ /arm_grasp_command (话题发布)
          │   zy_robot/arm_controller
          │       → MoveJ_Cmd()
          │       → Algo_Inverse_Kinematics()
          │       → [发布] /arm_status
          │
          ├─→ /gripper_control (服务调用)
          │   zy_robot/gripper_server
          │       → gripper_position()
          │
          └─→ /executor/status (话题发布)
              zy_comm/ctu_communication
                  ↓ (转换为 TCP 协议)
              CTU 设备 (0x80: GRASP_COUNT, 0x81: GRASP_START, 0x82: GRASP_OVER)
  ```

---

## 对比：合并 vs 拆分

### 场景 1：边缘设备部署

**合并 (zy_vision 包含 camera_node)**：

- 需要：pyrealsense2 + PyTorch + MMDetection + CUDA
- 内存：~2GB RAM + 1GB GPU
- 启动时间：~8s
- 无法部署：对于边缘设备太重

**拆分 (zy_camera + zy_vision)**：

- zy_camera 需要：仅 pyrealsense2
- 内存：~200MB
- 启动时间：~2s
- 部署：可以在边缘设备上部署（Jetson Nano 等）

### 场景 2：开发工作流

**合并**：

- 更改相机参数：需要重建视觉包
- 添加新模型：需要重建视觉包（包括相机）
- GPU 调试：整个包重启

**拆分**：

- 更改相机参数：仅重建 zy_camera
- 添加新模型：仅重建 zy_vision
- GPU 调试：仅重启 zy_vision（相机保持运行）

### 场景 3：硬件升级

**合并**：

- 升级相机：需要重新部署整个视觉栈
- 切换推理设备：重新部署整个视觉栈

**拆分**：

- 升级相机：仅重新部署 zy_camera
- 切换到边缘推理：保留 zy_camera，移动 zy_vision

**胜者**：拆分方法仅增加 1 个包，提供了显著的灵活性。

---

## 实现检查清单

### 阶段 1：创建新包

- [ ] 创建 zy_camera 包结构
- [ ] 创建 zy_comm 包结构
- [ ] 为新包设置 package.xml 和 setup.py

### 阶段 2：相机包开发（仅配置）

- [ ] 创建 zy_camera 包结构（package.xml, CMakeLists.txt, launch/）
- [ ] 安装 realsense2-camera 驱动：
  - [ ] Ubuntu: `sudo apt install ros-${ROS_DISTRO}-realsense2-camera`
  - [ ] Jetson: 从源码编译（librealsense2 + realsense-ros）
- [ ] 创建 launch/camera_bringup.launch.py
  - [ ] 使用 D435i 参数配置 realsense2_camera_node
  - [ ] 设置分辨率：color=640x480@30, depth=640x480@90
  - [ ] 启用 IMU (gyro, accel) 用于 D435i
  - [ ] 启用深度到彩色对齐
  - [ ] 启用空间/时间滤波器
- [ ] 创建 launch/camera_params.yaml（替代内联配置的备选）
- [ ] 更新 zy_camera/package.xml 添加依赖：
  - [ ] realsense2-camera (exec_depend)
  - [ ] ament_cmake (buildtool_depend)
- [ ] 验证 realsense-ros 安装：
  - [ ] `ros2 run realsense2_camera realsense2_camera_node --ros-args --help`
  - [ ] `ros2 pkg xml realsense2_camera | grep realsense`
- [ ] 测试相机包：
  - [ ] `ros2 launch zy_camera camera_bringup.launch.py`
  - [ ] 验证主题：`/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`
  - [ ] 验证 CameraInfo：`ros2 topic echo /camera/camera/color/camera_info`
  - [ ] 验证 TF：`ros2 tf2 ls`

### 阶段 3：视觉节点重构

- [ ] 创建 zy_vision 包结构（package.xml, setup.py, setup.cfg, resource/zy_vision）
- [ ] 创建 detection_server.py
  - [ ] 订阅 `/camera/color/image_raw`
  - [ ] 从 `models/mmdetection/configs/myconfig_zy.py` 加载 MMDetection 模型
  - [ ] 实现 `/detect_objects` 服务 (DetectObjects.srv)
  - [ ] 执行 NMS 过滤（score_threshold=0.8, iou_threshold=0.9）
- [ ] 创建 grasp_generator.py
  - [ ] 订阅 `/camera/color/image_raw`, `/camera/depth/image_raw`
  - [ ] 订阅 `/camera/color/camera_info`, `/camera/depth/camera_info`
  - [ ] 添加嵌入式工具函数：
    - [ ] in_paint() - 深度图空洞填充
    - [ ] letterbox() - 图像缩放填充
    - [ ] scale_coords() - 坐标缩放
  - [ ] 实现 `/generate_grasp` 服务 (GenerateGrasp.srv)
  - [ ] 从 `models/weights/epoch_20_last.pth` 加载 AugmentCNN 模型权重
- [ ] 创建 zy_vision/config/inference_params.yaml
- [ ] 创建 zy_vision/launch/inference_bringup.launch.py
- [ ] 更新 zy_vision/package.xml 添加依赖：
  - [ ] zy_interfaces (msg, srv)
  - [ ] sensor_msgs
  - [ ] cv_bridge
  - [ ] torch (PyTorch)
  - [ ] mmdet (MMDetection)
- [ ] 测试：验证检测和抓取生成服务独立工作

### 阶段 4：通信节点开发

- [ ] 创建 ctu_orchestrator.py（合并 ctu_conn.py + 编排逻辑）
- [ ] 迁移 ctu_protocol.py 到 protocols/
- [ ] 添加服务客户端：/detect_objects, /generate_grasp, /gripper_control
- [ ] 添加话题发布：/arm_grasp_command
- [ ] 添加 CTU TCP 服务器
- [ ] 实现状态管理（grasp_running, grasp_count, inverse_failures）
- [ ] 实现重试逻辑（max_attempts=2, max_inverse_failures=3）
- [ ] 嵌入 GOOGS_MAPPING
- [ ] 创建 comm_params.yaml
- [ ] 创建 comm_bringup.launch.py

### 阶段 5：机械臂控制器增强

- [ ] 添加订阅到 /arm_grasp_command 话题
- [ ] 实现完整抓取序列（mid → grasp → place）
- [ ] 在抓取序列中实现碰撞恢复
- [ ] 添加反馈发布（/arm_status）

### 阶段 6：系统启动文件

- [ ] 创建 static_tfs.launch.py
- [ ] 创建 camera_calib.launch.py
- [ ] 创建 grasp_system.launch.py
- [ ] 更新 zy_robot, zy_vision 启动文件

### 阶段 7：测试

- [ ] 测试 zy_camera 独立运行（图像 + CameraInfo）
- [ ] 测试 zy_vision 独立运行（检测 + 抓取）
- [ ] 测试 TF 发布 (ros2 run tf2_ros static_transform_publisher)
- [ ] 测试完整系统 (grasp_system.launch.py)
- [ ] 验证所有主题/服务连接
- [ ] 验证从 YAML 加载参数

---

## 迁移策略

### 从原始 Python 系统

| 组件                                    | 到包                         | 说明                      |
 |---------------------------------------|----------------------------|-------------------------|
| `camera.py` (RS class)                | zy_camera                  | 添加 CameraInfo 发布        |
| `ctu_conn.py`                         | zy_comm                    | 合并编排逻辑                  |
| `ctu_protocol.py`                     | zy_comm protocols/         | 保留协议逻辑                  |
| `gripper_zhiyuan.py`                  | zy_robot gripper_server    | 添加服务接口                  |
| `RoboticArm.py`                       | zy_robot arm_controller    | 添加 ROS2 话题，添加抓取序列       |
| `grasp_zy_zhiyuan1215.py` (obj_grasp) | zy_robot arm_controller    | 在 arm_controller 实现抓取序列 |
| `grasp_zy_zhiyuan1215.py` (workflow)  | zy_comm/ctu_orchestrator   | 实现编排（count、loop、retry）  |
| `MMDetection`                         | zy_vision detection_server | 服务接口                    |
| `AugmentCNN`                          | zy_vision grasp_generator  | 服务接口                    |
| `utils/utils.py` (子集)                 | zy_vision grasp_generator  | 嵌入节点中                   |

### 从原始 Python 系统（完整迁移）

| 组件                                    | 到包                      | 迁移任务                                                                 |
|---------------------------------------|----------------------------|--------------------------------------------------------------------------|
| `camera.py` (RS类)                | **不迁移** - 使用官方 realsense-ros 驱动 | 不需要自定义相机节点；通过启动参数配置                     |
| 图像预处理（inpaint, 裁剪）   | zy_vision/grasp_generator.py  | 将图像预处理逻辑从 camera.py 迁移到视觉节点                                |
| `ctu_conn.py`                         | zy_comm/ctu_communication.py | 提取 TCP 通信，分离协议逻辑                              |
| `ctu_protocol.py`                     | zy_comm/protocols/ctu_protocol.py | 保留协议解析逻辑完整                                           |
| `gripper_zhiyuan.py`                  | zy_robot/gripper_server.py    | 将 Modbus RTU 封装在 ROS2 服务中 (GripperControl.srv)                        |
| `RoboticArm.py`                       | zy_robot/arm_controller.py    | 将 SDK 封装在 ROS2 节点中，添加话题接口，实现抓取序列                          |
| `grasp_zy_zhiyuan1215.py` (grasp)   | zy_robot/arm_controller.py    | 将抓取序列集成到 arm_controller                                         |
| `grasp_zy_zhiyuan1215.py` (workflow) | zy_executor/ctu_orchestrator.py | 提取状态机、服务客户端、CTU 命令处理                                   |
| `MMDetection`                         | zy_vision/detection_server.py | 封装在 ROS2 服务中，添加 NMS 过滤                                       |
| `AugmentCNN`                          | zy_vision/grasp_generator.py  | 封装在 ROS2 服务中，添加坐标变换逻辑                                       |
| `utils/utils.py` (子集)                 | zy_vision/grasp_generator.py  | 将 in_paint(), letterbox(), scale_coords() 直接嵌入节点中                      |
| `config.py` (Tcam2base, Rbase2cam)    | zy_bringup/launch/static_tfs.launch.py | 转换为 TF2 静态变换                                                  |
| `config.py` (内参)                    | **不需要** - 由 realsense-ros 发布 | CameraInfo 由官方驱动自动发布                                   |
| `config.py` (臂位姿)                   | zy_bringup/config/robot_params.yaml  | 提取预定义位姿（init, mid, place 等）                                  |
| `config.py` (网络配置)                  | zy_bringup/config/comm_params.yaml  | 提取 CTU IP/端口、机器人 IP/端口                                             |

**关键变更**：
- **无自定义相机节点**：使用 Intel 经过实战测试的 `realsense2-camera` 驱动
- **CameraInfo 自动发布**：无需手动 CameraInfo 发布
- **预处理在视觉包中**：图像处理（inpaint, 裁剪）移至 zy_vision
- **更简单的 zy_camera**：纯配置包（launch + YAML）

---

## 配置策略

### 类型 1：运行时参数（YAML）

**位置**：`zy_bringup/config/*.yaml`
**示例**：

- 相机分辨率，发布频率
- 机器人 IP，端口，速度
- 模型路径，设备选择
- 阈值（NMS，置信度）

**访问模式**：

```python
# Node declares and loads
self.declare_parameter('robot_ip', '192.168.127.101')
robot_ip = self.get_parameter('robot_ip').value

# YAML passed in launch file
parameters = ['config/robot_params.yaml']
```

### 类型 2：校准数据（TF2）

**位置**：`zy_bringup/launch/static_tfs.launch.py`
**示例**：

- Tcam2base 变换矩阵
- TCP 补偿偏移

**访问模式**：

```python
# Other nodes query TF2
from tf2_ros import TransformListener

tf_buffer = tf2_ros.Buffer()
transform = tf_buffer.lookup_transform('camera_color_optical_frame', 'base_link', rospy.Time())
```

### 类型 3：嵌入式常量（Python）

**位置**：节点代码（zy_comm）
**示例**：

- 补偿角度（EDGE_ANGLE, SLOPE_ANGLE）
- TCP 偏移（TCP_COMPENSATE）
- 物品映射（GOODS_MAPPING）
- 裁剪区域（CROP_LEFT, CROP_RIGHT 等）

**理由**：简单常量，而非需要类型安全或单独配置文件的复杂校准矩阵。

---

## 依赖关系

### 包依赖

| 包             | zy_interfaces | zy_camera         | zy_vision                  | zy_robot       | zy_comm           |
 |---------------|---------------|-------------------|----------------------------|----------------|-------------------|
| zy_interfaces | -             | msg               | msg                        | msg            | msg               |
| zy_camera     | -             | -                 | -                          | -              | Image, CameraInfo |
| zy_vision     | -             | Image, CameraInfo | -                          | -              | -                 |
| zy_robot      | -             | -                 | -                          | -              | -                 |
| zy_comm       | -             | -                 | DetectionResult, GraspPose | GripperControl | -                 |

### 外部依赖

| 依赖                    | 使用方                  | 用途                |
 |-----------------------|----------------------|-------------------|
| `pyrealsense2`        | zy_camera            | RealSense D435 驱动 |
| `sensor_msgs`         | 所有                   | 标准 ROS2 消息        |
| `geometry_msgs`       | zy_robot             | 位姿/点消息            |
| `std_msgs`            | zy_comm, zy_robot    | 基本数据类型            |
| `cv_bridge`           | zy_camera, zy_vision | OpenCV-ROS2 桥接    |
| `tf2_ros`             | zy_bringup           | TF2 静态变换发布器       |
| `PyTorch`             | zy_vision            | 深度学习框架            |
| `MMDetection`         | zy_vision            | 物体检测              |
| `robotic_arm_package` | zy_robot             | RM65 SDK          |

---

## 优势总结

1. **模块化**：每个包具有单一、明确的职责
2. **标准合规**：遵循 ROS2 约定（TF2, CameraInfo 等）
3. **可部署**：可以在不同硬件上运行子集
4. **可维护**：更改隔离到相关包
5. **可测试**：每个包可以独立测试
6. **可扩展**：易于添加新功能（传感器、算法、机器人）
7. **无循环依赖**：无包间 Python 模块依赖

---

## 风险和缓解措施

### 风险 1：包数量增加

**风险**：6 个包（由于合并，从初始 7 个减少）
**缓解**：清晰的组织和启动文件简化启动；包数量比原始设计更少

### 风险 2：启动文件复杂性

**风险**：grasp_system.launch.py 中多个包含文件
**缓解**：文档齐全的结构，包特定的子启动文件，降低复杂度（无 executor 启动文件）

### 风险 3：配置分散

**风险**：参数分散在 4 个 YAML 文件中
**缓解**：集中在 `zy_bringup/config/` 中，清晰的命名约定（从 5 个文件减少）

### 风险 4：数据流复杂性

**风险**：包间通信有多个主题/服务
**缓解**：清晰的文档，使用 `rqt_graph` 可视化，简化数据流（无中间 executor 层）

---

## 详细实现说明

### 包目录模板

#### zy_camera
```
zy_camera/
├── package.xml                    # 必需：realsense2-camera (exec_depend)
├── CMakeLists.txt                # ament_cmake（无 Python 代码）
└── launch/
    ├── camera_bringup.launch.py    # 启动 realsense2_camera_node
    └── camera_params.yaml         # 备选 YAML 配置
```

**关键实现**（launch/camera_bringup.launch.py）：
```python
# 启动 Intel 官方 realsense-ros 驱动
# 不需要自定义 camera_node.py！

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense2_camera_node',
            parameters=[{
                'camera_name': 'camera',
                'camera_namespace': 'camera',
                'enable_color': True,
                'enable_depth': True,
                'enable_sync': True,
                'enable_gyro': True,  # D435i
                'enable_accel': True,  # D435i
                'rgb_camera.color_profile': '640x480x30',
                'depth_module.depth_profile': '640x480x90',  # 高帧率
                'align_depth.enable': True,
                'spatial_filter.enable': True,
                'temporal_filter.enable': True,
            }],
            output='screen'
        )
    ])
```

**不需要 Python 节点代码** - 仅配置！

#### zy_vision
```
zy_vision/
├── package.xml                    # 必需：zy_interfaces, sensor_msgs, cv_bridge, torch, mmdet
├── setup.py                      # 入口点：detection_server, grasp_generator
├── setup.cfg
├── resource/
│   └── zy_vision
├── zy_vision/
│   ├── __init__.py
│   ├── detection_server.py
│   └── grasp_generator.py
└── launch/
    └── inference_bringup.launch.py
```

**关键实现**：
```python
# grasp_generator.py - 使用 CameraInfo 进行坐标变换
from sensor_msgs.msg import CameraInfo

def grasp_img2real(self, row, col, depth, camera_info):
    # 使用 CameraInfo 的内参矩阵（由 realsense-ros 发布）
    K = np.array(camera_info.k).reshape(3,3)

    # 像素到相机坐标系
    x = (col - camera_info.k[2]) * depth / camera_info.k[0]
    y = (row - camera_info.k[5]) * depth / camera_info.k[4]
    z = depth

    # 使用 TF2 变换到基座坐标系
    point_camera = PointStamped()
    point_camera.header.frame_id = "camera_color_optical_frame"
    point_camera.point.x = x
    point_camera.point.y = y
    point_camera.point.z = z

    point_base = self.tf_buffer.transform(
        point_camera,
        "base_link",
        timeout=Duration(seconds=0.1)
    )

    return [point_base.point.x, point_base.point.y, point_base.point.z]

# 图像预处理（从原始系统迁移）
def preprocess_images(self, color_msg, depth_msg):
    # 订阅 realsense-ros 主题
    # /camera/camera/color/image_raw
    # /camera/camera/aligned_depth_to_color/image_raw

    # 将 ROS 消息转换为 OpenCV
    cv_bridge = CvBridge()
    color_image = cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
    depth_image = cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    # 应用图像预处理（来自原始 camera.py）
    # 1. 裁剪深度图像（去除左右边缘）
    depth_image = depth_image[:, 80:560, :]  # 裁剪到 480x480

    # 2. Inpaint 深度空洞（填充零值）
    mask = (depth_image == 0).astype(np.uint8)
    depth_image = cv2.inpaint(depth_image, mask, 3, cv2.INPAINT_NS)

    # 3. 转换深度为米（从 mm）
    depth_image = depth_image.astype(np.float32) / 1000.0

    return color_image, depth_image
```

**关键实现**：
```python
# grasp_generator.py - 使用 CameraInfo 进行坐标变换
from sensor_msgs.msg import CameraInfo

def grasp_img2real(self, row, col, depth, camera_info):
    # 使用 CameraInfo 的内参矩阵
    K = np.array(camera_info.k).reshape(3, 3)

    # 像素到相机坐标系
    x = (col - camera_info.k[2]) * depth / camera_info.k[0]
    y = (row - camera_info.k[5]) * depth / camera_info.k[4]
    z = depth

    # 使用 TF2 变换到基座坐标系
    point_camera = PointStamped()
    point_camera.header.frame_id = "camera_color_optical_frame"
    point_camera.point.x = x
    point_camera.point.y = y
    point_camera.point.z = z

    point_base = self.tf_buffer.transform(
        point_camera,
        "base_link",
        timeout=Duration(seconds=0.1)
    )

    return [point_base.point.x, point_base.point.y, point_base.point.z]
```

#### zy_robot
```
zy_robot/
├── package.xml                    # 必需：sensor_msgs, std_msgs, geometry_msgs
├── setup.py                      # 入口点：arm_controller, gripper_server
├── setup.cfg
├── resource/
│   └── zy_robot
├── zy_robot/
│   ├── __init__.py
│   ├── arm_controller.py
│   └── gripper_server.py
└── launch/
    └── robot_bringup.launch.py
```

**关键实现**：
```python
# arm_controller.py - 抓取序列
from zy_interfaces.msg import GraspPose
from std_msgs.msg import String

def execute_grasp_sequence(self, grasp_pose: GraspPose):
    # 1. 移动到中间位姿
    self.move_to_pose(self.mid_pose)
    self.wait_for_completion()

    # 2. 移动到接近位姿（抓取上方）
    approach_pose = grasp_pose.copy()
    approach_pose.position.z += 0.05  # 抓取上方 5cm
    self.move_to_pose(approach_pose)

    # 3. 移动到抓取位姿
    self.move_to_pose(grasp_pose)

    # 4. 关闭夹爪
    self.gripper_client.call_async(GripperControl.Request(position=0))

    # 5. 提升到放置位姿
    self.move_to_pose(self.place_mid_pose)
    self.move_to_pose(self.place_last_pose)

    # 6. 打开夹爪
    self.gripper_client.call_async(GripperControl.Request(position=1))
```

#### zy_comm
```
zy_comm/
├── package.xml                    # 必需：std_msgs, zy_interfaces
├── setup.py                      # 入口点：ctu_communication
├── setup.cfg
├── resource/
│   └── zy_comm
├── zy_comm/
│   ├── __init__.py
│   ├── protocols/
│   │   └── ctu_protocol.py
│   └── ctu_communication.py
└── launch/
    └── comm_bringup.launch.py
```

**关键实现**：
```python
# ctu_communication.py - TCP 到 ROS2 桥接
import socket
from zy_interfaces.msg import CTUCommand

class CTUCommunication(Node):
    def __init__(self):
        super().__init__('ctu_communication')
        self.ctu_pub = self.create_publisher(CTUCommand, '/ctu/command', 10)
        self.executor_status_sub = self.create_subscription(
            ExecutorStatus,
            '/executor/status',
            self.executor_status_callback,
            10
        )

        # TCP 连接
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((ctu_ip, ctu_port))

        # 线程
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop)
        self.listen_thread = threading.Thread(target=self.listen_loop)
        self.heartbeat_thread.start()
        self.listen_thread.start()

    def listen_loop(self):
        while rclpy.ok():
            # 接收 TCP 数据
            data = self.socket.recv(1024)
            if not data:
                break

            # 解析二进制协议（SOF + LEN + DATA + CRC16）
            command = self.parse_ctu_protocol(data)

            # 发布到 ROS2
            self.ctu_pub.publish(command)
```

#### zy_executor
```
zy_executor/
├── package.xml                    # 必需：std_msgs, zy_interfaces
├── setup.py                      # 入口点：ctu_orchestrator
├── setup.cfg
├── resource/
│   └── zy_executor
├── zy_executor/
│   ├── __init__.py
│   └── ctu_orchestrator.py
└── launch/
    └── executor_bringup.launch.py
```

**关键实现**：
```python
# ctu_orchestrator.py - 状态机
from enum import Enum
from zy_interfaces.msg import CTUCommand, ExecutorStatus

class State(Enum):
    IDLE = 0
    DETECTING = 1
    PLANNING = 2
    EXECUTING = 3

class CTUOrchestrator(Node):
    def __init__(self):
        super().__init__('ctu_orchestrator')
        self.state = State.IDLE
        self.grasp_running = False
        self.grasp_count = 0
        self.inverse_failures = 0

        # 服务客户端
        self.detect_client = self.create_client(DetectObjects, '/detect_objects')
        self.grasp_client = self.create_client(GenerateGrasp, '/generate_grasp')
        self.gripper_client = self.create_client(GripperControl, '/gripper_control')

        # 话题发布者/订阅者
        self.arm_cmd_pub = self.create_publisher(GraspCommand, '/arm_grasp_command', 10)
        self.executor_status_pub = self.create_publisher(ExecutorStatus, '/executor/status', 10)
        self.ctu_cmd_sub = self.create_subscription(CTUCommand, '/ctu/command', self.ctu_callback, 10)
        self.arm_status_sub = self.create_subscription(String, '/arm_status', self.arm_status_callback, 10)

    def ctu_callback(self, msg: CTUCommand):
        if msg.command == CTUCommand.CMD_START_SORTING:
            self.run_grasp_sequence()

    def run_grasp_sequence(self):
        if self.grasp_running:
            return

        self.grasp_running = True
        self.state = State.DETECTING

        # 调用检测服务
        future = self.detect_client.call_async(DetectObjects.Request())

        # 等待结果
        rclpy.spin_until_future_complete(self, future)
        detection_result = future.result()

        if detection_result.success:
            self.state = State.PLANNING
            # 调用抓取生成
            future = self.grasp_client.call_async(GenerateGrasp.Request(
                label=detection_result.label,
                bbox=detection_result.bbox
            ))
            # ... 继续抓取执行
```

#### zy_interfaces
```
zy_interfaces/
├── package.xml                    # 必需：rosidl_default_generators, std_msgs
├── CMakeLists.txt
└── zy_interfaces/
    ├── msg/
    │   ├── DetectionResult.msg
    │   ├── GraspPose.msg
    │   ├── GraspCommand.msg
    │   ├── CTUCommand.msg
    │   └── ExecutorStatus.msg
    ├── srv/
    │   ├── DetectObjects.srv
    │   ├── GenerateGrasp.srv
    │   └── GripperControl.srv
    └── action/
        └── ExecuteGrasp.action
```

**消息定义**：
```yaml
# msg/CTUCommand.msg
uint8 CMD_START_SORTING=0x70
uint8 CMD_ADJUST_SPEED=0x71
uint8 CMD_START_GRASP=0x81
uint8 CMD_GRASP_COMPLETE=0x82

uint8 command
string[] parameters  # 基于命令的可变参数
```

```yaml
# msg/ExecutorStatus.msg
uint8 STATUS_IDLE=0
uint8 STATUS_DETECTING=1
uint8 STATUS_PLANNING=2
uint8 STATUS_EXECUTING=3
uint8 STATUS_ERROR=4

uint8 status
int32 grasp_count
int32 inverse_failures
string error_message
```

#### zy_bringup
```
zy_bringup/
├── package.xml                    # 必需：ament_cmake
├── CMakeLists.txt
├── launch/
│   ├── grasp_system.launch.py
│   ├── static_tfs.launch.py
│   └── camera_calib.launch.py
└── config/
    ├── camera_params.yaml
    ├── robot_params.yaml
    ├── comm_params.yaml
    └── inference_params.yaml
```

### 构建和测试验证

**构建命令**：
```bash
cd Ros2
colcon build --symlink-install --packages-select zy_interfaces
colcon build --symlink-install --packages-select zy_camera
colcon build --symlink-install --packages-select zy_vision
colcon build --symlink-install --packages-select zy_robot
colcon build --symlink-install --packages-select zy_comm
colcon build --symlink-install --packages-select zy_executor
colcon build --symlink-install --packages-select zy_bringup
```

**验证步骤**：
1. **检查包构建成功**：
   ```bash
   ls install/zy_camera/lib/zy_camera/camera_node
   ls install/zy_vision/lib/zy_vision/detection_server
   ```

2. **验证话题/服务列表**：
   ```bash
   ros2 topic list  # 应显示设计中的所有话题
   ros2 service list  # 应显示设计中的所有服务
   ```

3. **测试相机节点**：
   ```bash
   ros2 launch zy_camera camera_bringup.launch.py
   # 在另一个终端：
   ros2 topic echo /camera/color/image_raw
   ros2 topic echo /camera/color/camera_info
   ```

4. **测试检测服务**：
   ```bash
   ros2 launch zy_vision inference_bringup.launch.py
   # 在另一个终端：
   ros2 service call /detect_objects zy_interfaces/srv/DetectObjects "{image: ...}"
   ```

5. **测试完整系统**：
   ```bash
   ros2 launch zy_bringup grasp_system.launch.py
   # 验证所有节点运行中：
   ros2 node list
   ```

### 关键依赖检查

**开始实现前**：
```bash
# Python 包
python3 -c "import pyrealsense2; print('pyrealsense2: OK')"
python3 -c "import torch; print('torch:', torch.__version__)"
python3 -c "import mmdet; print('mmdet:', mmdet.__version__)"
python3 -c "import cv2; print('opencv:', cv2.__version__)"

# ROS2 包
ros2 pkg list | grep zy_  # 构建后应列出所有 zy_* 包
ros2 interface show sensor_msgs/msg/Image  # 应显示消息定义
```

**缺失依赖（如果有）**：
```bash
# 在 Ubuntu/Jetson 上安装
sudo apt install ros-${ROS_DISTRO}-sensor-msgs
sudo apt install ros-${ROS_DISTRO}-cv-bridge
sudo apt install ros-${ROS_DISTRO}-tf2-ros
sudo apt install ros-${ROS_DISTRO}-geometry-msgs
```

---

## 未来增强

1. **添加参数描述符**：为参数添加类型和范围验证
2. **生命周期节点**：转换为托管节点以获得更好的状态管理
3. **诊断**：添加诊断发布器以进行系统健康监控
4. **启动文件参数**：为常用参数添加命令行覆盖
5. **多相机支持**：结构化以轻松添加深度/红外相机
6. **模块化推理**：为不同的检测模型添加单独的包（YOLO 等）

---

## 结论

此架构提供：

- **清晰的关注点分离**：每个包具有单一、明确的用途
- **ROS2 标准合规**：内参使用 CameraInfo，外参使用 TF2
- **部署灵活性**：可以在边缘设备上仅运行相机
- **可维护性**：具有清晰边界的独立包
- **可扩展性**：易于添加新功能而不影响现有代码

**总包数**：7
**总启动文件**：1 个主文件 + 6 个包特定文件
**总配置文件**：zy_bringup/config/ 中的 4 个文件

此架构在通信和编排之间提供了清晰的关注点分离，使得：

- **独立部署** CTU 通信和任务编排
- **松耦合** 通过 ROS2 主题/服务而非直接函数调用
- **清晰的测试边界** 用于通信和编排逻辑
- **可扩展性** 以支持除 CTU 以外的多个任务源
- **灵活性** 独立升级通信协议或编排算法

独立的编排包遵循 ROS2 最佳实践，并匹配工业机器人架构，其中通信、编排和控制是不同的、可独立测试的组件。
