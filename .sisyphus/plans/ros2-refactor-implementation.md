# ROS2 重构实施计划

## 项目概述

**目标**: 将单体式 Python 视觉抓取系统重构为模块化 ROS2 架构
**当前状态**: 原始 Python 系统（无 ROS2）
**目标平台**: Jetson Orin NX, Ubuntu 20.04
**ROS2 版本**: Humble (推荐) 或 Foxy

---

## 设计要点总结

### 1. 包架构（7个包）

```
Ros2/
├── zy_interfaces/          # 接口定义（msg/srv/action）
├── zy_camera/              # 相机驱动（使用 realsense-ros）
├── zy_vision/              # 视觉推理（detection_node + grasp_node）
├── zy_robot/               # 机器人控制（arm_controller_node + gripper_node）
├── zy_comm/                # CTU 通信（comm_node + protocols）
├── zy_executor/            # 任务编排（executor_node）
└── zy_bringup/             # 系统启动器（launch + 配置）
```

### 2. 关键设计决策

| 决策 | 理由 | 影响 |
|------|------|------|
| **相机/视觉分离** | 依赖隔离，灵活部署 | zy_camera 仅依赖 pyrealsense2 |
| **无校准包** | CameraInfo 由 realsense-ros 发布，TF2 处理外参 | 减少包数量 |
| **工具函数内联** | 仅 3 个函数被使用，避免复杂依赖 | in_paint, letterbox, scale_coords 嵌入节点 |
| **编排/通信分离** | 单一职责，独立部署，松耦合 | zy_comm (TCP) ↔ zy_executor (ROS2 topics) |
| **YAML 集中配置** | 统一管理，易于修改 | 所有运行时参数在 zy_bringup/config/ |

### 3. 迁移映射表

| 原系统组件 | 目标包 | 迁移方式 |
|-----------|--------|---------|
| `camera.py` (RS类) | zy_camera | **不迁移** - 使用 Intel 官方 realsense-ros 驱动 |
| 图像预处理 (inpaint, 裁剪) | zy_vision/grasp_generator.py | 迁移预处理逻辑 |
| `ctu_conn.py` | zy_comm/comm_node.py | 提取 TCP 通信 |
| `ctu_protocol.py` | zy_comm/protocols/ctu_protocol.py | 保持协议解析逻辑 |
| `gripper_zhiyuan.py` | zy_robot/gripper_node.py | 封装为 ROS2 服务 |
| `RoboticArm.py` | zy_robot/arm_controller_node.py | 封装为 ROS2 节点 |
| `grasp_zy_zhiyuan1215.py` (workflow) | zy_executor/executor_node.py | 提取状态机和编排逻辑 |
| `grasp_zy_zhiyuan1215.py` (grasp sequence) | zy_robot/arm_controller_node.py | 集成抓取序列 |
| MMDetection | zy_vision/detection_node.py | 封装为 ROS2 服务 |
| AugmentCNN | zy_vision/grasp_node.py | 封装为 ROS2 服务 |
| `utils/utils.py` (子集) | zy_vision/grasp_node.py | 嵌入 3 个函数 |
| `config.py` (Tcam2base) | zy_bringup/launch/static_tfs.launch.py | 转换为 TF2 |
| `config.py` (arm poses) | zy_bringup/config/robot_params.yaml | 提取预定义位姿 |
| `config.py` (network) | zy_bringup/config/comm_params.yaml | 提取 IP/端口 |

### 4. 关键配置参数

| 参数 | 来源 | 目标位置 |
|------|------|---------|
| **Tcam2base 变换矩阵** | config.py | zy_bringup/launch/static_tfs.launch.py (TF2) |
| **机械臂预定义位姿** | grasp_zy_zhiyuan1215.py | zy_bringup/config/robot_params.yaml |
| **TCP 补偿** | config.py (0.018m) | zy_vision/grasp_generator.py (内联常量) |
| **边缘倾斜角度** | config.py (π/7) | zy_vision/grasp_generator.py (内联常量) |
| **物体映射** | ctu_conn.py | zy_executor/executor_node.py (内联字典) |
| **相机分辨率** | camera.py | zy_bringup/config/camera_params.yaml |
| **机械臂速度** | config.py (20) | zy_bringup/config/robot_params.yaml |
| **CTU IP/端口** | ctu_conn.py | zy_bringup/config/comm_params.yaml |
| **模型路径** | grasp_zy_zhiyuan1215.py | zy_bringup/config/inference_params.yaml |

---

## 实施计划

### 阶段 0: 环境准备

**目标**: 搭建 ROS2 开发环境和依赖

**任务**:

1. **安装 ROS2**
   ```bash
   # Ubuntu 20.04 - ROS2 Humble
   sudo apt install software-properties-common
   sudo add-apt-repository universe
   sudo apt update && sudo apt install curl -y
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   sudo apt update
   sudo apt install ros-humble-desktop
   ```

2. **创建工作空间**
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   source /opt/ros/humble/setup.bash
   ```

3. **安装 realsense-ros**
   ```bash
   # 方法 1: apt 安装 (简单，但可能版本旧)
   sudo apt install ros-humble-realsense2-camera

   # 方法 2: 源码编译 (推荐 Jetson)
   git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2
   cd realsense-ros
   sudo apt install libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
   sudo apt install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
   colcon build
   source install/setup.bash
   ```

4. **准备 Python 依赖**
   ```bash
   # 在 conda 环境中安装
   conda activate zy_torch
   pip install rclpy sensor-msgs cv-bridge
   ```

**验收标准**:
- [ ] `ros2 --version` 输出正确版本
- [ ] `colcon build --help` 可用
- [ ] `colcon build --symlink` 编译无错误（警告可以忽略）
- [ ] Python 可以 import rclpy, sensor_msgs

---

### 阶段 1: 创建包结构

**目标**: 创建 7 个 ROS2 包的基础结构

**任务**:

1. **创建工作空间目录**
   ```bash
   cd ~/ros2_ws/src
   mkdir -p zy_interfaces zy_camera zy_vision zy_robot zy_comm zy_executor zy_bringup
   ```

2. **创建 zy_interfaces (消息/服务定义)**
   - `package.xml`
   - `CMakeLists.txt`
   - `msg/`:
     - `DetectionResult.msg`
     - `GraspPose.msg`
     - `CTUCommand.msg`
     - `ExecutorStatus.msg`
   - `srv/`:
     - `DetectObjects.srv`
     - `GenerateGrasp.srv`
     - `GripperControl.srv`

3. **创建 zy_camera (纯配置包)**
   - `package.xml`
   - `CMakeLists.txt`
   - `launch/camera_bringup.launch.py`

4. **创建 zy_vision (Python 包)**
   - `package.xml`
   - `setup.py`
   - `setup.cfg`
   - `zy_vision/__init__.py`
   - `zy_vision/detection_node.py`
   - `zy_vision/grasp_node.py`
   - `launch/inference_bringup.launch.py`

5. **创建 zy_robot (Python 包)**
   - `package.xml`
   - `setup.py`
   - `setup.cfg`
   - `zy_robot/__init__.py`
   - `zy_robot/arm_controller_node.py`
   - `zy_robot/gripper_node.py`
   - `launch/robot_bringup.launch.py`

6. **创建 zy_comm (Python 包)**
   - `package.xml`
   - `setup.py`
   - `setup.cfg`
   - `zy_comm/__init__.py`
   - `zy_comm/protocols/ctu_protocol.py`
   - `zy_comm/comm_node.py`
   - `launch/comm_bringup.launch.py`

7. **创建 zy_executor (Python 包)**
   - `package.xml`
   - `setup.py`
   - `setup.cfg`
   - `zy_executor/__init__.py`
   - `zy_executor/executor_node.py`
   - `launch/executor_bringup.launch.py`

8. **创建 zy_bringup (配置包)**
   - `package.xml`
   - `CMakeLists.txt`
   - `launch/`:
     - `static_tfs.launch.py`
     - `grasp_system.launch.py`
   - `config/`:
     - `camera_params.yaml`
     - `robot_params.yaml`
     - `comm_params.yaml`
     - `inference_params.yaml`

**验收标准**:
- [ ] 所有包的 `package.xml` 和 `setup.py`/`CMakeLists.txt` 存在
- [ ] `colcon build --symlink` 编译无错误（警告可以忽略）
- [ ] `ros2 pkg list` 显示所有 7 个包
- [ ] 每个包的 entry_points 可执行（例如 `ros2 run zy_vision detection_node` 可用）

---

### 阶段 2: 接口定义 (zy_interfaces)

**目标**: 定义 ROS2 消息、服务和 Action

**任务**:

1. **定义消息**

   **DetectionResult.msg**:
   ```
   std_msgs/Header header
   int32 label              # 物体类别 ID (1-5)
   float32 score            # 置信度
   int32[4] bbox           # [x1, y1, x2, y2]
   string class_name        # 物体类别名称
   ```

   **GraspPose.msg**:
   ```
   std_msgs/Header header
   float64[3] position     # [x, y, z] in base_link
   float64[4] orientation  # [qx, qy, qz, qw]
   float32 grasp_width     # 夹爪开合宽度
   float64 approach_dist   # 接近距离
   ```

   **CTUCommand.msg**:
   ```
   std_msgs/Header header
   uint8 cmd_id            # 命令字
   uint8[7] data           # 数据段（最多 7 字节）
   ```

   **ExecutorStatus.msg**:
   ```
   std_msgs/Header header
   uint8 state             # IDLE=0, DETECTING=1, PLANNING=2, EXECUTING=3
   uint32 grasp_count      # 已抓取次数
   uint8 error_code        # 错误码
   string message          # 状态消息
   ```

2. **定义服务**

   **DetectObjects.srv**:
   ```
   ---
   bool success
   DetectionResult[] results
   string message
   ```

   **GenerateGrasp.srv**:
   ```
   int32 label
   int32[4] bbox
   bool visualize
   ---
   bool success
   GraspPose[] grasps
   string message
   ```

   **GripperControl.srv**:
   ```
   float32 position        # 0.0 (close) - 1.0 (open)
   float32 velocity        # 可选速度
   ---
   bool success
   float32 current_position
   string message
   ```

3. **编译接口包**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select zy_interfaces
   source install/setup.bash
   ```

**验收标准**:
- [ ] `ros2 interface show zy_interfaces/msg/DetectionResult` 正常显示
- [ ] `ros2 interface show zy_interfaces/srv/DetectObjects` 正常显示
- [ ] 接口包编译成功，生成 Python 和 C++ 支持文件

---

### 阶段 3: 相机包开发 (zy_camera)

**目标**: 配置和启动 realsense-ros 驱动

**关键决策**: **不写自定义 camera_node**，使用 Intel 官方驱动

**任务**:

1. **创建 camera_bringup.launch.py**

   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory
   import os

   def generate_launch_description():
       pkg_share = get_package_share_directory('zy_camera')
       config_file = os.path.join(pkg_share, 'config', 'camera_params.yaml')

       return LaunchDescription([
           Node(
               package='realsense2_camera',
               executable='realsense2_camera_node',
               name='realsense2_camera_node',
               parameters=[config_file],
               output='screen'
           )
       ])
   ```

2. **创建 camera_params.yaml**

   ```yaml
   camera_name: camera
   camera_namespace: camera

   # Enable streams
   enable_color: true
   enable_depth: true
   enable_sync: true
   enable_gyro: true
   enable_accel: true
   unite_imu_method: 2

   # Resolution and FPS (D435i optimized)
   rgb_camera:
     color_profile: 640x480x30
   depth_module:
     depth_profile: 640x480x90

   # Post-processing
   align_depth:
     enable: true
   spatial_filter:
       enable: true
     temporal_filter:
       enable: true

   # Coordinate frames
   base_frame_id: camera_link
   publish_tf: true
   ```

3. **创建 package.xml 和 CMakeLists.txt**

   **package.xml**:
   ```xml
   <?xml version="1.0"?>
   <package format="3">
     <name>zy_camera</name>
     <version>1.0.0</version>
     <description>RealSense D435i camera wrapper (uses realsense-ros)</description>
     <maintainer email="dev@example.com">Developer</maintainer>
     <license>MIT</license>

     <buildtool_depend>ament_cmake</buildtool_depend>

     <exec_depend>realsense2_camera</exec_depend>

     <export>
       <build_type>ament_cmake</build_type>
     </export>
   </package>
   ```

   **CMakeLists.txt**:
   ```cmake
   cmake_minimum_required(VERSION 3.8)
   project(zy_camera)

   if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     add_compile_options(-Wall -Wextra -Wpedantic)
   endif()

   find_package(ament_cmake REQUIRED)

   # Install launch files
   install(DIRECTORY
     launch
     config
     DESTINATION share/${PROJECT_NAME}
   )

   ament_package()
   ```

4. **编译和测试**
   ```bash
   colcon build --packages-select zy_camera
   source install/setup.bash
   ros2 launch zy_camera camera_bringup.launch.py
   ```

5. **验证**
   ```bash
   # 检查主题
   ros2 topic list | grep camera

   # 检查 CameraInfo
   ros2 topic echo /camera/camera/color/camera_info --once

   # 检查 TF
   ros2 tf2 ls
   ```

**验收标准**:
- [ ] `ros2 launch zy_camera camera_bringup.launch.py` 启动成功
- [ ] `ros2 node list` 显示 `realsense2_camera_node`
- [ ] `ros2 topic list` 显示相机相关主题（image_raw, camera_info）

---

### 阶段 4: 视觉包开发 (zy_vision)

**目标**: 实现物体检测和抓取点生成服务

**关键迁移**:
- MMDetection → detection_node.py
- AugmentCNN → grasp_node.py
- 图像预处理 → grasp_node.py
- utils/utils.py (in_paint, letterbox, scale_coords) → 内联

**任务**:

1. **创建 detection_node.py**

   **功能**:
   - 订阅 `/camera/camera/color/image_raw`
   - 加载 MMDetection 模型 (`models/mmdetection/configs/myconfig_zy.py`, `models/weights/epoch_20_last.pth`)
   - 提供 `/detect_objects` 服务
   - NMS 过滤

   **关键代码结构**:
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image
   from zy_interfaces.srv import DetectObjects
   from cv_bridge import CvBridge
   from mmdet.apis import init_detector, inference_detector

   class DetectionNode(Node):
       def __init__(self):
           super().__init__('detection_node')
           self.cv_bridge = CvBridge()

           # Load model
           config_file = 'models/mmdetection/configs/myconfig_zy.py'
           checkpoint = 'models/weights/epoch_20_last.pth'
           self.model = init_detector(config_file, checkpoint, device='cuda')

           # Create service
           self.srv = self.create_service(DetectObjects, '/detect_objects', self.detect_callback)

       def detect_callback(self, request, response):
           # Get latest image
           msg = self.image_subscriber.get_latest_message()
           if msg is None:
               response.success = False
               return response

           # Convert to numpy
           img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

           # Inference
           result = inference_detector(self.model, img)

           # Convert to ROS response
           # ...

           return response

   def main(args=None):
       rclpy.init(args=args)
       node = DetectionNode()
       rclpy.spin(node)
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **创建 grasp_node.py**

   **功能**:
   - 订阅 `/camera/camera/color/image_raw`, `/camera/depth/image_raw`
   - 订阅 `/camera/camera/color/camera_info`, `/camera/depth/camera_info`
   - 调用 `/detect_objects` 服务
   - 加载 AugmentCNN 模型 (`doc/single_new.txt`, `models/test_250927_1644__zoneyung_/epoch_84_accuracy_1.00`)
   - 提供 `/generate_grasp` 服务
   - 坐标变换: 像素 → 相机坐标系 → 基座坐标系 (使用 TF2)
   - 边缘补偿 (π/7), 深度补偿 (-0.18m)

   **嵌入工具函数**:
   ```python
   # From utils/utils.py - in_paint
   def in_paint(self, depth_image, mask, inpaint_radius=3):
       return cv2.inpaint(depth_image, mask, inpaint_radius, cv2.INPAINT_NS)

   # From utils/utils.py - letterbox
   def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
       # Resize with padding
       pass

   # From utils/utils.py - scale_coords
   def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
       # Scale bounding boxes
       pass
   ```

   **图像预处理** (从 camera.py 迁移):
   ```python
   def preprocess_images(self, color_msg, depth_msg):
       # Convert ROS messages to OpenCV
       color_img = self.cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
       depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

       # Crop depth image (remove left/right edges)
       depth_img = depth_img[:, 80:560, :]  # 480x480

       # Inpaint depth holes
       mask = (depth_img == 0).astype(np.uint8)
       depth_img = self.in_paint(depth_img, mask, 3)

       # Convert depth to meters
       depth_img = depth_img.astype(np.float32) / 1000.0

       return color_img, depth_img
   ```

   **坐标变换** (使用 TF2):
   ```python
   from tf2_ros import TransformException

   def grasp_img2real(self, row, col, depth, camera_info):
       # Pixel to camera frame using intrinsic parameters
       K = np.array(camera_info.k).reshape(3, 3)
       x = (col - camera_info.k[2]) * depth / camera_info.k[0]
       y = (row - camera_info.k[5]) * depth / camera_info.k[4]
       z = depth

       # Transform to base frame using TF2
       point_camera = PointStamped()
       point_camera.header.frame_id = "camera_color_optical_frame"
       point_camera.point.x = x
       point_camera.point.y = y
       point_camera.point.z = z

       try:
           point_base = self.tf_buffer.transform(
               point_camera,
               "base_link",
               timeout=Duration(seconds=0.1)
           )
           return [point_base.point.x, point_base.point.y, point_base.point.z]
       except TransformException as ex:
           self.get_logger().error(f'Transform failed: {ex}')
           return None
   ```

3. **创建 inference_params.yaml**

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

4. **创建 package.xml, setup.py, setup.cfg**

   **package.xml**:
   ```xml
   <?xml version="1.0"?>
   <package format="3">
     <name>zy_vision</name>
     <version>1.0.0</version>
     <description>Vision inference (detection + grasp generation)</description>

     <buildtool_depend>ament_python</buildtool_depend>

     <depend>zy_interfaces</depend>
     <depend>sensor_msgs</depend>
     <depend>cv_bridge</depend>

     <exec_depend>python3-numpy</exec_depend>
     <exec_depend>python3-opencv</exec_depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

   **setup.py**:
   ```python
   from setuptools import setup

   package_name = 'zy_vision'

   setup(
       name=package_name,
       version='1.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           ('share/' + package_name + '/launch', ['launch/inference_bringup.launch.py']),
           ('share/' + package_name + '/config', ['config/inference_params.yaml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Developer',
       maintainer_email='dev@example.com',
       description='Vision inference (detection + grasp generation)',
       license='MIT',
       tests_require=['pytest'],
        entry_points={
            'console_scripts': [
                'detection_node = zy_vision.detection_node:main',
                'grasp_node = zy_vision.grasp_node:main',
            ],
        },
   )
   ```

5. **创建 inference_bringup.launch.py**

   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory
   import os

   def generate_launch_description():
       pkg_share = get_package_share_directory('zy_vision')
       config_file = os.path.join(pkg_share, 'config', 'inference_params.yaml')

        return LaunchDescription([
            Node(
                package='zy_vision',
                executable='detection_node',
                name='detection_node',
                parameters=[config_file],
                output='screen'
            ),
            Node(
                package='zy_vision',
                executable='grasp_node',
                name='grasp_node',
                parameters=[config_file],
                output='screen'
            )
        ])
   ```

6. **编译和测试**
    ```bash
    colcon build --packages-select zy_vision
    source install/setup.bash
    ros2 launch zy_vision inference_bringup.launch.py
    ```

7. **验证**
    ```bash
    # 检查服务
    ros2 service list | grep detection
    ros2 service list | grep grasp

    # 检查节点
    ros2 node list | grep detection
    ros2 node list | grep grasp
    ```

**验收标准**:
- [ ] `ros2 launch zy_vision inference_bringup.launch.py` 启动成功
- [ ] `ros2 node list` 显示 `detection_node` 和 `grasp_node`
- [ ] `ros2 service list` 显示 `/detect_objects` 和 `/generate_grasp` 服务
- [ ] 节点能独立运行（`ros2 run zy_vision detection_node` 可用）

---

### 阶段 5: 机器人包开发 (zy_robot)

**目标**: 实现机械臂控制和夹爪控制

**关键迁移**:
- `RoboticArm.py` → arm_controller_node.py
- `gripper_zhiyuan.py` → gripper_node.py
- 抓取序列 → arm_controller_node.py (mid → grasp → place)

**任务**:

1. **创建 gripper_node.py**

   **功能**:
   - 提供服务 `/gripper_control` (GripperControl.srv)
   - Modbus RTU 通信 (地址 0x01)
   - 位置控制 (0-1 范围，映射到 0-256000)

   **关键代码**:
   ```python
   import rclpy
   from rclpy.node import Node
   from zy_interfaces.srv import GripperControl
   from robotic_arm_package.robotic_arm import Arm, RM65
   import time

   class GripperNode(Node):
       def __init__(self):
           super().__init__('gripper_node')
           self.robot = Arm(RM65, "192.168.127.101", 8080)

           # Gripper parameters
           self.max_position = 256000  # 5 turns * 51200
           self.voltage = 3

           # Create service
           self.srv = self.create_service(GripperControl, '/gripper_control', self.control_callback)

       def control_callback(self, request, response):
           # Map 0-1 to 0-256000
           position = int(request.position * self.max_position)

           # Send Modbus command
           # Function code: 0x2a (control) or 0x2b (position)
           # ...

           response.success = True
           response.current_position = request.position
           return response

   def main(args=None):
       rclpy.init(args=args)
       node = GripperNode()
       rclpy.spin(node)
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **创建 arm_controller_node.py**

   **功能**:
   - 订阅 `/arm_grasp_command` (GraspCommand.msg)
   - 发布 `/arm_status` (String.msg)
   - 逆运动学求解
   - 碰撞检测和恢复
   - 执行完整抓取序列 (mid → approach → grasp → lift → place)

   **关键代码**:
   ```python
   import rclpy
   from rclpy.node import Node
   from zy_interfaces.msg import GraspCommand, GraspPose
   from std_msgs.msg import String
   from robotic_arm_package.robotic_arm import Arm, RM65
   import numpy as np

   class ArmControllerNode(Node):
       def __init__(self):
           super().__init__('arm_controller_node')

           # Robot initialization
           self.robot = Arm(RM65, "192.168.127.101", 8080)
           self.robot.Set_Collision_Stage(5)

           # Predefined poses (from config.py and grasp_zy_zhiyuan1215.py)
           self.init_pose = [86, -129, 127, -0.8, 71, -81]
           self.mid_pose = [0, -129, 127, -0.7, 71, -81]
           self.mid_pose1 = [0, -129, 80, -0.7, 100, -81]
           self.lift2init_pose = [65, -129, 127, -0.7, 77, 1]
           self.place_mid_pose = [65, -129, 60, 0, 121, 1]
           self.place_mid_pose2 = [69, -129, 60, 0, 9, 1]
           self.place_last_pose = [69, -104, 38, -2, 9, 1]

           # Robot speed
           self.robot_speed = 20

           # Create publishers/subscribers
           self.arm_cmd_sub = self.create_subscription(
               GraspCommand,
               '/arm_grasp_command',
               self.grasp_command_callback,
               10
           )
           self.arm_status_pub = self.create_publisher(String, '/arm_status', 10)

           # Gripper service client
           self.gripper_client = self.create_client(GripperControl, '/gripper_control')

       def grasp_command_callback(self, msg):
           """Execute grasp sequence based on received GraspPose"""
           try:
               self.execute_grasp_sequence(msg.grasp_pose)
           except Exception as e:
               self.get_logger().error(f'Grasp failed: {e}')
               self.arm_status_pub.publish(String(data=f'ERROR: {e}'))

       def execute_grasp_sequence(self, grasp_pose: GraspPose):
           """Execute full grasp sequence: mid → approach → grasp → place"""
           # 1. Move to mid pose
           self.move_to_pose(self.mid_pose)
           self.wait_for_completion()

           # 2. Move to approach pose (5cm above grasp)
           approach_pose = grasp_pose.copy()
           approach_pose.position.z += 0.05
           self.move_to_pose(approach_pose)
           self.wait_for_completion()

           # 3. Move to grasp pose
           self.move_to_pose(grasp_pose)
           self.wait_for_completion()

           # 4. Close gripper
           self.gripper_client.call_async(GripperControl.Request(position=0))

           # 5. Lift to place poses
           self.move_to_pose(self.mid_pose)
           self.move_to_pose(self.lift2init_pose)
           self.move_to_pose(self.place_mid_pose)
           self.move_to_pose(self.place_mid_pose2)
           self.move_to_pose(self.place_last_pose)
           self.wait_for_completion()

           # 6. Open gripper
           self.gripper_client.call_async(GripperControl.Request(position=1))

           # 7. Return to init
           self.move_to_pose(self.mid_pose)
           self.move_to_pose(self.init_pose)
           self.wait_for_completion()

           # Publish status
           self.arm_status_pub.publish(String(data='GRASP_COMPLETE'))

       def move_to_pose(self, pose):
           """Move robot to joint pose"""
           self.robot.Movej_Cmd(pose, self.robot_speed, 0)

       def wait_for_completion(self):
           """Wait for robot motion to complete"""
           # Robot SDK specific implementation
           time.sleep(0.5)

   def main(args=None):
       rclpy.init(args=args)
       node = ArmControllerNode()
       rclpy.spin(node)
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **创建 robot_params.yaml**

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

4. **创建 package.xml, setup.py, setup.cfg**

   **package.xml**:
   ```xml
   <?xml version="1.0"?>
   <package format="3">
     <name>zy_robot</name>
     <version>1.0.0</version>
     <description>Robot arm and gripper control</description>

     <buildtool_depend>ament_python</buildtool_depend>

     <depend>zy_interfaces</depend>
     <depend>sensor_msgs</depend>
     <depend>std_msgs</depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

   **setup.py**:
   ```python
   from setuptools import setup

   package_name = 'zy_robot'

   setup(
       name=package_name,
       version='1.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           ('share/' + package_name + '/launch', ['launch/robot_bringup.launch.py']),
           ('share/' + package_name + '/config', ['config/robot_params.yaml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
        entry_points={
            'console_scripts': [
                'arm_controller_node = zy_robot.arm_controller_node:main',
                'gripper_node = zy_robot.gripper_node:main',
            ],
        },
   )
   ```

5. **创建 robot_bringup.launch.py**

    ```python
    from launch import LaunchDescription
    from launch_ros.actions import Node
    from ament_index_python.packages import get_package_share_directory
    import os

    def generate_launch_description():
        pkg_share = get_package_share_directory('zy_robot')
        config_file = os.path.join(pkg_share, 'config', 'robot_params.yaml')

        return LaunchDescription([
            Node(
                package='zy_robot',
                executable='arm_controller_node',
                name='arm_controller_node',
                parameters=[config_file],
                output='screen'
            ),
            Node(
                package='zy_robot',
                executable='gripper_node',
                name='gripper_node',
                parameters=[config_file],
                output='screen'
            )
        ])
    ```

**验收标准**:
- [ ] `ros2 launch zy_robot robot_bringup.launch.py` 启动成功
- [ ] `ros2 node list` 显示 `arm_controller_node` 和 `gripper_node`
- [ ] `ros2 service list` 显示 `/gripper_control` 服务
- [ ] `ros2 topic list` 显示 `/arm_status` 主题
- [ ] 节点能独立运行（`ros2 run zy_robot gripper_node` 可用）

---

### 阶段 6: 通信包开发 (zy_comm)

**目标**: 实现 CTU TCP 通信和协议解析

**关键迁移**:
- `ctu_conn.py` (TCP 连接 + 心跳) → comm_node.py
- `ctu_protocol.py` (协议解析) → protocols/ctu_protocol.py

**任务**:

1. **创建 protocols/ctu_protocol.py**

   **功能**: 保持协议解析逻辑不变

   **关键代码**:
   ```python
   import struct
   from enum import IntEnum

   class CmdID(IntEnum):
       # CTU -> Robot
       CTU_GRASP_START = 0x70
       CTU_GRASP_SPEED = 0x71
       CTU_GRASP_STOP = 0x78
       CTU_GRASP_RELEASE = 0x79

       # Robot -> CTU
       GRASP_COUNT = 0x80
       GRASP_START = 0x81
       GRASP_OVER = 0x82

       # System
       HEARTBEAT = 0x99
       SLAM_OK = 0xE0
       GRASP_OK = 0xF0

   class CTUProtocol:
       SOF = b'\x55\xAA'
       CRC_POLYNOMIAL = 0xA001

       @staticmethod
       def build_segment(cmd_id, data):
           """Build protocol segment"""
           seg_len = len(data) + 1  # cmd_id + data
           segment = struct.pack('!BH', cmd_id, seg_len) + data
           return segment

       @staticmethod
       def build_frame(segments):
           """Build complete frame with CRC16"""
           payload = b''.join(segments)
           frame = CTUProtocol.SOF + payload + CTUProtocol.calculate_crc(payload)
           return frame

       @staticmethod
       def calculate_crc(data):
           """Calculate CRC16"""
           crc = 0
           for byte in data:
               crc ^= byte
               for _ in range(8):
                   if crc & 0x0001:
                       crc = (crc >> 1) ^ CTUProtocol.CRC_POLYNOMIAL
                   else:
                       crc >>= 1
           return struct.pack('!H', crc)

       @staticmethod
       def decode_frame(data):
           """Decode received frame"""
           # Validate SOF
           if data[:2] != CTUProtocol.SOF:
               raise ValueError("Invalid SOF")

           # Extract payload
           payload = data[2:-2]

           # Validate CRC
           received_crc = data[-2:]
           calculated_crc = CTUProtocol.calculate_crc(payload)
           if received_crc != calculated_crc:
               raise ValueError("CRC mismatch")

           # Parse segments
           segments = []
           offset = 0
           while offset < len(payload):
               cmd_id = struct.unpack('!B', payload[offset:offset+1])[0]
               seg_len = struct.unpack('!H', payload[offset+1:offset+3])[0]
               data_bytes = payload[offset+3:offset+3+seg_len-1]
               segments.append({'cmd_id': cmd_id, 'data': data_bytes})
               offset += 3 + seg_len - 1

           return segments
   ```

2. **创建 comm_node.py**

   **功能**:
   - TCP 连接到 CTU (192.168.127.253:8899)
   - 心跳机制 (10秒间隔)
   - 协议解析
   - 发布 CTU 命令到 `/ctu/command`
   - 订阅 `/executor/status` (反馈状态到 CTU)

   **关键代码**:
   ```python
   import rclpy
   from rclpy.node import Node
   from zy_interfaces.msg import CTUCommand, ExecutorStatus
   from std_msgs.msg import String
   import socket
   import threading
   import time
   from .protocols.ctu_protocol import CTUProtocol, CmdID

   class CommNode(Node):
       def __init__(self):
           super().__init__('comm_node')

           # Load parameters
           self.declare_parameter('ctu_ip', '192.168.127.253')
           self.declare_parameter('ctu_port', 8899)
           self.declare_parameter('heartbeat_interval', 10)
           self.ctu_ip = self.get_parameter('ctu_ip').value
           self.ctu_port = self.get_parameter('ctu_port').value
           self.heartbeat_interval = self.get_parameter('heartbeat_interval').value

           # Create publishers/subscribers
           self.ctu_pub = self.create_publisher(CTUCommand, '/ctu/command', 10)
           self.executor_status_sub = self.create_subscription(
               ExecutorStatus,
               '/executor/status',
               self.executor_status_callback,
               10
           )

           # TCP connection
           self.sock = None
           self.running = False

           # Connect to CTU
           self.connect()

           # Start threads
           self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
           self.listen_thread = threading.Thread(target=self.listen_loop, daemon=True)
           self.heartbeat_thread.start()
           self.listen_thread.start()

       def connect(self):
           while not self.running:
               try:
                   self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                   self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY,1)
                   self.sock.connect((self.ctu_ip, self.ctu_port))
                   self.running = True
                   self.get_logger().info(f'Connected to CTU at {self.ctu_ip}:{self.ctu_port}')
               except Exception as e:
                   self.get_logger().warning(f'Connection failed: {e}, retrying...')
                   time.sleep(1)

       def heartbeat_loop(self):
           """Send heartbeat every 10 seconds"""
           while self.running:
               try:
                   frame = CTUProtocol.build_frame([
                       CTUProtocol.build_segment(CmdID.HEARTBEAT, b'')
                   ])
                   self.safe_send(frame)
                   time.sleep(self.heartbeat_interval)
               except Exception as e:
                   self.get_logger().error(f'Heartbeat failed: {e}')
                   self.reconnect()

       def listen_loop(self):
           """Listen for CTU commands"""
           while self.running:
               try:
                   buffer = self.sock.recv(64)
                   if not buffer:
                       break
                   self.process_data(buffer)
               except Exception as e:
                   self.get_logger().error(f'Receive error: {e}')
                   self.reconnect()

       def process_data(self, data):
           """Parse CTU protocol and publish to ROS2"""
           try:
               segments = CTUProtocol.decode_frame(data)
               for seg in segments:
                   # Publish to ROS2 topic
                   msg = CTUCommand()
                   msg.cmd_id = seg['cmd_id']
                   msg.data = list(seg['data'])
                   self.ctu_pub.publish(msg)
           except Exception as e:
               self.get_logger().error(f'Parse error: {e}')

       def safe_send(self, frame):
           """Send frame with error handling"""
           try:
               self.sock.sendall(frame)
           except Exception as e:
               self.get_logger().error(f'Send error: {e}')
               self.reconnect()

       def reconnect(self):
           """Reconnect to CTU"""
           self.running = False
           self.sock.close()
           time.sleep(1)
           self.connect()

       def executor_status_callback(self, msg):
           """Send executor status back to CTU"""
           # Convert executor status to CTU protocol
           frame = CTUProtocol.build_frame([
               CTUProtocol.build_segment(CmdID.GRASP_COUNT, struct.pack('!I', msg.grasp_count))
           ])
           self.safe_send(frame)

   def main(args=None):
       rclpy.init(args=args)
       node = CommNode()
       rclpy.spin(node)
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **创建 comm_params.yaml**

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

4. **创建 package.xml, setup.py, setup.cfg**

   **package.xml**:
   ```xml
   <?xml version="1.0"?>
   <package format="3">
     <name>zy_comm</name>
     <version>1.0.0</version>
     <description>CTU communication (TCP + protocol parsing)</description>

     <buildtool_depend>ament_python</buildtool_depend>

     <depend>zy_interfaces</depend>
     <depend>std_msgs</depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

   **setup.py**:
   ```python
   from setuptools import setup

   package_name = 'zy_comm'

   setup(
       name=package_name,
       version='1.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           ('share/' + package_name + '/launch', ['launch/comm_bringup.launch.py']),
           ('share/' + package_name + '/config', ['config/comm_params.yaml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
        entry_points={
            'console_scripts': [
                'comm_node = zy_comm.comm_node:main',
            ],
        },
   )
   ```

5. **创建 comm_bringup.launch.py**

    ```python
    from launch import LaunchDescription
    from launch_ros.actions import Node
    from ament_index_python.packages import get_package_share_directory
    import os

    def generate_launch_description():
        pkg_share = get_package_share_directory('zy_comm')
        config_file = os.path.join(pkg_share, 'config', 'comm_params.yaml')

        return LaunchDescription([
            Node(
                package='zy_comm',
                executable='comm_node',
                name='comm_node',
                parameters=[config_file],
                output='screen'
            )
        ])
    ```

**验收标准**:
- [ ] `ros2 launch zy_comm comm_bringup.launch.py` 启动成功
- [ ] `ros2 node list` 显示 `comm_node`
- [ ] `ros2 topic list` 显示 `/ctu/command` 和 `/executor/status` 主题
- [ ] 节点能独立运行（`ros2 run zy_comm comm_node` 可用）

---

### 阶段 7: 执行器包开发 (zy_executor)

**目标**: 实现任务编排逻辑

**关键迁移**:
- `ctu_conn.py` (go_grasp) → executor_node.py (状态机)
- `grasp_zy_zhiyuan1215.py` (workflow) → executor_node.py (服务协调)

**任务**:

1. **创建 executor_node.py**

   **功能**:
   - 订阅 `/ctu/command` (CTU 命令)
   - 协调 `/detect_objects`, `/generate_grasp`, `/gripper_control` 服务
   - 发布 `/arm_grasp_command` (抓取位姿)
   - 发布 `/executor/status` (状态反馈)
   - 状态管理 (IDLE, DETECTING, PLANNING, EXECUTING)
   - 重试逻辑 (max_attempts=2, max_inverse_failures=3)

   **关键代码**:
   ```python
   import rclpy
   from rclpy.node import Node
   from zy_interfaces.msg import CTUCommand, ExecutorStatus
   from zy_interfaces.srv import DetectObjects, GenerateGrasp, GripperControl
   from enum import Enum
   import time

   class State(Enum):
       IDLE = 0
       DETECTING = 1
       PLANNING = 2
       EXECUTING = 3

   class ExecutorNode(Node):
       def __init__(self):
           super().__init__('executor_node')

           # State management
           self.state = State.IDLE
           self.grasp_running = False
           self.grasp_count = 0
           self.inverse_failures = 0
           self.max_attempts = 2
           self.max_inverse_failures = 3

           # Embedded goods mapping
           self.GOODS_MAPPING = {
               "1": "soap",
               "2": "interrupter",
               "3": "terminal",
               "4": "limit",
               "5": "voltage"
           }

           # Service clients
           self.detect_client = self.create_client(DetectObjects, '/detect_objects')
           self.grasp_client = self.create_client(GenerateGrasp, '/generate_grasp')
           self.gripper_client = self.create_client(GripperControl, '/gripper_control')

           # Topic publishers/subscribers
           self.arm_cmd_pub = self.create_publisher(ExecutorStatus, '/executor/status', 10)
           self.ctu_cmd_sub = self.create_subscription(
               CTUCommand,
               '/ctu/command',
               self.ctu_callback,
               10
           )

           # Wait for services
           self.detect_client.wait_for_service()
           self.grasp_client.wait_for_service()
           self.gripper_client.wait_for_service()

       def ctu_callback(self, msg):
           """Handle CTU commands"""
           if msg.cmd_id == 0x70:  # CTU_GRASP_START
               if not self.grasp_running:
                   self.run_grasp_sequence(msg.data)

       def run_grasp_sequence(self, data):
           """Execute grasp sequence with retry logic"""
           self.grasp_running = True
           self.state = State.DETECTING

           # Decode goods ID from data
           goods_id = chr(data[0]) if data else "1"
           goods_name = self.GOODS_MAPPING.get(goods_id, "soap")
           self.get_logger().info(f'Starting grasp for: {goods_name}')

           # Detect objects
           self.state = State.DETECTING
           future = self.detect_client.call_async(DetectObjects.Request())
           rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

           if not future.done():
               self.get_logger().error('Detection timeout')
               self.grasp_running = False
               return

           detection_result = future.result()
           if not detection_result.success:
               self.get_logger().error('Detection failed')
               self.grasp_running = False
               return

           # Generate grasp
           self.state = State.PLANNING
           for result in detection_result.results:
               future = self.grasp_client.call_async(
                   GenerateGrasp.Request(
                       label=result.label,
                       bbox=result.bbox.tolist(),
                       visualize=False
                   )
               )
               rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

               if not future.done():
                   self.get_logger().error('Grasp generation timeout')
                   self.inverse_failures += 1
                   continue

               grasp_result = future.result()
               if grasp_result.success:
                   # Execute grasp
                   self.state = State.EXECUTING
                   self.execute_grasp(grasp_result.grasps[0])
                   self.grasp_count += 1
                   break
               else:
                   self.get_logger().warning(f'Grasp generation failed: {grasp_result.message}')
                   self.inverse_failures += 1

               if self.inverse_failures >= self.max_inverse_failures:
                   self.get_logger().error('Max inverse failures reached')
                   break

           self.grasp_running = False
           self.state = State.IDLE

           # Publish final status
           status_msg = ExecutorStatus()
           status_msg.state = self.state.value
           status_msg.grasp_count = self.grasp_count
           status_msg.message = 'SEQUENCE_COMPLETE'
           self.arm_cmd_pub.publish(status_msg)

       def execute_grasp(self, grasp_pose):
           """Execute single grasp"""
           # Publish grasp command to arm_controller
           # arm_controller will handle the full sequence
           from zy_interfaces.msg import GraspCommand

           cmd = GraspCommand()
           cmd.grasp_pose = grasp_pose
           self.arm_cmd_pub.publish(cmd)

           # Wait for completion (subscribe to /arm_status)
           # ...

   def main(args=None):
       rclpy.init(args=args)
       node = ExecutorNode()
       rclpy.spin(node)
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **创建 executor_params.yaml**

   ```yaml
   grasp:
     max_attempts: 2
     max_inverse_failures: 3

   orchestration:
     service_timeout: 10.0  # seconds
     arm_timeout: 30.0      # seconds
   ```

3. **创建 package.xml, setup.py, setup.cfg**

   **package.xml**:
   ```xml
   <?xml version="1.0"?>
   <package format="3">
     <name>zy_executor</name>
     <version>1.0.0</version>
     <description>Task orchestration (coordinates vision services and robot control)</description>

     <buildtool_depend>ament_python</buildtool_depend>

     <depend>zy_interfaces</depend>
     <depend>std_msgs</depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

   **setup.py**:
   ```python
   from setuptools import setup

   package_name = 'zy_executor'

   setup(
       name=package_name,
       version='1.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           ('share/' + package_name + '/launch', ['launch/executor_bringup.launch.py']),
           ('share/' + package_name + '/config', ['config/executor_params.yaml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
        entry_points={
            'console_scripts': [
                'executor_node = zy_executor.executor_node:main',
            ],
        },
   )
   ```

**验收标准**:
- [ ] `ros2 launch zy_executor executor_bringup.launch.py` 启动成功
- [ ] `ros2 node list` 显示 `executor_node`
- [ ] `ros2 topic list` 显示 `/arm_grasp_command` 和 `/executor/status` 主题
- [ ] 节点能独立运行（`ros2 run zy_executor executor_node` 可用）

---

### 阶段 8: 系统启动器开发 (zy_bringup)

**目标**: 创建统一的系统启动文件和配置管理

**任务**:

1. **创建 static_tfs.launch.py**

   **功能**: 发布静态坐标变换 (Tcam2base, TCP 补偿)

   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node
   import numpy as np
   from scipy.spatial.transform import Rotation as R

   def generate_launch_description():
       # Tcam2base matrix (from config.py)
       T_cam2base = np.array([
           [0.02742095, -0.99940903, -0.0207286, 0.20841901],
           [0.9995487, 0.02766746, -0.01170045, -0.02848768],
           [0.01226705, -0.02039841, 0.99971667, 0.03739014],
           [0., 0., 0., 1., ]
       ])

       # Extract translation
       translation = T_cam2base[:3, 3]

       # Convert rotation matrix to quaternion
       rotation_matrix = T_cam2base[:3, :3]
       r = R.from_matrix(rotation_matrix)
       qx, qy, qz, qw = r.as_quat()

       return LaunchDescription([
           # Camera to base transform
           Node(
               package='tf2_ros',
               executable='static_transform_publisher',
               name='camera_to_base_tf',
               arguments=[
                   str(translation[0]), str(translation[1]), str(translation[2]),
                   str(qx), str(qy), str(qz), str(qw),
                   'base_link',
                   'camera_color_optical_frame'
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

2. **创建 grasp_system.launch.py**

   **功能**: 启动完整系统 (所有包)

   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from launch.actions import IncludeLaunchDescription, TimerAction
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       return LaunchDescription([
           # 1. Static TFs
           IncludeLaunchDescription(
               PythonLaunchDescriptionSource([
                   get_package_share_directory('zy_bringup') + '/launch/static_tfs.launch.py'
               ])
           ),

           # 2. Camera node
           TimerAction(
               period=0.0,
               actions=[
                   IncludeLaunchDescription(
                       PythonLaunchDescriptionSource([
                           get_package_share_directory('zy_camera') + '/launch/camera_bringup.launch.py'
                       ])
                   )
               ]
           ),

           # 3. Detection service (delayed 1s)
           TimerAction(
               period=1.0,
               actions=[
                   IncludeLaunchDescription(
                       PythonLaunchDescriptionSource([
                           get_package_share_directory('zy_vision') + '/launch/inference_bringup.launch.py'
                       ])
                   )
               ]
           ),

           # 4. Arm controller (delayed 1s)
           TimerAction(
               period=1.0,
               actions=[
                   IncludeLaunchDescription(
                       PythonLaunchDescriptionSource([
                           get_package_share_directory('zy_robot') + '/launch/robot_bringup.launch.py'
                       ])
                   )
               ]
           ),

            # 5. Gripper service (delayed 2s)
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package='zy_robot',
                        executable='gripper_node',
                        name='gripper_node',
                        parameters=[get_package_share_directory('zy_robot') + '/config/robot_params.yaml'],
                        output='screen'
                    )
                ]
            ),

            # 6. CTU communication (delayed 3s)
            TimerAction(
                period=3.0,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource([
                            get_package_share_directory('zy_comm') + '/launch/comm_bringup.launch.py'
                        ])
                    )
                ]
            ),

            # 7. Executor orchestrator (delayed 4s)
            TimerAction(
                period=4.0,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource([
                            get_package_share_directory('zy_executor') + '/launch/executor_bringup.launch.py'
                        ])
                    )
                ]
            )
       ])
   ```

3. **创建配置文件**

   **camera_params.yaml** (已在阶段 3 创建)

   **robot_params.yaml** (已在阶段 5 创建)

   **comm_params.yaml** (已在阶段 6 创建)

   **inference_params.yaml** (已在阶段 4 创建)

4. **创建 package.xml 和 CMakeLists.txt**

   **package.xml**:
   ```xml
   <?xml version="1.0"?>
   <package format="3">
     <name>zy_bringup</name>
     <version>1.0.0</version>
     <description>System launcher and configuration hub</description>

     <buildtool_depend>ament_cmake</buildtool_depend>

     <exec_depend>zy_camera</exec_depend>
     <exec_depend>zy_vision</exec_depend>
     <exec_depend>zy_robot</exec_depend>
     <exec_depend>zy_comm</exec_depend>
     <exec_depend>zy_executor</exec_depend>
     <exec_depend>tf2_ros</exec_depend>

     <export>
       <build_type>ament_cmake</build_type>
     </export>
   </package>
   ```

   **CMakeLists.txt**:
   ```cmake
   cmake_minimum_required(VERSION 3.8)
   project(zy_bringup)

   if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     add_compile_options(-Wall -Wextra -Wpedantic)
   endif()

   find_package(ament_cmake REQUIRED)

   # Install launch files and configs
   install(DIRECTORY
     launch
     config
     DESTINATION share/${PROJECT_NAME}
   )

   ament_package()
   ```

**验收标准**:
- [ ] `ros2 launch zy_bringup grasp_system.launch.py` 启动所有节点
- [ ] 静态 TF 正确发布 (`ros2 tf2 ls`)
- [ ] 所有节点正常运行 (`ros2 node list`)
- [ ] 所有主题和服务可用

---

### 阶段 9: 集成测试

**目标**: 测试完整系统功能

**任务**:

1. **单元测试**

   **相机包**:
   ```bash
   # 启动相机
   ros2 launch zy_camera camera_bringup.launch.py

   # 验证主题
   ros2 topic hz /camera/camera/color/image_raw
   ros2 topic hz /camera/camera/depth/image_raw
   ros2 topic echo /camera/camera/color/camera_info --once
   ```

   **视觉包**:
   ```bash
   # 启动视觉服务
   ros2 launch zy_vision inference_bringup.launch.py

   # 测试检测服务
   ros2 service call /detect_objects zy_interfaces/srv/DetectObjects "{}"

   # 测试抓取生成服务
   ros2 service call /generate_grasp zy_interfaces/srv/GenerateGrasp "{label: 1, bbox: [100, 100, 200, 200], visualize: false}"
   ```

   **机器人包**:
   ```bash
   # 启动机器人节点
   ros2 launch zy_robot robot_bringup.launch.py

   # 测试夹爪控制
   ros2 service call /gripper_control zy_interfaces/srv/GripperControl "{position: 0.5}"

   # 测试机械臂命令
   ros2 topic pub /arm_grasp_command zy_interfaces/msg/GraspCommand "{grasp_pose: {position: [0.1, 0.1, 0.3], orientation: [0.0, 0.0, 0.0, 1.0], grasp_width: 0.05, approach_dist: 0.05}}" --once
   ```

   **通信包**:
   ```bash
   # 启动 CTU 通信
   ros2 launch zy_comm comm_bringup.launch.py

   # 验证 CTU 命令发布
   ros2 topic echo /ctu/command
   ```

2. **集成测试**

   **启动完整系统**:
   ```bash
   ros2 launch zy_bringup grasp_system.launch.py
   ```

   **验证节点连接**:
   ```bash
   # 检查所有节点
   ros2 node list

   # 检查主题连接 (使用 rqt_graph)
   ros2 run rqt_graph rqt_graph
   ```

   **测试抓取流程**:
   - 手动发送 CTU 命令 (通过模拟 CTU 或测试脚本)
   - 验证检测 → 抓取生成 → 机械臂执行 → 放置
   - 验证状态反馈到 CTU

3. **性能测试**

   - 单次抓取时间: < 5 秒
   - 模型推理时间: < 100ms
   - 通信延迟: < 100ms
   - 系统启动时间: < 30 秒

4. **压力测试**

   - 连续抓取 10 次
   - 验证重试逻辑 (max_inverse_failures)
   - 验证碰撞恢复

**验收标准**:
- [ ] 所有单元测试通过
- [ ] 集成测试通过
- [ ] 性能指标满足要求
- [ ] 无内存泄漏
- [ ] 稳定运行 1 小时无崩溃

---

## 完善缺失部分

### 1. 模型权重准备

**问题**: `models/weights/epoch_20_last.pth` 文件不存在

**解决方案**:
1. 检查原始系统是否在其他位置有模型权重
2. 如果不存在，需要重新训练模型或从备份恢复
3. 确认模型路径配置正确

**操作**:
```bash
# 查找所有 .pth 文件
find /Users/wk/programs/grasp_zy -name "*.pth"

# 查找 epoch 相关文件
find /Users/wk/programs/grasp_zy -name "*epoch*"
```

### 2. 模型路径配置

**问题**: 模型路径是相对路径，在 ROS2 包中需要绝对路径或资源路径

**解决方案**:
1. 使用 `ament_index_python` 获取包共享路径
2. 在 YAML 配置中使用绝对路径或变量

**示例**:
```python
from ament_index_python.packages import get_package_share_directory

pkg_share = get_package_share_directory('grasp_zy')
model_path = os.path.join(pkg_share, '..', '..', 'models', 'weights', 'epoch_20_last.pth')
```

### 3. robotic_arm_package 集成

**问题**: robotic_arm_package 是独立 SDK，需要正确集成到 ROS2 包

**解决方案**:
1. 将 robotic_arm_package 安装到系统 Python 路径
2. 或在 ROS2 包中添加相对导入路径
3. 确保 RM65 SDK 正确初始化

**操作**:
```bash
# 方案 1: 安装到系统
cd /Users/wk/programs/grasp_zy/robotic_arm_package
pip install -e .

# 方案 2: 在 ROS2 包中添加路径
# 在 setup.py 中添加 sys.path
import sys
sys.path.append('/path/to/robotic_arm_package')
```

### 4. 数据集路径配置

**问题**: `doc/single_new.txt`, `dataset/cornell.data` 等路径需要确认

**解决方案**:
1. 确认这些文件存在于原始系统
2. 在 ROS2 包中正确配置路径
3. 使用绝对路径或资源路径

### 5. 日志和调试

**问题**: 需要统一的日志和调试机制

**解决方案**:
1. 使用 ROS2 的 logging 系统 (`self.get_logger()`)
2. 定义统一的日志级别 (DEBUG, INFO, WARNING, ERROR)
3. 添加关键节点的调试输出

---

## 时间估算

| 阶段 | 工作量 | 依赖 |
|------|--------|------|
| 阶段 0: 环境准备 | 1 天 | 无 |
| 阶段 1: 创建包结构 | 0.5 天 | 阶段 0 |
| 阶段 2: 接口定义 | 0.5 天 | 阶段 1 |
| 阶段 3: 相机包开发 | 1 天 | 阶段 2 |
| 阶段 4: 视觉包开发 | 3 天 | 阶段 2, 3 |
| 阶段 5: 机器人包开发 | 2 天 | 阶段 2 |
| 阶段 6: 通信包开发 | 1.5 天 | 阶段 2 |
| 阶段 7: 执行器包开发 | 2 天 | 阶段 2, 4, 5, 6 |
| 阶段 8: 系统启动器开发 | 0.5 天 | 阶段 3, 4, 5, 6, 7 |
| 阶段 9: 集成测试 | 2 天 | 阶段 8 |
| **总计** | **14 天** | |

**注意**: 这是一个估算，实际时间可能因熟悉程度和调试时间而异。

---

## 风险和缓解措施

### 风险 1: 模型权重文件缺失

**风险**: 模型权重文件不存在或路径错误

**影响**: 无法进行检测和抓取生成

**缓解措施**:
1. 提前确认模型文件位置
2. 备份原始模型权重
3. 准备模型重新训练计划

### 风险 2: realsense-ros 版本不兼容

**风险**: realsense-ros 在 Jetson 上的版本问题

**影响**: 相机无法启动或数据异常

**缓解措施**:
1. 测试 realsense-ros 安装
2. 准备备用版本
3. 参考官方 Jetson 安装文档

### 风险 3: robotic_arm_package 兼容性

**风险**: RM65 SDK 与 ROS2 不兼容

**影响**: 无法控制机械臂

**缓解措施**:
1. 提前测试 SDK 导入
2. 准备备用机械臂控制方案
3. 与 SDK 供应商沟通

### 风险 4: 坐标变换错误

**风险**: TF2 变换不正确导致抓取失败

**影响**: 抓取精度下降或失败

**缓解措施**:
1. 验证 Tcam2base 矩阵正确性
2. 使用 rviz2 可视化坐标系
3. 准备标定工具

### 风险 5: 性能不达标

**风险**: ROS2 开销导致性能下降

**影响**: 抓取时间超过 5 秒

**缓解措施**:
1. 优化节点间通信
2. 使用共享内存 (ZeroMQ)
3. 考虑使用 ROS2 性能优化选项

---

## 后续改进

### 短期改进 (1-2 周)

1. **添加单元测试**: 使用 pytest 或 gtest
2. **添加文档**: 每个节点的 README 和 API 文档
3. **添加可视化**: 使用 rviz2 显示抓取点
4. **性能优化**: 分析瓶颈并优化

### 中期改进 (1-2 月)

1. **添加监控**: 使用 rqt_console, rqt_plot 监控状态
2. **添加日志**: 使用 ros2 bag 记录数据
3. **添加诊断**: 使用 diagnostic_updater
4. **添加热重载**: 支持参数动态加载

### 长期改进 (3-6 月)

1. **添加仿真**: 使用 Gazebo 或 Webots 进行仿真
2. **添加 UI**: 使用 rqt_gui 或 Qt 添加控制界面
3. **添加扩展**: 支持多相机、多机器人
4. **添加 AI**: 使用强化学习优化抓取策略

---

## 参考资源

### ROS2 文档

- [ROS2 Humble 官方文档](https://docs.ros.org/en/humble/)
- [ROS2 Python 节点教程](https://docs.ros.org/en/humble/Tutorials/Python-Programming-Nodes.html)
- [ROS2 服务教程](https://docs.ros.org/en/humble/Tutorials/Python-Programming-Services.html)

### 相关项目

- [Intel RealSense ROS2](https://github.com/IntelRealSense/realsense-ros)
- [MoveIt2](https://moveit.ros.org/)
- [Nav2](https://navigation.ros.org/)

### 深度学习

- [MMDetection 文档](https://mmdetection.readthedocs.io/)
- [PyTorch ROS2 集成](https://github.com/pytorch/pytorch/tree/master/torch/utils/tensorboard)

---

## ROS2 命名更新总结

### 文件名对照表（已应用新命名）

| 原设计文件名 | 新文件名 | 类名 | 可执行名 |
|-------------|---------|------|---------|
| `detection_server.py` | `detection_node.py` | `DetectionNode` | `detection_node` |
| `grasp_generator.py` | `grasp_node.py` | `GraspNode` | `grasp_node` |
| `arm_controller.py` | `arm_controller_node.py` | `ArmControllerNode` | `arm_controller_node` |
| `gripper_server.py` | `gripper_node.py` | `GripperNode` | `gripper_node` |
| `ctu_communication.py` | `comm_node.py` | `CommNode` | `comm_node` |
| `ctu_orchestrator.py` | `executor_node.py` | `ExecutorNode` | `executor_node` |
| `ctu_protocol.py` | `ctu_protocol.py` | `CTUProtocol` | N/A (非节点文件) |

### ROS2 命令示例（更新后）

#### 启动单个节点
```bash
# 检测节点
ros2 run zy_vision detection_node

# 抓取节点
ros2 run zy_vision grasp_node

# 机械臂控制节点
ros2 run zy_robot arm_controller_node

# 夹爪节点
ros2 run zy_robot gripper_node

# 通信节点
ros2 run zy_comm comm_node

# 编排节点
ros2 run zy_executor executor_node
```

#### 列出节点
```bash
ros2 node list
# 输出示例：
# /detection_node
# /grasp_node
# /arm_controller_node
# /gripper_node
# /comm_node
# /executor_node
```

#### 测试服务
```bash
# 检查检测服务
ros2 service list | grep detection
ros2 service list | grep grasp
ros2 service list | grep gripper

# 测试检测服务
ros2 service call /detect_objects zy_interfaces/srv/DetectObjects "{}"

# 测试抓取生成服务
ros2 service call /generate_grasp zy_interfaces/srv/GenerateGrasp "{label: 1, bbox: [100, 100, 200, 200], visualize: false}"

# 测试夹爪控制服务
ros2 service call /gripper_control zy_interfaces/srv/GripperControl "{position: 0.5}"
```

---

## 总结

本计划提供了一个详细的 ROS2 重构路线图，从环境准备到集成测试的完整流程。关键设计决策包括:

1. **相机/视觉分离**: 提高部署灵活性
2. **编排/通信分离**: 单一职责，松耦合
3. **使用标准组件**: realsense-ros, TF2
4. **集中配置管理**: YAML 文件统一管理参数

通过遵循本计划，可以逐步将单体式 Python 系统重构为模块化、可维护、可扩展的 ROS2 系统。
