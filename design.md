# ROS2 Package Architecture Design Document

## Project Overview

**Project**: Visual Robotic Grasping System (grasp_zy)
**Target Platform**: ROS2 on Jetson Orin NX
**Goal**: Refactor monolithic Python system into modular ROS2 packages

---

## Design Rationale

### Key Design Decisions

#### 1. Separate Camera and Inference Packages

**Decision**: Split `zy_vision` into `zy_camera` and `zy_vision`

**Rationale**:

- **Dependency Isolation**: Camera package only needs pyrealsense2; Vision package needs PyTorch
- **Flexible Deployment**: Can deploy camera-only to edge devices without GPU
- **Resource Optimization**: Devices without inference don't load heavy libraries
- **Follows ROS2 Standards**: Matches real sense drivers (real sense2_camera), yolo_ros, fetch_robotics

**Package Count**: 7 packages (separated orchestration into dedicated package)

#### 2. No Separate Calibration Package

**Decision**: Do NOT create `zy_calib` package

**Rationale**:

- **Intrinsic Parameters**: Published via `sensor_msgs/CameraInfo` by camera_node
- **Extrinsic Parameters**: Published via TF2 static_transform_publisher
- **Compensation Parameters**: Embedded inline in nodes or via YAML parameters
- **Avoids Confusion**: `zy_calib` implies a calibration toolkit, which this is not

**Alternative Considered and Rejected**:

- `zy_config` (too generic)
- `zy_common` (too broad, unclear scope)
- `zy_calib` (implies calibration toolkit functionality)

#### 3. Inline Utility Functions

**Decision**: Embed frequently used utility functions directly in nodes that use them

**Rationale**:

- **Usage Analysis**: Only 2 functions from utils/utils.py (770 lines) are actually used:
    - `in_paint` - used only in grasp_generator
    - `letterbox` / `letterbox1` - used only in grasp_generator
- **Code Quality**: 10+ functions never called (dead code)
- **Dependencies**: Avoids complex inter-package Python module dependencies

**Functions Migrated**:

- `in_paint()` → zy_vision/grasp_generator.py
- `letterbox()` → zy_vision/grasp_generator.py
- `scale_coords()` → zy_vision/grasp_generator.py

**Functions Not Migrated** (dead code):

- `float3`, `init_seeds`, `model_info`, `weights_init_normal`, `coco_class_weights`, etc.

#### 4. Centralized YAML Configuration

**Decision**: All YAML configs in `zy_bringup/config/`

**Rationale**:

- **Unified Management**: Single location for all runtime parameters
- **ROS2 Standard**: Follows ros2 parameter loading patterns
- **Easy Updates**: Modify parameters without code changes
- **Type Safety**: Calibration data in Python (if needed) or static TF

#### 5. No Separate TF Package

**Decision**: Use launch file to publish static TFs, not a `zy_tf` package

**Rationale**:

- **Simpler**: `tf2_ros` package provides the node, no need to wrap it
- **Standard**: All ROS2 projects use static_transform_publisher directly in launch files
- **Less Code**: No custom Python code needed for TF publishing

#### 6. Separate Orchestration Package

**Decision**: Create dedicated `zy_executor` package for task orchestration, **NOT** merge into `zy_comm`

**Rationale**:

- **Single Responsibility**: `zy_comm` handles TCP communication only; `zy_executor` handles orchestration logic only
- **Independent Deployment**: CTU communication can be deployed on edge device near CTU; orchestration runs on main
  robot
- **Clear Testing Boundary**: Each package can be tested independently (mock CTU commands, mock service responses)
- **Flexible Upgrades**: Communication protocol changes don't affect orchestration logic; orchestration algorithm
  changes don't affect communication
- **Scalability**: Can easily add multiple task sources (not just CTU) or multiple executors in the future
- **Standard ROS2 Pattern**: Matches real-world architectures (Nav2's separate BT executors, industrial systems with
  separate communication/orchestration layers)

**Package Separation Benefits**:

- **zy_comm**: Pure TCP Socket + binary protocol handling → lightweight, minimal dependencies
- **zy_executor**: ROS2 orchestration logic → service clients, topic publishers, state management
- **Loose Coupling**: Communication via ROS2 topics (`/ctu/command`), not function calls

**Data Flow**:

  ```
  CTU Device
      ↓ (TCP binary protocol)
  zy_comm/ctu_communication
      ↓ (publishes ROS2 topic)
  /ctu/command (CTUCommand.msg)
      ↓
  zy_executor/ctu_orchestrator
      ↓ (subscribes + orchestrates services)
  ├─→ /detect_objects (service call)
  ├─→ /generate_grasp (service call)
  ├─→ /gripper_control (service call)
  └─→ /arm_grasp_command (topic publish)
  ```

---

## Package Structure

```
Ros2/
├── zy_interfaces/              # Interface definitions
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
├── zy_camera/                   # Camera driver (NEW)
│   ├── nodes/
│   │   └── camera_node.py
│   ├── launch/
│   │   └── camera_bringup.launch.py
│   └── config/
│       └── camera_params.yaml
│
├── zy_vision/                 # Vision inference (MODIFIED)
│   ├── nodes/
│   │   ├── detection_server.py
│   │   └── grasp_generator.py
│   ├── launch/
│   │   └── inference_bringup.launch.py
│   └── config/
│       └── inference_params.yaml
│
├── zy_robot/                  # Robot control
│   ├── nodes/
│   │   ├── arm_controller.py
│   │   └── gripper_server.py
│   ├── launch/
│   │   └── robot_bringup.launch.py
│   └── config/
│       └── robot_params.yaml
│
  ├── zy_comm/                   # CTU communication only (NEW)
  │   ├── nodes/
  │   │   └── ctu_communication.py
  │   ├── protocols/
  │   │   └── ctu_protocol.py
  │   ├── launch/
  │   │   └── comm_bringup.launch.py
  │   └── config/
  │       └── comm_params.yaml
  │
  ├── zy_executor/               # Task orchestration (NEW)
  │   ├── nodes/
  │   │   └── ctu_orchestrator.py
  │   ├── launch/
  │   │   └── executor_bringup.launch.py
  │   └── config/
  │       └── executor_params.yaml
  │
 └── zy_bringup/               # System launcher
     ├── launch/
     │   ├── grasp_system.launch.py      # Full system
     │   ├── static_tfs.launch.py       # TF publishing
     │   └── camera_calib.launch.py     # CameraInfo parameters
     └── config/
         ├── camera_params.yaml           # Camera intrinsics
         ├── robot_params.yaml            # Robot config
         └── comm_params.yaml            # Communication config
 ```

---

## Package Responsibilities

### zy_interfaces

**Role**: Define communication contracts
**Contains**: msg, srv, action definitions
**Dependencies**: None (pure definitions)

---

### zy_camera

**Role**: RealSense camera driver
**Primary Node**: `camera_node.py`

**Functionality**:

- Publish RGB images: `/camera/color/image_raw` (sensor_msgs/Image)
- Publish depth images: `/camera/depth/image_raw` (sensor_msgs/Image)
- Publish CameraInfo: `/camera/color/camera_info` (sensor_msgs/CameraInfo)
- Publish CameraInfo: `/camera/depth/camera_info` (sensor_msgs/CameraInfo)

**Configuration** (`camera_params.yaml`):

```yaml
camera:
  width: 640
  height: 480
  publish_rate: 10.0  # Hz

intrinsics:
  color:
    frame_id: "camera_color_optical_frame"
    fx: 604.335
    fy: 604.404
    cx: 316.187
    cy: 248.611

  depth:
    frame_id: "camera_depth_optical_frame"
    fx: 604.335
    fy: 604.404
    cx: 316.187
    cy: 248.611
```

**Dependencies**:

- `sensor_msgs` (standard ROS2)
- `cv_bridge` (standard ROS2)
- `pyrealsense2` (RealSense SDK)

**Key Design**: Publishes CameraInfo with intrinsic parameters so other nodes can perform pixel-to-3D conversion without
hardcoded calibration data.

---

### zy_vision

**Role**: Visual inference (detection + grasp generation)
**Primary Nodes**: `detection_server.py`, `grasp_generator.py`

**Functionality**:

#### detection_server.py

- Subscribe: `/camera/color/image_raw`
- Provide service: `/detect_objects` (DetectObjects.srv)
- Load MMDetection model
- Perform NMS filtering

#### grasp_generator.py

- Subscribe: `/camera/color/image_raw`, `/camera/depth/image_raw`
- Subscribe: `/camera/color/camera_info`, `/camera/depth/camera_info`
- Subscribe service: `/detect_objects`
- Provide service: `/generate_grasp` (GenerateGrasp.srv)
- Load AugmentCNN model
- Embedded utility functions: `in_paint()`, `letterbox()`, `scale_coords()`

**Configuration** (`inference_params.yaml`):

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

**Dependencies**:

- `sensor_msgs` (standard ROS2)
- `cv_bridge` (standard ROS2)
- `PyTorch`
- `MMDetection`
- Custom models (AugmentCNN)

**Key Design**:

- Separated from camera driver for independent deployment
- Utility functions embedded to avoid complex Python module dependencies
- Uses CameraInfo from camera_node for accurate coordinate transformation

---

### zy_robot

**Role**: Robot arm and gripper control
**Primary Nodes**: `arm_controller.py`, `gripper_server.py`

**Functionality**:

#### arm_controller.py

- Subscribe: `/arm_grasp_command` (grasp pose commands)
- Publish: `/arm_status` (state updates)
- Provide inverse kinematics
- Collision detection and recovery
- Predefined poses (init, mid, place, etc.)
- Execute full grasp sequence (grasp → place) when receiving grasp commands

#### gripper_server.py

- Provide service: `/gripper_control` (GripperControl.srv)
- Modbus RTU communication
- Position control (0-1 range)

**Configuration** (`robot_params.yaml`):

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

**Dependencies**:

- `sensor_msgs` (standard ROS2)
- `std_msgs` (standard ROS2)
- `robotic_arm_package` (RM65 SDK)

**Key Design**: Hardware abstraction through ROS2 services - other nodes don't need to know about SDK details.

---

### zy_comm

**Role**: CTU device communication (TCP Socket + protocol parsing only)
**Primary Node**: `ctu_communication.py`

**Functionality**:

- TCP connection to CTU (192.168.127.253:8899)
- Heartbeat mechanism (10s interval)
- Protocol parsing (binary + CRC16)
- Publish CTU commands to ROS2 topic: `/ctu/command`
- Subscribe executor status: `/executor/status` (for feedback to CTU)

**Configuration** (`comm_params.yaml`):

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

**Dependencies**:

- `std_msgs` (standard ROS2)
- `zy_interfaces` (message definitions: CTUCommand)

**Key Design**:

- **Pure Communication**: No orchestration logic, only TCP + protocol handling
- **ROS2 Bridge**: Converts CTU binary protocol to ROS2 messages
- **Stateless**: No orchestration state; executor maintains all state
- **Independent**: Can be deployed separately from orchestration

  ---

### zy_executor

**Role**: Task orchestration (coordinates vision services and robot control)
**Primary Node**: `ctu_orchestrator.py`

**Functionality**:

- Subscribe CTU commands: `/ctu/command` (from zy_comm)
- Coordinate vision services and robot control
- State management: grasp_running flag, attempt counters, inverse failure tracking
- Service clients: `/detect_objects`, `/generate_grasp`, `/gripper_control`
- Topic publisher: `/arm_grasp_command` (sends grasp poses to arm_controller)
- Topic publisher: `/executor/status` (reports orchestration state to CTU)
- Subscribe arm status: `/arm_status` (waits for arm execution completion)

**Embedded Mapping** (no separate config file needed):

  ```python
  GOODS_MAPPING = {
    "1": "soap",
    "2": "interrupter",
    "3": "terminal",
    "4": "limit",
    "5": "voltage"
}
  ```

**Embedded State Variables** (simple flags, no complex state machine):

  ```python
  # Orchestration state
self.grasp_running = False  # Prevent concurrent commands
self.grasp_count = 0  # Completed grasp count
self.inverse_failures = 0  # IK failure count
self.max_attempts = 2  # Single grasp retry limit
self.max_inverse_failures = 3  # Total IK failure limit
  ```

**Configuration** (`executor_params.yaml`):

  ```yaml
  grasp:
    max_attempts: 2
    max_inverse_failures: 3

  orchestration:
    service_timeout: 10.0  # seconds
    arm_timeout: 30.0      # seconds
  ```

**Dependencies**:

- `std_msgs` (standard ROS2)
- `zy_interfaces` (message definitions)

**Key Design**:

- **Pure Orchestration**: No TCP communication, only ROS2 topics/services
- **Decision-Driven Loop**: While loop based on detection results, not fixed iteration
- **State Machine**: Simple flag-based state (IDLE, DETECTING, PLANNING, EXECUTING)
- **Feedback to CTU**: Publishes status to `/executor/status` for CTU communication node
- **Service Coordination**: Sequentially calls detect → generate → grasp services

 ---

### zy_bringup

**Role**: System launcher and configuration hub
**Launch Files**:

#### static_tfs.launch.py

Publishes static coordinate transforms using `tf2_ros/static_transform_publisher`:

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

**Key Design**:

- Uses standard `tf2_ros` package
- No custom Python code needed
- Extrinsic calibration (Tcam2base) matrix converted to quaternion
- Separate transforms for camera-base and TCP compensation

#### camera_calib.launch.py

Launches camera_node with intrinsic parameters from YAML:

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

**Key Design**:

- Intrinsic parameters passed from launch file to camera_node
- CameraInfo published by camera_node using these parameters
- Other nodes subscribe to `/camera/color/camera_info` to get intrinsics

#### grasp_system.launch.py

Main system launch file - orchestrates all packages:

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

        # 5. CTU communication + orchestration (delayed)
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

**Key Design**:

- Uses IncludeLaunchDescription to reference package-specific launch files
- Package independence: each package has its own bringup launch
- Timed startup: ensures dependencies are ready
- Central orchestration: single command starts entire system

---

## Communication Architecture

### ROS2 Topics and Services

| Topic/Service               | Type                             | Publisher                    | Subscriber                                              | Purpose                        |
 |-----------------------------|----------------------------------|------------------------------|---------------------------------------------------------|--------------------------------|
| `/camera/color/image_raw`   | sensor_msgs/Image                | zy_camera/camera_node        | zy_vision/detection_server, zy_vision/grasp_generator   | RGB image data                 |
| `/camera/color/camera_info` | sensor_msgs/CameraInfo           | zy_camera/camera_node        | zy_vision/detection_server, zy_vision/grasp_generator   | Intrinsic parameters           |
| `/camera/depth/image_raw`   | sensor_msgs/Image                | zy_camera/camera_node        | zy_vision/grasp_generator                               | Depth image data               |
| `/camera/depth/camera_info` | sensor_msgs/CameraInfo           | zy_camera/camera_node        | zy_vision/grasp_generator                               | Intrinsic parameters           |
| `/detect_objects`           | zy_interfaces/srv/DetectObjects  | zy_vision/detection_server   | zy_vision/grasp_generator, zy_executor/ctu_orchestrator | Object detection service       |
| `/generate_grasp`           | zy_interfaces/srv/GenerateGrasp  | zy_vision/grasp_generator    | zy_executor/ctu_orchestrator                            | Grasp pose generation service  |
| `/arm_grasp_command`        | zy_interfaces/msg/GraspCommand   | zy_executor/ctu_orchestrator | zy_robot/arm_controller                                 | Arm grasp pose command         |
| `/gripper_control`          | zy_interfaces/srv/GripperControl | zy_robot/gripper_server      | zy_robot/arm_controller, zy_executor/ctu_orchestrator   | Gripper control service        |
| `/arm_status`               | std_msgs/String                  | zy_robot/arm_controller      | zy_executor/ctu_orchestrator                            | Robot arm status updates       |
| `/ctu/command`              | zy_interfaces/msg/CTUCommand     | zy_comm/ctu_communication    | zy_executor/ctu_orchestrator                            | CTU commands from TCP          |
| `/executor/status`          | zy_interfaces/msg/ExecutorStatus | zy_executor/ctu_orchestrator | zy_comm/ctu_communication                               | Executor state feedback to CTU |

### ROS2 Services

| Service            | Server                     | Client                                                  | Purpose                  |
  |--------------------|----------------------------|---------------------------------------------------------|--------------------------|
| `/detect_objects`  | zy_vision/detection_server | zy_vision/grasp_generator, zy_executor/ctu_orchestrator | Object detection         |
| `/generate_grasp`  | zy_vision/grasp_generator  | zy_executor/ctu_orchestrator                            | Grasp pose generation    |
| `/gripper_control` | zy_robot/gripper_server    | zy_robot/arm_controller, zy_executor/ctu_orchestrator   | Gripper position control |

### TF2 Transforms

| Parent Frame | Child Frame                  | Type   | Publisher            |
|--------------|------------------------------|--------|----------------------|
| `base_link`  | `camera_color_optical_frame` | Static | static_tfs.launch.py |
| `base_link`  | `camera_depth_optical_frame` | Static | static_tfs.launch.py |
| `ee_link`    | `tcp_link`                   | Static | static_tfs.launch.py |

---

## Data Flow

  ```
  CTU Device (192.168.127.253:8899)
      ↓ (TCP binary protocol: SOF + LEN + DATA + CRC16)
  zy_comm/ctu_communication
      ├─→ Heartbeat thread (10s interval)
      ├─→ Listener thread (receives commands)
      └─→ /ctu/command (CTUCommand.msg)
          ↓
      zy_executor/ctu_orchestrator
          ↓ (subscribes CTU commands)
          ├─→ /detect_objects (service call)
          │   zy_vision/detection_server ← /camera/color/image_raw
          │                           ← /camera/color/camera_info
          │
          ├─→ /generate_grasp (service call)
          │   zy_vision/grasp_generator ← /camera/color/image_raw
          │                           ← /camera/depth/image_raw
          │                           ← /camera/color/camera_info
          │                           ← /camera/depth/camera_info
          │                           ← [call service] /detect_objects
          │
          ├─→ /arm_grasp_command (topic publish)
          │   zy_robot/arm_controller
          │       → MoveJ_Cmd()
          │       → Algo_Inverse_Kinematics()
          │       → [publish] /arm_status
          │
          ├─→ /gripper_control (service call)
          │   zy_robot/gripper_server
          │       → gripper_position()
          │
          └─→ /executor/status (topic publish)
              zy_comm/ctu_communication
                  ↓ (converts to TCP protocol)
              CTU Device (0x80: GRASP_COUNT, 0x81: GRASP_START, 0x82: GRASP_OVER)
  ```

---

## Comparison: Merged vs Split

### Scenario 1: Edge Device Deployment

**Merged (zy_vision contains camera_node)**:

- Requires: pyrealsense2 + PyTorch + MMDetection + CUDA
- Memory: ~2GB RAM + 1GB GPU
- Startup time: ~8s
- Can't deploy: Too heavy for edge devices

**Split (zy_camera + zy_vision)**:

- zy_camera requires: pyrealsense2 only
- Memory: ~200MB
- Startup time: ~2s
- Deployment: Possible on edge devices (Jetson Nano, etc.)

### Scenario 2: Development Workflow

**Merged**:

- Changing camera parameters: Need to rebuild vision package
- Adding new model: Need to rebuild vision package (includes camera)
- GPU debugging: Entire package restart

**Split**:

- Changing camera parameters: Rebuild only zy_camera
- Adding new model: Rebuild only zy_vision
- GPU debugging: Restart only zy_vision (camera keeps running)

### Scenario 3: Hardware Upgrades

**Merged**:

- Upgrading camera: Need to redeploy entire vision stack
- Switching inference device: Redeploy entire vision stack

**Split**:

- Upgrading camera: Redeploy only zy_camera
- Switching to edge inference: Keep zy_camera, move zy_vision

**Winner**: Split approach provides significant flexibility with only 1 additional package.

---

## Implementation Checklist

### Phase 1: Create New Packages

- [ ] Create zy_camera package structure
- [ ] Create zy_comm package structure
- [ ] Setup package.xml and setup.py for new packages

### Phase 2: Camera Node Development

- [ ] Extract camera_node.py from zy_vision
- [ ] Add CameraInfo publishing to camera_node.py
- [ ] Add intrinsic parameters (fx, fy, cx, cy)
- [ ] Create camera_params.yaml
- [ ] Create camera_bringup.launch.py

### Phase 3: Vision Node Refactoring

- [ ] Remove camera_node.py from zy_vision
- [ ] Add embedded utility functions to grasp_generator.py:
    - [ ] in_paint()
    - [ ] letterbox()
    - [ ] scale_coords()
- [ ] Update detection_server.py (no changes needed)
- [ ] Create inference_params.yaml

### Phase 4: Communication Node Development

- [ ] Create ctu_orchestrator.py (merging ctu_conn.py + orchestration logic)
- [ ] Migrate ctu_protocol.py to protocols/
- [ ] Add service clients: /detect_objects, /generate_grasp, /gripper_control
- [ ] Add topic publisher: /arm_grasp_command
- [ ] Add TCP server for CTU communication
- [ ] Implement state management (grasp_running, grasp_count, inverse_failures)
- [ ] Implement retry logic (max_attempts=2, max_inverse_failures=3)
- [ ] Embed GOOGS_MAPPING
- [ ] Create comm_params.yaml
- [ ] Create comm_bringup.launch.py

### Phase 5: Arm Controller Enhancement

- [ ] Add subscription to /arm_grasp_command topic
- [ ] Implement full grasp sequence (mid → grasp → place)
- [ ] Implement collision recovery in grasp sequence
- [ ] Add feedback publishing (/arm_status)

### Phase 6: System Launch Files

- [ ] Create static_tfs.launch.py
- [ ] Create camera_calib.launch.py
- [ ] Create grasp_system.launch.py
- [ ] Update zy_robot, zy_vision bringup launches

### Phase 7: Testing

- [ ] Test zy_camera standalone (images + CameraInfo)
- [ ] Test zy_vision standalone (detection + grasp)
- [ ] Test TF publishing (ros2 run tf2_ros static_transform_publisher)
- [ ] Test full system (grasp_system.launch.py)
- [ ] Verify all topic/service connections
- [ ] Verify parameter loading from YAML

---

## Migration Strategy

### From Original Python System

| Component                             | To Package                 | Notes                                        |
 |---------------------------------------|----------------------------|----------------------------------------------|
| `camera.py` (RS class)                | zy_camera                  | Add CameraInfo publishing                    |
| `ctu_conn.py`                         | zy_comm                    | Merge with orchestration logic               |
| `ctu_protocol.py`                     | zy_comm protocols/         | Keep protocol logic                          |
| `gripper_zhiyuan.py`                  | zy_robot gripper_server    | Add service interface                        |
| `RoboticArm.py`                       | zy_robot arm_controller    | Add ROS2 topics, add grasp sequence          |
| `grasp_zy_zhiyuan1215.py` (obj_grasp) | zy_robot arm_controller    | Implement grasp sequence in arm_controller   |
| `grasp_zy_zhiyuan1215.py` (workflow)  | zy_comm/ctu_orchestrator   | Implement orchestration (count, loop, retry) |
| `MMDetection`                         | zy_vision detection_server | Service interface                            |
| `AugmentCNN`                          | zy_vision grasp_generator  | Service interface                            |
| `utils/utils.py` (subset)             | zy_vision grasp_generator  | Embed in node                                |

### From Existing ROS2

| Component                   | Action                            |
|-----------------------------|-----------------------------------|
| grasp_vision/camera_node.py | Move to zy_camera                 |
| grasp_vision package        | Remove camera_node, update config |
| All packages                | Update package.xml dependencies   |
| All packages                | Update import paths               |

---

## Configuration Strategy

### Type 1: Runtime Parameters (YAML)

**Location**: `zy_bringup/config/*.yaml`
**Examples**:

- Camera resolution, publish rate
- Robot IP, port, speed
- Model paths, device selection
- Thresholds (NMS, confidence)

**Access Pattern**:

```python
# Node declares and loads
self.declare_parameter('robot_ip', '192.168.127.101')
robot_ip = self.get_parameter('robot_ip').value

# YAML passed in launch file
parameters = ['config/robot_params.yaml']
```

### Type 2: Calibration Data (TF2)

**Location**: `zy_bringup/launch/static_tfs.launch.py`
**Examples**:

- Tcam2base transformation matrix
- TCP compensation offset

**Access Pattern**:

```python
# Other nodes query TF2
from tf2_ros import TransformListener

tf_buffer = tf2_ros.Buffer()
transform = tf_buffer.lookup_transform('camera_color_optical_frame', 'base_link', rospy.Time())
```

### Type 3: Embedded Constants (Python)

**Location**: Node code (zy_comm)
**Examples**:

- Compensation angles (EDGE_ANGLE, SLOPE_ANGLE)
- TCP offset (TCP_COMPENSATE)
- Goods mapping (GOODS_MAPPING)
- Crop regions (CROP_LEFT, CROP_RIGHT, etc.)

**Rationale**: Simple constants, not complex calibration matrices requiring type safety or separate config files.

---

## Dependencies

### Package Dependencies

| Package       | zy_interfaces | zy_camera         | zy_vision                  | zy_robot       | zy_comm           |
 |---------------|---------------|-------------------|----------------------------|----------------|-------------------|
| zy_interfaces | -             | msg               | msg                        | msg            | msg               |
| zy_camera     | -             | -                 | -                          | -              | Image, CameraInfo |
| zy_vision     | -             | Image, CameraInfo | -                          | -              | -                 |
| zy_robot      | -             | -                 | -                          | -              | -                 |
| zy_comm       | -             | -                 | DetectionResult, GraspPose | GripperControl | -                 |

### External Dependencies

| Dependency            | Used By              | Purpose                        |
 |-----------------------|----------------------|--------------------------------|
| `pyrealsense2`        | zy_camera            | RealSense D435 driver          |
| `sensor_msgs`         | All                  | Standard ROS2 messages         |
| `geometry_msgs`       | zy_robot             | Pose/Point messages            |
| `std_msgs`            | zy_comm, zy_robot    | Basic data types               |
| `cv_bridge`           | zy_camera, zy_vision | OpenCV-ROS2 bridge             |
| `tf2_ros`             | zy_bringup           | TF2 static transform publisher |
| `PyTorch`             | zy_vision            | Deep learning framework        |
| `MMDetection`         | zy_vision            | Object detection               |
| `robotic_arm_package` | zy_robot             | RM65 SDK                       |

---

## Advantages Summary

1. **Modular**: Each package has single, well-defined responsibility
2. **Standard-compliant**: Follows ROS2 conventions (TF2, CameraInfo, etc.)
3. **Deployable**: Can run subsets on different hardware
4. **Maintainable**: Changes isolated to relevant packages
5. **Testable**: Each package can be tested independently
6. **Extensible**: Easy to add new features (sensors, algorithms, robots)
7. **No Circular Dependencies**: No inter-package Python module dependencies

---

## Risks and Mitigations

### Risk 1: Increased Package Count

**Risk**: 6 packages (reduced from initial 7 due to consolidation)
**Mitigation**: Clear organization and bringup file simplifies launching; fewer packages than original design

### Risk 2: Launch File Complexity

**Risk**: Multiple include files in grasp_system.launch.py
**Mitigation**: Well-documented structure, package-specific sub-launch files, reduced complexity (no executor launch)

### Risk 3: Configuration Fragmentation

**Risk**: Parameters spread across 4 YAML files
**Mitigation**: Centralized in `zy_bringup/config/`, clear naming conventions (reduced from 5 files)

### Risk 4: Data Flow Complexity

**Risk**: Multiple topics/services for inter-package communication
**Mitigation**: Clear documentation, visualization with `rqt_graph`, simplified data flow (no intermediate executor
layer)

---

## Future Enhancements

1. **Add Parameter Descriptors**: Add type and range validation for parameters
2. **Lifecycle Nodes**: Convert to managed nodes for better state management
3. **Diagnostics**: Add diagnostic publishers for system health monitoring
4. **Launch File Arguments**: Add command-line overrides for common parameters
5. **Multiple Camera Support**: Structure for easy addition of depth/IR cameras
6. **Modular Inference**: Add separate packages for different detection models (YOLO, etc.)

---

## Conclusion

This architecture provides:

- **Clear separation of concerns**: Each package has a single, well-defined purpose
- **ROS2 standard compliance**: CameraInfo for intrinsics, TF2 for extrinsics
- **Deployment flexibility**: Can run camera-only on edge devices
- **Maintainability**: Independent packages with clear boundaries
- **Extensibility**: Easy to add new features without affecting existing code

**Total Packages**: 7
**Total Launch Files**: 1 main + 6 package-specific
**Total Config Files**: 4 in zy_bringup/config/

This architecture provides clear separation of concerns between communication and orchestration, enabling:

- **Independent deployment** of CTU communication and task orchestration
- **Loose coupling** via ROS2 topics/services instead of direct function calls
- **Clear testing boundaries** for communication and orchestration logic
- **Scalability** to support multiple task sources beyond CTU
- **Flexibility** to upgrade communication protocol or orchestration algorithm independently

The separated orchestration package follows ROS2 best practices and matches industrial robot architectures where
communication, orchestration, and control are distinct, independently testable components.
