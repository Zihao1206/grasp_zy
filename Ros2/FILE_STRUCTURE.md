# ROS2 抓取系统文件结构

## 完整目录树

```
Ros2/
├── README.md                       # 主文档
├── QUICKSTART.md                   # 快速开始指南
├── ARCHITECTURE.md                 # 架构说明
├── COMPARISON.md                   # 与原版本对比
├── FILE_STRUCTURE.md              # 本文件
│
├── build.sh                        # 编译脚本
├── setup_env.sh                    # 环境设置脚本
├── install_service.sh              # 服务安装脚本
├── ros2_grasp.service             # systemd服务文件
│
├── examples/                       # 示例代码
│   ├── grasp_client.py            # 抓取动作客户端示例
│   └── test_services.py           # 服务测试脚本
│
├── grasp_interfaces/              # 接口定义包
│   ├── package.xml
│   ├── CMakeLists.txt
│   ├── msg/                       # 消息定义
│   │   ├── GraspPose.msg         # 抓取姿态
│   │   └── DetectionResult.msg   # 检测结果
│   ├── srv/                       # 服务定义
│   │   ├── DetectObjects.srv     # 目标检测
│   │   ├── GenerateGrasp.srv     # 抓取姿态生成
│   │   └── GripperControl.srv    # 夹爪控制
│   └── action/                    # 动作定义
│       └── ExecuteGrasp.action   # 执行抓取
│
├── grasp_vision/                  # 视觉处理包
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   ├── resource/
│   │   └── grasp_vision
│   └── grasp_vision/
│       ├── __init__.py
│       ├── camera_node.py        # 相机节点
│       ├── detection_server.py   # 检测服务
│       └── grasp_generator.py    # 抓取姿态生成
│
├── grasp_control/                 # 控制包
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   ├── resource/
│   │   └── grasp_control
│   └── grasp_control/
│       ├── __init__.py
│       ├── arm_controller.py     # 机械臂控制器
│       └── gripper_server.py     # 夹爪服务
│
├── grasp_main/                    # 主控制包
│   ├── package.xml
│   ├── setup.py
│   ├── setup.cfg
│   ├── resource/
│   │   └── grasp_main
│   └── grasp_main/
│       ├── __init__.py
│       └── grasp_executor.py     # 抓取执行器
│
└── grasp_bringup/                 # 启动配置包
    ├── package.xml
    ├── CMakeLists.txt
    ├── launch/                    # 启动文件
    │   ├── grasp_system.launch.py        # 完整系统
    │   └── vision_only.launch.py         # 仅视觉模块
    └── config/                    # 配置文件
        └── grasp_params.yaml     # 参数配置
```

## 文件说明

### 根目录文件

| 文件 | 说明 |
|------|------|
| `README.md` | 完整的系统文档，包含安装、使用、调试等 |
| `QUICKSTART.md` | 快速开始指南，适合新手 |
| `ARCHITECTURE.md` | 详细的架构说明和数据流图 |
| `COMPARISON.md` | 与原Python脚本版本的详细对比 |
| `FILE_STRUCTURE.md` | 本文件，项目结构说明 |
| `build.sh` | 一键编译脚本 |
| `setup_env.sh` | 环境设置脚本 |
| `install_service.sh` | 安装systemd服务 |
| `ros2_grasp.service` | systemd服务配置文件 |

### grasp_interfaces 包

接口定义包，定义了系统中所有自定义的消息、服务和动作。

**消息 (msg/)**:
- `GraspPose.msg`: 抓取姿态，包含图像坐标、机器人坐标、姿态、质量等
- `DetectionResult.msg`: 目标检测结果，包含边界框、掩码、类别等

**服务 (srv/)**:
- `DetectObjects.srv`: 目标检测服务
  - Request: 彩色图像
  - Response: 检测结果
- `GenerateGrasp.srv`: 抓取姿态生成服务
  - Request: 彩色图像、深度图像、检测结果、目标标签
  - Response: 抓取姿态列表
- `GripperControl.srv`: 夹爪控制服务
  - Request: 位置 (0/1)
  - Response: 成功/失败

**动作 (action/)**:
- `ExecuteGrasp.action`: 执行完整抓取流程
  - Goal: 目标标签、是否可视化
  - Result: 成功/失败
  - Feedback: 状态、进度

### grasp_vision 包

视觉处理相关节点。

**节点**:

1. **camera_node.py**
   - 功能: 发布相机图像
   - 发布话题:
     - `/camera/color/image_raw`
     - `/camera/depth/image_raw`
   - 参数:
     - `width`: 640
     - `height`: 480
     - `publish_rate`: 10.0

2. **detection_server.py**
   - 功能: 提供目标检测服务
   - 服务: `/detect_objects`
   - 依赖: MMDetection, PyTorch
   - 参数:
     - `config_file`: MMDetection配置
     - `checkpoint`: 模型权重
     - `device`: cuda/cpu
     - `nms_score_threshold`: 0.8
     - `nms_iou_threshold`: 0.9

3. **grasp_generator.py**
   - 功能: 生成抓取姿态
   - 服务: `/generate_grasp`
   - 依赖: PyTorch, transforms3d
   - 参数:
     - `gene_file`: 网络架构文件
     - `cornell_data`: Cornell数据集配置
     - `model_weights`: 抓取模型权重
     - `device`: cuda/cpu

### grasp_control 包

机械臂和夹爪控制节点。

**节点**:

1. **arm_controller.py**
   - 功能: 机械臂控制
   - 发布话题: `/arm_status`
   - 依赖: robotic_arm_package
   - 参数:
     - `robot_ip`: 192.168.127.101
     - `robot_port`: 8080
     - `robot_speed`: 20
     - `collision_stage`: 5

2. **gripper_server.py**
   - 功能: 夹爪控制服务
   - 服务: `/gripper_control`
   - 依赖: gripper_zhiyuan
   - 参数:
     - `robot_ip`: 192.168.127.101
     - `robot_port`: 8080

### grasp_main 包

主控制逻辑。

**节点**:

1. **grasp_executor.py**
   - 功能: 协调整个抓取流程
   - 动作服务器: `/execute_grasp`
   - 订阅话题:
     - `/camera/color/image_raw`
     - `/camera/depth/image_raw`
   - 服务客户端:
     - `/detect_objects`
     - `/generate_grasp`
     - `/gripper_control`
   - 参数:
     - `robot_ip`: 192.168.127.101
     - `robot_port`: 8080
     - `robot_speed`: 20
     - `max_attempts`: 2

### grasp_bringup 包

启动文件和配置。

**启动文件**:

1. **grasp_system.launch.py**
   - 启动完整系统
   - 启动顺序:
     1. camera_node (立即)
     2. detection_server (延迟1秒)
     3. grasp_generator (延迟2秒)
     4. gripper_server (延迟3秒)
     5. grasp_executor (延迟4秒)
   - 参数:
     - `robot_ip`
     - `robot_port`
     - `robot_speed`

2. **vision_only.launch.py**
   - 仅启动视觉相关节点
   - 用于调试视觉功能

**配置文件**:

1. **grasp_params.yaml**
   - 集中管理所有参数
   - 包括机械臂、相机、检测、抓取生成等参数

### examples/ 目录

示例代码和测试脚本。

1. **grasp_client.py**
   - 抓取动作客户端示例
   - 用法: `python3 grasp_client.py carrot`

2. **test_services.py**
   - 服务测试脚本
   - 用法:
     - `python3 test_services.py gripper 0`
     - `python3 test_services.py detect image.jpg`

## 编译产物

编译后会生成以下目录：

```
build/          # 编译临时文件
install/        # 安装文件（包含可执行文件）
log/            # 编译日志
```

**install/** 目录结构:
```
install/
├── setup.bash                    # 环境设置脚本
├── local_setup.bash             # 本地环境设置
├── grasp_interfaces/            # 编译后的接口
│   ├── lib/
│   └── share/
├── grasp_vision/                # 安装的Python包
│   ├── lib/grasp_vision/
│   └── share/
├── grasp_control/
├── grasp_main/
└── grasp_bringup/
    └── share/grasp_bringup/
        ├── launch/              # 安装的启动文件
        └── config/              # 安装的配置文件
```

## 依赖关系

```
grasp_interfaces  (基础接口)
    ↑
    ├─────────────┬─────────────┐
    │             │             │
grasp_vision  grasp_control  grasp_main
    │             │             │
    └─────────────┴─────────────┘
                  ↑
            grasp_bringup (启动)
```

## 外部依赖

**系统依赖**:
- ROS2 Humble/Foxy/Galactic
- Python 3.8+
- CMake 3.8+

**ROS2包依赖**:
- `rclpy` - ROS2 Python客户端
- `std_msgs` - 标准消息
- `sensor_msgs` - 传感器消息
- `geometry_msgs` - 几何消息
- `cv_bridge` - OpenCV桥接
- `action_msgs` - 动作消息

**Python包依赖**:
- `torch` - PyTorch
- `torchvision` - PyTorch视觉
- `opencv-python` - OpenCV
- `pyrealsense2` - RealSense SDK
- `mmcv` - MMDetection基础库
- `mmdet` - MMDetection
- `transforms3d` - 3D变换
- `scikit-image` - 图像处理
- `numpy` - 数值计算

**硬件SDK**:
- `robotic_arm_package` - RM65机械臂SDK
- `gripper_zhiyuan` - 夹爪控制模块

## 关键路径

编译后的可执行文件路径：
```
install/lib/grasp_vision/camera_node
install/lib/grasp_vision/detection_server
install/lib/grasp_vision/grasp_generator
install/lib/grasp_control/gripper_server
install/lib/grasp_main/grasp_executor
```

启动文件路径：
```
install/share/grasp_bringup/launch/grasp_system.launch.py
install/share/grasp_bringup/launch/vision_only.launch.py
```

配置文件路径：
```
install/share/grasp_bringup/config/grasp_params.yaml
```

## 数据流路径

```
相机硬件
    ↓
camera_node
    ↓ [话题]
/camera/color/image_raw
/camera/depth/image_raw
    ↓
grasp_executor (订阅)
    ↓ [服务调用]
detection_server
    ↓ [服务调用]
grasp_generator
    ↓ [逆运动学]
机械臂SDK
    ↓ [服务调用]
gripper_server
    ↓
物理执行
```

## 日志路径

**ROS2日志**:
- 位置: `~/.ros/log/`
- 查看: `ros2 run rqt_console rqt_console`

**Systemd日志** (如果安装了服务):
- 查看: `sudo journalctl -u ros2_grasp.service -f`

## 开发建议

### 添加新节点

1. 确定节点所属的包
2. 在对应包的Python模块中创建节点文件
3. 在 `setup.py` 中添加入口点
4. 重新编译: `colcon build --packages-select <package_name>`

### 修改接口

1. 修改 `grasp_interfaces` 中的定义
2. 重新编译: `colcon build --packages-select grasp_interfaces`
3. 重新编译依赖此接口的包

### 调试单个节点

```bash
# 直接运行节点（需要先source环境）
ros2 run grasp_vision camera_node

# 带参数运行
ros2 run grasp_vision camera_node --ros-args -p width:=1280 -p height:=720

# 带日志级别
ros2 run grasp_vision camera_node --ros-args --log-level debug
```

### 测试工作流

1. 启动所需的节点
2. 使用 `ros2 topic echo` 查看话题数据
3. 使用 `ros2 service call` 测试服务
4. 使用 `ros2 action send_goal` 测试动作
5. 使用 `examples/` 中的脚本进行集成测试

## 版本管理建议

建议使用Git进行版本管理，`.gitignore` 应包含：

```gitignore
# ROS2编译产物
build/
install/
log/

# Python缓存
__pycache__/
*.pyc
*.pyo

# IDE
.vscode/
.idea/

# 临时文件
*.swp
*.swo
*~
```

## 备份建议

关键文件备份：
- 所有 `*.py` 源文件
- 所有 `*.msg`, `*.srv`, `*.action` 接口定义
- 所有 `package.xml`, `CMakeLists.txt`, `setup.py`
- 所有 `*.launch.py` 启动文件
- 所有 `*.yaml` 配置文件
- 文档文件 (`*.md`)

可忽略：
- `build/`, `install/`, `log/` 目录（可重新编译）
- `__pycache__/` 目录（自动生成）

