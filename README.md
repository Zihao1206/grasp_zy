# GraspZY - Visual Robot Grasping System

> **Project is migrating to ROS2 architecture**

Vision-based robot grasping system for Jetson Orin NX deployment. CTU sorting task with Intel RealSense D435 camera, PyTorch/MMDetection detection, and RM65 robot arm control.

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| `legacy/` | Maintenance | Original monolithic Python implementation (production) |
| ROS2 packages | Development | Modular architecture with MoveIt2 integration |

## Directory Structure

```
grasp_zy/                          # Git repository root
├── legacy/                        # Production code (Python)
│   ├── grasp_zy_zhiyuan1215.py   # Main grasping pipeline
│   ├── ctu_conn.py               # CTU TCP communication
│   ├── camera.py                 # RealSense wrapper
│   ├── robotic_arm_package/      # RM65 SDK
│   └── ...
│
├── zy_interfaces/                # ROS2: Custom messages/services
├── zy_camera/                    # ROS2: Camera driver wrapper
├── zy_vision/                    # ROS2: Detection + grasp generation
├── zy_robot/                     # ROS2: Robot arm + gripper control
├── zy_comm/                      # ROS2: CTU communication
├── zy_executor/                  # ROS2: Task orchestration
├── zy_bringup/                   # ROS2: Launch configurations
├── grasp_zy/                     # ROS2: Metapackage
│
├── grasp_zy.repos                # External dependencies (vcstool)
├── setup_workspace.sh            # Workspace initialization script
├── AGENTS.md                     # Detailed documentation
└── README.md                     # This file
```

---

## Legacy System (Production)

### Run Legacy Code

```bash
conda activate zy_torch

# Main grasping program
python legacy/grasp_zy_zhiyuan1215.py

# CTU communication service
python legacy/ctu_conn.py

# Debug tools
python legacy/robotic_arm_package/RoboticArm.py
python legacy/camera.py
```

### Legacy Tech Stack

Python 3.x | PyTorch + MMDetection | pyrealsense2 | RM65 SDK | Ubuntu 20.04 | Jetson Orin NX 8GB

---

## ROS2 System (Development)

### Package Responsibilities

| Package | Responsibility | Legacy Source |
|---------|---------------|---------------|
| `zy_interfaces` | Custom msg/srv/action definitions | - |
| `zy_camera` | RealSense camera driver | camera.py |
| `zy_vision` | Object detection + grasp planning | grasp_zy_zhiyuan1215.py (detection) |
| `zy_robot` | Robot arm + gripper control | robotic_arm.py, gripper_zhiyuan.py |
| `zy_comm` | CTU TCP communication | ctu_conn.py, ctu_protocol.py |
| `zy_executor` | Task orchestration state machine | grasp_zy_zhiyuan1215.py (workflow) |
| `zy_bringup` | System launch configurations | - |

### Quick Start

```bash
# 1. Create workspace
mkdir -p ~/grasp_zy_ws/src
cd ~/grasp_zy_ws/src

# 2. Clone repository (or use symbolic link for local development)
git clone <repo-url> .

# 3. Import external dependencies
vcs import . < grasp_zy.repos

# 4. Install system dependencies
rosdep install --from-paths . --ignore-src -y --rosdistro humble

# 5. Build
cd ..
colcon build --symlink-install

# 6. Source and run
source install/setup.bash
ros2 launch zy_bringup grasp_system.launch.py
```

### External Dependencies (grasp_zy.repos)

| Dependency | Version | Description |
|------------|---------|-------------|
| ros2_rm_robot | humble | Official RM65/RM75 ROS2 driver |
| realsense-ros | 4.55.1 | Intel RealSense ROS2 driver |

### Data Flow Architecture

```
CTU Device (192.168.127.253:8899)
    |
    | TCP binary protocol
    v
zy_comm/ctu_communication
    |
    | Publishes: /ctu/command
    v
zy_executor/ctu_orchestrator
    |
    | Coordinates service calls
    +--> /detect_objects (zy_vision)
    +--> /generate_grasp (zy_vision)
    +--> /gripper_control (zy_robot)
    +--> /arm_grasp_command (zy_robot)
          |
          v
    rm_driver/rm_control/rm_moveit2_config
          |
          | Official driver stack
          v
    RM65 Robot Arm
```

---

## Hardware Configuration

| Device | IP Address | Port |
|--------|------------|------|
| CTU | 192.168.127.253 | 8899 |
| RM65 Robot Arm | 192.168.127.101 | 8080 |
| Jetson (wired) | 192.168.127.102 | - |
| Jetson (wireless) | 192.168.2.51 | - |

## Deployment Environment

- **Platform**: Jetson Orin NX 8GB
- **OS**: Ubuntu 22.04
- **ROS2**: Humble Hawksbill
- **Python**: 3.10
- **Conda env**: zy_torch

## Documentation

- [AGENTS.md](AGENTS.md) - Detailed codebase documentation
- [legacy/design.md](legacy/design.md) - ROS2 architecture design (English)
- [legacy/design.zh.md](legacy/design.zh.md) - ROS2 architecture design (Chinese)

## References

- [ros2_rm_robot](https://github.com/RealManRobot/ros2_rm_robot) - Official RM robot ROS2 driver
- [realsense-ros](https://github.com/IntelRealSense/realsense-ros) - RealSense ROS2 driver
- [MoveIt2](https://moveit.picknik.ai/humble/index.html) - MoveIt2 documentation
- [ros2_control](https://control.ros.org/humble/index.html) - ros2_control documentation
