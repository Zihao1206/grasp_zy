# Robotic Arm SDK Guide

**概述**: 基于ctypes调用本地动态链接库(`libRM_Base.so`或`RM_Base.dll`)的机械臂SDK

**核心功能**: TCP Socket通信、6/7-DOF控制、逆运动学、碰撞检测

---

## Arm类初始化

```python
from robotic_arm_package.robotic_arm import Arm

# 连接机械臂
robot = Arm(RM65, "192.168.127.101", 8080)
robot.Set_Collision_Stage(5)  # 0-8, 默认5
```

## 核心API

### 运动控制

```python
# Movej: 关节空间运动
robot.Movej_Cmd([j1,j2,j3,j4,j5,j6], speed=20, r=0, block=True)

# Movel: 笛卡尔空间直线运动
robot.Movel_Cmd([x,y,z,rx,ry,rz], speed=20, r=0, block=True)

# Movec: 圆弧运动
robot.Movec_Cmd(mid_pose, end_pose, speed=20, r=0, block=True)
```

### 状态查询

```python
joints = robot.Get_Joint()  # 当前关节角度
pose = robot.Get_Current_Arm_State()  # 当前位姿(ArmState结构体)
joint_state = robot.Get_Joint_State()  # 关节状态(JOINT_STATE结构体)
```

### 逆运动学

```python
joints, tag = robot.Algo_Inverse_Kinematics(
    current_joint,  # 当前关节
    target_pose      # 目标位姿[x,y,z,rx,ry,rz]
)
# tag: 0=成功, 非0=失败
```

### 夹爪控制

```python
from robotic_arm_package.robotic_arm import GripperZhiyuan

gripper = GripperZhiyuan(robot)
gripper.gripper_initial()  # 初始化
gripper.gripper_position(0)  # 0=闭合, 1=打开
gripper.gripper_velocity(51200)  # 51200 = 1圈/秒
```

### 系统控制

```python
robot.Move_Stop_Cmd()  # 紧急停止
robot.Clear_System_Err(True)  # 清除错误
robot.Back_Zero_Point()  # 回零点
```

---

## 关键结构体

### JOINT_STATE - 关节状态
```python
class JOINT_STATE(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float * 7),  # 温度
        ("voltage", ctypes.c_float * 7),      # 电压
        ("current", ctypes.c_float * 7),      # 电流
        ("en_state", ctypes.c_byte * 7),      # 使能
        ("err_flag", ctypes.c_uint16 * 7),    # 错误
        ("sys_err", ctypes.c_uint16),          # 系统错误
    ]
```

### Pose - 笛卡尔位姿
```python
class Pose(ctypes.Structure):
    _fields_ = [
        ("position", Pos),
        ("quaternion", Quat),
        ("euler", Euler),
    ]
```

---

## 机械臂型号

| 常量 | 值 | 说明 |
|-------|-----|------|
| RM65 | 65 | RM65机械臂(6-DOF) |
| RM75 | 75 | RM75机械臂 |
| ECO65 | 651 | ECO65型 |

---

## 错误码（关键）

| 错误码 | 说明 |
|--------|------|
| 4 | INIT_SOCKET_ERR |
| 5 | SOCKET_CONNECT_ERR |
| 14 | ARM_ABNORMAL_STOP |
| 21 | CALCULATION_FAILED |
| 23 | FORCE_AUTO_STOP |

---

## 碰撞检测级别

| 级别 | 灵敏度 | 用途 |
|------|--------|------|
| 0-1 | 不灵敏 | 调试 |
| 4-5 | 中 | 正常(默认) |
| 6-8 | 高 | 安全模式 |

---

## 重要注意事项

1. **速度限制**: 0-50级，**不是百分比**
2. **TCP_NODELAY**: 已禁用Nagle算法降低延迟
3. **阻塞调用**: Movej/Movel默认block=True
4. **逆运动学**: 可能失败，需检查返回tag
5. **碰撞恢复**: 碰撞后需Clear_System_Err(True)
6. **回零点**: 首次使用或停机后建议Back_Zero_Point()
7. **线程安全**: SDK调用**不是线程安全**，ROS2需加互斥锁

---

## ROS2集成示例

```python
class ArmControllerNode(Node):
    def __init__(self):
        super().__init__('arm_controller')
        self.robot = Arm(RM65, robot_ip, robot_port)
        self.mutex = threading.Lock()

    def move_to_pose(self, pose):
        with self.mutex:
            tag, actual_pose = self.robot.Movel_Cmd(pose, speed=20, block=True)
```

---

## 调试工具

```bash
python RoboticArm.py  # 关节调试GUI
python RoboticGripper.py  # 夹爪调试GUI
```
