#!/usr/bin/env python3
"""
测试ROS2接口：测试消息、服务定义是否正确
无需硬件，仅测试接口编译和导入
"""
import sys

def test_imports():
    """测试是否能成功导入所有接口"""
    print("=" * 60)
    print("测试 ROS2 接口导入")
    print("=" * 60)
    
    try:
        # 测试消息导入
        print("\n[1/4] 测试消息导入...")
        from grasp_interfaces.msg import GraspPose, DetectionResult
        print("✓ GraspPose 导入成功")
        print("✓ DetectionResult 导入成功")
        
        # 测试服务导入
        print("\n[2/4] 测试服务导入...")
        from grasp_interfaces.srv import DetectObjects, GenerateGrasp, GripperControl
        print("✓ DetectObjects 导入成功")
        print("✓ GenerateGrasp 导入成功")
        print("✓ GripperControl 导入成功")
        
        # 测试动作导入
        print("\n[3/4] 测试动作导入...")
        from grasp_interfaces.action import ExecuteGrasp
        print("✓ ExecuteGrasp 导入成功")
        
        # 测试标准消息导入
        print("\n[4/4] 测试标准消息导入...")
        from sensor_msgs.msg import Image
        from geometry_msgs.msg import Point, Vector3
        from std_msgs.msg import String, Header
        print("✓ sensor_msgs 导入成功")
        print("✓ geometry_msgs 导入成功")
        print("✓ std_msgs 导入成功")
        
        print("\n" + "=" * 60)
        print("✓✓✓ 所有接口测试通过！")
        print("=" * 60)
        return True
        
    except ImportError as e:
        print(f"\n✗✗✗ 导入失败: {e}")
        print("\n提示: 请确保已编译工作空间并source环境")
        print("  cd /home/zh/zh/grasp_zy_zhiyuan/Ros2")
        print("  ./build.sh")
        print("  source install/setup.bash")
        return False

def test_message_creation():
    """测试消息创建"""
    print("\n" + "=" * 60)
    print("测试消息创建")
    print("=" * 60)
    
    try:
        from grasp_interfaces.msg import GraspPose
        from geometry_msgs.msg import Point, Vector3
        from std_msgs.msg import Header
        
        # 创建抓取姿态消息
        grasp = GraspPose()
        grasp.header = Header()
        grasp.row = 100
        grasp.column = 200
        grasp.angle = 1.57
        grasp.width = 50.0
        grasp.height = 25.0
        grasp.position = Point(x=0.3, y=0.2, z=0.5)
        grasp.orientation = Vector3(x=0.0, y=0.0, z=1.57)
        grasp.gripper_width = 0.05
        grasp.quality = 0.95
        grasp.slope_flag = True
        
        print(f"✓ 创建 GraspPose 消息成功")
        print(f"  - 位置: ({grasp.row}, {grasp.column})")
        print(f"  - 角度: {grasp.angle:.2f}")
        print(f"  - 质量: {grasp.quality:.2f}")
        
        print("\n✓✓✓ 消息创建测试通过！")
        return True
        
    except Exception as e:
        print(f"✗✗✗ 消息创建失败: {e}")
        return False


if __name__ == '__main__':
    success = True
    
    success &= test_imports()
    success &= test_message_creation()
    
    if success:
        print("\n🎉 所有测试通过！ROS2接口工作正常")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
        sys.exit(1)

