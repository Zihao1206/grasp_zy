from Robotic_Arm.rm_robot_interface import *

robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("192.168.127.101", 8080)


rm_set_realtime_push_config = rm_realtime_push_config_t(cycle=5, enable=True, port=8090, ip="192.168.127.100")
robot.rm_set_realtime_push(rm_set_realtime_push_config)
print("机械臂ID：", handle.id)
print(robot.rm_get_realtime_push())

robot.rm_set_collision_state(5)

software_info = robot.rm_get_arm_software_info()
if software_info[0] == 0:
    print("\n================== Arm Software Information ==================")
    print("Arm Model: ", software_info[1]['product_version'])
    print("Algorithm Library Version: ", software_info[1]['algorithm_info']['version'])
    print("Control Layer Software Version: ", software_info[1]['ctrl_info']['version'])
    print("Dynamics Version: ", software_info[1]['dynamic_info']['model_version'])
    print("Planning Layer Software Version: ", software_info[1]['plan_info']['version'])
    print("==============================================================\n")
else:
    print("\nFailed to get arm software information, Error code: ", software_info[0], "\n")