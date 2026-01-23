
from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t
from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t

class GripperZhiyuan():
    def __init__(self, robot):
        self.robot = robot
        # self.gripper_initial()

    def gripper_initial(self):
        self.robot.rm_set_tool_voltage(3)
        self.robot.rm_set_modbus_mode(1, 115200, 10)
        write_params = rm_peripheral_read_write_params_t(port=1, address=0x0A, device=0x01, num=6)
        self.robot.rm_write_registers(write_params, [0x00,0x00,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0x01])
        return
     
    def Motor_Close(self):
        self.robot.rm_set_tool_voltage(0)
        self.robot.rm_close_modbus_mode(1)

    def gripper_velocity(self, v):
        v_1 = (v >> 24) & 0b11111111
        v_2 = (v >> 16) & 0b11111111
        v_3 = (v >> 8) & 0b11111111
        v_4 = v & 0b11111111
        self.robot.rm_write_registers(
            rm_peripheral_read_write_params_t(port=1, address=0x0A, device=0x01, num=6),
            [v_1, v_2, v_3, v_4]
        )
        return

    def gripper_position(self, position):  # 设置夹爪的张开程度, 0为闭合，1为最大张开
        """
        控制夹爪的开合程度
        
        Args:
            position: 张开程度，范围[0, 1]
                     0 = 完全闭合
                     1 = 最大张开
        """
        # 限制参数范围在0-1之间
        if position > 1:
            position = 1
        if position < 0:
            position = 0

        # 将0-1映射到0-127的寄存器值（255*0.5）
        position = int(position * 255 * 0.5)
 
        self.robot.rm_write_registers(
            rm_peripheral_read_write_params_t(port=1, address=0x0A, device=0x01, num=6),
            [0x00, position, 0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,0x01]
        )
        return


if __name__ == '__main__':
    import time
    g = Gripper()
    g.gripper_initial()