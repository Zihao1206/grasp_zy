
class GripperZhiyuan():
    def __init__(self, robot):
        self.robot = robot
        # self.gripper_initial()

    def gripper_initial(self):  # 上电
        self.robot.Set_Tool_Voltage(3) # Set Voltage
        self.robot.Set_Modbus_Mode(1, 115200, 1) # Set Modbus Mode
        self.robot.Write_Registers(port=1, device=0x01, address=0x0A, num=6, single_data=[
            0x00,0x00,
            0x00,0xFF,
            0x00,0xFF,
            0x00,0xFF,
            0x00,0xFF,
            0x00,0x01,
        ])
        return
     
    def Motor_Close(self):
        self.robot.Set_Tool_Voltage(0)
        self.robot.Close_Modbus_Mode(1, False)

    def gripper_velocity(self, v):  # 设置速度，1-9
        # v = v * 51200
        # v_1 = (v >> 24) & 0b11111111
        # v_2 = (v >> 16) & 0b11111111
        # v_3 = (v >> 8) & 0b11111111
        # v_4 = v & 0b11111111
        # self.robot.Write_Registers(port=1, device=0x01, address=0x0A, num=0x0002, single_data=[v_1, v_2, v_3, v_4])
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

        self.robot.Write_Registers(port=1, device=0x01, address=0x0A, num=6, single_data=[
            0x00, position,
            0x00,0xFF,
            0x00,0xFF,
            0x00,0xFF,
            0x00,0xFF,
            0x00,0x01,
        ])
        return


if __name__ == '__main__':
    import time
    g = Gripper()
    g.gripper_initial()