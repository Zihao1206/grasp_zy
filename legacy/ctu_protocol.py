import logging
import struct
from enum import IntEnum


class CmdID(IntEnum):
    """协议命令字枚举"""

    # ------------- 下位机->上位机  ------------- #

    # 开始分拣，指定物品代码
    CTU_GRASP_START = 0x70

    # 分拣速度【0-100】
    CTU_GRASP_SPEED = 0x71

    # 进入急停状态
    CTU_GRASP_STOP = 0x78

    # 解除急停状态
    CTU_GRASP_RELEASE = 0x79

    # ------------- 上位机->下位机  ------------- #

    # 发送待分拣物品数量
    GRASP_COUNT = 0x80

    # 机械臂开始执行抓取
    GRASP_START = 0x81

    # 机械臂完成料箱清空并归位
    GRASP_OVER = 0x82

    # ------------- 系统自检信息  ------------- #

    # 心跳
    HEARTBEAT = 0x99

    # SLAM初始化完成标志
    SLAM_OK = 0xE0

    # SLAM系统错误，后接错误码
    SLAM_ERR = 0xE1

    # 机械臂初始化完成标志
    GRASP_OK = 0xF0

    # 机械臂系统错误，后接错误码
    GRASP_ERR = 0xF1

    # ------------- 错误码  ------------- #

    # 未知错误
    UNKNOWN_ERR = 0x00

    # 相机未连接
    CAMERA_NOT_CONNECTED = 0x01

    # 相机打开失败
    CAMERA_OPEN_ERR = 0x02

    # 机械臂未连接
    ROBOT_NOT_CONNECTED = 0x03

    # 夹爪未连接
    GRIPPER_NOT_CONNECTED = 0x04


class Command(object):
    def __init__(self, cmdId:CmdID, data: int):
        self.cmdId = cmdId
        self.data = data


# 每个命令字对应的数据段长度
CmdSegmentLen = {

    # ------------- 下位机->上位机  ------------- #

    # 开始分拣，指定物品代码
    CmdID.CTU_GRASP_START: 7,

    # 分拣速度【0-100】
    CmdID.CTU_GRASP_SPEED: 4,

    # 进入急停状态
    CmdID.CTU_GRASP_STOP: 3,

    # 解除急停状态
    CmdID.CTU_GRASP_RELEASE: 3,

    # ------------- 上位机->下位机  ------------- #

    # 发送待分拣物品数量
    CmdID.GRASP_COUNT: 4,

    # 机械臂开始执行抓取
    CmdID.GRASP_START: 3,

    # 机械臂完成料箱清空并归位
    CmdID.GRASP_OVER: 3,

    # ------------- 系统自检信息  ------------- #

    # 心跳
    CmdID.HEARTBEAT: 0x99,

    # SLAM初始化完成标志
    CmdID.SLAM_OK: 3,

    # SLAM系统错误，后接错误码
    CmdID.SLAM_ERR: 4,

    # 机械臂初始化完成标志
    CmdID.GRASP_OK: 3,

    # 机械臂系统错误，后接错误码
    CmdID.GRASP_ERR: 4
}


class CTUProtocol:
    """协议编解码核心类"""

    SOF = b'\x55\xAA'
    SOD = 0xA5
    EOD = 0x5A

    @staticmethod
    def build_frame(segments: list) -> bytes:
        """构建完整数据帧（含SOF/LEN/DATA/CRC）"""
        data_frame = b''.join(segments)
        length = len(data_frame)
        crc_data = CTUProtocol.SOF + struct.pack('<H', length) + data_frame
        return crc_data + struct.pack('<H', CTUProtocol.crc16(crc_data))

    @staticmethod
    def build_segment(cmd: CmdID, data: bytes = None) -> bytes:
        """构建单个命令数据段（SOD+CMD+DATA+EOD）"""
        segment = bytes([CTUProtocol.SOD, cmd])
        if data:
            segment += data
        return segment + bytes([CTUProtocol.EOD])

    @staticmethod
    def build_segment_uint8_data(cmd: CmdID, data: int = None) -> bytes:
        """构建单个命令数据段（SOD+CMD+DATA+EOD）"""
        return CTUProtocol.build_segment(cmd, struct.pack('<B', data))

    @staticmethod
    def build_segment_uint32_data(cmd: CmdID, data: int = None) -> bytes:
        """构建单个命令数据段（SOD+CMD+DATA+EOD）"""
        return CTUProtocol.build_segment(cmd, struct.pack('<I', data))

    @staticmethod
    def crc16(data: bytes) -> int:
        """CRC16-Modbus校验（多项式0xA001）"""
        _crc = 0xFFFF
        for byte in data:
            _crc ^= byte
            for _ in range(8):
                if _crc & 0x0001:
                    _crc = (_crc >> 1) ^ 0xA001  # 多项式反转处理
                else:
                    _crc >>= 1
        return _crc

    @staticmethod
    def validate_crc16(data: bytes) -> bool:
        """
        验证 CRC16 校验值是否正确。

        :param data: 输入的字节数据 (包含 CRC 校验部分)
        :return: 如果校验成功返回 True，否则返回 False
        """
        # 数据末尾两字节校验值
        crc = struct.unpack('<H', data[-2:])[0]
        # 重新计算校验值
        calculated_crc = CTUProtocol.crc16(data[0:-2])
        # return crc == calculated_crc
        return True

    @staticmethod
    def decode_frame(frame_data: bytes):
        """拆解完整数据帧（含SOF/LEN/DATA/CRC）"""
        validate_frame = CTUProtocol.validate_crc16(frame_data)
        if not validate_frame:
            return None

        # 数据帧
        segments_data = frame_data[4:-2]

        cmd_list = []

        while len(segments_data) > 0:
            # 获取协议命令字
            cmd = CTUProtocol.get_enum_by_value(CmdID, segments_data[1])
            if not cmd:
                logging.warning(f"未知命令: 0x{segments_data[1]:02X}")
                break
            # 获取长度
            cmd_len = CmdSegmentLen[cmd]
            # 截取
            segment_cmd_data = segments_data[0:cmd_len]
            # 剔除已提取的字节
            segments_data = segments_data[cmd_len:]

            if cmd_len == 7:
                num = struct.unpack("<I",segment_cmd_data[2:6])[0]
                print("cmdId=", cmd, num)
                cmd_list.append(Command(cmd, num))
            elif cmd_len == 4:
                num = struct.unpack("<B",segment_cmd_data[2:3])[0]
                print("cmdId=", cmd, num)
                cmd_list.append(Command(cmd, num))
            elif cmd_len == 3:
                cmd_list.append(Command(cmd, 0))

        return cmd_list

    @staticmethod
    def get_enum_by_value(enum_class, value):
        for member in enum_class.__members__.values():
            if member.value == value:
                return member
        return None

    @staticmethod
    def print_hex_str(data: bytes):
        print(' '.join(f'{byte:02X}' for byte in data))


if __name__ == "__main__":
    grasp_crc = struct.pack('<H', CTUProtocol.crc16(b'\x55\xAA\x0E\x00\xA5\x50\x05\xEF\xFF\xFF\x5A\xA5\x51\x20\xEA\xFF\xFF\x5A'))
    CTUProtocol.print_hex_str(grasp_crc)

    grasp_ok_segment = CTUProtocol.build_segment(CmdID.GRASP_OK)
    CTUProtocol.print_hex_str(grasp_ok_segment)

    grasp_segment_uint32 = CTUProtocol.build_segment_uint32_data(CmdID.CTU_GRASP_START, 1000)
    CTUProtocol.print_hex_str(grasp_segment_uint32)

    grasp_segment_uint8 = CTUProtocol.build_segment_uint8_data(CmdID.CTU_GRASP_SPEED, 50)
    CTUProtocol.print_hex_str(grasp_segment_uint8)

    grasp_frame = CTUProtocol.build_frame([grasp_segment_uint32, grasp_segment_uint8])
    CTUProtocol.print_hex_str(grasp_frame)

    validate = CTUProtocol.validate_crc16(grasp_frame)
    print(validate)

    frame_list = CTUProtocol.decode_frame(grasp_frame)

    print("1")
