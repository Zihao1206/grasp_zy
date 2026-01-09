import logging
import socket
import threading
import time
import traceback
from grasp_zy_test import Grasp
from ctu_protocol import CmdID, CTUProtocol, Command

GoogsMapping = {
    ## 肥皂
    "1": "soap",
    # 空气开关（白色）
    "2": "interrupter",
    # 接线端子（黑色） 
    "3": "terminal",
    # 限位开关 
    "4": "limit",
    # 电压采集模块
    "5": "voltage"
}


class CTUConn(object):
    def __init__(self, grasp, server_ip='192.168.127.253', port=8899):
        self.grasp = grasp
        self.server_ip = server_ip
        self.port = port
        self.sock = None
        self.reconnect_interval = 1
        self.max_interval = 30
        self.running = False
        # 是否在抓取中
        self.grasp_running_flag = False
        logging.basicConfig(format='%(asctime)s [CLIENT] %(message)s', level=logging.INFO)

    def connect(self):
        while not self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 设置 TCP_NODELAY 选项来关闭 Nagle 算法
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.connect((self.server_ip, self.port))
                self.running = True
                self.start_heartbeat()
                self.start_listener()
                logging.info("连接成功")
                self.reconnect_interval = 1
            except Exception as e:
                logging.warning(f"连接失败: {str(e)}，{self.reconnect_interval}s后重试")
                time.sleep(self.reconnect_interval)
                self.reconnect_interval = min(self.reconnect_interval * 2, self.max_interval)

    def start_heartbeat(self):
        def loop():
            while self.running:
                try:
                    frame = CTUProtocol.build_frame([CTUProtocol.build_segment(CmdID.HEARTBEAT)])
                    self.safe_send(frame)
                    time.sleep(10)
                except Exception as e:
                    logging.warning(f"连接失败: {str(e)}，{self.reconnect_interval}s后重试")
                    self.reconnect()

        threading.Thread(target=loop, daemon=True).start()

    def start_listener(self):
        def loop():
            while self.running:
                logging.info("start_listener_loop_running")
                try:
                    buffer = self.sock.recv(64)
                    logging.info("接收到数据")
                    if not buffer:
                        break
                    self.process_data(buffer)
                except Exception as e:
                    logging.error(f"接收异常: {str(e)}")
                    traceback.print_exc()

        threading.Thread(target=loop, daemon=True).start()

    def process_data(self, data: bytes):
        try:
            cmd_list = CTUProtocol.decode_frame(data)
            for cmd in cmd_list:
                if cmd.cmdId == CmdID.CTU_GRASP_START and self.grasp_running_flag == False:
                    # 开始分拣
                    self.go_grasp(cmd)
                elif cmd.cmdId == CmdID.CTU_GRASP_SPEED:
                    # 机械臂调速(0-100)
                    self.grasp.change_robot_speed(cmd.data)
        except Exception as e:
            logging.error(f"解析异常: {str(e)}")
            traceback.print_exc()


    def go_grasp(self, cmd):
        try:
            # 设置运行中标识
            self.grasp_running_flag = True
            label = GoogsMapping[str(cmd.data)]
            logging.info(f"开始抓取物品流程: {label}")
            # 初始化夹爪
            grasp.init_gripper()
            # 统计目标物品数量
            count = grasp.detect_obj(label)
            logging.info(f"待抓取物品[{label}]数量: {count}")
            # 发送待分拣物品数量
            self.safe_send(CTUProtocol.build_frame([CTUProtocol.build_segment_uint8_data(CmdID.GRASP_COUNT,count)]))
            # 开始抓取
            self.safe_send(CTUProtocol.build_frame([CTUProtocol.build_segment(CmdID.GRASP_START)]))
            logging.info(f"开始循环抓取物品: {label}")
            # 最大抓取次数（物品数 + 3）
            grasp_max_count = count + 5
            # 当前抓取统计
            grasp_count = 0
            # 最大逆解失败数
            grasp_inverse_max_count = 3
            # 逆解失败计数
            grasp_inverse_count = 0
            while grasp.detect_obj(label) > 0 and grasp_count < grasp_max_count and grasp_inverse_count < grasp_inverse_max_count:
                grasp_count = grasp_count + 1
                grasp_result = grasp.obj_grasp(label)
                if grasp_result == False:
                    grasp_inverse_count = grasp_inverse_count + 1
                logging.info(f"待抓取物品:{label}，数量:{count}，第[{grasp_count}]次，抓取结果: {grasp_result}，逆解失败数: {grasp_inverse_count}")
            # 抓取结束
            self.safe_send(CTUProtocol.build_frame([CTUProtocol.build_segment(CmdID.GRASP_OVER)]))
            # 退出运行中标识
            self.grasp_running_flag = False

        except Exception as e:
                logging.error(f"抓取异常: {str(e)}")
                traceback.print_exc()
                # 退出运行中标识
                self.grasp_running_flag = False


    def safe_send(self, data: bytes):
        try:
            time.sleep(0.2)
            self.sock.sendall(data)
            logging.info(f"发送数据: {data.hex()}")
        except Exception as e:
            print("发送异常：", str(e))
            self.reconnect()

    def reconnect(self):
        self.running = False
        if self.sock:
            self.sock.close()
        self.connect()


if __name__ == '__main__':

    # 分拣程序
    grasp = Grasp(hardware=True)

    # 连接到CTU
    client = CTUConn(grasp, server_ip='192.168.127.253', port=8899)
    client.connect()

    # 通知CTU启动完成
    client.safe_send(CTUProtocol.build_frame([CTUProtocol.build_segment(CmdID.GRASP_OK)]))

    try:
        while True:
            # client.go_grasp(Command(CmdID.CTU_GRASP_START, 1))
            time.sleep(10)
    except KeyboardInterrupt:
        client.running = False
