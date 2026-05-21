import logging
import socket
import threading
import time
import traceback
from grasp_zy_zhiyuanv1 import Grasp
from ctu_protocol import CmdID, CTUProtocol, Command

GoogsMapping = {
    # 白萝卜
    "1": "daikon",
    # 空气开关（白色）
    "2": "interrupter",
    # 接线端子（黑色） 
    "3": "terminal",
    # 限位开关 
    "4": "limit",
    # 电压采集模块
    "5": "voltage"
    ## 肥皂
    # "1": "soap",
}


class CTUConn(object):
    # ---- 防碰撞参数（可按现场调） ----
    # 发完 GRASP_OVER 后多少秒内拒收新的 CTU_GRASP_START
    GRASP_COOLDOWN_S = 2.0
    # 发 GRASP_OVER 前强制归位 + 等待机械臂静止的最长时间
    SETTLE_TIMEOUT_S = 5.0
    # 静止判定连续轮询次数（变化小于阈值才认定为停下）
    SETTLE_STABLE_HITS = 3
    SETTLE_POLL_INTERVAL_S = 0.05
    SETTLE_JOINT_EPS = 0.05  # 关节角变化阈值（度）

    def __init__(self, grasp, server_ip='192.168.127.253', port=8899):
        self.grasp = grasp
        self.server_ip = server_ip
        self.port = port
        self.sock = None
        self.reconnect_interval = 1
        self.max_interval = 30
        self.running = False
        self.grasp_running_flag = False
        # GRASP_OVER 之后的冷却截止时间（防止 CTU 立刻重发 START 触发再一次抓取）
        self.grasp_cooldown_until = 0.0
        self._send_lock = threading.Lock()
        logging.basicConfig(format='%(asctime)s [CLIENT] %(message)s', level=logging.INFO)

    def connect(self):
        while not self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
                if cmd.cmdId == CmdID.CTU_GRASP_START:
                    # 碰撞修复①：抓取进行中，直接丢弃后到的 START（典型场景：上一轮 go_grasp
                    # 仍在跑，TCP 缓冲区已有重发的 START）
                    if self.grasp_running_flag:
                        logging.warning("丢弃 CTU_GRASP_START：上一轮抓取仍在进行")
                        continue
                    # 碰撞修复②：刚发完 GRASP_OVER 的冷却窗内拒收，避免 CTU 重发 START
                    # 紧贴着 GRASP_OVER 又触发一轮抓取（此时小车可能正在移动）
                    remain = self.grasp_cooldown_until - time.monotonic()
                    if remain > 0:
                        logging.warning(f"丢弃 CTU_GRASP_START：处于 GRASP_OVER 冷却窗（剩余 {remain:.2f}s）")
                        continue
                    self.grasp_running_flag = True
                    threading.Thread(target=self._go_grasp_in_thread, args=(cmd,), daemon=True).start()
                elif cmd.cmdId == CmdID.CTU_GRASP_SPEED:
                    self.grasp.change_robot_speed(cmd.data)
        except Exception as e:
            logging.error(f"解析异常: {str(e)}")
            traceback.print_exc()

    def _go_grasp_in_thread(self, cmd):
        """独立线程中执行抓取，避免阻塞监听线程导致 TCP 缓冲区积压"""
        try:
            self.go_grasp(cmd)
        except Exception as e:
            logging.error(f"抓取线程异常: {str(e)}")
            traceback.print_exc()
        finally:
            # 即便异常，也要进入冷却窗（防止异常瞬间清旗再被 START 触发）
            self.grasp_cooldown_until = max(
                self.grasp_cooldown_until,
                time.monotonic() + self.GRASP_COOLDOWN_S,
            )
            self.grasp_running_flag = False

    def _settle_arm_and_send_over(self):
        """碰撞修复③：发 GRASP_OVER 前先强制归位并等待机械臂真正静止，最后再上报。"""
        # 1. 强制回 init_pose（不依赖单步 rm_movej 的 block 行为）
        try:
            robot = getattr(self.grasp, "robot", None)
            init_pose = getattr(self.grasp, "init_pose", None)
            speed = getattr(self.grasp, "robot_speed", 20)
            if robot is not None and init_pose is not None:
                try:
                    robot.rm_movej(init_pose, speed, 0, 0, 1)
                except Exception as e:
                    logging.warning(f"强制归位 rm_movej 异常: {e}")
            # 2. 轮询机械臂状态，确认已停下
            self._wait_arm_idle(robot)
        except Exception as e:
            logging.warning(f"归位/静止确认异常: {e}")

        # 3. 上报 GRASP_OVER 并开启冷却窗
        self.safe_send(CTUProtocol.build_frame([CTUProtocol.build_segment(CmdID.GRASP_OVER)]))
        self.grasp_cooldown_until = time.monotonic() + self.GRASP_COOLDOWN_S
        logging.info(f"GRASP_OVER 已上报，进入 {self.GRASP_COOLDOWN_S}s 冷却窗")

    def _wait_arm_idle(self, robot):
        """轮询关节角直到连续多次基本不变，判定为停止。SDK 不可用时退化为固定 sleep。"""
        if robot is None or not hasattr(robot, "rm_get_current_arm_state"):
            time.sleep(0.5)
            return

        deadline = time.monotonic() + self.SETTLE_TIMEOUT_S
        last_joint = None
        stable_hits = 0
        while time.monotonic() < deadline:
            try:
                ret, state = robot.rm_get_current_arm_state()
                if ret != 0 or not isinstance(state, dict):
                    time.sleep(self.SETTLE_POLL_INTERVAL_S)
                    continue
                joint = None
                for key in ("joint", "joints", "joint_angle", "joint_angles", "joint_position"):
                    val = state.get(key)
                    if val is not None:
                        try:
                            joint = [float(x) for x in val[:6]]
                        except (TypeError, ValueError):
                            joint = None
                        if joint is not None:
                            break
                if joint is None:
                    time.sleep(self.SETTLE_POLL_INTERVAL_S)
                    continue
                if last_joint is not None and all(
                    abs(a - b) < self.SETTLE_JOINT_EPS for a, b in zip(joint, last_joint)
                ):
                    stable_hits += 1
                    if stable_hits >= self.SETTLE_STABLE_HITS:
                        logging.info("机械臂已静止")
                        return
                else:
                    stable_hits = 0
                last_joint = joint
            except Exception as e:
                logging.warning(f"查询机械臂状态异常: {e}")
            time.sleep(self.SETTLE_POLL_INTERVAL_S)
        logging.warning(f"等待机械臂静止超时（>{self.SETTLE_TIMEOUT_S}s），仍按已停下处理")

    def go_grasp(self, cmd):
        sent_over = False
        try:
            label = GoogsMapping[str(cmd.data)]
            logging.info(f"开始抓取物品流程: {label}")
            self.grasp.init_gripper()
            count = self.grasp.detect_obj(label)
            logging.info(f"待抓取物品[{label}]数量: {count}")
            if count <= 0:
                logging.info("未检测到物品，归位并上报 GRASP_OVER")
                self._settle_arm_and_send_over()
                sent_over = True
                return
            self.safe_send(CTUProtocol.build_frame([CTUProtocol.build_segment_uint8_data(CmdID.GRASP_COUNT, count)]))
            self.safe_send(CTUProtocol.build_frame([CTUProtocol.build_segment(CmdID.GRASP_START)]))
            logging.info(f"开始循环抓取物品: {label}")
            grasp_max_count = count + 5
            grasp_count = 0
            grasp_inverse_max_count = 3
            grasp_inverse_count = 0
            # 碰撞修复：至少尝试初始检测到的数量，防止 detect_obj 两次结果不一致
            min_attempts = max(count, 1) if count > 0 else 0
            while grasp_count < grasp_max_count and grasp_inverse_count < grasp_inverse_max_count:
                if grasp_count >= min_attempts:
                    if self.grasp.detect_obj(label) <= 0:
                        logging.info(f"detect_obj返回0，已尝试{grasp_count}次(最少{min_attempts}次)，退出循环")
                        break
                grasp_count = grasp_count + 1
                grasp_result = self.grasp.obj_grasp(label)
                if grasp_result == False:
                    grasp_inverse_count = grasp_inverse_count + 1
                logging.info(f"待抓取物品:{label}，数量:{count}，第[{grasp_count}]次，抓取结果: {grasp_result}，逆解失败数: {grasp_inverse_count}")

            # 正常路径：归位 + 静止确认后再上报 GRASP_OVER
            self._settle_arm_and_send_over()
            sent_over = True

        except Exception as e:
            logging.error(f"抓取异常: {str(e)}")
            traceback.print_exc()
            # 碰撞修复④：异常路径也必须给 CTU 一个明确结束态，否则下游会僵死后再被超时重发
            # 触发，叠加成原来的问题。先尝试停车再归位，最后上报 GRASP_ERR（拿不到错误码就发
            # GRASP_OVER 兜底）。
            try:
                robot = getattr(self.grasp, "robot", None)
                if robot is not None and hasattr(robot, "rm_set_arm_stop"):
                    try:
                        robot.rm_set_arm_stop()
                    except Exception:
                        pass
                self._settle_arm_and_send_over_on_error()
                sent_over = True
            except Exception as e2:
                logging.error(f"异常恢复失败: {e2}")
                traceback.print_exc()
        finally:
            # 兜底：极端情况下都没发出 GRASP_OVER 时，至少进入冷却窗
            if not sent_over:
                self.grasp_cooldown_until = max(
                    self.grasp_cooldown_until,
                    time.monotonic() + self.GRASP_COOLDOWN_S,
                )

    def _settle_arm_and_send_over_on_error(self):
        """异常恢复：尝试归位 + 上报 GRASP_ERR，拿不到 ERR 段时退化为 GRASP_OVER。"""
        try:
            robot = getattr(self.grasp, "robot", None)
            init_pose = getattr(self.grasp, "init_pose", None)
            speed = getattr(self.grasp, "robot_speed", 20)
            if robot is not None and init_pose is not None:
                try:
                    robot.rm_movej(init_pose, speed, 0, 0, 1)
                except Exception as e:
                    logging.warning(f"异常恢复归位失败: {e}")
            self._wait_arm_idle(robot)
        except Exception as e:
            logging.warning(f"异常恢复期间状态查询失败: {e}")
        try:
            err_frame = CTUProtocol.build_frame([
                CTUProtocol.build_segment_uint8_data(CmdID.GRASP_ERR, int(CmdID.UNKNOWN_ERR)),
            ])
            self.safe_send(err_frame)
            logging.info("已上报 GRASP_ERR")
        except Exception as e:
            logging.warning(f"GRASP_ERR 上报失败，退化为 GRASP_OVER: {e}")
            self.safe_send(CTUProtocol.build_frame([CTUProtocol.build_segment(CmdID.GRASP_OVER)]))
        self.grasp_cooldown_until = time.monotonic() + self.GRASP_COOLDOWN_S

    def safe_send(self, data: bytes):
        try:
            time.sleep(0.2)
            with self._send_lock:
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
            client.go_grasp(Command(CmdID.CTU_GRASP_START, 1))
            time.sleep(10)
    except KeyboardInterrupt:
        client.running = False
