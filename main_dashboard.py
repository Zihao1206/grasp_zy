"""机器人协同控制系统 — 上位机主界面。

本文件是一个独立的 PyQt5 主控台，整体采用 2x2 布局：
    ┌──────────────┬──────────────┐
    │   功能区     │  机械臂控制  │
    ├──────────────┼──────────────┤
    │   日志显示   │   实时图像   │
    └──────────────┴──────────────┘

运行方式：
    python main_dashboard.py
要求 Python 3.8+，依赖 PyQt5；如需相机画面接入，可调用
``MainDashboard.update_camera_frame(bgr_image)`` 把 OpenCV 的 BGR ndarray
推到右下方实时图像区。
"""
from __future__ import annotations

import os
import sys
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QImage, QPainter, QPixmap, QResizeEvent
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# 当前文件所在目录，用于定位同目录下的子界面脚本
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 关节限位（单位：度）— 严格按需求要求
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "J1": (-178.0, 178.0),
    "J2": (-130.0, 130.0),
    "J3": (-135.0, 135.0),
    "J4": (-178.0, 178.0),
    "J5": (-128.0, 128.0),
    "J6": (-360.0, 360.0),
}

# 关节加减按钮的步长
JOINT_STEP = 1.0


# ---------------------------------------------------------------------------
# 工业风格全局样式表
# ---------------------------------------------------------------------------
INDUSTRIAL_QSS = """
QMainWindow, QWidget {
    background-color: #f0f3f5;
    color: #1a1a1a;
    font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #c5ccd2;
    border-radius: 4px;
    margin-top: 14px;
    padding: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #2b5a82;
    background-color: #f0f3f5;
}
QPushButton {
    background-color: #3b78b3;
    color: white;
    border: 1px solid #2b5a82;
    border-radius: 3px;
    padding: 6px 12px;
    min-height: 24px;
}
QPushButton:hover { background-color: #4a8ac8; }
QPushButton:pressed { background-color: #2b5a82; }
QPushButton:disabled {
    background-color: #b8c0c8;
    color: #f0f3f5;
    border-color: #99a3ad;
}
QPushButton#PrimaryBig {
    font-size: 15px;
    font-weight: 600;
    min-height: 56px;
}
/* ─── 功能区瓷砖按钮 (FunctionTile) ───
 *   ▸ 想改背景色 / 字色 / 圆角 / 边框 → 改下面四组的 background-color、color 等
 *   ▸ 想改"图标到按钮顶部 + 文字到按钮底部"的留白 → 改 padding (顺序: 上 右 下 左)
 *     - padding 第 1 个值变大 → 图标距顶部更远
 *     - padding 第 3 个值变大 → 文字距底部更远
 *   ▸ 想改图标和文字之间的间距 → 见 FunctionTile 类里的 self.setStyleSheet 备注
 *   ▸ 想改文字大小 / 粗细 → 改 font-size / font-weight
 */
QToolButton#FunctionTile {
    background-color: #dde6ef;
    color: #1a3656;
    border: 1px solid #b6c2cf;
    border-radius: 6px;
    padding: 8px 6px 8px 6px;
    font-size: 14px;
    font-weight: 600;
}
QToolButton#FunctionTile:hover {
    background-color: #cfdfee;
    border-color: #3b78b3;
    color: #14304f;
}
QToolButton#FunctionTile:pressed {
    background-color: #b9d2eb;
    border-color: #2b5a82;
}
QToolButton#FunctionTile:disabled {
    background-color: #d8dde2;
    color: #8a96a3;
    border-color: #c0c8d0;
}
QPushButton#JointStep {
    background-color: #5d6d7e;
    border-color: #34495e;
    min-width: 26px;
    max-width: 26px;
    min-height: 22px;
    max-height: 22px;
    padding: 0;
    font-weight: 700;
}
QPushButton#JointStep:hover { background-color: #7a8a9b; }
QPushButton#Danger {
    background-color: #b53737;
    border-color: #7a2424;
}
QPushButton#Danger:hover { background-color: #cf4747; }

QLineEdit, QDoubleSpinBox, QPlainTextEdit {
    background-color: #ffffff;
    border: 1px solid #b8c0c8;
    border-radius: 3px;
    padding: 3px 6px;
    selection-background-color: #3b78b3;
}
QPlainTextEdit { font-family: Consolas, "Courier New", monospace; }
QLabel#FieldLabel { color: #2b3a48; font-weight: 500; }
QLabel#CameraPlaceholder {
    background-color: #232a30;
    color: #f5f5f5;
    border: 1px solid #34495e;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 2px;
}
QFrame#VLine {
    background-color: #d0d6dc;
    max-width: 1px;
    min-width: 1px;
}
"""


# ---------------------------------------------------------------------------
# 工具函数：cv2(BGR) ndarray → QPixmap
# ---------------------------------------------------------------------------
def cv2_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """把 OpenCV BGR ndarray 转换为可直接 setPixmap 的 QPixmap。

    输入若为单通道灰度图也会被自动按 BGR 处理。
    """
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
        return QPixmap()
    if bgr.ndim == 2:
        bgr = np.stack([bgr] * 3, axis=-1)
    rgb = bgr[:, :, ::-1].copy()  # BGR → RGB
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# 功能区按钮：图标在上、文字在下；图标使用 SVG 矢量渲染，HiDPI 友好
# ---------------------------------------------------------------------------
# 调整功能区按钮"图案大小 / 位置 / 间距"的全部参数都集中在这一段：
#
#   1. 图标边长（逻辑像素）。改大 → 图案更大，改小 → 图案更小。
TILE_ICON_SIZE = 56
#
#   2. 按钮整体最小高度。撑大它图案与文字会被往中间居中，整块按钮更显眼。
TILE_MIN_HEIGHT = 110
#
#   3. 图标到文字的间距（像素）。Qt 的 QToolButton 默认 spacing≈4，
#      想让图文更紧凑就调小，想拉开就调大。
TILE_ICON_TEXT_SPACING = 2
#
# 备注：图标到"按钮顶部"以及文字到"按钮底部"的留白，由 QSS 中
#       QToolButton#FunctionTile 的 padding 控制（顺序：上 右 下 左）。
#       例如想让图标更贴近顶部，就把 padding 第 1 个值改小。


def _render_svg_to_pixmap(svg_path: str, size: int) -> QPixmap:
    """把 SVG 文件渲染为 ``size×size`` 的高质量 ``QPixmap``。

    若 SVG 加载失败则返回空 ``QPixmap``，调用方应自行处理（按钮会退化为纯文字）。
    HiDPI 下 ``QApplication`` 已经设置了 ``AA_UseHighDpiPixmaps``，这里基于
    ``devicePixelRatio`` 渲染高分辨率位图，避免按钮上图标发虚。
    """
    if not os.path.isfile(svg_path):
        return QPixmap()
    renderer = QSvgRenderer(svg_path)
    if not renderer.isValid():
        return QPixmap()
    dpr = max(1.0, QApplication.primaryScreen().devicePixelRatio() if QApplication.instance() else 1.0)
    pixel = max(1, int(round(size * dpr)))
    pix = QPixmap(pixel, pixel)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    renderer.render(painter)
    painter.end()
    pix.setDevicePixelRatio(dpr)
    return pix


class FunctionTile(QToolButton):
    """主控台功能区瓷砖按钮：上图下字、间距由 TILE_ICON_TEXT_SPACING 控制。

    Qt 自带的 ``ToolButtonTextUnderIcon`` 由系统 style 决定图文间距，无法精确
    调整；这里改为 ``ToolButtonIconOnly`` + 自定义 :meth:`paintEvent`，把图标
    与文字按设定间距居中绘制，从而支持"图文紧凑"的视觉需求。
    """

    def __init__(self, text: str, svg_path: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("FunctionTile")
        # ToolButtonIconOnly 让 Qt 不自动按 "图下文" 排版；同时下方把
        # text/icon 都置空，避免 super().paintEvent() 在背景之上再画一份
        # 自带的图标/文字 → 这就是出现"重影"的原因。
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)
        super().setText("")
        self._display_text = text
        self._icon: QIcon = QIcon()
        self.setMinimumHeight(TILE_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # iconSize 即便 icon 为空也仍被一些 style 用于估算尺寸，保留它无害
        self.setIconSize(QSize(TILE_ICON_SIZE, TILE_ICON_SIZE))
        self.setFocusPolicy(Qt.StrongFocus)
        self._svg_path = svg_path
        self._refresh_icon()

    # ── 公共接口 ──
    def set_svg_icon(self, svg_path: str) -> None:
        self._svg_path = svg_path
        self._refresh_icon()

    # ── 内部 ──
    def _refresh_icon(self) -> None:
        if not self._svg_path:
            self._icon = QIcon()
        else:
            pix = _render_svg_to_pixmap(self._svg_path, TILE_ICON_SIZE)
            self._icon = QIcon(pix) if not pix.isNull() else QIcon()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        # 1) 让 QSS 把 background-color / border / :hover / :pressed 等画出来。
        #    因为 self 不持有 icon/text，super 不会重复绘制图标和文字 → 不再重影。
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 2) 计算"图标 + 间距 + 文字"整体在垂直方向上的居中位置。
        rect = self.contentsRect()
        text_to_draw = self._display_text or ""
        font_metrics = painter.fontMetrics()
        text_h = font_metrics.height() if text_to_draw else 0
        icon_h = TILE_ICON_SIZE if not self._icon.isNull() else 0
        gap = TILE_ICON_TEXT_SPACING if (icon_h and text_h) else 0
        total_h = icon_h + gap + text_h
        top = rect.top() + max(0, (rect.height() - total_h) // 2)

        # 3) 绘制图标
        if icon_h:
            icon_rect_x = rect.left() + (rect.width() - TILE_ICON_SIZE) // 2
            self._icon.paint(
                painter,
                icon_rect_x,
                top,
                TILE_ICON_SIZE,
                TILE_ICON_SIZE,
                Qt.AlignCenter,
                QIcon.Normal if self.isEnabled() else QIcon.Disabled,
            )

        # 4) 绘制文字（颜色跟随当前 palette，禁用时变灰）
        if text_h:
            text_top = top + icon_h + gap
            text_rect = rect.adjusted(0, text_top - rect.top(), 0, 0)
            painter.setPen(self.palette().buttonText().color()
                           if self.isEnabled() else self.palette().mid().color())
            painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, text_to_draw)

        painter.end()


# 资源目录：项目内 ``dataset/`` 下的 SVG 图标
ASSETS_DIR = os.path.join(PROJECT_DIR, "dataset")


# ---------------------------------------------------------------------------
# 摄像头占位组件：始终保持深灰底白字 “Camera View”，可以被外部 setPixmap 覆盖
# ---------------------------------------------------------------------------
class CameraView(QLabel):
    """实时图像占位 / 显示组件。

    - 默认显示「Camera View」白字、深灰底；
    - 调用 :meth:`update_frame` 传入 BGR ndarray 时切换为图像显示；
    - 缩放时按比例展示，不会拉伸变形。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CameraPlaceholder")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("Camera View")
        self._raw_pixmap: Optional[QPixmap] = None

    def update_frame(self, bgr_image: np.ndarray) -> None:
        """OpenCV 接入入口：把外部 BGR ndarray 渲染到此组件。"""
        pix = cv2_to_qpixmap(bgr_image)
        if pix.isNull():
            return
        self._raw_pixmap = pix
        self._render()

    def clear_frame(self) -> None:
        """清空已渲染的图像，恢复占位文字。"""
        self._raw_pixmap = None
        self.setPixmap(QPixmap())
        self.setText("Camera View")

    def _render(self) -> None:
        if self._raw_pixmap is None or self._raw_pixmap.isNull():
            return
        scaled = self._raw_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render()


# ---------------------------------------------------------------------------
# 硬件 Worker：相机初始化 / 视频流 / 机械臂连接 / 状态轮询 / 运动执行
# ---------------------------------------------------------------------------
class RealSenseCamera:
    """轻量 RealSense 封装，提供与项目其它 GUI 一致的 get_img/stop 接口。"""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        for _ in range(5):
            self._pipeline.wait_for_frames()

    def get_img(self) -> Tuple[np.ndarray, np.ndarray]:
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        depth = np.asanyarray(aligned.get_depth_frame().get_data())
        color = np.asanyarray(aligned.get_color_frame().get_data())
        return depth, color

    def stop(self) -> None:
        self._pipeline.stop()


class CameraInitWorker(QObject):
    """后台初始化 RealSense，避免启动相机时卡住主界面。"""

    init_done = pyqtSignal(bool, object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps

    def run(self) -> None:
        try:
            cam = RealSenseCamera(self.width, self.height, self.fps)
            self.init_done.emit(True, cam)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.init_done.emit(False, None)
        finally:
            self.finished.emit()


class CameraStreamWorker(QObject):
    """持续读取相机画面并发送到 UI。"""

    frame_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, camera: RealSenseCamera) -> None:
        super().__init__()
        self.camera = camera
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                _, color = self.camera.get_img()
                self.frame_ready.emit(color)
            except Exception as exc:
                self.error_occurred.emit(str(exc))
                time.sleep(0.2)
            time.sleep(0.03)
        self.finished.emit()

    def stop(self) -> None:
        self._running = False


class RobotConnectWorker(QObject):
    """连接 RealMan 机械臂并初始化夹爪。"""

    connected = pyqtSignal(bool, object, object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, ip: str = "192.168.127.101", port: int = 8080) -> None:
        super().__init__()
        self.ip = ip
        self.port = port

    def run(self) -> None:
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
            from gripper_zhiyuan import GripperZhiyuan

            robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            handle = robot.rm_create_robot_arm(self.ip, self.port)
            robot.rm_set_collision_state(5)

            gripper = GripperZhiyuan(robot)
            try:
                gripper.gripper_initial()
            except Exception as exc:
                # 夹爪初始化失败不应阻断机械臂状态查看；后续夹爪按钮会继续报具体错误。
                self.error_occurred.emit(f"夹爪初始化失败: {exc}")

            self.connected.emit(True, robot, gripper)
            self.error_occurred.emit(f"机械臂连接句柄: {handle}")
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.connected.emit(False, None, None)
        finally:
            self.finished.emit()


class RobotStateWorker(QObject):
    """轮询机械臂当前状态，用于示教器式实时显示。"""

    state_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, robot: Any, interval_s: float = 0.2) -> None:
        super().__init__()
        self.robot = robot
        self.interval_s = interval_s
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                ret, state = self.robot.rm_get_current_arm_state()
                if ret != 0:
                    self.error_occurred.emit(f"获取机械臂状态失败，错误码: {ret}")
                else:
                    self.state_ready.emit(state)
            except Exception as exc:
                self.error_occurred.emit(str(exc))
            time.sleep(self.interval_s)
        self.finished.emit()

    def stop(self) -> None:
        self._running = False


class RobotCommandWorker(QObject):
    """执行一次机械臂命令，避免运动命令阻塞 UI。"""

    command_done = pyqtSignal(bool, str)
    finished = pyqtSignal()

    def __init__(self, robot: Any, command: str, payload: object = None, speed: int = 20) -> None:
        super().__init__()
        self.robot = robot
        self.command = command
        self.payload = payload
        self.speed = speed

    def run(self) -> None:
        try:
            if self.command == "movej":
                joints = list(self.payload or [])
                ret = self.robot.rm_movej(joints, self.speed, 0, 0, 1)
                ok = ret == 0
                msg = "关节运动完成" if ok else f"关节运动失败，返回码: {ret}"
            elif self.command == "movel":
                pose = list(self.payload or [])
                ret = self.robot.rm_movel(pose, self.speed, 0, 0, 1)
                ok = ret == 0
                msg = "位姿直线运动完成" if ok else f"位姿直线运动失败，返回码: {ret}"
            elif self.command == "stop":
                ret = self.robot.rm_set_arm_stop()
                ok = ret == 0
                msg = "停止命令已发送" if ok else f"停止失败，返回码: {ret}"
            else:
                ok = False
                msg = f"未知命令: {self.command}"
            self.command_done.emit(ok, msg)
        except Exception as exc:
            self.command_done.emit(False, str(exc))
        finally:
            self.finished.emit()


class GripperWorker(QObject):
    """夹爪开合命令。"""

    command_done = pyqtSignal(bool, str)
    finished = pyqtSignal()

    def __init__(self, gripper: Any, position: float) -> None:
        super().__init__()
        self.gripper = gripper
        self.position = position

    def run(self) -> None:
        try:
            self.gripper.gripper_position(self.position)
            text = "夹爪打开完成" if self.position > 0 else "夹爪关闭完成"
            self.command_done.emit(True, text)
        except Exception as exc:
            self.command_done.emit(False, str(exc))
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# 关节单元（标签 + 数值框 + “-” / “+” 按钮，带限位）
# ---------------------------------------------------------------------------
class JointRow(QWidget):
    """单关节控制行：标签 / 数值框 / 加减按钮。

    - 数值框可手动编辑（也会被范围裁剪）；
    - 加减按钮按 :data:`JOINT_STEP` 步长调整；
    - 触及上下限时按钮自动禁用，防止越界；
    - 任意数值变化通过 :attr:`value_changed` 信号对外暴露：(name, new_value)。
    """

    value_changed = pyqtSignal(str, float)

    def __init__(self, name: str, low: float, high: float, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._name = name
        self._low = float(low)
        self._high = float(high)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(f"{name}:")
        lbl.setObjectName("FieldLabel")
        lbl.setMinimumWidth(28)
        layout.addWidget(lbl)

        # 用 QDoubleSpinBox 但隐藏上下箭头，避免与自定义 +/- 按钮冲突
        self._spin = QDoubleSpinBox()
        self._spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self._spin.setDecimals(2)
        self._spin.setRange(self._low, self._high)
        self._spin.setSingleStep(JOINT_STEP)
        self._spin.setValue(0.0)
        self._spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._spin.setMinimumWidth(80)
        self._spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._spin.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self._spin, stretch=1)

        self._btn_minus = QPushButton("−")
        self._btn_minus.setObjectName("JointStep")
        self._btn_minus.clicked.connect(lambda: self._step(-JOINT_STEP))
        layout.addWidget(self._btn_minus)

        self._btn_plus = QPushButton("+")
        self._btn_plus.setObjectName("JointStep")
        self._btn_plus.clicked.connect(lambda: self._step(+JOINT_STEP))
        layout.addWidget(self._btn_plus)

        # 范围标签：让用户一眼看清允许区间
        rng_lbl = QLabel(f"[{int(self._low)}, {int(self._high)}]")
        rng_lbl.setStyleSheet("color: #6b7785; font-size: 11px;")
        layout.addWidget(rng_lbl)

        self._refresh_button_state()

    # ── 内部槽 ──
    def _on_value_changed(self, val: float) -> None:
        self._refresh_button_state()
        self.value_changed.emit(self._name, float(val))

    def _step(self, delta: float) -> None:
        new_val = self._spin.value() + delta
        new_val = max(self._low, min(self._high, new_val))
        if new_val != self._spin.value():
            self._spin.setValue(new_val)  # 会触发 valueChanged

    def _refresh_button_state(self) -> None:
        v = self._spin.value()
        # 留 1e-6 容差，浮点边界更稳
        self._btn_minus.setEnabled(v - JOINT_STEP >= self._low - 1e-6)
        self._btn_plus.setEnabled(v + JOINT_STEP <= self._high + 1e-6)

    # ── 公共接口 ──
    def value(self) -> float:
        return float(self._spin.value())

    def set_value(self, v: float) -> None:
        v = max(self._low, min(self._high, float(v)))
        self._spin.blockSignals(True)
        self._spin.setValue(v)
        self._spin.blockSignals(False)
        self._refresh_button_state()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class MainDashboard(QMainWindow):
    """机器人协同控制系统主界面。"""

    # 暴露给外部用于传入相机帧的便捷信号（线程安全：可跨线程 emit）
    camera_frame_received = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("机器人协同控制系统")
        self.resize(1280, 800)

        # 子进程句柄：避免重复启动外部 GUI
        self._sub_procs: Dict[str, subprocess.Popen] = {}
        self._robot: Optional[Any] = None
        self._gripper: Optional[Any] = None
        self._camera: Optional[RealSenseCamera] = None
        self._robot_connected = False
        self._target_mode = "joint"  # joint / pose：用户最后一次编辑的目标类型
        self._manual_override = False  # 用户编辑后暂停用实时状态覆盖目标输入
        self._updating_from_robot = False
        self._last_state_log_ts = 0.0

        self._robot_connect_thread: Optional[QThread] = None
        self._robot_state_thread: Optional[QThread] = None
        self._robot_state_worker: Optional[RobotStateWorker] = None
        self._cmd_thread: Optional[QThread] = None
        self._gripper_thread: Optional[QThread] = None
        self._camera_init_thread: Optional[QThread] = None
        self._camera_stream_thread: Optional[QThread] = None
        self._camera_stream_worker: Optional[CameraStreamWorker] = None

        self._build_ui()
        self._wire_signals()

        self.append_log("系统初始化完成", level="INFO")
        self._sync_hardware_buttons()
        self._start_camera()

    # ─────────────────────────── UI 构建 ───────────────────────────
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        grid = QGridLayout(central)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # 左上：功能区
        grid.addWidget(self._build_function_group(), 0, 0)
        # 左下：日志
        grid.addWidget(self._build_log_group(), 1, 0)
        # 右上：机械臂控制
        grid.addWidget(self._build_robot_group(), 0, 1)
        # 右下：实时图像
        grid.addWidget(self._build_camera_group(), 1, 1)

        # 左右对半
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        # 上下比例自动适应（默认 1:1）
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

    # ── 左上：功能区 ──
    def _build_function_group(self) -> QGroupBox:
        """左上方功能区：2x2 瓷砖按钮，每个按钮图标在上、文字在下。

        布局：
            ┌──────────────┬──────────────┐
            │  手眼标定     │  抓取标签     │
            ├──────────────┼──────────────┤
            │ 物流场景抓取  │ 语义引导抓取  │
            └──────────────┴──────────────┘
        """
        grp = QGroupBox("功能区")
        layout = QGridLayout(grp)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)

        self.btn_calib = FunctionTile("手眼标定", os.path.join(ASSETS_DIR, "手眼标定.svg"))
        self.btn_label = FunctionTile("抓取标注", os.path.join(ASSETS_DIR, "矩形框1.svg"))
        self.btn_grasp = FunctionTile("物流场景分拣", os.path.join(ASSETS_DIR, "料箱到人空箱出库.svg"))
        self.btn_semantic = FunctionTile("语义引导抓取", os.path.join(ASSETS_DIR, "AI智选.svg"))

        layout.addWidget(self.btn_calib, 0, 0)
        layout.addWidget(self.btn_label, 0, 1)
        layout.addWidget(self.btn_grasp, 1, 0)
        layout.addWidget(self.btn_semantic, 1, 1)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        return grp

    # ── 左下：日志 ──
    def _build_log_group(self) -> QGroupBox:
        grp = QGroupBox("日志显示")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(12, 18, 12, 12)

        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(5000)  # 限制最多 5000 行，避免内存膨胀
        layout.addWidget(self.log_widget)
        return grp

    # ── 右上：机械臂控制 ──
    def _build_robot_group(self) -> QGroupBox:
        grp = QGroupBox("机械臂控制")
        outer = QVBoxLayout(grp)
        outer.setContentsMargins(12, 18, 12, 12)
        outer.setSpacing(10)

        # 顶部按钮：连接 / 断开 / 夹爪开 / 夹爪关
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        self.btn_connect = QPushButton("连接")
        self.btn_disconnect = QPushButton("断开连接")
        self.btn_gripper_open = QPushButton("夹爪开")
        self.btn_gripper_close = QPushButton("夹爪关")
        for b in (self.btn_connect, self.btn_disconnect,
                  self.btn_gripper_open, self.btn_gripper_close):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            top_row.addWidget(b)
        outer.addLayout(top_row)

        # 中部：左右两列对比 — 工作空间 / 关节空间
        mid = QHBoxLayout()
        mid.setSpacing(12)

        # 左列：工作空间
        ws_box = QGroupBox("工作空间 (位姿)")
        ws_layout = QGridLayout(ws_box)
        ws_layout.setContentsMargins(10, 16, 10, 10)
        ws_layout.setHorizontalSpacing(6)
        ws_layout.setVerticalSpacing(6)
        self.pose_inputs: Dict[str, QLineEdit] = {}
        pose_names = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
        for i, name in enumerate(pose_names):
            lbl = QLabel(f"{name}:")
            lbl.setObjectName("FieldLabel")
            edit = QLineEdit()
            edit.setPlaceholderText("0.000")
            edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            edit.setMinimumWidth(90)
            self.pose_inputs[name] = edit
            ws_layout.addWidget(lbl, i, 0)
            ws_layout.addWidget(edit, i, 1)
        ws_layout.setColumnStretch(1, 1)
        mid.addWidget(ws_box, stretch=1)

        # 中间分割线（仅装饰，不必要可移除）
        vline = QFrame()
        vline.setObjectName("VLine")
        vline.setFrameShape(QFrame.VLine)
        mid.addWidget(vline)

        # 右列：关节空间
        js_box = QGroupBox("关节空间 (角度 °)")
        js_layout = QVBoxLayout(js_box)
        js_layout.setContentsMargins(10, 16, 10, 10)
        js_layout.setSpacing(4)
        self.joint_rows: Dict[str, JointRow] = {}
        for jname, (lo, hi) in JOINT_LIMITS.items():
            row = JointRow(jname, lo, hi)
            self.joint_rows[jname] = row
            js_layout.addWidget(row)
        js_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        mid.addWidget(js_box, stretch=1)

        outer.addLayout(mid)

        # 底部：移动 / 停止
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        self.btn_move = QPushButton("移动")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("Danger")
        for b in (self.btn_move, self.btn_stop):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumHeight(34)
        bottom_row.addWidget(self.btn_move)
        bottom_row.addWidget(self.btn_stop)
        outer.addLayout(bottom_row)

        return grp

    # ── 右下：实时图像 ──
    def _build_camera_group(self) -> QGroupBox:
        grp = QGroupBox("实时图像")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(12, 18, 12, 12)
        self.camera_view = CameraView()
        layout.addWidget(self.camera_view)
        return grp

    # ─────────────────────────── 信号连接 ───────────────────────────
    def _wire_signals(self) -> None:
        # 功能区：4 个独立占位函数
        self.btn_calib.clicked.connect(self.on_btn_calibration_clicked)
        self.btn_label.clicked.connect(self.on_btn_labeling_clicked)
        self.btn_grasp.clicked.connect(self.on_btn_grasp_clicked)
        self.btn_semantic.clicked.connect(self.on_btn_semantic_clicked)

        # 机械臂顶部按钮
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_disconnect.clicked.connect(self._on_disconnect_clicked)
        self.btn_gripper_open.clicked.connect(lambda: self._run_gripper(1.0))
        self.btn_gripper_close.clicked.connect(lambda: self._run_gripper(0.0))

        # 工作空间输入框：编辑结束后打日志
        for name, edit in self.pose_inputs.items():
            edit.editingFinished.connect(
                lambda n=name, e=edit: self._on_pose_edited(n, e.text()))

        # 关节空间：每个 JointRow 都把变化广播到日志
        for jname, row in self.joint_rows.items():
            row.value_changed.connect(self._on_joint_changed)

        # 底部移动 / 停止
        self.btn_move.clicked.connect(self._on_move_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        # 相机帧信号 → CameraView
        self.camera_frame_received.connect(self.camera_view.update_frame)

    # ─────────────────────────── 日志接口 ───────────────────────────
    def append_log(self, message: str, level: str = "INFO") -> None:
        """追加一条带时间戳的日志到日志区。

        :param message: 日志正文
        :param level: 日志级别（INFO / WARN / ERROR / DEBUG），决定显示前缀
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} [{level.upper()}] {message}"
        self.log_widget.appendPlainText(line)
        sb = self.log_widget.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ─────────────────────────── 硬件：相机 ───────────────────────────
    def _start_camera(self) -> None:
        """启动右下角 RealSense 实时图像。

        示教器类界面需要持续反馈现场画面；这里在主窗口启动后自动初始化相机，
        初始化失败时保留 Camera View 占位，并把错误写入日志。
        """
        if self._camera is not None or (
            self._camera_init_thread is not None and self._camera_init_thread.isRunning()
        ):
            return
        self.append_log("正在启动实时相机预览...", level="INFO")
        self._camera_init_thread = QThread(self)
        worker = CameraInitWorker(640, 480, 30)
        worker.moveToThread(self._camera_init_thread)
        self._camera_init_thread.started.connect(worker.run)
        worker.init_done.connect(self._on_camera_init_done)
        worker.error_occurred.connect(lambda m: self.append_log(f"相机启动失败: {m}", "ERROR"))
        worker.finished.connect(self._camera_init_thread.quit)
        self._camera_init_worker = worker
        self._camera_init_thread.start()

    def _on_camera_init_done(self, ok: bool, camera: object) -> None:
        if not ok or camera is None:
            self.append_log("实时相机未启动，保持占位画面", level="WARN")
            return
        self._camera = camera
        self.append_log("实时相机启动成功", level="INFO")
        self._camera_stream_thread = QThread(self)
        worker = CameraStreamWorker(self._camera)
        worker.moveToThread(self._camera_stream_thread)
        self._camera_stream_thread.started.connect(worker.run)
        worker.frame_ready.connect(self.update_camera_frame)
        worker.error_occurred.connect(lambda m: self.append_log(f"相机取流错误: {m}", "ERROR"))
        worker.finished.connect(self._camera_stream_thread.quit)
        self._camera_stream_worker = worker
        self._camera_stream_thread.start()

    # ─────────────────────────── 硬件：机械臂连接与状态 ───────────────────────────
    def _on_connect_clicked(self) -> None:
        if self._robot_connected:
            self.append_log("机械臂已经连接", level="WARN")
            return
        self.append_log("正在连接机械臂 192.168.127.101:8080 ...", level="INFO")
        self.btn_connect.setEnabled(False)
        self._robot_connect_thread = QThread(self)
        worker = RobotConnectWorker("192.168.127.101", 8080)
        worker.moveToThread(self._robot_connect_thread)
        self._robot_connect_thread.started.connect(worker.run)
        worker.connected.connect(self._on_robot_connected)
        worker.error_occurred.connect(lambda m: self.append_log(m, "INFO"))
        worker.finished.connect(self._robot_connect_thread.quit)
        self._robot_connect_worker = worker
        self._robot_connect_thread.start()

    def _on_robot_connected(self, ok: bool, robot: object, gripper: object) -> None:
        if not ok or robot is None:
            self._robot_connected = False
            self._robot = None
            self._gripper = None
            self.append_log("机械臂连接失败", level="ERROR")
            self._sync_hardware_buttons()
            return
        self._robot = robot
        self._gripper = gripper
        self._robot_connected = True
        self._manual_override = False
        self.append_log("机械臂连接成功，开始实时刷新位姿/关节状态", level="INFO")
        self._start_robot_state_polling()
        self._sync_hardware_buttons()

    def _start_robot_state_polling(self) -> None:
        if self._robot is None:
            return
        self._stop_robot_state_polling()
        self._robot_state_thread = QThread(self)
        worker = RobotStateWorker(self._robot, interval_s=0.2)
        worker.moveToThread(self._robot_state_thread)
        self._robot_state_thread.started.connect(worker.run)
        worker.state_ready.connect(self._on_robot_state_ready)
        worker.error_occurred.connect(self._on_robot_state_error)
        worker.finished.connect(self._robot_state_thread.quit)
        self._robot_state_worker = worker
        self._robot_state_thread.start()

    def _stop_robot_state_polling(self) -> None:
        if self._robot_state_worker is not None:
            self._robot_state_worker.stop()
        if self._robot_state_thread is not None:
            self._robot_state_thread.quit()
            self._robot_state_thread.wait(1500)
        self._robot_state_worker = None
        self._robot_state_thread = None

    def _on_robot_state_error(self, message: str) -> None:
        # 状态轮询可能在网络抖动时频繁失败，限频写日志。
        now = time.time()
        if now - self._last_state_log_ts > 2.0:
            self._last_state_log_ts = now
            self.append_log(f"机械臂状态刷新异常: {message}", level="WARN")

    def _on_robot_state_ready(self, state: object) -> None:
        pose, joints = self._extract_pose_and_joints(state)
        if pose is None and joints is None:
            self._on_robot_state_error(f"无法解析状态字段: {state}")
            return
        if self._manual_override:
            # 用户正在编辑目标值时不覆盖输入框；移动/停止后会重新恢复实时刷新。
            return
        self._updating_from_robot = True
        try:
            if pose is not None:
                for name, value in zip(("X", "Y", "Z", "Rx", "Ry", "Rz"), pose):
                    self.pose_inputs[name].setText(f"{float(value):.6f}")
            if joints is not None:
                for name, value in zip(("J1", "J2", "J3", "J4", "J5", "J6"), joints):
                    self.joint_rows[name].set_value(float(value))
        finally:
            self._updating_from_robot = False

    @staticmethod
    def _extract_pose_and_joints(state: object) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
        """从 RealMan 返回的状态字典中兼容解析 pose / joints。

        不同 SDK 版本字段名可能略有差异，因此这里做宽松匹配。
        """
        if not isinstance(state, dict):
            return None, None

        def as_six(value: object) -> Optional[Tuple[float, ...]]:
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)) and len(value) >= 6:
                try:
                    return tuple(float(x) for x in value[:6])
                except (TypeError, ValueError):
                    return None
            return None

        pose = None
        for key in ("pose", "arm_pose", "tool_pose", "tcp_pose", "end_pose"):
            pose = as_six(state.get(key))
            if pose is not None:
                break

        joints = None
        for key in ("joint", "joints", "joint_angle", "joint_angles", "joint_pos", "joint_position"):
            joints = as_six(state.get(key))
            if joints is not None:
                break

        # 某些 SDK 会把关节状态包在 joint_status / arm_state 里。
        for parent_key in ("joint_status", "arm_state", "status"):
            parent = state.get(parent_key)
            if isinstance(parent, dict):
                if joints is None:
                    for key in ("joint", "joints", "joint_angle", "joint_angles", "joint_position"):
                        joints = as_six(parent.get(key))
                        if joints is not None:
                            break
                if pose is None:
                    for key in ("pose", "arm_pose", "tool_pose", "tcp_pose"):
                        pose = as_six(parent.get(key))
                        if pose is not None:
                            break
        return pose, joints

    def _on_disconnect_clicked(self) -> None:
        self.append_log("正在断开机械臂连接...", level="INFO")
        self._stop_robot_state_polling()
        if self._gripper is not None:
            close_fn = getattr(self._gripper, "Motor_Close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    self.append_log(f"夹爪关闭通信失败: {exc}", level="WARN")
        if self._robot is not None:
            for name in ("rm_delete_robot_arm", "rm_close_robot_arm"):
                close_fn = getattr(self._robot, name, None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception as exc:
                        self.append_log(f"机械臂关闭接口 {name} 异常: {exc}", level="WARN")
                    break
        self._robot = None
        self._gripper = None
        self._robot_connected = False
        self._manual_override = False
        self._sync_hardware_buttons()
        self.append_log("机械臂已断开", level="INFO")

    def _sync_hardware_buttons(self) -> None:
        self.btn_connect.setEnabled(not self._robot_connected)
        for btn in (
            self.btn_disconnect,
            self.btn_gripper_open,
            self.btn_gripper_close,
            self.btn_move,
            self.btn_stop,
        ):
            btn.setEnabled(self._robot_connected)

    # ─────────────────────────── 功能区槽函数 ───────────────────────────
    def on_btn_calibration_clicked(self) -> None:
        """启动手眼标定界面 (calibration_gui.py)。"""
        self.append_log("点击 [手眼标定] — 准备启动手眼标定界面", level="INFO")
        self._launch_subprocess(
            tag="calibration",
            script_name="calibration_gui.py",
            label="手眼标定",
        )

    def on_btn_labeling_clicked(self) -> None:
        """启动抓取矩形标注界面 (labeling_gui.py)。"""
        self.append_log("点击 [抓取标签] — 准备启动抓取矩形标注工具", level="INFO")
        self._launch_subprocess(
            tag="labeling",
            script_name="labeling_gui.py",
            label="抓取标签",
        )

    def on_btn_grasp_clicked(self) -> None:
        """启动物流场景抓取界面 (grasp_gui_v2.py)。"""
        self.append_log("点击 [物流场景抓取] — 准备启动物流抓取界面", level="INFO")
        self._launch_subprocess(
            tag="grasp",
            script_name="grasp_gui_v2.py",
            label="物流场景抓取",
        )

    def on_btn_semantic_clicked(self) -> None:
        """占位：语义引导抓取（功能未实现，仅记录日志）。"""
        self.append_log("点击 [语义引导抓取] — 该功能暂未实现，敬请期待", level="DEBUG")

    # ─────────────────────────── 内部槽 ───────────────────────────
    def _on_pose_edited(self, name: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            self.append_log(f"工作空间 {name} 被清空", level="DEBUG")
            return
        try:
            val = float(text)
        except ValueError:
            self.append_log(f"工作空间 {name} 输入非法: {text!r}", level="ERROR")
            return
        self._target_mode = "pose"
        self._manual_override = True
        self.append_log(f"工作空间 {name} = {val:.3f}", level="INFO")

    def _on_joint_changed(self, name: str, value: float) -> None:
        if self._updating_from_robot:
            return
        self._target_mode = "joint"
        self._manual_override = True
        self.append_log(f"关节空间 {name} = {value:.2f}°", level="INFO")

    def _on_move_clicked(self) -> None:
        if not self._robot_connected or self._robot is None:
            self.append_log("机械臂未连接，无法移动", level="ERROR")
            return
        if self._cmd_thread is not None and self._cmd_thread.isRunning():
            self.append_log("已有机械臂运动命令正在执行", level="WARN")
            return

        if self._target_mode == "pose":
            target = self._collect_pose_target()
            if target is None:
                return
            target_str = ", ".join(f"{v:.6f}" for v in target)
            self.append_log(f"点击 [移动] — 目标位姿: {target_str}", level="INFO")
            self._run_robot_command("movel", target)
            return

        joint_state = {n: r.value() for n, r in self.joint_rows.items()}
        target = [joint_state[f"J{i}"] for i in range(1, 7)]
        joint_str = ", ".join(f"{k}={v:.2f}" for k, v in joint_state.items())
        self.append_log(f"点击 [移动] — 目标关节角: {joint_str}", level="INFO")
        self._run_robot_command("movej", target)

    def _collect_pose_target(self) -> Optional[list[float]]:
        values = []
        for name in ("X", "Y", "Z", "Rx", "Ry", "Rz"):
            text = self.pose_inputs[name].text().strip()
            try:
                values.append(float(text))
            except ValueError:
                self.append_log(f"目标位姿 {name} 不是有效数字: {text!r}", level="ERROR")
                return None
        return values

    def _on_stop_clicked(self) -> None:
        if not self._robot_connected or self._robot is None:
            self.append_log("机械臂未连接，无法停止", level="ERROR")
            return
        self.append_log("点击 [停止] — 立即停止机械臂运动", level="WARN")
        self._manual_override = False
        self._run_robot_command("stop", None)

    def _run_robot_command(self, command: str, payload: object) -> None:
        if self._robot is None:
            return
        self._cmd_thread = QThread(self)
        worker = RobotCommandWorker(self._robot, command, payload, speed=20)
        worker.moveToThread(self._cmd_thread)
        self._cmd_thread.started.connect(worker.run)
        worker.command_done.connect(self._on_robot_command_done)
        worker.finished.connect(self._cmd_thread.quit)
        self._cmd_worker = worker
        self._cmd_thread.start()

    def _on_robot_command_done(self, ok: bool, message: str) -> None:
        self.append_log(message, level="INFO" if ok else "ERROR")
        if ok:
            self._manual_override = False

    def _run_gripper(self, position: float) -> None:
        if not self._robot_connected or self._gripper is None:
            self.append_log("机械臂/夹爪未连接，无法执行夹爪命令", level="ERROR")
            return
        if self._gripper_thread is not None and self._gripper_thread.isRunning():
            self.append_log("已有夹爪命令正在执行", level="WARN")
            return
        action = "夹爪开" if position > 0 else "夹爪关"
        self.append_log(f"点击 [{action}]", level="INFO")
        self._gripper_thread = QThread(self)
        worker = GripperWorker(self._gripper, position)
        worker.moveToThread(self._gripper_thread)
        self._gripper_thread.started.connect(worker.run)
        worker.command_done.connect(
            lambda ok, msg: self.append_log(msg, "INFO" if ok else "ERROR"))
        worker.finished.connect(self._gripper_thread.quit)
        self._gripper_worker = worker
        self._gripper_thread.start()

    # ─────────────────────────── 子界面进程管理 ───────────────────────────
    def _launch_subprocess(self, tag: str, script_name: str, label: str) -> None:
        """启动指定脚本作为独立进程，避免 Qt 事件循环冲突。"""
        proc = self._sub_procs.get(tag)
        if proc is not None and proc.poll() is None:
            self.append_log(
                f"[{label}] 已在运行 (PID={proc.pid})，请先关闭旧窗口", level="WARN")
            return

        script_path = os.path.join(PROJECT_DIR, script_name)
        if not os.path.exists(script_path):
            self.append_log(
                f"[{label}] 启动失败：未找到脚本 {script_path}", level="ERROR")
            return

        try:
            new_proc = subprocess.Popen(
                [sys.executable, script_path],
                cwd=PROJECT_DIR,
            )
        except Exception as exc:
            self.append_log(f"[{label}] 启动异常：{exc}", level="ERROR")
            return

        self._sub_procs[tag] = new_proc
        self.append_log(f"[{label}] 已启动 (PID={new_proc.pid})", level="INFO")

    # ─────────────────────────── 对外公开接口 ───────────────────────────
    def update_camera_frame(self, bgr_image: np.ndarray) -> None:
        """供外部模块（例如 OpenCV 采集线程）推送实时画面。

        线程安全：内部使用 Qt 信号在主线程渲染。
        """
        self.camera_frame_received.emit(bgr_image)

    # ─────────────────────────── 关闭事件 ───────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_robot_state_polling()
        if self._camera_stream_worker is not None:
            self._camera_stream_worker.stop()
        if self._camera_stream_thread is not None:
            self._camera_stream_thread.quit()
            self._camera_stream_thread.wait(1500)
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as exc:
                self.append_log(f"相机关闭异常: {exc}", level="WARN")
        if self._robot_connected:
            self._on_disconnect_clicked()

        # 主窗口关闭时不强制关闭子界面进程（让用户自己决定是否保留），
        # 仅做日志记录；如需强杀，可解除下方注释。
        for tag, proc in self._sub_procs.items():
            if proc.poll() is None:
                self.append_log(
                    f"主界面关闭，子进程 [{tag}] (PID={proc.pid}) 仍在运行",
                    level="WARN",
                )
                # proc.terminate()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    # 高 DPI 友好（在创建 QApplication 之前调用）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(INDUSTRIAL_QSS)
    # 默认字体微调
    app.setFont(QFont("Microsoft YaHei", 10))

    win = MainDashboard()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
