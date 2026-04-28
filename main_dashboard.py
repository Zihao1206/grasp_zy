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
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap, QResizeEvent
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
        self._spin.setValue(v)


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

        self._build_ui()
        self._wire_signals()

        self.append_log("系统初始化完成", level="INFO")

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
        grp = QGroupBox("功能区")
        layout = QGridLayout(grp)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)

        self.btn_grasp = QPushButton("机械臂抓取")
        self.btn_calib = QPushButton("手眼标定")
        self.btn_spare1 = QPushButton("备用功能 1")
        self.btn_spare2 = QPushButton("备用功能 2")

        for b in (self.btn_grasp, self.btn_calib, self.btn_spare1, self.btn_spare2):
            b.setObjectName("PrimaryBig")
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.btn_grasp, 0, 0)
        layout.addWidget(self.btn_calib, 0, 1)
        layout.addWidget(self.btn_spare1, 1, 0)
        layout.addWidget(self.btn_spare2, 1, 1)

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
        self.btn_grasp.clicked.connect(self.on_btn_grasp_clicked)
        self.btn_calib.clicked.connect(self.on_btn_calibration_clicked)
        self.btn_spare1.clicked.connect(self.on_btn_spare1_clicked)
        self.btn_spare2.clicked.connect(self.on_btn_spare2_clicked)

        # 机械臂顶部按钮
        self.btn_connect.clicked.connect(
            lambda: self.append_log("点击 [连接] — 尝试连接机械臂", level="INFO"))
        self.btn_disconnect.clicked.connect(
            lambda: self.append_log("点击 [断开连接] — 释放机械臂连接", level="INFO"))
        self.btn_gripper_open.clicked.connect(
            lambda: self.append_log("点击 [夹爪开]", level="INFO"))
        self.btn_gripper_close.clicked.connect(
            lambda: self.append_log("点击 [夹爪关]", level="INFO"))

        # 工作空间输入框：编辑结束后打日志
        for name, edit in self.pose_inputs.items():
            edit.editingFinished.connect(
                lambda n=name, e=edit: self._on_pose_edited(n, e.text()))

        # 关节空间：每个 JointRow 都把变化广播到日志
        for jname, row in self.joint_rows.items():
            row.value_changed.connect(self._on_joint_changed)

        # 底部移动 / 停止
        self.btn_move.clicked.connect(self._on_move_clicked)
        self.btn_stop.clicked.connect(
            lambda: self.append_log("点击 [停止] — 立即停止机械臂运动", level="WARN"))

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

    # ─────────────────────────── 功能区占位函数 ───────────────────────────
    def on_btn_grasp_clicked(self) -> None:
        """占位：机械臂抓取（启动 grasp_gui_v2.py）。"""
        self.append_log("点击 [机械臂抓取] — 准备启动抓取界面", level="INFO")
        self._launch_subprocess(
            tag="grasp",
            script_name="grasp_gui_v2.py",
            label="机械臂抓取",
        )

    def on_btn_calibration_clicked(self) -> None:
        """占位：手眼标定（启动 calibration_gui.py）。"""
        self.append_log("点击 [手眼标定] — 准备启动手眼标定界面", level="INFO")
        self._launch_subprocess(
            tag="calibration",
            script_name="calibration_gui.py",
            label="手眼标定",
        )

    def on_btn_spare1_clicked(self) -> None:
        """占位：备用功能 1。"""
        self.append_log("点击 [备用功能 1] — 该功能暂未实现", level="DEBUG")

    def on_btn_spare2_clicked(self) -> None:
        """占位：备用功能 2。"""
        self.append_log("点击 [备用功能 2] — 该功能暂未实现", level="DEBUG")

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
        self.append_log(f"工作空间 {name} = {val:.3f}", level="INFO")

    def _on_joint_changed(self, name: str, value: float) -> None:
        self.append_log(f"关节空间 {name} = {value:.2f}°", level="INFO")

    def _on_move_clicked(self) -> None:
        joint_state = {n: r.value() for n, r in self.joint_rows.items()}
        joint_str = ", ".join(f"{k}={v:.2f}" for k, v in joint_state.items())
        self.append_log(f"点击 [移动] — 目标关节角: {joint_str}", level="INFO")

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
