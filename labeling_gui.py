"""抓取矩形（grasp rectangle）标注工具 — PyQt5 工业风重写版。

来源：``prepare/grasp-rectangle-labelling-master/main.py`` (Tkinter 版)。
本版本保持原有标注流程与文件格式：

1. 在画布上 **三次点击** 完成一个有方向的抓取矩形：
   * 第 1 次点击：起点 P1
   * 第 2 次点击：终点 P2 — 与 P1 共同确定夹爪一侧（红色边）的方向
   * 第 3 次点击：投影点 — 决定矩形的"宽度"（蓝色边长度），
     P3、P4 由几何投影自动算出。
2. 颜色规则与原版一致：
   * P1→P2 / P3→P4 红色 (夹爪两侧)
   * P2→P3 / P4→P1 蓝色 (夹爪开口方向)
3. 标签格式：``Labels/<dataset>/<image>.txt``，每行 8 个整数
   ``x1 y1 x2 y2 x3 y3 x4 y4``，与原工具完全兼容。

快捷键：

* ``A`` / ``←`` — 上一张
* ``D`` / ``→`` — 下一张
* ``S`` — 保存
* ``X`` / ``Delete`` — 删除当前选中的矩形
* ``P`` — 导出当前画布为带标注的 ``_labeled.jpg``
* ``Esc`` — 取消当前正在绘制的矩形
* ``Ctrl+滚轮`` — 在画布上缩放
"""

from __future__ import annotations

import os
import sys
import glob
import random
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
EXAMPLE_PREVIEW_SIZE = 240          # Examples 缩略图最长边
DEFAULT_LABEL_DIR_NAME = "Labels"   # 默认标签输出目录名
DEFAULT_EXAMPLES_DIR_NAME = "Examples"  # 默认参考图目录名

# 与原版颜色对应：第 1/3 条边红色（夹爪两侧），第 2/4 条边蓝色（开口方向）
EDGE_COLORS = ("#e23030", "#1f7ad6")
HIGHLIGHT_BORDER = QColor("#f7c948")

NORMAL_PEN_WIDTH = 2.0
HIGHLIGHT_PEN_WIDTH = 5.0

# ---------------------------------------------------------------------------
# 工业风格 QSS — 与项目其它 GUI 保持一致
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
QPushButton#Danger {
    background-color: #b53737;
    border-color: #7a2424;
}
QPushButton#Danger:hover { background-color: #cf4747; }

QLineEdit, QSpinBox, QPlainTextEdit, QListWidget {
    background-color: #ffffff;
    border: 1px solid #b8c0c8;
    border-radius: 3px;
    padding: 3px 6px;
    selection-background-color: #3b78b3;
}
QPlainTextEdit { font-family: Consolas, "Courier New", monospace; }
QListWidget::item:selected {
    background-color: #cfe2f3;
    color: #1a1a1a;
}
QLabel#FieldLabel { color: #2b3a48; font-weight: 500; }
QLabel#StatusBadge {
    color: #ffffff;
    background-color: #2b5a82;
    border-radius: 3px;
    padding: 2px 8px;
}
QLabel#ExampleSlot {
    background-color: #232a30;
    color: #c5ccd2;
    border: 1px solid #34495e;
    min-height: 120px;
    qproperty-alignment: AlignCenter;
}
"""


# ---------------------------------------------------------------------------
# 几何工具：根据投影点 (xr, yr) 求出 P3、P4
# ---------------------------------------------------------------------------
def complete_rectangle_with_projection_point(
    x1: float, y1: float, x2: float, y2: float, xr: float, yr: float
) -> Tuple[float, float, float, float]:
    """从原版工具复刻：给定 P1、P2 与 P3 的"投影点"，算出第三、第四个角点。

    数学含义：P2→P3 必须与 P1→P2 垂直，P3→P4 必须与 P1→P2 平行且等长。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.true_divide(y2 - y1, x2 - x1)
        m_perp = np.true_divide(-1.0, m)

    if m == 0:
        x3 = x2
        y3 = yr
        x4 = x1
        y4 = yr
    elif m_perp == 0:
        x3 = xr
        y3 = y2
        x4 = xr
        y4 = y1
    else:
        x3 = (yr - y2 + m_perp * x2 - m * xr) / (m_perp - m)
        y3 = y2 + m_perp * (x3 - x2)
        l = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if y2 > y1:
            y4 = y3 - np.sqrt((m ** 2 * l ** 2) / (1.0 + m ** 2))
        else:
            y4 = y3 + np.sqrt((m ** 2 * l ** 2) / (1.0 + m ** 2))
        x4 = x3 + (y4 - y3) / m
    return float(x3), float(y3), float(x4), float(y4)


# ---------------------------------------------------------------------------
# 标注画布 — QGraphicsView 子类
# ---------------------------------------------------------------------------
class LabelCanvas(QGraphicsView):
    """支持三点确定矩形 + 滚轮缩放 + 键盘取消的标注画布。

    * 信号 :attr:`mouse_moved` — 实时报告鼠标在 *原图像素坐标系* 中的位置。
    * 信号 :attr:`rectangle_committed` — 完成一个矩形时发出 ``[(x1,y1),...,(x4,y4)]``。
    * 信号 :attr:`status_changed` — 状态文本（点击数、当前操作提示）。
    """

    mouse_moved = pyqtSignal(int, int)
    rectangle_committed = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints() | 0x02)  # SmoothPixmapTransform
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setAlignment(Qt.AlignCenter)
        self.setBackgroundBrush(QBrush(QColor("#1a1a1a")))
        self.setStyleSheet("border: 1px solid #34495e;")

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._image_size = (0, 0)  # (w, h)

        # 三点法状态机
        self._click_state = 1
        self._p1: Optional[Tuple[float, float]] = None
        self._p2: Optional[Tuple[float, float]] = None

        # 预览线（鼠标移动时绘制）
        self._preview_edge1: Optional[QGraphicsLineItem] = None
        self._preview_edge2: Optional[QGraphicsLineItem] = None

        # 已确认矩形：每项是 ``(coords, [4 个 QGraphicsLineItem])``
        self._rectangles: List[Tuple[List[Tuple[float, float]], List[QGraphicsLineItem]]] = []
        self._highlight_index: int = -1

        self._scale_factor = 1.0

    # ── 公共：图像加载与清理 ──
    def load_image(self, path: str) -> bool:
        """加载图片并清空当前矩形。返回是否加载成功。"""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)

        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pix)
        self._pixmap_item.setZValue(-10)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self._image_size = (w, h)

        self._click_state = 1
        self._p1 = None
        self._p2 = None
        self._preview_edge1 = None
        self._preview_edge2 = None
        self._rectangles = []
        self._highlight_index = -1
        self._scale_factor = 1.0
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._emit_status()
        return True

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def rectangles(self) -> List[List[Tuple[int, int]]]:
        """返回已确认矩形的整数像素坐标。"""
        return [
            [(int(round(x)), int(round(y))) for x, y in coords]
            for coords, _ in self._rectangles
        ]

    # ── 标注操作 ──
    def add_rectangle(self, coords: List[Tuple[float, float]]) -> None:
        """从外部加载（如读 .txt 时）追加一个矩形。"""
        items = self._draw_edges(coords, NORMAL_PEN_WIDTH)
        self._rectangles.append((list(coords), items))

    def clear_rectangles(self) -> None:
        for _, items in self._rectangles:
            for it in items:
                self._scene.removeItem(it)
        self._rectangles = []
        self._highlight_index = -1

    def delete_rectangle(self, index: int) -> bool:
        if not (0 <= index < len(self._rectangles)):
            return False
        _, items = self._rectangles.pop(index)
        for it in items:
            self._scene.removeItem(it)
        if self._highlight_index == index:
            self._highlight_index = -1
        elif self._highlight_index > index:
            self._highlight_index -= 1
        return True

    def highlight_rectangle(self, index: int) -> None:
        """把指定矩形显示为加粗 + 高亮，便于在画面中定位。"""
        # 先恢复之前高亮的
        if 0 <= self._highlight_index < len(self._rectangles):
            _, items = self._rectangles[self._highlight_index]
            for i, it in enumerate(items):
                pen = it.pen()
                pen.setWidthF(NORMAL_PEN_WIDTH)
                pen.setColor(QColor(EDGE_COLORS[i % 2]))
                it.setPen(pen)
        if not (0 <= index < len(self._rectangles)):
            self._highlight_index = -1
            return
        self._highlight_index = index
        _, items = self._rectangles[index]
        for it in items:
            pen = it.pen()
            pen.setWidthF(HIGHLIGHT_PEN_WIDTH)
            pen.setColor(HIGHLIGHT_BORDER)
            it.setPen(pen)

    def cancel_current(self) -> None:
        """取消正在绘制的矩形（清掉预览线，回到初始点击状态）。"""
        self._remove_preview_lines()
        self._click_state = 1
        self._p1 = None
        self._p2 = None
        self._emit_status()

    # ── 鼠标交互 ──
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or not self.has_image():
            super().mousePressEvent(event)
            return
        scene_pt = self.mapToScene(event.pos())
        x, y = self._clamp(scene_pt.x(), scene_pt.y())

        if self._click_state == 1:
            self._p1 = (x, y)
            self._click_state = 2
        elif self._click_state == 2:
            self._p2 = (x, y)
            self._click_state = 3
        elif self._click_state == 3:
            assert self._p1 is not None and self._p2 is not None
            x1, y1 = self._p1
            x2, y2 = self._p2
            x3, y3, x4, y4 = complete_rectangle_with_projection_point(
                x1, y1, x2, y2, x, y)
            coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

            self._remove_preview_lines()
            items = self._draw_edges(coords, NORMAL_PEN_WIDTH)
            self._rectangles.append((coords, items))
            self.rectangle_committed.emit(coords)

            self._click_state = 1
            self._p1 = None
            self._p2 = None
        self._emit_status()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self.has_image():
            super().mouseMoveEvent(event)
            return
        scene_pt = self.mapToScene(event.pos())
        x, y = self._clamp(scene_pt.x(), scene_pt.y())
        self.mouse_moved.emit(int(round(x)), int(round(y)))

        if self._click_state == 1:
            self._remove_preview_lines()
        elif self._click_state == 2 and self._p1 is not None:
            x1, y1 = self._p1
            self._update_preview_edge1(x1, y1, x, y)
        elif self._click_state == 3 and self._p1 is not None and self._p2 is not None:
            x1, y1 = self._p1
            x2, y2 = self._p2
            x3, y3, _x4, _y4 = complete_rectangle_with_projection_point(
                x1, y1, x2, y2, x, y)
            self._update_preview_edge2(x2, y2, x3, y3)

        super().mouseMoveEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        # 按住 Ctrl 滚轮缩放
        if event.modifiers() & Qt.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.15 if angle > 0 else (1.0 / 1.15)
            new_scale = self._scale_factor * factor
            new_scale = max(0.1, min(10.0, new_scale))
            real_factor = new_scale / self._scale_factor
            self._scale_factor = new_scale
            self.scale(real_factor, real_factor)
            event.accept()
            return
        super().wheelEvent(event)

    # ── 私有辅助 ──
    def _clamp(self, x: float, y: float) -> Tuple[float, float]:
        w, h = self._image_size
        if w == 0 or h == 0:
            return x, y
        return max(0.0, min(float(w), x)), max(0.0, min(float(h), y))

    def _remove_preview_lines(self) -> None:
        for attr in ("_preview_edge1", "_preview_edge2"):
            line = getattr(self, attr)
            if line is not None:
                self._scene.removeItem(line)
                setattr(self, attr, None)

    def _update_preview_edge1(self, x1: float, y1: float, x: float, y: float) -> None:
        if self._preview_edge1 is None:
            self._preview_edge1 = self._scene.addLine(
                x1, y1, x, y, QPen(QColor(EDGE_COLORS[0]), NORMAL_PEN_WIDTH))
        else:
            self._preview_edge1.setLine(x1, y1, x, y)

    def _update_preview_edge2(self, x2: float, y2: float, x3: float, y3: float) -> None:
        if self._preview_edge2 is None:
            self._preview_edge2 = self._scene.addLine(
                x2, y2, x3, y3, QPen(QColor(EDGE_COLORS[1]), NORMAL_PEN_WIDTH))
        else:
            self._preview_edge2.setLine(x2, y2, x3, y3)

    def _draw_edges(
        self, coords: List[Tuple[float, float]], pen_width: float
    ) -> List[QGraphicsLineItem]:
        items: List[QGraphicsLineItem] = []
        for i in range(4):
            x_a, y_a = coords[i]
            x_b, y_b = coords[(i + 1) % 4]
            color = QColor(EDGE_COLORS[i % 2])
            line = self._scene.addLine(
                x_a, y_a, x_b, y_b, QPen(color, pen_width, Qt.SolidLine, Qt.RoundCap))
            line.setZValue(10)
            items.append(line)
        return items

    def _emit_status(self) -> None:
        if not self.has_image():
            self.status_changed.emit("尚未加载图像")
            return
        msg = {
            1: "点击 #1：选择第一条红边的起点",
            2: "点击 #2：选择第一条红边的终点",
            3: "点击 #3：选择第三条红边方向上的投影点",
        }.get(self._click_state, "")
        self.status_changed.emit(msg)


# ---------------------------------------------------------------------------
# Examples 缩略预览面板
# ---------------------------------------------------------------------------
class ExamplePanel(QWidget):
    """显示 0~3 张随机示例图，用于辅助标注。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        title = QLabel("示例参考")
        title.setObjectName("FieldLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self._slots: List[QLabel] = []
        for _ in range(3):
            lbl = QLabel("(空)")
            lbl.setObjectName("ExampleSlot")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(lbl, stretch=1)
            self._slots.append(lbl)
        layout.addStretch(0)

    def show_examples(self, paths: List[str]) -> None:
        for slot in self._slots:
            slot.clear()
            slot.setText("(空)")
            slot.setPixmap(QPixmap())
        for slot, path in zip(self._slots, paths):
            img = cv2.imread(path)
            if img is None:
                slot.setText("加载失败")
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                EXAMPLE_PREVIEW_SIZE, EXAMPLE_PREVIEW_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            slot.setPixmap(scaled)
            slot.setText("")


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class LabelingMainWindow(QMainWindow):
    """抓取矩形标注主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("抓取矩形标注工具")
        self.resize(1400, 900)

        self._image_dir: str = ""
        self._label_dir: str = ""
        self._image_paths: List[str] = []
        self._cur_index: int = -1
        self._dirty: bool = False  # 当前图是否有未保存改动

        self._build_ui()
        self._setup_shortcuts()
        self._wire_signals()

        self.append_log("标注工具已启动。请先点击 [选择图片目录] 加载数据。", level="INFO")
        self._sync_buttons()

    # ─────────────────────────── UI ───────────────────────────
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        outer.addWidget(self._build_top_toolbar())

        # 中部：左侧示例 | 中间画布 | 右侧操作
        middle_splitter = QSplitter(Qt.Horizontal)
        middle_splitter.setChildrenCollapsible(False)

        # 左：示例
        left_box = QGroupBox("Examples")
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(8, 14, 8, 8)
        self.example_panel = ExamplePanel()
        left_layout.addWidget(self.example_panel)
        middle_splitter.addWidget(left_box)

        # 中：画布
        canvas_box = QGroupBox("标注画布")
        canvas_layout = QVBoxLayout(canvas_box)
        canvas_layout.setContentsMargins(8, 14, 8, 8)
        self.canvas = LabelCanvas()
        canvas_layout.addWidget(self.canvas, stretch=1)
        # 状态条：左侧操作提示，右侧鼠标坐标
        bar = QHBoxLayout()
        self.lbl_canvas_status = QLabel("尚未加载图像")
        self.lbl_canvas_status.setObjectName("StatusBadge")
        self.lbl_mouse_pos = QLabel("x: -, y: -")
        bar.addWidget(self.lbl_canvas_status)
        bar.addStretch(1)
        bar.addWidget(self.lbl_mouse_pos)
        canvas_layout.addLayout(bar)
        middle_splitter.addWidget(canvas_box)

        # 右：操作
        right_box = QGroupBox("矩形列表 / 操作")
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(8, 14, 8, 8)
        right_layout.setSpacing(8)

        self.list_rectangles = QListWidget()
        self.list_rectangles.setSelectionMode(QListWidget.SingleSelection)
        right_layout.addWidget(self.list_rectangles, stretch=1)

        self.btn_delete = QPushButton("删除选中 (X)")
        self.btn_delete.setObjectName("Danger")
        self.btn_save = QPushButton("保存 (S)")
        self.btn_export = QPushButton("导出标注图 (P)")
        self.btn_clear = QPushButton("清空全部")
        right_layout.addWidget(self.btn_delete)
        right_layout.addWidget(self.btn_save)
        right_layout.addWidget(self.btn_export)
        right_layout.addWidget(self.btn_clear)

        middle_splitter.addWidget(right_box)
        middle_splitter.setStretchFactor(0, 1)
        middle_splitter.setStretchFactor(1, 4)
        middle_splitter.setStretchFactor(2, 1)
        middle_splitter.setSizes([260, 900, 260])
        outer.addWidget(middle_splitter, stretch=4)

        # 翻页栏
        outer.addWidget(self._build_navigation_bar())

        # 底部日志
        log_box = QGroupBox("日志显示")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(8, 14, 8, 8)
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(5000)
        self.log_widget.setMinimumHeight(120)
        log_layout.addWidget(self.log_widget)
        outer.addWidget(log_box, stretch=1)

    def _build_top_toolbar(self) -> QGroupBox:
        box = QGroupBox("数据集")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)

        self.btn_choose_dir = QPushButton("选择图片目录…")
        layout.addWidget(self.btn_choose_dir)

        layout.addWidget(self._mk_label("图片目录:"))
        self.le_image_dir = QLineEdit()
        self.le_image_dir.setReadOnly(True)
        self.le_image_dir.setPlaceholderText("尚未选择")
        layout.addWidget(self.le_image_dir, stretch=2)

        layout.addWidget(self._mk_label("标签目录:"))
        self.le_label_dir = QLineEdit()
        self.le_label_dir.setPlaceholderText("默认: 同级 Labels/<目录名>")
        layout.addWidget(self.le_label_dir, stretch=2)

        self.lbl_progress = QLabel("0/0")
        self.lbl_progress.setObjectName("StatusBadge")
        layout.addWidget(self.lbl_progress)
        return box

    def _build_navigation_bar(self) -> QGroupBox:
        box = QGroupBox("翻页 / 跳转")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)

        self.btn_prev = QPushButton("<< 上一张 (A)")
        self.btn_next = QPushButton("下一张 (D) >>")
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.btn_next)

        layout.addWidget(self._mk_label("跳转到第"))
        self.spin_goto = QSpinBox()
        self.spin_goto.setRange(1, 1)
        self.spin_goto.setValue(1)
        layout.addWidget(self.spin_goto)
        layout.addWidget(self._mk_label("张"))
        self.btn_goto = QPushButton("Go")
        layout.addWidget(self.btn_goto)

        layout.addStretch(1)
        self.lbl_filename = QLabel("(无)")
        self.lbl_filename.setStyleSheet("color: #5d6d7e;")
        layout.addWidget(self.lbl_filename)
        return box

    @staticmethod
    def _mk_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    # ─────────────────────────── 信号 / 快捷键 ───────────────────────────
    def _wire_signals(self) -> None:
        self.btn_choose_dir.clicked.connect(self._on_choose_dir)
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next.clicked.connect(self._on_next)
        self.btn_goto.clicked.connect(self._on_goto)

        self.btn_delete.clicked.connect(self._on_delete_selected)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_clear.clicked.connect(self._on_clear_all)

        self.canvas.mouse_moved.connect(self._on_mouse_moved)
        self.canvas.status_changed.connect(self.lbl_canvas_status.setText)
        self.canvas.rectangle_committed.connect(self._on_rectangle_committed)

        self.list_rectangles.currentRowChanged.connect(self._on_list_row_changed)

        self.le_label_dir.editingFinished.connect(self._on_label_dir_edited)

    def _setup_shortcuts(self) -> None:
        # 快捷键作用于整个窗口（无需画布拥有焦点）
        self._add_shortcut("A", self._on_prev)
        self._add_shortcut(QKeySequence(Qt.Key_Left), self._on_prev)
        self._add_shortcut("D", self._on_next)
        self._add_shortcut(QKeySequence(Qt.Key_Right), self._on_next)
        self._add_shortcut("S", self._on_save)
        self._add_shortcut("X", self._on_delete_selected)
        self._add_shortcut(QKeySequence(Qt.Key_Delete), self._on_delete_selected)
        self._add_shortcut("P", self._on_export)
        self._add_shortcut(QKeySequence(Qt.Key_Escape), self._on_cancel_current)

    def _add_shortcut(self, sequence, handler) -> None:
        sc = QShortcut(QKeySequence(sequence), self)
        sc.setContext(Qt.ApplicationShortcut)
        sc.activated.connect(handler)

    # ─────────────────────────── 日志 ───────────────────────────
    def append_log(self, message: str, level: str = "INFO") -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} [{level.upper()}] {message}"
        self.log_widget.appendPlainText(line)
        sb = self.log_widget.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ─────────────────────────── 槽函数 ───────────────────────────
    def _on_choose_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if not d:
            return
        self._image_dir = d
        self.le_image_dir.setText(d)
        # 默认标签目录：同级目录下 Labels/<basename>
        parent = os.path.dirname(os.path.abspath(d))
        base = os.path.basename(os.path.abspath(d)) or "default"
        default_label_dir = os.path.join(parent, DEFAULT_LABEL_DIR_NAME, base)
        if not self.le_label_dir.text().strip():
            self.le_label_dir.setText(default_label_dir)
        self._label_dir = self.le_label_dir.text().strip() or default_label_dir
        os.makedirs(self._label_dir, exist_ok=True)

        self._scan_images()
        self._load_examples()
        if not self._image_paths:
            QMessageBox.warning(self, "未找到图片",
                                f"目录 {d} 下未发现任何图片 ({', '.join(SUPPORTED_EXTS)})")
            self.append_log(f"目录 {d} 下没有支持的图片文件", level="WARN")
            self._sync_buttons()
            return

        self.append_log(
            f"已加载 {len(self._image_paths)} 张图片，标签目录: {self._label_dir}", level="INFO")
        self._cur_index = 0
        self.spin_goto.setRange(1, len(self._image_paths))
        self._load_current_image()
        self._sync_buttons()

    def _on_label_dir_edited(self) -> None:
        path = self.le_label_dir.text().strip()
        if not path:
            return
        if path == self._label_dir:
            return
        self._label_dir = path
        try:
            os.makedirs(path, exist_ok=True)
            self.append_log(f"标签目录已切换为: {path}", level="INFO")
        except OSError as exc:
            self.append_log(f"标签目录创建失败: {exc}", level="ERROR")

    def _on_prev(self) -> None:
        if not self._image_paths:
            return
        self._save_if_dirty(silent=True)
        self._cur_index = (self._cur_index - 1) % len(self._image_paths)
        self._load_current_image()

    def _on_next(self) -> None:
        if not self._image_paths:
            return
        self._save_if_dirty(silent=True)
        self._cur_index = (self._cur_index + 1) % len(self._image_paths)
        self._load_current_image()

    def _on_goto(self) -> None:
        if not self._image_paths:
            return
        target = self.spin_goto.value() - 1
        if target == self._cur_index:
            return
        self._save_if_dirty(silent=True)
        self._cur_index = max(0, min(len(self._image_paths) - 1, target))
        self._load_current_image()

    def _on_delete_selected(self) -> None:
        if not self.canvas.has_image():
            return
        row = self.list_rectangles.currentRow()
        if row < 0:
            self.append_log("尚未选择要删除的矩形", level="WARN")
            return
        ok = self.canvas.delete_rectangle(row)
        if ok:
            self.list_rectangles.takeItem(row)
            self._dirty = True
            self.append_log(f"已删除第 {row + 1} 个矩形", level="INFO")
        self._sync_buttons()

    def _on_clear_all(self) -> None:
        if not self.canvas.has_image() or not self.canvas.rectangles():
            return
        if QMessageBox.question(
            self, "清空", "确定要清空当前图片的全部矩形吗？") != QMessageBox.Yes:
            return
        self.canvas.clear_rectangles()
        self.list_rectangles.clear()
        self._dirty = True
        self.append_log("已清空当前图片的所有矩形", level="INFO")
        self._sync_buttons()

    def _on_save(self) -> None:
        if not self.canvas.has_image() or self._cur_index < 0:
            return
        try:
            self._save_current_label_file()
            self._dirty = False
            self.append_log(
                f"已保存第 {self._cur_index + 1} 张图片的标签 "
                f"({len(self.canvas.rectangles())} 个矩形)", level="INFO")
        except OSError as exc:
            self.append_log(f"保存失败: {exc}", level="ERROR")
            QMessageBox.critical(self, "保存失败", str(exc))

    def _on_export(self) -> None:
        """导出当前画布为带标注的图片（不含工具 UI）。"""
        if not self.canvas.has_image() or self._cur_index < 0:
            return
        path = self._labeled_jpg_path(self._image_paths[self._cur_index])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._render_labeled_image(path)
            self.append_log(f"已导出带标注图片到: {path}", level="INFO")
        except Exception as exc:
            self.append_log(f"导出失败: {exc}", level="ERROR")
            QMessageBox.critical(self, "导出失败", str(exc))

    def _on_cancel_current(self) -> None:
        if not self.canvas.has_image():
            return
        self.canvas.cancel_current()
        self.append_log("已取消当前正在绘制的矩形", level="DEBUG")

    def _on_mouse_moved(self, x: int, y: int) -> None:
        self.lbl_mouse_pos.setText(f"x: {x}, y: {y}")

    def _on_rectangle_committed(self, coords: object) -> None:
        if not isinstance(coords, list):
            return
        idx = len(self.canvas.rectangles())
        text = self._format_rectangle_text(coords, idx)
        self.list_rectangles.addItem(QListWidgetItem(text))
        self.list_rectangles.setCurrentRow(idx - 1)
        self._dirty = True
        self.append_log(f"新增矩形 #{idx}", level="DEBUG")
        self._sync_buttons()

    def _on_list_row_changed(self, row: int) -> None:
        self.canvas.highlight_rectangle(row)

    # ─────────────────────────── 数据加载 / 保存 ───────────────────────────
    def _scan_images(self) -> None:
        self._image_paths = []
        if not self._image_dir:
            return
        for root, _dirs, files in os.walk(self._image_dir):
            for f in sorted(files):
                if f.lower().endswith(SUPPORTED_EXTS):
                    self._image_paths.append(os.path.join(root, f))
        self._image_paths.sort()

    def _load_examples(self) -> None:
        if not self._image_dir:
            return
        parent = os.path.dirname(os.path.abspath(self._image_dir))
        base = os.path.basename(os.path.abspath(self._image_dir))
        candidate_dir = os.path.join(parent, DEFAULT_EXAMPLES_DIR_NAME, base)
        examples: List[str] = []
        if os.path.isdir(candidate_dir):
            files = []
            for ext in SUPPORTED_EXTS:
                files.extend(glob.glob(os.path.join(candidate_dir, f"**/*{ext}"), recursive=True))
            random.shuffle(files)
            examples = files[:3]
        self.example_panel.show_examples(examples)
        if examples:
            self.append_log(f"已从 {candidate_dir} 加载 {len(examples)} 张示例", level="DEBUG")

    def _load_current_image(self) -> None:
        if not (0 <= self._cur_index < len(self._image_paths)):
            return
        path = self._image_paths[self._cur_index]
        if not self.canvas.load_image(path):
            self.append_log(f"加载失败: {path}", level="ERROR")
            return
        self.list_rectangles.clear()
        # 读取已有标签
        label_path = self._label_path_for(path)
        loaded = 0
        if os.path.isfile(label_path):
            try:
                with open(label_path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        nums = [float(v) for v in line.split()]
                        if len(nums) >= 8:
                            coords = [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
                            self.canvas.add_rectangle(coords)
                            self.list_rectangles.addItem(
                                self._format_rectangle_text(coords, loaded + 1))
                            loaded += 1
            except OSError as exc:
                self.append_log(f"读取标签 {label_path} 失败: {exc}", level="ERROR")
        self._dirty = False
        self.lbl_filename.setText(path)
        self.lbl_progress.setText(f"{self._cur_index + 1}/{len(self._image_paths)}")
        self.spin_goto.blockSignals(True)
        self.spin_goto.setValue(self._cur_index + 1)
        self.spin_goto.blockSignals(False)
        self.append_log(
            f"加载图像 [{self._cur_index + 1}/{len(self._image_paths)}] "
            f"{os.path.basename(path)}，已读出 {loaded} 个矩形", level="INFO")
        self._sync_buttons()

    def _save_if_dirty(self, silent: bool = False) -> None:
        """切换图片前自动保存（参考原工具行为）。"""
        if not self._dirty or self._cur_index < 0:
            return
        try:
            self._save_current_label_file()
            self._dirty = False
            if not silent:
                self.append_log("已自动保存当前图标签", level="INFO")
        except OSError as exc:
            self.append_log(f"自动保存失败: {exc}", level="ERROR")

    def _save_current_label_file(self) -> None:
        path = self._image_paths[self._cur_index]
        label_path = self._label_path_for(path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        rectangles = self.canvas.rectangles()
        with open(label_path, "w", encoding="utf-8") as fp:
            for rect in rectangles:
                flat = " ".join(str(v) for pt in rect for v in pt)
                fp.write(flat + "\n")

    def _label_path_for(self, image_path: str) -> str:
        rel = os.path.relpath(image_path, self._image_dir) if self._image_dir else os.path.basename(image_path)
        rel_no_ext = os.path.splitext(rel)[0] + ".txt"
        if not self._label_dir:
            return os.path.join(os.path.dirname(image_path), rel_no_ext)
        return os.path.join(self._label_dir, rel_no_ext)

    def _labeled_jpg_path(self, image_path: str) -> str:
        rel = os.path.relpath(image_path, self._image_dir) if self._image_dir else os.path.basename(image_path)
        rel_labeled = os.path.splitext(rel)[0] + "_labeled.jpg"
        base = self._label_dir or os.path.dirname(image_path)
        return os.path.join(base, rel_labeled)

    def _render_labeled_image(self, output_path: str) -> None:
        """直接在原图上以 OpenCV 画矩形并保存，避免 Qt 截屏的字体/缩放误差。"""
        path = self._image_paths[self._cur_index]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"无法读取原图: {path}")
        rectangles = self.canvas.rectangles()
        # OpenCV 颜色为 BGR
        red_bgr = (48, 48, 226)   # ≈ #e23030
        blue_bgr = (214, 122, 31)  # ≈ #1f7ad6
        for rect in rectangles:
            for i in range(4):
                p_a = rect[i]
                p_b = rect[(i + 1) % 4]
                color = red_bgr if i % 2 == 0 else blue_bgr
                cv2.line(img, p_a, p_b, color, 2, lineType=cv2.LINE_AA)
        cv2.imwrite(output_path, img)

    @staticmethod
    def _format_rectangle_text(coords: List[Tuple[float, float]], idx: int) -> str:
        ints = [(int(round(x)), int(round(y))) for x, y in coords]
        return f"#{idx:02d}  " + " | ".join(f"({x},{y})" for x, y in ints)

    def _sync_buttons(self) -> None:
        has_imgs = bool(self._image_paths)
        has_image = self.canvas.has_image()
        has_rect = bool(self.canvas.rectangles())
        self.btn_prev.setEnabled(has_imgs)
        self.btn_next.setEnabled(has_imgs)
        self.btn_goto.setEnabled(has_imgs)
        self.spin_goto.setEnabled(has_imgs)
        self.btn_save.setEnabled(has_image)
        self.btn_export.setEnabled(has_image and has_rect)
        self.btn_clear.setEnabled(has_image and has_rect)
        self.btn_delete.setEnabled(has_rect)

    # ─────────────────────────── 关闭事件 ───────────────────────────
    def closeEvent(self, event) -> None:  # noqa: N802
        if self._dirty:
            ret = QMessageBox.question(
                self, "未保存",
                "当前图片有未保存的标注，是否保存？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if ret == QMessageBox.Cancel:
                event.ignore()
                return
            if ret == QMessageBox.Yes:
                try:
                    self._save_current_label_file()
                except OSError as exc:
                    self.append_log(f"关闭前保存失败: {exc}", level="ERROR")
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(INDUSTRIAL_QSS)
    app.setFont(QFont("Microsoft YaHei", 10))

    win = LabelingMainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
