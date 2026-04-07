from __future__ import annotations

import argparse
import io
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Dict, Optional, Protocol

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QMutex, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


OBJECT_LABELS = ["terminal", "limit", "voltage", "soap", "banana", "carrot", "daikon", "relay"]


class CameraLike(Protocol):
    def get_img(self) -> tuple[np.ndarray, np.ndarray]: ...


class RobotLike(Protocol):
    def rm_set_arm_stop(self) -> int: ...


class GraspLike(Protocol):
    @property
    def camera(self) -> CameraLike: ...

    @property
    def robot(self) -> RobotLike: ...

    @property
    def robot_speed(self) -> int: ...

    def plan_grasp(self, label: str, vis_callback: Optional[Callable] = None) -> bool: ...

    def execute_grasp(self, resume_event: object = None, cancel_event: object = None) -> bool: ...

    def obj_grasp(self, label: str, vis: bool = False, vis_callback: Optional[Callable] = None) -> bool: ...

    def init_gripper(self) -> None: ...


class GUIState(Enum):
    STARTUP = auto()
    IDLE = auto()
    INITIALIZING = auto()
    READY = auto()
    PREVIEW = auto()
    GRASPING = auto()
    PAUSED = auto()
    STOPPING = auto()
    FAULT = auto()
    CLOSING = auto()


@dataclass
class ButtonState:
    button_id: str
    enabled: bool


class StateMachine:
    _BUTTON_RULES: Dict[GUIState, Dict[str, bool]] = {
        GUIState.STARTUP: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
        GUIState.IDLE: {"init": True, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": True},
        GUIState.INITIALIZING: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
        GUIState.READY: {"init": True, "pre_grasp": True, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": True},
        GUIState.PREVIEW: {"init": False, "pre_grasp": False, "confirm": True, "replan": True, "cancel_preview": True, "pause": False, "resume": False, "stop": False, "object_select": False},
        GUIState.GRASPING: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": True, "resume": False, "stop": True, "object_select": False},
        GUIState.PAUSED: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": True, "stop": True, "object_select": False},
        GUIState.STOPPING: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
        GUIState.FAULT: {"init": True, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
        GUIState.CLOSING: {"init": False, "pre_grasp": False, "confirm": False, "replan": False, "cancel_preview": False, "pause": False, "resume": False, "stop": False, "object_select": False},
    }

    _TRANSITIONS = {
        GUIState.STARTUP: {GUIState.IDLE, GUIState.FAULT, GUIState.CLOSING},
        GUIState.IDLE: {GUIState.INITIALIZING, GUIState.READY, GUIState.FAULT, GUIState.CLOSING},
        GUIState.INITIALIZING: {GUIState.READY, GUIState.FAULT, GUIState.STOPPING, GUIState.CLOSING},
        GUIState.READY: {GUIState.PREVIEW, GUIState.STOPPING, GUIState.FAULT, GUIState.CLOSING, GUIState.IDLE},
        GUIState.PREVIEW: {GUIState.GRASPING, GUIState.PREVIEW, GUIState.READY, GUIState.STOPPING, GUIState.FAULT, GUIState.CLOSING},
        GUIState.GRASPING: {GUIState.PAUSED, GUIState.STOPPING, GUIState.READY, GUIState.FAULT, GUIState.CLOSING},
        GUIState.PAUSED: {GUIState.GRASPING, GUIState.STOPPING, GUIState.FAULT, GUIState.CLOSING},
        GUIState.STOPPING: {GUIState.IDLE, GUIState.READY, GUIState.FAULT, GUIState.CLOSING},
        GUIState.FAULT: {GUIState.INITIALIZING, GUIState.IDLE, GUIState.CLOSING},
        GUIState.CLOSING: set(),
    }

    def __init__(self, initial_state: GUIState = GUIState.STARTUP):
        self._current_state = initial_state
        self._previous_state = initial_state
        self._state_change_callbacks: list[Callable[[GUIState, GUIState], None]] = []

    @property
    def current_state(self) -> GUIState:
        return self._current_state

    @property
    def previous_state(self) -> GUIState:
        return self._previous_state

    def get_button_states(self) -> Dict[str, bool]:
        return dict(self._BUTTON_RULES[self._current_state])

    def can_transition_to(self, new_state: GUIState) -> bool:
        return new_state in self._TRANSITIONS[self._current_state]

    def transition_to(self, new_state: GUIState) -> bool:
        if not self.can_transition_to(new_state):
            return False
        old = self._current_state
        self._previous_state = old
        self._current_state = new_state
        for callback in self._state_change_callbacks:
            callback(old, new_state)
        return True

    def force_state(self, new_state: GUIState) -> None:
        old = self._current_state
        self._previous_state = old
        self._current_state = new_state
        for callback in self._state_change_callbacks:
            callback(old, new_state)

    def on_state_change(self, callback: Callable[[GUIState, GUIState], None]) -> None:
        self._state_change_callbacks.append(callback)


class HardwareLock:
    def __init__(self) -> None:
        self._mutex = QMutex()

    def acquire(self) -> None:
        self._mutex.lock()

    def release(self) -> None:
        self._mutex.unlock()

    def lock(self) -> None:
        self.acquire()

    def unlock(self) -> None:
        self.release()


class LogBridge(QObject):
    message_emitted = pyqtSignal(str)

    def __init__(self) -> None:
        QObject.__init__(self)
        self._buffer = io.StringIO()
        self._lock = QMutex()

    def write(self, text: str) -> int:
        self._lock.lock()
        try:
            self._buffer.write(text)
            if text and text.strip():
                stamped = f"{datetime.now().strftime('[%H:%M:%S] ')}{text.rstrip()}"
                self.message_emitted.emit(stamped)
            return len(text)
        finally:
            self._lock.unlock()

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return self._buffer.getvalue()


class VideoWorker(QObject):
    frame_ready = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, camera: Optional[CameraLike] = None, mock: bool = False) -> None:
        super().__init__()
        self._running = False
        self.mock = mock
        if mock:
            from tests.gui.mocks import MockCamera

            self.camera: Optional[CameraLike] = MockCamera()
        else:
            self.camera = camera

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                if self.camera is None:
                    raise RuntimeError("Camera is not configured")
                depth, color = self.camera.get_img()
                self.frame_ready.emit(depth, color)
            except Exception as exc:
                self.error_occurred.emit(str(exc))
            time.sleep(0.033)
        self.finished.emit()

    def stop(self) -> None:
        self._running = False


class _CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False
        self._lock = QMutex()

    def cancel(self) -> None:
        self._lock.lock()
        try:
            self._cancelled = True
        finally:
            self._lock.unlock()

    @property
    def is_cancelled(self) -> bool:
        self._lock.lock()
        try:
            return self._cancelled
        finally:
            self._lock.unlock()


class GraspWorker(QObject):
    grasp_finished = pyqtSignal(bool)
    grasp_vis_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        grasp: Optional[GraspLike],
        label: str,
        mock: bool = False,
        cancel_token: Optional[_CancellationToken] = None,
    ) -> None:
        super().__init__()
        self.grasp = grasp
        self.label = label
        self.mock = mock
        self._cancel_token = cancel_token

    def run(self) -> None:
        try:
            if self._cancel_token is not None and self._cancel_token.is_cancelled:
                self.grasp_finished.emit(False)
                return
            if self.mock:
                result = True
            else:
                if self.grasp is None:
                    raise RuntimeError("Grasp backend is not configured")
                result = self.grasp.obj_grasp(self.label, vis=False, vis_callback=self.grasp_vis_ready.emit)
            if self._cancel_token is not None and self._cancel_token.is_cancelled:
                self.grasp_finished.emit(False)
                return
            self.grasp_finished.emit(bool(result))
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.grasp_finished.emit(False)
        finally:
            self.finished.emit()


class PlanWorker(QObject):
    plan_ready = pyqtSignal(bool)
    plan_vis_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, grasp: Optional[GraspLike], label: str, mock: bool = False) -> None:
        super().__init__()
        self.grasp = grasp
        self.label = label
        self.mock = mock

    def run(self) -> None:
        try:
            if self.mock:
                self.plan_vis_ready.emit({
                    'bboxes': [[100, 100, 200, 200]],
                    'labels': [0],
                    'classes': OBJECT_LABELS,
                    'grasp_rect': [[100, 100], [200, 100], [200, 200], [100, 200]],
                    'grasp_center': (150, 150),
                    'crop_offset': 80,
                    'target_label': self.label,
                })
                result = True
            else:
                if self.grasp is None:
                    raise RuntimeError("Grasp backend is not configured")
                result = self.grasp.plan_grasp(self.label, vis_callback=self.plan_vis_ready.emit)
            self.plan_ready.emit(bool(result))
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.plan_ready.emit(False)
        finally:
            self.finished.emit()


class ExecuteWorker(QObject):
    execute_finished = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        grasp: Optional[GraspLike],
        mock: bool = False,
        resume_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__()
        self.grasp = grasp
        self.mock = mock
        self._resume_event = resume_event
        self._cancel_event = cancel_event

    def run(self) -> None:
        try:
            if self._cancel_event is not None and self._cancel_event.is_set():
                self.execute_finished.emit(False)
                return
            if self.mock:
                time.sleep(0.5)
                result = True
            else:
                if self.grasp is None:
                    raise RuntimeError("Grasp backend is not configured")
                result = self.grasp.execute_grasp(
                    resume_event=self._resume_event,
                    cancel_event=self._cancel_event,
                )
            if self._cancel_event is not None and self._cancel_event.is_set():
                self.execute_finished.emit(False)
                return
            self.execute_finished.emit(bool(result))
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.execute_finished.emit(False)
        finally:
            self.finished.emit()


class InitWorker(QObject):
    init_finished = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, grasp: Optional[GraspLike], mock: bool = False) -> None:
        super().__init__()
        self.grasp = grasp
        self.mock = mock

    def run(self) -> None:
        try:
            if not self.mock and self.grasp is not None:
                self.grasp.init_gripper()
            self.init_finished.emit(True)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.init_finished.emit(False)
        finally:
            self.finished.emit()


class StopWorker(QObject):
    stop_finished = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, grasp: Optional[GraspLike], mock: bool = False) -> None:
        super().__init__()
        self.grasp = grasp
        self.mock = mock

    def run(self) -> None:
        try:
            if self.grasp is not None:
                stop_fn = getattr(self.grasp.robot, "rm_set_arm_stop", None)
                if callable(stop_fn):
                    stop_fn()
            self.stop_finished.emit(True)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.stop_finished.emit(False)
        finally:
            self.finished.emit()


def cv2_to_qpixmap(color_image: Optional[np.ndarray]) -> QPixmap:
    if color_image is None:
        return QPixmap()
    if not isinstance(color_image, np.ndarray) or color_image.size == 0:
        return QPixmap()
    if color_image.ndim != 3 or color_image.shape[2] != 3:
        return QPixmap()

    height, width, channels = color_image.shape
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    bytes_per_line = channels * width
    qimage = QImage(rgb_image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class VideoWidget(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")
        self._overlay: Optional[dict] = None

    def set_overlay(self, vis_data: object) -> None:
        if vis_data is None:
            return
        self._overlay = vis_data

    def clear_overlay(self) -> None:
        self._overlay = None

    def update_frame(self, depth: np.ndarray, color: Optional[np.ndarray]) -> None:
        if color is None or not isinstance(color, np.ndarray) or color.size == 0:
            return
        display = color.copy()
        cv2.circle(display, (320, 240), 5, (0, 0, 255), -1)
        if self._overlay is not None:
            self._draw_overlay(display)
        pixmap = cv2_to_qpixmap(display)
        if pixmap.isNull():
            return
        target_size = self.size() if self.width() > 0 and self.height() > 0 else pixmap.size()
        self.setPixmap(
            pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _draw_overlay(self, display: np.ndarray) -> None:
        data = self._overlay
        offset_x = data.get('crop_offset', 0)
        classes = data.get('classes', [])
        target_label = data.get('target_label', '')

        bboxes = data.get('bboxes', [])
        det_labels = data.get('labels', [])
        for i, bbox in enumerate(bboxes):
            x1 = int(bbox[0] + offset_x)
            y1 = int(bbox[1])
            x2 = int(bbox[2] + offset_x)
            y2 = int(bbox[3])
            label_idx = int(det_labels[i]) if i < len(det_labels) else -1
            class_name = classes[label_idx] if 0 <= label_idx < len(classes) else ''
            is_target = class_name == target_label
            color = (0, 255, 0) if is_target else (128, 128, 128)
            thickness = 2 if is_target else 1
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            if class_name:
                cv2.putText(display, class_name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        grasp_rect = data.get('grasp_rect')
        if grasp_rect is not None:
            pts = np.array([[p[0] + offset_x, p[1]] for p in grasp_rect], dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)

        center = data.get('grasp_center')
        if center is not None:
            cx = center[0] + offset_x
            cy = center[1]
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(display, (cx, cy), 8, (0, 255, 255), 2)


class ControlPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.btn_init = QPushButton("初始化")
        self.btn_pre_grasp = QPushButton("预抓取")
        self.btn_confirm = QPushButton("确认抓取")
        self.btn_replan = QPushButton("重新预抓取")
        self.btn_cancel_preview = QPushButton("取消")
        self.btn_pause = QPushButton("暂停")
        self.btn_resume = QPushButton("继续抓取")
        self.btn_stop = QPushButton("紧急停止")
        self.combo_objects = QComboBox()
        self.combo_objects.addItems(OBJECT_LABELS)

        layout.addWidget(self.btn_init)
        layout.addWidget(QLabel("物体选择:"))
        layout.addWidget(self.combo_objects)
        layout.addWidget(self.btn_pre_grasp)
        layout.addWidget(self.btn_confirm)
        layout.addWidget(self.btn_replan)
        layout.addWidget(self.btn_cancel_preview)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_resume)
        layout.addWidget(self.btn_stop)
        layout.addStretch(1)


class LogWidget(QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("font-family: monospace;")

    def append_log(self, message: str) -> None:
        self.append(message)
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class StatusWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.speed_label = QLabel("机械臂速度: --")
        self.object_label = QLabel("选中物体: --")
        layout.addWidget(self.speed_label)
        layout.addWidget(self.object_label)
        layout.addStretch(1)

    def update_status(self, speed: object, object_name: str) -> None:
        self.speed_label.setText(f"机械臂速度: {speed}")
        self.object_label.setText(f"选中物体: {object_name}")


class GraspGUI(QMainWindow):
    def __init__(self, grasp: Optional[GraspLike] = None, mock: bool = False) -> None:
        super().__init__()
        self.grasp = grasp
        self.mock = mock

        self.state_machine = StateMachine(initial_state=GUIState.STARTUP)
        self.hw_lock = HardwareLock()

        self.video_thread: Optional[QThread] = None
        self.video_worker: Optional[VideoWorker] = None
        self.init_thread: Optional[QThread] = None
        self.init_worker: Optional[InitWorker] = None
        self.stop_thread: Optional[QThread] = None
        self.stop_worker: Optional[StopWorker] = None
        self.plan_thread: Optional[QThread] = None
        self.plan_worker: Optional[PlanWorker] = None
        self.execute_thread: Optional[QThread] = None
        self.execute_worker: Optional[ExecuteWorker] = None
        self._resume_event: Optional[threading.Event] = None
        self._cancel_event: Optional[threading.Event] = None
        self._closing = False

        self._original_stdout = sys.stdout
        self.log_bridge = LogBridge()

        self._setup_ui()
        self._setup_connections()

        self.log_bridge.message_emitted.connect(self.log_widget.append_log)
        sys.stdout = self.log_bridge

        self.state_machine.force_state(GUIState.IDLE)
        self._sync_controls_to_state()
        self._on_object_changed(self.control_panel.combo_objects.currentText())

    def _setup_ui(self) -> None:
        self.setWindowTitle("机械臂抓取控制系统")
        self.resize(1280, 720)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)

        self.control_panel = ControlPanel()
        self.video_widget = VideoWidget()
        self.log_widget = LogWidget()
        self.status_widget = StatusWidget()

        layout.addWidget(self.control_panel, 0, 0)
        layout.addWidget(self.video_widget, 0, 1)
        layout.addWidget(self.log_widget, 1, 0)
        layout.addWidget(self.status_widget, 1, 1)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        layout.setRowStretch(0, 2)
        layout.setRowStretch(1, 1)

    def _setup_connections(self) -> None:
        self.control_panel.btn_init.clicked.connect(self._on_init)
        self.control_panel.btn_pre_grasp.clicked.connect(self._on_pre_grasp)
        self.control_panel.btn_confirm.clicked.connect(self._on_confirm_grasp)
        self.control_panel.btn_replan.clicked.connect(self._on_replan)
        self.control_panel.btn_cancel_preview.clicked.connect(self._on_cancel_preview)
        self.control_panel.btn_pause.clicked.connect(self._on_pause)
        self.control_panel.btn_resume.clicked.connect(self._on_resume)
        self.control_panel.btn_stop.clicked.connect(self._on_stop)
        self.control_panel.combo_objects.currentTextChanged.connect(self._on_object_changed)
        self.state_machine.on_state_change(lambda _old, _new: self._sync_controls_to_state())

    def _sync_controls_to_state(self) -> None:
        states = self.state_machine.get_button_states()
        self.control_panel.btn_init.setEnabled(states["init"])
        self.control_panel.btn_pre_grasp.setEnabled(states["pre_grasp"])
        self.control_panel.btn_confirm.setEnabled(states["confirm"])
        self.control_panel.btn_replan.setEnabled(states["replan"])
        self.control_panel.btn_cancel_preview.setEnabled(states["cancel_preview"])
        self.control_panel.btn_pause.setEnabled(states["pause"])
        self.control_panel.btn_resume.setEnabled(states["resume"])
        self.control_panel.btn_stop.setEnabled(states["stop"])
        self.control_panel.combo_objects.setEnabled(states["object_select"])

    def start_video(self) -> None:
        if self.video_thread is not None and self.video_thread.isRunning():
            return

        camera = None
        if not self.mock and self.grasp is not None:
            camera = self.grasp.camera

        self.video_thread = QThread(self)
        self.video_worker = VideoWorker(camera=camera, mock=(self.mock or camera is None))
        self.video_worker.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.video_worker.run)
        self.video_worker.frame_ready.connect(self.video_widget.update_frame)
        self.video_worker.error_occurred.connect(lambda msg: print(f"[VideoWorker] {msg}"))
        self.video_worker.finished.connect(self.video_thread.quit)
        self.video_thread.start()

    def _on_init(self) -> None:
        if self.state_machine.current_state not in {GUIState.IDLE, GUIState.FAULT}:
            return
        self.state_machine.force_state(GUIState.INITIALIZING)
        print("初始化...")

        self.init_thread = QThread(self)
        self.init_worker = InitWorker(self.grasp, mock=self.mock)
        self.init_worker.moveToThread(self.init_thread)
        self.init_thread.started.connect(self.init_worker.run)
        self.init_worker.init_finished.connect(self._on_init_finished)
        self.init_worker.error_occurred.connect(lambda msg: print(f"初始化错误: {msg}"))
        self.init_worker.finished.connect(self.init_thread.quit)
        self.init_thread.start()

    def _on_init_finished(self, success: bool) -> None:
        if self._closing:
            return
        if success:
            print("初始化完成")
            self.state_machine.force_state(GUIState.READY)
        else:
            print("初始化失败")
            self.state_machine.force_state(GUIState.FAULT)

    def _on_pre_grasp(self) -> None:
        if self.state_machine.current_state != GUIState.READY:
            return
        label = self.control_panel.combo_objects.currentText()
        self.state_machine.force_state(GUIState.PREVIEW)
        print(f"预抓取: {label}")

        self.plan_thread = QThread(self)
        self.plan_worker = PlanWorker(self.grasp, label, mock=self.mock)
        self.plan_worker.moveToThread(self.plan_thread)
        self.plan_thread.started.connect(self.plan_worker.run)
        self.plan_worker.plan_ready.connect(self._on_plan_ready)
        self.plan_worker.plan_vis_ready.connect(self._on_grasp_vis_ready)
        self.plan_worker.error_occurred.connect(lambda msg: print(f"规划错误: {msg}"))
        self.plan_worker.finished.connect(self.plan_thread.quit)
        self.plan_thread.start()

    def _on_plan_ready(self, success: bool) -> None:
        if self._closing:
            return
        if success:
            print("预抓取规划完成，请确认或重新规划")
        else:
            print("预抓取规划失败")
            self.video_widget.clear_overlay()
            self.state_machine.force_state(GUIState.READY)

    def _on_confirm_grasp(self) -> None:
        if self.state_machine.current_state != GUIState.PREVIEW:
            return
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._cancel_event = threading.Event()
        self.state_machine.force_state(GUIState.GRASPING)
        print("开始执行抓取...")

        self.execute_thread = QThread(self)
        self.execute_worker = ExecuteWorker(
            self.grasp, mock=self.mock,
            resume_event=self._resume_event,
            cancel_event=self._cancel_event,
        )
        self.execute_worker.moveToThread(self.execute_thread)
        self.execute_thread.started.connect(self.execute_worker.run)
        self.execute_worker.execute_finished.connect(self._on_execute_finished)
        self.execute_worker.error_occurred.connect(lambda msg: print(f"执行错误: {msg}"))
        self.execute_worker.finished.connect(self.execute_thread.quit)
        self.execute_thread.start()

    def _on_execute_finished(self, success: bool) -> None:
        if self._closing:
            return
        if self._cancel_event is not None and self._cancel_event.is_set():
            return
        print("抓取成功!" if success else "抓取失败!")
        self.video_widget.clear_overlay()
        self.state_machine.force_state(GUIState.READY if success else GUIState.FAULT)

    def _on_replan(self) -> None:
        if self.state_machine.current_state != GUIState.PREVIEW:
            return
        self.video_widget.clear_overlay()
        label = self.control_panel.combo_objects.currentText()
        print(f"重新预抓取: {label}")

        self.plan_thread = QThread(self)
        self.plan_worker = PlanWorker(self.grasp, label, mock=self.mock)
        self.plan_worker.moveToThread(self.plan_thread)
        self.plan_thread.started.connect(self.plan_worker.run)
        self.plan_worker.plan_ready.connect(self._on_plan_ready)
        self.plan_worker.plan_vis_ready.connect(self._on_grasp_vis_ready)
        self.plan_worker.error_occurred.connect(lambda msg: print(f"规划错误: {msg}"))
        self.plan_worker.finished.connect(self.plan_thread.quit)
        self.plan_thread.start()

    def _on_cancel_preview(self) -> None:
        if self.state_machine.current_state != GUIState.PREVIEW:
            return
        self.video_widget.clear_overlay()
        print("取消预抓取")
        self.state_machine.force_state(GUIState.READY)

    def _on_pause(self) -> None:
        if self.state_machine.current_state != GUIState.GRASPING:
            return
        if self._resume_event is not None:
            self._resume_event.clear()
        self.state_machine.force_state(GUIState.PAUSED)
        print("抓取已暂停")

    def _on_resume(self) -> None:
        if self.state_machine.current_state != GUIState.PAUSED:
            return
        if self._resume_event is not None:
            self._resume_event.set()
        self.state_machine.force_state(GUIState.GRASPING)
        print("继续抓取...")

    def _on_grasp_vis_ready(self, vis_data: object) -> None:
        if self._closing:
            return
        self.video_widget.set_overlay(vis_data)

    def _on_stop(self) -> None:
        self.state_machine.force_state(GUIState.STOPPING)
        print("紧急停止!")
        self.video_widget.clear_overlay()

        if self._resume_event is not None:
            self._resume_event.set()
        if self._cancel_event is not None:
            self._cancel_event.set()

        self.stop_thread = QThread(self)
        self.stop_worker = StopWorker(self.grasp, mock=self.mock)
        self.stop_worker.moveToThread(self.stop_thread)
        self.stop_thread.started.connect(self.stop_worker.run)
        self.stop_worker.stop_finished.connect(self._on_stop_finished)
        self.stop_worker.error_occurred.connect(lambda msg: print(f"停止错误: {msg}"))
        self.stop_worker.finished.connect(self.stop_thread.quit)
        self.stop_thread.start()

    def _on_stop_finished(self, success: bool) -> None:
        if self._closing:
            return
        self._resume_event = None
        self._cancel_event = None
        self.state_machine.force_state(GUIState.IDLE if success else GUIState.FAULT)

    def _on_object_changed(self, object_name: str) -> None:
        speed = getattr(self.grasp, "robot_speed", 30)
        self.status_widget.update_status(speed=speed, object_name=object_name)

    def closeEvent(self, a0) -> None:  # noqa: N802
        self._closing = True

        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._resume_event is not None:
            self._resume_event.set()

        self.state_machine.force_state(GUIState.CLOSING)

        if self.video_worker is not None:
            self.video_worker.stop()
        for thread in (self.execute_thread, self.plan_thread, self.stop_thread, self.init_thread):
            if thread is not None:
                thread.quit()
                thread.wait(3000)

        if self.grasp is not None:
            stop_robot = getattr(self.grasp.robot, "rm_set_arm_stop", None)
            if callable(stop_robot):
                try:
                    stop_robot()
                except Exception:
                    pass
            stop_camera = getattr(self.grasp.camera, "stop", None)
            if callable(stop_camera):
                try:
                    stop_camera()
                except Exception:
                    pass

        if self.video_thread is not None:
            self.video_thread.quit()
            self.video_thread.wait(2000)

        sys.stdout = self._original_stdout
        super().closeEvent(a0)


class _hardware_guard:
    def __init__(self, lock: HardwareLock):
        self._lock = lock

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    if args.mock:
        from tests.gui.mocks import MockGrasp

        grasp = MockGrasp(hardware=False)
    else:
        from grasp_zy_zhiyuan1215 import Grasp

        grasp = Grasp(hardware=True)

    window = GraspGUI(grasp=grasp, mock=args.mock)
    window.show()
    window.start_video()
    return app.exec_()


__all__ = [
    "GUIState",
    "StateMachine",
    "HardwareLock",
    "LogBridge",
    "VideoWorker",
    "GraspWorker",
    "PlanWorker",
    "ExecuteWorker",
    "InitWorker",
    "StopWorker",
    "_CancellationToken",
    "cv2_to_qpixmap",
    "VideoWidget",
    "ControlPanel",
    "LogWidget",
    "StatusWidget",
    "GraspGUI",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
