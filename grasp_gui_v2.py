from __future__ import annotations

import argparse
import io
import sys
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

    def obj_grasp(self, label: str, vis: bool = False) -> bool: ...

    def init_gripper(self) -> None: ...


class GUIState(Enum):
    STARTUP = auto()
    IDLE = auto()
    INITIALIZING = auto()
    READY = auto()
    GRASPING = auto()
    STOPPING = auto()
    FAULT = auto()
    CLOSING = auto()


@dataclass
class ButtonState:
    button_id: str
    enabled: bool


class StateMachine:
    _BUTTON_RULES: Dict[GUIState, Dict[str, bool]] = {
        GUIState.STARTUP: {"init": False, "start_grasp": False, "stop": False, "object_select": False},
        GUIState.IDLE: {"init": True, "start_grasp": False, "stop": False, "object_select": True},
        GUIState.INITIALIZING: {"init": False, "start_grasp": False, "stop": False, "object_select": False},
        GUIState.READY: {"init": True, "start_grasp": True, "stop": True, "object_select": True},
        GUIState.GRASPING: {"init": False, "start_grasp": False, "stop": True, "object_select": False},
        GUIState.STOPPING: {"init": False, "start_grasp": False, "stop": False, "object_select": False},
        GUIState.FAULT: {"init": True, "start_grasp": False, "stop": False, "object_select": False},
        GUIState.CLOSING: {"init": False, "start_grasp": False, "stop": False, "object_select": False},
    }

    _TRANSITIONS = {
        GUIState.STARTUP: {GUIState.IDLE, GUIState.FAULT, GUIState.CLOSING},
        GUIState.IDLE: {GUIState.INITIALIZING, GUIState.READY, GUIState.FAULT, GUIState.CLOSING},
        GUIState.INITIALIZING: {GUIState.READY, GUIState.FAULT, GUIState.STOPPING, GUIState.CLOSING},
        GUIState.READY: {GUIState.GRASPING, GUIState.STOPPING, GUIState.FAULT, GUIState.CLOSING, GUIState.IDLE},
        GUIState.GRASPING: {GUIState.STOPPING, GUIState.READY, GUIState.FAULT, GUIState.CLOSING},
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

            self.camera = MockCamera()
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


class GraspWorker(QObject):
    grasp_finished = pyqtSignal(bool)
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
                result = True
            else:
                if self.grasp is None:
                    raise RuntimeError("Grasp backend is not configured")
                result = self.grasp.obj_grasp(self.label, vis=False)
            self.grasp_finished.emit(bool(result))
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.grasp_finished.emit(False)
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

    def update_frame(self, depth: np.ndarray, color: Optional[np.ndarray]) -> None:
        if color is None or not isinstance(color, np.ndarray) or color.size == 0:
            return
        display = color.copy()
        cv2.circle(display, (320, 240), 5, (0, 0, 255), -1)
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


class ControlPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.btn_init = QPushButton("初始化")
        self.btn_start = QPushButton("开始抓取")
        self.btn_stop = QPushButton("紧急停止")
        self.combo_objects = QComboBox()
        self.combo_objects.addItems(OBJECT_LABELS)

        layout.addWidget(self.btn_init)
        layout.addWidget(QLabel("物体选择:"))
        layout.addWidget(self.combo_objects)
        layout.addWidget(self.btn_start)
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
        self.grasp_thread: Optional[QThread] = None
        self.grasp_worker: Optional[GraspWorker] = None
        self._stop_requested = False
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
        self.control_panel.btn_start.clicked.connect(self._on_start_grasp)
        self.control_panel.btn_stop.clicked.connect(self._on_stop)
        self.control_panel.combo_objects.currentTextChanged.connect(self._on_object_changed)
        self.state_machine.on_state_change(lambda _old, _new: self._sync_controls_to_state())

    def _sync_controls_to_state(self) -> None:
        states = self.state_machine.get_button_states()
        self.control_panel.btn_init.setEnabled(states["init"])
        self.control_panel.btn_start.setEnabled(states["start_grasp"])
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
        try:
            if self.grasp is not None:
                self.grasp.init_gripper()
            print("初始化完成")
            self.state_machine.force_state(GUIState.READY)
        except Exception as exc:
            print(f"初始化失败: {exc}")
            self.state_machine.force_state(GUIState.FAULT)

    def _on_start_grasp(self) -> None:
        if self.state_machine.current_state != GUIState.READY:
            print("当前状态不允许开始抓取")
            return
        label = self.control_panel.combo_objects.currentText()
        self._stop_requested = False
        self.state_machine.force_state(GUIState.GRASPING)
        print(f"开始抓取: {label}")

        self.grasp_thread = QThread(self)
        self.grasp_worker = GraspWorker(self.grasp, label, mock=self.mock)
        self.grasp_worker.moveToThread(self.grasp_thread)
        self.grasp_thread.started.connect(self.grasp_worker.run)
        self.grasp_worker.grasp_finished.connect(self._on_grasp_finished)
        self.grasp_worker.error_occurred.connect(lambda msg: print(f"抓取错误: {msg}"))
        self.grasp_worker.finished.connect(self.grasp_thread.quit)
        self.grasp_thread.start()

    def _on_grasp_finished(self, success: bool) -> None:
        if self._closing or self._stop_requested or self.state_machine.current_state == GUIState.STOPPING:
            return
        print("抓取成功!" if success else "抓取失败!")
        self.state_machine.force_state(GUIState.READY if success else GUIState.FAULT)

    def _on_stop(self) -> None:
        self._stop_requested = True
        self.state_machine.force_state(GUIState.STOPPING)
        print("紧急停止!")
        try:
            if self.grasp is not None:
                with _hardware_guard(self.hw_lock):
                    self.grasp.robot.rm_set_arm_stop()
            self.state_machine.force_state(GUIState.IDLE)
        except Exception as exc:
            print(f"停止失败: {exc}")
            self.state_machine.force_state(GUIState.FAULT)

    def _on_object_changed(self, object_name: str) -> None:
        speed = getattr(self.grasp, "robot_speed", 30)
        self.status_widget.update_status(speed=speed, object_name=object_name)

    def closeEvent(self, a0) -> None:  # noqa: N802
        self._closing = True
        self._stop_requested = True
        self.state_machine.force_state(GUIState.CLOSING)
        if self.video_worker is not None:
            self.video_worker.stop()
        if self.grasp is not None:
            stop_robot = getattr(self.grasp.robot, "rm_set_arm_stop", None)
            if callable(stop_robot):
                try:
                    with _hardware_guard(self.hw_lock):
                        stop_robot()
                except Exception:
                    pass
        if self.grasp is not None:
            stop_camera = getattr(self.grasp.camera, "stop", None)
            if callable(stop_camera):
                stop_camera()
        if self.video_thread is not None:
            self.video_thread.quit()
            self.video_thread.wait(2000)
        if self.grasp_thread is not None:
            self.grasp_thread.quit()
            self.grasp_thread.wait(2000)

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
