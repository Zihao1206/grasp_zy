"""Hand-Eye Calibration GUI — 机械臂手眼标定工具

与 grasp_gui_v2.py 共享 QThread+Worker / Protocol / LogBridge / StateMachine 架构。
支持 ``--mock`` 命令行参数进行无硬件自动测试。
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QMutex, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Protocol interfaces
# ---------------------------------------------------------------------------

class CameraLike(Protocol):
    """Abstract camera with depth + colour and intrinsics retrieval."""

    def get_img(self) -> Tuple[np.ndarray, np.ndarray]: ...

    def get_intrinsics(self) -> np.ndarray:
        """Return 3×3 intrinsic matrix."""
        ...

    def stop(self) -> None: ...


class RobotLike(Protocol):
    """Abstract robot arm providing end-effector pose query."""

    def rm_get_current_arm_state(self) -> Tuple[int, dict]: ...
    def rm_set_arm_stop(self) -> int: ...


# ---------------------------------------------------------------------------
# GUI state machine
# ---------------------------------------------------------------------------

class GUIState(Enum):
    STARTUP = auto()
    IDLE = auto()
    CAMERA_INIT = auto()
    READY = auto()
    COLLECTING = auto()
    CALCULATING = auto()
    FAULT = auto()
    CLOSING = auto()


BUTTON_IDS = [
    "get_intrinsics", "choose_dir", "save", "undo", "calculate",
    "calib_type", "resolution", "format", "fps",
]


class StateMachine:
    _BUTTON_RULES: Dict[GUIState, Dict[str, bool]] = {
        GUIState.STARTUP:    {b: False for b in BUTTON_IDS},
        GUIState.IDLE:       {"get_intrinsics": False, "choose_dir": True,  "save": False, "undo": False, "calculate": False, "calib_type": True,  "resolution": True,  "format": True,  "fps": True},
        GUIState.CAMERA_INIT:{"get_intrinsics": False, "choose_dir": False, "save": False, "undo": False, "calculate": False, "calib_type": False, "resolution": False, "format": False, "fps": False},
        GUIState.READY:      {"get_intrinsics": True,  "choose_dir": True,  "save": True,  "undo": True,  "calculate": True,  "calib_type": False, "resolution": False, "format": False, "fps": False},
        GUIState.COLLECTING: {"get_intrinsics": False, "choose_dir": False, "save": False, "undo": False, "calculate": False, "calib_type": False, "resolution": False, "format": False, "fps": False},
        GUIState.CALCULATING:{"get_intrinsics": False, "choose_dir": False, "save": False, "undo": False, "calculate": False, "calib_type": False, "resolution": False, "format": False, "fps": False},
        GUIState.FAULT:      {"get_intrinsics": False, "choose_dir": True,  "save": False, "undo": False, "calculate": False, "calib_type": True,  "resolution": True,  "format": True,  "fps": True},
        GUIState.CLOSING:    {b: False for b in BUTTON_IDS},
    }

    _TRANSITIONS = {
        GUIState.STARTUP:     {GUIState.IDLE, GUIState.FAULT, GUIState.CLOSING},
        GUIState.IDLE:        {GUIState.CAMERA_INIT, GUIState.READY, GUIState.FAULT, GUIState.CLOSING},
        GUIState.CAMERA_INIT: {GUIState.READY, GUIState.FAULT, GUIState.CLOSING},
        GUIState.READY:       {GUIState.COLLECTING, GUIState.CALCULATING, GUIState.IDLE, GUIState.FAULT, GUIState.CLOSING},
        GUIState.COLLECTING:  {GUIState.READY, GUIState.FAULT, GUIState.CLOSING},
        GUIState.CALCULATING: {GUIState.READY, GUIState.FAULT, GUIState.CLOSING},
        GUIState.FAULT:       {GUIState.IDLE, GUIState.CLOSING},
        GUIState.CLOSING:     set(),
    }

    def __init__(self, initial: GUIState = GUIState.STARTUP) -> None:
        self._state = initial
        self._prev = initial
        self._cbs: List[Callable[[GUIState, GUIState], None]] = []

    @property
    def current(self) -> GUIState:
        return self._state

    @property
    def previous(self) -> GUIState:
        return self._prev

    def get_button_states(self) -> Dict[str, bool]:
        return dict(self._BUTTON_RULES[self._state])

    def can_transition(self, target: GUIState) -> bool:
        return target in self._TRANSITIONS[self._state]

    def transition(self, target: GUIState) -> bool:
        if not self.can_transition(target):
            return False
        self._fire(target)
        return True

    def force(self, target: GUIState) -> None:
        self._fire(target)

    def on_change(self, cb: Callable[[GUIState, GUIState], None]) -> None:
        self._cbs.append(cb)

    def _fire(self, target: GUIState) -> None:
        old = self._state
        self._prev = old
        self._state = target
        for cb in self._cbs:
            cb(old, target)


# ---------------------------------------------------------------------------
# Log bridge (stdout → QTextEdit)
# ---------------------------------------------------------------------------

class LogBridge(QObject):
    message_emitted = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._buf = io.StringIO()
        self._lock = QMutex()

    def write(self, text: str) -> int:
        self._lock.lock()
        try:
            self._buf.write(text)
            if text and text.strip():
                stamped = f"{datetime.now().strftime('[%H:%M:%S]')} {text.rstrip()}"
                self.message_emitted.emit(stamped)
            return len(text)
        finally:
            self._lock.unlock()

    def flush(self) -> None:
        return None


# ---------------------------------------------------------------------------
# cv2 → QPixmap helper
# ---------------------------------------------------------------------------

def cv2_to_qpixmap(bgr: Optional[np.ndarray]) -> QPixmap:
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
        return QPixmap()
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return QPixmap()
    h, w, ch = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# Workers (all QObject, run on QThread)
# ---------------------------------------------------------------------------

class VideoWorker(QObject):
    frame_ready = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, camera: Optional[CameraLike] = None, mock: bool = False) -> None:
        super().__init__()
        self._running = False
        self.mock = mock
        if mock:
            self.camera: Optional[CameraLike] = _MockCamera()
        else:
            self.camera = camera

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                if self.camera is None:
                    raise RuntimeError("Camera not available")
                depth, color = self.camera.get_img()
                self.frame_ready.emit(depth, color)
            except Exception as exc:
                self.error_occurred.emit(str(exc))
            time.sleep(0.033)
        self.finished.emit()

    def stop(self) -> None:
        self._running = False


class CameraInitWorker(QObject):
    """Initialise RealSense pipeline on a background thread."""
    init_done = pyqtSignal(bool, object)  # success, camera_or_None
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, width: int, height: int, fmt: str, fps: int, mock: bool = False) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.fmt = fmt
        self.fps = fps
        self.mock = mock

    def run(self) -> None:
        try:
            if self.mock:
                cam = _MockCamera(self.width, self.height)
                self.init_done.emit(True, cam)
                return
            import pyrealsense2 as rs
            pipeline = rs.pipeline()
            config = rs.config()
            rs_fmt = rs.format.bgr8 if self.fmt == "BGR8" else rs.format.rgb8
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs_fmt, self.fps)
            profile = pipeline.start(config)
            align = rs.align(rs.stream.color)
            for _ in range(5):
                pipeline.wait_for_frames()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            cam = _RealCamera(pipeline, align, intr, self.width, self.height)
            self.init_done.emit(True, cam)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.init_done.emit(False, None)
        finally:
            self.finished.emit()


class IntrinsicsWorker(QObject):
    intrinsics_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, camera: Optional[CameraLike], mock: bool = False) -> None:
        super().__init__()
        self.camera = camera
        self.mock = mock

    def run(self) -> None:
        try:
            if self.mock or self.camera is None:
                mat = np.array([[604.335, 0, 316.187],
                                [0, 604.404, 248.611],
                                [0, 0, 1]])
            else:
                mat = self.camera.get_intrinsics()
            self.intrinsics_ready.emit(mat)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()


class SaveDataWorker(QObject):
    """Capture image + pose, save to disk, then run chessboard detection.

    Emits ``preview_ready`` with a BGR ``np.ndarray`` rendered with detected
    corners (or the raw frame if detection fails). Detection failure is
    reported via ``log_message`` but does not abort the save flow.
    """

    save_done = pyqtSignal(bool, int)
    preview_ready = pyqtSignal(object, bool)  # rendered_image, detected_ok
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        camera: Optional[CameraLike],
        robot: Optional[RobotLike],
        save_dir: str,
        index: int,
        rows: int,
        cols: int,
        mock: bool = False,
    ) -> None:
        super().__init__()
        self.camera = camera
        self.robot = robot
        self.save_dir = save_dir
        self.index = index
        # OpenCV: patternSize = Size(inner_corners_per_row, inner_corners_per_column).
        # 与 out_of_hand_homogeneous_matrix/main.py 中 (XX, YY) 一致：默认 12×9 方格 → (11, 8)。
        self._inner_row = rows - 1
        self._inner_col = cols - 1
        self.mock = mock

    def run(self) -> None:
        try:
            if self.mock:
                color = self._mock_chessboard_image()
                pose = [-0.226, -0.003, 0.523, -0.01, 0.028, 2.65]
            else:
                if self.camera is None:
                    raise RuntimeError("Camera not available")
                _, color = self.camera.get_img()
                if self.robot is None:
                    raise RuntimeError("Robot not available")
                ret, state = self.robot.rm_get_current_arm_state()
                if ret != 0:
                    raise RuntimeError(f"获取机械臂位姿失败, 错误码: {ret}")
                pose = state.get("pose", [0.0] * 6)

            img_path = os.path.join(self.save_dir, f"{self.index}.jpg")
            cv2.imwrite(img_path, color)

            pose_file = os.path.join(self.save_dir, "poses.txt")
            with open(pose_file, "a") as fp:
                fp.write(",".join(str(v) for v in pose) + "\n")

            preview, detected = self._render_chessboard(color)
            self.preview_ready.emit(preview, detected)

            self.save_done.emit(True, self.index)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.save_done.emit(False, self.index)
        finally:
            self.finished.emit()

    def _render_chessboard(self, color: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Run ``findChessboardCorners`` and overlay them on a copy.

        Returns ``(rendered_image, success)``. On failure, returns the raw
        frame with a banner so the operator can still see the captured image.
        """
        display = color.copy()
        try:
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, (self._inner_row, self._inner_col),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH,
            )
            if ret:
                cv2.drawChessboardCorners(
                    display, (self._inner_row, self._inner_col), corners, ret)
                cv2.putText(display, f"#{self.index} OK", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                self.log_message.emit(
                    f"图片 {self.index} 角点检测成功 ({self._inner_row}x{self._inner_col})")
                return display, True
            self.log_message.emit(
                f"⚠ 图片 {self.index} 未检测到 {self._inner_row}x{self._inner_col} 角点 "
                f"(图片已保存，但建议撤销并重新拍摄)")
            cv2.putText(display, f"#{self.index} NO CORNERS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return display, False
        except Exception as exc:
            self.log_message.emit(f"⚠ 角点检测异常: {exc}")
            return display, False

    @staticmethod
    def _mock_chessboard_image() -> np.ndarray:
        """Render a deterministic chessboard for mock mode (so detection succeeds)."""
        h, w = 480, 640
        img = np.full((h, w, 3), 230, dtype=np.uint8)
        sq = 35
        rows, cols = 9, 12
        x0 = (w - cols * sq) // 2
        y0 = (h - rows * sq) // 2
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    cv2.rectangle(img,
                                  (x0 + c * sq, y0 + r * sq),
                                  (x0 + (c + 1) * sq, y0 + (r + 1) * sq),
                                  (20, 20, 20), -1)
        return img


class CalibrationWorker(QObject):
    calc_done = pyqtSignal(bool, object)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        images_dir: str,
        poses_file: str,
        intrinsics: np.ndarray,
        rows: int,
        cols: int,
        spacing: float,
        save_dir: str,
        mock: bool = False,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.poses_file = poses_file
        self.intrinsics = intrinsics
        # 与 main.py 中 XX=行方格数-1, YY=列方格数-1 一致（默认 12×9 → 11×8）
        self.xx = rows - 1
        self.yy = cols - 1
        self.spacing = spacing
        self.save_dir = save_dir
        self.mock = mock

    def run(self) -> None:
        try:
            if self.mock:
                R_mat = np.eye(3)
                t_vec = np.array([[0.1], [0.2], [0.3]])
                homo = np.eye(4)
                homo[:3, :3] = R_mat
                homo[:3, 3] = t_vec[:, 0]
                time.sleep(0.5)
                self.log_message.emit("旋转矩阵 (Rotation Matrix):")
                self.log_message.emit(f"\n{np.array2string(R_mat, precision=8, suppress_small=True)}")
                self.log_message.emit("平移向量 (Translation Vector):")
                self.log_message.emit(f"\n{np.array2string(t_vec, precision=8, suppress_small=True)}")
                self.log_message.emit("齐次变换矩阵 (Homogeneous Matrix 4×4):")
                self.log_message.emit(f"\n{np.array2string(homo, precision=8, suppress_small=True)}")
                self._save_matrix(homo)
                self.calc_done.emit(True, homo)
                return

            R_mat, t_vec = self._compute()
            homo = np.eye(4)
            homo[:3, :3] = R_mat
            homo[:3, 3] = t_vec[:, 0]

            self.log_message.emit("旋转矩阵 (Rotation Matrix):")
            self.log_message.emit(f"\n{np.array2string(R_mat, precision=8, suppress_small=True)}")
            self.log_message.emit("平移向量 (Translation Vector):")
            self.log_message.emit(f"\n{np.array2string(t_vec, precision=8, suppress_small=True)}")
            self.log_message.emit("齐次变换矩阵 (Homogeneous Matrix 4×4):")
            self.log_message.emit(f"\n{np.array2string(homo, precision=8, suppress_small=True)}")
            self._save_matrix(homo)
            self.calc_done.emit(True, homo)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.calc_done.emit(False, None)
        finally:
            self.finished.emit()

    # ---- algorithm (ported from out_of_hand_homogeneous_matrix/main.py) ----

    def _compute(self) -> Tuple[np.ndarray, np.ndarray]:
        objp = np.zeros((self.xx * self.yy, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.xx, 0:self.yy].T.reshape(-1, 2)
        objp *= self.spacing

        R_obj2cam_list: List[np.ndarray] = []
        t_obj2cam_list: List[np.ndarray] = []
        valid_indices: List[int] = []

        for i in range(200):
            for ext in (".jpg", ".png"):
                img_path = os.path.join(self.images_dir, f"{i}{ext}")
                if os.path.exists(img_path):
                    break
            else:
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.xx, self.yy),
                                                     flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
            if not ret:
                self.log_message.emit(f"图片 {i} 未检测到角点，跳过")
                continue

            _, rvec, tvec = cv2.solvePnP(objp.astype(np.float32),
                                         corners.astype(np.float32),
                                         self.intrinsics, None)
            R_obj2cam_list.append(rvec)
            t_obj2cam_list.append(tvec)
            valid_indices.append(i)

        n = len(valid_indices)
        if n < 3:
            raise RuntimeError(f"有效标定图片仅 {n} 张，至少需要 3 张")

        self.log_message.emit(f"成功处理 {n} 张标定图片 (索引: {valid_indices})")

        R_tool, t_tool = self._load_robot_poses(n)
        R_result, t_result = cv2.calibrateHandEye(
            R_tool, t_tool, R_obj2cam_list, t_obj2cam_list)
        return R_result, t_result

    def _load_robot_poses(self, expected_n: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        with open(self.poses_file, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        flat = [float(v) for line in lines for v in line.strip().split(",") if v.strip()]
        if len(flat) % 6 != 0:
            raise RuntimeError(f"poses.txt 格式错误: 数据点数 {len(flat)} 不是 6 的倍数")
        n_poses = len(flat) // 6
        if n_poses < expected_n:
            raise RuntimeError(f"poses.txt 仅有 {n_poses} 条位姿，但有 {expected_n} 张有效图片")

        R_list: List[np.ndarray] = []
        t_list: List[np.ndarray] = []
        for i in range(expected_n):
            pose = flat[i * 6:(i + 1) * 6]
            H = self._pose_to_homogeneous(pose)
            H_inv = self._inverse_homogeneous(H)
            R_list.append(H_inv[:3, :3])
            t_list.append(H_inv[:3, 3])
        return R_list, t_list

    @staticmethod
    def _pose_to_homogeneous(pose: List[float]) -> np.ndarray:
        x, y, z, rx, ry, rz = pose
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = [x, y, z]
        return H

    @staticmethod
    def _inverse_homogeneous(T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    def _save_matrix(self, matrix: np.ndarray) -> None:
        path = os.path.join(self.save_dir, "calibration_matrix.txt")
        np.savetxt(path, matrix, fmt="%.8f")
        self.log_message.emit(f"齐次变换矩阵已保存到: {path}")


# ---------------------------------------------------------------------------
# Real camera wrapper
# ---------------------------------------------------------------------------

class _RealCamera:
    def __init__(self, pipeline: Any, align: Any, intr: Any, w: int, h: int) -> None:
        self._pipeline = pipeline
        self._align = align
        self._intr_raw = intr
        self.width = w
        self.height = h

    def get_img(self) -> Tuple[np.ndarray, np.ndarray]:
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        depth = np.asanyarray(aligned.get_depth_frame().get_data())
        color = np.asanyarray(aligned.get_color_frame().get_data())
        return depth, color

    def get_intrinsics(self) -> np.ndarray:
        i = self._intr_raw
        return np.array([[i.fx, 0, i.ppx],
                         [0, i.fy, i.ppy],
                         [0, 0, 1]])

    def stop(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


class _MockCamera:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height

    def get_img(self) -> Tuple[np.ndarray, np.ndarray]:
        depth = np.random.randint(0, 65536, (self.height, self.width), dtype=np.uint16)
        color = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        pattern_size = 40
        for r in range(12):
            for c in range(9):
                x, y = 80 + c * pattern_size, 60 + r * pattern_size
                cv2.rectangle(color, (x, y), (x + pattern_size // 2, y + pattern_size // 2),
                              (0, 0, 0) if (r + c) % 2 == 0 else (255, 255, 255), -1)
        return depth, color

    def get_intrinsics(self) -> np.ndarray:
        return np.array([[604.335, 0, 316.187],
                         [0, 604.404, 248.611],
                         [0, 0, 1]])

    def stop(self) -> None:
        pass


class _MockRobot:
    def rm_get_current_arm_state(self) -> Tuple[int, dict]:
        pose = [-0.226 + np.random.uniform(-0.01, 0.01),
                -0.003 + np.random.uniform(-0.01, 0.01),
                0.523 + np.random.uniform(-0.01, 0.01),
                -0.01 + np.random.uniform(-0.005, 0.005),
                0.028 + np.random.uniform(-0.005, 0.005),
                2.65 + np.random.uniform(-0.05, 0.05)]
        return 0, {"pose": pose}

    def rm_set_arm_stop(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Qt Widgets
# ---------------------------------------------------------------------------

class VideoWidget(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")
        self._chessboard_overlay: Optional[np.ndarray] = None

    def update_frame(self, depth: np.ndarray, color: Optional[np.ndarray]) -> None:
        if color is None or not isinstance(color, np.ndarray) or color.size == 0:
            return
        display = color.copy()
        cv2.circle(display, (display.shape[1] // 2, display.shape[0] // 2), 4, (0, 0, 255), -1)
        pix = cv2_to_qpixmap(display)
        if pix.isNull():
            return
        target = self.size() if self.width() > 0 and self.height() > 0 else pix.size()
        self.setPixmap(pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))


class LogWidget(QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("font-family: monospace;")

    def append_log(self, message: str) -> None:
        self.append(message)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


class ResultPreviewWidget(QWidget):
    """Shows the most recently captured frame with detected chessboard corners.

    The widget exposes :meth:`update_preview` which accepts a BGR ``ndarray``
    plus a ``success`` flag; the underlying QLabel auto-scales the pixmap
    while preserving aspect ratio. Resize events trigger a re-render so the
    image always fills the available area.
    """

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._title = QLabel("最近一次采集预览")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)

        self._image_label = QLabel("尚无预览")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #1a1a1a; color: #888;")
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setMinimumSize(160, 120)
        layout.addWidget(self._image_label, stretch=1)

        self._status = QLabel("—")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        self._raw_pixmap: Optional[QPixmap] = None

    def update_preview(self, image: object, success: bool) -> None:
        if not isinstance(image, np.ndarray):
            return
        pix = cv2_to_qpixmap(image)
        if pix.isNull():
            return
        self._raw_pixmap = pix
        self._render_scaled()
        if success:
            self._status.setText("✓ 角点检测成功")
            self._status.setStyleSheet("color: #2c7a2c;")
        else:
            self._status.setText("✗ 未检测到角点")
            self._status.setStyleSheet("color: #b00020;")

    def clear(self) -> None:
        self._raw_pixmap = None
        self._image_label.setPixmap(QPixmap())
        self._image_label.setText("尚无预览")
        self._status.setText("—")
        self._status.setStyleSheet("")

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_scaled()

    def _render_scaled(self) -> None:
        if self._raw_pixmap is None or self._raw_pixmap.isNull():
            return
        target = self._image_label.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        self._image_label.setPixmap(self._raw_pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))


class ControlPanel(QWidget):
    """Left-side control panel with three QGroupBox modules."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(320)
        root = QVBoxLayout(self)

        # ── Module 1: Calibration Settings ──
        grp1 = QGroupBox("手眼标定设置")
        lay1 = QVBoxLayout(grp1)

        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("标定类型:"))
        self.combo_calib_type = QComboBox()
        self.combo_calib_type.addItems(["Eye-to-Hand (眼在手外)", "Eye-in-Hand (眼在手内)"])
        row_type.addWidget(self.combo_calib_type)
        lay1.addLayout(row_type)

        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("分辨率:"))
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(["640x480", "1280x720"])
        row_res.addWidget(self.combo_resolution)
        lay1.addLayout(row_res)

        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel("格式:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["BGR8", "RGB8"])
        row_fmt.addWidget(self.combo_format)
        lay1.addLayout(row_fmt)

        row_fps = QHBoxLayout()
        row_fps.addWidget(QLabel("帧率:"))
        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["15", "30", "60"])
        self.combo_fps.setCurrentText("30")
        row_fps.addWidget(self.combo_fps)
        lay1.addLayout(row_fps)

        lay1.addWidget(QLabel("相机内参 (3×3, 逗号/空格分隔):"))
        self.txt_intrinsics = QPlainTextEdit()
        self.txt_intrinsics.setMaximumHeight(72)
        self.txt_intrinsics.setPlaceholderText("604.335, 0, 316.187\n0, 604.404, 248.611\n0, 0, 1")
        lay1.addWidget(self.txt_intrinsics)
        self.btn_get_intrinsics = QPushButton("获取相机内参")
        lay1.addWidget(self.btn_get_intrinsics)

        root.addWidget(grp1)

        # ── Module 2: Data Collection ──
        grp2 = QGroupBox("数据采集")
        lay2 = QVBoxLayout(grp2)

        row_dir = QHBoxLayout()
        self.le_save_dir = QLineEdit()
        self.le_save_dir.setReadOnly(True)
        self.le_save_dir.setPlaceholderText("请选择保存目录…")
        row_dir.addWidget(self.le_save_dir)
        self.btn_choose_dir = QPushButton("选择文件夹")
        row_dir.addWidget(self.btn_choose_dir)
        lay2.addLayout(row_dir)

        row_btns = QHBoxLayout()
        self.btn_save = QPushButton("保存数据 (Save)")
        self.btn_undo = QPushButton("撤销 (Undo)")
        row_btns.addWidget(self.btn_save)
        row_btns.addWidget(self.btn_undo)
        lay2.addLayout(row_btns)

        self.lbl_progress = QLabel("已采集数据: 0 / 20")
        lay2.addWidget(self.lbl_progress)

        root.addWidget(grp2)

        # ── Module 3: Matrix Calculation ──
        grp3 = QGroupBox("计算标定矩阵")
        lay3 = QVBoxLayout(grp3)

        row_rc = QHBoxLayout()
        row_rc.addWidget(QLabel("行角点数:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(3, 30)
        self.spin_rows.setValue(12)
        row_rc.addWidget(self.spin_rows)
        row_rc.addWidget(QLabel("列角点数:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(3, 30)
        self.spin_cols.setValue(9)
        row_rc.addWidget(self.spin_cols)
        lay3.addLayout(row_rc)

        row_sp = QHBoxLayout()
        row_sp.addWidget(QLabel("物理间距 (m):"))
        self.spin_spacing = QDoubleSpinBox()
        self.spin_spacing.setDecimals(4)
        self.spin_spacing.setRange(0.001, 1.0)
        self.spin_spacing.setSingleStep(0.005)
        self.spin_spacing.setValue(0.03)
        row_sp.addWidget(self.spin_spacing)
        lay3.addLayout(row_sp)

        self.btn_calculate = QPushButton("计算 (Calculate)")
        lay3.addWidget(self.btn_calculate)

        root.addWidget(grp3)
        root.addStretch(1)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class CalibrationGUI(QMainWindow):
    def __init__(self, robot: Optional[RobotLike] = None, mock: bool = False) -> None:
        super().__init__()
        self.mock = mock
        self.robot = robot

        self.sm = StateMachine(GUIState.STARTUP)
        self._camera: Optional[CameraLike] = None
        self._save_dir: str = ""
        self._collected: int = 0
        self._closing = False

        self._video_thread: Optional[QThread] = None
        self._video_worker: Optional[VideoWorker] = None
        self._cam_init_thread: Optional[QThread] = None
        self._intr_thread: Optional[QThread] = None
        self._save_thread: Optional[QThread] = None
        self._calc_thread: Optional[QThread] = None

        self._original_stdout = sys.stdout
        self.log_bridge = LogBridge()

        self._setup_ui()
        self._setup_connections()

        self.log_bridge.message_emitted.connect(self.log_widget.append_log)
        sys.stdout = self.log_bridge

        self.sm.force(GUIState.IDLE)
        self._sync_ui()
        print("就绪 — 请配置相机参数后选择保存目录")

    # ── UI ──

    def _setup_ui(self) -> None:
        self.setWindowTitle("机械臂手眼标定工具")
        self.resize(1280, 720)

        central = QWidget(self)
        self.setCentralWidget(central)
        grid = QGridLayout(central)

        self.panel = ControlPanel()
        self.video_widget = VideoWidget()
        self.log_widget = LogWidget()
        self.preview_widget = ResultPreviewWidget()

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self.log_widget)
        bottom_splitter.addWidget(self.preview_widget)
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 2)
        bottom_splitter.setSizes([600, 400])

        grid.addWidget(self.panel, 0, 0, 2, 1)
        grid.addWidget(self.video_widget, 0, 1)
        grid.addWidget(bottom_splitter, 1, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)

    def _setup_connections(self) -> None:
        p = self.panel
        p.btn_get_intrinsics.clicked.connect(self._on_get_intrinsics)
        p.btn_choose_dir.clicked.connect(self._on_choose_dir)
        p.btn_save.clicked.connect(self._on_save)
        p.btn_undo.clicked.connect(self._on_undo)
        p.btn_calculate.clicked.connect(self._on_calculate)
        self.sm.on_change(lambda _o, _n: self._sync_ui())

    def _sync_ui(self) -> None:
        bs = self.sm.get_button_states()
        p = self.panel
        p.btn_get_intrinsics.setEnabled(bs["get_intrinsics"])
        p.btn_choose_dir.setEnabled(bs["choose_dir"])
        p.btn_save.setEnabled(bs["save"] and bool(self._save_dir))
        p.btn_undo.setEnabled(bs["undo"] and self._collected > 0)
        p.btn_calculate.setEnabled(bs["calculate"] and self._collected >= 3 and bool(self._save_dir))
        p.combo_calib_type.setEnabled(bs["calib_type"])
        p.combo_resolution.setEnabled(bs["resolution"])
        p.combo_format.setEnabled(bs["format"])
        p.combo_fps.setEnabled(bs["fps"])

    # ── Choose directory & auto-init camera ──

    def _on_choose_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not d:
            return
        self._save_dir = d
        self.panel.le_save_dir.setText(d)
        print(f"保存目录: {d}")

        poses_file = os.path.join(d, "poses.txt")
        if os.path.exists(poses_file):
            with open(poses_file, "r") as fp:
                count = sum(1 for line in fp if line.strip())
            existing_imgs = sum(
                1 for i in range(count)
                if os.path.exists(os.path.join(d, f"{i}.jpg"))
                or os.path.exists(os.path.join(d, f"{i}.png"))
            )
            self._collected = min(count, existing_imgs)
            print(f"检测到已有 {self._collected} 组数据")
        else:
            self._collected = 0
        self._update_progress()

        if self._camera is None:
            self._init_camera()

    def _init_camera(self) -> None:
        self.sm.force(GUIState.CAMERA_INIT)
        print("正在初始化相机…")

        res = self.panel.combo_resolution.currentText()
        w, h = (int(v) for v in res.split("x"))
        fmt = self.panel.combo_format.currentText()
        fps = int(self.panel.combo_fps.currentText())

        self._cam_init_thread = QThread(self)
        worker = CameraInitWorker(w, h, fmt, fps, mock=self.mock)
        worker.moveToThread(self._cam_init_thread)
        self._cam_init_thread.started.connect(worker.run)
        worker.init_done.connect(self._on_camera_init_done)
        worker.error_occurred.connect(lambda m: print(f"相机初始化错误: {m}"))
        worker.finished.connect(self._cam_init_thread.quit)
        self._cam_init_worker = worker
        self._cam_init_thread.start()

    def _on_camera_init_done(self, success: bool, camera: Optional[CameraLike]) -> None:
        if self._closing:
            return
        if success and camera is not None:
            self._camera = camera
            print("相机初始化成功")
            self._start_video()
            self.sm.force(GUIState.READY)
        else:
            print("相机初始化失败")
            self.sm.force(GUIState.FAULT)

    # ── Video ──

    def _start_video(self) -> None:
        if self._video_thread is not None and self._video_thread.isRunning():
            return
        self._video_thread = QThread(self)
        self._video_worker = VideoWorker(camera=self._camera, mock=self.mock)
        self._video_worker.moveToThread(self._video_thread)
        self._video_thread.started.connect(self._video_worker.run)
        self._video_worker.frame_ready.connect(self.video_widget.update_frame)
        self._video_worker.error_occurred.connect(lambda m: print(f"[Video] {m}"))
        self._video_worker.finished.connect(self._video_thread.quit)
        self._video_thread.start()

    # ── Get intrinsics ──

    def _on_get_intrinsics(self) -> None:
        print("获取相机内参…")
        self._intr_thread = QThread(self)
        worker = IntrinsicsWorker(self._camera, mock=self.mock)
        worker.moveToThread(self._intr_thread)
        self._intr_thread.started.connect(worker.run)
        worker.intrinsics_ready.connect(self._on_intrinsics_ready)
        worker.error_occurred.connect(lambda m: print(f"获取内参失败: {m}"))
        worker.finished.connect(self._intr_thread.quit)
        self._intr_worker = worker
        self._intr_thread.start()

    def _on_intrinsics_ready(self, mat: np.ndarray) -> None:
        lines = []
        for row in mat:
            lines.append(", ".join(f"{v:.6g}" for v in row))
        text = "\n".join(lines)
        self.panel.txt_intrinsics.setPlainText(text)
        print("相机内参矩阵:")
        print(text)

    # ── Save data ──

    def _on_save(self) -> None:
        if not self._save_dir:
            print("请先选择保存目录")
            return
        self.sm.force(GUIState.COLLECTING)

        rows = self.panel.spin_rows.value()
        cols = self.panel.spin_cols.value()

        self._save_thread = QThread(self)
        worker = SaveDataWorker(
            camera=self._camera, robot=self.robot,
            save_dir=self._save_dir, index=self._collected,
            rows=rows, cols=cols, mock=self.mock,
        )
        worker.moveToThread(self._save_thread)
        self._save_thread.started.connect(worker.run)
        worker.save_done.connect(self._on_save_done)
        worker.preview_ready.connect(self.preview_widget.update_preview)
        worker.log_message.connect(lambda m: print(m))
        worker.error_occurred.connect(lambda m: print(f"保存错误: {m}"))
        worker.finished.connect(self._save_thread.quit)
        self._save_worker = worker
        self._save_thread.start()

    def _on_save_done(self, success: bool, index: int) -> None:
        if self._closing:
            return
        if success:
            self._collected = index + 1
            self._update_progress()
            print(f"已保存第 {index} 组数据")
        else:
            print(f"保存第 {index} 组数据失败")
        self.sm.force(GUIState.READY)

    # ── Undo ──

    def _on_undo(self) -> None:
        if self._collected <= 0:
            print("无数据可撤销")
            return
        last = self._collected - 1
        for ext in (".jpg", ".png"):
            p = os.path.join(self._save_dir, f"{last}{ext}")
            if os.path.exists(p):
                os.remove(p)
                break

        poses_file = os.path.join(self._save_dir, "poses.txt")
        if os.path.exists(poses_file):
            with open(poses_file, "r") as fp:
                lines = fp.readlines()
            if lines:
                with open(poses_file, "w") as fp:
                    fp.writelines(lines[:-1])

        self._collected = last
        self._update_progress()
        if self._collected == 0:
            self.preview_widget.clear()
        print(f"已撤销第 {last} 组数据")
        self._sync_ui()

    # ── Calculate ──

    def _on_calculate(self) -> None:
        intrinsics = self._parse_intrinsics()
        if intrinsics is None:
            print("内参矩阵格式错误，请检查输入（3行，每行3个数字）")
            return

        poses_file = os.path.join(self._save_dir, "poses.txt")
        if not os.path.exists(poses_file):
            print("未找到 poses.txt，请先采集数据")
            return

        self.sm.force(GUIState.CALCULATING)
        print("开始计算标定矩阵…")

        rows = self.panel.spin_rows.value()
        cols = self.panel.spin_cols.value()
        spacing = self.panel.spin_spacing.value()

        self._calc_thread = QThread(self)
        worker = CalibrationWorker(
            images_dir=self._save_dir,
            poses_file=poses_file,
            intrinsics=intrinsics,
            rows=rows,
            cols=cols,
            spacing=spacing,
            save_dir=self._save_dir,
            mock=self.mock,
        )
        worker.moveToThread(self._calc_thread)
        self._calc_thread.started.connect(worker.run)
        worker.calc_done.connect(self._on_calc_done)
        worker.log_message.connect(lambda m: print(m))
        worker.error_occurred.connect(lambda m: print(f"标定计算错误: {m}"))
        worker.finished.connect(self._calc_thread.quit)
        self._calc_worker = worker
        self._calc_thread.start()

    def _on_calc_done(self, success: bool, matrix: Optional[np.ndarray]) -> None:
        if self._closing:
            return
        if success:
            print("标定计算完成")
        else:
            print("标定计算失败")
        self.sm.force(GUIState.READY)

    # ── helpers ──

    def _parse_intrinsics(self) -> Optional[np.ndarray]:
        text = self.panel.txt_intrinsics.toPlainText().strip()
        if not text:
            return None
        try:
            rows = []
            for line in text.splitlines():
                vals = [float(v) for v in line.replace(",", " ").split() if v.strip()]
                rows.append(vals)
            mat = np.array(rows, dtype=np.float64)
            if mat.shape != (3, 3):
                return None
            return mat
        except Exception:
            return None

    def _update_progress(self) -> None:
        self.panel.lbl_progress.setText(f"已采集数据: {self._collected} / 20")

    # ── close ──

    def closeEvent(self, event: Any) -> None:
        self._closing = True
        self.sm.force(GUIState.CLOSING)

        if self._video_worker is not None:
            self._video_worker.stop()
        for t in (self._video_thread, self._cam_init_thread,
                  self._intr_thread, self._save_thread, self._calc_thread):
            if t is not None:
                t.quit()
                t.wait(3000)

        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass

        sys.stdout = self._original_stdout
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="机械臂手眼标定工具")
    parser.add_argument("--mock", action="store_true", help="无硬件模拟模式")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    robot: Optional[RobotLike] = None
    if args.mock:
        robot = _MockRobot()
    else:
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
            arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            arm.rm_create_robot_arm("192.168.127.101", 8080)
            arm.rm_set_collision_state(5)
            robot = arm
        except Exception as exc:
            print(f"机械臂连接失败: {exc}，进入仅相机模式")

    window = CalibrationGUI(robot=robot, mock=args.mock)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
