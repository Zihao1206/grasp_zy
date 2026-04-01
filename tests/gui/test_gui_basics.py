import os

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from grasp_gui_v2 import (
    OBJECT_LABELS,
    ControlPanel,
    GraspWorker,
    HardwareLock,
    LogWidget,
    StatusWidget,
    VideoWidget,
    VideoWorker,
    cv2_to_qpixmap,
)


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_control_panel_uses_required_labels(app):
    panel = ControlPanel()
    actual = [panel.combo_objects.itemText(i) for i in range(panel.combo_objects.count())]
    assert actual == OBJECT_LABELS


def test_cv2_to_qpixmap_handles_none_and_empty(app):
    assert cv2_to_qpixmap(None).isNull()
    assert cv2_to_qpixmap(np.array([], dtype=np.uint8)).isNull()


def test_video_widget_update_frame_sets_pixmap(app):
    widget = VideoWidget()
    widget.resize(640, 480)
    depth = np.zeros((480, 640), dtype=np.uint16)
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    widget.update_frame(depth, color)
    pixmap = widget.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()


def test_log_widget_is_read_only_and_appends(app):
    widget = LogWidget()
    assert widget.isReadOnly()
    widget.append_log("hello")
    assert "hello" in widget.toPlainText()


def test_status_widget_updates_visible_labels(app):
    widget = StatusWidget()
    widget.update_status(42, "banana")
    assert "42" in widget.speed_label.text()
    assert "banana" in widget.object_label.text()


def test_video_worker_mock_emits_frame():
    worker = VideoWorker(mock=True)
    frames = []

    def _on_frame(depth, color):
        frames.append((depth, color))
        worker.stop()

    worker.frame_ready.connect(_on_frame)
    worker.run()
    assert len(frames) == 1
    depth, color = frames[0]
    assert depth.shape == (480, 640)
    assert color.shape == (480, 640, 3)


def test_grasp_worker_mock_finishes_successfully():
    worker = GraspWorker(grasp=None, label="terminal", mock=True)
    results = []
    finished = []

    worker.grasp_finished.connect(results.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()

    assert results == [True]
    assert finished == [True]


def test_hardware_lock_exposes_acquire_release():
    lock = HardwareLock()
    lock.acquire()
    lock.release()
