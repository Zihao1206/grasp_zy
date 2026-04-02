import os
import sys

import pytest
from PyQt5.QtWidgets import QApplication

import grasp_gui_v2
from grasp_gui_v2 import GUIState, GraspGUI, main
from tests.gui.mocks import MockGrasp


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_close_event_restores_stdout_and_cleans_threads(app):
    grasp = MockGrasp(hardware=False)
    camera_stop_calls = []
    robot_stop_calls = []

    def _camera_stop():
        camera_stop_calls.append(True)

    def _robot_stop():
        robot_stop_calls.append(True)
        return 0

    grasp.camera.stop = _camera_stop
    grasp.robot.rm_set_arm_stop = _robot_stop
    gui = GraspGUI(grasp=grasp, mock=True)
    gui.start_video()
    QApplication.processEvents()

    gui.state_machine.force_state(GUIState.READY)
    gui._sync_controls_to_state()
    gui.control_panel.btn_start.click()
    QApplication.processEvents()

    original_stdout = gui._original_stdout
    gui.close()
    QApplication.processEvents()

    assert sys.stdout is original_stdout
    assert camera_stop_calls == [True]
    assert robot_stop_calls == [True]
    if gui.video_thread is not None:
        assert not gui.video_thread.isRunning()
    if gui.grasp_thread is not None:
        assert not gui.grasp_thread.isRunning()


def test_main_mock_path_avoids_hardware_import_and_non_blocking(monkeypatch, app):
    shown = []
    video_started = []

    class FakeApp:
        def __init__(self, argv):
            self.argv = list(argv)

        @staticmethod
        def instance():
            return None

        def exec_(self):
            return 123

    def _show(self):
        shown.append(True)

    def _start_video(self):
        video_started.append(True)

    monkeypatch.setattr(grasp_gui_v2, "QApplication", FakeApp)
    monkeypatch.setattr(grasp_gui_v2.GraspGUI, "show", _show)
    monkeypatch.setattr(grasp_gui_v2.GraspGUI, "start_video", _start_video)
    monkeypatch.setattr(sys, "argv", ["prog", "--mock"])

    had_hardware_module = "grasp_zy_zhiyuan1215" in sys.modules
    prev_hardware_module = sys.modules.get("grasp_zy_zhiyuan1215")
    if had_hardware_module:
        del sys.modules["grasp_zy_zhiyuan1215"]

    try:
        exit_code = main()
    finally:
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, GraspGUI):
                widget.close()

    assert exit_code == 123
    assert shown == [True]
    assert video_started == [True]
    assert "grasp_zy_zhiyuan1215" not in sys.modules

    if had_hardware_module and prev_hardware_module is not None:
        sys.modules["grasp_zy_zhiyuan1215"] = prev_hardware_module
