import os
import time

import pytest
from PyQt5.QtWidgets import QApplication

from grasp_gui_v2 import GUIState, GraspGUI, _CancellationToken
from tests.gui.mocks import MockGrasp


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met before timeout")


def _assert_controls_follow_state(gui: GraspGUI):
    expected = gui.state_machine.get_button_states()
    assert gui.control_panel.btn_init.isEnabled() == expected["init"]
    assert gui.control_panel.btn_start.isEnabled() == expected["start_grasp"]
    assert gui.control_panel.btn_stop.isEnabled() == expected["stop"]
    assert gui.control_panel.combo_objects.isEnabled() == expected["object_select"]


def test_grasp_gui_builds_with_mock_and_mounts_main_widgets(app):
    gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
    try:
        layout = gui.centralWidget().layout()
        assert gui.control_panel is not None
        assert gui.video_widget is not None
        assert gui.log_widget is not None
        assert gui.status_widget is not None
        assert layout.indexOf(gui.control_panel) != -1
        assert layout.indexOf(gui.video_widget) != -1
        assert layout.indexOf(gui.log_widget) != -1
        assert layout.indexOf(gui.status_widget) != -1
    finally:
        gui.close()


def test_initial_state_is_idle_and_controls_match_rules(app):
    gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
    try:
        assert gui.state_machine.current_state == GUIState.IDLE
        _assert_controls_follow_state(gui)
    finally:
        gui.close()


def test_all_state_machine_rules_are_reflected_in_gui_controls(app):
    gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
    try:
        for state in GUIState:
            gui.state_machine.force_state(state)
            QApplication.processEvents()
            _assert_controls_follow_state(gui)
    finally:
        gui.close()


def test_object_selection_updates_status_widget(app):
    gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
    try:
        gui.control_panel.combo_objects.setCurrentText("banana")
        QApplication.processEvents()
        assert "30" in gui.status_widget.speed_label.text()
        assert "banana" in gui.status_widget.object_label.text()
    finally:
        gui.close()


def test_init_click_transitions_to_ready_in_mock_backend(app):
    gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
    transitions = []
    gui.state_machine.on_state_change(lambda old, new: transitions.append((old, new)))
    try:
        gui.control_panel.btn_init.click()
        _wait_until(lambda: gui.state_machine.current_state == GUIState.READY)
        assert (GUIState.IDLE, GUIState.INITIALIZING) in transitions
        assert gui.state_machine.current_state == GUIState.READY
        _assert_controls_follow_state(gui)
    finally:
        gui.close()


def test_start_button_disabled_until_ready(app):
    gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
    try:
        assert gui.state_machine.current_state == GUIState.IDLE
        assert gui.control_panel.btn_start.isEnabled() is False
        gui.control_panel.btn_start.click()
        QApplication.processEvents()
        assert gui.state_machine.current_state == GUIState.IDLE
    finally:
        gui.close()


def test_start_grasp_mock_success_returns_to_ready(app):
    gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
    transitions = []
    gui.state_machine.on_state_change(lambda old, new: transitions.append((old, new)))
    try:
        gui.control_panel.btn_init.click()
        _wait_until(lambda: gui.state_machine.current_state == GUIState.READY)
        gui.control_panel.btn_start.click()
        assert gui.state_machine.current_state == GUIState.GRASPING
        _wait_until(lambda: gui.state_machine.current_state == GUIState.READY)
        assert (GUIState.READY, GUIState.GRASPING) in transitions
        assert any(new == GUIState.READY for _, new in transitions)
        _assert_controls_follow_state(gui)
    finally:
        gui.close()


def test_emergency_stop_calls_robot_stop_and_transitions_consistently(app):
    grasp = MockGrasp(hardware=False)
    stop_calls = []

    def _stop():
        stop_calls.append(True)
        return 0

    grasp.robot.rm_set_arm_stop = _stop
    gui = GraspGUI(grasp=grasp, mock=True)
    try:
        gui.state_machine.force_state(GUIState.READY)
        gui._sync_controls_to_state()
        gui.control_panel.btn_stop.click()
        _wait_until(lambda: len(stop_calls) > 0)
        _wait_until(lambda: gui.state_machine.current_state in {GUIState.IDLE, GUIState.FAULT})
        assert stop_calls == [True]
        assert gui.state_machine.current_state in {GUIState.IDLE, GUIState.FAULT}
        _assert_controls_follow_state(gui)
    finally:
        gui.close()


def test_late_grasp_completion_does_not_override_stop_state(app):
    gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
    try:
        gui.state_machine.force_state(GUIState.GRASPING)
        gui._grasp_cancel = _CancellationToken()
        gui._grasp_cancel.cancel()
        gui._on_grasp_finished(True)
        assert gui.state_machine.current_state == GUIState.GRASPING
    finally:
        gui.close()
