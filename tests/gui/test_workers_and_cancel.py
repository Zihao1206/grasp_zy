import os
import time

import pytest
from PyQt5.QtWidgets import QApplication

from grasp_gui_v2 import (
    GUIState,
    GraspGUI,
    GraspWorker,
    InitWorker,
    StopWorker,
    _CancellationToken,
)
from tests.gui.mocks import MockGrasp


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met before timeout")


class TestCancellationToken:
    def test_initial_state_is_not_cancelled(self):
        token = _CancellationToken()
        assert not token.is_cancelled

    def test_cancel_sets_flag(self):
        token = _CancellationToken()
        token.cancel()
        assert token.is_cancelled

    def test_cancel_is_idempotent(self):
        token = _CancellationToken()
        token.cancel()
        token.cancel()
        assert token.is_cancelled


class TestInitWorkerDirectRun:
    def test_emits_true_on_success(self, app):
        worker = InitWorker(MockGrasp(hardware=False), mock=False)
        results = []
        worker.init_finished.connect(results.append)
        worker.run()
        QApplication.processEvents()
        assert results == [True]

    def test_emits_false_on_failure(self, app):
        class _FailGrasp:
            def init_gripper(self):
                raise RuntimeError("gripper error")

        worker = InitWorker(_FailGrasp(), mock=False)
        results = []
        worker.init_finished.connect(results.append)
        worker.run()
        QApplication.processEvents()
        assert results == [False]

    def test_skips_call_when_mock(self, app):
        worker = InitWorker(None, mock=True)
        results = []
        worker.init_finished.connect(results.append)
        worker.run()
        QApplication.processEvents()
        assert results == [True]


class TestStopWorkerDirectRun:
    def test_emits_true_on_success(self, app):
        grasp = MockGrasp(hardware=False)
        stop_calls = []
        grasp.robot.rm_set_arm_stop = lambda: stop_calls.append(True) or 0
        worker = StopWorker(grasp, mock=False)
        results = []
        worker.stop_finished.connect(results.append)
        worker.run()
        QApplication.processEvents()
        assert stop_calls == [True]
        assert results == [True]

    def test_emits_false_on_failure(self, app):
        class _FailGrasp:
            class robot:
                @staticmethod
                def rm_set_arm_stop():
                    raise OSError("serial error")

        worker = StopWorker(_FailGrasp(), mock=False)
        results = []
        worker.stop_finished.connect(results.append)
        worker.run()
        QApplication.processEvents()
        assert results == [False]

    def test_handles_none_grasp(self, app):
        worker = StopWorker(None, mock=False)
        results = []
        worker.stop_finished.connect(results.append)
        worker.run()
        QApplication.processEvents()
        assert results == [True]


class TestGraspWorkerCancellation:
    def test_cancelled_before_run_emits_false(self, app):
        token = _CancellationToken()
        token.cancel()
        worker = GraspWorker(None, "soap", mock=True, cancel_token=token)
        results = []
        worker.grasp_finished.connect(results.append)
        worker.run()
        QApplication.processEvents()
        assert results == [False]

    def test_not_cancelled_emits_true(self, app):
        token = _CancellationToken()
        worker = GraspWorker(None, "soap", mock=True, cancel_token=token)
        results = []
        worker.grasp_finished.connect(results.append)
        worker.run()
        QApplication.processEvents()
        assert results == [True]

    def test_no_token_works_normally(self, app):
        worker = GraspWorker(None, "soap", mock=True)
        results = []
        worker.grasp_finished.connect(results.append)
        worker.run()
        QApplication.processEvents()
        assert results == [True]


class TestGUIInitStopIntegration:
    def test_stop_cancels_active_grasp(self, app):
        gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
        try:
            gui.control_panel.btn_init.click()
            _wait_until(lambda: gui.state_machine.current_state == GUIState.READY)

            gui.control_panel.btn_start.click()
            assert gui.state_machine.current_state == GUIState.GRASPING
            assert gui._grasp_cancel is not None
            assert not gui._grasp_cancel.is_cancelled

            gui.control_panel.btn_stop.click()
            assert gui._grasp_cancel.is_cancelled
            _wait_until(lambda: gui.state_machine.current_state == GUIState.IDLE)
        finally:
            gui.close()

    def test_init_worker_thread_cleaned_up_after_completion(self, app):
        gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
        try:
            gui.control_panel.btn_init.click()
            _wait_until(lambda: gui.state_machine.current_state == GUIState.READY)
            _wait_until(lambda: gui.init_thread is not None and not gui.init_thread.isRunning())
        finally:
            gui.close()

    def test_stop_worker_thread_cleaned_up_after_completion(self, app):
        gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
        try:
            gui.state_machine.force_state(GUIState.READY)
            gui._sync_controls_to_state()
            gui.control_panel.btn_stop.click()
            _wait_until(lambda: gui.state_machine.current_state == GUIState.IDLE)
            _wait_until(lambda: gui.stop_thread is not None and not gui.stop_thread.isRunning())
        finally:
            gui.close()
