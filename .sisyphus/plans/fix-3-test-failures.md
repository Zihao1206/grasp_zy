# Fix 3 Failing GUI Tests

## TL;DR

> **Quick Summary**: Fix 3 test failures caused by the new 2-step grasp workflow (preview→confirm→execute) changing button names and state machine rules. All tests reference the old `btn_start` or assume `btn_stop` is enabled in READY state.
>
> **Deliverables**:
> - 3 test files updated to match new GUI structure
> - All 49/49 tests passing
>
> **Estimated Effort**: Quick
> **Parallel Execution**: YES - 1 wave
> **Critical Path**: All 3 tasks independent

---

## Context

### Original Request
Continuation of the pre-grasp + pause/resume feature. 46/49 tests pass; 3 remain failing due to stale references to old GUI structure (`btn_start`, `grasp_thread`, state machine rules that changed).

### Interview Summary
**Key Discussions**:
- Root cause analysis: All 3 tests written for old 1-step workflow, now 2-step (preview→confirm→execute)
- `btn_start` replaced by `btn_pre_grasp` + `btn_confirm`
- `grasp_thread` replaced by `execute_thread` + `plan_thread`
- State machine: `btn_stop` now only enabled in GRASPING and PAUSED (was previously enabled more broadly)

### Metis Review
**Identified Gaps** (addressed):
- Scope creep risk: Don't add new tests, only fix existing ones
- Guardrail: Don't change production code (`grasp_gui_v2.py`) — only test files
- Edge case: Test #3 (`test_stop_worker_thread_cleaned_up_after_completion`) needs to verify `stop_thread` is cleaned up, which already works — just needs the right flow to trigger stop

---

## Work Objectives

### Core Objective
Fix 3 failing tests to align with the new 2-step grasp GUI workflow.

### Concrete Deliverables
- `tests/gui/test_grasp_gui_window.py` — 1 test fixed
- `tests/gui/test_shutdown.py` — 1 test fixed
- `tests/gui/test_workers_and_cancel.py` — 1 test fixed

### Definition of Done
- [ ] `QT_QPA_PLATFORM=offscreen pytest tests/gui/ -v` → 49 passed, 0 failed

### Must Have
- All 49 tests passing
- No changes to production code (`grasp_gui_v2.py`, `grasp_zy_zhiyuan1215.py`)

### Must NOT Have (Guardrails)
- NO changes to `grasp_gui_v2.py` — production code is stable
- NO changes to `grasp_zy_zhiyuan1215.py` — backend is stable
- NO new test cases added — only fix existing ones
- NO changes to `tests/gui/mocks.py` — mocks are correct
- NO changes to passing tests — only touch the 3 failing tests
- NO AI slop: no excessive comments, no refactoring of working test code

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: Tests-after (the fixes ARE the tests)
- **Framework**: pytest + pytest-qt

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — all independent):
├── Task 1: Fix test_emergency_stop in test_grasp_gui_window.py [quick]
├── Task 2: Fix test_close_event in test_shutdown.py [quick]
└── Task 3: Fix test_stop_worker_thread in test_workers_and_cancel.py [quick]

Wave FINAL (After ALL tasks — verification):
└── Task F1: Run full test suite and verify 49/49 pass [quick]

Parallel Speedup: ~70% faster than sequential
Max Concurrent: 3
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | — | F1 |
| 2 | — | F1 |
| 3 | — | F1 |
| F1 | 1, 2, 3 | — |

### Agent Dispatch Summary

- **Wave 1**: **3** — T1→`quick`, T2→`quick`, T3→`quick`
- **FINAL**: **1** — F1→`quick`

---

## TODOs

- [x] 1. Fix `test_emergency_stop_calls_robot_stop_and_transitions_consistently`

  **What to do**:
  - In `tests/gui/test_grasp_gui_window.py`, fix the test at line 133
  - **Root cause**: Test forces state to READY then clicks btn_stop, but btn_stop is disabled in READY state (state machine rules: `READY.stop = False`)
  - **Fix**: Change the test to exercise the real flow: init→READY→pre_grasp→PREVIEW→confirm→GRASPING→stop
  - Replace lines 143-148 with:
    ```python
    gui.control_panel.btn_init.click()
    _wait_until(lambda: gui.state_machine.current_state == GUIState.READY)
    gui.control_panel.btn_pre_grasp.click()
    _wait_until(lambda: gui.state_machine.current_state == GUIState.PREVIEW)
    gui.control_panel.btn_confirm.click()
    _wait_until(lambda: gui.state_machine.current_state == GUIState.GRASPING)
    gui.control_panel.btn_stop.click()
    _wait_until(lambda: len(stop_calls) > 0)
    _wait_until(lambda: gui.state_machine.current_state in {GUIState.IDLE, GUIState.FAULT})
    ```
  - Remove the old `force_state` and `_sync_controls_to_state` calls (lines 144-145)
  - Keep the assertions at lines 149-151 unchanged

  **Must NOT do**:
  - Don't change the state machine rules in `grasp_gui_v2.py`
  - Don't enable stop button in READY state
  - Don't modify any other test in this file

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: F1
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `tests/gui/test_grasp_gui_window.py:115-130` — `test_pre_grasp_then_confirm_returns_to_ready` — This test already demonstrates the correct flow: init→READY→pre_grasp→PREVIEW→confirm→READY. Use the same pattern but add stop after GRASPING.
  - `tests/gui/test_grasp_gui_window.py:156-171` — `test_cancel_event_set_on_stop_during_execute` — This test already does the full flow with stop from GRASPING. Follow this exact pattern.

  **Why Each Reference Matters**:
  - Lines 115-130: Proves the init→pre_grasp→confirm sequence works. Copy the button clicks and _wait_until calls.
  - Lines 156-171: Proves stopping from GRASPING state works. This is the exact same flow needed for the failing test, just needs to also verify `stop_calls`.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Emergency stop test passes
    Tool: Bash (pytest)
    Preconditions: No code changes in progress
    Steps:
      1. Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/gui/test_grasp_gui_window.py::test_emergency_stop_calls_robot_stop_and_transitions_consistently -v
      2. Check exit code is 0
      3. Check output contains "PASSED"
    Expected Result: Test passes, robot stop was called, state transitioned to IDLE
    Failure Indicators: Exit code != 0, "FAILED" in output, "timeout" in output
    Evidence: .sisyphus/evidence/task-1-emergency-stop-pass.txt

  Scenario: No other tests broken by the change
    Tool: Bash (pytest)
    Preconditions: Task 1 change applied
    Steps:
      1. Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/gui/test_grasp_gui_window.py -v
      2. Count passed and failed tests
    Expected Result: All tests in this file pass (should be ~9 tests)
    Failure Indicators: Any test shows FAILED
    Evidence: .sisyphus/evidence/task-1-all-window-tests.txt
  ```

  **Commit**: YES (groups with 2, 3)
  - Message: `fix(tests): update 3 tests for new 2-step grasp workflow`
  - Files: `tests/gui/test_grasp_gui_window.py`, `tests/gui/test_shutdown.py`, `tests/gui/test_workers_and_cancel.py`
  - Pre-commit: `QT_QPA_PLATFORM=offscreen python -m pytest tests/gui/ -v`

---

- [x] 2. Fix `test_close_event_restores_stdout_and_cleans_threads`

  **What to do**:
  - In `tests/gui/test_shutdown.py`, fix the test at line 20
  - **Root cause**: Line 40 references `gui.control_panel.btn_start.click()` — `btn_start` no longer exists (replaced by `btn_pre_grasp` + `btn_confirm`)
  - Also line 52-53 references `gui.grasp_thread` — replaced by `gui.execute_thread`
  - **Fix**: Replace the old start-a-grasp flow with the new 2-step flow:
    ```python
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

        gui.control_panel.btn_init.click()
        _wait_until(lambda: gui.state_machine.current_state == GUIState.READY)
        gui.control_panel.btn_pre_grasp.click()
        _wait_until(lambda: gui.state_machine.current_state == GUIState.PREVIEW)
        gui.control_panel.btn_confirm.click()
        _wait_until(lambda: gui.state_machine.current_state == GUIState.GRASPING)

        original_stdout = gui._original_stdout
        gui.close()
        QApplication.processEvents()

        assert sys.stdout is original_stdout
        assert camera_stop_calls == [True]
        assert robot_stop_calls == [True]
        if gui.video_thread is not None:
            assert not gui.video_thread.isRunning()
        if gui.execute_thread is not None:
            assert not gui.execute_thread.isRunning()
    ```
  - Add `_wait_until` helper function (copy from other test files) and import `time`
  - Replace `gui.grasp_thread` check with `gui.execute_thread` check (line 52-53)

  **Must NOT do**:
  - Don't change `test_main_mock_path_avoids_hardware_import_and_non_blocking` — it already passes
  - Don't modify the `closeEvent` method in `grasp_gui_v2.py`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: F1
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `tests/gui/test_grasp_gui_window.py:19-26` — `_wait_until` helper function — Copy this exact implementation into test_shutdown.py
  - `tests/gui/test_grasp_gui_window.py:115-130` — `test_pre_grasp_then_confirm_returns_to_ready` — The correct flow pattern: init→READY→pre_grasp→PREVIEW→confirm
  - `grasp_gui_v2.py:551-571` — GraspGUI thread attributes — Shows new attribute names: `execute_thread` (not `grasp_thread`), `plan_thread`, `_resume_event`, `_cancel_event`

  **Why Each Reference Matters**:
  - test_grasp_gui_window.py `_wait_until`: Need this exact helper since shutdown test doesn't have it yet
  - test_grasp_gui_window.py flow: Proves the init→pre_grasp→confirm sequence works
  - grasp_gui_v2.py attributes: Confirms the correct attribute names to check in assertions

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Shutdown test passes
    Tool: Bash (pytest)
    Preconditions: No code changes in progress
    Steps:
      1. Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/gui/test_shutdown.py::test_close_event_restores_stdout_and_cleans_threads -v
      2. Check exit code is 0
      3. Check output contains "PASSED"
    Expected Result: Test passes, stdout restored, camera/robot stopped, threads cleaned
    Failure Indicators: Exit code != 0, "FAILED" in output, "btn_start" error, "grasp_thread" error
    Evidence: .sisyphus/evidence/task-2-shutdown-test-pass.txt

  Scenario: Other shutdown test still passes
    Tool: Bash (pytest)
    Preconditions: Task 2 change applied
    Steps:
      1. Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/gui/test_shutdown.py -v
      2. Check all tests pass
    Expected Result: Both shutdown tests pass
    Failure Indicators: Any test shows FAILED
    Evidence: .sisyphus/evidence/task-2-all-shutdown-tests.txt
  ```

  **Commit**: YES (groups with 1, 3)
  - Message: `fix(tests): update 3 tests for new 2-step grasp workflow`
  - Files: `tests/gui/test_grasp_gui_window.py`, `tests/gui/test_shutdown.py`, `tests/gui/test_workers_and_cancel.py`

---

- [x] 3. Fix `test_stop_worker_thread_cleaned_up_after_completion`

  **What to do**:
  - In `tests/gui/test_workers_and_cancel.py`, fix the test at line 176
  - **Root cause**: Test forces state to READY then clicks btn_stop, but btn_stop is disabled in READY state. The click does nothing, `_on_stop` never fires, state never transitions to IDLE.
  - **Fix**: Exercise the real flow to get to a state where stop is enabled, then click stop:
    ```python
    def test_stop_worker_thread_cleaned_up_after_completion(self, app):
        gui = GraspGUI(grasp=MockGrasp(hardware=False), mock=True)
        try:
            gui.control_panel.btn_init.click()
            _wait_until(lambda: gui.state_machine.current_state == GUIState.READY)
            gui.control_panel.btn_pre_grasp.click()
            _wait_until(lambda: gui.state_machine.current_state == GUIState.PREVIEW)
            gui.control_panel.btn_confirm.click()
            _wait_until(lambda: gui.state_machine.current_state == GUIState.GRASPING)
            gui.control_panel.btn_stop.click()
            _wait_until(lambda: gui.state_machine.current_state == GUIState.IDLE)
            _wait_until(lambda: gui.stop_thread is not None and not gui.stop_thread.isRunning())
        finally:
            gui.close()
    ```

  **Must NOT do**:
  - Don't change the `_wait_until` timeout (3.0s is fine for mock mode)
  - Don't modify passing tests in this file
  - Don't change `grasp_gui_v2.py` to enable stop in READY state

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: F1
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `tests/gui/test_workers_and_cancel.py:149-165` — `test_stop_sets_cancel_event` — This test already does the exact flow: init→READY→pre_grasp→PREVIEW→confirm→GRASPING→stop. Follow this pattern.
  - `tests/gui/test_workers_and_cancel.py:167-174` — `test_init_worker_thread_cleaned_up_after_completion` — Shows the thread cleanup assertion pattern (`thread is not None and not thread.isRunning()`)

  **Why Each Reference Matters**:
  - Lines 149-165: Proves the full flow works for stopping. This IS the correct pattern — the failing test just needs to use the same flow and then check stop_thread cleanup instead of cancel_event.
  - Lines 167-174: Shows the thread cleanup assertion pattern to use.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Stop worker thread test passes
    Tool: Bash (pytest)
    Preconditions: No code changes in progress
    Steps:
      1. Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/gui/test_workers_and_cancel.py::TestGUIInitStopIntegration::test_stop_worker_thread_cleaned_up_after_completion -v
      2. Check exit code is 0
      3. Check output contains "PASSED"
    Expected Result: Test passes, stop_thread is created and cleaned up after completion
    Failure Indicators: Exit code != 0, "FAILED" in output, "timeout" in output
    Evidence: .sisyphus/evidence/task-3-stop-thread-test.txt

  Scenario: All workers-and-cancel tests still pass
    Tool: Bash (pytest)
    Preconditions: Task 3 change applied
    Steps:
      1. Run: QT_QPA_PLATFORM=offscreen python -m pytest tests/gui/test_workers_and_cancel.py -v
      2. Check all tests pass
    Expected Result: All tests in this file pass
    Failure Indicators: Any test shows FAILED
    Evidence: .sisyphus/evidence/task-3-all-workers-tests.txt
  ```

  **Commit**: YES (groups with 1, 2)
  - Message: `fix(tests): update 3 tests for new 2-step grasp workflow`
  - Files: `tests/gui/test_grasp_gui_window.py`, `tests/gui/test_shutdown.py`, `tests/gui/test_workers_and_cancel.py`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

- [x] F1. **Full Test Suite Verification** — `quick`
  Run the complete GUI test suite: `QT_QPA_PLATFORM=offscreen pytest tests/gui/ -v`
  Verify: 49 passed, 0 failed. Save output to evidence.
  Output: `Tests [49/49 PASS] | VERDICT`

---

## Commit Strategy

- **Single commit**: `fix(tests): update 3 tests for new 2-step grasp workflow`
  - Files: `tests/gui/test_grasp_gui_window.py`, `tests/gui/test_shutdown.py`, `tests/gui/test_workers_and_cancel.py`
  - Pre-commit: `QT_QPA_PLATFORM=offscreen pytest tests/gui/ -v`

---

## Success Criteria

### Verification Commands
```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/gui/ -v
# Expected: 49 passed, 0 failed
```

### Final Checklist
- [ ] All 3 previously-failing tests now pass
- [ ] No previously-passing tests broke
- [ ] No changes to production code (grasp_gui_v2.py, grasp_zy_zhiyuan1215.py)
- [ ] No new tests added
- [ ] All 49/49 tests pass
