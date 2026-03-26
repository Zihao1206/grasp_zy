# CTU-Arm Interaction Fix

## TL;DR
> **Summary**: Fix the missing `grasp_zy_test.py` import and `detect_obj` method to restore CTU-triggered robotic arm grasping.
> **Deliverables**: Updated `grasp_zy_zhiyuan1215.py` with `detect_obj` method, fixed import in `ctu_conn.py`, agent-executable verification tests
> **Effort**: Short
> **Parallel**: NO - sequential dependency chain
> **Critical Path**: Audit dependencies → Add detect_obj → Fix import → Verify

## Context

### Original Request
After system reinstall, CTU can control cart movement but cannot trigger robotic arm grasping. Single-step grasping works via `grasp_zy_zhiyuan1215.py`, but CTU serial commands fail to execute grasping.

### Root Cause Analysis
1. **Missing method**: `ctu_conn.py:108,123` calls `grasp.detect_obj(label)` but this method does not exist in `grasp_zy_zhiyuan1215.py`
2. **Import path**: Already corrected to `from grasp_zy_zhiyuan1215 import Grasp` (verified)

### Metis Review (gaps addressed)
- Dependency audit required before porting `detect_obj`
- Helper methods `in_paint`, `get_index`, `find_num_count_np` must exist in target class
- Item-id to label mapping confirmed: `1=soap, 2=interrupter, 3=terminal, 4=limit, 5=voltage`
- QA must be agent-executable, not manual

## Work Objectives

### Core Objective
Restore CTU-to-arm communication by fixing the import path and adding the missing `detect_obj` method.

### Deliverables
1. `grasp_zy_zhiyuan1215.py` - Add `detect_obj` method with required helper methods
2. `ctu_conn.py` - Update import from `grasp_zy_test` to `grasp_zy_zhiyuan1215`
3. Verification tests - Agent-executable syntax and contract tests

### Definition of Done (verifiable conditions with commands)
```bash
# 1. Import smoke test
python -c "from grasp_zy_zhiyuan1215 import Grasp; print(hasattr(Grasp, 'detect_obj'))"
# Expected: True

# 2. Syntax validation
python -m py_compile ctu_conn.py grasp_zy_zhiyuan1215.py
# Expected: exit code 0

# 3. CTU connection starts without import error
python -c "from ctu_conn import CTUConn; print('OK')"
# Expected: OK
```

### Must Have
- `detect_obj(label)` method that returns integer count of detected objects
- Import path that resolves correctly
- All existing helper methods (`in_paint`, `get_index`, `find_num_count_np`) available

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- NO refactoring of `obj_grasp`, robot motion, or CTU protocol
- NO changes to ROS2 code
- NO environment/service configuration changes
- NO redesign of existing working code

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (compatibility restoration, not new feature)
- QA policy: Every task has agent-executed scenarios
- Evidence: .sisyphus/evidence/task-{N}-{slug}.{ext}

## Execution Strategy

### Parallel Execution Waves
> Sequential execution required due to dependency chain.

Wave 1: [dependency audit + method port]
Wave 2: [import fix]
Wave 3: [verification tests]

### Dependency Matrix
| Task | Blocked By |
|------|------------|
| Add detect_obj method | Dependency audit |
| Fix import path | Add detect_obj method |
| Verification tests | Fix import path |

### Agent Dispatch Summary
- Wave 1: 2 tasks (audit, add method) - category: quick
- Wave 2: 1 task (import fix) - category: quick
- Wave 3: 2 tasks (syntax test, contract test) - category: quick

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Dependency Audit for detect_obj Method

  **What to do**: Analyze `others/grasp_zy_zhiyuan08281.py:641-683` to identify all helper methods and attributes required by `detect_obj`. Compare against `grasp_zy_zhiyuan1215.py` to verify all dependencies exist.

  **Must NOT do**: Do not modify any files during this audit phase.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Read-only analysis task
  - Skills: [] — No special skills needed
  - Omitted: [] — None

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: Task 2 | Blocked By: None

  **References** (executor has NO interview context — be exhaustive):
  - Source: `others/grasp_zy_zhiyuan08281.py:641-683` — Original detect_obj implementation
  - Target: `grasp_zy_zhiyuan1215.py` — Where method will be added
  - Helper methods to verify: `self.camera.get_img()`, `self.in_paint()`, `self.det_model`, `self.get_index()`, `self.find_num_count_np()`

  **Acceptance Criteria** (agent-executable only):
  - [ ] List of all helper methods/attributes used by detect_obj documented
  - [ ] Confirmation that all dependencies exist in grasp_zy_zhiyuan1215.py OR list of missing methods

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Audit completeness
    Tool: Bash
    Steps: grep -n "self\." others/grasp_zy_zhiyuan08281.py | grep -A50 "def detect_obj"
    Expected: All self.X references identified
    Evidence: .sisyphus/evidence/task-1-audit.txt
  ```

  **Commit**: NO — Read-only audit task

- [ ] 2. Add detect_obj Method to Grasp Class

  **What to do**: 
  1. Copy `detect_obj` method from `others/grasp_zy_zhiyuan08281.py:641-683` to `grasp_zy_zhiyuan1215.py`
  2. Copy any missing helper methods (`get_index`, `find_num_count_np`) if not present
  3. Verify method signature matches: `def detect_obj(self, label) -> int`

  **Must NOT do**: 
  - Do not modify existing methods in grasp_zy_zhiyuan1215.py
  - Do not change the detection logic or thresholds

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Simple method copy with minimal changes
  - Skills: [] — No special skills needed
  - Omitted: [] — None

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: Task 3 | Blocked By: Task 1

  **References** (executor has NO interview context — be exhaustive):
  - Source: `others/grasp_zy_zhiyuan08281.py:641-683` — Method to port
  - Target: `grasp_zy_zhiyuan1215.py` — Insert after `obj_grasp` method or before `if __name__`
  - Dependencies: `get_index` (lines 633-637), `find_num_count_np` (lines 629-631) if missing

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -c "from grasp_zy_zhiyuan1215 import Grasp; print(hasattr(Grasp, 'detect_obj'))"` outputs `True`
  - [ ] `python -m py_compile grasp_zy_zhiyuan1215.py` exits with code 0

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Method exists and is callable
    Tool: Bash
    Steps: python -c "from grasp_zy_zhiyuan1215 import Grasp; g = Grasp(hardware=False); print(callable(getattr(g, 'detect_obj', None)))"
    Expected: True
    Evidence: .sisyphus/evidence/task-2-method-exists.txt

  Scenario: Syntax validation
    Tool: Bash
    Steps: python -m py_compile grasp_zy_zhiyuan1215.py && echo "PASS"
    Expected: PASS
    Evidence: .sisyphus/evidence/task-2-syntax.txt
  ```

  **Commit**: YES | Message: `fix: add detect_obj method to Grasp class for CTU compatibility` | Files: `grasp_zy_zhiyuan1215.py`

- [ ] 3. Verify Import Path in ctu_conn.py

  **What to do**: 
  1. Verify line 6 of `ctu_conn.py` contains `from grasp_zy_zhiyuan1215 import Grasp`
  2. Confirm no references to `grasp_zy_test` exist in the file

  **Must NOT do**: 
  - Do not modify any logic in ctu_conn.py unless import is incorrect

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Simple verification task
  - Skills: [] — No special skills needed
  - Omitted: [] — None

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: Task 4 | Blocked By: Task 2

  **References** (executor has NO interview context — be exhaustive):
  - Target: `ctu_conn.py:6` — Line to verify
  - Expected: `from grasp_zy_zhiyuan1215 import Grasp`
  - Note: Import was already corrected in a previous session; this is a verification gate

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -c "from ctu_conn import CTUConn; print('OK')"` outputs `OK`
  - [ ] No references to `grasp_zy_test` in ctu_conn.py

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Import resolves successfully
    Tool: Bash
    Steps: python -c "from ctu_conn import CTUConn; print('PASS')"
    Expected: PASS
    Evidence: .sisyphus/evidence/task-3-import-resolves.txt

  Scenario: No orphan references
    Tool: Bash
    Steps: grep -c "grasp_zy_test" ctu_conn.py || echo "0"
    Expected: 0
    Evidence: .sisyphus/evidence/task-3-no-orphans.txt
  ```

  **Commit**: NO — Verification only (import already correct)

- [ ] 4. Contract Verification Test

  **What to do**: Create and run a verification script that confirms all methods called by ctu_conn.py exist on the Grasp class.

  **Must NOT do**: Do not modify production code during verification.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: Simple verification script
  - Skills: [] — No special skills needed
  - Omitted: [] — None

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Final Verification | Blocked By: Task 3

  **References** (executor has NO interview context — be exhaustive):
  - Contract methods: `change_robot_speed`, `init_gripper`, `detect_obj`, `obj_grasp`
  - Reference: `ctu_conn.py:91,104,108,123` — All Grasp method calls

  **Acceptance Criteria** (agent-executable only):
  - [ ] All contract methods exist on Grasp class
  - [ ] Method signatures are callable

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: All CTU contract methods exist
    Tool: Bash
    Steps: python -c "
from grasp_zy_zhiyuan1215 import Grasp
methods = ['change_robot_speed', 'init_gripper', 'detect_obj', 'obj_grasp']
missing = [m for m in methods if not hasattr(Grasp, m)]
print('MISSING:', missing if missing else 'NONE')
"
    Expected: MISSING: NONE
    Evidence: .sisyphus/evidence/task-4-contract.txt
  ```

  **Commit**: NO — Verification only

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
Atomic commits with single intent:
1. `fix: add detect_obj method to Grasp class for CTU compatibility`

## Success Criteria
1. `python -c "from ctu_conn import CTUConn"` succeeds without error
2. `detect_obj(label)` returns integer count for any valid label
3. CTU serial command `wifi Rb_Pick 1 30` triggers arm grasping sequence
