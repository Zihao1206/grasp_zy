# 机械臂抓取 PyQt5 GUI 开发计划

## TL;DR

> **快速摘要**：为机械臂抓取项目开发一个功能完整的 PyQt5 可视化界面，支持实时视频流、物体选择、抓取控制、日志显示和状态监控，同时支持 Mock 模式和自动化测试。
>
> **交付物**：
> - `grasp_gui_v2.py`：主 GUI 应用程序
> - `tests/gui/`：自动化测试套件
> - `requirements.txt`：更新依赖（添加 PyQt5, pytest-qt）
>
> **预计工作量**：Large
> **并行执行**：YES - 3 waves
> **关键路径**：Mock 接口 → GUI 状态机 → 视频线程 → 抓取线程 → 集成测试

---

## Context

### 原始需求
用户需要一个 PyQt5 可视化界面，用于控制机械臂抓取系统。核心类 `Grasp` 已经实现，需要为其开发 GUI 前端。

### 访谈摘要
**关键讨论**：
- 初始化按钮行为：仅调用机械臂/夹爪初始化函数（快速重置）
- 错误显示方式：在日志区显示（不打断用户）
- 测试策略：添加自动化测试（Mock 硬件接口）
- Mock 模式：支持 `--mock` 参数，允许无硬件启动

**研究发现**：
- Grasp 类没有明显的线程安全机制，需要定义硬件所有权模型
- Camera 和 Robot 接口需要序列化访问
- 需要定义明确的 GUI 状态机和按钮启用/禁用规则

### Metis 审查
**识别的差距**（已解决）：
- 线程安全问题：添加 GUI 状态机和硬件访问序列化
- 测试策略：添加自动化测试（pytest + pytest-qt）
- Mock 模式：支持 `--mock` 参数
- 资源生命周期：定义窗口关闭时的清理契约

---

## Work Objectives

### 核心目标
开发一个功能完整、线程安全、可测试的 PyQt5 GUI 应用程序，用于控制机械臂抓取系统。

### 具体交付物
- `grasp_gui_v2.py`：主 GUI 应用程序（~800-1000 行代码）
- `tests/gui/test_state_machine.py`：状态机测试
- `tests/gui/test_video_worker.py`：视频线程测试
- `tests/gui/test_grasp_worker.py`：抓取线程测试
- `tests/gui/test_log_bridge.py`：日志桥接测试
- `tests/gui/test_shutdown.py`：关闭处理测试
- `requirements.txt`：添加 PyQt5, pytest-qt 依赖

### 完成定义
- [ ] GUI 启动并显示实时视频流（带中心红点）
- [ ] 所有控制按钮正常工作（初始化、开始抓取、紧急停止）
- [ ] 日志区实时显示系统输出
- [ ] 状态区显示机械臂速度和选中物体
- [ ] 所有自动化测试通过（pytest）
- [ ] Mock 模式正常工作（`--mock` 参数）
- [ ] 窗口关闭时正确清理资源

### 必须有
- 实时视频流显示（30 FPS）
- 物体选择下拉框（8 种物体）
- 初始化按钮（快速重置）
- 开始抓取按钮（子线程执行）
- 紧急停止按钮（立即停止机械臂）
- 日志区（sys.stdout 重定向）
- 状态区（速度 + 选中物体）
- Mock 模式支持（`--mock` 参数）
- 自动化测试（pytest + pytest-qt）
- 线程安全（硬件访问序列化）
- GUI 状态机（按钮启用/禁用规则）

### 必须没有（Guardrails）
- ❌ 不要修改现有的 `Grasp` 类实现
- ❌ 不要添加深度图视图（超出 v1 范围）
- ❌ 不要添加多摄像头支持
- ❌ 不要添加持久化设置
- ❌ 不要添加校准工具
- ❌ 不要添加批处理抓取功能
- ❌ 不要添加自动重连逻辑（超出简单定义行为）
- ❌ 不要添加模型管理 UI
- ❌ 不要添加主题/样式工作（超出功能布局）
- ❌ 不要允许抓取过程中改变物体选择（按钮禁用）
- ❌ 不要允许抓取过程中点击初始化（按钮禁用）
- ❌ 不要允许重复点击开始抓取（忽略 + 日志）

---

## Verification Strategy (MANDATORY)

> **零人工干预** — 所有验证均由代理执行。无例外。
> 禁止使用"用户手动测试/确认"作为验收标准。

### 测试决策
- **基础设施存在**：NO（需要创建 tests/ 目录）
- **自动化测试**：YES (TDD)
- **框架**：pytest + pytest-qt
- **TDD**：每个任务遵循 RED（失败测试）→ GREEN（最小实现）→ REFACTOR

### QA 策略
每个任务必须包含代理执行的 QA 场景（见下方 TODO 模板）。
证据保存到 `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`。

- **Frontend/UI**：使用 Playwright（playwright skill）— 导航、交互、断言 DOM、截图
- **TUI/CLI**：使用 interactive_bash (tmux) — 运行命令、发送按键、验证输出
- **API/Backend**：使用 Bash (curl) — 发送请求、断言状态 + 响应字段
- **Library/Module**：使用 Bash (bun/node REPL) — 导入、调用函数、比较输出

---

## Execution Strategy

### 并行执行波

> 通过将独立任务分组到并行波中来最大化吞吐量。
> 每个波在下一个波开始之前完成。
> 目标：每波 5-8 个任务。少于 3 个（除最后一个）= 拆分不足。

```
Wave 1（立即开始 — Mock 接口 + 状态机）：
├── Task 1: 更新依赖 + 测试框架 [quick]
├── Task 2: 创建 Mock 接口（Grasp, Camera, Robot, Gripper）[unspecified-high]
├── Task 3: GUI 状态机设计 + 测试 [deep]
├── Task 4: 日志桥接 + sys.stdout 重定向 [quick]
└── Task 5: 主窗口骨架 + QGridLayout [visual-engineering]

Wave 2（Wave 1 之后 — 核心线程）：
├── Task 6: 视频工作线程 + 测试 [unspecified-high]
├── Task 7: 图像转换工具（OpenCV → QPixmap）[quick]
├── Task 8: 视频显示组件 + 中心红点绘制 [visual-engineering]
├── Task 9: 抓取工作线程 + 测试 [unspecified-high]
├── Task 10: 硬件访问序列化机制 [deep]
└── Task 11: 控制面板（按钮 + 下拉框）[visual-engineering]

Wave 3（Wave 2 之后 — 集成 + 清理）：
├── Task 12: 日志显示组件 + QTextEdit [visual-engineering]
├── Task 13: 状态显示组件 [visual-engineering]
├── Task 14: 按钮状态管理（启用/禁用规则）[unspecified-high]
├── Task 15: 窗口关闭处理 + 资源清理 [unspecified-high]
├── Task 16: Mock 模式集成 + 测试 [unspecified-high]
└── Task 17: 集成测试 + 文档 [writing]

Wave FINAL（所有任务之后 — 4 个并行审查，然后用户确认）：
├── Task F1: 计划合规性审计（oracle）
├── Task F2: 代码质量审查（unspecified-high）
├── Task F3: 真实手动 QA（unspecified-high + playwright）
└── Task F4: 范围保真度检查（deep）
-> 呈现结果 -> 获取明确用户确认

关键路径：Task 1 → Task 2 → Task 3 → Task 6 → Task 9 → Task 10 → Task 16 → Task 17 → F1-F4 → 用户确认
并行加速：比顺序快约 70%
最大并发：5（Wave 1）
```

### 依赖矩阵（缩写 — 在生成的计划中显示所有任务）

- **1-5**: — — 6-11, 1
- **6**: 2, 7 — 8, 2
- **9**: 2, 10 — 14, 2
- **10**: 3 — 9, 2
- **11**: 3 — 14, 2
- **14**: 3, 9, 11 — 15, 2
- **15**: 14 — 16, 2
- **16**: 1, 2, 6, 9, 15 — 17, 2

> 这是缩写供参考。您生成的计划必须包含所有任务的完整矩阵。

### 代理调度摘要

- **1**: **5** — T1 → `quick`, T2 → `unspecified-high`, T3 → `deep`, T4 → `quick`, T5 → `visual-engineering`
- **2**: **6** — T6 → `unspecified-high`, T7 → `quick`, T8 → `visual-engineering`, T9 → `unspecified-high`, T10 → `deep`, T11 → `visual-engineering`
- **3**: **6** — T12 → `visual-engineering`, T13 → `visual-engineering`, T14 → `unspecified-high`, T15 → `unspecified-high`, T16 → `unspecified-high`, T17 → `writing`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> 实现 + 测试 = 一个任务。永远不要分离。
> 每个任务必须有：推荐代理配置 + 并行化信息 + QA 场景。
> **没有 QA 场景的任务是不完整的。无例外。**

- [ ] 1. 更新依赖 + 测试框架

  **做什么**：
  - 更新 `requirements.txt`，添加 PyQt5 和 pytest-qt 依赖
  - 创建 `tests/` 目录结构
  - 创建 `tests/gui/` 子目录
  - 创建 `tests/conftest.py`（pytest 配置）
  - 验证测试框架可以运行

  **必须不做**：
  - 不要修改现有的依赖版本
  - 不要添加其他不必要的测试依赖

  **推荐代理配置**：
  > 选择 category + skills 基于任务领域。证明每个选择。
  - **Category**: `quick`
    - 原因：简单的文件更新和目录创建，不需要复杂的逻辑
  - **Skills**: []
    - 无需特定技能，基础文件操作即可

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 1（与 Tasks 2, 3, 4, 5）
  - **阻塞**: Tasks 2-17（需要测试框架）
  - **被阻塞**: None（可以立即开始）

  **参考**（CRITICAL - 要详尽）：
  > 执行者没有来自您访谈的上下文。参考是他们唯一的指南。
  > 每个参考必须回答："我应该看什么以及为什么？"

  **模式参考**（要遵循的现有代码）：
  - `requirements.txt:1-37` - 现有依赖格式和结构

  **外部参考**（库和框架）：
  - PyQt5 官方文档：https://www.riverbankcomputing.com/static/Docs/PyQt5/
  - pytest-qt 文档：https://pytest-qt.readthedocs.io/

  **为什么每个参考重要**（解释相关性）：
  - `requirements.txt` - 需要遵循现有的依赖格式（版本约束、分组）

  **验收标准**：
  - [ ] `requirements.txt` 包含 PyQt5>=5.15.0
  - [ ] `requirements.txt` 包含 pytest-qt>=4.2.0
  - [ ] `tests/` 目录存在
  - [ ] `tests/gui/` 目录存在
  - [ ] `tests/conftest.py` 文件存在
  - [ ] `pytest tests/` 命令运行成功（即使没有测试）

  **QA 场景**（MANDATORY — 没有这些任务是不完整的）：

  > **这不是可选的。没有 QA 场景的任务将被拒绝。**
  >
  > 编写验证您构建内容的实际行为的场景测试。
  > 最少：每个任务 1 个快乐路径 + 1 个失败/边缘情况。
  > 每个场景 = 确切工具 + 确切步骤 + 确切断言 + 证据路径。

  ```
  场景：验证测试框架设置
    工具：Bash
    前置条件：项目根目录
    步骤：
      1. 运行 `pytest tests/ --collect-only`
      2. 检查输出不包含错误
    预期结果：命令成功退出，代码 0
    失败指标：命令失败或输出包含 "ERROR"
    证据：.sisyphus/evidence/task-01-framework-setup.txt

  场景：验证依赖安装
    工具：Bash
    前置条件：虚拟环境激活
    步骤：
      1. 运行 `python -c "import PyQt5; print(PyQt5.Qt.PYQT_VERSION_STR)"`
      2. 运行 `python -c "import pytest_qt; print(pytest_qt.__version__)"`
    预期结果：两个命令都输出版本号，无错误
    失败指标：ImportError 或 ModuleNotFoundError
    证据：.sisyphus/evidence/task-01-deps-check.txt
  ```

  **要捕获的证据**：
  - [ ] 每个证据文件命名为：task-01-{scenario-slug}.{ext}
  - [ ] 命令输出的文本日志

  **提交**: YES
  - 消息：`chore(deps): add PyQt5 and pytest-qt to requirements`
  - 文件：`requirements.txt`, `tests/conftest.py`
  - 预提交：`pytest tests/ --collect-only`

- [ ] 2. 创建 Mock 接口（Grasp, Camera, Robot, Gripper）

  **做什么**：
  - 创建 `tests/gui/mocks.py` 文件
  - 实现 `MockGrasp` 类，模拟 `Grasp` 的所有关键方法
  - 实现 `MockCamera` 类，模拟 `camera.RS` 的接口
  - 实现 `MockRobot` 类，模拟机械臂接口
  - 实现 `MockGripper` 类，模拟夹爪接口
  - Mock 对象应该返回合理的测试数据（不是 None）
  - Mock 对象应该记录方法调用（用于测试验证）

  **必须不做**：
  - 不要修改现有的 `Grasp` 类
  - 不要创建真实的硬件连接
  - 不要实现复杂的行为逻辑（只需返回合理的静态数据）

  **推荐代理配置**：
  > 选择 category + skills 基于任务领域。证明每个选择。
  - **Category**: `unspecified-high`
    - 原因：需要理解现有接口并创建准确的 Mock，需要较高的技术理解
  - **Skills**: []
    - 无需特定技能，但需要仔细阅读现有代码

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 1（与 Tasks 1, 3, 4, 5）
  - **阻塞**: Tasks 6, 9, 16（需要 Mock 接口）
  - **被阻塞**: Task 1（需要测试框架）

  **参考**（CRITICAL - 要详尽）：

  **模式参考**（要遵循的现有代码）：
  - `grasp_zy_zhiyuan1215.py:33-617` - Grasp 类的完整接口
  - `camera.py:12-84` - Camera 类的接口
  - `gripper_zhiyuan.py:5-60` - Gripper 类的接口

  **API/类型参考**（要实现的契约）：
  - `Grasp.__init__(hardware=True)` - 初始化方法
  - `Grasp.obj_grasp(label, vis=False)` - 抓取方法
  - `Grasp.camera.get_img()` - 返回 (depth, color) 元组
  - `Grasp.robot.rm_set_arm_stop()` - 紧急停止
  - `Grasp.robot_speed` - 速度属性

  **为什么每个参考重要**（解释相关性）：
  - `grasp_zy_zhiyuan1215.py` - 需要准确 Mock Grasp 类的所有公共方法
  - `camera.py` - 需要知道 `get_img()` 返回的确切数据类型和形状
  - `gripper_zhiyuan.py` - 需要知道夹爪的接口

  **验收标准**：
  - [ ] `tests/gui/mocks.py` 文件存在
  - [ ] `MockGrasp` 类实现所有公共方法
  - [ ] `MockCamera.get_img()` 返回 (depth, color) 元组
  - [ ] `MockRobot.rm_set_arm_stop()` 可调用
  - [ ] `MockGripper.gripper_position()` 可调用
  - [ ] Mock 对象记录方法调用（可以使用 `unittest.mock.Mock`）

  **QA 场景**（MANDATORY）：

  ```
  场景：验证 Mock Grasp 接口
    工具：Bash (Python REPL)
    前置条件：tests/gui/mocks.py 存在
    步骤：
      1. `python -c "from tests.gui.mocks import MockGrasp; g = MockGrasp(); print(g.robot_speed)"`
      2. `python -c "from tests.gui.mocks import MockGrasp; g = MockGrasp(); result = g.obj_grasp('test'); print(result)"`
    预期结果：第一个命令输出 30，第二个命令输出 True 或 False
    失败指标：AttributeError 或方法不存在
    证据：.sisyphus/evidence/task-02-mock-grasp.txt

  场景：验证 Mock Camera 接口
    工具：Bash (Python REPL)
    前置条件：tests/gui/mocks.py 存在
    步骤：
      1. `python -c "from tests.gui.mocks import MockCamera; c = MockCamera(); depth, color = c.get_img(); print(depth.shape, color.shape)"`
    预期结果：输出 (480, 640) (480, 640, 3)
    失败指标：方法不存在或返回类型错误
    证据：.sisyphus/evidence/task-02-mock-camera.txt
  ```

  **要捕获的证据**：
  - [ ] 每个证据文件命名为：task-02-{scenario-slug}.{ext}
  - [ ] Python REPL 输出日志

  **提交**: YES
  - 消息：`feat(gui): add mock interfaces for testing`
  - 文件：`tests/gui/mocks.py`
  - 预提交：`python -m pytest tests/gui/mocks.py -v`

- [ ] 3. GUI 状态机设计 + 测试

  **做什么**：
  - 在 `grasp_gui_v2.py` 中创建 `GUIState` 枚举类
  - 定义状态：`STARTUP`, `IDLE`, `INITIALIZING`, `READY`, `GRASPING`, `STOPPING`, `FAULT`, `CLOSING`
  - 创建 `StateMachine` 类管理状态转换
  - 定义每个状态下按钮的启用/禁用规则
  - 编写单元测试验证状态转换逻辑
  - 创建 `tests/gui/test_state_machine.py`

  **必须不做**：
  - 不要实现复杂的自动状态转换（保持显式）
  - 不要在状态机中包含业务逻辑

  **推荐代理配置**：
  > 选择 category + skills 基于任务领域。证明每个选择。
  - **Category**: `deep`
    - 原因：需要深入理解状态机设计和 GUI 交互逻辑
  - **Skills**: []
    - 无需特定技能，但需要理解状态机模式

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 1（与 Tasks 1, 2, 4, 5）
  - **阻塞**: Tasks 11, 14（需要状态机）
  - **被阻塞**: Task 1（需要测试框架）

  **参考**（CRITICAL - 要详尽）：

  **模式参考**（要遵循的现有代码）：
  - 无现有状态机代码（需要从零设计）

  **外部参考**（库和框架）：
  - Python enum 文档：https://docs.python.org/3/library/enum.html
  - 状态机模式：https://refactoring.guru/design-patterns/state

  **为什么每个参考重要**（解释相关性）：
  - Python enum - 用于定义状态枚举
  - 状态机模式 - 用于理解状态转换设计

  **验收标准**：
  - [ ] `GUIState` 枚举定义了所有 8 个状态
  - [ ] `StateMachine` 类实现了状态转换方法
  - [ ] `StateMachine` 包含 `get_button_states()` 方法返回按钮启用状态
  - [ ] `tests/gui/test_state_machine.py` 包含至少 10 个测试用例
  - [ ] 所有测试通过：`pytest tests/gui/test_state_machine.py -v`

  **QA 场景**（MANDATORY）：

  ```
  场景：验证状态转换逻辑
    工具：Bash (pytest)
    前置条件：tests/gui/test_state_machine.py 存在
    步骤：
      1. 运行 `pytest tests/gui/test_state_machine.py -v`
      2. 检查所有测试通过
    预期结果：所有测试通过，输出显示 "X passed"
    失败指标：任何测试失败或错误
    证据：.sisyphus/evidence/task-03-state-tests.txt

  场景：验证按钮状态规则
    工具：Bash (Python REPL)
    前置条件：grasp_gui_v2.py 存在
    步骤：
      1. `python -c "from grasp_gui_v2 import StateMachine, GUIState; sm = StateMachine(); sm.transition(GUIState.GRASPING); print(sm.get_button_states())"`
    预期结果：输出包含按钮状态字典，start_grasp 按钮为 False
    失败指标：方法不存在或返回值不正确
    证据：.sisyphus/evidence/task-03-button-states.txt
  ```

  **要捕获的证据**：
  - [ ] 每个证据文件命名为：task-03-{scenario-slug}.{ext}
  - [ ] pytest 输出日志
  - [ ] Python REPL 输出

  **提交**: YES
  - 消息：`feat(gui): implement state machine with tests`
  - 文件：`grasp_gui_v2.py`, `tests/gui/test_state_machine.py`
  - 预提交：`pytest tests/gui/test_state_machine.py -v`

- [ ] 4. 日志桥接 + sys.stdout 重定向

  **做什么**：
  - 创建 `LogBridge` 类继承 `io.StringIO`
  - 实现 `write()` 方法，发射 PyQt 信号
  - 创建 `pyqtSignal` 用于传递日志文本
  - 实现时间戳添加功能（[HH:MM:SS] 格式）
  - 编写单元测试验证日志重定向
  - 创建 `tests/gui/test_log_bridge.py`

  **必须不做**：
  - 不要修改全局 sys.stdout（只在 GUI 实例中重定向）
  - 不要在日志中包含敏感信息

  **推荐代理配置**：
  > 选择 category + skills 基于任务领域。证明每个选择。
  - **Category**: `quick`
    - 原因：相对简单的 IO 重定向和信号发射
  - **Skills**: []
    - 无需特定技能

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 1（与 Tasks 1, 2, 3, 5）
  - **阻塞**: Task 12（需要日志桥接）
  - **被阻塞**: Task 1（需要测试框架）

  **参考**（CRITICAL - 要详尽）：

  **模式参考**（要遵循的现有代码）：
  - 无现有日志重定向代码

  **外部参考**（库和框架）：
  - Python io 模块：https://docs.python.org/3/library/io.html
  - PyQt5 信号和槽：https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html

  **为什么每个参考重要**（解释相关性）：
  - Python io - 用于理解 StringIO 和自定义流
  - PyQt5 信号 - 用于线程安全的日志传递

  **验收标准**：
  - [ ] `LogBridge` 类继承 `io.StringIO`
  - [ ] `LogBridge` 包含 `message_emitted` pyqtSignal
  - [ ] `write()` 方法发射信号并添加时间戳
  - [ ] `tests/gui/test_log_bridge.py` 包含至少 5 个测试用例
  - [ ] 所有测试通过：`pytest tests/gui/test_log_bridge.py -v`

  **QA 场景**（MANDATORY）：

  ```
  场景：验证日志重定向
    工具：Bash (pytest)
    前置条件：tests/gui/test_log_bridge.py 存在
    步骤：
      1. 运行 `pytest tests/gui/test_log_bridge.py -v`
      2. 检查所有测试通过
    预期结果：所有测试通过
    失败指标：任何测试失败
    证据：.sisyphus/evidence/task-04-log-tests.txt

  场景：验证时间戳格式
    工具：Bash (Python REPL)
    前置条件：grasp_gui_v2.py 存在
    步骤：
      1. `python -c "from grasp_gui_v2 import LogBridge; import re; lb = LogBridge(); lb.write('test'); # Check signal contains timestamp pattern [\\d{2}:\\d{2}:\\d{2}]"`
    预期结果：日志消息包含时间戳
    失败指标：时间戳格式不正确
    证据：.sisyphus/evidence/task-04-timestamp.txt
  ```

  **要捕获的证据**：
  - [ ] 每个证据文件命名为：task-04-{scenario-slug}.{ext}
  - [ ] pytest 输出日志

  **提交**: YES
  - 消息：`feat(gui): add log bridge for stdout redirection`
  - 文件：`grasp_gui_v2.py`, `tests/gui/test_log_bridge.py`
  - 预提交：`pytest tests/gui/test_log_bridge.py -v`

- [ ] 5. 主窗口骨架 + QGridLayout

  **做什么**：
  - 创建 `GraspGUI` 主窗口类继承 `QMainWindow`
  - 实现中央部件和 QGridLayout 布局
  - 创建 4 个区域的占位符（右上、左上、左下、右下）
  - 设置窗口标题和大小（建议 1280x720）
  - 创建占位符 QLabel 和 QWidget 用于后续填充

  **必须不做**：
  - 不要实现具体的功能组件（只是骨架）
  - 不要添加样式或主题

  **推荐代理配置**：
  > 选择 category + skills 基于任务领域。证明每个选择。
  - **Category**: `visual-engineering`
    - 原因：GUI 布局和视觉设计
  - **Skills**: [`/frontend-ui-ux`]
    - `/frontend-ui-ux` - 用于创建清晰、专业的 GUI 布局

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 1（与 Tasks 1, 2, 3, 4）
  - **阻塞**: Tasks 8, 11, 12, 13（需要窗口骨架）
  - **被阻塞**: None（可以立即开始）

  **参考**（CRITICAL - 要详尽）：

  **模式参考**（要遵循的现有代码）：
  - 无现有 PyQt5 代码（这是新的）

  **外部参考**（库和框架）：
  - PyQt5 QMainWindow：https://www.riverbankcomputing.com/static/Docs/PyQt5/widgets/qmainwindow.html
  - QGridLayout：https://www.riverbankcomputing.com/static/Docs/PyQt5/layout.html#grid-layout

  **为什么每个参考重要**（解释相关性）：
  - QMainWindow - 理解主窗口结构
  - QGridLayout - 理解网格布局

  **验收标准**：
  - [ ] `GraspGUI` 类继承 `QMainWindow`
  - [ ] 窗口标题为 "机械臂抓取控制系统"
  - [ ] 窗口默认大小为 1280x720
  - [ ] 中央部件使用 QGridLayout
  - [ ] 4 个区域都有占位符部件
  - [ ] 窗口可以启动并显示（测试中）

  **QA 场景**（MANDATORY）：

  ```
  场景：验证窗口启动
    工具：Bash (pytest-qt)
    前置条件：grasp_gui_v2.py 存在
    步骤：
      1. 创建测试 `tests/gui/test_main_window.py`
      2. 使用 `qtbot` 创建窗口实例
      3. 验证窗口标题和大小
    预期结果：窗口创建成功，标题和大小正确
    失败指标：窗口创建失败或属性不正确
    证据：.sisyphus/evidence/task-05-window-startup.txt

  场景：验证布局结构
    工具：Bash (pytest-qt)
    前置条件：grasp_gui_v2.py 存在
    步骤：
      1. 检查中央部件存在
      2. 检查布局是 QGridLayout
      3. 检查有 4 个子部件
    预期结果：布局结构正确
    失败指标：布局类型错误或子部件数量不正确
    证据：.sisyphus/evidence/task-05-layout-structure.txt
  ```

  **要捕获的证据**：
  - [ ] 每个证据文件命名为：task-05-{scenario-slug}.{ext}
  - [ ] pytest 输出日志

  **提交**: YES
  - 消息：`feat(gui): create main window skeleton with grid layout`
  - 文件：`grasp_gui_v2.py`, `tests/gui/test_main_window.py`
  - 预提交：`pytest tests/gui/test_main_window.py -v`

- [ ] 6. 视频工作线程 + 测试

  **做什么**：
  - 创建 `VideoWorker` 类继承 `QObject`
  - 实现 `run()` 方法，循环获取摄像头图像
  - 创建 `frame_ready` pyqtSignal 发射图像
  - 创建 `error_occurred` pyqtSignal 发射错误
  - 实现线程安全标志（`running`, `paused`）
  - 支持 Mock 模式（使用 MockCamera）
  - 编写单元测试
  - 创建 `tests/gui/test_video_worker.py`

  **必须不做**：
  - 不要在 UI 线程中运行
  - 不要直接访问 UI 组件

  **推荐代理配置**：
  > 选择 category + skills 基于任务领域。证明每个选择。
  - **Category**: `unspecified-high`
    - 原因：多线程编程需要较高的技术理解
  - **Skills**: []
    - 无需特定技能

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 2（与 Tasks 7, 8, 9, 10, 11）
  - **阻塞**: Task 8（需要视频线程）
  - **被阻塞**: Tasks 2, 5（需要 Mock 接口和窗口骨架）

  **参考**（CRITICAL - 要详尽）：

  **模式参考**（要遵循的现有代码）：
  - `camera.py:51-65` - get_img() 方法实现

  **API/类型参考**（要实现的契约）：
  - `camera.RS.get_img()` - 返回 (depth, color) 元组

  **外部参考**（库和框架）：
  - PyQt5 QThread：https://www.riverbankcomputing.com/static/Docs/PyQt5/qthread.html
  - PyQt5 信号：https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html

  **为什么每个参考重要**（解释相关性）：
  - `camera.py` - 理解 get_img() 的返回类型
  - QThread - 理解如何创建工作线程
  - 信号 - 理解线程间通信

  **验收标准**：
  - [ ] `VideoWorker` 类继承 `QObject`
  - [ ] `frame_ready` 信号发射 (depth, color) 元组
  - [ ] `error_occurred` 信号发射错误消息
  - [ ] `run()` 方法在循环中调用 camera.get_img()
  - [ ] 支持 Mock 模式
  - [ ] `tests/gui/test_video_worker.py` 包含至少 8 个测试用例
  - [ ] 所有测试通过

  **QA 场景**（MANDATORY）：

  ```
  场景：验证视频线程启动和停止
    工具：Bash (pytest-qt)
    前置条件：tests/gui/test_video_worker.py 存在
    步骤：
      1. 创建 VideoWorker 实例（Mock 模式）
      2. 启动线程
      3. 等待至少 1 个帧信号
      4. 停止线程
      5. 验证线程停止
    预期结果：至少收到 1 个帧信号，线程正常停止
    失败指标：没有收到帧信号或线程无法停止
    证据：.sisyphus/evidence/task-06-video-thread.txt

  场景：验证帧数据格式
    工具：Bash (pytest-qt)
    前置条件：VideoWorker 可以运行
    步骤：
      1. 启动 VideoWorker（Mock 模式）
      2. 捕获 frame_ready 信号
      3. 验证数据是 (depth, color) 元组
      4. 验证 depth 和 color 的形状
    预期结果：depth 形状 (480, 640)，color 形状 (480, 640, 3)
    失败指标：数据格式不正确
    证据：.sisyphus/evidence/task-06-frame-format.txt
  ```

  **要捕获的证据**：
  - [ ] 每个证据文件命名为：task-06-{scenario-slug}.{ext}
  - [ ] pytest 输出日志

  **提交**: YES
  - 消息：`feat(gui): implement video worker thread`
  - 文件：`grasp_gui_v2.py`, `tests/gui/test_video_worker.py`
  - 预提交：`pytest tests/gui/test_video_worker.py -v`

- [ ] 7. 图像转换工具（OpenCV → QPixmap）

  **做什么**：
  - 创建 `cv2_to_qpixmap(color_image)` 函数
  - 实现 BGR → RGB 转换（cv2.cvtColor）
  - 实现 numpy array → QImage 转换
  - 实现 QImage → QPixmap 转换
  - 处理空图像和无效数据
  - 添加单元测试

  **必须不做**：
  - 不要修改原始图像数据
  - 不要在转换中进行图像处理（如缩放、旋转）

  **推荐代理配置**：
  > 选择 category + skills 基于任务领域。证明每个选择。
  - **Category**: `quick`
    - 原因：简单的图像格式转换
  - **Skills**: []
    - 无需特定技能

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 2（与 Tasks 6, 8, 9, 10, 11）
  - **阻塞**: Task 8（需要转换工具）
  - **被阻塞**: Task 2（需要 Mock 数据格式）

  **参考**（CRITICAL - 要详尽）：

  **模式参考**（要遵循的现有代码）：
  - `camera.py:62-63` - color_image 的数据格式

  **外部参考**（库和框架）：
  - OpenCV 颜色转换：https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
  - PyQt5 QImage：https://www.riverbankcomputing.com/static/Docs/PyQt5/gui/qimage.html

  **为什么每个参考重要**（解释相关性）：
  - OpenCV - 理解 BGR 到 RGB 转换
  - QImage - 理解如何从 numpy array 创建图像

  **验收标准**：
  - [ ] `cv2_to_qpixmap()` 函数存在
  - [ ] 函数接受 numpy array (H, W, 3) BGR 格式
  - [ ] 函数返回 QPixmap
  - [ ] 处理 None 和空数组情况
  - [ ] 单元测试覆盖正常和异常情况

  **QA 场景**（MANDATORY）：

  ```
  场景：验证图像转换
    工具：Bash (Python REPL)
    前置条件：grasp_gui_v2.py 存在
    步骤：
      1. 创建测试 BGR 图像 numpy array (480, 640, 3)
      2. 调用 cv2_to_qpixmap(image)
      3. 验证返回 QPixmap
      4. 验证 QPixmap 大小正确
    预期结果：返回有效的 QPixmap，大小为 640x480
    失败指标：返回类型错误或大小不正确
    证据：.sisyphus/evidence/task-07-image-convert.txt

  场景：验证空图像处理
    工具：Bash (Python REPL)
    前置条件：grasp_gui_v2.py 存在
    步骤：
      1. 调用 cv2_to_qpixmap(None)
      2. 验证返回空 QPixmap 或抛出异常
    预期结果：函数优雅处理 None，不崩溃
    失败指标：函数崩溃或抛出未捕获异常
    证据：.sisyphus/evidence/task-07-empty-image.txt
  ```

  **要捕获的证据**：
  - [ ] 每个证据文件命名为：task-07-{scenario-slug}.{ext}
  - [ ] Python REPL 输出

  **提交**: YES
  - 消息：`feat(gui): add image conversion utilities`
  - 文件：`grasp_gui_v2.py`
  - 预提交：`python -c "from grasp_gui_v2 import cv2_to_qpixmap; import numpy as np; cv2_to_qpixmap(np.zeros((480,640,3), dtype=np.uint8))"`

- [ ] 8. 视频显示组件 + 中心红点绘制

  **做什么**：
  - 创建 `VideoWidget` 类继承 `QLabel`
  - 实现 `update_frame(depth, color)` 方法
  - 在显示前在图像上绘制中心红点（320, 240，半径 5）
  - 使用 cv2.circle() 绘制红点
  - 调用 cv2_to_qpixmap() 转换图像
  - 设置缩放模式（保持宽高比）
  - 连接 VideoWorker 的 frame_ready 信号

  **必须不做**：
  - 不要修改原始图像数据（在副本上绘制）
  - 不要实现复杂的图像处理

  **推荐代理配置**：
  > 选择 category + skills 基于任务领域。证明每个选择。
  - **Category**: `visual-engineering`
    - 原因：GUI 组件和图像显示
  - **Skills**: [`/frontend-ui-ux`]
    - `/frontend-ui-ux` - 用于创建视觉组件

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 2（与 Tasks 6, 7, 9, 10, 11）
  - **阻塞**: None
  - **被阻塞**: Tasks 5, 6, 7（需要窗口骨架、视频线程、转换工具）

  **参考**（CRITICAL - 要详尽）：

  **模式参考**（要遵循的现有代码）：
  - 无现有视频显示组件

  **API/类型参考**（要实现的契约）：
  - QLabel - PyQt5 标签部件
  - QPixmap - PyQt5 图像类

  **外部参考**（库和框架）：
  - OpenCV 绘图：https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
  - PyQt5 QLabel：https://www.riverbankcomputing.com/static/Docs/PyQt5/widgets/qlabel.html

  **为什么每个参考重要**（解释相关性）：
  - OpenCV - 理解如何在图像上绘制圆
  - QLabel - 理解如何显示图像

  **验收标准**：
  - [ ] `VideoWidget` 类继承 `QLabel`
  - [ ] `update_frame()` 方法接受 (depth, color) 参数
  - [ ] 红点绘制在图像中心（320, 240）
  - [ ] 红点颜色为 (0, 0, 255) BGR 格式
  - [ ] 图像保持宽高比缩放
  - [ ] 可以连接到 VideoWorker 信号

  **QA 场景**（MANDATORY）：

  ```
  场景：验证视频显示
    工具：Bash (pytest-qt)
    前置条件：VideoWidget 存在
    步骤：
      1. 创建 VideoWidget 实例
      2. 调用 update_frame(depth, color) 使用测试数据
      3. 验证 QLabel 有 pixmap
      4. 验证 pixmap 大小合理
    预期结果：QLabel 显示图像
    失败指标：QLabel 没有 pixmap 或 pixmap 为空
    证据：.sisyphus/evidence/task-08-video-display.txt

  场景：验证中心红点
    工具：Bash (pytest-qt + 图像检查)
    前置条件：VideoWidget 可以显示图像
    步骤：
      1. 创建 VideoWidget 实例
      2. 使用纯色图像调用 update_frame
      3. 检查图像中心是否有红点（通过 pixmap 转换回图像并检查像素）
    预期结果：中心像素为红色
    失败指标：中心像素不是红色
    证据：.sisyphus/evidence/task-08-red-dot.txt
  ```

  **要捕获的证据**：
  - [ ] 每个证据文件命名为：task-08-{scenario-slug}.{ext}
  - [ ] pytest 输出日志
  - [ ] 截图（如果可能）

  **提交**: YES
  - 消息：`feat(gui): create video display widget with center dot`
  - 文件：`grasp_gui_v2.py`
  - 预提交：`pytest tests/gui/ -k video_widget -v`

- [ ] 9. 抓取工作线程 + 测试

  **做什么**：
  - 创建 `GraspWorker` 类继承 `QObject`
  - 实现 `run()` 方法执行 `grasp.obj_grasp(label, vis=False)`
  - 创建 `grasp_finished` pyqtSignal（参数：success: bool）
  - 创建 `error_occurred` pyqtSignal（参数：error_msg: str）
  - 支持 Mock 模式
  - 编写单元测试
  - 创建 `tests/gui/test_grasp_worker.py`

  **必须不做**：
  - 不要在 UI 线程中运行
  - 不要直接访问 UI 组件

  **推荐代理配置**：
  - **Category**: `unspecified-high`
    - 原因：多线程编程和硬件交互需要较高的技术理解
  - **Skills**: []

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 2（与 Tasks 6, 7, 8, 10, 11）
  - **阻塞**: Task 14（需要抓取线程）
  - **被阻塞**: Tasks 2, 5（需要 Mock 接口和窗口骨架）

  **参考**：
  - `grasp_zy_zhiyuan1215.py:419-617` - obj_grasp() 方法实现

  **验收标准**：
  - [ ] `GraspWorker` 类继承 `QObject`
  - [ ] `grasp_finished` 信号发射 bool（成功/失败）
  - [ ] `error_occurred` 信号发射错误消息
  - [ ] `run()` 方法调用 `grasp.obj_grasp(label)`
  - [ ] 支持 Mock 模式
  - [ ] `tests/gui/test_grasp_worker.py` 包含至少 6 个测试用例
  - [ ] 所有测试通过

  **QA 场景**：

  ```
  场景：验证抓取线程启动和完成
    工具：Bash (pytest-qt)
    前置条件：tests/gui/test_grasp_worker.py 存在
    步骤：
      1. 创建 GraspWorker 实例（Mock 模式）
      2. 启动线程
      3. 等待 grasp_finished 信号
      4. 验证信号参数是 bool
    预期结果：收到 grasp_finished 信号，参数是 True 或 False
    失败指标：没有收到信号或参数类型错误
    证据：.sisyphus/evidence/task-09-grasp-thread.txt
  ```

  **提交**: YES
  - 消息：`feat(gui): implement grasp worker thread`
  - 文件：`grasp_gui_v2.py`, `tests/gui/test_grasp_worker.py`
  - 预提交：`pytest tests/gui/test_grasp_worker.py -v`

- [ ] 10. 硬件访问序列化机制

  **做什么**：
  - 创建 `HardwareLock` 类管理硬件访问
  - 使用 `QMutex` 实现互斥锁
  - 为 Camera 和 Robot 操作提供序列化访问
  - 防止 VideoWorker 和 GraspWorker 同时访问硬件
  - 添加单元测试

  **必须不做**：
  - 不要在 UI 线程中持有锁过长时间
  - 不要实现复杂的锁策略（保持简单）

  **推荐代理配置**：
  - **Category**: `deep`
    - 原因：线程同步和锁机制需要深入理解
  - **Skills**: []

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 2（与 Tasks 6, 7, 8, 9, 11）
  - **阻塞**: Task 9（需要序列化机制）
  - **被阻塞**: Task 3（需要状态机）

  **参考**：
  - PyQt5 QMutex：https://www.riverbankcomputing.com/static/Docs/PyQt5/qmutex.html

  **验收标准**：
  - [ ] `HardwareLock` 类使用 `QMutex`
  - [ ] 提供 `acquire()` 和 `release()` 方法
  - [ ] VideoWorker 使用锁访问 camera
  - [ ] GraspWorker 使用锁访问 robot
  - [ ] 单元测试验证互斥行为

  **QA 场景**：

  ```
  场景：验证硬件访问互斥
    工具：Bash (pytest)
    前置条件：HardwareLock 实现
    步骤：
      1. 创建 HardwareLock 实例
      2. 模拟两个线程同时请求锁
      3. 验证只有一个线程获得锁
    预期结果：锁的互斥行为正确
    失败指标：两个线程同时获得锁
    证据：.sisyphus/evidence/task-10-hardware-lock.txt
  ```

  **提交**: YES
  - 消息：`feat(gui): add hardware access serialization`
  - 文件：`grasp_gui_v2.py`
  - 预提交：`pytest tests/gui/ -k hardware -v`

- [ ] 11. 控制面板（按钮 + 下拉框）

  **做什么**：
  - 创建 `ControlPanel` 类继承 `QWidget`
  - 添加 [初始化] 按钮
  - 添加 [物体选择] QComboBox（8 种物体）
  - 添加 [开始抓取] 按钮
  - 添加 [紧急停止] 按钮
  - 连接按钮点击信号到槽函数
  - 实现按钮启用/禁用逻辑

  **必须不做**：
  - 不要在控制面板中实现业务逻辑（只处理 UI）
  - 不要硬编码物体列表（从配置或模型获取）

  **推荐代理配置**：
  - **Category**: `visual-engineering`
    - 原因：GUI 组件布局和交互
  - **Skills**: [`/frontend-ui-ux`]

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 2（与 Tasks 6, 7, 8, 9, 10）
  - **阻塞**: Task 14（需要控制面板）
  - **被阻塞**: Tasks 3, 5（需要状态机和窗口骨架）

  **参考**：
  - `grasp_zy_zhiyuan1215.py:624-637` - 物体列表定义
  - PyQt5 QPushButton：https://www.riverbankcomputing.com/static/Docs/PyQt5/widgets/qpushbutton.html
  - PyQt5 QComboBox：https://www.riverbankcomputing.com/static/Docs/PyQt5/widgets/qcombobox.html

  **验收标准**：
  - [ ] `ControlPanel` 类继承 `QWidget`
  - [ ] 包含 4 个按钮
  - [ ] 包含 1 个 QComboBox
  - [ ] QComboBox 包含 8 种物体
  - [ ] 所有按钮有点击信号

  **QA 场景**：

  ```
  场景：验证控制面板布局
    工具：Bash (pytest-qt)
    前置条件：ControlPanel 存在
    步骤：
      1. 创建 ControlPanel 实例
      2. 查找所有 QPushButton
      3. 查找 QComboBox
      4. 验证数量正确
    预期结果：4 个按钮，1 个下拉框
    失败指标：控件数量不正确
    证据：.sisyphus/evidence/task-11-control-panel.txt
  ```

  **提交**: YES
  - 消息：`feat(gui): create control panel with buttons and combo box`
  - 文件：`grasp_gui_v2.py`
  - 预提交：`pytest tests/gui/ -k control_panel -v`

- [ ] 12. 日志显示组件

  **做什么**：
  - 创建 `LogWidget` 类继承 `QTextEdit`
  - 设置为只读模式
  - 实现 `append_log(message)` 方法
  - 连接 LogBridge 的 `message_emitted` 信号
  - 自动滚动到底部
  - 设置最大行数限制（防止内存溢出）

  **必须不做**：
  - 不要在日志中显示敏感信息
  - 不要实现复杂的日志过滤（超出 v1 范围）

  **推荐代理配置**：
  - **Category**: `visual-engineering`
    - 原因：GUI 组件
  - **Skills**: [`/frontend-ui-ux`]

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 3（与 Tasks 13, 14, 15, 16, 17）
  - **阻塞**: None
  - **被阻塞**: Tasks 4, 5（需要日志桥接和窗口骨架）

  **参考**：
  - PyQt5 QTextEdit：https://www.riverbankcomputing.com/static/Docs/PyQt5/widgets/qtextedit.html

  **验收标准**：
  - [ ] `LogWidget` 类继承 `QTextEdit`
  - [ ] 设置为只读
  - [ ] `append_log()` 方法追加日志
  - [ ] 自动滚动到底部
  - [ ] 最大行数限制（如 1000 行）

  **QA 场景**：

  ```
  场景：验证日志显示
    工具：Bash (pytest-qt)
    前置条件：LogWidget 存在
    步骤：
      1. 创建 LogWidget 实例
      2. 调用 append_log("测试消息")
      3. 验证文本包含 "测试消息"
    预期结果：日志显示在组件中
    失败指标：日志不显示
    证据：.sisyphus/evidence/task-12-log-widget.txt
  ```

  **提交**: YES
  - 消息：`feat(gui): create log display widget`
  - 文件：`grasp_gui_v2.py`
  - 预提交：`pytest tests/gui/ -k log_widget -v`

- [ ] 13. 状态显示组件

  **做什么**：
  - 创建 `StatusWidget` 类继承 `QWidget`
  - 显示机械臂速度（QLabel）
  - 显示选中物体（QLabel）
  - 实现 `update_status(speed, object_name)` 方法
  - 使用 QGridLayout 布局

  **必须不做**：
  - 不要显示过多状态信息（保持简洁）
  - 不要实现实时刷新（按需更新）

  **推荐代理配置**：
  - **Category**: `visual-engineering`
    - 原因：GUI 组件
  - **Skills**: [`/frontend-ui-ux`]

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 3（与 Tasks 12, 14, 15, 16, 17）
  - **阻塞**: None
  - **被阻塞**: Tasks 5（需要窗口骨架）

  **参考**：
  - `grasp_zy_zhiyuan1215.py:36` - robot_speed 默认值

  **验收标准**：
  - [ ] `StatusWidget` 类继承 `QWidget`
  - [ ] 显示速度和物体名称
  - [ ] `update_status()` 方法更新显示
  - [ ] 初始显示默认值

  **QA 场景**：

  ```
  场景：验证状态显示
    工具：Bash (pytest-qt)
    前置条件：StatusWidget 存在
    步骤：
      1. 创建 StatusWidget 实例
      2. 调用 update_status(30, "banana")
      3. 验证显示包含 "30" 和 "banana"
    预期结果：状态正确显示
    失败指标：状态不显示或不正确
    证据：.sisyphus/evidence/task-13-status-widget.txt
  ```

  **提交**: YES
  - 消息：`feat(gui): create status display widget`
  - 文件：`grasp_gui_v2.py`
  - 预提交：`pytest tests/gui/ -k status_widget -v`

- [ ] 14. 按钮状态管理

  **做什么**：
  - 实现 `update_button_states()` 方法
  - 根据 GUI 状态启用/禁用按钮
  - 状态机规则：
    - IDLE：初始化启用，开始抓取启用，紧急停止禁用
    - INITIALIZING：所有按钮禁用
    - READY：开始抓取启用，紧急停止启用
    - GRASPING：所有按钮禁用（除紧急停止）
    - STOPPING：所有按钮禁用
    - FAULT：初始化启用，其他禁用
  - 连接状态机的 `state_changed` 信号

  **必须不做**：
  - 不要允许在 GRASPING 状态下点击开始抓取（按钮禁用）
  - 不要允许在 GRASPING 状态下点击初始化（按钮禁用）

  **推荐代理配置**：
  - **Category**: `unspecified-high`
    - 原因：状态管理逻辑需要仔细设计
  - **Skills**: []

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 3（与 Tasks 12, 13, 15, 16, 17）
  - **阻塞**: None
  - **被阻塞**: Tasks 3, 9, 11（需要状态机、抓取线程、控制面板）

  **参考**：
  - 计划中的状态机定义（GUIState 枚举）

  **验收标准**：
  - [ ] `update_button_states()` 方法存在
  - [ ] 每个状态的按钮状态正确
  - [ ] 状态变化时按钮状态自动更新
  - [ ] 单元测试覆盖所有状态

  **QA 场景**：

  ```
  场景：验证按钮状态规则
    工具：Bash (pytest-qt)
    前置条件：按钮状态管理实现
    步骤：
      1. 设置状态为 GRASPING
      2. 检查开始抓取按钮是否禁用
      3. 检查紧急停止按钮是否启用
    预期结果：按钮状态符合规则
    失败指标：按钮状态不正确
    证据：.sisyphus/evidence/task-14-button-states.txt
  ```

  **提交**: YES
  - 消息：`feat(gui): implement button state management`
  - 文件：`grasp_gui_v2.py`
  - 预提交：`pytest tests/gui/ -k button_state -v`

- [ ] 15. 窗口关闭处理 + 资源清理

  **做什么**：
  - 重写 `closeEvent(event)` 方法
  - 停止 VideoWorker 线程
  - 停止 GraspWorker 线程（如果正在运行）
  - 调用 `camera.stop()`（如果存在）
  - 调用 `robot.rm_delete_robot_arm()`（如果存在）
  - 恢复 sys.stdout
  - 编写单元测试
  - 创建 `tests/gui/test_shutdown.py`

  **必须不做**：
  - 不要在有线程运行时直接关闭窗口（等待线程停止）
  - 不要忽略清理错误（记录到日志）

  **推荐代理配置**：
  - **Category**: `unspecified-high`
    - 原因：资源清理和线程管理需要仔细处理
  - **Skills**: []

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 3（与 Tasks 12, 13, 14, 16, 17）
  - **阻塞**: Task 16（需要关闭处理）
  - **被阻塞**: Tasks 6, 9（需要视频和抓取线程）

  **参考**：
  - `camera.py:82-83` - camera.stop() 方法
  - PyQt5 closeEvent：https://www.riverbankcomputing.com/static/Docs/PyQt5/qcloseevent.html

  **验收标准**：
  - [ ] `closeEvent()` 方法重写
  - [ ] 停止所有工作线程
  - [ ] 调用 camera.stop()
  - [ ] 恢复 sys.stdout
  - [ ] `tests/gui/test_shutdown.py` 包含至少 5 个测试用例
  - [ ] 所有测试通过

  **QA 场景**：

  ```
  场景：验证窗口关闭清理
    工具：Bash (pytest-qt)
    前置条件：关闭处理实现
    步骤：
      1. 创建 GUI 实例
      2. 启动视频线程
      3. 关闭窗口
      4. 验证线程已停止
      5. 验证资源已清理
    预期结果：所有资源正确清理
    失败指标：资源泄漏或线程未停止
    证据：.sisyphus/evidence/task-15-shutdown.txt
  ```

  **提交**: YES
  - 消息：`feat(gui): add window close handling and resource cleanup`
  - 文件：`grasp_gui_v2.py`, `tests/gui/test_shutdown.py`
  - 预提交：`pytest tests/gui/test_shutdown.py -v`

- [ ] 16. Mock 模式集成 + 测试

  **做什么**：
  - 添加 `--mock` 命令行参数解析（argparse）
  - 在 Mock 模式下使用 MockGrasp, MockCamera, MockRobot
  - 在真实模式下使用真实硬件
  - 更新 GUI 初始化逻辑
  - 编写集成测试
  - 创建 `tests/gui/test_integration.py`

  **必须不做**：
  - 不要在真实模式下使用 Mock 对象
  - 不要在 Mock 模式下尝试连接真实硬件

  **推荐代理配置**：
  - **Category**: `unspecified-high`
    - 原因：需要集成多个组件并处理命令行参数
  - **Skills**: []

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 3（与 Tasks 12, 13, 14, 15, 17）
  - **阻塞**: Task 17（需要 Mock 模式）
  - **被阻塞**: Tasks 1, 2, 6, 9, 15（需要依赖、Mock 接口、线程、关闭处理）

  **参考**：
  - `tests/gui/mocks.py` - Mock 类实现

  **验收标准**：
  - [ ] `--mock` 参数解析正确
  - [ ] Mock 模式使用 Mock 对象
  - [ ] 真实模式使用真实硬件
  - [ ] `tests/gui/test_integration.py` 包含至少 4 个测试用例
  - [ ] 所有测试通过

  **QA 场景**：

  ```
  场景：验证 Mock 模式启动
    工具：Bash
    前置条件：grasp_gui_v2.py 存在
    步骤：
      1. 运行 `python grasp_gui_v2.py --mock`
      2. 检查 GUI 启动成功
      3. 检查不连接真实硬件
    预期结果：GUI 在 Mock 模式下启动
    失败指标：GUI 启动失败或尝试连接硬件
    证据：.sisyphus/evidence/task-16-mock-mode.txt
  ```

  **提交**: YES
  - 消息：`feat(gui): integrate mock mode with CLI argument`
  - 文件：`grasp_gui_v2.py`
  - 预提交：`pytest tests/gui/test_integration.py -v`

- [ ] 17. 集成测试 + 文档

  **做什么**：
  - 创建完整的集成测试套件
  - 测试所有组件协同工作
  - 测试 Mock 模式下的完整流程
  - 创建 `tests/gui/test_integration.py`
  - 更新 `README.md` 添加 GUI 使用说明
  - 添加依赖安装说明
  - 添加运行命令说明

  **必须不做**：
  - 不要在集成测试中依赖真实硬件
  - 不要添加过于详细的 API 文档（保持简洁）

  **推荐代理配置**：
  - **Category**: `writing`
    - 原因：测试和文档编写
  - **Skills**: []

  **并行化**：
  - **可以并行运行**: YES
  - **并行组**: Wave 3（与 Tasks 12, 13, 14, 15, 16）
  - **阻塞**: None
  - **被阻塞**: Tasks 1-16（需要所有组件完成）

  **参考**：
  - 现有 README.md 格式

  **验收标准**：
  - [ ] `tests/gui/test_integration.py` 包含至少 10 个测试用例
  - [ ] 所有集成测试通过
  - [ ] README.md 包含 GUI 使用说明
  - [ ] 包含依赖安装说明
  - [ ] 包含运行命令说明

  **QA 场景**：

  ```
  场景：验证完整集成测试
    工具：Bash (pytest)
    前置条件：所有组件完成
    步骤：
      1. 运行 `pytest tests/gui/ -v`
      2. 检查所有测试通过
    预期结果：所有测试通过，0 失败
    失败指标：任何测试失败
    证据：.sisyphus/evidence/task-17-integration-tests.txt

  场景：验证文档完整性
    工具：Bash
    前置条件：README.md 更新
    步骤：
      1. 检查 README.md 包含 "GUI" 关键字
      2. 检查包含安装说明
      3. 检查包含运行命令
    预期结果：文档包含所有必需信息
    失败指标：缺少关键信息
    证据：.sisyphus/evidence/task-17-documentation.txt
  ```

  **提交**: YES
  - 消息：`test(gui): add integration tests and documentation`
  - 文件：`tests/gui/test_integration.py`, `README.md`
  - 预提交：`pytest tests/gui/ -v`

---

## Final Verification Wave (MANDATORY — 在所有实现任务之后)

> 4 个审查代理并行运行。所有必须批准。向用户呈现综合结果并在完成前获取明确的"确认"。
>
> **不要在验证后自动继续。等待用户的明确批准后再将工作标记为完成。**
> **永远不要在获取用户确认之前将 F1-F4 标记为已检查。** 拒绝或用户反馈 -> 修复 -> 重新运行 -> 再次呈现 -> 等待确认。

- [ ] F1. **计划合规性审计** — `oracle`
  通读计划。对于每个"必须有"：验证实现存在（读文件、curl 端点、运行命令）。对于每个"必须没有"：搜索代码库中的禁止模式 — 如果发现则拒绝并提供 file:line。检查 .sisyphus/evidence/ 中是否存在证据文件。将交付物与计划进行比较。
  输出：`必须有 [N/N] | 必须没有 [N/N] | 任务 [N/N] | 结论：批准/拒绝`

- [ ] F2. **代码质量审查** — `unspecified-high`
  运行 `tsc --noEmit` + linter + `bun test`。审查所有更改的文件：`as any`/`@ts-ignore`、空 catch、console.log 在生产中、注释掉的代码、未使用的导入。检查 AI slop：过度注释、过度抽象、通用名称（data/result/item/temp）。
  输出：`构建 [通过/失败] | Lint [通过/失败] | 测试 [N 通过/N 失败] | 文件 [N 干净/N 问题] | 结论`

- [ ] F3. **真实手动 QA** — `unspecified-high`（+ `playwright` skill 如果 UI）
  从干净状态开始。执行每个任务的每个 QA 场景 — 遵循确切步骤、捕获证据。测试跨任务集成（功能协同工作，而非隔离）。测试边缘情况：空状态、无效输入、快速操作。保存到 `.sisyphus/evidence/final-qa/`。
  输出：`场景 [N/N 通过] | 集成 [N/N] | 边缘情况 [N 测试] | 结论`

- [ ] F4. **范围保真度检查** — `deep`
  对于每个任务：读"做什么"，读实际差异（git log/diff）。验证 1:1 — 规范中的所有内容都已构建（无遗漏），规范之外没有构建任何内容（无蔓延）。检查"必须没有"合规性。检测跨任务污染：任务 N 触及任务 M 的文件。标记未说明的更改。
  输出：`任务 [N/N 合规] | 污染 [干净/N 问题] | 未说明 [干净/N 文件] | 结论`

---

## Commit Strategy

- **1**: `chore(deps): add PyQt5 and pytest-qt to requirements` — requirements.txt
- **2**: `feat(gui): add mock interfaces for testing` — tests/gui/mocks.py
- **3**: `feat(gui): implement state machine with tests` — grasp_gui_v2.py, tests/gui/test_state_machine.py
- **4**: `feat(gui): add log bridge for stdout redirection` — grasp_gui_v2.py, tests/gui/test_log_bridge.py
- **5**: `feat(gui): create main window skeleton with grid layout` — grasp_gui_v2.py
- **6**: `feat(gui): implement video worker thread` — grasp_gui_v2.py, tests/gui/test_video_worker.py
- **7**: `feat(gui): add image conversion utilities` — grasp_gui_v2.py
- **8**: `feat(gui): create video display widget with center dot` — grasp_gui_v2.py
- **9**: `feat(gui): implement grasp worker thread` — grasp_gui_v2.py, tests/gui/test_grasp_worker.py
- **10**: `feat(gui): add hardware access serialization` — grasp_gui_v2.py
- **11**: `feat(gui): create control panel with buttons and combo box` — grasp_gui_v2.py
- **12**: `feat(gui): create log display widget` — grasp_gui_v2.py
- **13**: `feat(gui): create status display widget` — grasp_gui_v2.py
- **14**: `feat(gui): implement button state management` — grasp_gui_v2.py
- **15**: `feat(gui): add window close handling and resource cleanup` — grasp_gui_v2.py, tests/gui/test_shutdown.py
- **16**: `feat(gui): integrate mock mode with CLI argument` — grasp_gui_v2.py
- **17**: `test(gui): add integration tests and documentation` — tests/gui/test_integration.py, README.md

---

## Success Criteria

### 验证命令
```bash
# 运行所有自动化测试
pytest tests/gui/ -v

# 在 Mock 模式下启动 GUI（无硬件）
python grasp_gui_v2.py --mock

# 在真实硬件模式下启动 GUI
python grasp_gui_v2.py

# 检查代码质量
flake8 grasp_gui_v2.py tests/gui/
mypy grasp_gui_v2.py
```

### 最终清单
- [ ] 所有"必须有"存在
- [ ] 所有"必须没有"不存在
- [ ] 所有测试通过
- [ ] Mock 模式正常工作
- [ ] 真实硬件模式正常工作
- [ ] 窗口关闭时正确清理资源
- [ ] 线程安全得到保证
- [ ] GUI 状态机正确工作
- [ ] 所有按钮状态管理正确
