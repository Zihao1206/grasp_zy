# 编写 RM65 机械臂视觉抓取系统使用说明书

## TL;DR

> **Quick Summary**: 为 `grasp_zy_zhiyuan1215.py` 和 `ctu_conn.py` 编写一份面向非技术企业人员的全中文独立使用说明书，使不懂代码的现场人员也能在新的 RM65 机械臂设备上完成部署、配置和日常使用。
> 
> **Deliverables**:
> - 一份独立的中文 Markdown 使用说明书：`USAGE_GUIDE.md`（项目根目录）
> - 包含截图占位符、配置参数对照表、操作步骤、故障排查等完整内容
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - 顺序写作，单个文档输出
> **Critical Path**: 文档规划 → 逐章节编写 → 审校 → 完成

---

## Context

### Original Request
用户要求针对 `grasp_zy_zhiyuan1215.py` 和 `ctu_conn.py` 编写使用说明，让不懂代码的企业人员也能在别的机械臂上调通并使用。CTU 小车通过 `ctu_conn.py` 发送指令进行外部控制。

### Interview Summary
**Key Discussions**:
- 目标场景：同型号 RM65 机械臂 + 同型号相机的复现部署（换到另一台同型号设备上）
- 文档位置：独立新文档，不合并到现有 README.md
- 语言：全中文
- 标定：简化版——参数说明 + 简化的标定指引（指向工具或已有文档）
- 受众：不懂代码的企业现场操作人员

**Research Findings**:
- 项目已有文档：README.md（技术向）、CTUArm.md（基础操作）、ctu_protocol.md（协议规范）、doc/RunGraspd.service_manual.md（服务配置）
- 关键硬编码参数分布在 4 个文件中：`grasp_zy_zhiyuan1215.py`、`ctu_conn.py`、`config.py`、`camera.py`
- 系统运行链路：RunGraspd.service → ctu_conn.py → Grasp(hardware=True) → 机械臂+相机+模型
- 现有 CTUArm.md 已有基础操作步骤（小车操作），但缺少参数配置和系统部署说明

### Self Gap Analysis（替代 Metis）
**Identified Gaps**（已处理）:
- Gap: 需要明确文档是给"部署工程师"还是"日常操作工"看 → Decision: 面向部署工程师（需要配置参数）+ 日常操作工（需要启停系统）两个角色
- Gap: 换设备后哪些参数必须改、哪些可以不改 → 已通过代码审查梳理出完整清单
- Gap: 标定步骤多复杂 → 用户选择简化版，指向工具即可
- Gap: 是否需要包含物品类别扩展指南 → 需要，因为 GoogsMapping 是企业最常修改的配置
- Gap: 故障排查需要覆盖哪些场景 → 基于代码中的错误处理逻辑梳理

---

## Work Objectives

### Core Objective
编写一份结构清晰、图文并茂的全中文使用说明书，让非技术背景的企业人员能够：
1. 理解系统的整体工作原理（不超过一页纸的概述）
2. 在新设备上完成全部硬件连接和网络配置
3. 正确修改所有必要的配置参数
4. 完成简化版标定（相机内参 + 手眼标定）
5. 启动/停止系统并进行日常操作
6. 排查常见的部署和运行问题

### Concrete Deliverables
- `USAGE_GUIDE.md` — 项目根目录下的独立使用说明书

### Definition of Done
- [ ] `USAGE_GUIDE.md` 文件存在于项目根目录
- [ ] 文档覆盖所有必要的参数配置项（每个参数标注：在哪个文件、第几行、改成什么）
- [ ] 文档包含从零开始的完整部署流程
- [ ] 文档包含故障排查章节
- [ ] 文档语言为全中文，无技术术语未解释的情况

### Must Have
- 每个需要修改的参数都标注：文件名、行号、当前值、如何获取新值
- 网络配置的完整步骤（含截图占位符）
- 系统启停的明确操作步骤
- CTU 通信流程的通俗解释
- 物品类别扩展方法
- 常见问题排查表
- 简化版标定指引

### Must NOT Have (Guardrails)
- 不要写代码实现细节（受众不懂代码）
- 不要在文档中使用英文术语而不加中文解释
- 不要假设读者有 Python 或 Linux 基础
- 不要照搬 README.md 的技术文档内容（那是给开发者看的）
- 不要包含 GUI 相关内容（grasp_gui_v2.py 不在本次范围内）
- 不要修改任何现有代码文件——本次只产出文档
- 避免过度冗长——每个章节聚焦实操步骤，不展开理论

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: N/A（文档任务）
- **Automated tests**: None
- **Framework**: N/A

### QA Policy
每个任务包含 agent-executed QA scenarios：
- 使用 Bash 工具读取生成的文档，验证关键章节存在
- 使用 Grep 搜索文档，验证关键参数引用的准确性
- 验证文件路径、行号引用与实际代码一致

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (单任务 — 文档规划):
└── Task 1: 文档结构规划与参数清单整理 [quick]

Wave 2 (核心写作 — 顺序执行):
├── Task 2: 编写第1-2章（系统概述 + 硬件连接） [writing]
├── Task 3: 编写第3-4章（网络配置 + 软件环境） [writing]
├── Task 4: 编写第5章（参数配置指南 — 核心章节） [writing]
├── Task 5: 编写第6章（标定指引） [writing]
├── Task 6: 编写第7章（系统启停与CTU通信） [writing]
├── Task 7: 编写第8章（物品类别扩展） [writing]
└── Task 8: 编写第9章（故障排查 + 附录） [writing]

Wave FINAL (验证):
└── Task F1: 文档完整性验证 + 参数引用准确性检查 [quick]
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1    | -         | 2-8    |
| 2    | 1         | F1     |
| 3    | 1         | F1     |
| 4    | 1         | F1     |
| 5    | 1         | F1     |
| 6    | 1         | F1     |
| 7    | 1         | F1     |
| 8    | 1         | F1     |
| F1   | 2-8       | -      |

### Agent Dispatch Summary

- **Wave 1**: 1 task — T1 → `quick`
- **Wave 2**: 7 tasks — T2-T8 → `writing`
- **FINAL**: 1 task — F1 → `quick`

---

## TODOs

- [x] 1. 文档结构规划与参数清单整理

  **What to do**:
  - 阅读以下代码文件，提取所有硬编码的关键参数，整理成"参数清单表"：
    - `grasp_zy_zhiyuan1215.py` — 机械臂IP(46行)、初始位姿(41行)、各关节位姿(56-64行)、Tcam2base矩阵(66-71行)、末端位姿(50-55行)、模型路径(84/86-87行)、图像裁剪范围(385-386/301-303行)、抓取补偿参数(296-297/300行)
    - `ctu_conn.py` — CTU IP和端口(163行)、GoogsMapping物品映射(9-22行)
    - `config.py` — Tcam2base(4-8行)、pose2/pose2_2(36-37行)、robot_speed(25/34行)、边缘范围(21-23行)、angle(14行)
    - `camera.py` — 相机内参矩阵(38-42行)
  - 设计文档的完整章节目录结构
  - 确定每个章节的写作要点和目标读者（部署工程师 vs 操作工）
  - 将参数按"必须修改"、"可能需要修改"、"一般不改"三级分类

  **Must NOT do**:
  - 不要开始写正式文档内容，只做规划
  - 不要修改任何代码文件

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 信息整理任务，需要精确读取代码文件并提取参数
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `writing`: 此步骤是信息整理，不是文案写作

  **Parallelization**:
  - **Can Run In Parallel**: NO（后续写作依赖此步骤的输出）
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: Tasks 2-8
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `grasp_zy_zhiyuan1215.py:36-94` — Grasp.__init__() 中所有硬编码参数
  - `ctu_conn.py:9-22` — GoogsMapping 物品类别映射表
  - `ctu_conn.py:157-174` — __main__ 中的连接配置
  - `config.py:1-38` — 全局配置参数
  - `camera.py:37-42` — 相机内参矩阵

  **External References**:
  - `CTUArm.md` — 已有的基础操作指引，新文档需要补充但不要重复
  - `ctu_protocol.md` — 通信协议详细说明，新文档只需通俗概括

  **WHY Each Reference Matters**:
  - Grasp.__init__ 是参数最集中的位置，换设备后必须调整的参数几乎都在这里
  - GoogsMapping 决定了 CTU 指定的物品代码对应什么检测类别
  - config.py 中的参数被 grasp_zy_zhiyuan1215.py 引用，需要同时说明两处关系

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 参数清单完整性检查
    Tool: Bash (grep)
    Preconditions: 参数清单已整理完毕（可以写在文档开头作为注释或单独输出）
    Steps:
      1. 在项目根目录 grep -n "192.168" grasp_zy_zhiyuan1215.py ctu_conn.py
      2. 验证找到的 IP 地址条目数 >= 2（机械臂IP + CTU IP）
      3. grep -n "init_pose\|mid_pose\|place.*pose" grasp_zy_zhiyuan1215.py
      4. 验证找到的位姿定义 >= 6 个
      5. grep -n "Tcam2base" grasp_zy_zhiyuan1215.py config.py
      6. 验证找到的变换矩阵定义 >= 2 处
    Expected Result: 所有硬编码参数位置被确认，清单覆盖无遗漏
    Evidence: .sisyphus/evidence/task-1-param-audit.txt

  Scenario: 章节结构合理性检查
    Tool: Bash (grep)
    Preconditions: 章节目录已设计
    Steps:
      1. 检查章节目录包含以下关键词：概述、硬件、网络、配置、标定、启停、类别、故障
      2. 验证每个章节有明确的目标读者标注
    Expected Result: 9个关键主题全部覆盖
    Evidence: .sisyphus/evidence/task-1-outline.txt
  ```

  **Commit**: NO（规划阶段不提交）

- [x] 2. 编写第1-2章：系统概述 + 硬件连接

  **What to do**:
  - 编写 USAGE_GUIDE.md 的第1章"系统概述"：
    - 用通俗语言描述系统做什么（CTU小车到达指定位置 → 识别物品 → 机械臂抓取 → 放到指定位置）
    - 列出系统包含的硬件清单（RM65机械臂、Intel RealSense相机、Jetson Orin NX开发板、CTU小车）
    - 用简化的流程图描述工作流程（文字描述即可，不需要实际画图）
    - 说明两个核心文件的分工（ctu_conn.py 负责与CTU通信，grasp_zy_zhiyuan1215.py 负责识别和抓取）
  - 编写第2章"硬件连接"：
    - 硬件清单表格（设备名、型号、数量、备注）
    - 连接拓扑图描述（哪些设备用网线连、哪些用USB连）
    - 机械臂上电步骤（参考 CTUArm.md 第1节）
    - 相机安装位置说明（固定在机械臂末端附近，朝下拍摄）
    - 开发板放置位置
    - 所有网线连接关系（开发板↔机械臂、开发板↔CTU交换机/MOXA）

  **Must NOT do**:
  - 不要写代码实现细节
  - 不要假设读者知道什么是 TCP/IP、串口等概念——需要用通俗语言解释
  - 不要复制粘贴 CTUArm.md 原文——用自己的话重写

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 纯文案写作任务，需要通俗易懂的中文表达
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 3 可并行，但建议顺序写以保持文风一致）
  - **Parallel Group**: Wave 2
  - **Blocks**: F1
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `CTUArm.md:1-134` — 已有的硬件操作步骤，可作为参考但不能照搬
  - `README.md:1-30` — 项目概述部分，提取通俗化的要素

  **WHY Each Reference Matters**:
  - CTUArm.md 有实际操作经验和截图描述，新文档需要把这些转化成更易懂的文字
  - README.md 的项目概述偏向技术，需要用非技术人员能懂的语言重新描述

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 系统概述可读性检查
    Tool: Bash (grep)
    Preconditions: USAGE_GUIDE.md 已创建并写入第1-2章
    Steps:
      1. grep -c "CTU" USAGE_GUIDE.md，验证 CTU 概念被提及 >= 3 次
      2. grep -c "机械臂" USAGE_GUIDE.md，验证机械臂被提及 >= 5 次
      3. grep -c "抓取" USAGE_GUIDE.md，验证抓取流程被描述 >= 3 次
      4. 检查文档开头不包含代码块或技术术语堆砌
    Expected Result: 文档开头用通俗语言描述系统功能，无未解释的技术术语
    Evidence: .sisyphus/evidence/task-2-chapter1-2.md

  Scenario: 硬件清单完整性
    Tool: Bash (grep)
    Preconditions: 第2章已写入
    Steps:
      1. grep "RM65\|RealSense\|Jetson\|CTU" USAGE_GUIDE.md
      2. 验证这4个设备名都被提及
    Expected Result: 4个核心硬件全部列出
    Evidence: .sisyphus/evidence/task-2-hardware-check.txt
  ```

  **Commit**: NO（等全部章节完成后一起提交）

- [x] 3. 编写第3-4章：网络配置 + 软件环境

  **What to do**:
  - 编写第3章"网络配置"：
    - 网络拓扑说明（所有设备在 192.168.127.x 网段）
    - 设备IP地址对照表（开发板、机械臂、CTU各自的IP和端口）
    - 如何确认网络连通（ping 命令步骤）
    - 如何修改各设备的IP（config.py、ctu_conn.py、RunGraspd.service中涉及IP的位置）
    - MOXA配置步骤（简化版，参考 CTUArm.md 第3节）
    - 网线连接图（文字描述 + 占位符说明实际布线）
  - 编写第4章"软件环境安装"：
    - 前提条件确认（Ubuntu系统版本、CUDA版本）
    - Conda 环境创建步骤（zy_torch 环境，Python 版本）
    - 依赖安装命令（pip install -r requirements.txt）
    - 机械臂 SDK 安装说明（robotic_arm_package 目录）
    - 验证安装是否成功的检查命令
    - 项目文件部署（把代码放到开发板的哪个目录，路径说明）

  **Must NOT do**:
  - 不要省略网络配置的任何一步——这是企业部署最容易出错的地方
  - 不要写 GUI 相关的安装步骤

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 步骤性文档写作
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 2 可并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: F1
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `CTUArm.md:23-62` — MOXA配置步骤
  - `CTUArm.md:63-88` — 网线连接和SSH登录步骤
  - `README.md:25-45` — 安装步骤和快速开始
  - `RunGraspd.service:1-22` — 服务中的IP和路径配置
  - `RunGrasp.sh:1-24` — 启动脚本中的路径配置
  - `doc/RunGraspd.service_manual.md:1-276` — 服务配置完整手册
  - `requirements.txt:1-43` — 依赖列表

  **API/Type References**:
  - `grasp_zy_zhiyuan1215.py:46` — 机械臂IP "192.168.127.101"
  - `ctu_conn.py:163` — CTU IP "192.168.127.253", port 8899
  - `RunGraspd.service:13-15` — WorkingDirectory 和 ExecStart 中的路径

  **WHY Each Reference Matters**:
  - 网络配置是企业部署的第一道门槛，IP地址散落在多个文件中
  - 服务配置手册已有详细步骤，新文档需要简化提取关键操作
  - requirements.txt 决定了依赖安装命令

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: IP地址一致性验证
    Tool: Bash (grep)
    Preconditions: 第3章已写入
    Steps:
      1. grep "192.168.127.101" USAGE_GUIDE.md，验证机械臂IP被正确引用
      2. grep "192.168.127.253" USAGE_GUIDE.md，验证CTU IP被正确引用
      3. grep "8899" USAGE_GUIDE.md，验证CTU端口被正确引用
      4. grep "8080" USAGE_GUIDE.md，验证机械臂端口被正确引用
    Expected Result: 文档中的IP和端口与代码完全一致
    Evidence: .sisyphus/evidence/task-3-ip-check.txt

  Scenario: 安装步骤可执行性
    Tool: Bash (grep)
    Preconditions: 第4章已写入
    Steps:
      1. grep "conda" USAGE_GUIDE.md，验证包含conda环境创建步骤
      2. grep "pip install" USAGE_GUIDE.md，验证包含依赖安装命令
      3. grep "requirements" USAGE_GUIDE.md，验证引用了正确的依赖文件
    Expected Result: 包含完整的环境搭建步骤
    Evidence: .sisyphus/evidence/task-3-install-check.txt
  ```

  **Commit**: NO（等全部章节完成后一起提交）

- [x] 4. 编写第5章：参数配置指南（核心章节）

  **What to do**:
  - 这是整份文档最关键的章节，需要用表格形式列出所有换设备后可能需要修改的参数
  - 按以下结构组织：

  **5.1 参数总览表**
  - 表格列：参数名 | 所在文件 | 行号 | 当前值 | 何时需要改 | 如何获取新值
  - 将参数按"必须修改"/"可能需要修改"/"一般不改"分类标注

  **5.2 网络参数配置**
  - 机械臂 IP 地址（grasp_zy_zhiyuan1215.py 第46行）
  - CTU IP 和端口（ctu_conn.py 第163行）
  - 修改方法和验证步骤

  **5.3 相机参数配置**
  - Tcam2base 相机到基座变换矩阵（grasp_zy_zhiyuan1215.py 第66-71行 + config.py 第4-8行）
  - 相机内参矩阵（camera.py 第38-42行）
  - 图像裁剪范围（grasp_zy_zhiyuan1215.py 第385-386行，当前为 [:, 80:560]）
  - 每个参数的修改位置、格式说明

  **5.4 机械臂位姿参数**
  - 初始位姿 init_pose（grasp_zy_zhiyuan1215.py 第56行）
  - 中间位姿 mid_pose / mid_pose1（第57-58行）
  - 回转位姿 lift2init_pose（第61行）
  - 放置位姿 place_mid_pose / place_mid_pose2 / place_last_pose（第62-64行 + config.py 第1行）
  - 如何获取这些位姿值（示教方式或读取当前关节角）
  - Z轴高度限制（config.py 第36-37行 pose2/pose2_2）

  **5.5 抓取补偿参数**
  - 边缘倾斜角度 slope_angle（grasp_zy_zhiyuan1215.py 第300行，当前 π/8）
  - 边缘检测范围（第301-303行 column_left/right, row_up/down）
  - TCP补偿（第296-297行 t_tcp_flange, tcp_compensate）
  - 深度偏移（第288行 +70 像素偏移）
  - 说明：这些参数一般不需要改，除非抓取精度有问题

  **5.6 模型路径配置**
  - 抓取模型权重（grasp_zy_zhiyuan1215.py 第84行）
  - 检测模型配置文件（第86行）
  - 检测模型权重（第87行）
  - 基因描述文件（第76行 doc/single_new.txt）

  **5.7 速度与安全参数**
  - 机械臂速度（grasp_zy_zhiyuan1215.py 第39行 + config.py 第25/34行，范围0-50）
  - 碰撞检测等级（第47行）
  - 最大抓取次数（ctu_conn.py 第116行，物品数+5）
  - 最大逆解失败次数（ctu_conn.py 第120行，3次）

  - 每个参数都提供：
    - 具体的文件路径和行号
    - 当前值的含义解释（用通俗语言）
    - 什么情况下需要修改
    - 修改后的验证方法

  **Must NOT do**:
  - 不要遗漏任何一个硬编码参数——这张表是企业部署的唯一参考
  - 不要用技术术语解释参数——用"抓取位置"、"相机朝向"等通俗说法
  - 不要修改任何代码文件

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 核心章节写作，需要精确的参数引用 + 通俗解释
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES（与 Task 2-3 可并行）
  - **Parallel Group**: Wave 2
  - **Blocks**: F1
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `grasp_zy_zhiyuan1215.py:36-94` — Grasp.__init__() 全部硬编码参数
  - `grasp_zy_zhiyuan1215.py:260-350` — grasp_img2real_yolo 中的补偿参数
  - `grasp_zy_zhiyuan1215.py:380-430` — 图像裁剪和检测范围
  - `config.py:1-38` — 全局配置参数
  - `camera.py:37-42` — 相机内参
  - `ctu_conn.py:9-22` — 物品映射
  - `ctu_conn.py:99-138` — 抓取流程中的参数
  - `gripper_zhiyuan.py:10-14` — 夹爪初始化参数（Modbus地址、波特率等）

  **WHY Each Reference Matters**:
  - 参数配置是换设备后最关键的操作，每个参数的位置和含义必须精确
  - config.py 和 grasp_zy_zhiyuan1215.py 中有重复定义的参数（如Tcam2base），两处都要标注
  - gripper_zhiyuan.py 中的 Modbus 参数如果换了夹爪也需要改

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 参数行号准确性抽查
    Tool: Bash (grep + read)
    Preconditions: 第5章已写入
    Steps:
      1. 从文档中提取5个标注了行号的参数
      2. 用 Read 工具到对应文件的实际行号查看内容
      3. 验证行号指向的内容确实是文档中描述的参数
      4. 具体验证：
         - grasp_zy_zhiyuan1215.py 第46行 → 应包含 "192.168.127.101"
         - grasp_zy_zhiyuan1215.py 第66行 → 应包含 Tcam2base 矩阵
         - ctu_conn.py 第163行 → 应包含 "192.168.127.253"
         - config.py 第1行 → 应包含 place_last_pose
         - camera.py 第38行 → 应包含内参矩阵
    Expected Result: 5个参数的行号全部准确
    Failure Indicators: 行号偏移 > 2行，或内容不匹配
    Evidence: .sisyphus/evidence/task-4-lineno-check.txt

  Scenario: 参数分类完整性
    Tool: Bash (grep)
    Preconditions: 参数总览表已写入
    Steps:
      1. grep -c "必须修改\|可能需要\|一般不改" USAGE_GUIDE.md
      2. 验证三类标签都出现
      3. grep "Tcam2base\|init_pose\|192.168\|intr\|GoogsMapping" USAGE_GUIDE.md
      4. 验证5个核心参数全部在总览表中
    Expected Result: 参数分类完整，核心参数无遗漏
    Evidence: .sisyphus/evidence/task-4-param-completeness.txt
  ```

  **Commit**: NO（等全部章节完成后一起提交）

- [x] 5. 编写第6章：标定指引

  **What to do**:
  - 编写简化版的标定操作指引，包括：

  **6.1 相机内参标定**
  - 解释什么是相机内参（简单说：相机"看到"的画面和真实世界的对应关系）
  - 指出 camera.py 第38-42行的内参矩阵
  - 如何获取新的内参（推荐使用 RealSense SDK 自带工具，或 OpenCV 标定工具）
  - 标定后的数值应该填在哪里、怎么填
  - 注意事项：相机分辨率必须是 640×480

  **6.2 手眼标定（Tcam2base 变换矩阵）**
  - 解释什么是手眼标定（简单说：告诉系统相机拍到的位置对应机械臂的什么位置）
  - 指出参数在 grasp_zy_zhiyuan1215.py 第66-71行 和 config.py 第4-8行
  - 推荐的标定方法（如使用机械臂 SDK 的标定工具，或 OpenCV hand-eye calibration）
  - 简化的标定步骤：
    1. 在机械臂工作区域放置一个标记物
    2. 控制机械臂末端到标记物位置，记录关节角
    3. 用相机拍摄标记物，记录像素位置
    4. 使用标定工具计算变换矩阵
  - 标定结果填写格式说明（4×4矩阵的每个元素对应哪行哪列）
  - 验证标定是否准确的方法（手动抓取测试）

  **6.3 不需要重新标定的情况**
  - 相机位置没有变化
  - 机械臂基座位置没有变化
  - 只有物品类别或速度需要调整

  **Must NOT do**:
  - 不要写复杂的数学公式或理论推导
  - 不要深入讲解四元数、旋转矩阵等概念
  - 保持步骤化、可操作

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 需要将复杂概念简化为可操作步骤
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: F1
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `grasp_zy_zhiyuan1215.py:66-72` — Tcam2base 矩阵和 Rbase2cam 的使用方式
  - `camera.py:37-42` — 相机内参矩阵
  - `grasp_zy_zhiyuan1215.py:286-294` — 坐标变换的实际使用，帮助理解变换矩阵含义

  **External References**:
  - RealSense SDK 自带标定工具（rs-enumerate-devices）
  - OpenCV hand-eye calibration（如果企业有标定工具）

  **WHY Each Reference Matters**:
  - Tcam2base 是手眼标定的核心输出，文档需要让读者理解这个矩阵填在哪里
  - 相机内参决定了深度图的像素-米转换精度
  - 坐标变换代码展示了矩阵如何被使用，帮助写出正确的标定指引

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 标定指引可操作性检查
    Tool: Bash (grep)
    Preconditions: 第6章已写入
    Steps:
      1. grep "内参" USAGE_GUIDE.md，验证相机内参标定步骤存在
      2. grep "手眼\|Tcam2base\|变换矩阵" USAGE_GUIDE.md，验证手眼标定步骤存在
      3. grep "验证" USAGE_GUIDE.md，验证标定验证方法存在
      4. 检查不包含复杂的数学公式（如旋转矩阵推导）
    Expected Result: 标定步骤以操作步骤形式呈现，无复杂公式
    Evidence: .sisyphus/evidence/task-5-calibration-check.txt
  ```

  **Commit**: NO（等全部章节完成后一起提交）

- [x] 6. 编写第7章：系统启停与 CTU 通信流程

  **What to do**:
  - 编写系统日常启停操作步骤：

  **7.1 首次启动（完整流程）**
  - Step 1: 确认硬件连接（网线、电源、相机USB）
  - Step 2: 机械臂上电（松开急停、后面开关向左打）
  - Step 3: SSH 登录开发板（ssh jet@192.168.127.102，密码为空格）
  - Step 4: 手动启动程序测试（cd 项目目录 → conda activate zy_torch → python ctu_conn.py）
  - Step 5: 观察日志输出确认正常（看到"连接成功"和机械臂回到初始位置）
  - Step 6: 配置 systemd 服务实现开机自启（参考 doc/RunGraspd.service_manual.md）

  **7.2 日常启停**
  - 启动：sudo systemctl start RunGraspd.service
  - 停止：sudo systemctl stop RunGraspd.service
  - 重启：sudo systemctl restart RunGraspd.service
  - 查看状态：sudo systemctl status RunGraspd.service
  - 查看日志：sudo journalctl -u RunGraspd.service -f

  **7.3 CTU 通信流程（通俗解释）**
  - 用通俗的"对话"方式解释 CTU 和机械臂系统的交互：
    1. 系统启动后，程序自动连接 CTU，并发送"我准备好了"（GRASP_OK）
    2. 系统每 10 秒发送一次心跳，告诉 CTU"我还在线"
    3. CTU 到达指定位置后，发送"开始分拣第X类物品"（CTU_GRASP_START）
    4. 程序先拍照识别，告诉 CTU"我看到N个目标物品"（GRASP_COUNT）
    5. 程序开始逐个抓取，每次告诉 CTU"我开始抓了"（GRASP_START）
    6. 全部抓完后，告诉 CTU"抓完了，我回去了"（GRASP_OVER）
    7. CTU 还可以随时调整速度（CTU_GRASP_SPEED）
    8. CTU 可以紧急停止（CTU_GRASP_STOP）和解除急停（CTU_GRASP_RELEASE）
  - 用表格列出物品代码与名称的对应关系（引用 ctu_conn.py 的 GoogsMapping）

  **7.4 手动单步抓取测试**
  - 参考 CTUArm.md 第6.1节的单步测试命令
  - wifi Rb_Pick [物品类型] [速度]
  - 解释测试结果含义

  **Must NOT do**:
  - 不要列出完整的协议帧格式（那是 ctu_protocol.md 的事）
  - 不要解释 CRC16、字节序等技术细节

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 操作步骤文档写作
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: F1
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `ctu_conn.py:38-53` — 连接和重连逻辑
  - `ctu_conn.py:55-66` — 心跳机制
  - `ctu_conn.py:84-96` — 命令处理逻辑
  - `ctu_conn.py:99-138` — 抓取流程（核心业务逻辑）
  - `ctu_conn.py:157-174` — __main__ 启动流程
  - `CTUArm.md:63-88` — SSH 登录和服务重启步骤
  - `CTUArm.md:106-132` — 单步抓取测试和故障恢复
  - `doc/RunGraspd.service_manual.md:192-234` — 常用服务命令

  **WHY Each Reference Matters**:
  - ctu_conn.py 的 go_grasp 函数是核心业务流程，文档需要用通俗语言复述这个流程
  - CTUArm.md 已有部分操作步骤，新文档需要整合并补充说明
  - 服务手册有完整的服务管理命令，直接引用即可

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 启停命令准确性
    Tool: Bash (grep)
    Preconditions: 第7章已写入
    Steps:
      1. grep "systemctl" USAGE_GUIDE.md，验证包含 systemctl 命令
      2. grep "RunGraspd" USAGE_GUIDE.md，验证服务名正确
      3. grep "GRASP_OK\|心跳\|CTU_GRASP_START\|GRASP_COUNT\|GRASP_START\|GRASP_OVER" USAGE_GUIDE.md
      4. 验证通信流程中的6个步骤都被提及
    Expected Result: 启停命令准确，通信流程完整
    Evidence: .sisyphus/evidence/task-6-startstop-check.txt
  ```

  **Commit**: NO（等全部章节完成后一起提交）

- [x] 7. 编写第8章：物品类别扩展

  **What to do**:
  - 编写如何添加新的物品类别：

  **8.1 当前支持的物品类别**
  - 表格：物品代码 → 物品名称（来自 ctu_conn.py GoogsMapping）
  - 当前支持：1=白萝卜, 2=空气开关, 3=接线端子, 4=限位开关, 5=电压采集模块

  **8.2 如何添加新物品类别**
  - Step 1: 在 ctu_conn.py 第9-22行的 GoogsMapping 字典中添加新映射
  - Step 2: 在检测模型中训练新类别（概述流程，不展开训练细节）
  - Step 3: 更新检测模型配置文件（models/mmdetection/configs/myconfig_zy.py 中的类别列表）
  - Step 4: 替换模型权重文件
  - Step 5: 修改模型路径（grasp_zy_zhiyuan1215.py 第86-87行）
  - Step 6: 测试验证

  **8.3 如何修改物品代码映射**
  - GoogsMapping 格式说明（"数字字符串": "英文标签名"）
  - 英文标签名必须与检测模型训练时的类别名一致
  - 修改后需要重启程序

  **Must NOT do**:
  - 不要深入讲解模型训练流程（那需要专门的开发文档）
  - 只需要告诉操作人员"新物品需要开发人员训练模型并提供权重文件"

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 操作说明写作
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: F1
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `ctu_conn.py:9-22` — GoogsMapping 字典
  - `grasp_zy_zhiyuan1215.py:86-87` — 检测模型配置和权重路径

  **WHY Each Reference Matters**:
  - GoogsMapping 是 CTU 指令到检测类别的桥梁，企业最常修改
  - 模型路径告诉操作人员新权重文件应该替换到哪里

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 物品映射一致性
    Tool: Bash (grep)
    Preconditions: 第8章已写入
    Steps:
      1. grep "白萝卜\|空气开关\|接线端子\|限位开关\|电压采集" USAGE_GUIDE.md
      2. 验证5个物品类别全部列出
      3. 对比文档中的物品代码与 ctu_conn.py GoogsMapping 是否一致
    Expected Result: 物品类别和代码与代码完全一致
    Evidence: .sisyphus/evidence/task-7-goods-check.txt
  ```

  **Commit**: NO（等全部章节完成后一起提交）

- [x] 8. 编写第9章：故障排查 + 附录

  **What to do**:
  - 编写故障排查章节：

  **9.1 常见问题排查表**
  - 表格格式：问题现象 | 可能原因 | 排查步骤 | 解决方法
  - 覆盖以下场景（基于代码中的错误处理逻辑）：
    - 程序启动失败（找不到模型文件、Conda环境不对）
    - 机械臂连接不上（IP不通、SDK未安装）
    - 相机打开失败（未连接、驱动问题）
    - CTU 连接不上（网络不通、端口不对）
    - 抓取失败-逆解失败（物品位置超出机械臂工作范围）
    - 抓取失败-碰撞检测（碰撞后自动恢复流程）
    - 物品检测不到（光照问题、物品不在相机视野中）
    - 夹爪不动作（Modbus配置问题）
    - systemd 服务异常退出

  **9.2 碰撞恢复操作**
  - 程序自动恢复流程的通俗说明（代码中 _recover_from_collision 已实现）
  - 如果自动恢复失败，手动操作步骤（参考 CTUArm.md 第6.2节，按绿色按钮手动转回初始位）
  - 恢复后执行 sudo systemctl restart RunGraspd.service

  **9.3 日志查看指南**
  - 程序运行时的关键日志信息含义：
    - "连接成功" → CTU 连接正常
    - "开始抓取物品流程" → 收到 CTU 指令
    - "待抓取物品数量: N" → 检测到N个物品
    - "机械臂逆解失败" → 物品位置不可达
    - "碰撞检测到" → 触发碰撞保护
    - "抓取成功完成" → 单次抓取成功
    - "机械臂完成料箱清空" → 全部抓完

  **附录A：配置文件快速对照表**
  - 汇总所有需要修改的文件和关键参数的速查表

  **附录B：设备IP地址速查表**
  - 所有设备的默认IP和端口

  **附录C：相关文档索引**
  - CTUArm.md — CTU小车操作手册
  - ctu_protocol.md — 通信协议详细说明
  - doc/RunGraspd.service_manual.md — 系统服务配置手册
  - README.md — 技术开发文档

  **Must NOT do**:
  - 不要遗漏碰撞恢复步骤——这是现场最常遇到的问题
  - 不要假设操作人员会看代码中的错误信息

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 故障排查表需要基于代码中的实际错误处理来编写
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: F1
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `grasp_zy_zhiyuan1215.py:105-144` — _movej_safe 和 _recover_from_collision 碰撞恢复逻辑
  - `grasp_zy_zhiyuan1215.py:518-622` — execute_grasp 中的错误处理和重试逻辑
  - `ctu_conn.py:38-53` — 连接重试机制
  - `CTUArm.md:104-134` — 故障恢复操作步骤
  - `doc/RunGraspd.service_manual.md:236-270` — 服务常见问题

  **WHY Each Reference Matters**:
  - 碰撞恢复是最常见的现场问题，代码中已有自动恢复，但文档需要说明手动恢复的备选方案
  - 错误处理代码中的 print 语句就是操作人员会看到的日志，需要翻译成通俗说明
  - CTUArm.md 已有部分故障排查步骤，需要整合到新文档中

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 故障排查覆盖度检查
    Tool: Bash (grep)
    Preconditions: 第9章已写入
    Steps:
      1. grep -c "问题\|故障\|失败\|异常\|排查" USAGE_GUIDE.md
      2. 验证故障排查条目 >= 8 个
      3. grep "碰撞" USAGE_GUIDE.md，验证碰撞恢复步骤存在
      4. grep "日志" USAGE_GUIDE.md，验证日志说明存在
      5. grep "逆解" USAGE_GUIDE.md，验证逆解失败说明存在
    Expected Result: 常见故障场景全部覆盖
    Evidence: .sisyphus/evidence/task-8-troubleshooting-check.txt

  Scenario: 附录完整性
    Tool: Bash (grep)
    Preconditions: 附录已写入
    Steps:
      1. grep "附录" USAGE_GUIDE.md，验证附录章节存在
      2. grep "CTUArm\|ctu_protocol\|RunGraspd" USAGE_GUIDE.md，验证引用了相关文档
    Expected Result: 附录包含速查表和相关文档索引
    Evidence: .sisyphus/evidence/task-8-appendix-check.txt
  ```

  **Commit**: YES（全部章节完成后提交）
  - Message: `docs: 添加 RM65 视觉抓取系统使用说明书`
  - Files: `USAGE_GUIDE.md`
  - Pre-commit: 无（纯文档）

---

## Final Verification Wave

- [x] F1. **文档完整性验证** — `quick`
  读取生成的 `USAGE_GUIDE.md`，验证以下内容：
  1. 所有必要章节都存在（对照 Task 1 的结构规划）
  2. 文档中引用的每个文件路径在项目中真实存在
  3. 文档中引用的行号与实际代码行号一致（抽查 5 个关键参数）
  4. 文档中无未解释的英文技术术语
  5. 物品类别映射表与 `ctu_conn.py` 中 GoogsMapping 一致
  6. IP 地址、端口号与代码中实际配置一致
  Output: `章节 [N/N] | 路径引用 [N/N] | 行号引用 [N/N] | 术语 [CLEAN] | VERDICT: APPROVE/REJECT`

---

## Commit Strategy

- **单次提交**: `docs: 添加 RM65 视觉抓取系统使用说明书`
  - `USAGE_GUIDE.md`
  - Pre-commit: 无（纯文档）

---

## Success Criteria

### Verification Commands
```bash
# 文档文件存在
ls -la USAGE_GUIDE.md

# 包含所有关键章节
grep -c "## " USAGE_GUIDE.md  # Expected: >= 9

# 包含参数配置章节（核心）
grep "参数配置" USAGE_GUIDE.md  # Expected: match

# 包含故障排查
grep "故障排查" USAGE_GUIDE.md  # Expected: match
```

### Final Checklist
- [ ] 文档覆盖系统概述、硬件连接、网络配置、软件环境、参数配置、标定、启停、类别扩展、故障排查
- [ ] 每个配置参数标注了文件名和行号
- [ ] 全中文，无未解释的英文术语
- [ ] 包含简化的标定指引
- [ ] 不包含代码实现细节
