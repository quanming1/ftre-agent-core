# PRD-B1-Permission-State-Tracing

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | B1 |
| 名称 | Permission + State + Tracing |
| 状态 | 已验收 |
| 创建日期 | 2026-08-12 |
| 定稿日期 | 2026-08-12 |
| 验收日期 | 2026-08-12 |
| 关联文档 | docs/TODO.yaml 阶段 B1；AGENTS.md |

## 1. 背景与目标

- **背景**：核心引擎（A 组）已具备执行能力，需要补齐工具权限控制、Agent 状态管理与调用追踪三块横切能力，才能支撑受控运行与可观测性。
- **目标**：提供权限判定引擎（allow/deny/ask）、可序列化的 AgentState、线程安全工具与全链路 Tracer（含 SQLite 导出）。
- **非目标**：不做用户级会话管理（属 ftre 后端职责）；不做分布式追踪协议对接。

## 2. 需求范围

### 2.1 功能需求

- [x] FR1：permission 权限引擎，按规则（allow / deny / ask）判定工具/动作是否放行，ask 模式进入待确认状态
- [x] FR2：AgentState 状态管理，支持运行状态读取与持久化（序列化/反序列化）
- [x] FR3：threading 线程安全工具，提供跨线程的锁/安全访问原语
- [x] FR4：Tracer 调用追踪，记录完整调用链（Agent 运行 → 工具调用 → LLM 调用），支持导出到 SQLite

### 2.2 非功能需求

- 性能：权限判定为 O(规则数) 且不引入明显延迟；Tracer 记录为异步/低开销写入
- 安全：deny 规则优先于 allow（默认拒绝语义），规则可配置
- 可观测：追踪记录含完整上下文（时间、调用方、参数摘要、结果状态）

## 3. 技术方案

- 模块设计：
  - `src/ftre_agent_core/permission/`：权限引擎（规则匹配器 + allow/deny/ask 决策 + ask 回调）
  - `src/ftre_agent_core/state/`：AgentState（运行状态模型 + 序列化）
  - `src/ftre_agent_core/threading.py`：线程安全原语封装
  - `src/ftre_agent_core/tracing.py`：Tracer（span/事件记录 + SQLite 导出器）
- 关键数据结构：权限规则（action, target, decision）、状态快照、追踪 span（id/父 id/类型/时间）
- 依赖选型：sqlite3 标准库导出；threading 标准库

## 4. 接口定义

- `PermissionEngine.check(action, target) -> allow | deny | ask`；ask 结果可回填
- `AgentState.to_dict()` / `from_dict()` 持久化；状态读取线程安全
- `Tracer.start_span(...)` / `end_span(...)`；`Tracer.export_sqlite(path)`

## 5. 验收标准

- [x] AC1：配置 allow/deny/ask 混合规则后，权限判定结果与规则语义一致（deny 优先、ask 进入待确认）
- [x] AC2：AgentState 序列化后经反序列化 round-trip 与原始状态等价
- [x] AC3：Tracer 记录完整调用链（Agent → 工具 → LLM 的父子 span 关系正确），SQLite 导出后可查询
- [x] AC4：并发场景下 threading 工具保证状态访问无竞争
- [x] AC5：相关单元测试全部通过

## 6. 测试计划

- 单元测试覆盖：权限规则优先级与 ask 流程、AgentState round-trip、Tracer span 树与 SQLite 导出、并发读写
- 手动验证：完整跑一轮带权限拦截的 Agent 会话，追踪记录在 SQLite 中可回放

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-12 | 初始定稿 | — |
