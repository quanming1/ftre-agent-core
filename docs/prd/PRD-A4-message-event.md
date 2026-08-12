# PRD-A4-Message-Event

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | A4 |
| 名称 | Message + Event |
| 状态 | 已验收 |
| 创建日期 | 2026-08-12 |
| 定稿日期 | 2026-08-12 |
| 验收日期 | 2026-08-12 |
| 关联文档 | docs/TODO.yaml 阶段 A4；AGENTS.md |

## 1. 背景与目标

- **背景**：Agent 循环（A1）、工具（A2）、Hook（A3）之间需要统一的 Message 与 Event 数据模型，才能稳定传递内容与结构化事件（供 Tracer 捕获、前端渲染、调试导出）。
- **目标**：提供完整可序列化的 Msg 数据模型、ContentBlock 内容块类型体系与事件基类。
- **非目标**：不做事件总线/分发器；不做消息持久化存储（状态见 B1 阶段）。

## 2. 需求范围

### 2.1 功能需求

- [x] FR1：`Msg` 数据模型（role / content / metadata），支持序列化/反序列化
- [x] FR2：`MsgName` 枚举，统一标识消息类型/来源
- [x] FR3：`ContentBlock` 类型体系：text / toolCall / toolResult / image / think 五类内容块
- [x] FR4：`EventBase` / `CustomEvent` 事件基类，事件可被 Tracer 捕获与导出

### 2.2 非功能需求

- 兼容性：Msg/ContentBlock 序列化格式稳定，可跨进程/跨版本交换
- 健壮性：未知/畸形内容块反序列化有明确错误或降级策略
- 可扩展：新增内容块类型无需改动既有类型

## 3. 技术方案

- 模块设计：
  - `src/ftre_agent_core/message/_msg.py`：`Msg`（role/content/metadata）与 `MsgName` 枚举
  - `src/ftre_agent_core/message/_block.py`：`ContentBlock` 及 text/toolCall/toolResult/image/think 子类型
  - `src/ftre_agent_core/event/_event.py`：`EventBase` 基类与 `CustomEvent`
- 关键数据结构：Msg 的 content 为 ContentBlock 列表；Event 携带类型、时间戳与载荷
- 依赖选型：Pydantic 或等效序列化模型；标准库 datetime

## 4. 接口定义

- `Msg(role, content: list[ContentBlock], metadata)`；`Msg.model_dump()` / 反序列化构造
- `ContentBlock` 子类型按 `type` 判别（text/toolCall/toolResult/image/think）
- `EventBase(timestamp, type, payload)`；`CustomEvent` 供业务自定义载荷

## 5. 验收标准

- [x] AC1：Msg 序列化后经反序列化 round-trip 与原始对象等价（role/content/metadata 完整）
- [x] AC2：ContentBlock 五类类型（text/toolCall/toolResult/image/think）均可构造、序列化并按 type 正确判别
- [x] AC3：EventBase/CustomEvent 生成后可被 Tracer（B1）捕获并写入追踪记录
- [x] AC4：畸形内容块反序列化抛出明确错误而非静默损坏数据
- [x] AC5：相关单元测试全部通过

## 6. 测试计划

- 单元测试覆盖：Msg round-trip、五类 ContentBlock 判别与序列化、Event 载荷、畸形输入错误路径
- 手动验证：ReActAgent 全程产生的消息以 Msg/Event 形态导出，结构与预期一致

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-12 | 初始定稿 | — |
