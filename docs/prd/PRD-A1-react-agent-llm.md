# PRD-A1-ReActAgent-LLM-Handler

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | A1 |
| 名称 | ReActAgent + LLM Handler |
| 状态 | 已验收 |
| 创建日期 | 2026-08-12 |
| 定稿日期 | 2026-08-12 |
| 验收日期 | 2026-08-12 |
| 关联文档 | docs/TODO.yaml 阶段 A1；AGENTS.md |

## 1. 背景与目标

- **背景**：ftre-agent-core 需要 ReAct（Reasoning + Acting）循环驱动的 Agent 执行引擎，以及统一的 LLM 调用层，为后续 Tool、Hook 等阶段提供执行骨架。
- **目标**：提供可独立运行的 ReActAgent（思考→工具→观察→再思考）与 LLMHandler 调用封装，支持流式输出与取消。
- **非目标**：不实现具体工具（属 A2 阶段）；不做多 Agent 编排（属后续阶段）。

## 2. 需求范围

### 2.1 功能需求

- [x] FR1：ReActAgent 实现 ReAct 循环（思考 → 工具调用 → 观察结果 → 再思考），直至生成最终回答或达到终止条件
- [x] FR2：LLMHandler 封装 LLM API 调用（模型/参数/消息组装），支持同步与异步两种调用方式
- [x] FR3：支持流式输出（stream），流式 token 通过事件/回调逐条透出且不丢失
- [x] FR4：支持 `cancel_nowait` 取消机制，取消在 `await` 挂起点立即生效，清理在途任务
- [x] FR5：支持 LLM 工具调用（function/tool calling）响应解析，产出结构化工具调用参数

### 2.2 非功能需求

- 性能：单轮 ReAct 循环开销低，无冗余序列化
- 兼容性：LLM 层与具体模型供应商解耦（可通过适配层接入不同后端）
- 健壮性：LLM 调用失败可重试/可降级，取消后无悬挂任务

## 3. 技术方案

- 模块设计：
  - `src/ftre_agent_core/agent/react.py`：ReActAgent 主循环（思考/工具/观察状态机、终止判定、回合管理）
  - `src/ftre_agent_core/llm/completion.py`：LLMHandler（模型配置、补全/流式补全、工具调用解析）
  - `src/ftre_agent_core/llm/utils.py`：LLM 层公共工具（消息转换、参数归一、错误分类）
- 关键数据结构：ReAct 回合上下文、LLM 调用参数、工具调用解析结果
- 依赖选型：基于 asyncio 异步模型；LLM 后端经统一接口注入

## 4. 接口定义

- `ReActAgent.run(messages) -> 最终响应`；内部维护思考/工具调用/观察循环
- `LLMHandler.complete(messages, tools=None)` / `complete_stream(...)` 异步流式变体
- `cancel_nowait()`：非阻塞发起取消，await 点收到 CancelledError 后清理资源

## 5. 验收标准

- [x] AC1：构造多步 ReAct 场景（思考→工具→观察→再思考），循环按预期顺序正确执行并产出最终回答
- [x] AC2：流式输出场景下逐条接收 token，断言无丢失、顺序正确
- [x] AC3：调用 `cancel_nowait` 后，进行中的 await 调用在最近挂起点立即抛出取消并清理，无悬挂任务
- [x] AC4：LLM 返回工具调用时，解析结果与原始调用声明一致，可交给 Tool 层执行
- [x] AC5：相关单元测试全部通过

## 6. 测试计划

- 单元测试覆盖：ReAct 循环分支（工具结果分支/终止分支）、流式 token 完整性、取消时序、工具调用解析边界
- 手动验证：真实 LLM 后端跑通一轮带工具调用的 ReAct 对话

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-12 | 初始定稿 | — |
