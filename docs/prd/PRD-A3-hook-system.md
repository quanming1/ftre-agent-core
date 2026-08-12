# PRD-A3-Hook-System

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | A3 |
| 名称 | Hook 系统 |
| 状态 | 已验收 |
| 创建日期 | 2026-08-12 |
| 定稿日期 | 2026-08-12 |
| 验收日期 | 2026-08-12 |
| 关联文档 | docs/TODO.yaml 阶段 A3；AGENTS.md |

## 1. 背景与目标

- **背景**：Agent 执行（A1）与工具（A2）链路已就绪，需要在执行前后插入可扩展逻辑（消息改写、配置调整、观测等），且不侵入核心循环代码。
- **目标**：提供统一的 Hook 管理器，支持在 Agent 执行关键时机挂载异步钩子，形成可组合的 filter chain。
- **非目标**：不做插件热加载与动态安装；不做事件总线（消息与事件见 A4 阶段）。

## 2. 需求范围

### 2.1 功能需求

- [x] FR1：`FtreCoreHookManager` 提供 hook 注册与 `trigger`/`await` 触发能力
- [x] FR2：`before_messages_build` hook：在消息构建前触发，可修改/注入消息
- [x] FR3：`before_agent_run` hook：在 Agent 运行前触发，可修改运行配置（config）
- [x] FR4：异步 filter chain：多个 hook 按注册顺序串联，coroutine 返回值被正确 `await` 后传给下一个

### 2.2 非功能需求

- 性能：hook 数量较多时仍保持线性开销，无重复 await
- 健壮性：单个 hook 抛错不影响链上其余 hook（可隔离）
- 可组合：hook 与具体 Agent 解耦，同一 hook 可挂到多个时机

## 3. 技术方案

- 模块设计：`src/ftre_agent_core/hooks.py` 内实现 `FtreCoreHookManager`（注册表 + 触发链）
- 关键数据结构：时机名 → hook 列表；每个 hook 为可调用对象（支持 async def）
- 依赖选型：asyncio 原生协程；无外部依赖

## 4. 接口定义

- `FtreCoreHookManager.register(name, hook)` / `trigger(name, *args)`（异步执行链）
- 内置时机名：`before_messages_build`、`before_agent_run`
- 每个 hook 的返回值 await 后作为下一个 hook 的输入，链尾返回值返回调用方

## 5. 验收标准

- [x] AC1：在 `before_messages_build` 注册的 hook 在消息构建前正确触发，且能修改 messages（改动对下游可见）
- [x] AC2：在 `before_agent_run` 注册的 hook 能修改 config，改动对 Agent 运行生效
- [x] AC3：注册 async hook 时，coroutine 返回值被 await，后续 hook 收到的是 await 后的值而非协程对象
- [x] AC4：多个 hook 按注册顺序执行形成 filter chain，前序输出正确传入后续
- [x] AC5：相关单元测试全部通过

## 6. 测试计划

- 单元测试覆盖：注册/触发顺序、async hook 的 await 语义、messages/config 修改生效、hook 抛错隔离
- 手动验证：在 ReActAgent 上挂 before_messages_build 注入系统提示词，观察行为变化

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-12 | 初始定稿 | — |
