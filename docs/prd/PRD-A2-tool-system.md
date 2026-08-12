# PRD-A2-Tool-System

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | A2 |
| 名称 | Tool 体系 |
| 状态 | 已验收 |
| 创建日期 | 2026-08-12 |
| 定稿日期 | 2026-08-12 |
| 验收日期 | 2026-08-12 |
| 关联文档 | docs/TODO.yaml 阶段 A2；AGENTS.md |

## 1. 背景与目标

- **背景**：ReAct 循环（A1）需要可注册、可查找、可执行的工具层来承载实际动作，Agent 才能与外接能力交互。
- **目标**：提供 Tool 基类、参数定义、依赖注入与注册执行机制，作为 Agent 工具调用的统一入口。
- **非目标**：不实现具体业务工具；不做工具权限判定（属 B1 阶段）。

## 2. 需求范围

### 2.1 功能需求

- [x] FR1：Tool 基类 + ToolParameter 参数定义（名称/描述/类型/必填），支撑自动生成工具 Schema
- [x] FR2：Injected 依赖注入标记，运行期由框架注入（如上下文/会话）而非来自 LLM 参数
- [x] FR3：ToolRegistry 支持注册（按名称去重）、查找（按名称/条件）、执行（run_one）工具
- [x] FR4：cancellation 取消机制，长时工具在执行中被取消时可响应并释放资源

### 2.2 非功能需求

- 性能：注册/查找为常量级或近常量级开销
- 健壮性：重复注册、未知工具名等异常路径有明确错误语义
- 可扩展：新工具只需继承 Tool 并注册即可接入，不改动框架代码

## 3. 技术方案

- 模块设计：
  - `src/ftre_agent_core/tool/base.py`：`Tool` 基类、`ToolParameter` 参数描述、`Injected` 注入标记
  - `src/ftre_agent_core/tool/registry.py`：`ToolRegistry`（注册表 + 查找 + `run_one` 执行入口）
  - `src/ftre_agent_core/tool/cancellation.py`：工具执行取消的协作机制
- 关键数据结构：工具名称 → 工具实例映射；参数 Schema 描述结构
- 依赖选型：无外部强依赖，纯标准库 + 类型标注

## 4. 接口定义

- `ToolRegistry.register(tool)` / `ToolRegistry.get(name)` / `ToolRegistry.run_one(name, args)`
- `ToolParameter(name, description, type, required)`；`Injected` 标记的字段不进入 LLM 可见参数
- 取消：执行中检查取消信号，收到后抛取消异常并清理

## 5. 验收标准

- [x] AC1：工具可注册并可按名称查找，未知名称查找返回明确错误
- [x] AC2：Injected 标记的参数在 `run_one` 时自动注入正确值，且不出现在 LLM 工具 Schema 中
- [x] AC3：`ToolRegistry.run_one` 传入合法名称与参数时执行结果正确，传入非法参数时按 ToolParameter 校验报错
- [x] AC4：取消场景下工具执行被中断且无资源泄漏
- [x] AC5：相关单元测试全部通过

## 6. 测试计划

- 单元测试覆盖：注册/查找/重复注册、Injected 注入、参数校验边界、取消路径
- 手动验证：ReActAgent（A1）接入注册工具后跑通端到端工具调用

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-12 | 初始定稿 | — |
