# ReActAgent 权限引擎归属设计

> 日期：2026-08-03  
> 状态：设计草案，尚未实施

## 背景

当前 Core 的 `ReActAgent` 同时接收两类权限数据：

```python
ReActAgent(
    ...,
    state=state,
    permission_engine=PermissionEngine(),
)
```

其中：

- 可持久化的 `PermissionRule` 和 `default_behavior` 保存在
  `AgentState.permission_context` 中；
- 是否真正执行权限检查，却取决于构造 `ReActAgent` 时是否传入
  `PermissionEngine` 实例。

FTRE 因此既要准备规则数据，又要了解 Core 的引擎类并实例化它。
这使应用层与 Core 内部算法产生了不必要的耦合。

## 当前设计的问题

### 1. 存在两个权限事实源

`AgentState.permission_context` 表示“权限规则是什么”，
`permission_engine is None` 表示“规则是否生效”。两者可以相互矛盾：

| State 状态 | Engine 状态 | 实际结果 |
|---|---|---|
| 存在 ASK/DENY 规则 | `None` | 规则被完全忽略 |
| 空规则 | 存在 | 走 `default_behavior` |
| 存在规则 | 存在 | 正常求值 |

一个持久化 State 本身无法完整表达 Agent 恢复后的权限行为。

### 2. 应用层依赖 Core 的算法实现

`PermissionEngine` 是 Core 的纯决策算法，不是 FTRE 的业务配置。
FTRE 只需提供规则、默认行为和恢复后的工具状态，不应负责组装引擎。

### 3. `PermissionEngine` 并不适合作为注入点

当前 `PermissionEngine` 是无状态的具体类：

```python
decision = engine.evaluate(request, rules, default_behavior)
```

它既不持有会话数据，也没有抽象接口或多个生产实现。
将这种具体、无状态算法作为公开构造参数，并没有获得有效的依赖倒置收益。

### 4. 恢复路径容易遗漏引擎

新建 Agent 和从 `state.json` 恢复 Agent 时，调用方都必须记得再传一次
`PermissionEngine()`。任意一条构建路径遗漏该参数，都会在无明显异常的情况下
绕过全部权限规则。

## AgentScope 对照

AgentScope 的 Agent 构造函数不接收 `PermissionEngine` 或独立的规则列表。
它的关键结构是：

```python
self.state = state or AgentState()
self._engine = PermissionEngine(self.state.permission_context)
```

`AgentState.permission_context` 是类型化、可序列化的持久化数据，内含模式、
工作目录和 allow/deny/ask 规则。Engine 始终由 Agent 内部创建。

可参考的本地代码：

- `E:\agentscope\src\agentscope\agent\_agent.py`
- `E:\agentscope\src\agentscope\state\_state.py`
- `E:\agentscope\src\agentscope\permission\_context.py`
- `E:\agentscope\src\agentscope\permission\_engine.py`

AgentScope 的核心边界是：

```text
应用层 / 存储层
        │
        │  AgentState(permission_context=规则数据)
        ▼
    ReActAgent
        │
        │  内部创建
        ▼
 PermissionEngine
        │
        ▼
 PermissionDecision
```

## 设计决策

### 决策 1：删除 `ReActAgent.permission_engine` 构造参数

对外 API 不再允许调用方传入 Engine 实例。

```python
# Before
ReActAgent(
    ...,
    state=state,
    permission_engine=PermissionEngine(),
)

# After
ReActAgent(
    ...,
    state=state,
)
```

`ReActAgent` 内部始终创建 Core 的 `PermissionEngine`：

```python
self._state = state or AgentState()
self._permission_engine = PermissionEngine()
```

### 决策 2：规则的唯一事实源是 `AgentState`

应用层传入的是 `PermissionRule` 数据，但这些数据必须统一放入
`AgentState.permission_context`，不再同时作为 `ReActAgent` 的第二个参数传入。

原因是新 Agent 和恢复 Agent 都必须以 State 为准。若同时接收
`state.permission_context` 与 `permission_rules` 参数，就必须再定义覆盖顺序，
重新引入两个事实源。

FTRE 的构造方式应为：

```python
state = AgentState(
    permission_context=PermissionContext(
        permission_rules=[
            PermissionRule(
                id="default-bash-ask",
                tool_name="bash",
                behavior=PermissionBehavior.ASK,
            ),
        ],
        default_behavior=PermissionBehavior.ALLOW,
    ),
)

agent = ReActAgent(
    ...,
    state=state,
)
```

### 决策 3：引入类型化 `PermissionContext`

当前 `AgentState.permission_context` 是 `dict[str, Any]`，其字段约定散落在注释和
`ActingExecutor._load_permission_config()` 中。建议改为 Pydantic 模型：

```python
class PermissionContext(BaseModel):
    permission_rules: list[PermissionRule] = Field(default_factory=list)
    default_behavior: PermissionBehavior = PermissionBehavior.ALLOW


class AgentState(BaseModel):
    context: list[Msg] = Field(default_factory=list)
    permission_context: PermissionContext = Field(
        default_factory=PermissionContext,
    )
```

保留 `permission_rules` 和 `default_behavior` 现有 JSON 字段名，旧的
`state.json` 对象可由 Pydantic 直接解析，不需要额外迁移字段。

### 决策 4：用 `default_behavior=ALLOW` 表达不启用拦截

不再用 `permission_engine=None` 表达“跳过权限”。
空规则加 `default_behavior=ALLOW` 即可完整、可持久化地表达同样的行为：

```json
{
  "permission_rules": [],
  "default_behavior": "allow"
}
```

该默认值也保持了目前“未传入 Engine 时工具直接执行”的兼容行为。
若未来要提供更安全的 Core 默认值，应单独作为破坏性行为变更讨论，
不与本次依赖边界重构混在一起。

### 决策 5：`PermissionEngine` 保持纯算法

Engine 仍不读写 `AgentState`，不负责持久化，不保存会话级规则。
`ActingExecutor` 从 `AgentState.permission_context` 获取类型化数据，再显式交给
Engine 求值：

```python
ctx = self.agent.state.permission_context
decision = self.permission_engine.evaluate(
    request,
    ctx.permission_rules,
    ctx.default_behavior,
)
```

这样保持了 Core 算法的无状态和可单测性，同时不再将算法实例暴露给 FTRE。

## 数据所有权

| 数据/能力 | 所有者 | 是否持久化 |
|---|---|---|
| `PermissionRule` | `AgentState.permission_context` | 是 |
| `default_behavior` | `AgentState.permission_context` | 是 |
| ToolCall 的 ASKING/ALLOWED/FINISHED 状态 | `AgentState.context` | 是 |
| 规则匹配和冲突解决算法 | `PermissionEngine` | 否，纯代码 |
| 暂停、确认和恢复编排 | `ReActRunner` | 状态投影到 AgentState |
| 默认业务规则的选择 | FTRE | 通过 AgentState 传入 |

## 持久化与恢复

新流程：

```text
FTRE 创建 PermissionRule
  → 写入 AgentState.permission_context
  → ReActAgent 内部创建 PermissionEngine
  → Engine 对 ToolCall 求值
  → ASKING/ALLOWED/FINISHED 写入 AgentState.context
  → FTRE 持久化 AgentState 投影
```

恢复流程：

```text
FTRE 读取 state.json
  → 重建 AgentState（规则 + ToolCall 状态）
  → 构造 ReActAgent(state=state)
  → Agent 内部自动获得 PermissionEngine
  → UserConfirmResultEvent 继续原工具调用
```

调用方恢复 Agent 时不再需要“记得重新注入 Engine”。

## 拟议修改范围

### Core

| 文件 | 修改 |
|---|---|
| `permission/_context.py` | 新增类型化 `PermissionContext` |
| `permission/__init__.py` | 导出 `PermissionContext` |
| `state/_agent_state.py` | `permission_context` 由 dict 改为类型化模型 |
| `agent/react.py` | 删除 `permission_engine` 参数，内部创建 Engine |
| `agent/runner/react_runner.py` | 仅使用 Agent 内部 Engine |
| `agent/runner/_execute_acting.py` | 直接读取类型化 PermissionContext，删除 dict 解析 |
| `tests/test_permission*.py` | 用 AgentState 注入规则，不再注入 Engine |

### FTRE

| 文件 | 修改 |
|---|---|
| `src/ftre/agent/agent_manager.py` | 删除 `PermissionEngine` import 和 `permission_engine=...` |
| `src/ftre/agent/agent_manager.py` | `_default_agent_state()` 直接构造类型化 PermissionContext |
| `src/ftre/agent/turn_executor.py` | 验证新建/恢复路径都只传 State |

## 兼容策略

1. `PermissionContext` 保留现有 JSON 字段名，允许 Pydantic 从 dict 直接校验。
2. Core 的空权限配置默认为 `ALLOW`，保持原未注入 Engine 时的行为。
3. 恢复时以持久化 State 为唯一事实源，不从构造参数覆盖规则。
4. `PermissionEngine.evaluate()` 的纯函数语义保持不变。
5. `RequireUserConfirmEvent` / `UserConfirmResultEvent` 协议和 ToolCall 状态机保持不变。

## 不采用的方案

### 继续注入 `PermissionEngine`

拒绝原因：暴露 Core 内部算法，无实际多实现需求，且与持久化规则形成双事实源。

### `ReActAgent(state=..., permission_rules=...)`

拒绝作为主 API。虽然直接传规则比传 Engine 更合理，但它与
`state.permission_context` 重复。恢复时若两者不同，必须额外规定优先级。

如未来需要便捷 API，可提供明确的 State 工厂函数，而不是给 Agent 增加第二份规则：

```python
state = AgentState.with_permissions(
    rules=[...],
    default_behavior=PermissionBehavior.ALLOW,
)
```

### Engine 直接持有可变 `AgentState`

拒绝原因：会让纯权限算法依赖 Agent 持久化模型，降低独立测试性，
也不利于未来在非 ReAct 场景复用 Engine。

## 验收标准

1. `ReActAgent.__init__` 不再暴露 `permission_engine` 参数。
2. FTRE 不再 import 或实例化 `PermissionEngine`。
3. 权限规则和默认行为只存在于 `AgentState.permission_context`。
4. 新 Agent 在空规则下保持目前的默认放行行为。
5. ASK 工具能够暂停，用户同意后能够恢复执行。
6. DENY 工具不执行，并产生配对的 DENIED ToolResultBlock。
7. 多工具部分确认后重建 Agent，未确认的 ASKING ToolCall 仍可继续展示和确认。
8. 旧 `permission_context` JSON 可直接恢复为类型化模型。
9. Core 权限单测、暂停/恢复测试和 FTRE 后端集成测试全部通过。

## 非目标

本设计不改变以下内容：

- `PermissionRule` 的正则匹配语义；
- ASK/DENY/ALLOW 的冲突优先级；
- 批量工具的整批暂停语义；
- `RequireUserConfirmEvent` 和 `UserConfirmResultEvent` 字段；
- FTRE WebSocket 协议和客户端确认 UI；
- 默认 Bash 规则的具体内容。

