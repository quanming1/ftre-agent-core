# Re: Feature Request: ftre-agent-core 支持 LiteLLM Responses API

> **收件人：** ai-base 后端团队  
> **发件人：** ftre-agent-core 开发团队  
> **日期：** 2026-03-31  

---

## 状态：已完成 ✅

你们提出的 Responses API 支持需求已实现并合并到 main 分支。

## 改动内容

### 1. Agent 构造参数新增 `api_type`

```python
agent = ReActAgent(
    model="openai/gpt-5.4",
    api_key="sk-xxx",
    api_base="http://ai.xiamai.top/v1",
    api_type="responses",  # 新增参数，默认 "completions"
    system_prompt="...",
    tools=[...],
)
```

### 2. 支持的协议类型

| api_type | 底层调用 | 适用场景 |
|----------|----------|----------|
| `"completions"` | `litellm.completion()` | 默认，兼容所有厂商 |
| `"responses"` | `litellm.responses()` | xiamai、respyun 等 Responses 协议厂商 |

### 3. 架构重构

将协议适配逻辑拆分为独立的适配器模块：

```
handler/llm/
├── base.py            # StreamAdapter 抽象基类
├── completion.py      # CompletionAdapter
├── responses.py       # ResponsesAdapter
└── handler.py         # LLMHandler (facade)
```

未来新增协议只需实现 `StreamAdapter` 接口。

## 使用示例

```python
from ftre_agent_core.agent import ReActAgent

# xiamai 厂商配置
agent = ReActAgent(
    model="openai/gpt-5.4",
    api_key="sk-xxx",
    api_base="http://ai.xiamai.top/v1",
    api_type="responses",
    tools=[...],
)

# 正常使用，与 completions 模式行为一致
for event in agent.run("你好"):
    print(event)
```

## 相关提交

```
74d5c53 refactor: extract LLM adapters into separate module
84fbb36 feat: support api_type=responses for LiteLLM Responses API
537ebbb refactor: migrate from OpenAI SDK to LiteLLM
```

## 向下兼容

- `api_type` 默认值为 `"completions"`，现有代码无需修改
- 所有现有测试通过

## 待确认

请 ai-base 团队使用 xiamai/respyun 厂商进行实际测试，如有问题请反馈。特别关注：

1. 流式文本输出是否正常
2. Tool calling 是否正常
3. 多轮对话上下文是否保持

---

如有问题请联系 ftre-agent-core 团队。
