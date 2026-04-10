# TTFT 性能诊断

> LLM 调用前的同步开销分析与诊断方法

## 问题现象

- TTFT（Time To First Token）显著高于直接调用 API
- 同一 API key 在其他 agent 系统表现正常

## 诊断测试

用于对比框架开销 vs 裸 HTTP 的两个测试：

| 测试文件 | 类型 | 覆盖范围 |
|----------|------|----------|
| `src/tests/test_deepminer_curl.py` | 裸 HTTP | 绕过 LiteLLM 和框架，直接 httpx 流式请求 |
| `src/tests/test_llm_only.py` | 纯 LLM 调用 | 对比 `litellm` vs `httpx` 直连，无 ReAct 逻辑 |
| `src/tests/test_deepminer_speed.py` | 完整框架 | 通过 ReActAgent 走完整链路 |

**用法：**
```bash
# 测试网关本身速度
python src/tests/test_deepminer_curl.py

# 对比 litellm vs httpx 性能
python src/tests/test_llm_only.py

# 测试框架开销
pytest src/tests/test_deepminer_speed.py::TestFirstTokenLatency -v
```

## 诊断决策树

```
运行 curl 测试 ───┬── TTFT 慢（>3s）→ 问题在网关/模型，与框架无关
                  └── TTFT 快（<2s）→ 运行框架测试
                                        ↓
                            框架 TTFT 慢 → 问题在框架内部开销
                            框架 TTFT 快 → 可能是偶发或环境问题
```

## 核心文件

| 文件 | 职责 |
|------|------|
| `src/ftre_agent_core/agent/runner/handler/llm/utils.py` | `dump_llm_input` — 写调试日志（已异步化） |
| `src/ftre_agent_core/agent/runner/handler/llm/completion.py` | completions 适配器 |
| `src/ftre_agent_core/agent/runner/handler/llm/responses.py` | responses 适配器 |
| `src/ftre_agent_core/threading.py` | `thread_pool.io` — 16 worker 的 IO 线程池 |
| `src/ftre_agent_core/memory/token.py` | `TokenUsage` — 纯内存累加器 |
| `src/ftre_agent_core/__init__.py` | LiteLLM 日志级别配置入口 |

## 瓶颈分析

### 已确认非瓶颈

| 组件 | 原因 |
|------|------|
| **LiteLLM** | `litellm.completion()` 是薄的 HTTP 封装，无 retry、无 prompt 预处理、无 token 计数 |
| **TokenUsage** | 纯内存累加器，无外部 API 调用（`src/ftre_agent_core/memory/token.py`） |
| **MCP** | 延迟连接（lazy connect），不开启 `connect()` 时无开销 |

### 实际瓶颈（按影响排序）

1. **LiteLLM DEBUG 日志**（影响最大）
   - 现象：TTFT 2-5秒 → 毫秒级
   - 原因：DEBUG 日志输出大量 SQL 日志，积累后导致性能下降
   - 方案：在 `src/ftre_agent_core/__init__.py` 设置 `LITELLM_LOG=WARNING`

2. **`dump_llm_input` 同步写文件**（已修复）
   - 原实现：同步 `mkdir` + `json.dumps` + `write_text`
   - 方案：`thread_pool.io.submit()` 异步化

3. **`to_openai_tools()` 无缓存**
   - 每次迭代重建工具 schema（~1-5ms）
   - 工具列表通常不变，可在 `ToolRegistry` 层面缓存

## 性能对比结论

| 方式 | TTFT | 稳定性 | 备注 |
|------|------|--------|------|
| **httpx 直连** | 快 5-10 秒 | 低 | 频繁出现 503/SSL EOF 错误 |
| **litellm** | 慢 5-10 秒 | 高 | 内置重试机制，推荐生产使用 |

**结论**：虽然 litellm 比直连慢，但内置的重试和错误处理在生产环境更可靠。

## 调用链路

```
react_runner.py:_step()
  → memory.get_messages()
  → agent.tools.to_openai_tools()      ← 可缓存优化
  → completion.py:stream_completion()
    → utils.py:dump_llm_input()         ← 已异步化
    → litellm.completion()              ← LiteLLM DEBUG 日志配置
```

## 配置优化

### LiteLLM 环境变量（性能换时间）

在 `src/ftre_agent_core/__init__.py` 中配置：

```python
import os

# 关闭 DEBUG 日志（最大性能提升）
os.environ.setdefault("LITELLM_LOG", "WARNING")

# 其他性能优化配置
os.environ.setdefault("LITELLM_DROP_PARAMS", "True")       # 跳过参数验证
os.environ.setdefault("LITELLM_MAX_RETRIES", "1")          # 减少重试次数
os.environ.setdefault("LITELLM_REQUEST_TIMEOUT", "60")     # 请求超时
```

其他可选性能配置：

| 环境变量 | 作用 | 建议 |
|----------|------|------|
| `LITELLM_DROP_PARAMS` | 跳过参数验证 | 开启（除非需要严格校验） |
| `LITELLM_MAX_RETRIES` | 最大重试次数 | 设为 0 或 1（用应用层重试替代） |
| `LITELLM_CACHE` | 启用响应缓存 | 非流式场景可开启 |
| `LITELLM_REQUEST_TIMEOUT` | 请求超时 | 根据网关 SLA 设置合理值 |
| `LITELLM_DISABLE_VERIFY_FILE` | 禁用文件类型验证 | 确认输入安全后可开启 |

## 设计决策

- **LiteLLM DEBUG 日志是大头**：配置不当会导致 TTFT 从毫秒级恶化到秒级
- **LiteLLM 本身不背锅**：TTFT 慢的原因通常是日志配置或框架内部同步操作
- **对比测试是诊断关键**：curl 测试 vs 框架测试的 TTFT 差值即为框架开销
- **异步化优先**：I/O 操作（写日志、网络请求）应走 `thread_pool.io`
- **litellm vs 直连权衡**：litellm 慢但更稳定，生产环境推荐 litellm

## 注意事项

- LiteLLM DEBUG 日志默认可能开启，需在 `src/ftre_agent_core/__init__.py` 显式设置 `LITELLM_LOG=WARNING`
- `thread_pool.io` 有 16 个 worker，适合轻量 I/O 任务
- `dump_llm_input` 是 fire-and-forget，失败静默不影响主流程
- 诊断时应先跑 curl 测试确认网关/模型本身不慢
- TokenUsage 纯内存，恢复 checkpoint 时也不会触发外部调用
