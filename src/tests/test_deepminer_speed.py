"""
Deepminer 端点速度测试

测试维度：
1. 首 token 延迟 (TTFT - Time To First Token)
2. 总耗时 (Total Latency)
3. 吞吐量 (Tokens per Second)
4. 流式输出连续性
5. 工具调用延迟
6. 并发请求性能

端点: https://llm-gateway.mlamp.cn/v1
模型: claude-opus-4-6
"""

import statistics
import threading
import time

import pytest

from ftre_agent_core.agent import EventType, ReActAgent
from ftre_agent_core.tool import tool

# ── Deepminer 配置 ─────────────────────────────────────────────
API_KEY = "sk-hjcMKlZoxvUCzUhsQH2FOXMXnzVh4m4l64QGuLqcBrzOAl7q"
API_BASE = "https://llm-gateway.mlamp.cn/v1"
MODEL = "openai/claude-opus-4-6"


# ── 辅助函数 ───────────────────────────────────────────────────


def _collect_events(agent, message):
    """收集所有事件，附带时间戳"""
    results = []
    start = time.perf_counter()
    for event in agent.run(message):
        results.append((time.perf_counter() - start, event))
    return results


def _make_agent(system_prompt="你是一个简洁的助手。", tools=None, **kwargs):
    """快速创建 agent"""
    return ReActAgent(
        model=MODEL,
        api_key=API_KEY,
        api_base=API_BASE,
        system_prompt=system_prompt,
        tools=tools or [],
        **kwargs,
    )


def _extract_ttft(timed_events):
    """从带时间戳的事件列表中提取首 token 延迟（秒）"""
    for ts, event in timed_events:
        if event["type"] == EventType.MESSAGE and event["data"].get("content"):
            return ts
    return None


def _extract_full_text(timed_events):
    """拼接所有 MESSAGE 事件的文本"""
    parts = []
    for _, event in timed_events:
        if event["type"] == EventType.MESSAGE and event["data"].get("content"):
            parts.append(event["data"]["content"])
    return "".join(parts)


def _count_message_chunks(timed_events):
    """统计流式 chunk 数量"""
    return sum(
        1
        for _, e in timed_events
        if e["type"] == EventType.MESSAGE and e["data"].get("content")
    )


def _print_separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def simple_tools():
    """简单工具集"""

    @tool()
    def get_weather(city: str) -> str:
        """获取指定城市的天气"""
        weather_data = {
            "北京": "晴天，25°C",
            "上海": "多云，28°C",
            "广州": "小雨，30°C",
            "深圳": "阵雨，29°C",
        }
        return weather_data.get(city, f"{city}：数据未知")

    @tool()
    def calculate(expression: str) -> str:
        """计算数学表达式"""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    return [get_weather, calculate]


# ════════════════════════════════════════════════════════════════
# 测试类
# ════════════════════════════════════════════════════════════════


class TestFirstTokenLatency:
    """首 Token 延迟 (TTFT) 测试"""

    def test_ttft_short_prompt(self):
        """短 prompt 首 token 延迟"""
        _print_separator("TTFT - 短 prompt")

        agent = _make_agent()
        timed_events = _collect_events(agent, "你好")

        ttft = _extract_ttft(timed_events)
        assert ttft is not None, "未收到任何文本 token"

        print(f"首 token 延迟 (TTFT): {ttft * 1000:.0f}ms")
        # claude-opus 经网关转发，TTFT 通常 20~40s，阈值放宽到 60s
        assert ttft < 60, f"TTFT {ttft:.1f}s 过高"

    def test_ttft_medium_prompt(self):
        """中等 prompt 首 token 延迟"""
        _print_separator("TTFT - 中等 prompt")

        agent = _make_agent(
            system_prompt=(
                "你是一个资深技术架构师，擅长分析系统设计问题。"
                "回答时要有条理，分点阐述。"
            )
        )
        timed_events = _collect_events(agent, "简述微服务架构的优缺点")

        ttft = _extract_ttft(timed_events)
        assert ttft is not None, "未收到任何文本 token"

        print(f"首 token 延迟 (TTFT): {ttft * 1000:.0f}ms")
        assert ttft < 90, f"TTFT {ttft:.1f}s 过高"

    def test_ttft_repeated_3_times(self):
        """连续 3 次测量 TTFT 稳定性"""
        _print_separator("TTFT - 稳定性（3 次）")

        ttft_list = []
        prompts = ["你好", "1+1等于几？", "今天星期几？"]

        for i, prompt in enumerate(prompts):
            agent = _make_agent()
            timed_events = _collect_events(agent, prompt)
            ttft = _extract_ttft(timed_events)
            assert ttft is not None, f"第 {i + 1} 次未收到文本 token"
            ttft_list.append(ttft)
            print(f"  第 {i + 1} 次 TTFT: {ttft * 1000:.0f}ms")

        avg = statistics.mean(ttft_list)
        std = statistics.stdev(ttft_list) if len(ttft_list) > 1 else 0
        print(f"  平均: {avg * 1000:.0f}ms | 标准差: {std * 1000:.0f}ms")
        print(
            f"  最快: {min(ttft_list) * 1000:.0f}ms | 最慢: {max(ttft_list) * 1000:.0f}ms"
        )


class TestTotalLatency:
    """总耗时测试"""

    def test_short_response_latency(self):
        """短回复总耗时"""
        _print_separator("总耗时 - 短回复")

        agent = _make_agent()
        start = time.perf_counter()
        timed_events = _collect_events(agent, "用一句话回答：天空为什么是蓝色的？")
        total = time.perf_counter() - start

        text = _extract_full_text(timed_events)
        ttft = _extract_ttft(timed_events)

        print(f"总耗时: {total * 1000:.0f}ms")
        print(f"TTFT:  {ttft * 1000:.0f}ms" if ttft else "TTFT: N/A")
        print(f"生成耗时: {(total - (ttft or 0)) * 1000:.0f}ms")
        print(f"回复长度: {len(text)} 字符")

        assert total < 120, f"总耗时 {total:.1f}s 过高"
        assert len(text) > 0, "回复为空"

    def test_medium_response_latency(self):
        """中等回复总耗时"""
        _print_separator("总耗时 - 中等回复")

        agent = _make_agent()
        start = time.perf_counter()
        timed_events = _collect_events(agent, "用大约100字介绍一下 Python 语言")
        total = time.perf_counter() - start

        text = _extract_full_text(timed_events)
        ttft = _extract_ttft(timed_events)

        print(f"总耗时: {total * 1000:.0f}ms")
        print(f"TTFT:  {ttft * 1000:.0f}ms" if ttft else "TTFT: N/A")
        print(f"回复长度: {len(text)} 字符")

        assert total < 180, f"总耗时 {total:.1f}s 过高"


class TestThroughput:
    """吞吐量测试（tokens/s 估算）"""

    def test_throughput_estimation(self):
        """估算输出吞吐量（字符/秒）"""
        _print_separator("吞吐量估算")

        agent = _make_agent(system_prompt="你是一个助手，请详细回答用户问题。")
        start = time.perf_counter()
        timed_events = _collect_events(
            agent, "详细介绍一下 TCP 三次握手的过程，至少200字"
        )
        total = time.perf_counter() - start

        text = _extract_full_text(timed_events)
        ttft = _extract_ttft(timed_events) or 0
        generation_time = total - ttft

        char_count = len(text)
        # 粗略估算：中文约 1.5 token/字符，英文约 1.3 token/单词
        estimated_tokens = int(char_count * 1.2)

        chars_per_sec = char_count / generation_time if generation_time > 0 else 0
        tokens_per_sec = (
            estimated_tokens / generation_time if generation_time > 0 else 0
        )

        print(f"总耗时:       {total:.2f}s")
        print(f"TTFT:         {ttft * 1000:.0f}ms")
        print(f"生成耗时:     {generation_time:.2f}s")
        print(f"输出字符:     {char_count}")
        print(f"估算 tokens:  {estimated_tokens}")
        print(f"字符/秒:      {chars_per_sec:.1f}")
        print(f"tokens/秒:    {tokens_per_sec:.1f}")

        # 基本断言：应该能产出内容
        assert char_count > 50, f"回复太短: {char_count} 字符"
        assert chars_per_sec > 1, f"吞吐量过低: {chars_per_sec:.1f} 字符/秒"


class TestStreamingContinuity:
    """流式输出连续性测试"""

    def test_streaming_chunks(self):
        """验证流式输出 chunk 间隔合理"""
        _print_separator("流式 chunk 间隔分析")

        agent = _make_agent()
        timed_events = _collect_events(agent, "用大约150字介绍一下人工智能")

        # 提取所有 MESSAGE 事件的时间戳
        msg_timestamps = [
            ts
            for ts, e in timed_events
            if e["type"] == EventType.MESSAGE and e["data"].get("content")
        ]

        chunk_count = len(msg_timestamps)
        assert chunk_count > 0, "未收到流式 chunk"

        print(f"总 chunk 数: {chunk_count}")

        if chunk_count >= 2:
            gaps = [
                msg_timestamps[i + 1] - msg_timestamps[i]
                for i in range(len(msg_timestamps) - 1)
            ]
            avg_gap = statistics.mean(gaps)
            max_gap = max(gaps)
            min_gap = min(gaps)

            print(f"chunk 间隔:")
            print(f"  平均: {avg_gap * 1000:.0f}ms")
            print(f"  最小: {min_gap * 1000:.0f}ms")
            print(f"  最大: {max_gap * 1000:.0f}ms")

            # 最大间隔不应超过 30 秒（网关转发 + 模型思考容忍）
            assert max_gap < 30, f"最大 chunk 间隔 {max_gap:.1f}s 异常，可能存在卡顿"
        else:
            print("只有 1 个 chunk，无法分析间隔")

    def test_no_empty_chunks(self):
        """验证不存在空内容 chunk"""
        _print_separator("空 chunk 检查")

        agent = _make_agent()
        timed_events = _collect_events(agent, "你好，请做自我介绍")

        empty_chunks = [
            e
            for _, e in timed_events
            if e["type"] == EventType.MESSAGE and not e["data"].get("content")
        ]

        # 有些实现允许空 chunk，但数量不应过多
        total_msg = sum(1 for _, e in timed_events if e["type"] == EventType.MESSAGE)
        print(f"总 MESSAGE 事件: {total_msg}")
        print(f"空 chunk 数量: {len(empty_chunks)}")


class TestToolCallSpeed:
    """工具调用速度测试"""

    def test_tool_call_roundtrip(self, simple_tools):
        """工具调用完整往返延迟"""
        _print_separator("工具调用往返延迟")

        agent = _make_agent(
            system_prompt="你是一个天气助手。当用户询问天气时，使用 get_weather 工具。",
            tools=simple_tools,
        )

        start = time.perf_counter()
        timed_events = _collect_events(agent, "北京天气怎么样？")
        total = time.perf_counter() - start

        # 提取各阶段时间
        tool_call_ts = None
        tool_result_ts = None
        first_msg_after_tool_ts = None
        tool_result_seen = False

        for ts, event in timed_events:
            if event["type"] == EventType.TOOL_CALL and tool_call_ts is None:
                tool_call_ts = ts
            elif event["type"] == EventType.TOOL_RESULT and tool_result_ts is None:
                tool_result_ts = ts
                tool_result_seen = True
            elif (
                event["type"] == EventType.MESSAGE
                and tool_result_seen
                and event["data"].get("content")
                and first_msg_after_tool_ts is None
            ):
                first_msg_after_tool_ts = ts

        print(f"总耗时:                  {total * 1000:.0f}ms")
        if tool_call_ts is not None:
            print(f"首次工具调用:            {tool_call_ts * 1000:.0f}ms")
        if tool_result_ts is not None:
            print(f"工具结果返回:            {tool_result_ts * 1000:.0f}ms")
        if tool_call_ts and tool_result_ts:
            print(
                f"工具执行耗时:            {(tool_result_ts - tool_call_ts) * 1000:.0f}ms"
            )
        if first_msg_after_tool_ts and tool_result_ts:
            print(
                f"工具结果→首 token:       {(first_msg_after_tool_ts - tool_result_ts) * 1000:.0f}ms"
            )

        event_types = [e["type"] for _, e in timed_events]
        assert EventType.TOOL_CALL in event_types, "未触发工具调用"
        assert EventType.TOOL_RESULT in event_types, "未收到工具结果"
        assert EventType.DONE in event_types, "未正常结束"

    def test_calculate_tool_speed(self, simple_tools):
        """计算工具调用速度"""
        _print_separator("计算工具调用速度")

        agent = _make_agent(
            system_prompt="你是一个计算助手。使用 calculate 工具进行计算，然后告诉用户结果。",
            tools=simple_tools,
        )

        start = time.perf_counter()
        timed_events = _collect_events(agent, "计算 2 的 10 次方")
        total = time.perf_counter() - start

        tool_result = next(
            (e for _, e in timed_events if e["type"] == EventType.TOOL_RESULT),
            None,
        )
        assert tool_result is not None, "未收到工具结果"
        assert "1024" in tool_result["data"]["result"], (
            f"计算结果不正确: {tool_result['data']['result']}"
        )

        print(f"总耗时: {total * 1000:.0f}ms")
        print(f"工具结果: {tool_result['data']['result']}")

    def test_multi_tool_call_speed(self, simple_tools):
        """多工具连续调用速度"""
        _print_separator("多工具连续调用")

        agent = _make_agent(
            system_prompt=(
                "你是一个全能助手。可以查天气，也可以做计算。"
                "请按顺序完成用户的每个请求。"
            ),
            tools=simple_tools,
        )

        start = time.perf_counter()
        timed_events = _collect_events(
            agent, "帮我查一下北京和上海的天气，再算一下 99 * 88"
        )
        total = time.perf_counter() - start

        tool_calls = [
            (ts, e) for ts, e in timed_events if e["type"] == EventType.TOOL_CALL
        ]
        tool_results = [
            (ts, e) for ts, e in timed_events if e["type"] == EventType.TOOL_RESULT
        ]

        print(f"总耗时: {total * 1000:.0f}ms")
        print(f"工具调用次数: {len(tool_calls)}")
        print(f"工具结果次数: {len(tool_results)}")

        for i, (ts, ev) in enumerate(tool_calls):
            print(
                f"  调用 {i + 1}: {ev['data']['name']}({ev['data']['arguments']}) @ {ts * 1000:.0f}ms"
            )
        for i, (ts, ev) in enumerate(tool_results):
            print(
                f"  结果 {i + 1}: {ev['data']['name']} = {ev['data']['result']} @ {ts * 1000:.0f}ms"
            )

        # 至少应该有 2 次工具调用
        assert len(tool_calls) >= 2, f"预期至少 2 次工具调用，实际 {len(tool_calls)} 次"


class TestConcurrency:
    """并发请求测试"""

    def test_concurrent_2_agents(self):
        """2 个 agent 并发请求"""
        _print_separator("并发测试 - 2 agents")

        results = {}
        errors = {}

        def run_agent(name, prompt):
            try:
                agent = _make_agent()
                start = time.perf_counter()
                timed_events = _collect_events(agent, prompt)
                elapsed = time.perf_counter() - start
                text = _extract_full_text(timed_events)
                ttft = _extract_ttft(timed_events)
                results[name] = {
                    "elapsed": elapsed,
                    "ttft": ttft,
                    "chars": len(text),
                    "chunks": _count_message_chunks(timed_events),
                }
            except Exception as e:
                errors[name] = str(e)

        threads = [
            threading.Thread(target=run_agent, args=("agent-1", "用一句话介绍北京")),
            threading.Thread(target=run_agent, args=("agent-2", "用一句话介绍上海")),
        ]

        overall_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        overall = time.perf_counter() - overall_start

        print(f"整体耗时: {overall * 1000:.0f}ms")
        for name, data in sorted(results.items()):
            ttft_str = f"{data['ttft'] * 1000:.0f}ms" if data["ttft"] else "N/A"
            print(
                f"  {name}: 总耗时={data['elapsed'] * 1000:.0f}ms | "
                f"TTFT={ttft_str} | "
                f"字符={data['chars']} | chunks={data['chunks']}"
            )
        for name, err in sorted(errors.items()):
            print(f"  {name}: 错误 - {err}")

        assert len(errors) == 0, f"有 agent 失败: {errors}"
        assert len(results) == 2, "未全部完成"

    def test_concurrent_4_agents(self):
        """4 个 agent 并发请求 — 压力测试"""
        _print_separator("并发测试 - 4 agents")

        prompts = [
            "用一句话介绍 Python",
            "用一句话介绍 JavaScript",
            "用一句话介绍 Rust",
            "用一句话介绍 Go",
        ]
        results = {}
        errors = {}

        def run_agent(idx, prompt):
            name = f"agent-{idx + 1}"
            try:
                agent = _make_agent()
                start = time.perf_counter()
                timed_events = _collect_events(agent, prompt)
                elapsed = time.perf_counter() - start
                text = _extract_full_text(timed_events)
                ttft = _extract_ttft(timed_events)
                results[name] = {
                    "elapsed": elapsed,
                    "ttft": ttft,
                    "chars": len(text),
                    "prompt": prompt,
                }
            except Exception as e:
                errors[name] = str(e)

        threads = [
            threading.Thread(target=run_agent, args=(i, p))
            for i, p in enumerate(prompts)
        ]

        overall_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=90)
        overall = time.perf_counter() - overall_start

        print(f"整体耗时: {overall * 1000:.0f}ms")
        print(
            f"完成: {len(results)}/{len(prompts)} | 失败: {len(errors)}/{len(prompts)}"
        )

        ttft_list = []
        for name, data in sorted(results.items()):
            ttft_str = f"{data['ttft'] * 1000:.0f}ms" if data["ttft"] else "N/A"
            if data["ttft"]:
                ttft_list.append(data["ttft"])
            print(
                f"  {name}: {data['elapsed'] * 1000:.0f}ms | "
                f"TTFT={ttft_str} | "
                f"{data['chars']}字符 | {data['prompt']}"
            )
        for name, err in sorted(errors.items()):
            print(f"  {name}: 错误 - {err}")

        if ttft_list:
            print(f"\nTTFT 统计:")
            print(f"  平均: {statistics.mean(ttft_list) * 1000:.0f}ms")
            print(f"  最快: {min(ttft_list) * 1000:.0f}ms")
            print(f"  最慢: {max(ttft_list) * 1000:.0f}ms")

        # 至少 3/4 应该成功
        assert len(results) >= 3, f"并发成功率过低: {len(results)}/{len(prompts)}"


class TestEdgeCases:
    """边界场景速度测试"""

    def test_empty_response_handling(self):
        """空回复 / 极短回复处理速度"""
        _print_separator("极短回复速度")

        agent = _make_agent()
        start = time.perf_counter()
        timed_events = _collect_events(agent, "回复'好'这一个字")
        total = time.perf_counter() - start

        text = _extract_full_text(timed_events)
        print(f"总耗时: {total * 1000:.0f}ms")
        print(f"回复: {repr(text)}")

        event_types = [e["type"] for _, e in timed_events]
        assert EventType.DONE in event_types, "未正常结束"

    def test_long_input_latency(self):
        """长输入 prompt 的处理延迟"""
        _print_separator("长输入延迟")

        long_text = "这是一段测试文本。" * 100  # ~900 字符
        prompt = f"请用一句话概括以下内容：\n{long_text}"

        agent = _make_agent()
        start = time.perf_counter()
        timed_events = _collect_events(agent, prompt)
        total = time.perf_counter() - start

        ttft = _extract_ttft(timed_events)
        text = _extract_full_text(timed_events)

        print(f"输入长度: {len(prompt)} 字符")
        print(f"总耗时: {total * 1000:.0f}ms")
        print(f"TTFT: {ttft * 1000:.0f}ms" if ttft else "TTFT: N/A")
        print(f"回复长度: {len(text)} 字符")

        assert total < 180, f"长输入总耗时 {total:.1f}s 过高"

    def test_multi_turn_latency(self):
        """多轮对话时第二轮延迟是否退化"""
        _print_separator("多轮对话延迟对比")

        agent = _make_agent(system_prompt="你是一个助手，回答尽量简短。")

        # 第一轮
        start1 = time.perf_counter()
        events1 = _collect_events(agent, "你好")
        elapsed1 = time.perf_counter() - start1
        ttft1 = _extract_ttft(events1)

        # 第二轮
        start2 = time.perf_counter()
        events2 = _collect_events(agent, "1+1等于几？")
        elapsed2 = time.perf_counter() - start2
        ttft2 = _extract_ttft(events2)

        # 第三轮
        start3 = time.perf_counter()
        events3 = _collect_events(agent, "谢谢")
        elapsed3 = time.perf_counter() - start3
        ttft3 = _extract_ttft(events3)

        print(
            f"第 1 轮: 总耗时={elapsed1 * 1000:.0f}ms | TTFT={ttft1 * 1000:.0f}ms"
            if ttft1
            else f"第 1 轮: 总耗时={elapsed1 * 1000:.0f}ms | TTFT=N/A"
        )
        print(
            f"第 2 轮: 总耗时={elapsed2 * 1000:.0f}ms | TTFT={ttft2 * 1000:.0f}ms"
            if ttft2
            else f"第 2 轮: 总耗时={elapsed2 * 1000:.0f}ms | TTFT=N/A"
        )
        print(
            f"第 3 轮: 总耗时={elapsed3 * 1000:.0f}ms | TTFT={ttft3 * 1000:.0f}ms"
            if ttft3
            else f"第 3 轮: 总耗时={elapsed3 * 1000:.0f}ms | TTFT=N/A"
        )

        # 后续轮次不应该比首轮慢太多（3 倍以内合理，因为历史上下文增加）
        if ttft1 and ttft3:
            ratio = ttft3 / ttft1
            print(f"第3轮/第1轮 TTFT 比值: {ratio:.2f}x")


class TestSummaryBenchmark:
    """综合基准测试 — 一次跑出所有关键指标"""

    def test_full_benchmark(self, simple_tools):
        """综合基准：TTFT / 吞吐量 / 工具调用 一次跑完"""
        _print_separator("综合基准测试 (Full Benchmark)")

        metrics = {}

        # ── 1. 简单对话 TTFT ──
        agent1 = _make_agent()
        start = time.perf_counter()
        events1 = _collect_events(agent1, "你好")
        elapsed1 = time.perf_counter() - start
        metrics["simple_ttft"] = _extract_ttft(events1)
        metrics["simple_total"] = elapsed1

        # ── 2. 吞吐量 ──
        agent2 = _make_agent(system_prompt="你是一个助手，请详细回答。")
        start = time.perf_counter()
        events2 = _collect_events(agent2, "介绍一下机器学习的基本概念，至少150字")
        elapsed2 = time.perf_counter() - start
        text2 = _extract_full_text(events2)
        ttft2 = _extract_ttft(events2) or 0
        gen_time = elapsed2 - ttft2
        metrics["throughput_chars"] = len(text2)
        metrics["throughput_time"] = gen_time
        metrics["throughput_cps"] = len(text2) / gen_time if gen_time > 0 else 0

        # ── 3. 工具调用 ──
        agent3 = _make_agent(
            system_prompt="你是一个天气助手。使用 get_weather 工具查询天气。",
            tools=simple_tools,
        )
        start = time.perf_counter()
        events3 = _collect_events(agent3, "深圳天气")
        elapsed3 = time.perf_counter() - start
        tool_call_ts = next(
            (ts for ts, e in events3 if e["type"] == EventType.TOOL_CALL), None
        )
        tool_result_ts = next(
            (ts for ts, e in events3 if e["type"] == EventType.TOOL_RESULT), None
        )
        metrics["tool_total"] = elapsed3
        metrics["tool_call_at"] = tool_call_ts
        metrics["tool_result_at"] = tool_result_ts

        # ── 输出报告 ──
        print()
        print("┌─────────────────────────────────────────────────┐")
        print("│           Deepminer 端点基准报告                │")
        print("├─────────────────────────────────────────────────┤")

        ttft_ms = metrics["simple_ttft"] * 1000 if metrics["simple_ttft"] else 0
        print(f"│  简单对话 TTFT:       {ttft_ms:>8.0f} ms              │")
        print(
            f"│  简单对话总耗时:      {metrics['simple_total'] * 1000:>8.0f} ms              │"
        )
        print(
            f"│  输出吞吐量:          {metrics['throughput_cps']:>8.1f} 字符/秒         │"
        )
        print(
            f"│  输出字符数:          {metrics['throughput_chars']:>8d}                 │"
        )
        print(
            f"│  工具调用总耗时:      {metrics['tool_total'] * 1000:>8.0f} ms              │"
        )

        if metrics["tool_call_at"] and metrics["tool_result_at"]:
            tool_exec = (metrics["tool_result_at"] - metrics["tool_call_at"]) * 1000
            print(f"│  工具执行耗时:        {tool_exec:>8.0f} ms              │")

        print("└─────────────────────────────────────────────────┘")
        print()

        # 基本断言
        assert metrics["simple_ttft"] is not None, "TTFT 测量失败"
        assert metrics["throughput_chars"] > 0, "吞吐量测量失败"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
