"""
Curl vs LiteLLM 性能对比测试

测试维度：
1. TTFT (首 token 延迟)
2. 总耗时
3. 吞吐量

对比方案：
- curl: 直接 httpx HTTP 请求
- litellm: ReActAgent (litellm SDK)
"""

import json
import statistics
import time

import httpx

from ftre_agent_core.agent import EventType, ReActAgent

# ── 配置 ───────────────────────────────────────────────────────
API_KEY = "sk-REDACTED"
API_BASE = "https://llm-gateway.REDACTED.example.com/v1"
MODEL_CURL = "claude-opus-4-6"
MODEL_LITELLM = "openai/claude-opus-4-6"

CONNECT_TIMEOUT = 60.0
READ_TIMEOUT = 300.0

# 测试 prompt
TEST_PROMPTS = [
    ("短回复", "你好"),
    ("中等回复", "用大约50字介绍一下Python"),
    ("长回复", "用大约200字介绍一下Python语言的特点"),
]


# ════════════════════════════════════════════════════════════════
# 方式1: 直接 curl (httpx)
# ════════════════════════════════════════════════════════════════


def curl_stream(prompt: str, max_tokens: int = 512) -> dict:
    """直接 HTTP 请求，模拟 curl"""
    result = {
        "ttft": None,
        "total_time": 0.0,
        "generation_time": 0.0,
        "text": "",
        "chunk_count": 0,
        "error": None,
    }

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": MODEL_CURL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()

    try:
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=READ_TIMEOUT,
            write=60.0,
            pool=60.0,
        )
        with httpx.Client(timeout=timeout, http2=False) as client, client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    result["error"] = f"HTTP {resp.status_code}"
                    return result

                buffer = ""
                for raw_bytes in resp.iter_bytes():
                    now = time.perf_counter()
                    buffer += raw_bytes.decode("utf-8", errors="replace")

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue

                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            continue

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content")

                        if content:
                            if result["ttft"] is None:
                                result["ttft"] = now - start
                            result["text"] += content
                            result["chunk_count"] += 1

    except Exception as e:  # noqa: BLE001 - diagnostic script reports failure
        result["error"] = f"{type(e).__name__}: {e}"

    result["total_time"] = time.perf_counter() - start
    if result["ttft"] is not None:
        result["generation_time"] = result["total_time"] - result["ttft"]

    return result


# ════════════════════════════════════════════════════════════════
# 方式2: LiteLLM (ReActAgent)
# ════════════════════════════════════════════════════════════════


def litellm_stream(prompt: str, max_tokens: int = 512) -> dict:
    """使用 ReActAgent (litellm SDK)"""
    result = {
        "ttft": None,
        "total_time": 0.0,
        "generation_time": 0.0,
        "text": "",
        "chunk_count": 0,
        "error": None,
    }

    agent = ReActAgent(
        model=MODEL_LITELLM,
        api_key=API_KEY,
        api_base=API_BASE,
        system_prompt="你是一个简洁的助手。",
        tools=[],
    )

    start = time.perf_counter()
    first_chunk = True
    chunks = 0

    try:
        for event in agent.run(prompt):
            if event["type"] == EventType.MESSAGE:
                data = event["data"]
                if data.get("content"):
                    if first_chunk:
                        result["ttft"] = time.perf_counter() - start
                        first_chunk = False
                    result["text"] += data["content"]
                    chunks += 1

        result["chunk_count"] = chunks

    except Exception as e:  # noqa: BLE001 - diagnostic script reports failure
        result["error"] = f"{type(e).__name__}: {e}"

    result["total_time"] = time.perf_counter() - start
    if result["ttft"] is not None:
        result["generation_time"] = result["total_time"] - result["ttft"]

    return result


# ════════════════════════════════════════════════════════════════
# 对比测试
# ════════════════════════════════════════════════════════════════


def run_comparison():
    """运行对比测试"""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         Curl vs LiteLLM 性能对比测试                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    summary = []

    for name, prompt in TEST_PROMPTS:
        print(f"\n{'=' * 60}")
        print(f"  📍 测试: {name}")
        print(f"{'=' * 60}")

        # Curl
        print("\n  [curl] 正在测试...")
        curl_result = curl_stream(prompt)
        curl_ttft = curl_result["ttft"] * 1000 if curl_result["ttft"] else 0
        curl_total = curl_result["total_time"] * 1000
        print(f"  [curl] TTFT={curl_ttft:.0f}ms, 总耗时={curl_total:.0f}ms, 字符={len(curl_result['text'])}")

        # LiteLLM
        print("\n  [litellm] 正在测试...")
        llm_result = litellm_stream(prompt)
        llm_ttft = llm_result["ttft"] * 1000 if llm_result["ttft"] else 0
        llm_total = llm_result["total_time"] * 1000
        print(f"  [litellm] TTFT={llm_ttft:.0f}ms, 总耗时={llm_total:.0f}ms, 字符={len(llm_result['text'])}")

        # 对比
        ttft_diff = llm_ttft - curl_ttft
        total_diff = llm_total - curl_total

        print("\n  📊 对比结果:")
        print(f"     TTFT 差异:   {ttft_diff:+.0f}ms ({'litellm慢' if ttft_diff > 0 else 'litellm快'})")
        print(f"     总耗时差异:  {total_diff:+.0f}ms ({'litellm慢' if total_diff > 0 else 'litellm快'})")

        summary.append({
            "name": name,
            "curl_ttft": curl_ttft,
            "curl_total": curl_total,
            "llm_ttft": llm_ttft,
            "llm_total": llm_total,
            "ttft_diff": ttft_diff,
            "total_diff": total_diff,
        })

        # 错误检查
        if curl_result["error"]:
            print(f"\n  ⚠️ curl 错误: {curl_result['error']}")
        if llm_result["error"]:
            print(f"\n  ⚠️ litellm 错误: {llm_result['error']}")

    # 汇总
    print(f"\n\n{'═' * 60}")
    print("  📊 汇总")
    print(f"{'═' * 60}")
    print(f"  {'测试':<10} {'curl TTFT':<12} {'litellm TTFT':<14} {'差异':<10}")
    print(f"  {'-' * 50}")

    for s in summary:
        print(f"  {s['name']:<10} {s['curl_ttft']:>8.0f}ms  {s['llm_ttft']:>10.0f}ms  {s['ttft_diff']:>+8.0f}ms")

    if summary:
        avg_ttft_diff = statistics.mean([s["ttft_diff"] for s in summary])
        avg_total_diff = statistics.mean([s["total_diff"] for s in summary])
        print(f"  {'-' * 50}")
        print(f"  {'平均':<10} {'':<12} {'':<14} {avg_ttft_diff:>+8.0f}ms")
        print(f"  {'平均总耗时差异':<10} {'':<12} {'':<14} {avg_total_diff:>+8.0f}ms")

    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    run_comparison()
