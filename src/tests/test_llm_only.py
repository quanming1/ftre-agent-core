"""
纯 LLM 调用对比：直接 httpx vs litellm.completion()

排除 ReActAgent 的框架开销，只对比底层 LLM 调用。
"""

import json
import statistics
import time

import httpx
import litellm

# ── 配置 ───────────────────────────────────────────────────────
API_KEY = "sk-hjcMKlZoxvUCzUhsQH2FOXMXnzVh4m4l64QGuLqcBrzOAl7q"
API_BASE = "https://llm-gateway.mlamp.cn/v1"
MODEL = "openai/claude-opus-4-6"

CONNECT_TIMEOUT = 60.0
READ_TIMEOUT = 300.0

# 测试 prompt
TEST_PROMPTS = [
    ("短回复", "你好", 64),
    ("中等回复", "用大约50字介绍一下Python", 256),
    ("长回复", "用大约200字介绍一下Python语言的特点", 512),
]


# ════════════════════════════════════════════════════════════════
# 方式1: 直接 httpx
# ════════════════════════════════════════════════════════════════


def httpx_stream(prompt: str, max_tokens: int) -> dict:
    """直接 HTTP 请求"""
    result = {
        "ttft": None,
        "total_time": 0.0,
        "text": "",
        "error": None,
    }

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": MODEL,  # httpx 直接用 litellm 的模型名
        "messages": [{"role": "user", "content": prompt}],
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
        with httpx.Client(timeout=timeout, http2=False) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
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

                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            continue

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            if result["ttft"] is None:
                                result["ttft"] = now - start
                            result["text"] += content

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    result["total_time"] = time.perf_counter() - start
    return result


# ════════════════════════════════════════════════════════════════
# 方式2: litellm.completion()
# ════════════════════════════════════════════════════════════════


def litellm_stream(prompt: str, max_tokens: int) -> dict:
    """使用 litellm.completion()"""
    result = {
        "ttft": None,
        "total_time": 0.0,
        "text": "",
        "error": None,
    }

    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()

    try:
        response = litellm.completion(
            model=MODEL,
            messages=messages,
            stream=True,
            api_key=API_KEY,
            api_base=API_BASE,
        )

        for chunk in response:
            now = time.perf_counter()
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                if result["ttft"] is None:
                    result["ttft"] = now - start
                result["text"] += delta.content

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    result["total_time"] = time.perf_counter() - start
    return result


# ════════════════════════════════════════════════════════════════
# 对比测试
# ════════════════════════════════════════════════════════════════


def run_comparison():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       纯 LLM 调用对比: httpx vs litellm.completion()          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    summary = []

    for name, prompt, max_tokens in TEST_PROMPTS:
        print(f"\n{'=' * 60}")
        print(f"  📍 {name}")
        print(f"{'=' * 60}")

        # httpx
        print("  [httpx]...", end=" ", flush=True)
        h_result = httpx_stream(prompt, max_tokens)
        h_ttft = h_result["ttft"] * 1000 if h_result["ttft"] else 0
        h_total = h_result["total_time"] * 1000
        print(f"TTFT={h_ttft:.0f}ms, 总={h_total:.0f}ms")
        if h_result["error"]:
            print(f"        ⚠️ {h_result['error']}")

        # litellm
        print("  [litellm]...", end=" ", flush=True)
        l_result = litellm_stream(prompt, max_tokens)
        l_ttft = l_result["ttft"] * 1000 if l_result["ttft"] else 0
        l_total = l_result["total_time"] * 1000
        print(f"TTFT={l_ttft:.0f}ms, 总={l_total:.0f}ms")
        if l_result["error"]:
            print(f"         ⚠️ {l_result['error']}")

        # 对比
        diff_ttft = l_ttft - h_ttft
        diff_total = l_total - h_total

        print(f"\n  📊 差异:")
        print(f"     TTFT:   {diff_ttft:>+8.0f}ms")
        print(f"     总耗时: {diff_total:>+8.0f}ms")

        if h_result["ttft"] and l_result["ttft"]:
            summary.append({
                "name": name,
                "httpx_ttft": h_ttft,
                "httpx_total": h_total,
                "litellm_ttft": l_ttft,
                "litellm_total": l_total,
                "diff_ttft": diff_ttft,
                "diff_total": diff_total,
            })

    # 汇总
    if summary:
        print(f"\n\n{'═' * 60}")
        print(f"  📊 汇总")
        print(f"{'═' * 60}")
        print(f"  {'测试':<10} {'httpx':<18} {'litellm':<18} {'TTFT差异':<12}")
        print(f"  {'-' * 60}")

        for s in summary:
            print(f"  {s['name']:<10} {s['httpx_ttft']:>8.0f}ms / {s['httpx_total']:>8.0f}ms  "
                  f"{s['litellm_ttft']:>8.0f}ms / {s['litellm_total']:>8.0f}ms  {s['diff_ttft']:>+8.0f}ms")

        avg_ttft = statistics.mean([s["diff_ttft"] for s in summary])
        avg_total = statistics.mean([s["diff_total"] for s in summary])
        print(f"  {'-' * 60}")
        print(f"  {'平均':<10} {'':<18} {'':<18} {avg_ttft:>+8.0f}ms")
        print(f"  {'平均总耗时':<10} {'':<18} {'':<18} {avg_total:>+8.0f}ms")

        if avg_ttft < 50:
            print(f"\n  ✅ 差异 < 50ms，可忽略")
        elif avg_ttft < 500:
            print(f"\n  ⚠️ 差异 50~500ms，轻微开销")
        else:
            print(f"\n  ❌ 差异 > 500ms，litellm 开销明显")

    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    run_comparison()
