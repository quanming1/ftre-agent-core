"""
Deepminer 端点 TTFT 直测脚本 — 绕过 LiteLLM，用原始 HTTP 请求

测试维度：
1. 原始 TTFT（首 token 延迟）
2. 总耗时 / 生成吞吐
3. 多次采样稳定性

直接用 httpx 流式 SSE 解析，排除框架开销。

使用方式：
    py tests/test_deepminer_curl.py
"""

import json
import ssl
import statistics
import time

import httpx

# ── 配置 ───────────────────────────────────────────────────────
API_KEY = "sk-hjcMKlZoxvUCzUhsQH2FOXMXnzVh4m4l64QGuLqcBrzOAl7q"
API_BASE = "https://llm-gateway.mlamp.cn/v1"
DEFAULT_MODEL = "claude-opus-4-6"

# 超时配置
CONNECT_TIMEOUT = 60.0
READ_TIMEOUT = 300.0

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 2.0


def stream_chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str = "你是一个简洁的助手。",
    max_tokens: int = 1024,
    retry: int = 0,
) -> dict:
    """
    直接 HTTP 流式请求 OpenAI 兼容端点，手动解析 SSE。
    返回包含 ttft, total_time, text, error 等字段的字典。
    """
    result = {
        "ttft": None,
        "total_time": 0.0,
        "generation_time": 0.0,
        "text": "",
        "chunk_count": 0,
        "error": None,
        "http_status": None,
        "model": model,
    }

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
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
        # 禁用 HTTP/2，使用 HTTP/1.1 避免 SSL 问题
        with httpx.Client(timeout=timeout, http2=False) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                result["http_status"] = resp.status_code

                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    result["error"] = f"HTTP {resp.status_code}: {body[:500]}"
                    result["total_time"] = time.perf_counter() - start
                    return result

                # 逐行解析 SSE
                buffer = ""
                for raw_bytes in resp.iter_bytes():
                    now = time.perf_counter()
                    buffer += raw_bytes.decode("utf-8", errors="replace")

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue

                        data_str = line[len("data:") :].strip()
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
                            elapsed = now - start
                            if result["ttft"] is None:
                                result["ttft"] = elapsed
                            result["text"] += content
                            result["chunk_count"] += 1

    except (httpx.ConnectError, httpx.ConnectTimeout, ssl.SSLError) as e:
        # 连接错误，尝试重试
        if retry < MAX_RETRIES:
            print(
                f"    ⚠️ 连接失败，{RETRY_DELAY}s 后重试 ({retry + 1}/{MAX_RETRIES})..."
            )
            time.sleep(RETRY_DELAY)
            return stream_chat(prompt, model, system_prompt, max_tokens, retry + 1)
        result["error"] = f"连接失败 (重试 {MAX_RETRIES} 次后): {type(e).__name__}"
    except httpx.ReadTimeout:
        result["error"] = "读取超时"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    result["total_time"] = time.perf_counter() - start
    if result["ttft"] is not None:
        result["generation_time"] = result["total_time"] - result["ttft"]

    return result


def print_result(result: dict, label: str = ""):
    """格式化打印结果"""
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")

    if result["error"]:
        print(f"  ❌ 错误: {result['error']}")
        return

    ttft_ms = result["ttft"] * 1000 if result["ttft"] else 0
    char_count = len(result["text"])
    gen_time = result["generation_time"]
    cps = char_count / gen_time if gen_time > 0 else 0

    print(f"  模型:       {result['model']}")
    print(f"  TTFT:       {ttft_ms:.0f} ms  ({result['ttft']:.2f}s)")
    print(
        f"  总耗时:     {result['total_time'] * 1000:.0f} ms  ({result['total_time']:.2f}s)"
    )
    print(f"  生成耗时:   {gen_time * 1000:.0f} ms")
    print(f"  回复长度:   {char_count} 字符")
    print(f"  chunk 数:   {result['chunk_count']}")
    print(f"  吞吐量:     {cps:.1f} 字符/秒")

    # 回复预览
    preview = result["text"][:100].replace("\n", "\\n")
    print(f"  回复预览:   {preview}{'...' if char_count > 100 else ''}")


def run_all():
    """运行所有测试"""
    print()
    print("╔═══════════════════════════════════════════════════════╗")
    print("║   Deepminer 端点 TTFT 直测 (Raw HTTP/1.1)            ║")
    print("║   端点: llm-gateway.mlamp.cn                          ║")
    print(f"║   模型: {DEFAULT_MODEL:<44s}║")
    print("╚═══════════════════════════════════════════════════════╝")

    ttft_samples = []

    # ── 测试 1: 简单 TTFT ──
    print("\n\n📍 测试 1: 简单 TTFT")
    r = stream_chat("你好", max_tokens=64)
    print_result(r, "简单问候")
    if r["ttft"]:
        ttft_samples.append(r["ttft"])

    # ── 测试 2: 稍复杂问题 ──
    print("\n\n📍 测试 2: 稍复杂问题")
    r = stream_chat("1+1等于几？请直接回答数字", max_tokens=32)
    print_result(r, "简单计算")
    if r["ttft"]:
        ttft_samples.append(r["ttft"])

    # ── 测试 3: 长回复吞吐量 ──
    print("\n\n📍 测试 3: 长回复吞吐量")
    r = stream_chat(
        "用大约200字介绍一下 Python 语言的特点",
        system_prompt="你是一个技术专家，请详细回答。",
        max_tokens=1024,
    )
    print_result(r, "长回复")
    if r["ttft"]:
        ttft_samples.append(r["ttft"])

    # ── 汇总 ──
    print(f"\n\n{'═' * 55}")
    print(f"  📊 TTFT 汇总统计")
    print(f"{'═' * 55}")

    if ttft_samples:
        print(f"  采样次数:   {len(ttft_samples)}")
        print(f"  平均 TTFT:  {statistics.mean(ttft_samples) * 1000:.0f} ms")
        print(f"  最快 TTFT:  {min(ttft_samples) * 1000:.0f} ms")
        print(f"  最慢 TTFT:  {max(ttft_samples) * 1000:.0f} ms")
        if len(ttft_samples) > 1:
            print(f"  标准差:     {statistics.stdev(ttft_samples) * 1000:.0f} ms")
    else:
        print("  ❌ 无成功采样")

    print(f"\n{'═' * 55}")
    print("  ✅ 测试完成")
    print(f"{'═' * 55}\n")


if __name__ == "__main__":
    run_all()
