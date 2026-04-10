"""
重试机制测试 — 仅验证事件定义

注：重试逻辑已移至 ai-base 层（/chat/retry 路由），
core 层仅保留 EventType.RETRY、RetryData、retry_event() 定义。
"""

def test_retry_event_structure():
    """验证 RETRY Event 结构"""
    from ftre_agent_core.agent.event import retry_event, EventType

    event = retry_event(code="timeout", message="Request timed out", attempt=1, max_attempts=3)

    assert event["type"] == EventType.RETRY
    assert event["data"]["code"] == "timeout"
    assert event["data"]["message"] == "Request timed out"
    assert event["data"]["attempt"] == 1
    assert event["data"]["max_attempts"] == 3
    print("✅ RETRY Event 结构正确")


def test_llm_error_classify():
    """验证 LLMError.classify() 分类"""
    from ftre_agent_core.agent.runner.handler.llm.types import LLMError

    # 用 Exception 代替具体 litellm 异常类
    err = LLMError.classify(Exception("test"))
    assert err.code == "unknown"
    print("✅ LLMError.classify() 分类正确")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  重试事件定义测试")
    print("=" * 50 + "\n")

    test_retry_event_structure()
    test_llm_error_classify()

    print("\n" + "=" * 50)
    print("  ✅ 所有测试通过")
    print("=" * 50 + "\n")
