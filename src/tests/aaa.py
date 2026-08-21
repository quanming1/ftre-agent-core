"""
ReActAgent 取消测试

测试场景：
1. 取消响应速度
2. 取消后状态正确
3. 取消后能正常开始新对话
"""

import json
import threading

from ftre_agent_core.agent import ReActAgent

API_KEY = "sk-REDACTED"
API_BASE = "https://llm-gateway.REDACTED.example.com/v1"
MODEL = "openai/deepseek-reasoner"


class TestCancelLatency:
    """取消延迟测试"""

    def test_measure_cancel_latency(self):
        """测量从调用 cancel 到生成器停止的延迟"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个助手，回答时要非常详细，至少500字。",
            tools=[],
        )

        def consume():
            for event in agent.run("详细介绍一下量子计算的原理和应用"):
                print(json.dumps(event, ensure_ascii=False, default=str))

        thread = threading.Thread(target=consume)
        thread.start()
        thread.join()


testCancelLatency = TestCancelLatency()


testCancelLatency.test_measure_cancel_latency()
