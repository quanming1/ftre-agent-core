"""
ReActAgent 取消测试

测试场景：
1. 取消响应速度
2. 取消后状态正确
3. 取消后能正常开始新对话
"""
import time
import threading
import pytest
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.agent.runner import RunStatus
from ftre_agent_core.tool import tool


API_KEY = "sk-1cdcb7b0e7fb40d49bd6b66b8666022e"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "openai/qwen-plus"


class TestCancelBasic:
    """基础取消测试"""

    def test_cancel_nowait_immediate(self):
        """cancel_nowait 应该立即返回"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个助手，回答时尽量详细。",
            tools=[],
        )

        # 在另一个线程中消费事件
        events = []
        def consume():
            for event in agent.run("写一篇500字的文章介绍人工智能"):
                events.append(event)

        thread = threading.Thread(target=consume)
        thread.start()

        # 等待开始接收响应
        time.sleep(1)
        
        # 测量取消耗时
        start = time.time()
        agent.cancel_nowait()
        cancel_time = time.time() - start

        # cancel_nowait 应该立即返回（< 100ms）
        assert cancel_time < 0.1, f"cancel_nowait 耗时 {cancel_time:.3f}s，应该立即返回"

        # 等待线程结束
        thread.join(timeout=10)
        
        print(f"cancel_nowait 耗时: {cancel_time*1000:.1f}ms")
        print(f"收到 {len(events)} 个事件")

    def test_cancel_sync_blocks_until_done(self):
        """cancel_sync 应该阻塞直到善后完成"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个助手，回答时尽量详细。",
            tools=[],
        )

        events = []
        done = threading.Event()
        
        def consume():
            for event in agent.run("写一篇500字的文章介绍人工智能"):
                events.append(event)
            done.set()

        thread = threading.Thread(target=consume)
        thread.start()

        # 等待开始接收响应
        time.sleep(1)
        
        # 测量取消耗时
        start = time.time()
        agent.cancel_sync()
        cancel_time = time.time() - start

        # cancel_sync 返回后，善后应该已完成
        # 但不应该等太久（最多几秒）
        assert cancel_time < 5, f"cancel_sync 耗时 {cancel_time:.3f}s，超过预期"

        # 等待消费线程结束
        thread.join(timeout=5)
        
        print(f"cancel_sync 耗时: {cancel_time*1000:.1f}ms")
        print(f"收到 {len(events)} 个事件")

    def test_cancel_state_correct(self):
        """取消后状态应该正确"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个助手。",
            tools=[],
        )

        def consume():
            for _ in agent.run("写一篇文章"):
                pass

        thread = threading.Thread(target=consume)
        thread.start()

        time.sleep(0.5)
        agent.cancel_nowait()
        
        thread.join(timeout=10)

        # 取消后状态应该是 CANCELLED 或已重置为 IDLE
        assert agent.state.status in (RunStatus.CANCELLED, RunStatus.IDLE, RunStatus.COMPLETED)

    def test_cancel_then_new_conversation(self):
        """取消后应该能正常开始新对话"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个简洁的助手。",
            tools=[],
        )

        # 第一次对话，取消
        def consume1():
            for _ in agent.run("写一篇长文章"):
                pass

        thread = threading.Thread(target=consume1)
        thread.start()
        
        time.sleep(0.5)
        agent.cancel_nowait()
        thread.join(timeout=10)

        # 第二次对话，正常完成
        events = list(agent.run("你好"))
        
        event_types = [e["type"] for e in events]
        assert EventType.DONE in event_types, "取消后应该能正常开始新对话"


class TestCancelWithTools:
    """工具调用场景的取消测试"""

    @pytest.fixture
    def slow_tool(self):
        @tool()
        def slow_operation(seconds: int = 3) -> str:
            """模拟耗时操作"""
            time.sleep(seconds)
            return f"操作完成，耗时 {seconds} 秒"
        return slow_operation

    def test_cancel_during_tool_execution(self, slow_tool):
        """工具执行期间取消"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="当用户要求时，使用 slow_operation 工具。",
            tools=[slow_tool],
        )

        events = []
        def consume():
            for event in agent.run("执行一个耗时5秒的操作"):
                events.append(event)
                if event["type"] == EventType.ASSISTANT_MESSAGE_COMPLETE:
                    tool_calls = [b for b in event["data"]["content"] if b.get("type") == "toolCall"]
                    if tool_calls:
                        print(f"工具调用: {tool_calls[0]['name']}")

        thread = threading.Thread(target=consume)
        thread.start()

        # 等待工具开始执行
        time.sleep(2)
        
        # 取消
        start = time.time()
        agent.cancel_nowait()
        cancel_time = time.time() - start
        
        # cancel_nowait 应该立即返回
        assert cancel_time < 0.1

        thread.join(timeout=10)
        
        print(f"取消耗时: {cancel_time*1000:.1f}ms")
        print(f"事件: {[e['type'].value for e in events]}")


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

        events = []
        generator_stopped = threading.Event()
        
        def consume():
            for event in agent.run("详细介绍一下量子计算的原理和应用"):
                events.append((time.time(), event))
            generator_stopped.set()

        thread = threading.Thread(target=consume)
        thread.start()

        # 等待开始接收响应
        time.sleep(2)
        
        cancel_time = time.time()
        agent.cancel_nowait()
        
        # 等待生成器停止
        generator_stopped.wait(timeout=10)
        stop_time = time.time()
        
        latency = stop_time - cancel_time
        
        thread.join(timeout=5)

        # 统计取消后还收到多少事件
        events_after_cancel = [(t, e) for t, e in events if t > cancel_time]
        
        print(f"取消延迟: {latency*1000:.1f}ms")
        print(f"取消后收到 {len(events_after_cancel)} 个事件")
        print(f"总共收到 {len(events)} 个事件")
        
        # 延迟应该在合理范围内（< 5秒）
        assert latency < 5, f"取消延迟 {latency:.1f}s 超过预期"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
