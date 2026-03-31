"""
Pytest 配置和共享 fixtures
"""
import pytest


@pytest.fixture
def mock_openai_response():
    """模拟 OpenAI 响应（无工具调用）"""
    class MockChoice:
        def __init__(self):
            self.delta = MockDelta()
            self.finish_reason = None
    
    class MockDelta:
        def __init__(self):
            self.content = "Hello, I'm an AI assistant."
            self.tool_calls = None
    
    class MockChunk:
        def __init__(self):
            self.choices = [MockChoice()]
    
    return [MockChunk()]


@pytest.fixture
def mock_tool_call_response():
    """模拟 OpenAI 响应（带工具调用）"""
    class MockFunction:
        def __init__(self):
            self.name = "get_weather"
            self.arguments = '{"city": "Beijing"}'
    
    class MockToolCall:
        def __init__(self):
            self.id = "call_123"
            self.type = "function"
            self.function = MockFunction()
    
    class MockDelta:
        def __init__(self):
            self.content = None
            self.tool_calls = [MockToolCall()]
    
    class MockChoice:
        def __init__(self):
            self.delta = MockDelta()
            self.finish_reason = "tool_calls"
    
    class MockChunk:
        def __init__(self):
            self.choices = [MockChoice()]
    
    return [MockChunk()]
