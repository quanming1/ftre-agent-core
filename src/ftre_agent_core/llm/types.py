"""
模拟 OpenAI SDK 对象结构

所有适配器共用这些数据类来构造返回值，
让上层代码（LLMHandler）能用和真实 OpenAI SDK 一样的属性访问方式。
"""
from dataclasses import dataclass


@dataclass
class FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list | None = None

@dataclass
class FakeChoice:
    message: FakeMessage = None
    finish_reason: str = "stop"

@dataclass
class FakeResponse:
    choices: list = None
    usage: FakeUsage = None

@dataclass
class FakeDelta:
    content: str | None = None
    tool_calls: list | None = None

@dataclass
class FakeFunctionDelta:
    name: str | None = None
    arguments: str | None = None

@dataclass
class FakeToolCallDelta:
    index: int = 0
    id: str | None = None
    function: FakeFunctionDelta = None

@dataclass
class FakeStreamChoice:
    delta: FakeDelta = None

@dataclass
class FakeChunk:
    choices: list = None
    usage: FakeUsage = None
