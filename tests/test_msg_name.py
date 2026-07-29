"""MsgName 枚举、序列化、往返一致性测试。"""
import json

import pytest
from ftre_agent_core.message import (
    AssistantMsg,
    Msg,
    MsgName,
    SystemMsg,
    TextBlock,
    UserMsg,
)


class TestMsgNameEnum:
    """MsgName 枚举值与字符串映射。"""

    def test_enum_values(self):
        assert MsgName.DEFAULT == "default"
        assert MsgName.COMPACT == "compact"
        assert str(MsgName.DEFAULT) == "default"
        assert str(MsgName.COMPACT) == "compact"

    def test_from_string(self):
        assert MsgName("default") is MsgName.DEFAULT
        assert MsgName("compact") is MsgName.COMPACT

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            MsgName("agent_x")


class TestMsgNameSerialization:
    """Msg.name 序列化为字符串，往返一致。"""

    def test_default_serializes_to_string(self):
        msg = UserMsg(name=MsgName.DEFAULT, content="hi")
        dumped = msg.model_dump(mode="json")
        assert dumped["name"] == "default"

    def test_compact_serializes_to_string(self):
        msg = UserMsg(
            name=MsgName.COMPACT,
            content="摘要正文",
            metadata={"hide": True},
        )
        dumped = msg.model_dump(mode="json")
        assert dumped["name"] == "compact"

    def test_json_roundtrip_default(self):
        msg = AssistantMsg(name=MsgName.DEFAULT, content="hello", id="r1")
        payload = json.loads(msg.model_dump_json())
        assert payload["name"] == "default"
        restored = Msg.model_validate(payload)
        assert restored.name == MsgName.DEFAULT

    def test_json_roundtrip_compact(self):
        msg = UserMsg(
            name=MsgName.COMPACT,
            content="摘要",
            metadata={"hide": True, "context_compact": {"tokens_before": 100}},
        )
        payload = json.loads(msg.model_dump_json())
        assert payload["name"] == "compact"
        restored = Msg.model_validate(payload)
        assert restored.name == MsgName.COMPACT


class TestFactoryDefaults:
    """工厂函数默认 name=MsgName.DEFAULT。"""

    def test_usermsg_default_name(self):
        msg = UserMsg(content="hi")
        assert msg.name == MsgName.DEFAULT

    def test_assistantmsg_default_name(self):
        msg = AssistantMsg(content="hi")
        assert msg.name == MsgName.DEFAULT

    def test_systemmsg_default_name(self):
        msg = SystemMsg(content="sys")
        assert msg.name == MsgName.DEFAULT

    def test_usermsg_explicit_compact(self):
        msg = UserMsg(name=MsgName.COMPACT, content="摘要")
        assert msg.name == MsgName.COMPACT

    def test_msg_direct_default(self):
        msg = Msg(content=[TextBlock(text="x")], role="user")
        assert msg.name == MsgName.DEFAULT


class TestBackwardCompatStringInput:
    """合法枚举字符串值输入仍被接受（从磁盘/JSON 反序列化场景）。"""

    def test_string_default_accepted(self):
        msg = Msg(name="default", content=[TextBlock(text="x")], role="user")
        assert msg.name == MsgName.DEFAULT

    def test_string_compact_accepted(self):
        msg = Msg(name="compact", content=[TextBlock(text="x")], role="user")
        assert msg.name == MsgName.COMPACT

    def test_arbitrary_string_rejected(self):
        with pytest.raises(Exception):
            Msg(name="agent_id", content=[TextBlock(text="x")], role="user")
