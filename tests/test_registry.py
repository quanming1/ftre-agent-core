"""registry 工厂测试（PRD-B2 AC3）。"""

from __future__ import annotations

import pytest

from ftre_agent_core.llm import (
    LLMError,
    OpenAICompletionsAdapter,
    OpenAIResponsesAdapter,
    create_llm_handler,
    supported_protocols,
)


class TestFactory:
    def test_dispatches_completions(self):
        handler = create_llm_handler("completions", model="m", api_key="k")
        assert isinstance(handler, OpenAICompletionsAdapter)

    def test_dispatches_responses(self):
        handler = create_llm_handler("responses", model="m", api_key="k")
        assert isinstance(handler, OpenAIResponsesAdapter)

    def test_default_is_completions(self):
        handler = create_llm_handler(model="m", api_key="k")
        assert isinstance(handler, OpenAICompletionsAdapter)

    def test_unknown_api_type_raises_with_supported_list(self):
        with pytest.raises(LLMError) as exc_info:
            create_llm_handler("banana", model="m", api_key="k")
        assert exc_info.value.code == "INVALID_API_TYPE"
        assert "completions" in exc_info.value.message
        assert "responses" in exc_info.value.message
        assert "banana" in exc_info.value.message

    def test_supported_protocols_stable_order(self):
        assert supported_protocols() == ["completions", "responses"]
