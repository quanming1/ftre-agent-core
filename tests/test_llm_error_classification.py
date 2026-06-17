from ftre_agent_core.llm.completion import LLMError


class StatusError(Exception):
    status_code = 400


def test_status_400_is_bad_request():
    err = LLMError.classify(StatusError("bad input"))

    assert err.code == "bad_request"


def test_invalid_parameter_message_is_bad_request():
    err = LLMError.classify(
        Exception(
            "<400> InternalError.Algo.InvalidParameter: "
            "Repetitive tool calls detected in the conversation history"
        )
    )

    assert err.code == "bad_request"
