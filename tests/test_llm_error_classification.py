from ftre_agent_core.llm import LLMError


class StatusError(Exception):
    status_code = 400


class BillingStatusError(Exception):
    status_code = 402


def test_status_400_is_bad_request():
    err = LLMError.classify(StatusError("bad input"))

    assert err.code == "bad_request"


def test_status_402_is_unretryable_bad_request():
    err = LLMError.classify(BillingStatusError("Insufficient Balance"))

    assert err.code == "bad_request"


def test_invalid_parameter_message_is_bad_request():
    err = LLMError.classify(
        Exception(
            "<400> InternalError.Algo.InvalidParameter: "
            "Repetitive tool calls detected in the conversation history"
        )
    )

    assert err.code == "bad_request"
