"""MessageContext stateless helper tests."""
from ftre_agent_core.message import Msg
from ftre_agent_core.message_context import MessageContext
from ftre_agent_core.state import AgentState


def test_message_context_updates_agent_state_context():
    state = AgentState()

    MessageContext.add_user(state.context, "hello")
    MessageContext.add_assistant(state.context, "world", reasoning="thinking")

    assert all(isinstance(message, Msg) for message in state.context)
    assert [message.role for message in state.context] == ["user", "assistant"]
    assert MessageContext.messages(state.context)[1]["reasoning_content"] == "thinking"


def test_get_messages_prepends_system_prompt_without_storing_it():
    state = AgentState()
    MessageContext.add_user(state.context, "hello")

    messages = MessageContext.get_messages(state.context, "system")

    assert messages[0] == {"role": "system", "content": "system"}
    assert len(messages) == 2
    assert len(state.context) == 1


def test_set_and_clear_messages_mutate_the_supplied_context():
    state = AgentState()

    MessageContext.set_messages(
        state.context,
        [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}],
    )
    assert [message.get_text_content() for message in state.context] == ["one", "two"]

    MessageContext.clear(state.context)
    assert state.context == []
