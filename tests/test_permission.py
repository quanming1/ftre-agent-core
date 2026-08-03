# -*- coding: utf-8 -*-
"""Unit tests for the pure PermissionEngine decision model."""
import pytest
from pydantic import ValidationError

from ftre_agent_core import (
    PermissionBehavior,
    PermissionDecision,
    PermissionEngine,
    PermissionRequest,
    PermissionRule,
)


@pytest.fixture
def engine() -> PermissionEngine:
    return PermissionEngine()


def test_default_behavior_when_no_rule_matches(engine: PermissionEngine) -> None:
    decision = engine.evaluate(
        PermissionRequest(tool_name="read_file"),
        rules=[],
        default_behavior=PermissionBehavior.ASK,
    )

    assert decision.behavior == PermissionBehavior.ASK
    assert decision.rule_id is None
    assert decision.reason == "No permission rule matched"


def test_default_behavior_defaults_to_ask(engine: PermissionEngine) -> None:
    decision = engine.evaluate(PermissionRequest(tool_name="x"), rules=[])

    assert decision.behavior == PermissionBehavior.ASK


def test_exact_tool_name_match(engine: PermissionEngine) -> None:
    rules = [
        PermissionRule(
            id="allow-read",
            tool_name="read_file",
            behavior=PermissionBehavior.ALLOW,
        ),
    ]
    decision = engine.evaluate(PermissionRequest(tool_name="read_file"), rules)

    assert decision.behavior == PermissionBehavior.ALLOW
    assert decision.rule_id == "allow-read"


def test_exact_rule_does_not_match_other_tool(engine: PermissionEngine) -> None:
    rules = [
        PermissionRule(
            id="allow-read",
            tool_name="read_file",
            behavior=PermissionBehavior.ALLOW,
        ),
    ]
    decision = engine.evaluate(
        PermissionRequest(tool_name="delete_file"),
        rules,
        default_behavior=PermissionBehavior.DENY,
    )

    assert decision.behavior == PermissionBehavior.DENY
    assert decision.rule_id is None


def test_argument_regex_fullmatch(engine: PermissionEngine) -> None:
    rules = [
        PermissionRule(
            id="allow-git-status",
            tool_name="bash",
            argument_regex={"command": r"git status(?:\s+.*)?"},
            behavior=PermissionBehavior.ALLOW,
            priority=10,
        ),
        PermissionRule(
            id="ask-bash",
            tool_name="bash",
            behavior=PermissionBehavior.ASK,
        ),
    ]

    allowed = engine.evaluate(
        PermissionRequest(
            tool_name="bash",
            arguments={"command": "git status --short"},
        ),
        rules,
    )
    asked = engine.evaluate(
        PermissionRequest(
            tool_name="bash",
            arguments={"command": "git stash pop"},
        ),
        rules,
    )

    assert allowed.behavior == PermissionBehavior.ALLOW
    assert allowed.rule_id == "allow-git-status"
    assert asked.behavior == PermissionBehavior.ASK
    assert asked.rule_id == "ask-bash"


def test_argument_regex_requires_argument_and_valid_pattern(
    engine: PermissionEngine,
) -> None:
    missing = PermissionRule(
        id="missing",
        tool_name="bash",
        argument_regex={"command": r"dir"},
        behavior=PermissionBehavior.ALLOW,
    )
    invalid = PermissionRule(
        id="invalid",
        tool_name="bash",
        argument_regex={"command": "("},
        behavior=PermissionBehavior.ALLOW,
    )

    missing_decision = engine.evaluate(
        PermissionRequest(tool_name="bash"),
        [missing],
        default_behavior=PermissionBehavior.ASK,
    )
    invalid_decision = engine.evaluate(
        PermissionRequest(tool_name="bash", arguments={"command": "dir"}),
        [invalid],
        default_behavior=PermissionBehavior.ASK,
    )

    assert missing_decision.behavior == PermissionBehavior.ASK
    assert invalid_decision.behavior == PermissionBehavior.ASK


def test_wildcard_rule_matches_any_tool(engine: PermissionEngine) -> None:
    rules = [
        PermissionRule(
            id="deny-all",
            tool_name="*",
            behavior=PermissionBehavior.DENY,
        ),
    ]
    decision = engine.evaluate(PermissionRequest(tool_name="anything"), rules)

    assert decision.behavior == PermissionBehavior.DENY
    assert decision.rule_id == "deny-all"


def test_disabled_rule_is_ignored(engine: PermissionEngine) -> None:
    rules = [
        PermissionRule(
            id="allow-read",
            tool_name="read_file",
            behavior=PermissionBehavior.ALLOW,
            enabled=False,
        ),
    ]
    decision = engine.evaluate(
        PermissionRequest(tool_name="read_file"),
        rules,
        default_behavior=PermissionBehavior.ASK,
    )

    assert decision.behavior == PermissionBehavior.ASK
    assert decision.rule_id is None


def test_higher_priority_rule_wins(engine: PermissionEngine) -> None:
    rules = [
        PermissionRule(
            id="wildcard-deny",
            tool_name="*",
            behavior=PermissionBehavior.DENY,
            priority=0,
        ),
        PermissionRule(
            id="specific-allow",
            tool_name="read_file",
            behavior=PermissionBehavior.ALLOW,
            priority=100,
        ),
    ]
    decision = engine.evaluate(PermissionRequest(tool_name="read_file"), rules)

    assert decision.behavior == PermissionBehavior.ALLOW
    assert decision.rule_id == "specific-allow"


def test_same_priority_conflict_fails_safe_to_deny(
    engine: PermissionEngine,
) -> None:
    rules = [
        PermissionRule(
            id="allow-read",
            tool_name="read_file",
            behavior=PermissionBehavior.ALLOW,
            priority=10,
        ),
        PermissionRule(
            id="ask-read",
            tool_name="*",
            behavior=PermissionBehavior.ASK,
            priority=10,
        ),
    ]
    decision = engine.evaluate(PermissionRequest(tool_name="read_file"), rules)

    assert decision.behavior == PermissionBehavior.DENY
    assert decision.rule_id is None
    assert decision.reason == "Conflicting permission rules at the same priority"


def test_same_priority_same_behavior_is_not_a_conflict(
    engine: PermissionEngine,
) -> None:
    rules = [
        PermissionRule(
            id="allow-read",
            tool_name="read_file",
            behavior=PermissionBehavior.ALLOW,
            priority=10,
        ),
        PermissionRule(
            id="allow-any",
            tool_name="*",
            behavior=PermissionBehavior.ALLOW,
            priority=10,
        ),
    ]
    decision = engine.evaluate(PermissionRequest(tool_name="read_file"), rules)

    assert decision.behavior == PermissionBehavior.ALLOW
    assert decision.rule_id in {"allow-read", "allow-any"}


def test_evaluate_does_not_mutate_inputs(engine: PermissionEngine) -> None:
    request = PermissionRequest(tool_name="read_file", arguments={"path": "a"})
    rules = [
        PermissionRule(
            id="allow-read",
            tool_name="read_file",
            behavior=PermissionBehavior.ALLOW,
        ),
    ]
    rules_before = [r.model_dump() for r in rules]

    engine.evaluate(request, rules)

    assert [r.model_dump() for r in rules] == rules_before
    assert request.arguments == {"path": "a"}


def test_decision_is_json_serializable(engine: PermissionEngine) -> None:
    rules = [
        PermissionRule(
            id="deny-all",
            tool_name="*",
            behavior=PermissionBehavior.DENY,
        ),
    ]
    decision = engine.evaluate(PermissionRequest(tool_name="x"), rules)

    dumped = decision.model_dump(mode="json")
    assert dumped["behavior"] == "deny"
    assert dumped["rule_id"] == "deny-all"

    restored = PermissionDecision.model_validate(dumped)
    assert restored.behavior == PermissionBehavior.DENY


def test_request_requires_tool_name() -> None:
    with pytest.raises(ValidationError):
        PermissionRequest()


def test_rule_requires_id_tool_name_and_behavior() -> None:
    with pytest.raises(ValidationError):
        PermissionRule(tool_name="x", behavior=PermissionBehavior.ALLOW)
    with pytest.raises(ValidationError):
        PermissionRule(id="r", behavior=PermissionBehavior.ALLOW)
    with pytest.raises(ValidationError):
        PermissionRule(id="r", tool_name="x")


def test_rules_loaded_from_agent_state_permission_context(
    engine: PermissionEngine,
) -> None:
    """规则以类型化模型存在 AgentState.permission_context，Core 取出后传给引擎。"""
    from ftre_agent_core.permission import PermissionContext
    from ftre_agent_core.state import AgentState

    state = AgentState(
        permission_context=PermissionContext(
            permission_rules=[
                PermissionRule(
                    id="deny-delete",
                    tool_name="delete_file",
                    behavior=PermissionBehavior.DENY,
                ),
            ],
            default_behavior=PermissionBehavior.ALLOW,
        ),
    )

    ctx = state.permission_context
    denied = engine.evaluate(
        PermissionRequest(tool_name="delete_file"), ctx.permission_rules,
        ctx.default_behavior,
    )
    assert denied.behavior == PermissionBehavior.DENY
    assert denied.rule_id == "deny-delete"

    allowed = engine.evaluate(
        PermissionRequest(tool_name="read_file"), ctx.permission_rules,
        ctx.default_behavior,
    )
    assert allowed.behavior == PermissionBehavior.ALLOW
    assert allowed.rule_id is None


def test_permission_context_roundtrips_legacy_json():
    """旧 state.json 的 permission_context dict 可直接恢复为类型化模型。"""
    from ftre_agent_core.permission import PermissionContext
    from ftre_agent_core.state import AgentState

    legacy = AgentState.model_validate({
        "context": [],
        "permission_context": {
            "permission_rules": [
                {
                    "id": "ask-bash",
                    "tool_name": "bash",
                    "argument_regex": {},
                    "behavior": "ask",
                    "priority": 0,
                    "enabled": True,
                },
            ],
            "default_behavior": "allow",
        },
    })

    assert isinstance(legacy.permission_context, PermissionContext)
    assert legacy.permission_context.permission_rules[0].id == "ask-bash"
    assert legacy.permission_context.permission_rules[0].behavior == PermissionBehavior.ASK
    assert legacy.permission_context.default_behavior == PermissionBehavior.ALLOW

    # 再序列化回 JSON，字段名不变
    dumped = legacy.model_dump(mode="json")
    assert dumped["permission_context"]["default_behavior"] == "allow"
    assert dumped["permission_context"]["permission_rules"][0]["behavior"] == "ask"


def test_permission_context_defaults_to_empty_allow():
    """默认 PermissionContext = 空规则 + ALLOW（不启用拦截的完整表达）。"""
    from ftre_agent_core.permission import PermissionContext

    ctx = PermissionContext()
    assert ctx.permission_rules == []
    assert ctx.default_behavior == PermissionBehavior.ALLOW
