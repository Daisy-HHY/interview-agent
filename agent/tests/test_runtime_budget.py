"""动态运行预算策略测试。"""

from agent.runtime_budget import decide_budget


def test_budget_allows_soft_and_hard_limit_extension():
    decision = decide_budget(
        step=3,
        soft_limit=2,
        hard_limit=8,
        elapsed_ms=100,
        context_tokens=100,
        context_limit=1000,
        tool_failures=0,
    )

    assert decision.allow is True
    assert decision.reason == "extended_within_hard_limit"


def test_budget_stops_on_context_pressure_and_tool_failures():
    context_decision = decide_budget(
        step=2,
        soft_limit=2,
        hard_limit=8,
        elapsed_ms=100,
        context_tokens=950,
        context_limit=1000,
        tool_failures=0,
    )
    failure_decision = decide_budget(
        step=2,
        soft_limit=2,
        hard_limit=8,
        elapsed_ms=100,
        context_tokens=100,
        context_limit=1000,
        tool_failures=2,
    )

    assert context_decision.allow is False
    assert context_decision.reason == "context_pressure"
    assert failure_decision.allow is False
    assert failure_decision.reason == "repeated_tool_failure"
