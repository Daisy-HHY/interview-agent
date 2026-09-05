"""Pi runtime 的最小、可解释动态预算决策。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetDecision:
    """一次预算检查的结果。"""

    allow: bool
    reason: str


def decide_budget(
    *,
    step: int,
    soft_limit: int,
    hard_limit: int,
    elapsed_ms: int,
    context_tokens: int,
    context_limit: int | None,
    tool_failures: int,
    max_elapsed_ms: int = 600_000,
    max_tool_failures: int = 2,
    context_pressure_limit: float = 0.9,
) -> BudgetDecision:
    """按确定性阈值决定是否允许进入下一次模型调用。"""
    if step >= hard_limit:
        return BudgetDecision(False, "hard_limit")
    if elapsed_ms >= max_elapsed_ms:
        return BudgetDecision(False, "elapsed_limit")
    if tool_failures >= max_tool_failures:
        return BudgetDecision(False, "repeated_tool_failure")
    if context_limit and context_tokens / context_limit >= context_pressure_limit:
        return BudgetDecision(False, "context_pressure")
    if step < soft_limit:
        return BudgetDecision(True, "within_soft_budget")
    return BudgetDecision(True, "extended_within_hard_limit")
