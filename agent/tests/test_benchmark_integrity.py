"""评测数据不能因缺失指标、旧历史或预算回退而虚高。"""

from agent import runtime_benchmark as benchmark
from agent.session import SessionStore


def test_budget_uses_only_valid_denominator():
    """没有预算数据不是未命中。"""
    result = benchmark.summarize_budget_rows([
        {"runtime": "native", "budget": {}},
        {"runtime": "pi", "budget": {
            "hit": True, "steps_used": 32, "reason": "hard_limit"}},
    ])
    assert result["samples"] == 1
    assert result["missing_samples"] == 1
    assert result["hit_rate"] == 1
    assert benchmark.summarize_budget_rows([{"budget": {}}])["hit_rate"] is None


def test_benchmark_does_not_touch_business_sessions(tmp_path, monkeypatch):
    """禁止恢复或保存用户会话；两次评测起始上下文相同。"""
    def forbidden(*args, **kwargs):
        """任何访问业务会话文件都使本用例失败。"""
        raise AssertionError("business session accessed")

    monkeypatch.setattr(SessionStore, "_restore", forbidden)
    monkeypatch.setattr(SessionStore, "save", forbidden)
    rows = [benchmark.run_once(str(tmp_path), "fake", "fake", None, "pi", 1,
                               fake_llm=True) for _ in range(2)]
    assert all(row["status"] == "done" for row in rows)
    assert rows[0]["estimated_tokens"] == rows[1]["estimated_tokens"]
    assert not (tmp_path / ".sessions").exists()


def test_hard_limit_is_not_success(tmp_path, monkeypatch):
    """安全阀正常返回也不等于完成任务。"""
    from agent.pi_runtime import PiAgentRuntime

    def exhausted(self, *args, **kwargs):
        """模拟预算耗尽后正常返回。"""
        self._last_budget = {"hit": True, "reason": "hard_limit", "steps_used": 32}
        return "budget exhausted"

    monkeypatch.setattr(PiAgentRuntime, "run", exhausted)
    row = benchmark.run_once(str(tmp_path), "fake", "fake", None, "pi", 1, fake_llm=True)
    assert row["status"] == "budget_exhausted"
    assert row["benchmark_status"] != "done"
