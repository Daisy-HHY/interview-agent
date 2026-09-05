"""runtime benchmark 入口的轻量测试。"""

from agent import runtime_benchmark


class FakeRuntime:
    last_model_elapsed_ms = 7

    def __init__(
        self,
        runtime_name: str = "native",
        use_tool: bool = True,
        compaction=None,
    ) -> None:
        self.runtime_name = runtime_name
        self.use_tool = use_tool
        self.last_compaction = compaction or {"state": "disabled"}
        self.last_budget = {
            "steps_used": 1,
            "hit": False,
            "reason": "natural_completion",
            "soft_limit": 8,
            "hard_limit": 32,
        }
        self.messages = [{"role": "system", "content": "s"}]

    def run(self, text, on_delta=None, on_tool_call=None):
        if self.use_tool and on_tool_call:
            on_tool_call("read_file", {"path": "app.py"}, "start", "")
            on_tool_call("read_file", {"path": "app.py"}, "end", "content")
        if on_delta:
            on_delta("回答")
        self.messages.append({"role": "user", "content": text})
        self.messages.append({"role": "assistant", "content": "回答"})
        return "回答"


class FakeStore:
    def configure(self, **kwargs):
        self.config = kwargs
        self.available_tools = [
            "list_directory",
            "search_code",
            "read_file",
            "lookup_questions",
        ]
        self.enabled_tools = kwargs.get("enabled_tools") or self.available_tools

    def get_or_create(self, session):
        self.session = session
        compaction = (
            {"state": "not_needed", "before_tokens": 8, "after_tokens": 8,
             "compaction_elapsed_ms": 1}
            if self.config.get("compaction_enabled")
            else {"state": "disabled"}
        )
        return FakeRuntime(self.config["agent_runtime"], compaction=compaction)

    def save(self, session):
        self.saved = session


class FakeStoreWithFactory(FakeStore):
    def __init__(self, llm_factory=None):
        self.llm_factory = llm_factory


def test_run_once_returns_sanitized_metric(monkeypatch):
    """benchmark 输出指标，不包含 API Key 和完整工具结果。"""
    monkeypatch.setattr(runtime_benchmark, "SessionStore", FakeStore)

    row = runtime_benchmark.run_once(
        workspace="/project",
        api_key="sk-secret",
        model="gpt-test",
        base_url="https://example.test",
        runtime="native",
        index=1,
    )

    assert row["runtime"] == "native"
    assert row["status"] == "done"
    assert row["model_elapsed_ms"] == 7
    assert row["first_delta_ms"] is not None
    assert row["tools"][0]["tool"] == "read_file"
    assert row["tools"][0]["result_chars"] == len("content")
    assert row["benchmark_status"] == "done"
    assert row["tool_call_count"] == 1
    assert row["required_tool_used"] is True
    assert row["tool_sequence"] == ["read_file"]
    assert row["available_tools"] == [
        "list_directory",
        "search_code",
        "read_file",
        "lookup_questions",
    ]
    assert row["enabled_tools"] == row["available_tools"]
    assert "sk-secret" not in str(row)
    assert "content" not in str(row)


def test_run_once_accepts_pi_runtime(monkeypatch):
    """benchmark 可单独跑 pi runtime。"""
    monkeypatch.setattr(runtime_benchmark, "SessionStore", FakeStore)

    row = runtime_benchmark.run_once(
        workspace="/project",
        api_key="sk-secret",
        model="gpt-test",
        base_url=None,
        runtime="pi",
        index=1,
    )

    assert row["runtime"] == "pi"
    assert row["configured_runtime"] == "pi"


def test_run_once_passes_enabled_tools(monkeypatch):
    """benchmark 可指定启用工具清单。"""
    monkeypatch.setattr(runtime_benchmark, "SessionStore", FakeStore)

    row = runtime_benchmark.run_once(
        workspace="/project",
        api_key="sk-secret",
        model="gpt-test",
        base_url=None,
        runtime="pi",
        index=1,
        enabled_tools=["list_directory"],
    )

    assert row["enabled_tools"] == ["list_directory"]


def test_run_once_marks_insufficient_tool_use(monkeypatch):
    """benchmark 应标记未使用项目读取工具的样本。"""
    class NoToolStore(FakeStore):
        def get_or_create(self, session):
            self.session = session
            return FakeRuntime(self.config["agent_runtime"], use_tool=False)

    monkeypatch.setattr(runtime_benchmark, "SessionStore", NoToolStore)

    row = runtime_benchmark.run_once(
        workspace="/project",
        api_key="sk-secret",
        model="gpt-test",
        base_url=None,
        runtime="pi",
        index=1,
    )

    assert row["status"] == "done"
    assert row["benchmark_status"] == "insufficient_tool_use"
    assert row["tool_call_count"] == 0
    assert row["required_tool_used"] is False
    assert row["tool_sequence"] == []


def test_run_once_accepts_fake_llm_mode(monkeypatch):
    """FakeLLM benchmark 模式可在不提供真实 key 时运行。"""
    monkeypatch.setattr(runtime_benchmark, "SessionStore", FakeStoreWithFactory)

    row = runtime_benchmark.run_once(
        workspace="/project",
        api_key="",
        model="fake",
        base_url=None,
        runtime="pi",
        index=1,
        fake_llm=True,
    )

    assert row["runtime"] == "pi"
    assert row["status"] == "done"


def test_run_once_reports_compaction_metrics_and_mode(monkeypatch):
    """benchmark 支持压缩对照所需的脱敏指标。"""
    monkeypatch.setattr(runtime_benchmark, "SessionStore", FakeStore)

    row = runtime_benchmark.run_once(
        workspace="/project",
        api_key="sk-secret",
        model="gpt-test",
        base_url=None,
        runtime="pi",
        index=1,
        compaction_enabled=True,
        compaction_trigger_tokens=10,
        compaction_keep_messages=2,
    )

    assert row["compaction_enabled"] is True
    assert row["compaction"]["state"] == "not_needed"
    assert row["compression_token_saved"] is None
    assert "sk-secret" not in str(row)


def test_summarize_compaction_rows_returns_comparable_aggregates():
    rows = [
        {
            "runtime": "pi",
            "compaction_enabled": True,
            "compression_token_saved": 40,
            "compression_ratio": 0.4,
            "compression_elapsed_ms": 3,
            "total_elapsed_ms": 100,
            "status": "done",
        },
        {
            "runtime": "pi",
            "compaction_enabled": True,
            "compression_token_saved": 0,
            "compression_ratio": 0.0,
            "compression_elapsed_ms": 4,
            "total_elapsed_ms": 120,
            "status": "error",
        },
    ]

    summary = runtime_benchmark.summarize_compaction_rows(rows)

    assert summary["samples"] == 2
    assert summary["completed"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["avg_token_saved"] == 20
    assert summary["avg_compression_elapsed_ms"] == 3.5


def test_summarize_budget_rows_reports_hit_rate():
    rows = [
        {"budget": {"steps_used": 2, "hit": True, "reason": "hard_limit"}},
        {"budget": {"steps_used": 1, "hit": False, "reason": "natural_completion"}},
    ]

    summary = runtime_benchmark.summarize_budget_rows(rows)

    assert summary == {
        "samples": 2,
        "hit_count": 1,
        "hit_rate": 0.5,
        "natural_completion_count": 1,
        "avg_steps_used": 1.5,
        "reasons": {"hard_limit": 1, "natural_completion": 1},
    }
