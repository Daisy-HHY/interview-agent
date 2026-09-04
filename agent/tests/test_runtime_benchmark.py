"""runtime benchmark 入口的轻量测试。"""

from agent import runtime_benchmark


class FakeRuntime:
    last_model_elapsed_ms = 7

    def __init__(self, runtime_name: str = "native", use_tool: bool = True) -> None:
        self.runtime_name = runtime_name
        self.use_tool = use_tool
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
        return FakeRuntime(self.config["agent_runtime"])

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
