"""runtime benchmark 入口的轻量测试。"""

from agent import runtime_benchmark


class FakeRuntime:
    runtime_name = "native"
    last_model_elapsed_ms = 7

    def __init__(self) -> None:
        self.messages = [{"role": "system", "content": "s"}]

    def run(self, text, on_delta=None, on_tool_call=None):
        if on_tool_call:
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

    def get_or_create(self, session):
        self.session = session
        return FakeRuntime()

    def save(self, session):
        self.saved = session


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
    assert "sk-secret" not in str(row)
    assert "content" not in str(row)
