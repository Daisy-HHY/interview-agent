"""协议层测试（设计第 7.2.2 节，第 2 层测试——不碰 LLM）。

覆盖：
- parse_message 解析各种合法/非法输入
- notify 系列：流式、工具调用、完成、错误 通知的格式正确
- handle_message 分发到对应 handler
"""

import json

import agent.protocol as protocol

# ──────────────────────────────────────────────
# parse_message：入站消息解析
# ──────────────────────────────────────────────


class TestParseMessage:
    def test_parses_chat_message(self):
        """能正确解析 chat 消息（设计第 1.5.2 节）。"""
        line = '{"jsonrpc":"2.0","method":"chat","params":{"session":"s1","text":"hi"}}'
        msg = protocol.parse_message(line)

        assert msg is not None
        assert msg["method"] == "chat"
        assert msg["params"]["text"] == "hi"

    def test_parses_init_message(self):
        """能解析 init 消息。"""
        line = (
            '{"jsonrpc":"2.0","method":"init",'
            '"params":{"workspace":"/proj","api_key":"sk-x","model":"gpt-4o-mini"}}'
        )
        msg = protocol.parse_message(line)

        assert msg is not None
        assert msg["method"] == "init"
        assert msg["params"]["workspace"] == "/proj"

    def test_parses_stop_message(self):
        """能解析 stop 消息。"""
        line = '{"jsonrpc":"2.0","method":"stop","params":{"session":"s1"}}'
        msg = protocol.parse_message(line)

        assert msg is not None
        assert msg["method"] == "stop"

    def test_tolerates_trailing_newline(self):
        """尾随换行/空白也能解析（stdin 读到行通常带 \\n）。"""
        line = '{"method":"chat","params":{"text":"x"}}\n'
        msg = protocol.parse_message(line)

        assert msg is not None
        assert msg["method"] == "chat"

    def test_empty_line_returns_none(self):
        """空行返回 None（不崩溃）。"""
        assert protocol.parse_message("") is None
        assert protocol.parse_message("   \n") is None

    def test_malformed_json_returns_none(self):
        """格式错误的 JSON 返回 None（容错，不抛异常）。"""
        assert protocol.parse_message("{not json}") is None
        assert protocol.parse_message("随机文字") is None
        assert protocol.parse_message('{"method":}') is None

    def test_missing_method_returns_none(self):
        """没有 method 字段的消息视为非法，返回 None。"""
        assert protocol.parse_message('{"jsonrpc":"2.0","params":{}}') is None

    def test_non_object_returns_none(self):
        """顶层数组/数字等非法消息返回 None。"""
        assert protocol.parse_message("[1,2,3]") is None
        assert protocol.parse_message("42") is None

    def test_params_optional(self):
        """params 字段可选（如 stop 可能不传 params）。"""
        line = '{"method":"stop"}'
        msg = protocol.parse_message(line)

        assert msg is not None
        assert msg["method"] == "stop"

    def test_chinese_text_preserved(self):
        """中文字符不被转义破坏（ensure_ascii=False 在出站，入站靠 json.loads）。"""
        line = '{"method":"chat","params":{"text":"我做了一个选课系统"}}'
        msg = protocol.parse_message(line)

        assert msg is not None
        assert msg["params"]["text"] == "我做了一个选课系统"


# ──────────────────────────────────────────────
# notify 系列：出站通知格式
# 用 monkeypatch 截获 stdout 写入，验证格式正确
# ──────────────────────────────────────────────


class TestNotify:
    def test_notify_writes_one_json_line(self, monkeypatch):
        """notify 写一行合法 JSON-RPC。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        # flush 是 stdout 对象的方法，monkeypatch 后需要确保存在
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify("stream", {"session": "s1", "delta": "你"})

        assert len(written) == 1  # 只写了一行（含 \n）
        line = written[0]
        assert line.endswith("\n")  # 必须以换行结尾（分帧约定）
        msg = json.loads(line)
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "stream"
        assert msg["params"] == {"session": "s1", "delta": "你"}

    def test_notify_keeps_chinese_unescaped(self, monkeypatch):
        """中文不被 \\uXXXX 转义（ensure_ascii=False）。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify("stream", {"delta": "你好"})

        assert "你好" in written[0]  # 原样中文，不是 \uXXXX

    def test_notify_calls_flush(self, monkeypatch):
        """每次 notify 后必须 flush（设计第 1.6 节，流式不能攒缓冲）。"""
        flush_count = [0]
        monkeypatch.setattr(protocol.sys.stdout, "write", lambda s: None)
        monkeypatch.setattr(
            protocol.sys.stdout, "flush", lambda: flush_count.__setitem__(0, flush_count[0] + 1)
        )

        protocol.notify("done", {"session": "s1"})

        assert flush_count[0] == 1

    def test_notify_stream(self, monkeypatch):
        """notify_stream 格式正确。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_stream("s1", "你好")

        msg = json.loads(written[0])
        assert msg["method"] == "stream"
        assert msg["params"] == {"session": "s1", "delta": "你好"}

    def test_notify_tool_call_start(self, monkeypatch):
        """工具调用 start 通知（含 args）。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_tool_call("s1", "search_code", "start", args={"keyword": "redis"})

        msg = json.loads(written[0])
        assert msg["method"] == "tool_call"
        assert msg["params"]["tool"] == "search_code"
        assert msg["params"]["phase"] == "start"
        assert msg["params"]["args"] == {"keyword": "redis"}

    def test_notify_tool_call_end(self, monkeypatch):
        """工具调用 end 通知（含 result）。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_tool_call("s1", "search_code", "end", result="找到3处")

        msg = json.loads(written[0])
        assert msg["params"]["phase"] == "end"
        assert msg["params"]["result"] == "找到3处"

    def test_notify_runtime_metric(self, monkeypatch):
        """运行指标通知不携带正文，只携带耗时和统计字段。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_runtime_metric("s1", {
            "runtime": "native",
            "status": "done",
            "model_elapsed_ms": 12,
            "first_delta_ms": 3,
            "total_elapsed_ms": 15,
            "estimated_tokens": 20,
            "error_kind": None,
            "tools": [{"tool": "read_file", "elapsed_ms": 2, "result_chars": 8}],
        })

        msg = json.loads(written[0])
        assert msg["method"] == "runtime_metric"
        assert msg["params"]["session"] == "s1"
        assert msg["params"]["runtime"] == "native"
        assert msg["params"]["tools"][0]["result_chars"] == 8

    def test_notify_agent_event_redacts_context_and_tool_content(self, monkeypatch):
        """agent_event 只输出安全摘要，不泄漏上下文、参数值和工具结果。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_agent_event("s1", {
            "type": "tool_execution_end",
            "event_seq": 3,
            "elapsed_ms": 12,
            "tool_name": "read_file",
            "tool_call_id": "call_1",
            "args": {"path": "secret.env"},
            "result": type("Result", (), {"content": "API_KEY=secret"})(),
            "is_error": False,
            "messages": [{"content": "full prompt secret"}],
        })

        msg = json.loads(written[0])
        assert msg["method"] == "agent_event"
        assert msg["params"]["schema_version"] == 1
        assert msg["params"]["event"] == "tool_execution_end"
        assert msg["params"]["args_keys"] == ["path"]
        assert msg["params"]["result_chars"] == len("API_KEY=secret")
        assert "secret.env" not in written[0]
        assert "API_KEY=secret" not in written[0]
        assert "full prompt secret" not in written[0]

    def test_notify_agent_event_includes_only_compaction_metrics(self, monkeypatch):
        """上下文压缩事件只发送状态和数量统计。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_agent_event("s1", {
            "type": "context_compaction",
            "state": "fallback",
            "before_messages": 8,
            "after_messages": 8,
            "before_tokens": 100,
            "after_tokens": 100,
            "summary": "完整摘要不应输出",
        })

        line = written[0]
        msg = json.loads(line)
        assert msg["params"]["schema_version"] == 1
        assert msg["params"]["state"] == "fallback"
        assert msg["params"]["before_messages"] == 8
        assert "完整摘要不应输出" not in line

    def test_agent_event_rejects_unknown_type_and_non_integer_metrics(self, monkeypatch):
        """事件协议只接受稳定事件和整数指标。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_agent_event("s1", {
            "type": "future_event",
            "event_seq": "1",
        })
        assert written == []

        protocol.notify_agent_event("s1", {
            "type": "agent_start",
            "event_seq": True,
            "elapsed_ms": 3.5,
            "step": 0,
        })
        params = json.loads(written[0])["params"]
        assert params["schema_version"] == 1
        assert "event_seq" not in params
        assert "elapsed_ms" not in params
        assert params["step"] == 0

    def test_notify_done(self, monkeypatch):
        """完成通知。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_done("s1")

        msg = json.loads(written[0])
        assert msg["method"] == "done"
        assert msg["params"] == {"session": "s1"}

    def test_notify_cancelled(self, monkeypatch):
        """停止通知。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_cancelled("s1", "部分回答")

        msg = json.loads(written[0])
        assert msg["method"] == "cancelled"
        assert msg["params"] == {"session": "s1", "partial": "部分回答"}

    def test_notify_error(self, monkeypatch):
        """错误通知。"""
        written = []
        monkeypatch.setattr(protocol.sys.stdout, "write", written.append)
        monkeypatch.setattr(protocol.sys.stdout, "flush", lambda: None)

        protocol.notify_error("s1", "API key 无效")

        msg = json.loads(written[0])
        assert msg["method"] == "error"
        assert msg["params"]["message"] == "API key 无效"


# ──────────────────────────────────────────────
# handle_message：消息分发
# ──────────────────────────────────────────────


class TestHandleMessage:
    def test_dispatches_to_correct_handler(self):
        """method 匹配则调对应 handler。"""
        called = []

        handlers = {
            "init": lambda p: called.append(("init", p)),
            "chat": lambda p: called.append(("chat", p)),
        }

        protocol.handle_message(
            {"method": "chat", "params": {"text": "hi"}},
            handlers,
        )

        assert called == [("chat", {"text": "hi"})]

    def test_passes_params_to_handler(self):
        """params 正确传给 handler。"""
        received = []
        handlers = {"init": lambda p: received.append(p)}

        protocol.handle_message(
            {"method": "init", "params": {"workspace": "/x", "model": "m"}},
            handlers,
        )

        assert received == [{"workspace": "/x", "model": "m"}]

    def test_missing_params_passes_empty_dict(self):
        """没有 params 时 handler 收到空 dict（不崩）。"""
        received = []
        handlers = {"stop": lambda p: received.append(p)}

        protocol.handle_message({"method": "stop"}, handlers)

        assert received == [{}]

    def test_null_params_treated_as_empty(self):
        """params 为 null 时当空 dict 处理。"""
        received = []
        handlers = {"stop": lambda p: received.append(p)}

        protocol.handle_message({"method": "stop", "params": None}, handlers)

        assert received == [{}]

    def test_unknown_method_silent(self):
        """未知 method 静默忽略，不崩、不报错。"""
        handlers = {"init": lambda p: None}

        # 不该抛异常
        protocol.handle_message({"method": "unknown_method", "params": {}}, handlers)

    def test_no_handler_silent(self):
        """handlers 为空 dict 时也不崩。"""
        protocol.handle_message({"method": "chat", "params": {}}, {})
