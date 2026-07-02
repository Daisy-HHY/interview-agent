"""入口测试（设计第 7.2.2 节，第 2 层测试）。

测 main.py 的消息分发和"回调 → 通知"的转换。
用注入 FakeLLM 的 SessionStore + 截获 stdout，不碰真 LLM、不依赖真 stdin。
"""

import io
import json

import agent.main as main_mod
import agent.protocol as protocol
from agent.llm_client import FakeLLM, make_text_response, make_tool_call_response
from agent.session import SessionStore

# ──────────────────────────────────────────────
# 测试辅助：截获 stdout，注入带 FakeLLM 的 store
# ──────────────────────────────────────────────


def run_input(lines, store):
    """喂几行输入给 main，返回捕获的 stdout 通知。

    lines 可以是字符串（一行）或 dict（自动序列化为 JSON 行）。
    """
    # 统一成字符串行
    str_lines = []
    for line in lines:
        if isinstance(line, dict):
            str_lines.append(json.dumps(line, ensure_ascii=False))
        else:
            str_lines.append(line)

    stream = io.StringIO("".join(line + "\n" for line in str_lines))
    output = io.StringIO()

    # 截获 protocol.notify 的 stdout 写入
    original_write = protocol.sys.stdout.write
    original_flush = protocol.sys.stdout.flush
    protocol.sys.stdout.write = output.write
    protocol.sys.stdout.flush = lambda: None
    try:
        main_mod.main(stream=stream, store=store)
    finally:
        protocol.sys.stdout.write = original_write
        protocol.sys.stdout.flush = original_flush

    return [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]


def make_configured_store(tmp_path, script=None):
    """建一个已 init、用 FakeLLM 的 store。"""
    if script is None:
        script = [make_text_response("好的")]
    fake = FakeLLM(script)
    store = SessionStore(llm_factory=lambda: fake)
    store._sessions_dir = str(tmp_path / ".sessions")  # noqa: SLF001
    store.configure(
        workspace=str(tmp_path),
        api_key="sk-test",
        model="gpt-4o-mini",
    )
    return store, fake


# ──────────────────────────────────────────────
# init 处理
# ──────────────────────────────────────────────


class TestInit:
    def test_init_configures_store(self, tmp_path):
        """init 消息正确配置 store。"""
        store = SessionStore()
        store._sessions_dir = str(tmp_path / ".sessions")  # noqa: SLF001

        run_input(
            [{
                "jsonrpc": "2.0", "method": "init",
                "params": {"workspace": str(tmp_path), "api_key": "sk-x", "model": "m"},
            }],
            store,
        )

        assert store.is_configured
        assert store.workspace == str(tmp_path)

    def test_init_missing_params_sends_error(self, tmp_path):
        """init 缺 workspace 或 key 时发 error 通知。"""
        store = SessionStore()
        notifications = run_input(
            [{"method": "init", "params": {"workspace": "/x"}}],  # 缺 key
            store,
        )

        assert any(n["method"] == "error" for n in notifications)
        assert not store.is_configured


# ──────────────────────────────────────────────
# chat 处理：回调转通知
# ──────────────────────────────────────────────


class TestChat:
    def test_chat_direct_answer_emits_stream_then_done(self, tmp_path):
        """直接回答：发 stream 通知（含回答）+ done 通知。"""
        store, _ = make_configured_store(
            tmp_path, [make_text_response("你用了什么数据库？")]
        )

        notifications = run_input(
            [{
                "method": "chat",
                "params": {"session": "s1", "text": "我做了一个选课系统"},
            }],
            store,
        )

        methods = [n["method"] for n in notifications]
        assert "stream" in methods
        assert methods[-1] == "done"  # 最后是 done

        # stream 通知含回答内容
        stream_msg = next(n for n in notifications if n["method"] == "stream")
        assert "数据库" in stream_msg["params"]["delta"]

    def test_chat_tool_calls_emit_tool_call_notifications(self, tmp_path):
        """调工具：发 tool_call 的 start/end 通知。"""
        store, _ = make_configured_store(tmp_path, [
            make_tool_call_response("read_file", {"path": "app.py"}),  # 第 1 轮调工具
            make_text_response("看到代码了"),                            # 第 2 轮回答
        ])

        notifications = run_input(
            [{"method": "chat", "params": {"session": "s1", "text": "看看代码"}}],
            store,
        )

        tool_msgs = [n for n in notifications if n["method"] == "tool_call"]
        # start + end 两个 tool_call 通知
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["params"]["phase"] == "start"
        assert tool_msgs[0]["params"]["tool"] == "read_file"
        assert tool_msgs[1]["params"]["phase"] == "end"

    def test_chat_persists_history(self, tmp_path):
        """chat 后历史落盘（设计第 6.4.3 节）。"""
        store, _ = make_configured_store(
            tmp_path, [make_text_response("回答")]
        )

        run_input(
            [{"method": "chat", "params": {"session": "s1", "text": "问题"}}],
            store,
        )

        assert (tmp_path / ".sessions" / "s1.json").exists()

    def test_chat_before_init_sends_error(self, tmp_path):
        """未 init 就 chat：发 error 通知。"""
        store = SessionStore()
        store._sessions_dir = str(tmp_path / ".sessions")  # noqa: SLF001

        notifications = run_input(
            [{"method": "chat", "params": {"session": "s1", "text": "hi"}}],
            store,
        )

        assert any(n["method"] == "error" for n in notifications)


# ──────────────────────────────────────────────
# 选中代码注入（设计第 5.3.3 节）
# ──────────────────────────────────────────────


class TestAttachedCode:
    def test_attached_code_injected_into_message(self, tmp_path):
        """attached_code 被拼进发给 Agent 的消息。"""
        # FakeLLM 第一轮捕获 messages，看 attached 是否进来
        captured = []
        fake = FakeLLM([make_text_response("ok")])
        original_chat = fake.chat

        def spy_chat(messages, tools, on_delta=None, cancel_event=None):
            captured.append(messages)
            return original_chat(messages, tools)

        fake.chat = spy_chat

        store = SessionStore(llm_factory=lambda: fake)
        store._sessions_dir = str(tmp_path / ".sessions")  # noqa: SLF001
        store.configure(workspace=str(tmp_path), api_key="sk-test")

        run_input([{
            "method": "chat",
            "params": {
                "session": "s1",
                "text": "这段代码怎么样",
                "attached_code": {"file": "db.py", "content": "connect()"},
            },
        }], store)

        # 用户消息里应含选中代码
        user_msgs = [m for m in captured[0] if m.get("role") == "user"]
        assert any("connect()" in m["content"] for m in user_msgs)
        assert any("db.py" in m["content"] for m in user_msgs)

    def test_no_attached_code_uses_plain_text(self, tmp_path):
        """没有 attached_code 时用户消息就是原话。"""
        captured = []
        fake = FakeLLM([make_text_response("ok")])
        original_chat = fake.chat

        def spy_chat(messages, tools, on_delta=None, cancel_event=None):
            captured.append(messages)
            return original_chat(messages, tools)

        fake.chat = spy_chat

        store = SessionStore(llm_factory=lambda: fake)
        store._sessions_dir = str(tmp_path / ".sessions")  # noqa: SLF001
        store.configure(workspace=str(tmp_path), api_key="sk-test")

        run_input([{
            "method": "chat",
            "params": {"session": "s1", "text": "普通问题"},
        }], store)

        user_msgs = [m for m in captured[0] if m.get("role") == "user"]
        assert user_msgs[0]["content"] == "普通问题"


# ──────────────────────────────────────────────
# 容错：脏输入 / 异常不杀进程
# ──────────────────────────────────────────────


class TestRobustness:
    def test_malformed_line_skipped(self, tmp_path):
        """格式错误的行被跳过，不崩、不输出。"""
        store, _ = make_configured_store(tmp_path)
        notifications = run_input(["这不是json", "{bad"], store)
        assert notifications == []

    def test_unknown_method_skipped(self, tmp_path):
        """未知 method 静默忽略。"""
        store, _ = make_configured_store(tmp_path)
        notifications = run_input(
            [{"method": "mystery", "params": {}}], store,
        )
        assert notifications == []

    def test_stop_is_noop(self, tmp_path):
        """stop 消息不崩（MVP 占位）。"""
        store, _ = make_configured_store(tmp_path)
        # 不该抛异常
        run_input([{"method": "stop", "params": {"session": "s1"}}], store)

    def test_empty_input_exits_cleanly(self, tmp_path):
        """空输入正常退出。"""
        store, _ = make_configured_store(tmp_path)
        main_mod.main(stream=io.StringIO(""), store=store)  # 不该抛异常

    def test_handler_exception_sends_error_not_crash(self, tmp_path, monkeypatch):
        """handler 内部抛异常时发 error 通知，不杀进程。"""
        store, _ = make_configured_store(tmp_path)

        # 让 chat handler 抛异常（异步架构下 chat 入队走 _enqueue_chat）
        def boom(params):
            raise RuntimeError("故意的")

        monkeypatch.setitem(
            main_mod.__dict__, "_enqueue_chat",
            lambda chat_queue, get_cancel, params: boom(params),
        )

        # 重新设置 handlers 引用 —— main 里每次都重建 handlers dict，所以 monkeypatch 生效
        notifications = run_input(
            [{"method": "chat", "params": {"session": "s1", "text": "x"}}],
            store,
        )

        # 应该有 error 通知，而不是进程崩溃
        assert any(n["method"] == "error" for n in notifications)
        assert any("故意的" in n["params"]["message"] for n in notifications)


# ──────────────────────────────────────────────
# 多消息序列
# ──────────────────────────────────────────────


class TestMultipleMessages:
    def test_init_then_chat_sequence(self, tmp_path):
        """完整序列：init → chat。"""
        fake = FakeLLM([make_text_response("面试回答")])
        store = SessionStore(llm_factory=lambda: fake)
        store._sessions_dir = str(tmp_path / ".sessions")  # noqa: SLF001

        notifications = run_input([
            {"method": "init", "params": {
                "workspace": str(tmp_path), "api_key": "sk-test"}},
            {"method": "chat", "params": {"session": "s1", "text": "你好"}},
        ], store)

        assert store.is_configured
        methods = [n["method"] for n in notifications]
        assert "stream" in methods
        assert methods[-1] == "done"

    def test_multi_turn_in_same_session(self, tmp_path):
        """同一 session 多轮 chat，历史累积。"""
        fake = FakeLLM([
            make_text_response("第一轮"),
            make_text_response("第二轮"),
        ])
        store = SessionStore(llm_factory=lambda: fake)
        store._sessions_dir = str(tmp_path / ".sessions")  # noqa: SLF001
        store.configure(workspace=str(tmp_path), api_key="sk-test")

        run_input([
            {"method": "chat", "params": {"session": "s1", "text": "问1"}},
            {"method": "chat", "params": {"session": "s1", "text": "问2"}},
        ], store)

        loop = store.get_or_create("s1")
        # 两轮历史累积：system + user1 + asst1 + user2 + asst2
        assert len(loop.messages) == 5


# ──────────────────────────────────────────────
# 异步分发（#8：chat 不阻塞主线程，stop 才能及时到达）
# ──────────────────────────────────────────────


class TestAsyncDispatch:
    """chat 在 worker 线程执行，主线程持续读 stdin（#8 的前提）。

    配合 TestCancel（loop 步骤边界取消）和 TestStreamingCancel（流式 chunk
    取消），覆盖 stop 端到端：主线程读到 stop → set cancel_event → loop 停。
    """

    def test_chat_does_not_block_main_thread(self, tmp_path):
        """chat 慢（worker 阻塞）时，主线程仍能读到后续消息——证明异步分发。"""
        import io
        import json
        import threading

        gate = threading.Event()
        init_seen = threading.Event()

        class GatedLLM:
            def chat(self, messages, tools, on_delta=None, cancel_event=None):
                gate.wait(timeout=5)  # 阻塞 worker，模拟慢 LLM
                return make_text_response("答")

        store = SessionStore(llm_factory=lambda: GatedLLM())
        store._sessions_dir = str(tmp_path / ".sessions")  # noqa: SLF001
        store.configure(workspace=str(tmp_path), api_key="sk-test")

        # 拦截 _handle_init：若它在 chat 完成前被调，说明主线程没被 chat 阻塞
        real_init = main_mod._handle_init

        def spy_init(s, params):
            init_seen.set()
            real_init(s, params)

        main_mod._handle_init = spy_init  # noqa: SLF001
        try:
            chat_line = json.dumps(
                {"method": "chat", "params": {"session": "s1", "text": "问"}},
            )
            init_line = json.dumps({
                "method": "init",
                "params": {"workspace": str(tmp_path), "api_key": "sk-x"},
            })
            stream = io.StringIO(chat_line + "\n" + init_line + "\n")
            output = io.StringIO()
            ow = protocol.sys.stdout.write
            of = protocol.sys.stdout.flush
            protocol.sys.stdout.write = output.write
            protocol.sys.stdout.flush = lambda: None
            try:
                t = threading.Thread(
                    target=main_mod.main,
                    kwargs={"stream": stream, "store": store},
                    daemon=True,
                )
                t.start()
                # 主线程应能在 chat(worker 阻塞) 期间读到后续 init
                assert init_seen.wait(timeout=3), "主线程被 chat 阻塞，未读到后续 init"
                gate.set()  # release worker，让 chat 完成、main 退出
                t.join(timeout=5)
            finally:
                protocol.sys.stdout.write = ow
                protocol.sys.stdout.flush = of
        finally:
            main_mod._handle_init = real_init  # noqa: SLF001

        assert not t.is_alive(), "main 未退出"
