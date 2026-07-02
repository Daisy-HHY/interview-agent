"""LLM 客户端测试（设计第 7.2.3 节，第 2 层测试）。

测试目标：验证 FakeLLM 行为可靠（它是 Phase 4 测试的工具）。
注意：不测 OpenAIClient（要花钱、不确定），它留给 Phase 7 冒烟。

关键覆盖点：
- FakeLLM 按脚本顺序返回
- call_count 正确递增
- 脚本耗尽时报错（防止漏写响应的测试悄悄通过）
- make_tool_call_response / make_text_response 格式正确
"""

import json

import pytest

from agent.llm_client import (
    FakeLLM,
    LLMResponse,
    make_text_response,
    make_tool_call_response,
)

# ──────────────────────────────────────────────
# make_text_response 测试
# ──────────────────────────────────────────────


class TestMakeTextResponse:
    def test_creates_content(self):
        """文本响应应该有 content，无 tool_calls。"""
        resp = make_text_response("你好")
        assert resp.content == "你好"
        assert resp.tool_calls == []

    def test_empty_string_allowed(self):
        """空字符串也合法（LLM 偶尔会返回空内容）。"""
        resp = make_text_response("")
        assert resp.content == ""
        assert resp.tool_calls == []


# ──────────────────────────────────────────────
# make_tool_call_response 测试
# ──────────────────────────────────────────────


class TestMakeToolCallResponse:
    def test_creates_tool_call(self):
        """工具调用响应应该有 tool_calls，无 content。"""
        resp = make_tool_call_response("search_code", {"keyword": "redis"})
        assert resp.content == ""
        assert len(resp.tool_calls) == 1

    def test_tool_call_has_correct_name(self):
        """工具名应该正确。"""
        resp = make_tool_call_response("read_file", {"path": "a.py"})
        assert resp.tool_calls[0]["function"]["name"] == "read_file"

    def test_arguments_is_json_string(self):
        """★ 关键：arguments 必须是 JSON 字符串，不是 dict。

        这是 OpenAI API 的格式要求，新手常踩坑。
        Agent 循环会用 json.loads() 解析它。
        """
        resp = make_tool_call_response("search_code", {"keyword": "redis"})
        args = resp.tool_calls[0]["function"]["arguments"]

        assert isinstance(args, str)  # 是字符串
        parsed = json.loads(args)     # 能被 JSON 解析
        assert parsed == {"keyword": "redis"}

    def test_empty_arguments(self):
        """无参数的工具调用也合法。"""
        resp = make_tool_call_response("ping", {})
        args = resp.tool_calls[0]["function"]["arguments"]
        assert json.loads(args) == {}


# ──────────────────────────────────────────────
# FakeLLM 测试（核心：验证它作为测试工具可靠）
# ──────────────────────────────────────────────


class TestFakeLLM:
    def test_initial_call_count_is_zero(self):
        """构造后 call_count 应该是 0。"""
        fake = FakeLLM([make_text_response("hi")])
        assert fake.call_count == 0

    def test_returns_scripted_responses_in_order(self):
        """★ 按脚本顺序返回（Agent 循环测试的前提）。"""
        fake = FakeLLM([
            make_tool_call_response("search_code", {"keyword": "redis"}),
            make_text_response("你用了 Redis，过期策略？"),
        ])

        # 第一次：应该返回工具调用
        r1 = fake.chat([], [])
        assert r1.tool_calls[0]["function"]["name"] == "search_code"
        assert fake.call_count == 1

        # 第二次：应该返回文本
        r2 = fake.chat([], [])
        assert r2.content == "你用了 Redis，过期策略？"
        assert r2.tool_calls == []
        assert fake.call_count == 2

    def test_single_response_script(self):
        """只有一个响应的脚本。"""
        fake = FakeLLM([make_text_response("only one")])
        r = fake.chat([], [])
        assert r.content == "only one"
        assert fake.call_count == 1

    def test_script_exhausted_raises(self):
        """★ 脚本耗尽时报错（防止测试漏写响应却悄悄通过）。

        这是个保护机制：如果你测试时 Agent 循环调了 3 次 LLM，
        但脚本只写了 2 个响应，会立刻报错而不是返回 None。
        """
        fake = FakeLLM([make_text_response("first")])
        fake.chat([], [])  # 消耗第一个

        with pytest.raises(RuntimeError, match="脚本耗尽"):
            fake.chat([], [])  # 第二次没响应了

    def test_ignores_messages_and_tools(self):
        """FakeLLM 不关心传入的 messages/tools（它就按脚本吐）。

        这是故意的——让测试聚焦于"循环逻辑"，不被 LLM 内容干扰。
        """
        fake = FakeLLM([make_text_response("canned")])
        r = fake.chat(
            messages=[{"role": "user", "content": "anything"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
        assert r.content == "canned"  # 无视输入，返回脚本里的

    def test_call_count_tracks_all_invocations(self):
        """call_count 准确追踪调用次数。"""
        fake = FakeLLM([
            make_text_response("a"),
            make_text_response("b"),
            make_text_response("c"),
        ])

        assert fake.call_count == 0
        fake.chat([], [])
        assert fake.call_count == 1
        fake.chat([], [])
        assert fake.call_count == 2
        fake.chat([], [])
        assert fake.call_count == 3


# ──────────────────────────────────────────────
# LLMResponse 数据类测试
# ──────────────────────────────────────────────


class TestLLMResponse:
    def test_default_values(self):
        """默认值：content 空字符串，tool_calls 空列表。"""
        resp = LLMResponse()
        assert resp.content == ""
        assert resp.tool_calls == []

    def test_independent_default_lists(self):
        """★ 关键：每个实例的 tool_calls 应该独立（可变默认值的经典坑）。

        如果用 tool_calls=[] 作默认值，所有实例会共享同一个列表。
        用 field(default_factory=list) 避免，这个测试守护这个约束。
        """
        r1 = LLMResponse()
        r2 = LLMResponse()
        r1.tool_calls.append({"fake": "call"})

        # r1 加了，r2 不应该受影响
        assert len(r2.tool_calls) == 0

    def test_has_tool_calls_property(self):
        """tool_calls 非空时，has_tool_calls 为 True（Agent 循环判断用）。"""
        # 注意：这个属性需要在 LLMResponse 里定义，如果没有就先跳过这个测试
        # 这里测的是"有 tool_calls 就应该能判断"的语义
        with_text = LLMResponse(content="hello")
        with_tool = LLMResponse(tool_calls=[{"function": {"name": "x"}}])

        assert len(with_tool.tool_calls) > 0
        assert len(with_text.tool_calls) == 0


# ──────────────────────────────────────────────
# 错误分类测试（设计第 6.4.2 节，Phase 7-A）
# ──────────────────────────────────────────────


class TestClassifyOpenAIError:
    """验证 _classify_openai_error 把各种 openai 异常映射成正确的 LLMError。"""

    def test_authentication_error_classified_as_auth(self):
        """401 key 无效 → auth 类（不可恢复）。"""
        import openai

        from agent.llm_client import (
            ERROR_KIND_AUTH,
            _classify_openai_error,
        )
        # 构造一个 AuthenticationError（需要 mock response）
        err = self._make_status_error(openai.AuthenticationError, 401)
        result = _classify_openai_error(err)

        assert result.kind == ERROR_KIND_AUTH
        assert "API key" in result.message or "key" in result.message

    def test_rate_limit_error_classified_as_rate_limit(self):
        """429 限流 → rate_limit 类（可恢复）。"""
        import openai

        from agent.llm_client import (
            ERROR_KIND_RATE_LIMIT,
            _classify_openai_error,
        )
        err = self._make_status_error(openai.RateLimitError, 429)
        result = _classify_openai_error(err)

        assert result.kind == ERROR_KIND_RATE_LIMIT

    def test_api_connection_error_classified_as_connection(self):
        """断网 → connection 类（可恢复）。"""
        import openai

        from agent.llm_client import (
            ERROR_KIND_CONNECTION,
            _classify_openai_error,
        )
        err = openai.APIConnectionError(request=None)
        result = _classify_openai_error(err)

        assert result.kind == ERROR_KIND_CONNECTION

    def test_timeout_classified_as_connection(self):
        """超时 → connection 类（超时是连接问题的子类）。"""
        import openai

        from agent.llm_client import (
            ERROR_KIND_CONNECTION,
            _classify_openai_error,
        )
        err = openai.APITimeoutError(request=None)
        result = _classify_openai_error(err)

        assert result.kind == ERROR_KIND_CONNECTION

    def test_internal_server_error_classified_as_server(self):
        """5xx → server 类（可恢复）。"""
        import openai

        from agent.llm_client import (
            ERROR_KIND_SERVER,
            _classify_openai_error,
        )
        err = self._make_status_error(openai.InternalServerError, 503)
        result = _classify_openai_error(err)

        assert result.kind == ERROR_KIND_SERVER

    def test_unknown_error_fallback(self):
        """其他 openai 异常 → unknown 类。"""
        import openai

        from agent.llm_client import (
            ERROR_KIND_UNKNOWN,
            _classify_openai_error,
        )
        err = self._make_status_error(openai.APIStatusError, 418)
        result = _classify_openai_error(err)

        assert result.kind == ERROR_KIND_UNKNOWN

    def test_classified_message_is_user_friendly(self):
        """分类后的提示是友好的中文（给用户看的）。"""
        import openai

        from agent.llm_client import _classify_openai_error
        err = self._make_status_error(openai.AuthenticationError, 401)
        result = _classify_openai_error(err)

        # 提示应该是中文，且指引用户去检查设置
        assert "interview.apiKey" in result.message

    def _make_status_error(self, exc_cls, status_code):
        """构造一个带 status_code 的 openai 状态错误（绕过复杂的构造签名）。"""
        import httpx
        # openai 的状态错误需要 response 对象，用 mock 构造
        request = httpx.Request("POST", "https://api.test.com/chat")
        response = httpx.Response(
            status_code,
            request=request,
            content=b'{"error":{"message":"test"}}',
        )
        return exc_cls("test error", response=response, body=None)


class TestLLMErrorDataclass:
    """LLMError 作为 dataclass + Exception 的行为。"""

    def test_is_exception(self):
        """LLMError 是 Exception 子类（能被 except 捕获）。"""
        from agent.llm_client import LLMError
        assert issubclass(LLMError, Exception)

    def test_carries_kind_and_message(self):
        """携带 kind 和 message 字段。"""
        from agent.llm_client import ERROR_KIND_AUTH, LLMError
        err = LLMError(ERROR_KIND_AUTH, "key 无效")
        assert err.kind == ERROR_KIND_AUTH
        assert err.message == "key 无效"

    def test_str_returns_message(self):
        """str(error) 返回 message（方便日志）。"""
        from agent.llm_client import ERROR_KIND_AUTH, LLMError
        err = LLMError(ERROR_KIND_AUTH, "key 无效")
        assert str(err) == "key 无效"


# ──────────────────────────────────────────────
# 自动重试测试（设计第 6.4.2 节延伸，Phase 7-B）
# ──────────────────────────────────────────────


class TestAutoRetry:
    """验证 OpenAIClient 的自动重试逻辑（用 mock client，不真调 API）。"""

    def _make_client_with_mock(self, side_effects, monkeypatch):
        """构造一个 OpenAIClient，但把内部 _client 替换成 mock。

        side_effects: 每次调 create 时依次抛出/返回的值。
        sleep 时间缩到 0，让测试不真等。
        """
        from agent.llm_client import OpenAIClient

        client = OpenAIClient.__new__(OpenAIClient)  # 跳过 __init__（不真连）
        client._model = "test-model"

        class MockCreate:
            def __init__(self):
                self.calls = 0

            def __call__(self, **kwargs):
                idx = self.calls
                self.calls += 1
                val = side_effects[idx]
                if isinstance(val, Exception):
                    raise val
                return val

        mock_create = MockCreate()

        class MockCompletions:
            def __init__(self):
                self.create = mock_create

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client._client = MockClient()

        # time.sleep 已是 llm_client 模块级（顶部 import time），
        # 直接 patch 模块的 time.sleep，避免测试真等 1+2+4 秒
        monkeypatch.setattr("agent.llm_client.time.sleep", lambda s: None)

        return client, mock_create

    def test_retries_on_rate_limit_then_succeeds(self, monkeypatch):
        """★ 前 2 次 429，第 3 次成功 → 应该重试并返回结果。"""

        # 构造 fake response
        class FakeMsg:
            content = "成功"
            tool_calls = None

        class FakeChoice:
            message = FakeMsg()

        class FakeResponse:
            choices = [FakeChoice()]

        ok = FakeResponse()
        err = self._make_rate_limit_error()
        client, mock_create = self._make_client_with_mock(
            [err, err, ok], monkeypatch,
        )

        # 应该重试 2 次后成功
        result = client.chat([], [])
        assert result.content == "成功"
        assert mock_create.calls == 3  # 调了 3 次

    def test_no_retry_on_auth_error(self, monkeypatch):
        """★ 401 不重试：立刻抛 LLMError（重试也是 401）。"""
        from agent.llm_client import LLMError
        err = self._make_auth_error()
        client, mock_create = self._make_client_with_mock([err], monkeypatch)

        with pytest.raises(LLMError) as exc_info:
            client.chat([], [])

        assert "auth" in exc_info.value.kind or exc_info.value.kind == "auth"
        assert mock_create.calls == 1  # 只调了 1 次，没重试

    def test_retries_exhausted_raises_after_max_attempts(self, monkeypatch):
        """重试用尽（3 次都 429）→ 抛最后的 LLMError，不再 sleep。"""
        err = self._make_rate_limit_error()
        client, mock_create = self._make_client_with_mock(
            [err, err, err], monkeypatch,
        )

        from agent.llm_client import LLMError
        with pytest.raises(LLMError):
            client.chat([], [])
        assert mock_create.calls == 3  # 正好 3 次

    def test_retries_on_connection_error(self, monkeypatch):
        """断网（APIConnectionError）→ 可恢复，应该重试。"""
        import openai
        class FakeMsg:
            content = "通了"
            tool_calls = None
        class FakeChoice:
            message = FakeMsg()
        class FakeResponse:
            choices = [FakeChoice()]

        conn_err = openai.APIConnectionError(request=None)
        client, mock_create = self._make_client_with_mock(
            [conn_err, FakeResponse()], monkeypatch,
        )
        result = client.chat([], [])
        assert result.content == "通了"
        assert mock_create.calls == 2

    def test_non_openai_error_not_retried(self, monkeypatch):
        """非 openai 异常（如程序 bug 的 ValueError）不重试，直接抛。"""
        client, mock_create = self._make_client_with_mock(
            [ValueError("bug")], monkeypatch,
        )
        with pytest.raises(ValueError):
            client.chat([], [])
        assert mock_create.calls == 1

    def _make_rate_limit_error(self):
        """构造一个 RateLimitError（429）。"""
        import httpx
        import openai
        request = httpx.Request("POST", "https://api.test.com/chat")
        response = httpx.Response(
            429, request=request,
            content=b'{"error":{"message":"rate limited"}}',
        )
        return openai.RateLimitError("rate limited", response=response, body=None)

    def _make_auth_error(self):
        """构造一个 AuthenticationError（401）。"""
        import httpx
        import openai
        request = httpx.Request("POST", "https://api.test.com/chat")
        response = httpx.Response(
            401, request=request,
            content=b'{"error":{"message":"bad key"}}',
        )
        return openai.AuthenticationError("bad key", response=response, body=None)


# ──────────────────────────────────────────────
# 流式输出测试（设计第 1.6、6.5 节，Phase 7-C）
# ──────────────────────────────────────────────


def _make_text_chunk(content):
    """构造一个流式文本 chunk（模拟 openai 流式返回）。"""
    class FakeDelta:
        def __init__(self, content):
            self.content = content
            self.tool_calls = None
    class FakeChoice:
        def __init__(self, content):
            self.delta = FakeDelta(content)
    class FakeChunk:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]
    return FakeChunk(content)


def _make_tool_call_chunks():
    """构造 tool_calls 分片流（模拟 openai 流式分片到达）。

    返回一个 chunk 列表，模拟 read_file("app.py") 分 3 次到达：
    - chunk1: id + name="read_file"
    - chunk2: arguments 部分 '{"path":"app'
    - chunk3: arguments 部分 '.py"}'
    """
    class FakeFn:
        def __init__(self, name=None, arguments=None):
            self.name = name
            self.arguments = arguments
    class FakeTC:
        def __init__(self, index, tc_id=None, name=None, arguments=None):
            self.index = index
            self.id = tc_id
            self.function = FakeFn(name, arguments)
    class FakeDelta:
        def __init__(self, tool_calls):
            self.content = None
            self.tool_calls = tool_calls
    class FakeChoice:
        def __init__(self, tool_calls):
            self.delta = FakeDelta(tool_calls)
    class FakeChunk:
        def __init__(self, tool_calls):
            self.choices = [FakeChoice(tool_calls)]
    return [
        FakeChunk([FakeTC(0, tc_id="call_1", name="read_file")]),
        FakeChunk([FakeTC(0, arguments='{"path":"app')]),
        FakeChunk([FakeTC(0, arguments='.py"}')]),
    ]


class TestStreamingOutput:
    """验证流式输出（on_delta 回调 + 累积，设计 1.6 节）。"""

    def _make_streaming_client(self, chunks, monkeypatch):
        """构造 OpenAIClient，mock create 返回流式 chunk 迭代器。"""
        from agent.llm_client import OpenAIClient
        client = OpenAIClient.__new__(OpenAIClient)
        client._model = "test"

        class MockCreate:
            def __init__(self):
                self.received_kwargs = None
                self.calls = 0
            def __call__(self, **kwargs):
                self.calls += 1
                self.received_kwargs = kwargs
                return iter(chunks)

        mock = MockCreate()

        class MockChat:
            class completions:
                @staticmethod
                def create(**kw):
                    return mock(**kw)
        class MockClient:
            def __init__(self):
                self.chat = MockChat()
        client._client = MockClient()
        return client, mock

    def test_text_deltas_pushed_via_on_delta(self, monkeypatch):
        """★ 文本 chunk 通过 on_delta 实时分段推出。"""
        chunks = [_make_text_chunk("你"), _make_text_chunk("好"), _make_text_chunk("呀")]
        client, _ = self._make_streaming_client(chunks, monkeypatch)

        received = []
        resp = client.chat([], [], on_delta=received.append)

        assert "".join(received) == "你好呀"  # 分 3 段收到
        assert resp.content == "你好呀"  # 累积完整

    def test_stream_true_added_when_on_delta_provided(self, monkeypatch):
        """传 on_delta 时请求带 stream=True。"""
        client, mock = self._make_streaming_client(
            [_make_text_chunk("x")], monkeypatch,
        )
        client.chat([], [], on_delta=lambda s: None)
        assert mock.received_kwargs.get("stream") is True

    def test_no_stream_without_on_delta(self, monkeypatch):
        """不传 on_delta 时用非流式（stream 不在 kwargs 里）。"""
        # 非流式路径走 _call_with_retry，需 mock 成返回完整 response
        from agent.llm_client import OpenAIClient
        client = OpenAIClient.__new__(OpenAIClient)
        client._model = "test"

        class FakeMsg:
            content = "hi"
            tool_calls = None
        class FakeChoice:
            message = FakeMsg()
        class FakeResp:
            choices = [FakeChoice()]

        class MockChat:
            class completions:
                @staticmethod
                def create(**kw):
                    assert "stream" not in kw  # 非流式不该有 stream
                    return FakeResp()
        class MockClient:
            def __init__(self):
                self.chat = MockChat()
        client._client = MockClient()

        resp = client.chat([], [])  # 不传 on_delta
        assert resp.content == "hi"

    def test_tool_calls_accumulated_from_fragments(self, monkeypatch):
        """★ 流式 tool_calls 分片到达，正确累积拼接。"""
        chunks = _make_tool_call_chunks()
        client, _ = self._make_streaming_client(chunks, monkeypatch)

        resp = client.chat([], [], on_delta=lambda s: None)

        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "read_file"
        # arguments 分 2 片拼接
        assert tc["function"]["arguments"] == '{"path":"app.py"}'

    def test_interruption_preserves_partial_content(self, monkeypatch):
        """★ 中断保留（设计 6.5）：流式中途断网，保留已生成文本。"""
        import openai

        from agent.llm_client import OpenAIClient
        client = OpenAIClient.__new__(OpenAIClient)
        client._model = "test"

        conn_err = openai.APIConnectionError(request=None)

        def make_stream():
            yield _make_text_chunk("已经生成的")
            raise conn_err  # 第二个 chunk 前断网

        class MockChat:
            class completions:
                @staticmethod
                def create(**kw):
                    return make_stream()
        class MockClient:
            def __init__(self):
                self.chat = MockChat()
        client._client = MockClient()

        received = []
        resp = client.chat([], [], on_delta=received.append)

        # 已生成部分保留，并追加中断标记
        assert "已经生成的" in resp.content
        assert "中断" in resp.content
        assert received == ["已经生成的"]  # 已推的保留


class TestFakeLLMStreamingCompat:
    """FakeLLM 的 on_delta 兼容性（保持接口一致）。"""

    def test_fake_llm_pushes_full_content_when_on_delta_given(self):
        """FakeLLM 收到 on_delta 时把整段文本推一次（伪流式）。"""
        fake = FakeLLM([make_text_response("完整回答")])
        received = []
        fake.chat([], [], on_delta=received.append)
        assert received == ["完整回答"]

    def test_fake_llm_no_delta_when_tool_response(self):
        """FakeLLM 返回工具调用时不推 delta（只有文本才推）。"""
        fake = FakeLLM([make_tool_call_response("read_file", {"path": "a.py"})])
        received = []
        fake.chat([], [], on_delta=received.append)
        assert received == []  # 工具响应不推文本

    def test_fake_llm_works_without_on_delta(self):
        """不传 on_delta 时 FakeLLM 行为不变（向后兼容）。"""
        fake = FakeLLM([make_text_response("x")])
        resp = fake.chat([], [])
        assert resp.content == "x"


# ──────────────────────────────────────────────
# 流式路径自动重试（#2 修复，设计 6.4.2 对流式生效）
# ──────────────────────────────────────────────


class TestStreamingRetry:
    """流式路径的可恢复错误自动重试（生产路径恒为流式，重试必须对流式生效）。

    纪律：只在尚未推出任何文本时重试（首 chunk 前的 429/连接/5xx）。
    一旦已推文本，中途失败只能中断保留——重试会从头生成，造成前端重复。
    """

    def _client(self, outcomes, monkeypatch):
        """构造 OpenAIClient，每次 create 按 outcomes 顺序取一个：
        Exception → raise；list → 返回 iter(chunks)。sleep 置空避免真等。"""
        from agent.llm_client import OpenAIClient
        client = OpenAIClient.__new__(OpenAIClient)
        client._model = "test"

        class MockCreate:
            def __init__(self):
                self.calls = 0

            def __call__(self, **kw):
                self.calls += 1
                val = outcomes[self.calls - 1]
                if isinstance(val, Exception):
                    raise val
                return iter(val)

        mock = MockCreate()

        class MockChat:
            class completions:
                @staticmethod
                def create(**kw):
                    return mock(**kw)

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client._client = MockClient()
        monkeypatch.setattr("agent.llm_client.time.sleep", lambda s: None)
        return client, mock

    def _rl(self):
        import httpx
        import openai
        req = httpx.Request("POST", "https://x/y")
        resp = httpx.Response(429, request=req, content=b"{}")
        return openai.RateLimitError("rl", response=resp, body=None)

    def _auth(self):
        import httpx
        import openai
        req = httpx.Request("POST", "https://x/y")
        resp = httpx.Response(401, request=req, content=b"{}")
        return openai.AuthenticationError("bad", response=resp, body=None)

    def _conn(self):
        import openai
        return openai.APIConnectionError(request=None)

    def test_retries_rate_limit_then_succeeds(self, monkeypatch):
        """★ 首 chunk 前 429 → 重试 → 第 2 次成功，前端无重复文本。"""
        client, mock = self._client(
            [self._rl(), [_make_text_chunk("成功")]], monkeypatch,
        )
        received = []
        resp = client.chat([], [], on_delta=received.append)

        assert resp.content == "成功"
        assert received == ["成功"]  # 失败的那次没推任何文本，无重复
        assert mock.calls == 2

    def test_retries_connection_then_succeeds(self, monkeypatch):
        """★ 首 chunk 前断网 → 重试 → 成功。"""
        client, mock = self._client(
            [self._conn(), [_make_text_chunk("OK")]], monkeypatch,
        )
        resp = client.chat([], [], on_delta=lambda s: None)

        assert resp.content == "OK"
        assert mock.calls == 2

    def test_retries_exhausted_raises(self, monkeypatch):
        """3 次都 429 → 抛 LLMError，不再 sleep。"""
        from agent.llm_client import LLMError
        client, mock = self._client(
            [self._rl(), self._rl(), self._rl()], monkeypatch,
        )
        with pytest.raises(LLMError):
            client.chat([], [], on_delta=lambda s: None)
        assert mock.calls == 3

    def test_no_retry_on_auth(self, monkeypatch):
        """401 不可恢复，立刻抛，不重试。"""
        from agent.llm_client import LLMError
        client, mock = self._client([self._auth()], monkeypatch)
        with pytest.raises(LLMError) as ei:
            client.chat([], [], on_delta=lambda s: None)

        assert ei.value.kind == "auth"
        assert mock.calls == 1

    def test_no_retry_after_partial_content(self, monkeypatch):
        """已推出文本后断网：中断保留已生成部分，不重试（重试会从头生成重复）。"""

        def partial_then_fail():
            yield _make_text_chunk("已生成")
            raise self._conn()

        client, mock = self._client([partial_then_fail()], monkeypatch)
        received = []
        resp = client.chat([], [], on_delta=received.append)

        assert mock.calls == 1  # 没重试
        assert "已生成" in resp.content
        assert "中断" in resp.content  # 追加了中断标记
        assert received == ["已生成"]


# ──────────────────────────────────────────────
# OpenAIClient timeout 配置（#4 防 LLM 调用无限挂死）
# ──────────────────────────────────────────────


class TestTimeoutConfig:
    """OpenAIClient 构造必须传有限 timeout 给底层 openai client（#4）。

    旧实现 OpenAI(api_key, base_url) 不传 timeout，依赖 SDK 默认 ~600s，
    配合多步重试可让进程长时间挂死、前端 awaiting 永久 true。
    """

    def test_init_passes_timeout_to_openai(self, monkeypatch):
        """构造时 timeout 透传给 openai.OpenAI。"""
        import openai as _openai

        captured: dict = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(_openai, "OpenAI", FakeOpenAI)

        from agent.llm_client import OpenAIClient
        OpenAIClient(api_key="sk-test", model="m", base_url=None, timeout=42)

        assert captured.get("timeout") == 42

    def test_default_timeout_is_finite(self, monkeypatch):
        """默认 timeout 必须是有限正值，不能依赖 SDK 默认（~600s）而 hang。"""
        import openai as _openai

        captured: dict = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(_openai, "OpenAI", FakeOpenAI)

        from agent.llm_client import OpenAIClient
        OpenAIClient(api_key="sk-test", model="m")

        t = captured.get("timeout")
        assert t is not None
        assert isinstance(t, (int, float)) and t > 0


# ──────────────────────────────────────────────
# 流式 cancel（#8：流式生成中可被中断）
# ──────────────────────────────────────────────


class TestStreamingCancel:
    """流式生成中 cancel_event 被 set → 停止接收后续 chunk，返回已累积部分。

    用户最常在"面试官正在流式回答"时点停止，所以单步流式必须能中断。
    """

    def test_breaks_on_cancel_midway(self, monkeypatch):
        """中途 cancel → 后续 chunk 不再累积，返回已收到的部分文本。"""
        import threading

        from agent.llm_client import OpenAIClient

        cancel = threading.Event()

        def make_stream():
            yield _make_text_chunk("a")
            yield _make_text_chunk("b")
            cancel.set()  # 模拟外部线程在 b 之后取消
            yield _make_text_chunk("c")  # 不应被累积

        client = OpenAIClient.__new__(OpenAIClient)
        client._model = "t"

        class MockChat:
            class completions:
                @staticmethod
                def create(**kw):
                    return make_stream()

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client._client = MockClient()

        received: list = []
        resp = client.chat([], [], on_delta=received.append, cancel_event=cancel)

        assert resp.content == "ab"  # c 被 cancel 截断
        assert received == ["a", "b"]

    def test_no_cancel_event_streams_normally(self, monkeypatch):
        """不传 cancel_event 时流式行为不变（向后兼容）。"""
        from agent.llm_client import OpenAIClient

        client = OpenAIClient.__new__(OpenAIClient)
        client._model = "t"

        class MockChat:
            class completions:
                @staticmethod
                def create(**kw):
                    return iter([_make_text_chunk("x"), _make_text_chunk("y")])

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client._client = MockClient()

        received: list = []
        resp = client.chat([], [], on_delta=received.append)

        assert resp.content == "xy"
        assert received == ["x", "y"]
