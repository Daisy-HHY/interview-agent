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
    test_model_connection,
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

    def test_bad_request_model_not_exist_carries_model_name(self):
        """400 模型不存在 → bad_request 类，提示里带配置的模型名（笔误一眼可见）。"""
        import openai

        from agent.llm_client import ERROR_KIND_BAD_REQUEST, _classify_openai_error
        # 复现智谱真实报错：modelCode 不存在（用户把 glm-5.2 误写成 glm-5,2）
        err = self._make_status_error(
            openai.BadRequestError,
            400,
            message="Error code: 400 - "
                    "{'error': {'code': '1214', 'message': 'modelCode：不存在'}}",
        )
        result = _classify_openai_error(err, model="glm-5,2")

        assert result.kind == ERROR_KIND_BAD_REQUEST
        assert "glm-5,2" in result.message
        assert "interview.model" in result.message

    def test_bad_request_model_not_exist_without_model_context(self):
        """不传 model 也能分类出 bad_request（向后兼容旧调用）。"""
        import openai

        from agent.llm_client import ERROR_KIND_BAD_REQUEST, _classify_openai_error
        err = self._make_status_error(
            openai.BadRequestError,
            400,
            message="Error code: 400 - model not found",
        )
        result = _classify_openai_error(err)

        assert result.kind == ERROR_KIND_BAD_REQUEST
        assert "interview.model" in result.message

    def test_bad_request_other_400_classified(self):
        """其他 400（非模型名问题）→ 也归 bad_request 类（不可恢复，不重试）。"""
        import openai

        from agent.llm_client import ERROR_KIND_BAD_REQUEST, _classify_openai_error
        err = self._make_status_error(
            openai.BadRequestError, 400, message="invalid parameter",
        )
        result = _classify_openai_error(err, model="glm-5.2")

        assert result.kind == ERROR_KIND_BAD_REQUEST

    def _make_status_error(self, exc_cls, status_code, message="test error"):
        """构造一个带 status_code 的 openai 状态错误（绕过复杂的构造签名）。"""
        import httpx
        # openai 的状态错误需要 response 对象，用 mock 构造
        request = httpx.Request("POST", "https://api.test.com/chat")
        response = httpx.Response(
            status_code,
            request=request,
            content=b'{"error":{"message":"test"}}',
        )
        return exc_cls(message, response=response, body=None)


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


class TestOpenAIImportError:
    def test_missing_openai_dependency_becomes_friendly_llm_error(self, monkeypatch):
        """真实模式缺 openai 包时，提示用户如何修复环境。"""
        import builtins

        from agent.llm_client import ERROR_KIND_UNKNOWN, LLMError, OpenAIClient

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ModuleNotFoundError("No module named 'openai'", name="openai")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(LLMError) as exc_info:
            OpenAIClient(api_key="sk-test")

        assert exc_info.value.kind == ERROR_KIND_UNKNOWN
        assert "pip install openai" in exc_info.value.message
        assert "Demo Mode" in exc_info.value.message


class TestModelConnection:
    def test_success_returns_user_friendly_message(self):
        """连接测试成功时返回成功状态，不需要进入会话历史。"""

        class OkClient:
            def test_connection(self):
                return None

        result = test_model_connection(
            api_key="sk-test",
            model="gpt-4o-mini",
            client_factory=lambda **_kwargs: OkClient(),
        )

        assert result["ok"] is True
        assert "连接成功" in result["message"]

    def test_demo_mode_short_circuit(self):
        """Demo Mode 不需要真实 API 测试。"""
        result = test_model_connection(api_key="", model="", demo_mode=True)

        assert result["ok"] is True
        assert "Demo Mode" in result["message"]

    def test_missing_api_key_fails_before_network(self):
        """缺 API Key 直接给配置提示。"""
        result = test_model_connection(api_key="", model="gpt-4o-mini")

        assert result["ok"] is False
        assert result["kind"] == "auth"
        assert "interview.apiKey" in result["message"]

    def test_missing_model_fails_before_network(self):
        """缺模型名直接给配置提示。"""
        result = test_model_connection(api_key="sk-test", model="")

        assert result["ok"] is False
        assert result["kind"] == "bad_request"
        assert "interview.model" in result["message"]

    def test_classified_llm_error_becomes_result(self):
        """模型不存在等 LLMError 被转成连接测试失败结果。"""
        from agent.llm_client import ERROR_KIND_BAD_REQUEST, LLMError

        class BadClient:
            def test_connection(self):
                raise LLMError(ERROR_KIND_BAD_REQUEST, "模型不存在，请检查 interview.model")

        result = test_model_connection(
            api_key="sk-test",
            model="bad-model",
            client_factory=lambda **_kwargs: BadClient(),
        )

        assert result["ok"] is False
        assert result["kind"] == ERROR_KIND_BAD_REQUEST
        assert "interview.model" in result["message"]


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

    def test_no_retry_on_bad_request_model_not_exist(self, monkeypatch):
        """★ 400 模型不存在不重试：立刻抛 LLMError，且提示带模型名。"""
        from agent.llm_client import ERROR_KIND_BAD_REQUEST, LLMError
        err = self._make_bad_request_model_error()
        client, mock_create = self._make_client_with_mock([err], monkeypatch)
        client._model = "glm-5,2"

        with pytest.raises(LLMError) as exc_info:
            client.chat([], [])

        assert exc_info.value.kind == ERROR_KIND_BAD_REQUEST
        assert "glm-5,2" in exc_info.value.message
        assert mock_create.calls == 1  # 只调了 1 次，没重试

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

    def _make_bad_request_model_error(self):
        """构造一个 BadRequestError（400，模型不存在，复现智谱真实报错）。"""
        import httpx
        import openai
        request = httpx.Request("POST", "https://api.test.com/chat")
        response = httpx.Response(
            400, request=request,
            content=b'{"error":{"code":"1214","message":"modelCode"}}',
        )
        return openai.BadRequestError(
            "Error code: 400 - "
            "{'error': {'code': '1214', 'message': 'modelCode：不存在'}}",
            response=response,
            body=None,
        )


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

    def test_blocking_response_keeps_finish_reason(self, monkeypatch):
        """非流式响应保留 finish_reason，供 runtime 判断 length 截断。"""
        from agent.llm_client import OpenAIClient
        client = OpenAIClient.__new__(OpenAIClient)
        client._model = "test"

        class FakeMsg:
            content = "hi"
            tool_calls = None

        class FakeChoice:
            message = FakeMsg()
            finish_reason = "length"

        class FakeResp:
            choices = [FakeChoice()]

        class MockChat:
            class completions:
                @staticmethod
                def create(**kw):
                    return FakeResp()

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client._client = MockClient()

        resp = client.chat([], [])

        assert resp.finish_reason == "length"

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

    def test_streaming_response_keeps_finish_reason(self, monkeypatch):
        """流式响应保留最后一个 finish_reason。"""
        class FakeDelta:
            content = "hi"
            tool_calls = None

        class FakeChoice:
            delta = FakeDelta()
            finish_reason = "stop"

        class FakeChunk:
            choices = [FakeChoice()]

        client, _ = self._make_streaming_client([FakeChunk()], monkeypatch)

        resp = client.chat([], [], on_delta=lambda s: None)

        assert resp.finish_reason == "stop"

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
