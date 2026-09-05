"""LLM 客户端层（设计第 7.2.3 节）。

定义统一的 LLMClient 协议，让 Agent 循环不关心是真实 API 还是 FakeLLM。
测试用 FakeLLM（零费用、确定），真实运行用 OpenAIClient。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


def _strip_surrogates(text: str) -> str:
    """清除字符串里的孤立代理项。

    Windows 文件名/文件内容可能含 \\udcaa 等代理项字符，UTF-8 编不出，
    openai 库序列化请求时会抛 UnicodeEncodeError。
    用 surrogatepass 编成原始字节，再 ignore 解码丢弃。
    """
    try:
        return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _clean_surrogates(obj: Any) -> Any:
    """递归清理数据结构里所有字符串的孤立代理项。

    对 dict / list / tuple 递归深入，对 str 清理，其他类型原样返回。
    这样 messages + tools 里无论坏字符藏多深（嵌套的 function.arguments 字符串等）
    都能在发给 openai 前清干净。
    """
    if isinstance(obj, str):
        return _strip_surrogates(obj)
    if isinstance(obj, dict):
        return {k: _clean_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_surrogates(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_clean_surrogates(item) for item in obj)
    return obj


@dataclass
class LLMResponse:
    """LLM 一次调用的结构化响应。

    content:     LLM 输出的文本（可能为空，当它选择调工具时）
    tool_calls:  LLM 要调的工具列表（可能为空，当它直接回答时）
    finish_reason: 模型停止原因，用于识别 length 截断等特殊场景
    """
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


# ──────────────────────────────────────────────
# 错误分类（设计第 6.4.2 节）
# ──────────────────────────────────────────────

# 错误类型：决定给用户什么提示、是否重试
# rate_limit / auth = 用户可感知原因；connection / server = 临时故障可重试
ERROR_KIND_RATE_LIMIT = "rate_limit"      # 429：请求太频繁 / 余额不足
ERROR_KIND_AUTH = "auth"                  # 401：key 无效（不可恢复，重试无意义）
ERROR_KIND_CONNECTION = "connection"      # 断网 / 超时 / DNS（临时，可重试）
ERROR_KIND_SERVER = "server"              # 5xx：服务端临时故障（可重试）
ERROR_KIND_BAD_REQUEST = "bad_request"    # 400：模型名/参数错误（不可恢复）
ERROR_KIND_UNKNOWN = "unknown"


@dataclass
class LLMError(Exception):
    """LLM 调用失败的统一异常，携带错误类型和友好提示。

    把 openai 的各种异常归一成 kind + message，让上层（main.py）能据此
    给用户不同的提示（设计第 6.4.2 节），重试逻辑也能据此判断是否重试。
    """
    kind: str
    message: str

    def __post_init__(self) -> None:
        # dataclass + Exception：需要手动调 super().__init__ 让 message 成为异常信息
        super().__init__(self.message)


class AgentCancelled(Exception):
    """当前生成被用户停止，partial 保存已经生成的文本。"""

    def __init__(self, partial: str = "") -> None:
        super().__init__("生成已停止")
        self.partial = partial


def _classify_openai_error(e: Exception, model: str | None = None) -> LLMError:
    """把 openai 异常映射成 LLMError（设计第 6.4.2 节）。

    区分可恢复（rate_limit/connection/server，可重试）和不可恢复（auth，让用户改 key）。
    model（可选）：当前配置的模型名，bad_request 时拼进提示，方便用户看出拼写错误。
    """
    # 延迟导入 openai 异常类：测试环境可能用注入的 mock，不强依赖 openai 装载
    import openai as _openai

    # 顺序敏感：子类要在父类前判断（APITimeoutError 是 APIConnectionError 子类）
    if isinstance(e, _openai.AuthenticationError):
        return LLMError(
            ERROR_KIND_AUTH,
            "API key 无效或余额不足。请在设置里检查 interview.apiKey 和余额。",
        )
    if isinstance(e, _openai.RateLimitError):
        return LLMError(
            ERROR_KIND_RATE_LIMIT,
            "请求太频繁或余额不足。请稍等几秒再试，或检查账户额度。",
        )
    if isinstance(e, (_openai.APITimeoutError, _openai.APIConnectionError)):
        return LLMError(
            ERROR_KIND_CONNECTION,
            "网络连接失败或超时。请检查网络或 interview.baseUrl 配置。",
        )
    if isinstance(e, _openai.InternalServerError):
        return LLMError(
            ERROR_KIND_SERVER,
            "服务端临时故障（5xx）。请稍后重试。",
        )
    if isinstance(e, _openai.APIStatusError) and e.status_code == 400:
        text = str(e)
        lowered = text.lower()
        model_hint = f"「{model}」" if model else ""
        if (
            "modelcode" in lowered
            or ("model" in lowered and (
                "不存在" in text
                or "not found" in lowered
                or "not exist" in lowered
                or "does not exist" in lowered
                or "invalid model" in lowered
            ))
        ):
            return LLMError(
                ERROR_KIND_BAD_REQUEST,
                f"模型配置错误：模型{model_hint}在当前 Base URL 对应的服务中不存在。"
                "请检查 interview.model 是否拼写正确、与 interview.baseUrl 的服务商匹配"
                "（常见笔误：glm-5,2 应为 glm-5.2）。",
            )
        return LLMError(
            ERROR_KIND_BAD_REQUEST,
            f"请求参数错误：模型服务拒绝了本次请求。"
            f"请检查 interview.model{model_hint}与 interview.baseUrl 是否匹配。",
        )
    # 其他 openai 异常（APIError、APIStatusError 等）
    return LLMError(
        ERROR_KIND_UNKNOWN,
        f"调用失败: {type(e).__name__}: {e}",
    )


class LLMClient(Protocol):
    """统一的 LLM 客户端接口（鸭子类型）。

    Agent 循环只认这个接口，不关心是 OpenAIClient 还是 FakeLLM。
    """

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        on_delta: Any = None,  # 可选：流式文本回调 Callable[[str], None]
        should_cancel: Any = None,
    ) -> LLMResponse:
        """调一次 LLM，返回结构化响应。

        on_delta（可选）：流式输出时，每收到一段文本就调一次回调
        （设计第 1.6 节打字效果）。非流式实现（FakeLLM）可忽略此参数。
        """
        ...


class FakeLLM:
    """假 LLM，按预设脚本返回响应（设计第 7.2.3 节）。

    用法：构造时传一个响应列表，每次 chat() 消耗一个。
    测试时能精确控制 LLM"说什么"，零费用、完全确定。

    示例：
        fake = FakeLLM([
            make_tool_call_response("search_code", {"keyword": "redis"}),
            make_text_response("你用了 Redis，过期策略是什么？"),
        ])
        # 第一次 chat() 返回调工具，第二次返回回答
    """

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = script
        self._index = 0
        self.call_count = 0  # 测试用：验证 Agent 循环调了几次

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        on_delta: Any = None,
        should_cancel: Any = None,
    ) -> LLMResponse:
        if should_cancel is not None and should_cancel():
            raise AgentCancelled()
        self.call_count += 1
        if self._index >= len(self._script):
            raise RuntimeError(
                f"FakeLLM 脚本耗尽：第 {self.call_count} 次调用，"
                f"但脚本只有 {len(self._script)} 个响应"
            )
        response = self._script[self._index]
        self._index += 1
        # 伪流式：传了 on_delta 且是文本响应时，整段推一次（FakeLLM 不真分段，
        # 但保持接口兼容，让流式链路在测试里也能走通）
        if on_delta is not None and response.content and not response.tool_calls:
            on_delta(response.content)
            if should_cancel is not None and should_cancel():
                raise AgentCancelled(response.content)
        return response
    

def make_tool_call_response(
    tool_name: str,
    arguments: dict[str, Any],
) -> LLMResponse:
    """构造一个"调用工具"的响应（供 FakeLLM 脚本用）。

    格式模仿 OpenAI API 返回的 tool_calls 结构。
    """
    import json
    return LLMResponse(
        tool_calls=[{
            "id": f"call_{tool_name}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments),
            },
        }]
    )


def make_text_response(text: str) -> LLMResponse:
    """构造一个"直接回答文本"的响应（供 FakeLLM 脚本用）。"""
    return LLMResponse(content=text)



class OpenAIClient:
    """真实 OpenAI 兼容 API 客户端（设计第 7.2.3 节）。

    MVP 阶段用非流式（简单可靠）。Phase 7 联调时再加流式输出。
    支持 OpenAI 兼容的 API（DeepSeek、Moonshot 等都行）。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        # 延迟导入：只在真实使用时才 import openai
        # 这样测试代码 import llm_client 时不会强依赖 openai 库
        try:
            from openai import OpenAI
        except ModuleNotFoundError as e:
            if e.name != "openai":
                raise
            raise LLMError(
                ERROR_KIND_UNKNOWN,
                "当前 Python 环境未安装 openai 依赖。\n"
                "解决方法任选其一：\n"
                "1. 在设置 interview.pythonPath 里填写目标项目 venv 的 python 完整路径\n"
                "2. 用当前解释器执行：pip install openai\n"
                "3. 勾选 Demo Mode 体验完整流程（不需要 API 和依赖）",
            ) from e
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.usage_totals: dict[str, int | None] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 0,
            "missing_usage_calls": 0,
        }

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        on_delta: Any = None,
        should_cancel: Any = None,
    ) -> LLMResponse:
        if should_cancel is not None and should_cancel():
            raise AgentCancelled()
        # 终极防御：清掉所有数据里的孤立代理项。
        # Windows 文件名/文件内容可能含 \udcaa 等代理项，openai 序列化请求
        # body 时会抛 UnicodeEncodeError。在调用前递归清理整个数据结构。
        messages = _clean_surrogates(messages)
        tools = _clean_surrogates(tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:  # 没工具时不传 tools 参数（有些模型会报错）
            kwargs["tools"] = tools

        # 流式 vs 非流式（设计第 1.6、6.5 节）：
        # 传了 on_delta 回调 → 用 stream=True，边收边推 delta（打字效果）
        # 没传 → 非流式（简单，FakeLLM/测试默认路径）
        if on_delta is not None:
            response = self._chat_streaming(kwargs, on_delta, should_cancel=should_cancel)
        else:
            if should_cancel is not None and should_cancel():
                raise AgentCancelled()
            response = self._chat_blocking(kwargs)
        self._record_usage(response.usage)
        return response

    def _record_usage(self, usage: dict[str, int] | None) -> None:
        """累计当前客户端实际用量；任一调用缺失时不推算总费用。"""
        totals = getattr(self, "usage_totals", {
            "input_tokens": 0, "output_tokens": 0, "model_calls": 0, "missing_usage_calls": 0,
        })
        totals["model_calls"] += 1
        if usage is None:
            totals["missing_usage_calls"] += 1
            totals["input_tokens"] = totals["output_tokens"] = None
        elif not totals["missing_usage_calls"]:
            totals["input_tokens"] += usage["input_tokens"]
            totals["output_tokens"] += usage["output_tokens"]
        self.usage_totals = totals

    @staticmethod
    def _usage(value: Any) -> dict[str, int] | None:
        """仅接受服务返回的非负整数用量。"""
        inputs = getattr(value, "prompt_tokens", None)
        outputs = getattr(value, "completion_tokens", None)
        if type(inputs) is int and type(outputs) is int and inputs >= 0 and outputs >= 0:
            return {"input_tokens": inputs, "output_tokens": outputs}
        return None

    def _chat_blocking(self, kwargs: dict[str, Any]) -> LLMResponse:
        """非流式调用（默认路径，FakeLLM/单测用）。"""
        response = self._call_with_retry(kwargs)
        choice = response.choices[0]
        llm_response = self._build_response(choice.message)
        llm_response.finish_reason = getattr(choice, "finish_reason", None)
        llm_response.usage = self._usage(getattr(response, "usage", None))
        return llm_response

    def test_connection(self) -> None:
        """发起一次极短请求，用于验证 API Key、Base URL 和模型名。"""
        self._call_with_retry({
            "model": self._model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
        })

    def _chat_streaming(
        self, kwargs: dict[str, Any], on_delta: Any, should_cancel: Any = None,
    ) -> LLMResponse:
        """流式调用（设计第 1.6、6.5 节）。

        - 边收边把文本 delta 通过 on_delta 推出去（打字效果）
        - tool_calls 分片到达，累积拼接（openai 流式的 arguments 是分段的）
        - 中断保留（设计 6.5）：网络断了保留已收部分，追加错误标记，不丢文字

        返回累积后的完整 LLMResponse（和 _chat_blocking 相同接口）。
        """
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        accumulated_content = ""
        finish_reason = None
        usage = None
        stream = None
        # tool_calls 分片累积：{index: {id, name, arguments_parts}}
        tool_calls_acc: dict[int, dict] = {}

        try:
            try:
                stream = self._client.chat.completions.create(**kwargs)
            except Exception as e:
                # 仅在服务明确不接受该可选字段、且还未开始流式输出时降级。
                if getattr(e, "status_code", None) != 400 or "stream_options" not in str(e):
                    raise
                kwargs.pop("stream_options")
                stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                if should_cancel is not None and should_cancel():
                    raise AgentCancelled(accumulated_content)
                usage = self._usage(getattr(chunk, "usage", None)) or usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = choice.delta

                # 文本 delta：累积 + 实时推出
                if delta.content:
                    accumulated_content += delta.content
                    try:
                        on_delta(delta.content)
                    except Exception:
                        # 回调失败不影响主流程
                        pass
                    if should_cancel is not None and should_cancel():
                        raise AgentCancelled(accumulated_content)

                # tool_calls 分片：按 index 累积
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        slot = tool_calls_acc[idx]
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments

        except Exception as e:
            if isinstance(e, AgentCancelled):
                raise
            # 中断保留（设计第 6.5 节）：流式中途出错（断网等）
            # 保留已生成的文本，追加错误标记，让用户知道生成被中断
            import openai as _openai
            if isinstance(e, _openai.APIError):
                err = _classify_openai_error(e, model=self._model)
                # 可恢复错误且已有内容：保留 + 标记中断
                if accumulated_content and err.kind in (
                    ERROR_KIND_CONNECTION, ERROR_KIND_SERVER, ERROR_KIND_RATE_LIMIT,
                ):
                    return LLMResponse(
                        content=accumulated_content
                        + "\n\n⚠️ 生成中断（"
                        + err.message
                        + "）。已生成部分保留，可重新发送。",
                        finish_reason="interrupted",
                    )
                # 不可恢复或无内容：抛错（走外层错误处理）
                raise err
            raise
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        # 组装 tool_calls（按 index 排序）
        tool_calls = []
        for idx in sorted(tool_calls_acc):
            slot = tool_calls_acc[idx]
            tool_calls.append({
                "id": slot["id"],
                "type": "function",
                "function": {
                    "name": slot["name"],
                    "arguments": slot["arguments"],
                },
            })

        return LLMResponse(
            content=accumulated_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _build_response(message: Any) -> LLMResponse:
        """从 openai message 对象构造 LLMResponse（非流式路径用）。"""
        content = message.content or ""
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
        return LLMResponse(content=content, tool_calls=tool_calls)

    def _call_with_retry(self, kwargs: dict[str, Any]):
        """带自动重试的 LLM 调用（设计第 6.4.2 节延伸）。

        策略：
        - 可恢复错误（rate_limit / connection / server）：重试，最多 3 次
        - 不可恢复错误（auth）：立刻抛 LLMError，不重试
        - 指数退避：1s → 2s → 4s
        - 重试用尽后仍失败：抛最后一次的 LLMError（已分类）

        返回原始 openai response 对象（chat 方法再解析）。
        """
        max_attempts = 3
        delay = 1.0

        last_error: LLMError | None = None
        for attempt in range(max_attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                # 用延迟导入避免 openai 未安装时（纯 FakeLLM 测试）报错
                import openai as _openai
                # 非 openai 异常（如程序 bug）：不重试，直接抛
                if not isinstance(e, _openai.APIError):
                    raise

                last_error = _classify_openai_error(e, model=self._model)

                # 不可恢复错误：重试不会改变结果
                if last_error.kind in (ERROR_KIND_AUTH, ERROR_KIND_BAD_REQUEST):
                    raise last_error

                # 可恢复错误：最后一次也直接抛（不再 sleep）
                if attempt == max_attempts - 1:
                    raise last_error

                # 中间次失败：指数退避后重试
                time.sleep(delay)
                delay *= 2

        # 理论上不会到这（上面 return 或 raise），保险起见
        assert last_error is not None
        raise last_error


def test_model_connection(
    api_key: str,
    model: str,
    base_url: str | None = None,
    demo_mode: bool = False,
    client_factory: Any = None,
) -> dict[str, Any]:
    """测试当前模型配置，不创建会话、不写历史。"""
    if demo_mode:
        return {
            "ok": True,
            "kind": "demo",
            "message": "Demo Mode 使用内置 FakeLLM，无需测试真实模型连接。",
        }
    if not api_key.strip():
        return {
            "ok": False,
            "kind": ERROR_KIND_AUTH,
            "message": "还未配置 API Key。请在设置里填写 interview.apiKey，或开启 Demo Mode。",
        }
    if not model.strip():
        return {
            "ok": False,
            "kind": ERROR_KIND_BAD_REQUEST,
            "message": "还未配置模型名。请填写 interview.model 后再测试连接。",
        }

    factory = client_factory or OpenAIClient
    try:
        client = factory(api_key=api_key, model=model.strip(), base_url=base_url or None)
        client.test_connection()
    except LLMError as e:
        return {"ok": False, "kind": e.kind, "message": e.message}
    except Exception as e:
        return {
            "ok": False,
            "kind": ERROR_KIND_UNKNOWN,
            "message": f"模型连接测试失败：{type(e).__name__}: {e}",
        }

    return {
        "ok": True,
        "kind": "ok",
        "message": "模型连接成功。当前 API Key、Base URL 和模型名可用。",
    }


test_model_connection.__test__ = False
