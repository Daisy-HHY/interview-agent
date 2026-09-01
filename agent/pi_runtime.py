"""Pi-style Agent runtime.

这个 runtime 借鉴 Pi 的事件流和上下文外置结构，但仍运行在当前 Python
子进程内，并实现项目现有的 AgentRuntime.run() 边界。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.agent_loop import _sanitize_surrogates
from agent.history import compress_history, enforce_token_limit
from agent.llm_client import AgentCancelled, LLMClient, LLMResponse
from agent.runtime import CancelCallback, ResponseCallback, ToolCallCallback
from agent.tools.base import ToolRegistry

MAX_STEPS_FALLBACK = "（已达到最大推理步数，本轮停止。你可以继续描述你的项目。）"

AgentEvent = dict[str, Any]
EventSink = Callable[[AgentEvent], None]
BeforeToolCall = Callable[
    [dict[str, Any], dict[str, Any], "AgentContext"],
    "AgentToolResult | None",
]
AfterToolCall = Callable[
    [dict[str, Any], "AgentToolResult", "AgentContext"],
    "AgentToolResult | None",
]
TransformContext = Callable[[list[dict[str, Any]], CancelCallback | None], list[dict[str, Any]]]
ConvertToLlm = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
PrepareNextTurn = Callable[[dict[str, Any]], "AgentLoopTurnUpdate | AgentContext | None"]
ShouldStopAfterTurn = Callable[[dict[str, Any]], bool]


@dataclass
class AgentContext:
    """Pi-style 外置上下文，持有消息、工具和系统提示。"""

    system_prompt: str
    messages: list[dict[str, Any]]
    tools: ToolRegistry


@dataclass
class AgentToolResult:
    """结构化工具结果，写回历史时再转成 OpenAI 兼容 tool message。"""

    content: str
    is_error: bool = False
    terminate: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentLoopTurnUpdate:
    """`prepare_next_turn` 返回的下一轮运行状态更新。"""

    context: AgentContext | None = None


@dataclass
class AgentLoopConfig:
    """Pi-style loop 配置钩子。"""

    transform_context: TransformContext | None = None
    convert_to_llm: ConvertToLlm | None = None
    prepare_next_turn: PrepareNextTurn | None = None
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None
    should_stop_after_turn: ShouldStopAfterTurn | None = None


class PiToolExecutor:
    """Pi-style 工具执行器，负责工具生命周期而不是主循环编排。"""

    def __init__(
        self,
        context: AgentContext,
        config: AgentLoopConfig,
        emit: EventSink,
    ) -> None:
        self._context = context
        self._config = config
        self._emit = emit

    def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        assistant_stop_reason: str | None = None,
        on_tool_call: ToolCallCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> tuple[list[dict[str, Any]], list[AgentToolResult]]:
        """按 assistant 原顺序串行执行工具，返回 tool messages 和结构化结果。"""
        if assistant_stop_reason == "length":
            return self._fail_truncated_tool_calls(tool_calls, on_tool_call)

        messages: list[dict[str, Any]] = []
        results: list[AgentToolResult] = []
        for index, tool_call in enumerate(tool_calls):
            if should_cancel is not None and should_cancel():
                cancelled = self._cancelled_tool_messages(tool_calls[index:])
                return messages + cancelled, results
            name, args, parse_error = self._parse_tool_call(tool_call)
            self._emit({
                "type": "tool_execution_start",
                "tool_call": tool_call,
                "tool_call_id": tool_call.get("id", ""),
                "tool_name": name,
                "args": args,
            })
            if on_tool_call:
                on_tool_call(name, args, "start", "")

            result = parse_error or self._execute_tool_call(tool_call, name, args)
            result = self._after_tool_call(tool_call, args, result)
            self._emit({
                "type": "tool_execution_end",
                "tool_call": tool_call,
                "tool_call_id": tool_call.get("id", ""),
                "tool_name": name,
                "result": result,
                "is_error": result.is_error,
            })
            if on_tool_call:
                on_tool_call(name, args, "end", result.content)

            message = self._create_tool_result_message(tool_call, name, result)
            self._emit({"type": "message_start", "message": message})
            self._emit({"type": "message_end", "message": message})
            messages.append(message)
            results.append(result)
        return messages, results

    def _fail_truncated_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        on_tool_call: ToolCallCallback | None,
    ) -> tuple[list[dict[str, Any]], list[AgentToolResult]]:
        """输出被截断时不执行工具，避免用不完整参数访问文件或外部能力。"""
        messages: list[dict[str, Any]] = []
        results: list[AgentToolResult] = []
        for tool_call in tool_calls:
            name, args, _parse_error = self._parse_tool_call(tool_call)
            self._emit({
                "type": "tool_execution_start",
                "tool_call": tool_call,
                "tool_call_id": tool_call.get("id", ""),
                "tool_name": name,
                "args": args,
            })
            if on_tool_call:
                on_tool_call(name, args, "start", "")
            result = AgentToolResult(
                f"工具调用 '{name}' 未执行：模型输出因 length 截断，"
                "参数可能不完整，请重新发起完整工具调用。",
                is_error=True,
            )
            self._emit({
                "type": "tool_execution_end",
                "tool_call": tool_call,
                "tool_call_id": tool_call.get("id", ""),
                "tool_name": name,
                "result": result,
                "is_error": True,
            })
            if on_tool_call:
                on_tool_call(name, args, "end", result.content)
            message = self._create_tool_result_message(tool_call, name, result)
            self._emit({"type": "message_start", "message": message})
            self._emit({"type": "message_end", "message": message})
            messages.append(message)
            results.append(result)
        return messages, results

    def _parse_tool_call(
        self,
        tool_call: dict[str, Any],
    ) -> tuple[str, dict[str, Any], AgentToolResult | None]:
        """解析 OpenAI tool_call 参数，解析失败时返回错误结果。"""
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "")
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return name, {}, AgentToolResult(
                f"工具参数解析出错: JSONDecodeError: {e.msg}",
                is_error=True,
            )
        if not isinstance(args, dict):
            return name, {}, AgentToolResult(
                "工具参数解析出错: arguments 必须是对象。",
                is_error=True,
            )
        return name, args, None

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        name: str,
        args: dict[str, Any],
    ) -> AgentToolResult:
        """执行一个工具调用，错误以结构化结果返回。"""
        if self._config.before_tool_call:
            blocked = self._config.before_tool_call(tool_call, args, self._context)
            if blocked is not None:
                return blocked

        tool = self._context.tools.get(name)
        if tool is None:
            return AgentToolResult(
                f"错误：不存在名为 '{name}' 的工具。可用工具："
                f"{[t for t in self._context.tools._tools]}",
                is_error=True,
            )
        try:
            return AgentToolResult(_sanitize_surrogates(tool.execute(**args)))
        except Exception as e:
            return AgentToolResult(
                _sanitize_surrogates(f"工具执行出错: {type(e).__name__}: {e}"),
                is_error=True,
            )

    def _after_tool_call(
        self,
        tool_call: dict[str, Any],
        args: dict[str, Any],
        result: AgentToolResult,
    ) -> AgentToolResult:
        """执行工具后钩子，默认不改结果。"""
        if not self._config.after_tool_call:
            return result
        override = self._config.after_tool_call(tool_call, result, self._context)
        return override or result

    @staticmethod
    def _create_tool_result_message(
        tool_call: dict[str, Any],
        name: str,
        result: AgentToolResult,
    ) -> dict[str, Any]:
        """创建当前项目 LLMClient 兼容的 tool result message。"""
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "content": result.content,
        }

    @staticmethod
    def _cancelled_tool_messages(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """取消时补齐 pending tool 结果，保持 OpenAI 历史格式合法。"""
        return [
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": "工具调用已停止。",
            }
            for tool_call in tool_calls
        ]


class PiAgentRuntime:
    """基于 Pi loop 结构的 Python runtime。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
        max_steps: int = 8,
        max_history_tokens: int | None = None,
        max_kept_full: int | None = None,
        config: AgentLoopConfig | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._max_history_tokens = max_history_tokens
        self._max_kept_full = max_kept_full
        self._config = config or AgentLoopConfig()
        self._event_sink = event_sink
        self._events: list[AgentEvent] = []
        self._last_model_elapsed_ms = 0
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._context = AgentContext(system_prompt, self._messages, tools)

    @property
    def runtime_name(self) -> str:
        """返回实际运行的 runtime 名称。"""
        return "pi"

    @property
    def messages(self) -> list[dict]:
        """返回当前会话历史，保持现有 .sessions JSON 格式。"""
        return self._messages

    @property
    def events(self) -> list[AgentEvent]:
        """返回本实例已发出的内部事件，供测试和后续诊断复用。"""
        return self._events

    @property
    def last_model_elapsed_ms(self) -> int:
        """返回最近一轮模型调用累计耗时。"""
        return self._last_model_elapsed_ms

    def run(
        self,
        user_text: str,
        on_tool_call: ToolCallCallback | None = None,
        on_response: ResponseCallback | None = None,
        on_delta: Any = None,
        should_cancel: CancelCallback | None = None,
    ) -> str:
        """执行一轮 pi-style Agent 循环，并映射回现有回调协议。"""
        self._last_model_elapsed_ms = 0
        self._sync_context()
        self._check_cancel(should_cancel)
        new_messages: list[dict[str, Any]] = []
        last_completed_turn: dict[str, Any] | None = None
        user_message = {"role": "user", "content": user_text}
        self._messages.append(user_message)
        new_messages.append(user_message)
        self._emit({"type": "agent_start"})
        self._emit({"type": "turn_start"})
        self._emit({"type": "message_start", "message": user_message})
        self._emit({"type": "message_end", "message": user_message})

        for step in range(self._max_steps):
            self._check_cancel(should_cancel)
            if last_completed_turn is not None:
                self._prepare_next_turn(last_completed_turn)
                self._emit({"type": "turn_start"})
            assistant_message = {"role": "assistant", "content": ""}
            self._emit({"type": "message_start", "message": assistant_message})
            response = self._call_llm(on_delta, should_cancel)
            assistant_message = {
                "role": "assistant",
                "content": response.content or "",
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = response.tool_calls
            self._messages.append(assistant_message)
            new_messages.append(assistant_message)
            self._emit({"type": "message_end", "message": assistant_message})

            if response.tool_calls:
                tool_messages, tool_results = self._tool_executor().execute_tool_calls(
                    response.tool_calls,
                    assistant_stop_reason=response.finish_reason,
                    on_tool_call=on_tool_call,
                    should_cancel=should_cancel,
                )
                self._messages.extend(tool_messages)
                new_messages.extend(tool_messages)
                self._sync_context()
                turn = {
                    "message": assistant_message,
                    "tool_results": tool_results,
                    "context": self._context,
                    "new_messages": list(new_messages),
                    "step": step,
                }
                self._emit({"type": "turn_end", **turn})
                last_completed_turn = turn
                if self._should_stop_after_turn(turn):
                    final = tool_results[-1].content if tool_results else response.content
                    self._emit({"type": "agent_end", "messages": list(new_messages)})
                    if on_response:
                        on_response(final)
                    return final
                continue

            content = response.content
            if on_response:
                on_response(content)
            turn = {
                "message": assistant_message,
                "tool_results": [],
                "context": self._context,
                "new_messages": list(new_messages),
                "step": step,
            }
            self._emit({"type": "turn_end", **turn})
            self._emit({"type": "agent_end", "messages": list(new_messages)})
            return content

        fallback = MAX_STEPS_FALLBACK
        fallback_message = {"role": "assistant", "content": fallback}
        self._messages.append(fallback_message)
        new_messages.append(fallback_message)
        self._emit({"type": "message_start", "message": fallback_message})
        self._emit({"type": "message_end", "message": fallback_message})
        if on_response:
            on_response(fallback)
        self._emit({"type": "agent_end", "messages": list(new_messages)})
        return fallback

    def _sync_context(self) -> None:
        """同步可能由 SessionStore 恢复替换过的消息列表。"""
        self._context = AgentContext(self._system_prompt, self._messages, self._tools)

    def _prepare_next_turn(self, turn: dict[str, Any]) -> None:
        """在 turn_end 之后、下一轮开始前应用 prepare_next_turn 更新。"""
        update = None
        if self._config.prepare_next_turn:
            update = self._config.prepare_next_turn(turn)
        if isinstance(update, AgentContext):
            self._context = update
            self._messages = update.messages
        elif isinstance(update, AgentLoopTurnUpdate) and update.context is not None:
            self._context = update.context
            self._messages = update.context.messages

    def _default_transform_context(
        self,
        messages: list[dict[str, Any]],
        should_cancel: CancelCallback | None,
    ) -> list[dict[str, Any]]:
        """默认上下文窗口压缩策略，只影响本次 LLM 请求。"""
        self._check_cancel(should_cancel)
        if self._max_kept_full is not None:
            transformed = compress_history(messages, self._max_kept_full)
        else:
            transformed = compress_history(messages)
        if self._max_history_tokens is not None:
            return enforce_token_limit(transformed, self._max_history_tokens)
        return enforce_token_limit(transformed)

    @staticmethod
    def _default_convert_to_llm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """默认转换：当前项目内部消息已是 OpenAI 兼容格式，直接浅拷贝。"""
        return list(messages)

    def _call_llm(self, on_delta: Any, should_cancel: CancelCallback | None) -> LLMResponse:
        """调用现有 LLMClient，并把流式片段转成内部 message_update 事件。"""
        tools_schema = self._tools.all_schemas()
        transform_context = self._config.transform_context or self._default_transform_context
        convert_to_llm = self._config.convert_to_llm or self._default_convert_to_llm
        transformed_messages = transform_context(self._context.messages, should_cancel)
        llm_messages = convert_to_llm(transformed_messages)
        started = time.perf_counter()

        def emit_delta(delta: str) -> None:
            self._emit({"type": "message_update", "delta": delta})
            if on_delta:
                on_delta(delta)

        try:
            response = self._llm.chat(
                llm_messages,
                tools_schema,
                on_delta=emit_delta if on_delta is not None else None,
                should_cancel=should_cancel,
            )
        except AgentCancelled as e:
            self._last_model_elapsed_ms += int((time.perf_counter() - started) * 1000)
            if e.partial:
                partial = {"role": "assistant", "content": e.partial}
                self._messages.append(partial)
                self._emit({"type": "message_end", "message": partial})
            raise
        except Exception:
            self._last_model_elapsed_ms += int((time.perf_counter() - started) * 1000)
            raise
        self._last_model_elapsed_ms += int((time.perf_counter() - started) * 1000)
        return response

    def _tool_executor(self) -> PiToolExecutor:
        """创建当前 turn 使用的工具执行器。"""
        return PiToolExecutor(self._context, self._config, self._emit)

    def _should_stop_after_turn(self, turn: dict[str, Any]) -> bool:
        """判断本轮完成后是否提前结束。"""
        tool_results: list[AgentToolResult] = turn.get("tool_results") or []
        if tool_results and all(result.terminate for result in tool_results):
            return True
        if self._config.should_stop_after_turn:
            return self._config.should_stop_after_turn(turn)
        return False

    def _emit(self, event: AgentEvent) -> None:
        """记录并发送内部事件。"""
        self._events.append(event)
        if self._event_sink:
            self._event_sink(event)

    @staticmethod
    def _check_cancel(should_cancel: CancelCallback | None) -> None:
        if should_cancel is not None and should_cancel():
            raise AgentCancelled()
