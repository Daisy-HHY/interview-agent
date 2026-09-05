"""Pi-style Agent runtime.

这个 runtime 借鉴 Pi 的事件流和上下文外置结构，但仍运行在当前 Python
子进程内，并实现项目现有的 AgentRuntime.run() 边界。
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.agent_loop import _sanitize_surrogates
from agent.history import compress_history, count_tokens, enforce_token_limit, message_groups
from agent.llm_client import AgentCancelled, LLMClient, LLMResponse
from agent.runtime import CancelCallback, ResponseCallback, ToolCallCallback
from agent.runtime_budget import decide_budget
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
CompactionFn = Callable[[list[dict[str, Any]], CancelCallback | None], list[dict[str, Any]]]


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
    compaction_enabled: bool = False
    compaction_trigger_tokens: int | None = None
    compaction_keep_messages: int = 6
    compaction_fn: CompactionFn | None = None
    dynamic_budget_enabled: bool = False
    dynamic_budget_soft_steps: int = 8
    dynamic_budget_max_elapsed_ms: int = 600_000


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
        max_steps: int = 32,
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
        self._event_sequence = 0
        self._event_started_at = time.perf_counter()
        self._last_model_elapsed_ms = 0
        self._last_compaction: dict[str, Any] = {"state": "disabled"}
        self._last_budget: dict[str, Any] = {}
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

    def restore_messages(self, messages: list[dict]) -> None:
        """恢复 SessionStore 读取的历史消息，并同步 AgentContext。"""
        self._messages = messages
        self._sync_context()

    @property
    def events(self) -> list[AgentEvent]:
        """返回本实例已发出的内部事件，供测试和后续诊断复用。"""
        return self._events

    @property
    def last_model_elapsed_ms(self) -> int:
        """返回最近一轮模型调用累计耗时。"""
        return self._last_model_elapsed_ms

    @property
    def last_compaction(self) -> dict[str, Any]:
        """返回最近一次请求前上下文 checkpoint 的脱敏状态。"""
        return dict(self._last_compaction)

    @property
    def last_budget(self) -> dict[str, Any]:
        """返回最近一轮运行预算的脱敏统计。"""
        return dict(self._last_budget)

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
        self._event_sequence = 0
        self.last_stop_reason = "running"
        self._event_started_at = time.perf_counter()
        self._last_budget = {
            "enabled": self._config.dynamic_budget_enabled,
            "soft_limit": max(1, min(self._config.dynamic_budget_soft_steps, self._max_steps)),
            "hard_limit": self._max_steps,
            "steps_used": 0,
            "hit": False,
            "reason": "running",
        }
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
            if self._config.dynamic_budget_enabled:
                decision = self._check_dynamic_budget(step, last_completed_turn)
                self._last_budget["reason"] = decision.reason
                if not decision.allow:
                    self._last_budget["hit"] = True
                    self._last_budget["steps_used"] = step
                    return self._return_budget_fallback(new_messages, on_response)
            self._last_budget["steps_used"] = step + 1
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
                    self._last_budget["reason"] = "tool_termination"
                    return final
                continue

            content = response.content
            self.last_stop_reason = response.finish_reason or "natural_completion"
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
            self._last_budget["reason"] = "natural_completion"
            return content

        self._last_budget["hit"] = True
        self._last_budget["reason"] = "hard_limit"
        self._last_budget["steps_used"] = self._max_steps
        return self._return_budget_fallback(new_messages, on_response)

    def _return_budget_fallback(
        self,
        new_messages: list[dict[str, Any]],
        on_response: ResponseCallback | None,
    ) -> str:
        """在动态或固定预算耗尽时写入明确的 fallback。"""
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

    def _check_dynamic_budget(
        self,
        step: int,
        last_completed_turn: dict[str, Any] | None,
    ):
        """基于耗时、上下文压力和最近工具失败检查动态预算。"""
        tool_failures = sum(
            1 for result in (last_completed_turn or {}).get("tool_results", [])
            if getattr(result, "is_error", False)
        )
        context_limit = self._max_history_tokens
        context_tokens = count_tokens(self._messages)
        return decide_budget(
            step=step,
            soft_limit=max(1, min(self._config.dynamic_budget_soft_steps, self._max_steps)),
            hard_limit=self._max_steps,
            elapsed_ms=int((time.perf_counter() - self._event_started_at) * 1000),
            context_tokens=context_tokens,
            context_limit=context_limit,
            tool_failures=tool_failures,
            max_elapsed_ms=self._config.dynamic_budget_max_elapsed_ms,
        )

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

    def _compact_context(
        self,
        messages: list[dict[str, Any]],
        should_cancel: CancelCallback | None,
    ) -> list[dict[str, Any]]:
        """按配置创建请求级 checkpoint，不直接改写会话历史。"""
        if not self._config.compaction_enabled:
            self._last_compaction = {"state": "disabled"}
            return messages

        before_tokens = count_tokens(messages)
        trigger = self._config.compaction_trigger_tokens
        started_at = time.perf_counter()
        base = {
            "before_messages": len(messages),
            "before_tokens": before_tokens,
        }
        if trigger is None or trigger <= 0 or before_tokens < trigger:
            self._last_compaction = {
                "state": "not_needed",
                "compaction_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                **base,
            }
            self._emit({"type": "context_compaction", **self._last_compaction})
            return messages

        self._check_cancel(should_cancel)
        checkpoint = deepcopy(messages)
        try:
            compact = (
                self._config.compaction_fn(checkpoint, should_cancel)
                if self._config.compaction_fn
                else self._default_compaction(checkpoint)
            )
            self._check_cancel(should_cancel)
            if not self._is_valid_context(compact):
                raise ValueError("压缩结果不是合法消息序列")
        except AgentCancelled:
            self._last_compaction = {
                "state": "cancelled",
                "compaction_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                **base,
            }
            self._emit({"type": "context_compaction", **self._last_compaction})
            raise
        except Exception:
            self._last_compaction = {
                "state": "fallback",
                "compaction_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                **base,
                "after_messages": len(messages),
            }
            self._emit({"type": "context_compaction", **self._last_compaction})
            return messages

        self._last_compaction = {
            "state": "completed",
            "compaction_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            **base,
            "after_messages": len(compact),
            "after_tokens": count_tokens(compact),
        }
        self._emit({"type": "context_compaction", **self._last_compaction})
        return compact

    def _default_compaction(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按完整消息组保留最近窗口，同时保留当前问题及本轮工具链。"""
        keep = max(1, self._config.compaction_keep_messages)
        current = next((i for i in range(len(messages) - 1, -1, -1)
                        if messages[i].get("role") == "user"), len(messages) - 1)
        start = min(current, max(0, len(messages) - keep))
        compact = []
        position = 0
        for group in message_groups(messages):
            position += len(group)
            if group[0].get("role") == "system" or position > start:
                compact.extend(group)
        return compact

    @staticmethod
    def _repair_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """移除没有成对 assistant tool_call 的 tool 消息，避免制造假历史。"""
        result: list[dict[str, Any]] = []
        available_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system", "user", "assistant", "tool",
            }:
                continue
            if message.get("role") == "tool":
                call_id = str(message.get("tool_call_id") or "")
                if call_id not in available_ids:
                    continue
            result.append(message)
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    if isinstance(call, dict) and call.get("id"):
                        available_ids.add(str(call["id"]))
        return result

    @classmethod
    def _is_valid_context(cls, messages: Any) -> bool:
        """检查压缩结果是否仍是当前 LLM 可接受的基本消息序列。"""
        if not isinstance(messages, list) or not messages:
            return False
        if not all(isinstance(message, dict) for message in messages):
            return False
        if messages[0].get("role") != "system":
            return False
        repaired = cls._repair_tool_messages(messages)
        if repaired != messages:
            return False
        pending_ids: set[str] = set()
        for message in messages:
            if message.get("role") == "assistant":
                pending_ids.update(
                    str(call["id"])
                    for call in message.get("tool_calls") or []
                    if isinstance(call, dict) and call.get("id")
                )
            elif message.get("role") == "tool":
                pending_ids.discard(str(message.get("tool_call_id") or ""))
        return not pending_ids

    @staticmethod
    def _default_convert_to_llm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """默认转换：当前项目内部消息已是 OpenAI 兼容格式，直接浅拷贝。"""
        return list(messages)

    def _call_llm(self, on_delta: Any, should_cancel: CancelCallback | None) -> LLMResponse:
        """调用现有 LLMClient，并把流式片段转成内部 message_update 事件。"""
        tools_schema = self._tools.all_schemas()
        transform_context = self._config.transform_context or self._default_transform_context
        convert_to_llm = self._config.convert_to_llm or self._default_convert_to_llm
        checkpoint_messages = self._compact_context(deepcopy(self._context.messages), should_cancel)
        transformed_messages = transform_context(checkpoint_messages, should_cancel)
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
        self._event_sequence += 1
        record = dict(event)
        record["event_seq"] = self._event_sequence
        record["elapsed_ms"] = int((time.perf_counter() - self._event_started_at) * 1000)
        self._events.append(record)
        if self._event_sink:
            self._event_sink(record)

    @staticmethod
    def _check_cancel(should_cancel: CancelCallback | None) -> None:
        if should_cancel is not None and should_cancel():
            raise AgentCancelled()
