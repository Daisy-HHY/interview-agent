"""Pi-style runtime 测试。"""

import json

import pytest

from agent.llm_client import AgentCancelled, FakeLLM, LLMResponse, make_text_response
from agent.pi_runtime import AgentContext, AgentLoopConfig, AgentToolResult, PiAgentRuntime
from agent.tools.base import ToolRegistry


class EchoTool:
    """测试用工具：回显参数。"""

    name = "echo"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "测试工具",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, text: str = "") -> str:
        self.calls.append(text)
        return f"echo: {text}"


class FailingTool:
    """测试用工具：执行时抛异常。"""

    name = "failing_tool"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "失败工具",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def execute(self) -> str:
        raise RuntimeError("故意失败")


def registry_with(*tools) -> ToolRegistry:
    """创建测试工具注册表。"""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def tool_response(*calls: tuple[str, dict]) -> LLMResponse:
    """创建一个包含多个 OpenAI 风格 tool_calls 的响应。"""
    return LLMResponse(tool_calls=[
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for index, (name, args) in enumerate(calls, start=1)
    ])


def invalid_args_response() -> LLMResponse:
    """创建参数 JSON 损坏的工具响应。"""
    return LLMResponse(tool_calls=[{
        "id": "call_bad",
        "type": "function",
        "function": {"name": "echo", "arguments": "{bad"},
    }])


def test_pi_direct_answer_emits_events_and_records_history():
    fake = FakeLLM([make_text_response("直接回答")])
    loop = PiAgentRuntime(fake, registry_with(), "sys")

    result = loop.run("你好")

    assert result == "直接回答"
    assert loop.runtime_name == "pi"
    assert fake.call_count == 1
    assert [m["role"] for m in loop.messages] == ["system", "user", "assistant"]
    assert [event["type"] for event in loop.events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert loop.events[0]["event_seq"] == 1
    assert isinstance(loop.events[-1]["elapsed_ms"], int)


def test_pi_restore_messages_uses_public_history_boundary():
    """恢复历史通过公开方法完成，并同步 AgentContext。"""
    loop = PiAgentRuntime(FakeLLM([make_text_response("ok")]), registry_with(), "sys")

    restored = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "恢复的问题"},
    ]
    loop.restore_messages(restored)

    assert loop.messages is restored
    assert loop._context.messages is restored  # noqa: SLF001


def test_pi_tool_call_then_answer_triggers_callbacks():
    fake = FakeLLM([
        tool_response(("echo", {"text": "项目"})),
        make_text_response("看到项目了"),
    ])
    tool = EchoTool()
    loop = PiAgentRuntime(fake, registry_with(tool), "sys")
    events = []

    result = loop.run("看看", on_tool_call=lambda name, args, phase, result: events.append(
        (name, args, phase, result),
    ))

    assert result == "看到项目了"
    assert fake.call_count == 2
    assert tool.calls == ["项目"]
    assert events[0][:3] == ("echo", {"text": "项目"}, "start")
    assert events[1][0] == "echo"
    assert events[1][2] == "end"
    assert "项目" in events[1][3]


def test_pi_multiple_tools_in_source_order():
    fake = FakeLLM([
        tool_response(("echo", {"text": "A"}), ("echo", {"text": "B"})),
        make_text_response("done"),
    ])
    loop = PiAgentRuntime(fake, registry_with(EchoTool()), "sys")

    loop.run("开始")

    tool_messages = [m for m in loop.messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2"]
    assert [m["content"] for m in tool_messages] == ["echo: A", "echo: B"]


def test_pi_tool_failure_is_error_observation():
    fake = FakeLLM([
        tool_response(("failing_tool", {})),
        make_text_response("换个方式问"),
    ])
    loop = PiAgentRuntime(fake, registry_with(FailingTool()), "sys")

    result = loop.run("调用失败工具")

    assert result == "换个方式问"
    tool_message = next(m for m in loop.messages if m.get("role") == "tool")
    assert "工具执行出错" in tool_message["content"]
    tool_end = next(event for event in loop.events if event["type"] == "tool_execution_end")
    assert tool_end["result"].is_error is True


def test_pi_invalid_json_arguments_does_not_execute_tool():
    fake = FakeLLM([invalid_args_response(), make_text_response("已恢复")])
    tool = EchoTool()
    loop = PiAgentRuntime(fake, registry_with(tool), "sys")

    result = loop.run("坏参数")

    assert result == "已恢复"
    assert tool.calls == []
    tool_message = next(m for m in loop.messages if m.get("role") == "tool")
    assert "工具参数解析出错" in tool_message["content"]


def test_pi_truncated_tool_call_does_not_execute_tool():
    """对齐 Pi：length 截断的 tool calls 不执行，只写错误 observation。"""
    fake = FakeLLM([
        LLMResponse(
            tool_calls=tool_response(("echo", {"text": "x"})).tool_calls,
            finish_reason="length",
        ),
        make_text_response("已重试"),
    ])
    tool = EchoTool()
    loop = PiAgentRuntime(fake, registry_with(tool), "sys")

    result = loop.run("截断")

    assert result == "已重试"
    assert tool.calls == []
    tool_message = next(m for m in loop.messages if m.get("role") == "tool")
    assert "length 截断" in tool_message["content"]
    tool_end = next(event for event in loop.events if event["type"] == "tool_execution_end")
    assert tool_end["is_error"] is True


def test_pi_nonexistent_tool_call_records_error():
    fake = FakeLLM([tool_response(("ghost_tool", {})), make_text_response("换个问法")])
    loop = PiAgentRuntime(fake, registry_with(EchoTool()), "sys")

    result = loop.run("不存在的工具")

    assert result == "换个问法"
    tool_message = next(m for m in loop.messages if m.get("role") == "tool")
    assert "不存在" in tool_message["content"]


def test_pi_streaming_delta_passes_through():
    fake = FakeLLM([make_text_response("流式回答")])
    loop = PiAgentRuntime(fake, registry_with(), "sys")
    deltas = []

    result = loop.run("问", on_delta=deltas.append)

    assert result == "流式回答"
    assert deltas == ["流式回答"]
    assert any(event["type"] == "message_update" for event in loop.events)


def test_pi_stops_at_max_steps():
    fake = FakeLLM([
        tool_response(("echo", {"text": "1"})),
        tool_response(("echo", {"text": "2"})),
    ])
    loop = PiAgentRuntime(fake, registry_with(EchoTool()), "sys", max_steps=2)

    result = loop.run("开始")

    assert "最大推理步数" in result
    assert fake.call_count == 2


def test_pi_default_max_steps_is_32():
    fake = FakeLLM([make_text_response("答")])
    loop = PiAgentRuntime(fake, registry_with(), "sys")

    assert loop._max_steps == 32  # noqa: SLF001


def test_pi_continues_after_native_eight_step_budget():
    """Pi 不复用 native 的 8 回合限制，直到模型无工具调用才结束。"""
    fake = FakeLLM([
        *(tool_response(("echo", {"text": str(index)})) for index in range(9)),
        make_text_response("完整回答"),
    ])
    tool = EchoTool()
    loop = PiAgentRuntime(fake, registry_with(tool), "sys")

    result = loop.run("请完整查看项目")

    assert result == "完整回答"
    assert fake.call_count == 10
    assert len(tool.calls) == 9


def test_pi_cancel_before_tool_keeps_history_valid():
    fake = FakeLLM([tool_response(("echo", {"text": "x"}))])
    loop = PiAgentRuntime(fake, registry_with(EchoTool()), "sys")
    calls = 0

    def should_cancel():
        nonlocal calls
        calls += 1
        return calls >= 5

    with pytest.raises(AgentCancelled):
        loop.run("问", should_cancel=should_cancel)

    assert loop.messages[-1]["role"] == "tool"
    assert "已停止" in loop.messages[-1]["content"]


def test_pi_history_limits_keep_system_prompt():
    fake = FakeLLM([
        make_text_response("答1"),
        make_text_response("答2"),
        make_text_response("答3"),
    ])
    loop = PiAgentRuntime(fake, registry_with(), "重要系统提示", max_history_tokens=50)

    loop.run("问1")
    loop.run("问2")
    loop.run("问3")

    assert loop.messages[0] == {"role": "system", "content": "重要系统提示"}
    assert loop.messages[-1]["role"] == "assistant"


def test_pi_transform_context_runs_before_convert_to_llm():
    """LLM 边界按 transform_context -> convert_to_llm -> chat 执行。"""
    calls = []

    class SpyLLM:
        def chat(self, messages, tools, on_delta=None, should_cancel=None):
            calls.append(("chat", [m["role"] for m in messages]))
            return make_text_response("答")

    def transform_context(messages, should_cancel=None):
        calls.append(("transform", [m["role"] for m in messages]))
        return messages + [{"role": "user", "content": "transform 注入"}]

    def convert_to_llm(messages):
        calls.append(("convert", [m["role"] for m in messages]))
        return [m for m in messages if m.get("content") != "transform 注入"]

    loop = PiAgentRuntime(
        SpyLLM(),
        registry_with(),
        "sys",
        config=AgentLoopConfig(
            transform_context=transform_context,
            convert_to_llm=convert_to_llm,
        ),
    )

    loop.run("问")

    assert [name for name, _roles in calls] == ["transform", "convert", "chat"]
    assert calls[1][1] == ["system", "user", "user"]
    assert calls[2][1] == ["system", "user"]


def test_pi_prepare_next_turn_runs_after_turn_end_before_next_turn():
    """prepare_next_turn 只在上一轮 turn_end 后、下一轮 turn_start 前执行。"""
    fake = FakeLLM([
        tool_response(("echo", {"text": "x"})),
        make_text_response("done"),
    ])
    order = []

    def event_sink(event):
        if event["type"] in {"turn_end", "turn_start"}:
            order.append(event["type"])

    def prepare_next_turn(turn):
        order.append("prepare_next_turn")
        assert turn["tool_results"][0].content == "echo: x"
        return None

    loop = PiAgentRuntime(
        fake,
        registry_with(EchoTool()),
        "sys",
        config=AgentLoopConfig(prepare_next_turn=prepare_next_turn),
        event_sink=event_sink,
    )

    loop.run("问")

    assert order == ["turn_start", "turn_end", "prepare_next_turn", "turn_start", "turn_end"]


def test_pi_new_messages_contains_only_current_run_messages():
    """turn_end / agent_end 的 new_messages 不包含本次 run 前已有历史。"""
    fake = FakeLLM([
        make_text_response("第一轮"),
        make_text_response("第二轮"),
    ])
    loop = PiAgentRuntime(fake, registry_with(), "sys")
    loop.run("旧问题")
    before_count = len(loop.messages)

    loop.run("新问题")

    turn_end = [event for event in loop.events if event["type"] == "turn_end"][-1]
    agent_end = [event for event in loop.events if event["type"] == "agent_end"][-1]
    assert len(turn_end["new_messages"]) == 2
    assert len(agent_end["messages"]) == 2
    assert loop.messages[:before_count][-1]["content"] == "第一轮"
    assert [m["content"] for m in turn_end["new_messages"]] == ["新问题", "第二轮"]


def test_pi_before_after_tool_hooks_can_override_result():
    fake = FakeLLM([tool_response(("echo", {"text": "x"}))])

    def before_tool_call(tool_call, args, context):
        return AgentToolResult("blocked", is_error=True, terminate=True)

    def after_tool_call(tool_call, result, context):
        return AgentToolResult(f"after: {result.content}", terminate=result.terminate)

    loop = PiAgentRuntime(
        fake,
        registry_with(EchoTool()),
        "sys",
        config=AgentLoopConfig(
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
        ),
    )

    result = loop.run("问")

    assert result == "after: blocked"
    assert loop.messages[-1]["content"] == "after: blocked"


def test_pi_tool_executor_can_be_used_directly():
    """工具执行器脱离 runtime 主循环也能产生 tool result message。"""
    from agent.pi_runtime import PiToolExecutor

    context = AgentContext("sys", [{"role": "system", "content": "sys"}], registry_with(EchoTool()))
    events = []
    callbacks = []
    executor = PiToolExecutor(context, AgentLoopConfig(), events.append)
    response = tool_response(("echo", {"text": "x"}))

    messages, results = executor.execute_tool_calls(
        response.tool_calls,
        on_tool_call=lambda name, args, phase, result: callbacks.append(
            (name, phase, result),
        ),
    )

    assert messages == [{"role": "tool", "tool_call_id": "call_1", "content": "echo: x"}]
    assert results[0].content == "echo: x"
    assert [event["type"] for event in events] == [
        "tool_execution_start",
        "tool_execution_end",
        "message_start",
        "message_end",
    ]
    assert [item[1] for item in callbacks] == ["start", "end"]


def test_pi_compaction_is_disabled_by_default():
    fake = FakeLLM([make_text_response("答")])
    loop = PiAgentRuntime(fake, registry_with(), "sys")

    loop.run("问题")

    assert not [event for event in loop.events if event["type"] == "context_compaction"]


def test_pi_compaction_uses_checkpoint_without_mutating_history():
    class RecordingLLM:
        def __init__(self):
            self.messages = None

        def chat(self, messages, tools, on_delta=None, should_cancel=None):
            self.messages = messages
            return make_text_response("答")

    llm = RecordingLLM()
    loop = PiAgentRuntime(
        llm,
        registry_with(),
        "sys",
        config=AgentLoopConfig(
            compaction_enabled=True,
            compaction_trigger_tokens=5,
            compaction_keep_messages=2,
        ),
    )
    loop.restore_messages([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "旧问题 " * 10},
        {"role": "assistant", "content": "旧回答 " * 10},
    ])
    before = list(loop.messages)

    loop.run("新问题")

    event = [event for event in loop.events if event["type"] == "context_compaction"][-1]
    assert event["state"] == "completed"
    assert event["before_messages"] > event["after_messages"]
    assert llm.messages[0] == {"role": "system", "content": "sys"}
    assert llm.messages[-1] == {"role": "user", "content": "新问题"}
    assert loop.messages[: len(before)] == before


def test_pi_compaction_invalid_result_falls_back_to_checkpoint():
    fake = FakeLLM([make_text_response("答")])
    loop = PiAgentRuntime(
        fake,
        registry_with(),
        "sys",
        config=AgentLoopConfig(
            compaction_enabled=True,
            compaction_trigger_tokens=1,
            compaction_fn=lambda _messages, _cancel: [{"role": "tool", "content": "孤立"}],
        ),
    )

    loop.run("问题")

    event = [event for event in loop.events if event["type"] == "context_compaction"][-1]
    assert event["state"] == "fallback"
    assert event["after_messages"] == event["before_messages"]
