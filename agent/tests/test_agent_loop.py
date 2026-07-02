"""Agent 循环测试（设计第 7.2.3 节，第 3 层测试——最有价值）。

用 FakeLLM 精确控制 LLM 输出，验证 Agent 循环所有行为分支。
零费用、确定、能覆盖所有情况。

关键覆盖点：
- 直接回答（不调工具）
- 调工具后回答（循环 2 次）
- 达到 MAX_STEPS 安全阀
- 工具失败自我恢复（设计 3.9 / 6.6）
- 幻觉调用不存在的工具
- 工具结果正确进入历史
- 回调被正确触发
"""


from agent.agent_loop import AgentLoop
from agent.llm_client import FakeLLM, make_text_response, make_tool_call_response
from agent.tools.base import ToolRegistry

# ──────────────────────────────────────────────
# 测试辅助：假工具 + 装配注册表
# ──────────────────────────────────────────────


class EchoTool:
    """测试用假工具：把参数原样返回（方便断言）。"""

    def __init__(self, name: str = "echo"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": "测试用假工具",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    },
                    "required": ["text"],
                },
            },
        }

    def execute(self, text: str = "") -> str:
        return f"echo: {text}"


class FailingTool:
    """测试用假工具：永远抛异常（测错误恢复）。"""

    name = "failing_tool"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "failing_tool",
                "description": "永远失败的测试工具",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    def execute(self) -> str:
        raise RuntimeError("故意失败")


def build_registry(*tools) -> ToolRegistry:
    """装配一个含指定工具的注册表。"""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


SYSTEM_PROMPT = "你是面试官"


# ──────────────────────────────────────────────
# 基础循环行为测试
# ──────────────────────────────────────────────


class TestBasicLoop:
    def test_direct_answer_no_tool(self):
        """LLM 直接回答：不调工具，循环 1 次结束。"""
        fake = FakeLLM([make_text_response("你好，我是面试官")])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(EchoTool()),
            system_prompt=SYSTEM_PROMPT,
        )

        result = loop.run("你好")

        assert result == "你好，我是面试官"
        assert fake.call_count == 1  # 只调了 1 次 LLM

    def test_tool_call_then_answer(self):
        """LLM 调工具后回答：循环 2 次。"""
        fake = FakeLLM([
            make_tool_call_response("echo", {"text": "项目信息"}),  # 第 1 轮：调工具
            make_text_response("我看到你的项目了"),                   # 第 2 轮：回答
        ])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(EchoTool()),
            system_prompt=SYSTEM_PROMPT,
        )

        result = loop.run("看看我的项目")

        assert result == "我看到你的项目了"
        assert fake.call_count == 2

    def test_multiple_tool_calls_in_sequence(self):
        """LLM 连续调多个工具（每轮一个），最后回答。"""
        fake = FakeLLM([
            make_tool_call_response("echo", {"text": "第一次"}),
            make_tool_call_response("echo", {"text": "第二次"}),
            make_text_response("完成了"),
        ])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(EchoTool()),
            system_prompt=SYSTEM_PROMPT,
        )

        result = loop.run("开始")

        assert result == "完成了"
        assert fake.call_count == 3

    def test_multiple_tools_in_one_response(self):
        """一轮返回多个工具调用（tool_calls 列表有多个）。"""
        # 构造一个含 2 个工具调用的响应
        import json

        from agent.llm_client import LLMResponse
        multi_call = LLMResponse(tool_calls=[
            {"id": "call_1", "type": "function",
             "function": {"name": "echo", "arguments": json.dumps({"text": "A"})}},
            {"id": "call_2", "type": "function",
             "function": {"name": "echo", "arguments": json.dumps({"text": "B"})}},
        ])
        fake = FakeLLM([multi_call, make_text_response("done")])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(EchoTool()),
            system_prompt=SYSTEM_PROMPT,
        )

        result = loop.run("test")

        assert result == "done"
        # 历史里应该有 2 个 tool 结果
        tool_msgs = [m for m in loop.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2


# ──────────────────────────────────────────────
# 安全阀测试（设计第 2.5 节）
# ──────────────────────────────────────────────


class TestMaxStepsSafety:
    def test_stops_at_max_steps(self):
        """达到 MAX_STEPS 必须停，返回兜底文本。"""
        # 脚本：无限调工具，永远不直接回答
        infinite_tool_calls = [
            make_tool_call_response("echo", {"text": str(i)})
            for i in range(20)
        ]
        fake = FakeLLM(infinite_tool_calls)
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(EchoTool()),
            system_prompt=SYSTEM_PROMPT,
            max_steps=3,  # 设小一点方便测
        )

        result = loop.run("开始")

        assert "最大推理步数" in result
        assert fake.call_count == 3  # 恰好调 3 次就停

    def test_max_steps_default_is_8(self):
        """默认 MAX_STEPS 是 8（设计第 2.5 节）。"""
        fake = FakeLLM([make_text_response("done")])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(),
            system_prompt=SYSTEM_PROMPT,
        )
        assert loop._max_steps == 8


# ──────────────────────────────────────────────
# 错误恢复测试（设计第 3.9 / 6.6 节）
# ──────────────────────────────────────────────


class TestErrorRecovery:
    def test_tool_failure_recovers(self):
        """工具失败时，Agent 不崩，错误进历史让 LLM 自我恢复。"""
        fake = FakeLLM([
            make_tool_call_response("failing_tool", {}),  # 调会失败的工具
            make_text_response("看到你出错了，换个方式问"),  # LLM 看到错误后调整
        ])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(FailingTool()),
            system_prompt=SYSTEM_PROMPT,
        )

        result = loop.run("调用会失败的工具")

        # 没崩，正常返回了 LLM 的第二轮回答
        assert result == "看到你出错了，换个方式问"
        assert fake.call_count == 2

        # 错误信息应该进了历史（作为 tool 结果）
        tool_msgs = [m for m in loop.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "出错" in tool_msgs[0]["content"] or "失败" in tool_msgs[0]["content"]

    def test_nonexistent_tool_call(self):
        """LLM 幻觉调用不存在的工具：返回错误文本，不崩。"""
        fake = FakeLLM([
            make_tool_call_response("ghost_tool", {}),  # 不存在的工具
            make_text_response("抱歉，换个问法"),
        ])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(EchoTool()),  # 只有 echo，没有 ghost_tool
            system_prompt=SYSTEM_PROMPT,
        )

        result = loop.run("test")

        assert result == "抱歉，换个问法"
        # 错误信息进了历史
        tool_msgs = [m for m in loop.messages if m.get("role") == "tool"]
        assert "不存在" in tool_msgs[0]["content"]


# ──────────────────────────────────────────────
# 历史管理测试（验证 Phase 2 被正确调用）
# ──────────────────────────────────────────────


class TestHistoryManagement:
    def test_system_prompt_always_first(self):
        """系统提示永远是历史第一条（设计第 6.2.3 节）。"""
        fake = FakeLLM([make_text_response("hi")])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(),
            system_prompt="重要系统提示",
        )

        loop.run("用户问题")

        assert loop.messages[0]["role"] == "system"
        assert loop.messages[0]["content"] == "重要系统提示"

    def test_tool_results_enter_history(self):
        """工具执行结果正确进入历史（role==tool）。"""
        fake = FakeLLM([
            make_tool_call_response("echo", {"text": "数据"}),
            make_text_response("done"),
        ])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(EchoTool()),
            system_prompt=SYSTEM_PROMPT,
        )

        loop.run("test")

        tool_msgs = [m for m in loop.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "数据" in tool_msgs[0]["content"]

    def test_history_persists_across_runs(self):
        """多次 run 之间历史共享（多轮对话）。"""
        fake = FakeLLM([
            make_text_response("第一轮回答"),
            make_text_response("第二轮回答"),
        ])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(),
            system_prompt=SYSTEM_PROMPT,
        )

        loop.run("第一轮问题")
        loop.run("第二轮问题")

        # 历史应该累积：系统 + 用户1 + 助手1 + 用户2 + 助手2
        assert len(loop.messages) == 5
        assert loop.messages[0]["role"] == "system"
        assert loop.messages[1]["content"] == "第一轮问题"
        assert loop.messages[2]["content"] == "第一轮回答"
        assert loop.messages[3]["content"] == "第二轮问题"
        assert loop.messages[4]["content"] == "第二轮回答"


# ──────────────────────────────────────────────
# 回调测试
# ──────────────────────────────────────────────


class TestCallbacks:
    def test_response_callback_triggered(self):
        """LLM 直接回答时，on_response 被触发。"""
        fake = FakeLLM([make_text_response("最终回答")])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(),
            system_prompt=SYSTEM_PROMPT,
        )

        captured = []
        loop.run("test", on_response=captured.append)

        assert captured == ["最终回答"]

    def test_tool_call_callback_triggered(self):
        """调工具时，on_tool_call 被触发（start 和 end 两次）。"""
        fake = FakeLLM([
            make_tool_call_response("echo", {"text": "hi"}),
            make_text_response("done"),
        ])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(EchoTool()),
            system_prompt=SYSTEM_PROMPT,
        )

        events = []

        def on_tool_call(name, args, phase, result):
            events.append((name, phase, result))

        loop.run("test", on_tool_call=on_tool_call)

        # 应该有 start 和 end 两个事件
        assert len(events) == 2
        assert events[0][0] == "echo"       # 工具名
        assert events[0][1] == "start"      # 开始
        assert events[1][1] == "end"        # 结束
        assert "hi" in events[1][2]         # 结果含 echo 内容

    def test_callbacks_not_required(self):
        """不传回调也能正常运行（回调是可选的）。"""
        fake = FakeLLM([make_text_response("done")])
        loop = AgentLoop(
            llm=fake,
            tools=build_registry(),
            system_prompt=SYSTEM_PROMPT,
        )

        # 不传任何回调，应该不报错
        result = loop.run("test")
        assert result == "done"


# ──────────────────────────────────────────────
# 流式透传测试（设计第 1.6 节，Phase 7-C）
# ──────────────────────────────────────────────


class TestStreamingPassthrough:
    """验证 agent_loop.run 把 on_delta 透传给 LLMClient.chat。"""

    def test_on_delta_passed_to_chat(self):
        """★ run 传 on_delta → chat 收到（通过自定义 LLM 验证）。"""
        from agent.agent_loop import AgentLoop
        from agent.llm_client import LLMResponse
        from agent.tools.base import ToolRegistry

        received_delta_kwargs = []

        class SpyLLM:
            def __init__(self):
                self.received_delta = None
            def chat(self, messages, tools, on_delta=None, cancel_event=None):
                received_delta_kwargs.append(on_delta)
                self.received_delta = on_delta
                # 模拟伪流式：推一次文本
                if on_delta is not None:
                    on_delta("流式片段")
                return LLMResponse(content="最终回答")

        spy = SpyLLM()
        loop = AgentLoop(
            llm=spy, tools=ToolRegistry(), system_prompt="sys",
        )
        deltas = []
        loop.run("问", on_delta=deltas.append)

        assert spy.received_delta is not None  # chat 收到了 on_delta
        assert deltas == ["流式片段"]  # delta 被推到了 run 的回调

    def test_no_on_delta_passes_none_to_chat(self):
        """不传 on_delta → chat 收到 None（非流式）。"""
        from agent.agent_loop import AgentLoop
        from agent.llm_client import LLMResponse
        from agent.tools.base import ToolRegistry

        class SpyLLM:
            def __init__(self):
                self.received_delta = "unset"
            def chat(self, messages, tools, on_delta="unset", cancel_event=None):
                self.received_delta = on_delta
                return LLMResponse(content="答")

        spy = SpyLLM()
        loop = AgentLoop(
            llm=spy, tools=ToolRegistry(), system_prompt="sys",
        )
        loop.run("问")  # 不传 on_delta

        assert spy.received_delta is None  # None 表示非流式

    def test_streaming_works_with_tool_calls(self):
        """流式模式下工具调用轮次正常（不推 delta，正常循环）。"""
        import tempfile

        from agent.agent_loop import AgentLoop
        from agent.llm_client import (
            FakeLLM,
            make_text_response,
            make_tool_call_response,
        )
        from agent.tools.base import ToolRegistry
        from agent.tools.builtin import ListDirectoryTool

        registry = ToolRegistry()
        registry.register(ListDirectoryTool(tempfile.gettempdir()))

        fake = FakeLLM([
            make_tool_call_response("list_directory", {"path": "."}),
            make_text_response("流式最终回答"),
        ])
        loop = AgentLoop(
            llm=fake, tools=registry, system_prompt="sys",
        )
        deltas = []
        result = loop.run("看看", on_delta=deltas.append)

        assert result == "流式最终回答"
        # 工具调用轮不推文本 delta；最终回答轮推一次（FakeLLM 伪流式）
        assert deltas == ["流式最终回答"]


# ──────────────────────────────────────────────
# cancel（中断）机制（#8 stop 生效的基础）
# ──────────────────────────────────────────────


class TestCancel:
    """loop.run 支持 cancel_event，被 set 后尽快停止（#8）。

    stop 按钮要生效，loop 必须能在生成过程中被外部取消。cancel_event 是
    threading.Event，loop 在每个步骤边界检查，set 了就停止后续 LLM 调用。
    """

    def test_cancel_before_first_step_skips_llm(self):
        """run 前 cancel_event 已 set → 不调 LLM，返回中断提示。"""
        import threading

        fake = FakeLLM([make_text_response("不应到达")])
        loop = AgentLoop(
            llm=fake, tools=build_registry(), system_prompt=SYSTEM_PROMPT,
        )
        event = threading.Event()
        event.set()

        result = loop.run("问", cancel_event=event)

        assert fake.call_count == 0  # 没调 LLM
        assert "停止" in result

    def test_cancel_between_steps_stops_loop(self):
        """第一步工具完成后 cancel → 不进入第二步的 LLM 调用。"""
        import threading

        event = threading.Event()
        fake = FakeLLM([
            make_tool_call_response("echo", {"text": "x"}),
            make_text_response("不应到达"),
        ])
        loop = AgentLoop(
            llm=fake, tools=build_registry(EchoTool()), system_prompt=SYSTEM_PROMPT,
        )

        def on_tool_call(name, args, phase, result):
            if phase == "end":
                event.set()  # 第一步工具完成 → 触发 cancel

        result = loop.run("问", on_tool_call=on_tool_call, cancel_event=event)

        assert fake.call_count == 1  # 只调了第一步的 LLM，第二步被 cancel 拦下
        assert "停止" in result

    def test_cancel_pushes_marker_via_on_delta(self):
        """cancel 时通过 on_delta 推中断标记，让前端流式气泡可见。"""
        import threading

        fake = FakeLLM([make_tool_call_response("echo", {"text": "x"})])
        loop = AgentLoop(
            llm=fake, tools=build_registry(EchoTool()), system_prompt=SYSTEM_PROMPT,
        )
        event = threading.Event()
        deltas = []

        def on_tool_call(name, args, phase, result):
            if phase == "end":
                event.set()

        loop.run("问", on_tool_call=on_tool_call, on_delta=deltas.append,
                 cancel_event=event)

        assert any("停止" in d for d in deltas)

    def test_no_cancel_event_runs_normally(self):
        """不传 cancel_event（None）时行为不变（向后兼容）。"""
        fake = FakeLLM([make_text_response("正常回答")])
        loop = AgentLoop(
            llm=fake, tools=build_registry(), system_prompt=SYSTEM_PROMPT,
        )
        result = loop.run("问")
        assert result == "正常回答"
