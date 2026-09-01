"""LangChain runtime 的最小契约测试。"""

import sys
import types

from agent.langchain_runtime import LangChainAgentRuntime
from agent.langchain_tools import build_langchain_tools
from agent.llm_client import ERROR_KIND_BAD_REQUEST, LLMError
from agent.tools.base import ToolRegistry


class EchoTool:
    name = "echo"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "回显文本",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }

    def execute(self, text: str = "") -> str:
        return f"echo: {text}"


class FakeMcpLikeTool:
    """模拟未来 MCP adapter 产出的 Tool，不依赖任何 MCP SDK。"""

    name = "mcp_lookup"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "查询外部能力",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }

    def execute(self, query: str = "") -> str:
        return f"mcp-like: {query}"


def install_fake_langchain_tools(monkeypatch):
    """安装测试用 LangChain tool decorator，避免单测依赖真实包。"""
    langchain = types.ModuleType("langchain")
    tools_mod = types.ModuleType("langchain.tools")

    def tool(name, description="", args_schema=None):
        def decorate(fn):
            fn.name = name
            fn.description = description
            fn.args_schema = args_schema
            return fn

        return decorate

    tools_mod.tool = tool
    monkeypatch.setitem(sys.modules, "langchain", langchain)
    monkeypatch.setitem(sys.modules, "langchain.tools", tools_mod)


def registry_with_echo() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


def test_langchain_tool_adapter_accepts_mcp_like_tool(monkeypatch):
    """未来 MCP 工具只要适配成 Tool，就能复用同一个 adapter。"""
    install_fake_langchain_tools(monkeypatch)
    registry = ToolRegistry()
    registry.register(FakeMcpLikeTool())

    tools = build_langchain_tools(registry)

    assert tools[0].name == "mcp_lookup"
    assert tools[0](query="runtime") == "mcp-like: runtime"


def test_langchain_tool_adapter_reuses_existing_tool(monkeypatch):
    """adapter 应复用 ToolRegistry 内工具，并触发 start/end 通知。"""
    install_fake_langchain_tools(monkeypatch)
    events = []
    tools = build_langchain_tools(
        registry_with_echo(),
        on_tool_call=lambda name, args, phase, result: events.append(
            (name, phase, args, result)
        ),
    )

    result = tools[0](text="hi")

    assert result == "echo: hi"
    assert tools[0].name == "echo"
    assert events[0][0:2] == ("echo", "start")
    assert events[1][0:2] == ("echo", "end")
    assert events[1][3] == "echo: hi"


class FakeMessageChunk:
    def __init__(self, text):
        self.text = text


class FakeStream:
    def __init__(self, items, output):
        self._items = items
        self.output = output

    def interleave(self, *names):
        return iter(self._items)


class FakeAgent:
    def __init__(self, tools):
        self._tools = tools

    def stream_events(self, payload, config=None, version="v3"):
        tool = self._tools[0]
        tool_result = (
            tool.invoke({"text": "项目"})
            if hasattr(tool, "invoke")
            else tool(text="项目")
        )
        output = {"messages": [{"role": "assistant", "content": f"最终：{tool_result}"}]}
        return FakeStream(
            [
                ("messages", FakeMessageChunk(["最终：", tool_result])),
            ],
            output,
        )


class FakeConfigAgent:
    def __init__(self):
        self.config = None

    def stream_events(self, payload, config=None, version="v3"):
        self.config = config
        return FakeStream(
            [("messages", FakeMessageChunk(["ok"]))],
            {"messages": [{"role": "assistant", "content": "ok"}]},
        )


class GraphRecursionError(Exception):
    pass


class FakeRecursionStream:
    output = None

    def interleave(self, *names):
        raise GraphRecursionError("Recursion limit of 16 reached")


class FakeRecursionAgent:
    def stream_events(self, payload, config=None, version="v3"):
        return FakeRecursionStream()


class OpenAIInvalidRequestError(Exception):
    pass


class FakeBadRequestStream:
    output = None

    def interleave(self, *names):
        raise OpenAIInvalidRequestError("Model does not exist. Please check it carefully.")


class FakeBadRequestAgent:
    def stream_events(self, payload, config=None, version="v3"):
        return FakeBadRequestStream()


def test_langchain_runtime_streams_and_records_history(monkeypatch):
    """LangChain runtime 应输出 delta、通知工具，并保持可落盘 messages。"""
    install_fake_langchain_tools(monkeypatch)
    deltas = []
    events = []

    runtime = LangChainAgentRuntime(
        api_key="sk-test",
        model="gpt-test",
        base_url="https://example.test",
        tools=registry_with_echo(),
        system_prompt="你是面试官",
        model_factory=lambda: object(),
        agent_factory=lambda model, tools, system_prompt: FakeAgent(tools),
    )

    answer = runtime.run(
        "看看项目",
        on_delta=deltas.append,
        on_tool_call=lambda name, args, phase, result: events.append(
            (name, phase, args, result)
        ),
    )

    assert answer == "最终：echo: 项目"
    assert deltas == ["最终：", "echo: 项目"]
    assert [event[1] for event in events] == ["start", "end"]
    assert runtime.messages[0]["role"] == "system"
    assert runtime.messages[-1] == {"role": "assistant", "content": answer}


def test_langchain_runtime_uses_wider_langgraph_recursion_limit(monkeypatch):
    """LangGraph super-step 上限不能直接等于 native max_steps 的两倍。"""
    install_fake_langchain_tools(monkeypatch)
    fake_agent = FakeConfigAgent()
    runtime = LangChainAgentRuntime(
        api_key="sk-test",
        model="gpt-test",
        base_url="https://example.test",
        tools=registry_with_echo(),
        system_prompt="你是面试官",
        max_steps=8,
        model_factory=lambda: object(),
        agent_factory=lambda model, tools, system_prompt: fake_agent,
    )

    runtime.run("看看项目")

    assert fake_agent.config["recursion_limit"] == 64


def test_langchain_runtime_recursion_limit_returns_fallback(monkeypatch):
    """LangGraph 递归上限应返回可读兜底，不把 traceback 直接抛给 UI。"""
    install_fake_langchain_tools(monkeypatch)
    deltas = []

    runtime = LangChainAgentRuntime(
        api_key="sk-test",
        model=" gpt-test ",
        base_url=" https://example.test ",
        tools=registry_with_echo(),
        system_prompt="你是面试官",
        model_factory=lambda: object(),
        agent_factory=lambda model, tools, system_prompt: FakeRecursionAgent(),
    )

    answer = runtime.run("看看项目", on_delta=deltas.append)

    assert "最大推理步数" in answer
    assert deltas == [answer]
    assert runtime.messages[-1] == {"role": "assistant", "content": answer}
    assert runtime._model == "gpt-test"  # noqa: SLF001
    assert runtime._base_url == "https://example.test"  # noqa: SLF001


def test_langchain_runtime_model_not_found_uses_friendly_error(monkeypatch):
    """LangChain 包装后的模型不存在错误应复用项目友好错误分类。"""
    install_fake_langchain_tools(monkeypatch)
    runtime = LangChainAgentRuntime(
        api_key="sk-test",
        model="missing-model",
        base_url="https://example.test",
        tools=registry_with_echo(),
        system_prompt="你是面试官",
        model_factory=lambda: object(),
        agent_factory=lambda model, tools, system_prompt: FakeBadRequestAgent(),
    )

    try:
        runtime.run("看看项目")
        assert False, "应该抛 LLMError"
    except LLMError as e:
        assert e.kind == ERROR_KIND_BAD_REQUEST
        assert "missing-model" in e.message
