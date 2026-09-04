import pytest

from agent.mcp_adapter import (
    McpToolAdapter,
    McpToolCallResult,
    McpToolDefinition,
    register_mcp_tools,
)
from agent.tools.base import ToolRegistry


class FakeMcpAdapter:
    def __init__(self, definitions):
        self.definitions = definitions
        self.calls = []
        self.closed = False
        self.result = McpToolCallResult("fake result")

    def list_tools(self):
        return self.definitions

    def call_tool(self, name, arguments, should_cancel=None):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("cancelled")
        self.calls.append((name, arguments))
        return self.result

    def close(self):
        self.closed = True


def definition(name="remote_echo"):
    return McpToolDefinition(
        name=name,
        description="fake MCP tool",
        input_schema={"type": "object", "properties": {}},
    )


def test_register_mcp_tools_wraps_adapter_without_network():
    adapter = FakeMcpAdapter([definition()])
    registry = ToolRegistry()

    assert register_mcp_tools(registry, adapter) == ["remote_echo"]
    assert registry.get("remote_echo").execute() == "fake result"
    assert adapter.calls == [("remote_echo", {})]


def test_register_mcp_tools_rejects_duplicate_names():
    adapter = FakeMcpAdapter([definition(), definition()])

    with pytest.raises(ValueError, match="重复"):
        register_mcp_tools(ToolRegistry(), adapter)


def test_mcp_adapter_contract_has_no_default_remote_execution():
    adapter = FakeMcpAdapter([definition()])
    assert isinstance(adapter, McpToolAdapter)
    adapter.close()
    assert adapter.closed is True


def test_mcp_adapter_limits_error_and_oversized_results():
    adapter = FakeMcpAdapter([definition()])
    adapter.result = McpToolCallResult("x" * 20_001, is_error=True, error_kind="timeout")
    registry = ToolRegistry()
    register_mcp_tools(registry, adapter)

    result = registry.get("remote_echo").execute()

    assert result.startswith("MCP 工具错误（timeout）：")
    assert len(result) < 20_100


def test_mcp_adapter_can_cancel_before_provider_call():
    adapter = FakeMcpAdapter([definition()])

    with pytest.raises(RuntimeError, match="cancelled"):
        adapter.call_tool("remote_echo", {}, should_cancel=lambda: True)

    assert adapter.calls == []
