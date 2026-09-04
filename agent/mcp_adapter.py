"""MCP-style 工具 adapter 合同。

本模块只提供本地接口和 ToolRegistry wrapper，不建立网络连接、不引入 MCP 依赖。
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent.tools.base import ToolRegistry

MAX_MCP_RESULT_CHARS = 20_000


@dataclass(frozen=True)
class McpToolDefinition:
    """provider 暴露给 runtime 的工具描述。"""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class McpToolCallResult:
    """provider 调用结果，正文只在本地工具 wrapper 内使用。"""

    content: str
    is_error: bool = False
    error_kind: str | None = None
    retryable: bool = False


@runtime_checkable
class McpToolAdapter(Protocol):
    """未来 provider 必须实现的最小生命周期合同。"""

    def list_tools(self) -> list[McpToolDefinition]: ...

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        should_cancel: Any = None,
    ) -> McpToolCallResult: ...

    def close(self) -> None: ...


class _McpTool:
    """把显式注册的 adapter 工具包装成现有 Tool 契约。"""

    def __init__(self, adapter: McpToolAdapter, definition: McpToolDefinition) -> None:
        self._adapter = adapter
        self._definition = definition
        self.name = definition.name

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._definition.description,
                "parameters": self._definition.input_schema,
            },
        }

    def execute(self, **kwargs: Any) -> str:
        """显式调用 provider 工具，并限制返回正文大小。"""
        result = self._adapter.call_tool(self.name, kwargs)
        if not isinstance(result, McpToolCallResult):
            raise TypeError("MCP adapter 必须返回 McpToolCallResult")
        content = str(result.content or "")
        if len(content) > MAX_MCP_RESULT_CHARS:
            content = content[:MAX_MCP_RESULT_CHARS] + "\n...(MCP 结果已截断)..."
        if result.is_error:
            prefix = result.error_kind or "provider"
            return f"MCP 工具错误（{prefix}）：{content}"
        return content


def register_mcp_tools(registry: ToolRegistry, adapter: McpToolAdapter) -> list[str]:
    """将显式 provider 描述注册到现有 ToolRegistry，不自动发现远程 provider。"""
    definitions = adapter.list_tools()
    names: set[str] = set()
    for definition in definitions:
        if not isinstance(definition, McpToolDefinition):
            raise TypeError("MCP 工具描述必须是 McpToolDefinition")
        if not definition.name or definition.name in names or registry.get(definition.name):
            raise ValueError(f"MCP 工具名重复或为空：{definition.name}")
        if not isinstance(definition.input_schema, dict):
            raise ValueError(f"MCP 工具 schema 非法：{definition.name}")
        names.add(definition.name)

    for definition in definitions:
        registry.register(_McpTool(adapter, definition))
    return [definition.name for definition in definitions]
