from typing import Any, Protocol


class Tool(Protocol):
    """所有 runtime 共用的最小工具接口。

    无论工具来源是内置实现、LangChain adapter 还是未来 MCP adapter，
    runtime 只依赖这三个契约：
    - name: 工具名，必须与 schema.function.name 一致。
    - schema: OpenAI function calling 格式的参数 schema。
    - execute: 接收 JSON schema 对应的 kwargs，返回可喂给 LLM 的字符串。
    """

    @property
    def name(self) -> str: ...

    @property
    def schema(self) -> dict: ...

    def execute(self, **kwargs: Any) -> str: ...


class ToolRegistry:
    """工具注册表。Agent 循环只跟它打交道，不关心工具是内置还是 MCP。

    核心职责：
    - 存工具（register）
    - 取工具（get）
    - 给 LLM 列工具（all_schemas）
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具。重复注册会覆盖（方便测试时替换）。"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """按名字取工具。不存在返回 None（Agent 循环要处理这种情况）。"""
        return self._tools.get(name)

    def all_schemas(self) -> list[dict]:
        """组装喂给 OpenAI 的 tools 参数。注册了几个就有几个。"""
        return [tool.schema for tool in self._tools.values()]
