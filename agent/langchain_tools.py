"""把现有 ToolRegistry 适配成 LangChain tools。"""

from typing import Any

from agent.llm_client import AgentCancelled
from agent.runtime import CancelCallback, ToolCallCallback
from agent.tools.base import ToolRegistry


def build_langchain_tools(
    registry: ToolRegistry,
    on_tool_call: ToolCallCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> list[Any]:
    """把 OpenAI function schema 风格工具包装成 LangChain 工具。

    参数：
        registry: 当前项目已有工具注册表。
        on_tool_call: 工具开始/结束通知回调，沿用现有思考球展示。
        should_cancel: 停止按钮状态检查。

    返回：LangChain `create_agent` 可接收的工具列表。
    """
    try:
        from langchain.tools import tool as langchain_tool
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("langchain"):
            raise RuntimeError(_missing_langchain_message()) from e
        raise

    tools = []
    for schema in registry.all_schemas():
        function = schema.get("function", {})
        name = function.get("name", "")
        if not name:
            continue
        description = function.get("description", name)
        args_schema = function.get("parameters") or {"type": "object", "properties": {}}
        tools.append(
            langchain_tool(
                name,
                description=description,
                args_schema=args_schema,
            )(_make_tool_runner(registry, name, on_tool_call, should_cancel))
        )
    return tools


def _make_tool_runner(
    registry: ToolRegistry,
    name: str,
    on_tool_call: ToolCallCallback | None,
    should_cancel: CancelCallback | None,
):
    def run_tool(**kwargs: Any) -> str:
        """执行 Interview Agent 内置工具。"""
        if should_cancel is not None and should_cancel():
            raise AgentCancelled()

        if on_tool_call:
            on_tool_call(name, kwargs, "start", "")

        result = _safe_execute(registry, name, kwargs)

        if on_tool_call:
            on_tool_call(name, kwargs, "end", result)

        if should_cancel is not None and should_cancel():
            raise AgentCancelled()
        return result

    run_tool.__name__ = name
    return run_tool


def _safe_execute(registry: ToolRegistry, name: str, args: dict[str, Any]) -> str:
    tool = registry.get(name)
    if tool is None:
        return f"错误：不存在名为 '{name}' 的工具。"
    try:
        return str(tool.execute(**args))
    except Exception as e:
        return f"工具执行出错: {type(e).__name__}: {e}"


def _missing_langchain_message() -> str:
    return (
        "当前 Python 环境未安装 LangChain 运行时依赖。"
        "请安装 agent-framework 可选依赖，或将 interview.agentRuntime 改回 native。"
    )

