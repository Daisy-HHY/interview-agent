"""工具注册装配层。

runtime 只依赖 ToolRegistry；本模块负责决定有哪些工具、哪些被启用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.tools.base import Tool, ToolRegistry
from agent.tools.builtin import (
    ListDirectoryTool,
    LookupQuestionsTool,
    ReadFileTool,
    SearchCodeTool,
)


@dataclass(frozen=True)
class ToolSpec:
    """可注册工具的元信息和创建函数。"""

    name: str
    provider: str
    description: str
    risk_level: str
    enabled_by_default: bool
    factory: Callable[[str], Tool]


def builtin_tool_specs() -> list[ToolSpec]:
    """返回基础内置工具清单。"""
    return [
        ToolSpec(
            name="list_directory",
            provider="builtin",
            description="列出当前工作区内的目录结构。",
            risk_level="read",
            enabled_by_default=True,
            factory=lambda workspace: ListDirectoryTool(workspace),
        ),
        ToolSpec(
            name="search_code",
            provider="builtin",
            description="在当前工作区源码和文本文件中搜索关键词。",
            risk_level="read",
            enabled_by_default=True,
            factory=lambda workspace: SearchCodeTool(workspace),
        ),
        ToolSpec(
            name="read_file",
            provider="builtin",
            description="读取当前工作区内的单个文件内容。",
            risk_level="read",
            enabled_by_default=True,
            factory=lambda workspace: ReadFileTool(workspace),
        ),
        ToolSpec(
            name="lookup_questions",
            provider="builtin",
            description="按技术点读取内置面试追问题库。",
            risk_level="local",
            enabled_by_default=True,
            factory=lambda _workspace: LookupQuestionsTool(),
        ),
    ]


def available_tool_names() -> list[str]:
    """返回当前可用工具名，供诊断和 benchmark 输出。"""
    return [spec.name for spec in builtin_tool_specs()]


def default_enabled_tool_names() -> list[str]:
    """返回默认启用工具名。"""
    return [spec.name for spec in builtin_tool_specs() if spec.enabled_by_default]


def normalize_enabled_tools(enabled_tools: list[str] | None) -> list[str]:
    """过滤未知工具并去重；未配置时返回默认启用工具。"""
    available = set(available_tool_names())
    requested = enabled_tools if enabled_tools is not None else default_enabled_tool_names()
    normalized: list[str] = []
    for name in requested:
        if name in available and name not in normalized:
            normalized.append(name)
    return normalized


def build_tool_registry(
    workspace: str,
    enabled_tools: list[str] | None = None,
) -> ToolRegistry:
    """按启用清单装配工具注册表。"""
    enabled = set(normalize_enabled_tools(enabled_tools))
    registry = ToolRegistry()
    for spec in builtin_tool_specs():
        if spec.name in enabled:
            registry.register(spec.factory(workspace))
    return registry
