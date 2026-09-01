"""工具注册装配层测试。"""

from agent.tools.registry import (
    available_tool_names,
    build_tool_registry,
    default_enabled_tool_names,
    normalize_enabled_tools,
)


def test_build_tool_registry_uses_default_tools(tmp_path):
    """未配置时注册默认基础工具。"""
    registry = build_tool_registry(str(tmp_path))

    schemas = {schema["function"]["name"] for schema in registry.all_schemas()}
    assert schemas == {
        "list_directory",
        "search_code",
        "read_file",
        "lookup_questions",
    }


def test_build_tool_registry_filters_enabled_tools(tmp_path):
    """配置启用清单时，只注册指定工具。"""
    registry = build_tool_registry(
        str(tmp_path),
        enabled_tools=["list_directory", "search_code"],
    )

    schemas = {schema["function"]["name"] for schema in registry.all_schemas()}
    assert schemas == {"list_directory", "search_code"}


def test_normalize_enabled_tools_ignores_unknown_and_deduplicates():
    """未知工具忽略，重复工具去重。"""
    assert normalize_enabled_tools(["read_file", "ghost", "read_file"]) == ["read_file"]


def test_available_and_default_tool_names_match_builtin_tools():
    """当前默认工具均来自可用工具清单。"""
    assert set(default_enabled_tool_names()).issubset(set(available_tool_names()))
