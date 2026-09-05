"""真实临时文件验证读取权限和资源上限。"""

from unittest.mock import mock_open, patch

import pytest

from agent.tools.builtin import ListDirectoryTool, ReadFileTool, SearchCodeTool


@pytest.mark.parametrize("name", [".env", ".env.production", "id_rsa", "private.pem",
                                   ".sessions/s1.json", ".interview-agent/reports/a.md"])
def test_sensitive_files_never_enter_tools(tmp_path, name):
    """同一敏感文件不能通过读取、搜索或目录入口绕过。"""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("SYNTHETIC_SECRET", encoding="utf-8")
    with pytest.raises(ValueError, match="敏感"):
        ReadFileTool(str(tmp_path)).execute(name)
    assert "SYNTHETIC_SECRET" not in SearchCodeTool(str(tmp_path)).execute("SYNTHETIC")
    assert path.name not in ListDirectoryTool(str(tmp_path)).execute(".")


def test_search_cap_is_global(tmp_path):
    """跨目录命中总数也必须不超过二十。"""
    for folder in ["a", "b", "c/deep"]:
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        (directory / "code.py").write_text("needle\n" * 25)
    result = SearchCodeTool(str(tmp_path)).execute("needle")
    assert sum(": needle" in line for line in result.splitlines()) == 20


def test_empty_search_is_rejected(tmp_path):
    """空关键词不触发全仓扫描。"""
    assert "错误" in SearchCodeTool(str(tmp_path)).execute("  ")


def test_read_has_byte_bound_even_for_single_line(tmp_path):
    """读取计数证明不是先载入全文，再截断输出。"""
    path = tmp_path / "large.py"
    path.write_text("x" * 2_000_000)
    fake = mock_open(read_data=b"x" * 65_537)
    with patch("builtins.open", fake):
        result = ReadFileTool(str(tmp_path)).execute("large.py")
    fake().read.assert_called_once_with(65_537)
    fake().readlines.assert_not_called()
    assert len(result) < 66_000
    assert "已截断" in result


def test_source_and_hidden_public_config_stay_readable(tmp_path):
    """公开配置不因为是隐藏目录或 JSON 而被拒绝。"""
    directory = tmp_path / ".github"
    directory.mkdir()
    (directory / "config.json").write_text('{"public":true}')
    assert "public" in ReadFileTool(str(tmp_path)).execute(".github/config.json")


def test_search_link_cannot_leave_workspace(tmp_path):
    """搜索也检查链接真实目标；不具备链接权限时只跳过该平台用例。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("SYNTHETIC_OUTSIDE")
    try:
        (workspace / "link.py").symlink_to(outside)
    except OSError:
        pytest.skip("当前 Windows 进程不具备创建符号链接权限")
    assert "SYNTHETIC_OUTSIDE" not in SearchCodeTool(str(workspace)).execute("SYNTHETIC")
    with pytest.raises(ValueError, match="路径越界"):
        ReadFileTool(str(workspace)).execute("link.py")
