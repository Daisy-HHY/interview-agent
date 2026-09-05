"""0.2.4 长历史与磁盘失败回归，使用合成数据，不访问真实会话。"""

import json
from copy import deepcopy

import pytest

from agent.agent_loop import AgentLoop
from agent.history import count_tokens, enforce_token_limit
from agent.llm_client import FakeLLM, make_text_response, make_tool_call_response
from agent.pi_runtime import PiAgentRuntime
from agent.session import SessionStore
from agent.tools.base import ToolRegistry


def tool_group():
    """构造两个调用及其完整结果。"""
    calls = make_tool_call_response("read_file", {"path": "a.py"}).tool_calls
    calls += make_tool_call_response("search_code", {"keyword": "test"}).tool_calls
    return [{"role": "assistant", "content": "", "tool_calls": calls}] + [
        {"role": "tool", "tool_call_id": call["id"], "content": "x" * 300}
        for call in calls
    ]


def test_trim_drops_complete_old_tool_group():
    """旧工具组整体裁剪，不留下孤立结果。"""
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "old"}] + tool_group() + [
                    {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": "current"}]
    original = deepcopy(messages)
    trimmed = enforce_token_limit(messages, 15)
    assert not any(m["role"] == "tool" for m in trimmed)
    assert trimmed[-1]["content"] == "current"
    assert count_tokens(trimmed) <= 15
    assert messages == original


def test_current_user_and_tool_group_cannot_be_silently_discarded():
    """本轮最小上下文仍超限时应拒绝请求。"""
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "current"}] + tool_group()
    with pytest.raises(ValueError, match="上下文"):
        enforce_token_limit(messages, 5)


@pytest.mark.parametrize("runtime", [AgentLoop, PiAgentRuntime])
def test_runtime_preserves_archive_and_isolates_model_mutation(runtime):
    """模型获得裁剪副本，不能通过可变参数改写原始存档。"""
    fake = FakeLLM([make_text_response("answer")])
    seen = []
    original_chat = fake.chat

    def chat(messages, tools, **kwargs):
        """保存请求快照并模拟调用边界修改参数。"""
        seen.append(deepcopy(messages))
        messages[0]["content"] = "mutated"
        return original_chat(messages, tools, **kwargs)

    fake.chat = chat
    loop = runtime(fake, ToolRegistry(), "sys", max_history_tokens=100)
    old = [{"role": "system", "content": "sys"},
           {"role": "user", "content": "JD-ORIGINAL " * 300},
           {"role": "assistant", "content": "previous"}]
    loop.restore_messages(deepcopy(old))
    loop.run("current")
    assert loop.messages[:3] == old
    assert count_tokens(seen[0]) <= 100


@pytest.mark.parametrize("failure", ["serialize", "write", "replace"])
def test_failed_save_preserves_previous_file(tmp_path, monkeypatch, failure):
    """序列化、写入和原子替换失败都不得损坏旧文件。"""
    store = SessionStore(llm_factory=lambda: FakeLLM([make_text_response("answer")]))
    store.configure(str(tmp_path), "fake")
    loop = store.get_or_create("s1")
    loop.run("original")
    store.save("s1")
    path = tmp_path / ".sessions" / "s1.json"
    before = path.read_bytes()
    loop.messages.append({"role": "user", "content": "new"})

    def fail(*args, **kwargs):
        """模拟本地写入失败。"""
        if failure == "write":
            args[1].write("partial")
        raise OSError("synthetic disk failure")

    monkeypatch.setattr("agent.session.os.replace" if failure == "replace"
                        else "agent.session.json.dump", fail)
    with pytest.raises(OSError):
        store.save("s1")
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("data", ['{broken', '{}', 'null', '[null]', '[]',
                                   '[{"role":"user","content":"wrong first"}]'])
def test_bad_session_cannot_be_overwritten(tmp_path, data):
    """损坏与错误形状都应显式拒绝恢复，不创建可覆盖原文件的新循环。"""
    directory = tmp_path / ".sessions"
    directory.mkdir()
    path = directory / "s1.json"
    path.write_text(data, encoding="utf-8")
    store = SessionStore(llm_factory=lambda: FakeLLM([]))
    store.configure(str(tmp_path), "fake")
    with pytest.raises(ValueError, match="会话"):
        store.get_or_create("s1")
    store.save("s1")
    assert path.read_text(encoding="utf-8") == data


def test_archive_survives_trim_save_and_restart(tmp_path):
    """真实临时文件往返，原始背景不会随模型请求裁剪消失。"""
    store = SessionStore(llm_factory=lambda: FakeLLM([make_text_response("answer")]))
    store.configure(str(tmp_path), "fake", max_history_tokens=2000)
    loop = store.get_or_create("s1")
    loop.messages.extend([{"role": "user", "content": "JD-RESUME " * 2000},
                          {"role": "assistant", "content": "old answer"}])
    loop.run("next")
    store.save("s1")
    new = SessionStore(llm_factory=lambda: FakeLLM([]))
    new.configure(str(tmp_path), "fake")
    restored = new.get_or_create("s1").messages
    assert restored[1]["content"] == "JD-RESUME " * 2000
    assert restored == json.loads((tmp_path / ".sessions/s1.json").read_text("utf-8"))
