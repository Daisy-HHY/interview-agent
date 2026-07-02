"""历史管理测试（设计第 7.2.1 节，第 1 层纯逻辑测试）。

测试目标：覆盖 compress_history 和 enforce_token_limit 的边界情况。
关键覆盖点：
- 摘要保留最近 N 个完整（设计 6.2.2）
- 系统提示和最新消息永不丢（设计 6.2.3）★ 核心安全设计
- 短结果不压缩、无工具结果时不处理
- token 超限时从最老的开始丢
"""

from agent.history import (
    compress_history,
    compress_tool_result,
    count_tokens,
    enforce_token_limit,
)

# ──────────────────────────────────────────────
# 辅助函数：构造测试用的消息
# ──────────────────────────────────────────────


def _system(msg: str) -> dict:
    return {"role": "system", "content": msg}


def _user(msg: str) -> dict:
    return {"role": "user", "content": msg}


def _assistant(msg: str) -> dict:
    return {"role": "assistant", "content": msg}


def _tool(content: str) -> dict:
    """构造一个 tool 结果消息（role=="tool"）"""
    return {"role": "tool", "content": content}


# ──────────────────────────────────────────────
# count_tokens 测试
# ──────────────────────────────────────────────


class TestCountTokens:
    def test_empty_list(self):
        assert count_tokens([]) == 0

    def test_single_message(self):
        # 6 字符 / 3 = 2 tokens
        assert count_tokens([_user("abcdef")]) == 2

    def test_multiple_messages_accumulate(self):
        messages = [_user("abc"), _assistant("def")]  # 共 6 字符
        assert count_tokens(messages) == 2


# ──────────────────────────────────────────────
# compress_tool_result 测试（单条压缩）
# ──────────────────────────────────────────────


class TestCompressToolResult:
    def test_short_content_unchanged(self):
        """短内容不应被压缩。"""
        msg = _tool("短内容，不压缩")
        result = compress_tool_result(msg)
        assert result["content"] == "短内容，不压缩"

    def test_long_content_compressed(self):
        """超过 200 字符的内容应被压缩。"""
        long_content = "x" * 500
        msg = _tool(long_content)
        result = compress_tool_result(msg)

        assert "已压缩" in result["content"]
        assert "500" in result["content"]  # 提示原长度
        assert len(result["content"]) < 500  # 确实变短了

    def test_preserves_head_and_tail(self):
        """压缩后保留头 100 字符和尾 50 字符（让 LLM 知道看过什么）。"""
        head = "HEAD" + "a" * 96   # 头部标记 + 凑到 100 字符
        middle = "M" * 300
        tail = "b" * 46 + "TAIL"   # 尾部凑到 50 字符
        msg = _tool(head + middle + tail)

        result = compress_tool_result(msg)
        assert "HEAD" in result["content"]  # 头部保留
        assert "TAIL" in result["content"]  # 尾部保留

    def test_original_message_not_mutated(self):
        """压缩不应修改原消息（不可变，方便测试）。"""
        long_content = "x" * 500
        msg = _tool(long_content)
        original_content = msg["content"]

        compress_tool_result(msg)
        assert msg["content"] == original_content  # 原消息没变


# ──────────────────────────────────────────────
# compress_history 测试（批量压缩）
# ──────────────────────────────────────────────


class TestCompressHistory:
    def test_no_tool_results_unchanged(self):
        """没有工具结果时，历史原样返回。"""
        messages = [_system("sys"), _user("hi"), _assistant("hello")]
        result = compress_history(messages)
        assert result == messages

    def test_fewer_than_threshold_not_compressed(self):
        """工具结果数 <= 阈值时，不压缩（保留全部完整）。"""
        messages = [
            _system("sys"),
            _user("q1"),
            _tool("结果1" * 100),  # 1 个工具结果
            _assistant("a1"),
        ]
        result = compress_history(messages, max_tool_results_kept_full=3)
        assert result == messages  # 原样返回

    def test_keeps_recent_n_full(self):
        """超过阈值时，保留最近 N 个完整，老的压缩。"""
        long = "x" * 500
        messages = [
            _system("sys"),
            _tool(long + "_OLD1"),   # 老的，应被压缩
            _tool(long + "_OLD2"),   # 老的，应被压缩
            _tool(long + "_RECENT1"),  # 最近的，应完整
            _tool(long + "_RECENT2"),  # 最近的，应完整
            _tool(long + "_RECENT3"),  # 最近的，应完整
            _user("question"),
        ]

        result = compress_history(messages, max_tool_results_kept_full=3)

        # 最近 3 个保持完整
        assert result[3]["content"] == long + "_RECENT1"
        assert result[4]["content"] == long + "_RECENT2"
        assert result[5]["content"] == long + "_RECENT3"

        # 老 2 个被压缩
        assert "已压缩" in result[1]["content"]
        assert "已压缩" in result[2]["content"]

    def test_non_tool_messages_not_compressed(self):
        """非 tool 消息（system/user/assistant）永远不被压缩。"""
        long = "x" * 500
        messages = [
            _system(long),       # system，不应被压缩
            _user(long),         # user，不应被压缩
            _tool(long),         # tool，可能被压缩
            _assistant(long),    # assistant，不应被压缩
        ]
        result = compress_history(messages, max_tool_results_kept_full=0)

        # 只有 tool 消息（index 2）被压缩
        assert result[0]["content"] == long  # system 完整
        assert result[1]["content"] == long  # user 完整
        assert "已压缩" in result[2]["content"]  # tool 被压缩
        assert result[3]["content"] == long  # assistant 完整


# ──────────────────────────────────────────────
# enforce_token_limit 测试（硬裁剪）
# ──────────────────────────────────────────────


class TestEnforceTokenLimit:
    def test_under_limit_unchanged(self):
        """token 数低于上限时，原样返回。"""
        messages = [_system("short"), _user("hi")]
        result = enforce_token_limit(messages, max_tokens=10000)
        assert result == messages

    def test_drops_oldest_middle_messages(self):
        """超过上限时，从最老的中间消息开始丢。"""
        # 每条 100 字符，10 条 = 1000 字符 ≈ 333 tokens
        big = "x" * 100
        messages = [
            _system("sys"),         # [0] 永远保留
            _assistant(big),        # [1] 应被丢
            _assistant(big),        # [2] 应被丢
            _assistant(big),        # [3] 可能被丢
            _user("latest"),        # [-1] 永远保留
        ]
        # 设个很小的上限，强制裁剪
        result = enforce_token_limit(messages, max_tokens=50)

        assert result[0] == _system("sys")        # 系统提示保留
        assert result[-1] == _user("latest")      # 最新消息保留
        assert len(result) < len(messages)        # 确实裁剪了

    def test_system_prompt_never_dropped(self):
        """★ 核心安全设计：系统提示永不丢（Agent 灵魂）。"""
        big = "x" * 500
        messages = [
            _system("重要系统提示，不能丢"),
            _user(big),
            _assistant(big),
            _user(big),
        ]
        result = enforce_token_limit(messages, max_tokens=10)
        assert result[0]["content"] == "重要系统提示，不能丢"

    def test_latest_message_never_dropped(self):
        """★ 核心安全设计：最新用户消息永不丢（当前要回答的问题）。"""
        big = "x" * 500
        messages = [
            _system("sys"),
            _user(big),
            _assistant(big),
            _user("这是最新问题，必须保留"),
        ]
        result = enforce_token_limit(messages, max_tokens=10)
        assert result[-1]["content"] == "这是最新问题，必须保留"

    def test_keeps_system_and_latest_at_minimum(self):
        """裁剪至少保留 system 和最新消息（不删光、不破坏配对）。

        旧 pop(1) 实现因 `len(result) > 4` 的停止条件硬保留 4 条（魔数副产物）；
        配对感知裁剪改为"按组删、保护首末"，下限是首组（system）+ 末组（最新），
        最少 2 条。核心保证是不删光、不破坏 tool_calls↔tool 配对。
        """
        big = "x" * 1000
        messages = [_system("s"), _user(big), _user(big), _user(big), _user(big), _user("last")]
        result = enforce_token_limit(messages, max_tokens=5)

        assert result[0] == _system("s")       # 系统提示保留
        assert result[-1] == _user("last")     # 最新消息保留
        assert len(result) >= 2                # 不删光
        assert not _has_orphan_tool(result)    # 无配对破坏（此例无 tool）

    def test_original_list_not_mutated(self):
        """裁剪不应修改原 list。"""
        messages = [_system("s"), _user("x" * 500), _user("q")]
        original_len = len(messages)
        enforce_token_limit(messages, max_tokens=10)
        assert len(messages) == original_len  # 原 list 没变


# ──────────────────────────────────────────────
# tool_calls ↔ tool 配对保护（critical 修复）
#
# OpenAI 要求每个 assistant(tool_calls) 紧跟对应 role=tool，
# 反之每个 role=tool 前必须有带 matching tool_call_id 的 assistant。
# 裁剪破坏任一方 → 400。旧 pop(1) 盲删会破坏配对。
# ──────────────────────────────────────────────


def _tc_msg(call_id: str) -> dict:
    """带 tool_calls 的 assistant 消息（OpenAI 格式）。"""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": "x", "arguments": "{}"}}]}


def _tool_msg(call_id: str) -> dict:
    """role=tool 的工具结果消息（带 tool_call_id，OpenAI 格式）。"""
    return {"role": "tool", "tool_call_id": call_id, "content": "结果"}


def _has_orphan_tool(messages: list[dict]) -> bool:
    """是否有孤立的 role=tool（前面缺带 tool_calls 的 assistant）。"""
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        j = i - 1
        while j >= 0 and messages[j].get("role") == "tool":
            j -= 1
        prev = messages[j] if j >= 0 else None
        if not (prev and prev.get("role") == "assistant" and prev.get("tool_calls")):
            return True
    return False


def _has_orphan_tool_calls(messages: list[dict]) -> bool:
    """是否有 assistant(tool_calls) 后面缺对应 role=tool。"""
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        ids = {tc.get("id") for tc in m["tool_calls"]}
        seen = set()
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            seen.add(messages[j].get("tool_call_id"))
            j += 1
        if not ids.issubset(seen):
            return True
    return False


class TestToolPairingPreservation:
    """★ critical：裁剪绝不能破坏 assistant(tool_calls) ↔ role=tool 配对。"""

    def test_no_orphan_tool_after_trim(self):
        """裁剪不能留下孤立的 role=tool（删了它对应的 assistant(tool_calls)）。

        旧 pop(1) 盲删这个用例会删掉 assistant(tool_calls)，留下孤立 tool。
        """
        messages = [
            _system("sys"),
            _tc_msg("call_X"),       # 带 tool_calls
            _tool_msg("call_X"),     # 对应 tool
            _assistant("中间回答"),
            _user("最新问题"),
        ]
        result = enforce_token_limit(messages, max_tokens=1)

        assert not _has_orphan_tool(result), f"留下孤立 tool: {result}"
        assert not _has_orphan_tool_calls(result), f"留下孤立 tool_calls: {result}"

    def test_multi_turn_tool_history_stays_paired(self):
        """多轮工具历史裁剪后，保留的部分配对仍完整。"""
        messages = [
            _system("sys"),
            _tc_msg("call_1"), _tool_msg("call_1"),
            _assistant("第一轮回答"),
            _tc_msg("call_2"), _tool_msg("call_2"),
            _assistant("第二轮回答"),
            _user("最新问题"),
        ]
        result = enforce_token_limit(messages, max_tokens=1)

        assert not _has_orphan_tool(result), f"留下孤立 tool: {result}"
        assert not _has_orphan_tool_calls(result), f"留下孤立 tool_calls: {result}"

    def test_system_and_latest_always_kept(self):
        """裁剪后 system 仍在首、最新消息仍在尾。"""
        messages = [
            _system("重要系统提示"),
            _tc_msg("call_X"), _tool_msg("call_X"),
            _user("最新问题"),
        ]
        result = enforce_token_limit(messages, max_tokens=1)

        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "最新问题"

    def test_unpaired_history_not_broken(self):
        """无 tool 的普通历史裁剪仍正常工作（不误伤）。"""
        messages = [
            _system("sys"),
            _user("x" * 500),
            _assistant("y" * 500),
            _user("最新"),
        ]
        result = enforce_token_limit(messages, max_tokens=5)

        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "最新"
        assert not _has_orphan_tool(result)
