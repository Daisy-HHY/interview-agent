"""对话历史管理（设计第 6.2 节）。

三层防线的第 2、3 层：摘要 + 硬裁剪。
防止工具结果（整份代码）撑爆对话历史。
"""



from copy import deepcopy

# 第 3 层硬裁剪的 token 上限（设计第 6.2.3 节）
# 预留空间给系统提示 + 新回复，所以设得比模型上限小很多
MAX_HISTORY_TOKENS = 20000


def count_tokens(messages: list[dict]) -> int:
    """粗略估算 messages 的 token 数（设计第 6.2.3 节）。

    MVP 用"字符数 / 3"粗估（中文约 1 字 = 1 token，英文约 4 字符 = 1 token，
    折中取 3）。准确估算要装 tiktoken，MVP 阶段粗估够用。
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        # tool_calls 也算（function calling 的结构化调用）
        if "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                total_chars += len(str(tc))
    return total_chars // 3


def compress_tool_result(msg: dict) -> dict:
    """把一个完整的工具结果压成摘要（设计第 6.2.2 节）。

    超过 200 字符的，保留头 100 字 + 省略提示 + 尾 50 字。
    让 LLM 知道"之前看过什么"，但不用带着完整内容。
    """
    content = msg.get("content", "")
    if len(content) <= 200:
        return msg
    summary = (
        content[:100]
        + f"\n...(已压缩，原 {len(content)} 字符)...\n"
        + content[-50:]
    )
    return {**msg, "content": summary}


def compress_history(
    messages: list[dict],
    max_tool_results_kept_full: int = 3,
) -> list[dict]:
    """第 2 层防线：老的 tool_result 压缩，保留最近 N 个完整。

    设计第 6.2.2 节：最近的 Observation 对当前决策最相关，
    老的细节 LLM 大概率用不上。
    """
    # 找出所有 tool 结果的位置（OpenAI 格式：role == "tool"）
    tool_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool"
    ]

    # 要压缩的：超出"保留最近 N 个"之外的较早结果
    if len(tool_indices) <= max_tool_results_kept_full:
        return messages  # 没超过，不用压缩
    # 注意：不能用 [:-0]，因为 -0 == 0，list[:-0] 是空列表（Python 经典坑）
    if max_tool_results_kept_full == 0:
        to_compress = set(tool_indices)
    else:
        to_compress = set(tool_indices[:-max_tool_results_kept_full])

    return [
        compress_tool_result(m) if i in to_compress else m
        for i, m in enumerate(messages)
    ]


def enforce_token_limit(
    messages: list[dict],
    max_tokens: int = MAX_HISTORY_TOKENS,
) -> list[dict]:
    """裁剪旧消息组，保留系统及当前用户轮次；最小上下文超限时显式拒绝。

    返回深拷贝，模型和请求转换不能改写业务存档。token 仍为字符估算值。
    """
    groups = message_groups(messages)
    current = next((i for i in range(len(groups) - 1, -1, -1)
                    if groups[i][0].get("role") == "user"), len(groups) - 1)
    total = count_tokens(messages)
    kept = []
    for index, group in enumerate(groups):
        if total > max_tokens and index < current and group[0].get("role") != "system":
            total -= count_tokens(group)
        else:
            kept.extend(group)
    if count_tokens(kept) > max_tokens:
        raise ValueError("当前问题及工具链的最小上下文超过预算，请缩短输入或提高历史上限。")
    return deepcopy(kept)


def message_groups(messages: list[dict]) -> list[list[dict]]:
    """按连续工具调用及全部结果分组；拒绝孤立、重复或缺失的工具结果。"""
    groups: list[list[dict]] = []
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("工具上下文存在孤立或重复结果，请使用完整会话。")
            groups[-1].append(message)
            pending.remove(call_id)
            continue
        if pending:
            raise ValueError("工具上下文缺少调用结果，请使用完整会话。")
        groups.append([message])
        calls = message.get("tool_calls") or []
        if calls:
            ids = [call.get("id") for call in calls]
            if role != "assistant" or not all(isinstance(i, str) and i for i in ids):
                raise ValueError("工具上下文的调用标识无效。")
            pending = set(ids)
            if len(pending) != len(ids):
                raise ValueError("工具上下文存在重复调用标识。")
    if pending:
        raise ValueError("工具上下文缺少调用结果，请使用完整会话。")
    return groups
