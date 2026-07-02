"""协议层（设计第 1.4、1.5 节）。

负责 stdio 上的 JSON-RPC 收发与消息分帧。

约定（设计第 1.4.1 节）：
- stdin/stdout 是连续字节流，没有天然消息边界
- 约定"一行 = 一条消息"作为分帧方式（JSON 字符串内不含裸换行符）

两类消息（设计第 1.5.1 节）：
- Request（VS Code → Python）：init / chat / stop
- Notification（Python → VS Code）：stream / tool_call / done / error
"""

import json
import sys
import threading

# ──────────────────────────────────────────────
# 入站：解析 VS Code 发来的消息
# ──────────────────────────────────────────────


def parse_message(line: str) -> dict | None:
    """解析一行 JSON-RPC 消息（设计第 1.4.3 节）。

    对格式错误的输入容错：解析失败返回 None，而不是抛异常。
    这样"喂脏数据"不会让整个子进程崩溃。

    参数：
        line: stdin 读到的一行字符串（含尾随换行符也无所谓）

    返回：
        解析后的 dict（含 method/params 等字段），或 None（格式错误）
    """
    line = line.strip()
    if not line:
        return None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    # 一条合法消息至少要有 method 字段
    if not isinstance(msg, dict) or "method" not in msg:
        return None
    return msg


# ──────────────────────────────────────────────
# 出站：往 stdout 写通知（Python → VS Code）
# ──────────────────────────────────────────────


def _sanitize(text: str) -> str:
    """清除字符串里的孤立代理项（surrogate）。

    实现复用 agent_loop._sanitize_surrogates（同一套清理逻辑，避免重复）。
    """
    from agent.agent_loop import _sanitize_surrogates

    return _sanitize_surrogates(text)


def notify(method: str, params: dict) -> None:
    """往 stdout 写一行 JSON 通知（设计第 1.5.2 节）。

    每条通知一行，写完立刻 flush——流式输出必须实时推出，
    不能攒在缓冲区里（设计第 1.6 节 stdout 缓冲区陷阱）。

    参数：
        method: 通知类型（stream / tool_call / done / error）
        params: 通知参数
    """
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    # 序列化后清掉孤立代理项，防止 Windows 文件名含坏字符导致编码崩溃
    line = _sanitize(json.dumps(msg, ensure_ascii=False)) + "\n"
    # 多线程写 stdout（#8：chat 在 worker 线程，与主线程的 error 通知并发）：
    # 加锁防止两行的字节交错，破坏 TS 侧的按行分帧。
    with _stdout_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


# stdout 写入锁：notify 可能从 worker 线程（流式/工具通知）和主线程
# （error 通知）并发调用，必须串行化以保证"一行一条消息"分帧不被破坏（#8）。
_stdout_lock = threading.Lock()


# 以下是 4 种出站通知的便捷封装，对应设计第 1.5.2 节的消息清单。
# 封装出来是为了让 main.py 调用处语义清晰，也方便测试 mock。


def notify_stream(session: str, delta: str) -> None:
    """流式输出：LLM 吐一段文字就推一段（设计第 1.6 节）。"""
    notify("stream", {"session": session, "delta": delta})


def notify_tool_call(
    session: str,
    tool: str,
    phase: str,
    args: dict | None = None,
    result: str = "",
) -> None:
    """工具调用通知：让 UI 显示"正在搜代码"气泡（设计第 1.5.2 节）。

    参数：
        session: 会话 id
        tool:    工具名（search_code / read_file / ...）
        phase:   "start"（开始）或 "end"（结束）
        args:    工具参数（start 时用，end 时可省）
        result:  工具结果（end 时用，start 时为空）
    """
    params: dict = {"session": session, "tool": tool, "phase": phase}
    if args is not None:
        params["args"] = args
    if result:
        params["result"] = result
    notify("tool_call", params)


def notify_done(session: str) -> None:
    """本轮 Agent 循环结束（设计第 1.5.3 节时序末尾）。"""
    notify("done", {"session": session})


def notify_error(session: str, message: str) -> None:
    """错误通知：API 失效、网络断、工具报错（设计第 1.7 节）。"""
    notify("error", {"session": session, "message": message})


# ──────────────────────────────────────────────
# 消息分发器（设计第 1.4.3 节骨架的工程化）
# ──────────────────────────────────────────────


def handle_message(
    msg: dict,
    handlers: dict[str, callable],  # type: ignore[type-arg]
) -> None:
    """根据消息的 method 字段，调对应 handler。

    设计第 1.4.3 节的骨架是 `result = handle(msg)`，这里把它拆成
    "按 method 分发"——每种 method 一个 handler，注册进 handlers dict。

    参数：
        msg:      parse_message 解析出来的消息
        handlers: {method: handler_func(params) -> None}
                  method 有 init/chat/stop 三种（设计第 1.5.2 节）
                  未知 method 静默忽略（不崩溃，容错）
    """
    method = msg.get("method")
    params = msg.get("params", {}) or {}
    handler = handlers.get(method)
    if handler is None:
        # 未知 method：静默忽略，不杀进程
        return
    handler(params)
