"""Agent 内核入口（设计第 1.4.3 节骨架 + 第 5D 节）。

这是 Python 子进程的 main——读 stdin 一行行 JSON-RPC，分发到 handler，
把 Agent 循环的结果通过 stdout 通知推回去。

设计第 1.4.3 节骨架：
    for line in sys.stdin:
        msg = json.loads(line)
        result = handle(msg)
        sys.stdout.write(json.dumps(result) + "\\n")
        sys.stdout.flush()

把"handle"拆成按 method 分发（init/chat/stop），三种 Request 各自的处理逻辑。
"""

import sys

# Windows 默认 stdout/stderr/stdin 用系统编码（GBK/cp936），会导致中文输出
# 在某些字符上抛 UnicodeEncodeError。强制改为 UTF-8，与协议约定一致
# （设计第 1.6 节：stdio 用 UTF-8 JSON）。
# stdin 尤其关键：Node 侧写入的是 UTF-8，若按 GBK 解码，用户输入的中文
# 会变成乱码进入对话历史并落盘（历史会话标题乱码的根因）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

import agent.protocol as protocol
from agent.llm_client import AgentCancelled
from agent.session import SessionStore


def main(
    stream=None,
    store: SessionStore | None = None,
) -> None:
    """主循环：读 stdin，分发消息，输出通知。

    参数：
        stream: 输入流（默认 sys.stdin）。测试可传 io.StringIO。
        store:  会话仓库（默认新建）。测试可注入带 FakeLLM 的 store。

    退出条件：stdin 关闭（读到 EOF）即结束——VS Code 关闭时 stdin 会关。
    """
    if stream is None:
        stream = sys.stdin
    if store is None:
        store = SessionStore()

    import threading

    active_thread: threading.Thread | None = None
    cancel_event = threading.Event()

    def is_busy() -> bool:
        return active_thread is not None and active_thread.is_alive()

    def handle_init(params: dict) -> None:
        if is_busy():
            protocol.notify_error(
                params.get("session", "unknown"),
                "当前回答仍在生成，请先停止当前回答后再调整配置。",
            )
            return
        _handle_init(store, params)

    def handle_chat(params: dict) -> None:
        nonlocal active_thread, cancel_event
        session = params.get("session", "default")
        if is_busy():
            protocol.notify_error(session, "请先停止当前回答后再发送新消息。")
            return
        cancel_event = threading.Event()

        def worker() -> None:
            try:
                _handle_chat(store, params, should_cancel=cancel_event.is_set)
            except Exception as e:
                protocol.notify_error(session, f"内部错误: {type(e).__name__}: {e}")

        active_thread = threading.Thread(target=worker, daemon=True)
        active_thread.start()

    def handle_stop(params: dict) -> None:
        _handle_stop(cancel_event, params)

    handlers = {
        "init": handle_init,
        "chat": handle_chat,
        "stop": handle_stop,
    }

    for line in stream:
        msg = protocol.parse_message(line)
        if msg is None:
            # 格式错误：静默跳过（设计第 1.7 节容错，不杀进程）
            continue
        try:
            protocol.handle_message(msg, handlers)
        except Exception as e:
            # 业务层错误：发 error 通知，不杀进程（设计第 6.4.2 节）
            session = msg.get("params", {}).get("session", "unknown")
            protocol.notify_error(session, f"内部错误: {type(e).__name__}: {e}")

    if active_thread is not None:
        active_thread.join()


# ──────────────────────────────────────────────
# 三种 Request 的 handler
# ──────────────────────────────────────────────


def _handle_init(store: SessionStore, params: dict) -> None:
    """处理 init 消息：记录 workspace / api_key / model（设计第 1.5.2 节）。

    init 是会话开始时发一次的消息，给 Python 工作区路径和配置。
    """
    workspace = params.get("workspace")
    api_key = params.get("api_key")
    model = params.get("model", "gpt-4o-mini")
    base_url = params.get("base_url")
    resume = params.get("resume")
    # 调优参数（Phase 7-D，可选；TS 侧传整数，不传则 None 用默认）
    max_steps = params.get("max_steps")
    max_history_tokens = params.get("max_history_tokens")
    max_kept_full = params.get("max_kept_full")
    agent_runtime = params.get("agent_runtime", "native")

    if not workspace or not api_key:
        protocol.notify_error(
            params.get("session", "unknown"),
            "init 缺少必要参数（workspace 或 api_key）",
        )
        return

    store.configure(
        workspace=workspace,
        api_key=api_key,
        model=model,
        base_url=base_url,
        resume=resume,
        max_steps=max_steps,
        max_history_tokens=max_history_tokens,
        max_kept_full=max_kept_full,
        agent_runtime=agent_runtime,
    )


def _handle_chat(
    store: SessionStore,
    params: dict,
    should_cancel=None,
) -> None:
    """处理 chat 消息：跑一轮 Agent 循环（设计第 1.5.3 节时序）。

    chat 是最常用的消息——用户说的话。整个 Agent 循环在这里触发：
    1. 取（或新建）session 的 AgentLoop
    2. 跑 run()，回调把工具调用/回答转成通知
    3. 落盘历史（设计第 6.4.3 节）
    4. 发 done 通知
    """
    session = params.get("session", "default")
    text = params.get("text", "")

    # 拼上选中代码（设计第 5.3.3 节 attached_code）
    attached = params.get("attached_code")
    if attached and isinstance(attached, dict) and attached.get("content"):
        text = _inject_attached_code(text, attached)

    if not store.is_configured:
        protocol.notify_error(session, "未初始化：请先发送 init 消息")
        return

    loop = store.get_or_create(session)

    # 回调：把 Agent 循环的内部事件转成协议通知
    def on_tool_call(name, args, phase, result):
        protocol.notify_tool_call(
            session, name, phase,
            args=args if phase == "start" else None,
            result=result,
        )

    # 流式回调（设计第 1.6 节，Phase 7-C）：每段文本实时推给前端。
    # on_delta 已经分段推了，所以下面的 on_response 不再整段重复发。
    def on_delta(delta):
        protocol.notify_stream(session, delta)

    def on_response(content):
        # 流式模式下文本已由 on_delta 分段推出，这里不再整段重发。
        # （content 用于 loop 内部记录历史，不用来推通知。）
        pass

    try:
        loop.run(
            text,
            on_tool_call=on_tool_call,
            on_response=on_response,
            on_delta=on_delta,
            should_cancel=should_cancel,
        )
    except AgentCancelled as e:
        store.save(session)
        protocol.notify_cancelled(session, e.partial)
        return
    except Exception as e:
        # LLM 调用失败等：发 error，不杀进程（设计第 6.4.1 节）
        import traceback

        from agent.llm_client import LLMError

        tb = traceback.format_exc()
        # 完整堆栈打到 stderr（显示在 VS Code 的 Interview Agent 输出通道，便于诊断）
        sys.stderr.write(tb)
        sys.stderr.flush()

        # 按错误类型给用户不同提示（设计第 6.4.2 节）。
        # LLMError 携带友好中文提示；其他异常用通用格式。
        if isinstance(e, LLMError):
            protocol.notify_error(session, e.message)
        else:
            protocol.notify_error(
                session, f"Agent 执行失败: {type(e).__name__}: {e}"
            )
        return

    # 落盘历史（每轮对话后存一次，设计第 6.4.3 节）
    store.save(session)

    # 本轮结束
    protocol.notify_done(session)


def _handle_stop(cancel_event, params: dict) -> None:
    """处理 stop 消息：中断当前生成（设计第 1.5.2 节）。

    空闲时 stop 不报错；运行中由主循环设置 cancel_event。
    """
    cancel_event.set()


def _inject_attached_code(text: str, attached: dict) -> str:
    """把选中的代码拼进用户消息（设计第 5.3.3 节）。

    格式：在用户原话前附上代码块，让面试官能针对这段代码追问。
    """
    file = attached.get("file", "未知文件")
    content = attached.get("content", "")
    return f"[选中代码 {file}]\n```\n{content}\n```\n\n我的问题：{text}"


if __name__ == "__main__":
    main()
