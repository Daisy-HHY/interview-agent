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

import queue as queue_mod
import sys
import threading

# Windows 默认 stdout/stderr 用系统编码（GBK/cp936），会导致中文输出
# 在某些字符上抛 UnicodeEncodeError。强制改为 UTF-8，与协议约定一致
# （设计第 1.6 节：stdio 用 UTF-8 JSON）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import agent.protocol as protocol
from agent.session import SessionStore


def main(
    stream=None,
    store: SessionStore | None = None,
) -> None:
    """主循环：读 stdin，分发消息，输出通知。

    异步分发（#8）：chat 在独立 worker 线程串行执行，主线程持续读 stdin，
    这样"停止"消息能在生成期间及时到达并 set cancel_event，中断 loop。
    EOF（stdin 关闭）后等 worker 处理完队列再返回（测试确定性 + 生产优雅退出）。

    参数：
        stream: 输入流（默认 sys.stdin）。测试可传 io.StringIO。
        store:  会话仓库（默认新建）。测试可注入带 FakeLLM 的 store。

    退出条件：stdin 关闭（读到 EOF）即结束——VS Code 关闭时 stdin 会关。
    """
    if stream is None:
        stream = sys.stdin
    if store is None:
        store = SessionStore()

    # 异步 chat 基础设施（#8）：单 worker 串行跑 chat；每 session 一个
    # cancel_event，stop handler set 它，loop.run / 流式 chunk 检查它来中断。
    chat_queue: "queue_mod.Queue[tuple | None]" = queue_mod.Queue()
    cancel_events: dict[str, threading.Event] = {}
    state_lock = threading.Lock()

    def get_cancel(session: str) -> threading.Event:
        with state_lock:
            if session not in cancel_events:
                cancel_events[session] = threading.Event()
            return cancel_events[session]

    def worker() -> None:
        while True:
            task = chat_queue.get()
            if task is None:
                return  # 关闭信号
            session, text, cancel_event = task
            try:
                _run_chat(store, session, text, cancel_event)
            except Exception as e:  # worker 兜底，绝不死
                protocol.notify_error(
                    session, f"内部错误: {type(e).__name__}: {e}",
                )

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    handlers = {
        "init": lambda params: _handle_init(store, params),
        "chat": lambda params: _enqueue_chat(chat_queue, get_cancel, params),
        "stop": lambda params: get_cancel(
            params.get("session", "default"),
        ).set(),
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

    # 优雅关闭：让 worker 处理完队列里已入队的 chat 再退出
    # （测试里 run_input 喂完后等通知完整；生产里 EOF 后等当前 chat 完）
    chat_queue.put(None)
    worker_thread.join()


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
    # 落盘目录（#3 修复）：TS 侧传插件数据目录（globalStorageUri.fsPath），
    # 让历史落盘位置稳定、可预测，不依赖子进程 cwd。
    storage_dir = params.get("storage_dir")

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
        storage_dir=storage_dir,
    )


def _enqueue_chat(chat_queue, get_cancel, params) -> None:
    """chat handler：把任务塞进 worker 队列（不阻塞主线程读 stdin，#8）。

    入队前 clear 该 session 的 cancel_event（清掉上轮 stop 的残留 set）；
    worker 跑 loop.run 时带着它，stop handler 随时 set 即可中断。
    """
    session = params.get("session", "default")
    text = params.get("text", "")

    # 拼上选中代码（设计第 5.3.3 节 attached_code）
    attached = params.get("attached_code")
    if attached and isinstance(attached, dict) and attached.get("content"):
        text = _inject_attached_code(text, attached)

    cancel_event = get_cancel(session)
    cancel_event.clear()
    chat_queue.put((session, text, cancel_event))


def _run_chat(
    store: SessionStore,
    session: str,
    text: str,
    cancel_event: "threading.Event",
) -> None:
    """worker 线程里跑一轮 Agent 循环（原 _handle_chat 的执行主体）。

    cancel_event 由 stop handler set；loop.run 在步骤边界、流式 chunk 间检查，
    被取消时尽快收尾并仍发 done（#8）。
    """
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

    # 流式回调（设计第 1.6 节，Phase 7-C）：每段文本实时推给前端
    def on_delta(delta):
        protocol.notify_stream(session, delta)

    def on_response(content):
        # 流式模式下文本已由 on_delta 分段推出，这里不再整段重发
        pass

    try:
        loop.run(
            text,
            on_tool_call=on_tool_call,
            on_response=on_response,
            on_delta=on_delta,
            cancel_event=cancel_event,
        )
    except Exception as e:
        # LLM 调用失败等：发 error，不杀 worker（设计第 6.4.1 节）
        import traceback

        from agent.llm_client import LLMError

        tb = traceback.format_exc()
        # 完整堆栈打到 stderr（VS Code 输出通道诊断用）
        sys.stderr.write(tb)
        sys.stderr.flush()

        # 按错误类型给用户不同提示（设计第 6.4.2 节）
        if isinstance(e, LLMError):
            protocol.notify_error(session, e.message)
        else:
            protocol.notify_error(
                session, f"Agent 执行失败: {type(e).__name__}: {e}",
            )
        return

    # 落盘历史（每轮对话后存一次，设计第 6.4.3 节）
    store.save(session)

    # 本轮结束（含被 cancel 收尾的情况——loop.run 也会正常 return）
    protocol.notify_done(session)


def _inject_attached_code(text: str, attached: dict) -> str:
    """把选中的代码拼进用户消息（设计第 5.3.3 节）。

    格式：在用户原话前附上代码块，让面试官能针对这段代码追问。
    """
    file = attached.get("file", "未知文件")
    content = attached.get("content", "")
    return f"[选中代码 {file}]\n```\n{content}\n```\n\n我的问题：{text}"


if __name__ == "__main__":
    main()
