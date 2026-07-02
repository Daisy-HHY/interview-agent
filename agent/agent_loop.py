"""Agent 循环（设计第 2 节）。

项目心脏：让 LLM 能"多走几步"——每一步要么调工具，
要么直接回答。调工具就把结果塞回对话历史，让 LLM 再想下一步。
"""

from typing import Any, Callable

from agent.history import compress_history, enforce_token_limit
from agent.llm_client import LLMClient, LLMResponse
from agent.tools.base import ToolRegistry


def _sanitize_surrogates(text: str) -> str:
    """清除字符串里的孤立代理项（surrogate）。

    Windows 上 os.listdir / 文件名 / 文件内容可能返回含孤立代理项的字符串，
    这些字符 UTF-8 无法编码，会导致 OpenAI 序列化请求、notify 输出时抛
    UnicodeEncodeError。

    先用 surrogatepass 把代理项编成原始字节，再用 ignore 解码丢弃坏字节。
    """
    try:
        return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

# 回调类型：让外部（Phase 5 的协议层）能知道"循环在干什么"
# on_tool_call: 工具被调用时通知（UI 显示"正在搜代码"气泡）
# on_response: LLM 给出最终文本回答时通知（UI 流式输出）
ToolCallCallback = Callable[[str, dict[str, Any], str, str], None]
ResponseCallback = Callable[[str], None]


class AgentLoop:
    """Agent 循环（设计第 2 节）。

    核心逻辑：
        while 还没达到最大步数:
            调 LLM
            if LLM 想调工具:
                执行工具，结果塞回历史，继续循环
            else:
                输出回答，结束

    用法：
        loop = AgentLoop(llm=fake_llm, tools=registry)
        answer = loop.run("我做了一个选课系统")
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
        max_steps: int = 8,  # 设计第 2.5 节安全阀
        # 历史管理参数（设计第 6.2 节，Phase 7-D 可配化）。
        # None 表示用 history 模块的默认值（20000 / 3）。
        max_history_tokens: int | None = None,
        max_kept_full: int | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._max_history_tokens = max_history_tokens
        self._max_kept_full = max_kept_full
        # 对话历史：整个 session 复用一份
        # 第一条永远是系统提示（设计第 6.2.3 节：永不删除）
        self._messages: list[dict] = [{"role": "system", "content": system_prompt}]

    @property
    def messages(self) -> list[dict]:
        """暴露历史（Phase 5 落盘用）。"""
        return self._messages
    

    def run(
        self,
        user_text: str,
        on_tool_call: ToolCallCallback | None = None,
        on_response: ResponseCallback | None = None,
        on_delta: Any = None,
        cancel_event: Any = None,
    ) -> str:
        """跑一轮 Agent 循环。

        on_delta（可选）：流式文本回调（设计第 1.6 节，Phase 7-C）。
        每收到一段 LLM 文本就调一次，实现打字效果。
        传了它，chat 内部用 stream=True；不传则非流式。

        cancel_event（可选，#8）：threading.Event，被 set 时 loop 在下一个步骤
        边界停止后续 LLM 调用——让前端的"停止"按钮能中断生成。

        参数：
            user_text:    用户这一轮说的话
            on_tool_call: 工具调用回调（可选）
            on_response:  最终回答回调（可选）
            on_delta:     流式文本片段回调（可选，Phase 7-C）
            cancel_event: 取消事件（可选，#8）

        返回：Agent 的最终文本回答
        """
        # 把用户消息加入历史
        self._messages.append({"role": "user", "content": user_text})

        # 工具 schema：每次取一次（注册的工具可能变化）
        tools_schema = self._tools.all_schemas()

        for step in range(self._max_steps):
            # ── cancel 检查（#8 stop 生效）：步骤边界看是否被取消 ──
            if cancel_event is not None and cancel_event.is_set():
                return self._finish_cancelled(on_response, on_delta)

            # ── 每轮调 LLM 前：管理历史（设计第 6.2 节）──
            # 参数透传：None 时用 history 模块默认值（Phase 7-D 可配化）
            if self._max_kept_full is not None:
                self._messages = compress_history(self._messages, self._max_kept_full)
            else:
                self._messages = compress_history(self._messages)
            if self._max_history_tokens is not None:
                self._messages = enforce_token_limit(self._messages, self._max_history_tokens)
            else:
                self._messages = enforce_token_limit(self._messages)

            # ── 调 LLM（传 on_delta 启用流式，Phase 7-C；cancel_event 支持 #8）──
            response = self._llm.chat(
                self._messages, tools_schema,
                on_delta=on_delta, cancel_event=cancel_event,
            )

            if response.tool_calls:
                # LLM 想调工具：处理所有工具调用，继续循环
                self._handle_tool_calls(response, on_tool_call)
                # 工具执行后也查 cancel（及时性：不必等到下一步循环开头）
                if cancel_event is not None and cancel_event.is_set():
                    return self._finish_cancelled(on_response, on_delta)
                # 不 return，继续下一轮——LLM 拿到工具结果会再想下一步
            else:
                # LLM 直接回答了：输出文本，结束循环
                self._messages.append(
                    {"role": "assistant", "content": response.content}
                )
                if on_response:
                    on_response(response.content)
                return response.content

        # 循环跑满 max_steps 还没结束：触发安全阀
        fallback = "（已达到最大推理步数，本轮停止。你可以继续描述你的项目。）"
        self._messages.append({"role": "assistant", "content": fallback})
        if on_response:
            on_response(fallback)
        return fallback

    def _finish_cancelled(
        self,
        on_response: ResponseCallback | None,
        on_delta: Any,
    ) -> str:
        """cancel 后的收尾：记历史 + 通知前端（#8）。

        流式下 on_response 是 no-op（main.py），所以用 on_delta 把"已停止"
        标记推给前端气泡，让用户看到生成被中断。
        """
        msg = "（已停止）"
        self._messages.append({"role": "assistant", "content": msg})
        if on_delta:
            on_delta("\n\n" + msg)
        if on_response:
            on_response(msg)
        return msg


    def _handle_tool_calls(
        self,
        response: LLMResponse,
        on_tool_call: ToolCallCallback | None,
    ) -> None:
        """处理 LLM 的所有工具调用，把结果塞回对话历史。

        设计第 3.9 节：工具失败不杀循环，错误当 Observation 喂回去。
        """
        # 先把 assistant 的工具调用意图加入历史（OpenAI 格式要求）
        self._messages.append({
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": response.tool_calls,
        })

        # 逐个执行工具
        import json
        for tc in response.tool_calls:
            func = tc["function"]
            name = func["name"]
            # arguments 是 JSON 字符串（Phase 3 守护的格式）
            try:
                args = json.loads(func["arguments"])
            except json.JSONDecodeError:
                args = {}

            # 通知外部：工具开始
            if on_tool_call:
                on_tool_call(name, args, "start", "")

            # 执行工具（设计第 3.9 节：safe_execute 错误兜底）
            result = self._safe_execute(name, args)

            # 通知外部：工具结束
            if on_tool_call:
                on_tool_call(name, args, "end", result)

            # 把工具结果塞回历史（OpenAI 格式：role=="tool"，带 tool_call_id）
            self._messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    def _safe_execute(self, name: str, args: dict[str, Any]) -> str:
        """安全执行工具（设计第 3.9 节）。

        工具失败不杀循环，把错误变成给 LLM 的文本，让它自我恢复。
        """
        tool = self._tools.get(name)
        if tool is None:
            # LLM 幻觉调用了不存在的工具
            return f"错误：不存在名为 '{name}' 的工具。可用工具：{[t for t in self._tools._tools]}"

        try:
            result = tool.execute(**args)
        except Exception as e:
            # 错误当 Observation 喂回去，LLM 会自己调整策略
            result = f"工具执行出错: {type(e).__name__}: {e}"

        # 清理工具结果里的孤立代理项（surrogate）。
        # Windows 文件名可能含坏字符（如 .venv 残留），这些字符 UTF-8 编不出，
        # 会导致后续 OpenAI 序列化请求 / notify 输出时抛 UnicodeEncodeError。
        # 在结果进入历史前清掉，根治所有下游崩溃。
        return _sanitize_surrogates(result)
        
    