"""Agent runtime 协议。

0.1.11 只抽一层最小边界：一轮对话、消息历史和现有回调。
"""

from typing import Any, Callable, Protocol

from agent.agent_loop import AgentLoop

NativeAgentRuntime = AgentLoop

ToolCallCallback = Callable[[str, dict[str, Any], str, str], None]
ResponseCallback = Callable[[str], None]
CancelCallback = Callable[[], bool]


class AgentRuntime(Protocol):
    """可替换的 Agent 运行时接口。"""

    @property
    def runtime_name(self) -> str:
        """返回实际运行的 runtime 名称。"""
        ...

    @property
    def messages(self) -> list[dict]:
        """返回当前会话消息历史，用于原 JSON 格式落盘。"""
        ...

    @property
    def last_model_elapsed_ms(self) -> int:
        """返回最近一轮模型调用累计耗时。"""
        ...

    def run(
        self,
        user_text: str,
        on_tool_call: ToolCallCallback | None = None,
        on_response: ResponseCallback | None = None,
        on_delta: Any = None,
        should_cancel: CancelCallback | None = None,
    ) -> str:
        """执行一轮对话并返回最终回答。"""
        ...
