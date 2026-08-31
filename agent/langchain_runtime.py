"""LangChain / LangGraph 运行时实现。"""

from typing import Any, Callable

from agent.history import compress_history, enforce_token_limit
from agent.langchain_tools import build_langchain_tools
from agent.llm_client import AgentCancelled
from agent.runtime import CancelCallback, ResponseCallback, ToolCallCallback
from agent.tools.base import ToolRegistry


class LangChainAgentRuntime:
    """基于 LangChain `create_agent` 的可选 Agent runtime。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None,
        tools: ToolRegistry,
        system_prompt: str,
        max_steps: int = 8,
        max_history_tokens: int | None = None,
        max_kept_full: int | None = None,
        model_factory: Callable[[], Any] | None = None,
        agent_factory: Callable[[Any, list[Any], str], Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._tools = tools
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._max_history_tokens = max_history_tokens
        self._max_kept_full = max_kept_full
        self._model_factory = model_factory
        self._agent_factory = agent_factory
        self._messages: list[dict] = [{"role": "system", "content": system_prompt}]

    @property
    def messages(self) -> list[dict]:
        """暴露历史，保持 SessionStore 原 JSON 落盘格式。"""
        return self._messages

    def run(
        self,
        user_text: str,
        on_tool_call: ToolCallCallback | None = None,
        on_response: ResponseCallback | None = None,
        on_delta: Any = None,
        should_cancel: CancelCallback | None = None,
    ) -> str:
        """跑一轮 LangChain Agent，并映射回现有回调协议。"""
        self._check_cancel(should_cancel)
        self._messages.append({"role": "user", "content": user_text})
        self._prepare_history()

        langchain_tools = build_langchain_tools(
            self._tools,
            on_tool_call=on_tool_call,
            should_cancel=should_cancel,
        )
        agent = self._create_agent(langchain_tools)

        try:
            answer = self._invoke_agent(agent, on_delta, should_cancel)
        except AgentCancelled as e:
            if e.partial:
                self._messages.append({"role": "assistant", "content": e.partial})
            raise

        self._messages.append({"role": "assistant", "content": answer})
        if on_response:
            on_response(answer)
        return answer

    def _prepare_history(self) -> None:
        if self._max_kept_full is not None:
            self._messages = compress_history(self._messages, self._max_kept_full)
        else:
            self._messages = compress_history(self._messages)
        if self._max_history_tokens is not None:
            self._messages = enforce_token_limit(self._messages, self._max_history_tokens)
        else:
            self._messages = enforce_token_limit(self._messages)

    def _create_agent(self, langchain_tools: list[Any]) -> Any:
        model = self._model_factory() if self._model_factory else self._create_model()
        if self._agent_factory:
            return self._agent_factory(model, langchain_tools, self._system_prompt)

        try:
            from langchain.agents import create_agent
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith("langchain"):
                raise RuntimeError(_missing_langchain_message()) from e
            raise

        return create_agent(
            model=model,
            tools=langchain_tools,
            system_prompt=self._system_prompt,
        )

    def _create_model(self) -> Any:
        try:
            from langchain.chat_models import init_chat_model
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith(("langchain", "langchain_openai")):
                raise RuntimeError(_missing_langchain_message()) from e
            raise

        kwargs: dict[str, Any] = {
            "model": self._model,
            "model_provider": "openai",
            "api_key": self._api_key,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return init_chat_model(**kwargs)

    def _invoke_agent(
        self,
        agent: Any,
        on_delta: Any,
        should_cancel: CancelCallback | None,
    ) -> str:
        self._check_cancel(should_cancel)
        payload = {"messages": self._langchain_input_messages()}
        config = {
            "recursion_limit": max(self._max_steps * 2, 2),
            "configurable": {"thread_id": "interview-agent"},
        }

        if hasattr(agent, "stream_events"):
            stream = agent.stream_events(payload, config=config, version="v3")
            answer = self._consume_stream(stream, on_delta, should_cancel)
            if answer:
                return answer

        result = agent.invoke(payload, config=config)
        answer = _extract_final_text(result)
        if on_delta and answer:
            on_delta(answer)
        self._check_cancel(should_cancel, answer)
        return answer

    def _consume_stream(
        self,
        stream: Any,
        on_delta: Any,
        should_cancel: CancelCallback | None,
    ) -> str:
        answer = ""

        if hasattr(stream, "interleave"):
            for kind, item in stream.interleave("messages", "tool_calls"):
                self._check_cancel(should_cancel, answer)
                if kind != "messages":
                    continue
                for token in _message_tokens(item):
                    answer += token
                    if on_delta:
                        on_delta(token)
            return answer or _extract_final_text(getattr(stream, "output", None))

        values = getattr(stream, "values", None)
        if values is not None:
            latest = ""
            for snapshot in values:
                self._check_cancel(should_cancel, latest)
                latest = _extract_final_text(snapshot) or latest
            if latest and on_delta:
                on_delta(latest)
            return latest

        return ""

    def _langchain_input_messages(self) -> list[dict]:
        messages = []
        for message in self._messages:
            role = message.get("role")
            if role == "system" or role == "tool":
                continue
            messages.append({"role": role, "content": message.get("content", "")})
        return messages

    @staticmethod
    def _check_cancel(
        should_cancel: CancelCallback | None,
        partial: str = "",
    ) -> None:
        if should_cancel is not None and should_cancel():
            raise AgentCancelled(partial)


def _message_tokens(item: Any) -> list[str]:
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return [text] if text else []
    if text is not None:
        return [str(part) for part in text if part]

    content = getattr(item, "content", None)
    if isinstance(content, str):
        return [content] if content else []
    return []


def _extract_final_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            return _message_content(messages[-1])
    return _message_content(result)


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content) if content else ""


def _missing_langchain_message() -> str:
    return (
        "当前 Python 环境未安装 LangChain 运行时依赖。"
        "请安装 agent-framework 可选依赖，或将 interview.agentRuntime 改回 native。"
    )

