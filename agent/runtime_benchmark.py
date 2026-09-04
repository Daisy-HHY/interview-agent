"""真实 runtime 指标评估入口。

用环境变量读取模型配置，避免把 API Key 写入命令参数、日志或输出。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent.history import count_tokens
from agent.llm_client import FakeLLM, LLMError, make_text_response, make_tool_call_response
from agent.session import SessionStore

PROMPT = (
    "请先使用 list_directory 或 search_code 读取当前项目结构和 Agent runtime 相关代码，"
    "必要时再用 read_file 查看关键文件；然后基于真实代码，"
    "追问一个和工具调用或会话恢复相关的技术问题。"
)
REQUIRED_TOOL_NAMES = {"list_directory", "search_code", "read_file"}


def main() -> None:
    """运行 runtime 对比 benchmark，输出 JSONL。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=os.getcwd())
    parser.add_argument(
        "--runtime",
        choices=["native", "langchain", "pi", "both", "all"],
        default="both",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output")
    parser.add_argument(
        "--fake-llm",
        action="store_true",
        help="使用固定 FakeLLM，不要求 INTERVIEW_API_KEY。",
    )
    parser.add_argument(
        "--enabled-tools",
        help="逗号分隔的启用工具名；不传则使用默认基础工具。",
    )
    args = parser.parse_args()

    api_key = os.environ.get("INTERVIEW_API_KEY", "").strip()
    if not api_key and not args.fake_llm:
        raise SystemExit("缺少 INTERVIEW_API_KEY，未执行真实 runtime benchmark。")
    if args.fake_llm:
        api_key = "fake"

    if args.runtime == "both":
        runtimes = ["native", "langchain"]
    elif args.runtime == "all":
        runtimes = ["native", "langchain", "pi"]
    else:
        runtimes = [args.runtime]
    rows = []
    for runtime in runtimes:
        for index in range(args.rounds):
            row = run_once(
                workspace=args.workspace,
                api_key=api_key,
                model=os.environ.get("INTERVIEW_MODEL", "gpt-4o-mini"),
                base_url=os.environ.get("INTERVIEW_BASE_URL") or None,
                runtime=runtime,
                index=index + 1,
                enabled_tools=_parse_enabled_tools(args.enabled_tools),
                fake_llm=args.fake_llm,
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    if args.output:
        Path(args.output).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )


def run_once(
    workspace: str,
    api_key: str,
    model: str,
    base_url: str | None,
    runtime: str,
    index: int,
    enabled_tools: list[str] | None = None,
    fake_llm: bool = False,
) -> dict[str, Any]:
    """执行一轮 runtime benchmark，可选使用无网络 FakeLLM。"""
    with _fake_llm_environment(fake_llm):
        return _run_once(
            workspace=workspace,
            api_key=api_key,
            model=model,
            base_url=base_url,
            runtime=runtime,
            index=index,
            enabled_tools=enabled_tools,
            fake_llm=fake_llm,
        )


def _run_once(
    workspace: str,
    api_key: str,
    model: str,
    base_url: str | None,
    runtime: str,
    index: int,
    enabled_tools: list[str] | None = None,
    fake_llm: bool = False,
) -> dict[str, Any]:
    """执行一轮模型调用，并返回不含敏感正文的指标。"""
    store = SessionStore(llm_factory=_build_fake_llm) if fake_llm else SessionStore()
    store.configure(
        workspace=workspace,
        api_key=api_key or ("fake" if fake_llm else api_key),
        model=model,
        base_url=base_url,
        agent_runtime=runtime,
        enabled_tools=enabled_tools,
    )
    session = f"benchmark-{runtime}-{index}"
    loop = store.get_or_create(session)
    started_at = time.perf_counter()
    first_delta_ms: int | None = None
    tool_starts: dict[str, list[tuple[float, int]]] = {}
    tools: list[dict[str, Any]] = []

    def on_delta(_delta: str) -> None:
        nonlocal first_delta_ms
        if first_delta_ms is None:
            first_delta_ms = int((time.perf_counter() - started_at) * 1000)

    def on_tool_call(name: str, args: dict[str, Any], phase: str, result: str) -> None:
        if phase == "start":
            tools.append({
                "tool": name,
                "elapsed_ms": None,
                "args_keys": sorted(args.keys()),
            })
            tool_starts.setdefault(name, []).append((time.perf_counter(), len(tools) - 1))
            return
        if phase == "end" and tool_starts.get(name):
            tool_started_at, metric_index = tool_starts[name].pop()
            tools[metric_index]["elapsed_ms"] = int(
                (time.perf_counter() - tool_started_at) * 1000
            )
            tools[metric_index]["result_chars"] = len(result or "")

    status = "done"
    error_kind = None
    try:
        loop.run(PROMPT, on_delta=on_delta, on_tool_call=on_tool_call)
        store.save(session)
    except LLMError as e:
        status = "error"
        error_kind = e.kind
    except Exception as e:
        status = "error"
        error_kind = type(e).__name__

    tool_sequence = [tool["tool"] for tool in tools]
    required_tool_used = any(name in REQUIRED_TOOL_NAMES for name in tool_sequence)
    benchmark_status = (
        "insufficient_tool_use"
        if status == "done" and not required_tool_used
        else status
    )

    return {
        "runtime": getattr(loop, "runtime_name", runtime),
        "configured_runtime": runtime,
        "round": index,
        "status": status,
        "benchmark_status": benchmark_status,
        "model": model,
        "base_url_configured": bool(base_url),
        "available_tools": store.available_tools,
        "enabled_tools": store.enabled_tools,
        "model_elapsed_ms": getattr(loop, "last_model_elapsed_ms", 0),
        "first_delta_ms": first_delta_ms,
        "total_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        "estimated_tokens": count_tokens(loop.messages),
        "error_kind": error_kind,
        "tool_call_count": len(tools),
        "required_tool_used": required_tool_used,
        "tool_sequence": tool_sequence,
        "tools": tools,
    }


def _build_fake_llm() -> FakeLLM:
    """构造固定 benchmark LLM，先调用项目读取工具再结束。"""
    return FakeLLM([
        make_tool_call_response("list_directory", {"path": "."}),
        make_text_response("FakeLLM benchmark 完成"),
    ])


@contextmanager
def _fake_llm_environment(enabled: bool):
    """在单轮 benchmark 期间启用 FakeLLM 环境，并恢复原值。"""
    if not enabled:
        yield
        return
    previous = os.environ.get("INTERVIEW_FAKE_LLM")
    os.environ["INTERVIEW_FAKE_LLM"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("INTERVIEW_FAKE_LLM", None)
        else:
            os.environ["INTERVIEW_FAKE_LLM"] = previous


def _parse_enabled_tools(value: str | None) -> list[str] | None:
    """解析逗号分隔的启用工具名。"""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
