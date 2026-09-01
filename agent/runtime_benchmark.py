"""真实 runtime 指标评估入口。

用环境变量读取模型配置，避免把 API Key 写入命令参数、日志或输出。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from agent.history import count_tokens
from agent.llm_client import LLMError
from agent.session import SessionStore

PROMPT = (
    "请读取当前项目结构，结合项目中的 Agent runtime 和工具实现，"
    "追问一个和工具调用或会话恢复相关的技术问题。"
)


def main() -> None:
    """运行 native/langchain 对比 benchmark，输出 JSONL。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=os.getcwd())
    parser.add_argument("--runtime", choices=["native", "langchain", "both"], default="both")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    api_key = os.environ.get("INTERVIEW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("缺少 INTERVIEW_API_KEY，未执行真实 runtime benchmark。")

    runtimes = ["native", "langchain"] if args.runtime == "both" else [args.runtime]
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
) -> dict[str, Any]:
    """执行一轮真实模型调用，并返回不含敏感正文的指标。"""
    store = SessionStore()
    store.configure(
        workspace=workspace,
        api_key=api_key,
        model=model,
        base_url=base_url,
        agent_runtime=runtime,
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

    return {
        "runtime": getattr(loop, "runtime_name", runtime),
        "configured_runtime": runtime,
        "round": index,
        "status": status,
        "model": model,
        "base_url_configured": bool(base_url),
        "model_elapsed_ms": getattr(loop, "last_model_elapsed_ms", 0),
        "first_delta_ms": first_delta_ms,
        "total_elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        "estimated_tokens": count_tokens(loop.messages),
        "error_kind": error_kind,
        "tools": tools,
    }


if __name__ == "__main__":
    main()
