"""模型实际用量的流式与缺失场景，不发真实网络请求。"""

from types import SimpleNamespace as NS

import pytest

from agent.llm_client import OpenAIClient


@pytest.mark.parametrize("streaming", [False, True])
def test_actual_usage_is_collected(streaming):
    """usage-only 最后分片也应计入实际用量。"""
    client = OpenAIClient.__new__(OpenAIClient)
    client._model = "fake"
    usage = NS(prompt_tokens=31, completion_tokens=7)
    choice = NS(finish_reason="stop", message=NS(content="answer", tool_calls=[]),
                delta=NS(content="answer", tool_calls=[]))
    response = ([NS(choices=[choice], usage=None), NS(choices=[], usage=usage)]
                if streaming else NS(choices=[choice], usage=usage))
    client._client = NS(chat=NS(completions=NS(create=lambda **kw: response)))
    client.chat([{"role": "user", "content": "test"}], [],
                on_delta=(lambda _: None) if streaming else None)
    assert client.usage_totals == {
        "input_tokens": 31, "output_tokens": 7, "model_calls": 1, "missing_usage_calls": 0,
    }


def test_missing_usage_does_not_mean_zero_cost():
    """缺失计数明确标记，不能把服务未返回用量写成零消费。"""
    client = OpenAIClient.__new__(OpenAIClient)
    client._model = "fake"
    response = NS(choices=[NS(finish_reason="stop", message=NS(content="ok", tool_calls=[]))])
    client._client = NS(chat=NS(completions=NS(create=lambda **kw: response)))
    client.chat([], [])
    assert client.usage_totals["missing_usage_calls"] == 1
    assert client.usage_totals["input_tokens"] is None
