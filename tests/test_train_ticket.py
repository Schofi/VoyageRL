"""Exercise ticket responses and timeout handling without network requests."""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import dotenv
import httpx
import pytest
from openai import AsyncOpenAI


@pytest.fixture
def ticket_module(monkeypatch):
    # Isolate framework registration; execute the real tool and OpenAI client.
    base = ModuleType("qwen_agent.tools.base")

    class BaseTool:
        def __init__(self, cfg=None):
            pass

    base.BaseTool = BaseTool
    base.register_tool = lambda *args, **kwargs: lambda cls: cls
    for name in ("qwen_agent", "qwen_agent.tools"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "qwen_agent.tools.base", base)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL_ID", "test-model")

    path = Path(__file__).resolve().parents[1] / "tools" / "tool_train_ticket.py"
    spec = importlib.util.spec_from_file_location("train_ticket_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def use_transport(module, monkeypatch, handler):
    def make_client(**kwargs):
        kwargs["max_retries"] = 0
        kwargs["http_client"] = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(module, "AsyncOpenAI", make_client)


def test_timeout_returns_original_error_instead_of_crashing(
    ticket_module, monkeypatch, caplog
):
    def timeout(request):
        raise httpx.ReadTimeout("read timed out", request=request)

    use_transport(ticket_module, monkeypatch, timeout)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(ticket_module.TrainTicketsSearch().call({
            "date": "2026-10-20", "from_city": "Chengdu", "to_city": "Beijing"
        }))

    assert "APITimeoutError" in result
    assert "Request timed out" in result
    assert "Request timed out" in caplog.text
    assert "test-key" not in result + caplog.text


def test_success_returns_model_content(ticket_module, monkeypatch):
    content = '["simulated train result"]'

    def success(request):
        return httpx.Response(200, json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f" {content} "},
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }, request=request)

    use_transport(ticket_module, monkeypatch, success)
    result = asyncio.run(ticket_module.TrainTicketsSearch().call({
        "date": "2026-10-20", "from_city": "Chengdu", "to_city": "Beijing"
    }))

    assert result == content
