"""Offline checks for single-sample output and interrupted batch recovery."""

import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessage


@pytest.fixture
def distill(monkeypatch):
    # The model framework is not installed locally; tools must not call services.
    for name in ("qwen_agent", "qwen_agent.agents", "qwen_agent.llm"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name, attribute in (
        ("qwen_agent.agents.fncall_agent", "FnCallAgent"),
        ("qwen_agent.llm.schema", "Message"),
        ("tools.tool_web_search", "WebSearch"),
        ("tools.tool_visit", "Visit"),
        ("tools.tool_weather", "WeatherSearch"),
        ("tools.tool_transport", "FlightsSearch"),
        ("tools.tool_train_ticket", "TrainTicketsSearch"),
        ("tools.tool_route_planning", "RoutePlanning"),
        ("tools.tool_poi_search", "POISearch"),
        ("tools.tool_around_search", "AroundSearch"),
    ):
        module = ModuleType(name)
        setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, name, module)
    tokenizer_module = ModuleType("transformers")
    tokenizer_module.AutoTokenizer = object
    tokenizer_module.PreTrainedTokenizerFast = object
    monkeypatch.setitem(sys.modules, "transformers", tokenizer_module)

    path = Path(__file__).resolve().parents[1] / "data_distill.py"
    spec = importlib.util.spec_from_file_location("data_distill_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agent(distill):
    instance = distill.MultiTurnReactAgent.__new__(distill.MultiTurnReactAgent)
    instance.max_llm_calls_per_run = 2
    instance.max_tokens_per_turn = 1000
    instance.max_total_tokens_before_finishing = 1000
    instance.max_total_tokens = 2000
    instance.count_tokens = lambda messages: 0

    async def answer(client, messages):
        return object(), ChatCompletionMessage(
            role="assistant", content="<answer>经济路线</answer>"
        )

    instance.call_server = answer
    return instance


def completed(question="重庆路线"):
    # Match legacy files, which do not contain a qid field.
    return {
        "question": question,
        "answer": "",
        "messages": [{"role": "assistant", "content": "<answer>旧路线</answer>"}],
        "prediction": "旧路线",
        "termination": "answer",
        "stats": {"turns": 1},
    }


def test_single_sample_returns_result_and_creates_output_directory(agent, tmp_path):
    output = tmp_path / "new" / "directory"
    result = asyncio.run(agent.make_trajectory(
        {"qid": "123", "question": "重庆路线"}, None, str(output)
    ))

    assert result["prediction"] == "经济路线"
    saved = json.loads((output / "123.json").read_text(encoding="utf-8"))
    assert saved["prediction"] == result["prediction"]
    assert "completions" not in saved


def test_resume_preserves_completed_legacy_result(agent, tmp_path):
    output = tmp_path / "123.json"
    original = json.dumps(completed(), ensure_ascii=False)
    output.write_text(original, encoding="utf-8")

    async def must_not_call(client, messages):
        raise AssertionError("A completed sample was regenerated")

    agent.call_server = must_not_call
    summary = asyncio.run(agent.make_trajectory(
        [{"qid": "123", "question": "重庆路线"}], None, str(tmp_path), resume=True
    ))

    assert summary == {"total": 1, "completed": 0, "skipped": 1, "failed": 0}
    assert output.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("saved", [
    '{"question":',
    "[]",
    json.dumps({**completed(), "termination": "exceed available llm calls"}),
    json.dumps({**completed(), "question": "其他问题"}),
    json.dumps({**completed(), "prediction": ""}),
    json.dumps({**completed(), "messages": []}),
    json.dumps({**completed(), "messages": [None]}),
    json.dumps({**completed(), "messages": [{"role": "assistant", "content": None}]}),
    json.dumps({**completed(), "messages": [{"role": "user", "content": "重庆路线"}]}),
])
def test_resume_regenerates_incomplete_or_mismatched_results(agent, tmp_path, saved):
    output = tmp_path / "123.json"
    output.write_text(saved, encoding="utf-8")
    summary = asyncio.run(agent.make_trajectory(
        [{"qid": "123", "question": "重庆路线"}], None, str(tmp_path), resume=True
    ))

    assert summary == {"total": 1, "completed": 1, "skipped": 0, "failed": 0}
    assert json.loads(output.read_text(encoding="utf-8"))["prediction"] == "经济路线"


def test_batch_continues_after_malformed_row_during_resume(agent, tmp_path):
    summary = asyncio.run(agent.make_trajectory([
        {"question": "缺少编号的记录"},
        {"qid": "123", "question": "重庆路线"},
    ], None, str(tmp_path), resume=True))

    assert summary == {"total": 2, "completed": 1, "skipped": 0, "failed": 1}
    assert (tmp_path / "123.json").is_file()


def test_interrupted_batch_resumes_only_remaining_samples(agent, tmp_path):
    data = [
        {"qid": "1", "question": "第一条路线"},
        {"qid": "2", "question": "第二条路线"},
    ]
    normal_answer = agent.call_server

    async def interrupted_answer(client, messages):
        if messages[1]["content"] == "第二条路线":
            raise KeyboardInterrupt
        return await normal_answer(client, messages)

    agent.call_server = interrupted_answer
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(agent.make_trajectory(data, None, str(tmp_path)))
    first_result = (tmp_path / "1.json").read_bytes()
    assert not (tmp_path / "2.json").exists()

    agent.call_server = normal_answer
    summary = asyncio.run(agent.make_trajectory(
        data, None, str(tmp_path), resume=True
    ))

    assert summary == {"total": 2, "completed": 1, "skipped": 1, "failed": 0}
    assert (tmp_path / "1.json").read_bytes() == first_result
    assert (tmp_path / "2.json").is_file()


def test_failed_write_does_not_damage_previous_result(agent, distill, tmp_path, monkeypatch):
    output = tmp_path / "123.json"
    original = json.dumps(completed(), ensure_ascii=False)
    output.write_text(original, encoding="utf-8")

    def failed_dump(value, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError("simulated interruption while saving")

    monkeypatch.setattr(distill.json, "dump", failed_dump)
    with pytest.raises(OSError, match="simulated interruption"):
        asyncio.run(agent.make_trajectory(
            {"qid": "123", "question": "重庆路线"}, None, str(tmp_path)
        ))

    assert output.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("batch", [False, True])
def test_token_limit_answer_is_saved_for_resume(agent, tmp_path, batch):
    agent.max_total_tokens_before_finishing = -1
    calls = 0

    async def answer_after_limit(client, messages):
        nonlocal calls
        calls += 1
        content = "正在规划" if calls == 1 else "<answer>最终路线</answer>"
        return object(), ChatCompletionMessage(role="assistant", content=content)

    agent.call_server = answer_after_limit
    data = {"qid": "123", "question": "重庆路线"}
    result = asyncio.run(agent.make_trajectory(
        [data] if batch else data, None, str(tmp_path)
    ))

    if batch:
        assert result == {"total": 1, "completed": 1, "skipped": 0, "failed": 0}
    else:
        assert result["prediction"] == "最终路线"
    saved = json.loads((tmp_path / "123.json").read_text(encoding="utf-8"))
    assert saved["termination"] == "generate an answer as token limit reached"
    assert saved["messages"][-1]["content"] == "<answer>最终路线</answer>"


def test_default_main_runs_only_one_sample(distill, agent, tmp_path, monkeypatch):
    async def run(data, save_path, resume=False):
        return await agent.make_trajectory(data, None, save_path, resume=resume)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["data_distill.py"])
    monkeypatch.setattr(distill, "arun_episode", run)
    distill.main()

    outputs = list(tmp_path.rglob("*.json"))
    assert outputs == [tmp_path / "saved" / "tmp" / "123.json"]
    assert not (tmp_path / "saved" / "logs").exists()


@pytest.mark.parametrize("no_resume", [False, True])
def test_batch_main_preserves_ids_and_supports_resume_override(
    distill, agent, tmp_path, monkeypatch, no_resume
):
    (tmp_path / "data").mkdir()
    rows = [
        {"qid": "123", "question": "重庆路线"},
        {"question": "第二条路线"},
    ]
    (tmp_path / "data" / "examples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n\n",
        encoding="utf-8",
    )
    output = tmp_path / "saved" / "examples"
    output.mkdir(parents=True)
    (output / "123.json").write_text(json.dumps(completed()), encoding="utf-8")
    summaries = []

    async def run(data, save_path, resume=False):
        summary = await agent.make_trajectory(data, None, save_path, resume=resume)
        summaries.append(summary)
        return summary

    argv = ["data_distill.py", "--mode", "batch", "--filename", "examples.jsonl"]
    if no_resume:
        argv.append("--no-resume")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(distill, "arun_episode", run)
    distill.main()

    assert summaries == [{
        "total": 2, "completed": 2 if no_resume else 1,
        "skipped": 0 if no_resume else 1, "failed": 0,
    }]
    expected_prediction = "经济路线" if no_resume else "旧路线"
    assert json.loads((output / "123.json").read_text(encoding="utf-8"))["prediction"] == expected_prediction
    generated_id = hashlib.md5("第二条路线".encode("utf-8")).hexdigest()
    assert (output / f"{generated_id}.json").is_file()


def test_batch_save_dir_keeps_existing_teacher_outputs_separate(
    distill, agent, tmp_path, monkeypatch
):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "train_expansion.jsonl").write_text(
        json.dumps({"qid": "sample", "question": "Plan a trip"}), encoding="utf-8"
    )
    previous_dir = tmp_path / "saved" / "train_expansion"
    previous_dir.mkdir(parents=True)
    previous_file = previous_dir / "sample.json"
    previous_file.write_text("old teacher output", encoding="utf-8")

    async def run(data, save_path, resume=False):
        return await agent.make_trajectory(data, None, save_path, resume=resume)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(distill, "arun_episode", run)
    monkeypatch.setattr(sys, "argv", [
        "data_distill.py", "--mode", "batch", "--filename", "train_expansion.jsonl",
        "--save-dir", "saved/train_expansion-ds",
    ])
    distill.main()

    assert previous_file.read_text(encoding="utf-8") == "old teacher output"
    assert (tmp_path / "saved" / "train_expansion-ds" / "sample.json").is_file()


def test_batch_log_contains_debug_progress_errors_and_both_datasets(
    distill, agent, tmp_path, monkeypatch
):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    datasets = {
        "test": [
            {"qid": "1", "question": "重庆路线"},
            {"qid": "2", "question": "触发失败"},
        ],
        "train_expansion": [{"qid": "3", "question": "已完成路线"}],
    }
    for name, rows in datasets.items():
        (input_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
            encoding="utf-8",
        )
    saved_dir = tmp_path / "saved" / "train_expansion"
    saved_dir.mkdir(parents=True)
    (saved_dir / "3.json").write_text(
        json.dumps(completed("已完成路线"), ensure_ascii=False), encoding="utf-8"
    )
    normal_answer = agent.call_server

    async def answer_or_fail(client, messages):
        if messages[1]["content"] == "触发失败":
            raise RuntimeError("模拟批量报错")
        return await normal_answer(client, messages)

    async def run(data, save_path, resume=False):
        return await agent.make_trajectory(data, None, save_path, resume=resume)

    agent.call_server = answer_or_fail
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["data_distill.py", "--mode", "batch"])
    monkeypatch.setattr(distill, "arun_episode", run)
    distill.main()

    logs = list((tmp_path / "saved" / "logs").glob("batch_*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "[DEBUG]" in text
    assert "Result dumped to" in text
    assert "test.jsonl" in text and "train_expansion.jsonl" in text
    assert "[1/2] Run QID 1" in text
    assert "[1/2] Completed QID 1" in text
    assert "Skip completed QID 3" in text
    assert "Traceback (most recent call last):" in text
    assert "RuntimeError: 模拟批量报错" in text
    assert "'failed': 1" in text


def test_batch_log_records_input_file_errors(distill, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "data_distill.py", "--mode", "batch", "--filename", "missing.jsonl"
    ])
    with pytest.raises(FileNotFoundError):
        distill.main()

    logs = list((tmp_path / "saved" / "logs").glob("batch_*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "FileNotFoundError" in text
    assert "missing.jsonl" in text


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_batch_logs_survive_restarts_and_handlers_are_closed(
    distill, tmp_path, monkeypatch, error
):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "example.jsonl").write_text(
        json.dumps({"qid": "1", "question": "重庆路线"}), encoding="utf-8"
    )

    async def interrupted(data, save_path, resume=False):
        raise error("模拟中断")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "data_distill.py", "--mode", "batch", "--filename", "example.jsonl"
    ])
    monkeypatch.setattr(distill, "arun_episode", interrupted)
    root = distill.agent_logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = distill.logger.level
    for _ in range(2):
        with pytest.raises(error):
            distill.main()
        assert root.handlers == original_handlers
        assert distill.logger.level == original_level

    logs = list((tmp_path / "saved" / "logs").glob("batch_*.log"))
    assert len(logs) == 2
    for path in logs:
        text = path.read_text(encoding="utf-8")
        assert "example.jsonl" in text
        assert "interrupted" in text.lower() if error is KeyboardInterrupt else "RuntimeError" in text
        # An open FileHandler would prevent renaming the file on Windows.
        path.rename(path.with_suffix(".closed.log"))


def completion_with_usage(usage, content="<answer>经济路线</answer>"):
    return ChatCompletion.model_validate({
        "id": "chatcmpl-offline", "object": "chat.completion", "created": 0,
        "model": "deepseek-flash", "usage": usage,
        "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
    })


@pytest.mark.parametrize("cache_fields, expected_cache", [
    ({"prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200},
     {"cache_reported_requests": 1, "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200}),
    ({"prompt_tokens_details": {"cached_tokens": 800}},
     {"cache_reported_requests": 1, "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200}),
    ({"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 1000},
     {"cache_reported_requests": 1, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 1000}),
    ({}, {"cache_reported_requests": 0}),
])
def test_api_cache_usage_is_saved_without_inventing_missing_counts(
    distill, agent, tmp_path, cache_fields, expected_cache
):
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100, **cache_fields}

    async def answer(client, messages):
        completion = completion_with_usage(usage)
        return completion, completion.choices[0].message

    agent.call_server = answer
    result = asyncio.run(agent.make_trajectory(
        {"qid": "usage", "question": "重庆路线"}, None, str(tmp_path)
    ))
    expected = {
        "requests": 1, "usage_reported_requests": 1,
        "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
        **expected_cache,
    }
    assert result["stats"]["usage"] == expected
    saved = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert saved["stats"]["usage"] == expected


def test_usage_accumulates_extra_token_limit_call(agent, tmp_path):
    agent.max_total_tokens_before_finishing = -1
    calls = 0

    async def answer(client, messages):
        nonlocal calls
        calls += 1
        completion = completion_with_usage({
            "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
            "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200,
        }, "正在规划" if calls == 1 else "<answer>经济路线</answer>")
        return completion, completion.choices[0].message

    agent.call_server = answer
    result = asyncio.run(agent.make_trajectory(
        {"qid": "usage", "question": "重庆路线"}, None, str(tmp_path)
    ))
    usage = result["stats"]["usage"]
    assert usage["requests"] == usage["cache_reported_requests"] == 2
    assert usage["prompt_tokens"] == 2000
    assert usage["prompt_cache_hit_tokens"] == 1600


def test_batch_usage_includes_responses_from_eventually_failed_sample(
    distill, agent, tmp_path
):
    async def answer(client, messages):
        if len(messages) > 2:
            raise RuntimeError("response after a billed call failed")
        content = "正在规划" if messages[1]["content"] == "失败样本" else "<answer>经济路线</answer>"
        completion = completion_with_usage({
            "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
            "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200,
        }, content)
        return completion, completion.choices[0].message

    agent.call_server = answer
    with distill.batch_logging(str(tmp_path / "logs")) as log_path:
        summary = asyncio.run(agent.make_trajectory([
            {"qid": "1", "question": "重庆路线"},
            {"qid": "2", "question": "失败样本"},
        ], None, str(tmp_path / "results")))
    assert summary == {"total": 2, "completed": 1, "skipped": 0, "failed": 1}
    assert agent.usage_totals["prompt_tokens"] == 2000
    text = log_path.read_text(encoding="utf-8")
    assert "cache_hit_rate=80.00%" in text
    assert "Dataset token usage" in text
    assert "'prompt_tokens': 2000" in text


@pytest.mark.parametrize("base_url", ["https://api.deepseek.com/v1", "https://example.invalid/v1"])
def test_api_requests_use_provider_parameters_and_preserve_shared_prefix(
    distill, agent, monkeypatch, base_url
):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion_with_usage({
            "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
            "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200,
        }).model_dump(exclude_none=True))

    agent.tools = []
    monkeypatch.setenv("AGENT_MODEL_ID", "deepseek-flash")

    async def run():
        async with AsyncOpenAI(
            api_key="offline-test-key", base_url=base_url,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            for question in ("重庆路线", "成都路线"):
                await distill.MultiTurnReactAgent.call_server(agent, client, [
                    {"role": "system", "content": "相同的固定系统提示词"},
                    {"role": "user", "content": question},
                ])

    asyncio.run(run())
    assert requests[0]["messages"][0] == requests[1]["messages"][0]
    assert requests[0]["tools"] == requests[1]["tools"]
    assert "cache" not in requests[0] and "prompt_cache_key" not in requests[0]
    if "api.deepseek.com" in base_url:
        assert requests[0]["max_tokens"] == agent.max_tokens_per_turn
        assert requests[0]["thinking"] == {"type": "enabled"}
        assert "reasoning_split" not in requests[0]
    else:
        assert requests[0]["max_completion_tokens"] == agent.max_tokens_per_turn
        assert requests[0]["enable_thinking"] is True


def test_runtime_retries_are_bounded(distill, agent, monkeypatch):
    calls = 0

    async def fail(**kwargs):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise AssertionError("Exceeded configured retry budget")
        raise RuntimeError("temporary failure")

    agent.tools = []
    monkeypatch.setenv("AGENT_MODEL_ID", "deepseek-flash")
    client = SimpleNamespace(
        base_url="https://api.deepseek.com",
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail)),
    )
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        asyncio.run(distill.MultiTurnReactAgent.call_server(agent, client, [], max_attempts=2))
    assert calls == 2
