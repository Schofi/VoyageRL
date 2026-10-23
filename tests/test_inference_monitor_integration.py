"""No model download, GPU, network calls, or environment secret loading."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    for name in ('torch', 'transformers', 'dotenv'):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['transformers'].AutoModelForCausalLM = object
    sys.modules['transformers'].AutoTokenizer = object
    sys.modules['dotenv'].load_dotenv = lambda: None
    path = Path(__file__).resolve().parents[1] / 'run_tool_loop_infer.py'
    spec = importlib.util.spec_from_file_location('monitored_inference_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(tmp_path):
    dataset = tmp_path / 'test.jsonl'
    dataset.write_text(json.dumps({'id': 'sample', 'messages': [
        {'role': 'system', 'content': 'You are a test agent.'},
        {'role': 'user', 'content': 'private travel preferences'},
    ]}) + '\n', encoding='utf-8')
    return SimpleNamespace(
        metrics_output=str(tmp_path / 'metrics.jsonl'), metrics_run_id='integration-run',
        infer_backend='transformers', model_dir='test/model', tools_dir=str(tmp_path / 'tools'),
        dataset_path=str(dataset), override_system_prompt_file='', system_max_tool_calls=3,
        quiet=True, max_turns=3, tool_first_enforce=True, max_no_tool_no_answer_retries=1,
        tool_response_max_chars=5000, max_invalid_tool_rounds=2, max_same_tool_call_rounds=2,
        force_answer_after_turns=3, save_full_messages=False,
    )


def test_tool_loop_emits_one_success_event_and_preserves_result(runner, args, monkeypatch):
    responses = iter(['<tool_call>{"name":"weather_search","arguments":{"city":"test"}}</tool_call>', '<answer>private answer</answer>'])
    monkeypatch.setattr(runner, 'infer_once', lambda *a: next(responses))
    class Tool:
        async def call(self, params):
            return 'sunny'
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text)
    monkeypatch.setattr(runner, 'load_tool', lambda *a: Tool())
    result = asyncio.run(runner.run_single(args, Tokenizer(), None, None, 0))
    assert result['status'] == 'answer' and result['prediction'] == 'private answer'
    events = Path(args.metrics_output).read_text().splitlines()
    assert len(events) == 1
    event = json.loads(events[0])
    assert event['success'] and event['turns'] == 2
    assert event['tool_calls'] == 1 and event['tool_errors'] == 0
    assert event['model_s'] >= 0 and event['tool_s'] >= 0
    assert event['e2e_s'] >= event['model_s'] + event['tool_s']
    assert event['output_tokens'] > 0 and event['ttft_s'] is None
    assert 'private' not in events[0]


def test_failed_generation_still_records_one_terminal_event(runner, args, monkeypatch):
    def fail(*unused):
        raise RuntimeError('private model error')
    monkeypatch.setattr(runner, 'infer_once', fail)
    with pytest.raises(RuntimeError):
        asyncio.run(runner.run_single(args, None, None, None, 0))
    event = json.loads(Path(args.metrics_output).read_text())
    assert event['status'] == 'error' and event['error_type'] == 'RuntimeError'
    assert event['success'] is False and event['turns'] == 1
    assert 'private' not in json.dumps(event)


def test_tool_error_count_and_disabled_monitoring(runner, args, monkeypatch):
    responses = iter(['<tool_call>{"name":"weather_search","arguments":{"city":"test"}}</tool_call>', '<answer>answer</answer>'])
    monkeypatch.setattr(runner, 'infer_once', lambda *a: next(responses))
    class BrokenTool:
        async def call(self, params):
            raise TimeoutError('tool unavailable')
    monkeypatch.setattr(runner, 'load_tool', lambda *a: BrokenTool())
    asyncio.run(runner.run_single(args, None, None, None, 0))
    assert json.loads(Path(args.metrics_output).read_text())['tool_errors'] == 1
    before = Path(args.metrics_output).read_bytes()
    args.metrics_output = ''
    args.tool_first_enforce = False
    monkeypatch.setattr(runner, 'infer_once', lambda *a: '<answer>answer</answer>')
    assert asyncio.run(runner.run_single(args, None, None, None, 0))['status'] == 'answer'
    assert (Path(args.dataset_path).parent / 'metrics.jsonl').read_bytes() == before


def test_invalid_sample_is_an_observed_failure(runner, args):
    Path(args.dataset_path).write_text('{"messages": []}\n')
    result = asyncio.run(runner.run_single(args, None, None, None, 0))
    assert result['status'] == 'invalid_sample'
    event = json.loads(Path(args.metrics_output).read_text())
    assert event['status'] == 'invalid_sample' and not event['success']
