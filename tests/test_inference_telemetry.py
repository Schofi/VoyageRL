import json
import math

import pytest

from utils.inference_telemetry import RequestMetrics, tool_failed


def test_event_has_only_metrics_and_separates_model_and_tool_time(tmp_path):
    ticks = iter([10, 11, 14, 15, 19, 21])
    metrics = RequestMetrics(tmp_path / 'metrics.jsonl', 'run-1', 'req-1', 'transformers', 'model', clock=lambda: next(ticks))
    with metrics.stage('model_s'):
        pass
    with metrics.stage('tool_s'):
        metrics.record_tool_result('TOOL_ERROR: secret diagnostic')
    metrics.finish({'status': 'answer', 'turns': 2, 'prediction': 'private answer', 'query': 'private query'})
    event = json.loads((tmp_path / 'metrics.jsonl').read_text())
    assert event['e2e_s'] == 11
    assert event['model_s'] == 3 and event['tool_s'] == 4
    assert event['tool_calls'] == event['tool_errors'] == 1
    assert event['success'] is True and event['turns'] == 2
    assert event['ttft_s'] is None and event['tpot_s'] is None
    assert event['output_tokens'] is None
    assert 'private' not in json.dumps(event) and 'secret' not in json.dumps(event)


def test_exception_event_and_append_without_truncating(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    for request in ['a', 'b']:
        metric = RequestMetrics(path, 'run', request, 'vllm', 'test')
        with pytest.raises(RuntimeError), metric.stage('tool_s'):
            raise RuntimeError('private exception')
        metric.finish(error_type='RuntimeError')
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(events) == 2
    assert all(e['status'] == 'error' and e['success'] is False and e['tool_errors'] == 1 for e in events)
    assert all(math.isfinite(e['e2e_s']) and e['e2e_s'] >= 0 for e in events)


def test_token_estimate_is_labeled_and_failed_count_is_unknown(tmp_path):
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text)
    metric = RequestMetrics(tmp_path / 'events', 'run', 'r', 'vllm', 'test')
    metric.record_output('abc', Tokenizer())
    metric.record_output('de', Tokenizer())
    event = metric.finish({'status': 'answer'})
    assert event['output_tokens'] == 5
    assert event['output_tokens_source'] == 'retokenized_estimate'
    metric2 = RequestMetrics(tmp_path / 'events', 'run', 'r2', 'vllm', 'test')
    metric2.record_output('abc', object())
    metric2.record_output('de', Tokenizer())
    assert metric2.finish({'status': 'answer'})['output_tokens'] is None


def test_io_failure_does_not_interrupt_inference(tmp_path, caplog):
    metric = RequestMetrics(tmp_path, 'run', 'r', 'vllm', 'test')  # directory is not writable as a file
    assert metric.finish({'status': 'answer'})['success'] is True
    assert 'telemetry' in caplog.text.lower()


@pytest.mark.parametrize('result,failed', [('TOOL_ERROR: timeout', True), ('INVALID_PARAMS', True), ('weather fine', False), ({'error': 'bad'}, True), ({'status': 'error'}, True), ({'status': 'ok'}, False)])
def test_tool_result_classification(result, failed):
    assert tool_failed(result) is failed


@pytest.mark.parametrize('result', [
    'API response error: INVALID_USER_KEY',
    '[WeatherSearch] Invalid request format: Input must be a JSON object',
    '[POISearch] Invalid request format: address missing',
    'Firecrawl search Timeout or error (TimeoutError: timeout); return None',
    'valid result\n=======\nFirecrawl search Timeout or error (TimeoutError: timeout)',
    '[visit] Failed to read page.',
    '[train_tickets_search] Failed to get response: TimeoutError: timeout',
    'Error fetching content https://example.test: timeout',
    'The useful information in https://example.test for user goal test as follows: \n\nEvidence in page: \nThe provided webpage content could not be accessed. Please check the URL or file format.\n\nSummary: \nThe webpage content could not be processed, and therefore, no information is available.',
])
def test_existing_tool_protocol_errors_are_failures(result):
    assert tool_failed(result)


@pytest.mark.parametrize('result', [
    "No results found for query: 'test'. Use a less specific query.",
    'No POI data available for test.',
    'Article about API response error: how to troubleshoot',
])
def test_empty_search_or_error_words_in_content_are_not_failures(result):
    assert not tool_failed(result)
