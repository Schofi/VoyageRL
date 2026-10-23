"""Optional, content-free telemetry for offline inference (standard library only)."""

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterator

LOGGER = logging.getLogger(__name__)


def tool_failed(result: Any) -> bool:
    """Count protocol/runtime failures, not whether returned facts are useful."""
    if isinstance(result, dict):
        return bool(result.get('error')) or result.get('status') in ('error', 'failed')
    text = str(result).lstrip()
    # These prefixes are emitted by the existing travel tools. Do not classify
    # arbitrary occurrences of 'error' in search results or 'no results' as failures.
    if text.startswith(('TOOL_ERROR', 'INVALID_PARAMS', 'API response error:')):
        return True
    if re.match(r'^\[[A-Za-z_]+\] (Invalid request format:|Failed to get response:)', text):
        return True
    return any(
        part.strip().startswith(('Firecrawl search Timeout or error (', '[visit] Failed to read page.', 'Error fetching content '))
        or (part.strip().startswith('The useful information in ')
            and 'The provided webpage content could not be accessed. Please check the URL or file format.' in part
            and 'The webpage content could not be processed, and therefore, no information is available.' in part)
        for part in text.split('\n=======\n')
    )


class RequestMetrics:
    """One terminal JSONL event per request, including failed requests.

    A single inference process owns each log file. Multiple processes should use
    separate files configured as separate sources in GPU-Monitor.
    """

    def __init__(self, path: str | Path, run_id: str, request_id: str,
                 backend: str, model: str, *, clock: Callable[[], float] = time.perf_counter):
        self.path = Path(path)
        self.clock = clock
        self.started = clock()
        self.event: dict[str, Any] = {
            'schema_version': 1, 'run_id': run_id, 'request_id': request_id,
            'backend': backend, 'model': model, 'model_s': 0.0, 'tool_s': 0.0,
            'output_tokens': None, 'output_tokens_source': 'retokenized_estimate',
            'tool_calls': 0, 'tool_errors': 0, 'turns': 0,
            'ttft_s': None, 'tpot_s': None,
        }
        self.token_count = 0
        self.token_count_valid = True
        self.outputs_seen = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in ('model_s', 'tool_s'):
            raise ValueError('Unknown inference measurement stage')
        started = self.clock()
        if name == 'tool_s':
            self.event['tool_calls'] += 1
        try:
            yield
        except Exception:
            if name == 'tool_s':
                self.event['tool_errors'] += 1
            raise
        finally:
            self.event[name] += max(0.0, self.clock() - started)

    def record_output(self, text: str, tokenizer: Any) -> None:
        self.outputs_seen += 1
        try:
            self.token_count += len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            self.token_count_valid = False

    def record_tool_result(self, result: Any) -> None:
        if tool_failed(result):
            self.event['tool_errors'] += 1

    def finish(self, result: dict[str, Any] | None = None, *, error_type: str | None = None) -> dict[str, Any]:
        result = result or {}
        event = {
            **self.event,
            'timestamp': time.time(),
            'status': 'error' if error_type else result.get('status', 'error'),
            'success': error_type is None and result.get('status') == 'answer',
            'e2e_s': max(0.0, self.clock() - self.started),
            'turns': result.get('turns', self.event['turns']),
            'output_tokens': self.token_count if self.outputs_seen and self.token_count_valid else None,
        }
        if error_type:
            event['error_type'] = error_type  # Never include exception messages or prompts.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        except (OSError, ValueError, TypeError):
            LOGGER.warning('Inference telemetry could not be written; inference continues.')
        return event
