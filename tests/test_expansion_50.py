"""离线验证 50 条上限和停止条件，不调用真实 API。"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError


PROJECT_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "expansion_50_under_test", PROJECT_DIR / "test_expand_50.py"
)
expansion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(expansion)


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.answer(len(self.requests), kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )


@pytest.fixture
def paths(tmp_path):
    seeds = tmp_path / "seed.jsonl"
    output = tmp_path / "expansion.jsonl"
    seeds.write_text(
        "".join(json.dumps({"question": f"种子问题{i}"}, ensure_ascii=False) + "\n"
                for i in range(60)),
        encoding="utf-8",
    )
    return seeds, output


def read_questions(output):
    return [json.loads(line)["question"]
            for line in output.read_text(encoding="utf-8").splitlines()]


def test_stops_after_exactly_50_new_questions(paths):
    client = FakeClient(lambda n, _: "\n".join(f"{i + 1}. 新问题{n}-{i}" for i in range(5)))
    seeds, output = paths
    count = expansion.expand_to_50(client, seeds, output)
    assert count == 50
    assert len(read_questions(output)) == 50
    assert len(client.requests) == 10
    assert all(request["model"] == "ZHIPU/GLM-5.3-Flash"
               for request in client.requests)


def test_extra_lines_are_trimmed_and_final_request_asks_only_for_remainder(paths):
    client = FakeClient(lambda n, _: "\n".join(f"新问题{n}-{i}" for i in range(7)))
    seeds, output = paths
    assert expansion.expand_to_50(client, seeds, output) == 50
    assert len(read_questions(output)) == 50
    assert len(client.requests) == 8
    assert "生成1个" in client.requests[-1]["messages"][-1]["content"]


def test_deduplicates_and_excludes_all_original_seeds(paths):
    def answer(n, _):
        return f"1. 种子问题59\n2. 重复问题\n3. 重复问题\n4. 新问题{n}"

    client = FakeClient(answer)
    seeds, output = paths
    assert expansion.expand_to_50(client, seeds, output, max_api_calls=4) == 5
    questions = read_questions(output)
    assert len(questions) == len(set(questions)) == 5
    assert "种子问题59" not in questions
    assert len(client.requests) == 4


def test_no_progress_stops_after_three_calls(paths):
    client = FakeClient(lambda n, _: "种子问题0\n\n种子问题0")
    seeds, output = paths
    assert expansion.expand_to_50(client, seeds, output) == 0
    assert read_questions(output) == []
    assert len(client.requests) == 3


def test_request_errors_are_bounded_and_keep_partial_output(paths):
    def answer(n, _):
        if n == 1:
            return "\n".join(f"新问题{i}" for i in range(5))
        raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))

    client = FakeClient(answer)
    seeds, output = paths
    assert expansion.expand_to_50(client, seeds, output) == 5
    assert len(read_questions(output)) == 5
    assert len(client.requests) == 4


def test_authentication_error_stops_immediately(paths):
    def answer(n, _):
        response = httpx.Response(
            401, request=httpx.Request("POST", "https://example.invalid")
        )
        raise AuthenticationError("invalid credential", response=response, body={})

    client = FakeClient(answer)
    seeds, output = paths
    assert expansion.expand_to_50(client, seeds, output) == 0
    assert len(client.requests) == 1


def test_rejects_same_input_and_output_without_changing_seed(paths):
    seeds, _ = paths
    before = seeds.read_bytes()
    client = FakeClient(lambda n, _: "新问题")
    with pytest.raises(ValueError, match="不同"):
        expansion.expand_to_50(client, seeds, seeds)
    assert seeds.read_bytes() == before
    assert not client.requests


def test_client_uses_timeout_and_disables_sdk_retries():
    with patch.dict(os.environ, {}, clear=True), \
         patch.object(expansion, "API_KEY", "unit-test-key"), \
         patch.object(expansion, "BASE_URL", "https://example.invalid/v1"), \
         patch.object(expansion, "OpenAI") as constructor:
        expansion.build_client()
    assert constructor.call_args.kwargs["api_key"] == "unit-test-key"
    assert constructor.call_args.kwargs["base_url"] == "https://example.invalid/v1"
    assert constructor.call_args.kwargs["max_retries"] == 0
    assert constructor.call_args.kwargs["timeout"] == 60.0


def test_main_returns_failure_when_fewer_than_50_are_generated():
    with patch.object(expansion, "build_client"), \
         patch.object(expansion, "expand_to_50", return_value=12):
        assert expansion.main() == 1


def test_main_uses_inline_model_even_if_environment_has_another_model():
    with patch.dict(os.environ, {"GLM_MODEL": "unrelated-model"}), \
         patch.object(expansion, "build_client"), \
         patch.object(expansion, "expand_to_50", return_value=50) as expand:
        assert expansion.main() == 0
    assert expand.call_args.kwargs["model"] == expansion.MODEL
