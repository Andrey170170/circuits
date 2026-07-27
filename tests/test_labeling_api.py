from __future__ import annotations

from circuits.labeling.api import anthropic_usage, openai_usage, parse_json_output
from circuits.labeling.batch import anthropic_batch_request, openai_batch_line
from circuits.labeling.schema import ChatMessage, GenerationRequest


def _request(provider: str) -> GenerationRequest:
    return GenerationRequest(
        request_id="request-1",
        run_id="run-1",
        recipe_id="recipe-1",
        stage="candidate_generation",
        state="primary",
        cluster_id=7,
        sample_index=2,
        evidence_partition_id="generation",
        provider=provider,
        model="model-1",
        transport="native_batch",
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="user"),
        ],
        max_output_tokens=100,
        temperature=0.5,
        reasoning={"effort": "low"},
        prompt_template_version="v1",
        prompt_sha256="a" * 64,
        evidence_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )


def test_json_parser_accepts_fenced_output_and_rejects_wrong_schema() -> None:
    parsed, status = parse_json_output(
        '```json\n{"description": "input-copying feature"}\n```',
        "candidate_generation",
    )
    assert status == "success"
    assert parsed == {"description": "input-copying feature"}
    assert parse_json_output('{"label": "x"}', "cluster_summary")[1] == "invalid_json"


def test_openai_batch_line_targets_responses() -> None:
    line = openai_batch_line(_request("openai"))
    assert line["custom_id"] == "request-1"
    assert line["url"] == "/v1/responses"
    assert line["body"]["reasoning"] == {"effort": "low"}
    assert line["body"]["input"][0]["role"] == "system"


def test_anthropic_batch_request_separates_system_message() -> None:
    value = anthropic_batch_request(_request("anthropic"))
    assert value["params"]["system"] == "system"
    assert value["params"]["messages"] == [{"role": "user", "content": "user"}]
    assert value["params"]["thinking"] == {"effort": "low"}


def test_usage_normalization() -> None:
    openai = openai_usage(
        {
            "input_tokens": 120,
            "output_tokens": 30,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 10},
        }
    )
    assert openai.uncached_input_tokens == 100
    assert openai.cache_read_tokens == 20
    assert openai.reasoning_tokens == 10

    anthropic = anthropic_usage(
        {
            "input_tokens": 100,
            "output_tokens": 30,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 20,
        }
    )
    assert anthropic.uncached_input_tokens == 100
    assert anthropic.cache_write_tokens == 5
    assert anthropic.cache_read_tokens == 20
