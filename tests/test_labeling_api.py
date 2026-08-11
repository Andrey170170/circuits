from __future__ import annotations

import asyncio
import json

from circuits.labeling.api import (
    FakeBackend,
    anthropic_usage,
    openai_usage,
    parse_json_output,
)
from circuits.labeling.batch import anthropic_batch_request, openai_batch_line
from circuits.labeling.config import ModelRoleConfig
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


def test_width_one_parser_requires_explicit_limit_and_status_fields() -> None:
    assert (
        parse_json_output(
            '{"description":"x"}',
            "candidate_generation",
            "bonafide-width-one-cluster-candidate-v2",
        )[1]
        == "invalid_json"
    )
    parsed, status = parse_json_output(
        '{"label":"x","rationale":"r","confidence":0.3,'
        '"background_or_confound":"b","limitations":"l",'
        '"status":"provisional_label"}',
        "cluster_summary",
        "bonafide-width-one-cluster-summary-v2",
    )
    assert status == "success"
    assert parsed is not None
    assert parsed["status"] == "provisional_label"
    assert (
        parse_json_output(
            '{"label":"","rationale":"r","confidence":0.3,'
            '"background_or_confound":"b","limitations":"l",'
            '"status":"provisional_label"}',
            "cluster_summary",
            "bonafide-width-one-cluster-summary-v2",
        )[1]
        == "invalid_json"
    )


def test_hybrid_parser_requires_exact_candidate_and_summary_schemas() -> None:
    candidate_version = "bonafide-hybrid-candidate-cluster-candidate-v1"
    assert (
        parse_json_output(
            '{"description":"x"}', "candidate_generation", candidate_version
        )[1]
        == "invalid_json"
    )
    candidate = (
        '{"description":"x","localized_evidence":"local",'
        '"candidate_comparison_evidence":"top5","limitations":"exploratory"}'
    )
    assert parse_json_output(candidate, "candidate_generation", candidate_version)[
        1
    ] == ("success")
    summary_version = "bonafide-hybrid-candidate-cluster-summary-v1"
    summary = (
        '{"label":"feature","rationale":"r","confidence":0.5,'
        '"candidate_comparison_evidence":"top5","limitations":"l",'
        '"status":"provisional_label"}'
    )
    assert parse_json_output(summary, "cluster_summary", summary_version)[1] == (
        "success"
    )
    invalid = json.loads(summary)
    invalid["unexpected"] = "field"
    assert (
        parse_json_output(json.dumps(invalid), "cluster_summary", summary_version)[1]
        == "invalid_json"
    )


def test_fake_backend_emits_strict_hybrid_candidate_schema() -> None:
    request = _request("fake").model_copy(
        update={
            "prompt_template_version": (
                "bonafide-hybrid-candidate-cluster-candidate-v1"
            )
        }
    )
    backend = FakeBackend(
        ModelRoleConfig(provider="fake", model="fake", max_output_tokens=100)
    )
    result = asyncio.run(backend.generate(request))
    assert result.parse_status == "success"
    assert set(result.parsed or {}) == {
        "description",
        "localized_evidence",
        "candidate_comparison_evidence",
        "limitations",
    }
    summary_request = request.model_copy(
        update={
            "stage": "cluster_summary",
            "sample_index": None,
            "prompt_template_version": "bonafide-hybrid-candidate-cluster-summary-v1",
        }
    )
    summary_result = asyncio.run(backend.generate(summary_request))
    assert summary_result.parse_status == "success"
    assert set(summary_result.parsed or {}) == {
        "label",
        "rationale",
        "confidence",
        "candidate_comparison_evidence",
        "limitations",
        "status",
    }


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
