from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuits.analysis.bonafide.candidate_labeling_openai_batch import (
    openai_candidate_batch_line,
    openai_schema_name,
    parse_openai_candidate_batch_row,
    prepare_openai_candidate_batch_input,
)
from circuits.analysis.bonafide.candidate_labeling_renderer import (
    HELDOUT_FORBIDDEN_INPUTS,
    STATUS_ENUM,
    TYPED_OUTPUT_FIELDS,
)
from circuits.analysis.bonafide.candidate_labeling_runtime import (
    PreparedCandidateLabelingRequest,
)
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.labeling.schema import ChatMessage


def _schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(TYPED_OUTPUT_FIELDS),
        "properties": {
            "input_localization_hypothesis": {"type": "string", "minLength": 1},
            "exploratory_candidate_description": {
                "type": "string",
                "const": "not_available",
            },
            "background_or_confound": {"type": "string", "minLength": 1},
            "limitations": {"type": "string", "minLength": 1},
            "status": {"type": "string", "enum": list(STATUS_ENUM)},
        },
    }


def _request(**overrides: object) -> PreparedCandidateLabelingRequest:
    payload: dict[str, object] = {
        "request_id": "semantic_generation-1",
        "stage_id": "semantic_generation",
        "model_role": "semantic_generator",
        "logical_prompt_id": "arm_1_width_only:w64:00",
        "arm_id": "arm_1_width_only",
        "arm_sha256": "a" * 64,
        "anchor_index": 0,
        "cluster_id": 7,
        "sample_index": 0,
        "family_partition": "generation",
        "generation_only": True,
        "selection_audit_visible": False,
        "forbidden_input_fields": list(HELDOUT_FORBIDDEN_INPUTS),
        "source_renderer_manifest_sha256": "b" * 64,
        "source_prompt_sha256": "c" * 64,
        "source_message_payload_sha256": "d" * 64,
        "width_evidence_sha256": "e" * 64,
        "rendered_candidate_witness_sha256_in_order": None,
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "transport": "native_batch",
        "endpoint": None,
        "endpoints_resolved": False,
        "calls_made": False,
        "max_output_tokens": 3200,
        "temperature": 0.7,
        "reasoning": {"effort": "low"},
        "provider_parameters": {"store": False},
        "role_config_sha256": "f" * 64,
        "messages": [
            ChatMessage(role="system", content="bounded generation evidence"),
            ChatMessage(role="user", content="analyze these witnesses"),
        ],
        "expected_output_json_schema": _schema(),
        "typed_output_fields": list(TYPED_OUTPUT_FIELDS),
        "status_enum": list(STATUS_ENUM),
    }
    payload.update(overrides)
    payload["messages"] = [
        message.model_dump(mode="json") if isinstance(message, ChatMessage) else message
        for message in payload["messages"]  # type: ignore[union-attr]
    ]
    payload["request_sha256"] = canonical_sha256(payload)
    return PreparedCandidateLabelingRequest.model_validate(payload)


def _valid_output() -> dict[str, str]:
    return {
        "input_localization_hypothesis": "Repeated punctuation before the token.",
        "exploratory_candidate_description": "not_available",
        "background_or_confound": "The template may explain the pattern.",
        "limitations": "Only local single-target evidence is displayed.",
        "status": "provisional_description",
    }


def _row(
    request: PreparedCandidateLabelingRequest,
    *,
    text: str | None = None,
    status: str = "completed",
    content: list[dict[str, str]] | None = None,
) -> dict:
    if content is None:
        content = [] if text is None else [{"type": "output_text", "text": text}]
    return {
        "custom_id": request.request_id,
        "response": {
            "status_code": 200,
            "request_id": "batch-request-1",
            "body": {
                "id": "response-1",
                "model": request.model,
                "status": status,
                "output": [
                    {"type": "message", "status": "completed", "content": content}
                ],
                "usage": {
                    "input_tokens": 101,
                    "input_tokens_details": {"cached_tokens": 11},
                    "output_tokens": 29,
                    "output_tokens_details": {"reasoning_tokens": 7},
                },
            },
        },
        "error": None,
    }


def test_batch_line_uses_responses_structured_outputs_and_omits_temperature() -> None:
    request = _request()

    line = openai_candidate_batch_line(request)

    assert line == {
        "custom_id": request.request_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-5.6-luna",
            "input": [message.model_dump() for message in request.messages],
            "max_output_tokens": 3200,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": openai_schema_name(request),
                    "schema": request.expected_output_json_schema,
                    "strict": True,
                }
            },
            "reasoning": {"effort": "low"},
        },
    }
    assert "temperature" not in line["body"]
    assert openai_schema_name(request).startswith(
        "candidate_labeling_semantic_generation_"
    )


def test_batch_line_includes_temperature_without_reasoning() -> None:
    line = openai_candidate_batch_line(_request(reasoning={}))

    assert line["body"]["temperature"] == 0.7
    assert "reasoning" not in line["body"]


def test_prepare_is_ordered_jsonl_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    first = _request()
    second = _request(
        request_id="semantic_generation-2",
        sample_index=1,
    )
    destination = tmp_path / "batch.jsonl"

    prepare_openai_candidate_batch_input([first, second], destination)

    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["custom_id"] for row in rows] == [first.request_id, second.request_id]
    manifest = json.loads((tmp_path / "batch.jsonl.manifest.json").read_text())
    assert manifest["request_bindings_in_order"] == [
        {"request_id": first.request_id, "request_sha256": first.request_sha256},
        {"request_id": second.request_id, "request_sha256": second.request_sha256},
    ]
    with pytest.raises(ValueError, match="repeats request_id"):
        prepare_openai_candidate_batch_input([first, first], tmp_path / "bad.jsonl")
    with pytest.raises(ValueError, match="share stage and model"):
        prepare_openai_candidate_batch_input(
            [first, _request(request_id="other", model="gpt-5.6-terra")],
            tmp_path / "mixed-model.jsonl",
        )


def test_batch_line_enforces_generation_fence_and_reserved_parameters() -> None:
    with pytest.raises(ValueError, match="forbidden audit inputs"):
        openai_candidate_batch_line(
            _request(
                messages=[
                    ChatMessage(role="user", content="audit_evidence must stay hidden")
                ]
            )
        )
    with pytest.raises(ValueError, match="managed OpenAI fields"):
        openai_candidate_batch_line(_request(provider_parameters={"text": {}}))
    with pytest.raises(ValueError, match="unsupported.*provider_parameters"):
        openai_candidate_batch_line(
            _request(provider_parameters={"instructions": "audit data"})
        )


def test_parse_valid_output_validates_schema_usage_and_hashes() -> None:
    request = _request()
    row = _row(request, text=json.dumps(_valid_output()))

    result = parse_openai_candidate_batch_row(row, request)

    assert result.validation_status == "success"
    assert result.parsed_output == _valid_output()
    assert result.request_sha256 == request.request_sha256
    assert result.provider_request_id == "batch-request-1"
    assert result.usage.input_tokens == 101
    assert result.usage.uncached_input_tokens == 90
    assert result.usage.cache_read_tokens == 11
    assert result.usage.output_tokens == 29
    assert result.usage.reasoning_tokens == 7
    assert result.raw_text_sha256 is not None
    assert result.raw_response_sha256 is not None
    assert result.raw_row_sha256 == canonical_sha256(row)


@pytest.mark.parametrize(
    ("text", "expected_status"),
    [
        ("not json", "invalid_json"),
        (json.dumps({**_valid_output(), "status": "unsupported"}), "schema_invalid"),
        (json.dumps({**_valid_output(), "extra": "field"}), "schema_invalid"),
    ],
)
def test_parse_rejects_malformed_or_schema_invalid_output(
    text: str, expected_status: str
) -> None:
    request = _request()

    result = parse_openai_candidate_batch_row(_row(request, text=text), request)

    assert result.validation_status == expected_status
    assert result.error_type == expected_status


def test_parse_refusal_does_not_accept_text_as_output() -> None:
    request = _request()
    row = _row(
        request,
        content=[{"type": "refusal", "refusal": "I cannot do that."}],
    )

    result = parse_openai_candidate_batch_row(row, request)

    assert result.validation_status == "refusal"
    assert result.refusal == "I cannot do that."
    assert result.stop_reason == "refusal"
    assert result.parsed_output is None


def test_parse_empty_refusal_still_fails_closed() -> None:
    request = _request()
    row = _row(
        request,
        content=[
            {"type": "output_text", "text": json.dumps(_valid_output())},
            {"type": "refusal", "refusal": ""},
        ],
    )

    result = parse_openai_candidate_batch_row(row, request)

    assert result.validation_status == "refusal"
    assert result.refusal == ""
    assert result.parsed_output is None


def test_parse_rejects_incomplete_message_under_completed_response() -> None:
    request = _request()
    row = _row(request, text=json.dumps(_valid_output()))
    row["response"]["body"]["output"][0]["status"] = "incomplete"

    result = parse_openai_candidate_batch_row(row, request)

    assert result.validation_status == "incomplete"
    assert result.error_type == "incomplete_message"


def test_parse_incomplete_preserves_reason_without_accepting_partial_json() -> None:
    request = _request()
    row = _row(request, text='{"input_localization_hypothesis":', status="incomplete")
    row["response"]["body"]["incomplete_details"] = {  # type: ignore[index]
        "reason": "max_output_tokens"
    }

    result = parse_openai_candidate_batch_row(row, request)

    assert result.validation_status == "incomplete"
    assert result.stop_reason == "max_output_tokens"
    assert result.raw_text is not None
    assert result.parsed_output is None


def test_parse_requires_explicit_completed_response_status() -> None:
    request = _request()
    row = _row(request, text=json.dumps(_valid_output()))
    row["response"]["body"].pop("status")

    result = parse_openai_candidate_batch_row(row, request)

    assert result.validation_status == "provider_error"
    assert result.error_type == "response_status:missing"
    assert result.parsed_output is None


@pytest.mark.parametrize(
    "row",
    [
        {
            "custom_id": "semantic_generation-1",
            "response": None,
            "error": {"code": "rate_limit_exceeded", "message": "retry"},
        },
        {
            "custom_id": "semantic_generation-1",
            "response": {"status_code": 400, "body": {"error": "bad request"}},
            "error": None,
        },
    ],
)
def test_parse_provider_error_rows(row: dict) -> None:
    result = parse_openai_candidate_batch_row(row, _request())

    assert result.validation_status == "provider_error"
    assert result.error_type is not None
    assert result.raw_row_sha256 == canonical_sha256(row)
