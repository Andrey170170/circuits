"""Pure OpenAI Batch formatting and parsing for candidate-labeling requests.

This module deliberately has no client construction or network operations.  It
turns an already provenance-checked, generation-only request into one Responses
Batch JSONL row and validates a returned row against that request's exact output
schema.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.api import openai_usage
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl
from circuits.labeling.schema import ChatMessage, StrictModel, Usage

RESULT_SCHEMA = "adag.bonafide.candidate-labeling-openai-batch-result.v1"
BATCH_INPUT_MANIFEST_SCHEMA = "adag.bonafide.candidate-labeling-openai-batch-input.v1"
OPENAI_BATCH_ENDPOINT = "/v1/responses"
_RESERVED_BODY_FIELDS = {
    "input",
    "max_output_tokens",
    "model",
    "reasoning",
    "temperature",
    "text",
}
_ALLOWED_PROVIDER_PARAMETERS = {
    "metadata",
    "prompt_cache_key",
    "safety_identifier",
    "service_tier",
    "store",
    "user",
}
_SCHEMA_NAME_FRAGMENT = re.compile(r"[^A-Za-z0-9_-]+")


class CandidateBatchRequest(Protocol):
    """Structural request surface shared by initial and future rewrite calls."""

    request_id: str
    request_sha256: str
    stage_id: str
    logical_prompt_id: str
    provider: str
    model: str
    transport: str
    generation_only: bool
    selection_audit_visible: bool
    forbidden_input_fields: list[str]
    max_output_tokens: int
    temperature: float | None
    reasoning: dict[str, Any]
    provider_parameters: dict[str, Any]
    messages: list[ChatMessage]
    expected_output_json_schema: dict[str, Any]


class CandidateOpenAIBatchResult(StrictModel):
    """Deterministic, typed interpretation of one OpenAI Batch result row."""

    schema_version: Literal[
        "adag.bonafide.candidate-labeling-openai-batch-result.v1"
    ] = RESULT_SCHEMA
    request_id: str
    request_sha256: str
    stage_id: str
    logical_prompt_id: str
    provider: Literal["openai"] = "openai"
    provider_request_id: str | None = None
    model_requested: str
    model_resolved: str | None = None
    response_status: str | None = None
    validation_status: Literal[
        "success",
        "empty",
        "invalid_json",
        "schema_invalid",
        "refusal",
        "incomplete",
        "provider_error",
    ]
    parsed_output: dict[str, Any] | None = None
    raw_text: str | None = None
    raw_text_sha256: str | None = None
    refusal: str | None = None
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    error_type: str | None = None
    error_message: str | None = None
    raw_response_sha256: str | None = None
    raw_row_sha256: str


def _request_messages(request: CandidateBatchRequest) -> list[dict[str, Any]]:
    return [message.model_dump(mode="json") for message in request.messages]


def _validate_generation_fence(request: CandidateBatchRequest) -> None:
    if request.provider != "openai":
        raise ValueError("OpenAI Batch formatting requires provider='openai'")
    if request.transport != "native_batch":
        raise ValueError("OpenAI Batch formatting requires native_batch transport")
    if request.generation_only is not True:
        raise ValueError("candidate-labeling batch request must be generation-only")
    if request.selection_audit_visible is not False:
        raise ValueError("selection/audit evidence must remain hidden from generation")
    serialized_messages = json.dumps(
        _request_messages(request), sort_keys=True, separators=(",", ":")
    )
    leaked = sorted(
        field
        for field in request.forbidden_input_fields
        if field in serialized_messages
    )
    if leaked:
        raise ValueError(
            "candidate-labeling messages contain forbidden audit inputs: "
            + ", ".join(leaked)
        )


def openai_schema_name(request: CandidateBatchRequest) -> str:
    """Return a stable valid name that distinguishes differing arm schemas."""

    stage = _SCHEMA_NAME_FRAGMENT.sub("_", request.stage_id).strip("_-") or "stage"
    schema_suffix = canonical_sha256(request.expected_output_json_schema)[:12]
    return f"candidate_labeling_{stage}_{schema_suffix}"[:64]


def openai_candidate_batch_line(
    request: CandidateBatchRequest,
) -> dict[str, Any]:
    """Format one candidate-labeling request for OpenAI's Responses Batch API."""

    _validate_generation_fence(request)
    overlap = sorted(_RESERVED_BODY_FIELDS.intersection(request.provider_parameters))
    if overlap:
        raise ValueError(
            "provider_parameters may not override managed OpenAI fields: "
            + ", ".join(overlap)
        )
    unsupported = sorted(
        set(request.provider_parameters) - _ALLOWED_PROVIDER_PARAMETERS
    )
    if unsupported:
        raise ValueError(
            "unsupported candidate-labeling OpenAI provider_parameters: "
            + ", ".join(unsupported)
        )
    body: dict[str, Any] = {
        "model": request.model,
        "input": _request_messages(request),
        "max_output_tokens": request.max_output_tokens,
        **request.provider_parameters,
        "text": {
            "format": {
                "type": "json_schema",
                "name": openai_schema_name(request),
                "schema": request.expected_output_json_schema,
                "strict": True,
            }
        },
    }
    if request.reasoning:
        body["reasoning"] = request.reasoning
    elif request.temperature is not None:
        body["temperature"] = request.temperature
    return {
        "custom_id": request.request_id,
        "method": "POST",
        "url": OPENAI_BATCH_ENDPOINT,
        "body": body,
    }


def prepare_openai_candidate_batch_input(
    requests: Iterable[CandidateBatchRequest], destination: Path
) -> None:
    """Atomically write deterministic JSONL in caller-specified request order."""

    values = list(requests)
    if not values:
        raise ValueError("candidate-labeling batch input cannot be empty")
    request_ids = [request.request_id for request in values]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("candidate-labeling batch input repeats request_id")
    stage_models = {(request.stage_id, request.model) for request in values}
    if len(stage_models) != 1:
        raise ValueError(
            "one candidate-labeling batch input must share stage and model"
        )
    atomic_write_jsonl(
        destination,
        (openai_candidate_batch_line(request) for request in values),
    )
    manifest = {
        "schema_version": BATCH_INPUT_MANIFEST_SCHEMA,
        "stage_id": values[0].stage_id,
        "model": values[0].model,
        "request_count": len(values),
        "request_bindings_in_order": [
            {
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
            }
            for request in values
        ],
        "input_file": destination.name,
        "input_file_sha256": file_sha256(destination),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(
        destination.with_name(f"{destination.name}.manifest.json"), manifest
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text_and_refusal(
    body: Mapping[str, Any],
) -> tuple[str, str | None, list[str | None]]:
    texts: list[str] = []
    refusals: list[str] = []
    message_statuses: list[str | None] = []
    output = body.get("output", [])
    if not isinstance(output, list):
        return "", None, message_statuses
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        message_statuses.append(
            str(item["status"]) if item.get("status") is not None else None
        )
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, Mapping):
                continue
            block_type = block.get("type")
            if block_type in {"output_text", "text"}:
                texts.append(str(block.get("text", "")))
            elif block_type == "refusal":
                refusals.append(str(block.get("refusal", "")))
    refusal = "\n".join(refusals) if refusals else None
    return "".join(texts), refusal, message_statuses


def _candidate_schema_error(
    value: Mapping[str, Any], schema: Mapping[str, Any]
) -> str | None:
    """Validate the deliberately small, frozen candidate-labeling schema subset."""

    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or not isinstance(schema.get("required"), list)
        or not isinstance(schema.get("properties"), Mapping)
    ):
        return "unsupported candidate-labeling output schema"
    required = schema["required"]
    properties = schema["properties"]
    if (
        any(not isinstance(field, str) for field in required)
        or set(required) != set(properties)
        or set(value) != set(required)
    ):
        return "output fields do not exactly match the required schema fields"
    for field in required:
        rule = properties[field]
        if not isinstance(rule, Mapping):
            return f"unsupported schema rule for {field}"
        item = value[field]
        allowed_keys = {"type", "minLength", "enum", "const"}
        if set(rule) - allowed_keys:
            return f"unsupported schema keyword for {field}"
        if "const" in rule:
            if set(rule) != {"const"} or item != rule["const"]:
                return f"{field} does not match its required constant"
            continue
        if rule.get("type") != "string" or not isinstance(item, str):
            return f"{field} must be a string"
        minimum = rule.get("minLength")
        if minimum is not None and (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or len(item) < minimum
        ):
            return f"{field} is shorter than minLength"
        allowed = rule.get("enum")
        if allowed is not None and (
            not isinstance(allowed, list) or item not in allowed
        ):
            return f"{field} is outside its enum"
    return None


def _provider_error(
    *,
    row: Mapping[str, Any],
    request: CandidateBatchRequest,
    detail: Any,
    error_type: str,
    provider_request_id: str | None = None,
    raw_response_sha256: str | None = None,
) -> CandidateOpenAIBatchResult:
    return CandidateOpenAIBatchResult(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
        stage_id=request.stage_id,
        logical_prompt_id=request.logical_prompt_id,
        provider_request_id=provider_request_id,
        model_requested=request.model,
        validation_status="provider_error",
        error_type=error_type,
        error_message=json.dumps(detail, sort_keys=True, default=str)[:2000],
        raw_response_sha256=raw_response_sha256,
        raw_row_sha256=canonical_sha256(row),
    )


def parse_openai_candidate_batch_row(
    row: Mapping[str, Any], request: CandidateBatchRequest
) -> CandidateOpenAIBatchResult:
    """Parse one output/error row without accepting partial or refused content."""

    _validate_generation_fence(request)
    if row.get("custom_id") != request.request_id:
        raise ValueError("OpenAI batch row custom_id does not match request_id")
    response = row.get("response")
    row_error = row.get("error")
    if row_error is not None or not isinstance(response, Mapping):
        code = row_error.get("code") if isinstance(row_error, Mapping) else None
        error_type = "batch_request_error" + (f":{code}" if code else "")
        return _provider_error(
            row=row,
            request=request,
            detail=row_error if row_error is not None else row,
            error_type=error_type,
        )
    provider_request_id = response.get("request_id")
    if provider_request_id is not None:
        provider_request_id = str(provider_request_id)
    body = response.get("body")
    body_hash = canonical_sha256(body) if isinstance(body, Mapping) else None
    status_code = response.get("status_code")
    if status_code != 200 or not isinstance(body, Mapping):
        return _provider_error(
            row=row,
            request=request,
            detail=body if body is not None else response,
            error_type="batch_request_error",
            provider_request_id=provider_request_id,
            raw_response_sha256=body_hash,
        )

    provider_request_id = provider_request_id or (
        str(body["id"]) if body.get("id") is not None else None
    )
    response_status = str(body["status"]) if body.get("status") is not None else None
    text, refusal, message_statuses = _text_and_refusal(body)
    text_hash = _sha256_text(text) if text else None
    usage = openai_usage(body.get("usage"))
    common: dict[str, Any] = {
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "stage_id": request.stage_id,
        "logical_prompt_id": request.logical_prompt_id,
        "provider_request_id": provider_request_id,
        "model_requested": request.model,
        "model_resolved": (
            str(body["model"]) if body.get("model") is not None else None
        ),
        "response_status": response_status,
        "raw_text": text or None,
        "raw_text_sha256": text_hash,
        "refusal": refusal,
        "usage": usage,
        "raw_response_sha256": body_hash,
        "raw_row_sha256": canonical_sha256(row),
    }
    if refusal is not None:
        return CandidateOpenAIBatchResult(
            **common,
            validation_status="refusal",
            stop_reason="refusal",
            error_type="model_refusal",
            error_message=refusal[:2000],
        )
    incomplete_details = body.get("incomplete_details")
    if response_status == "incomplete":
        reason = (
            incomplete_details.get("reason")
            if isinstance(incomplete_details, Mapping)
            else None
        )
        return CandidateOpenAIBatchResult(
            **common,
            validation_status="incomplete",
            stop_reason=str(reason or "incomplete"),
            error_type="incomplete_response",
            error_message=(
                json.dumps(incomplete_details, sort_keys=True)[:2000]
                if incomplete_details is not None
                else None
            ),
        )
    if response_status != "completed":
        return CandidateOpenAIBatchResult(
            **common,
            validation_status="provider_error",
            stop_reason=response_status or "missing",
            error_type=f"response_status:{response_status or 'missing'}",
            error_message=json.dumps(body.get("error"), sort_keys=True)[:2000],
        )
    if not message_statuses or any(
        status != "completed" for status in message_statuses
    ):
        return CandidateOpenAIBatchResult(
            **common,
            validation_status="incomplete",
            stop_reason="message_status",
            error_type="incomplete_message",
            error_message=json.dumps(message_statuses),
        )
    if not text:
        return CandidateOpenAIBatchResult(
            **common,
            validation_status="empty",
            stop_reason=response_status,
            error_type="empty_response",
        )
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return CandidateOpenAIBatchResult(
            **common,
            validation_status="invalid_json",
            stop_reason=response_status,
            error_type="invalid_json",
            error_message=str(error)[:2000],
        )
    if not isinstance(value, dict):
        return CandidateOpenAIBatchResult(
            **common,
            validation_status="schema_invalid",
            stop_reason=response_status,
            error_type="schema_invalid",
            error_message="structured output is not a JSON object",
        )
    schema_error = _candidate_schema_error(value, request.expected_output_json_schema)
    if schema_error is not None:
        return CandidateOpenAIBatchResult(
            **common,
            validation_status="schema_invalid",
            parsed_output=dict(value),
            stop_reason=response_status,
            error_type="schema_invalid",
            error_message=schema_error,
        )
    return CandidateOpenAIBatchResult(
        **common,
        validation_status="success",
        parsed_output=dict(value),
        stop_reason=response_status,
    )
