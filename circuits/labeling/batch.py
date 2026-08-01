"""Native Batch API request formatting, submission, status, and collection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from circuits.labeling.api import anthropic_usage, openai_usage, parse_json_output
from circuits.labeling.config import ModelRoleConfig
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl
from circuits.labeling.schema import GenerationRequest, GenerationResult


def openai_batch_line(request: GenerationRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "input": [message.model_dump() for message in request.messages],
        "max_output_tokens": request.max_output_tokens,
        **request.provider_parameters,
    }
    if request.temperature is not None and not request.reasoning:
        body["temperature"] = request.temperature
    if request.reasoning:
        body["reasoning"] = request.reasoning
    return {
        "custom_id": request.request_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def anthropic_batch_request(request: GenerationRequest) -> dict[str, Any]:
    system = "\n\n".join(
        message.content for message in request.messages if message.role == "system"
    )
    params: dict[str, Any] = {
        "model": request.model,
        "system": system,
        "messages": [
            message.model_dump()
            for message in request.messages
            if message.role != "system"
        ],
        "max_tokens": request.max_output_tokens,
        **request.provider_parameters,
    }
    if request.temperature is not None and not request.reasoning:
        params["temperature"] = request.temperature
    if request.reasoning:
        params["thinking"] = request.reasoning
    return {"custom_id": request.request_id, "params": params}


def prepare_batch_input(
    requests: Iterable[GenerationRequest], destination: Path, provider: str
) -> None:
    values = list(requests)
    if not values:
        raise ValueError("batch input cannot be empty")
    if len({request.model for request in values}) != 1:
        raise ValueError("one native batch input may target only one model")
    if provider == "openai":
        atomic_write_jsonl(destination, (openai_batch_line(request) for request in values))
    elif provider == "anthropic":
        atomic_write_json(
            destination,
            {
                "schema_version": "adag.labeling.anthropic-batch-input.v1",
                "requests": [anthropic_batch_request(request) for request in values],
            },
        )
    else:
        raise ValueError(f"native batch is unsupported for provider {provider!r}")


def submit_openai_batch(
    input_path: Path, *, run_id: str, stage: str, key_env: str = "OPENAI_API_KEY"
) -> dict[str, Any]:
    from openai import OpenAI

    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"required API key environment variable is missing: {key_env}")
    client = OpenAI(api_key=api_key)
    with input_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"run_id": run_id, "stage": stage},
    )
    return {
        "schema_version": "adag.labeling.provider-batch.v1",
        "provider": "openai",
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


def submit_anthropic_batch(
    input_path: Path, *, key_env: str = "ANTHROPIC_API_KEY"
) -> dict[str, Any]:
    from anthropic import Anthropic

    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"required API key environment variable is missing: {key_env}")
    value = json.loads(input_path.read_text(encoding="utf-8"))
    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.create(requests=value["requests"])
    return {
        "schema_version": "adag.labeling.provider-batch.v1",
        "provider": "anthropic",
        "batch_id": batch.id,
        "status": batch.processing_status,
    }


def retrieve_batch(
    provider: str, batch_id: str, config: ModelRoleConfig
) -> dict[str, Any]:
    key_name = config.api_key_env or (
        "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    )
    api_key = os.environ.get(key_name)
    if not api_key:
        raise ValueError(f"required API key environment variable is missing: {key_name}")
    if provider == "openai":
        from openai import OpenAI

        batch = OpenAI(api_key=api_key).batches.retrieve(batch_id)
        return {
            "schema_version": "adag.labeling.provider-batch.v1",
            "provider": provider,
            "batch_id": batch.id,
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "request_counts": (
                batch.request_counts.model_dump() if batch.request_counts else None
            ),
        }
    from anthropic import Anthropic

    batch = Anthropic(api_key=api_key).messages.batches.retrieve(batch_id)
    return {
        "schema_version": "adag.labeling.provider-batch.v1",
        "provider": provider,
        "batch_id": batch.id,
        "status": batch.processing_status,
        "request_counts": batch.request_counts.model_dump(),
        "ended_at": str(batch.ended_at) if batch.ended_at else None,
    }


def collect_openai_batch(
    batch_id: str,
    requests: dict[str, GenerationRequest],
    *,
    key_env: str = "OPENAI_API_KEY",
) -> tuple[dict[str, GenerationResult], str]:
    from openai import OpenAI

    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"required API key environment variable is missing: {key_env}")
    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed" or not batch.output_file_id:
        raise ValueError(f"OpenAI batch is not collectable: status={batch.status!r}")
    content = client.files.content(batch.output_file_id).text
    results: dict[str, GenerationResult] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = row.get("custom_id")
        if request_id not in requests:
            raise ValueError(f"batch output contains unknown custom_id: {request_id!r}")
        if request_id in results:
            raise ValueError(f"batch output repeats custom_id: {request_id!r}")
        results[request_id] = parse_openai_batch_row(row, requests[request_id])
    return results, content


def collect_anthropic_batch(
    batch_id: str,
    requests: dict[str, GenerationRequest],
    *,
    key_env: str = "ANTHROPIC_API_KEY",
) -> dict[str, GenerationResult]:
    from anthropic import Anthropic

    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"required API key environment variable is missing: {key_env}")
    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        raise ValueError(
            f"Anthropic batch is not collectable: status={batch.processing_status!r}"
        )
    results: dict[str, GenerationResult] = {}
    for item in client.messages.batches.results(batch_id):
        request_id = item.custom_id
        if request_id not in requests:
            raise ValueError(f"batch output contains unknown custom_id: {request_id!r}")
        if request_id in results:
            raise ValueError(f"batch output repeats custom_id: {request_id!r}")
        results[request_id] = parse_anthropic_batch_result(item, requests[request_id])
    return results


def parse_openai_batch_row(
    row: dict[str, Any], request: GenerationRequest
) -> GenerationResult:
    error = row.get("error")
    response = row.get("response")
    if error or not response or int(response.get("status_code", 0)) != 200:
        detail = error or (response or {}).get("body")
        return GenerationResult(
            request_id=request.request_id,
            provider="openai",
            model_requested=request.model,
            parse_status="provider_error",
            error_type="batch_request_error",
            error_message=json.dumps(detail, sort_keys=True)[:2000],
        )
    body = response["body"]
    text = _openai_response_text(body)
    parsed, status = parse_json_output(text, request.stage)
    return GenerationResult(
        request_id=request.request_id,
        provider="openai",
        provider_request_id=response.get("request_id") or body.get("id"),
        model_requested=request.model,
        model_resolved=body.get("model"),
        raw_text=text,
        raw_response_sha256=_sha256(text),
        parsed=parsed,
        parse_status=status,  # type: ignore[arg-type]
        usage=openai_usage(body.get("usage")),
        stop_reason=body.get("status"),
    )


def parse_anthropic_batch_result(
    item: Any, request: GenerationRequest
) -> GenerationResult:
    result = item.result
    if result.type != "succeeded":
        return GenerationResult(
            request_id=request.request_id,
            provider="anthropic",
            model_requested=request.model,
            parse_status="provider_error",
            error_type=f"batch_{result.type}",
            error_message=str(result)[:2000],
        )
    message = result.message
    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    parsed, status = parse_json_output(text, request.stage)
    return GenerationResult(
        request_id=request.request_id,
        provider="anthropic",
        provider_request_id=message.id,
        model_requested=request.model,
        model_resolved=message.model,
        raw_text=text,
        raw_response_sha256=_sha256(text),
        parsed=parsed,
        parse_status=status,  # type: ignore[arg-type]
        usage=anthropic_usage(message.usage),
        stop_reason=message.stop_reason,
    )


def _openai_response_text(body: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text"):
                texts.append(str(content.get("text", "")))
    return "".join(texts)


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
