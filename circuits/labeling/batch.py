"""Native Batch API request formatting, submission, status, and collection."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from circuits.labeling.api import (
    anthropic_usage,
    openai_stop_reason,
    openai_usage,
    parse_json_output,
)
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
        atomic_write_jsonl(
            destination, (openai_batch_line(request) for request in values)
        )
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
    metadata = {"run_id": run_id, "stage": stage}
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata,
    )
    return {
        "schema_version": "adag.labeling.provider-batch.v1",
        "provider": "openai",
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "endpoint": "/v1/responses",
        "completion_window": "24h",
        "metadata": dict(getattr(batch, "metadata", None) or metadata),
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
        raise ValueError(
            f"required API key environment variable is missing: {key_name}"
        )
    if provider == "openai":
        from openai import OpenAI

        batch = OpenAI(api_key=api_key).batches.retrieve(batch_id)
        return {
            "schema_version": "adag.labeling.provider-batch.v1",
            "provider": provider,
            "batch_id": batch.id,
            "input_file_id": getattr(batch, "input_file_id", None),
            "endpoint": getattr(batch, "endpoint", None),
            "completion_window": getattr(batch, "completion_window", None),
            "metadata": dict(getattr(batch, "metadata", None) or {}),
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
) -> tuple[dict[str, GenerationResult], dict[str, dict[str, Any]]]:
    _, raw_files = download_openai_batch_files(batch_id, key_env=key_env)
    results: dict[str, GenerationResult] = {}
    for source in ("output", "error"):
        item = raw_files.get(source)
        if item is None:
            continue
        file_id = item["file_id"]
        content = item["content"]
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"OpenAI batch {source} file is not valid UTF-8: {file_id!r}"
            ) from error
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"OpenAI batch {source} file has invalid JSON on line {line_number}"
                ) from error
            request_id = row.get("custom_id")
            if request_id not in requests:
                raise ValueError(
                    f"OpenAI batch {source} file contains unknown custom_id: "
                    f"{request_id!r}"
                )
            if request_id in results:
                raise ValueError(
                    "OpenAI batch files repeat custom_id across their union: "
                    f"{request_id!r}"
                )
            request = requests[request_id]
            results[request_id] = (
                parse_openai_batch_error_row(row, request)
                if source == "error"
                else parse_openai_batch_row(row, request)
            )
    missing = sorted(set(requests) - set(results))
    if missing:
        raise ValueError(
            "OpenAI batch output/error file union omitted request results: "
            + ", ".join(missing)
        )
    return results, raw_files


def download_openai_batch_files(
    batch_id: str, *, key_env: str = "OPENAI_API_KEY"
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Retrieve one completed batch and return its exact opaque result files."""

    from openai import OpenAI

    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"required API key environment variable is missing: {key_env}")
    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)
    observed_batch_id = getattr(batch, "id", batch_id)
    if observed_batch_id != batch_id:
        raise ValueError("OpenAI batch retrieval returned a different batch id")
    if batch.status != "completed" or not (batch.output_file_id or batch.error_file_id):
        raise ValueError(f"OpenAI batch is not collectable: status={batch.status!r}")
    request_counts = getattr(batch, "request_counts", None)
    snapshot = {
        "schema_version": "adag.labeling.provider-batch.v1",
        "provider": "openai",
        "batch_id": observed_batch_id,
        "input_file_id": getattr(batch, "input_file_id", None),
        "endpoint": getattr(batch, "endpoint", None),
        "completion_window": getattr(batch, "completion_window", None),
        "metadata": dict(getattr(batch, "metadata", None) or {}),
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "request_counts": request_counts.model_dump() if request_counts else None,
    }
    raw_files: dict[str, dict[str, Any]] = {}
    for source, file_id in (
        ("output", batch.output_file_id),
        ("error", batch.error_file_id),
    ):
        if file_id:
            raw_files[source] = {
                "file_id": file_id,
                "content": _openai_file_bytes(client.files.content(file_id)),
            }
    return snapshot, raw_files


def download_openai_file_bytes(
    file_id: str, *, key_env: str = "OPENAI_API_KEY"
) -> bytes:
    """Download one exact OpenAI file by immutable provider file id."""

    from openai import OpenAI

    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"required API key environment variable is missing: {key_env}")
    return _openai_file_bytes(OpenAI(api_key=api_key).files.content(file_id))


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
    parsed, status = parse_json_output(
        text, request.stage, request.prompt_template_version
    )
    return GenerationResult(
        request_id=request.request_id,
        provider="openai",
        provider_request_id=response.get("request_id") or body.get("id"),
        model_requested=request.model,
        model_resolved=body.get("model"),
        raw_text=text,
        raw_response_sha256=_sha256(text),
        parsed=parsed,
        parse_status=status,
        usage=openai_usage(body.get("usage")),
        stop_reason=openai_stop_reason(body),
    )


def parse_openai_batch_error_row(
    row: dict[str, Any], request: GenerationRequest
) -> GenerationResult:
    """Represent every error-file row as an explicit failed request result."""

    error = row.get("error")
    response = row.get("response") or {}
    detail = error or response.get("body") or row
    error_type = "batch_request_error"
    if isinstance(error, dict) and error.get("code"):
        error_type = f"batch_request_error:{error['code']}"
    return GenerationResult(
        request_id=request.request_id,
        provider_request_id=response.get("request_id"),
        provider="openai",
        model_requested=request.model,
        parse_status="provider_error",
        error_type=error_type,
        error_message=json.dumps(detail, sort_keys=True)[:2000],
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
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    parsed, status = parse_json_output(
        text, request.stage, request.prompt_template_version
    )
    return GenerationResult(
        request_id=request.request_id,
        provider="anthropic",
        provider_request_id=message.id,
        model_requested=request.model,
        model_resolved=message.model,
        raw_text=text,
        raw_response_sha256=_sha256(text),
        parsed=parsed,
        parse_status=status,
        usage=anthropic_usage(message.usage),
        stop_reason=message.stop_reason,
    )


def _openai_response_text(body: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        texts.extend(
            str(content.get("text", ""))
            for content in item.get("content", [])
            if content.get("type") in ("output_text", "text")
        )
    return "".join(texts)


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _openai_file_bytes(value: Any) -> bytes:
    content = value.content
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    raise TypeError("OpenAI file content response did not contain bytes")
