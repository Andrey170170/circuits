"""Receipt-bound OpenAI Batch transport for graph occurrence-role labels."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.graph_labeling.runtime import (
    _load_label_set,
    _load_run,
    _method,
    _packets_by_occurrence,
    _requests_for_method,
    ingest_results,
    label_set_identity,
    normalize_structured_label,
)
from circuits.graph_labeling.schema import ExternalResultRow, MethodSpec, PromptRequest
from circuits.labeling.api import openai_usage
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)
from circuits.labeling.pricing import load_price_snapshot
from circuits.labeling.schema import Usage

ENDPOINT = "/v1/responses"
COMPLETION_WINDOW = "24h"
PRICE_SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "scripts/bonafide/configs/labeling/prices-2026-08-25-graph-labeling.json"
)


class OpenAIBatchTransport(Protocol):
    def upload_batch_input(self, path: Path) -> Mapping[str, Any]: ...

    def retrieve_file(self, file_id: str) -> Mapping[str, Any]: ...

    def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str],
    ) -> Mapping[str, Any]: ...

    def retrieve_batch(self, batch_id: str) -> Mapping[str, Any]: ...

    def download_file(self, file_id: str) -> bytes: ...


def _provider_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    raise ValueError("OpenAI provider response is not an object")


class SDKOpenAIBatchTransport:
    """Thin official-SDK adapter; the API key is read only at construction."""

    def __init__(self, *, key_env: str = "OPENAI_API_KEY") -> None:
        api_key = os.environ.get(key_env)
        if not api_key:
            raise ValueError(
                f"required API key environment variable is missing: {key_env}"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, max_retries=0)

    def upload_batch_input(self, path: Path) -> Mapping[str, Any]:
        with path.open("rb") as handle:
            return _provider_dict(
                self._client.files.create(file=handle, purpose="batch")
            )

    def retrieve_file(self, file_id: str) -> Mapping[str, Any]:
        return _provider_dict(self._client.files.retrieve(file_id))

    def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str],
    ) -> Mapping[str, Any]:
        return _provider_dict(
            self._client.batches.create(
                input_file_id=input_file_id,
                endpoint=endpoint,
                completion_window=completion_window,
                metadata=metadata,
            )
        )

    def retrieve_batch(self, batch_id: str) -> Mapping[str, Any]:
        return _provider_dict(self._client.batches.retrieve(batch_id))

    def download_file(self, file_id: str) -> bytes:
        value = self._client.files.content(file_id)
        if hasattr(value, "read"):
            content = value.read()
        elif hasattr(value, "content"):
            content = value.content
        else:
            content = value
        if not isinstance(content, bytes):
            raise ValueError("OpenAI file content is not bytes")
        return content


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _self_hashed(value: dict[str, Any], field: str = "content_hash") -> dict[str, Any]:
    return {**value, field: canonical_sha256(value)}


def _load_hashed(path: Path, field: str = "content_hash") -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable OpenAI Batch receipt: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"OpenAI Batch receipt is not an object: {path}")
    core = dict(value)
    recorded = core.pop(field, None)
    if recorded != canonical_sha256(core):
        raise ValueError(f"OpenAI Batch receipt hash drift: {path}")
    return value


def _write_or_verify_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _load_hashed(path) != value:
            raise ValueError(f"persisted OpenAI Batch receipt drift: {path}")
        return
    atomic_write_json(path, value)


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        for row in rows
    )


def _attempt_id() -> str:
    return f"attempt-{uuid.uuid4().hex}"


def _strict_cost(
    rates: Mapping[str, Any], usage: Usage, *, context: str
) -> dict[str, Any]:
    """Conservatively price all inclusive input at the dearer input/write rate."""

    if usage.input_tokens is None or usage.output_tokens is None:
        raise ValueError(f"OpenAI Batch usage is incomplete: {context}")
    input_rate = float(rates["input_per_million"])
    cache_write_rate = float(rates.get("cache_write_per_million", input_rate))
    strict_input_rate = max(input_rate, cache_write_rate)
    output_rate = float(rates["output_per_million"])
    input_cost = usage.input_tokens * strict_input_rate / 1_000_000
    output_cost = usage.output_tokens * output_rate / 1_000_000
    return {
        "pricing_policy": "max_input_or_cache_write_rate_plus_output_v1",
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "strict_input_rate_per_million": strict_input_rate,
        "output_rate_per_million": output_rate,
        "strict_input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }


def _output_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "status",
            "label",
            "reads_from",
            "cited_evidence_ids",
            "claim_citations",
            "apparent_role",
            "target_effect",
            "rationale",
            "alternative_hypothesis",
            "limitations",
            "confidence",
        ],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["provisional_label", "insufficient_evidence"],
            },
            "label": nullable_string,
            "reads_from": {"type": "array", "items": {"type": "string"}},
            "cited_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "claim_citations": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "label",
                    "reads_from",
                    "apparent_role",
                    "target_effect",
                    "rationale",
                    "alternative_hypothesis",
                ],
                "properties": {
                    key: {"type": "array", "items": {"type": "string"}}
                    for key in (
                        "label",
                        "reads_from",
                        "apparent_role",
                        "target_effect",
                        "rationale",
                        "alternative_hypothesis",
                    )
                },
            },
            "apparent_role": nullable_string,
            "target_effect": {
                "type": "string",
                "enum": ["supports", "suppresses", "mixed", "unclear"],
            },
            "rationale": nullable_string,
            "alternative_hypothesis": nullable_string,
            "limitations": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        },
    }


def _batch_line(request: PromptRequest) -> dict[str, Any]:
    generation = request.generation
    if generation.provider != "openai":
        raise ValueError("OpenAI Batch requires an OpenAI labeling method")
    reserved = {
        "model",
        "input",
        "max_output_tokens",
        "reasoning",
        "temperature",
        "store",
        "text",
    }
    overlap = sorted(reserved & set(generation.provider_parameters))
    if overlap:
        raise ValueError(f"provider parameters override Batch-owned fields: {overlap}")
    body: dict[str, Any] = {
        "model": generation.model,
        "input": request.messages,
        "max_output_tokens": generation.max_output_tokens,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "graph_occurrence_role_v1",
                "strict": True,
                "schema": _output_schema(),
            }
        },
        **generation.provider_parameters,
    }
    if generation.reasoning:
        body["reasoning"] = generation.reasoning
    elif generation.temperature is not None:
        body["temperature"] = generation.temperature
    return {
        "custom_id": request.request_id,
        "method": "POST",
        "url": ENDPOINT,
        "body": body,
    }


def _context(
    run_root: Path, method_id: str
) -> tuple[Path, dict[str, Any], MethodSpec, list[PromptRequest], str]:
    root = run_root.expanduser().resolve()
    manifest = _load_run(root)
    method = _method(manifest["spec"], method_id)
    if method.kind != "structured_llm_graph_role_v1" or method.labeler is None:
        raise ValueError("OpenAI Batch supports only structured graph-role methods")
    if method.labeler.provider != "openai":
        raise ValueError("selected graph-role method is not configured for OpenAI")
    requests = sorted(
        _requests_for_method(root, manifest, method).values(),
        key=lambda item: item.request_id,
    )
    if not requests:
        raise ValueError("selected method has no materialized requests")
    label_set_id = label_set_identity(manifest["study_sha256"], method.identity_sha256)
    return root, manifest, method, requests, label_set_id


def _batch_root(root: Path, label_set_id: str) -> Path:
    return root / "openai-batches" / label_set_id


def prepare_openai_batch(
    run_root: Path,
    method_id: str,
    *,
    max_cost_usd: float,
    price_snapshot: Path = PRICE_SNAPSHOT,
) -> dict[str, Any]:
    """Materialize and cost-guard one method-level Batch input without API calls."""

    if max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive")
    root, manifest, method, requests, label_set_id = _context(run_root, method_id)
    rows = [_batch_line(request) for request in requests]
    data = _jsonl_bytes(rows)
    prices = load_price_snapshot(price_snapshot)
    model = method.labeler.model  # type: ignore[union-attr]
    try:
        rates = prices["rates"]["openai"][model]["native_batch"]
    except KeyError as error:
        raise ValueError(
            f"price snapshot lacks OpenAI native_batch rate for {model}"
        ) from error
    input_proxy = sum(
        len(json.dumps(row["body"], ensure_ascii=False).encode()) + 256 for row in rows
    )
    output_ceiling = sum(request.generation.max_output_tokens for request in requests)
    strict_input_rate = max(
        float(rates["input_per_million"]),
        float(rates.get("cache_write_per_million", rates["input_per_million"])),
    )
    projected = (
        input_proxy * strict_input_rate
        + output_ceiling * float(rates["output_per_million"])
    ) / 1_000_000
    if projected > max_cost_usd + 1e-12:
        raise ValueError(
            f"cost guard ${max_cost_usd:.6f} is below projected Batch ceiling ${projected:.6f}"
        )
    batch_root = _batch_root(root, label_set_id)
    input_path = batch_root / "input.jsonl"
    plan_path = batch_root / "plan.json"
    if input_path.exists() or plan_path.exists():
        if not input_path.is_file() or not plan_path.is_file():
            raise ValueError("partial persisted OpenAI Batch plan")
        observed = _load_hashed(plan_path)
        created_at = observed.get("created_at")
    else:
        created_at = _now()
    batch_identity = canonical_sha256(
        {
            "study_sha256": manifest["study_sha256"],
            "method_sha256": method.identity_sha256,
            "request_bindings": [
                {
                    "request_id": request.request_id,
                    "logical_request_sha256": request.logical_request_sha256,
                    "evidence_sha256": request.evidence_sha256,
                }
                for request in requests
            ],
            "input_sha256": canonical_sha256(rows),
            "endpoint": ENDPOINT,
            "completion_window": COMPLETION_WINDOW,
        }
    )
    core = {
        "schema_version": "adag.graph-labeling.openai-batch-plan.v1",
        "created_at": created_at,
        "batch_identity_sha256": batch_identity,
        "run_manifest_sha256": manifest["content_hash"],
        "study_sha256": manifest["study_sha256"],
        "method_id": method.method_id,
        "method_sha256": method.identity_sha256,
        "label_set_id": label_set_id,
        "model": model,
        "endpoint": ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
        "request_count": len(requests),
        "request_bindings": [
            {
                "request_id": request.request_id,
                "logical_request_sha256": request.logical_request_sha256,
                "evidence_sha256": request.evidence_sha256,
            }
            for request in requests
        ],
        "input_file": "input.jsonl",
        "input_file_sha256": hashlib.sha256(data).hexdigest(),
        "input_bytes": len(data),
        "input_token_proxy_total": input_proxy,
        "max_output_tokens_total": output_ceiling,
        "planning_method": (
            "utf8_response_body_bytes_plus_256_input_proxy_charged_at_"
            "max_uncached_or_cache_write_rate_plus_full_output_ceiling"
        ),
        "price_snapshot_id": prices["snapshot_id"],
        "price_snapshot_file_sha256": file_sha256(price_snapshot),
        "projected_cost_ceiling_usd": projected,
        "caller_max_cost_usd": max_cost_usd,
    }
    plan = _self_hashed(core)
    if input_path.exists():
        if input_path.read_bytes() != data or _load_hashed(plan_path) != plan:
            raise ValueError("persisted OpenAI Batch plan or input drift")
    else:
        batch_root.mkdir(parents=True, exist_ok=False)
        atomic_write_bytes(input_path, data)
        atomic_write_json(plan_path, plan)
    return plan


def _metadata(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        "graph_label_set": str(plan["label_set_id"]),
        "batch_identity": str(plan["batch_identity_sha256"])[:32],
    }


def _validate_remote(
    snapshot: Mapping[str, Any], submission: Mapping[str, Any]
) -> None:
    if (
        snapshot.get("id") != submission.get("batch_id")
        or snapshot.get("input_file_id") != submission.get("input_file_id")
        or snapshot.get("endpoint") != ENDPOINT
        or snapshot.get("completion_window") != COMPLETION_WINDOW
        or dict(snapshot.get("metadata") or {}) != submission.get("metadata")
    ):
        raise ValueError("OpenAI Batch remote identity drift")


def _load_bound_plan(
    batch_root: Path,
    manifest: Mapping[str, Any],
    method: MethodSpec,
    requests: list[PromptRequest],
    label_set_id: str,
) -> dict[str, Any]:
    plan = _load_hashed(batch_root / "plan.json")
    input_path = batch_root / "input.jsonl"
    expected_bindings = [
        {
            "request_id": request.request_id,
            "logical_request_sha256": request.logical_request_sha256,
            "evidence_sha256": request.evidence_sha256,
        }
        for request in requests
    ]
    expected_input = _jsonl_bytes([_batch_line(request) for request in requests])
    if (
        plan.get("run_manifest_sha256") != manifest["content_hash"]
        or plan.get("study_sha256") != manifest["study_sha256"]
        or plan.get("method_id") != method.method_id
        or plan.get("method_sha256") != method.identity_sha256
        or plan.get("label_set_id") != label_set_id
        or plan.get("request_bindings") != expected_bindings
        or plan.get("endpoint") != ENDPOINT
        or plan.get("completion_window") != COMPLETION_WINDOW
        or not input_path.is_file()
        or input_path.read_bytes() != expected_input
        or plan.get("input_file_sha256") != hashlib.sha256(expected_input).hexdigest()
    ):
        raise ValueError("OpenAI Batch plan or frozen request binding drift")
    return plan


def _load_bound_upload(
    batch_root: Path, plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    upload_intent = _load_hashed(batch_root / "upload-intent.json")
    upload = _load_hashed(batch_root / "upload.json")
    attempt_id = upload_intent.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or not attempt_id.startswith("attempt-")
        or upload_intent.get("batch_identity_sha256") != plan["batch_identity_sha256"]
        or upload_intent.get("plan_sha256") != plan["content_hash"]
        or upload_intent.get("input_file_sha256") != plan["input_file_sha256"]
        or upload_intent.get("purpose") != "batch"
        or upload.get("attempt_id") != attempt_id
        or upload.get("upload_intent_sha256") != upload_intent["content_hash"]
        or upload.get("batch_identity_sha256") != plan["batch_identity_sha256"]
        or upload.get("input_file_sha256") != plan["input_file_sha256"]
        or upload.get("purpose") != "batch"
        or not isinstance(upload.get("input_file_id"), str)
        or not upload["input_file_id"]
    ):
        raise ValueError("OpenAI Batch local upload binding drift")
    return upload_intent, upload


def _load_bound_submission(batch_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    submission = _load_hashed(batch_root / "submission.json")
    upload_intent, upload = _load_bound_upload(batch_root, plan)
    create_intent = _load_hashed(batch_root / "create-intent.json")
    attempt_id = upload_intent.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or not attempt_id.startswith("attempt-")
        or upload_intent.get("schema_version")
        != "adag.graph-labeling.openai-batch-upload-intent.v1"
        or submission.get("batch_identity_sha256") != plan["batch_identity_sha256"]
        or submission.get("plan_sha256") != plan["content_hash"]
        or submission.get("upload_intent_sha256") != upload_intent["content_hash"]
        or submission.get("create_intent_sha256") != create_intent["content_hash"]
        or submission.get("study_sha256") != plan["study_sha256"]
        or submission.get("method_sha256") != plan["method_sha256"]
        or submission.get("request_bindings") != plan["request_bindings"]
        or submission.get("input_file_sha256") != plan["input_file_sha256"]
        or submission.get("endpoint") != ENDPOINT
        or submission.get("completion_window") != COMPLETION_WINDOW
        or submission.get("upload_receipt_sha256") != upload["content_hash"]
        or submission.get("input_file_id") != upload.get("input_file_id")
        or upload.get("batch_identity_sha256") != plan["batch_identity_sha256"]
        or upload.get("input_file_sha256") != plan["input_file_sha256"]
        or upload.get("purpose") != "batch"
        or upload.get("attempt_id") != attempt_id
        or upload.get("upload_intent_sha256") != upload_intent["content_hash"]
        or upload_intent.get("batch_identity_sha256") != plan["batch_identity_sha256"]
        or upload_intent.get("plan_sha256") != plan["content_hash"]
        or upload_intent.get("input_file_sha256") != plan["input_file_sha256"]
        or upload_intent.get("purpose") != "batch"
        or create_intent.get("batch_identity_sha256") != plan["batch_identity_sha256"]
        or create_intent.get("plan_sha256") != plan["content_hash"]
        or create_intent.get("upload_receipt_sha256") != upload["content_hash"]
        or create_intent.get("input_file_id") != upload.get("input_file_id")
        or create_intent.get("endpoint") != ENDPOINT
        or create_intent.get("completion_window") != COMPLETION_WINDOW
        or create_intent.get("metadata") != _metadata(plan)
        or submission.get("attempt_id") != attempt_id
        or create_intent.get("attempt_id") != attempt_id
        or submission.get("receipt_mode") not in {"direct", "recovered"}
    ):
        raise ValueError("OpenAI Batch local submission binding drift")
    return submission


def _status_chain(
    batch_root: Path, submission: Mapping[str, Any]
) -> list[dict[str, Any]]:
    status_root = batch_root / "status"
    if not status_root.exists():
        return []
    values: list[dict[str, Any]] = []
    previous: str | None = None
    for index, path in enumerate(sorted(status_root.iterdir())):
        if path.name != f"receipt-{index:04d}.json":
            raise ValueError("OpenAI Batch status receipt sequence drift")
        value = _load_hashed(path)
        if (
            value.get("submission_sha256") != submission["content_hash"]
            or value.get("batch_identity_sha256") != submission["batch_identity_sha256"]
            or value.get("batch_id") != submission["batch_id"]
            or value.get("previous_status_sha256") != previous
        ):
            raise ValueError("OpenAI Batch status receipt chain drift")
        values.append(value)
        previous = value["content_hash"]
    return values


def _submit_openai_batch_unlocked(
    run_root: Path,
    method_id: str,
    *,
    max_cost_usd: float,
    transport: OpenAIBatchTransport | None = None,
) -> dict[str, Any]:
    """Upload and submit exactly once; an intent-only state fails closed."""

    plan = prepare_openai_batch(run_root, method_id, max_cost_usd=max_cost_usd)
    root = Path(run_root).expanduser().resolve()
    batch_root = _batch_root(root, str(plan["label_set_id"]))
    receipt_path = batch_root / "submission.json"
    upload_intent_path = batch_root / "upload-intent.json"
    upload_path = batch_root / "upload.json"
    create_intent_path = batch_root / "create-intent.json"
    if (batch_root / "abandoned.json").exists():
        raise RuntimeError("OpenAI Batch attempt was explicitly abandoned")
    if receipt_path.exists():
        return _load_bound_submission(batch_root, plan)
    live = transport or SDKOpenAIBatchTransport()
    if upload_path.exists():
        upload_intent, upload_receipt = _load_bound_upload(batch_root, plan)
        attempt_id = str(upload_intent["attempt_id"])
    else:
        if upload_intent_path.exists():
            upload_intent = _load_hashed(upload_intent_path)
            if (
                upload_intent.get("batch_identity_sha256")
                != plan["batch_identity_sha256"]
                or upload_intent.get("plan_sha256") != plan["content_hash"]
                or upload_intent.get("input_file_sha256") != plan["input_file_sha256"]
            ):
                raise ValueError("OpenAI upload intent drift")
            raise RuntimeError(
                "OpenAI upload is indeterminate; recover it by input file id or "
                "explicitly abandon the attempt"
            )
        attempt_id = _attempt_id()
        upload_intent = _self_hashed(
            {
                "schema_version": "adag.graph-labeling.openai-batch-upload-intent.v1",
                "created_at": _now(),
                "attempt_id": attempt_id,
                "batch_identity_sha256": plan["batch_identity_sha256"],
                "plan_sha256": plan["content_hash"],
                "input_file_sha256": plan["input_file_sha256"],
                "purpose": "batch",
            }
        )
        atomic_write_json(upload_intent_path, upload_intent)
        uploaded = dict(live.upload_batch_input(batch_root / "input.jsonl"))
        input_file_id = uploaded.get("id")
        if not isinstance(input_file_id, str) or not input_file_id:
            raise ValueError("OpenAI upload did not return an input file id")
        upload_receipt = _self_hashed(
            {
                "schema_version": "adag.graph-labeling.openai-batch-upload.v1",
                "created_at": _now(),
                "attempt_id": attempt_id,
                "upload_intent_sha256": upload_intent["content_hash"],
                "batch_identity_sha256": plan["batch_identity_sha256"],
                "input_file_sha256": plan["input_file_sha256"],
                "input_file_id": input_file_id,
                "purpose": uploaded.get("purpose"),
                "remote": uploaded,
            }
        )
        if upload_receipt["purpose"] != "batch":
            raise ValueError("OpenAI upload purpose drift")
        atomic_write_json(upload_path, upload_receipt)
    input_file_id = upload_receipt.get("input_file_id")
    if not isinstance(input_file_id, str) or not input_file_id:
        raise ValueError("OpenAI upload receipt lacks an input file id")
    if create_intent_path.exists():
        create_intent = _load_hashed(create_intent_path)
        if (
            create_intent.get("attempt_id") != attempt_id
            or create_intent.get("batch_identity_sha256")
            != plan["batch_identity_sha256"]
            or create_intent.get("plan_sha256") != plan["content_hash"]
            or create_intent.get("upload_receipt_sha256")
            != upload_receipt["content_hash"]
        ):
            raise ValueError("OpenAI Batch create intent drift")
        raise RuntimeError(
            "OpenAI Batch creation is indeterminate; recover or explicitly abandon the attempt"
        )
    metadata = _metadata(plan)
    create_intent = _self_hashed(
        {
            "schema_version": "adag.graph-labeling.openai-batch-create-intent.v1",
            "created_at": _now(),
            "attempt_id": attempt_id,
            "batch_identity_sha256": plan["batch_identity_sha256"],
            "plan_sha256": plan["content_hash"],
            "upload_receipt_sha256": upload_receipt["content_hash"],
            "input_file_id": input_file_id,
            "endpoint": ENDPOINT,
            "completion_window": COMPLETION_WINDOW,
            "metadata": metadata,
        }
    )
    atomic_write_json(create_intent_path, create_intent)
    remote = dict(
        live.create_batch(
            input_file_id=input_file_id,
            endpoint=ENDPOINT,
            completion_window=COMPLETION_WINDOW,
            metadata=metadata,
        )
    )
    batch_id = remote.get("id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("OpenAI Batch creation did not return a batch id")
    receipt = _self_hashed(
        {
            "schema_version": "adag.graph-labeling.openai-batch-submission.v1",
            "created_at": _now(),
            "attempt_id": attempt_id,
            "batch_identity_sha256": plan["batch_identity_sha256"],
            "plan_sha256": plan["content_hash"],
            "upload_intent_sha256": upload_receipt["upload_intent_sha256"],
            "create_intent_sha256": create_intent["content_hash"],
            "study_sha256": plan["study_sha256"],
            "method_sha256": plan["method_sha256"],
            "request_bindings": plan["request_bindings"],
            "input_file_sha256": plan["input_file_sha256"],
            "input_file_id": input_file_id,
            "upload_receipt_sha256": upload_receipt["content_hash"],
            "batch_id": batch_id,
            "endpoint": ENDPOINT,
            "completion_window": COMPLETION_WINDOW,
            "metadata": metadata,
            "status_at_submission": remote.get("status"),
            "remote": remote,
            "receipt_mode": "direct",
        }
    )
    _validate_remote(remote, receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


def _recover_openai_batch_unlocked(
    run_root: Path,
    method_id: str,
    batch_id: str,
    *,
    transport: OpenAIBatchTransport | None = None,
) -> dict[str, Any]:
    """Recover an intent-only submission after proving the remote input identity."""

    if not batch_id:
        raise ValueError("OpenAI Batch recovery requires a batch id")
    root, manifest, method, requests, label_set_id = _context(run_root, method_id)
    batch_root = _batch_root(root, label_set_id)
    if (batch_root / "abandoned.json").exists():
        raise RuntimeError("cannot recover an explicitly abandoned attempt")
    plan = _load_bound_plan(batch_root, manifest, method, requests, label_set_id)
    receipt_path = batch_root / "submission.json"
    if receipt_path.exists():
        receipt = _load_bound_submission(batch_root, plan)
        if receipt.get("batch_id") != batch_id:
            raise ValueError("recovery batch id differs from persisted submission")
        return receipt
    upload_intent = _load_hashed(batch_root / "upload-intent.json")
    if (
        upload_intent.get("batch_identity_sha256") != plan["batch_identity_sha256"]
        or upload_intent.get("plan_sha256") != plan["content_hash"]
    ):
        raise ValueError("OpenAI Batch recovery intent drift")
    live = transport or SDKOpenAIBatchTransport()
    remote = dict(live.retrieve_batch(batch_id))
    expected = {
        "batch_id": batch_id,
        "input_file_id": remote.get("input_file_id"),
        "metadata": _metadata(plan),
    }
    _validate_remote(remote, expected)
    input_file_id = remote.get("input_file_id")
    if not isinstance(input_file_id, str) or not input_file_id:
        raise ValueError("recovery Batch lacks an input file id")
    remote_input = live.download_file(input_file_id)
    if remote_input != (batch_root / "input.jsonl").read_bytes():
        raise ValueError("recovery Batch input differs from the frozen local JSONL")
    upload_receipt = _self_hashed(
        {
            "schema_version": "adag.graph-labeling.openai-batch-upload.v1",
            "created_at": _now(),
            "attempt_id": upload_intent["attempt_id"],
            "upload_intent_sha256": upload_intent["content_hash"],
            "batch_identity_sha256": plan["batch_identity_sha256"],
            "input_file_sha256": plan["input_file_sha256"],
            "input_file_id": input_file_id,
            "purpose": "batch",
            "remote": {"recovered_from_batch_id": batch_id},
        }
    )
    upload_path = batch_root / "upload.json"
    if upload_path.exists():
        _existing_intent, existing_upload = _load_bound_upload(batch_root, plan)
        if existing_upload.get("input_file_id") != input_file_id:
            raise ValueError("recovery upload receipt drift")
        upload_receipt = existing_upload
    else:
        atomic_write_json(upload_path, upload_receipt)
    create_intent_path = batch_root / "create-intent.json"
    if create_intent_path.exists():
        create_intent = _load_hashed(create_intent_path)
        if (
            create_intent.get("attempt_id") != upload_intent["attempt_id"]
            or create_intent.get("batch_identity_sha256")
            != plan["batch_identity_sha256"]
            or create_intent.get("plan_sha256") != plan["content_hash"]
            or create_intent.get("upload_receipt_sha256")
            != upload_receipt["content_hash"]
            or create_intent.get("input_file_id") != input_file_id
            or create_intent.get("metadata") != _metadata(plan)
        ):
            raise ValueError("OpenAI Batch recovery create intent drift")
    else:
        create_intent = _self_hashed(
            {
                "schema_version": "adag.graph-labeling.openai-batch-create-intent.v1",
                "created_at": _now(),
                "attempt_id": upload_intent["attempt_id"],
                "batch_identity_sha256": plan["batch_identity_sha256"],
                "plan_sha256": plan["content_hash"],
                "upload_receipt_sha256": upload_receipt["content_hash"],
                "input_file_id": input_file_id,
                "endpoint": ENDPOINT,
                "completion_window": COMPLETION_WINDOW,
                "metadata": _metadata(plan),
                "recovery_materialized": True,
            }
        )
        atomic_write_json(create_intent_path, create_intent)
    receipt = _self_hashed(
        {
            "schema_version": "adag.graph-labeling.openai-batch-submission.v1",
            "created_at": _now(),
            "attempt_id": upload_intent["attempt_id"],
            "batch_identity_sha256": plan["batch_identity_sha256"],
            "plan_sha256": plan["content_hash"],
            "upload_intent_sha256": upload_intent["content_hash"],
            "create_intent_sha256": create_intent["content_hash"],
            "study_sha256": plan["study_sha256"],
            "method_sha256": plan["method_sha256"],
            "request_bindings": plan["request_bindings"],
            "input_file_sha256": plan["input_file_sha256"],
            "input_file_id": input_file_id,
            "upload_receipt_sha256": upload_receipt["content_hash"],
            "batch_id": batch_id,
            "endpoint": ENDPOINT,
            "completion_window": COMPLETION_WINDOW,
            "metadata": _metadata(plan),
            "status_at_submission": remote.get("status"),
            "remote": remote,
            "receipt_mode": "recovered",
        }
    )
    _validate_remote(remote, receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


def _recover_openai_upload_unlocked(
    run_root: Path,
    method_id: str,
    input_file_id: str,
    *,
    transport: OpenAIBatchTransport | None = None,
) -> dict[str, Any]:
    """Recover an upload-intent-only attempt after byte and purpose verification."""

    if not input_file_id:
        raise ValueError("OpenAI upload recovery requires an input file id")
    root, manifest, method, requests, label_set_id = _context(run_root, method_id)
    batch_root = _batch_root(root, label_set_id)
    if (batch_root / "abandoned.json").exists():
        raise RuntimeError("cannot recover an explicitly abandoned attempt")
    plan = _load_bound_plan(batch_root, manifest, method, requests, label_set_id)
    upload_path = batch_root / "upload.json"
    if upload_path.exists():
        _intent, upload = _load_bound_upload(batch_root, plan)
        if upload["input_file_id"] != input_file_id:
            raise ValueError("recovery input file id differs from persisted upload")
        return upload
    upload_intent = _load_hashed(batch_root / "upload-intent.json")
    if (
        upload_intent.get("batch_identity_sha256") != plan["batch_identity_sha256"]
        or upload_intent.get("plan_sha256") != plan["content_hash"]
        or upload_intent.get("input_file_sha256") != plan["input_file_sha256"]
        or upload_intent.get("purpose") != "batch"
    ):
        raise ValueError("OpenAI upload recovery intent drift")
    live = transport or SDKOpenAIBatchTransport()
    metadata = dict(live.retrieve_file(input_file_id))
    if metadata.get("id") != input_file_id or metadata.get("purpose") != "batch":
        raise ValueError("recovery upload identity or purpose drift")
    remote_input = live.download_file(input_file_id)
    if remote_input != (batch_root / "input.jsonl").read_bytes():
        raise ValueError("recovery upload differs from the frozen local JSONL")
    upload = _self_hashed(
        {
            "schema_version": "adag.graph-labeling.openai-batch-upload.v1",
            "created_at": _now(),
            "attempt_id": upload_intent["attempt_id"],
            "upload_intent_sha256": upload_intent["content_hash"],
            "batch_identity_sha256": plan["batch_identity_sha256"],
            "input_file_sha256": plan["input_file_sha256"],
            "input_file_id": input_file_id,
            "purpose": "batch",
            "remote": metadata,
            "receipt_mode": "recovered",
        }
    )
    atomic_write_json(upload_path, upload)
    return upload


def _openai_batch_status_unlocked(
    run_root: Path,
    method_id: str,
    *,
    transport: OpenAIBatchTransport | None = None,
) -> dict[str, Any]:
    root, manifest, method, requests, label_set_id = _context(run_root, method_id)
    batch_root = _batch_root(root, label_set_id)
    plan = _load_bound_plan(batch_root, manifest, method, requests, label_set_id)
    submission = _load_bound_submission(batch_root, plan)
    snapshot = dict(
        (transport or SDKOpenAIBatchTransport()).retrieve_batch(
            str(submission["batch_id"])
        )
    )
    _validate_remote(snapshot, submission)
    status_root = batch_root / "status"
    status_root.mkdir(parents=True, exist_ok=True)
    previous = _status_chain(batch_root, submission)
    previous_sha256 = previous[-1]["content_hash"] if previous else None
    receipt = _self_hashed(
        {
            "schema_version": "adag.graph-labeling.openai-batch-status.v1",
            "observed_at": _now(),
            "submission_sha256": submission["content_hash"],
            "batch_identity_sha256": submission["batch_identity_sha256"],
            "batch_id": submission["batch_id"],
            "previous_status_sha256": previous_sha256,
            "status": snapshot.get("status"),
            "request_counts": snapshot.get("request_counts"),
            "usage": snapshot.get("usage"),
            "output_file_id": snapshot.get("output_file_id"),
            "error_file_id": snapshot.get("error_file_id"),
            "remote": snapshot,
        }
    )
    atomic_write_json(
        status_root
        / f"receipt-{len(list(status_root.glob('receipt-*.json'))):04d}.json",
        receipt,
    )
    return receipt


def _response_text(body: Mapping[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str):
        return output_text
    texts = [
        content["text"]
        for item in body.get("output", [])
        if isinstance(item, Mapping)
        for content in item.get("content", [])
        if isinstance(content, Mapping)
        and content.get("type") == "output_text"
        and isinstance(content.get("text"), str)
    ]
    if len(texts) != 1:
        raise ValueError(
            "Responses Batch row does not contain exactly one structured output text"
        )
    return texts[0]


def _parse_output(
    data: bytes, requests: dict[str, PromptRequest]
) -> tuple[list[ExternalResultRow], dict[str, Any]]:
    try:
        lines = data.decode().splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("OpenAI Batch output is not UTF-8") from error
    rows: dict[str, ExternalResultRow] = {}
    usage: dict[str, Any] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"malformed OpenAI Batch output line {line_number}"
            ) from error
        custom_id = item.get("custom_id") if isinstance(item, dict) else None
        if custom_id not in requests:
            raise ValueError(
                f"OpenAI Batch output contains unknown custom_id: {custom_id!r}"
            )
        if custom_id in rows:
            raise ValueError(f"OpenAI Batch output repeats custom_id: {custom_id!r}")
        response = item.get("response")
        if (
            item.get("error") is not None
            or not isinstance(response, dict)
            or response.get("status_code") != 200
        ):
            raise ValueError(f"OpenAI Batch request failed: {custom_id}")
        body = response.get("body")
        if (
            not isinstance(body, dict)
            or body.get("status") != "completed"
            or body.get("error") is not None
        ):
            raise ValueError(
                f"Responses API result is incomplete or failed: {custom_id}"
            )
        try:
            payload = json.loads(_response_text(body))
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed structured output for {custom_id}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"structured output is not an object for {custom_id}")
        request = requests[custom_id]
        observed_model = body.get("model")
        if not isinstance(observed_model, str) or not (
            observed_model == request.generation.model
            or observed_model.startswith(request.generation.model + "-")
        ):
            raise ValueError(f"Responses API result model drift: {custom_id}")
        parsed_usage = openai_usage(body.get("usage"))
        if parsed_usage.input_tokens is None or parsed_usage.output_tokens is None:
            raise ValueError(f"Responses API result usage is incomplete: {custom_id}")
        raw_usage = body.get("usage")
        raw_total = (
            raw_usage.get("total_tokens") if isinstance(raw_usage, dict) else None
        )
        if raw_total is not None and raw_total != (
            parsed_usage.input_tokens + parsed_usage.output_tokens
        ):
            raise ValueError(f"Responses API total-token usage mismatch: {custom_id}")
        response_id = body.get("id")
        provider_request_id = response.get("request_id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(f"Responses API result lacks response id: {custom_id}")
        if not isinstance(provider_request_id, str) or not provider_request_id:
            raise ValueError(
                f"Responses Batch row lacks provider request id: {custom_id}"
            )
        rows[custom_id] = ExternalResultRow(
            request_id=custom_id,
            logical_request_sha256=request.logical_request_sha256,
            evidence_sha256=request.evidence_sha256,
            method_sha256=request.method_sha256,
            raw_payload=payload,
            raw_response_sha256=canonical_sha256(payload),
        )
        usage[custom_id] = {
            "response_id": response_id,
            "model": body.get("model"),
            "service_tier": body.get("service_tier"),
            "usage": parsed_usage.model_dump(mode="json"),
            "provider_request_id": provider_request_id,
        }
    missing = sorted(set(requests) - set(rows))
    if missing:
        raise ValueError(f"OpenAI Batch output omitted custom_ids: {missing}")
    return [rows[key] for key in sorted(rows)], usage


def _load_collection(
    collection_root: Path,
    *,
    manifest: Mapping[str, Any],
    method: MethodSpec,
    plan: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _load_hashed(collection_root / "receipt.json")
    output_path = collection_root / "provider-output.jsonl"
    results_path = collection_root / "external-results.jsonl"
    if (
        receipt.get("submission_sha256") != submission["content_hash"]
        or receipt.get("run_manifest_sha256") != manifest["content_hash"]
        or receipt.get("study_sha256") != manifest["study_sha256"]
        or receipt.get("method_sha256") != method.identity_sha256
        or receipt.get("batch_id") != submission["batch_id"]
        or receipt.get("input_file_id") != submission["input_file_id"]
        or receipt.get("request_bindings") != submission["request_bindings"]
        or receipt.get("price_snapshot_file_sha256")
        != plan["price_snapshot_file_sha256"]
        or not output_path.is_file()
        or hashlib.sha256(output_path.read_bytes()).hexdigest()
        != receipt.get("output_file_sha256")
        or not results_path.is_file()
        or hashlib.sha256(results_path.read_bytes()).hexdigest()
        != receipt.get("external_results_sha256")
    ):
        raise ValueError("canonical OpenAI Batch collection binding drift")
    return receipt


def _finalize_collection(
    *,
    root: Path,
    manifest: dict[str, Any],
    method: MethodSpec,
    method_id: str,
    collection_root: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    finalization_path = collection_root / "finalization.json"
    result_source = {
        "kind": "openai_batch_collection_v1",
        "provider": "openai",
        "configured_model": receipt["configured_model"],
        "provider_exact_models": receipt["provider_exact_models"],
        "provider_response_bindings_sha256": receipt[
            "provider_response_bindings_sha256"
        ],
        "collection_receipt_sha256": receipt["content_hash"],
        "output_file_sha256": receipt["output_file_sha256"],
        "batch_id": receipt["batch_id"],
        "input_file_id": receipt["input_file_id"],
        "output_file_id": receipt["output_file_id"],
        "batch_usage": receipt["batch_usage"],
        "strict_actual_cost": receipt["strict_actual_cost"],
        "price_snapshot_id": receipt["price_snapshot_id"],
    }
    results_path = collection_root / "external-results.jsonl"
    ingested = ingest_results(
        root, method_id, results_path, result_source=result_source
    )
    label_set_manifest, _labels = _load_label_set(root, manifest, method)
    label_root = root / "label-sets" / ingested.label_set_id
    label_finalization = _load_hashed(label_root / "finalization-receipt.json")
    core = {
        "schema_version": "adag.graph-labeling.openai-batch-finalization.v1",
        "collection_receipt_sha256": receipt["content_hash"],
        "output_file_sha256": receipt["output_file_sha256"],
        "batch_id": receipt["batch_id"],
        "provider_exact_models": receipt["provider_exact_models"],
        "label_set_id": ingested.label_set_id,
        "label_count": ingested.label_count,
        "label_set_manifest_sha256": label_set_manifest["content_hash"],
        "label_set_labels_file_sha256": label_set_manifest["labels_file_sha256"],
        "label_set_finalization_receipt_sha256": label_finalization["content_hash"],
        "result_source_sha256": canonical_sha256(result_source),
    }
    finalization = _self_hashed(core)
    _write_or_verify_json(finalization_path, finalization)
    return {**receipt, "label_set_receipt": ingested.model_dump(mode="json")}


def _collect_openai_batch_unlocked(
    run_root: Path,
    method_id: str,
    *,
    transport: OpenAIBatchTransport | None = None,
    finalize: bool = False,
) -> dict[str, Any]:
    """Download, validate, normalize, and optionally ingest a completed Batch."""

    root, manifest, method, request_values, label_set_id = _context(run_root, method_id)
    requests = {request.request_id: request for request in request_values}
    batch_root = _batch_root(root, label_set_id)
    plan = _load_bound_plan(batch_root, manifest, method, request_values, label_set_id)
    submission = _load_bound_submission(batch_root, plan)
    collection_root = batch_root / "collection"
    if collection_root.exists():
        receipt = _load_collection(
            collection_root,
            manifest=manifest,
            method=method,
            plan=plan,
            submission=submission,
        )
        if finalize:
            return _finalize_collection(
                root=root,
                manifest=manifest,
                method=method,
                method_id=method_id,
                collection_root=collection_root,
                receipt=receipt,
            )
        return receipt
    live = transport or SDKOpenAIBatchTransport()
    attempts_root = batch_root / "collection-attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_id = _attempt_id()
    attempt_root = attempts_root / attempt_id
    attempt_root.mkdir()
    attempt_intent = _self_hashed(
        {
            "schema_version": "adag.graph-labeling.openai-batch-collection-intent.v1",
            "created_at": _now(),
            "attempt_id": attempt_id,
            "submission_sha256": submission["content_hash"],
            "batch_identity_sha256": submission["batch_identity_sha256"],
            "batch_id": submission["batch_id"],
        }
    )
    atomic_write_json(attempt_root / "intent.json", attempt_intent)
    snapshot = dict(live.retrieve_batch(str(submission["batch_id"])))
    snapshot_receipt = _self_hashed(
        {
            "schema_version": "adag.graph-labeling.openai-batch-download-snapshot.v1",
            "observed_at": _now(),
            "attempt_id": attempt_id,
            "attempt_intent_sha256": attempt_intent["content_hash"],
            "submission_sha256": submission["content_hash"],
            "remote": snapshot,
        }
    )
    atomic_write_json(attempt_root / "snapshot.json", snapshot_receipt)
    output_file_id = snapshot.get("output_file_id")
    error_file_id = snapshot.get("error_file_id")
    output = (
        live.download_file(output_file_id) if isinstance(output_file_id, str) else b""
    )
    errors = (
        live.download_file(error_file_id) if isinstance(error_file_id, str) else b""
    )
    for name, content in (("output.jsonl", output), ("errors.jsonl", errors)):
        path = attempt_root / name
        if content:
            atomic_write_bytes(path, content)
    download_receipt = _self_hashed(
        {
            "schema_version": "adag.graph-labeling.openai-batch-download.v1",
            "recorded_at": _now(),
            "attempt_id": attempt_id,
            "attempt_intent_sha256": attempt_intent["content_hash"],
            "snapshot_sha256": snapshot_receipt["content_hash"],
            "output_file_id": output_file_id,
            "output_file_sha256": hashlib.sha256(output).hexdigest(),
            "error_file_id": error_file_id,
            "error_file_sha256": hashlib.sha256(errors).hexdigest(),
        }
    )
    atomic_write_json(attempt_root / "download.json", download_receipt)
    _validate_remote(snapshot, submission)
    counts = snapshot.get("request_counts") or {}
    if not isinstance(counts, Mapping):
        raise ValueError("OpenAI Batch request counts are malformed")
    if (
        snapshot.get("status") != "completed"
        or counts.get("total") != len(requests)
        or counts.get("completed") != len(requests)
        or counts.get("failed") != 0
        or not output
        or error_file_id is not None
        or errors
    ):
        raise ValueError("OpenAI Batch is partial, failed, or contains request errors")
    external, usage = _parse_output(output, requests)
    aggregate_usage = openai_usage(snapshot.get("usage"))
    if aggregate_usage.input_tokens is None or aggregate_usage.output_tokens is None:
        raise ValueError("OpenAI Batch aggregate usage is incomplete")
    per_request_usage = [Usage.model_validate(item["usage"]) for item in usage.values()]
    for field in (
        "input_tokens",
        "uncached_input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
    ):
        aggregate_value = getattr(aggregate_usage, field)
        if aggregate_value is None:
            continue
        if aggregate_value != sum(
            getattr(item, field) or 0 for item in per_request_usage
        ):
            raise ValueError(f"OpenAI Batch aggregate usage mismatch: {field}")
    raw_aggregate_usage = snapshot.get("usage")
    if isinstance(raw_aggregate_usage, Mapping):
        total = raw_aggregate_usage.get("total_tokens")
        if total is not None and total != (
            aggregate_usage.input_tokens + aggregate_usage.output_tokens
        ):
            raise ValueError("OpenAI Batch aggregate total-token usage mismatch")
    prices = load_price_snapshot(PRICE_SNAPSHOT)
    if file_sha256(PRICE_SNAPSHOT) != plan["price_snapshot_file_sha256"]:
        raise ValueError("OpenAI Batch price snapshot drift")
    rates = prices["rates"]["openai"][plan["model"]]["native_batch"]
    cost_rows = {
        request_id: _strict_cost(
            rates,
            Usage.model_validate(usage[request_id]["usage"]),
            context=request_id,
        )
        for request_id in sorted(requests)
    }
    aggregate_cost = _strict_cost(rates, aggregate_usage, context="batch aggregate")
    actual_cost = sum(float(row["total_cost_usd"]) for row in cost_rows.values())
    if abs(actual_cost - float(aggregate_cost["total_cost_usd"])) > 1e-12:
        raise ValueError("OpenAI Batch aggregate cost reconciliation mismatch")
    if (
        actual_cost > float(plan["caller_max_cost_usd"]) + 1e-12
        or actual_cost > float(plan["projected_cost_ceiling_usd"]) + 1e-12
    ):
        raise ValueError("OpenAI Batch actual cost exceeds authorized ceiling")
    packets = _packets_by_occurrence(root, manifest)
    for row in external:
        request = requests[row.request_id]
        normalize_structured_label(
            row.raw_payload,
            packet=packets[request.occurrence_id],
            method=method,
            logical_request_sha256=row.logical_request_sha256,
            result_sha256=row.raw_response_sha256,
        )
    result_rows = [row.model_dump(mode="json") for row in external]
    expected_results = _jsonl_bytes(result_rows)
    provider_exact_models = sorted({str(item["model"]) for item in usage.values()})
    if len(provider_exact_models) != 1:
        raise ValueError("OpenAI Batch returned mixed exact provider models")
    response_bindings = {
        request_id: {
            "response_id": item["response_id"],
            "provider_request_id": item["provider_request_id"],
            "exact_model": item["model"],
            "usage": item["usage"],
        }
        for request_id, item in sorted(usage.items())
    }
    staging = Path(tempfile.mkdtemp(prefix=".collection-", dir=batch_root))
    results_path = staging / "external-results.jsonl"
    atomic_write_jsonl(results_path, result_rows)
    atomic_write_bytes(staging / "provider-output.jsonl", output)
    receipt_core = {
        "schema_version": "adag.graph-labeling.openai-batch-collection.v1",
        "collected_at": _now(),
        "submission_sha256": submission["content_hash"],
        "run_manifest_sha256": manifest["content_hash"],
        "study_sha256": manifest["study_sha256"],
        "method_sha256": method.identity_sha256,
        "batch_id": submission["batch_id"],
        "input_file_id": submission["input_file_id"],
        "output_file_id": output_file_id,
        "error_file_id": error_file_id,
        "output_file_sha256": hashlib.sha256(output).hexdigest(),
        "external_results_sha256": hashlib.sha256(expected_results).hexdigest(),
        "request_count": len(requests),
        "request_bindings": submission["request_bindings"],
        "configured_model": plan["model"],
        "provider_exact_models": provider_exact_models,
        "provider_response_bindings_sha256": canonical_sha256(response_bindings),
        "provider_response_bindings": response_bindings,
        "download_attempt_id": attempt_id,
        "download_attempt_intent_sha256": attempt_intent["content_hash"],
        "download_snapshot_sha256": snapshot_receipt["content_hash"],
        "download_receipt_sha256": download_receipt["content_hash"],
        "batch_usage": aggregate_usage.model_dump(mode="json"),
        "per_request_remote_metadata": usage,
        "price_snapshot_id": prices["snapshot_id"],
        "price_snapshot_file_sha256": plan["price_snapshot_file_sha256"],
        "per_request_strict_costs": cost_rows,
        "strict_actual_cost": aggregate_cost,
        "actual_cost_usd": actual_cost,
        "authorized_cost_guard_usd": plan["caller_max_cost_usd"],
        "projected_cost_ceiling_usd": plan["projected_cost_ceiling_usd"],
    }
    receipt = _self_hashed(receipt_core)
    atomic_write_json(staging / "receipt.json", receipt)
    os.replace(staging, collection_root)
    if finalize:
        return _finalize_collection(
            root=root,
            manifest=manifest,
            method=method,
            method_id=method_id,
            collection_root=collection_root,
            receipt=receipt,
        )
    return receipt


@contextmanager
def _lifecycle_gate(run_root: Path, method_id: str):
    root, _, _, _, label_set_id = _context(run_root, method_id)
    locks_root = root / "openai-batches" / ".locks"
    locks_root.mkdir(parents=True, exist_ok=True)
    lock = locks_root / f"{label_set_id}.lock"
    handle = lock.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("OpenAI Batch lifecycle gate is held") from error
        owner = {
            "schema_version": "adag.graph-labeling.openai-batch-lock.v1",
            "claimed_at": _now(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "method_id": method_id,
            "label_set_id": label_set_id,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def submit_openai_batch(
    run_root: Path,
    method_id: str,
    *,
    max_cost_usd: float,
    transport: OpenAIBatchTransport | None = None,
) -> dict[str, Any]:
    with _lifecycle_gate(run_root, method_id):
        return _submit_openai_batch_unlocked(
            run_root,
            method_id,
            max_cost_usd=max_cost_usd,
            transport=transport,
        )


def abandon_openai_attempt(
    run_root: Path, method_id: str, *, reason: str
) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("abandonment requires a nonempty reason")
    with _lifecycle_gate(run_root, method_id):
        root, manifest, method, requests, label_set_id = _context(run_root, method_id)
        batch_root = _batch_root(root, label_set_id)
        plan = _load_bound_plan(batch_root, manifest, method, requests, label_set_id)
        if (batch_root / "submission.json").exists():
            raise ValueError("cannot abandon a receipt-bound submission")
        abandonment_path = batch_root / "abandoned.json"
        if abandonment_path.exists():
            return _load_hashed(abandonment_path)
        intents = [
            _load_hashed(path)["content_hash"]
            for path in (
                batch_root / "upload-intent.json",
                batch_root / "create-intent.json",
            )
            if path.exists()
        ]
        if not intents:
            raise ValueError("no indeterminate OpenAI Batch attempt exists")
        core = {
            "schema_version": "adag.graph-labeling.openai-batch-abandonment.v1",
            "created_at": _now(),
            "batch_identity_sha256": plan["batch_identity_sha256"],
            "plan_sha256": plan["content_hash"],
            "intent_receipt_sha256s": intents,
            "reason": reason,
            "terminal": True,
        }
        receipt = _self_hashed(core)
        _write_or_verify_json(abandonment_path, receipt)
        return receipt


def recover_openai_batch(
    run_root: Path,
    method_id: str,
    batch_id: str,
    *,
    transport: OpenAIBatchTransport | None = None,
) -> dict[str, Any]:
    with _lifecycle_gate(run_root, method_id):
        return _recover_openai_batch_unlocked(
            run_root, method_id, batch_id, transport=transport
        )


def recover_openai_upload(
    run_root: Path,
    method_id: str,
    input_file_id: str,
    *,
    transport: OpenAIBatchTransport | None = None,
) -> dict[str, Any]:
    with _lifecycle_gate(run_root, method_id):
        return _recover_openai_upload_unlocked(
            run_root, method_id, input_file_id, transport=transport
        )


def openai_batch_status(
    run_root: Path,
    method_id: str,
    *,
    transport: OpenAIBatchTransport | None = None,
) -> dict[str, Any]:
    with _lifecycle_gate(run_root, method_id):
        return _openai_batch_status_unlocked(run_root, method_id, transport=transport)


def collect_openai_batch(
    run_root: Path,
    method_id: str,
    *,
    transport: OpenAIBatchTransport | None = None,
    finalize: bool = False,
) -> dict[str, Any]:
    with _lifecycle_gate(run_root, method_id):
        return _collect_openai_batch_unlocked(
            run_root, method_id, transport=transport, finalize=finalize
        )
