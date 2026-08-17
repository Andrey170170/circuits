"""Receipt-bound native OpenAI Batch lifecycle for coarse qualification v2."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import validate_decisions
from circuits.analysis.bonafide.coarse_sampling_annotation_v2 import (
    ARM_IDS,
    load_v2_qualification,
)
from circuits.labeling.api import openai_stop_reason, openai_usage
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)
from circuits.labeling.pricing import estimate_cost, load_price_snapshot
from circuits.labeling.schema import Usage

RUN_SCHEMA = "adag.process-witness.coarse-openai-batch-run.v2"
SUBMISSION_INTENT_SCHEMA = (
    "adag.process-witness.coarse-openai-batch-submission-intent.v2"
)
SUBMISSION_SCHEMA = "adag.process-witness.coarse-openai-batch-submission.v2"
STATUS_SCHEMA = "adag.process-witness.coarse-openai-batch-status.v2"
EVENT_SCHEMA = "adag.process-witness.coarse-openai-batch-event.v2"
COLLECTION_SCHEMA = "adag.process-witness.coarse-openai-batch-collection.v2"
STAGE = "coarse_qualification_v2"

_BOUND_RUN_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_openai_batch_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/api.py",
    "circuits/labeling/io.py",
    "circuits/labeling/pricing.py",
    "circuits/labeling/schema.py",
    "scripts/bonafide/process_witness_coarse_openai_batch_v2.py",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _self_hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = dict(value)
    if field in payload:
        raise ValueError(f"payload already contains {field}")
    payload[field] = canonical_sha256(payload)
    return payload


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("coarse v2 run repository root drift")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("coarse v2 submission requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_RUN_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse v2 run source is untracked: {relative}")
        path = root / relative
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        committed = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        expected = hashlib.sha256(committed).hexdigest()
        if file_sha256(path) != expected:
            raise ValueError(f"coarse v2 run source differs from HEAD: {relative}")
        files.append({"path": relative, "git_blob": blob, "sha256": expected})
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "files": files,
    }


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _openai_client() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "required API key environment variable is missing: OPENAI_API_KEY"
        )
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        project=os.environ.get("OPENAI_PROJECT_ID"),
        organization=os.environ.get("OPENAI_ORG_ID"),
        timeout=180.0,
        max_retries=0,
    )


def _provider_batch_dict(batch: Any) -> dict[str, Any]:
    counts = getattr(batch, "request_counts", None)
    usage = getattr(batch, "usage", None)
    return {
        "schema_version": "adag.labeling.provider-batch.v1",
        "provider": "openai",
        "batch_id": batch.id,
        "input_file_id": batch.input_file_id,
        "endpoint": batch.endpoint,
        "completion_window": batch.completion_window,
        "metadata": dict(getattr(batch, "metadata", None) or {}),
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "request_counts": counts.model_dump() if counts else None,
        "model": getattr(batch, "model", None),
        "usage": usage.model_dump() if usage else None,
    }


def _submit_openai_batch_v2(
    input_path: Path, *, run_id: str, stage: str, key_env: str
) -> dict[str, Any]:
    if key_env != "OPENAI_API_KEY":
        raise ValueError("coarse v2 API key environment drift")
    client = _openai_client()
    with input_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    metadata = {"run_id": run_id, "stage": stage}
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata,
    )
    if getattr(batch, "input_file_id", None) != uploaded.id:
        raise ValueError("coarse v2 created Batch input file id drift")
    return _provider_batch_dict(batch)


def _retrieve_openai_batch_v2(batch_id: str, *, run_id: str) -> dict[str, Any]:
    batch = _openai_client().batches.retrieve(batch_id)
    return _provider_batch_dict(batch)


def _openai_file_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if hasattr(value, "read"):
        observed = value.read()
        if isinstance(observed, bytes):
            return observed
    if hasattr(value, "content") and isinstance(value.content, bytes):
        return value.content
    raise TypeError("OpenAI file content is not bytes")


def _download_openai_batch_v2(
    batch_id: str, *, run_id: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    client = _openai_client()
    batch = client.batches.retrieve(batch_id)
    snapshot = _provider_batch_dict(batch)
    if batch.id != batch_id or batch.status != "completed":
        raise ValueError("coarse v2 provider Batch is not collectable")
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
    if not raw_files:
        raise ValueError("coarse v2 completed Batch has no output or error file")
    return snapshot, raw_files


def _validate_provider_snapshot(
    value: Mapping[str, Any],
    *,
    batch_id: str | None,
    input_file_id: str | None,
    run_id: str,
) -> None:
    if value.get("provider") != "openai":
        raise ValueError("coarse v2 provider receipt provider drift")
    observed_model = value.get("model")
    if observed_model is not None and (
        not isinstance(observed_model, str)
        or not observed_model.startswith("gpt-5.6-luna")
    ):
        raise ValueError("coarse v2 provider receipt model drift")
    if not isinstance(value.get("batch_id"), str) or not isinstance(
        value.get("input_file_id"), str
    ):
        raise ValueError("coarse v2 provider receipt lacks batch or input file id")
    if batch_id is not None and value.get("batch_id") != batch_id:
        raise ValueError("coarse v2 provider receipt batch id drift")
    if input_file_id is not None and value.get("input_file_id") != input_file_id:
        raise ValueError("coarse v2 provider receipt input file id drift")
    if (
        value.get("endpoint") != "/v1/responses"
        or value.get("completion_window") != "24h"
        or value.get("metadata") != {"run_id": run_id, "stage": STAGE}
    ):
        raise ValueError("coarse v2 provider metadata, endpoint, or window drift")


def submit_v2_batch(
    *,
    qualification_root: Path,
    run_root: Path,
    maximum_authorized_cost_usd: float,
    authorization_note: str,
    submitter: Callable[..., dict[str, Any]] = _submit_openai_batch_v2,
) -> dict[str, Any]:
    """Persist intent, upload exact JSONL, and create one Batch exactly once."""

    loaded = load_v2_qualification(qualification_root)
    projected = float(loaded["cost_plan"]["projected_upper_bound_usd"])
    if not 0 < maximum_authorized_cost_usd <= 20.0:
        raise ValueError("coarse v2 authorization must be in (0, 20]")
    if projected > maximum_authorized_cost_usd:
        raise ValueError("coarse v2 persisted cost ceiling exceeds authorization")
    source_revision = _source_revision()
    if run_root.exists():
        raise FileExistsError(f"coarse v2 run already exists: {run_root}")
    run_root.mkdir(parents=True)
    try:
        input_path = run_root / "input.jsonl"
        shutil.copyfile(qualification_root / "batch-input.jsonl", input_path)
        if file_sha256(input_path) != file_sha256(
            qualification_root / "batch-input.jsonl"
        ):
            raise ValueError("coarse v2 run input copy drift")
        intent = _self_hashed(
            {
                "schema_version": RUN_SCHEMA,
                "status": "intent_persisted_before_provider_calls",
                "created_at": _utc_now(),
                "qualification_root": str(qualification_root.resolve()),
                "qualification_manifest_sha256": loaded["manifest"]["manifest_sha256"],
                "cost_plan_sha256": loaded["cost_plan"]["cost_plan_sha256"],
                "projected_upper_bound_usd": projected,
                "maximum_authorized_cost_usd": maximum_authorized_cost_usd,
                "authorization_note": authorization_note,
                "input_jsonl_sha256": file_sha256(input_path),
                "request_ids_in_order": [
                    request["request_id"] for request in loaded["requests"]
                ],
                "environment": {
                    "hostname": platform.node(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "endpoint_identity": "https://api.openai.com/v1",
                    "openai_project_sha256": (
                        hashlib.sha256(
                            os.environ["OPENAI_PROJECT_ID"].encode()
                        ).hexdigest()
                        if os.environ.get("OPENAI_PROJECT_ID")
                        else None
                    ),
                    "openai_organization_sha256": (
                        hashlib.sha256(os.environ["OPENAI_ORG_ID"].encode()).hexdigest()
                        if os.environ.get("OPENAI_ORG_ID")
                        else None
                    ),
                },
                "source_revision": source_revision,
            },
            "run_intent_sha256",
        )
        atomic_write_json(run_root / "run-intent.json", intent)
        submission_intent = _self_hashed(
            {
                "schema_version": SUBMISSION_INTENT_SCHEMA,
                "created_at": _utc_now(),
                "run_intent_sha256": intent["run_intent_sha256"],
                "input_jsonl_sha256": intent["input_jsonl_sha256"],
                "request_count": len(loaded["requests"]),
                "stage": STAGE,
            },
            "submission_intent_sha256",
        )
        atomic_write_json(run_root / "submission-intent.json", submission_intent)
        provider = submitter(
            input_path,
            run_id=intent["run_intent_sha256"],
            stage=STAGE,
            key_env=loaded["config"]["provider"]["api_key_env"],
        )
        _validate_provider_snapshot(
            provider,
            batch_id=None,
            input_file_id=None,
            run_id=intent["run_intent_sha256"],
        )
        receipt = _self_hashed(
            {
                "schema_version": SUBMISSION_SCHEMA,
                "recorded_at": _utc_now(),
                "run_intent_sha256": intent["run_intent_sha256"],
                "submission_intent_sha256": submission_intent[
                    "submission_intent_sha256"
                ],
                "provider_response": provider,
            },
            "submission_sha256",
        )
        atomic_write_json(run_root / "submission.json", receipt)
        return receipt
    except BaseException as error:
        # Once a submission intent exists, provider state may be indeterminate.  The
        # retained run root blocks an automatic duplicate upload/submission.
        intent_path = run_root / "run-intent.json"
        submission_intent_path = run_root / "submission-intent.json"
        if intent_path.is_file() and submission_intent_path.is_file():
            intent = _load_object(intent_path)
            submission_intent = _load_object(submission_intent_path)
            failure = _self_hashed(
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "status": "failed_closed_indeterminate_provider_state",
                    "recorded_at": _utc_now(),
                    "run_intent_sha256": intent["run_intent_sha256"],
                    "submission_intent_sha256": submission_intent[
                        "submission_intent_sha256"
                    ],
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                    "automatic_retry_permitted": False,
                },
                "submission_failure_sha256",
            )
            atomic_write_json(run_root / "submission-failure.json", failure)
            raise RuntimeError(
                "coarse v2 provider state is indeterminate; automatic retry is forbidden"
            ) from error
        raise


def _load_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    intent = _load_object(run_root / "run-intent.json")
    _verify_self_hash(intent, "run_intent_sha256", "coarse v2 run intent")
    qualification = load_v2_qualification(Path(intent["qualification_root"]))
    if (
        qualification["manifest"]["manifest_sha256"]
        != intent["qualification_manifest_sha256"]
        or qualification["cost_plan"]["cost_plan_sha256"] != intent["cost_plan_sha256"]
        or file_sha256(run_root / "input.jsonl") != intent["input_jsonl_sha256"]
    ):
        raise ValueError("coarse v2 run/qualification binding drift")
    if intent["request_ids_in_order"] != [
        request["request_id"] for request in qualification["requests"]
    ]:
        raise ValueError("coarse v2 run request order drift")
    submission_intent = _load_object(run_root / "submission-intent.json")
    _verify_self_hash(
        submission_intent,
        "submission_intent_sha256",
        "coarse v2 submission intent",
    )
    if (
        submission_intent["run_intent_sha256"] != intent["run_intent_sha256"]
        or submission_intent["input_jsonl_sha256"] != intent["input_jsonl_sha256"]
        or submission_intent["request_count"] != len(qualification["requests"])
    ):
        raise ValueError("coarse v2 submission-intent binding drift")
    submission = _load_object(run_root / "submission.json")
    _verify_self_hash(submission, "submission_sha256", "coarse v2 submission")
    if (
        submission["run_intent_sha256"] != intent["run_intent_sha256"]
        or submission["submission_intent_sha256"]
        != submission_intent["submission_intent_sha256"]
    ):
        raise ValueError("coarse v2 submission run binding drift")
    provider = submission["provider_response"]
    _validate_provider_snapshot(
        provider,
        batch_id=provider["batch_id"],
        input_file_id=provider["input_file_id"],
        run_id=intent["run_intent_sha256"],
    )
    return intent, qualification, submission


def check_v2_batch(
    *,
    run_root: Path,
    retriever: Callable[..., dict[str, Any]] = _retrieve_openai_batch_v2,
) -> dict[str, Any]:
    """Retrieve and hash-chain one provider status observation."""

    intent, _qualification, submission = _load_run(run_root)
    provider = submission["provider_response"]
    observed = retriever(provider["batch_id"], run_id=intent["run_intent_sha256"])
    _validate_provider_snapshot(
        observed,
        batch_id=provider["batch_id"],
        input_file_id=provider["input_file_id"],
        run_id=intent["run_intent_sha256"],
    )
    status_root = run_root / "status"
    status_root.mkdir(exist_ok=True)
    prior = sorted(status_root.glob("receipt-*.json"))
    previous_sha = None
    for index, path in enumerate(prior):
        if path.name != f"receipt-{index:04d}.json":
            raise ValueError("coarse v2 status receipt numbering drift")
        previous = _load_object(path)
        _verify_self_hash(previous, "status_sha256", "coarse v2 status")
        if (
            previous.get("previous_status_sha256") != previous_sha
            or previous.get("submission_sha256") != submission["submission_sha256"]
            or previous.get("run_intent_sha256") != intent["run_intent_sha256"]
        ):
            raise ValueError("coarse v2 status receipt chain drift")
        previous_sha = previous["status_sha256"]
    receipt = _self_hashed(
        {
            "schema_version": STATUS_SCHEMA,
            "recorded_at": _utc_now(),
            "run_intent_sha256": intent["run_intent_sha256"],
            "submission_sha256": submission["submission_sha256"],
            "previous_status_sha256": previous_sha,
            "provider_response": observed,
        },
        "status_sha256",
    )
    atomic_write_json(status_root / f"receipt-{len(prior):04d}.json", receipt)
    return receipt


def _response_text(body: Mapping[str, Any]) -> tuple[str, str | None, list[str | None]]:
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
        for block in item.get("content", []):
            if not isinstance(block, Mapping):
                continue
            if block.get("type") in {"output_text", "text"}:
                texts.append(str(block.get("text", "")))
            elif block.get("type") == "refusal":
                refusals.append(str(block.get("refusal", "")))
    return "".join(texts), ("\n".join(refusals) if refusals else None), message_statuses


def parse_v2_batch_row(
    row: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    """Interpret one exact Batch row without accepting partial target coverage."""

    if row.get("custom_id") != request["request_id"]:
        raise ValueError("coarse v2 Batch custom_id does not match request")
    common = {
        "schema_version": EVENT_SCHEMA,
        "request_id": request["request_id"],
        "arm_id": request["arm_id"],
        "body_sha256": request["body_sha256"],
        "source_v1_request_id": request["source_v1_request_id"],
        "repeat_of_request_id": request["repeat_of_request_id"],
        "raw_row_sha256": canonical_sha256(row),
    }
    response = row.get("response")
    row_error = row.get("error")
    if row_error is not None or not isinstance(response, Mapping):
        return {
            **common,
            "validation_status": "provider_error",
            "error_type": "batch_request_error",
            "error_message": json.dumps(row_error or row, sort_keys=True)[:2000],
            "usage": openai_usage(None).model_dump(mode="json"),
            "decisions": None,
        }
    body = response.get("body")
    if response.get("status_code") != 200 or not isinstance(body, Mapping):
        return {
            **common,
            "validation_status": "provider_error",
            "error_type": "batch_http_error",
            "error_message": json.dumps(body or response, sort_keys=True)[:2000],
            "usage": openai_usage(None).model_dump(mode="json"),
            "decisions": None,
        }
    usage = openai_usage(body.get("usage")).model_dump(mode="json")
    text, refusal, message_statuses = _response_text(body)
    details = {
        **common,
        "provider_request_id": response.get("request_id") or body.get("id"),
        "model_requested": request["provider_body"]["model"],
        "model_resolved": body.get("model"),
        "response_status": body.get("status"),
        "stop_reason": openai_stop_reason(body),
        "raw_response_sha256": canonical_sha256(body),
        "raw_text": text or None,
        "raw_text_sha256": (
            hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        ),
        "usage": usage,
    }
    if refusal is not None:
        return {
            **details,
            "validation_status": "refusal",
            "error_type": "model_refusal",
            "error_message": refusal[:2000],
            "decisions": None,
        }
    if body.get("status") != "completed":
        return {
            **details,
            "validation_status": "incomplete",
            "error_type": "incomplete_response",
            "error_message": json.dumps(body.get("incomplete_details"), sort_keys=True)[
                :2000
            ],
            "decisions": None,
        }
    resolved_model = body.get("model")
    requested_model = request["provider_body"]["model"]
    if not isinstance(resolved_model, str) or not resolved_model.startswith(
        requested_model
    ):
        return {
            **details,
            "validation_status": "provider_error",
            "error_type": "resolved_model_drift",
            "error_message": (
                f"requested {requested_model!r}, received {resolved_model!r}"
            ),
            "decisions": None,
        }
    if any(status != "completed" for status in message_statuses):
        return {
            **details,
            "validation_status": "incomplete",
            "error_type": "incomplete_message",
            "error_message": json.dumps(message_statuses),
            "decisions": None,
        }
    try:
        value = json.loads(text)
        decisions = validate_decisions(value, focal_unit_ids=request["focal_unit_ids"])
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return {
            **details,
            "validation_status": "invalid_output",
            "error_type": type(error).__name__,
            "error_message": str(error)[:2000],
            "decisions": None,
        }
    return {
        **details,
        "validation_status": "success",
        "error_type": None,
        "error_message": None,
        "decisions": decisions,
    }


def _jsonl_rows(content: bytes, source: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"coarse v2 {source} file is not UTF-8") from error
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"coarse v2 {source} file has invalid JSON at line {line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"coarse v2 {source} row is not an object")
        rows.append(value)
    return rows


def _estimate_v2_actual_cost(
    prices: dict[str, Any], *, model: str, usage: Usage
) -> tuple[Any, bool]:
    """Price receipt usage, applying the documented long-context multipliers."""

    adjusted = prices
    applied = False
    long_context = prices.get("long_context", {}).get(model)
    if isinstance(long_context, Mapping) and usage.input_tokens is not None:
        threshold = int(long_context["threshold_input_tokens_exclusive"])
        if usage.input_tokens > threshold:
            adjusted = copy.deepcopy(prices)
            rates = adjusted["rates"]["openai"][model]["native_batch"]
            input_multiplier = float(long_context["input_multiplier"])
            output_multiplier = float(long_context["output_multiplier"])
            for field in (
                "input_per_million",
                "cache_read_per_million",
                "cache_write_per_million",
            ):
                rates[field] = float(rates[field]) * input_multiplier
            rates["output_per_million"] = (
                float(rates["output_per_million"]) * output_multiplier
            )
            applied = True
    return (
        estimate_cost(
            adjusted,
            provider="openai",
            model=model,
            transport="native_batch",
            usage=usage,
        ),
        applied,
    )


def _load_or_create_collection_intent(
    *,
    run_root: Path,
    intent: Mapping[str, Any],
    submission: Mapping[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    path = run_root / "collection-intent.json"
    if path.exists():
        existing = _load_object(path)
        _verify_self_hash(
            existing, "collection_intent_sha256", "coarse v2 collection intent"
        )
        if (
            existing.get("run_intent_sha256") != intent["run_intent_sha256"]
            or existing.get("submission_sha256") != submission["submission_sha256"]
            or existing.get("batch_id") != batch_id
        ):
            raise ValueError("coarse v2 collection intent binding drift")
        return existing
    value = _self_hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "status": "collection_intent_persisted",
            "recorded_at": _utc_now(),
            "run_intent_sha256": intent["run_intent_sha256"],
            "submission_sha256": submission["submission_sha256"],
            "batch_id": batch_id,
        },
        "collection_intent_sha256",
    )
    atomic_write_json(path, value)
    return value


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_object(path) != value:
            raise ValueError(f"retained coarse v2 JSON artifact drift: {path}")
        return
    atomic_write_json(path, value)


def _write_or_verify_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"retained coarse v2 byte artifact drift: {path}")
        return
    atomic_write_bytes(path, value)


def collect_v2_batch(
    *,
    run_root: Path,
    downloader: Callable[..., tuple[dict[str, Any], dict[str, dict[str, Any]]]] = (
        _download_openai_batch_v2
    ),
) -> dict[str, Any]:
    """Download opaque files, require exact row coverage, validate, and price."""

    intent, qualification, submission = _load_run(run_root)
    if (run_root / "collection-manifest.json").exists():
        raise FileExistsError("coarse v2 Batch has already been collected")
    provider = submission["provider_response"]
    collection_intent = _load_or_create_collection_intent(
        run_root=run_root,
        intent=intent,
        submission=submission,
        batch_id=provider["batch_id"],
    )
    snapshot, raw_files = downloader(
        provider["batch_id"], run_id=intent["run_intent_sha256"]
    )
    _validate_provider_snapshot(
        snapshot,
        batch_id=provider["batch_id"],
        input_file_id=provider["input_file_id"],
        run_id=intent["run_intent_sha256"],
    )
    if snapshot.get("status") != "completed":
        raise ValueError("coarse v2 Batch is not completed")
    raw_root = run_root / "raw"
    raw_root.mkdir(exist_ok=True)
    _write_or_verify_json(raw_root / "provider-snapshot.json", snapshot)
    raw_bindings = [
        {
            "source": "provider_snapshot",
            "file_id": None,
            "path": "raw/provider-snapshot.json",
            "sha256": file_sha256(raw_root / "provider-snapshot.json"),
            "bytes": (raw_root / "provider-snapshot.json").stat().st_size,
        }
    ]
    rows_by_id: dict[str, dict[str, Any]] = {}
    for source in ("output", "error"):
        item = raw_files.get(source)
        if item is None:
            continue
        content = item["content"]
        if not isinstance(content, bytes):
            raise ValueError("coarse v2 downloaded provider content is not bytes")
        path = raw_root / f"{source}.jsonl"
        if item.get("file_id") != snapshot.get(f"{source}_file_id"):
            raise ValueError(f"coarse v2 {source} file id receipt drift")
        _write_or_verify_bytes(path, content)
        raw_bindings.append(
            {
                "source": source,
                "file_id": item["file_id"],
                "path": str(path.relative_to(run_root)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        for row in _jsonl_rows(content, source):
            request_id = row.get("custom_id")
            if not isinstance(request_id, str):
                raise ValueError("coarse v2 Batch row has no string custom_id")
            if request_id in rows_by_id:
                raise ValueError("coarse v2 Batch output/error union repeats custom_id")
            rows_by_id[request_id] = row

    requests = qualification["requests"]
    requests_by_id = {request["request_id"]: request for request in requests}
    unknown = sorted(set(rows_by_id) - set(requests_by_id))
    missing = sorted(set(requests_by_id) - set(rows_by_id))
    if unknown or missing:
        failure = _self_hashed(
            {
                **collection_intent,
                "status": "failed_closed_row_coverage",
                "completed_at": _utc_now(),
                "unknown_custom_ids": unknown,
                "missing_custom_ids": missing,
                "raw_file_bindings": raw_bindings,
            },
            "collection_manifest_sha256",
        )
        atomic_write_json(run_root / "collection-manifest.json", failure)
        _readonly_tree(run_root)
        raise ValueError("coarse v2 Batch row union does not exactly cover requests")

    price_path = Path(qualification["cost_plan"]["price_snapshot_path"])
    prices = load_price_snapshot(price_path)
    events = []
    total_cost = 0.0
    cost_complete = True
    usage_fields = (
        "input_tokens",
        "uncached_input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
    )
    usage_totals = dict.fromkeys(usage_fields, 0)
    usage_complete = dict.fromkeys(usage_fields, True)
    for request in requests:
        event = parse_v2_batch_row(rows_by_id[request["request_id"]], request)
        usage = Usage.model_validate(event["usage"])
        cost, long_context_applied = _estimate_v2_actual_cost(
            prices,
            model=request["provider_body"]["model"],
            usage=usage,
        )
        event["price_snapshot_id"] = prices["snapshot_id"]
        event["price_snapshot_sha256"] = file_sha256(price_path)
        event["cost"] = cost.model_dump(mode="json")
        event["long_context_price_multiplier_applied"] = long_context_applied
        event["event_sha256"] = canonical_sha256(event)
        events.append(event)
        if cost.total_cost is None:
            cost_complete = False
        else:
            total_cost += float(cost.total_cost)
        for field in usage_fields:
            value = getattr(usage, field)
            if value is None:
                usage_complete[field] = False
            else:
                usage_totals[field] += int(value)

    atomic_write_jsonl(run_root / "events.jsonl", events, overwrite=True)
    success = [event for event in events if event["validation_status"] == "success"]
    unique_arm_targets = {
        (event["arm_id"], decision["unit_id"])
        for event in success
        for decision in event["decisions"]
    }
    arm_success = {
        arm: sum(
            event["validation_status"] == "success" and event["arm_id"] == arm
            for event in events
        )
        for arm in ARM_IDS
    }
    all_success = len(success) == len(requests)
    exact_unique_coverage = len(unique_arm_targets) == 144
    authorization_exceeded = cost_complete and total_cost > float(
        intent["maximum_authorized_cost_usd"]
    )
    ready = (
        all_success
        and exact_unique_coverage
        and cost_complete
        and not authorization_exceeded
    )
    complete = _self_hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "status": (
                "complete"
                if ready
                else (
                    "failed_closed_authorization_exceeded"
                    if authorization_exceeded
                    else (
                        "failed_closed_unpriceable_usage"
                        if not cost_complete
                        else "failed_closed_provider_results"
                    )
                )
            ),
            "completed_at": _utc_now(),
            "run_intent_sha256": intent["run_intent_sha256"],
            "submission_sha256": submission["submission_sha256"],
            "collection_intent_sha256": collection_intent["collection_intent_sha256"],
            "batch_id": provider["batch_id"],
            "request_count": len(requests),
            "success_count": len(success),
            "failure_count": len(requests) - len(success),
            "arm_success_counts": arm_success,
            "unique_arm_target_coverage": len(unique_arm_targets),
            "expected_unique_arm_target_coverage": 144,
            "exact_target_coverage": exact_unique_coverage,
            "qualification_decisions_ready": ready,
            "maximum_authorized_cost_usd": intent["maximum_authorized_cost_usd"],
            "authorization_exceeded": authorization_exceeded,
            "usage_totals": {
                field: (usage_totals[field] if usage_complete[field] else None)
                for field in usage_fields
            },
            "known_priced_cost_usd": total_cost,
            "cost_complete": cost_complete,
            "actual_total_cost_usd": total_cost if cost_complete else None,
            "raw_file_bindings": raw_bindings,
            "events_jsonl_sha256": file_sha256(run_root / "events.jsonl"),
            "event_bindings_in_order": [
                {
                    "request_id": event["request_id"],
                    "event_sha256": event["event_sha256"],
                }
                for event in events
            ],
        },
        "collection_manifest_sha256",
    )
    atomic_write_json(run_root / "collection-manifest.json", complete)
    _readonly_tree(run_root)
    return complete
