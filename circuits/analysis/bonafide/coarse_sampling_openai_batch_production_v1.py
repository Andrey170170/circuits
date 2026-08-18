"""Receipt-bound multi-shard Batch lifecycle for coarse production v1.

All provider mutations are explicit calls.  Building or loading the campaign is
network free.  Submission intent is durable before upload/create; ambiguous
state forbids automatic retries and must be reconciled by metadata discovery.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import validate_decisions
from circuits.analysis.bonafide.coarse_sampling_openai_batch_v4 import (
    _estimate_v4_actual_cost,
    _openai_client,
    _openai_file_bytes,
    _provider_batch_dict,
    _response_text,
)
from circuits.analysis.bonafide.coarse_sampling_production_v1 import (
    iter_shard_requests,
    load_production_bundle,
    proposal_from_votes,
    sampling_groups,
)
from circuits.labeling.api import openai_stop_reason, openai_usage
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
)
from circuits.labeling.pricing import load_price_snapshot
from circuits.labeling.schema import Usage

CAMPAIGN_RUN_SCHEMA = "adag.process-witness.coarse-production-run.v1"
SUBMISSION_SCHEMA = "adag.process-witness.coarse-production-submission.v1"
STATUS_SCHEMA = "adag.process-witness.coarse-production-status.v1"
COLLECTION_SCHEMA = "adag.process-witness.coarse-production-collection.v1"
EVENT_SCHEMA = "adag.process-witness.coarse-production-event.v1"
UPLOAD_SCHEMA = "adag.process-witness.coarse-production-upload.v1"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(value)
    output[field] = canonical_sha256(output)
    return output


def _verify(value: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


@contextmanager
def _submission_gate(run_root: Path):
    """Serialize spend/queue checks and submission-intent creation across processes."""

    lock = run_root / ".submission-gate"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "coarse production submission gate is already held or stale; "
            "inspect provider state before manual lock removal"
        ) from error
    try:
        atomic_write_json(
            lock / "owner.json",
            {
                "schema_version": "adag.process-witness.coarse-production-lock.v1",
                "created_at": _now(),
                "hostname": platform.node(),
                "pid": os.getpid(),
            },
        )
        yield
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


def initialize_campaign_run(
    *,
    bundle_root: Path,
    run_root: Path,
    maximum_authorized_cost_usd: float,
    authorization_note: str,
    provider_queued_input_token_limit: int,
    maximum_concurrent_shards: int,
) -> dict[str, Any]:
    """Bind launch authorization and queue capacity without provider calls."""

    bundle = load_production_bundle(bundle_root, load_units=False)
    if run_root.exists():
        raise FileExistsError(f"coarse production run exists: {run_root}")
    if maximum_authorized_cost_usd <= 0 or not authorization_note.strip():
        raise ValueError(
            "coarse production requires an explicit positive spend authorization"
        )
    if not 1 <= maximum_concurrent_shards <= len(bundle["shards"]):
        raise ValueError("maximum concurrent shards is outside the frozen shard census")
    largest_forecasts = sorted(
        (
            int(shard["queued_input_tokens_empirical_forecast"])
            for shard in bundle["shards"]
        ),
        reverse=True,
    )[:maximum_concurrent_shards]
    if provider_queued_input_token_limit < sum(largest_forecasts):
        raise ValueError(
            "active API-tier queue limit cannot hold the requested shard concurrency"
        )
    run_root.mkdir(parents=True)
    try:
        shard_bindings = []
        for shard in bundle["shards"]:
            shard_root = run_root / "shards" / shard["shard_id"]
            shard_root.mkdir(parents=True)
            source = bundle_root / shard["path"]
            destination = shard_root / "input.jsonl"
            shutil.copyfile(source, destination)
            if file_sha256(destination) != shard["sha256"]:
                raise ValueError("coarse production run shard copy drift")
            shard_bindings.append(
                {
                    "shard_id": shard["shard_id"],
                    "input_relative_path": str(destination.relative_to(run_root)),
                    "input_sha256": shard["sha256"],
                    "bytes": shard["bytes"],
                    "request_count": shard["request_count"],
                    "direct_v4_cost_forecast_usd": shard[
                        "direct_v4_cost_forecast_usd"
                    ],
                    "queued_input_tokens_empirical_forecast": shard[
                        "queued_input_tokens_empirical_forecast"
                    ],
                }
            )
        intent = _hashed(
            {
                "schema_version": CAMPAIGN_RUN_SCHEMA,
                "status": "initialized_no_provider_calls",
                "created_at": _now(),
                "bundle_root": str(bundle_root.resolve()),
                "bundle_manifest_sha256": bundle["manifest"]["manifest_sha256"],
                "cost_plan_sha256": bundle["cost_plan"]["cost_plan_sha256"],
                "maximum_authorized_cost_usd": maximum_authorized_cost_usd,
                "authorization_note": authorization_note,
                "provider_queued_input_token_limit": provider_queued_input_token_limit,
                "maximum_concurrent_shards": maximum_concurrent_shards,
                "queue_policy": "active frozen shards must fit recorded concurrency and queued-token cap",
                "shards": shard_bindings,
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
                "network_calls_made": 0,
            },
            "campaign_run_sha256",
        )
        atomic_write_json(run_root / "campaign-intent.json", intent)
        return intent
    except BaseException:
        shutil.rmtree(run_root, ignore_errors=True)
        raise


def _campaign(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    intent = _load_object(run_root / "campaign-intent.json")
    _verify(intent, "campaign_run_sha256", "coarse production campaign intent")
    bundle = load_production_bundle(
        Path(intent["bundle_root"]), load_units=False, strict_topology=False
    )
    if bundle["manifest"]["manifest_sha256"] != intent["bundle_manifest_sha256"]:
        raise ValueError("coarse production run/bundle binding drift")
    repo_root = Path(__file__).resolve().parents[3]
    for binding in bundle["manifest"]["source_revision"]["files"]:
        path = repo_root / binding["path"]
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise ValueError(
                f"coarse production executing source drift: {binding['path']}"
            )
    for binding in intent["shards"]:
        path = run_root / binding["input_relative_path"]
        if file_sha256(path) != binding["input_sha256"]:
            raise ValueError("coarse production run input drift")
    return intent, bundle


def _metadata(
    intent: Mapping[str, Any], shard_id: str, generation: str
) -> dict[str, str]:
    return {
        "campaign": str(intent["campaign_run_sha256"])[:40],
        "shard": shard_id,
        "generation": generation,
    }


def _upload_provider(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        uploaded = _openai_client().files.create(file=handle, purpose="batch")
    return {
        "schema_version": UPLOAD_SCHEMA,
        "provider": "openai",
        "input_file_id": uploaded.id,
        "purpose": "batch",
    }


def _create_provider(
    input_file_id: str, *, metadata: dict[str, str]
) -> dict[str, Any]:
    batch = _openai_client().batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata,
    )
    if getattr(batch, "input_file_id", None) != input_file_id:
        raise ValueError("coarse production created Batch input file id drift")
    return _provider_batch_dict(batch)


def _validate_upload(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != UPLOAD_SCHEMA
        or value.get("provider") != "openai"
        or value.get("purpose") != "batch"
        or not isinstance(value.get("input_file_id"), str)
        or not value["input_file_id"]
    ):
        raise ValueError("coarse production provider upload snapshot drift")


def _validate_snapshot(
    value: Mapping[str, Any],
    *,
    metadata: Mapping[str, str],
    batch_id: str | None = None,
    input_file_id: str | None = None,
) -> None:
    if (
        value.get("provider") != "openai"
        or value.get("endpoint") != "/v1/responses"
        or value.get("completion_window") != "24h"
        or value.get("metadata") != dict(metadata)
        or not isinstance(value.get("input_file_id"), str)
        or not isinstance(value.get("batch_id"), str)
        or (batch_id is not None and value.get("batch_id") != batch_id)
        or (
            input_file_id is not None
            and value.get("input_file_id") != input_file_id
        )
    ):
        raise ValueError("coarse production provider snapshot drift")


def _attempted_primary_forecast(
    run_root: Path, intent: Mapping[str, Any]
) -> tuple[float, bool]:
    """Return actual-or-reserved primary spend and whether every receipt is priced."""

    total = 0.0
    complete = True
    for shard in intent["shards"]:
        root = run_root / "shards" / shard["shard_id"]
        collection_path = root / "collection.json"
        if collection_path.exists():
            collection = _load_object(collection_path)
            _verify(collection, "collection_sha256", "coarse production collection")
            total += float(collection["known_priced_cost_usd"])
            complete = complete and bool(collection["cost_complete"])
        elif (root / "submission-intent.json").exists():
            total += float(shard["direct_v4_cost_forecast_usd"])
    return total, complete


def _active_primary_queue(
    run_root: Path, intent: Mapping[str, Any]
) -> tuple[list[str], int]:
    active: list[str] = []
    queued_tokens = 0
    terminal = {"completed", "failed", "expired", "cancelled"}
    for shard in intent["shards"]:
        root = run_root / "shards" / shard["shard_id"]
        if (root / "collection.json").exists():
            continue
        state = None
        status_paths = sorted((root / "status").glob("*.json"))
        if status_paths:
            state = _load_object(status_paths[-1])["provider_response"].get("status")
        elif (root / "submission.json").exists():
            state = _load_object(root / "submission.json")["provider_response"].get(
                "status"
            )
        attempted = (root / "submission-intent.json").exists()
        if attempted and state not in terminal:
            active.append(str(shard["shard_id"]))
            queued_tokens += int(shard["queued_input_tokens_empirical_forecast"])
    return active, queued_tokens


def submit_shard(
    *,
    run_root: Path,
    shard_id: str,
    uploader: Callable[[Path], dict[str, Any]] = _upload_provider,
    creator: Callable[..., dict[str, Any]] = _create_provider,
) -> dict[str, Any]:
    intent, bundle = _campaign(run_root)
    shard = next((s for s in bundle["shards"] if s["shard_id"] == shard_id), None)
    if shard is None:
        raise ValueError("unknown coarse production shard")
    shard_root = run_root / "shards" / shard_id
    with _submission_gate(run_root):
        if (shard_root / "submission-intent.json").exists():
            raise FileExistsError(
                "coarse production shard submission was already attempted"
            )
        attempted_cost, cost_complete = _attempted_primary_forecast(run_root, intent)
        if not cost_complete:
            raise ValueError("prior collected campaign cost is not fully priced")
        intent_shard = next(
            item for item in intent["shards"] if item["shard_id"] == shard_id
        )
        candidate_cost = float(intent_shard["direct_v4_cost_forecast_usd"])
        if attempted_cost + candidate_cost > float(
            intent["maximum_authorized_cost_usd"]
        ):
            raise ValueError(
                "prospective actual-or-reserved campaign cost exceeds authorization"
            )
        active, active_queue_tokens = _active_primary_queue(run_root, intent)
        if len(active) >= int(intent["maximum_concurrent_shards"]):
            raise ValueError(f"recorded shard concurrency is already full: {active}")
        this_queue = int(intent_shard["queued_input_tokens_empirical_forecast"])
        if active_queue_tokens + this_queue > int(
            intent["provider_queued_input_token_limit"]
        ):
            raise ValueError("recorded queued-input-token capacity would be exceeded")
        metadata = _metadata(intent, shard_id, "primary")
        input_path = shard_root / "input.jsonl"
        submission_intent = _hashed(
            {
                "schema_version": SUBMISSION_SCHEMA,
                "status": "intent_persisted_before_provider_calls",
                "created_at": _now(),
                "campaign_run_sha256": intent["campaign_run_sha256"],
                "shard_id": shard_id,
                "generation": "primary",
                "input_sha256": file_sha256(input_path),
                "request_count": shard["request_count"],
                "direct_v4_cost_forecast_usd": candidate_cost,
                "prospective_campaign_cost_usd": attempted_cost + candidate_cost,
                "metadata": metadata,
            },
            "submission_intent_sha256",
        )
        atomic_write_json(shard_root / "submission-intent.json", submission_intent)
        try:
            upload = uploader(input_path)
            _validate_upload(upload)
            atomic_write_json(shard_root / "provider-upload-response.json", upload)
            provider = creator(upload["input_file_id"], metadata=metadata)
            atomic_write_json(shard_root / "provider-create-response.json", provider)
            _validate_snapshot(
                provider,
                metadata=metadata,
                input_file_id=upload["input_file_id"],
            )
            receipt = _hashed(
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "status": "submitted",
                    "recorded_at": _now(),
                    "campaign_run_sha256": intent["campaign_run_sha256"],
                    "submission_intent_sha256": submission_intent[
                        "submission_intent_sha256"
                    ],
                    "provider_upload_response_sha256": file_sha256(
                        shard_root / "provider-upload-response.json"
                    ),
                    "provider_response": provider,
                },
                "submission_sha256",
            )
            atomic_write_json(shard_root / "submission.json", receipt)
            return receipt
        except BaseException as error:
            failure = _hashed(
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "status": "failed_closed_indeterminate_provider_state",
                    "recorded_at": _now(),
                    "submission_intent_sha256": submission_intent[
                        "submission_intent_sha256"
                    ],
                    "upload_receipt_persisted": (
                        shard_root / "provider-upload-response.json"
                    ).exists(),
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                    "automatic_retry_permitted": False,
                },
                "submission_failure_sha256",
            )
            atomic_write_json(shard_root / "submission-failure.json", failure)
            raise RuntimeError(
                "provider state is indeterminate; automatic retry is forbidden"
            ) from error


def recover_shard_submission(
    *,
    run_root: Path,
    shard_id: str,
    discoverer: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    intent, _bundle = _campaign(run_root)
    shard_root = run_root / "shards" / shard_id
    if (shard_root / "submission.json").exists():
        raise FileExistsError("coarse production submission receipt already exists")
    submission_intent = _load_object(shard_root / "submission-intent.json")
    _verify(submission_intent, "submission_intent_sha256", "submission intent")
    upload = _load_object(shard_root / "provider-upload-response.json")
    _validate_upload(upload)
    metadata = _metadata(intent, shard_id, "primary")
    create = shard_root / "provider-create-response.json"
    if create.exists():
        provider = _load_object(create)
        recovered_by = "immediate_create_snapshot"
    else:
        if discoverer is None:

            def discoverer(**_: Any) -> list[dict[str, Any]]:
                return [
                    _provider_batch_dict(batch)
                    for batch in _openai_client().batches.list(limit=100)
                    if dict(getattr(batch, "metadata", None) or {}) == metadata
                ]

        matches = discoverer(metadata=metadata)
        if len(matches) != 1:
            raise ValueError(
                f"expected one metadata-matched Batch, found {len(matches)}"
            )
        provider = matches[0]
        atomic_write_json(create, provider)
        recovered_by = "unique_provider_metadata_discovery"
    _validate_snapshot(
        provider, metadata=metadata, input_file_id=upload["input_file_id"]
    )
    receipt = _hashed(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "status": "submitted",
            "recorded_at": _now(),
            "recovered_by": recovered_by,
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "submission_intent_sha256": submission_intent["submission_intent_sha256"],
            "provider_response": provider,
        },
        "submission_sha256",
    )
    atomic_write_json(shard_root / "submission.json", receipt)
    return receipt


def _submission(run_root: Path, shard_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    intent, _bundle = _campaign(run_root)
    value = _load_object(run_root / "shards" / shard_id / "submission.json")
    _verify(value, "submission_sha256", "coarse production submission")
    if value["campaign_run_sha256"] != intent["campaign_run_sha256"]:
        raise ValueError("coarse production submission campaign drift")
    upload_path = run_root / "shards" / shard_id / "provider-upload-response.json"
    upload = _load_object(upload_path)
    _validate_upload(upload)
    if file_sha256(upload_path) != value["provider_upload_response_sha256"]:
        raise ValueError("coarse production submission upload binding drift")
    metadata = _metadata(intent, shard_id, "primary")
    _validate_snapshot(
        value["provider_response"],
        metadata=metadata,
        input_file_id=upload["input_file_id"],
    )
    return intent, value


def check_shard(
    *,
    run_root: Path,
    shard_id: str,
    retriever: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    intent, submission = _submission(run_root, shard_id)
    provider = submission["provider_response"]
    if retriever is None:

        def retriever(batch_id: str) -> dict[str, Any]:
            return _provider_batch_dict(_openai_client().batches.retrieve(batch_id))

    observed = retriever(provider["batch_id"])
    metadata = _metadata(intent, shard_id, "primary")
    _validate_snapshot(
        observed,
        metadata=metadata,
        batch_id=provider["batch_id"],
        input_file_id=provider["input_file_id"],
    )
    status_root = run_root / "shards" / shard_id / "status"
    status_root.mkdir(exist_ok=True)
    prior = sorted(status_root.glob("receipt-*.json"))
    previous = None
    for path in prior:
        row = _load_object(path)
        _verify(row, "status_sha256", "coarse production status")
        if row["previous_status_sha256"] != previous:
            raise ValueError("coarse production status chain drift")
        previous = row["status_sha256"]
    receipt = _hashed(
        {
            "schema_version": STATUS_SCHEMA,
            "recorded_at": _now(),
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "submission_sha256": submission["submission_sha256"],
            "previous_status_sha256": previous,
            "provider_response": observed,
        },
        "status_sha256",
    )
    atomic_write_json(status_root / f"receipt-{len(prior):04d}.json", receipt)
    return receipt


def _download(batch_id: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    batch = _openai_client().batches.retrieve(batch_id)
    snapshot = _provider_batch_dict(batch)
    files = {}
    for source in ("output", "error"):
        file_id = snapshot.get(f"{source}_file_id")
        if file_id:
            files[source] = {
                "file_id": file_id,
                "content": _openai_file_bytes(_openai_client().files.content(file_id)),
            }
    return snapshot, files


def _parse_row(row: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    common = {
        "schema_version": EVENT_SCHEMA,
        "request_id": request["request_id"],
        "shard_id": request["shard_id"],
        "window_id": request["window_id"],
        "window_index": request["window_index"],
        "response_id": request["response_id"],
        "replica_index": request["replica_index"],
        "body_sha256": request["body_sha256"],
        "focal_unit_ids": request["focal_unit_ids"],
        "raw_row_sha256": canonical_sha256(row),
    }
    response = row.get("response")
    if row.get("error") is not None or not isinstance(response, Mapping):
        return {
            **common,
            "validation_status": "provider_error",
            "error_type": "batch_request_error",
            "usage": openai_usage(None).model_dump(mode="json"),
            "decisions": None,
        }
    body = response.get("body")
    if response.get("status_code") != 200 or not isinstance(body, Mapping):
        return {
            **common,
            "validation_status": "provider_error",
            "error_type": "batch_http_error",
            "usage": openai_usage(None).model_dump(mode="json"),
            "decisions": None,
        }
    usage = openai_usage(body.get("usage")).model_dump(mode="json")
    text, refusal, statuses = _response_text(body)
    details = {
        **common,
        "provider_request_id": response.get("request_id") or body.get("id"),
        "model_resolved": body.get("model"),
        "response_status": body.get("status"),
        "stop_reason": openai_stop_reason(body),
        "raw_response_sha256": canonical_sha256(body),
        "raw_text": text or None,
        "usage": usage,
    }
    if refusal is not None:
        return {
            **details,
            "validation_status": "refusal",
            "error_type": "model_refusal",
            "decisions": None,
        }
    if body.get("status") != "completed" or any(
        status != "completed" for status in statuses
    ):
        return {
            **details,
            "validation_status": "incomplete",
            "error_type": "incomplete_response",
            "decisions": None,
        }
    if not isinstance(body.get("model"), str) or not body["model"].startswith(
        "gpt-5.6-luna"
    ):
        return {
            **details,
            "validation_status": "provider_error",
            "error_type": "resolved_model_drift",
            "decisions": None,
        }
    try:
        decisions = validate_decisions(
            json.loads(text), focal_unit_ids=request["focal_unit_ids"]
        )
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return {
            **details,
            "validation_status": "invalid_output",
            "error_type": type(error).__name__,
            "decisions": None,
        }
    return {
        **details,
        "validation_status": "success",
        "error_type": None,
        "decisions": decisions,
    }


def _price_events(
    *,
    events: list[dict[str, Any]],
    snapshot: Mapping[str, Any],
    prices: dict[str, Any],
) -> tuple[float, bool, str]:
    """Price a Batch from aggregate usage, or typed per-row/zero evidence."""

    aggregate_usage = openai_usage(snapshot.get("usage"))
    aggregate_cost, aggregate_long = _estimate_v4_actual_cost(
        prices, model="gpt-5.6-luna", usage=aggregate_usage
    )
    event_total = 0.0
    event_complete = True
    for event in events:
        usage = Usage.model_validate(event["usage"])
        cost, long_context = _estimate_v4_actual_cost(
            prices, model="gpt-5.6-luna", usage=usage
        )
        if cost.total_cost is None and event["validation_status"] in {
            "provider_error",
            "missing",
        }:
            # No provider response/usage exists for this request. The provider Batch
            # receipt is the evidence that it did not produce a billable model result.
            zero = Usage(
                input_tokens=0,
                uncached_input_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
            )
            cost, long_context = _estimate_v4_actual_cost(
                prices, model="gpt-5.6-luna", usage=zero
            )
            event["pricing_basis"] = "no_provider_result_or_usage_priced_zero"
        else:
            event["pricing_basis"] = "request_usage"
        event["cost"] = cost.model_dump(mode="json")
        event["long_context_price_multiplier_applied"] = long_context
        if cost.total_cost is None:
            event_complete = False
        else:
            event_total += float(cost.total_cost)
    if aggregate_cost.complete and aggregate_cost.total_cost is not None:
        total = float(aggregate_cost.total_cost)
        complete = True
        basis = "provider_batch_aggregate_usage"
        for event in events:
            event["collection_pricing_basis"] = basis
            event["batch_aggregate_long_context_price_multiplier_applied"] = (
                aggregate_long
            )
    else:
        total = event_total
        complete = event_complete
        basis = "request_usage_plus_typed_zero_for_no_provider_result"
        for event in events:
            event["collection_pricing_basis"] = basis
    for event in events:
        event["event_sha256"] = canonical_sha256(event)
    return total, complete, basis


def collect_shard(
    *,
    run_root: Path,
    shard_id: str,
    downloader: Callable[
        [str], tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ] = _download,
) -> dict[str, Any]:
    intent, submission = _submission(run_root, shard_id)
    shard_root = run_root / "shards" / shard_id
    if (shard_root / "collection.json").exists():
        raise FileExistsError("coarse production shard already collected")
    provider = submission["provider_response"]
    collection_intent = _hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "status": "intent_persisted",
            "recorded_at": _now(),
            "submission_sha256": submission["submission_sha256"],
            "batch_id": provider["batch_id"],
        },
        "collection_intent_sha256",
    )
    atomic_write_json(shard_root / "collection-intent.json", collection_intent)
    snapshot, files = downloader(provider["batch_id"])
    _validate_snapshot(
        snapshot,
        metadata=_metadata(intent, shard_id, "primary"),
        batch_id=provider["batch_id"],
        input_file_id=provider["input_file_id"],
    )
    terminal_status = snapshot.get("status")
    if terminal_status not in {"completed", "failed", "expired", "cancelled"}:
        raise ValueError("coarse production Batch is not terminal")
    raw_root = shard_root / "raw"
    raw_root.mkdir(exist_ok=True)
    atomic_write_json(raw_root / "provider-snapshot.json", snapshot)
    rows = {}
    raw_bindings = [
        {
            "source": "provider_snapshot",
            "file_id": None,
            "path": str((raw_root / "provider-snapshot.json").relative_to(run_root)),
            "sha256": file_sha256(raw_root / "provider-snapshot.json"),
            "bytes": (raw_root / "provider-snapshot.json").stat().st_size,
        }
    ]
    for source, item in files.items():
        content = item["content"]
        if item["file_id"] != snapshot.get(f"{source}_file_id"):
            raise ValueError("coarse production provider file receipt drift")
        path = raw_root / f"{source}.jsonl"
        atomic_write_bytes(path, content)
        raw_bindings.append(
            {
                "source": source,
                "file_id": item["file_id"],
                "path": str(path.relative_to(run_root)),
                "sha256": file_sha256(path),
                "bytes": len(content),
            }
        )
        for row in read_jsonl(path):
            request_id = row.get("custom_id")
            if not isinstance(request_id, str) or request_id in rows:
                raise ValueError("coarse production duplicate or missing custom_id")
            rows[request_id] = row
    requests = list(iter_shard_requests(Path(intent["bundle_root"]), shard_id))
    expected = {r["request_id"] for r in requests}
    unknown = sorted(set(rows) - expected)
    if unknown:
        raise ValueError("coarse production provider output contains unknown custom_id")
    prices = load_price_snapshot(Path(intent["bundle_root"]) / "price-snapshot.json")
    events = []
    for request in requests:
        if request["request_id"] not in rows:
            event = {
                "schema_version": EVENT_SCHEMA,
                "request_id": request["request_id"],
                "shard_id": shard_id,
                "window_id": request["window_id"],
                "window_index": request["window_index"],
                "response_id": request["response_id"],
                "replica_index": request["replica_index"],
                "body_sha256": request["body_sha256"],
                "focal_unit_ids": request["focal_unit_ids"],
                "validation_status": "missing",
                "error_type": f"terminal_batch_{terminal_status}_without_request_row",
                "usage": openai_usage(None).model_dump(mode="json"),
                "decisions": None,
            }
        else:
            event = _parse_row(rows[request["request_id"]], request)
        events.append(event)
    total, complete_cost, pricing_basis = _price_events(
        events=events, snapshot=snapshot, prices=prices
    )
    atomic_write_jsonl(shard_root / "events.jsonl", events)
    success = sum(e["validation_status"] == "success" for e in events)
    prior_cost = 0.0
    for other in intent["shards"]:
        path = run_root / "shards" / other["shard_id"] / "collection.json"
        if path.exists():
            prior = _load_object(path)
            _verify(prior, "collection_sha256", "coarse production collection")
            prior_cost += float(prior["known_priced_cost_usd"])
    cumulative_cost = prior_cost + total
    authorization_exceeded = complete_cost and cumulative_cost > float(
        intent["maximum_authorized_cost_usd"]
    )
    result = _hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "status": (
                "failed_closed_authorization_exceeded"
                if authorization_exceeded
                else (
                    "complete"
                    if success == len(events) and complete_cost
                    else "complete_with_failed_requests_recovery_eligible"
                )
            ),
            "completed_at": _now(),
            "collection_intent_sha256": collection_intent["collection_intent_sha256"],
            "request_count": len(events),
            "success_count": success,
            "failure_count": len(events) - success,
            "known_priced_cost_usd": total,
            "cumulative_known_priced_cost_usd": cumulative_cost,
            "cost_complete": complete_cost,
            "pricing_basis": pricing_basis,
            "provider_terminal_status": terminal_status,
            "maximum_authorized_cost_usd": intent["maximum_authorized_cost_usd"],
            "authorization_exceeded": authorization_exceeded,
            "raw_file_bindings": raw_bindings,
            "events_sha256": file_sha256(shard_root / "events.jsonl"),
        },
        "collection_sha256",
    )
    atomic_write_json(shard_root / "collection.json", result)
    return result


def prepare_failed_only_recovery(*, run_root: Path) -> dict[str, Any]:
    """Freeze one failed-only recovery wave, partitioned by primary shard."""

    intent, bundle = _campaign(run_root)
    recovery_root = run_root / "recovery-000"
    if recovery_root.exists():
        raise FileExistsError("coarse production recovery wave already exists")
    failed = []
    successful = set()
    for shard in bundle["shards"]:
        shard_root = run_root / "shards" / shard["shard_id"]
        events_path = shard_root / "events.jsonl"
        collection_path = shard_root / "collection.json"
        if not events_path.exists() or not collection_path.exists():
            raise ValueError(
                "all primary shards must be collected before recovery freeze"
            )
        collection = _load_object(collection_path)
        _verify(collection, "collection_sha256", "coarse production collection")
        if (
            file_sha256(events_path) != collection["events_sha256"]
            or not collection["cost_complete"]
            or collection["authorization_exceeded"]
        ):
            raise ValueError("primary collection is not recovery-eligible")
        for event in read_jsonl(events_path):
            if event["validation_status"] == "success":
                successful.add(event["request_id"])
            else:
                failed.append(event["request_id"])
    failed_set = set(failed)
    if not failed_set or failed_set & successful:
        raise ValueError(
            "recovery requires failed-only non-overlapping request identities"
        )
    source_lines: dict[str, tuple[str, dict[str, Any]]] = {}
    for shard in bundle["shards"]:
        for line in read_jsonl(Path(intent["bundle_root"]) / shard["path"]):
            if line["custom_id"] in failed_set:
                source_lines[line["custom_id"]] = (shard["shard_id"], line)
    if set(source_lines) != failed_set:
        raise ValueError("recovery failed request body coverage drift")
    recovery_root.mkdir()
    recovery_shards = []
    for primary in bundle["shards"]:
        ordered = [
            source_lines[request_id][1]
            for request_id in primary["request_ids_in_order"]
            if request_id in failed_set
        ]
        if not ordered:
            continue
        recovery_shard_id = primary["shard_id"]
        shard_root = recovery_root / "shards" / recovery_shard_id
        shard_root.mkdir(parents=True)
        atomic_write_jsonl(shard_root / "input.jsonl", ordered)
        input_path = shard_root / "input.jsonl"
        if input_path.stat().st_size >= 180_000_000:
            raise ValueError("coarse production recovery shard violates byte guard")
        provider_body_bytes = sum(
            len(
                json.dumps(
                    row["body"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            )
            for row in ordered
        )
        empirical = bundle["config"]["empirical_calibration"]
        recovery_shards.append(
            {
                "shard_id": recovery_shard_id,
                "input_relative_path": str(input_path.relative_to(run_root)),
                "input_sha256": file_sha256(input_path),
                "bytes": input_path.stat().st_size,
                "request_count": len(ordered),
                "direct_v4_cost_forecast_usd": (
                    float(empirical["source_actual_cost_usd"])
                    * len(ordered)
                    / int(empirical["source_request_count"])
                ),
                "queued_input_tokens_empirical_forecast": round(
                    provider_body_bytes
                    * float(empirical["source_input_tokens"])
                    / float(empirical["source_provider_body_utf8_bytes"])
                ),
                "request_ids_in_order": [row["custom_id"] for row in ordered],
            }
        )
    manifest = _hashed(
        {
            "schema_version": "adag.process-witness.coarse-production-recovery.v1",
            "status": "prepared_offline_failed_only",
            "created_at": _now(),
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "recovery_wave": 0,
            "request_count": len(failed_set),
            "shard_count": len(recovery_shards),
            "shards": recovery_shards,
            "successful_requests_rerun": 0,
            "provider_bodies_byte_identical": True,
            "additional_recovery_waves_permitted": False,
        },
        "recovery_manifest_sha256",
    )
    atomic_write_json(recovery_root / "manifest.json", manifest)
    return manifest


def _recovery_binding(
    run_root: Path, shard_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    intent, bundle = _campaign(run_root)
    manifest = _load_object(run_root / "recovery-000" / "manifest.json")
    _verify(manifest, "recovery_manifest_sha256", "coarse production recovery")
    if manifest["campaign_run_sha256"] != intent["campaign_run_sha256"]:
        raise ValueError("coarse production recovery campaign drift")
    binding = next((s for s in manifest["shards"] if s["shard_id"] == shard_id), None)
    if binding is None:
        raise ValueError("unknown coarse production recovery shard")
    shard_root = run_root / "recovery-000" / "shards" / shard_id
    if file_sha256(shard_root / "input.jsonl") != binding["input_sha256"]:
        raise ValueError("coarse production recovery input drift")
    return intent, bundle, binding, shard_root


def submit_recovery_shard(
    *,
    run_root: Path,
    shard_id: str,
    uploader: Callable[[Path], dict[str, Any]] = _upload_provider,
    creator: Callable[..., dict[str, Any]] = _create_provider,
) -> dict[str, Any]:
    intent, _bundle, binding, shard_root = _recovery_binding(run_root, shard_id)
    with _submission_gate(run_root):
        if (shard_root / "submission-intent.json").exists():
            raise FileExistsError(
                "coarse production recovery submission was already attempted"
            )
        primary_cost, primary_complete = _attempted_primary_forecast(run_root, intent)
        if not primary_complete:
            raise ValueError("primary campaign cost is not fully priced")
        recovery_manifest = _load_object(run_root / "recovery-000" / "manifest.json")
        recovery_cost = 0.0
        active: list[str] = []
        active_queue = 0
        terminal = {"completed", "failed", "expired", "cancelled"}
        for other in recovery_manifest["shards"]:
            other_root = run_root / "recovery-000" / "shards" / other["shard_id"]
            collection_path = other_root / "collection.json"
            if collection_path.exists():
                collection = _load_object(collection_path)
                _verify(collection, "collection_sha256", "recovery collection")
                if not collection["cost_complete"]:
                    raise ValueError("prior recovery cost is not fully priced")
                recovery_cost += float(collection["known_priced_cost_usd"])
                continue
            if (other_root / "submission-intent.json").exists():
                recovery_cost += float(other["direct_v4_cost_forecast_usd"])
                state = None
                statuses = sorted((other_root / "status").glob("*.json"))
                if statuses:
                    state = _load_object(statuses[-1])["provider_response"].get(
                        "status"
                    )
                elif (other_root / "submission.json").exists():
                    state = _load_object(other_root / "submission.json")[
                        "provider_response"
                    ].get("status")
                if state not in terminal:
                    active.append(other["shard_id"])
                    active_queue += int(
                        other["queued_input_tokens_empirical_forecast"]
                    )
        candidate_cost = float(binding["direct_v4_cost_forecast_usd"])
        if primary_cost + recovery_cost + candidate_cost > float(
            intent["maximum_authorized_cost_usd"]
        ):
            raise ValueError(
                "prospective actual-or-reserved recovery cost exceeds authorization"
            )
        if len(active) >= int(intent["maximum_concurrent_shards"]):
            raise ValueError(f"recorded recovery concurrency is already full: {active}")
        candidate_queue = int(binding["queued_input_tokens_empirical_forecast"])
        if active_queue + candidate_queue > int(
            intent["provider_queued_input_token_limit"]
        ):
            raise ValueError("recovery queued-input-token capacity would be exceeded")
        metadata = _metadata(intent, shard_id, "recovery-000")
        input_path = shard_root / "input.jsonl"
        submission_intent = _hashed(
            {
                "schema_version": SUBMISSION_SCHEMA,
                "status": "intent_persisted_before_provider_calls",
                "created_at": _now(),
                "campaign_run_sha256": intent["campaign_run_sha256"],
                "shard_id": shard_id,
                "generation": "recovery-000",
                "input_sha256": binding["input_sha256"],
                "request_count": binding["request_count"],
                "direct_v4_cost_forecast_usd": candidate_cost,
                "prospective_campaign_cost_usd": (
                    primary_cost + recovery_cost + candidate_cost
                ),
                "metadata": metadata,
            },
            "submission_intent_sha256",
        )
        atomic_write_json(shard_root / "submission-intent.json", submission_intent)
        try:
            upload = uploader(input_path)
            _validate_upload(upload)
            atomic_write_json(shard_root / "provider-upload-response.json", upload)
            provider = creator(upload["input_file_id"], metadata=metadata)
            atomic_write_json(shard_root / "provider-create-response.json", provider)
            _validate_snapshot(
                provider,
                metadata=metadata,
                input_file_id=upload["input_file_id"],
            )
            receipt = _hashed(
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "status": "submitted",
                    "recorded_at": _now(),
                    "campaign_run_sha256": intent["campaign_run_sha256"],
                    "submission_intent_sha256": submission_intent[
                        "submission_intent_sha256"
                    ],
                    "provider_upload_response_sha256": file_sha256(
                        shard_root / "provider-upload-response.json"
                    ),
                    "provider_response": provider,
                },
                "submission_sha256",
            )
            atomic_write_json(shard_root / "submission.json", receipt)
            return receipt
        except BaseException as error:
            failure = _hashed(
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "status": "failed_closed_indeterminate_provider_state",
                    "recorded_at": _now(),
                    "submission_intent_sha256": submission_intent[
                        "submission_intent_sha256"
                    ],
                    "upload_receipt_persisted": (
                        shard_root / "provider-upload-response.json"
                    ).exists(),
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                    "automatic_retry_permitted": False,
                },
                "submission_failure_sha256",
            )
            atomic_write_json(shard_root / "submission-failure.json", failure)
            raise RuntimeError(
                "recovery provider state is indeterminate; automatic retry is forbidden"
            ) from error


def recover_recovery_submission(
    *,
    run_root: Path,
    shard_id: str,
    discoverer: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    intent, _bundle, _binding, shard_root = _recovery_binding(run_root, shard_id)
    if (shard_root / "submission.json").exists():
        raise FileExistsError("coarse production recovery receipt already exists")
    submission_intent = _load_object(shard_root / "submission-intent.json")
    _verify(submission_intent, "submission_intent_sha256", "recovery submission intent")
    upload = _load_object(shard_root / "provider-upload-response.json")
    _validate_upload(upload)
    metadata = _metadata(intent, shard_id, "recovery-000")
    create = shard_root / "provider-create-response.json"
    if create.exists():
        provider = _load_object(create)
        recovered_by = "immediate_create_snapshot"
    else:
        if discoverer is None:

            def discoverer(**_: Any) -> list[dict[str, Any]]:
                return [
                    _provider_batch_dict(batch)
                    for batch in _openai_client().batches.list(limit=100)
                    if dict(getattr(batch, "metadata", None) or {}) == metadata
                ]

        matches = discoverer(metadata=metadata)
        if len(matches) != 1:
            raise ValueError(
                f"expected one metadata-matched recovery Batch, found {len(matches)}"
            )
        provider = matches[0]
        atomic_write_json(create, provider)
        recovered_by = "unique_provider_metadata_discovery"
    _validate_snapshot(
        provider, metadata=metadata, input_file_id=upload["input_file_id"]
    )
    receipt = _hashed(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "status": "submitted",
            "recorded_at": _now(),
            "recovered_by": recovered_by,
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "submission_intent_sha256": submission_intent["submission_intent_sha256"],
            "provider_response": provider,
        },
        "submission_sha256",
    )
    atomic_write_json(shard_root / "submission.json", receipt)
    return receipt


def check_recovery_shard(
    *,
    run_root: Path,
    shard_id: str,
    retriever: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    intent, _bundle, _binding, shard_root = _recovery_binding(run_root, shard_id)
    submission = _load_object(shard_root / "submission.json")
    _verify(submission, "submission_sha256", "recovery submission")
    upload_path = shard_root / "provider-upload-response.json"
    upload = _load_object(upload_path)
    _validate_upload(upload)
    if file_sha256(upload_path) != submission["provider_upload_response_sha256"]:
        raise ValueError("recovery submission upload binding drift")
    provider = submission["provider_response"]
    if retriever is None:

        def retriever(batch_id: str) -> dict[str, Any]:
            return _provider_batch_dict(_openai_client().batches.retrieve(batch_id))

    observed = retriever(provider["batch_id"])
    _validate_snapshot(
        observed,
        metadata=_metadata(intent, shard_id, "recovery-000"),
        batch_id=provider["batch_id"],
        input_file_id=upload["input_file_id"],
    )
    status_root = shard_root / "status"
    status_root.mkdir(exist_ok=True)
    prior = sorted(status_root.glob("receipt-*.json"))
    previous = None
    for path in prior:
        row = _load_object(path)
        _verify(row, "status_sha256", "recovery status")
        if row["previous_status_sha256"] != previous:
            raise ValueError("coarse production recovery status chain drift")
        previous = row["status_sha256"]
    receipt = _hashed(
        {
            "schema_version": STATUS_SCHEMA,
            "recorded_at": _now(),
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "submission_sha256": submission["submission_sha256"],
            "previous_status_sha256": previous,
            "provider_response": observed,
        },
        "status_sha256",
    )
    atomic_write_json(status_root / f"receipt-{len(prior):04d}.json", receipt)
    return receipt


def collect_recovery_shard(
    *,
    run_root: Path,
    shard_id: str,
    downloader: Callable[
        [str], tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ] = _download,
) -> dict[str, Any]:
    intent, _bundle, binding, shard_root = _recovery_binding(run_root, shard_id)
    if (shard_root / "collection.json").exists():
        raise FileExistsError("coarse production recovery shard already collected")
    submission = _load_object(shard_root / "submission.json")
    _verify(submission, "submission_sha256", "recovery submission")
    upload_path = shard_root / "provider-upload-response.json"
    upload = _load_object(upload_path)
    _validate_upload(upload)
    if file_sha256(upload_path) != submission["provider_upload_response_sha256"]:
        raise ValueError("recovery submission upload binding drift")
    provider = submission["provider_response"]
    collection_intent = _hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "status": "intent_persisted",
            "recorded_at": _now(),
            "submission_sha256": submission["submission_sha256"],
            "batch_id": provider["batch_id"],
        },
        "collection_intent_sha256",
    )
    atomic_write_json(shard_root / "collection-intent.json", collection_intent)
    snapshot, files = downloader(provider["batch_id"])
    _validate_snapshot(
        snapshot,
        metadata=_metadata(intent, shard_id, "recovery-000"),
        batch_id=provider["batch_id"],
        input_file_id=upload["input_file_id"],
    )
    terminal_status = snapshot.get("status")
    if terminal_status not in {"completed", "failed", "expired", "cancelled"}:
        raise ValueError("coarse production recovery Batch is not terminal")
    raw_root = shard_root / "raw"
    raw_root.mkdir(exist_ok=True)
    atomic_write_json(raw_root / "provider-snapshot.json", snapshot)
    rows = {}
    raw_bindings = [
        {
            "source": "provider_snapshot",
            "file_id": None,
            "path": str((raw_root / "provider-snapshot.json").relative_to(run_root)),
            "sha256": file_sha256(raw_root / "provider-snapshot.json"),
            "bytes": (raw_root / "provider-snapshot.json").stat().st_size,
        }
    ]
    for source, item in files.items():
        if item["file_id"] != snapshot.get(f"{source}_file_id"):
            raise ValueError("coarse production recovery file receipt drift")
        path = raw_root / f"{source}.jsonl"
        atomic_write_bytes(path, item["content"])
        raw_bindings.append(
            {
                "source": source,
                "file_id": item["file_id"],
                "path": str(path.relative_to(run_root)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
        for row in read_jsonl(path):
            request_id = row.get("custom_id")
            if not isinstance(request_id, str) or request_id in rows:
                raise ValueError("coarse production recovery duplicate custom_id")
            rows[request_id] = row
    expected_ids = set(binding["request_ids_in_order"])
    if set(rows) - expected_ids:
        raise ValueError("coarse production recovery contains unknown custom_id")
    primary_requests = {
        request["request_id"]: request
        for request in iter_shard_requests(Path(intent["bundle_root"]), shard_id)
        if request["request_id"] in expected_ids
    }
    prices = load_price_snapshot(Path(intent["bundle_root"]) / "price-snapshot.json")
    events = []
    for request_id in binding["request_ids_in_order"]:
        request = primary_requests[request_id]
        if request_id in rows:
            event = _parse_row(rows[request_id], request)
        else:
            event = {
                "schema_version": EVENT_SCHEMA,
                "request_id": request_id,
                "shard_id": shard_id,
                "window_id": request["window_id"],
                "window_index": request["window_index"],
                "response_id": request["response_id"],
                "replica_index": request["replica_index"],
                "body_sha256": request["body_sha256"],
                "focal_unit_ids": request["focal_unit_ids"],
                "validation_status": "missing",
                "error_type": f"terminal_batch_{terminal_status}_without_request_row",
                "usage": openai_usage(None).model_dump(mode="json"),
                "decisions": None,
            }
        event["generation"] = "recovery-000"
        events.append(event)
    total, complete_cost, pricing_basis = _price_events(
        events=events, snapshot=snapshot, prices=prices
    )
    atomic_write_jsonl(shard_root / "events.jsonl", events)
    success = sum(e["validation_status"] == "success" for e in events)
    campaign_cost = (
        total
        + sum(
            float(_load_object(path)["known_priced_cost_usd"])
            for path in run_root.glob("shards/*/collection.json")
        )
        + sum(
            float(_load_object(path)["known_priced_cost_usd"])
            for path in run_root.glob("recovery-000/shards/*/collection.json")
            if path.parent.name != shard_id
        )
    )
    authorization_exceeded = complete_cost and campaign_cost > float(
        intent["maximum_authorized_cost_usd"]
    )
    result = _hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "status": (
                "failed_closed_authorization_exceeded"
                if authorization_exceeded
                else (
                    "complete"
                    if success == len(events) and complete_cost
                    else "failed_closed_recovery_exhausted"
                )
            ),
            "completed_at": _now(),
            "collection_intent_sha256": collection_intent["collection_intent_sha256"],
            "request_count": len(events),
            "success_count": success,
            "failure_count": len(events) - success,
            "known_priced_cost_usd": total,
            "cumulative_known_priced_cost_usd": campaign_cost,
            "cost_complete": complete_cost,
            "pricing_basis": pricing_basis,
            "provider_terminal_status": terminal_status,
            "authorization_exceeded": authorization_exceeded,
            "raw_file_bindings": raw_bindings,
            "events_sha256": file_sha256(shard_root / "events.jsonl"),
        },
        "collection_sha256",
    )
    atomic_write_json(shard_root / "collection.json", result)
    return result


def _copy_campaign_evidence(
    *,
    run_root: Path,
    temporary: Path,
    intent: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> None:
    """Make final evidence independent of the mutable run and source bundle roots."""

    copied_bundle = temporary / "campaign-bundle"
    shutil.copytree(Path(intent["bundle_root"]), copied_bundle)
    shutil.copyfile(run_root / "campaign-intent.json", temporary / "campaign-intent.json")
    for shard in bundle["shards"]:
        source = run_root / "shards" / shard["shard_id"]
        target = temporary / "shards" / shard["shard_id"]
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns("input.jsonl"),
        )
        target_input = target / "input.jsonl"
        copied_input = copied_bundle / shard["path"]
        target_input.symlink_to(os.path.relpath(copied_input, target))
    recovery = run_root / "recovery-000"
    if recovery.exists():
        shutil.copytree(recovery, temporary / "recovery-000")


def _write_evidence_inventory(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in {
            root / "manifest.json",
            root / "evidence-inventory.json",
        }:
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("coarse production final evidence has external symlink") from error
        row = {
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        if path.is_symlink():
            row["symlink_target"] = os.readlink(path)
        files.append(row)
    inventory = _hashed(
        {
            "schema_version": "adag.process-witness.coarse-production-evidence-inventory.v1",
            "files": files,
        },
        "evidence_inventory_sha256",
    )
    atomic_write_json(root / "evidence-inventory.json", inventory)
    return inventory


def load_frozen_proposal_bank(root: Path) -> dict[str, Any]:
    """Strictly validate a finalized proposal bank without its original run roots."""

    manifest = _load_object(root / "manifest.json")
    _verify(manifest, "proposal_bank_manifest_sha256", "proposal bank manifest")
    inventory = _load_object(root / "evidence-inventory.json")
    _verify(inventory, "evidence_inventory_sha256", "proposal evidence inventory")
    if (
        inventory["evidence_inventory_sha256"]
        != manifest["evidence_inventory_sha256"]
    ):
        raise ValueError("proposal evidence inventory/manifest drift")
    expected = {row["path"]: row for row in inventory["files"]}
    observed = {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file()
        and path
        not in {root / "manifest.json", root / "evidence-inventory.json"}
    }
    if set(expected) != set(observed):
        raise ValueError("proposal evidence inventory coverage drift")
    for relative, path in observed.items():
        row = expected[relative]
        if file_sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"proposal evidence file drift: {relative}")
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("proposal evidence external symlink drift") from error
    bundle = load_production_bundle(root / "campaign-bundle", load_units=True)
    if bundle["manifest"]["manifest_sha256"] != manifest["bundle_manifest_sha256"]:
        raise ValueError("proposal copied bundle binding drift")
    intent = _load_object(root / "campaign-intent.json")
    _verify(intent, "campaign_run_sha256", "copied campaign intent")
    if intent["campaign_run_sha256"] != manifest["campaign_run_sha256"]:
        raise ValueError("proposal copied campaign binding drift")
    for binding in manifest["collection_bindings"]:
        attempt = (
            root / "shards" / binding["shard_id"]
            if binding["generation"] == "primary"
            else root / "recovery-000" / "shards" / binding["shard_id"]
        )
        for name in (
            "input.jsonl",
            "submission-intent.json",
            "provider-upload-response.json",
            "provider-create-response.json",
            "submission.json",
            "collection-intent.json",
            "collection.json",
            "events.jsonl",
            "raw/provider-snapshot.json",
        ):
            if not (attempt / name).is_file():
                raise ValueError(f"proposal evidence misses {binding['generation']}/{binding['shard_id']}/{name}")
        collection = _load_object(attempt / "collection.json")
        _verify(collection, "collection_sha256", "copied collection")
        submission_intent = _load_object(attempt / "submission-intent.json")
        _verify(
            submission_intent,
            "submission_intent_sha256",
            "copied submission intent",
        )
        if file_sha256(attempt / "input.jsonl") != submission_intent["input_sha256"]:
            raise ValueError("proposal copied provider input binding drift")
        upload_path = attempt / "provider-upload-response.json"
        upload = _load_object(upload_path)
        _validate_upload(upload)
        submission = _load_object(attempt / "submission.json")
        _verify(submission, "submission_sha256", "copied submission")
        if (
            file_sha256(upload_path)
            != submission["provider_upload_response_sha256"]
        ):
            raise ValueError("proposal copied upload receipt binding drift")
        provider = submission["provider_response"]
        _validate_snapshot(
            provider,
            metadata=provider["metadata"],
            input_file_id=upload["input_file_id"],
        )
        if _load_object(attempt / "provider-create-response.json") != provider:
            raise ValueError("proposal copied create receipt binding drift")
        collection_intent = _load_object(attempt / "collection-intent.json")
        _verify(
            collection_intent,
            "collection_intent_sha256",
            "copied collection intent",
        )
        if (
            collection_intent["collection_intent_sha256"]
            != collection["collection_intent_sha256"]
        ):
            raise ValueError("proposal copied collection intent binding drift")
        if (
            collection["collection_sha256"] != binding["collection_sha256"]
            or file_sha256(attempt / "events.jsonl") != binding["events_sha256"]
        ):
            raise ValueError("proposal copied collection binding drift")
        for raw in collection["raw_file_bindings"]:
            raw_path = root / raw["path"]
            if (
                not raw_path.is_file()
                or file_sha256(raw_path) != raw["sha256"]
                or raw_path.stat().st_size != raw["bytes"]
            ):
                raise ValueError("proposal copied raw provider evidence drift")
        raw_snapshot = _load_object(attempt / "raw/provider-snapshot.json")
        _validate_snapshot(
            raw_snapshot,
            metadata=provider["metadata"],
            batch_id=provider["batch_id"],
            input_file_id=upload["input_file_id"],
        )
        prior = None
        for status_path in sorted((attempt / "status").glob("*.json")):
            status = _load_object(status_path)
            _verify(status, "status_sha256", "copied provider status")
            if status["previous_status_sha256"] != prior:
                raise ValueError("proposal copied provider status chain drift")
            prior = status["status_sha256"]
    for filename, field in (
        ("effective-events.jsonl", "effective_events_sha256"),
        ("proposals.jsonl", "proposals_sha256"),
        ("sampling-groups.jsonl", "sampling_groups_sha256"),
    ):
        if file_sha256(root / filename) != manifest[field]:
            raise ValueError(f"proposal final output drift: {filename}")
    return {"manifest": manifest, "inventory": inventory, "bundle": bundle}


def finalize_campaign(*, run_root: Path, destination: Path) -> dict[str, Any]:
    """Union complete primary results and freeze atom proposals plus sampling groups."""

    intent, bundle = _campaign(run_root)
    if destination.exists():
        raise FileExistsError(
            f"coarse production proposal destination exists: {destination}"
        )
    primary_events = []
    collection_bindings = []
    total_cost = 0.0
    cost_complete = True
    for shard in bundle["shards"]:
        shard_root = run_root / "shards" / shard["shard_id"]
        collection = _load_object(shard_root / "collection.json")
        _verify(collection, "collection_sha256", "coarse production collection")
        shard_events = read_jsonl(shard_root / "events.jsonl")
        if file_sha256(shard_root / "events.jsonl") != collection["events_sha256"]:
            raise ValueError("coarse production primary event binding drift")
        primary_events.extend(shard_events)
        total_cost += float(collection["known_priced_cost_usd"])
        cost_complete = cost_complete and bool(collection["cost_complete"])
        collection_bindings.append(
            {
                "generation": "primary",
                "shard_id": shard["shard_id"],
                "collection_sha256": collection["collection_sha256"],
                "events_sha256": collection["events_sha256"],
            }
        )
    request_ids = [row["request_id"] for row in bundle["request_index"]]
    if len(primary_events) != len(request_ids) or {
        e["request_id"] for e in primary_events
    } != set(request_ids):
        raise ValueError("coarse production campaign union request coverage drift")
    events_by_id = {event["request_id"]: event for event in primary_events}
    recovery_root = run_root / "recovery-000"
    if recovery_root.exists():
        recovery = _load_object(recovery_root / "manifest.json")
        _verify(recovery, "recovery_manifest_sha256", "coarse production recovery")
        for shard in recovery["shards"]:
            shard_root = recovery_root / "shards" / shard["shard_id"]
            collection = _load_object(shard_root / "collection.json")
            _verify(collection, "collection_sha256", "recovery collection")
            recovery_events = read_jsonl(shard_root / "events.jsonl")
            if file_sha256(shard_root / "events.jsonl") != collection["events_sha256"]:
                raise ValueError("coarse production recovery event binding drift")
            for event in recovery_events:
                primary = events_by_id[event["request_id"]]
                if primary["validation_status"] == "success":
                    raise ValueError(
                        "coarse production recovery reran a successful request"
                    )
                events_by_id[event["request_id"]] = event
            total_cost += float(collection["known_priced_cost_usd"])
            cost_complete = cost_complete and bool(collection["cost_complete"])
            collection_bindings.append(
                {
                    "generation": "recovery-000",
                    "shard_id": shard["shard_id"],
                    "collection_sha256": collection["collection_sha256"],
                    "events_sha256": collection["events_sha256"],
                }
            )
    events = [events_by_id[request_id] for request_id in request_ids]
    if any(event["validation_status"] != "success" for event in events):
        raise ValueError(
            "coarse production finalization requires recovery-resolved success coverage"
        )
    if not cost_complete or total_cost > float(intent["maximum_authorized_cost_usd"]):
        raise ValueError(
            "coarse production finalization cost is incomplete or unauthorized"
        )
    votes_by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        for decision in event["decisions"]:
            votes_by_unit[decision["unit_id"]].append(
                {
                    "request_id": event["request_id"],
                    "replica_index": event["replica_index"],
                    **decision,
                }
            )
    units = load_production_bundle(Path(intent["bundle_root"]), load_units=True)[
        "units"
    ]
    proposals = [
        proposal_from_votes(unit, votes_by_unit.get(unit["unit_id"], []))
        for unit in units
    ]
    groups = sampling_groups(units, proposals)
    temporary = destination.parent / f".{destination.name}.finalizing-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(
            f"coarse production temporary destination exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    atomic_write_jsonl(temporary / "effective-events.jsonl", events)
    atomic_write_jsonl(temporary / "proposals.jsonl", proposals)
    atomic_write_jsonl(temporary / "sampling-groups.jsonl", groups)
    _copy_campaign_evidence(
        run_root=run_root,
        temporary=temporary,
        intent=intent,
        bundle=bundle,
    )
    inventory = _write_evidence_inventory(temporary)
    result = _hashed(
        {
            "schema_version": "adag.process-witness.coarse-proposal-bank.v1",
            "status": "frozen_sampling_proposals_not_semantic_truth",
            "created_at": _now(),
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "bundle_manifest_sha256": bundle["manifest"]["manifest_sha256"],
            "proposal_count": len(proposals),
            "sampling_group_count": len(groups),
            "provider_pending_atoms_with_three_votes": len(votes_by_unit),
            "actual_total_cost_usd": total_cost,
            "maximum_authorized_cost_usd": intent["maximum_authorized_cost_usd"],
            "effective_events_sha256": file_sha256(
                temporary / "effective-events.jsonl"
            ),
            "proposals_sha256": file_sha256(temporary / "proposals.jsonl"),
            "sampling_groups_sha256": file_sha256(temporary / "sampling-groups.jsonl"),
            "evidence_inventory_sha256": inventory[
                "evidence_inventory_sha256"
            ],
            "collection_bindings": collection_bindings,
            "claim_boundary": bundle["config"]["claim_boundary"],
        },
        "proposal_bank_manifest_sha256",
    )
    atomic_write_json(temporary / "manifest.json", result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.rename(destination)
    _readonly_tree(destination)
    load_frozen_proposal_bank(destination)
    return result
