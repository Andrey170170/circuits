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


def _submit_provider(path: Path, *, metadata: dict[str, str]) -> dict[str, Any]:
    with path.open("rb") as handle:
        uploaded = _openai_client().files.create(file=handle, purpose="batch")
    batch = _openai_client().batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata,
    )
    if getattr(batch, "input_file_id", None) != uploaded.id:
        raise ValueError("coarse production created Batch input file id drift")
    return _provider_batch_dict(batch)


def _validate_snapshot(
    value: Mapping[str, Any],
    *,
    metadata: Mapping[str, str],
    batch_id: str | None = None,
) -> None:
    if (
        value.get("provider") != "openai"
        or value.get("endpoint") != "/v1/responses"
        or value.get("completion_window") != "24h"
        or value.get("metadata") != dict(metadata)
        or not isinstance(value.get("input_file_id"), str)
        or not isinstance(value.get("batch_id"), str)
        or (batch_id is not None and value.get("batch_id") != batch_id)
    ):
        raise ValueError("coarse production provider snapshot drift")


def submit_shard(
    *,
    run_root: Path,
    shard_id: str,
    submitter: Callable[..., dict[str, Any]] = _submit_provider,
) -> dict[str, Any]:
    intent, bundle = _campaign(run_root)
    shard = next((s for s in bundle["shards"] if s["shard_id"] == shard_id), None)
    if shard is None:
        raise ValueError("unknown coarse production shard")
    shard_root = run_root / "shards" / shard_id
    if (shard_root / "submission-intent.json").exists():
        raise FileExistsError(
            "coarse production shard submission was already attempted"
        )
    collected_cost = 0.0
    active = []
    active_queue_tokens = 0
    for other in intent["shards"]:
        other_root = run_root / "shards" / other["shard_id"]
        collection_path = other_root / "collection.json"
        if collection_path.exists():
            collection = _load_object(collection_path)
            _verify(collection, "collection_sha256", "coarse production collection")
            collected_cost += float(collection["known_priced_cost_usd"])
        status_paths = sorted((other_root / "status").glob("*.json"))
        state = None
        if status_paths:
            state = _load_object(status_paths[-1])["provider_response"].get("status")
        elif (other_root / "submission.json").exists() and not collection_path.exists():
            state = _load_object(other_root / "submission.json")[
                "provider_response"
            ].get("status")
        if state is not None and state in {
            "validating",
            "in_progress",
            "finalizing",
            "cancelling",
        }:
            active.append(other["shard_id"])
            active_queue_tokens += int(other["queued_input_tokens_empirical_forecast"])
    if collected_cost >= float(intent["maximum_authorized_cost_usd"]):
        raise ValueError("collected campaign cost has exhausted the authorization")
    if len(active) >= int(intent["maximum_concurrent_shards"]):
        raise ValueError(f"recorded shard concurrency is already full: {active}")
    this_forecast = next(
        int(item["queued_input_tokens_empirical_forecast"])
        for item in intent["shards"]
        if item["shard_id"] == shard_id
    )
    if active_queue_tokens + this_forecast > int(
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
            "metadata": metadata,
        },
        "submission_intent_sha256",
    )
    atomic_write_json(shard_root / "submission-intent.json", submission_intent)
    try:
        provider = submitter(input_path, metadata=metadata)
        atomic_write_json(shard_root / "provider-create-response.json", provider)
        _validate_snapshot(provider, metadata=metadata)
        receipt = _hashed(
            {
                "schema_version": SUBMISSION_SCHEMA,
                "status": "submitted",
                "recorded_at": _now(),
                "campaign_run_sha256": intent["campaign_run_sha256"],
                "submission_intent_sha256": submission_intent[
                    "submission_intent_sha256"
                ],
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
    _validate_snapshot(provider, metadata=metadata)
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
    metadata = _metadata(intent, shard_id, "primary")
    _validate_snapshot(value["provider_response"], metadata=metadata)
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
    _validate_snapshot(observed, metadata=metadata, batch_id=provider["batch_id"])
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
    )
    if snapshot.get("status") != "completed":
        raise ValueError("coarse production Batch is not completed")
    raw_root = shard_root / "raw"
    raw_root.mkdir(exist_ok=True)
    atomic_write_json(raw_root / "provider-snapshot.json", snapshot)
    rows = {}
    raw_bindings = []
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
    total = 0.0
    complete_cost = True
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
                "error_type": "missing_provider_row",
                "usage": openai_usage(None).model_dump(mode="json"),
                "decisions": None,
            }
        else:
            event = _parse_row(rows[request["request_id"]], request)
        usage = Usage.model_validate(event["usage"])
        cost, long_context = _estimate_v4_actual_cost(
            prices, model="gpt-5.6-luna", usage=usage
        )
        event["cost"] = cost.model_dump(mode="json")
        event["long_context_price_multiplier_applied"] = long_context
        event["event_sha256"] = canonical_sha256(event)
        events.append(event)
        if cost.total_cost is None:
            complete_cost = False
        else:
            total += float(cost.total_cost)
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
        events_path = run_root / "shards" / shard["shard_id"] / "events.jsonl"
        if not events_path.exists():
            raise ValueError(
                "all primary shards must be collected before recovery freeze"
            )
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
        recovery_shards.append(
            {
                "shard_id": recovery_shard_id,
                "input_relative_path": str(input_path.relative_to(run_root)),
                "input_sha256": file_sha256(input_path),
                "bytes": input_path.stat().st_size,
                "request_count": len(ordered),
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
    submitter: Callable[..., dict[str, Any]] = _submit_provider,
) -> dict[str, Any]:
    intent, _bundle, binding, shard_root = _recovery_binding(run_root, shard_id)
    if (shard_root / "submission-intent.json").exists():
        raise FileExistsError(
            "coarse production recovery submission was already attempted"
        )
    primary_cost = sum(
        float(_load_object(path)["known_priced_cost_usd"])
        for path in run_root.glob("shards/*/collection.json")
    )
    recovery_cost = sum(
        float(_load_object(path)["known_priced_cost_usd"])
        for path in run_root.glob("recovery-000/shards/*/collection.json")
    )
    if primary_cost + recovery_cost >= float(intent["maximum_authorized_cost_usd"]):
        raise ValueError("collected campaign cost has exhausted recovery authorization")
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
            "metadata": metadata,
        },
        "submission_intent_sha256",
    )
    atomic_write_json(shard_root / "submission-intent.json", submission_intent)
    try:
        provider = submitter(input_path, metadata=metadata)
        atomic_write_json(shard_root / "provider-create-response.json", provider)
        _validate_snapshot(provider, metadata=metadata)
        receipt = _hashed(
            {
                "schema_version": SUBMISSION_SCHEMA,
                "status": "submitted",
                "recorded_at": _now(),
                "campaign_run_sha256": intent["campaign_run_sha256"],
                "submission_intent_sha256": submission_intent[
                    "submission_intent_sha256"
                ],
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
    _validate_snapshot(provider, metadata=metadata)
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
    provider = submission["provider_response"]
    if retriever is None:

        def retriever(batch_id: str) -> dict[str, Any]:
            return _provider_batch_dict(_openai_client().batches.retrieve(batch_id))

    observed = retriever(provider["batch_id"])
    _validate_snapshot(
        observed,
        metadata=_metadata(intent, shard_id, "recovery-000"),
        batch_id=provider["batch_id"],
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
    )
    if snapshot.get("status") != "completed":
        raise ValueError("coarse production recovery Batch is not completed")
    raw_root = shard_root / "raw"
    raw_root.mkdir(exist_ok=True)
    atomic_write_json(raw_root / "provider-snapshot.json", snapshot)
    rows = {}
    for source, item in files.items():
        if item["file_id"] != snapshot.get(f"{source}_file_id"):
            raise ValueError("coarse production recovery file receipt drift")
        path = raw_root / f"{source}.jsonl"
        atomic_write_bytes(path, item["content"])
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
    total = 0.0
    complete_cost = True
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
                "error_type": "missing_provider_row",
                "usage": openai_usage(None).model_dump(mode="json"),
                "decisions": None,
            }
        event["generation"] = "recovery-000"
        usage = Usage.model_validate(event["usage"])
        cost, long_context = _estimate_v4_actual_cost(
            prices, model="gpt-5.6-luna", usage=usage
        )
        event["cost"] = cost.model_dump(mode="json")
        event["long_context_price_multiplier_applied"] = long_context
        event["event_sha256"] = canonical_sha256(event)
        events.append(event)
        if cost.total_cost is None:
            complete_cost = False
        else:
            total += float(cost.total_cost)
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
            "authorization_exceeded": authorization_exceeded,
            "events_sha256": file_sha256(shard_root / "events.jsonl"),
        },
        "collection_sha256",
    )
    atomic_write_json(shard_root / "collection.json", result)
    return result


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
    evidence_root = temporary / "collection-evidence"
    for binding in collection_bindings:
        source_root = (
            run_root / "shards" / binding["shard_id"]
            if binding["generation"] == "primary"
            else recovery_root / "shards" / binding["shard_id"]
        )
        target = evidence_root / binding["generation"] / binding["shard_id"]
        target.mkdir(parents=True)
        shutil.copyfile(source_root / "collection.json", target / "collection.json")
        shutil.copyfile(source_root / "events.jsonl", target / "events.jsonl")
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
            "collection_bindings": collection_bindings,
            "claim_boundary": bundle["config"]["claim_boundary"],
        },
        "proposal_bank_manifest_sha256",
    )
    atomic_write_json(temporary / "manifest.json", result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.rename(destination)
    _readonly_tree(destination)
    return result
