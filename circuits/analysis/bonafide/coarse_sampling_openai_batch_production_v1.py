"""Receipt-bound multi-shard Batch lifecycle for coarse production v1.

All provider mutations are explicit calls.  Building or loading the campaign is
network free.  Submission intent is durable before upload/create; ambiguous
state forbids automatic retries and must be reconciled by metadata discovery.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import (
    canonical_json,
    canonical_sha256,
    file_sha256,
)
from circuits.analysis.bonafide.coarse_sampling_annotation import validate_decisions
from circuits.analysis.bonafide.coarse_sampling_openai_batch_v4 import (
    _estimate_v4_actual_cost,
    _openai_client,
    _openai_file_bytes,
    _response_text,
)
from circuits.analysis.bonafide.coarse_sampling_openai_batch_v4 import (
    _provider_batch_dict as _base_provider_batch_dict,
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
from circuits.labeling.pricing import estimate_cost, load_price_snapshot
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


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_object(path) != dict(value):
            raise ValueError(f"retained coarse production JSON drift: {path}")
        return
    atomic_write_json(path, value)


def _write_or_verify_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"retained coarse production bytes drift: {path}")
        return
    atomic_write_bytes(path, value)


def _write_or_verify_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    expected = b"".join(
        (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for value in values
    )
    _write_or_verify_bytes(path, expected)


def _load_or_create_collection_intent(
    *, path: Path, submission: Mapping[str, Any]
) -> dict[str, Any]:
    if path.exists():
        intent = _load_object(path)
        _verify(intent, "collection_intent_sha256", "collection intent")
        if (
            intent.get("submission_sha256") != submission["submission_sha256"]
            or intent.get("batch_id") != submission["provider_response"]["batch_id"]
        ):
            raise ValueError("retained collection intent binding drift")
        return intent
    intent = _hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "status": "intent_persisted",
            "recorded_at": _now(),
            "submission_sha256": submission["submission_sha256"],
            "batch_id": submission["provider_response"]["batch_id"],
        },
        "collection_intent_sha256",
    )
    atomic_write_json(path, intent)
    return intent


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


@contextmanager
def _collection_gate(shard_root: Path):
    lock = shard_root / ".collection-gate"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "coarse production collection gate is already held or stale; "
            "inspect retained collection evidence before manual lock removal"
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


def _active_collection_locks(run_root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(run_root))
        for pattern in (
            "shards/*/.collection-gate",
            "recovery-000/shards/*/.collection-gate",
        )
        for path in run_root.glob(pattern)
    )


def initialize_campaign_run(
    *,
    bundle_root: Path,
    run_root: Path,
    authorized_primary_shard_ids: list[str],
    forecast_budget_usd: float,
    forecast_budget_authorization_note: str,
    acknowledged_strict_worst_case_exposure_usd: float,
    strict_exposure_acknowledgement_note: str,
    primary_actual_spend_limit_usd: float,
    provider_queued_input_token_limit: int,
    maximum_concurrent_shards: int,
) -> dict[str, Any]:
    """Bind launch authorization and queue capacity without provider calls."""

    bundle = load_production_bundle(bundle_root, load_units=False)
    if run_root.exists():
        raise FileExistsError(f"coarse production run exists: {run_root}")
    full_campaign_strict_exposure = float(
        bundle["cost_plan"]["strict_no_cache_full_output_exposure_usd"]
    )
    if (
        not authorized_primary_shard_ids
        or any(not isinstance(value, str) for value in authorized_primary_shard_ids)
        or len(authorized_primary_shard_ids) != len(set(authorized_primary_shard_ids))
    ):
        raise ValueError(
            "coarse production requires a nonempty unique authorized primary shard subset"
        )
    requested = set(authorized_primary_shard_ids)
    known = {str(shard["shard_id"]) for shard in bundle["shards"]}
    if not requested <= known:
        raise ValueError("authorized primary shard subset contains unknown shard")
    selected_shards = [
        shard for shard in bundle["shards"] if shard["shard_id"] in requested
    ]
    frozen_authorized_ids = [str(shard["shard_id"]) for shard in selected_shards]
    subset_forecast = sum(
        float(shard["direct_v4_cost_forecast_usd"]) for shard in selected_shards
    )
    strict_exposure = sum(
        float(shard["strict_no_cache_full_output_exposure_usd"])
        for shard in selected_shards
    )
    if not all(
        math.isfinite(value)
        for value in (
            forecast_budget_usd,
            acknowledged_strict_worst_case_exposure_usd,
            primary_actual_spend_limit_usd,
            subset_forecast,
            strict_exposure,
            full_campaign_strict_exposure,
        )
    ):
        raise ValueError("coarse production authorization costs must be finite")
    if forecast_budget_usd <= 0 or not forecast_budget_authorization_note.strip():
        raise ValueError(
            "coarse production requires an explicit positive forecast budget"
        )
    if forecast_budget_usd + 1e-9 < subset_forecast:
        raise ValueError(
            "coarse production forecast budget is below the frozen authorized subset forecast"
        )
    if (
        abs(acknowledged_strict_worst_case_exposure_usd - strict_exposure) > 1e-9
        or not strict_exposure_acknowledgement_note.strip()
    ):
        raise ValueError(
            "coarse production requires exact acknowledgement that actual spend may "
            "exceed the forecast budget up to the strict worst-case exposure"
        )
    if primary_actual_spend_limit_usd <= strict_exposure:
        raise ValueError(
            "primary actual spend limit must be strictly above the frozen subset strict exposure"
        )
    if not 1 <= maximum_concurrent_shards <= len(selected_shards):
        raise ValueError(
            "maximum concurrent shards is outside the authorized primary subset"
        )
    if requested != known and maximum_concurrent_shards != 1:
        raise ValueError("primary shard-subset authorization requires concurrency one")
    largest_forecasts = sorted(
        (
            int(shard["queued_input_tokens_empirical_forecast"])
            for shard in selected_shards
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
        for shard in selected_shards:
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
                    "direct_v4_cost_forecast_usd": shard["direct_v4_cost_forecast_usd"],
                    "strict_no_cache_full_output_exposure_usd": shard[
                        "strict_no_cache_full_output_exposure_usd"
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
                "authorization_scope_schema_version": (
                    "adag.process-witness.coarse-production-primary-scope.v1"
                ),
                "authorization_scope": "explicit_primary_shard_subset",
                "authorized_primary_shard_ids": frozen_authorized_ids,
                "authorized_primary_direct_v4_cost_forecast_usd": subset_forecast,
                "full_campaign_authorized": requested == known,
                "full_campaign_strict_worst_case_exposure_usd": (
                    full_campaign_strict_exposure
                ),
                "forecast_budget_usd": forecast_budget_usd,
                "forecast_budget_authorization_note": forecast_budget_authorization_note,
                "forecast_budget_is_hard_spend_cap": False,
                "strict_worst_case_exposure_usd": strict_exposure,
                "acknowledged_strict_worst_case_exposure_usd": (
                    acknowledged_strict_worst_case_exposure_usd
                ),
                "strict_exposure_acknowledgement_note": (
                    strict_exposure_acknowledgement_note
                ),
                "primary_actual_spend_limit_usd": primary_actual_spend_limit_usd,
                "primary_actual_spend_must_be_strictly_below_limit": True,
                "primary_actual_limit_protected_by_frozen_request_caps": True,
                "provider_queued_input_token_limit": provider_queued_input_token_limit,
                "maximum_concurrent_shards": maximum_concurrent_shards,
                "queue_policy": "active frozen shards must fit recorded concurrency and queued-token cap",
                "shards": shard_bindings,
                "environment": {
                    "hostname": platform.node(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "endpoint_identity": "https://api.openai.com/v1",
                    "python_version": platform.python_version(),
                    "openai_sdk_version": importlib.metadata.version("openai"),
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


def _validate_runtime_environment(environment: Any) -> None:
    project = os.environ.get("OPENAI_PROJECT_ID")
    organization = os.environ.get("OPENAI_ORG_ID")
    project_sha256 = hashlib.sha256(project.encode()).hexdigest() if project else None
    organization_sha256 = (
        hashlib.sha256(organization.encode()).hexdigest() if organization else None
    )
    if (
        not isinstance(environment, Mapping)
        or environment.get("python_version") != platform.python_version()
        or environment.get("openai_sdk_version") != importlib.metadata.version("openai")
        or environment.get("openai_project_sha256") != project_sha256
        or environment.get("openai_organization_sha256") != organization_sha256
    ):
        raise ValueError("coarse production runtime environment drift")


def _campaign(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    intent = _load_object(run_root / "campaign-intent.json")
    _verify(intent, "campaign_run_sha256", "coarse production campaign intent")
    bundle = load_production_bundle(
        Path(intent["bundle_root"]), load_units=False, strict_topology=False
    )
    if bundle["manifest"]["manifest_sha256"] != intent["bundle_manifest_sha256"]:
        raise ValueError("coarse production run/bundle binding drift")
    _validate_primary_authorization_scope(intent, bundle)
    _validate_runtime_environment(intent.get("environment"))
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


def _validate_primary_authorization_scope(
    intent: Mapping[str, Any], bundle: Mapping[str, Any]
) -> None:
    raw_authorized_ids = intent.get("authorized_primary_shard_ids")
    raw_bindings = intent.get("shards")
    if (
        not isinstance(raw_authorized_ids, list)
        or not raw_authorized_ids
        or any(not isinstance(value, str) for value in raw_authorized_ids)
        or not isinstance(raw_bindings, list)
        or any(not isinstance(binding, Mapping) for binding in raw_bindings)
    ):
        raise ValueError("coarse production primary authorization scope drift")
    authorized_ids = [value for value in raw_authorized_ids if isinstance(value, str)]
    bindings = [binding for binding in raw_bindings if isinstance(binding, Mapping)]
    if (
        intent.get("authorization_scope_schema_version")
        != "adag.process-witness.coarse-production-primary-scope.v1"
        or intent.get("authorization_scope") != "explicit_primary_shard_subset"
        or len(authorized_ids) != len(set(authorized_ids))
        or [binding.get("shard_id") for binding in bindings] != authorized_ids
    ):
        raise ValueError("coarse production primary authorization scope drift")
    by_id = {str(shard["shard_id"]): shard for shard in bundle["shards"]}
    if any(
        not isinstance(value, str) or value not in by_id for value in authorized_ids
    ):
        raise ValueError("coarse production authorized primary shard identity drift")
    selected = [by_id[value] for value in authorized_ids]
    expected_order = [
        str(shard["shard_id"])
        for shard in bundle["shards"]
        if shard["shard_id"] in set(authorized_ids)
    ]
    subset_forecast = sum(
        float(shard["direct_v4_cost_forecast_usd"]) for shard in selected
    )
    subset_strict = sum(
        float(shard["strict_no_cache_full_output_exposure_usd"]) for shard in selected
    )
    full_ids = {str(shard["shard_id"]) for shard in bundle["shards"]}
    numeric_scope_values = (
        intent.get("authorized_primary_direct_v4_cost_forecast_usd"),
        intent.get("strict_worst_case_exposure_usd"),
        intent.get("acknowledged_strict_worst_case_exposure_usd"),
        intent.get("full_campaign_strict_worst_case_exposure_usd"),
        intent.get("primary_actual_spend_limit_usd"),
        intent.get("forecast_budget_usd"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_scope_values
    ):
        raise ValueError("coarse production primary authorization cost drift")
    maximum_concurrent = intent.get("maximum_concurrent_shards")
    queued_limit = intent.get("provider_queued_input_token_limit")
    largest_queue_reservations = sorted(
        (int(shard["queued_input_tokens_empirical_forecast"]) for shard in selected),
        reverse=True,
    )[: maximum_concurrent if isinstance(maximum_concurrent, int) else 0]
    if (
        authorized_ids != expected_order
        or abs(
            float(intent.get("authorized_primary_direct_v4_cost_forecast_usd", -1))
            - subset_forecast
        )
        > 1e-9
        or abs(float(intent.get("strict_worst_case_exposure_usd", -1)) - subset_strict)
        > 1e-9
        or abs(
            float(intent.get("acknowledged_strict_worst_case_exposure_usd", -1))
            - subset_strict
        )
        > 1e-9
        or intent.get("full_campaign_authorized") != (set(authorized_ids) == full_ids)
        or not isinstance(maximum_concurrent, int)
        or not 1 <= maximum_concurrent <= len(selected)
        or (set(authorized_ids) != full_ids and maximum_concurrent != 1)
        or not isinstance(queued_limit, int)
        or queued_limit < sum(largest_queue_reservations)
        or float(intent.get("forecast_budget_usd", -1)) + 1e-9 < subset_forecast
        or not isinstance(intent.get("forecast_budget_authorization_note"), str)
        or not intent["forecast_budget_authorization_note"].strip()
        or not isinstance(intent.get("strict_exposure_acknowledgement_note"), str)
        or not intent["strict_exposure_acknowledgement_note"].strip()
        or abs(
            float(intent.get("full_campaign_strict_worst_case_exposure_usd", -1))
            - float(bundle["cost_plan"]["strict_no_cache_full_output_exposure_usd"])
        )
        > 1e-9
        or float(intent.get("primary_actual_spend_limit_usd", -1)) <= subset_strict
        or intent.get("primary_actual_spend_must_be_strictly_below_limit") is not True
        or intent.get("primary_actual_limit_protected_by_frozen_request_caps")
        is not True
    ):
        raise ValueError("coarse production primary authorization semantics drift")
    for binding, shard in zip(bindings, selected, strict=True):
        if (
            binding.get("input_relative_path")
            != f"shards/{shard['shard_id']}/input.jsonl"
            or binding.get("input_sha256") != shard["sha256"]
            or binding.get("bytes") != shard["bytes"]
            or binding.get("request_count") != shard["request_count"]
            or abs(
                float(binding.get("direct_v4_cost_forecast_usd", -1))
                - float(shard["direct_v4_cost_forecast_usd"])
            )
            > 1e-9
            or abs(
                float(binding.get("strict_no_cache_full_output_exposure_usd", -1))
                - float(shard["strict_no_cache_full_output_exposure_usd"])
            )
            > 1e-9
            or binding.get("queued_input_tokens_empirical_forecast")
            != shard["queued_input_tokens_empirical_forecast"]
        ):
            raise ValueError("coarse production authorized primary binding drift")


def _authorized_primary_shard(
    intent: Mapping[str, Any], shard_id: str
) -> Mapping[str, Any]:
    for shard in intent["shards"]:
        if shard["shard_id"] == shard_id:
            return shard
    raise ValueError(f"primary shard is not authorized for this run: {shard_id}")


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


def _provider_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _provider_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_provider_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _provider_json_value(model_dump(mode="json"))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _provider_json_value(to_dict())
    raise TypeError(f"unsupported provider receipt value: {type(value).__name__}")


def _production_provider_batch_dict(batch: Any) -> dict[str, Any]:
    """Retain the base Batch receipt plus structured Batch-level errors."""

    value = _base_provider_batch_dict(batch)
    raw = _provider_json_value(batch)
    value["provider_model_dump"] = raw
    value["errors"] = (
        raw.get("errors")
        if isinstance(raw, Mapping) and "errors" in raw
        else _provider_json_value(getattr(batch, "errors", None))
    )
    return value


def _create_provider(input_file_id: str, *, metadata: dict[str, str]) -> dict[str, Any]:
    batch = _openai_client().batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata,
    )
    if getattr(batch, "input_file_id", None) != input_file_id:
        raise ValueError("coarse production created Batch input file id drift")
    return _production_provider_batch_dict(batch)


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
        or (input_file_id is not None and value.get("input_file_id") != input_file_id)
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
    intent_shard = _authorized_primary_shard(intent, shard_id)
    shard = next((s for s in bundle["shards"] if s["shard_id"] == shard_id), None)
    if shard is None:
        raise ValueError("unknown coarse production shard")
    shard_root = run_root / "shards" / shard_id
    with _submission_gate(run_root):
        collection_locks = _active_collection_locks(run_root)
        if collection_locks:
            raise RuntimeError(
                f"collection materialization is active: {collection_locks}"
            )
        if (shard_root / "submission-intent.json").exists():
            raise FileExistsError(
                "coarse production shard submission was already attempted"
            )
        attempted_cost, cost_complete = _attempted_primary_forecast(run_root, intent)
        if not cost_complete:
            raise ValueError("prior collected campaign cost is not fully priced")
        if attempted_cost > float(intent["strict_worst_case_exposure_usd"]):
            raise ValueError(
                "prior primary actual cost exceeds acknowledged strict exposure"
            )
        if attempted_cost >= float(intent["primary_actual_spend_limit_usd"]):
            raise ValueError("prior primary actual cost reached its strict spend limit")
        candidate_cost = float(intent_shard["direct_v4_cost_forecast_usd"])
        if attempted_cost + candidate_cost > float(intent["forecast_budget_usd"]):
            raise ValueError(
                "prospective actual-or-reserved campaign cost exceeds forecast budget"
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


def _recover_submission_failure(
    *,
    shard_root: Path,
    intent: Mapping[str, Any],
    shard_id: str,
    generation: str,
    discoverer: Callable[..., list[dict[str, Any]]] | None,
    uploader: Callable[[Path], dict[str, Any]],
    creator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if (shard_root / "submission.json").exists():
        raise FileExistsError("coarse production submission receipt already exists")
    submission_intent = _load_object(shard_root / "submission-intent.json")
    _verify(submission_intent, "submission_intent_sha256", "submission intent")
    if submission_intent.get("generation") != generation:
        raise ValueError("coarse production recovery generation drift")
    metadata = _metadata(intent, shard_id, generation)
    if discoverer is None:

        def discoverer(**_: Any) -> list[dict[str, Any]]:
            return [
                _production_provider_batch_dict(batch)
                for batch in _openai_client().batches.list(limit=100)
                if dict(getattr(batch, "metadata", None) or {}) == metadata
            ]

    upload_path = shard_root / "provider-upload-response.json"
    if not upload_path.exists():
        if (shard_root / "provider-create-response.json").exists():
            raise ValueError(
                "provider create snapshot exists without its upload receipt"
            )
        matches = discoverer(metadata=metadata)
        if matches:
            raise ValueError(
                "upload receipt is absent but provider metadata discovery found Batch(es)"
            )
        failure_path = shard_root / "submission-failure.json"
        original_failure = None
        if failure_path.exists():
            original_failure = _load_object(failure_path)
            _verify(
                original_failure,
                "submission_failure_sha256",
                "ambiguous upload failure",
            )
            if (
                original_failure.get("submission_intent_sha256")
                != submission_intent["submission_intent_sha256"]
                or original_failure.get("upload_receipt_persisted") is not False
            ):
                raise ValueError("ambiguous upload failure/submission intent drift")
        orphan = _hashed(
            {
                "schema_version": "adag.process-witness.coarse-production-orphan-upload.v1",
                "recorded_at": _now(),
                "submission_failure_sha256": (
                    original_failure["submission_failure_sha256"]
                    if original_failure is not None
                    else None
                ),
                "submission_intent_sha256": submission_intent[
                    "submission_intent_sha256"
                ],
                "failure_receipt_present": original_failure is not None,
                "metadata_matches_before_reupload": 0,
                "original_upload_may_be_orphaned": True,
                "batch_create_could_not_run_without_an_upload_receipt": True,
            },
            "orphan_upload_state_sha256",
        )
        atomic_write_json(shard_root / "orphan-upload-state.json", orphan)
        upload = uploader(shard_root / "input.jsonl")
        _validate_upload(upload)
        atomic_write_json(upload_path, upload)
        provider = creator(upload["input_file_id"], metadata=metadata)
        atomic_write_json(shard_root / "provider-create-response.json", provider)
        recovered_by = "safe_reupload_after_zero_batch_metadata_matches"
    else:
        upload = _load_object(upload_path)
        _validate_upload(upload)
        create = shard_root / "provider-create-response.json"
        if create.exists():
            provider = _load_object(create)
            recovered_by = "immediate_create_snapshot"
        else:
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
            "provider_upload_response_sha256": file_sha256(
                shard_root / "provider-upload-response.json"
            ),
            "provider_response": provider,
        },
        "submission_sha256",
    )
    atomic_write_json(shard_root / "submission.json", receipt)
    return receipt


def recover_shard_submission(
    *,
    run_root: Path,
    shard_id: str,
    discoverer: Callable[..., list[dict[str, Any]]] | None = None,
    uploader: Callable[[Path], dict[str, Any]] = _upload_provider,
    creator: Callable[..., dict[str, Any]] = _create_provider,
) -> dict[str, Any]:
    intent, _bundle = _campaign(run_root)
    _authorized_primary_shard(intent, shard_id)
    with _submission_gate(run_root):
        return _recover_submission_failure(
            shard_root=run_root / "shards" / shard_id,
            intent=intent,
            shard_id=shard_id,
            generation="primary",
            discoverer=discoverer,
            uploader=uploader,
            creator=creator,
        )


def _submission(run_root: Path, shard_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    intent, _bundle = _campaign(run_root)
    _authorized_primary_shard(intent, shard_id)
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
            return _production_provider_batch_dict(
                _openai_client().batches.retrieve(batch_id)
            )

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
    snapshot = _production_provider_batch_dict(batch)
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
        provider_error = row.get("error")
        error_code = (
            provider_error.get("code") if isinstance(provider_error, Mapping) else None
        )
        error_message = (
            provider_error.get("message")
            if isinstance(provider_error, Mapping)
            else provider_error
        )
        return {
            **common,
            "validation_status": "provider_error",
            "error_type": "batch_request_error",
            "provider_error_code": str(error_code) if error_code is not None else None,
            "error_message": (
                str(error_message)[:2000] if error_message is not None else None
            ),
            "usage": openai_usage(None).model_dump(mode="json"),
            "decisions": None,
        }
    body = response.get("body")
    if response.get("status_code") != 200 or not isinstance(body, Mapping):
        body_error = body.get("error") if isinstance(body, Mapping) else None
        error_code = body_error.get("code") if isinstance(body_error, Mapping) else None
        error_message = (
            body_error.get("message") if isinstance(body_error, Mapping) else body_error
        )
        return {
            **common,
            "validation_status": "provider_error",
            "error_type": "batch_http_error",
            "provider_error_code": str(error_code) if error_code is not None else None,
            "error_message": (
                str(error_message)[:2000] if error_message is not None else None
            ),
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
    aggregate_fallback_long_context_impossible: bool = False,
) -> tuple[float, bool, str]:
    """Price row usage first; use aggregate usage only with a safe tier proof."""

    aggregate_raw = snapshot.get("usage")
    aggregate_usage = openai_usage(aggregate_raw)
    event_total = 0.0
    event_complete = True
    for event in events:
        usage = Usage.model_validate(event["usage"])
        cost, long_context = _estimate_v4_actual_cost(
            prices, model="gpt-5.6-luna", usage=usage
        )
        event["pricing_basis"] = "request_usage"
        event["cost"] = cost.model_dump(mode="json")
        event["long_context_price_multiplier_applied"] = long_context
        if cost.total_cost is None:
            event_complete = False
        else:
            event_total += float(cost.total_cost)
    total = event_total
    complete = event_complete
    basis = "per_request_usage_with_per_request_long_context_threshold"
    if isinstance(aggregate_raw, Mapping) and event_complete:
        input_details = aggregate_raw.get("input_tokens_details")
        presence = {
            "input_tokens": "input_tokens" in aggregate_raw,
            "cache_read_tokens": isinstance(input_details, Mapping)
            and "cached_tokens" in input_details,
            "cache_write_tokens": isinstance(input_details, Mapping)
            and "cache_write_tokens" in input_details,
            "output_tokens": "output_tokens" in aggregate_raw,
        }
        for field, present in presence.items():
            if not present:
                continue
            expected = getattr(aggregate_usage, field)
            values = [getattr(Usage.model_validate(e["usage"]), field) for e in events]
            if (
                expected is not None
                and all(value is not None for value in values)
                and sum(int(value) for value in values if value is not None) != expected
            ):
                complete = False
                basis = "failed_closed_batch_aggregate_usage_mismatch"
                break
        if complete:
            basis = "per_request_usage_reconciled_to_batch_aggregate"
    elif isinstance(aggregate_raw, Mapping) and not event_complete:
        input_details = aggregate_raw.get("input_tokens_details")
        presence = {
            "input_tokens": "input_tokens" in aggregate_raw,
            "cache_read_tokens": isinstance(input_details, Mapping)
            and "cached_tokens" in input_details,
            "cache_write_tokens": isinstance(input_details, Mapping)
            and "cache_write_tokens" in input_details,
            "output_tokens": "output_tokens" in aggregate_raw,
        }
        aggregate_pricing_fields_present = all(presence.values())
        aggregate_not_below_known_rows = True
        for field, present in presence.items():
            if not present:
                continue
            aggregate_value = getattr(aggregate_usage, field)
            known_values = [
                getattr(Usage.model_validate(event["usage"]), field) for event in events
            ]
            known_sum = sum(int(value) for value in known_values if value is not None)
            if aggregate_value is not None and aggregate_value < known_sum:
                aggregate_not_below_known_rows = False
                break
        if aggregate_pricing_fields_present:
            known_uncached_sum = sum(
                int(value)
                for value in (
                    Usage.model_validate(event["usage"]).uncached_input_tokens
                    for event in events
                )
                if value is not None
            )
            if (
                aggregate_usage.uncached_input_tokens is not None
                and aggregate_usage.uncached_input_tokens < known_uncached_sum
            ):
                aggregate_not_below_known_rows = False
        aggregate_not_below_known_cost = True
        threshold = int(
            prices["long_context"]["gpt-5.6-luna"]["threshold_input_tokens_exclusive"]
        )
        aggregate_below_threshold = (
            aggregate_usage.input_tokens is not None
            and aggregate_usage.input_tokens <= threshold
        )
        if (
            aggregate_pricing_fields_present
            and aggregate_not_below_known_rows
            and (
                aggregate_fallback_long_context_impossible or aggregate_below_threshold
            )
        ):
            aggregate_cost = estimate_cost(
                prices,
                provider="openai",
                model="gpt-5.6-luna",
                transport="native_batch",
                usage=aggregate_usage,
            )
            if aggregate_cost.total_cost is not None:
                aggregate_total = float(aggregate_cost.total_cost)
                if aggregate_total + 1e-12 < event_total:
                    aggregate_not_below_known_cost = False
                elif aggregate_total != 0.0 or _proven_pre_execution_failure(snapshot):
                    total = aggregate_total
                    complete = True
                    basis = (
                        "aggregate_batch_usage_with_per_request_byte_upper_bound"
                        if aggregate_fallback_long_context_impossible
                        else "aggregate_batch_usage_below_long_context_threshold"
                    )
        if not aggregate_not_below_known_rows:
            basis = "failed_closed_batch_aggregate_usage_below_known_rows"
        elif not aggregate_not_below_known_cost:
            basis = "failed_closed_batch_aggregate_cost_below_known_rows"
        elif not complete:
            basis = "cost_incomplete_per_request_and_aggregate_usage"
    elif not event_complete:
        basis = "cost_incomplete_per_request_usage_missing"
    for event in events:
        event["collection_pricing_basis"] = basis
    for event in events:
        event["event_sha256"] = canonical_sha256(event)
    return total, complete, basis


def _batch_error_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        data = value.get("data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, Mapping)]
        return [value] if value.get("code") else []
    if isinstance(value, list):
        return [row for row in value if isinstance(row, Mapping)]
    return []


def _proven_pre_execution_failure(snapshot: Mapping[str, Any]) -> bool:
    counts = snapshot.get("request_counts")
    usage_raw = snapshot.get("usage")
    errors = _batch_error_rows(snapshot.get("errors"))
    if (
        snapshot.get("status") != "failed"
        or not isinstance(counts, Mapping)
        or not isinstance(usage_raw, Mapping)
        or not errors
        or any(
            not isinstance(row.get("code"), str) or not row["code"] for row in errors
        )
    ):
        return False
    total = counts.get("total")
    completed = counts.get("completed")
    failed = counts.get("failed")
    if not isinstance(total, int) or total < 0 or completed != 0 or failed != total:
        return False
    input_details = usage_raw.get("input_tokens_details")
    if not isinstance(input_details, Mapping):
        return False
    required_zeroes = (
        usage_raw.get("input_tokens"),
        input_details.get("cached_tokens"),
        input_details.get("cache_write_tokens"),
        usage_raw.get("output_tokens"),
    )
    return required_zeroes == (0, 0, 0, 0)


def _input_byte_bound_excludes_long_context(
    *, input_path: Path, config: Mapping[str, Any], prices: Mapping[str, Any]
) -> bool:
    provider = config["provider"]
    threshold = int(
        prices["long_context"][provider["model"]]["threshold_input_tokens_exclusive"]
    )
    overhead = int(provider["input_token_overhead_per_request"])
    for row in read_jsonl(input_path):
        body_bytes = len(
            json.dumps(
                row["body"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        if body_bytes + overhead > threshold:
            return False
    return True


def _strict_exposure_for_provider_bodies(
    *, config: Mapping[str, Any], prices: Mapping[str, Any], body_bytes: list[int]
) -> float:
    provider = config["provider"]
    rates = prices["rates"]["openai"][provider["model"]]["native_batch"]
    overhead = int(provider["input_token_overhead_per_request"])
    threshold = int(
        prices["long_context"][provider["model"]]["threshold_input_tokens_exclusive"]
    )
    if any(value + overhead > threshold for value in body_bytes):
        raise ValueError(
            "recovery strict exposure needs per-request long-context pricing support"
        )
    input_rate = max(
        float(rates["input_per_million"]),
        float(rates["cache_write_per_million"]),
    )
    output_rate = float(rates["output_per_million"])
    return (
        sum(value + overhead for value in body_bytes) / 1_000_000 * input_rate
        + len(body_bytes) * int(provider["max_output_tokens"]) / 1_000_000 * output_rate
    )


def _collect_shard_locked(
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
    collection_intent = _load_or_create_collection_intent(
        path=shard_root / "collection-intent.json", submission=submission
    )
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
    _write_or_verify_json(raw_root / "provider-snapshot.json", snapshot)
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
        _write_or_verify_bytes(path, content)
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
    bundle = load_production_bundle(
        Path(intent["bundle_root"]), load_units=False, strict_topology=False
    )
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
        events=events,
        snapshot=snapshot,
        prices=prices,
        aggregate_fallback_long_context_impossible=(
            _input_byte_bound_excludes_long_context(
                input_path=shard_root / "input.jsonl",
                config=bundle["config"],
                prices=prices,
            )
        ),
    )
    _write_or_verify_jsonl(shard_root / "events.jsonl", events)
    success = sum(e["validation_status"] == "success" for e in events)
    prior_cost = 0.0
    for other in intent["shards"]:
        path = run_root / "shards" / other["shard_id"] / "collection.json"
        if path.exists():
            prior = _load_object(path)
            _verify(prior, "collection_sha256", "coarse production collection")
            prior_cost += float(prior["known_priced_cost_usd"])
    cumulative_cost = prior_cost + total
    forecast_budget_exceeded = complete_cost and cumulative_cost > float(
        intent["forecast_budget_usd"]
    )
    primary_actual_limit_exceeded = complete_cost and cumulative_cost >= float(
        intent["primary_actual_spend_limit_usd"]
    )
    primary_strict_exposure_exceeded = complete_cost and cumulative_cost > float(
        intent["strict_worst_case_exposure_usd"]
    )
    result = _hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "status": (
                "failed_closed_primary_actual_spend_limit_exceeded"
                if primary_actual_limit_exceeded
                else (
                    "failed_closed_primary_strict_exposure_exceeded"
                    if primary_strict_exposure_exceeded
                    else (
                        "complete_forecast_budget_exceeded_stop_further_primary_submission"
                        if forecast_budget_exceeded
                        else (
                            "complete"
                            if success == len(events) and complete_cost
                            else "complete_with_failed_requests_recovery_eligible"
                        )
                    )
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
            "forecast_budget_usd": intent["forecast_budget_usd"],
            "forecast_budget_is_hard_spend_cap": False,
            "forecast_budget_exceeded": forecast_budget_exceeded,
            "primary_actual_spend_limit_usd": intent["primary_actual_spend_limit_usd"],
            "primary_actual_spend_must_be_strictly_below_limit": True,
            "primary_actual_spend_limit_exceeded": primary_actual_limit_exceeded,
            "primary_strict_worst_case_exposure_usd": intent[
                "strict_worst_case_exposure_usd"
            ],
            "primary_strict_exposure_exceeded": primary_strict_exposure_exceeded,
            "raw_file_bindings": raw_bindings,
            "events_sha256": file_sha256(shard_root / "events.jsonl"),
        },
        "collection_sha256",
    )
    atomic_write_json(shard_root / "collection.json", result)
    return result


def collect_shard(
    *,
    run_root: Path,
    shard_id: str,
    downloader: Callable[
        [str], tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ] = _download,
) -> dict[str, Any]:
    intent, _bundle = _campaign(run_root)
    _authorized_primary_shard(intent, shard_id)
    shard_root = run_root / "shards" / shard_id
    with _collection_gate(shard_root), _submission_gate(run_root):
        return _collect_shard_locked(
            run_root=run_root,
            shard_id=shard_id,
            downloader=downloader,
        )


def prepare_failed_only_recovery(*, run_root: Path) -> dict[str, Any]:
    """Freeze one failed-only recovery wave, partitioned by primary shard."""

    intent, bundle = _campaign(run_root)
    recovery_root = run_root / "recovery-000"
    if recovery_root.exists():
        raise FileExistsError("coarse production recovery wave already exists")
    authorized_ids = [str(shard["shard_id"]) for shard in intent["shards"]]
    by_id = {str(shard["shard_id"]): shard for shard in bundle["shards"]}
    authorized_shards = [by_id[shard_id] for shard_id in authorized_ids]
    failed = []
    successful = set()
    primary_actual_cost = 0.0
    for shard in authorized_shards:
        shard_root = run_root / "shards" / shard["shard_id"]
        events_path = shard_root / "events.jsonl"
        collection_path = shard_root / "collection.json"
        if not events_path.exists() or not collection_path.exists():
            raise ValueError(
                "all authorized primary shards must be collected before recovery freeze"
            )
        collection = _load_object(collection_path)
        _verify(collection, "collection_sha256", "coarse production collection")
        if (
            file_sha256(events_path) != collection["events_sha256"]
            or not collection["cost_complete"]
        ):
            raise ValueError("primary collection is not recovery-eligible")
        primary_actual_cost += float(collection["known_priced_cost_usd"])
        for event in read_jsonl(events_path):
            if event["validation_status"] == "success":
                successful.add(event["request_id"])
            else:
                failed.append(event["request_id"])
    if primary_actual_cost > float(intent["strict_worst_case_exposure_usd"]):
        raise ValueError(
            "primary actual cost exceeds acknowledged strict exposure; recovery forbidden"
        )
    if primary_actual_cost >= float(intent["primary_actual_spend_limit_usd"]):
        raise ValueError("primary actual cost reached its strict spend limit")
    failed_set = set(failed)
    if not failed_set or failed_set & successful:
        raise ValueError(
            "recovery requires failed-only non-overlapping request identities"
        )
    source_lines: dict[str, tuple[str, dict[str, Any]]] = {}
    for shard in authorized_shards:
        for line in read_jsonl(Path(intent["bundle_root"]) / shard["path"]):
            if line["custom_id"] in failed_set:
                source_lines[line["custom_id"]] = (shard["shard_id"], line)
    if set(source_lines) != failed_set:
        raise ValueError("recovery failed request body coverage drift")
    recovery_root.mkdir()
    recovery_shards = []
    recovery_direct_forecast_total = 0.0
    recovery_strict_exposure_total = 0.0
    for primary in authorized_shards:
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
        provider_body_byte_values = [
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
        ]
        provider_body_bytes = sum(provider_body_byte_values)
        empirical = bundle["config"]["empirical_calibration"]
        prices = load_price_snapshot(
            Path(intent["bundle_root"]) / "price-snapshot.json"
        )
        direct_forecast = (
            float(empirical["source_actual_cost_usd"])
            * len(ordered)
            / int(empirical["source_request_count"])
        )
        strict_exposure = _strict_exposure_for_provider_bodies(
            config=bundle["config"],
            prices=prices,
            body_bytes=provider_body_byte_values,
        )
        recovery_direct_forecast_total += direct_forecast
        recovery_strict_exposure_total += strict_exposure
        recovery_shards.append(
            {
                "shard_id": recovery_shard_id,
                "input_relative_path": str(input_path.relative_to(run_root)),
                "input_sha256": file_sha256(input_path),
                "bytes": input_path.stat().st_size,
                "request_count": len(ordered),
                "direct_v4_cost_forecast_usd": direct_forecast,
                "strict_no_cache_full_output_exposure_usd": strict_exposure,
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
            "authorized_primary_shard_ids": authorized_ids,
            "recovery_wave": 0,
            "request_count": len(failed_set),
            "shard_count": len(recovery_shards),
            "shards": recovery_shards,
            "direct_v4_cost_forecast_usd": recovery_direct_forecast_total,
            "strict_no_cache_full_output_exposure_usd": (
                recovery_strict_exposure_total
            ),
            "fresh_recovery_forecast_budget_and_strict_exposure_ack_required": True,
            "successful_requests_rerun": 0,
            "provider_bodies_byte_identical": True,
            "additional_recovery_waves_permitted": False,
        },
        "recovery_manifest_sha256",
    )
    atomic_write_json(recovery_root / "manifest.json", manifest)
    return manifest


def authorize_recovery_wave(
    *,
    run_root: Path,
    recovery_forecast_budget_usd: float,
    forecast_budget_authorization_note: str,
    acknowledged_strict_worst_case_exposure_usd: float,
    strict_exposure_acknowledgement_note: str,
) -> dict[str, Any]:
    intent, bundle = _campaign(run_root)
    recovery_root = run_root / "recovery-000"
    manifest = _load_object(recovery_root / "manifest.json")
    _verify(manifest, "recovery_manifest_sha256", "coarse production recovery")
    _validate_recovery_manifest_inputs(
        run_root=run_root,
        authoritative_bundle_root=Path(intent["bundle_root"]),
        intent=intent,
        bundle=bundle,
        manifest=manifest,
    )
    path = recovery_root / "authorization.json"
    if path.exists():
        raise FileExistsError("coarse production recovery authorization exists")
    strict_exposure = float(manifest["strict_no_cache_full_output_exposure_usd"])
    direct_forecast = float(manifest["direct_v4_cost_forecast_usd"])
    if not all(
        math.isfinite(value)
        for value in (
            recovery_forecast_budget_usd,
            acknowledged_strict_worst_case_exposure_usd,
            strict_exposure,
            direct_forecast,
        )
    ):
        raise ValueError("recovery authorization costs must be finite")
    if (
        recovery_forecast_budget_usd <= 0
        or recovery_forecast_budget_usd + 1e-12 < direct_forecast
        or not forecast_budget_authorization_note.strip()
    ):
        raise ValueError("recovery requires a positive explicit forecast budget")
    if (
        abs(acknowledged_strict_worst_case_exposure_usd - strict_exposure) > 1e-9
        or not strict_exposure_acknowledgement_note.strip()
    ):
        raise ValueError(
            "recovery requires exact fresh acknowledgement that actual spend may "
            "exceed its forecast budget up to its strict worst-case exposure"
        )
    authorization = _hashed(
        {
            "schema_version": "adag.process-witness.coarse-production-recovery-authorization.v1",
            "created_at": _now(),
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "recovery_manifest_sha256": manifest["recovery_manifest_sha256"],
            "recovery_forecast_budget_usd": recovery_forecast_budget_usd,
            "forecast_budget_authorization_note": forecast_budget_authorization_note,
            "forecast_budget_is_hard_spend_cap": False,
            "strict_worst_case_exposure_usd": strict_exposure,
            "acknowledged_strict_worst_case_exposure_usd": (
                acknowledged_strict_worst_case_exposure_usd
            ),
            "strict_exposure_acknowledgement_note": (
                strict_exposure_acknowledgement_note
            ),
        },
        "recovery_authorization_sha256",
    )
    _validate_recovery_authorization(
        authorization=authorization,
        manifest=manifest,
        intent=intent,
    )
    atomic_write_json(path, authorization)
    return authorization


def _validate_recovery_authorization(
    *,
    authorization: Mapping[str, Any],
    manifest: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> None:
    _validate_recovery_manifest_scope(manifest=manifest, intent=intent)
    _verify(
        authorization,
        "recovery_authorization_sha256",
        "recovery authorization",
    )
    strict_exposure = float(manifest["strict_no_cache_full_output_exposure_usd"])
    direct_forecast = float(manifest["direct_v4_cost_forecast_usd"])
    recovery_budget = authorization.get("recovery_forecast_budget_usd")
    if (
        authorization.get("schema_version")
        != "adag.process-witness.coarse-production-recovery-authorization.v1"
        or authorization.get("campaign_run_sha256") != intent["campaign_run_sha256"]
        or authorization.get("recovery_manifest_sha256")
        != manifest["recovery_manifest_sha256"]
        or isinstance(recovery_budget, bool)
        or not isinstance(recovery_budget, (int, float))
        or not math.isfinite(float(recovery_budget))
        or float(recovery_budget) <= 0
        or float(recovery_budget) + 1e-12 < direct_forecast
        or not math.isfinite(strict_exposure)
        or not math.isfinite(direct_forecast)
        or not isinstance(authorization.get("forecast_budget_authorization_note"), str)
        or not authorization["forecast_budget_authorization_note"].strip()
        or authorization.get("forecast_budget_is_hard_spend_cap") is not False
        or authorization.get("strict_worst_case_exposure_usd") != strict_exposure
        or authorization.get("acknowledged_strict_worst_case_exposure_usd")
        != strict_exposure
        or not isinstance(
            authorization.get("strict_exposure_acknowledgement_note"), str
        )
        or not authorization["strict_exposure_acknowledgement_note"].strip()
    ):
        raise ValueError("recovery authorization semantic drift")


def _validate_recovery_manifest_scope(
    *, manifest: Mapping[str, Any], intent: Mapping[str, Any]
) -> None:
    authorized = intent.get("authorized_primary_shard_ids")
    recovery_shards = manifest.get("shards")
    recovery_shard_ids = []
    request_ids = []
    if isinstance(recovery_shards, list):
        for shard in recovery_shards:
            if isinstance(shard, Mapping):
                recovery_shard_ids.append(shard.get("shard_id"))
                shard_request_ids = shard.get("request_ids_in_order")
                if isinstance(shard_request_ids, list):
                    request_ids.extend(shard_request_ids)
    if (
        manifest.get("campaign_run_sha256") != intent.get("campaign_run_sha256")
        or manifest.get("authorized_primary_shard_ids") != authorized
        or not isinstance(authorized, list)
        or not isinstance(recovery_shards, list)
        or not recovery_shards
        or any(not isinstance(shard_id, str) for shard_id in recovery_shard_ids)
        or len(recovery_shard_ids) != len(set(recovery_shard_ids))
        or any(not isinstance(request_id, str) for request_id in request_ids)
        or len(request_ids) != len(set(request_ids))
        or manifest.get("shard_count") != len(recovery_shards)
        or manifest.get("request_count") != len(request_ids)
        or any(
            not isinstance(shard, Mapping)
            or shard.get("shard_id") not in authorized
            or not isinstance(shard.get("request_ids_in_order"), list)
            or shard.get("request_count") != len(shard["request_ids_in_order"])
            for shard in recovery_shards
        )
    ):
        raise ValueError("coarse production recovery primary-scope drift")


def _validate_recovery_manifest_inputs(
    *,
    run_root: Path,
    authoritative_bundle_root: Path,
    intent: Mapping[str, Any],
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Bind recovery rows to failed events and frozen rows before provider mutation."""

    _validate_recovery_manifest_scope(manifest=manifest, intent=intent)
    bundle_by_id = {str(shard["shard_id"]): shard for shard in bundle["shards"]}
    failed_by_shard: dict[str, list[str]] = {}
    original_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    primary_actual_cost = 0.0
    for shard_id in intent["authorized_primary_shard_ids"]:
        primary_root = run_root / "shards" / shard_id
        collection = _load_object(primary_root / "collection.json")
        _verify(collection, "collection_sha256", "primary recovery source collection")
        events_path = primary_root / "events.jsonl"
        if (
            file_sha256(events_path) != collection.get("events_sha256")
            or collection.get("cost_complete") is not True
        ):
            raise ValueError("recovery primary source collection binding drift")
        primary_actual_cost += float(collection["known_priced_cost_usd"])
        failed_by_shard[shard_id] = [
            str(event["request_id"])
            for event in read_jsonl(events_path)
            if event.get("validation_status") != "success"
        ]
        frozen_shard = bundle_by_id[shard_id]
        for row in read_jsonl(authoritative_bundle_root / frozen_shard["path"]):
            request_id = row.get("custom_id")
            if not isinstance(request_id, str) or request_id in original_by_id:
                raise ValueError("recovery frozen primary request identity drift")
            original_by_id[request_id] = (shard_id, row)
    if primary_actual_cost > float(intent["strict_worst_case_exposure_usd"]):
        raise ValueError(
            "primary actual cost exceeds acknowledged strict exposure; recovery forbidden"
        )
    if primary_actual_cost >= float(intent["primary_actual_spend_limit_usd"]):
        raise ValueError("primary actual cost reached its strict spend limit")
    expected_failed = {
        request_id
        for shard_request_ids in failed_by_shard.values()
        for request_id in shard_request_ids
    }
    expected_recovery_shard_ids = [
        shard_id
        for shard_id in intent["authorized_primary_shard_ids"]
        if failed_by_shard[shard_id]
    ]
    if [binding["shard_id"] for binding in manifest["shards"]] != (
        expected_recovery_shard_ids
    ):
        raise ValueError("recovery shard order/coverage drift")
    empirical = bundle["config"]["empirical_calibration"]
    prices = load_price_snapshot(authoritative_bundle_root / "price-snapshot.json")
    expected_direct_total = 0.0
    expected_strict_total = 0.0
    observed_failed: set[str] = set()
    for binding in manifest["shards"]:
        shard_id = str(binding["shard_id"])
        expected_relative_path = f"recovery-000/shards/{shard_id}/input.jsonl"
        if binding.get("input_relative_path") != expected_relative_path:
            raise ValueError("recovery input path binding drift")
        input_path = run_root / str(binding["input_relative_path"])
        if (
            not input_path.is_file()
            or file_sha256(input_path) != binding.get("input_sha256")
            or input_path.stat().st_size != binding.get("bytes")
        ):
            raise ValueError("recovery input file binding drift")
        rows = read_jsonl(input_path)
        request_ids = [row.get("custom_id") for row in rows]
        if (
            request_ids != binding["request_ids_in_order"]
            or len(rows) != binding["request_count"]
            or request_ids
            != [
                request_id
                for request_id in failed_by_shard[shard_id]
                if request_id in set(binding["request_ids_in_order"])
            ]
        ):
            raise ValueError("recovery failed request order/subset drift")
        provider_body_byte_values = [len(canonical_json(row["body"])) for row in rows]
        provider_body_bytes = sum(provider_body_byte_values)
        expected_direct = (
            float(empirical["source_actual_cost_usd"])
            * len(rows)
            / int(empirical["source_request_count"])
        )
        expected_strict = _strict_exposure_for_provider_bodies(
            config=bundle["config"],
            prices=prices,
            body_bytes=provider_body_byte_values,
        )
        expected_queue = round(
            provider_body_bytes
            * float(empirical["source_input_tokens"])
            / float(empirical["source_provider_body_utf8_bytes"])
        )
        if (
            not math.isclose(
                float(binding.get("direct_v4_cost_forecast_usd", math.nan)),
                expected_direct,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(
                    binding.get("strict_no_cache_full_output_exposure_usd", math.nan)
                ),
                expected_strict,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or binding.get("queued_input_tokens_empirical_forecast") != expected_queue
        ):
            raise ValueError("recovery shard forecast/exposure/queue binding drift")
        expected_direct_total += expected_direct
        expected_strict_total += expected_strict
        for request_id, row in zip(request_ids, rows, strict=True):
            if (
                not isinstance(request_id, str)
                or request_id not in original_by_id
                or original_by_id[request_id][0] != shard_id
                or canonical_json(original_by_id[request_id][1]) != canonical_json(row)
            ):
                raise ValueError("recovery request body or primary-shard binding drift")
            observed_failed.add(request_id)
    if observed_failed != expected_failed:
        raise ValueError(
            "recovery manifest does not exactly cover failed primary requests"
        )
    if (
        manifest.get("schema_version")
        != "adag.process-witness.coarse-production-recovery.v1"
        or manifest.get("status") != "prepared_offline_failed_only"
        or manifest.get("recovery_wave") != 0
        or not math.isclose(
            float(manifest.get("direct_v4_cost_forecast_usd", math.nan)),
            expected_direct_total,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(manifest.get("strict_no_cache_full_output_exposure_usd", math.nan)),
            expected_strict_total,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or manifest.get("successful_requests_rerun") != 0
        or manifest.get("provider_bodies_byte_identical") is not True
        or manifest.get(
            "fresh_recovery_forecast_budget_and_strict_exposure_ack_required"
        )
        is not True
        or manifest.get("additional_recovery_waves_permitted") is not False
    ):
        raise ValueError("recovery manifest forecast/exposure semantics drift")


def _recovery_binding(
    run_root: Path, shard_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    intent, bundle = _campaign(run_root)
    manifest = _load_object(run_root / "recovery-000" / "manifest.json")
    _verify(manifest, "recovery_manifest_sha256", "coarse production recovery")
    _validate_recovery_manifest_inputs(
        run_root=run_root,
        authoritative_bundle_root=Path(intent["bundle_root"]),
        intent=intent,
        bundle=bundle,
        manifest=manifest,
    )
    if manifest["campaign_run_sha256"] != intent["campaign_run_sha256"]:
        raise ValueError("coarse production recovery campaign drift")
    authorization = _load_object(run_root / "recovery-000" / "authorization.json")
    _validate_recovery_authorization(
        authorization=authorization,
        manifest=manifest,
        intent=intent,
    )
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
        collection_locks = _active_collection_locks(run_root)
        if collection_locks:
            raise RuntimeError(
                f"collection materialization is active: {collection_locks}"
            )
        if (shard_root / "submission-intent.json").exists():
            raise FileExistsError(
                "coarse production recovery submission was already attempted"
            )
        _primary_cost, primary_complete = _attempted_primary_forecast(run_root, intent)
        if not primary_complete:
            raise ValueError("primary campaign cost is not fully priced")
        recovery_manifest = _load_object(run_root / "recovery-000" / "manifest.json")
        recovery_authorization = _load_object(
            run_root / "recovery-000" / "authorization.json"
        )
        _validate_recovery_authorization(
            authorization=recovery_authorization,
            manifest=recovery_manifest,
            intent=intent,
        )
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
                    active_queue += int(other["queued_input_tokens_empirical_forecast"])
        candidate_cost = float(binding["direct_v4_cost_forecast_usd"])
        if recovery_cost + candidate_cost > float(
            recovery_authorization["recovery_forecast_budget_usd"]
        ):
            raise ValueError(
                "prospective actual-or-reserved recovery cost exceeds forecast budget"
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
                "prospective_recovery_cost_usd": (recovery_cost + candidate_cost),
                "recovery_authorization_sha256": recovery_authorization[
                    "recovery_authorization_sha256"
                ],
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
    uploader: Callable[[Path], dict[str, Any]] = _upload_provider,
    creator: Callable[..., dict[str, Any]] = _create_provider,
) -> dict[str, Any]:
    intent, _bundle, _binding, shard_root = _recovery_binding(run_root, shard_id)
    with _submission_gate(run_root):
        return _recover_submission_failure(
            shard_root=shard_root,
            intent=intent,
            shard_id=shard_id,
            generation="recovery-000",
            discoverer=discoverer,
            uploader=uploader,
            creator=creator,
        )


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
            return _production_provider_batch_dict(
                _openai_client().batches.retrieve(batch_id)
            )

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


def _collect_recovery_shard_locked(
    *,
    run_root: Path,
    shard_id: str,
    downloader: Callable[
        [str], tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ] = _download,
) -> dict[str, Any]:
    intent, bundle, binding, shard_root = _recovery_binding(run_root, shard_id)
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
    collection_intent = _load_or_create_collection_intent(
        path=shard_root / "collection-intent.json", submission=submission
    )
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
    _write_or_verify_json(raw_root / "provider-snapshot.json", snapshot)
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
        _write_or_verify_bytes(path, item["content"])
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
        events=events,
        snapshot=snapshot,
        prices=prices,
        aggregate_fallback_long_context_impossible=(
            _input_byte_bound_excludes_long_context(
                input_path=shard_root / "input.jsonl",
                config=bundle["config"],
                prices=prices,
            )
        ),
    )
    _write_or_verify_jsonl(shard_root / "events.jsonl", events)
    success = sum(e["validation_status"] == "success" for e in events)
    primary_cost = sum(
        float(_load_object(path)["known_priced_cost_usd"])
        for path in run_root.glob("shards/*/collection.json")
    )
    recovery_cumulative_cost = total + sum(
        float(_load_object(path)["known_priced_cost_usd"])
        for path in run_root.glob("recovery-000/shards/*/collection.json")
        if path.parent.name != shard_id
    )
    campaign_cost = primary_cost + recovery_cumulative_cost
    recovery_authorization = _load_object(
        run_root / "recovery-000" / "authorization.json"
    )
    recovery_manifest = _load_object(run_root / "recovery-000" / "manifest.json")
    _validate_recovery_authorization(
        authorization=recovery_authorization,
        manifest=recovery_manifest,
        intent=intent,
    )
    forecast_budget_exceeded = complete_cost and recovery_cumulative_cost > float(
        recovery_authorization["recovery_forecast_budget_usd"]
    )
    result = _hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "status": (
                "complete_forecast_budget_exceeded_stop_further_recovery_submission"
                if forecast_budget_exceeded
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
            "recovery_cumulative_known_priced_cost_usd": recovery_cumulative_cost,
            "cost_complete": complete_cost,
            "pricing_basis": pricing_basis,
            "provider_terminal_status": terminal_status,
            "forecast_budget_usd": recovery_authorization[
                "recovery_forecast_budget_usd"
            ],
            "forecast_budget_is_hard_spend_cap": False,
            "forecast_budget_exceeded": forecast_budget_exceeded,
            "raw_file_bindings": raw_bindings,
            "events_sha256": file_sha256(shard_root / "events.jsonl"),
        },
        "collection_sha256",
    )
    atomic_write_json(shard_root / "collection.json", result)
    return result


def collect_recovery_shard(
    *,
    run_root: Path,
    shard_id: str,
    downloader: Callable[
        [str], tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ] = _download,
) -> dict[str, Any]:
    shard_root = run_root / "recovery-000" / "shards" / shard_id
    with _collection_gate(shard_root), _submission_gate(run_root):
        return _collect_recovery_shard_locked(
            run_root=run_root,
            shard_id=shard_id,
            downloader=downloader,
        )


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
    shutil.copyfile(
        run_root / "campaign-intent.json", temporary / "campaign-intent.json"
    )
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
            raise ValueError(
                "coarse production final evidence has external symlink"
            ) from error
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


def _validate_readonly_modes(root: Path) -> None:
    if root.stat().st_mode & 0o777 != 0o555:
        raise ValueError("proposal evidence root mode drift")
    for path in root.rglob("*"):
        expected = 0o555 if path.is_dir() else 0o444
        if path.stat().st_mode & 0o777 != expected:
            raise ValueError(f"proposal evidence mode drift: {path.relative_to(root)}")


def load_frozen_proposal_bank(root: Path) -> dict[str, Any]:
    """Strictly validate a finalized proposal bank without its original run roots."""

    _validate_readonly_modes(root)
    manifest = _load_object(root / "manifest.json")
    _verify(manifest, "proposal_bank_manifest_sha256", "proposal bank manifest")
    if (
        manifest.get("schema_version") != "adag.process-witness.coarse-proposal-bank.v1"
        or manifest.get("status") != "frozen_sampling_proposals_not_semantic_truth"
    ):
        raise ValueError("proposal bank manifest semantic drift")
    inventory = _load_object(root / "evidence-inventory.json")
    _verify(inventory, "evidence_inventory_sha256", "proposal evidence inventory")
    if inventory["evidence_inventory_sha256"] != manifest["evidence_inventory_sha256"]:
        raise ValueError("proposal evidence inventory/manifest drift")
    expected = {row["path"]: row for row in inventory["files"]}
    observed = {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file()
        and path not in {root / "manifest.json", root / "evidence-inventory.json"}
    }
    if set(expected) != set(observed):
        raise ValueError("proposal evidence inventory coverage drift")
    for relative, path in observed.items():
        row = expected[relative]
        if (
            file_sha256(path) != row["sha256"]
            or path.stat().st_size != row["bytes"]
            or ("symlink_target" in row) != path.is_symlink()
            or (path.is_symlink() and os.readlink(path) != row.get("symlink_target"))
        ):
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
    if (
        intent.get("schema_version") != CAMPAIGN_RUN_SCHEMA
        or intent.get("status") != "initialized_no_provider_calls"
        or intent["campaign_run_sha256"] != manifest["campaign_run_sha256"]
        or intent.get("forecast_budget_is_hard_spend_cap") is not False
        or intent.get("acknowledged_strict_worst_case_exposure_usd")
        != intent.get("strict_worst_case_exposure_usd")
        or not isinstance(intent.get("strict_exposure_acknowledgement_note"), str)
        or not intent["strict_exposure_acknowledgement_note"].strip()
        or manifest.get("primary_forecast_budget_usd")
        != intent.get("forecast_budget_usd")
        or manifest.get("primary_forecast_budget_is_hard_spend_cap") is not False
        or manifest.get("primary_strict_worst_case_exposure_usd")
        != intent.get("strict_worst_case_exposure_usd")
        or manifest.get("primary_actual_spend_limit_usd")
        != intent.get("primary_actual_spend_limit_usd")
        or manifest.get("authorized_primary_shard_ids")
        != intent.get("authorized_primary_shard_ids")
    ):
        raise ValueError("proposal copied campaign binding drift")
    _validate_primary_authorization_scope(intent, bundle)
    recovery_binding = manifest.get("recovery_authorization")
    recovery_root = root / "recovery-000"
    if (recovery_binding is None) != (not recovery_root.exists()):
        raise ValueError("proposal recovery authorization presence drift")
    if recovery_binding is not None:
        recovery = _load_object(recovery_root / "manifest.json")
        _verify(recovery, "recovery_manifest_sha256", "copied recovery manifest")
        _validate_recovery_manifest_inputs(
            run_root=root,
            authoritative_bundle_root=root / "campaign-bundle",
            intent=intent,
            bundle=bundle,
            manifest=recovery,
        )
        authorization = _load_object(recovery_root / "authorization.json")
        _validate_recovery_authorization(
            authorization=authorization,
            manifest=recovery,
            intent=intent,
        )
        if (
            recovery_binding["recovery_manifest_sha256"]
            != recovery["recovery_manifest_sha256"]
            or recovery_binding["recovery_authorization_sha256"]
            != authorization["recovery_authorization_sha256"]
            or recovery_binding["recovery_forecast_budget_usd"]
            != authorization["recovery_forecast_budget_usd"]
            or recovery_binding["strict_worst_case_exposure_usd"]
            != authorization["strict_worst_case_exposure_usd"]
        ):
            raise ValueError("proposal copied recovery authorization binding drift")
    observed_primary_cost = 0.0
    observed_recovery_cost = 0.0
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
                raise ValueError(
                    f"proposal evidence misses {binding['generation']}/{binding['shard_id']}/{name}"
                )
        collection = _load_object(attempt / "collection.json")
        _verify(collection, "collection_sha256", "copied collection")
        if (
            collection.get("schema_version") != COLLECTION_SCHEMA
            or collection.get("cost_complete") is not True
            or collection.get("status")
            not in {
                "complete",
                "complete_with_failed_requests_recovery_eligible",
                "complete_forecast_budget_exceeded_stop_further_primary_submission",
                "complete_forecast_budget_exceeded_stop_further_recovery_submission",
            }
            or collection.get("provider_terminal_status")
            not in {"completed", "failed", "expired", "cancelled"}
        ):
            raise ValueError("proposal copied collection semantic drift")
        if binding["generation"] == "primary":
            observed_primary_cost += float(collection["known_priced_cost_usd"])
        else:
            observed_recovery_cost += float(collection["known_priced_cost_usd"])
        submission_intent = _load_object(attempt / "submission-intent.json")
        _verify(
            submission_intent,
            "submission_intent_sha256",
            "copied submission intent",
        )
        expected_generation = (
            "primary" if binding["generation"] == "primary" else "recovery-000"
        )
        expected_metadata = _metadata(intent, binding["shard_id"], expected_generation)
        if (
            submission_intent.get("schema_version") != SUBMISSION_SCHEMA
            or submission_intent.get("generation") != expected_generation
            or submission_intent.get("metadata") != expected_metadata
            or file_sha256(attempt / "input.jsonl") != submission_intent["input_sha256"]
        ):
            raise ValueError("proposal copied provider input binding drift")
        upload_path = attempt / "provider-upload-response.json"
        upload = _load_object(upload_path)
        _validate_upload(upload)
        submission = _load_object(attempt / "submission.json")
        _verify(submission, "submission_sha256", "copied submission")
        if (
            submission.get("schema_version") != SUBMISSION_SCHEMA
            or submission.get("status") != "submitted"
            or submission.get("campaign_run_sha256") != intent["campaign_run_sha256"]
            or submission.get("submission_intent_sha256")
            != submission_intent["submission_intent_sha256"]
            or file_sha256(upload_path) != submission["provider_upload_response_sha256"]
        ):
            raise ValueError("proposal copied upload receipt binding drift")
        provider = submission["provider_response"]
        _validate_snapshot(
            provider,
            metadata=expected_metadata,
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
            collection_intent.get("schema_version") != COLLECTION_SCHEMA
            or collection_intent.get("submission_sha256")
            != submission["submission_sha256"]
            or collection_intent.get("batch_id") != provider["batch_id"]
            or collection_intent["collection_intent_sha256"]
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
                raw["path"] not in expected
                or not raw_path.is_file()
                or file_sha256(raw_path) != raw["sha256"]
                or raw_path.stat().st_size != raw["bytes"]
            ):
                raise ValueError("proposal copied raw provider evidence drift")
        raw_snapshot = _load_object(attempt / "raw/provider-snapshot.json")
        _validate_snapshot(
            raw_snapshot,
            metadata=expected_metadata,
            batch_id=provider["batch_id"],
            input_file_id=upload["input_file_id"],
        )
        if raw_snapshot.get("status") != collection["provider_terminal_status"]:
            raise ValueError("proposal copied terminal status binding drift")
        prior = None
        status_paths = sorted((attempt / "status").glob("*.json"))
        if not status_paths:
            raise ValueError("proposal evidence misses provider status receipt")
        for status_path in status_paths:
            status = _load_object(status_path)
            _verify(status, "status_sha256", "copied provider status")
            if (
                status.get("schema_version") != STATUS_SCHEMA
                or status.get("campaign_run_sha256") != intent["campaign_run_sha256"]
                or status["previous_status_sha256"] != prior
                or status["submission_sha256"] != submission["submission_sha256"]
            ):
                raise ValueError("proposal copied provider status chain drift")
            _validate_snapshot(
                status["provider_response"],
                metadata=expected_metadata,
                batch_id=provider["batch_id"],
                input_file_id=upload["input_file_id"],
            )
            prior = status["status_sha256"]
    if (
        manifest.get("primary_actual_cost_usd") != observed_primary_cost
        or manifest.get("recovery_actual_cost_usd") != observed_recovery_cost
        or manifest.get("actual_total_cost_usd")
        != observed_primary_cost + observed_recovery_cost
        or observed_primary_cost
        > float(manifest["primary_strict_worst_case_exposure_usd"])
        or (
            recovery_binding is not None
            and observed_recovery_cost
            > float(recovery_binding["strict_worst_case_exposure_usd"])
        )
    ):
        raise ValueError("proposal copied cost binding drift")
    for filename, field in (
        ("effective-events.jsonl", "effective_events_sha256"),
        ("proposals.jsonl", "proposals_sha256"),
        ("sampling-groups.jsonl", "sampling_groups_sha256"),
    ):
        if file_sha256(root / filename) != manifest[field]:
            raise ValueError(f"proposal final output drift: {filename}")
    return {"manifest": manifest, "inventory": inventory, "bundle": bundle}


def _enforce_actual_within_strict_exposure(
    *, actual_cost_usd: float, strict_exposure_usd: float, generation: str
) -> None:
    if (
        not math.isfinite(actual_cost_usd)
        or not math.isfinite(strict_exposure_usd)
        or actual_cost_usd < 0
        or strict_exposure_usd < 0
    ):
        raise ValueError("coarse production exposure values must be finite nonnegative")
    if actual_cost_usd > strict_exposure_usd:
        raise ValueError(
            f"coarse production {generation} actual cost exceeds acknowledged strict exposure"
        )


def finalize_campaign(*, run_root: Path, destination: Path) -> dict[str, Any]:
    """Union complete primary results and freeze atom proposals plus sampling groups."""

    intent, bundle = _campaign(run_root)
    if intent.get("full_campaign_authorized") is not True:
        raise ValueError(
            "calibration/subset run cannot finalize the full production proposal corpus"
        )
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
    primary_actual_cost = total_cost
    _enforce_actual_within_strict_exposure(
        actual_cost_usd=primary_actual_cost,
        strict_exposure_usd=float(intent["strict_worst_case_exposure_usd"]),
        generation="primary",
    )
    events_by_id = {event["request_id"]: event for event in primary_events}
    recovery_root = run_root / "recovery-000"
    recovery_authorization_binding = None
    recovery_actual_cost = 0.0
    if recovery_root.exists():
        recovery = _load_object(recovery_root / "manifest.json")
        _verify(recovery, "recovery_manifest_sha256", "coarse production recovery")
        recovery_authorization = _load_object(recovery_root / "authorization.json")
        _validate_recovery_authorization(
            authorization=recovery_authorization,
            manifest=recovery,
            intent=intent,
        )
        recovery_authorization_binding = {
            "recovery_manifest_sha256": recovery["recovery_manifest_sha256"],
            "recovery_authorization_sha256": recovery_authorization[
                "recovery_authorization_sha256"
            ],
            "recovery_forecast_budget_usd": recovery_authorization[
                "recovery_forecast_budget_usd"
            ],
            "forecast_budget_is_hard_spend_cap": False,
            "strict_worst_case_exposure_usd": recovery_authorization[
                "strict_worst_case_exposure_usd"
            ],
        }
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
            recovery_actual_cost += float(collection["known_priced_cost_usd"])
            cost_complete = cost_complete and bool(collection["cost_complete"])
            collection_bindings.append(
                {
                    "generation": "recovery-000",
                    "shard_id": shard["shard_id"],
                    "collection_sha256": collection["collection_sha256"],
                    "events_sha256": collection["events_sha256"],
                }
            )
        _enforce_actual_within_strict_exposure(
            actual_cost_usd=recovery_actual_cost,
            strict_exposure_usd=float(
                recovery_authorization["strict_worst_case_exposure_usd"]
            ),
            generation="recovery",
        )
    events = [events_by_id[request_id] for request_id in request_ids]
    if any(event["validation_status"] != "success" for event in events):
        raise ValueError(
            "coarse production finalization requires recovery-resolved success coverage"
        )
    if not cost_complete:
        raise ValueError("coarse production finalization cost is incomplete")
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
            "primary_actual_cost_usd": primary_actual_cost,
            "recovery_actual_cost_usd": recovery_actual_cost,
            "primary_forecast_budget_usd": intent["forecast_budget_usd"],
            "primary_forecast_budget_is_hard_spend_cap": False,
            "primary_strict_worst_case_exposure_usd": intent[
                "strict_worst_case_exposure_usd"
            ],
            "primary_actual_spend_limit_usd": intent["primary_actual_spend_limit_usd"],
            "authorized_primary_shard_ids": intent["authorized_primary_shard_ids"],
            "recovery_authorization": recovery_authorization_binding,
            "effective_events_sha256": file_sha256(
                temporary / "effective-events.jsonl"
            ),
            "proposals_sha256": file_sha256(temporary / "proposals.jsonl"),
            "sampling_groups_sha256": file_sha256(temporary / "sampling-groups.jsonl"),
            "evidence_inventory_sha256": inventory["evidence_inventory_sha256"],
            "collection_bindings": collection_bindings,
            "claim_boundary": bundle["config"]["claim_boundary"],
        },
        "proposal_bank_manifest_sha256",
    )
    atomic_write_json(temporary / "manifest.json", result)
    _readonly_tree(temporary)
    load_frozen_proposal_bank(temporary)
    temporary.rename(destination)
    try:
        load_frozen_proposal_bank(destination)
    except BaseException:
        destination.rename(temporary)
        raise
    return result
