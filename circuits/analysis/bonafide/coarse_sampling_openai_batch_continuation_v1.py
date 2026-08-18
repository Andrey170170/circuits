"""Additive full-corpus continuation after the immutable shard-005 calibration.

The continuation never edits or relabels the calibration run.  It copies that
receipt tree as inherited evidence, extracts only its non-successful request
rows for a calibration-recovery attempt, and repacks only original shards
000--004 into Tier-3-safe tranches.  Every provider JSONL row is copied byte for
byte from the frozen v6 bundle.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 as production_v1
from circuits.analysis.bonafide.canonical import (
    canonical_json,
    canonical_sha256,
    file_sha256,
)
from circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 import (
    COLLECTION_SCHEMA,
    EVENT_SCHEMA,
    STATUS_SCHEMA,
    SUBMISSION_SCHEMA,
    _create_provider,
    _download,
    _parse_row,
    _price_events,
    _production_provider_batch_dict,
    _strict_exposure_for_provider_bodies,
    _upload_provider,
    _validate_snapshot,
    _validate_upload,
)
from circuits.analysis.bonafide.coarse_sampling_production_v1 import (
    load_production_bundle,
    proposal_from_votes,
    sampling_groups,
)
from circuits.labeling.api import openai_usage
from circuits.labeling.io import atomic_write_bytes, atomic_write_json, read_jsonl
from circuits.labeling.pricing import load_price_snapshot

CONTINUATION_SCHEMA = "adag.process-witness.coarse-production-continuation.v1"
FINAL_SCHEMA = "adag.process-witness.coarse-proposal-bank-continuation.v1"
CALIBRATION_RECOVERY_ID = "calibration-recovery-000"
TERMINAL_PROVIDER_STATUSES = {"completed", "failed", "expired", "cancelled"}
CALIBRATION_KNOWN_COST_USD = 3.99951985
CALIBRATION_DIRECT_FORECAST_USD = 3.207300838
CALIBRATION_BODY_BYTES = 102_828_486
CALIBRATION_REQUEST_COUNT = 6_439
CALIBRATION_ALL_INPUT_AND_CACHE_COST_USD = 1.47010105
CALIBRATION_OUTPUT_COST_USD = 2.52941880
CALIBRATED_RESERVATION_MARGIN = 1.25


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def _verify(value: Mapping[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _exact_jsonl_rows(path: Path) -> list[tuple[str, bytes, dict[str, Any]]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"provider input lacks terminal newline: {path}")
    output = []
    for line in raw.splitlines(keepends=True):
        value = json.loads(line)
        request_id = value.get("custom_id") if isinstance(value, Mapping) else None
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"provider input row lacks custom_id: {path}")
        output.append((request_id, line, dict(value)))
    if len({row[0] for row in output}) != len(output):
        raise ValueError(f"duplicate provider input custom_id: {path}")
    return output


def _tree_sha256(root: Path) -> str:
    rows = [
        {
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return canonical_sha256(rows)


def _current_source_revision(*, run_root: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "circuits/analysis/bonafide/coarse_sampling_openai_batch_continuation_v1.py",
        "scripts/bonafide/process_witness_coarse_openai_batch_continuation_v1.py",
        "pyproject.toml",
        "uv.lock",
    )
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status:
        raise ValueError(
            "continuation production preparation requires a clean tracked worktree"
        )
    snapshot_root = run_root / "execution-source"
    files = []
    for relative in relative_paths:
        source = repo_root / relative
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repo_root,
            capture_output=True,
        )
        if tracked.returncode != 0:
            raise ValueError(f"continuation source is not Git tracked: {relative}")
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        if source.read_bytes() != committed:
            raise ValueError(f"continuation source differs from HEAD: {relative}")
        git_blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(source),
                "bytes": source.stat().st_size,
                "git_blob": git_blob,
                "snapshot_path": str(destination.relative_to(run_root)),
            }
        )
    archive = subprocess.run(
        ["git", "archive", "--format=tar", git_head],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    archive_path = snapshot_root / "repository.tar"
    atomic_write_bytes(archive_path, archive)
    return {
        "git_head": git_head,
        "git_commit": git_head,
        "git_tree": git_tree,
        "tracked_worktree_clean": True,
        "tracked_worktree_status": tracked_status,
        "tracked_worktree_status_sha256": hashlib.sha256(
            tracked_status.encode("utf-8")
        ).hexdigest(),
        "files": files,
        "repository_archive_path": str(archive_path.relative_to(run_root)),
        "repository_archive_sha256": file_sha256(archive_path),
        "repository_archive_bytes": archive_path.stat().st_size,
        "snapshot_tree_sha256": _tree_sha256(snapshot_root),
    }


def _copy_calibration_evidence(*, source: Path, destination: Path) -> str:
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    return _tree_sha256(destination)


def _validated_calibration_source(
    *, bundle_root: Path, calibration_run_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    intent, bundle = production_v1._campaign(calibration_run_root)
    if Path(intent["bundle_root"]).resolve() != bundle_root.resolve():
        raise ValueError("calibration continuation bundle-root drift")
    if intent.get("authorized_primary_shard_ids") != ["shard-005"]:
        raise ValueError("continuation requires the shard-005-only calibration")
    shard_root = calibration_run_root / "shards/shard-005"
    collection = _load_object(shard_root / "collection.json")
    production_v1._verify(collection, "collection_sha256", "calibration collection")
    events_path = shard_root / "events.jsonl"
    if (
        file_sha256(events_path) != collection.get("events_sha256")
        or collection.get("request_count") != 6447
        or collection.get("success_count") != 6433
        or collection.get("failure_count") != 14
    ):
        raise ValueError("calibration collection census or event binding drift")
    frozen = next(s for s in bundle["shards"] if s["shard_id"] == "shard-005")
    if (
        file_sha256(shard_root / "input.jsonl") != frozen["sha256"]
        or (shard_root / "input.jsonl").read_bytes()
        != (bundle_root / frozen["path"]).read_bytes()
    ):
        raise ValueError("calibration provider input differs from frozen v6 bytes")
    return intent, bundle


def _validate_calibration_failure_ids(
    *,
    ordered_primary_ids: Sequence[str],
    events: Sequence[Mapping[str, Any]],
    recovery_ids: Sequence[str],
) -> None:
    event_by_id = {str(row.get("request_id")): row for row in events}
    expected = [
        request_id
        for request_id in ordered_primary_ids
        if request_id in event_by_id
        and event_by_id[request_id].get("validation_status") != "success"
    ]
    if (
        len(event_by_id) != len(events)
        or set(event_by_id) != set(ordered_primary_ids)
        or list(recovery_ids) != expected
    ):
        raise ValueError(
            "calibration recovery must contain exactly the non-successful requests"
        )


def _reconcile_inherited_calibration_cost(
    *, calibration_run_root: Path, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Prove the eight credit failures contributed zero aggregate usage."""

    snapshot_path = (
        calibration_run_root / "shards/shard-005/raw/provider-snapshot.json"
    )
    snapshot = _load_object(snapshot_path)
    counts = snapshot.get("request_counts")
    aggregate = snapshot.get("usage")
    details = aggregate.get("input_tokens_details") if isinstance(aggregate, Mapping) else None
    output_details = aggregate.get("output_tokens_details") if isinstance(aggregate, Mapping) else None
    credit_failures = [
        event
        for event in events
        if event.get("validation_status") == "provider_error"
        and event.get("provider_error_code") == "credit_balance_exhausted"
    ]
    usage_bearing = [
        event for event in events if event.get("validation_status") != "provider_error"
    ]
    sums = {
        field: sum(int(event.get("usage", {}).get(field) or 0) for event in usage_bearing)
        for field in (
            "input_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "uncached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
        )
    }
    if (
        snapshot.get("status") != "completed"
        or not isinstance(counts, Mapping)
        or counts.get("total") != len(events)
        or counts.get("completed") != len(usage_bearing)
        or counts.get("failed") != len(credit_failures)
        or len(credit_failures) != 8
        or len(usage_bearing) != 6439
        or any(
            event.get("usage", {}).get("input_tokens") is not None
            or event.get("usage", {}).get("output_tokens") is not None
            for event in credit_failures
        )
        or not isinstance(aggregate, Mapping)
        or not isinstance(details, Mapping)
        or not isinstance(output_details, Mapping)
        or aggregate.get("input_tokens") != sums["input_tokens"]
        or details.get("cached_tokens") != sums["cache_read_tokens"]
        or aggregate.get("output_tokens") != sums["output_tokens"]
        or output_details.get("reasoning_tokens") != sums["reasoning_tokens"]
        or sums["input_tokens"]
        != sums["cache_read_tokens"]
        + sums["cache_write_tokens"]
        + sums["uncached_input_tokens"]
    ):
        raise ValueError("inherited calibration zero-usage cost reconciliation failed")
    return _hashed(
        {
            "schema_version": "adag.process-witness.coarse-calibration-cost-reconciliation.v1",
            "status": "aggregate_exact_credit_failures_zero_usage",
            "created_at": _now(),
            "provider_snapshot_sha256": file_sha256(snapshot_path),
            "usage_bearing_request_count": len(usage_bearing),
            "credit_balance_exhausted_request_count": len(credit_failures),
            "aggregate_usage_equals_usage_bearing_row_sums": True,
            "credit_failures_zero_provider_usage": True,
            "adopted_actual_cost_usd": CALIBRATION_KNOWN_COST_USD,
            "cost_complete": True,
            "usage_sums": sums,
        },
        "cost_reconciliation_sha256",
    )


def _provider_body_bytes(row: Mapping[str, Any]) -> int:
    body = row.get("body")
    if not isinstance(body, Mapping):
        raise ValueError("provider input body drift")
    return len(canonical_json(body))


def _queue_tokens(*, body_bytes: int, bundle: Mapping[str, Any]) -> int:
    empirical = bundle["config"]["empirical_calibration"]
    return math.ceil(
        body_bytes
        * float(empirical["source_input_tokens"])
        / float(empirical["source_provider_body_utf8_bytes"])
    )


def _cost_metrics(
    *, bundle_root: Path, bundle: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    body_bytes = [_provider_body_bytes(row) for row in rows]
    empirical = bundle["config"]["empirical_calibration"]
    direct = (
        float(empirical["source_actual_cost_usd"])
        * len(rows)
        / int(empirical["source_request_count"])
    )
    prices = load_price_snapshot(bundle_root / "price-snapshot.json")
    strict = _strict_exposure_for_provider_bodies(
        config=bundle["config"], prices=prices, body_bytes=body_bytes
    )
    raw_calibrated = (
        CALIBRATION_ALL_INPUT_AND_CACHE_COST_USD
        * sum(body_bytes)
        / CALIBRATION_BODY_BYTES
        + CALIBRATION_OUTPUT_COST_USD
        * len(rows)
        / CALIBRATION_REQUEST_COUNT
    )
    scaled_direct = direct * (
        CALIBRATION_KNOWN_COST_USD / CALIBRATION_DIRECT_FORECAST_USD
    )
    return {
        "direct_v4_cost_forecast_usd": direct,
        "strict_no_cache_full_output_exposure_usd": strict,
        "calibrated_cost_reservation_usd": max(raw_calibrated, scaled_direct)
        * CALIBRATED_RESERVATION_MARGIN,
    }


def _attempt_binding(
    *,
    run_root: Path,
    bundle_root: Path,
    bundle: Mapping[str, Any],
    attempt_id: str,
    generation: str,
    source_shard_ids: Sequence[str],
    exact_rows: Sequence[tuple[str, bytes, dict[str, Any]]],
) -> dict[str, Any]:
    attempt_root = run_root / "attempts" / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=True)
    input_path = attempt_root / "input.jsonl"
    production_v1._write_or_verify_bytes(
        input_path, b"".join(row[1] for row in exact_rows)
    )
    values = [row[2] for row in exact_rows]
    body_bytes = sum(_provider_body_bytes(row) for row in values)
    overhead = int(bundle["config"]["provider"]["input_token_overhead_per_request"])
    metrics = _cost_metrics(bundle_root=bundle_root, bundle=bundle, rows=values)
    binding = {
        "attempt_id": attempt_id,
        "generation": generation,
        "input_relative_path": str(input_path.relative_to(run_root)),
        "input_sha256": file_sha256(input_path),
        "bytes": input_path.stat().st_size,
        "request_count": len(exact_rows),
        "request_ids_in_order": [row[0] for row in exact_rows],
        "source_shard_ids": list(source_shard_ids),
        "provider_body_utf8_bytes": body_bytes,
        "queued_input_tokens_empirical_forecast": _queue_tokens(
            body_bytes=body_bytes, bundle=bundle
        ),
        "strict_input_token_byte_upper_bound": body_bytes + overhead * len(values),
        **metrics,
    }
    production_v1._write_or_verify_json(
        attempt_root / "binding.json", _hashed(binding, "binding_sha256")
    )
    return binding


def _pack_response_affinity(
    *,
    source_rows: Sequence[tuple[str, bytes, dict[str, Any]]],
    request_metadata: Mapping[str, Mapping[str, Any]],
    bundle: Mapping[str, Any],
    queue_cap: int,
) -> list[list[tuple[str, bytes, dict[str, Any]]]]:
    blocks: list[list[tuple[str, bytes, dict[str, Any]]]] = []
    current_response = None
    for row in source_rows:
        response_id = request_metadata[row[0]]["response_id"]
        if response_id != current_response:
            blocks.append([])
            current_response = response_id
        blocks[-1].append(row)
    output: list[list[tuple[str, bytes, dict[str, Any]]]] = []
    for block in blocks:
        block_tokens = _queue_tokens(
            body_bytes=sum(_provider_body_bytes(row[2]) for row in block), bundle=bundle
        )
        if block_tokens > queue_cap:
            raise ValueError(
                "one response-affinity block exceeds the empirical queue tranche cap"
            )
        if not output:
            output.append([])
        candidate = output[-1] + block
        candidate_tokens = _queue_tokens(
            body_bytes=sum(_provider_body_bytes(row[2]) for row in candidate),
            bundle=bundle,
        )
        if output[-1] and candidate_tokens > queue_cap:
            output.append([])
        output[-1].extend(block)
    return output


def prepare_continuation(
    *,
    bundle_root: Path,
    calibration_run_root: Path,
    run_root: Path,
    provider_queued_input_token_limit: int,
    tranche_empirical_queue_cap: int,
    maximum_concurrent_attempts: int,
    authorized_forecast_budget_usd: float,
    warning_spend_threshold_usd: float,
    hard_campaign_stop_usd: float,
    authorization_note: str,
    calibration_observed_input_tokens: int,
    calibration_forecast_input_tokens: int,
) -> dict[str, Any]:
    """Prepare a network-free, non-mutating continuation artifact."""

    intent, bundle = _validated_calibration_source(
        bundle_root=bundle_root, calibration_run_root=calibration_run_root
    )
    if run_root.exists():
        raise FileExistsError(f"continuation run exists: {run_root}")
    if (
        provider_queued_input_token_limit < 1
        or tranche_empirical_queue_cap < 1
        or tranche_empirical_queue_cap > provider_queued_input_token_limit
        or maximum_concurrent_attempts != 1
        or calibration_forecast_input_tokens < 1
        or calibration_observed_input_tokens < 1
    ):
        raise ValueError("invalid continuation queue authorization")
    if (
        not math.isfinite(authorized_forecast_budget_usd)
        or not math.isfinite(warning_spend_threshold_usd)
        or not math.isfinite(hard_campaign_stop_usd)
        or authorized_forecast_budget_usd <= 0
        or warning_spend_threshold_usd <= 0
        or hard_campaign_stop_usd <= warning_spend_threshold_usd
        or authorized_forecast_budget_usd > hard_campaign_stop_usd
        or not authorization_note.strip()
    ):
        raise ValueError("invalid continuation spend authorization")
    run_root.mkdir(parents=True)
    try:
        inherited_tree = _copy_calibration_evidence(
            source=calibration_run_root,
            destination=run_root / "inherited-calibration-run",
        )
        collection = _load_object(
            calibration_run_root / "shards/shard-005/collection.json"
        )
        events = read_jsonl(calibration_run_root / "shards/shard-005/events.jsonl")
        cost_reconciliation = _reconcile_inherited_calibration_cost(
            calibration_run_root=calibration_run_root, events=events
        )
        atomic_write_json(
            run_root / "inherited-cost-reconciliation.json", cost_reconciliation
        )
        request_metadata = {
            str(row["request_id"]): row for row in bundle["request_index"]
        }
        shard_by_id = {str(row["shard_id"]): row for row in bundle["shards"]}
        calibration_rows = _exact_jsonl_rows(
            bundle_root / shard_by_id["shard-005"]["path"]
        )
        failure_ids = [
            str(event["request_id"])
            for event in events
            if event.get("validation_status") != "success"
        ]
        _validate_calibration_failure_ids(
            ordered_primary_ids=[row[0] for row in calibration_rows],
            events=events,
            recovery_ids=failure_ids,
        )
        failure_set = set(failure_ids)
        recovery_rows = [row for row in calibration_rows if row[0] in failure_set]
        if [row[0] for row in recovery_rows] != failure_ids:
            raise ValueError("inherited calibration failure order drift")
        remaining_rows: list[tuple[str, bytes, dict[str, Any]]] = []
        remaining_source_by_id: dict[str, str] = {}
        for shard in bundle["shards"]:
            shard_id = str(shard["shard_id"])
            if shard_id == "shard-005":
                continue
            for row in _exact_jsonl_rows(bundle_root / shard["path"]):
                remaining_rows.append(row)
                remaining_source_by_id[row[0]] = shard_id
        tranches = []
        packed = _pack_response_affinity(
            source_rows=remaining_rows,
            request_metadata=request_metadata,
            bundle=bundle,
            queue_cap=tranche_empirical_queue_cap,
        )
        for index, rows in enumerate(packed):
            source_ids = list(
                dict.fromkeys(remaining_source_by_id[row[0]] for row in rows)
            )
            binding = _attempt_binding(
                run_root=run_root,
                bundle_root=bundle_root,
                bundle=bundle,
                attempt_id=f"primary-tranche-{index:03d}",
                generation="continuation-primary",
                source_shard_ids=source_ids,
                exact_rows=rows,
            )
            if binding["queued_input_tokens_empirical_forecast"] > tranche_empirical_queue_cap:
                raise ValueError("continuation tranche exceeds empirical queue cap")
            tranches.append(binding)
        attempts = tranches
        direct_forecast = sum(
            float(row["direct_v4_cost_forecast_usd"]) for row in attempts
        ) + float(collection["known_priced_cost_usd"])
        calibrated_forecast = sum(
            float(row["calibrated_cost_reservation_usd"]) for row in attempts
        ) + float(collection["known_priced_cost_usd"])
        if direct_forecast > authorized_forecast_budget_usd:
            raise ValueError("continuation direct forecast exceeds authorization")
        if calibrated_forecast > hard_campaign_stop_usd:
            raise ValueError("continuation calibrated reservation forecast exceeds hard stop")
        actual_ratio = calibration_observed_input_tokens / calibration_forecast_input_tokens
        manifest = _hashed(
            {
                "schema_version": CONTINUATION_SCHEMA,
                "status": "prepared_offline_no_provider_calls",
                "created_at": _now(),
                "bundle_root": str(bundle_root.resolve()),
                "bundle_manifest_sha256": bundle["manifest"]["manifest_sha256"],
                "calibration_run_root": str(calibration_run_root.resolve()),
                "calibration_campaign_run_sha256": intent["campaign_run_sha256"],
                "calibration_collection_sha256": collection["collection_sha256"],
                "calibration_events_sha256": collection["events_sha256"],
                "inherited_calibration_tree_sha256": inherited_tree,
                "calibration_request_count": len(events),
                "calibration_success_count": len(events) - len(failure_ids),
                "calibration_failure_count": len(failure_ids),
                "calibration_known_priced_cost_usd": collection[
                    "known_priced_cost_usd"
                ],
                "inherited_cost_reconciliation_sha256": cost_reconciliation[
                    "cost_reconciliation_sha256"
                ],
                "calibration_cost_complete": True,
                "calibration_queue_forecast_input_tokens": calibration_forecast_input_tokens,
                "calibration_queue_observed_input_tokens": calibration_observed_input_tokens,
                "calibration_queue_actual_to_forecast_ratio": actual_ratio,
                "queue_forecast_underprediction_fraction": actual_ratio - 1.0,
                "provider_queued_input_token_limit": provider_queued_input_token_limit,
                "tranche_empirical_queue_cap": tranche_empirical_queue_cap,
                "queue_headroom_fraction": 1
                - tranche_empirical_queue_cap / provider_queued_input_token_limit,
                "maximum_concurrent_attempts": maximum_concurrent_attempts,
                "queue_rejection_policy": "receipt_bound_stop_no_further_submissions",
                "authorized_forecast_budget_usd": authorized_forecast_budget_usd,
                "warning_spend_threshold_usd": warning_spend_threshold_usd,
                "hard_campaign_stop_usd": hard_campaign_stop_usd,
                "hard_stop_submission_rule": "reject known_actual >= stop or known_actual_plus_reservations > stop",
                "one_active_attempt_bounded_overshoot_possible": True,
                "authorization_note": authorization_note,
                "direct_forecast_including_inherited_known_cost_usd": direct_forecast,
                "calibrated_reservation_forecast_including_inherited_actual_usd": (
                    calibrated_forecast
                ),
                "inherited_failure_request_ids": failure_ids,
                "deferred_failed_only_recovery_required": True,
                "maximum_failed_only_recovery_waves": 1,
                "primary_tranches": tranches,
                "attempts": attempts,
                "remaining_original_shard_ids": [
                    str(shard["shard_id"])
                    for shard in bundle["shards"]
                    if shard["shard_id"] != "shard-005"
                ],
                "request_bytes_and_custom_ids_preserved": True,
                "response_affinity_preserved": True,
                "network_calls_made": 0,
                "execution_source_revision": _current_source_revision(
                    run_root=run_root
                ),
                "environment": {
                    "python_version": platform.python_version(),
                    "openai_sdk_version": importlib.metadata.version("openai"),
                    "openai_project_sha256": (
                        hashlib.sha256(os.environ["OPENAI_PROJECT_ID"].encode()).hexdigest()
                        if os.environ.get("OPENAI_PROJECT_ID")
                        else None
                    ),
                    "openai_organization_sha256": (
                        hashlib.sha256(os.environ["OPENAI_ORG_ID"].encode()).hexdigest()
                        if os.environ.get("OPENAI_ORG_ID")
                        else None
                    ),
                },
            },
            "continuation_manifest_sha256",
        )
        atomic_write_json(run_root / "continuation-manifest.json", manifest)
        _append_cost_status(
            run_root=run_root, manifest=manifest, trigger="inherited_calibration_adopted"
        )
        _load_continuation(run_root)
        return manifest
    except BaseException:
        shutil.rmtree(run_root, ignore_errors=True)
        raise


def _load_continuation(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_object(run_root / "continuation-manifest.json")
    _verify(manifest, "continuation_manifest_sha256", "continuation manifest")
    if (
        manifest.get("schema_version") != CONTINUATION_SCHEMA
        or manifest.get("status") != "prepared_offline_no_provider_calls"
    ):
        raise ValueError("unsupported continuation manifest")
    bundle_root = Path(manifest["bundle_root"])
    bundle = load_production_bundle(bundle_root, load_units=False)
    if bundle["manifest"]["manifest_sha256"] != manifest["bundle_manifest_sha256"]:
        raise ValueError("continuation bundle binding drift")
    repo_root = Path(__file__).resolve().parents[3]
    for source in bundle["manifest"]["source_revision"]["files"]:
        path = repo_root / source["path"]
        if not path.is_file() or file_sha256(path) != source["sha256"]:
            raise ValueError(f"frozen v6 source revision drift: {source['path']}")
    execution_revision = manifest.get("execution_source_revision", {})
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_tracked_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    archive_path = run_root / str(execution_revision.get("repository_archive_path", ""))
    if (
        not isinstance(execution_revision.get("git_head"), str)
        or current_head != execution_revision.get("git_head")
        or current_head != execution_revision.get("git_commit")
        or current_tree != execution_revision.get("git_tree")
        or current_tracked_status
        or execution_revision.get("tracked_worktree_clean") is not True
        or execution_revision.get("tracked_worktree_status") != ""
        or hashlib.sha256(
            str(execution_revision.get("tracked_worktree_status", "")).encode("utf-8")
        ).hexdigest()
        != execution_revision.get("tracked_worktree_status_sha256")
        or not archive_path.is_file()
        or file_sha256(archive_path)
        != execution_revision.get("repository_archive_sha256")
        or archive_path.stat().st_size
        != execution_revision.get("repository_archive_bytes")
        or _tree_sha256(run_root / "execution-source")
        != execution_revision.get("snapshot_tree_sha256")
    ):
        raise ValueError("continuation execution-source provenance drift")
    for source in execution_revision.get("files", []):
        path = repo_root / source["path"]
        snapshot = run_root / source["snapshot_path"]
        current_blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{source['path']}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if (
            not path.is_file()
            or not snapshot.is_file()
            or file_sha256(path) != source["sha256"]
            or file_sha256(snapshot) != source["sha256"]
            or path.stat().st_size != source["bytes"]
            or snapshot.stat().st_size != source["bytes"]
            or current_blob != source["git_blob"]
        ):
            raise ValueError(f"continuation executing source drift: {source['path']}")
    if _tree_sha256(run_root / "inherited-calibration-run") != manifest[
        "inherited_calibration_tree_sha256"
    ]:
        raise ValueError("inherited calibration evidence drift")
    inherited_root = run_root / "inherited-calibration-run/shards/shard-005"
    inherited_collection = _load_object(inherited_root / "collection.json")
    production_v1._verify(
        inherited_collection, "collection_sha256", "inherited calibration collection"
    )
    inherited_events = read_jsonl(inherited_root / "events.jsonl")
    if (
        inherited_collection["collection_sha256"]
        != manifest["calibration_collection_sha256"]
        or file_sha256(inherited_root / "events.jsonl")
        != manifest["calibration_events_sha256"]
        or len(inherited_events) != manifest["calibration_request_count"]
        or sum(row.get("validation_status") == "success" for row in inherited_events)
        != manifest["calibration_success_count"]
        or sum(row.get("validation_status") != "success" for row in inherited_events)
        != manifest["calibration_failure_count"]
    ):
        raise ValueError("inherited calibration census drift")
    reconciliation = _load_object(run_root / "inherited-cost-reconciliation.json")
    _verify(
        reconciliation,
        "cost_reconciliation_sha256",
        "inherited cost reconciliation",
    )
    if (
        reconciliation["cost_reconciliation_sha256"]
        != manifest["inherited_cost_reconciliation_sha256"]
        or reconciliation.get("cost_complete") is not True
        or not math.isclose(
            float(reconciliation.get("adopted_actual_cost_usd", math.nan)),
            float(manifest["calibration_known_priced_cost_usd"]),
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("inherited calibration cost reconciliation drift")
    environment = manifest["environment"]
    project = os.environ.get("OPENAI_PROJECT_ID")
    organization = os.environ.get("OPENAI_ORG_ID")
    if (
        environment.get("python_version") != platform.python_version()
        or environment.get("openai_sdk_version") != importlib.metadata.version("openai")
        or environment.get("openai_project_sha256")
        != (hashlib.sha256(project.encode()).hexdigest() if project else None)
        or environment.get("openai_organization_sha256")
        != (hashlib.sha256(organization.encode()).hexdigest() if organization else None)
    ):
        raise ValueError("continuation runtime environment drift")
    bundle_root = Path(manifest["bundle_root"])
    source_line_sha: dict[str, str] = {}
    source_shard: dict[str, str] = {}
    for shard in bundle["shards"]:
        for request_id, raw_line, _value in _exact_jsonl_rows(bundle_root / shard["path"]):
            source_line_sha[request_id] = hashlib.sha256(raw_line).hexdigest()
            source_shard[request_id] = str(shard["shard_id"])
    inherited_ids = [
        str(row["request_id"])
        for row in inherited_events
        if row.get("validation_status") != "success"
    ]
    if inherited_ids != manifest["inherited_failure_request_ids"]:
        raise ValueError("inherited calibration failure identity drift")
    known_ids: set[str] = set()
    response_attempt: dict[str, str] = {}
    request_metadata = {
        str(row["request_id"]): row for row in bundle["request_index"]
    }
    for binding in manifest["attempts"]:
        attempt_root = run_root / "attempts" / binding["attempt_id"]
        retained = _load_object(attempt_root / "binding.json")
        _verify(retained, "binding_sha256", "continuation attempt binding")
        if {k: retained[k] for k in binding} != binding:
            raise ValueError("continuation retained attempt binding drift")
        input_path = run_root / binding["input_relative_path"]
        rows = _exact_jsonl_rows(input_path)
        if (
            file_sha256(input_path) != binding["input_sha256"]
            or [row[0] for row in rows] != binding["request_ids_in_order"]
            or binding["queued_input_tokens_empirical_forecast"]
            > manifest["tranche_empirical_queue_cap"]
            or known_ids.intersection(binding["request_ids_in_order"])
        ):
            raise ValueError("continuation attempt input or identity drift")
        if any(
            request_id not in source_line_sha
            or hashlib.sha256(raw_line).hexdigest() != source_line_sha[request_id]
            or source_shard[request_id] == "shard-005"
            for request_id, raw_line, _value in rows
        ):
            raise ValueError("continuation input is not byte-identical to v6 source")
        for request_id, _raw_line, _value in rows:
            response_id = str(request_metadata[request_id]["response_id"])
            previous_attempt = response_attempt.setdefault(
                response_id, str(binding["attempt_id"])
            )
            if previous_attempt != binding["attempt_id"]:
                raise ValueError("continuation response affinity was split")
        known_ids.update(binding["request_ids_in_order"])
    expected_primary = {
        request_id for request_id, shard_id in source_shard.items() if shard_id != "shard-005"
    }
    if known_ids != expected_primary:
        raise ValueError("continuation primary exact-union drift")
    recovery_path = run_root / "failed-only-recovery/manifest.json"
    if recovery_path.exists():
        recovery = _load_object(recovery_path)
        _verify(recovery, "recovery_manifest_sha256", "failed-only recovery manifest")
        recovery_intent = _load_object(
            recovery_path.parent / "preparation-intent.json"
        )
        _verify(
            recovery_intent,
            "recovery_intent_sha256",
            "failed-only recovery intent",
        )
        if (
            recovery.get("recovery_intent_sha256")
            != recovery_intent["recovery_intent_sha256"]
            or recovery_intent.get("continuation_manifest_sha256")
            != manifest["continuation_manifest_sha256"]
        ):
            raise ValueError("failed-only recovery intent binding drift")
        binding = recovery.get("attempt")
        if not isinstance(binding, Mapping):
            raise ValueError("failed-only recovery attempt binding drift")
        retained = _load_object(
            run_root / "attempts" / str(binding["attempt_id"]) / "binding.json"
        )
        _verify(retained, "binding_sha256", "failed-only recovery binding")
        if {key: retained[key] for key in binding} != dict(binding):
            raise ValueError("failed-only recovery retained binding drift")
        recovery_rows = _exact_jsonl_rows(
            run_root / str(binding["input_relative_path"])
        )
        expected_failed = list(manifest["inherited_failure_request_ids"])
        for primary in manifest["attempts"]:
            collection_path = (
                run_root / "attempts" / primary["attempt_id"] / "collection.json"
            )
            if not collection_path.exists():
                raise ValueError("recovery exists before every primary collection")
            collection = _load_object(collection_path)
            _verify(collection, "collection_sha256", "primary collection for recovery")
            if collection.get("cost_complete") is not True:
                raise ValueError("recovery source primary cost is incomplete")
            events_path = collection_path.parent / "events.jsonl"
            if file_sha256(events_path) != collection.get("events_sha256"):
                raise ValueError("recovery source primary event drift")
            expected_failed.extend(
                str(event["request_id"])
                for event in read_jsonl(events_path)
                if event.get("validation_status") != "success"
            )
        request_order = [str(row["request_id"]) for row in bundle["request_index"]]
        expected_failed_set = set(expected_failed)
        ordered_expected = [
            request_id for request_id in request_order if request_id in expected_failed_set
        ]
        if (
            recovery.get("continuation_manifest_sha256")
            != manifest["continuation_manifest_sha256"]
            or recovery.get("request_count") != len(ordered_expected)
            or recovery.get("successful_requests_rerun") != 0
            or recovery.get("additional_recovery_waves_permitted") is not False
            or [row[0] for row in recovery_rows] != ordered_expected
            or any(
                hashlib.sha256(raw_line).hexdigest() != source_line_sha[request_id]
                for request_id, raw_line, _row in recovery_rows
            )
        ):
            raise ValueError("failed-only recovery exact coverage or byte drift")
    cost_paths = sorted((run_root / "cost-status").glob("receipt-*.json"))
    if not cost_paths:
        raise ValueError("continuation cost-status chain is absent")
    previous = None
    sticky = False
    for path in cost_paths:
        status = _load_object(path)
        _verify(status, "cost_status_sha256", "continuation cost status")
        if (
            status.get("continuation_manifest_sha256")
            != manifest["continuation_manifest_sha256"]
            or status.get("previous_cost_status_sha256") != previous
            or (sticky and status.get("warning_threshold_reached") is not True)
        ):
            raise ValueError("continuation cost-status chain drift")
        sticky = sticky or bool(status.get("warning_threshold_reached"))
        previous = status["cost_status_sha256"]
    _validate_inherited_calibration_evidence(
        run_root=run_root,
        bundle=bundle,
        manifest=manifest,
    )
    return manifest, bundle


def _all_attempts(
    run_root: Path, manifest: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    attempts = list(manifest["attempts"])
    recovery_path = run_root / "failed-only-recovery/manifest.json"
    if recovery_path.exists():
        recovery = _load_object(recovery_path)
        _verify(recovery, "recovery_manifest_sha256", "failed-only recovery manifest")
        recovery_intent = _load_object(
            recovery_path.parent / "preparation-intent.json"
        )
        _verify(
            recovery_intent,
            "recovery_intent_sha256",
            "failed-only recovery intent",
        )
        if (
            recovery.get("continuation_manifest_sha256")
            != manifest["continuation_manifest_sha256"]
            or recovery.get("recovery_intent_sha256")
            != recovery_intent["recovery_intent_sha256"]
            or recovery.get("recovery_wave") != 0
            or recovery.get("additional_recovery_waves_permitted") is not False
        ):
            raise ValueError("failed-only recovery manifest drift")
        attempts.append(recovery["attempt"])
    return attempts


def _prepare_failed_only_recovery_under_gate(*, run_root: Path) -> dict[str, Any]:
    """Freeze the single recovery after every continuation primary is collected."""

    manifest, bundle = _load_continuation(run_root)
    recovery_root = run_root / "failed-only-recovery"
    if (recovery_root / "manifest.json").exists():
        raise FileExistsError("failed-only recovery is already prepared")
    _inherited_collection, inherited_events = _validate_inherited_calibration_evidence(
        run_root=run_root, bundle=bundle, manifest=manifest
    )
    failed = [
        str(row["request_id"])
        for row in inherited_events
        if row.get("validation_status") != "success"
    ]
    if failed != list(manifest["inherited_failure_request_ids"]):
        raise ValueError("failed-only recovery inherited failure identity drift")
    successful = {
        str(row["request_id"])
        for row in inherited_events
        if row.get("validation_status") == "success"
    }
    for binding in manifest["attempts"]:
        collection_path = (
            run_root / "attempts" / binding["attempt_id"] / "collection.json"
        )
        if not collection_path.is_file():
            raise ValueError("all continuation primaries must be collected before recovery")
        _collection, attempt_events = _validate_collected_attempt(
            run_root=run_root,
            manifest=manifest,
            bundle=bundle,
            binding=binding,
        )
        for event in attempt_events:
            request_id = str(event["request_id"])
            if event.get("validation_status") == "success":
                successful.add(request_id)
            else:
                failed.append(request_id)
    failed_set = set(failed)
    if not failed_set or failed_set.intersection(successful):
        raise ValueError("failed-only recovery overlaps a successful request")
    source_by_id: dict[str, tuple[str, bytes, dict[str, Any]]] = {}
    for shard in bundle["shards"]:
        for request_id, raw_line, row in _exact_jsonl_rows(
            Path(manifest["bundle_root"]) / shard["path"]
        ):
            source_by_id[request_id] = (str(shard["shard_id"]), raw_line, row)
    request_order = [str(row["request_id"]) for row in bundle["request_index"]]
    ordered_failed = [request_id for request_id in request_order if request_id in failed_set]
    if set(ordered_failed) != failed_set:
        raise ValueError("failed-only recovery source coverage drift")
    rows = [
        (request_id, source_by_id[request_id][1], source_by_id[request_id][2])
        for request_id in ordered_failed
    ]
    values = [row[2] for row in rows]
    body_bytes = sum(_provider_body_bytes(row) for row in values)
    queued_tokens = _queue_tokens(body_bytes=body_bytes, bundle=bundle)
    if queued_tokens > manifest["tranche_empirical_queue_cap"]:
        raise ValueError(
            "failed-only recovery exceeds queue cap; stop and prepare a separately "
            "reviewed version"
        )
    recovery_root.mkdir(exist_ok=True)
    intent_payload = {
        "schema_version": "adag.process-witness.coarse-continuation-recovery-intent.v1",
        "status": "validated_sources_before_materialization",
        "continuation_manifest_sha256": manifest["continuation_manifest_sha256"],
        "request_ids_in_order": ordered_failed,
        "source_event_sha256s": [
            manifest["calibration_events_sha256"],
            *[
                _load_object(
                    run_root
                    / "attempts"
                    / str(primary["attempt_id"])
                    / "collection.json"
                )["events_sha256"]
                for primary in manifest["attempts"]
            ],
        ],
        "queued_input_tokens_empirical_forecast": queued_tokens,
    }
    preparation_intent_path = recovery_root / "preparation-intent.json"
    if preparation_intent_path.exists():
        preparation_intent = _load_object(preparation_intent_path)
        _verify(
            preparation_intent,
            "recovery_intent_sha256",
            "failed-only recovery intent",
        )
        retained_payload = {
            key: preparation_intent.get(key) for key in intent_payload
        }
        if retained_payload != intent_payload:
            raise ValueError("failed-only recovery retained intent drift")
    else:
        preparation_intent = _hashed(
            {**intent_payload, "created_at": _now()},
            "recovery_intent_sha256",
        )
        atomic_write_json(preparation_intent_path, preparation_intent)
    binding = _attempt_binding(
        run_root=run_root,
        bundle_root=Path(manifest["bundle_root"]),
        bundle=bundle,
        attempt_id="failed-only-recovery-000",
        generation="failed-only-recovery",
        source_shard_ids=list(
            dict.fromkeys(source_by_id[request_id][0] for request_id in ordered_failed)
        ),
        exact_rows=rows,
    )
    if binding["queued_input_tokens_empirical_forecast"] != queued_tokens:
        raise ValueError("failed-only recovery queue forecast changed during materialization")
    result = _hashed(
        {
            "schema_version": "adag.process-witness.coarse-continuation-recovery.v1",
            "status": "prepared_offline_failed_only",
            "created_at": _now(),
            "continuation_manifest_sha256": manifest["continuation_manifest_sha256"],
            "recovery_intent_sha256": preparation_intent[
                "recovery_intent_sha256"
            ],
            "recovery_wave": 0,
            "attempt": binding,
            "request_count": len(ordered_failed),
            "inherited_calibration_failure_count": len(
                manifest["inherited_failure_request_ids"]
            ),
            "continuation_primary_failure_count": len(ordered_failed)
            - len(manifest["inherited_failure_request_ids"]),
            "successful_requests_rerun": 0,
            "provider_rows_byte_identical_to_v6": True,
            "additional_recovery_waves_permitted": False,
        },
        "recovery_manifest_sha256",
    )
    atomic_write_json(recovery_root / "manifest.json", result)
    return result


def prepare_failed_only_recovery(*, run_root: Path) -> dict[str, Any]:
    """Strictly validate and freeze the single aggregate recovery."""

    with _gate(run_root):
        return _prepare_failed_only_recovery_under_gate(run_root=run_root)


def _attempt(
    manifest: Mapping[str, Any], attempt_id: str, run_root: Path | None = None
) -> Mapping[str, Any]:
    attempts = (
        _all_attempts(run_root, manifest) if run_root is not None else manifest["attempts"]
    )
    match = next((row for row in attempts if row["attempt_id"] == attempt_id), None)
    if match is None:
        raise ValueError(f"unknown continuation attempt: {attempt_id}")
    return match


def _attempt_state(run_root: Path, attempt_id: str) -> str | None:
    root = run_root / "attempts" / attempt_id
    if (root / "collection.json").exists():
        return "collected"
    statuses = sorted((root / "status").glob("receipt-*.json"))
    if statuses:
        return str(_load_object(statuses[-1])["provider_response"].get("status"))
    if (root / "submission.json").exists():
        return str(_load_object(root / "submission.json")["provider_response"].get("status"))
    if (root / "submission-intent.json").exists():
        return "indeterminate"
    return None


def _cost_state(
    run_root: Path,
    manifest: Mapping[str, Any],
    *,
    exclude_attempt_id: str | None = None,
) -> tuple[float, float]:
    known = float(manifest["calibration_known_priced_cost_usd"])
    active_reservations = 0.0
    for binding in _all_attempts(run_root, manifest):
        if binding["attempt_id"] == exclude_attempt_id:
            continue
        root = run_root / "attempts" / binding["attempt_id"]
        if (root / "collection.json").exists():
            collection = _load_object(root / "collection.json")
            _verify(collection, "collection_sha256", "continuation collection")
            if collection.get("cost_complete") is not True:
                raise ValueError("continuation collected cost is incomplete")
            known += float(collection["known_priced_cost_usd"])
        elif (root / "submission-intent.json").exists():
            active_reservations += float(binding["calibrated_cost_reservation_usd"])
    return known, active_reservations


def _append_cost_status(
    *,
    run_root: Path,
    manifest: Mapping[str, Any],
    trigger: str,
    hard_stop_crossed_after_inflight_attempt: bool = False,
) -> dict[str, Any]:
    known, active_reservations = _cost_state(run_root, manifest)
    root = run_root / "cost-status"
    root.mkdir(exist_ok=True)
    prior_paths = sorted(root.glob("receipt-*.json"))
    previous = None
    warning_sticky = False
    for path in prior_paths:
        row = _load_object(path)
        _verify(row, "cost_status_sha256", "continuation cost status")
        if row.get("previous_cost_status_sha256") != previous:
            raise ValueError("continuation cost-status chain drift")
        previous = row["cost_status_sha256"]
        warning_sticky = warning_sticky or bool(row.get("warning_threshold_reached"))
    warning = warning_sticky or known >= float(manifest["warning_spend_threshold_usd"])
    result = _hashed(
        {
            "schema_version": "adag.process-witness.coarse-continuation-cost-status.v1",
            "recorded_at": _now(),
            "trigger": trigger,
            "continuation_manifest_sha256": manifest["continuation_manifest_sha256"],
            "previous_cost_status_sha256": previous,
            "known_actual_cost_usd": known,
            "active_calibrated_reservations_usd": active_reservations,
            "warning_spend_threshold_usd": manifest["warning_spend_threshold_usd"],
            "warning_threshold_reached": warning,
            "hard_campaign_stop_usd": manifest["hard_campaign_stop_usd"],
            "hard_stop_crossed_after_inflight_attempt": (
                hard_stop_crossed_after_inflight_attempt
                or known >= float(manifest["hard_campaign_stop_usd"])
            ),
        },
        "cost_status_sha256",
    )
    atomic_write_json(root / f"receipt-{len(prior_paths):04d}.json", result)
    return result


def _ensure_current_cost_status(
    *, run_root: Path, manifest: Mapping[str, Any], trigger: str
) -> dict[str, Any]:
    known, active = _cost_state(run_root, manifest)
    paths = sorted((run_root / "cost-status").glob("receipt-*.json"))
    if paths:
        latest = _load_object(paths[-1])
        _verify(latest, "cost_status_sha256", "continuation cost status")
        if math.isclose(
            float(latest["known_actual_cost_usd"]), known, rel_tol=0, abs_tol=1e-12
        ) and math.isclose(
            float(latest["active_calibrated_reservations_usd"]),
            active,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            return latest
    return _append_cost_status(run_root=run_root, manifest=manifest, trigger=trigger)


def _validate_candidate_admission(
    *,
    manifest: Mapping[str, Any],
    candidate_id: str,
    run_root: Path | None = None,
    active_attempt_ids: Sequence[str],
    known_actual_cost_usd: float,
    active_cost_reservations_usd: float,
) -> None:
    candidate = _attempt(manifest, candidate_id, run_root)
    if len(active_attempt_ids) >= int(manifest["maximum_concurrent_attempts"]):
        raise ValueError("continuation attempt concurrency is already full")
    active_queue = sum(
        int(
            _attempt(manifest, attempt_id, run_root)[
                "queued_input_tokens_empirical_forecast"
            ]
        )
        for attempt_id in active_attempt_ids
    )
    if active_queue + int(candidate["queued_input_tokens_empirical_forecast"]) > int(
        manifest["provider_queued_input_token_limit"]
    ):
        raise ValueError("continuation queued-input-token capacity would be exceeded")
    prospective = (
        known_actual_cost_usd
        + active_cost_reservations_usd
        + float(candidate["calibrated_cost_reservation_usd"])
    )
    stop = float(manifest["hard_campaign_stop_usd"])
    if known_actual_cost_usd >= stop or prospective > stop:
        raise ValueError("continuation calibrated spend reservation exceeds hard stop")


@contextmanager
def _gate(run_root: Path):
    lock = run_root / ".continuation-gate"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError("continuation mutation gate is held or stale") from error
    try:
        yield
    finally:
        lock.rmdir()


def _metadata(
    manifest: Mapping[str, Any], attempt_id: str, run_root: Path | None = None
) -> dict[str, str]:
    return {
        "campaign": str(manifest["continuation_manifest_sha256"])[:40],
        "shard": attempt_id,
        "generation": str(_attempt(manifest, attempt_id, run_root)["generation"]),
    }


def submit_attempt(
    *,
    run_root: Path,
    attempt_id: str,
    uploader: Callable[[Path], dict[str, Any]] = _upload_provider,
    creator: Callable[..., dict[str, Any]] = _create_provider,
) -> dict[str, Any]:
    manifest, _bundle = _load_continuation(run_root)
    binding = _attempt(manifest, attempt_id, run_root)
    root = run_root / "attempts" / attempt_id
    with _gate(run_root):
        if (root / "submission-intent.json").exists():
            raise FileExistsError("continuation attempt submission already attempted")
        attempt_order = [row["attempt_id"] for row in _all_attempts(run_root, manifest)]
        candidate_index = attempt_order.index(attempt_id)
        uncollected_prior = [
            prior_id
            for prior_id in attempt_order[:candidate_index]
            if _attempt_state(run_root, prior_id) != "collected"
        ]
        if uncollected_prior:
            raise ValueError(
                f"continuation attempts must be collected in order: {uncollected_prior}"
            )
        active = [
            row["attempt_id"]
            for row in _all_attempts(run_root, manifest)
            if _attempt_state(run_root, row["attempt_id"])
            not in {None, "collected", *TERMINAL_PROVIDER_STATUSES}
        ]
        known, active_reservations = _cost_state(run_root, manifest)
        _validate_candidate_admission(
            manifest=manifest,
            candidate_id=attempt_id,
            run_root=run_root,
            active_attempt_ids=active,
            known_actual_cost_usd=known,
            active_cost_reservations_usd=active_reservations,
        )
        metadata = _metadata(manifest, attempt_id, run_root)
        input_path = run_root / binding["input_relative_path"]
        submission_intent = _hashed(
            {
                "schema_version": SUBMISSION_SCHEMA,
                "status": "intent_persisted_before_provider_calls",
                "created_at": _now(),
                "continuation_manifest_sha256": manifest[
                    "continuation_manifest_sha256"
                ],
                "attempt_id": attempt_id,
                "generation": binding["generation"],
                "input_sha256": binding["input_sha256"],
                "request_count": binding["request_count"],
                "known_actual_cost_usd": known,
                "active_calibrated_reservations_usd": active_reservations,
                "candidate_calibrated_cost_reservation_usd": binding[
                    "calibrated_cost_reservation_usd"
                ],
                "projected_warning_threshold_reached": known
                + active_reservations
                + float(binding["calibrated_cost_reservation_usd"])
                >= float(manifest["warning_spend_threshold_usd"]),
                "metadata": metadata,
            },
            "submission_intent_sha256",
        )
        atomic_write_json(root / "submission-intent.json", submission_intent)
        provider_call_stage = "upload"
        try:
            upload = uploader(input_path)
            _validate_upload(upload)
            atomic_write_json(root / "provider-upload-response.json", upload)
            provider_call_stage = "create"
            provider = creator(upload["input_file_id"], metadata=metadata)
            _validate_snapshot(provider, metadata=metadata, input_file_id=upload["input_file_id"])
            atomic_write_json(root / "provider-create-response.json", provider)
            receipt = _hashed(
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "status": "submitted",
                    "recorded_at": _now(),
                    "continuation_manifest_sha256": manifest[
                        "continuation_manifest_sha256"
                    ],
                    "submission_intent_sha256": submission_intent[
                        "submission_intent_sha256"
                    ],
                    "provider_upload_response_sha256": file_sha256(
                        root / "provider-upload-response.json"
                    ),
                    "provider_response": provider,
                },
                "submission_sha256",
            )
            atomic_write_json(root / "submission.json", receipt)
            return receipt
        except BaseException as error:
            status_code = getattr(error, "status_code", None)
            explicit_rejection = isinstance(status_code, int) and status_code in {
                400,
                403,
                409,
                422,
                429,
            }
            atomic_write_json(
                root / "submission-failure.json",
                _hashed(
                    {
                        "schema_version": SUBMISSION_SCHEMA,
                        "status": (
                            "provider_explicit_rejection_no_accepted_batch"
                            if explicit_rejection
                            else "failed_closed_indeterminate_provider_state"
                        ),
                        "recorded_at": _now(),
                        "submission_intent_sha256": submission_intent[
                            "submission_intent_sha256"
                        ],
                        "continuation_manifest_sha256": manifest[
                            "continuation_manifest_sha256"
                        ],
                        "provider_call_stage": provider_call_stage,
                        "metadata": metadata,
                        "provider_upload_response_sha256": (
                            file_sha256(root / "provider-upload-response.json")
                            if (root / "provider-upload-response.json").is_file()
                            else None
                        ),
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:2000],
                        "provider_status_code": status_code,
                        "queue_rejection_campaign_stop": (
                            explicit_rejection and status_code == 429
                        ),
                        "automatic_retry_permitted": False,
                    },
                    "submission_failure_sha256",
                ),
            )
            if explicit_rejection:
                raise RuntimeError(
                    "provider explicitly rejected the attempt; the campaign is stopped"
                ) from error
            raise RuntimeError(
                "continuation provider state is indeterminate; reconcile before retry"
            ) from error


def _discover_batches_by_metadata(metadata: Mapping[str, str]) -> dict[str, Any]:
    page = production_v1._openai_client().batches.list(limit=100)
    snapshots: list[dict[str, Any]] = []
    total_scanned = 0
    page_count = 0
    for current in page.iter_pages():
        page_count += 1
        for batch in current.data:
            total_scanned += 1
            if dict(getattr(batch, "metadata", None) or {}) == dict(metadata):
                snapshots.append(_production_provider_batch_dict(batch))
    return {
        "exhaustive": True,
        "page_count": page_count,
        "total_scanned": total_scanned,
        "snapshots": snapshots,
    }


def recover_attempt_submission(
    *,
    run_root: Path,
    attempt_id: str,
    discoverer: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reconcile one ambiguous submission without a second scientific attempt."""

    manifest, _bundle = _load_continuation(run_root)
    _attempt(manifest, attempt_id, run_root)
    root = run_root / "attempts" / attempt_id
    with _gate(run_root):
        if (root / "submission.json").exists():
            raise FileExistsError("continuation submission receipt already exists")
        submission_intent = _load_object(root / "submission-intent.json")
        _verify(submission_intent, "submission_intent_sha256", "submission intent")
        failure = _load_object(root / "submission-failure.json")
        _verify(failure, "submission_failure_sha256", "submission failure")
        metadata = _metadata(manifest, attempt_id, run_root)
        upload_path = root / "provider-upload-response.json"
        expected_upload_sha = (
            file_sha256(upload_path) if upload_path.is_file() else None
        )
        if (
            failure.get("submission_intent_sha256")
            != submission_intent["submission_intent_sha256"]
            or failure.get("continuation_manifest_sha256")
            != manifest["continuation_manifest_sha256"]
            or failure.get("metadata") != metadata
            or failure.get("provider_upload_response_sha256")
            != expected_upload_sha
        ):
            raise ValueError("submission failure receipt binding drift")
        if failure.get("status") == "provider_explicit_rejection_no_accepted_batch":
            raise ValueError("explicit provider rejection stopped this campaign version")
        if discoverer is None:
            discoverer = _discover_batches_by_metadata
        create_path = root / "provider-create-response.json"
        if create_path.exists():
            if not upload_path.exists():
                raise ValueError("provider create receipt exists without upload receipt")
            provider = _load_object(create_path)
            upload = _load_object(upload_path)
            recovered_by = "immediate_retained_create_snapshot"
            discovery_sha256 = None
        else:
            discovery = dict(discoverer(metadata=metadata))
            matches = discovery.get("snapshots")
            if (
                discovery.get("exhaustive") is not True
                or not isinstance(discovery.get("page_count"), int)
                or discovery["page_count"] < 1
                or not isinstance(discovery.get("total_scanned"), int)
                or discovery["total_scanned"] < 0
                or not isinstance(matches, list)
                or discovery["total_scanned"] < len(matches)
            ):
                raise ValueError("provider discovery did not prove exhaustive enumeration")
            discovery_root = root / "discovery"
            discovery_root.mkdir(exist_ok=True)
            prior_discoveries = sorted(discovery_root.glob("receipt-*.json"))
            previous_discovery = None
            if prior_discoveries:
                previous = _load_object(prior_discoveries[-1])
                _verify(previous, "discovery_sha256", "provider discovery")
                previous_discovery = previous["discovery_sha256"]
            discovery_receipt = _hashed(
                {
                    "schema_version": "adag.process-witness.coarse-continuation-discovery.v1",
                    "recorded_at": _now(),
                    "continuation_manifest_sha256": manifest[
                        "continuation_manifest_sha256"
                    ],
                    "submission_failure_sha256": failure[
                        "submission_failure_sha256"
                    ],
                    "previous_discovery_sha256": previous_discovery,
                    "metadata": metadata,
                    "exhaustive": True,
                    "page_count": discovery["page_count"],
                    "total_scanned": discovery["total_scanned"],
                    "matching_batch_ids": [row.get("batch_id") for row in matches],
                    "matching_snapshots": matches,
                },
                "discovery_sha256",
            )
            atomic_write_json(
                discovery_root / f"receipt-{len(prior_discoveries):04d}.json",
                discovery_receipt,
            )
            if len(matches) != 1:
                raise ValueError(
                    "ambiguous submission requires exactly one metadata-matched Batch; "
                    f"found {len(matches)} and zero matches are not absence proof"
                )
            if not upload_path.exists():
                raise ValueError("matched Batch exists without retained upload receipt")
            provider = matches[0]
            upload = _load_object(upload_path)
            recovered_by = "unique_provider_metadata_discovery"
            discovery_sha256 = discovery_receipt["discovery_sha256"]
        _validate_upload(upload)
        _validate_snapshot(
            provider, metadata=metadata, input_file_id=upload["input_file_id"]
        )
        production_v1._write_or_verify_json(
            root / "provider-create-response.json", provider
        )
        receipt = _hashed(
            {
                "schema_version": SUBMISSION_SCHEMA,
                "status": "submitted",
                "recorded_at": _now(),
                "recovered_by": recovered_by,
                "discovery_sha256": discovery_sha256,
                "continuation_manifest_sha256": manifest[
                    "continuation_manifest_sha256"
                ],
                "submission_intent_sha256": submission_intent[
                    "submission_intent_sha256"
                ],
                "provider_upload_response_sha256": file_sha256(upload_path),
                "provider_response": provider,
            },
            "submission_sha256",
        )
        atomic_write_json(root / "submission.json", receipt)
        return receipt


def _validate_submission_discovery(
    *, root: Path, receipt: Mapping[str, Any]
) -> None:
    discovery_sha = receipt.get("discovery_sha256")
    if receipt.get("recovered_by") == "unique_provider_metadata_discovery":
        paths = sorted((root / "discovery").glob("receipt-*.json"))
        previous = None
        selected = None
        for path in paths:
            discovery = _load_object(path)
            _verify(discovery, "discovery_sha256", "provider discovery")
            if discovery.get("previous_discovery_sha256") != previous:
                raise ValueError("provider discovery receipt chain drift")
            previous = discovery["discovery_sha256"]
            if previous == discovery_sha:
                selected = discovery
        if (
            selected is None
            or selected.get("exhaustive") is not True
            or selected.get("matching_batch_ids")
            != [receipt["provider_response"]["batch_id"]]
        ):
            raise ValueError("submission discovery binding drift")
    elif discovery_sha is not None:
        raise ValueError("unexpected submission discovery binding")


def _submission(run_root: Path, attempt_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _bundle = _load_continuation(run_root)
    root = run_root / "attempts" / attempt_id
    receipt = _load_object(root / "submission.json")
    _verify(receipt, "submission_sha256", "continuation submission")
    upload = _load_object(root / "provider-upload-response.json")
    _validate_upload(upload)
    if file_sha256(root / "provider-upload-response.json") != receipt[
        "provider_upload_response_sha256"
    ]:
        raise ValueError("continuation upload binding drift")
    _validate_submission_discovery(root=root, receipt=receipt)
    _validate_snapshot(
        receipt["provider_response"],
        metadata=_metadata(manifest, attempt_id, run_root),
        input_file_id=upload["input_file_id"],
    )
    return manifest, receipt


def _record_status_snapshot(
    *,
    run_root: Path,
    attempt_id: str,
    manifest: Mapping[str, Any],
    submission: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    root = run_root / "attempts" / attempt_id / "status"
    root.mkdir(exist_ok=True)
    prior = sorted(root.glob("receipt-*.json"))
    previous = None
    for path in prior:
        row = _load_object(path)
        _verify(row, "status_sha256", "continuation status")
        if row["previous_status_sha256"] != previous:
            raise ValueError("continuation status chain drift")
        previous = row["status_sha256"]
    result = _hashed(
        {
            "schema_version": STATUS_SCHEMA,
            "recorded_at": _now(),
            "continuation_manifest_sha256": manifest["continuation_manifest_sha256"],
            "submission_sha256": submission["submission_sha256"],
            "previous_status_sha256": previous,
            "provider_response": dict(observed),
        },
        "status_sha256",
    )
    atomic_write_json(root / f"receipt-{len(prior):04d}.json", result)
    return result


def status_attempt(
    *, run_root: Path, attempt_id: str, retriever: Callable[..., dict[str, Any]] | None = None
) -> dict[str, Any]:
    manifest, submission = _submission(run_root, attempt_id)
    if retriever is None:
        def retriever(batch_id: str) -> dict[str, Any]:
            return _production_provider_batch_dict(
                production_v1._openai_client().batches.retrieve(batch_id)
            )
    provider = submission["provider_response"]
    observed = retriever(provider["batch_id"])
    _validate_snapshot(
        observed,
        metadata=_metadata(manifest, attempt_id, run_root),
        batch_id=provider["batch_id"],
        input_file_id=provider["input_file_id"],
    )
    return _record_status_snapshot(
        run_root=run_root,
        attempt_id=attempt_id,
        manifest=manifest,
        submission=submission,
        observed=observed,
    )


def _derive_events_from_provider_rows(
    *,
    rows: Mapping[str, Mapping[str, Any]],
    binding: Mapping[str, Any],
    bundle: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = set(binding["request_ids_in_order"])
    if set(rows) - expected:
        raise ValueError("continuation provider output contains unknown custom_id")
    metadata_by_id = {
        str(row["request_id"]): row for row in bundle["request_index"]
    }
    events = []
    for request_id in binding["request_ids_in_order"]:
        request = metadata_by_id[request_id]
        if request_id in rows:
            event = _parse_row(rows[request_id], request)
        else:
            event = {
                "schema_version": EVENT_SCHEMA,
                "request_id": request_id,
                "shard_id": request["shard_id"],
                "window_id": request["window_id"],
                "window_index": request["window_index"],
                "response_id": request["response_id"],
                "replica_index": request["replica_index"],
                "body_sha256": request["body_sha256"],
                "focal_unit_ids": request["focal_unit_ids"],
                "validation_status": "missing",
                "error_type": (
                    f"terminal_batch_{snapshot['status']}_without_request_row"
                ),
                "usage": openai_usage(None).model_dump(mode="json"),
                "decisions": None,
            }
        event["generation"] = binding["generation"]
        event["execution_attempt_id"] = binding["attempt_id"]
        events.append(event)
    return events


def collect_attempt(
    *,
    run_root: Path,
    attempt_id: str,
    downloader: Callable[[str], tuple[dict[str, Any], dict[str, dict[str, Any]]]] = _download,
) -> dict[str, Any]:
    manifest, bundle = _load_continuation(run_root)
    binding = _attempt(manifest, attempt_id, run_root)
    root = run_root / "attempts" / attempt_id
    with _gate(run_root):
        if (root / "collection.json").exists():
            collection, _events = _validate_collected_attempt(
                run_root=run_root,
                manifest=manifest,
                bundle=bundle,
                binding=binding,
            )
            _ensure_current_cost_status(
                run_root=run_root,
                manifest=manifest,
                trigger=f"collection_receipt_reconciled:{attempt_id}",
            )
            return collection
        _manifest, submission = _submission(run_root, attempt_id)
        provider = submission["provider_response"]
        collection_intent_path = root / "collection-intent.json"
        if collection_intent_path.exists():
            collection_intent = _load_object(collection_intent_path)
            _verify(
                collection_intent,
                "collection_intent_sha256",
                "continuation collection intent",
            )
            if collection_intent.get("submission_sha256") != submission["submission_sha256"]:
                raise ValueError("continuation collection intent binding drift")
        else:
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
            atomic_write_json(collection_intent_path, collection_intent)
        snapshot, files = downloader(provider["batch_id"])
        _validate_snapshot(
            snapshot,
            metadata=_metadata(manifest, attempt_id, run_root),
            batch_id=provider["batch_id"],
            input_file_id=provider["input_file_id"],
        )
        if snapshot.get("status") not in TERMINAL_PROVIDER_STATUSES:
            raise ValueError("continuation Batch is not terminal")
        _record_status_snapshot(
            run_root=run_root,
            attempt_id=attempt_id,
            manifest=manifest,
            submission=submission,
            observed=snapshot,
        )
        raw_root = root / "raw"
        raw_root.mkdir(exist_ok=True)
        production_v1._write_or_verify_json(
            raw_root / "provider-snapshot.json", snapshot
        )
        rows: dict[str, Mapping[str, Any]] = {}
        expected_raw_sources = {
            source
            for source in ("output", "error")
            if snapshot.get(f"{source}_file_id") is not None
        }
        if set(files) != expected_raw_sources:
            raise ValueError("continuation provider raw file set drift")
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
                raise ValueError("continuation provider file receipt drift")
            path = raw_root / f"{source}.jsonl"
            production_v1._write_or_verify_bytes(path, item["content"])
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
                    raise ValueError("continuation duplicate provider custom_id")
                rows[request_id] = row
        events = _derive_events_from_provider_rows(
            rows=rows, binding=binding, bundle=bundle, snapshot=snapshot
        )
        prices = load_price_snapshot(Path(manifest["bundle_root"]) / "price-snapshot.json")
        total, complete, basis = _price_events(
            events=events,
            snapshot=snapshot,
            prices=prices,
            aggregate_fallback_long_context_impossible=(
                production_v1._input_byte_bound_excludes_long_context(
                    input_path=run_root / binding["input_relative_path"],
                    config=bundle["config"],
                    prices=prices,
                )
            ),
        )
        event_bytes = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in events
        )
        production_v1._write_or_verify_bytes(root / "events.jsonl", event_bytes)
        success = sum(row["validation_status"] == "success" for row in events)
        result = _hashed(
            {
                "schema_version": COLLECTION_SCHEMA,
                "status": "complete" if success == len(events) and complete else "complete_with_failures",
                "completed_at": _now(),
                "collection_intent_sha256": collection_intent[
                    "collection_intent_sha256"
                ],
                "continuation_manifest_sha256": manifest["continuation_manifest_sha256"],
                "attempt_id": attempt_id,
                "generation": binding["generation"],
                "request_count": len(events),
                "success_count": success,
                "failure_count": len(events) - success,
                "known_priced_cost_usd": total,
                "cost_complete": complete,
                "pricing_basis": basis,
                "provider_terminal_status": snapshot["status"],
                "raw_file_bindings": raw_bindings,
                "events_sha256": file_sha256(root / "events.jsonl"),
            },
            "collection_sha256",
        )
        known_before, active_before = _cost_state(
            run_root, manifest, exclude_attempt_id=attempt_id
        )
        atomic_write_json(root / "collection.json", result)
        _append_cost_status(
            run_root=run_root,
            manifest=manifest,
            trigger=f"collection:{attempt_id}",
            hard_stop_crossed_after_inflight_attempt=(
                known_before + active_before + total
                >= float(manifest["hard_campaign_stop_usd"])
            ),
        )
        return result


def _validate_collected_attempt(
    *,
    run_root: Path,
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    binding: Mapping[str, Any],
    bundle_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempt_id = str(binding["attempt_id"])
    root = run_root / "attempts" / attempt_id
    collection = _load_object(root / "collection.json")
    _verify(collection, "collection_sha256", "continuation collection")
    events_path = root / "events.jsonl"
    events = read_jsonl(events_path)
    if (
        collection.get("continuation_manifest_sha256")
        != manifest["continuation_manifest_sha256"]
        or collection.get("attempt_id") != attempt_id
        or collection.get("generation") != binding["generation"]
        or collection.get("request_count") != len(events)
        or collection.get("cost_complete") is not True
        or file_sha256(events_path) != collection.get("events_sha256")
        or [str(event["request_id"]) for event in events]
        != binding["request_ids_in_order"]
    ):
        raise ValueError("continuation collected attempt binding drift")
    submission_intent = _load_object(root / "submission-intent.json")
    _verify(submission_intent, "submission_intent_sha256", "submission intent")
    submission = _load_object(root / "submission.json")
    _verify(submission, "submission_sha256", "submission")
    _validate_submission_discovery(root=root, receipt=submission)
    upload_path = root / "provider-upload-response.json"
    upload = _load_object(upload_path)
    _validate_upload(upload)
    if (
        submission_intent.get("input_sha256") != binding["input_sha256"]
        or submission_intent.get("request_count") != binding["request_count"]
        or submission.get("submission_intent_sha256")
        != submission_intent["submission_intent_sha256"]
        or submission.get("provider_upload_response_sha256") != file_sha256(upload_path)
    ):
        raise ValueError("continuation submission receipt drift")
    provider = submission["provider_response"]
    metadata = _metadata(manifest, attempt_id, run_root)
    _validate_snapshot(provider, metadata=metadata, input_file_id=upload["input_file_id"])
    if _load_object(root / "provider-create-response.json") != provider:
        raise ValueError("continuation provider create receipt drift")
    collection_intent = _load_object(root / "collection-intent.json")
    _verify(collection_intent, "collection_intent_sha256", "collection intent")
    if (
        collection_intent.get("submission_sha256") != submission["submission_sha256"]
        or collection.get("collection_intent_sha256")
        != collection_intent["collection_intent_sha256"]
    ):
        raise ValueError("continuation collection intent drift")
    raw_snapshot_path = root / "raw/provider-snapshot.json"
    raw_snapshot = _load_object(raw_snapshot_path)
    _validate_snapshot(
        raw_snapshot,
        metadata=metadata,
        batch_id=provider["batch_id"],
        input_file_id=provider["input_file_id"],
    )
    if raw_snapshot.get("status") != collection.get("provider_terminal_status"):
        raise ValueError("continuation terminal snapshot drift")
    raw_bindings = collection.get("raw_file_bindings")
    if not isinstance(raw_bindings, list):
        raise ValueError("continuation raw bindings are absent")
    raw_by_source = {row["source"]: row for row in raw_bindings}
    expected_raw_sources = {
        "provider_snapshot",
        *(
            source
            for source in ("output", "error")
            if raw_snapshot.get(f"{source}_file_id") is not None
        ),
    }
    if len(raw_by_source) != len(raw_bindings) or set(raw_by_source) != expected_raw_sources:
        raise ValueError("continuation raw snapshot is not bound")
    provider_rows: dict[str, Mapping[str, Any]] = {}
    for raw in raw_bindings:
        path = run_root / raw["path"]
        if (
            not path.is_file()
            or file_sha256(path) != raw["sha256"]
            or path.stat().st_size != raw["bytes"]
        ):
            raise ValueError("continuation raw provider evidence drift")
        source = str(raw["source"])
        if source == "provider_snapshot":
            if raw.get("file_id") is not None or path != raw_snapshot_path:
                raise ValueError("continuation raw snapshot receipt drift")
            continue
        if raw.get("file_id") != raw_snapshot.get(f"{source}_file_id"):
            raise ValueError("continuation raw provider file id drift")
        for row in read_jsonl(path):
            request_id = row.get("custom_id")
            if not isinstance(request_id, str) or request_id in provider_rows:
                raise ValueError("continuation duplicate raw provider custom_id")
            provider_rows[request_id] = row
    reconstructed = _derive_events_from_provider_rows(
        rows=provider_rows,
        binding=binding,
        bundle=bundle,
        snapshot=raw_snapshot,
    )
    prices = load_price_snapshot(
        (bundle_root or Path(manifest["bundle_root"])) / "price-snapshot.json"
    )
    total, complete, basis = _price_events(
        events=reconstructed,
        snapshot=raw_snapshot,
        prices=prices,
        aggregate_fallback_long_context_impossible=(
            production_v1._input_byte_bound_excludes_long_context(
                input_path=run_root / binding["input_relative_path"],
                config=bundle["config"],
                prices=prices,
            )
        ),
    )
    if canonical_sha256(reconstructed) != canonical_sha256(events):
        raise ValueError("continuation events do not reconstruct from raw evidence")
    success_count = sum(row["validation_status"] == "success" for row in reconstructed)
    expected_status = (
        "complete"
        if success_count == len(reconstructed) and complete
        else "complete_with_failures"
    )
    if (
        complete is not True
        or collection.get("cost_complete") is not True
        or collection.get("status") != expected_status
        or collection.get("success_count") != success_count
        or collection.get("failure_count") != len(reconstructed) - success_count
        or not math.isclose(
            float(collection.get("known_priced_cost_usd", math.nan)),
            float(total),
            rel_tol=0,
            abs_tol=1e-12,
        )
        or collection.get("pricing_basis") != basis
    ):
        raise ValueError("continuation collection census or pricing drift")
    previous = None
    status_paths = sorted((root / "status").glob("receipt-*.json"))
    if not status_paths:
        raise ValueError("continuation status chain is absent")
    for path in status_paths:
        status = _load_object(path)
        _verify(status, "status_sha256", "continuation status")
        if (
            status.get("previous_status_sha256") != previous
            or status.get("submission_sha256") != submission["submission_sha256"]
        ):
            raise ValueError("continuation status chain drift")
        _validate_snapshot(
            status["provider_response"],
            metadata=metadata,
            batch_id=provider["batch_id"],
            input_file_id=provider["input_file_id"],
        )
        previous = status["status_sha256"]
    request_by_id = {str(row["request_id"]): row for row in bundle["request_index"]}
    for event in events:
        source = request_by_id[str(event["request_id"])]
        if any(
            event.get(field) != source[field]
            for field in (
                "shard_id",
                "window_id",
                "window_index",
                "response_id",
                "replica_index",
                "body_sha256",
                "focal_unit_ids",
            )
        ):
            raise ValueError("continuation event/source request identity drift")
    return collection, events


def _validate_inherited_calibration_evidence(
    *,
    run_root: Path,
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any],
    bundle_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = run_root / "inherited-calibration-run/shards/shard-005"
    collection = _load_object(root / "collection.json")
    production_v1._verify(collection, "collection_sha256", "inherited collection")
    events = read_jsonl(root / "events.jsonl")
    shard = next(row for row in bundle["shards"] if row["shard_id"] == "shard-005")
    if (
        collection["collection_sha256"] != manifest["calibration_collection_sha256"]
        or file_sha256(root / "events.jsonl") != collection["events_sha256"]
        or (root / "input.jsonl").read_bytes()
        != ((bundle_root or Path(manifest["bundle_root"])) / shard["path"]).read_bytes()
    ):
        raise ValueError("inherited calibration collection/input drift")
    source_intent = _load_object(run_root / "inherited-calibration-run/campaign-intent.json")
    production_v1._verify(
        source_intent, "campaign_run_sha256", "inherited campaign intent"
    )
    submission = _load_object(root / "submission.json")
    production_v1._verify(submission, "submission_sha256", "inherited submission")
    upload = _load_object(root / "provider-upload-response.json")
    _validate_upload(upload)
    submission_intent = _load_object(root / "submission-intent.json")
    production_v1._verify(
        submission_intent, "submission_intent_sha256", "inherited submission intent"
    )
    collection_intent = _load_object(root / "collection-intent.json")
    production_v1._verify(
        collection_intent, "collection_intent_sha256", "inherited collection intent"
    )
    if (
        submission.get("submission_intent_sha256")
        != submission_intent["submission_intent_sha256"]
        or submission.get("provider_upload_response_sha256")
        != file_sha256(root / "provider-upload-response.json")
        or collection_intent.get("submission_sha256") != submission["submission_sha256"]
        or collection.get("collection_intent_sha256")
        != collection_intent["collection_intent_sha256"]
    ):
        raise ValueError("inherited submission/collection receipt drift")
    metadata = production_v1._metadata(source_intent, "shard-005", "primary")
    provider = submission["provider_response"]
    _validate_snapshot(provider, metadata=metadata, input_file_id=upload["input_file_id"])
    if _load_object(root / "provider-create-response.json") != provider:
        raise ValueError("inherited provider create receipt drift")
    raw_snapshot = _load_object(root / "raw/provider-snapshot.json")
    _validate_snapshot(
        raw_snapshot,
        metadata=metadata,
        batch_id=provider["batch_id"],
        input_file_id=provider["input_file_id"],
    )
    raw_bindings = collection.get("raw_file_bindings")
    if not isinstance(raw_bindings, list):
        raise ValueError("inherited raw provider bindings absent")
    raw_by_source = {row["source"]: row for row in raw_bindings}
    expected_raw_sources = {
        "provider_snapshot",
        *(
            source
            for source in ("output", "error")
            if raw_snapshot.get(f"{source}_file_id") is not None
        ),
    }
    if (
        len(raw_by_source) != len(raw_bindings)
        or set(raw_by_source) != expected_raw_sources
        or raw_snapshot.get("status") != collection.get("provider_terminal_status")
    ):
        raise ValueError("inherited raw snapshot binding absent")
    provider_rows: dict[str, Mapping[str, Any]] = {}
    for raw in raw_bindings:
        path = run_root / "inherited-calibration-run" / raw["path"]
        if (
            not path.is_file()
            or file_sha256(path) != raw["sha256"]
            or path.stat().st_size != raw["bytes"]
        ):
            raise ValueError("inherited raw provider evidence drift")
        source = str(raw["source"])
        if source == "provider_snapshot":
            if raw.get("file_id") is not None:
                raise ValueError("inherited raw snapshot receipt drift")
            continue
        if raw.get("file_id") != raw_snapshot.get(f"{source}_file_id"):
            raise ValueError("inherited raw provider file id drift")
        for row in read_jsonl(path):
            request_id = row.get("custom_id")
            if not isinstance(request_id, str) or request_id in provider_rows:
                raise ValueError("inherited duplicate raw provider custom_id")
            provider_rows[request_id] = row
    request_by_id = {str(row["request_id"]): row for row in bundle["request_index"]}
    ordered_ids = [row[0] for row in _exact_jsonl_rows(root / "input.jsonl")]
    if set(provider_rows) - set(ordered_ids):
        raise ValueError("inherited raw provider output has unknown custom_id")
    reconstructed = []
    for request_id in ordered_ids:
        request = request_by_id[request_id]
        if request_id in provider_rows:
            reconstructed.append(_parse_row(provider_rows[request_id], request))
        else:
            reconstructed.append(
                {
                    "schema_version": EVENT_SCHEMA,
                    "request_id": request_id,
                    "shard_id": request["shard_id"],
                    "window_id": request["window_id"],
                    "window_index": request["window_index"],
                    "response_id": request["response_id"],
                    "replica_index": request["replica_index"],
                    "body_sha256": request["body_sha256"],
                    "focal_unit_ids": request["focal_unit_ids"],
                    "validation_status": "missing",
                    "error_type": (
                        f"terminal_batch_{raw_snapshot['status']}_without_request_row"
                    ),
                    "usage": openai_usage(None).model_dump(mode="json"),
                    "decisions": None,
                }
            )
    prices_root = bundle_root or Path(manifest["bundle_root"])
    prices = load_price_snapshot(prices_root / "price-snapshot.json")
    total, complete, basis = _price_events(
        events=reconstructed,
        snapshot=raw_snapshot,
        prices=prices,
        aggregate_fallback_long_context_impossible=(
            production_v1._input_byte_bound_excludes_long_context(
                input_path=root / "input.jsonl",
                config=bundle["config"],
                prices=prices,
            )
        ),
    )
    if canonical_sha256(reconstructed) != canonical_sha256(events):
        raise ValueError("inherited events do not reconstruct from raw evidence")
    success_count = sum(row["validation_status"] == "success" for row in reconstructed)
    if (
        collection.get("cost_complete") is not complete
        or collection.get("success_count") != success_count
        or collection.get("failure_count") != len(reconstructed) - success_count
        or not math.isclose(
            float(collection.get("known_priced_cost_usd", math.nan)),
            float(total),
            rel_tol=0,
            abs_tol=1e-12,
        )
        or collection.get("pricing_basis") != basis
    ):
        raise ValueError("inherited collection census or pricing drift")
    previous = None
    for path in sorted((root / "status").glob("receipt-*.json")):
        status = _load_object(path)
        production_v1._verify(status, "status_sha256", "inherited status")
        if status.get("previous_status_sha256") != previous:
            raise ValueError("inherited status chain drift")
        previous = status["status_sha256"]
    if previous is None:
        raise ValueError("inherited status chain absent")
    for event in events:
        source = request_by_id[str(event["request_id"])]
        if any(
            event.get(field) != source[field]
            for field in (
                "shard_id",
                "window_id",
                "window_index",
                "response_id",
                "replica_index",
                "body_sha256",
                "focal_unit_ids",
            )
        ):
            raise ValueError("inherited event/source request drift")
    reconciliation = _load_object(run_root / "inherited-cost-reconciliation.json")
    _verify(reconciliation, "cost_reconciliation_sha256", "inherited cost reconciliation")
    if (
        reconciliation.get("cost_complete") is not True
        or reconciliation.get("credit_balance_exhausted_request_count") != 8
        or reconciliation.get("usage_bearing_request_count") != 6439
        or reconciliation["cost_reconciliation_sha256"]
        != manifest["inherited_cost_reconciliation_sha256"]
    ):
        raise ValueError("inherited cost reconciliation drift")
    return collection, events


def _finalize_continuation_under_gate(
    *, run_root: Path, destination: Path
) -> dict[str, Any]:
    """Freeze one effective success per v6 request after aggregate recovery."""

    manifest, bundle = _load_continuation(run_root)
    if destination.exists():
        raise FileExistsError(f"continuation destination exists: {destination}")
    attempts = _all_attempts(run_root, manifest)
    if len(attempts) != len(manifest["attempts"]) + 1:
        raise ValueError("failed-only recovery must be prepared before finalization")
    inherited_collection, inherited_events = _validate_inherited_calibration_evidence(
        run_root=run_root, bundle=bundle, manifest=manifest
    )
    effective = {str(row["request_id"]): row for row in inherited_events}
    collection_bindings = [
        {
            "generation": "inherited-calibration-primary",
            "attempt_id": "shard-005",
            "collection_sha256": inherited_collection["collection_sha256"],
            "events_sha256": inherited_collection["events_sha256"],
        }
    ]
    total_cost = float(manifest["calibration_known_priced_cost_usd"])
    for binding in attempts:
        collection, attempt_events = _validate_collected_attempt(
            run_root=run_root, manifest=manifest, bundle=bundle, binding=binding
        )
        for event in attempt_events:
            request_id = str(event["request_id"])
            if binding["generation"] == "failed-only-recovery":
                if request_id not in effective or effective[request_id]["validation_status"] == "success":
                    raise ValueError("failed-only recovery did not replace a failure")
            elif request_id in effective:
                raise ValueError("continuation primary duplicated request identity")
            effective[request_id] = event
        total_cost += float(collection["known_priced_cost_usd"])
        collection_bindings.append(
            {
                "generation": binding["generation"],
                "attempt_id": binding["attempt_id"],
                "collection_sha256": collection["collection_sha256"],
                "events_sha256": collection["events_sha256"],
            }
        )
    request_ids = [str(row["request_id"]) for row in bundle["request_index"]]
    if set(effective) != set(request_ids) or any(
        effective[request_id]["validation_status"] != "success"
        for request_id in request_ids
    ):
        raise ValueError("continuation finalization requires exact successful request coverage")
    events = [effective[request_id] for request_id in request_ids]
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
    units = load_production_bundle(Path(manifest["bundle_root"]), load_units=True)["units"]
    pending_ids = {
        str(unit["unit_id"])
        for unit in units
        if unit["assignment_route"] == "openai_pending"
    }
    if set(votes_by_unit) != pending_ids or any(
        len(votes_by_unit[unit_id]) != 3
        or sorted(int(vote["replica_index"]) for vote in votes_by_unit[unit_id])
        != [0, 1, 2]
        for unit_id in pending_ids
    ):
        raise ValueError("continuation finalization requires exactly three votes per pending unit")
    proposals = [proposal_from_votes(unit, votes_by_unit.get(unit["unit_id"], [])) for unit in units]
    groups = sampling_groups(units, proposals)
    final_cost_status = _ensure_current_cost_status(
        run_root=run_root,
        manifest=manifest,
        trigger="finalization_cost_reconciliation",
    )
    if (
        not math.isclose(
            float(final_cost_status["known_actual_cost_usd"]),
            total_cost,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or float(final_cost_status["active_calibrated_reservations_usd"]) != 0.0
        or bool(final_cost_status["hard_stop_crossed_after_inflight_attempt"])
        != (total_cost >= float(manifest["hard_campaign_stop_usd"]))
    ):
        raise ValueError("continuation final cost-status reconciliation drift")
    temporary = destination.parent / f".{destination.name}.finalizing-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"continuation temporary destination exists: {temporary}")
    temporary.mkdir(parents=True)
    for name, rows in (
        ("effective-events.jsonl", events),
        ("proposals.jsonl", proposals),
        ("sampling-groups.jsonl", groups),
    ):
        atomic_write_bytes(
            temporary / name,
            b"".join(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for row in rows
            ),
        )
    shutil.copytree(run_root, temporary / "continuation-evidence")
    shutil.copytree(Path(manifest["bundle_root"]), temporary / "campaign-bundle")
    inventory = production_v1._write_evidence_inventory(temporary)
    result = _hashed(
        {
            "schema_version": FINAL_SCHEMA,
            "status": "frozen_sampling_proposals_not_semantic_truth",
            "created_at": _now(),
            "continuation_manifest_sha256": manifest["continuation_manifest_sha256"],
            "bundle_manifest_sha256": manifest["bundle_manifest_sha256"],
            "request_count": len(events),
            "proposal_count": len(proposals),
            "sampling_group_count": len(groups),
            "pending_units_with_exactly_three_votes": len(pending_ids),
            "actual_total_cost_usd": total_cost,
            "cost_complete": True,
            "warning_spend_threshold_usd": manifest["warning_spend_threshold_usd"],
            "hard_campaign_stop_usd": manifest["hard_campaign_stop_usd"],
            "hard_stop_crossed_after_inflight_attempt": total_cost
            >= float(manifest["hard_campaign_stop_usd"]),
            "final_cost_status_sha256": final_cost_status["cost_status_sha256"],
            "collection_bindings": collection_bindings,
            "effective_events_sha256": file_sha256(temporary / "effective-events.jsonl"),
            "proposals_sha256": file_sha256(temporary / "proposals.jsonl"),
            "sampling_groups_sha256": file_sha256(temporary / "sampling-groups.jsonl"),
            "evidence_inventory_sha256": inventory["evidence_inventory_sha256"],
            "claim_boundary": bundle["config"]["claim_boundary"],
        },
        "proposal_bank_manifest_sha256",
    )
    atomic_write_json(temporary / "manifest.json", result)
    production_v1._readonly_tree(temporary)
    load_frozen_continuation_proposal_bank(temporary)
    temporary.rename(destination)
    try:
        load_frozen_continuation_proposal_bank(destination)
    except BaseException:
        destination.rename(temporary)
        raise
    return result


def finalize_continuation(*, run_root: Path, destination: Path) -> dict[str, Any]:
    """Freeze a fully collected continuation under the campaign mutation gate."""

    with _gate(run_root):
        return _finalize_continuation_under_gate(
            run_root=run_root, destination=destination
        )


def load_frozen_continuation_proposal_bank(root: Path) -> dict[str, Any]:
    """Strictly reload a finalized continuation without original source roots."""

    production_v1._validate_readonly_modes(root)
    result = _load_object(root / "manifest.json")
    _verify(result, "proposal_bank_manifest_sha256", "continuation proposal bank")
    if (
        result.get("schema_version") != FINAL_SCHEMA
        or result.get("status") != "frozen_sampling_proposals_not_semantic_truth"
        or result.get("cost_complete") is not True
    ):
        raise ValueError("continuation proposal-bank semantic drift")
    inventory = _load_object(root / "evidence-inventory.json")
    _verify(inventory, "evidence_inventory_sha256", "continuation evidence inventory")
    if inventory["evidence_inventory_sha256"] != result["evidence_inventory_sha256"]:
        raise ValueError("continuation evidence inventory binding drift")
    expected = {row["path"]: row for row in inventory["files"]}
    observed = {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file()
        and path not in {root / "manifest.json", root / "evidence-inventory.json"}
    }
    if set(expected) != set(observed):
        raise ValueError("continuation evidence inventory coverage drift")
    for relative, path in observed.items():
        row = expected[relative]
        if file_sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"continuation evidence file drift: {relative}")
    bundle = load_production_bundle(root / "campaign-bundle", load_units=True)
    if bundle["manifest"]["manifest_sha256"] != result["bundle_manifest_sha256"]:
        raise ValueError("continuation copied bundle drift")
    continuation_root = root / "continuation-evidence"
    manifest = _load_object(continuation_root / "continuation-manifest.json")
    _verify(manifest, "continuation_manifest_sha256", "copied continuation manifest")
    if manifest["continuation_manifest_sha256"] != result["continuation_manifest_sha256"]:
        raise ValueError("continuation copied intent drift")
    _validate_inherited_calibration_evidence(
        run_root=continuation_root,
        bundle=bundle,
        manifest=manifest,
        bundle_root=root / "campaign-bundle",
    )
    for filename, field in (
        ("effective-events.jsonl", "effective_events_sha256"),
        ("proposals.jsonl", "proposals_sha256"),
        ("sampling-groups.jsonl", "sampling_groups_sha256"),
    ):
        if file_sha256(root / filename) != result[field]:
            raise ValueError(f"continuation output drift: {filename}")
    events = read_jsonl(root / "effective-events.jsonl")
    request_by_id = {str(row["request_id"]): row for row in bundle["request_index"]}
    if (
        len(events) != result["request_count"]
        or [str(row["request_id"]) for row in events] != list(request_by_id)
        or any(row.get("validation_status") != "success" for row in events)
    ):
        raise ValueError("continuation effective-event exact coverage drift")
    votes: dict[str, list[int]] = defaultdict(list)
    for event in events:
        source = request_by_id[str(event["request_id"])]
        if any(
            event.get(field) != source[field]
            for field in (
                "shard_id",
                "window_id",
                "window_index",
                "response_id",
                "replica_index",
                "body_sha256",
                "focal_unit_ids",
            )
        ):
            raise ValueError("continuation frozen event/request drift")
        for decision in event["decisions"]:
            votes[str(decision["unit_id"])].append(int(event["replica_index"]))
    pending = {
        str(unit["unit_id"])
        for unit in bundle["units"]
        if unit["assignment_route"] == "openai_pending"
    }
    if set(votes) != pending or any(sorted(votes[unit_id]) != [0, 1, 2] for unit_id in pending):
        raise ValueError("continuation frozen three-vote coverage drift")
    # Attempt-aware raw/provider/status validation uses only copied evidence and bundle.
    for binding in _all_attempts(continuation_root, manifest):
        _validate_collected_attempt(
            run_root=continuation_root,
            manifest=manifest,
            bundle=bundle,
            binding=binding,
            bundle_root=root / "campaign-bundle",
        )
    source_line_sha: dict[str, str] = {}
    source_shard: dict[str, str] = {}
    for shard in bundle["shards"]:
        for request_id, raw_line, _row in _exact_jsonl_rows(
            root / "campaign-bundle" / shard["path"]
        ):
            source_line_sha[request_id] = hashlib.sha256(raw_line).hexdigest()
            source_shard[request_id] = str(shard["shard_id"])
    observed_primary: set[str] = set()
    response_attempt: dict[str, str] = {}
    request_metadata = {
        str(row["request_id"]): row for row in bundle["request_index"]
    }
    for binding in manifest["attempts"]:
        rows = _exact_jsonl_rows(
            continuation_root / str(binding["input_relative_path"])
        )
        for request_id, raw_line, _row in rows:
            response_id = str(request_metadata[request_id]["response_id"])
            prior_attempt = response_attempt.setdefault(
                response_id, str(binding["attempt_id"])
            )
            if (
                source_shard.get(request_id) == "shard-005"
                or hashlib.sha256(raw_line).hexdigest()
                != source_line_sha.get(request_id)
                or request_id in observed_primary
                or prior_attempt != binding["attempt_id"]
            ):
                raise ValueError("frozen primary byte/union/response-affinity drift")
            observed_primary.add(request_id)
    expected_primary = {
        request_id for request_id, shard_id in source_shard.items() if shard_id != "shard-005"
    }
    if observed_primary != expected_primary:
        raise ValueError("frozen primary exact-union drift")
    recovery = _load_object(continuation_root / "failed-only-recovery/manifest.json")
    _verify(recovery, "recovery_manifest_sha256", "frozen recovery manifest")
    recovery_binding = recovery["attempt"]
    recovery_rows = _exact_jsonl_rows(
        continuation_root / str(recovery_binding["input_relative_path"])
    )
    inherited_failures = [
        str(event["request_id"])
        for event in read_jsonl(
            continuation_root
            / "inherited-calibration-run/shards/shard-005/events.jsonl"
        )
        if event.get("validation_status") != "success"
    ]
    primary_failures = []
    for binding in manifest["attempts"]:
        primary_failures.extend(
            str(event["request_id"])
            for event in read_jsonl(
                continuation_root / "attempts" / binding["attempt_id"] / "events.jsonl"
            )
            if event.get("validation_status") != "success"
        )
    failed_set = set(inherited_failures + primary_failures)
    request_order = [str(row["request_id"]) for row in bundle["request_index"]]
    ordered_failed = [request_id for request_id in request_order if request_id in failed_set]
    if (
        [row[0] for row in recovery_rows] != ordered_failed
        or any(
            hashlib.sha256(raw_line).hexdigest() != source_line_sha.get(request_id)
            for request_id, raw_line, _row in recovery_rows
        )
    ):
        raise ValueError("frozen failed-only recovery byte/exact-union drift")
    cost_paths = sorted((continuation_root / "cost-status").glob("receipt-*.json"))
    previous = None
    warning_sticky = False
    last_cost = None
    for path in cost_paths:
        cost_status = _load_object(path)
        _verify(cost_status, "cost_status_sha256", "frozen cost status")
        if (
            cost_status.get("previous_cost_status_sha256") != previous
            or (warning_sticky and cost_status.get("warning_threshold_reached") is not True)
        ):
            raise ValueError("frozen cost-status chain drift")
        warning_sticky = warning_sticky or bool(
            cost_status.get("warning_threshold_reached")
        )
        previous = cost_status["cost_status_sha256"]
        last_cost = cost_status
    if (
        last_cost is None
        or last_cost["cost_status_sha256"] != result["final_cost_status_sha256"]
        or not math.isclose(
            float(last_cost["known_actual_cost_usd"]),
            float(result["actual_total_cost_usd"]),
            rel_tol=0,
            abs_tol=1e-9,
        )
        or float(last_cost["active_calibrated_reservations_usd"]) != 0.0
        or bool(last_cost["hard_stop_crossed_after_inflight_attempt"])
        != bool(result["hard_stop_crossed_after_inflight_attempt"])
    ):
        raise ValueError("frozen final cost-status binding drift")
    return {"manifest": result, "inventory": inventory, "bundle": bundle}
