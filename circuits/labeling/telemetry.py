"""Deterministic cost and GPU telemetry rollups for labeling runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.runtime import load_run_manifest
from circuits.labeling.schema import TelemetryRecord

USAGE_FIELDS = (
    "input_tokens",
    "uncached_input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
)


def _empty_api_bucket() -> dict[str, Any]:
    return {
        "event_count": 0,
        "parse_status_counts": {},
        "usage": dict.fromkeys(USAGE_FIELDS, 0),
        "usage_missing_event_counts": dict.fromkeys(USAGE_FIELDS, 0),
        "known_cost_usd": 0.0,
        "complete_cost_event_count": 0,
        "incomplete_cost_event_count": 0,
    }


def _add_api_event(bucket: dict[str, Any], telemetry: TelemetryRecord) -> None:
    bucket["event_count"] += 1
    statuses = Counter(bucket["parse_status_counts"])
    statuses[telemetry.parse_status] += 1
    bucket["parse_status_counts"] = dict(sorted(statuses.items()))
    for field in USAGE_FIELDS:
        value = getattr(telemetry.usage, field)
        if value is None:
            bucket["usage_missing_event_counts"][field] += 1
        else:
            bucket["usage"][field] += value
    cost = telemetry.cost
    if cost is not None and cost.complete and cost.total_cost is not None:
        bucket["known_cost_usd"] += cost.total_cost
        bucket["complete_cost_event_count"] += 1
    else:
        bucket["incomplete_cost_event_count"] += 1


def _api_sources(run_root: Path) -> list[tuple[str, Path]]:
    values: list[tuple[str, Path]] = []
    for stage in ("candidate_generation", "cluster_summary"):
        values.extend(
            ("canonical", path)
            for path in sorted((run_root / "telemetry" / stage).glob("*.json"))
        )
    retry_root = run_root / "provider_batches"
    for path in sorted(retry_root.glob("*/retries/**/*telemetry.json")):
        source = (
            "archived_original"
            if path.name == "original-telemetry.json"
            else "retry_attempt"
        )
        values.append((source, path))
    source_order = {"canonical": 0, "archived_original": 1, "retry_attempt": 2}
    return sorted(values, key=lambda item: (source_order[item[0]], item[1].as_posix()))


def _provider_summary(run_root: Path, run_id: str) -> dict[str, Any]:
    overall = _empty_api_bucket()
    by_stage: dict[str, dict[str, Any]] = {}
    by_provider: dict[str, dict[str, Any]] = {}
    source_file_counts: Counter[str] = Counter()
    included_source_counts: Counter[str] = Counter()
    seen: dict[str, str] = {}
    files: list[dict[str, Any]] = []

    for source, path in _api_sources(run_root):
        source_file_counts[source] += 1
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            telemetry = TelemetryRecord.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid generation telemetry: {path}") from error
        if telemetry.run_id != run_id:
            raise ValueError(f"generation telemetry run identity mismatch: {path}")
        identity = canonical_sha256(telemetry.model_dump(mode="json"))
        relative = path.relative_to(run_root).as_posix()
        duplicate_of = seen.get(identity)
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "source": source,
                "event_identity_sha256": identity,
                "included": duplicate_of is None,
                "duplicate_of": duplicate_of,
            }
        )
        if duplicate_of is not None:
            continue
        seen[identity] = relative
        included_source_counts[source] += 1
        _add_api_event(overall, telemetry)
        stage_bucket = by_stage.setdefault(telemetry.stage, _empty_api_bucket())
        _add_api_event(stage_bucket, telemetry)
        provider_bucket = by_provider.setdefault(telemetry.backend, _empty_api_bucket())
        _add_api_event(provider_bucket, telemetry)

    return {
        **overall,
        "source_file_counts": dict(sorted(source_file_counts.items())),
        "included_source_counts": dict(sorted(included_source_counts.items())),
        "duplicate_file_count": len(files) - overall["event_count"],
        "by_stage": dict(sorted(by_stage.items())),
        "by_provider": dict(sorted(by_provider.items())),
        "files": files,
    }


def _empty_gpu_bucket() -> dict[str, Any]:
    return {
        "record_count": 0,
        "elapsed_seconds": 0.0,
        "gpu_hours": 0.0,
        "completed_cluster_count": 0,
        "skipped_cluster_count": 0,
        "peak_hbm_bytes_max": None,
        "peak_host_rss_kib_max": None,
    }


def _add_gpu_record(bucket: dict[str, Any], value: dict[str, Any]) -> None:
    bucket["record_count"] += 1
    bucket["elapsed_seconds"] += float(value["elapsed_seconds"])
    bucket["gpu_hours"] += float(value["gpu_hours"])
    counts = value["counts"]
    bucket["completed_cluster_count"] += int(counts.get("completed", 0))
    bucket["skipped_cluster_count"] += int(counts.get("skipped", 0))
    for field in ("peak_hbm_bytes", "peak_host_rss_kib"):
        raw = value.get(field)
        if raw is not None:
            current = bucket[f"{field}_max"]
            bucket[f"{field}_max"] = (
                int(raw) if current is None else max(current, int(raw))
            )


def _local_scoring_summary(run_root: Path, run_id: str) -> dict[str, Any]:
    overall = _empty_gpu_bucket()
    by_phase: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    files: list[dict[str, Any]] = []
    for path in sorted((run_root / "telemetry" / "local_scoring").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid local scoring telemetry: {path}") from error
        if value.get("schema_version") != "adag.labeling.local-scoring-telemetry.v1":
            raise ValueError(f"unsupported local scoring telemetry: {path}")
        if value.get("run_id") != run_id:
            raise ValueError(f"local scoring telemetry run identity mismatch: {path}")
        for field in ("phase", "elapsed_seconds", "gpu_hours", "counts"):
            if field not in value:
                raise ValueError(f"local scoring telemetry lacks {field}: {path}")
        identity = canonical_sha256(value)
        relative = path.relative_to(run_root).as_posix()
        duplicate = identity in seen
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "event_identity_sha256": identity,
                "included": not duplicate,
            }
        )
        if duplicate:
            continue
        seen.add(identity)
        _add_gpu_record(overall, value)
        phase_bucket = by_phase.setdefault(str(value["phase"]), _empty_gpu_bucket())
        _add_gpu_record(phase_bucket, value)
    return {
        **overall,
        "duplicate_file_count": len(files) - overall["record_count"],
        "by_phase": dict(sorted(by_phase.items())),
        "files": files,
    }


def summarize_telemetry(*, run_root: Path) -> dict[str, Any]:
    """Read and aggregate all billable labeling and successful GPU telemetry."""

    manifest = load_run_manifest(run_root)
    return {
        "schema_version": "adag.labeling.telemetry-summary.v1",
        "run_id": manifest["run_id"],
        "source_run_manifest_sha256": manifest["manifest_sha256"],
        "provider_api": _provider_summary(run_root, manifest["run_id"]),
        "local_scoring": _local_scoring_summary(run_root, manifest["run_id"]),
    }
