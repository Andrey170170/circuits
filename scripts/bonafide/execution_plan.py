"""Build and validate a deterministic compound execution plan for final traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "bonafide-trace-execution-plan/v1"
FEATURE_NAMES = (
    "intercept",
    "candidate_mlp_edge_count",
    "planned_jacobian_target_chunk_executions",
    "input_token_count",
)
DEFAULT_SHARD_COUNT = 12
MINIMUM_PREDICTION_SECONDS = 1.0
PATHOLOGICAL_EDGE_THRESHOLD = 10_000_000
DEFAULT_HISTORICAL_SUMMARY = Path(
    "/scratch/general/vast/u1653998/circuits/results/bonafide/performance/"
    "wave2c-instrumented-summary.jsonl"
)
DEFAULT_HISTORICAL_SUMMARY_SHA256 = (
    "a63f1447e778f67be80112946ac927f1f167c03f4cfb83041bfb108171b68d03"
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _finite_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _counter(record: Mapping[str, Any], name: str) -> float:
    instrumentation = record.get("instrumentation")
    if not isinstance(instrumentation, Mapping):
        raise ValueError(f"record {record.get('artifact_id')!r} lacks instrumentation")
    counters = instrumentation.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError(f"record {record.get('artifact_id')!r} lacks counters")
    return _finite_number(counters.get(name), name)


def _training_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, float]]:
    complete = [record for record in records if record.get("status") == "complete"]
    if not complete:
        raise ValueError("historical summary contains no complete measured traces")
    seen: dict[str, tuple[float, ...]] = {}
    rows: list[dict[str, float]] = []
    for record in complete:
        source_id = record.get("source_artifact_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("complete historical trace lacks source_artifact_id")
        row = {
            "candidate_mlp_edge_count": _counter(record, "candidate_mlp_edge_count"),
            "planned_jacobian_target_chunk_executions": _counter(
                record, "planned_jacobian_target_chunk_executions"
            ),
            "input_token_count": _finite_number(
                record.get("input_token_count"), "input_token_count", minimum=1.0
            ),
            "trace_wall_seconds": _finite_number(
                record.get("trace_wall_seconds"), "trace_wall_seconds", minimum=0.0
            ),
        }
        signature = tuple(row.values())
        if source_id in seen:
            if seen[source_id] != signature:
                raise ValueError(f"conflicting historical records for {source_id}")
            raise ValueError(f"duplicate historical record for {source_id}")
        seen[source_id] = signature
        rows.append(row)
    return rows


def fit_cost_model(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _training_rows(records)
    x = np.asarray(
        [
            [
                1.0,
                row["candidate_mlp_edge_count"],
                row["planned_jacobian_target_chunk_executions"],
                row["input_token_count"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray([row["trace_wall_seconds"] for row in rows], dtype=np.float64)
    coefficients, _, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
    if rank != len(FEATURE_NAMES):
        raise ValueError(f"cost-model design matrix is rank deficient: {rank}")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(singular_values)):
        raise ValueError("cost-model fit produced non-finite values")
    predictions = x @ coefficients
    residual_sum = float(np.square(y - predictions).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    r_squared = 1.0 - residual_sum / total_sum
    if not math.isfinite(r_squared) or r_squared < 0.0:
        raise ValueError(f"cost-model fit has invalid R-squared: {r_squared}")
    ranges = {
        name: {
            "min": float(min(row[name] for row in rows)),
            "max": float(max(row[name] for row in rows)),
        }
        for name in FEATURE_NAMES[1:]
    }
    ranges["trace_wall_seconds"] = {"min": float(y.min()), "max": float(y.max())}
    return {
        "kind": "ordinary_least_squares",
        "target": "trace_wall_seconds",
        "feature_names": list(FEATURE_NAMES),
        "coefficients": {
            name: float(value) for name, value in zip(FEATURE_NAMES, coefficients, strict=True)
        },
        "r_squared_in_sample": float(r_squared),
        "training_record_count": len(rows),
        "training_ranges": ranges,
        "design_matrix_rank": int(rank),
        "minimum_prediction_seconds": MINIMUM_PREDICTION_SECONDS,
        "negative_or_small_predictions_are_clamped": True,
    }


def _predict(model: Mapping[str, Any], metrics: Mapping[str, float]) -> tuple[float, float, bool]:
    coefficients = model["coefficients"]
    raw = float(coefficients["intercept"]) + sum(
        float(coefficients[name]) * float(metrics[name]) for name in FEATURE_NAMES[1:]
    )
    if not math.isfinite(raw):
        raise ValueError("cost-model prediction is non-finite")
    minimum = float(model["minimum_prediction_seconds"])
    estimated = max(minimum, raw)
    return raw, estimated, estimated != raw


def _refinement_index(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    index: dict[str, dict[str, float]] = {}
    for record in records:
        if record.get("status") != "complete":
            continue
        artifact_id = record.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("complete refinement probe lacks artifact_id")
        metrics = {
            "candidate_mlp_edge_count": _counter(record, "candidate_mlp_edge_count"),
            "planned_jacobian_target_chunk_executions": _counter(
                record, "planned_jacobian_target_chunk_executions"
            ),
            "input_token_count": _finite_number(
                record.get("input_token_count"), "input_token_count", minimum=1.0
            ),
        }
        previous = index.get(artifact_id)
        if previous is not None:
            if previous != metrics:
                raise ValueError(f"conflicting refinement metrics for {artifact_id}")
            raise ValueError(f"duplicate complete refinement probe {artifact_id}")
        index[artifact_id] = metrics
    if not index:
        raise ValueError("refinement summary contains no complete probes")
    return index


def _manifest_items(manifest: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    wave_ids: set[str] = set()
    artifact_ids: set[str] = set()
    for wave in manifest.get("waves", []):
        wave_id = wave.get("wave_id")
        if not isinstance(wave_id, str) or not wave_id or wave_id in wave_ids:
            raise ValueError(f"invalid or duplicate source wave_id {wave_id!r}")
        wave_ids.add(wave_id)
        for item in wave.get("items", []):
            artifact_id = item.get("artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id or artifact_id in artifact_ids:
                raise ValueError(f"invalid or duplicate source artifact_id {artifact_id!r}")
            artifact_ids.add(artifact_id)
            pairs.append((wave, item))
    if not pairs:
        raise ValueError("final trace manifest contains no work items")
    return pairs


def _item_ref(
    wave: Mapping[str, Any],
    item: Mapping[str, Any],
    metrics: Mapping[str, float],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    final_selection = item.get("target_selection", {}).get("final_selection", {})
    probe_id = final_selection.get("source_refinement_probe_id")
    if not isinstance(probe_id, str) or not probe_id:
        raise ValueError(f"final target {item.get('artifact_id')} lacks source probe identity")
    diagnostics = final_selection.get("refinement_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError(f"final target {item.get('artifact_id')} lacks refinement diagnostics")
    diagnosed_edges = _finite_number(
        diagnostics.get("candidate_mlp_edge_count"), "diagnostic candidate_mlp_edge_count"
    )
    if diagnosed_edges != metrics["candidate_mlp_edge_count"]:
        raise ValueError(f"conflicting candidate-edge metrics for {item.get('artifact_id')}")
    raw, estimated, clamped = _predict(model, metrics)
    return {
        "source_wave_id": wave["wave_id"],
        "source_artifact_id": item["artifact_id"],
        "source_refinement_probe_id": probe_id,
        "example_id": item["example"]["example_id"],
        "corpus_role": wave.get("corpus_role"),
        "cluster_fit_eligible": wave.get("cluster_fit_eligible"),
        "target_response_positions": item["target_selection"]["response_token_positions"],
        "token_text": diagnostics.get("token_text"),
        "selection_reasons": final_selection.get("selection_reasons", []),
        "workload": {name: int(metrics[name]) for name in FEATURE_NAMES[1:]},
        "raw_estimated_seconds": raw,
        "estimated_seconds": estimated,
        "prediction_was_clamped": clamped,
    }


def _lpt_shards(items: list[dict[str, Any]], shard_count: int) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if shard_count > len(items):
        raise ValueError("shard_count cannot exceed routine target count")
    shards = [{"shard_index": i, "items": [], "estimated_seconds": 0.0} for i in range(shard_count)]
    ordered = sorted(
        items,
        key=lambda item: (
            -item["estimated_seconds"], item["source_wave_id"], item["source_artifact_id"]
        ),
    )
    for item in ordered:
        shard = min(shards, key=lambda value: (value["estimated_seconds"], value["shard_index"]))
        shard["items"].append(item)
        shard["estimated_seconds"] += item["estimated_seconds"]
    for shard in shards:
        shard["items"].sort(
            key=lambda item: (
                -item["estimated_seconds"], item["source_wave_id"], item["source_artifact_id"]
            )
        )
        shard["item_count"] = len(shard["items"])
    return shards


def build_execution_plan(
    *,
    manifest_path: Path,
    config_path: Path,
    historical_summary_path: Path,
    refinement_summary_path: Path,
    shard_count: int = DEFAULT_SHARD_COUNT,
    expected_historical_sha256: str = DEFAULT_HISTORICAL_SUMMARY_SHA256,
) -> dict[str, Any]:
    paths = [
        manifest_path.resolve(),
        config_path.resolve(),
        historical_summary_path.resolve(),
        refinement_summary_path.resolve(),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_path, config_path, historical_summary_path, refinement_summary_path = paths
    manifest_sha = sha256_file(manifest_path)
    config_sha = sha256_file(config_path)
    historical_sha = sha256_file(historical_summary_path)
    refinement_sha = sha256_file(refinement_summary_path)
    if historical_sha != expected_historical_sha256:
        raise ValueError("historical Wave2c summary SHA-256 drift")
    manifest = _load_json(manifest_path)
    config = _load_json(config_path)
    expected_refinement = (
        manifest.get("source_artifacts", {}).get("refinement_summary", {}).get("sha256")
    )
    if expected_refinement != refinement_sha:
        raise ValueError("refinement summary SHA-256 disagrees with final manifest")

    historical_records = _load_jsonl(historical_summary_path)
    refinement_records = _load_jsonl(refinement_summary_path)
    model = fit_cost_model(historical_records)
    probes = _refinement_index(refinement_records)
    routine: list[dict[str, Any]] = []
    extremes: list[dict[str, Any]] = []
    for wave, item in _manifest_items(manifest):
        probe_id = item.get("target_selection", {}).get("final_selection", {}).get(
            "source_refinement_probe_id"
        )
        if probe_id not in probes:
            raise ValueError(f"missing refinement metrics for final target {item.get('artifact_id')}")
        ref = _item_ref(wave, item, probes[probe_id], model)
        if wave.get("extreme_workload_isolation") is True:
            extremes.append(ref)
        else:
            routine.append(ref)
    if len(extremes) != 4:
        raise ValueError(f"expected exactly four manifest-marked extremes, found {len(extremes)}")

    pathological = [
        item
        for item in extremes
        if item["workload"]["candidate_mlp_edge_count"] >= PATHOLOGICAL_EDGE_THRESHOLD
    ]
    preflight = [item for item in extremes if item not in pathological]
    if len(pathological) != 1:
        raise ValueError("expected exactly one pathological extreme target")
    preflight.sort(
        key=lambda item: (
            item["estimated_seconds"], item["source_wave_id"], item["source_artifact_id"]
        )
    )
    pathological.sort(key=lambda item: (item["source_wave_id"], item["source_artifact_id"]))
    shards = _lpt_shards(routine, shard_count)
    sources = {
        "final_trace_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "canonical_sha256": hashlib.sha256(canonical_json(manifest)).hexdigest(),
        },
        "trace_run_config": {
            "path": str(config_path),
            "sha256": config_sha,
            "canonical_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
        },
        "historical_wave2c_summary": {
            "path": str(historical_summary_path),
            "sha256": historical_sha,
        },
        "refinement_probe_summary": {
            "path": str(refinement_summary_path),
            "sha256": refinement_sha,
        },
    }
    tasks = [
        {
            "task_index": shard["shard_index"],
            "task_kind": "routine",
            "source_index": shard["shard_index"],
            "item_count": shard["item_count"],
            "estimated_seconds": shard["estimated_seconds"],
        }
        for shard in shards
    ]
    for item in preflight:
        tasks.append(
            {
                "task_index": len(tasks),
                "task_kind": "extreme_preflight",
                "source_index": preflight.index(item),
                "item_count": 1,
                "estimated_seconds": item["estimated_seconds"],
            }
        )
    tasks.append(
        {
            "task_index": len(tasks),
            "task_kind": "pathological_manual",
            "source_index": 0,
            "item_count": 1,
            "estimated_seconds": pathological[0]["estimated_seconds"],
            "requires_explicit_manual_opt_in": True,
        }
    )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
        "cost_model": model,
        "sharding": {
            "algorithm": "longest_processing_time_first",
            "within_shard_order": "estimated_seconds_descending",
            "routine_shard_count": shard_count,
            "routine_target_count": len(routine),
            "routine_total_estimated_seconds": sum(item["estimated_seconds"] for item in routine),
            "shards": shards,
        },
        "extremes": {
            "manifest_marked_count": len(extremes),
            "preflight": preflight,
            "manual_pathological": pathological,
            "pathological_candidate_edge_threshold": PATHOLOGICAL_EDGE_THRESHOLD,
        },
        "tasks": tasks,
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json(plan)).hexdigest()
    validate_execution_plan(plan, manifest=manifest, verify_sources=True)
    return plan


def validate_execution_plan(
    plan: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    verify_sources: bool = True,
) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported execution-plan schema {plan.get('schema_version')!r}")
    core = dict(plan)
    plan_sha = core.pop("plan_sha256", None)
    actual_plan_sha = hashlib.sha256(canonical_json(core)).hexdigest()
    if plan_sha != actual_plan_sha:
        raise ValueError("execution plan SHA-256 is invalid")
    sources = plan.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("execution plan lacks sources")
    if verify_sources:
        for name in (
            "final_trace_manifest",
            "trace_run_config",
            "historical_wave2c_summary",
            "refinement_probe_summary",
        ):
            source = sources.get(name)
            if not isinstance(source, Mapping):
                raise ValueError(f"execution plan lacks source {name}")
            path = Path(str(source.get("path", "")))
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"execution-plan source must be an existing absolute path: {path}")
            if sha256_file(path) != source.get("sha256"):
                raise ValueError(f"execution-plan source hash drift: {name}")
            if name in ("final_trace_manifest", "trace_run_config"):
                canonical_sha = hashlib.sha256(canonical_json(_load_json(path))).hexdigest()
                if source.get("canonical_sha256") != canonical_sha:
                    raise ValueError(f"execution-plan canonical source hash drift: {name}")
    if manifest is None:
        manifest = _load_json(Path(sources["final_trace_manifest"]["path"]))
    manifest_pairs = _manifest_items(manifest)
    expected_routine = {
        (wave["wave_id"], item["artifact_id"])
        for wave, item in manifest_pairs
        if wave.get("extreme_workload_isolation") is not True
    }
    expected_extreme = {
        (wave["wave_id"], item["artifact_id"])
        for wave, item in manifest_pairs
        if wave.get("extreme_workload_isolation") is True
    }
    sharding = plan.get("sharding")
    extremes = plan.get("extremes")
    if not isinstance(sharding, Mapping) or not isinstance(extremes, Mapping):
        raise ValueError("execution plan lacks sharding or extremes")
    shards = sharding.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("execution plan has no routine shards")
    if [shard.get("shard_index") for shard in shards] != list(range(len(shards))):
        raise ValueError("routine shard indices must be contiguous and ordered")
    routine_items: list[Mapping[str, Any]] = []
    seen_routine_refs: set[tuple[Any, Any]] = set()
    for shard in shards:
        items = shard.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("routine shard must contain items")
        if shard.get("item_count") != len(items):
            raise ValueError("routine shard item_count is inconsistent")
        for item in items:
            ref = (item.get("source_wave_id"), item.get("source_artifact_id"))
            if ref in seen_routine_refs:
                raise ValueError("execution plan contains duplicate target assignments")
            seen_routine_refs.add(ref)
        estimates = [
            _finite_number(item.get("estimated_seconds"), "estimated_seconds")
            for item in items
        ]
        if estimates != sorted(estimates, reverse=True):
            raise ValueError("routine shard is not ordered high-cost-first")
        if not math.isclose(
            _finite_number(shard.get("estimated_seconds"), "shard estimated_seconds"),
            sum(estimates),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("routine shard estimated_seconds is inconsistent")
        routine_items.extend(items)
    extreme_items: list[Mapping[str, Any]] = []
    for group in ("preflight", "manual_pathological"):
        items = extremes.get(group)
        if not isinstance(items, list):
            raise ValueError(f"extreme group {group} must be a list")
        extreme_items.extend(items)
    all_items = [*routine_items, *extreme_items]
    actual = [(item.get("source_wave_id"), item.get("source_artifact_id")) for item in all_items]
    if len(actual) != len(set(actual)):
        raise ValueError("execution plan contains duplicate target assignments")
    routine_refs = {
        (item.get("source_wave_id"), item.get("source_artifact_id")) for item in routine_items
    }
    extreme_refs = {
        (item.get("source_wave_id"), item.get("source_artifact_id")) for item in extreme_items
    }
    if routine_refs != expected_routine:
        raise ValueError("routine shard membership disagrees with manifest non-extreme targets")
    if extreme_refs != expected_extreme:
        raise ValueError("extreme membership disagrees with manifest-marked targets")
    preflight = extremes["preflight"]
    pathological = extremes["manual_pathological"]
    if len(preflight) != 3 or len(pathological) != 1:
        raise ValueError("extreme groups must contain three preflights and one manual target")
    threshold = int(extremes.get("pathological_candidate_edge_threshold", -1))
    if threshold != PATHOLOGICAL_EDGE_THRESHOLD:
        raise ValueError("pathological threshold is inconsistent")
    if any(item["workload"]["candidate_mlp_edge_count"] >= threshold for item in preflight):
        raise ValueError("pathological target was assigned to preflight")
    if pathological[0]["workload"]["candidate_mlp_edge_count"] < threshold:
        raise ValueError("manual target does not meet pathological threshold")
    if int(sharding.get("routine_target_count", -1)) != sum(
        len(shard["items"]) for shard in shards
    ):
        raise ValueError("routine target count is inconsistent")
    aggregate = sum(float(shard["estimated_seconds"]) for shard in shards)
    if not math.isclose(
        _finite_number(
            sharding.get("routine_total_estimated_seconds"),
            "routine_total_estimated_seconds",
        ),
        aggregate,
        rel_tol=1e-12,
        abs_tol=1e-8,
    ):
        raise ValueError("routine aggregate estimate is inconsistent")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or [task.get("task_index") for task in tasks] != list(
        range(len(tasks))
    ):
        raise ValueError("execution tasks must have contiguous ordered indices")
    expected_task_count = len(shards) + len(preflight) + len(pathological)
    if len(tasks) != expected_task_count:
        raise ValueError("execution task count is inconsistent")
    for index, task in enumerate(tasks):
        if index < len(shards):
            expected_kind, expected_source, expected_items = "routine", index, shards[index]["item_count"]
        elif index < len(shards) + len(preflight):
            expected_kind = "extreme_preflight"
            expected_source = index - len(shards)
            expected_items = 1
        else:
            expected_kind, expected_source, expected_items = "pathological_manual", 0, 1
            if task.get("requires_explicit_manual_opt_in") is not True:
                raise ValueError("pathological task must require explicit manual opt-in")
        if (
            task.get("task_kind") != expected_kind
            or task.get("source_index") != expected_source
            or task.get("item_count") != expected_items
        ):
            raise ValueError(f"execution task {index} does not resolve to its source group")


def write_execution_plan(path: Path, plan: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(plan) + b"\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, default=DEFAULT_HISTORICAL_SUMMARY)
    parser.add_argument("--refinement-summary", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_execution_plan(
        manifest_path=args.manifest,
        config_path=args.config,
        historical_summary_path=args.historical_summary,
        refinement_summary_path=args.refinement_summary,
        shard_count=args.shard_count,
    )
    write_execution_plan(args.output, plan)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "routine_shards": plan["sharding"]["routine_shard_count"],
        "routine_targets": plan["sharding"]["routine_target_count"],
        "extreme_targets": plan["extremes"]["manifest_marked_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
