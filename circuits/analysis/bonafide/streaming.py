"""Per-response streaming builders for dense features and multiplex shards."""

from __future__ import annotations

import json
import os
import resource
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from circuits.analysis.bonafide.build_plan import (
    dense_fit_weights,
    task_records,
    validate_downstream_plan,
)
from circuits.analysis.bonafide.canonical import (
    canonical_json,
    canonical_sha256,
    file_sha256,
)
from circuits.analysis.bonafide.identity import OccurrenceKey, SignedBasisKey
from circuits.analysis.bonafide.multiplex import (
    BasisTargetSummary,
    TargetSlice,
    build_target_slice,
    validate_target_slice_round_trip,
)
from circuits.tracing.artifact import load_compact_trace

FEATURE_SHARD_SCHEMA = "adag.bonafide.dense-feature-shard.v1"
MULTIPLEX_SHARD_SCHEMA = "adag.bonafide.response-multiplex-shard.v1"
DEFAULT_PARQUET_BUFFER_ROWS = 25_000
DEFAULT_PARQUET_BUFFER_BATCHES = 16


def _basis_columns(prefix: str = "") -> list[pa.Field]:
    return [
        pa.field(f"{prefix}model_id", pa.string(), nullable=False),
        pa.field(f"{prefix}model_revision", pa.string(), nullable=False),
        pa.field(f"{prefix}layer", pa.int32(), nullable=False),
        pa.field(f"{prefix}neuron_index", pa.int64(), nullable=False),
        pa.field(f"{prefix}polarity", pa.string(), nullable=False),
    ]


TARGET_SCHEMA = pa.schema(
    [
        pa.field("atlas_trace_index", pa.int64(), nullable=False),
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("base_question_id", pa.string(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("prediction_position", pa.int32(), nullable=True),
        pa.field("target_token_id", pa.int64(), nullable=False),
        pa.field("target_token_text", pa.string(), nullable=True),
        pa.field("target_logit", pa.float64(), nullable=True),
        pa.field("target_probability", pa.float64(), nullable=True),
        pa.field("corpus_role", pa.string(), nullable=False),
        pa.field("cluster_fit_eligible", pa.bool_(), nullable=False),
        pa.field("fit_weight", pa.float64(), nullable=False),
        pa.field("artifact_manifest_sha256", pa.string(), nullable=False),
        pa.field("artifact_payload_sha256", pa.string(), nullable=False),
        pa.field("condition_json", pa.string(), nullable=False),
        pa.field("selection_reasons_json", pa.string(), nullable=False),
        pa.field("local_ci_count", pa.int32(), nullable=False),
        pa.field("local_labels", pa.list_(pa.string()), nullable=False),
    ]
)

FEATURE_OBSERVATION_SCHEMA = pa.schema(
    [
        *_basis_columns(),
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("base_question_id", pa.string(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("fit_weight", pa.float64(), nullable=False),
        pa.field("signed_attribution", pa.float64(), nullable=False),
        pa.field("absolute_attribution_mass", pa.float64(), nullable=False),
        pa.field("occurrence_count", pa.int32(), nullable=False),
        pa.field("mean_activation", pa.float64(), nullable=False),
        pa.field("attribution_profile", pa.list_(pa.float64()), nullable=False),
        pa.field("attribution_support", pa.list_(pa.bool_()), nullable=False),
        pa.field("contribution_profile", pa.list_(pa.float64()), nullable=False),
        pa.field("contribution_support", pa.list_(pa.bool_()), nullable=False),
        pa.field("in_degree", pa.int64(), nullable=False),
        pa.field("out_degree", pa.int64(), nullable=False),
    ]
)

TARGET_STATS_SCHEMA = pa.schema(
    [
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("node_count", pa.int64(), nullable=False),
        pa.field("edge_count", pa.int64(), nullable=False),
        pa.field("signed_basis_count", pa.int64(), nullable=False),
        pa.field("occurrence_count", pa.int64(), nullable=False),
        pa.field("attribution_profile_cell_count", pa.int64(), nullable=False),
        pa.field("attribution_profile_column_count", pa.int64(), nullable=False),
        pa.field("attribution_supported_cell_count", pa.int64(), nullable=False),
        pa.field("attribution_nonzero_count", pa.int64(), nullable=False),
        pa.field("contribution_profile_cell_count", pa.int64(), nullable=False),
        pa.field("contribution_profile_column_count", pa.int64(), nullable=False),
        pa.field("contribution_supported_cell_count", pa.int64(), nullable=False),
        pa.field("contribution_nonzero_count", pa.int64(), nullable=False),
        pa.field("source_payload_size_bytes", pa.int64(), nullable=False),
    ]
)

NODE_OCCURRENCE_SCHEMA = pa.schema(
    [
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("token_position", pa.int32(), nullable=False),
        *_basis_columns(),
        pa.field("attribution", pa.float64(), nullable=False),
        pa.field("activation", pa.float64(), nullable=False),
        pa.field("local_label", pa.string(), nullable=False),
    ]
)

EDGE_OCCURRENCE_SCHEMA = pa.schema(
    [
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("source_token_position", pa.int32(), nullable=False),
        *_basis_columns("source_"),
        pa.field("target_token_position", pa.int32(), nullable=False),
        *_basis_columns("target_"),
        pa.field("attribution", pa.float64(), nullable=False),
        pa.field("weight", pa.float64(), nullable=False),
        pa.field("local_label", pa.string(), nullable=False),
    ]
)

TRAJECTORY_SCHEMA = pa.schema(
    [
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        *_basis_columns(),
        pa.field("supported", pa.bool_(), nullable=False),
        pa.field("signed_attribution", pa.float64(), nullable=False),
        pa.field("absolute_attribution_mass", pa.float64(), nullable=False),
        pa.field("occurrence_count", pa.int32(), nullable=False),
        pa.field("mean_activation", pa.float64(), nullable=False),
        pa.field("in_degree", pa.int64(), nullable=False),
        pa.field("out_degree", pa.int64(), nullable=False),
    ]
)

LONGITUDINAL_SCHEMA = pa.schema(
    [
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("left_response_position", pa.int32(), nullable=False),
        pa.field("right_response_position", pa.int32(), nullable=False),
        pa.field("left_trace_unit_id", pa.string(), nullable=False),
        pa.field("right_trace_unit_id", pa.string(), nullable=False),
        *_basis_columns(),
        pa.field("left_token_positions", pa.list_(pa.int32()), nullable=False),
        pa.field("right_token_positions", pa.list_(pa.int32()), nullable=False),
        pa.field("correspondence_kind", pa.string(), nullable=False),
        pa.field("explicitly_noncausal", pa.bool_(), nullable=False),
    ]
)

AGGREGATED_NODE_SUPPORT_SCHEMA = pa.schema(
    [
        pa.field("response_id", pa.string(), nullable=False),
        *_basis_columns(),
        pa.field("support_target_count", pa.int32(), nullable=False),
        pa.field("support_response_positions", pa.list_(pa.int32()), nullable=False),
        pa.field("support_trace_unit_ids", pa.list_(pa.string()), nullable=False),
        pa.field(
            "mean_abs_attribution_over_supported_targets",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "mean_signed_attribution_over_supported_targets",
            pa.float64(),
            nullable=False,
        ),
        pa.field("occurrence_count_sum", pa.int64(), nullable=False),
    ]
)

AGGREGATED_EDGE_SUPPORT_SCHEMA = pa.schema(
    [
        pa.field("response_id", pa.string(), nullable=False),
        *_basis_columns("source_"),
        *_basis_columns("target_"),
        pa.field("support_target_count", pa.int32(), nullable=False),
        pa.field("support_response_positions", pa.list_(pa.int32()), nullable=False),
        pa.field("support_trace_unit_ids", pa.list_(pa.string()), nullable=False),
        pa.field("edge_occurrence_count", pa.int64(), nullable=False),
        pa.field(
            "mean_signed_attribution_over_edge_occurrences",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "mean_abs_attribution_over_edge_occurrences",
            pa.float64(),
            nullable=False,
        ),
        pa.field("mean_weight_over_edge_occurrences", pa.float64(), nullable=False),
    ]
)

BASIS_NODE_SCHEMA = pa.schema(
    [
        *_basis_columns(),
        pa.field("response_id", pa.string(), nullable=False),
    ]
)


class ParquetSink:
    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        *,
        max_buffer_rows: int = DEFAULT_PARQUET_BUFFER_ROWS,
        max_buffer_batches: int = DEFAULT_PARQUET_BUFFER_BATCHES,
    ) -> None:
        if max_buffer_rows < 1 or max_buffer_batches < 1:
            raise ValueError("Parquet buffer limits must be positive")
        self.path = path
        self.schema = schema
        self.max_buffer_rows = max_buffer_rows
        self.max_buffer_batches = max_buffer_batches
        self.writer: pq.ParquetWriter | None = None
        self.row_count = 0
        self.flush_count = 0
        self.write_seconds = 0.0
        self._buffer: list[Mapping[str, Any]] = []
        self._buffer_batches = 0
        self.closed = False

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self.closed:
            raise ValueError(f"cannot write to closed Parquet sink: {self.path}")
        if not rows:
            return
        self.row_count += len(rows)
        offset = 0
        while offset < len(rows):
            available = self.max_buffer_rows - len(self._buffer)
            take = min(available, len(rows) - offset)
            self._buffer.extend(rows[offset : offset + take])
            offset += take
            self._buffer_batches += 1
            if (
                len(self._buffer) >= self.max_buffer_rows
                or self._buffer_batches >= self.max_buffer_batches
            ):
                self.flush()

    def flush(self) -> None:
        if not self._buffer:
            self._buffer_batches = 0
            return
        started = perf_counter()
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        self.writer.write_table(table)
        self.flush_count += 1
        self._buffer.clear()
        self._buffer_batches = 0
        self.write_seconds += perf_counter() - started

    def close(self) -> None:
        if self.closed:
            return
        self.flush()
        if self.writer is None:
            started = perf_counter()
            pq.write_table(
                pa.Table.from_pylist([], schema=self.schema),
                self.path,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
            self.flush_count += 1
            self.write_seconds += perf_counter() - started
        else:
            started = perf_counter()
            self.writer.close()
            self.write_seconds += perf_counter() - started
        self.closed = True

    def performance_record(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "flush_count": self.flush_count,
            "write_seconds": self.write_seconds,
            "max_buffer_rows": self.max_buffer_rows,
            "max_buffer_batches": self.max_buffer_batches,
        }


@dataclass
class StageTimings:
    seconds: dict[str, float]
    calls: dict[str, int]

    @classmethod
    def empty(cls) -> StageTimings:
        return cls(seconds=defaultdict(float), calls=defaultdict(int))

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += perf_counter() - started
            self.calls[name] += 1

    def record(self, name: str, seconds: float, *, calls: int = 1) -> None:
        self.seconds[name] += seconds
        self.calls[name] += calls

    def to_record(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "wall_seconds": self.seconds[name],
                "call_count": self.calls[name],
            }
            for name in sorted(self.seconds)
        }


def _basis_record(basis: SignedBasisKey, prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}model_id": basis.model_id,
        f"{prefix}model_revision": basis.model_revision,
        f"{prefix}layer": basis.layer,
        f"{prefix}neuron_index": basis.neuron_index,
        f"{prefix}polarity": basis.polarity,
    }


def _basis_from_occurrence(
    occurrence: OccurrenceKey,
    target_slice: TargetSlice,
) -> SignedBasisKey:
    return SignedBasisKey(
        model_id=target_slice.model_id,
        model_revision=target_slice.model_revision,
        layer=occurrence.layer,
        neuron_index=occurrence.neuron_index,
        polarity=occurrence.polarity,
    )


def _target_row(
    record: Mapping[str, Any],
    fit_weight: float,
    *,
    local_labels: Sequence[str],
) -> dict[str, Any]:
    return {
        "atlas_trace_index": int(record["atlas_trace_index"]),
        "trace_unit_id": str(record["trace_unit_id"]),
        "source_artifact_id": str(record["source_artifact_id"]),
        "response_id": str(record["response_id"]),
        "base_question_id": str(record["base_question_id"]),
        "response_position": int(record["response_position"]),
        "prediction_position": record.get("prediction_position"),
        "target_token_id": int(record["target_token_id"]),
        "target_token_text": record.get("target_token_text"),
        "target_logit": record.get("target_logit"),
        "target_probability": record.get("target_probability"),
        "corpus_role": str(record["corpus_role"]),
        "cluster_fit_eligible": bool(record["cluster_fit_eligible"]),
        "fit_weight": fit_weight,
        "artifact_manifest_sha256": str(record["artifact_manifest_sha256"]),
        "artifact_payload_sha256": str(record["artifact_payload_sha256"]),
        "condition_json": canonical_json(record["condition"]).decode("utf-8"),
        "selection_reasons_json": canonical_json(record["selection_reasons"]).decode(
            "utf-8"
        ),
        "local_ci_count": len(local_labels),
        "local_labels": list(local_labels),
    }


def _feature_row(
    summary: BasisTargetSummary,
    record: Mapping[str, Any],
    fit_weight: float,
) -> dict[str, Any]:
    return {
        **_basis_record(summary.basis),
        "trace_unit_id": record["trace_unit_id"],
        "source_artifact_id": record["source_artifact_id"],
        "response_id": record["response_id"],
        "base_question_id": record["base_question_id"],
        "response_position": record["response_position"],
        "fit_weight": fit_weight,
        "signed_attribution": summary.signed_attribution,
        "absolute_attribution_mass": summary.absolute_attribution_mass,
        "occurrence_count": summary.occurrence_count,
        "mean_activation": summary.mean_activation,
        "attribution_profile": list(summary.attribution_map),
        "attribution_support": list(summary.attribution_support),
        "contribution_profile": list(summary.contribution_map),
        "contribution_support": list(summary.contribution_support),
        "in_degree": summary.in_degree,
        "out_degree": summary.out_degree,
    }


def _target_stats(
    target_slice: TargetSlice,
    record: Mapping[str, Any],
    *,
    payload_size_bytes: int,
) -> dict[str, Any]:
    attr_cells = [
        value
        for summary in target_slice.basis_summaries
        for value in summary.attribution_map
    ]
    contribution_cells = [
        value
        for summary in target_slice.basis_summaries
        for value in summary.contribution_map
    ]
    attribution_widths = {
        len(summary.attribution_map) for summary in target_slice.basis_summaries
    }
    contribution_widths = {
        len(summary.contribution_map) for summary in target_slice.basis_summaries
    }
    if len(attribution_widths) > 1 or len(contribution_widths) > 1:
        raise ValueError("one target contains inconsistent basis profile widths")
    return {
        "trace_unit_id": target_slice.trace_unit_id,
        "source_artifact_id": record["source_artifact_id"],
        "response_position": target_slice.target_response_position,
        "node_count": len(target_slice.nodes),
        "edge_count": len(target_slice.edges),
        "signed_basis_count": len(target_slice.basis_summaries),
        "occurrence_count": len(target_slice.nodes),
        "attribution_profile_cell_count": len(attr_cells),
        "attribution_profile_column_count": (
            attribution_widths.pop() if attribution_widths else 0
        ),
        "attribution_supported_cell_count": sum(
            value is not None for value in attr_cells
        ),
        "attribution_nonzero_count": sum(
            value is not None and value != 0.0 for value in attr_cells
        ),
        "contribution_profile_cell_count": len(contribution_cells),
        "contribution_profile_column_count": (
            contribution_widths.pop() if contribution_widths else 0
        ),
        "contribution_supported_cell_count": sum(
            value is not None for value in contribution_cells
        ),
        "contribution_nonzero_count": sum(
            value is not None and value != 0.0 for value in contribution_cells
        ),
        "source_payload_size_bytes": payload_size_bytes,
    }


@dataclass
class NodeSupport:
    positions: list[int]
    trace_ids: list[str]
    abs_attribution_sum: float = 0.0
    signed_attribution_sum: float = 0.0
    occurrence_count: int = 0


@dataclass
class EdgeSupport:
    positions: list[int]
    trace_ids: list[str]
    edge_count: int = 0
    attribution_sum: float = 0.0
    abs_attribution_sum: float = 0.0
    weight_sum: float = 0.0


def _validate_existing_shard(
    shard_path: Path,
    *,
    plan_sha256: str,
    task_index: int,
) -> dict[str, Any]:
    manifest_path = shard_path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"existing response shard lacks manifest: {shard_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("response shard manifest must be an object")
    core = dict(manifest)
    recorded_hash = core.pop("manifest_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("existing response shard manifest hash mismatch")
    if manifest.get("plan_sha256") != plan_sha256:
        raise ValueError("existing response shard belongs to another plan")
    if manifest.get("task_index") != task_index:
        raise ValueError("existing response shard task index mismatch")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("existing response shard file inventory is invalid")
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("existing shard file record is invalid")
        path = shard_path / str(item["path"])
        if path.stat().st_size != item.get("size_bytes"):
            raise ValueError(f"existing response shard file size mismatch: {path}")
        if file_sha256(path) != item.get("sha256"):
            raise ValueError(f"existing response shard file hash mismatch: {path}")
    return manifest


def _validate_compatible_build_plans(
    plans: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not plans:
        raise ValueError("at least one downstream build plan is required")
    validated = [
        validate_downstream_plan(
            plan,
            verify_inputs=True,
            verify_code=True,
        )
        for plan in plans
    ]
    lanes = [str(plan["lane"]) for plan in validated]
    if len(set(lanes)) != len(lanes):
        raise ValueError("joint downstream build plans contain a duplicate lane")
    reference = validated[0]
    compatibility_fields = (
        "development",
        "development_targets_per_response",
        "source_inventory",
        "repo_root",
        "code_revision",
        "runtime_environment",
        "uv_lock_sha256",
        "dense_summary",
        "tasks",
    )
    for candidate in validated[1:]:
        for field in compatibility_fields:
            if candidate.get(field) != reference.get(field):
                raise ValueError(f"joint downstream build plans disagree on {field}")
        if candidate["output_root"] == reference["output_root"]:
            raise ValueError("joint downstream lanes require distinct output roots")
    return validated


def build_response_shard(
    *,
    plan: Mapping[str, Any],
    task_index: int,
) -> dict[str, Any]:
    """Build one lane while retaining the joint builder's output contract."""

    lane = str(plan.get("lane"))
    return _build_response_shards(plans=[plan], task_index=task_index)[lane]


def build_joint_response_shards(
    *,
    feature_plan: Mapping[str, Any],
    multiplex_plan: Mapping[str, Any],
    task_index: int,
) -> dict[str, Any]:
    """Build feature and multiplex shards from one validated artifact pass."""

    results = _build_response_shards(
        plans=[feature_plan, multiplex_plan],
        task_index=task_index,
    )
    expected_lanes = {"dense_features", "dense_multiplex"}
    if set(results) != expected_lanes:
        raise ValueError("joint builder requires feature and multiplex plans")
    return {
        "status": (
            "skipped_complete"
            if all(
                result["status"] == "skipped_complete" for result in results.values()
            )
            else "complete"
        ),
        "lanes": results,
    }


def _build_response_shards(
    *,
    plans: Sequence[Mapping[str, Any]],
    task_index: int,
) -> dict[str, dict[str, Any]]:
    validated_plans = _validate_compatible_build_plans(plans)
    reference = validated_plans[0]
    task, records = task_records(reference, task_index=task_index)
    fit_weights = dense_fit_weights(reference)
    start = datetime.now(UTC)
    wall_started = perf_counter()
    timings = StageTimings.empty()
    results: dict[str, dict[str, Any]] = {}
    contexts: dict[str, dict[str, Any]] = {}

    try:
        for validated in validated_plans:
            lane = str(validated["lane"])
            output_root = Path(str(validated["output_root"]))
            shard_name = f"task-{task_index:03d}-{task['response_id']!s}"
            shard_path = output_root / "shards" / shard_name
            if shard_path.exists():
                manifest = _validate_existing_shard(
                    shard_path,
                    plan_sha256=str(validated["plan_sha256"]),
                    task_index=task_index,
                )
                results[lane] = {
                    "status": "skipped_complete",
                    "manifest": manifest,
                }
                continue

            shard_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = shard_path.parent / f".{shard_name}.tmp-{uuid.uuid4().hex}"
            temporary.mkdir()
            sinks: dict[str, ParquetSink] = {
                "targets.parquet": ParquetSink(
                    temporary / "targets.parquet",
                    TARGET_SCHEMA,
                ),
                "target-stats.parquet": ParquetSink(
                    temporary / "target-stats.parquet",
                    TARGET_STATS_SCHEMA,
                ),
            }
            if lane == "dense_features":
                sinks["basis-observations.parquet"] = ParquetSink(
                    temporary / "basis-observations.parquet",
                    FEATURE_OBSERVATION_SCHEMA,
                )
                shard_schema = FEATURE_SHARD_SCHEMA
            elif lane == "dense_multiplex":
                sinks.update(
                    {
                        "node-occurrences.parquet": ParquetSink(
                            temporary / "node-occurrences.parquet",
                            NODE_OCCURRENCE_SCHEMA,
                        ),
                        "edge-occurrences.parquet": ParquetSink(
                            temporary / "edge-occurrences.parquet",
                            EDGE_OCCURRENCE_SCHEMA,
                        ),
                        "trajectory-measurements.parquet": ParquetSink(
                            temporary / "trajectory-measurements.parquet",
                            TRAJECTORY_SCHEMA,
                        ),
                    }
                )
                shard_schema = MULTIPLEX_SHARD_SCHEMA
            else:
                raise ValueError(f"unsupported downstream lane: {lane}")
            contexts[lane] = {
                "plan": validated,
                "shard_path": shard_path,
                "temporary": temporary,
                "sinks": sinks,
                "shard_schema": shard_schema,
            }

        if not contexts:
            return results

        multiplex_active = "dense_multiplex" in contexts
        feature_active = "dense_features" in contexts
        basis_nodes: set[SignedBasisKey] = set()
        node_support: dict[SignedBasisKey, NodeSupport] = {}
        edge_support: dict[tuple[SignedBasisKey, SignedBasisKey], EdgeSupport] = {}
        longitudinal_rows: list[dict[str, Any]] = []
        prior_summaries: dict[SignedBasisKey, BasisTargetSummary] | None = None
        prior_position: int | None = None
        prior_trace_id: str | None = None
        totals: dict[str, int] = defaultdict(int)

        for record in records:
            source_id = str(record["source_artifact_id"])
            trace_id = str(record["trace_unit_id"])
            fit_weight = fit_weights[source_id]
            with timings.measure("artifact_load_validate"):
                loaded = load_compact_trace(str(record["artifact_path"]))
            if loaded.manifest.get("artifact_id") != trace_id:
                raise ValueError("inventory/artifact runtime identity mismatch")
            with timings.measure("dataframe_to_records"):
                node_rows = loaded.circuit_data.df_node.to_dict(orient="records")
                edge_rows = loaded.circuit_data.df_edge.to_dict(orient="records")
            with timings.measure("target_slice_build"):
                target_slice = build_target_slice(
                    response_id=str(record["response_id"]),
                    target_response_position=int(record["response_position"]),
                    trace_unit_id=trace_id,
                    model_id=str(record["model_id"]),
                    model_revision=str(record["model_revision"]),
                    node_rows=node_rows,
                    edge_rows=edge_rows,
                )
            with timings.measure("target_slice_round_trip"):
                validate_target_slice_round_trip(
                    target_slice,
                    source_node_rows=node_rows,
                    source_edge_rows=edge_rows,
                )
            with timings.measure("target_metadata_stats"):
                target_row = _target_row(
                    record,
                    fit_weight,
                    local_labels=loaded.circuit_data.labels,
                )
                stats = _target_stats(
                    target_slice,
                    record,
                    payload_size_bytes=int(loaded.manifest["data_size_bytes"]),
                )
                for context in contexts.values():
                    sinks = context["sinks"]
                    sinks["targets.parquet"].write([target_row])
                    sinks["target-stats.parquet"].write([stats])
                for field in (
                    "node_count",
                    "edge_count",
                    "signed_basis_count",
                    "occurrence_count",
                    "attribution_profile_cell_count",
                    "attribution_profile_column_count",
                    "attribution_supported_cell_count",
                    "attribution_nonzero_count",
                    "contribution_profile_cell_count",
                    "contribution_profile_column_count",
                    "contribution_supported_cell_count",
                    "contribution_nonzero_count",
                    "source_payload_size_bytes",
                ):
                    totals[field] += int(stats[field])

            if feature_active:
                with timings.measure("feature_rows"):
                    contexts["dense_features"]["sinks"][
                        "basis-observations.parquet"
                    ].write(
                        [
                            _feature_row(summary, record, fit_weight)
                            for summary in target_slice.basis_summaries
                        ]
                    )

            if multiplex_active:
                with timings.measure("multiplex_occurrence_rows"):
                    multiplex_sinks = contexts["dense_multiplex"]["sinks"]
                    multiplex_sinks["node-occurrences.parquet"].write(
                        [
                            {
                                "trace_unit_id": trace_id,
                                "response_id": target_slice.response_id,
                                "response_position": (
                                    target_slice.target_response_position
                                ),
                                "token_position": node.occurrence.token_position,
                                **_basis_record(node.basis),
                                "attribution": node.attribution,
                                "activation": node.activation,
                                "local_label": node.local_label,
                            }
                            for node in target_slice.nodes
                        ]
                    )
                    multiplex_sinks["edge-occurrences.parquet"].write(
                        [
                            {
                                "trace_unit_id": trace_id,
                                "response_id": target_slice.response_id,
                                "response_position": (
                                    target_slice.target_response_position
                                ),
                                "source_token_position": edge.source.token_position,
                                **_basis_record(
                                    _basis_from_occurrence(edge.source, target_slice),
                                    "source_",
                                ),
                                "target_token_position": edge.target.token_position,
                                **_basis_record(
                                    _basis_from_occurrence(edge.target, target_slice),
                                    "target_",
                                ),
                                "attribution": edge.attribution,
                                "weight": edge.weight,
                                "local_label": edge.local_label,
                            }
                            for edge in target_slice.edges
                        ]
                    )
                    multiplex_sinks["trajectory-measurements.parquet"].write(
                        [
                            {
                                "trace_unit_id": trace_id,
                                "response_id": target_slice.response_id,
                                "response_position": (
                                    target_slice.target_response_position
                                ),
                                **_basis_record(summary.basis),
                                "supported": True,
                                "signed_attribution": summary.signed_attribution,
                                "absolute_attribution_mass": (
                                    summary.absolute_attribution_mass
                                ),
                                "occurrence_count": summary.occurrence_count,
                                "mean_activation": summary.mean_activation,
                                "in_degree": summary.in_degree,
                                "out_degree": summary.out_degree,
                            }
                            for summary in target_slice.basis_summaries
                        ]
                    )

                with timings.measure("multiplex_support_aggregation"):
                    current_summaries = {
                        summary.basis: summary
                        for summary in target_slice.basis_summaries
                    }
                    if prior_summaries is not None:
                        longitudinal_rows.extend(
                            {
                                "response_id": target_slice.response_id,
                                "left_response_position": prior_position,
                                "right_response_position": (
                                    target_slice.target_response_position
                                ),
                                "left_trace_unit_id": prior_trace_id,
                                "right_trace_unit_id": trace_id,
                                **_basis_record(basis),
                                "left_token_positions": [
                                    occurrence.token_position
                                    for occurrence in prior_summaries[basis].occurrences
                                ],
                                "right_token_positions": [
                                    occurrence.token_position
                                    for occurrence in current_summaries[
                                        basis
                                    ].occurrences
                                ],
                                "correspondence_kind": "same_basis_at_next_target",
                                "explicitly_noncausal": True,
                            }
                            for basis in sorted(
                                prior_summaries.keys() & current_summaries.keys()
                            )
                        )
                    prior_summaries = current_summaries
                    prior_position = target_slice.target_response_position
                    prior_trace_id = trace_id

                    per_target_edges: dict[
                        tuple[SignedBasisKey, SignedBasisKey],
                        tuple[int, float, float, float],
                    ] = {}
                    for edge in target_slice.edges:
                        source_basis = _basis_from_occurrence(
                            edge.source,
                            target_slice,
                        )
                        target_basis = _basis_from_occurrence(
                            edge.target,
                            target_slice,
                        )
                        count, attr_sum, abs_sum, weight_sum = per_target_edges.get(
                            (source_basis, target_basis),
                            (0, 0.0, 0.0, 0.0),
                        )
                        per_target_edges[(source_basis, target_basis)] = (
                            count + 1,
                            attr_sum + edge.attribution,
                            abs_sum + abs(edge.attribution),
                            weight_sum + edge.weight,
                        )
                    for pair, values in per_target_edges.items():
                        support = edge_support.get(pair)
                        if support is None:
                            support = EdgeSupport([], [])
                            edge_support[pair] = support
                        support.positions.append(target_slice.target_response_position)
                        support.trace_ids.append(trace_id)
                        support.edge_count += values[0]
                        support.attribution_sum += values[1]
                        support.abs_attribution_sum += values[2]
                        support.weight_sum += values[3]

                    for summary in target_slice.basis_summaries:
                        basis_nodes.add(summary.basis)
                        support = node_support.get(summary.basis)
                        if support is None:
                            support = NodeSupport([], [])
                            node_support[summary.basis] = support
                        support.positions.append(target_slice.target_response_position)
                        support.trace_ids.append(trace_id)
                        support.abs_attribution_sum += summary.absolute_attribution_mass
                        support.signed_attribution_sum += summary.signed_attribution
                        support.occurrence_count += summary.occurrence_count

            del target_slice, node_rows, edge_rows, loaded

        if multiplex_active:
            with timings.measure("multiplex_final_aggregation"):
                multiplex_context = contexts["dense_multiplex"]
                temporary = multiplex_context["temporary"]
                multiplex_sinks = multiplex_context["sinks"]
                multiplex_sinks.update(
                    {
                        "basis-nodes.parquet": ParquetSink(
                            temporary / "basis-nodes.parquet",
                            BASIS_NODE_SCHEMA,
                        ),
                        "longitudinal-correspondence.parquet": ParquetSink(
                            temporary / "longitudinal-correspondence.parquet",
                            LONGITUDINAL_SCHEMA,
                        ),
                        "aggregated-node-support.parquet": ParquetSink(
                            temporary / "aggregated-node-support.parquet",
                            AGGREGATED_NODE_SUPPORT_SCHEMA,
                        ),
                        "aggregated-edge-support.parquet": ParquetSink(
                            temporary / "aggregated-edge-support.parquet",
                            AGGREGATED_EDGE_SUPPORT_SCHEMA,
                        ),
                    }
                )
                multiplex_sinks["basis-nodes.parquet"].write(
                    [
                        {
                            **_basis_record(basis),
                            "response_id": task["response_id"],
                        }
                        for basis in sorted(basis_nodes)
                    ]
                )
                multiplex_sinks["longitudinal-correspondence.parquet"].write(
                    longitudinal_rows
                )
                multiplex_sinks["aggregated-node-support.parquet"].write(
                    [
                        {
                            "response_id": task["response_id"],
                            **_basis_record(basis),
                            "support_target_count": len(support.positions),
                            "support_response_positions": support.positions,
                            "support_trace_unit_ids": support.trace_ids,
                            "mean_abs_attribution_over_supported_targets": (
                                support.abs_attribution_sum / len(support.positions)
                            ),
                            "mean_signed_attribution_over_supported_targets": (
                                support.signed_attribution_sum / len(support.positions)
                            ),
                            "occurrence_count_sum": support.occurrence_count,
                        }
                        for basis, support in sorted(node_support.items())
                    ]
                )
                multiplex_sinks["aggregated-edge-support.parquet"].write(
                    [
                        {
                            "response_id": task["response_id"],
                            **_basis_record(pair[0], "source_"),
                            **_basis_record(pair[1], "target_"),
                            "support_target_count": len(support.positions),
                            "support_response_positions": support.positions,
                            "support_trace_unit_ids": support.trace_ids,
                            "edge_occurrence_count": support.edge_count,
                            "mean_signed_attribution_over_edge_occurrences": (
                                support.attribution_sum / support.edge_count
                            ),
                            "mean_abs_attribution_over_edge_occurrences": (
                                support.abs_attribution_sum / support.edge_count
                            ),
                            "mean_weight_over_edge_occurrences": (
                                support.weight_sum / support.edge_count
                            ),
                        }
                        for pair, support in sorted(edge_support.items())
                    ]
                )

        for lane, context in contexts.items():
            with timings.measure(f"{lane}_parquet_finalize"):
                for sink in context["sinks"].values():
                    sink.close()
            timings.record(
                f"{lane}_parquet_encode_write",
                sum(sink.write_seconds for sink in context["sinks"].values()),
                calls=sum(sink.flush_count for sink in context["sinks"].values()),
            )

        plan_hashes = {
            str(plan["lane"]): str(plan["plan_sha256"]) for plan in validated_plans
        }
        for lane, context in contexts.items():
            validated = context["plan"]
            sinks = context["sinks"]
            temporary = context["temporary"]
            with timings.measure(f"{lane}_output_checksum"):
                file_records = [
                    {
                        "path": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                        "row_count": sinks[path.name].row_count,
                    }
                    for path in sorted(temporary.glob("*.parquet"))
                ]
            identity = {
                "schema_version": context["shard_schema"],
                "plan_sha256": validated["plan_sha256"],
                "lane": lane,
                "task_index": task_index,
                "response_id": task["response_id"],
                "base_question_id": task["base_question_id"],
                "target_count": task["target_count"],
                "target_identity_sha256": task["target_identity_sha256"],
            }
            manifest: dict[str, Any] = {
                **identity,
                "shard_identity_sha256": canonical_sha256(identity),
                "created_at": start.isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
                "build_mode": (
                    "joint_one_pass" if len(validated_plans) > 1 else "single_lane"
                ),
                "joint_plan_sha256s": plan_hashes,
                "code_revision": validated["code_revision"],
                "runtime_environment": validated["runtime_environment"],
                "slurm": {
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                    "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
                    "node": os.environ.get("SLURMD_NODENAME"),
                },
                "wall_seconds": perf_counter() - wall_started,
                "stage_timings": timings.to_record(),
                "parquet_sinks": {
                    name: sink.performance_record()
                    for name, sink in sorted(sinks.items())
                },
                "peak_rss_bytes": (
                    int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
                ),
                "totals": dict(sorted(totals.items())),
                "files": file_records,
            }
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            os.replace(temporary, context["shard_path"])
            results[lane] = {"status": "complete", "manifest": manifest}
        return results
    except BaseException:
        for context in contexts.values():
            for sink in context["sinks"].values():
                if not sink.closed and sink.writer is not None:
                    sink.writer.close()
            shutil.rmtree(context["temporary"], ignore_errors=True)
        raise
