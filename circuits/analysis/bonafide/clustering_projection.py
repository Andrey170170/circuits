"""Project provisional cluster assignments onto the dense response multiplex."""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.analysis.bonafide.clustering_evaluation import (
    LoadedClusterState,
    _validate_source_plan,
    load_cluster_states,
)
from circuits.analysis.bonafide.compaction import (
    COMPACTED_MULTIPLEX_SCHEMA,
    _validate_existing_compaction,
)

PROJECTION_SCHEMA = "adag.bonafide.cluster-multiplex-projection.v1"

TARGET_CLUSTER_SCHEMA = pa.schema(
    [
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("base_question_id", pa.string(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("response_target_ordinal", pa.int32(), nullable=False),
        pa.field("response_phase_bin", pa.int8(), nullable=False),
        pa.field("cluster_id", pa.int32(), nullable=False),
        pa.field("supported_basis_count", pa.int32(), nullable=False),
        pa.field("signed_attribution", pa.float64(), nullable=False),
        pa.field("absolute_attribution_mass", pa.float64(), nullable=False),
        pa.field("occurrence_count", pa.int64(), nullable=False),
        pa.field("occurrence_weighted_mean_activation", pa.float64(), nullable=False),
        pa.field("in_degree", pa.int64(), nullable=False),
        pa.field("out_degree", pa.int64(), nullable=False),
    ]
)

CLUSTER_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("cluster_id", pa.int32(), nullable=False),
        pa.field("member_basis_count", pa.int32(), nullable=False),
        pa.field("support_target_count", pa.int32(), nullable=False),
        pa.field("support_response_count", pa.int16(), nullable=False),
        pa.field("support_family_count", pa.int16(), nullable=False),
        pa.field("maximum_response_target_fraction", pa.float64(), nullable=False),
        pa.field("maximum_response_attribution_fraction", pa.float64(), nullable=False),
        pa.field("median_normalized_emergence", pa.float64(), nullable=False),
        pa.field("median_persistence_density", pa.float64(), nullable=False),
        pa.field("median_adjacent_continuity", pa.float64(), nullable=False),
        pa.field("mean_five_bin_temporal_cosine", pa.float64(), nullable=True),
        pa.field("labelable", pa.bool_(), nullable=False),
        pa.field("labeling_status", pa.string(), nullable=False),
    ]
)

CLUSTER_EDGE_SCHEMA = pa.schema(
    [
        pa.field("source_cluster_id", pa.int32(), nullable=False),
        pa.field("target_cluster_id", pa.int32(), nullable=False),
        pa.field("support_target_count", pa.int32(), nullable=False),
        pa.field("support_response_count", pa.int16(), nullable=False),
        pa.field("support_family_count", pa.int16(), nullable=False),
        pa.field("edge_occurrence_count", pa.int64(), nullable=False),
        pa.field("signed_attribution_sum", pa.float64(), nullable=False),
        pa.field("absolute_attribution_mass", pa.float64(), nullable=False),
        pa.field("weight_sum", pa.float64(), nullable=False),
        pa.field("recurrent_across_responses_and_families", pa.bool_(), nullable=False),
        pa.field("witness_trace_unit_ids", pa.list_(pa.string()), nullable=False),
        pa.field("support_trace_set_sha256", pa.string(), nullable=False),
    ]
)


def _basis_tuple(record: Mapping[str, Any], prefix: str = "") -> tuple[Any, ...]:
    return (
        str(record[f"{prefix}model_id"]),
        str(record[f"{prefix}model_revision"]),
        int(record[f"{prefix}layer"]),
        int(record[f"{prefix}neuron_index"]),
        str(record[f"{prefix}polarity"]),
    )


def _validate_multiplex_root(path: Path) -> dict[str, Any]:
    raw = load_json_object(path / "manifest.json")
    plan_sha256 = raw.get("plan_sha256")
    if not isinstance(plan_sha256, str):
        raise ValueError("multiplex manifest lacks plan_sha256")
    manifest = _validate_existing_compaction(path, plan_sha256=plan_sha256)
    if manifest.get("schema_version") != COMPACTED_MULTIPLEX_SCHEMA:
        raise ValueError("projection requires a compacted response multiplex")
    if manifest.get("lane") != "dense_multiplex":
        raise ValueError("projection input lane must be dense_multiplex")
    return manifest


def _phase_bin(ordinal: int, count: int) -> int:
    if count < 1 or ordinal < 0 or ordinal >= count:
        raise ValueError("response target ordinal is invalid")
    return min(4, (ordinal * 5) // count)


def _cosine(left: NDArray[np.float64], right: NDArray[np.float64]) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return None
    return float(np.dot(left, right) / denominator)


@dataclass
class _TargetClusterAccumulator:
    signed_attribution: float = 0.0
    absolute_attribution_mass: float = 0.0
    occurrence_count: int = 0
    activation_weighted_sum: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    supported_basis_count: int = 0


@dataclass
class _ClusterEdgeAccumulator:
    trace_mask: int = 0
    response_mask: int = 0
    family_mask: int = 0
    edge_occurrence_count: int = 0
    signed_attribution_sum: float = 0.0
    absolute_attribution_mass: float = 0.0
    weight_sum: float = 0.0


def _cluster_summaries(
    *,
    labels: NDArray[np.int64],
    target_cluster_rows: Sequence[Mapping[str, Any]],
    response_target_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    cluster_count = int(labels.max()) + 1
    member_counts = np.bincount(labels[labels >= 0], minlength=cluster_count)
    rows_by_cluster: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_cluster_rows:
        rows_by_cluster[int(row["cluster_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for cluster_id in range(cluster_count):
        rows = rows_by_cluster.get(cluster_id, [])
        response_target_support: dict[str, int] = defaultdict(int)
        response_abs_mass: dict[str, float] = defaultdict(float)
        positions_by_response: dict[str, list[int]] = defaultdict(list)
        bins_by_response: dict[str, NDArray[np.float64]] = {}
        families: set[str] = set()
        for row in rows:
            response_id = str(row["response_id"])
            response_target_support[response_id] += 1
            response_abs_mass[response_id] += float(row["absolute_attribution_mass"])
            positions_by_response[response_id].append(
                int(row["response_target_ordinal"])
            )
            families.add(str(row["base_question_id"]))
            bins_by_response.setdefault(
                response_id,
                np.zeros(5, dtype=np.float64),
            )[
                int(row["response_phase_bin"])
            ] += float(row["absolute_attribution_mass"])
        target_count = len(rows)
        total_abs_mass = sum(response_abs_mass.values())
        maximum_response_target_fraction = (
            max(response_target_support.values()) / target_count
            if target_count
            else 0.0
        )
        maximum_response_attribution_fraction = (
            max(response_abs_mass.values()) / total_abs_mass
            if total_abs_mass > 0
            else 0.0
        )
        emergences: list[float] = []
        densities: list[float] = []
        continuities: list[float] = []
        for response_id, raw_positions in positions_by_response.items():
            positions = sorted(set(raw_positions))
            denominator = max(1, response_target_counts[response_id] - 1)
            emergences.append(positions[0] / denominator)
            span = positions[-1] - positions[0] + 1
            densities.append(len(positions) / span)
            adjacent = sum(
                right == left + 1
                for left, right in zip(positions[:-1], positions[1:], strict=True)
            )
            continuities.append(adjacent / max(1, span - 1))
        temporal_cosines: list[float] = []
        for left, right in itertools.combinations(
            sorted(bins_by_response),
            2,
        ):
            cosine = _cosine(
                bins_by_response[left],
                bins_by_response[right],
            )
            if cosine is not None:
                temporal_cosines.append(cosine)
        response_count = len(response_target_support)
        family_count = len(families)
        labelable = (
            int(member_counts[cluster_id]) >= 8
            and target_count >= 20
            and response_count >= 3
            and family_count >= 3
        )
        summaries.append(
            {
                "cluster_id": cluster_id,
                "member_basis_count": int(member_counts[cluster_id]),
                "support_target_count": target_count,
                "support_response_count": response_count,
                "support_family_count": family_count,
                "maximum_response_target_fraction": (maximum_response_target_fraction),
                "maximum_response_attribution_fraction": (
                    maximum_response_attribution_fraction
                ),
                "median_normalized_emergence": (
                    median(emergences) if emergences else 0.0
                ),
                "median_persistence_density": (median(densities) if densities else 0.0),
                "median_adjacent_continuity": (
                    median(continuities) if continuities else 0.0
                ),
                "mean_five_bin_temporal_cosine": (
                    mean(temporal_cosines) if temporal_cosines else None
                ),
                "labelable": labelable,
                "labeling_status": (
                    "ready" if labelable else "insufficient_labeling_support"
                ),
            }
        )
    return summaries


def _decode_trace_mask(
    mask: int,
    trace_ids: Sequence[str],
    *,
    limit: int = 10,
) -> tuple[list[str], str]:
    indices: list[int] = []
    remaining = mask
    while remaining:
        bit = remaining & -remaining
        indices.append(bit.bit_length() - 1)
        remaining ^= bit
    support_ids = [trace_ids[index] for index in indices]
    return support_ids[:limit], canonical_sha256(support_ids)


def _write_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema
) -> None:
    pq.write_table(
        pa.Table.from_pylist(list(rows), schema=schema),
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def build_multiplex_projection(
    *,
    source_plan_path: Path,
    structural_report_path: Path,
    multiplex_root: Path,
    output_root: Path,
    code_revision: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = _validate_source_plan(source_plan_path)
    structural_report = load_json_object(structural_report_path)
    report_core = dict(structural_report)
    recorded_report_hash = report_core.pop("report_sha256", None)
    if recorded_report_hash != canonical_sha256(report_core):
        raise ValueError("structural report hash mismatch")
    if (
        structural_report.get("source_plan", {}).get("plan_sha256")
        != plan["plan_sha256"]
    ):
        raise ValueError("structural report belongs to another clustering plan")
    multiplex_root = multiplex_root.resolve()
    multiplex_manifest = _validate_multiplex_root(multiplex_root)
    states = load_cluster_states(plan)
    candidate_tasks = [
        int(candidate["medoid_task_index"])
        for candidate in structural_report["candidate_resolutions"]
        if bool(candidate["passes_preliminary_gates"])
    ]
    if len(candidate_tasks) < 2:
        raise ValueError("fewer than two structural candidates passed")
    candidate_states = [states[index] for index in candidate_tasks]

    feature_basis_rows = pq.read_table(
        Path(str(plan["feature_store"]["path"])) / "basis-index.parquet"
    ).to_pylist()
    feature_basis_to_index = {
        _basis_tuple(row): int(row["signed_basis_index"]) for row in feature_basis_rows
    }
    multiplex_basis_rows = pq.read_table(
        multiplex_root / "basis-index.parquet"
    ).to_pylist()
    if {_basis_tuple(row) for row in multiplex_basis_rows} != set(
        feature_basis_to_index
    ):
        raise ValueError("feature and multiplex signed-basis identities disagree")

    target_rows = pq.read_table(multiplex_root / "target-index.parquet").to_pylist()
    target_by_trace = {str(row["trace_unit_id"]): row for row in target_rows}
    ordered_targets_by_response: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        ordered_targets_by_response[str(row["response_id"])].append(row)
    ordinal_by_trace: dict[str, int] = {}
    response_target_counts: dict[str, int] = {}
    for response_id, rows in ordered_targets_by_response.items():
        rows.sort(
            key=lambda row: (
                int(row["response_position"]),
                str(row["trace_unit_id"]),
            )
        )
        response_target_counts[response_id] = len(rows)
        for ordinal, row in enumerate(rows):
            ordinal_by_trace[str(row["trace_unit_id"])] = ordinal

    trace_ids = sorted(target_by_trace)
    trace_index = {trace_id: index for index, trace_id in enumerate(trace_ids)}
    response_ids = sorted(ordered_targets_by_response)
    response_index = {
        response_id: index for index, response_id in enumerate(response_ids)
    }
    family_ids = sorted({str(row["base_question_id"]) for row in target_rows})
    family_index = {family_id: index for index, family_id in enumerate(family_ids)}
    family_by_response = {
        response_id: str(rows[0]["base_question_id"])
        for response_id, rows in ordered_targets_by_response.items()
    }

    target_accumulators: dict[int, dict[tuple[str, int], _TargetClusterAccumulator]] = {
        state.task_index: {} for state in candidate_states
    }
    edge_accumulators: dict[int, dict[tuple[int, int], _ClusterEdgeAccumulator]] = {
        state.task_index: {} for state in candidate_states
    }
    multiplex_output_root = multiplex_root.parent
    trajectory_columns = [
        "trace_unit_id",
        "response_id",
        "response_position",
        "model_id",
        "model_revision",
        "layer",
        "neuron_index",
        "polarity",
        "signed_attribution",
        "absolute_attribution_mass",
        "occurrence_count",
        "mean_activation",
        "in_degree",
        "out_degree",
    ]
    edge_columns = [
        "response_id",
        "source_model_id",
        "source_model_revision",
        "source_layer",
        "source_neuron_index",
        "source_polarity",
        "target_model_id",
        "target_model_revision",
        "target_layer",
        "target_neuron_index",
        "target_polarity",
        "support_trace_unit_ids",
        "edge_occurrence_count",
        "mean_signed_attribution_over_edge_occurrences",
        "mean_abs_attribution_over_edge_occurrences",
        "mean_weight_over_edge_occurrences",
    ]
    for shard in multiplex_manifest["shards"]:
        shard_path = multiplex_output_root / str(shard["path"])
        trajectory = pq.read_table(
            shard_path / "trajectory-measurements.parquet",
            columns=trajectory_columns,
        )
        for row in trajectory.to_pylist():
            try:
                basis_index = feature_basis_to_index[_basis_tuple(row)]
            except KeyError as error:
                raise ValueError(
                    "trajectory basis is absent from feature index"
                ) from error
            trace_unit_id = str(row["trace_unit_id"])
            for state in candidate_states:
                cluster_id = int(state.labels[basis_index])
                if cluster_id < 0:
                    continue
                key = (trace_unit_id, cluster_id)
                accumulator = target_accumulators[state.task_index].setdefault(
                    key,
                    _TargetClusterAccumulator(),
                )
                occurrence_count = int(row["occurrence_count"])
                accumulator.signed_attribution += float(row["signed_attribution"])
                accumulator.absolute_attribution_mass += float(
                    row["absolute_attribution_mass"]
                )
                accumulator.occurrence_count += occurrence_count
                accumulator.activation_weighted_sum += (
                    float(row["mean_activation"]) * occurrence_count
                )
                accumulator.in_degree += int(row["in_degree"])
                accumulator.out_degree += int(row["out_degree"])
                accumulator.supported_basis_count += 1

        edge_file = pq.ParquetFile(shard_path / "aggregated-edge-support.parquet")
        for batch in edge_file.iter_batches(
            batch_size=20_000,
            columns=edge_columns,
        ):
            for row in batch.to_pylist():
                try:
                    source_basis_index = feature_basis_to_index[
                        _basis_tuple(row, "source_")
                    ]
                    target_basis_index = feature_basis_to_index[
                        _basis_tuple(row, "target_")
                    ]
                except KeyError as error:
                    raise ValueError(
                        "aggregated edge basis is absent from feature index"
                    ) from error
                response_id = str(row["response_id"])
                response_mask = 1 << response_index[response_id]
                family_mask = 1 << family_index[family_by_response[response_id]]
                row_trace_mask = 0
                for trace_unit_id in row["support_trace_unit_ids"]:
                    row_trace_mask |= 1 << trace_index[str(trace_unit_id)]
                occurrence_count = int(row["edge_occurrence_count"])
                for state in candidate_states:
                    source_cluster = int(state.labels[source_basis_index])
                    target_cluster = int(state.labels[target_basis_index])
                    if source_cluster < 0 or target_cluster < 0:
                        continue
                    key = (source_cluster, target_cluster)
                    accumulator = edge_accumulators[state.task_index].setdefault(
                        key, _ClusterEdgeAccumulator()
                    )
                    accumulator.trace_mask |= row_trace_mask
                    accumulator.response_mask |= response_mask
                    accumulator.family_mask |= family_mask
                    accumulator.edge_occurrence_count += occurrence_count
                    accumulator.signed_attribution_sum += (
                        float(row["mean_signed_attribution_over_edge_occurrences"])
                        * occurrence_count
                    )
                    accumulator.absolute_attribution_mass += (
                        float(row["mean_abs_attribution_over_edge_occurrences"])
                        * occurrence_count
                    )
                    accumulator.weight_sum += (
                        float(row["mean_weight_over_edge_occurrences"])
                        * occurrence_count
                    )

    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    candidate_manifests: list[dict[str, Any]] = []
    try:
        for state in candidate_states:
            candidate_path = temporary / f"task-{state.task_index:03d}"
            candidate_path.mkdir()
            target_cluster_rows: list[dict[str, Any]] = []
            for (
                trace_unit_id,
                cluster_id,
            ), accumulator in sorted(target_accumulators[state.task_index].items()):
                target = target_by_trace[trace_unit_id]
                response_id = str(target["response_id"])
                ordinal = ordinal_by_trace[trace_unit_id]
                target_cluster_rows.append(
                    {
                        "trace_unit_id": trace_unit_id,
                        "response_id": response_id,
                        "base_question_id": str(target["base_question_id"]),
                        "response_position": int(target["response_position"]),
                        "response_target_ordinal": ordinal,
                        "response_phase_bin": _phase_bin(
                            ordinal,
                            response_target_counts[response_id],
                        ),
                        "cluster_id": cluster_id,
                        "supported_basis_count": (accumulator.supported_basis_count),
                        "signed_attribution": (accumulator.signed_attribution),
                        "absolute_attribution_mass": (
                            accumulator.absolute_attribution_mass
                        ),
                        "occurrence_count": accumulator.occurrence_count,
                        "occurrence_weighted_mean_activation": (
                            accumulator.activation_weighted_sum
                            / accumulator.occurrence_count
                            if accumulator.occurrence_count
                            else 0.0
                        ),
                        "in_degree": accumulator.in_degree,
                        "out_degree": accumulator.out_degree,
                    }
                )
            summaries = _cluster_summaries(
                labels=state.labels,
                target_cluster_rows=target_cluster_rows,
                response_target_counts=response_target_counts,
            )
            edge_rows: list[dict[str, Any]] = []
            for (
                source_cluster,
                target_cluster,
            ), accumulator in sorted(edge_accumulators[state.task_index].items()):
                witness_ids, support_hash = _decode_trace_mask(
                    accumulator.trace_mask,
                    trace_ids,
                )
                recurrent = (
                    accumulator.response_mask.bit_count() >= 2
                    and accumulator.family_mask.bit_count() >= 2
                )
                edge_rows.append(
                    {
                        "source_cluster_id": source_cluster,
                        "target_cluster_id": target_cluster,
                        "support_target_count": (accumulator.trace_mask.bit_count()),
                        "support_response_count": (
                            accumulator.response_mask.bit_count()
                        ),
                        "support_family_count": (accumulator.family_mask.bit_count()),
                        "edge_occurrence_count": (accumulator.edge_occurrence_count),
                        "signed_attribution_sum": (accumulator.signed_attribution_sum),
                        "absolute_attribution_mass": (
                            accumulator.absolute_attribution_mass
                        ),
                        "weight_sum": accumulator.weight_sum,
                        "recurrent_across_responses_and_families": recurrent,
                        "witness_trace_unit_ids": witness_ids,
                        "support_trace_set_sha256": support_hash,
                    }
                )
            target_path = candidate_path / "target-cluster-trajectories.parquet"
            summary_path = candidate_path / "cluster-summaries.parquet"
            edge_path = candidate_path / "cluster-edge-support.parquet"
            _write_parquet(
                target_path,
                target_cluster_rows,
                TARGET_CLUSTER_SCHEMA,
            )
            _write_parquet(summary_path, summaries, CLUSTER_SUMMARY_SCHEMA)
            _write_parquet(edge_path, edge_rows, CLUSTER_EDGE_SCHEMA)
            member_counts = {
                int(row["cluster_id"]): int(row["member_basis_count"])
                for row in summaries
            }
            labelable = [row for row in summaries if bool(row["labelable"])]
            labelable_mass = sum(
                member_counts[int(row["cluster_id"])] for row in labelable
            )
            assigned_count = int((state.labels >= 0).sum())
            total_edge_mass = sum(
                float(row["absolute_attribution_mass"]) for row in edge_rows
            )
            recurrent_edge_mass = sum(
                float(row["absolute_attribution_mass"])
                for row in edge_rows
                if bool(row["recurrent_across_responses_and_families"])
            )
            temporal_values = [
                float(row["mean_five_bin_temporal_cosine"])
                for row in summaries
                if row["mean_five_bin_temporal_cosine"] is not None
            ]
            metrics = {
                "cluster_count": len(summaries),
                "assigned_basis_count": assigned_count,
                "labelable_cluster_count": len(labelable),
                "labelable_cluster_fraction": (len(labelable) / len(summaries)),
                "labelable_assigned_mass_fraction": (labelable_mass / assigned_count),
                "median_maximum_response_target_fraction": median(
                    float(row["maximum_response_target_fraction"]) for row in summaries
                ),
                "p90_maximum_response_target_fraction": float(
                    np.quantile(
                        [
                            float(row["maximum_response_target_fraction"])
                            for row in summaries
                        ],
                        0.9,
                    )
                ),
                "median_temporal_profile_cosine": (
                    median(temporal_values) if temporal_values else None
                ),
                "cluster_edge_pair_count": len(edge_rows),
                "recurrent_cluster_edge_pair_fraction": (
                    sum(
                        bool(row["recurrent_across_responses_and_families"])
                        for row in edge_rows
                    )
                    / len(edge_rows)
                    if edge_rows
                    else 0.0
                ),
                "recurrent_cluster_edge_mass_fraction": (
                    recurrent_edge_mass / total_edge_mass
                    if total_edge_mass > 0
                    else 0.0
                ),
            }
            file_records = [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "row_count": pq.read_metadata(path).num_rows,
                }
                for path in (summary_path, edge_path, target_path)
            ]
            candidate_manifest: dict[str, Any] = {
                "task_index": state.task_index,
                "n_clusters": int(state.config["n_clusters"]),
                "source_state_manifest_sha256": state.manifest["manifest_sha256"],
                "metrics": metrics,
                "files": file_records,
                "descriptions_generated": False,
            }
            candidate_manifest["candidate_projection_sha256"] = canonical_sha256(
                candidate_manifest
            )
            with (candidate_path / "manifest.json").open(
                "x",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    candidate_manifest,
                    handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
            candidate_manifests.append(candidate_manifest)
        manifest: dict[str, Any] = {
            "schema_version": PROJECTION_SCHEMA,
            "source_plan": {
                "path": str(source_plan_path.resolve()),
                "plan_sha256": plan["plan_sha256"],
            },
            "source_structural_report": {
                "path": str(structural_report_path.resolve()),
                "report_sha256": structural_report["report_sha256"],
            },
            "source_multiplex": {
                "path": str(multiplex_root),
                "manifest_sha256": multiplex_manifest["manifest_sha256"],
            },
            "candidate_count": len(candidate_manifests),
            "candidates": candidate_manifests,
            "analysis_code_revision": (
                dict(code_revision) if code_revision is not None else None
            ),
            "analysis_environment": (
                dict(environment) if environment is not None else None
            ),
            "descriptions_generated": False,
            "scientific_cluster_state_selected": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if output_root.exists():
            raise FileExistsError(f"projection output exists: {output_root}")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
