"""Freeze primary and alternative cluster states for exploratory labeling."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.analysis.bonafide.clustering_evaluation import (
    LoadedClusterState,
    _affinity_for_config,
    _validate_source_plan,
    load_cluster_states,
)
from circuits.analysis.bonafide.clustering_store import load_pair_evidence

SELECTION_SCHEMA = "adag.bonafide.selected-cluster-states.v1"
SELECTED_STATE_SCHEMA = "adag.bonafide.selected-cluster-state.v1"

PROTOTYPE_SCHEMA = pa.schema(
    [
        pa.field("cluster_id", pa.int32(), nullable=False),
        pa.field("member_basis_count", pa.int32(), nullable=False),
        pa.field("labelable", pa.bool_(), nullable=False),
        pa.field(
            "prototype_signed_basis_indices",
            pa.list_(pa.int64()),
            nullable=False,
        ),
        pa.field(
            "prototype_internal_affinity_strengths",
            pa.list_(pa.float64()),
            nullable=False,
        ),
    ]
)


def _validated_hashed_object(path: Path, hash_field: str) -> dict[str, Any]:
    value = load_json_object(path)
    core = dict(value)
    recorded_hash = core.pop(hash_field, None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError(f"{path.name} hash mismatch")
    return value


def _percentile_ranks(
    values: Mapping[int, float],
    *,
    higher_is_better: bool,
) -> dict[int, float]:
    if not values:
        raise ValueError("percentile ranks require values")
    if len(values) == 1:
        return {next(iter(values)): 1.0}
    result: dict[int, float] = {}
    for task_index, value in values.items():
        if higher_is_better:
            lower = sum(other < value for other in values.values())
            equal = sum(other == value for other in values.values())
        else:
            lower = sum(other > value for other in values.values())
            equal = sum(other == value for other in values.values())
        result[task_index] = (lower + (equal - 1) / 2.0) / (len(values) - 1)
    return result


def _candidate_scores(
    *,
    structural: Mapping[str, Any],
    projection: Mapping[str, Any],
    resampling: Mapping[str, Any],
    recurrence_by_task: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    structural_by_task = {
        int(candidate["medoid_task_index"]): candidate
        for candidate in structural["candidate_resolutions"]
    }
    projection_by_task = {
        int(candidate["task_index"]): candidate
        for candidate in projection["candidates"]
    }
    resampling_by_task = {
        int(candidate["source_task_index"]): candidate
        for candidate in resampling["candidates"]
    }
    common = sorted(
        set(structural_by_task) & set(projection_by_task) & set(resampling_by_task)
    )
    if len(common) < 2:
        raise ValueError("selection requires at least two projected candidates")

    raw_metrics: dict[str, tuple[bool, dict[int, float]]] = {
        "seed_mean_ari": (
            True,
            {
                task: float(structural_by_task[task]["seed_stability"]["mean_ari"])
                for task in common
            },
        ),
        "family_jackknife_median_ari": (
            True,
            {
                task: float(resampling_by_task[task]["family_jackknife_median_ari"])
                for task in common
            },
        ),
        "family_jackknife_p10_ari": (
            True,
            {
                task: float(resampling_by_task[task]["family_jackknife_p10_ari"])
                for task in common
            },
        ),
        "checkpoint_median_ari": (
            True,
            {
                task: float(resampling_by_task[task]["checkpoint_median_ari"])
                for task in common
            },
        ),
        "modularity": (
            True,
            {
                task: float(structural_by_task[task]["graph"]["modularity"])
                for task in common
            },
        ),
        "affinity_enrichment": (
            True,
            {
                task: float(
                    structural_by_task[task]["graph"]["internal_affinity_enrichment"]
                )
                for task in common
            },
        ),
        "weighted_conductance": (
            False,
            {
                task: float(
                    structural_by_task[task]["graph"]["size_weighted_mean_conductance"]
                )
                for task in common
            },
        ),
        "size_entropy": (
            True,
            {
                task: float(structural_by_task[task]["size"]["normalized_size_entropy"])
                for task in common
            },
        ),
        "size_gini": (
            False,
            {
                task: float(structural_by_task[task]["size"]["size_gini"])
                for task in common
            },
        ),
        "labelable_mass": (
            True,
            {
                task: float(
                    projection_by_task[task]["metrics"][
                        "labelable_assigned_mass_fraction"
                    ]
                )
                for task in common
            },
        ),
        "labelable_clusters": (
            True,
            {
                task: float(
                    projection_by_task[task]["metrics"]["labelable_cluster_fraction"]
                )
                for task in common
            },
        ),
        "median_response_concentration": (
            False,
            {
                task: float(
                    projection_by_task[task]["metrics"][
                        "median_maximum_response_target_fraction"
                    ]
                )
                for task in common
            },
        ),
        "p90_response_concentration": (
            False,
            {
                task: float(
                    projection_by_task[task]["metrics"][
                        "p90_maximum_response_target_fraction"
                    ]
                )
                for task in common
            },
        ),
        "median_target_recurrence": (
            True,
            {
                task: float(recurrence_by_task[task]["median_support_target_count"])
                for task in common
            },
        ),
        "median_response_recurrence": (
            True,
            {
                task: float(recurrence_by_task[task]["median_support_response_count"])
                for task in common
            },
        ),
        "median_family_recurrence": (
            True,
            {
                task: float(recurrence_by_task[task]["median_support_family_count"])
                for task in common
            },
        ),
        "temporal_cosine": (
            True,
            {
                task: float(
                    projection_by_task[task]["metrics"][
                        "median_temporal_profile_cosine"
                    ]
                )
                for task in common
            },
        ),
        "recurrent_edge_pairs": (
            True,
            {
                task: float(
                    projection_by_task[task]["metrics"][
                        "recurrent_cluster_edge_pair_fraction"
                    ]
                )
                for task in common
            },
        ),
        "recurrent_edge_mass": (
            True,
            {
                task: float(
                    projection_by_task[task]["metrics"][
                        "recurrent_cluster_edge_mass_fraction"
                    ]
                )
                for task in common
            },
        ),
    }
    ranks = {
        name: _percentile_ranks(values, higher_is_better=direction)
        for name, (direction, values) in raw_metrics.items()
    }
    categories = {
        "stability": (
            0.25,
            (
                "seed_mean_ari",
                "family_jackknife_median_ari",
                "family_jackknife_p10_ari",
                "checkpoint_median_ari",
            ),
        ),
        "graph": (
            0.25,
            (
                "modularity",
                "affinity_enrichment",
                "weighted_conductance",
            ),
        ),
        "balance": (
            0.20,
            (
                "size_entropy",
                "size_gini",
                "labelable_mass",
                "labelable_clusters",
            ),
        ),
        "recurrence": (
            0.15,
            (
                "median_response_concentration",
                "p90_response_concentration",
                "median_target_recurrence",
                "median_response_recurrence",
                "median_family_recurrence",
            ),
        ),
        "temporal_and_edges": (
            0.15,
            (
                "temporal_cosine",
                "recurrent_edge_pairs",
                "recurrent_edge_mass",
            ),
        ),
    }
    records: list[dict[str, Any]] = []
    for task in common:
        hard_gates = {
            "preliminary": bool(structural_by_task[task]["passes_preliminary_gates"]),
            "labelable_assigned_mass": (
                float(
                    projection_by_task[task]["metrics"][
                        "labelable_assigned_mass_fraction"
                    ]
                )
                >= 0.90
            ),
            "labelable_cluster_fraction": (
                float(projection_by_task[task]["metrics"]["labelable_cluster_fraction"])
                >= 0.80
            ),
            "family_jackknife": bool(
                resampling_by_task[task]["passes_family_jackknife_gate"]
            ),
        }
        category_scores = {
            category: mean(ranks[name][task] for name in names)
            for category, (_, names) in categories.items()
        }
        composite = sum(
            weight * category_scores[category]
            for category, (weight, _) in categories.items()
        )
        records.append(
            {
                "source_task_index": task,
                "n_clusters": int(structural_by_task[task]["n_clusters"]),
                "hard_gates": hard_gates,
                "passes_all_hard_gates": all(hard_gates.values()),
                "raw_metrics": {
                    name: values[task] for name, (_, values) in raw_metrics.items()
                },
                "metric_percentile_ranks": {
                    name: values[task] for name, values in ranks.items()
                },
                "category_scores": category_scores,
                "composite_score": composite,
            }
        )
    return records


def _family_partitions(
    family_ids: Sequence[str],
    *,
    state_identity: str,
) -> dict[str, str]:
    ordered = sorted(
        family_ids,
        key=lambda family_id: (
            canonical_sha256(
                {
                    "state_identity": state_identity,
                    "family_id": family_id,
                }
            ),
            family_id,
        ),
    )
    roles = ("generation", "selection_scoring", "audit")
    return {
        family_id: roles[index % len(roles)] for index, family_id in enumerate(ordered)
    }


def _balanced_exemplars(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_partitions: Mapping[str, str],
    inventory_by_trace: Mapping[str, Mapping[str, Any]],
    limit_per_partition: int = 2,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for partition in ("generation", "selection_scoring", "audit"):
        candidates = [
            row
            for row in rows
            if family_partitions[str(row["base_question_id"])] == partition
        ]
        seen_families: set[str] = set()
        seen_responses: set[str] = set()
        seen_phases: set[int] = set()
        seen_conditions: set[str] = set()
        seen_tokens: set[str] = set()
        while len(seen_families) < limit_per_partition:
            eligible: list[tuple[tuple[Any, ...], Mapping[str, Any]]] = []
            for row in candidates:
                trace_unit_id = str(row["trace_unit_id"])
                inventory = inventory_by_trace[trace_unit_id]
                family_id = str(row["base_question_id"])
                response_id = str(row["response_id"])
                if family_id in seen_families or response_id in seen_responses:
                    continue
                phase = int(row["response_phase_bin"])
                condition = canonical_sha256(inventory["condition"])
                token = str(inventory["target_token_text"])
                novelty = (
                    int(phase not in seen_phases)
                    + int(condition not in seen_conditions)
                    + int(token not in seen_tokens)
                )
                eligible.append(
                    (
                        (
                            -novelty,
                            -float(row["absolute_attribution_mass"]),
                            trace_unit_id,
                        ),
                        row,
                    )
                )
            if not eligible:
                break
            _, chosen = min(eligible, key=lambda item: item[0])
            selected.append(chosen)
            inventory = inventory_by_trace[str(chosen["trace_unit_id"])]
            seen_families.add(str(chosen["base_question_id"]))
            seen_responses.add(str(chosen["response_id"]))
            seen_phases.add(int(chosen["response_phase_bin"]))
            seen_conditions.add(canonical_sha256(inventory["condition"]))
            seen_tokens.add(str(inventory["target_token_text"]))
    return selected


def _load_exemplar_context(
    trace_unit_id: str,
    *,
    inventory_by_trace: Mapping[str, Mapping[str, Any]],
    family_partitions: Mapping[str, str],
) -> dict[str, Any]:
    inventory = inventory_by_trace[trace_unit_id]
    manifest_path = Path(str(inventory["artifact_path"])) / "manifest.json"
    if file_sha256(manifest_path) != inventory["artifact_manifest_sha256"]:
        raise ValueError("exemplar trace manifest hash drift")
    manifest = load_json_object(manifest_path)
    example = manifest.get("bonafide_example")
    if not isinstance(example, Mapping):
        raise ValueError("exemplar trace lacks BonaFide context")
    family_id = str(inventory["base_question_id"])
    return {
        "trace_unit_id": trace_unit_id,
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": inventory["artifact_manifest_sha256"],
        "artifact_payload_sha256": inventory["artifact_payload_sha256"],
        "response_id": inventory["response_id"],
        "base_question_id": family_id,
        "family_partition": family_partitions[family_id],
        "response_position": int(inventory["response_position"]),
        "target_token_text": inventory["target_token_text"],
        "condition": inventory["condition"],
        "prompt": example.get("prompt"),
        "question": example.get("question"),
        "response": example.get("response"),
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_projection_candidate(
    candidate_path: Path,
    *,
    expected: Mapping[str, Any],
    state: LoadedClusterState,
) -> dict[str, Any]:
    manifest = _validated_hashed_object(
        candidate_path / "manifest.json",
        "candidate_projection_sha256",
    )
    if manifest != expected:
        raise ValueError("projection candidate differs from root manifest")
    if int(manifest["task_index"]) != state.task_index:
        raise ValueError("projection candidate task drift")
    if manifest["source_state_manifest_sha256"] != state.manifest["manifest_sha256"]:
        raise ValueError("projection candidate state drift")
    for record in manifest["files"]:
        path = candidate_path / str(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError("projection candidate file hash drift")
    return manifest


def _projection_recurrence_metrics(
    projection_root: Path,
    projection: Mapping[str, Any],
) -> dict[int, dict[str, float]]:
    metrics: dict[int, dict[str, float]] = {}
    for expected in projection["candidates"]:
        task_index = int(expected["task_index"])
        candidate_path = projection_root / f"task-{task_index:03d}"
        manifest = _validated_hashed_object(
            candidate_path / "manifest.json",
            "candidate_projection_sha256",
        )
        if manifest != expected:
            raise ValueError("projection candidate differs from root manifest")
        summary_record = next(
            record
            for record in manifest["files"]
            if record["path"] == "cluster-summaries.parquet"
        )
        summary_path = candidate_path / "cluster-summaries.parquet"
        if file_sha256(summary_path) != summary_record["sha256"]:
            raise ValueError("projection summary file hash drift")
        rows = pq.read_table(
            summary_path,
            columns=[
                "support_target_count",
                "support_response_count",
                "support_family_count",
            ],
        ).to_pylist()
        metrics[task_index] = {
            "median_support_target_count": float(
                median(int(row["support_target_count"]) for row in rows)
            ),
            "median_support_response_count": float(
                median(int(row["support_response_count"]) for row in rows)
            ),
            "median_support_family_count": float(
                median(int(row["support_family_count"]) for row in rows)
            ),
        }
    return metrics


def build_selected_states(
    *,
    source_plan_path: Path,
    structural_report_path: Path,
    projection_root: Path,
    resample_plan_path: Path,
    resample_report_path: Path,
    output_root: Path,
    code_revision: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    source_plan = _validate_source_plan(source_plan_path)
    structural = _validated_hashed_object(
        structural_report_path,
        "report_sha256",
    )
    projection = _validated_hashed_object(
        projection_root / "manifest.json",
        "manifest_sha256",
    )
    resampling = _validated_hashed_object(
        resample_report_path,
        "report_sha256",
    )
    resample_plan = _validated_hashed_object(
        resample_plan_path,
        "plan_sha256",
    )
    if structural["source_plan"]["plan_sha256"] != source_plan["plan_sha256"]:
        raise ValueError("selection structural-report source drift")
    if projection["source_plan"]["plan_sha256"] != source_plan["plan_sha256"]:
        raise ValueError("selection projection source drift")
    if resampling["plan_sha256"] != resample_plan["plan_sha256"]:
        raise ValueError("selection resampling report-plan drift")
    if resample_plan["source_plan"]["plan_sha256"] != source_plan["plan_sha256"]:
        raise ValueError("selection resampling source-plan drift")
    if (
        resample_plan["source_structural_report"]["report_sha256"]
        != structural["report_sha256"]
    ):
        raise ValueError("selection resampling structural-report drift")
    if (
        resample_plan["source_projection"]["manifest_sha256"]
        != projection["manifest_sha256"]
    ):
        raise ValueError("selection resampling projection drift")
    scores = _candidate_scores(
        structural=structural,
        projection=projection,
        resampling=resampling,
        recurrence_by_task=_projection_recurrence_metrics(
            projection_root,
            projection,
        ),
    )
    eligible = [record for record in scores if record["passes_all_hard_gates"]]
    if len({int(record["n_clusters"]) for record in eligible}) < 2:
        raise ValueError("fewer than two cluster counts pass labeling-readiness gates")
    ordered = sorted(
        eligible,
        key=lambda record: (
            -round(float(record["composite_score"]), 12),
            -float(record["raw_metrics"]["family_jackknife_median_ari"]),
            float(
                next(
                    candidate["size"]["singleton_cluster_fraction"]
                    for candidate in structural["candidate_resolutions"]
                    if int(candidate["medoid_task_index"])
                    == int(record["source_task_index"])
                )
            ),
            float(
                next(
                    candidate["size"]["tiny_cluster_fraction_lt5"]
                    for candidate in structural["candidate_resolutions"]
                    if int(candidate["medoid_task_index"])
                    == int(record["source_task_index"])
                )
            ),
            int(record["n_clusters"]),
            int(record["source_task_index"]),
        ),
    )
    primary = ordered[0]
    alternative = next(
        record
        for record in ordered[1:]
        if record["n_clusters"] != primary["n_clusters"]
    )
    selected = (("primary", primary), ("alternative", alternative))

    states = load_cluster_states(source_plan)
    evidence, support = load_pair_evidence(
        Path(str(source_plan["pair_evidence"][0]["output_path"]))
    )
    feature_root = Path(str(source_plan["feature_store"]["path"]))
    basis_rows = pq.read_table(feature_root / "basis-index.parquet").to_pylist()
    basis_rows.sort(key=lambda row: int(row["signed_basis_index"]))
    if [int(row["signed_basis_index"]) for row in basis_rows] != list(
        range(len(basis_rows))
    ):
        raise ValueError("basis index is not contiguous")
    feature_manifest = load_json_object(feature_root / "manifest.json")
    feature_file_by_name = {
        str(record["path"]): record for record in feature_manifest["files"]
    }
    target_index_path = feature_root / "target-index.parquet"
    if (
        file_sha256(target_index_path)
        != feature_file_by_name["target-index.parquet"]["sha256"]
    ):
        raise ValueError("feature target index hash drift")
    dense_target_rows = pq.read_table(
        target_index_path,
        columns=["trace_unit_id", "base_question_id"],
    ).to_pylist()
    dense_trace_ids = {str(record["trace_unit_id"]) for record in dense_target_rows}
    inventory_path = Path(str(feature_manifest["source_inventory"]["path"]))
    if (
        file_sha256(inventory_path)
        != feature_manifest["source_inventory"]["file_sha256"]
    ):
        raise ValueError("source inventory file hash drift")
    inventory = _validated_hashed_object(
        inventory_path,
        "inventory_sha256",
    )
    inventory_by_trace = {
        str(record["trace_unit_id"]): record
        for record in inventory["records"]
        if record["status"] == "discovery"
        and str(record["trace_unit_id"]) in dense_trace_ids
    }
    if set(inventory_by_trace) != dense_trace_ids:
        raise ValueError("dense target index and source inventory differ")
    family_ids = sorted(
        {str(record["base_question_id"]) for record in inventory_by_trace.values()}
    )

    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    selected_manifests: list[dict[str, Any]] = []
    partition_identity = canonical_sha256(
        {
            "source_plan_sha256": source_plan["plan_sha256"],
            "structural_report_sha256": structural["report_sha256"],
            "projection_manifest_sha256": projection["manifest_sha256"],
            "resample_report_sha256": resampling["report_sha256"],
        }
    )
    try:
        for role, score_record in selected:
            source_task_index = int(score_record["source_task_index"])
            state = states[source_task_index]
            state_path = temporary / role
            state_path.mkdir()
            family_partitions = _family_partitions(
                family_ids,
                state_identity=partition_identity,
            )
            affinity = _affinity_for_config(
                state.config,
                evidence=evidence,
                support=support,
            )
            projection_path = projection_root / f"task-{source_task_index:03d}"
            projected_by_task = {
                int(candidate["task_index"]): candidate
                for candidate in projection["candidates"]
            }
            projection_manifest = _validate_projection_candidate(
                projection_path,
                expected=projected_by_task[source_task_index],
                state=state,
            )
            summaries = pq.read_table(
                projection_path / "cluster-summaries.parquet"
            ).to_pylist()
            summary_by_cluster = {int(row["cluster_id"]): row for row in summaries}
            trajectory_rows = pq.read_table(
                projection_path / "target-cluster-trajectories.parquet"
            ).to_pylist()
            trajectories_by_cluster: dict[int, list[Mapping[str, Any]]] = defaultdict(
                list
            )
            for row in trajectory_rows:
                trajectories_by_cluster[int(row["cluster_id"])].append(row)
            edge_rows = pq.read_table(
                projection_path / "cluster-edge-support.parquet"
            ).to_pylist()
            edges_by_cluster: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for row in edge_rows:
                edges_by_cluster[int(row["source_cluster_id"])].append(row)
                if row["target_cluster_id"] != row["source_cluster_id"]:
                    edges_by_cluster[int(row["target_cluster_id"])].append(row)

            prototype_rows: list[dict[str, Any]] = []
            evidence_records: list[dict[str, Any]] = []
            cluster_count = int(state.labels.max()) + 1
            for cluster_id in range(cluster_count):
                members = np.flatnonzero(state.labels == cluster_id)
                internal_strength = np.asarray(
                    affinity[members][:, members].sum(axis=1)
                ).ravel()
                order = np.lexsort((members, -internal_strength))
                prototype_local = order[: min(5, len(order))]
                prototype_indices = members[prototype_local]
                prototype_strengths = internal_strength[prototype_local]
                summary = summary_by_cluster[cluster_id]
                prototype_rows.append(
                    {
                        "cluster_id": cluster_id,
                        "member_basis_count": len(members),
                        "labelable": bool(summary["labelable"]),
                        "prototype_signed_basis_indices": (
                            prototype_indices.astype(int).tolist()
                        ),
                        "prototype_internal_affinity_strengths": (
                            prototype_strengths.astype(float).tolist()
                        ),
                    }
                )
                exemplar_rows = _balanced_exemplars(
                    trajectories_by_cluster[cluster_id],
                    family_partitions=family_partitions,
                    inventory_by_trace=inventory_by_trace,
                )
                exemplars: list[dict[str, Any]] = []
                for row in exemplar_rows:
                    context = _load_exemplar_context(
                        str(row["trace_unit_id"]),
                        inventory_by_trace=inventory_by_trace,
                        family_partitions=family_partitions,
                    )
                    exemplars.append(
                        {
                            **context,
                            "cluster_projection": {
                                key: row[key]
                                for key in (
                                    "absolute_attribution_mass",
                                    "signed_attribution",
                                    "supported_basis_count",
                                    "occurrence_count",
                                    "in_degree",
                                    "out_degree",
                                    "response_phase_bin",
                                )
                            },
                        }
                    )
                exemplar_partition_counts = {
                    partition: sum(
                        exemplar["family_partition"] == partition
                        for exemplar in exemplars
                    )
                    for partition in (
                        "generation",
                        "selection_scoring",
                        "audit",
                    )
                }
                partition_supported = all(
                    count > 0 for count in exemplar_partition_counts.values()
                )
                top_edges = sorted(
                    edges_by_cluster[cluster_id],
                    key=lambda row: (
                        not bool(row["recurrent_across_responses_and_families"]),
                        -float(row["absolute_attribution_mass"]),
                        int(row["source_cluster_id"]),
                        int(row["target_cluster_id"]),
                    ),
                )[:5]
                evidence_records.append(
                    {
                        "schema_version": (
                            "adag.bonafide.cluster-labeling-evidence.v1"
                        ),
                        "state_role": role,
                        "cluster_id": cluster_id,
                        "labeling_status": (
                            "insufficient_partition_support"
                            if bool(summary["labelable"]) and not partition_supported
                            else summary["labeling_status"]
                        ),
                        "partition_supported": partition_supported,
                        "exemplar_partition_counts": (exemplar_partition_counts),
                        "member_basis_count": len(members),
                        "prototype_signed_bases": [
                            {
                                **basis_rows[int(index)],
                                "internal_affinity_strength": float(strength),
                            }
                            for index, strength in zip(
                                prototype_indices,
                                prototype_strengths,
                                strict=True,
                            )
                        ],
                        "multiplex_summary": summary,
                        "balanced_target_exemplars": exemplars,
                        "top_recurrent_cluster_edges": top_edges,
                        "descriptions_generated": False,
                    }
                )
            assignment_source = state.path / "assignments.parquet"
            assignment_path = state_path / "assignments.parquet"
            shutil.copyfile(assignment_source, assignment_path)
            prototype_path = state_path / "cluster-prototypes.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    prototype_rows,
                    schema=PROTOTYPE_SCHEMA,
                ),
                prototype_path,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
            evidence_path = state_path / "labeling-evidence.jsonl"
            _write_jsonl(evidence_path, evidence_records)
            partition_path = state_path / "family-partitions.json"
            partition_payload = {
                "schema_version": ("adag.bonafide.cluster-label-family-partitions.v1"),
                "state_role": role,
                "partitions": family_partitions,
            }
            partition_payload["partitions_sha256"] = canonical_sha256(partition_payload)
            with partition_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    partition_payload,
                    handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
            files = [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in (
                    assignment_path,
                    prototype_path,
                    evidence_path,
                    partition_path,
                )
            ]
            selected_manifest: dict[str, Any] = {
                "schema_version": SELECTED_STATE_SCHEMA,
                "state_role": role,
                "selection_status": ("frozen_label_free_for_exploratory_labeling"),
                "source_task_index": source_task_index,
                "n_clusters": int(score_record["n_clusters"]),
                "source_state": {
                    "path": str(state.path),
                    "manifest_sha256": state.manifest["manifest_sha256"],
                },
                "source_projection": {
                    "path": str(projection_path),
                    "candidate_projection_sha256": projection_manifest[
                        "candidate_projection_sha256"
                    ],
                },
                "score": score_record,
                "cluster_count": cluster_count,
                "labelable_cluster_count": sum(
                    bool(row["labelable"]) for row in summaries
                ),
                "partition_supported_cluster_count": sum(
                    bool(record["partition_supported"]) for record in evidence_records
                ),
                "prototype_definition": (
                    "top within-cluster sparse-affinity strength; "
                    "signed-basis identity preserved"
                ),
                "family_partition_definition": (
                    "selection-source-hashed family blocks shared by both "
                    "states; no family crosses generation, selection_scoring, "
                    "and audit"
                ),
                "family_partition_identity_sha256": partition_identity,
                "files": files,
                "code_revision": dict(code_revision),
                "environment": dict(environment),
                "descriptions_generated": False,
                "holdout_opened": False,
                "scientific_cluster_state": True,
            }
            selected_manifest["manifest_sha256"] = canonical_sha256(selected_manifest)
            with (state_path / "manifest.json").open(
                "x",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    selected_manifest,
                    handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
            selected_manifests.append(selected_manifest)
        manifest: dict[str, Any] = {
            "schema_version": SELECTION_SCHEMA,
            "selection_rule": (
                "frozen label-free percentile-rank composite from plan Section 8.5.2"
            ),
            "source_plan": {
                "path": str(source_plan_path.resolve()),
                "plan_sha256": source_plan["plan_sha256"],
            },
            "source_structural_report": {
                "path": str(structural_report_path.resolve()),
                "report_sha256": structural["report_sha256"],
            },
            "source_projection": {
                "path": str(projection_root.resolve()),
                "manifest_sha256": projection["manifest_sha256"],
            },
            "source_resampling_report": {
                "path": str(resample_report_path.resolve()),
                "report_sha256": resampling["report_sha256"],
            },
            "source_resampling_plan": {
                "path": str(resample_plan_path.resolve()),
                "plan_sha256": resample_plan["plan_sha256"],
            },
            "candidate_scores": scores,
            "selected_states": selected_manifests,
            "primary_source_task_index": int(primary["source_task_index"]),
            "alternative_source_task_index": int(alternative["source_task_index"]),
            "code_revision": dict(code_revision),
            "environment": dict(environment),
            "descriptions_generated": False,
            "holdout_opened": False,
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
            raise FileExistsError(f"selected-state output exists: {output_root}")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
