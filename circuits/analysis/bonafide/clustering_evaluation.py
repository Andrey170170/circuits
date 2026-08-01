"""Label-free structural evaluation of provisional BonaFide cluster states."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    load_json_object,
    write_hashed_json,
)
from circuits.analysis.bonafide.cluster_execution import (
    CLUSTER_PLAN_SCHEMA,
    _validate_existing_cluster_state,
)
from circuits.analysis.bonafide.clustering import (
    knn_affinity,
    mean_similarity_matrix,
)
from circuits.analysis.bonafide.clustering_store import (
    BasisSupport,
    PairEvidence,
    load_pair_evidence,
)

STRUCTURAL_REPORT_SCHEMA = "adag.bonafide.clustering-structural-report.v1"


@dataclass(frozen=True)
class LoadedClusterState:
    task_index: int
    path: Path
    manifest: Mapping[str, Any]
    labels: NDArray[np.int64]

    @property
    def config(self) -> Mapping[str, Any]:
        return self.manifest["config"]


def _validate_source_plan(path: Path) -> dict[str, Any]:
    plan = load_json_object(path)
    core = dict(plan)
    recorded_hash = core.pop("plan_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("source clustering plan hash mismatch")
    if plan.get("schema_version") != CLUSTER_PLAN_SCHEMA:
        raise ValueError("unsupported source clustering plan")
    configurations = plan.get("configurations")
    if not isinstance(configurations, list):
        raise ValueError("source clustering configurations are invalid")
    for task_index, config in enumerate(configurations):
        if not isinstance(config, Mapping) or config.get("task_index") != task_index:
            raise ValueError("source clustering task index is invalid")
        unhashed = dict(config)
        config_hash = unhashed.pop("config_sha256", None)
        unhashed.pop("task_index", None)
        if config_hash != canonical_sha256(unhashed):
            raise ValueError("source clustering configuration hash mismatch")
    return plan


def _load_assignment_labels(path: Path, *, basis_count: int) -> NDArray[np.int64]:
    rows = pq.read_table(
        path,
        columns=["signed_basis_index", "cluster_id"],
    ).to_pylist()
    rows.sort(key=lambda row: int(row["signed_basis_index"]))
    if [int(row["signed_basis_index"]) for row in rows] != list(range(basis_count)):
        raise ValueError("cluster assignment basis index is invalid")
    return np.asarray(
        [-1 if row["cluster_id"] is None else int(row["cluster_id"]) for row in rows],
        dtype=np.int64,
    )


def load_cluster_states(
    plan: Mapping[str, Any],
) -> dict[int, LoadedClusterState]:
    output_root = Path(str(plan["output_root"]))
    configurations = plan["configurations"]
    basis_count = int(
        load_json_object(
            Path(str(plan["pair_evidence"][0]["output_path"])) / "manifest.json"
        )["basis_count"]
    )
    states: dict[int, LoadedClusterState] = {}
    for path in sorted((output_root / "cluster-states").iterdir()):
        if not path.is_dir():
            continue
        manifest = _validate_existing_cluster_state(path)
        task_index = int(manifest["task_index"])
        if task_index in states:
            raise ValueError("duplicate provisional cluster task state")
        if manifest.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("cluster state belongs to another source plan")
        if manifest.get("config") != configurations[task_index]:
            raise ValueError("cluster state configuration drift")
        states[task_index] = LoadedClusterState(
            task_index=task_index,
            path=path,
            manifest=manifest,
            labels=_load_assignment_labels(
                path / "assignments.parquet",
                basis_count=basis_count,
            ),
        )
    if set(states) != set(range(len(configurations))):
        raise ValueError("provisional clustering sweep is incomplete")
    return states


def assignment_ari(
    left: NDArray[np.int64],
    right: NDArray[np.int64],
) -> float:
    if left.shape != right.shape:
        raise ValueError("assignment shapes disagree")
    shared = (left >= 0) & (right >= 0)
    if int(shared.sum()) < 2:
        raise ValueError("assignments have insufficient shared support")
    return float(adjusted_rand_score(left[shared], right[shared]))


def seed_stability(
    states: Sequence[LoadedClusterState],
) -> dict[str, Any]:
    if len(states) < 2:
        raise ValueError("seed stability requires at least two states")
    pair_records: list[dict[str, Any]] = []
    score_by_task: dict[int, list[float]] = defaultdict(list)
    for left, right in itertools.combinations(
        sorted(states, key=lambda state: state.task_index),
        2,
    ):
        score = assignment_ari(left.labels, right.labels)
        pair_records.append(
            {
                "left_task_index": left.task_index,
                "right_task_index": right.task_index,
                "ari": score,
            }
        )
        score_by_task[left.task_index].append(score)
        score_by_task[right.task_index].append(score)
    medoid_task_index = min(
        score_by_task,
        key=lambda task_index: (
            -mean(score_by_task[task_index]),
            task_index,
        ),
    )
    values = [record["ari"] for record in pair_records]
    return {
        "pairwise": pair_records,
        "minimum_ari": min(values),
        "mean_ari": mean(values),
        "maximum_ari": max(values),
        "medoid_task_index": medoid_task_index,
        "mean_ari_to_other_seeds_by_task": {
            str(task_index): mean(scores)
            for task_index, scores in sorted(score_by_task.items())
        },
    }


def cluster_size_metrics(labels: NDArray[np.int64]) -> dict[str, Any]:
    assigned = labels[labels >= 0]
    if not len(assigned):
        raise ValueError("cluster size metrics require assigned bases")
    unique, counts = np.unique(assigned, return_counts=True)
    expected = np.arange(len(unique), dtype=np.int64)
    if not np.array_equal(unique, expected):
        raise ValueError("cluster labels must be contiguous from zero")
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 1.0
    ordered = np.sort(counts.astype(np.float64))
    count = len(ordered)
    gini = float(
        (2.0 * np.sum(np.arange(1, count + 1) * ordered) / (count * ordered.sum()))
        - (count + 1) / count
    )
    return {
        "assigned_basis_count": int(counts.sum()),
        "cluster_count": len(counts),
        "minimum_cluster_size": int(counts.min()),
        "median_cluster_size": float(np.median(counts)),
        "maximum_cluster_size": int(counts.max()),
        "maximum_cluster_fraction": float(counts.max() / counts.sum()),
        "singleton_cluster_count": int(np.sum(counts == 1)),
        "singleton_cluster_fraction": float(np.mean(counts == 1)),
        "tiny_cluster_count_lt5": int(np.sum(counts < 5)),
        "tiny_cluster_fraction_lt5": float(np.mean(counts < 5)),
        "assigned_mass_in_clusters_ge8": float(
            counts[counts >= 8].sum() / counts.sum()
        ),
        "normalized_size_entropy": normalized_entropy,
        "size_gini": gini,
        "cluster_sizes": counts.astype(int).tolist(),
    }


def sparse_graph_partition_metrics(
    labels: NDArray[np.int64],
    affinity: csr_matrix,
) -> dict[str, Any]:
    if affinity.shape != (len(labels), len(labels)):
        raise ValueError("affinity shape does not match assignments")
    if (affinity - affinity.T).nnz:
        raise ValueError("partition affinity must be symmetric")
    active = labels >= 0
    if not np.any(active):
        raise ValueError("partition metrics require assigned bases")
    graph = affinity[active][:, active].tocsr()
    active_labels = labels[active]
    degree = np.asarray(graph.sum(axis=1)).ravel()
    total_weight = float(degree.sum() / 2.0)
    if total_weight <= 0:
        raise ValueError("partition affinity has no positive weight")

    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    columns = graph.indices
    same = active_labels[rows] == active_labels[columns]
    internal_weight = float(graph.data[same].sum() / 2.0)
    observed_internal_fraction = internal_weight / total_weight

    cluster_count = int(active_labels.max()) + 1
    volumes = np.bincount(
        active_labels,
        weights=degree,
        minlength=cluster_count,
    )
    expected_internal_fraction = float(np.sum((volumes / (2.0 * total_weight)) ** 2))
    modularity = observed_internal_fraction - expected_internal_fraction
    conductances: list[float] = []
    for cluster_id in range(cluster_count):
        mask = active_labels == cluster_id
        volume = float(degree[mask].sum())
        internal_twice = float(graph[mask][:, mask].sum())
        cut = max(0.0, volume - internal_twice)
        denominator = min(volume, 2.0 * total_weight - volume)
        conductances.append(cut / denominator if denominator > 0 else 0.0)
    cluster_sizes = np.bincount(active_labels, minlength=cluster_count)
    size_weighted_conductance = float(np.average(conductances, weights=cluster_sizes))
    return {
        "undirected_affinity_weight": total_weight,
        "observed_internal_affinity_fraction": observed_internal_fraction,
        "degree_volume_null_internal_fraction": expected_internal_fraction,
        "internal_affinity_enrichment": (
            observed_internal_fraction / expected_internal_fraction
            if expected_internal_fraction > 0
            else None
        ),
        "modularity": modularity,
        "minimum_conductance": min(conductances),
        "median_conductance": median(conductances),
        "maximum_conductance": max(conductances),
        "size_weighted_mean_conductance": size_weighted_conductance,
        "cluster_conductances": conductances,
    }


def _affinity_for_config(
    config: Mapping[str, Any],
    *,
    evidence: PairEvidence,
    support: BasisSupport,
) -> csr_matrix:
    eligible = (
        (support.target_counts >= int(config["basis_min_target_count"]))
        & (support.response_counts >= int(config["basis_min_response_count"]))
        & (support.family_counts >= int(config["basis_min_family_count"]))
    )
    if bool(config["exclude_boundary_layers"]):
        eligible &= ~support.boundary_mask
    similarity = mean_similarity_matrix(
        evidence,
        min_pair_target_overlap=int(config["pair_min_target_overlap"]),
        min_pair_response_overlap=int(config["pair_min_response_overlap"]),
        min_pair_family_overlap=int(config["pair_min_family_overlap"]),
        eligible_mask=eligible,
    )
    return knn_affinity(
        similarity,
        neighbors=int(config["neighbors"]),
        symmetrization=config["knn_symmetrization"],
        minimum_affinity=float(config["minimum_affinity"]),
    )


def _group_seed_states(
    states: Mapping[int, LoadedClusterState],
    *,
    sensitivity: str,
) -> dict[int, list[LoadedClusterState]]:
    grouped: dict[int, list[LoadedClusterState]] = defaultdict(list)
    for state in states.values():
        if state.config["sensitivity"] == sensitivity:
            grouped[int(state.config["n_clusters"])].append(state)
    return dict(grouped)


def _matching_sensitivity_states(
    states: Mapping[int, LoadedClusterState],
    *,
    n_clusters: int,
) -> list[LoadedClusterState]:
    return [
        state
        for state in states.values()
        if int(state.config["n_clusters"]) == n_clusters
        and state.config["sensitivity"]
        in {
            "basis_target_support_2",
            "basis_target_support_5",
            "neighbors_16",
            "neighbors_64",
            "pair_target_overlap_3",
        }
    ]


def build_structural_report(
    source_plan_path: Path,
    *,
    code_revision: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = _validate_source_plan(source_plan_path)
    states = load_cluster_states(plan)
    evidence_by_weighting: dict[str, tuple[PairEvidence, BasisSupport]] = {}
    pair_manifest_hashes: dict[str, str] = {}
    for item in plan["pair_evidence"]:
        weighting = str(item["weighting"])
        path = Path(str(item["output_path"]))
        evidence_by_weighting[weighting] = load_pair_evidence(path)
        pair_manifest_hashes[weighting] = load_json_object(path / "manifest.json")[
            "manifest_sha256"
        ]

    state_metrics: dict[int, dict[str, Any]] = {}
    for task_index, state in sorted(states.items()):
        config = state.config
        evidence, support = evidence_by_weighting[str(config["weighting"])]
        affinity = _affinity_for_config(
            config,
            evidence=evidence,
            support=support,
        )
        state_metrics[task_index] = {
            "task_index": task_index,
            "config_sha256": config["config_sha256"],
            "sensitivity": config["sensitivity"],
            "n_clusters": int(config["n_clusters"]),
            "random_seed": int(config["random_seed"]),
            "weighting": config["weighting"],
            "size": cluster_size_metrics(state.labels),
            "graph": sparse_graph_partition_metrics(state.labels, affinity),
        }

    primary_groups = _group_seed_states(states, sensitivity="primary")
    unweighted_groups = _group_seed_states(states, sensitivity="unweighted")
    candidate_records: list[dict[str, Any]] = []
    for n_clusters in sorted(primary_groups):
        primary_stability = seed_stability(primary_groups[n_clusters])
        unweighted_stability = seed_stability(unweighted_groups[n_clusters])
        medoid_task = int(primary_stability["medoid_task_index"])
        unweighted_medoid_task = int(unweighted_stability["medoid_task_index"])
        medoid_state = states[medoid_task]
        sensitivity_records = [
            {
                "task_index": state.task_index,
                "sensitivity": state.config["sensitivity"],
                "ari_on_shared_assigned_bases": assignment_ari(
                    medoid_state.labels,
                    state.labels,
                ),
            }
            for state in sorted(
                _matching_sensitivity_states(
                    states,
                    n_clusters=n_clusters,
                ),
                key=lambda item: item.task_index,
            )
        ]
        sensitivity_records.append(
            {
                "task_index": unweighted_medoid_task,
                "sensitivity": "unweighted_medoid",
                "ari_on_shared_assigned_bases": assignment_ari(
                    medoid_state.labels,
                    states[unweighted_medoid_task].labels,
                ),
            }
        )
        sensitivity_values = [
            float(record["ari_on_shared_assigned_bases"])
            for record in sensitivity_records
        ]
        size = state_metrics[medoid_task]["size"]
        graph = state_metrics[medoid_task]["graph"]
        preliminary_gates = {
            "assigned_coverage": (
                int(medoid_state.manifest["assigned_basis_count"])
                / int(medoid_state.manifest["eligible_basis_count"])
                >= 0.95
            ),
            "maximum_cluster_fraction": (
                float(size["maximum_cluster_fraction"]) <= 0.15
            ),
            "singleton_cluster_fraction": (
                float(size["singleton_cluster_fraction"]) <= 0.02
            ),
            "seed_mean_ari": float(primary_stability["mean_ari"]) >= 0.72,
            "seed_minimum_ari": (float(primary_stability["minimum_ari"]) >= 0.70),
            "modularity": float(graph["modularity"]) >= 0.20,
            "affinity_enrichment": (
                float(graph["internal_affinity_enrichment"]) >= 1.25
            ),
            "sensitivity_median_ari": median(sensitivity_values) >= 0.50,
        }
        candidate_records.append(
            {
                "n_clusters": n_clusters,
                "medoid_task_index": medoid_task,
                "source_state_manifest_sha256": medoid_state.manifest[
                    "manifest_sha256"
                ],
                "seed_stability": primary_stability,
                "unweighted_seed_stability": unweighted_stability,
                "sensitivity_agreement": sensitivity_records,
                "sensitivity_median_ari": median(sensitivity_values),
                "size": size,
                "graph": graph,
                "preliminary_gates": preliminary_gates,
                "passes_preliminary_gates": all(preliminary_gates.values()),
                "projection_status": "pending",
                "family_blocked_status": "pending",
            }
        )

    report: dict[str, Any] = {
        "schema_version": STRUCTURAL_REPORT_SCHEMA,
        "source_plan": {
            "path": str(source_plan_path.resolve()),
            "plan_sha256": plan["plan_sha256"],
        },
        "source_pair_evidence_manifest_sha256": pair_manifest_hashes,
        "state_count": len(states),
        "selection_rule": {
            "primary_pool": ("hierarchical input-profile primary configurations"),
            "seed_representative": "maximum mean ARI medoid",
            "semantic_inspection_used": False,
            "descriptions_generated": False,
        },
        "analysis_code_revision": (
            dict(code_revision) if code_revision is not None else None
        ),
        "analysis_environment": (
            dict(environment) if environment is not None else None
        ),
        "state_metrics": [state_metrics[index] for index in sorted(state_metrics)],
        "candidate_resolutions": candidate_records,
        "descriptions_generated": False,
        "scientific_cluster_state_selected": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def write_structural_report(path: Path, report: Mapping[str, Any]) -> None:
    write_hashed_json(path, report, hash_field="report_sha256")
