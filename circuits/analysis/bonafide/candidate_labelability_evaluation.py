"""Frozen label-free held-out evaluation for candidate-aware cluster states.

This adapter is deliberately downstream of a fully validated input bundle and
persisted clustering baseline.  It opens only profile, assignment, and frozen
partition data: labels, task outcomes, and generated descriptions are outside
its contract.  The report is explicitly preliminary until the independently
specified direction-null and generation-family jackknife analyses complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from circuits.analysis.bonafide import (
    candidate_clustering as candidate_clustering_module,
)
from circuits.analysis.bonafide import (
    candidate_clustering_execution as candidate_clustering_execution_module,
)
from circuits.analysis.bonafide import candidate_coherence as candidate_coherence_module
from circuits.analysis.bonafide import candidate_profiles as candidate_profiles_module
from circuits.analysis.bonafide import canonical as canonical_module
from circuits.analysis.bonafide import clustering as clustering_module
from circuits.analysis.bonafide import (
    clustering_evaluation as clustering_evaluation_module,
)
from circuits.analysis.bonafide import clustering_store as clustering_store_module
from circuits.analysis.bonafide.candidate_clustering import (
    CandidateClusterInputBundle,
    load_candidate_cluster_input_bundle,
)
from circuits.analysis.bonafide.candidate_clustering_execution import (
    LoadedCandidateClusteringBaseline,
    load_candidate_clustering_baseline,
)
from circuits.analysis.bonafide.candidate_coherence import (
    FROZEN_BOOTSTRAP_REPLICATES,
    FROZEN_CANDIDATE_STATES,
    FROZEN_PARTITION_FAMILY_COUNTS,
    FROZEN_WIDTH_STATES,
    candidate_coherence_bootstrap,
    cluster_support_readiness,
    evaluate_candidate_coherence,
    evaluate_width_one_coherence,
    generation_candidate_centroids,
)
from circuits.analysis.bonafide.candidate_profiles import (
    CANDIDATE_PROFILE_SCHEMA,
    WIDTH_PROFILE_SCHEMA,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)

CANDIDATE_LABELABILITY_EVALUATION_SCHEMA = (
    "adag.bonafide.candidate-labelability-evaluation.v1"
)
REPORT_FILENAME = "candidate-labelability-evaluation.json"

_PARTITIONS = ("generation", "selection_scoring", "audit")
_HELDOUT_PARTITIONS = ("selection_scoring", "audit")
_STATE_NAMES = tuple(FROZEN_CANDIDATE_STATES)
_COMPARISONS = ("C_minus_W", "F_minus_W", "C_minus_S", "F_minus_S")
type _IntArray = npt.NDArray[np.int64]

_SOURCE_BINDINGS = {
    "canonical": "circuits/analysis/bonafide/canonical.py",
    "clustering": "circuits/analysis/bonafide/clustering.py",
    "clustering_evaluation": "circuits/analysis/bonafide/clustering_evaluation.py",
    "clustering_store": "circuits/analysis/bonafide/clustering_store.py",
    "candidate_profiles": "circuits/analysis/bonafide/candidate_profiles.py",
    "candidate_clustering": "circuits/analysis/bonafide/candidate_clustering.py",
    "candidate_clustering_execution": (
        "circuits/analysis/bonafide/candidate_clustering_execution.py"
    ),
    "candidate_coherence": "circuits/analysis/bonafide/candidate_coherence.py",
    "candidate_labelability_evaluation": (
        "circuits/analysis/bonafide/candidate_labelability_evaluation.py"
    ),
    "candidate_labelability_cli": (
        "scripts/bonafide/candidate_labelability_evaluate.py"
    ),
    "frozen_protocol": "docs/CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md",
}

_RUNTIME_MODULE_BINDINGS = {
    "candidate_clustering": candidate_clustering_module,
    "candidate_clustering_execution": candidate_clustering_execution_module,
    "candidate_coherence": candidate_coherence_module,
    "candidate_profiles": candidate_profiles_module,
    "canonical": canonical_module,
    "clustering": clustering_module,
    "clustering_evaluation": clustering_evaluation_module,
    "clustering_store": clustering_store_module,
}


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise ValueError(
            f"unable to bind labelability source revision: {message}"
        ) from error
    return completed.stdout.strip()


def collect_candidate_labelability_revision(repo_root: Path) -> dict[str, Any]:
    """Bind the complete clean tracked tree and every evaluation source file."""

    repo_root = repo_root.resolve()
    if Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve() != repo_root:
        raise ValueError("labelability evaluation must run from the repository root")
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError("labelability evaluation requires a clean tracked worktree")
    validate_candidate_labelability_runtime_paths(repo_root)
    files: list[dict[str, str]] = []
    for role, relative in _SOURCE_BINDINGS.items():
        if _git(repo_root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(
                f"labelability evaluation source is not tracked: {relative}"
            )
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"labelability evaluation source is missing: {relative}")
        files.append({"role": role, "path": relative, "sha256": file_sha256(path)})
    return {
        "repo_root": str(repo_root),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "files": files,
    }


def validate_candidate_labelability_runtime_paths(repo_root: Path) -> None:
    """Reject editable-install leakage from a different worktree at runtime."""

    repo_root = repo_root.resolve()
    own_expected = repo_root / _SOURCE_BINDINGS["candidate_labelability_evaluation"]
    if Path(__file__).resolve() != own_expected.resolve():
        raise ValueError(
            "candidate labelability evaluation module was imported from another worktree"
        )
    for role, module in _RUNTIME_MODULE_BINDINGS.items():
        observed_raw = getattr(module, "__file__", None)
        if not isinstance(observed_raw, str):
            raise TypeError(
                f"candidate labelability runtime module has no path: {role}"
            )
        expected = repo_root / _SOURCE_BINDINGS[role]
        if Path(observed_raw).resolve() != expected.resolve():
            raise ValueError(
                f"candidate labelability runtime module came from another worktree: {role}"
            )


def _exact_parquet_rows(path: Path, schema: pa.Schema) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    # Parquet commonly round-trips Arrow list child names as ``element`` rather
    # than ``item``; semantic schemas, including nullability, must still match.
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"candidate labelability parquet schema drift: {path.name}")
    return table.to_pylist()


def _family_ids(bundle: CandidateClusterInputBundle) -> dict[str, tuple[str, ...]]:
    partitions = bundle.family_partitions.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != set(_PARTITIONS):
        raise ValueError("candidate labelability family partition inventory drift")
    result: dict[str, tuple[str, ...]] = {}
    all_ids: set[str] = set()
    for partition in _PARTITIONS:
        raw = partitions[partition]
        if not isinstance(raw, list) or any(
            not isinstance(value, str) or not value for value in raw
        ):
            raise ValueError(f"invalid family IDs for partition {partition!r}")
        values = tuple(raw)
        expected = FROZEN_PARTITION_FAMILY_COUNTS[partition]
        if len(values) != expected or len(set(values)) != expected:
            raise ValueError(
                f"partition {partition!r} must bind exactly {expected} unique families"
            )
        if all_ids.intersection(values):
            raise ValueError("candidate labelability family partitions overlap")
        all_ids.update(values)
        result[partition] = values
    return result


def extract_chosen_medoid_assignments(
    baseline: LoadedCandidateClusteringBaseline,
    *,
    basis_count: int,
) -> dict[str, _IntArray]:
    """Extract exactly one chosen, valid medoid assignment for W/C/F/S."""

    manifest = baseline.manifest
    if manifest.get("numerically_valid") is not True:
        raise ValueError("candidate clustering baseline is not numerically valid")
    chosen = manifest.get("chosen_cluster_count")
    if isinstance(chosen, bool) or not isinstance(chosen, int) or chosen <= 1:
        raise ValueError(
            "candidate clustering baseline has no valid chosen cluster count"
        )
    if int(manifest.get("basis_count", -1)) != basis_count:
        raise ValueError("candidate baseline and input basis counts disagree")

    rows = baseline.assignments.to_pylist()
    output: dict[str, _IntArray] = {}
    for state in _STATE_NAMES:
        selected = [
            row
            for row in rows
            if row["view"] == state
            and row["n_clusters"] == chosen
            and bool(row["is_medoid"])
        ]
        if len(selected) != basis_count:
            raise ValueError(f"state {state!r} lacks one complete chosen medoid block")
        selected.sort(key=lambda row: int(row["signed_basis_index"]))
        if [int(row["signed_basis_index"]) for row in selected] != list(
            range(basis_count)
        ):
            raise ValueError(f"state {state!r} medoid basis order is incomplete")
        state_indices = {int(row["state_index"]) for row in selected}
        seeds = {int(row["seed"]) for row in selected}
        if len(state_indices) != 1 or len(seeds) != 1:
            raise ValueError(f"state {state!r} mixes medoid state blocks")
        if any(
            not bool(row["fit_valid"]) or not bool(row["seed_valid"])
            for row in selected
        ):
            raise ValueError(f"state {state!r} chosen medoid is not valid")
        values = np.full(basis_count, -1, dtype=np.int64)
        for basis_index, row in enumerate(selected):
            assigned = bool(row["assigned"])
            cluster = row["cluster_id"]
            if assigned != (cluster is not None):
                raise ValueError(f"state {state!r} assignment nullability drift")
            if assigned:
                cluster_id = int(cluster)
                if not 0 <= cluster_id < chosen:
                    raise ValueError(f"state {state!r} cluster ID is out of range")
                values[basis_index] = cluster_id
        output[state] = values
    return output


def normalized_profile_records(
    bundle: CandidateClusterInputBundle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Independently read validated profile Parquets into explicit records."""

    target_by_case = {str(row["case_id"]): row for row in bundle.target_rows}
    if len(target_by_case) != len(bundle.target_rows):
        raise ValueError("candidate labelability target identities are not unique")
    basis_by_index = {int(row["signed_basis_index"]): row for row in bundle.basis_rows}
    if sorted(basis_by_index) != list(range(bundle.basis_count)):
        raise ValueError("candidate labelability basis index is not canonical")

    def identity(row: Mapping[str, Any]) -> dict[str, Any]:
        case_id = str(row["case_id"])
        if case_id not in target_by_case:
            raise ValueError("profile references an unknown target")
        target = target_by_case[case_id]
        basis_index = int(row["signed_basis_index"])
        basis = basis_by_index.get(basis_index)
        if basis is None:
            raise ValueError("profile references an unknown signed basis")
        for field in (
            "model_id",
            "model_revision",
            "layer",
            "neuron_index",
            "polarity",
        ):
            if row[field] != basis[field]:
                raise ValueError("profile signed-basis identity drift")
        return {
            "partition": str(target["family_partition"]),
            "family_id": str(target["base_question_id"]),
            "response_id": str(target["response_id"]),
            "target_id": case_id,
            "basis_index": basis_index,
        }

    candidate_rows = _exact_parquet_rows(
        bundle.root / "candidate-profiles.parquet", CANDIDATE_PROFILE_SCHEMA
    )
    candidate_records: list[dict[str, Any]] = []
    for row in candidate_rows:
        vector = np.asarray(row["candidate_contrast_profile"], dtype=np.float64)
        if vector.ndim != 1 or vector.size != 5 or not np.all(np.isfinite(vector)):
            raise ValueError("candidate direction must be a finite five-vector")
        candidate_records.append({**identity(row), "vector": vector.tolist()})

    width_rows = _exact_parquet_rows(
        bundle.root / "width-profiles.parquet", WIDTH_PROFILE_SCHEMA
    )
    width_records: list[dict[str, Any]] = []
    for row in width_rows:
        values_raw = row["attribution_profile"]
        support_raw = row["attribution_support"]
        if len(values_raw) == 0 or len(values_raw) != len(support_raw):
            raise ValueError("width profile values/support are not aligned")
        if any(type(value) is not bool for value in support_raw):
            raise TypeError("width profile support must contain actual booleans")
        support = np.asarray(support_raw, dtype=np.bool_)
        values = np.asarray(
            [0.0 if value is None else value for value in values_raw],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values[support])):
            raise ValueError("supported width profile coordinates must be finite")
        if any(
            value is None and bool(mask)
            for value, mask in zip(values_raw, support_raw, strict=True)
        ):
            raise ValueError("supported width profile coordinate is null")
        width_records.append(
            {
                **identity(row),
                "values": values.tolist(),
                "support": support.tolist(),
            }
        )
    return candidate_records, width_records


def _partition_records(
    records: Sequence[Mapping[str, Any]], partition: str
) -> list[Mapping[str, Any]]:
    return [record for record in records if record["partition"] == partition]


def _bootstrap_reports(coherence: Mapping[str, Any]) -> dict[str, Any]:
    partition = str(coherence["partition"])
    raw_family_ids = coherence["expected_family_ids"]
    if not isinstance(raw_family_ids, Sequence) or isinstance(raw_family_ids, str):
        raise TypeError("coherence report family IDs are invalid")
    family_ids: tuple[str, ...] = tuple(
        family_id for family_id in raw_family_ids if isinstance(family_id, str)
    )
    if len(family_ids) != len(raw_family_ids):
        raise TypeError("coherence report family IDs are invalid")
    expected_family_ids = set(family_ids)
    comparisons = coherence["comparisons"]
    result: dict[str, Any] = {}
    for comparison in _COMPARISONS:
        report = comparisons.get(comparison)
        if not isinstance(report, Mapping):
            raise TypeError(f"coherence report lacks comparison {comparison}")
        effects = report.get("per_family_effect")
        if not isinstance(effects, Mapping):
            raise TypeError(f"coherence comparison {comparison} lacks family effects")
        observed_family_ids: set[str] = set()
        for family_id in effects:
            if not isinstance(family_id, str):
                raise TypeError(
                    f"coherence comparison {comparison} has an invalid family ID"
                )
            observed_family_ids.add(family_id)
        if observed_family_ids != expected_family_ids:
            result[comparison] = {
                "available": False,
                "reason": "not_all_frozen_families_scoreable",
                "family_ids": sorted(observed_family_ids),
                "family_count": len(observed_family_ids),
                "missing_family_ids": sorted(expected_family_ids - observed_family_ids),
                "unexpected_family_ids": sorted(
                    observed_family_ids - expected_family_ids
                ),
                "replicates": FROZEN_BOOTSTRAP_REPLICATES,
                "mean_effect": report.get("mean_effect"),
                "ci_95_lower": None,
                "ci_95_upper": None,
            }
        else:
            result[comparison] = {
                "available": True,
                "reason": None,
                **candidate_coherence_bootstrap(
                    effects,
                    expected_family_ids=family_ids,
                    partition=partition,
                    replicates=FROZEN_BOOTSTRAP_REPLICATES,
                ),
            }
    return result


def _chosen_resolution(
    baseline: LoadedCandidateClusteringBaseline, state: str
) -> Mapping[str, Any]:
    chosen = int(baseline.manifest["chosen_cluster_count"])
    matches = [
        item
        for item in baseline.manifest["resolution_diagnostics"]
        if item["view"] == state and item["n_clusters"] == chosen
    ]
    if len(matches) != 1 or matches[0].get("valid") is not True:
        raise ValueError(f"chosen structural diagnostics unavailable for {state}")
    return matches[0]


def _condition(
    name: str, observed: Any, threshold: str, passed: bool
) -> dict[str, Any]:
    return {
        "condition": name,
        "observed": observed,
        "threshold": threshold,
        "satisfied": bool(passed),
    }


def _structural_gates(
    baseline: LoadedCandidateClusteringBaseline,
    readiness: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for state in _STATE_NAMES:
        resolution = _chosen_resolution(baseline, state)
        medoid_seed = resolution["medoid_seed"]
        medoid = next(
            item for item in resolution["seeds"] if item["seed"] == medoid_seed
        )
        size = resolution["size_metrics"]
        graph = resolution["graph_metrics"]
        conditions = [
            _condition(
                "assignment_fraction",
                medoid["assignment_fraction"],
                ">= 0.95",
                medoid["assignment_fraction"] >= 0.95,
            ),
            _condition(
                "largest_cluster_fraction",
                size["maximum_cluster_fraction"],
                "<= 0.15",
                size["maximum_cluster_fraction"] <= 0.15,
            ),
            _condition(
                "mean_seed_ari",
                resolution["mean_seed_ari"],
                ">= 0.72",
                resolution["mean_seed_ari"] >= 0.72,
            ),
            _condition(
                "minimum_seed_ari",
                resolution["minimum_seed_ari"],
                ">= 0.70",
                resolution["minimum_seed_ari"] >= 0.70,
            ),
            _condition(
                "modularity",
                graph["modularity"],
                ">= 0.20",
                graph["modularity"] >= 0.20,
            ),
            _condition(
                "affinity_enrichment",
                graph["internal_affinity_enrichment"],
                ">= 1.25",
                graph["internal_affinity_enrichment"] >= 1.25,
            ),
            _condition(
                "labeling_ready_cluster_fraction",
                readiness[state]["labeling_ready_cluster_fraction"],
                ">= 0.80",
                readiness[state]["labeling_ready_cluster_fraction"] >= 0.80,
            ),
        ]
        reports[state] = {
            "conditions": conditions,
            "all_pre_jackknife_conditions_satisfied": all(
                item["satisfied"] for item in conditions
            ),
            "jackknife_required": True,
            "final_structural_gate": None,
        }
    return reports


def _functional_gates(
    coherence_by_partition: Mapping[str, Mapping[str, Any]],
    bootstrap_by_partition: Mapping[str, Mapping[str, Any]],
    width_by_partition: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for state in ("C", "F"):
        all_conditions: list[dict[str, Any]] = []
        partitions: dict[str, Any] = {}
        for partition in _HELDOUT_PARTITIONS:
            coherence = coherence_by_partition[partition]
            comparisons = coherence["comparisons"]
            xw = comparisons[f"{state}_minus_W"]
            xs = comparisons[f"{state}_minus_S"]
            bootstrap = bootstrap_by_partition[partition][f"{state}_minus_W"]
            bootstrap_lower = bootstrap["ci_95_lower"]
            conditions = [
                _condition(
                    "lift_over_W",
                    xw["mean_effect"],
                    ">= 0.05",
                    xw["mean_effect"] is not None and xw["mean_effect"] >= 0.05,
                ),
                _condition(
                    "bootstrap_lower_bound",
                    bootstrap_lower,
                    "> 0",
                    bootstrap.get("available") is True
                    and bootstrap_lower is not None
                    and bootstrap_lower > 0.0,
                ),
                _condition(
                    "positive_family_count",
                    xw["positive_family_count"],
                    ">= 7 of 8",
                    xw["family_count"] == 8 and xw["positive_family_count"] >= 7,
                ),
                _condition(
                    "lift_over_S",
                    xs["mean_effect"],
                    "> 0",
                    xs["mean_effect"] is not None and xs["mean_effect"] > 0.0,
                ),
                _condition(
                    "all_families_scoreable",
                    coherence["all_families_scoreable"],
                    "true",
                    coherence["all_families_scoreable"] is True,
                ),
            ]
            if state == "F":
                width = width_by_partition[partition]
                width_effect = width["comparisons"]["F_minus_W"]["mean_effect"]
                conditions.extend(
                    [
                        _condition(
                            "width_one_coherence_loss",
                            width_effect,
                            ">= -0.05",
                            width_effect is not None and width_effect >= -0.05,
                        ),
                        _condition(
                            "width_all_families_scoreable",
                            width["all_families_scoreable"],
                            "true",
                            width["all_families_scoreable"] is True,
                        ),
                    ]
                )
            partitions[partition] = {
                "conditions": conditions,
                "all_pre_null_conditions_satisfied": all(
                    item["satisfied"] for item in conditions
                ),
            }
            all_conditions.extend(conditions)
        states[state] = {
            "partitions": partitions,
            "all_pre_null_conditions_satisfied": all(
                item["satisfied"] for item in all_conditions
            ),
            "direction_null_required": True,
            "generation_family_jackknife_required": True,
            "final_functional_gate": None,
        }
    return states


def _conservative_labeling_readiness(
    candidate_support: Mapping[str, Any], width_support: Mapping[str, Any]
) -> dict[str, Any]:
    """Require frozen witness thresholds in both evidence families.

    Candidate support strictly supersets width support in the frozen corpus, so
    candidate-only counts would overstate whether the paired W width-only arm
    has enough witnesses.  Separate reports remain visible for diagnosis while
    the joint result is the conservative anchor/guardrail policy.
    """

    cluster_count = int(candidate_support["cluster_count"])
    if int(width_support["cluster_count"]) != cluster_count:
        raise ValueError("candidate and width readiness cluster counts disagree")
    candidate_clusters = candidate_support["clusters"]
    width_clusters = width_support["clusters"]
    if len(candidate_clusters) != cluster_count or len(width_clusters) != cluster_count:
        raise ValueError("candidate or width readiness cluster inventory is incomplete")
    clusters: list[dict[str, Any]] = []
    for cluster_id, (candidate, width) in enumerate(
        zip(candidate_clusters, width_clusters, strict=True)
    ):
        if candidate["cluster_id"] != cluster_id or width["cluster_id"] != cluster_id:
            raise ValueError("candidate and width readiness cluster order drift")
        clusters.append(
            {
                "cluster_id": cluster_id,
                "candidate_support_ready": bool(candidate["labeling_ready"]),
                "width_support_ready": bool(width["labeling_ready"]),
                "labeling_ready": bool(
                    candidate["labeling_ready"] and width["labeling_ready"]
                ),
            }
        )
    ready_count = sum(item["labeling_ready"] for item in clusters)
    return {
        "support_policy": "candidate_and_width_frozen_thresholds",
        "cluster_count": cluster_count,
        "labeling_ready_cluster_count": ready_count,
        "labeling_ready_cluster_fraction": ready_count / cluster_count,
        "clusters": clusters,
        "candidate_support": candidate_support,
        "width_support": width_support,
    }


def evaluate_loaded_candidate_labelability(
    bundle: CandidateClusterInputBundle,
    baseline: LoadedCandidateClusteringBaseline,
) -> dict[str, Any]:
    """Evaluate validated artifacts without opening labels or outcomes."""

    source = baseline.manifest.get("source_input_bundle")
    if not isinstance(source, Mapping) or source.get(
        "manifest_sha256"
    ) != bundle.manifest.get("manifest_sha256"):
        raise ValueError("candidate baseline is not bound to the supplied input bundle")
    families = _family_ids(bundle)
    assignments = extract_chosen_medoid_assignments(
        baseline, basis_count=bundle.basis_count
    )
    candidate_records, width_records = normalized_profile_records(bundle)
    observed_candidate_families = {
        partition: {
            str(item["family_id"])
            for item in _partition_records(candidate_records, partition)
        }
        for partition in _PARTITIONS
    }
    observed_width_families = {
        partition: {
            str(item["family_id"])
            for item in _partition_records(width_records, partition)
        }
        for partition in _PARTITIONS
    }
    for partition in _PARTITIONS:
        expected = set(families[partition])
        if (
            observed_candidate_families[partition] != expected
            or observed_width_families[partition] != expected
        ):
            raise ValueError(
                f"profile family coverage differs from frozen {partition!r} binding"
            )

    chosen = int(baseline.manifest["chosen_cluster_count"])
    generation = _partition_records(candidate_records, "generation")
    centroids = {
        state: generation_candidate_centroids(
            generation, assignments[state], n_clusters=chosen
        )
        for state in _STATE_NAMES
    }
    if any(centroid.family_count != 18 for centroid in centroids.values()):
        raise ValueError("generation centroid adapter did not consume all 18 families")

    coherence: dict[str, Any] = {}
    bootstraps: dict[str, Any] = {}
    width: dict[str, Any] = {}
    for partition in _HELDOUT_PARTITIONS:
        coherence[partition] = evaluate_candidate_coherence(
            _partition_records(candidate_records, partition),
            assignments,
            centroids,
            partition=partition,
            expected_family_ids=families[partition],
        )
        bootstraps[partition] = _bootstrap_reports(coherence[partition])
        width[partition] = evaluate_width_one_coherence(
            _partition_records(width_records, partition),
            {state: assignments[state] for state in FROZEN_WIDTH_STATES},
            partition=partition,
            expected_family_ids=families[partition],
        )

    readiness = {}
    for state in _STATE_NAMES:
        candidate_readiness = cluster_support_readiness(
            candidate_records,
            assignments[state],
            expected_family_ids_by_partition=families,
            n_clusters=chosen,
        )
        width_readiness = cluster_support_readiness(
            width_records,
            assignments[state],
            expected_family_ids_by_partition=families,
            n_clusters=chosen,
        )
        readiness[state] = _conservative_labeling_readiness(
            candidate_readiness, width_readiness
        )
    centroid_reports = {
        state: {
            "cluster_count": value.cluster_count,
            "dimension": value.dimension,
            "family_count": value.family_count,
            "response_count": value.response_count,
            "target_count": value.target_count,
            "available_cluster_count": int(value.available.sum()),
            "clusters": list(value.cluster_reports),
        }
        for state, value in centroids.items()
    }
    return {
        "chosen_cluster_count": chosen,
        "record_counts": {
            "candidate_basis_target_occurrences": len(candidate_records),
            "width_basis_target_occurrences": len(width_records),
        },
        "frozen_family_ids_by_partition": {
            key: list(value) for key, value in families.items()
        },
        "assignment_counts": {
            state: {
                "assigned_basis_count": int(np.sum(values >= 0)),
                "unassigned_basis_count": int(np.sum(values < 0)),
            }
            for state, values in assignments.items()
        },
        "generation_centroids": centroid_reports,
        "heldout_candidate_coherence": coherence,
        "paired_family_bootstraps": bootstraps,
        "heldout_width_one_coherence": width,
        "cluster_labeling_readiness": readiness,
        "pre_null_pre_jackknife_gates": {
            "structural": _structural_gates(baseline, readiness),
            "functional": _functional_gates(coherence, bootstraps, width),
            "final_pass_claimed": False,
            "status": "pending_direction_null_and_generation_family_jackknife",
        },
    }


def _artifact_record(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    observed = load_json_object(path)
    if observed != dict(manifest):
        raise ValueError(f"source manifest changed during evaluation: {path}")
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "manifest_file_sha256": file_sha256(path),
        "schema_version": str(manifest["schema_version"]),
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Publish once with portable O_EXCL; an interrupted file fails validation."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace evaluation report: {path}"
        ) from error
    # An interrupted write deliberately leaves a partial exclusive file.  Its
    # missing or mismatched self-hash prevents consumption and silent retry.
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def run_candidate_labelability_evaluation(
    *, input_root: Path, baseline_root: Path, output_path: Path, repo_root: Path
) -> dict[str, Any]:
    """Full-validate inputs, evaluate, and exclusively publish a fail-closed report."""

    revision = collect_candidate_labelability_revision(repo_root)
    bundle = load_candidate_cluster_input_bundle(input_root)
    baseline = load_candidate_clustering_baseline(baseline_root, verify_source=True)
    evaluation = evaluate_loaded_candidate_labelability(bundle, baseline)
    # Close the payload and source TOCTOU window before publication.  These
    # loaders revalidate every bound Parquet, affinity, assignment, and hash.
    final_bundle = load_candidate_cluster_input_bundle(input_root)
    final_baseline = load_candidate_clustering_baseline(
        baseline_root, verify_source=True
    )
    if (
        final_bundle.root != bundle.root
        or dict(final_bundle.manifest) != dict(bundle.manifest)
        or final_baseline.root != baseline.root
        or dict(final_baseline.manifest) != dict(baseline.manifest)
    ):
        raise ValueError(
            "candidate labelability source artifacts changed during evaluation"
        )
    if collect_candidate_labelability_revision(repo_root) != revision:
        raise ValueError(
            "candidate labelability source revision changed during evaluation"
        )
    input_manifest_path = bundle.root / "manifest.json"
    baseline_manifest_path = baseline.root / "manifest.json"
    report: dict[str, Any] = {
        "schema_version": CANDIDATE_LABELABILITY_EVALUATION_SCHEMA,
        "purpose": "label_free_pre_null_pre_jackknife_candidate_labelability_evaluation",
        "source_input_bundle": _artifact_record(input_manifest_path, bundle.manifest),
        "source_clustering_baseline": _artifact_record(
            baseline_manifest_path, baseline.manifest
        ),
        "code_revision": revision,
        "firewall": {
            "outcomes_inspected": False,
            "labels_inspected": False,
            "descriptions_generated": False,
            "model_calls_made": False,
            "confirmatory_holdout_opened": False,
            "generation_only_centroids": True,
            "selection_and_audit_fit_influence": False,
            "final_pass_claimed": False,
        },
        "evaluation": evaluation,
    }
    report["manifest_sha256"] = canonical_sha256(report)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _write_exclusive(output_path, payload)
    return report


def _validate_artifact_binding(record: Any, *, label: str) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} binding is invalid")
    path = Path(str(record.get("manifest_path")))
    if not path.is_file() or file_sha256(path) != record.get("manifest_file_sha256"):
        raise ValueError(f"{label} manifest file drift")
    manifest = load_json_object(path)
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core) or recorded != record.get("manifest_sha256"):
        raise ValueError(f"{label} manifest content drift")
    if manifest.get("schema_version") != record.get("schema_version"):
        raise ValueError(f"{label} schema drift")


def _validate_revision(report: Mapping[str, Any]) -> None:
    revision = report.get("code_revision")
    if (
        not isinstance(revision, Mapping)
        or revision.get("tracked_worktree_clean") is not True
    ):
        raise ValueError("candidate labelability source revision is invalid")
    root = Path(str(revision.get("repo_root")))
    validate_candidate_labelability_runtime_paths(root)
    records = revision.get("files")
    if not isinstance(records, list) or len(records) != len(_SOURCE_BINDINGS):
        raise ValueError("candidate labelability source file inventory drift")
    by_role = {item.get("role"): item for item in records if isinstance(item, Mapping)}
    if set(by_role) != set(_SOURCE_BINDINGS):
        raise ValueError("candidate labelability source roles drift")
    for role, relative in _SOURCE_BINDINGS.items():
        item = by_role[role]
        if item.get("path") != relative or file_sha256(root / relative) != item.get(
            "sha256"
        ):
            raise ValueError(f"candidate labelability source hash drift: {role}")


def load_candidate_labelability_evaluation(
    path: Path, *, verify_sources: bool = True
) -> dict[str, Any]:
    """Load and fail-closed validate one persisted evaluation report."""

    report = load_json_object(path.resolve())
    core = dict(report)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core):
        raise ValueError("candidate labelability report self-hash mismatch")
    if report.get("schema_version") != CANDIDATE_LABELABILITY_EVALUATION_SCHEMA:
        raise ValueError("unsupported candidate labelability report schema")
    firewall = report.get("firewall")
    expected_false = (
        "outcomes_inspected",
        "labels_inspected",
        "descriptions_generated",
        "model_calls_made",
        "confirmatory_holdout_opened",
        "selection_and_audit_fit_influence",
        "final_pass_claimed",
    )
    if not isinstance(firewall, Mapping) or any(
        firewall.get(field) is not False for field in expected_false
    ):
        raise ValueError("candidate labelability firewall drift")
    if firewall.get("generation_only_centroids") is not True:
        raise ValueError("candidate labelability centroid firewall drift")
    evaluation = report.get("evaluation")
    gates = (
        evaluation.get("pre_null_pre_jackknife_gates")
        if isinstance(evaluation, Mapping)
        else None
    )
    if (
        not isinstance(gates, Mapping)
        or gates.get("final_pass_claimed") is not False
        or gates.get("status")
        != "pending_direction_null_and_generation_family_jackknife"
    ):
        raise ValueError("candidate labelability preliminary gate status drift")
    if verify_sources:
        _validate_revision(report)
        _validate_artifact_binding(
            report.get("source_input_bundle"), label="input bundle"
        )
        _validate_artifact_binding(
            report.get("source_clustering_baseline"), label="clustering baseline"
        )
        input_record = report["source_input_bundle"]
        baseline_record = report["source_clustering_baseline"]
        assert isinstance(input_record, Mapping)
        assert isinstance(baseline_record, Mapping)
        bundle = load_candidate_cluster_input_bundle(
            Path(str(input_record["manifest_path"])).resolve().parent
        )
        baseline = load_candidate_clustering_baseline(
            Path(str(baseline_record["manifest_path"])).resolve().parent,
            verify_source=True,
        )
        if bundle.manifest.get("manifest_sha256") != input_record.get(
            "manifest_sha256"
        ) or baseline.manifest.get("manifest_sha256") != baseline_record.get(
            "manifest_sha256"
        ):
            raise ValueError("candidate labelability deep source binding drift")
        source = baseline.manifest.get("source_input_bundle")
        if not isinstance(source, Mapping) or source.get(
            "manifest_sha256"
        ) != bundle.manifest.get("manifest_sha256"):
            raise ValueError("candidate labelability baseline/input binding drift")
        recomputed = evaluate_loaded_candidate_labelability(bundle, baseline)
        if evaluation != recomputed:
            raise ValueError("candidate labelability recomputed evaluation drift")
    return report
