"""Frozen exploratory labelability evaluation for hybrid candidate clusters.

Generation remains the only fit partition.  Held-out targets are reopened only
through their hash-bound candidate-union artifacts and are mapped into the
immutable generation basis universe before any statistic is computed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.sparse.linalg import ArpackNoConvergence
from sklearn.metrics import adjusted_rand_score

from circuits.analysis.bonafide.candidate_clustering import (
    MIN_BASIS_FAMILIES,
    MIN_BASIS_RESPONSES,
    MIN_BASIS_TARGETS,
    MIN_PAIR_FAMILIES,
    MIN_PAIR_RESPONSES,
    MIN_PAIR_TARGETS,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.clustering import mean_similarity_matrix
from circuits.analysis.bonafide.hybrid_candidate_clustering import (
    HybridEvidence,
    Representation,
    _fit_one,
    accumulate_fused_evidence,
    blocks_from_bundle,
    candidate_view,
)
from circuits.analysis.bonafide.hybrid_candidate_clustering_execution import (
    ASSIGNMENT_SCHEMA,
    load_hybrid_clustering_manifest,
)
from circuits.analysis.bonafide.hybrid_candidate_inputs import (
    HybridInputBundle,
    _locate_unions,
    extract_hybrid_target,
    load_hybrid_input_bundle,
)
from circuits.analysis.bonafide.identity import SignedBasisKey
from circuits.tracing.candidate_union import load_candidate_union_artifact

EVALUATION_SCHEMA = "adag.bonafide.hybrid-candidate-labelability.v1"
WITNESS_SCHEMA = "adag.bonafide.hybrid-candidate-witness-inventory.v1"
TARGET_SCHEMA = "adag.bonafide.hybrid-candidate-evaluation-targets.v1"
PROTOCOL_PATH = "docs/HYBRID_CANDIDATE_LABELABILITY_PROTOCOL.md"
BOOTSTRAP_REPLICATES = 10_000
PARTITIONS = ("generation", "selection_scoring", "audit")
HELDOUT_PARTITIONS = ("selection_scoring", "audit")
VIEWS = ("input", "candidate", "contrast")
READINESS_THRESHOLDS: Mapping[str, tuple[int, int]] = {
    "generation": (8, 4),
    "selection_scoring": (4, 2),
    "audit": (4, 2),
}
STATE_SPECS: Mapping[str, Mapping[str, Any]] = {
    "primary": {
        "representation": "raw_top5_plus_observed.v1",
        "representation_key": "raw",
        "affinity_mode": "full_positive",
        "n_clusters": 64,
        "seed": 17,
    },
    "alternative": {
        "representation": "paper_normalized_model_top5.v1",
        "representation_key": "paper_normalized",
        "affinity_mode": "full_positive",
        "n_clusters": 64,
        "seed": 29,
    },
}

OCCURRENCE_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("family_id", pa.string(), nullable=False),
        pa.field("partition", pa.string(), nullable=False),
        pa.field("basis_index", pa.int64(), nullable=False),
        pa.field("input_values", pa.list_(pa.float64()), nullable=False),
        pa.field("input_support", pa.list_(pa.bool_()), nullable=False),
        pa.field("paper_input_values", pa.list_(pa.float64()), nullable=False),
        pa.field("raw_candidate_values", pa.list_(pa.float64()), nullable=False),
        pa.field("paper_candidate_values", pa.list_(pa.float64()), nullable=False),
        pa.field("occurrence_count", pa.int32(), nullable=False),
    ]
)

EVALUATION_SOURCE_PATHS = (
    "circuits/analysis/bonafide/hybrid_candidate_labelability.py",
    "circuits/analysis/bonafide/hybrid_candidate_inputs.py",
    "circuits/analysis/bonafide/hybrid_candidate_clustering.py",
    "circuits/analysis/bonafide/hybrid_candidate_clustering_execution.py",
    "circuits/analysis/bonafide/clustering.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/tracing/candidate_union.py",
    "circuits/tracing/artifact.py",
    "scripts/bonafide/hybrid_candidate_labelability_evaluate.py",
    PROTOCOL_PATH,
    "pyproject.toml",
    "uv.lock",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def collect_evaluation_revision(repo_root: Path) -> dict[str, Any]:
    """Bind a clean commit and every executable evaluation source."""

    repo_root = repo_root.resolve()
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError("hybrid labelability requires a clean tracked worktree")
    files: list[dict[str, str]] = []
    for relative in EVALUATION_SOURCE_PATHS:
        if _git(repo_root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"evaluation source is not tracked: {relative}")
        path = repo_root / relative
        if _git(repo_root, "hash-object", relative) != _git(
            repo_root, "rev-parse", f"HEAD:{relative}"
        ):
            raise ValueError(f"evaluation source differs from HEAD: {relative}")
        files.append({"path": relative, "sha256": file_sha256(path)})
    return {
        "repo_root": str(repo_root),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "files": files,
    }


def _basis_map(bundle: HybridInputBundle) -> dict[SignedBasisKey, int]:
    result: dict[SignedBasisKey, int] = {}
    for row in bundle.basis_rows:
        key = SignedBasisKey(
            str(row["model_id"]),
            str(row["model_revision"]),
            int(row["layer"]),
            int(row["neuron_index"]),
            cast(Literal["+", "-"], row["polarity"]),
        )
        result[key] = int(row["signed_basis_index"])
    if len(result) != len(bundle.basis_rows):
        raise ValueError("generation signed-basis universe is not unique")
    return result


def _mapped_occurrence(
    row: Mapping[str, Any], metadata: Mapping[str, Any], basis_index: int
) -> dict[str, Any]:
    support = [bool(value) for value in row["input_attribution_support"]]
    raw_input = [
        0.0 if value is None else float(value)
        for value in row["input_attribution_profile"]
    ]
    paper_input = [
        0.0 if value is None else float(value)
        for value in row["paper_normalized_input_attribution_profile"]
    ]
    return {
        "target_id": str(metadata["case_id"]),
        "response_id": str(metadata["response_id"]),
        "family_id": str(metadata["base_question_id"]),
        "partition": str(metadata["family_partition"]),
        "basis_index": basis_index,
        "input_values": raw_input,
        "input_support": support,
        "paper_input_values": paper_input,
        "raw_candidate_values": [
            float(value) for value in row["raw_candidate_contribution"]
        ],
        "paper_candidate_values": [
            float(value)
            for value in row["paper_normalized_candidate_contribution"]
        ],
        "occurrence_count": int(row["occurrence_count"]),
    }


def extract_evaluation_occurrences(
    bundle: HybridInputBundle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reopen held-out unions and map them into the frozen generation universe."""

    basis_map = _basis_map(bundle)
    target_by_case = {str(row["case_id"]): dict(row) for row in bundle.target_rows}
    occurrences = [
        _mapped_occurrence(row, target_by_case[str(row["case_id"])], int(row["signed_basis_index"]))
        for row in bundle.profile_rows
    ]
    source = bundle.source_bundle
    union_root = Path(str(source.manifest["inputs"]["candidate_union_root"]))
    unions = _locate_unions(union_root)
    identity = {(row["model_id"], row["model_revision"]) for row in bundle.basis_rows}
    if len(identity) != 1:
        raise ValueError("generation basis universe mixes model identities")
    model_id, model_revision = next(iter(identity))
    heldout_only = 0
    heldout_mapped = 0
    for raw_target in sorted(source.target_rows, key=lambda row: str(row["case_id"])):
        if raw_target["family_partition"] == "generation":
            continue
        target = dict(raw_target)
        target["_model_id"] = model_id
        target["_model_revision"] = model_revision
        artifact_id = str(target["candidate_union_artifact_id"])
        if artifact_id not in unions:
            raise ValueError(f"candidate-union artifact is missing: {artifact_id}")
        rows, metadata = extract_hybrid_target(
            load_candidate_union_artifact(unions[artifact_id]), target=target
        )
        target_by_case[str(metadata["case_id"])] = metadata
        for row in rows:
            index = basis_map.get(row["basis"])
            if index is None:
                heldout_only += 1
                continue
            occurrences.append(_mapped_occurrence(row, metadata, index))
            heldout_mapped += 1
    expected_cases = {str(row["case_id"]) for row in source.target_rows}
    if set(target_by_case) != expected_cases:
        raise ValueError("evaluation target extraction is incomplete")
    occurrences.sort(key=lambda row: (row["target_id"], row["basis_index"]))
    return occurrences, [target_by_case[key] for key in sorted(target_by_case)], {
        "mapped_occurrence_count": len(occurrences),
        "heldout_mapped_occurrence_count": heldout_mapped,
        "heldout_only_basis_occurrence_count": heldout_only,
        "generation_basis_count": len(basis_map),
    }


def load_frozen_assignments(
    fit_root: Path,
    fit_manifest: Mapping[str, Any],
    *,
    basis_count: int,
) -> dict[str, NDArray[np.int64]]:
    """Load exactly the two protocol-frozen medoid assignment blocks."""

    table = pq.read_table(fit_root / "assignments.parquet")
    if not table.schema.equals(ASSIGNMENT_SCHEMA, check_metadata=False):
        raise ValueError("hybrid assignment schema drift")
    rows = table.to_pylist()
    output: dict[str, NDArray[np.int64]] = {}
    for role, spec in STATE_SPECS.items():
        fits = [
            fit
            for fit in fit_manifest["fits"]
            if fit["representation"] == spec["representation"]
            and fit["affinity_mode"] == spec["affinity_mode"]
            and fit["n_clusters"] == spec["n_clusters"]
        ]
        if (
            len(fits) != 1
            or fits[0].get("status") != "valid"
            or fits[0].get("medoid_seed") != spec["seed"]
        ):
            raise ValueError(f"frozen {role} fit identity drift")
        selected = [
            row
            for row in rows
            if row["representation"] == spec["representation"]
            and row["affinity_mode"] == spec["affinity_mode"]
            and row["n_clusters"] == spec["n_clusters"]
            and row["seed"] == spec["seed"]
        ]
        selected.sort(key=lambda row: int(row["signed_basis_index"]))
        if (
            len(selected) != basis_count
            or [int(row["signed_basis_index"]) for row in selected]
            != list(range(basis_count))
            or any(not row["is_medoid"] for row in selected)
        ):
            raise ValueError(f"frozen {role} assignment block is incomplete")
        labels = np.asarray(
            [-1 if row["cluster_id"] is None else int(row["cluster_id"]) for row in selected],
            dtype=np.int64,
        )
        if np.any(labels >= int(spec["n_clusters"])):
            raise ValueError(f"frozen {role} cluster ID is out of range")
        output[role] = labels
    return output


def _unit_cosine(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    support: NDArray[np.bool_] | None = None,
) -> float | None:
    if support is not None:
        left = left[support]
        right = right[support]
    if not len(left):
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def _target_vectors(
    rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    *,
    role: str,
    view: str,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    spec = STATE_SPECS[role]
    if view == "input":
        field = "paper_input_values" if role == "alternative" else "input_values"
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        support = np.asarray([row["input_support"] for row in rows], dtype=np.bool_)
        return values, support
    raw = np.asarray([row["raw_candidate_values"] for row in rows], dtype=np.float32)
    paper = np.asarray([row["paper_candidate_values"] for row in rows], dtype=np.float32)
    representation: Representation = (
        "contrast" if view == "contrast" else cast(Representation, spec["representation_key"])
    )
    values = candidate_view(
        raw,
        model_top5_indices=target["model_top5_indices"],
        observed_candidate_index=int(target["observed_candidate_index"]),
        representation=representation,
        paper_normalized=paper,
    ).astype(np.float64)
    return values, np.ones(values.shape, dtype=np.bool_)


def _target_effects(
    rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    assignments: Mapping[str, NDArray[np.int64]],
    *,
    view: str,
) -> dict[str, dict[str, Any] | None]:
    """Score one fixed valid pair pool shared by both frozen states."""

    ordered = sorted(rows, key=lambda row: int(row["basis_index"]))
    indices = np.asarray([row["basis_index"] for row in ordered], dtype=np.int64)
    vectors = {
        role: _target_vectors(ordered, target, role=role, view=view)
        for role in STATE_SPECS
    }
    pools: dict[str, list[tuple[int, int, float]]] = {role: [] for role in STATE_SPECS}
    for left, right in itertools.combinations(range(len(ordered)), 2):
        basis_left, basis_right = int(indices[left]), int(indices[right])
        if any(
            assignments[role][basis_left] < 0 or assignments[role][basis_right] < 0
            for role in STATE_SPECS
        ):
            continue
        scores: dict[str, float] = {}
        valid = True
        for role, (values, support) in vectors.items():
            shared = support[left] & support[right]
            score = _unit_cosine(values[left], values[right], shared)
            if score is None:
                valid = False
                break
            scores[role] = score
        if not valid:
            continue
        for role in STATE_SPECS:
            pools[role].append((basis_left, basis_right, scores[role]))
    output: dict[str, dict[str, Any] | None] = {}
    for role, pool in pools.items():
        same = [
            score
            for left, right, score in pool
            if assignments[role][left] == assignments[role][right]
        ]
        different = [
            score
            for left, right, score in pool
            if assignments[role][left] != assignments[role][right]
        ]
        output[role] = (
            None
            if not same or not different
            else {
                "effect": float(np.mean(same) - np.mean(different)),
                "same_pair_count": len(same),
                "different_pair_count": len(different),
                "common_pair_pool_count": len(pool),
            }
        )
    return output


def _aggregate_effects(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce equally target -> response -> family -> partition."""

    response_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        response_values[(str(row["family_id"]), str(row["response_id"]))].append(
            float(row["effect"])
        )
    family_values: dict[str, list[float]] = defaultdict(list)
    for (family, _), values in response_values.items():
        family_values[family].append(float(np.mean(values)))
    effects = {
        family: float(np.mean(values)) for family, values in sorted(family_values.items())
    }
    return {
        "effect": None if not effects else float(np.mean(list(effects.values()))),
        "per_family_effect": effects,
        "positive_family_count": sum(value > 0 for value in effects.values()),
        "scoreable_family_count": len(effects),
        "scoreable_response_count": len(response_values),
        "scoreable_target_count": len(records),
    }


def _bootstrap(
    effects: Mapping[str, float], *, protocol_sha256: str, partition: str, role: str, view: str
) -> dict[str, Any]:
    material = (
        protocol_sha256.encode()
        + b"\0"
        + partition.encode()
        + b"\0"
        + role.encode()
        + b"\0"
        + view.encode()
        + b"\0hybrid-coherence-bootstrap-v1"
    )
    digest = hashlib.sha256(material).digest()
    seed = int.from_bytes(digest[:8], "big")
    ordered = np.asarray([effects[key] for key in sorted(effects)], dtype=np.float64)
    if len(ordered) != 8:
        return {
            "available": False,
            "seed": seed,
            "replicates": BOOTSTRAP_REPLICATES,
            "ci_95_lower": None,
            "ci_95_upper": None,
        }
    draws = np.random.default_rng(seed).choice(
        ordered, size=(BOOTSTRAP_REPLICATES, len(ordered)), replace=True
    ).mean(axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5], method="linear")
    return {
        "available": True,
        "seed": seed,
        "replicates": BOOTSTRAP_REPLICATES,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def evaluate_coherence(
    occurrences: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, NDArray[np.int64]],
    *,
    protocol_sha256: str,
) -> dict[str, Any]:
    targets_by_id = {str(row["case_id"]): row for row in targets}
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in occurrences:
        by_target[str(row["target_id"])].append(row)
    raw: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    coverage: list[dict[str, Any]] = []
    target_statistics: list[dict[str, Any]] = []
    for target_id in sorted(by_target):
        target = targets_by_id[target_id]
        for view in VIEWS:
            effects = _target_effects(by_target[target_id], target, assignments, view=view)
            for role, result in effects.items():
                coverage.append(
                    {
                        "target_id": target_id,
                        "state_role": role,
                        "view": view,
                        "scoreable": result is not None,
                    }
                )
                if result is not None:
                    record = {
                        **result,
                        "partition": target["family_partition"],
                        "state_role": role,
                        "view": view,
                        "target_id": target_id,
                        "response_id": target["response_id"],
                        "family_id": target["base_question_id"],
                    }
                    raw[(str(target["family_partition"]), role, view)].append(record)
                    target_statistics.append(record)
    result: dict[str, Any] = {}
    for partition in PARTITIONS:
        result[partition] = {}
        for role in STATE_SPECS:
            result[partition][role] = {}
            for view in VIEWS:
                summary = _aggregate_effects(raw[(partition, role, view)])
                summary["bootstrap"] = _bootstrap(
                    summary["per_family_effect"],
                    protocol_sha256=protocol_sha256,
                    partition=partition,
                    role=role,
                    view=view,
                )
                result[partition][role][view] = summary
    return {
        "partitions": result,
        "target_coverage": coverage,
        "target_statistics": target_statistics,
    }


def _nonempty_support(
    row: Mapping[str, Any], target: Mapping[str, Any], *, role: str, evidence: str
) -> bool:
    values, support = _target_vectors([row], target, role=role, view=evidence)
    return _unit_cosine(values[0], values[0], support[0]) is not None


def build_witness_inventory(
    occurrences: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, NDArray[np.int64]],
    *,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Materialize the exact evidence inventory behind every readiness count."""

    target_by_id = {str(row["case_id"]): row for row in targets}
    states: dict[str, Any] = {}
    for role, spec in STATE_SPECS.items():
        clusters: list[dict[str, Any]] = []
        for cluster_id in range(64):
            cluster_ready = True
            partition_payload: dict[str, Any] = {}
            for partition, (target_minimum, family_minimum) in READINESS_THRESHOLDS.items():
                supported = [
                    row
                    for row in occurrences
                    if row["partition"] == partition
                    and assignments[role][int(row["basis_index"])] == cluster_id
                    and _nonempty_support(
                        row,
                        target_by_id[str(row["target_id"])],
                        role=role,
                        evidence="input",
                    )
                    and _nonempty_support(
                        row,
                        target_by_id[str(row["target_id"])],
                        role=role,
                        evidence="candidate",
                    )
                ]
                target_to_family = {
                    str(row["target_id"]): str(row["family_id"]) for row in supported
                }
                target_ids = sorted(target_to_family)
                response_ids = sorted({str(row["response_id"]) for row in supported})
                family_ids = sorted(set(target_to_family.values()))
                basis_indices = sorted({int(row["basis_index"]) for row in supported})
                ready = len(target_ids) >= target_minimum and len(family_ids) >= family_minimum
                cluster_ready = cluster_ready and ready
                frozen_ids: list[str] = []
                if ready:
                    family_order = sorted(
                        family_ids,
                        key=lambda family: hashlib.sha256(
                            f"{protocol_sha256}\0{role}\0{cluster_id}\0{partition}\0{family}".encode()
                        ).hexdigest(),
                    )[:family_minimum]
                    for family in family_order:
                        choices = [target for target in target_ids if target_to_family[target] == family]
                        frozen_ids.append(
                            min(
                                choices,
                                key=lambda target: hashlib.sha256(
                                    f"{protocol_sha256}\0{role}\0{cluster_id}\0{partition}\0{target}".encode()
                                ).hexdigest(),
                            )
                        )
                    remaining = [target for target in target_ids if target not in frozen_ids]
                    remaining.sort(
                        key=lambda target: hashlib.sha256(
                            f"{protocol_sha256}\0{role}\0{cluster_id}\0{partition}\0{target}".encode()
                        ).hexdigest()
                    )
                    frozen_ids.extend(remaining[: target_minimum - family_minimum])
                partition_payload[partition] = {
                    "target_ids": target_ids,
                    "response_ids": response_ids,
                    "family_ids": family_ids,
                    "basis_indices": basis_indices,
                    "target_count": len(target_ids),
                    "response_count": len(response_ids),
                    "family_count": len(family_ids),
                    "basis_count": len(basis_indices),
                    "required_target_count": target_minimum,
                    "required_family_count": family_minimum,
                    "ready": ready,
                    "frozen_target_ids": frozen_ids,
                    "frozen_target_selection_hashes": [
                        hashlib.sha256(
                            f"{protocol_sha256}\0{role}\0{cluster_id}\0{partition}\0{target}".encode()
                        ).hexdigest()
                        for target in frozen_ids
                    ],
                    "frozen_target_ids_sha256": canonical_sha256(frozen_ids),
                }
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "ready": cluster_ready,
                    "joint_witnesses": partition_payload,
                }
            )
        ready_count = sum(cluster["ready"] for cluster in clusters)
        states[role] = {
            "state": dict(spec),
            "cluster_count": 64,
            "ready_cluster_count": ready_count,
            "ready_cluster_fraction": ready_count / 64,
            "required_ready_cluster_count": 52,
            "passed": ready_count >= 52,
            "clusters": clusters,
        }
    return {"schema_version": WITNESS_SCHEMA, "states": states}


def structural_guardrails(fit_manifest: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role, spec in STATE_SPECS.items():
        matches = [
            fit
            for fit in fit_manifest["fits"]
            if fit["representation"] == spec["representation"]
            and fit["affinity_mode"] == spec["affinity_mode"]
            and fit["n_clusters"] == spec["n_clusters"]
        ]
        if len(matches) != 1 or matches[0].get("status") != "valid":
            raise ValueError(f"frozen {role} structural fit is unavailable")
        fit = matches[0]
        observed = {
            "assignment_fraction": float(fit["assignment_fraction_by_seed"][str(spec["seed"])]),
            "maximum_cluster_fraction": float(fit["cluster_size_metrics"]["maximum_cluster_fraction"]),
            "mean_seed_ari": float(fit["mean_seed_ari"]),
            "minimum_seed_ari": float(fit["minimum_seed_ari"]),
            "modularity": float(fit["partition_metrics"]["modularity"]),
            "internal_affinity_enrichment": float(
                fit["partition_metrics"]["internal_affinity_enrichment"]
            ),
        }
        conditions = {
            "assignment_fraction": observed["assignment_fraction"] >= 0.95,
            "maximum_cluster_fraction": observed["maximum_cluster_fraction"] <= 0.15,
            "mean_seed_ari": observed["mean_seed_ari"] >= 0.72,
            "minimum_seed_ari": observed["minimum_seed_ari"] >= 0.70,
            "modularity": observed["modularity"] >= 0.20,
            "internal_affinity_enrichment": observed["internal_affinity_enrichment"] >= 1.25,
        }
        output[role] = {"observed": observed, "conditions": conditions, "passed": all(conditions.values())}
    return output


def _jackknife_evidence(
    bundle: HybridInputBundle, *, representation: Representation, omitted_family: str
) -> HybridEvidence:
    blocks = [
        block
        for block in blocks_from_bundle(bundle, representation=representation)
        if block.base_question_id != omitted_family
    ]
    pair = accumulate_fused_evidence(blocks, basis_count=len(bundle.basis_rows))
    eligible = (
        (np.asarray(pair.overlap_count.diagonal()).ravel() >= MIN_BASIS_TARGETS)
        & (np.asarray(pair.response_overlap_count.diagonal()).ravel() >= MIN_BASIS_RESPONSES)
        & (np.asarray(pair.family_overlap_count.diagonal()).ravel() >= MIN_BASIS_FAMILIES)
    )
    similarity = mean_similarity_matrix(
        pair,
        min_pair_target_overlap=MIN_PAIR_TARGETS,
        min_pair_response_overlap=MIN_PAIR_RESPONSES,
        min_pair_family_overlap=MIN_PAIR_FAMILIES,
        eligible_mask=eligible,
    )
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    similarity.sort_indices()
    return HybridEvidence(representation, pair, eligible, similarity)


def generation_family_jackknife(
    bundle: HybridInputBundle,
    assignments: Mapping[str, NDArray[np.int64]],
) -> dict[str, Any]:
    families = sorted({str(row["base_question_id"]) for row in bundle.target_rows})
    if len(families) != 18:
        raise ValueError("generation jackknife requires exactly 18 families")
    output: dict[str, Any] = {}
    for role, spec in STATE_SPECS.items():
        replicates: list[dict[str, Any]] = []
        for family in families:
            try:
                evidence = _jackknife_evidence(
                    bundle,
                    representation=cast(Representation, spec["representation_key"]),
                    omitted_family=family,
                )
                fit = _fit_one(evidence, affinity_mode="full_positive", n_clusters=64)
                labels = fit.seeds[fit.medoid_seed].result.labels
                common = (labels >= 0) & (assignments[role] >= 0)
                full_assigned_count = int((assignments[role] >= 0).sum())
                common_fraction = int(common.sum()) / full_assigned_count
                if common_fraction < 0.80:
                    raise ValueError("jackknife common-assignment coverage is below 80%")
                ari = float(adjusted_rand_score(labels[common], assignments[role][common]))
                if not np.isfinite(ari) or fit.medoid_seed not in (17, 29, 43):
                    raise ValueError("jackknife fit produced a nonfinite or invalid result")
                replicates.append(
                    {
                        "omitted_family_id": family,
                        "valid": True,
                        "ari": ari,
                        "medoid_seed": fit.medoid_seed,
                        "active_basis_count": int((labels >= 0).sum()),
                        "common_assigned_basis_count": int(common.sum()),
                        "full_assigned_basis_count": full_assigned_count,
                        "common_assigned_fraction": common_fraction,
                    }
                )
            except (
                ArithmeticError,
                ArpackNoConvergence,
                KeyError,
                RuntimeError,
                ValueError,
            ) as error:
                replicates.append(
                    {
                        "omitted_family_id": family,
                        "valid": False,
                        "ari": None,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
        values = [float(row["ari"]) for row in replicates if row["valid"]]
        median_ari = None if not values else float(median(values))
        p10_ari = None if not values else float(np.percentile(values, 10, method="linear"))
        passed = (
            len(values) == 18
            and cast(float, median_ari) >= 0.60
            and cast(float, p10_ari) >= 0.45
        )
        output[role] = {
            "replicates": replicates,
            "valid_replicate_count": len(values),
            "median_ari": median_ari,
            "p10_ari": p10_ari,
            "passed": passed,
        }
    return output


def authorization_report(
    *,
    structural: Mapping[str, Any],
    jackknife: Mapping[str, Any],
    witness: Mapping[str, Any],
    coherence: Mapping[str, Any],
) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for role in STATE_SPECS:
        conditions: dict[str, bool] = {
            "structural_guardrails": bool(structural[role]["passed"]),
            "generation_family_jackknife": bool(jackknife[role]["passed"]),
            "witness_readiness": bool(witness["states"][role]["passed"]),
        }
        for partition in HELDOUT_PARTITIONS:
            for view in ("input", "candidate"):
                report = coherence["partitions"][partition][role][view]
                effects = report["per_family_effect"]
                conditions[f"{partition}_{view}_all_families_scoreable"] = len(effects) == 8
                conditions[f"{partition}_{view}_positive"] = (
                    report["effect"] is not None and report["effect"] > 0
                )
                lower = report["bootstrap"]["ci_95_lower"]
                conditions[f"{partition}_{view}_bootstrap_lower_positive"] = (
                    lower is not None and lower > 0
                )
                conditions[f"{partition}_{view}_seven_family_effects_positive"] = sum(
                    float(value) > 0 for value in effects.values()
                ) >= 7
        authorized = all(conditions.values())
        states[role] = {
            "state": dict(STATE_SPECS[role]),
            "conditions": conditions,
            "exploratory_labeling_authorized": authorized,
            "scientific_promotion_authorized": False,
            "witness_readiness": {
                key: witness["states"][role][key]
                for key in (
                    "cluster_count",
                    "ready_cluster_count",
                    "ready_cluster_fraction",
                    "required_ready_cluster_count",
                    "passed",
                )
            },
        }
    return {
        "states": states,
        "exploratory_labeling_authorized": any(
            report["exploratory_labeling_authorized"] for report in states.values()
        ),
        "scientific_promotion_authorized": False,
    }


def _validated_jackknife_summary(value: object) -> dict[str, Any]:
    """Recompute every jackknife aggregate and gate from its 18 records."""

    if not isinstance(value, Mapping) or set(value) != set(STATE_SPECS):
        raise ValueError("hybrid jackknife state inventory drift")
    states = cast(Mapping[str, Any], value)
    output: dict[str, Any] = {}
    for role in STATE_SPECS:
        report = states[role]
        if not isinstance(report, Mapping):
            raise TypeError("hybrid jackknife report is invalid")
        replicates = report.get("replicates")
        if not isinstance(replicates, list) or len(replicates) != 18:
            raise ValueError("hybrid jackknife requires 18 persisted replicates")
        families: set[str] = set()
        values: list[float] = []
        for record in replicates:
            if not isinstance(record, Mapping):
                raise TypeError("hybrid jackknife replicate is invalid")
            typed_record = cast(Mapping[str, Any], record)
            family = typed_record.get("omitted_family_id")
            if not isinstance(family, str) or not family or family in families:
                raise ValueError("hybrid jackknife family identity is invalid")
            families.add(family)
            if typed_record.get("valid") is True:
                ari = float(typed_record["ari"])
                fraction = float(typed_record["common_assigned_fraction"])
                if (
                    not np.isfinite(ari)
                    or not np.isfinite(fraction)
                    or fraction < 0.80
                    or typed_record.get("medoid_seed") not in (17, 29, 43)
                ):
                    raise ValueError("hybrid jackknife valid replicate contract drift")
                values.append(ari)
            elif typed_record.get("valid") is not False:
                raise ValueError("hybrid jackknife replicate status drift")
        median_ari = None if not values else float(median(values))
        p10_ari = None if not values else float(
            np.percentile(values, 10, method="linear")
        )
        recomputed = {
            "replicates": replicates,
            "valid_replicate_count": len(values),
            "median_ari": median_ari,
            "p10_ari": p10_ari,
            "passed": (
                len(values) == 18
                and cast(float, median_ari) >= 0.60
                and cast(float, p10_ari) >= 0.45
            ),
        }
        if dict(report) != recomputed:
            raise ValueError("hybrid jackknife aggregate drift")
        output[role] = recomputed
    return output


def _payload_binding(targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = [
        {
            "case_id": row["case_id"],
            "partition": row["family_partition"],
            "candidate_union_artifact_id": row["candidate_union_artifact_id"],
            "candidate_union_payload_sha256": row["candidate_union_payload_sha256"],
            "candidate_union_topology_sha256": row["candidate_union_topology_sha256"],
            "refinement_artifact_id": row["refinement_artifact_id"],
            "refinement_payload_sha256": row["refinement_payload_sha256"],
        }
        for row in targets
    ]
    return {
        "target_payload_set_sha256": canonical_sha256(
            sorted(records, key=lambda row: str(row["case_id"]))
        ),
        "case_set_sha256": canonical_sha256(sorted(str(row["case_id"]) for row in targets)),
        "partition_case_set_sha256": {
            partition: canonical_sha256(
                sorted(
                    str(row["case_id"])
                    for row in targets
                    if row["family_partition"] == partition
                )
            )
            for partition in PARTITIONS
        },
    }


def _validate_targets_against_source(
    targets: Sequence[Mapping[str, Any]], bundle: HybridInputBundle
) -> None:
    source = {str(row["case_id"]): row for row in bundle.source_bundle.target_rows}
    observed = {str(row.get("case_id")): row for row in targets}
    if len(observed) != len(targets) or set(observed) != set(source):
        raise ValueError("hybrid evaluation target identities drift from source")
    fields = (
        "response_id",
        "base_question_id",
        "family_partition",
        "candidate_count",
        "candidate_union_artifact_id",
        "candidate_union_payload_sha256",
        "candidate_union_topology_sha256",
    )
    for case_id, row in observed.items():
        if any(row.get(field) != source[case_id].get(field) for field in fields):
            raise ValueError("hybrid evaluation target/source provenance drift")


def _validate_occurrences(
    rows: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    basis_count: int,
) -> None:
    target_by_id = {str(row["case_id"]): row for row in targets}
    seen: set[tuple[str, int]] = set()
    covered: set[str] = set()
    for row in rows:
        target_id = str(row["target_id"])
        target = target_by_id.get(target_id)
        basis_index = int(row["basis_index"])
        key = (target_id, basis_index)
        if target is None or key in seen or not 0 <= basis_index < basis_count:
            raise ValueError("hybrid evaluation occurrence identity is invalid")
        seen.add(key)
        covered.add(target_id)
        if (
            row["partition"] != target["family_partition"]
            or row["family_id"] != target["base_question_id"]
            or row["response_id"] != target["response_id"]
        ):
            raise ValueError("hybrid evaluation occurrence hierarchy drift")
        input_values = np.asarray(row["input_values"], dtype=np.float64)
        paper_values = np.asarray(row["paper_input_values"], dtype=np.float64)
        support = np.asarray(row["input_support"], dtype=np.bool_)
        raw_candidate = np.asarray(row["raw_candidate_values"], dtype=np.float64)
        paper_candidate = np.asarray(row["paper_candidate_values"], dtype=np.float64)
        if (
            input_values.ndim != 1
            or not len(input_values)
            or paper_values.shape != input_values.shape
            or support.shape != input_values.shape
            or raw_candidate.shape != (int(target["candidate_count"]),)
            or paper_candidate.shape != raw_candidate.shape
            or not np.isfinite(input_values).all()
            or not np.isfinite(paper_values).all()
            or not np.isfinite(raw_candidate).all()
            or not np.isfinite(paper_candidate).all()
            or int(row["occurrence_count"]) <= 0
        ):
            raise ValueError("hybrid evaluation occurrence profile is invalid")
    if not covered:
        raise ValueError("hybrid evaluation has no mapped occurrences")


def run_hybrid_candidate_labelability(
    *,
    input_root: Path,
    fit_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run and atomically publish the protocol-frozen evaluation."""

    if output_root.exists():
        raise FileExistsError(f"hybrid labelability destination already exists: {output_root}")
    revision = collect_evaluation_revision(repo_root)
    protocol_sha256 = next(
        row["sha256"] for row in revision["files"] if row["path"] == PROTOCOL_PATH
    )
    bundle = load_hybrid_input_bundle(input_root)
    fit_manifest = load_hybrid_clustering_manifest(fit_root)
    fit_binding = fit_manifest["input_binding"]
    if fit_binding["manifest_sha256"] != bundle.manifest["manifest_sha256"]:
        raise ValueError("hybrid input and clustering fit bindings disagree")
    assignments = load_frozen_assignments(
        fit_root, fit_manifest, basis_count=len(bundle.basis_rows)
    )
    occurrences, targets, extraction = extract_evaluation_occurrences(bundle)
    _validate_occurrences(occurrences, targets, basis_count=len(bundle.basis_rows))
    coherence = evaluate_coherence(
        occurrences, targets, assignments, protocol_sha256=protocol_sha256
    )
    witness = build_witness_inventory(
        occurrences, targets, assignments, protocol_sha256=protocol_sha256
    )
    structural = structural_guardrails(fit_manifest)
    jackknife = generation_family_jackknife(bundle, assignments)
    authorization = authorization_report(
        structural=structural,
        jackknife=jackknife,
        witness=witness,
        coherence=coherence,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        pq.write_table(
            pa.Table.from_pylist(list(occurrences), schema=OCCURRENCE_SCHEMA),
            temporary / "occurrences.parquet",
            compression="zstd",
        )
        (temporary / "targets.json").write_text(
            json.dumps(
                {"schema_version": TARGET_SCHEMA, "targets": targets},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "witness-inventory.json").write_text(
            json.dumps(witness, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        files = [
            {"path": name, "sha256": file_sha256(temporary / name)}
            for name in ("occurrences.parquet", "targets.json", "witness-inventory.json")
        ]
        manifest: dict[str, Any] = {
            "schema_version": EVALUATION_SCHEMA,
            "purpose": "exploratory_label_free_hybrid_cluster_labelability",
            "source_input_binding": {
                "path": str(input_root.resolve()),
                "schema_version": bundle.manifest["schema_version"],
                "manifest_sha256": bundle.manifest["manifest_sha256"],
            },
            "source_fit_binding": {
                "path": str(fit_root.resolve()),
                "schema_version": fit_manifest["schema_version"],
                "manifest_sha256": fit_manifest["manifest_sha256"],
            },
            "code_revision": revision,
            "protocol": {"path": PROTOCOL_PATH, "sha256": protocol_sha256},
            "frozen_states": {role: dict(spec) for role, spec in STATE_SPECS.items()},
            "family_partitions": bundle.source_bundle.family_partitions,
            "target_payload_binding": _payload_binding(targets),
            "signed_basis_mapping": {
                "generation_basis_count": len(bundle.basis_rows),
                "generation_basis_index_sha256": canonical_sha256(list(bundle.basis_rows)),
                **extraction,
            },
            "coherence": coherence,
            "generation_family_jackknife": jackknife,
            "structural_guardrails": structural,
            "states": authorization["states"],
            "exploratory_labeling_authorized": authorization[
                "exploratory_labeling_authorized"
            ],
            "scientific_promotion_authorized": False,
            "confirmatory_holdout_opened": False,
            "firewall": {
                "generation_only_fit": True,
                "heldout_fit_influence": False,
                "labels_inspected": False,
                "task_outcomes_inspected": False,
                "descriptions_generated": False,
                "model_calls_made": False,
                "audit_witnesses_prompt_eligible": False,
            },
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_hybrid_candidate_labelability(
    root: Path, *, repo_root: Path | None = None
) -> Mapping[str, Any]:
    """Deep-load the immutable evaluation and fail closed on provenance drift."""

    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("hybrid labelability manifest must be an object")
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if manifest.get("schema_version") != EVALUATION_SCHEMA or recorded != canonical_sha256(core):
        raise ValueError("hybrid labelability manifest is invalid")
    if (
        manifest.get("scientific_promotion_authorized") is not False
        or manifest.get("confirmatory_holdout_opened") is not False
        or manifest.get("firewall")
        != {
            "generation_only_fit": True,
            "heldout_fit_influence": False,
            "labels_inspected": False,
            "task_outcomes_inspected": False,
            "descriptions_generated": False,
            "model_calls_made": False,
            "audit_witnesses_prompt_eligible": False,
        }
    ):
        raise ValueError("hybrid labelability scientific firewall drift")
    revision = manifest.get("code_revision")
    if not isinstance(revision, Mapping) or revision.get("tracked_worktree_clean") is not True:
        raise ValueError("hybrid labelability code revision is invalid")
    source_hashes = {
        row["path"]: row["sha256"] for row in revision.get("files", [])
    }
    if set(source_hashes) != set(EVALUATION_SOURCE_PATHS) or manifest.get("protocol") != {
        "path": PROTOCOL_PATH,
        "sha256": source_hashes.get(PROTOCOL_PATH),
    }:
        raise ValueError("hybrid labelability executable provenance drift")
    if repo_root is not None and collect_evaluation_revision(repo_root) != revision:
        raise ValueError("local evaluation source revision drift")
    input_binding = manifest.get("source_input_binding")
    fit_binding = manifest.get("source_fit_binding")
    if not isinstance(input_binding, Mapping) or not isinstance(fit_binding, Mapping):
        raise TypeError("hybrid labelability source bindings are absent")
    bundle = load_hybrid_input_bundle(Path(str(input_binding["path"])))
    fit_manifest = load_hybrid_clustering_manifest(Path(str(fit_binding["path"])))
    if input_binding != {
        "path": str(bundle.root),
        "schema_version": bundle.manifest["schema_version"],
        "manifest_sha256": bundle.manifest["manifest_sha256"],
    } or fit_binding != {
        "path": str(Path(str(fit_binding["path"])).resolve()),
        "schema_version": fit_manifest["schema_version"],
        "manifest_sha256": fit_manifest["manifest_sha256"],
    }:
        raise ValueError("hybrid labelability source manifest drift")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TypeError("hybrid labelability file inventory is invalid")
    by_name = {str(row.get("path")): row for row in records if isinstance(row, Mapping)}
    expected = {"occurrences.parquet", "targets.json", "witness-inventory.json"}
    if set(by_name) != expected:
        raise ValueError("hybrid labelability file inventory is incomplete")
    for name, row in by_name.items():
        if Path(name).name != name or file_sha256(root / name) != row.get("sha256"):
            raise ValueError(f"hybrid labelability file hash mismatch: {name}")
    table = pq.read_table(root / "occurrences.parquet")
    if not table.schema.equals(OCCURRENCE_SCHEMA, check_metadata=False):
        raise ValueError("hybrid labelability occurrence schema drift")
    targets_payload = json.loads((root / "targets.json").read_text(encoding="utf-8"))
    witness = json.loads((root / "witness-inventory.json").read_text(encoding="utf-8"))
    if targets_payload.get("schema_version") != TARGET_SCHEMA or witness.get("schema_version") != WITNESS_SCHEMA:
        raise ValueError("hybrid labelability sidecar schema drift")
    targets = targets_payload.get("targets")
    if not isinstance(targets, list):
        raise TypeError("hybrid labelability target inventory is invalid")
    _validate_targets_against_source(targets, bundle)
    occurrence_rows = table.to_pylist()
    _validate_occurrences(
        occurrence_rows, targets, basis_count=len(bundle.basis_rows)
    )
    if manifest.get("target_payload_binding") != _payload_binding(targets):
        raise ValueError("hybrid labelability target payload binding drift")
    if witness.get("states") is None:
        raise ValueError("hybrid labelability witness inventory is incomplete")
    # Recompute readiness and authorization-relevant witnesses from the hashed
    # occurrence rows, so rehashing a semantically altered sidecar still fails.
    assignments = load_frozen_assignments(
        Path(str(fit_binding["path"])), fit_manifest, basis_count=len(bundle.basis_rows)
    )
    recomputed_witness = build_witness_inventory(
        occurrence_rows,
        targets,
        assignments,
        protocol_sha256=str(manifest["protocol"]["sha256"]),
    )
    if recomputed_witness != witness:
        raise ValueError("hybrid labelability witness inventory numeric drift")
    protocol_sha256 = str(manifest["protocol"]["sha256"])
    recomputed_coherence = evaluate_coherence(
        occurrence_rows,
        targets,
        assignments,
        protocol_sha256=protocol_sha256,
    )
    if recomputed_coherence != manifest.get("coherence"):
        raise ValueError("hybrid labelability coherence numeric drift")
    recomputed_structural = structural_guardrails(fit_manifest)
    if recomputed_structural != manifest.get("structural_guardrails"):
        raise ValueError("hybrid labelability structural guardrail drift")
    recomputed_jackknife = _validated_jackknife_summary(
        manifest.get("generation_family_jackknife")
    )
    recomputed_authorization = authorization_report(
        structural=recomputed_structural,
        jackknife=recomputed_jackknife,
        witness=recomputed_witness,
        coherence=recomputed_coherence,
    )
    if (
        recomputed_authorization["states"] != manifest.get("states")
        or recomputed_authorization["exploratory_labeling_authorized"]
        is not manifest.get("exploratory_labeling_authorized")
        or manifest.get("scientific_promotion_authorized") is not False
    ):
        raise ValueError("hybrid labelability authorization numeric drift")
    if manifest.get("frozen_states") != {
        role: dict(spec) for role, spec in STATE_SPECS.items()
    }:
        raise ValueError("hybrid labelability frozen state drift")
    for role in STATE_SPECS:
        if manifest["states"][role]["witness_readiness"] != {
            key: witness["states"][role][key]
            for key in (
                "cluster_count",
                "ready_cluster_count",
                "ready_cluster_fraction",
                "required_ready_cluster_count",
                "passed",
            )
        }:
            raise ValueError("hybrid labelability readiness summary drift")
        if manifest["states"][role].get("scientific_promotion_authorized") is not False:
            raise ValueError("hybrid labelability state promotion status drift")
    return manifest
