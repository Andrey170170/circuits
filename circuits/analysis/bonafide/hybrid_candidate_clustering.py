"""Paper-style target-local multiview clustering over candidate-union traces."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence

from circuits.analysis.bonafide.candidate_clustering import (
    CLUSTER_COUNTS,
    MIN_ASSIGNMENT_FRACTION,
    MIN_BASIS_FAMILIES,
    MIN_BASIS_RESPONSES,
    MIN_BASIS_TARGETS,
    MIN_PAIR_FAMILIES,
    MIN_PAIR_RESPONSES,
    MIN_PAIR_TARGETS,
    RANDOM_SEEDS,
    choose_medoid_seed,
)
from circuits.analysis.bonafide.clustering import (
    PairEvidence,
    SparseSpectralResult,
    knn_affinity,
    mean_similarity_matrix,
    sparse_spectral_cluster,
    target_pairwise_profile_similarity,
)
from circuits.analysis.bonafide.hybrid_candidate_inputs import HybridInputBundle

Representation = Literal["raw", "paper_normalized", "top5", "contrast"]
REPRESENTATION_IDS: Mapping[Representation, str] = {
    "raw": "raw_top5_plus_observed.v1",
    "paper_normalized": "paper_normalized_model_top5.v1",
    "top5": "raw_model_top5.v1",
    "contrast": "top5_minus_observed.v1",
}
FUSION_ID = "clamp_nonnegative_harmonic_inside_target_then_hierarchical_mean.v1"


@dataclass(frozen=True)
class HybridTargetBlock:
    case_id: str
    response_id: str
    base_question_id: str
    basis_indices: NDArray[np.int64]
    attr_values: NDArray[np.float32]
    attr_support: NDArray[np.bool_]
    candidate_values: NDArray[np.float32]
    fit_weight: float


@dataclass(frozen=True)
class HybridEvidence:
    representation: Representation
    pair_evidence: PairEvidence
    eligible_mask: NDArray[np.bool_]
    similarity: csr_matrix


@dataclass(frozen=True)
class HybridSeedFit:
    seed: int
    result: SparseSpectralResult
    assignment_fraction: float


@dataclass(frozen=True)
class HybridResolutionFit:
    representation: Representation
    affinity_mode: str
    n_clusters: int
    affinity: csr_matrix
    seeds: Mapping[int, HybridSeedFit]
    medoid_seed: int
    pairwise_seed_ari: Mapping[tuple[int, int], float]
    mean_seed_ari: float
    minimum_seed_ari: float


@dataclass(frozen=True)
class HybridInvalidFit:
    representation: Representation
    affinity_mode: str
    n_clusters: int
    error_type: str
    error_message: str


def candidate_view(
    raw: NDArray[np.float32],
    *,
    model_top5_indices: Sequence[int],
    observed_candidate_index: int,
    representation: Representation,
    paper_normalized: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Create one target-local candidate view without cross-target alignment."""

    if raw.ndim != 2 or raw.shape[1] not in {5, 6}:
        raise ValueError("raw candidate profiles must have width five or six")
    if not np.isfinite(raw).all():
        raise ValueError("raw candidate profiles must be finite")
    if not 0 <= observed_candidate_index < raw.shape[1]:
        raise ValueError("observed candidate index is outside the candidate axis")
    indices = np.asarray(model_top5_indices, dtype=np.int64)
    observed_only = {observed_candidate_index} if raw.shape[1] == 6 else set()
    expected_indices = set(range(raw.shape[1])) - observed_only
    if indices.shape != (5,) or {int(value) for value in indices} != expected_indices:
        raise ValueError("model-top-five indices do not describe the candidate axis")
    if representation == "raw":
        return raw.copy()
    if representation == "paper_normalized":
        if paper_normalized is None or paper_normalized.shape != raw.shape:
            raise ValueError(
                "paper-normalized profiles must match raw candidate profiles"
            )
        if not np.isfinite(paper_normalized).all():
            raise ValueError("paper-normalized profiles must be finite")
        return paper_normalized[:, indices].copy()
    top5 = raw[:, indices]
    if representation == "top5":
        return top5.copy()
    if representation == "contrast":
        return top5 - raw[:, observed_candidate_index, None]
    raise ValueError(f"unsupported candidate representation: {representation}")


def target_fused_similarity(
    attr_values: NDArray[np.float32],
    attr_support: NDArray[np.bool_],
    candidate_values: NDArray[np.float32],
    *,
    epsilon: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Clamp and harmonic-fuse the two cosine views inside one target.

    A finite zero cosine remains valid evidence.  A missing overlap or a zero
    pair-specific norm is invalid and must not contribute to recurrence counts.
    """

    if (
        attr_values.ndim != 2
        or attr_support.shape != attr_values.shape
        or candidate_values.ndim != 2
        or candidate_values.shape[0] != attr_values.shape[0]
        or not np.isfinite(attr_values).all()
        or not np.isfinite(candidate_values).all()
    ):
        raise ValueError("hybrid target-local profile shapes or values are invalid")
    attr, attr_valid = target_pairwise_profile_similarity(
        attr_values, attr_support, epsilon=epsilon
    )
    candidate_support = np.ones(candidate_values.shape, dtype=np.bool_)
    contribution, contribution_valid = target_pairwise_profile_similarity(
        candidate_values, candidate_support, epsilon=epsilon
    )
    valid = attr_valid & contribution_valid
    attr = np.maximum(attr, 0.0)
    contribution = np.maximum(contribution, 0.0)
    denominator = attr + contribution
    fused = np.zeros_like(attr)
    positive = valid & (denominator > epsilon)
    fused[positive] = (
        2.0 * attr[positive] * contribution[positive] / denominator[positive]
    )
    return fused, valid


def blocks_from_bundle(
    bundle: HybridInputBundle,
    *,
    representation: Representation,
    partition: str = "generation",
) -> tuple[HybridTargetBlock, ...]:
    if partition != "generation":
        raise ValueError("hybrid clustering is restricted to the generation partition")
    if any(row.get("family_partition") != "generation" for row in bundle.target_rows):
        raise ValueError("hybrid input contains a non-generation target")
    targets = {
        str(row["case_id"]): row
        for row in bundle.target_rows
        if row["family_partition"] == partition
    }
    rows_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in bundle.profile_rows:
        case_id = str(row["case_id"])
        if case_id in targets:
            rows_by_case.setdefault(case_id, []).append(row)
    if set(rows_by_case) != set(targets):
        raise ValueError("hybrid profiles do not cover the selected target partition")
    blocks: list[HybridTargetBlock] = []
    for case_id in sorted(targets):
        target = targets[case_id]
        ordered = sorted(
            rows_by_case[case_id], key=lambda row: int(row["signed_basis_index"])
        )
        indices = np.asarray(
            [row["signed_basis_index"] for row in ordered], dtype=np.int64
        )
        if len(np.unique(indices)) != len(indices):
            raise ValueError("hybrid target contains duplicate signed bases")
        attr_field = (
            "paper_normalized_input_attribution_profile"
            if representation == "paper_normalized"
            else "input_attribution_profile"
        )
        raw_attr = [list(row[attr_field]) for row in ordered]
        support = np.asarray(
            [row["input_attribution_support"] for row in ordered], dtype=np.bool_
        )
        attr = np.asarray(
            [
                [0.0 if value is None else float(value) for value in row]
                for row in raw_attr
            ],
            dtype=np.float32,
        )
        if attr.shape != support.shape:
            raise ValueError("hybrid attribution values/support disagree")
        raw_candidate = np.asarray(
            [row["raw_candidate_contribution"] for row in ordered], dtype=np.float32
        )
        paper_normalized_candidate = np.asarray(
            [row["paper_normalized_candidate_contribution"] for row in ordered],
            dtype=np.float32,
        )
        if raw_candidate.shape[1] != int(target["candidate_count"]):
            raise ValueError("hybrid candidate profile width disagrees with target")
        values = candidate_view(
            raw_candidate,
            model_top5_indices=target["model_top5_indices"],
            observed_candidate_index=int(target["observed_candidate_index"]),
            representation=representation,
            paper_normalized=paper_normalized_candidate,
        )
        blocks.append(
            HybridTargetBlock(
                case_id=case_id,
                response_id=str(target["response_id"]),
                base_question_id=str(target["base_question_id"]),
                basis_indices=indices,
                attr_values=attr,
                attr_support=support,
                candidate_values=values,
                fit_weight=float(target["partition_hierarchical_weight"]),
            )
        )
    return tuple(blocks)


def _distinct_counts(
    groups: Mapping[str, list[tuple[NDArray[np.int32], NDArray[np.int32]]]],
    *,
    basis_count: int,
) -> csr_matrix:
    result = csr_matrix((basis_count, basis_count), dtype=np.int32)
    for group in sorted(groups):
        rows = np.concatenate([part[0] for part in groups[group]])
        columns = np.concatenate([part[1] for part in groups[group]])
        present = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, columns)),
            shape=(basis_count, basis_count),
        ).tocsr()
        present.sum_duplicates()
        present.data.fill(1)
        result += present.astype(np.int32)
    result.sum_duplicates()
    result.sort_indices()
    return result


def accumulate_fused_evidence(
    blocks: Sequence[HybridTargetBlock], *, basis_count: int
) -> PairEvidence:
    """Fuse target-local views first, then aggregate with hierarchical weights."""

    if not blocks:
        raise ValueError("hybrid evidence requires target blocks")
    seen: set[str] = set()
    row_parts: list[NDArray[np.int32]] = []
    column_parts: list[NDArray[np.int32]] = []
    similarity_parts: list[NDArray[np.float64]] = []
    weight_parts: list[NDArray[np.float64]] = []
    response_parts: dict[str, list[tuple[NDArray[np.int32], NDArray[np.int32]]]] = {}
    family_parts: dict[str, list[tuple[NDArray[np.int32], NDArray[np.int32]]]] = {}
    for block in blocks:
        if block.case_id in seen:
            raise ValueError("duplicate hybrid target block")
        seen.add(block.case_id)
        if not math.isfinite(block.fit_weight) or block.fit_weight <= 0:
            raise ValueError("hybrid target weight must be finite and positive")
        fused, valid = target_fused_similarity(
            block.attr_values, block.attr_support, block.candidate_values
        )
        local_rows, local_columns = np.nonzero(np.triu(valid))
        if not len(local_rows):
            continue
        rows = block.basis_indices[local_rows].astype(np.int32, copy=False)
        columns = block.basis_indices[local_columns].astype(np.int32, copy=False)
        if (
            np.any(rows < 0)
            or np.any(columns < 0)
            or np.any(rows >= basis_count)
            or np.any(columns >= basis_count)
        ):
            raise ValueError("hybrid basis index is outside the canonical basis")
        row_parts.append(rows)
        column_parts.append(columns)
        similarity_parts.append(fused[local_rows, local_columns] * block.fit_weight)
        weight_parts.append(np.full(len(rows), block.fit_weight, dtype=np.float64))
        response_parts.setdefault(block.response_id, []).append((rows, columns))
        family_parts.setdefault(block.base_question_id, []).append((rows, columns))
    if not row_parts:
        raise ValueError("no target contained a valid fused profile pair")
    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    shape = (basis_count, basis_count)
    weighted = coo_matrix(
        (np.concatenate(similarity_parts), (rows, columns)), shape=shape
    ).tocsr()
    weights = coo_matrix(
        (np.concatenate(weight_parts), (rows, columns)), shape=shape
    ).tocsr()
    overlap = coo_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, columns)), shape=shape
    ).tocsr()
    evidence = PairEvidence(
        weighted_similarity_sum=weighted,
        support_weight_sum=weights,
        overlap_count=overlap,
        response_overlap_count=_distinct_counts(
            response_parts, basis_count=basis_count
        ),
        family_overlap_count=_distinct_counts(family_parts, basis_count=basis_count),
        target_count=len(blocks),
        weighting="hierarchical",
        epsilon=1e-12,
    )
    evidence.validate()
    return evidence


def build_hybrid_evidence(
    bundle: HybridInputBundle, *, representation: Representation
) -> HybridEvidence:
    evidence = accumulate_fused_evidence(
        blocks_from_bundle(bundle, representation=representation),
        basis_count=len(bundle.basis_rows),
    )
    eligible = (
        (np.asarray(evidence.overlap_count.diagonal()).ravel() >= MIN_BASIS_TARGETS)
        & (
            np.asarray(evidence.response_overlap_count.diagonal()).ravel()
            >= MIN_BASIS_RESPONSES
        )
        & (
            np.asarray(evidence.family_overlap_count.diagonal()).ravel()
            >= MIN_BASIS_FAMILIES
        )
    )
    similarity = mean_similarity_matrix(
        evidence,
        min_pair_target_overlap=MIN_PAIR_TARGETS,
        min_pair_response_overlap=MIN_PAIR_RESPONSES,
        min_pair_family_overlap=MIN_PAIR_FAMILIES,
        eligible_mask=eligible,
    )
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    similarity.sort_indices()
    return HybridEvidence(representation, evidence, eligible, similarity)


def _fit_one(
    evidence: HybridEvidence, *, affinity_mode: str, n_clusters: int
) -> HybridResolutionFit:
    if affinity_mode == "full_positive":
        affinity = evidence.similarity.copy()
        affinity.data[affinity.data <= 0] = 0.0
        affinity.eliminate_zeros()
    elif affinity_mode == "knn32":
        affinity = knn_affinity(
            evidence.similarity, neighbors=32, symmetrization="union_max"
        )
    else:
        raise ValueError(f"unsupported affinity mode: {affinity_mode}")
    seed_fits: dict[int, HybridSeedFit] = {}
    eligible_count = int(evidence.eligible_mask.sum())
    for seed in RANDOM_SEEDS:
        result = sparse_spectral_cluster(
            affinity,
            n_clusters=n_clusters,
            random_seed=seed,
            self_loop_weight=1.0,
            eigen_tolerance=1e-6,
        )
        fraction = (
            float(
                np.sum((result.labels >= 0) & evidence.eligible_mask) / eligible_count
            )
            if eligible_count
            else 0.0
        )
        if fraction < MIN_ASSIGNMENT_FRACTION:
            raise ValueError("hybrid clustering assignment coverage is insufficient")
        labels = np.unique(result.labels[result.labels >= 0])
        if not np.array_equal(labels, np.arange(n_clusters, dtype=np.int64)):
            raise ValueError("hybrid clustering did not realize the requested clusters")
        seed_fits[seed] = HybridSeedFit(seed, result, fraction)
    medoid, pairwise = choose_medoid_seed(
        {seed: fit.result.labels for seed, fit in seed_fits.items()}
    )
    scores = list(pairwise.values())
    return HybridResolutionFit(
        evidence.representation,
        affinity_mode,
        n_clusters,
        affinity,
        seed_fits,
        medoid,
        pairwise,
        mean(scores),
        min(scores),
    )


def fit_hybrid_grid(
    bundle: HybridInputBundle,
    *,
    representations: Sequence[Representation] = (
        "raw",
        "paper_normalized",
        "top5",
        "contrast",
    ),
    affinity_modes: Sequence[str] = ("full_positive", "knn32"),
    cluster_counts: Sequence[int] = CLUSTER_COUNTS,
) -> dict[tuple[str, str, int], HybridResolutionFit | HybridInvalidFit]:
    """Fit fresh partitions while persisting invalid scientific grid cells."""

    fits: dict[tuple[str, str, int], HybridResolutionFit | HybridInvalidFit] = {}
    for representation in representations:
        evidence = build_hybrid_evidence(bundle, representation=representation)
        for affinity_mode, n_clusters in itertools.product(
            affinity_modes, cluster_counts
        ):
            try:
                fit: HybridResolutionFit | HybridInvalidFit = _fit_one(
                    evidence, affinity_mode=affinity_mode, n_clusters=int(n_clusters)
                )
            except (ArithmeticError, ArpackNoConvergence, RuntimeError, ValueError) as error:
                fit = HybridInvalidFit(
                    representation,
                    affinity_mode,
                    int(n_clusters),
                    type(error).__name__,
                    str(error),
                )
            fits[(representation, affinity_mode, int(n_clusters))] = fit
    return fits
