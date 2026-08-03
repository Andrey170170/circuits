"""Exact missing-aware sparse clustering primitives for the dense BonaFide atlas.

The production path computes input-profile similarity only within one target
trace. Unsupported profile coordinates are excluded from both dot products and
pair-specific norms; they are never filled with scientific zeros. Target-level
similarities are then accumulated with the frozen hierarchical fit weights.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

PAIR_EVIDENCE_SCHEMA = "adag.bonafide.pair-evidence.v1"
SPARSE_CLUSTER_SCHEMA = "adag.bonafide.sparse-cluster-state.v1"
DEFAULT_EPSILON = 1e-12

WeightingMode = Literal["hierarchical", "unweighted"]
KnnSymmetrization = Literal["union_max", "mutual_min"]


@dataclass(frozen=True)
class TargetProfileBlock:
    """Signed-basis profiles that coexist in one independently traced target."""

    trace_unit_id: str
    response_id: str
    base_question_id: str
    basis_indices: NDArray[np.int64]
    values: NDArray[np.float32]
    support: NDArray[np.bool_]
    fit_weight: float

    def validate(self) -> None:
        if not self.trace_unit_id:
            raise ValueError("target profile block requires trace_unit_id")
        if not self.response_id or not self.base_question_id:
            raise ValueError("target profile block requires response/family identity")
        if self.basis_indices.ndim != 1:
            raise ValueError("basis_indices must be one-dimensional")
        if self.values.ndim != 2 or self.support.ndim != 2:
            raise ValueError("profile values/support must be two-dimensional")
        if self.values.shape != self.support.shape:
            raise ValueError("profile values/support shapes disagree")
        if self.values.shape[0] != len(self.basis_indices):
            raise ValueError("profile row count does not match basis indices")
        if len({int(value) for value in self.basis_indices}) != len(self.basis_indices):
            raise ValueError("target profile block contains duplicate basis indices")
        if np.any(self.basis_indices < 0):
            raise ValueError("basis indices must be nonnegative")
        if not math.isfinite(self.fit_weight) or self.fit_weight <= 0:
            raise ValueError("fit_weight must be finite and positive")
        if not np.all(np.isfinite(self.values[self.support])):
            raise ValueError("supported profile values must be finite")


@dataclass(frozen=True)
class PairEvidence:
    """Upper-triangular accumulated pair evidence, including the diagonal."""

    weighted_similarity_sum: csr_matrix
    support_weight_sum: csr_matrix
    overlap_count: csr_matrix
    response_overlap_count: csr_matrix
    family_overlap_count: csr_matrix
    target_count: int
    weighting: WeightingMode
    epsilon: float

    def validate(self) -> None:
        matrices = (
            self.weighted_similarity_sum,
            self.support_weight_sum,
            self.overlap_count,
            self.response_overlap_count,
            self.family_overlap_count,
        )
        shape = matrices[0].shape
        if shape[0] != shape[1] or any(matrix.shape != shape for matrix in matrices):
            raise ValueError("pair-evidence matrices must be equally sized and square")
        if self.target_count < 1:
            raise ValueError("pair evidence requires at least one target")
        if self.weighting not in ("hierarchical", "unweighted"):
            raise ValueError("pair-evidence weighting is invalid")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("pair-evidence epsilon must be finite and positive")
        if self.support_weight_sum.nnz != self.overlap_count.nnz:
            raise ValueError("pair support-weight/count sparsity disagrees")
        if np.any(self.support_weight_sum.data <= 0):
            raise ValueError("pair support weights must be positive")
        if np.any(self.overlap_count.data <= 0):
            raise ValueError("pair overlap counts must be positive")
        if np.any(self.response_overlap_count.data <= 0):
            raise ValueError("pair response-overlap counts must be positive")
        if np.any(self.family_overlap_count.data <= 0):
            raise ValueError("pair family-overlap counts must be positive")
        response_minus_target = (
            self.response_overlap_count - self.overlap_count
        ).tocsr()
        if response_minus_target.nnz and np.any(response_minus_target.data > 0):
            raise ValueError("pair response overlap exceeds target overlap")
        family_minus_response = (
            self.family_overlap_count - self.response_overlap_count
        ).tocsr()
        if family_minus_response.nnz and np.any(family_minus_response.data > 0):
            raise ValueError("pair family overlap exceeds response overlap")

    @property
    def basis_count(self) -> int:
        return int(self.support_weight_sum.shape[0])

    @property
    def valid_profile_target_counts(self) -> NDArray[np.int64]:
        return np.asarray(self.overlap_count.diagonal(), dtype=np.int64)


@dataclass(frozen=True)
class SparseSpectralResult:
    labels: NDArray[np.int64]
    active_mask: NDArray[np.bool_]
    eigenvalues: NDArray[np.float64]
    connected_component_count: int
    cluster_sizes: dict[int, int]


class PairEvidenceAccumulator:
    """Incrementally collect target-local pair evidence before CSR reduction."""

    def __init__(
        self,
        *,
        basis_count: int,
        weighting: WeightingMode = "hierarchical",
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        if basis_count < 1:
            raise ValueError("basis_count must be positive")
        if weighting not in ("hierarchical", "unweighted"):
            raise ValueError("unsupported weighting mode")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.basis_count = basis_count
        self.weighting = weighting
        self.epsilon = epsilon
        self._row_parts: list[NDArray[np.int32]] = []
        self._column_parts: list[NDArray[np.int32]] = []
        self._weighted_similarity_parts: list[NDArray[np.float64]] = []
        self._weight_parts: list[NDArray[np.float64]] = []
        self._overlap_parts: list[NDArray[np.int32]] = []
        self._response_pair_parts: dict[
            str, list[tuple[NDArray[np.int32], NDArray[np.int32]]]
        ] = {}
        self._family_pair_parts: dict[
            str, list[tuple[NDArray[np.int32], NDArray[np.int32]]]
        ] = {}
        self._seen_trace_ids: set[str] = set()
        self._target_count = 0

    def add(self, block: TargetProfileBlock) -> None:
        block.validate()
        if block.trace_unit_id in self._seen_trace_ids:
            raise ValueError("duplicate target profile block")
        self._seen_trace_ids.add(block.trace_unit_id)
        self._target_count += 1
        if (
            len(block.basis_indices)
            and int(block.basis_indices.max()) >= self.basis_count
        ):
            raise ValueError("target profile block basis index is out of range")
        similarities, valid = target_pairwise_profile_similarity(
            block.values,
            block.support,
            epsilon=self.epsilon,
        )
        local_rows, local_columns = np.nonzero(np.triu(valid))
        if not len(local_rows):
            return
        global_rows = block.basis_indices[local_rows].astype(np.int32, copy=False)
        global_columns = block.basis_indices[local_columns].astype(
            np.int32,
            copy=False,
        )
        weight = block.fit_weight if self.weighting == "hierarchical" else 1.0
        pair_similarities = similarities[local_rows, local_columns]
        self._row_parts.append(global_rows)
        self._column_parts.append(global_columns)
        self._weighted_similarity_parts.append(pair_similarities * weight)
        self._weight_parts.append(np.full(len(local_rows), weight, dtype=np.float64))
        self._overlap_parts.append(np.ones(len(local_rows), dtype=np.int32))
        self._response_pair_parts.setdefault(block.response_id, []).append(
            (global_rows, global_columns)
        )
        self._family_pair_parts.setdefault(block.base_question_id, []).append(
            (global_rows, global_columns)
        )

    def _distinct_group_overlap(
        self,
        parts_by_group: dict[str, list[tuple[NDArray[np.int32], NDArray[np.int32]]]],
    ) -> csr_matrix:
        shape = (self.basis_count, self.basis_count)
        total = csr_matrix(shape, dtype=np.int32)
        for group in sorted(parts_by_group):
            group_parts = parts_by_group[group]
            rows = np.concatenate([part[0] for part in group_parts])
            columns = np.concatenate([part[1] for part in group_parts])
            present = coo_matrix(
                (np.ones(len(rows), dtype=np.int8), (rows, columns)),
                shape=shape,
            ).tocsr()
            present.sum_duplicates()
            present.data.fill(1)
            total += present.astype(np.int32)
        total.sum_duplicates()
        total.sort_indices()
        return total

    def finalize(self) -> PairEvidence:
        if self._target_count < 1:
            raise ValueError("pair evidence requires target blocks")
        if not self._row_parts:
            raise ValueError("no target block contained a valid profile pair")
        rows = np.concatenate(self._row_parts)
        columns = np.concatenate(self._column_parts)
        shape = (self.basis_count, self.basis_count)
        weighted_similarity_sum = coo_matrix(
            (
                np.concatenate(self._weighted_similarity_parts),
                (rows, columns),
            ),
            shape=shape,
            dtype=np.float64,
        ).tocsr()
        support_weight_sum = coo_matrix(
            (np.concatenate(self._weight_parts), (rows, columns)),
            shape=shape,
            dtype=np.float64,
        ).tocsr()
        overlap_count = coo_matrix(
            (np.concatenate(self._overlap_parts), (rows, columns)),
            shape=shape,
            dtype=np.int32,
        ).tocsr()
        response_overlap_count = self._distinct_group_overlap(self._response_pair_parts)
        family_overlap_count = self._distinct_group_overlap(self._family_pair_parts)
        for matrix in (
            weighted_similarity_sum,
            support_weight_sum,
            overlap_count,
        ):
            matrix.sum_duplicates()
            matrix.sort_indices()
        evidence = PairEvidence(
            weighted_similarity_sum=weighted_similarity_sum,
            support_weight_sum=support_weight_sum,
            overlap_count=overlap_count,
            response_overlap_count=response_overlap_count,
            family_overlap_count=family_overlap_count,
            target_count=self._target_count,
            weighting=self.weighting,
            epsilon=self.epsilon,
        )
        evidence.validate()
        return evidence


def target_pairwise_profile_similarity(
    values: NDArray[np.float32],
    support: NDArray[np.bool_],
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Compute exact pair-intersection cosine similarity inside one target."""

    if values.ndim != 2 or support.ndim != 2 or values.shape != support.shape:
        raise ValueError("values/support must be equally shaped matrices")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not np.all(np.isfinite(values[support])):
        raise ValueError("supported profile values must be finite")

    masked_values = np.where(support, values, 0.0).astype(np.float64, copy=False)
    support_float = support.astype(np.float64, copy=False)
    dot = masked_values @ masked_values.T
    restricted_squared_norms = (masked_values * masked_values) @ support_float.T
    denominator = np.sqrt(
        np.maximum(restricted_squared_norms * restricted_squared_norms.T, 0.0)
    )
    valid = denominator > epsilon
    similarities = np.zeros_like(dot)
    np.divide(dot, denominator, out=similarities, where=valid)
    np.clip(similarities, -1.0, 1.0, out=similarities)
    return similarities, valid


def accumulate_pair_evidence(
    blocks: Iterable[TargetProfileBlock],
    *,
    basis_count: int,
    weighting: WeightingMode = "hierarchical",
    epsilon: float = DEFAULT_EPSILON,
) -> PairEvidence:
    """Accumulate exact upper-triangular evidence over target blocks."""

    accumulator = PairEvidenceAccumulator(
        basis_count=basis_count,
        weighting=weighting,
        epsilon=epsilon,
    )
    for block in blocks:
        accumulator.add(block)
    return accumulator.finalize()


def mean_similarity_matrix(
    evidence: PairEvidence,
    *,
    min_pair_target_overlap: int,
    min_pair_response_overlap: int = 1,
    min_pair_family_overlap: int = 1,
    eligible_mask: NDArray[np.bool_] | None = None,
) -> csr_matrix:
    """Return a symmetric sparse mean-similarity matrix."""

    evidence.validate()
    if min_pair_target_overlap < 1:
        raise ValueError("min_pair_target_overlap must be positive")
    if min_pair_response_overlap < 1 or min_pair_family_overlap < 1:
        raise ValueError("pair response/family overlap thresholds must be positive")
    if eligible_mask is None:
        eligible_mask = np.ones(evidence.basis_count, dtype=np.bool_)
    if eligible_mask.shape != (evidence.basis_count,):
        raise ValueError("eligible_mask shape is invalid")

    inverse_weights = evidence.support_weight_sum.copy()
    inverse_weights.data = 1.0 / inverse_weights.data
    upper = evidence.weighted_similarity_sum.multiply(inverse_weights)
    overlap_mask = evidence.overlap_count.copy()
    overlap_mask.data = (overlap_mask.data >= min_pair_target_overlap).astype(np.int8)
    overlap_mask.eliminate_zeros()
    upper = upper.multiply(overlap_mask)
    response_mask = evidence.response_overlap_count.copy()
    response_mask.data = (response_mask.data >= min_pair_response_overlap).astype(
        np.int8
    )
    response_mask.eliminate_zeros()
    upper = upper.multiply(response_mask)
    family_mask = evidence.family_overlap_count.copy()
    family_mask.data = (family_mask.data >= min_pair_family_overlap).astype(np.int8)
    family_mask.eliminate_zeros()
    upper = upper.multiply(family_mask)
    eligibility = diags(eligible_mask.astype(np.int8), format="csr")
    upper = eligibility @ upper @ eligibility
    upper.eliminate_zeros()
    diagonal = diags(upper.diagonal(), format="csr")
    symmetric = (upper + upper.T - diagonal).tocsr()
    symmetric.sum_duplicates()
    symmetric.sort_indices()
    return symmetric


def knn_affinity(
    similarity: csr_matrix,
    *,
    neighbors: int,
    symmetrization: KnnSymmetrization = "union_max",
    minimum_affinity: float = 0.0,
) -> csr_matrix:
    """Create a deterministic positive kNN affinity graph."""

    if similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity matrix must be square")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if symmetrization not in ("union_max", "mutual_min"):
        raise ValueError("unsupported kNN symmetrization")
    if not math.isfinite(minimum_affinity) or minimum_affinity < 0:
        raise ValueError("minimum_affinity must be finite and nonnegative")

    positive = similarity.copy().tocsr()
    positive.setdiag(0.0)
    positive.data[positive.data <= minimum_affinity] = 0.0
    positive.eliminate_zeros()
    positive.sort_indices()
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row in range(positive.shape[0]):
        start, stop = positive.indptr[row : row + 2]
        row_columns = positive.indices[start:stop]
        row_values = positive.data[start:stop]
        if not len(row_values):
            continue
        order = np.lexsort((row_columns, -row_values))
        selected = order[:neighbors]
        rows.extend([row] * len(selected))
        columns.extend(int(value) for value in row_columns[selected])
        values.extend(float(value) for value in row_values[selected])
    directed = coo_matrix(
        (values, (rows, columns)),
        shape=positive.shape,
        dtype=np.float64,
    ).tocsr()
    if symmetrization == "union_max":
        affinity = directed.maximum(directed.T)
    else:
        affinity = directed.minimum(directed.T)
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    affinity.sort_indices()
    return affinity


def _canonicalize_cluster_labels(
    labels: NDArray[np.int64],
    basis_indices: NDArray[np.int64],
) -> NDArray[np.int64]:
    unique = sorted(
        (int(label) for label in np.unique(labels)),
        key=lambda label: int(basis_indices[labels == label].min()),
    )
    remap = {label: index for index, label in enumerate(unique)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int64)


def sparse_spectral_cluster(
    affinity: csr_matrix,
    *,
    n_clusters: int,
    random_seed: int,
    self_loop_weight: float = 1.0,
    eigen_tolerance: float = 1e-6,
) -> SparseSpectralResult:
    """Cluster the non-isolated portion of a sparse affinity graph."""

    if affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity matrix must be square")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least two")
    if not math.isfinite(self_loop_weight) or self_loop_weight < 0:
        raise ValueError("self_loop_weight must be finite and nonnegative")
    if not math.isfinite(eigen_tolerance) or eigen_tolerance <= 0:
        raise ValueError("eigen_tolerance must be finite and positive")
    if (affinity - affinity.T).nnz:
        raise ValueError("affinity matrix must be exactly symmetric")
    if np.any(affinity.data < 0):
        raise ValueError("affinity matrix must be nonnegative")

    degree_without_self = np.asarray(affinity.sum(axis=1)).ravel()
    active_mask = degree_without_self > 0
    active_indices = np.flatnonzero(active_mask).astype(np.int64)
    if len(active_indices) <= n_clusters:
        raise ValueError("n_clusters must be smaller than active basis count")
    active = affinity[active_indices][:, active_indices].astype(np.float64)
    component_count = int(
        connected_components(active, directed=False, return_labels=False)
    )
    if component_count > n_clusters:
        raise ValueError(
            "affinity graph has more connected components than requested clusters"
        )

    if self_loop_weight:
        active = active + diags(
            np.full(len(active_indices), self_loop_weight),
            format="csr",
        )
    degree = np.asarray(active.sum(axis=1)).ravel()
    inverse_sqrt_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inverse_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    normalized = (
        diags(inverse_sqrt_degree, format="csr")
        @ active
        @ diags(inverse_sqrt_degree, format="csr")
    ).tocsr()
    rng = np.random.default_rng(random_seed)
    eigenvalues, embedding = eigsh(
        normalized,
        k=n_clusters,
        which="LA",
        tol=eigen_tolerance,
        v0=rng.standard_normal(len(active_indices)),
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    embedding = np.asarray(embedding[:, order], dtype=np.float64)
    row_norms = np.linalg.norm(embedding, axis=1)
    nonzero_rows = row_norms > 0
    embedding[nonzero_rows] /= row_norms[nonzero_rows, None]
    raw_labels = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=20,
    ).fit_predict(embedding)
    active_labels = _canonicalize_cluster_labels(
        np.asarray(raw_labels, dtype=np.int64),
        active_indices,
    )
    labels = np.full(affinity.shape[0], -1, dtype=np.int64)
    labels[active_indices] = active_labels
    unique, counts = np.unique(active_labels, return_counts=True)
    return SparseSpectralResult(
        labels=labels,
        active_mask=active_mask,
        eigenvalues=eigenvalues,
        connected_component_count=component_count,
        cluster_sizes={
            int(label): int(count) for label, count in zip(unique, counts, strict=True)
        },
    )
