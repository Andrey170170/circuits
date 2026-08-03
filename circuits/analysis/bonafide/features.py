"""Missing-aware signed-basis input-profile features for discovery fitting."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.identity import SignedBasisKey
from circuits.analysis.bonafide.multiplex import TargetSlice
from circuits.analysis.bonafide.partition import (
    AnalysisTarget,
    hierarchical_fit_weights,
)

FEATURE_SCHEMA = "adag.bonafide.atlas-features.v1"
SMOKE_CLUSTER_SCHEMA = "adag.bonafide.smoke-cluster-state.v1"
DEFAULT_NORM_EPSILON = 1e-12


@dataclass(frozen=True)
class BasisProfileObservation:
    basis: SignedBasisKey
    trace_unit_id: str
    source_artifact_id: str
    base_question_id: str
    response_id: str
    target_response_position: int
    profile: tuple[float | None, ...]
    support: tuple[bool, ...]
    signed_attribution: float
    absolute_attribution_mass: float
    occurrence_count: int
    mean_activation: float
    fit_weight: float


def build_profile_observations(
    slices: Iterable[TargetSlice],
    *,
    fit_target_by_trace: Mapping[str, AnalysisTarget],
) -> tuple[BasisProfileObservation, ...]:
    """Build discovery-only observations without aligning unrelated token positions."""

    ordered_slices = tuple(
        sorted(
            slices,
            key=lambda item: (
                item.response_id,
                item.target_response_position,
                item.trace_unit_id,
            ),
        )
    )
    trace_ids = {target_slice.trace_unit_id for target_slice in ordered_slices}
    if trace_ids != set(fit_target_by_trace):
        missing = trace_ids - set(fit_target_by_trace)
        extra = set(fit_target_by_trace) - trace_ids
        raise ValueError(
            "target/slice mapping mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    targets = [fit_target_by_trace[trace_id] for trace_id in sorted(trace_ids)]
    weights = hierarchical_fit_weights(targets)

    observations: list[BasisProfileObservation] = []
    for target_slice in ordered_slices:
        target = fit_target_by_trace[target_slice.trace_unit_id]
        if target.response_id != target_slice.response_id:
            raise ValueError("analysis target/target slice response_id mismatch")
        if target.response_position != target_slice.target_response_position:
            raise ValueError("analysis target/target slice response position mismatch")
        observations.extend(
            BasisProfileObservation(
                basis=summary.basis,
                trace_unit_id=target_slice.trace_unit_id,
                source_artifact_id=target.source_artifact_id,
                base_question_id=target.base_question_id,
                response_id=target.response_id,
                target_response_position=target.response_position,
                profile=summary.attribution_map,
                support=summary.attribution_support,
                signed_attribution=summary.signed_attribution,
                absolute_attribution_mass=summary.absolute_attribution_mass,
                occurrence_count=summary.occurrence_count,
                mean_activation=summary.mean_activation,
                fit_weight=weights[target.source_artifact_id],
            )
            for summary in target_slice.basis_summaries
        )
    return tuple(
        sorted(
            observations,
            key=lambda item: (
                item.basis,
                item.response_id,
                item.target_response_position,
                item.trace_unit_id,
            ),
        )
    )


def _directional_similarity(
    left: BasisProfileObservation,
    right: BasisProfileObservation,
    *,
    epsilon: float,
) -> float | None:
    if left.trace_unit_id != right.trace_unit_id:
        raise ValueError("profile vectors are comparable only within one target slice")
    if len(left.profile) != len(right.profile):
        raise ValueError("within-target profile widths disagree")
    paired = [
        (left_value, right_value)
        for left_value, right_value in zip(
            left.profile,
            right.profile,
            strict=True,
        )
        if left_value is not None and right_value is not None
    ]
    if not paired:
        return None
    dot = sum(left_value * right_value for left_value, right_value in paired)
    left_norm = math.sqrt(sum(left_value**2 for left_value, _ in paired))
    right_norm = math.sqrt(sum(right_value**2 for _, right_value in paired))
    if left_norm <= epsilon or right_norm <= epsilon:
        return None
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def pairwise_profile_similarity(
    left_basis: SignedBasisKey,
    right_basis: SignedBasisKey,
    observations: Sequence[BasisProfileObservation],
    *,
    epsilon: float = DEFAULT_NORM_EPSILON,
) -> tuple[float | None, tuple[str, ...]]:
    """Aggregate cosine similarity only over co-supported target slices."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    by_basis: dict[SignedBasisKey, dict[str, BasisProfileObservation]] = defaultdict(
        dict
    )
    for observation in observations:
        prior = by_basis[observation.basis].get(observation.trace_unit_id)
        if prior is not None:
            raise ValueError("duplicate basis observation in one target slice")
        by_basis[observation.basis][observation.trace_unit_id] = observation

    common_traces = sorted(by_basis[left_basis].keys() & by_basis[right_basis].keys())
    weighted_sum = 0.0
    supported_weight = 0.0
    witnesses: list[str] = []
    for trace_unit_id in common_traces:
        left = by_basis[left_basis][trace_unit_id]
        right = by_basis[right_basis][trace_unit_id]
        similarity = _directional_similarity(left, right, epsilon=epsilon)
        if similarity is None:
            continue
        if not math.isclose(left.fit_weight, right.fit_weight, abs_tol=1e-15):
            raise ValueError("one target slice has inconsistent fit weights")
        weighted_sum += similarity * left.fit_weight
        supported_weight += left.fit_weight
        witnesses.append(trace_unit_id)
    if not witnesses:
        return None, ()
    return weighted_sum / supported_weight, tuple(witnesses)


def cluster_fully_supported_profiles(
    observations: Sequence[BasisProfileObservation],
    *,
    expected_trace_ids: Sequence[str],
    n_clusters: int,
    epsilon: float = DEFAULT_NORM_EPSILON,
) -> dict[str, Any]:
    """Run a deterministic, description-free vertical-slice clustering smoke.

    This intentionally restricts the smoke to bases with a valid profile in
    every selected target. It does not fill missing similarities with zeros.
    """

    from sklearn.cluster import AgglomerativeClustering

    trace_ids = tuple(sorted(set(expected_trace_ids)))
    if not trace_ids:
        raise ValueError("clustering smoke requires target traces")
    usable = fully_supported_profile_bases(
        observations,
        expected_trace_ids=trace_ids,
        epsilon=epsilon,
    )
    if isinstance(n_clusters, bool) or not 1 <= n_clusters <= len(usable):
        raise ValueError(
            f"n_clusters must be in [1, {len(usable)}] for fully supported bases"
        )

    similarities: list[list[float]] = []
    witnesses_by_pair: dict[str, list[str]] = {}
    for left_index, left_basis in enumerate(usable):
        row: list[float] = []
        for right_index, right_basis in enumerate(usable):
            similarity, witnesses = pairwise_profile_similarity(
                left_basis,
                right_basis,
                observations,
                epsilon=epsilon,
            )
            if similarity is None or witnesses != trace_ids:
                raise ValueError(
                    "fully supported clustering encountered missing pairwise support"
                )
            row.append(similarity)
            if left_index <= right_index:
                witnesses_by_pair[f"{left_index}:{right_index}"] = list(witnesses)
        similarities.append(row)

    distances = [
        [max(0.0, min(2.0, 1.0 - similarity)) for similarity in row]
        for row in similarities
    ]
    for index in range(len(distances)):
        distances[index][index] = 0.0
    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distances)

    state: dict[str, Any] = {
        "schema_version": SMOKE_CLUSTER_SCHEMA,
        "purpose": "vertical_slice_engineering_smoke",
        "descriptions_generated": False,
        "scientific_cluster_state": False,
        "feature_contract": {
            "schema_version": FEATURE_SCHEMA,
            "profile_reducer": "signed_sum_within_target_basis",
            "directional_norm": "l2_on_pairwise_support_intersection",
            "epsilon": epsilon,
            "missing_support": "missing_not_zero",
            "weighting": "equal_family_then_response_then_target",
        },
        "trace_unit_ids": list(trace_ids),
        "n_clusters": n_clusters,
        "eligible_signed_basis_count": len(usable),
        "signed_bases": [
            {
                "signed_basis_index": index,
                "basis_key": basis.to_record(),
                "cluster_id": int(labels[index]),
            }
            for index, basis in enumerate(usable)
        ],
        "similarity_matrix": similarities,
        "pair_support": witnesses_by_pair,
    }
    state["cluster_state_sha256"] = canonical_sha256(state)
    return state


def fully_supported_profile_bases(
    observations: Sequence[BasisProfileObservation],
    *,
    expected_trace_ids: Sequence[str],
    epsilon: float = DEFAULT_NORM_EPSILON,
) -> tuple[SignedBasisKey, ...]:
    """Return bases with a valid nonzero profile in every selected target."""

    trace_ids = tuple(sorted(set(expected_trace_ids)))
    if not trace_ids:
        raise ValueError("profile support selection requires target traces")
    support_counts = Counter(observation.basis for observation in observations)
    candidate_bases = sorted(
        basis for basis, count in support_counts.items() if count == len(trace_ids)
    )
    observed_traces_by_basis: dict[SignedBasisKey, set[str]] = defaultdict(set)
    for observation in observations:
        observed_traces_by_basis[observation.basis].add(observation.trace_unit_id)
    candidate_bases = [
        basis
        for basis in candidate_bases
        if observed_traces_by_basis[basis] == set(trace_ids)
    ]

    usable: list[SignedBasisKey] = []
    for basis in candidate_bases:
        self_similarity, witnesses = pairwise_profile_similarity(
            basis,
            basis,
            observations,
            epsilon=epsilon,
        )
        if self_similarity is not None and witnesses == trace_ids:
            usable.append(basis)
    return tuple(usable)
