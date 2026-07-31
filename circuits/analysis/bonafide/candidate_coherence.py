"""Pure held-out metrics for candidate-aware clustering.

The functions in this module deliberately know nothing about cluster fitting or
artifact I/O.  They consume reduced basis-target records and integer assignment
arrays, which keeps selection/audit evaluation downstream of a frozen fitted
state.  Records use the following normalized fields:

``family_id``, ``response_id``, ``target_id``, ``basis_index``
    Required identity fields.  A basis may occur at most once in a target.
``vector``
    A candidate-direction vector for centroid/coherence calculations.
``values`` and ``support``
    A width-one source-token profile and its missing-coordinate mask.
``partition``
    Required by every scientific helper and checked against its requested role.

Zero-norm candidate vectors and pairwise width profiles with zero restricted
norm are missing, never zero-filled.  Invalid identities, shapes, assignments,
or nonfinite supported values raise rather than silently changing coverage.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Record = Mapping[str, Any]

HELD_OUT_PARTITIONS = frozenset({"selection_scoring", "audit"})
FROZEN_HELD_OUT_FAMILY_COUNT = 8
FROZEN_BOOTSTRAP_REPLICATES = 10_000
FROZEN_PARTITION_FAMILY_COUNTS: Mapping[str, int] = {
    "generation": 18,
    "selection_scoring": 8,
    "audit": 8,
}
FROZEN_CANDIDATE_STATES = ("W", "C", "F", "S")
FROZEN_WIDTH_STATES = ("W", "F")
FROZEN_PROTOCOL_SHA256 = (
    "1e24d333fcf9b595bceea9ef42c12bbc0726af22c66ce2a161fd9a1ca45d7983"
)


@dataclass(frozen=True)
class CandidateCentroids:
    """Generation-only candidate centroids and explicit omission diagnostics."""

    values: np.ndarray
    available: np.ndarray
    cluster_reports: tuple[dict[str, Any], ...]
    dimension: int
    cluster_count: int
    family_count: int
    response_count: int
    target_count: int
    partition: str

    def __post_init__(self) -> None:
        if self.values.shape != (self.cluster_count, self.dimension):
            raise ValueError("centroid values have an invalid shape")
        if self.available.shape != (self.cluster_count,):
            raise ValueError("centroid availability has an invalid shape")
        if not np.all(np.isfinite(self.values[self.available])):
            raise ValueError("available centroids must be finite")
        if self.partition != "generation":
            raise ValueError("candidate centroids must be generation-only")


def _text(record: Record, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _basis_index(record: Record) -> int:
    value = record.get("basis_index")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("basis_index must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("basis_index must be nonnegative")
    return value


def _identity(record: Record) -> tuple[str, str, str, int]:
    return (
        _text(record, "family_id"),
        _text(record, "response_id"),
        _text(record, "target_id"),
        _basis_index(record),
    )


def _validate_hierarchy(records: Sequence[Record]) -> None:
    target_owner: dict[str, tuple[str, str]] = {}
    response_owner: dict[str, str] = {}
    seen: set[tuple[str, int]] = set()
    for record in records:
        family_id, response_id, target_id, basis_index = _identity(record)
        owner = target_owner.setdefault(target_id, (family_id, response_id))
        if owner != (family_id, response_id):
            raise ValueError(f"target {target_id!r} has inconsistent ownership")
        family = response_owner.setdefault(response_id, family_id)
        if family != family_id:
            raise ValueError(f"response {response_id!r} has inconsistent ownership")
        occurrence = (target_id, basis_index)
        if occurrence in seen:
            raise ValueError(f"duplicate reduced basis-target record: {occurrence!r}")
        seen.add(occurrence)


def _assignment_array(assignments: Sequence[int] | np.ndarray) -> np.ndarray:
    raw = np.asarray(assignments)
    if raw.ndim != 1:
        raise ValueError("assignments must be one-dimensional")
    if raw.dtype.kind not in "iu":
        if raw.dtype.kind != "f" or not np.all(np.isfinite(raw)):
            raise ValueError("assignments must contain finite integers")
        if not np.all(raw == np.floor(raw)):
            raise ValueError("assignments must contain integers")
    result = raw.astype(np.int64, copy=False)
    if np.any(result < -1):
        raise ValueError("only -1 may represent an unassigned basis")
    return result


def _assignment_maps(
    assignment_by_state: Mapping[str, Sequence[int] | np.ndarray],
    required_states: Sequence[str],
    max_basis_index: int,
) -> dict[str, np.ndarray]:
    if len(set(required_states)) != len(required_states) or not required_states:
        raise ValueError("required_states must be unique and non-empty")
    arrays: dict[str, np.ndarray] = {}
    for state in required_states:
        if state not in assignment_by_state:
            raise ValueError(f"missing assignments for state {state!r}")
        array = _assignment_array(assignment_by_state[state])
        if max_basis_index >= len(array):
            raise ValueError(f"state {state!r} does not cover every record basis")
        arrays[state] = array
    return arrays


def _validate_held_out_partition(partition: str) -> None:
    if partition not in HELD_OUT_PARTITIONS:
        raise ValueError(
            "held-out evaluation partition must be selection_scoring or audit"
        )


def _require_record_partition(records: Sequence[Record], expected: str) -> None:
    for record in records:
        observed = _text(record, "partition")
        if observed != expected:
            raise ValueError(
                f"partition firewall: expected {expected!r}, found {observed!r}"
            )


def _frozen_family_ids(
    values: Collection[str], *, expected_count: int, context: str
) -> frozenset[str]:
    materialized = tuple(values)
    if any(not isinstance(value, str) or not value for value in materialized):
        raise ValueError(f"{context} family IDs must be non-empty strings")
    result = frozenset(materialized)
    if len(result) != len(materialized):
        raise ValueError(f"{context} family IDs must be unique")
    if len(result) != expected_count:
        raise ValueError(
            f"{context} requires exactly {expected_count} frozen family IDs"
        )
    return result


def _boolean_mask(values: Sequence[bool], *, field: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    if raw.dtype.kind != "b":
        raise TypeError(f"{field} must contain actual boolean values")
    return raw.astype(np.bool_, copy=False)


def _candidate_vectors(records: Sequence[Record]) -> tuple[list[np.ndarray], int]:
    vectors: list[np.ndarray] = []
    dimension: int | None = None
    for record in records:
        vector = np.asarray(record.get("vector"), dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("candidate vector must be non-empty and one-dimensional")
        if not np.all(np.isfinite(vector)):
            raise ValueError("candidate vector must be finite")
        if dimension is None:
            dimension = int(vector.size)
        elif vector.size != dimension:
            raise ValueError("candidate vectors must share one dimension")
        vectors.append(vector)
    if dimension is None:
        raise ValueError("at least one candidate record is required")
    return vectors, dimension


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    result = vector / norm
    if not np.all(np.isfinite(result)):
        return None
    return result


def generation_candidate_centroids(
    records: Sequence[Record],
    assignments: Sequence[int] | np.ndarray,
    *,
    n_clusters: int | None = None,
) -> CandidateCentroids:
    """Build frozen generation centroids with equal mass at every hierarchy level.

    Each nonzero basis-target vector is normalized first.  Means are then taken
    over bases within target, targets within response, responses within family,
    and families.  Intermediate means are not normalized; only the final
    cluster vector is normalized, exactly preserving the frozen weighting.
    """

    materialized = tuple(records)
    _require_record_partition(materialized, "generation")
    _validate_hierarchy(materialized)
    vectors, dimension = _candidate_vectors(materialized)
    assignment = _assignment_array(assignments)
    max_basis = max(_basis_index(record) for record in materialized)
    if max_basis >= len(assignment):
        raise ValueError("assignments do not cover every record basis")
    inferred_clusters = int(assignment.max(initial=-1)) + 1
    if n_clusters is None:
        n_clusters = inferred_clusters
    if (
        isinstance(n_clusters, bool)
        or not isinstance(n_clusters, int)
        or n_clusters <= 0
    ):
        raise ValueError("n_clusters must be a positive integer")
    if inferred_clusters > n_clusters:
        raise ValueError("assignment contains a cluster outside n_clusters")

    families = sorted({_text(record, "family_id") for record in materialized})
    responses = sorted({_text(record, "response_id") for record in materialized})
    targets = sorted({_text(record, "target_id") for record in materialized})
    target_vectors: dict[int, dict[tuple[str, str, str], list[np.ndarray]]] = {
        cluster: defaultdict(list) for cluster in range(n_clusters)
    }
    assigned_occurrence_counts = [0] * n_clusters
    zero_norm_counts = [0] * n_clusters
    unassigned_count = 0

    for record, vector in zip(materialized, vectors, strict=True):
        basis_index = _basis_index(record)
        cluster = int(assignment[basis_index])
        if cluster < 0:
            unassigned_count += 1
            continue
        assigned_occurrence_counts[cluster] += 1
        unit = _unit(vector)
        if unit is None:
            zero_norm_counts[cluster] += 1
            continue
        key = (
            _text(record, "family_id"),
            _text(record, "response_id"),
            _text(record, "target_id"),
        )
        target_vectors[cluster][key].append(unit)

    values = np.zeros((n_clusters, dimension), dtype=np.float64)
    available = np.zeros(n_clusters, dtype=np.bool_)
    reports: list[dict[str, Any]] = []
    for cluster in range(n_clusters):
        target_means = {
            key: np.mean(np.stack(items), axis=0)
            for key, items in target_vectors[cluster].items()
        }
        by_response: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for (family_id, response_id, _), vector in target_means.items():
            by_response[(family_id, response_id)].append(vector)
        response_means = {
            key: np.mean(np.stack(items), axis=0) for key, items in by_response.items()
        }
        by_family: dict[str, list[np.ndarray]] = defaultdict(list)
        for (family_id, _), vector in response_means.items():
            by_family[family_id].append(vector)
        family_means = {
            family_id: np.mean(np.stack(items), axis=0)
            for family_id, items in by_family.items()
        }
        final_norm = 0.0
        if family_means:
            final = np.mean(np.stack(list(family_means.values())), axis=0)
            final_norm = float(np.linalg.norm(final))
            if math.isfinite(final_norm) and final_norm > 0.0:
                values[cluster] = final / final_norm
                available[cluster] = True
        supported_occurrences = sum(
            len(items) for items in target_vectors[cluster].values()
        )
        reports.append(
            {
                "cluster_id": cluster,
                "available": bool(available[cluster]),
                "assigned_basis_target_count": assigned_occurrence_counts[cluster],
                "supported_basis_target_count": supported_occurrences,
                "zero_norm_basis_target_count": zero_norm_counts[cluster],
                "nonempty_target_count": len(target_means),
                "empty_target_count": len(targets) - len(target_means),
                "nonempty_response_count": len(response_means),
                "empty_response_count": len(responses) - len(response_means),
                "nonempty_family_count": len(family_means),
                "empty_family_count": len(families) - len(family_means),
                "pre_normalization_l2_norm": final_norm,
            }
        )

    # The count is global rather than repeated in every cluster report.
    if unassigned_count + sum(assigned_occurrence_counts) != len(materialized):
        raise AssertionError("candidate centroid accounting mismatch")
    return CandidateCentroids(
        values=values,
        available=available,
        cluster_reports=tuple(reports),
        dimension=dimension,
        cluster_count=n_clusters,
        family_count=len(families),
        response_count=len(responses),
        target_count=len(targets),
        partition="generation",
    )


def _score_occurrence(
    vector: np.ndarray,
    basis_index: int,
    assignment: np.ndarray,
    centroids: CandidateCentroids,
) -> tuple[float, float] | None:
    cluster = int(assignment[basis_index])
    if cluster < 0:
        return None
    if cluster >= centroids.cluster_count:
        raise ValueError("assignment cluster is absent from its centroid state")
    unit = _unit(vector)
    if unit is None or not bool(centroids.available[cluster]):
        return None
    other = centroids.available.copy()
    other[cluster] = False
    if not bool(np.any(other)):
        return None
    own_cosine = float(np.dot(unit, centroids.values[cluster]))
    other_cosines = centroids.values[other] @ unit
    if not math.isfinite(own_cosine) or not np.all(np.isfinite(other_cosines)):
        raise ValueError("nonfinite candidate coherence score")
    margin = own_cosine - float(np.max(other_cosines))
    return own_cosine, margin


def _hierarchical_occurrence_weights(records: Sequence[Record]) -> np.ndarray:
    """Equal family/response/target/occurrence weights for coverage only."""

    by_family: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for index, record in enumerate(records):
        family_id, response_id, target_id, _ = _identity(record)
        by_family[family_id][response_id][target_id].append(index)
    weights = np.zeros(len(records), dtype=np.float64)
    family_mass = 1.0 / len(by_family)
    for responses in by_family.values():
        response_mass = family_mass / len(responses)
        for targets in responses.values():
            target_mass = response_mass / len(targets)
            for indices in targets.values():
                occurrence_mass = target_mass / len(indices)
                weights[indices] = occurrence_mass
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
        raise AssertionError("hierarchical occurrence weights do not sum to one")
    return weights


def _hierarchical_summary(
    records: Sequence[Record], values: Mapping[int, float]
) -> tuple[float | None, dict[str, float]]:
    by_target: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for index, value in values.items():
        record = records[index]
        by_target[
            (
                _text(record, "family_id"),
                _text(record, "response_id"),
                _text(record, "target_id"),
            )
        ].append(float(value))
    target_means = {key: float(np.mean(items)) for key, items in by_target.items()}
    by_response: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (family_id, response_id, _), value in target_means.items():
        by_response[(family_id, response_id)].append(value)
    response_means = {key: float(np.mean(items)) for key, items in by_response.items()}
    by_family: dict[str, list[float]] = defaultdict(list)
    for (family_id, _), value in response_means.items():
        by_family[family_id].append(value)
    family_means = {
        family_id: float(np.mean(items)) for family_id, items in by_family.items()
    }
    overall = float(np.mean(list(family_means.values()))) if family_means else None
    return overall, dict(sorted(family_means.items()))


def evaluate_candidate_coherence(
    records: Sequence[Record],
    assignment_by_state: Mapping[str, Sequence[int] | np.ndarray],
    centroid_by_state: Mapping[str, CandidateCentroids],
    *,
    partition: str,
    expected_family_ids: Collection[str],
    required_states: Sequence[str] = ("W", "C", "F", "S"),
) -> dict[str, Any]:
    """Evaluate four states on one fixed held-out occurrence intersection."""

    _validate_held_out_partition(partition)
    if tuple(required_states) != FROZEN_CANDIDATE_STATES:
        raise ValueError("candidate coherence states are frozen at W, C, F, S")
    frozen_family_ids = _frozen_family_ids(
        expected_family_ids,
        expected_count=FROZEN_HELD_OUT_FAMILY_COUNT,
        context=partition,
    )
    materialized = tuple(records)
    if not materialized:
        raise ValueError("candidate coherence requires occurrence records")
    _require_record_partition(materialized, partition)
    _validate_hierarchy(materialized)
    vectors, dimension = _candidate_vectors(materialized)
    families = sorted({_text(record, "family_id") for record in materialized})
    observed_family_ids = frozenset(families)
    if observed_family_ids != frozen_family_ids:
        raise ValueError(
            f"partition {partition!r} family IDs do not match the bound manifest; "
            f"missing={sorted(frozen_family_ids - observed_family_ids)}, "
            f"unexpected={sorted(observed_family_ids - frozen_family_ids)}"
        )
    max_basis = max(_basis_index(record) for record in materialized)
    assignments = _assignment_maps(assignment_by_state, required_states, max_basis)
    for state in required_states:
        if state not in centroid_by_state:
            raise ValueError(f"missing centroids for state {state!r}")
        if centroid_by_state[state].dimension != dimension:
            raise ValueError(f"candidate dimension mismatch for state {state!r}")
        if assignments[state].max(initial=-1) >= centroid_by_state[state].cluster_count:
            raise ValueError(
                f"assignment cluster is absent from state {state!r} centroids"
            )

    raw_scores: dict[str, dict[int, tuple[float, float]]] = {
        state: {} for state in required_states
    }
    scoreable_by_state: dict[str, int] = {}
    for state in required_states:
        for index, (record, vector) in enumerate(
            zip(materialized, vectors, strict=True)
        ):
            score = _score_occurrence(
                vector,
                _basis_index(record),
                assignments[state],
                centroid_by_state[state],
            )
            if score is not None:
                raw_scores[state][index] = score
        scoreable_by_state[state] = len(raw_scores[state])
    common = sorted(set.intersection(*(set(scores) for scores in raw_scores.values())))
    common_set = set(common)
    weights = _hierarchical_occurrence_weights(materialized)

    common_records = [materialized[index] for index in common]
    common_families = sorted({_text(record, "family_id") for record in common_records})
    common_responses = sorted(
        {_text(record, "response_id") for record in common_records}
    )
    common_targets = sorted({_text(record, "target_id") for record in common_records})
    missing_families = sorted(set(families) - set(common_families))
    state_reports: dict[str, Any] = {}
    per_family_margin: dict[str, dict[str, float]] = {}
    for state in required_states:
        own = {index: raw_scores[state][index][0] for index in common}
        margin = {index: raw_scores[state][index][1] for index in common}
        own_mean, own_family = _hierarchical_summary(materialized, own)
        margin_mean, margin_family = _hierarchical_summary(materialized, margin)
        if set(own_family) != set(margin_family):
            raise AssertionError("own-cosine and margin family coverage differ")
        per_family_margin[state] = margin_family
        state_reports[state] = {
            "scoreable_before_intersection_count": scoreable_by_state[state],
            "own_cluster_cosine": own_mean,
            "coherence_margin": margin_mean,
            "per_family_own_cluster_cosine": own_family,
            "per_family_coherence_margin": margin_family,
        }

    comparisons: dict[str, Any] = {}
    if "W" in required_states:
        baselines = ["W"] + (["S"] if "S" in required_states else [])
        for state in (
            candidate for candidate in ("C", "F") if candidate in required_states
        ):
            for baseline in baselines:
                effects = {
                    family_id: per_family_margin[state][family_id]
                    - per_family_margin[baseline][family_id]
                    for family_id in common_families
                }
                key = f"{state}_minus_{baseline}"
                comparisons[key] = {
                    "mean_effect": (
                        float(np.mean(list(effects.values()))) if effects else None
                    ),
                    "positive_family_count": sum(
                        value > 0.0 for value in effects.values()
                    ),
                    "family_count": len(effects),
                    "per_family_effect": dict(sorted(effects.items())),
                }

    coverage = {
        "basis_occurrence_count": len(common),
        "total_basis_occurrence_count": len(materialized),
        "basis_occurrence_fraction": len(common) / len(materialized),
        "hierarchical_weight_fraction": float(weights[common].sum()) if common else 0.0,
        "target_count": len(common_targets),
        "total_target_count": len(
            {_text(record, "target_id") for record in materialized}
        ),
        "response_count": len(common_responses),
        "total_response_count": len(
            {_text(record, "response_id") for record in materialized}
        ),
        "family_count": len(common_families),
        "total_family_count": len(families),
    }
    if not math.isclose(
        float(weights[list(common_set)].sum()) if common_set else 0.0,
        coverage["hierarchical_weight_fraction"],
        abs_tol=1e-15,
    ):
        raise AssertionError("common coverage accounting mismatch")
    return {
        "partition": partition,
        "required_states": list(required_states),
        "expected_family_count": FROZEN_HELD_OUT_FAMILY_COUNT,
        "expected_family_ids": sorted(frozen_family_ids),
        "all_families_scoreable": not missing_families,
        "missing_family_ids": missing_families,
        "coverage": coverage,
        "states": state_reports,
        "comparisons": comparisons,
    }


def candidate_coherence_bootstrap(
    family_effects: Mapping[str, float],
    *,
    expected_family_ids: Collection[str],
    protocol_sha256: str = FROZEN_PROTOCOL_SHA256,
    partition: str,
    replicates: int = FROZEN_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Apply the frozen paired family-block bootstrap and linear percentile CI."""

    if (
        len(protocol_sha256) != 64
        or protocol_sha256.lower() != protocol_sha256
        or any(character not in "0123456789abcdef" for character in protocol_sha256)
    ):
        raise ValueError("protocol_sha256 must be 64 lowercase hexadecimal characters")
    if protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise ValueError(
            "bootstrap protocol SHA-256 does not match the frozen protocol"
        )
    _validate_held_out_partition(partition)
    if replicates != FROZEN_BOOTSTRAP_REPLICATES:
        raise ValueError("candidate coherence bootstrap is frozen at 10,000 replicates")
    frozen_family_ids = _frozen_family_ids(
        expected_family_ids,
        expected_count=FROZEN_HELD_OUT_FAMILY_COUNT,
        context=partition,
    )
    if frozenset(family_effects) != frozen_family_ids:
        raise ValueError(
            "bootstrap effects do not match the bound partition family IDs"
        )
    family_ids = sorted(frozen_family_ids)
    effects = np.asarray(
        [family_effects[item] for item in family_ids], dtype=np.float64
    )
    if not np.all(np.isfinite(effects)):
        raise ValueError("bootstrap family effects must all be finite")
    digest = hashlib.sha256(
        protocol_sha256.encode("ascii")
        + b"\0"
        + partition.encode("utf-8")
        + b"\0candidate-coherence-bootstrap-v1"
    ).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        FROZEN_HELD_OUT_FAMILY_COUNT,
        size=(replicates, FROZEN_HELD_OUT_FAMILY_COUNT),
    )
    replicate_means = effects[draws].mean(axis=1)
    if replicate_means.shape != (replicates,) or not np.all(
        np.isfinite(replicate_means)
    ):
        raise AssertionError("invalid bootstrap replicate produced")
    lower, upper = np.quantile(replicate_means, [0.025, 0.975], method="linear")
    return {
        "family_ids": family_ids,
        "family_count": FROZEN_HELD_OUT_FAMILY_COUNT,
        "replicates": replicates,
        "seed": seed,
        "mean_effect": float(effects.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "method": "paired-family-block-linear-percentile",
    }


def missing_aware_cosine(
    left_values: Sequence[float],
    left_support: Sequence[bool],
    right_values: Sequence[float],
    right_support: Sequence[bool],
) -> float | None:
    """Cosine over jointly supported source coordinates, with no zero fill."""

    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    left_mask = _boolean_mask(left_support, field="left_support")
    right_mask = _boolean_mask(right_support, field="right_support")
    if left.ndim != 1 or left.size == 0 or right.shape != left.shape:
        raise ValueError("width profiles must be aligned non-empty vectors")
    if left_mask.shape != left.shape or right_mask.shape != left.shape:
        raise ValueError("width support masks must align with profile vectors")
    common = left_mask & right_mask
    if not bool(np.any(common)):
        return None
    left_restricted = left[common]
    right_restricted = right[common]
    if not np.all(np.isfinite(left_restricted)) or not np.all(
        np.isfinite(right_restricted)
    ):
        raise ValueError("supported width coordinates must be finite")
    left_norm = float(np.linalg.norm(left_restricted))
    right_norm = float(np.linalg.norm(right_restricted))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    result = float(np.dot(left_restricted, right_restricted) / (left_norm * right_norm))
    if not math.isfinite(result):
        return None
    return result


def evaluate_width_one_coherence(
    records: Sequence[Record],
    assignment_by_state: Mapping[str, Sequence[int] | np.ndarray],
    *,
    partition: str,
    expected_family_ids: Collection[str],
    required_states: Sequence[str] = ("W", "F"),
) -> dict[str, Any]:
    """Compare within-target input coherence on a common pair/target pool."""

    _validate_held_out_partition(partition)
    if tuple(required_states) != FROZEN_WIDTH_STATES:
        raise ValueError("width-one preservation states are frozen at W and F")
    frozen_family_ids = _frozen_family_ids(
        expected_family_ids,
        expected_count=FROZEN_HELD_OUT_FAMILY_COUNT,
        context=partition,
    )
    materialized = tuple(records)
    if not materialized:
        raise ValueError("width coherence requires profile records")
    _require_record_partition(materialized, partition)
    _validate_hierarchy(materialized)
    all_families = sorted({_text(record, "family_id") for record in materialized})
    observed_family_ids = frozenset(all_families)
    if observed_family_ids != frozen_family_ids:
        raise ValueError(
            f"partition {partition!r} family IDs do not match the bound manifest; "
            f"missing={sorted(frozen_family_ids - observed_family_ids)}, "
            f"unexpected={sorted(observed_family_ids - frozen_family_ids)}"
        )
    max_basis = max(_basis_index(record) for record in materialized)
    assignments = _assignment_maps(assignment_by_state, required_states, max_basis)
    profiles: list[tuple[np.ndarray, np.ndarray]] = []
    for record in materialized:
        values = np.asarray(record.get("values"), dtype=np.float64)
        support = _boolean_mask(record.get("support"), field="support")
        if values.ndim != 1 or values.size == 0 or support.shape != values.shape:
            raise ValueError("invalid width profile or support shape")
        if not np.all(np.isfinite(values[support])):
            raise ValueError("supported width coordinates must be finite")
        profiles.append((values, support))

    by_target: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(materialized):
        by_target[
            (
                _text(record, "family_id"),
                _text(record, "response_id"),
                _text(record, "target_id"),
            )
        ].append(index)

    valid_pair_count = 0
    common_pair_count = 0
    target_metrics: dict[str, dict[tuple[str, str, str], float]] = {
        state: {} for state in required_states
    }
    target_pair_counts: dict[tuple[str, str, str], int] = {}
    for target, indices in by_target.items():
        pairs: list[tuple[int, int, float]] = []
        for left_position, left_index in enumerate(indices):
            left_values, left_support = profiles[left_index]
            for right_index in indices[left_position + 1 :]:
                right_values, right_support = profiles[right_index]
                cosine = missing_aware_cosine(
                    left_values,
                    left_support,
                    right_values,
                    right_support,
                )
                if cosine is None:
                    continue
                valid_pair_count += 1
                left_basis = _basis_index(materialized[left_index])
                right_basis = _basis_index(materialized[right_index])
                if any(
                    assignments[state][left_basis] < 0
                    or assignments[state][right_basis] < 0
                    for state in required_states
                ):
                    continue
                pairs.append((left_basis, right_basis, cosine))
        common_pair_count += len(pairs)
        if not pairs:
            continue
        tentative: dict[str, float] = {}
        for state in required_states:
            same = [
                cosine
                for left_basis, right_basis, cosine in pairs
                if assignments[state][left_basis] == assignments[state][right_basis]
            ]
            different = [
                cosine
                for left_basis, right_basis, cosine in pairs
                if assignments[state][left_basis] != assignments[state][right_basis]
            ]
            if not same or not different:
                break
            tentative[state] = float(np.mean(same) - np.mean(different))
        if len(tentative) != len(required_states):
            continue
        target_pair_counts[target] = len(pairs)
        for state, value in tentative.items():
            target_metrics[state][target] = value

    scoreable_targets = sorted(target_pair_counts)
    scoreable_families = sorted({target[0] for target in scoreable_targets})
    missing_families = sorted(set(all_families) - set(scoreable_families))
    state_reports: dict[str, Any] = {}
    family_values: dict[str, dict[str, float]] = {}
    for state in required_states:
        # Each target already contains the frozen same-minus-different statistic;
        # reduce target -> response -> family without reweighting by pair count.
        pseudo_values: dict[int, float] = {}
        pseudo_records: list[Record] = []
        for index, target in enumerate(scoreable_targets):
            family_id, response_id, target_id = target
            pseudo_records.append(
                {
                    "family_id": family_id,
                    "response_id": response_id,
                    "target_id": target_id,
                    "basis_index": index,
                }
            )
            pseudo_values[index] = target_metrics[state][target]
        overall, per_family = _hierarchical_summary(pseudo_records, pseudo_values)
        family_values[state] = per_family
        state_reports[state] = {
            "coherence": overall,
            "per_family_coherence": per_family,
        }

    comparisons: dict[str, Any] = {}
    baseline = "W" if "W" in required_states else required_states[0]
    for state in required_states:
        if state == baseline:
            continue
        common_families = sorted(
            set(family_values[state]) & set(family_values[baseline])
        )
        effects = {
            family_id: family_values[state][family_id]
            - family_values[baseline][family_id]
            for family_id in common_families
        }
        comparisons[f"{state}_minus_{baseline}"] = {
            "mean_effect": float(np.mean(list(effects.values()))) if effects else None,
            "per_family_effect": effects,
        }
    scoreable_pair_count = sum(target_pair_counts.values())
    return {
        "partition": partition,
        "required_states": list(required_states),
        "expected_family_count": FROZEN_HELD_OUT_FAMILY_COUNT,
        "expected_family_ids": sorted(frozen_family_ids),
        "all_families_scoreable": not missing_families,
        "missing_family_ids": missing_families,
        "coverage": {
            "valid_pair_count": valid_pair_count,
            "common_assigned_pair_count": common_pair_count,
            "common_assigned_pair_fraction": (
                common_pair_count / valid_pair_count if valid_pair_count else 0.0
            ),
            "scoreable_target_count": len(scoreable_targets),
            "total_target_count": len(by_target),
            "scoreable_pair_count": scoreable_pair_count,
            "scoreable_pair_fraction": (
                scoreable_pair_count / valid_pair_count if valid_pair_count else 0.0
            ),
        },
        "states": state_reports,
        "comparisons": comparisons,
    }


DEFAULT_READINESS_THRESHOLDS: Mapping[str, tuple[int, int]] = {
    "generation": (8, 4),
    "selection_scoring": (4, 2),
    "audit": (4, 2),
}


def cluster_support_readiness(
    records: Sequence[Record],
    assignments: Sequence[int] | np.ndarray,
    *,
    expected_family_ids_by_partition: Mapping[str, Collection[str]],
    n_clusters: int | None = None,
    thresholds: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Count partition witnesses and identify labeling-ready clusters."""

    materialized = tuple(records)
    if not materialized:
        raise ValueError("readiness requires support records")
    _validate_hierarchy(materialized)
    assignment = _assignment_array(assignments)
    max_basis = max(_basis_index(record) for record in materialized)
    if max_basis >= len(assignment):
        raise ValueError("assignments do not cover every support record basis")
    inferred_clusters = int(assignment.max(initial=-1)) + 1
    if n_clusters is None:
        n_clusters = inferred_clusters
    if (
        isinstance(n_clusters, bool)
        or not isinstance(n_clusters, int)
        or n_clusters <= 0
    ):
        raise ValueError("n_clusters must be a positive integer")
    if inferred_clusters > n_clusters:
        raise ValueError("assignment contains a cluster outside n_clusters")
    if thresholds is None:
        thresholds = DEFAULT_READINESS_THRESHOLDS
    if dict(thresholds) != dict(DEFAULT_READINESS_THRESHOLDS):
        raise ValueError(
            "labeling-readiness thresholds are frozen and cannot be changed"
        )
    if set(expected_family_ids_by_partition) != set(FROZEN_PARTITION_FAMILY_COUNTS):
        raise ValueError(
            "bound family partitions must contain generation, selection_scoring, and audit"
        )
    frozen_families_by_partition = {
        partition: _frozen_family_ids(
            expected_family_ids_by_partition[partition],
            expected_count=expected_count,
            context=partition,
        )
        for partition, expected_count in FROZEN_PARTITION_FAMILY_COUNTS.items()
    }
    frozen_partitions = list(frozen_families_by_partition)
    for index, left in enumerate(frozen_partitions):
        for right in frozen_partitions[index + 1 :]:
            overlap = (
                frozen_families_by_partition[left] & frozen_families_by_partition[right]
            )
            if overlap:
                raise ValueError(
                    f"bound family partitions are not disjoint: {sorted(overlap)[:5]}"
                )
    normalized_thresholds: dict[str, tuple[int, int]] = {}
    for partition, threshold in thresholds.items():
        if not partition or len(threshold) != 2:
            raise ValueError("invalid readiness threshold")
        target_minimum, family_minimum = threshold
        if target_minimum <= 0 or family_minimum <= 0:
            raise ValueError("readiness minima must be positive")
        normalized_thresholds[partition] = (target_minimum, family_minimum)

    families_by_partition: dict[str, set[str]] = {
        partition: set() for partition in FROZEN_PARTITION_FAMILY_COUNTS
    }
    for record in materialized:
        partition = _text(record, "partition")
        if partition not in families_by_partition:
            raise ValueError(f"unexpected support partition {partition!r}")
        families_by_partition[partition].add(_text(record, "family_id"))
    for partition in FROZEN_PARTITION_FAMILY_COUNTS:
        if families_by_partition[partition] != frozen_families_by_partition[partition]:
            raise ValueError(
                f"partition {partition!r} family IDs do not match the bound manifest"
            )

    support: dict[int, dict[str, list[Record]]] = {
        cluster: {partition: [] for partition in normalized_thresholds}
        for cluster in range(n_clusters)
    }
    unassigned_count = 0
    for record in materialized:
        partition = _text(record, "partition")
        if partition not in normalized_thresholds:
            raise ValueError(f"unexpected support partition {partition!r}")
        cluster = int(assignment[_basis_index(record)])
        if cluster < 0:
            unassigned_count += 1
            continue
        support[cluster][partition].append(record)

    cluster_reports: list[dict[str, Any]] = []
    for cluster in range(n_clusters):
        partition_reports: dict[str, Any] = {}
        overall_ready = True
        for partition, (
            target_minimum,
            family_minimum,
        ) in normalized_thresholds.items():
            items = support[cluster][partition]
            target_count = len({_text(record, "target_id") for record in items})
            response_count = len({_text(record, "response_id") for record in items})
            family_count = len({_text(record, "family_id") for record in items})
            basis_count = len({_basis_index(record) for record in items})
            ready = target_count >= target_minimum and family_count >= family_minimum
            overall_ready = overall_ready and ready
            partition_reports[partition] = {
                "basis_occurrence_count": len(items),
                "distinct_basis_count": basis_count,
                "target_witness_count": target_count,
                "response_witness_count": response_count,
                "family_witness_count": family_count,
                "required_target_witness_count": target_minimum,
                "required_family_witness_count": family_minimum,
                "ready": ready,
            }
        cluster_reports.append(
            {
                "cluster_id": cluster,
                "labeling_ready": overall_ready,
                "partitions": partition_reports,
            }
        )
    ready_count = sum(report["labeling_ready"] for report in cluster_reports)
    return {
        "cluster_count": n_clusters,
        "labeling_ready_cluster_count": ready_count,
        "labeling_ready_cluster_fraction": ready_count / n_clusters,
        "unassigned_basis_occurrence_count": unassigned_count,
        "thresholds": {
            partition: {
                "target_witness_count": threshold[0],
                "family_witness_count": threshold[1],
            }
            for partition, threshold in normalized_thresholds.items()
        },
        "expected_family_ids_by_partition": {
            partition: sorted(family_ids)
            for partition, family_ids in frozen_families_by_partition.items()
        },
        "clusters": cluster_reports,
    }
