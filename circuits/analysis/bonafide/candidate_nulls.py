"""Deterministic candidate-direction null permutations.

The null preserves each target's signed-basis support and the exact multiset of
five-channel candidate vectors within every layer/activation-polarity stratum.
Only the association between a supported signed basis and its local competitive
direction is permuted.  This module deliberately has no persistence, clustering,
or scoring dependencies.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from circuits.analysis.bonafide.candidate_profiles import FROZEN_PROTOCOL_SHA256

DIRECTION_NULL_NAMESPACE = "direction-null-v1"
DIRECTION_WIDTH = 5
MINIMUM_MOVABLE_BLOCK_SIZE = 4
NULL_EFFECTIVENESS_THRESHOLD = 0.8

Vector5 = tuple[float, float, float, float, float]


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validated_vector(value: Sequence[float], field: str) -> Vector5:
    if isinstance(value, (str, bytes)) or len(value) != DIRECTION_WIDTH:
        raise ValueError(f"{field} must contain exactly five values")
    values = tuple(
        _finite_float(component, f"{field}[{index}]")
        for index, component in enumerate(value)
    )
    return values[0], values[1], values[2], values[3], values[4]


def _l2_mass(vector: Vector5) -> float:
    mass = float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))
    if not math.isfinite(mass):
        raise ValueError("candidate-vector L2 mass must be finite")
    return mass


@dataclass(frozen=True)
class CandidateDirectionTarget:
    """One target's candidate profiles aligned by occurrence position."""

    target_id: str
    signed_basis_indices: tuple[int, ...]
    layers: tuple[int, ...]
    polarities: tuple[str, ...]
    vectors: tuple[Vector5, ...]
    target_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError("target_id must be a non-empty string")
        count = len(self.signed_basis_indices)
        if count == 0:
            raise ValueError("a candidate-direction target must have support")
        if not (len(self.layers) == len(self.polarities) == len(self.vectors) == count):
            raise ValueError("target candidate arrays must have equal lengths")
        if len(set(self.signed_basis_indices)) != count:
            raise ValueError("signed_basis_indices must be unique within a target")
        for index, basis_index in enumerate(self.signed_basis_indices):
            if isinstance(basis_index, bool) or not isinstance(basis_index, int):
                raise TypeError(f"signed_basis_indices[{index}] must be an integer")
            if basis_index < 0:
                raise ValueError(f"signed_basis_indices[{index}] must be nonnegative")
        for index, layer in enumerate(self.layers):
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise TypeError(f"layers[{index}] must be an integer")
            if layer < 0:
                raise ValueError(f"layers[{index}] must be nonnegative")
        for index, polarity in enumerate(self.polarities):
            if polarity not in {"+", "-"}:
                raise ValueError(f"polarities[{index}] must be '+' or '-'")
        for index, vector in enumerate(self.vectors):
            _validated_vector(vector, f"vectors[{index}]")
        target_weight = _finite_float(self.target_weight, "target_weight")
        if target_weight <= 0.0:
            raise ValueError("target_weight must be positive")


@dataclass(frozen=True)
class PermutedCandidateDirectionTarget:
    """A null target with output rows aligned to the original support."""

    target_id: str
    signed_basis_indices: tuple[int, ...]
    layers: tuple[int, ...]
    polarities: tuple[str, ...]
    vectors: tuple[Vector5, ...]
    source_signed_basis_indices: tuple[int, ...]
    movable: tuple[bool, ...]
    mass_deciles: tuple[int, ...]
    block_ids: tuple[int | None, ...]
    target_weight: float


@dataclass(frozen=True)
class CandidateDirectionNullReport:
    """Effectiveness diagnostics for one deterministic null replicate.

    ``hierarchical_target_weight_fraction`` gives every target its supplied
    hierarchical weight, then weights that target's movable occurrence fraction.
    Consequently, targets with larger candidate supports do not silently receive
    more total mass than the frozen hierarchy assigned them.
    """

    replicate_index: int
    rng_seed: int
    total_basis_occurrence_count: int
    movable_basis_occurrence_count: int
    movable_basis_occurrence_fraction: float
    total_hierarchical_target_weight: float
    movable_hierarchical_target_weight: float
    hierarchical_target_weight_fraction: float
    effectiveness_threshold: float
    effective: bool


@dataclass(frozen=True)
class CandidateDirectionNull:
    """Permuted target profiles and their protocol effectiveness report."""

    targets: tuple[PermutedCandidateDirectionTarget, ...]
    report: CandidateDirectionNullReport


def direction_null_seed(protocol_sha256: str, replicate_index: int) -> int:
    """Derive the protocol-frozen unsigned 64-bit seed for one replicate."""

    if (
        not isinstance(protocol_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None
    ):
        raise ValueError("protocol_sha256 must be a lowercase SHA-256 hex digest")
    if protocol_sha256 != FROZEN_PROTOCOL_SHA256:
        raise ValueError("protocol_sha256 does not match the frozen protocol")
    if (
        isinstance(replicate_index, bool)
        or not isinstance(replicate_index, int)
        or not 0 <= replicate_index < 100
    ):
        raise ValueError("replicate_index must be between zero and 99")
    payload = b"\0".join(
        (
            protocol_sha256.encode("ascii"),
            DIRECTION_NULL_NAMESPACE.encode("ascii"),
            replicate_index.to_bytes(8, byteorder="big", signed=False),
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def mass_decile_assignments(
    signed_basis_indices: Sequence[int],
    vectors: Sequence[Sequence[float]],
) -> tuple[int, ...]:
    """Assign exact ordinal mass deciles, returned in input alignment."""

    count = len(signed_basis_indices)
    if count == 0:
        return ()
    if len(vectors) != count:
        raise ValueError("signed-basis and candidate-vector lengths disagree")
    if len(set(signed_basis_indices)) != count:
        raise ValueError("signed_basis_indices must be unique within a stratum")
    validated_vectors = tuple(
        _validated_vector(vector, f"vectors[{index}]")
        for index, vector in enumerate(vectors)
    )
    masses = tuple(_l2_mass(vector) for vector in validated_vectors)
    ordered_positions = sorted(
        range(count),
        key=lambda position: (
            masses[position],
            signed_basis_indices[position],
        ),
    )
    result = [0] * count
    for ordinal_rank, position in enumerate(ordered_positions):
        result[position] = min(9, (10 * ordinal_rank) // count)
    return tuple(result)


def merge_mass_decile_blocks(
    deciles: Sequence[int],
    *,
    minimum_size: int = MINIMUM_MOVABLE_BLOCK_SIZE,
) -> tuple[tuple[int, ...], ...]:
    """Merge consecutive nonempty deciles into disjoint protocol blocks.

    Returned members are positions into ``deciles``.  A sole final block smaller
    than ``minimum_size`` is returned but is fixed; when a prior block exists the
    final small block is merged backward exactly once.
    """

    if isinstance(minimum_size, bool) or not isinstance(minimum_size, int):
        raise TypeError("minimum_size must be an integer")
    if minimum_size <= 0:
        raise ValueError("minimum_size must be positive")
    members_by_decile: dict[int, list[int]] = defaultdict(list)
    for position, decile in enumerate(deciles):
        if isinstance(decile, bool) or not isinstance(decile, int):
            raise TypeError(f"deciles[{position}] must be an integer")
        if not 0 <= decile <= 9:
            raise ValueError(f"deciles[{position}] must be between zero and nine")
        members_by_decile[decile].append(position)

    blocks: list[list[int]] = []
    pending: list[int] = []
    for decile in range(10):
        members = members_by_decile.get(decile)
        if not members:
            continue
        pending.extend(members)
        if len(pending) >= minimum_size:
            blocks.append(pending)
            pending = []
    if pending:
        if blocks:
            blocks[-1].extend(pending)
        else:
            blocks.append(pending)
    return tuple(tuple(block) for block in blocks)


def _permute_one_target(
    target: CandidateDirectionTarget,
    rng: np.random.Generator,
) -> PermutedCandidateDirectionTarget:
    count = len(target.signed_basis_indices)
    source_positions = list(range(count))
    movable = [False] * count
    mass_deciles = [-1] * count
    block_ids: list[int | None] = [None] * count
    strata: dict[tuple[int, str], list[int]] = defaultdict(list)
    for position, (layer, polarity) in enumerate(
        zip(target.layers, target.polarities, strict=True)
    ):
        strata[(layer, polarity)].append(position)

    next_block_id = 0
    for stratum in sorted(strata):
        positions = strata[stratum]
        basis_indices = [
            target.signed_basis_indices[position] for position in positions
        ]
        vectors = [target.vectors[position] for position in positions]
        deciles = mass_decile_assignments(basis_indices, vectors)
        for position, decile in zip(positions, deciles, strict=True):
            mass_deciles[position] = decile

        canonical_local_positions = sorted(
            range(len(positions)),
            key=lambda local_position: (
                _l2_mass(vectors[local_position]),
                basis_indices[local_position],
            ),
        )
        canonical_deciles = [
            deciles[position] for position in canonical_local_positions
        ]
        for local_block in merge_mass_decile_blocks(canonical_deciles):
            block = tuple(
                positions[canonical_local_positions[local_position]]
                for local_position in local_block
            )
            if len(block) < MINIMUM_MOVABLE_BLOCK_SIZE:
                continue
            # The protocol requires one and only one RNG permutation per block.
            permuted_sources = rng.permutation(np.asarray(block, dtype=np.int64))
            for destination, source in zip(
                block, permuted_sources.tolist(), strict=True
            ):
                source_positions[destination] = int(source)
                movable[destination] = True
                block_ids[destination] = next_block_id
            next_block_id += 1

    if any(decile < 0 for decile in mass_deciles):
        raise AssertionError("every supported occurrence must receive a mass decile")
    vectors = tuple(target.vectors[position] for position in source_positions)
    source_basis_indices = tuple(
        target.signed_basis_indices[position] for position in source_positions
    )
    return PermutedCandidateDirectionTarget(
        target_id=target.target_id,
        signed_basis_indices=target.signed_basis_indices,
        layers=target.layers,
        polarities=target.polarities,
        vectors=vectors,
        source_signed_basis_indices=source_basis_indices,
        movable=tuple(movable),
        mass_deciles=tuple(mass_deciles),
        block_ids=tuple(block_ids),
        target_weight=float(target.target_weight),
    )


def generate_candidate_direction_null(
    targets: Sequence[CandidateDirectionTarget],
    *,
    protocol_sha256: str,
    replicate_index: int,
) -> CandidateDirectionNull:
    """Generate one deterministic protocol null over per-target profiles.

    RNG calls are made in ascending target ID, layer, polarity, mass, and signed-
    basis order.  Results retain the caller's target order and row alignment.
    """

    materialized = tuple(targets)
    if not materialized:
        raise ValueError("candidate-direction null requires at least one target")
    target_ids = [target.target_id for target in materialized]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target_id values must be unique")
    seed = direction_null_seed(protocol_sha256, replicate_index)
    rng = np.random.default_rng(seed)
    permuted_by_id = {
        target.target_id: _permute_one_target(target, rng)
        for target in sorted(materialized, key=lambda item: item.target_id)
    }
    permuted = tuple(permuted_by_id[target.target_id] for target in materialized)

    total_occurrences = sum(len(target.signed_basis_indices) for target in materialized)
    movable_occurrences = sum(sum(target.movable) for target in permuted)
    occurrence_fraction = movable_occurrences / total_occurrences
    total_target_weight = math.fsum(target.target_weight for target in materialized)
    movable_target_weight = math.fsum(
        target.target_weight * (sum(result.movable) / len(result.movable))
        for target, result in zip(materialized, permuted, strict=True)
    )
    target_weight_fraction = movable_target_weight / total_target_weight
    effective = (
        occurrence_fraction >= NULL_EFFECTIVENESS_THRESHOLD
        and target_weight_fraction >= NULL_EFFECTIVENESS_THRESHOLD
    )
    report = CandidateDirectionNullReport(
        replicate_index=replicate_index,
        rng_seed=seed,
        total_basis_occurrence_count=total_occurrences,
        movable_basis_occurrence_count=movable_occurrences,
        movable_basis_occurrence_fraction=occurrence_fraction,
        total_hierarchical_target_weight=total_target_weight,
        movable_hierarchical_target_weight=movable_target_weight,
        hierarchical_target_weight_fraction=target_weight_fraction,
        effectiveness_threshold=NULL_EFFECTIVENESS_THRESHOLD,
        effective=effective,
    )
    return CandidateDirectionNull(targets=permuted, report=report)
