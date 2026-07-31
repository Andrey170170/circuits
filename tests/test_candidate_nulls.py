from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from circuits.analysis.bonafide.candidate_nulls import (
    CandidateDirectionTarget,
    direction_null_seed,
    generate_candidate_direction_null,
    mass_decile_assignments,
    merge_mass_decile_blocks,
)

PROTOCOL_SHA256 = "1e24d333fcf9b595bceea9ef42c12bbc0726af22c66ce2a161fd9a1ca45d7983"


def _vector(value: float) -> tuple[float, float, float, float, float]:
    return (value, value + 0.1, value + 0.2, value + 0.3, value + 0.4)


def _target(
    target_id: str,
    count: int,
    *,
    layer: int = 2,
    polarity: str = "+",
    target_weight: float = 1.0,
) -> CandidateDirectionTarget:
    return CandidateDirectionTarget(
        target_id=target_id,
        signed_basis_indices=tuple(range(count)),
        layers=(layer,) * count,
        polarities=(polarity,) * count,
        vectors=tuple(_vector(float(index)) for index in range(count)),
        target_weight=target_weight,
    )


def test_direction_null_seed_matches_frozen_byte_recipe() -> None:
    replicate = 17
    expected = int.from_bytes(
        hashlib.sha256(
            PROTOCOL_SHA256.encode("ascii")
            + b"\0direction-null-v1\0"
            + replicate.to_bytes(8, "big")
        ).digest()[:8],
        "big",
    )
    assert direction_null_seed(PROTOCOL_SHA256, replicate) == expected
    assert direction_null_seed(PROTOCOL_SHA256, 0) != expected

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        direction_null_seed(PROTOCOL_SHA256.upper(), replicate)
    with pytest.raises(ValueError, match="frozen protocol"):
        direction_null_seed("0" * 64, replicate)
    with pytest.raises(ValueError, match="zero and 99"):
        direction_null_seed(PROTOCOL_SHA256, -1)
    with pytest.raises(ValueError, match="zero and 99"):
        direction_null_seed(PROTOCOL_SHA256, 100)


def test_generation_entry_point_rejects_protocol_and_replicate_drift() -> None:
    target = _target("target", 4)
    with pytest.raises(ValueError, match="frozen protocol"):
        generate_candidate_direction_null(
            [target], protocol_sha256="f" * 64, replicate_index=0
        )
    with pytest.raises(ValueError, match="zero and 99"):
        generate_candidate_direction_null(
            [target], protocol_sha256=PROTOCOL_SHA256, replicate_index=100
        )


def test_mass_deciles_use_basis_index_to_break_l2_ties() -> None:
    tied = (1.0, 0.0, 0.0, 0.0, 0.0)
    basis_indices = [70, 10, 50, 20, 90]
    deciles = mass_decile_assignments(basis_indices, [tied] * 5)
    by_basis = dict(zip(basis_indices, deciles, strict=True))
    assert by_basis == {10: 0, 20: 2, 50: 4, 70: 6, 90: 8}

    reordered = [90, 20, 70, 10, 50]
    reordered_deciles = mass_decile_assignments(reordered, [tied] * 5)
    assert dict(zip(reordered, reordered_deciles, strict=True)) == by_basis


def test_decile_blocks_merge_final_small_block_backward_once() -> None:
    # Four occurrences reach the threshold, another four form a second block,
    # and the final two must merge backward into only the second block.
    assert merge_mass_decile_blocks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == (
        (0, 1, 2, 3),
        (4, 5, 6, 7, 8, 9),
    )
    assert merge_mass_decile_blocks([0, 4, 9]) == ((0, 1, 2),)
    assert merge_mass_decile_blocks([]) == ()


def test_null_is_deterministic_and_independent_of_target_input_order() -> None:
    first = _target("target-b", 9, target_weight=0.75)
    second = _target("target-a", 8, target_weight=0.25)
    result = generate_candidate_direction_null(
        [first, second], protocol_sha256=PROTOCOL_SHA256, replicate_index=3
    )
    repeated = generate_candidate_direction_null(
        [first, second], protocol_sha256=PROTOCOL_SHA256, replicate_index=3
    )
    reversed_result = generate_candidate_direction_null(
        [second, first], protocol_sha256=PROTOCOL_SHA256, replicate_index=3
    )
    assert result == repeated
    assert {target.target_id: target.vectors for target in result.targets} == {
        target.target_id: target.vectors for target in reversed_result.targets
    }
    assert result.report.rng_seed == direction_null_seed(PROTOCOL_SHA256, 3)


def test_small_strata_stay_fixed_and_fail_effectiveness_gate() -> None:
    target = CandidateDirectionTarget(
        target_id="small-strata",
        signed_basis_indices=(0, 1, 2, 3, 4, 5),
        layers=(1, 1, 1, 2, 2, 2),
        polarities=("+", "+", "+", "-", "-", "-"),
        vectors=tuple(_vector(float(index)) for index in range(6)),
        target_weight=1.0,
    )
    result = generate_candidate_direction_null(
        [target], protocol_sha256=PROTOCOL_SHA256, replicate_index=0
    )
    permuted = result.targets[0]
    assert permuted.vectors == target.vectors
    assert permuted.source_signed_basis_indices == target.signed_basis_indices
    assert permuted.movable == (False,) * 6
    assert permuted.block_ids == (None,) * 6
    assert result.report.movable_basis_occurrence_fraction == 0.0
    assert result.report.hierarchical_target_weight_fraction == 0.0
    assert not result.report.effective


def test_null_preserves_support_and_vector_multisets_within_strata() -> None:
    target = CandidateDirectionTarget(
        target_id="mixed",
        signed_basis_indices=tuple(range(14)),
        layers=(1,) * 5 + (1,) * 4 + (2,) * 5,
        polarities=("+",) * 5 + ("-",) * 4 + ("+",) * 5,
        vectors=tuple(_vector(float(index)) for index in range(14)),
        target_weight=1.0,
    )
    result = generate_candidate_direction_null(
        [target], protocol_sha256=PROTOCOL_SHA256, replicate_index=41
    )
    permuted = result.targets[0]
    assert permuted.signed_basis_indices == target.signed_basis_indices
    assert permuted.layers == target.layers
    assert permuted.polarities == target.polarities
    assert all(permuted.movable)
    assert all(block_id is not None for block_id in permuted.block_ids)
    for stratum in {(1, "+"), (1, "-"), (2, "+")}:
        original = Counter(
            vector
            for layer, polarity, vector in zip(
                target.layers, target.polarities, target.vectors, strict=True
            )
            if (layer, polarity) == stratum
        )
        null = Counter(
            vector
            for layer, polarity, vector in zip(
                permuted.layers, permuted.polarities, permuted.vectors, strict=True
            )
            if (layer, polarity) == stratum
        )
        assert null == original

    assert result.report.total_basis_occurrence_count == 14
    assert result.report.movable_basis_occurrence_count == 14
    assert result.report.movable_basis_occurrence_fraction == 1.0
    assert result.report.hierarchical_target_weight_fraction == 1.0
    assert result.report.effective


def test_hierarchical_effectiveness_weights_per_target_movable_fraction() -> None:
    mostly_fixed = CandidateDirectionTarget(
        target_id="mostly-fixed",
        signed_basis_indices=tuple(range(10)),
        layers=(0,) * 4 + (1,) * 3 + (2,) * 3,
        polarities=("+",) * 10,
        vectors=tuple(_vector(float(index)) for index in range(10)),
        target_weight=0.9,
    )
    movable = _target("movable", 4, target_weight=0.1)
    result = generate_candidate_direction_null(
        [mostly_fixed, movable],
        protocol_sha256=PROTOCOL_SHA256,
        replicate_index=7,
    )
    assert result.report.movable_basis_occurrence_fraction == pytest.approx(8 / 14)
    assert result.report.movable_hierarchical_target_weight == pytest.approx(
        0.9 * 0.4 + 0.1
    )
    assert result.report.hierarchical_target_weight_fraction == pytest.approx(0.46)
    assert not result.report.effective
