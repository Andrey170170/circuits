from __future__ import annotations

import hashlib

import numpy as np
import pytest
from circuits.analysis.bonafide.candidate_coherence import (
    candidate_coherence_bootstrap,
    cluster_support_readiness,
    evaluate_candidate_coherence,
    evaluate_width_one_coherence,
    generation_candidate_centroids,
    missing_aware_cosine,
)


def _candidate_record(
    family: str,
    response: str,
    target: str,
    basis: int,
    vector: tuple[float, ...],
    *,
    partition: str = "generation",
) -> dict[str, object]:
    return {
        "partition": partition,
        "family_id": family,
        "response_id": response,
        "target_id": target,
        "basis_index": basis,
        "vector": vector,
    }


def test_generation_centroids_apply_each_hierarchical_mean_without_renormalizing() -> (
    None
):
    # Family a has two responses, one with two targets.  If basis-targets or
    # targets were pooled, its many +x vectors would overwhelm the hierarchy.
    records = [
        _candidate_record("a", "a1", "a1t1", 0, (2.0, 0.0)),
        _candidate_record("a", "a1", "a1t1", 1, (0.0, 3.0)),
        _candidate_record("a", "a1", "a1t2", 0, (4.0, 0.0)),
        _candidate_record("a", "a2", "a2t1", 0, (4.0, 0.0)),
        _candidate_record("b", "b1", "b1t1", 0, (0.0, 5.0)),
        _candidate_record("b", "b1", "b1t1", 2, (0.0, 0.0)),
    ]
    assignments = np.asarray([0, 0, 0])

    result = generation_candidate_centroids(records, assignments, n_clusters=2)

    # target a1t1=(.5,.5); response a1=(.75,.25); family a=(.875,.125);
    # family b=(0,1); final=(.4375,.5625), normalized.
    expected = np.asarray([0.4375, 0.5625])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result.values[0], expected)
    assert result.available.tolist() == [True, False]
    assert result.cluster_reports[0]["nonempty_target_count"] == 4
    assert result.cluster_reports[0]["empty_target_count"] == 0
    assert result.cluster_reports[0]["zero_norm_basis_target_count"] == 1
    assert result.cluster_reports[1]["empty_family_count"] == 2


def test_generation_centroid_reports_cancellation_as_unavailable() -> None:
    records = [
        _candidate_record("a", "a1", "a1t", 0, (1.0, 0.0)),
        _candidate_record("b", "b1", "b1t", 0, (-1.0, 0.0)),
    ]
    result = generation_candidate_centroids(records, [0], n_clusters=1)
    assert not result.available[0]
    assert result.cluster_reports[0]["pre_normalization_l2_norm"] == 0.0


def _two_state_centroids() -> tuple[list[dict[str, object]], dict[str, object]]:
    generation = [
        _candidate_record("g1", "gr1", "gt1", 0, (1.0, 0.0)),
        _candidate_record("g1", "gr1", "gt1", 1, (0.0, 1.0)),
        _candidate_record("g2", "gr2", "gt2", 0, (1.0, 0.0)),
        _candidate_record("g2", "gr2", "gt2", 1, (0.0, 1.0)),
    ]
    good = generation_candidate_centroids(generation, [0, 1], n_clusters=2)
    swapped = generation_candidate_centroids(generation, [1, 0], n_clusters=2)
    return generation, {"W": swapped, "C": good, "F": good, "S": swapped}


def test_candidate_coherence_uses_one_four_state_intersection_and_family_hierarchy() -> (
    None
):
    _, centroids = _two_state_centroids()
    records = [
        _candidate_record(
            "f1", "r1", "t1", 0, (1.0, 0.0), partition="selection_scoring"
        ),
        _candidate_record(
            "f1", "r1", "t1", 1, (0.0, 1.0), partition="selection_scoring"
        ),
        _candidate_record(
            "f1", "r1", "t2", 0, (1.0, 0.0), partition="selection_scoring"
        ),
        _candidate_record(
            "f2", "r2", "t3", 0, (1.0, 0.0), partition="selection_scoring"
        ),
        # Zero norm is unscoreable in every state and excluded from intersection.
        _candidate_record(
            "f2", "r2", "t3", 1, (0.0, 0.0), partition="selection_scoring"
        ),
        *[
            _candidate_record(
                f"f{index}",
                f"r{index}",
                f"t{index + 2}",
                0,
                (1.0, 0.0),
                partition="selection_scoring",
            )
            for index in range(3, 9)
        ],
    ]
    assignments = {
        "W": np.asarray([0, 1]),
        "C": np.asarray([0, 1]),
        "F": np.asarray([0, 1]),
        "S": np.asarray([0, 1]),
    }

    report = evaluate_candidate_coherence(
        records,
        assignments,
        centroids,  # type: ignore[arg-type]
        partition="selection_scoring",
        expected_family_ids={f"f{index}" for index in range(1, 9)},
    )

    assert report["all_families_scoreable"]
    assert report["coverage"]["basis_occurrence_count"] == 10
    assert report["coverage"]["basis_occurrence_fraction"] == pytest.approx(10 / 11)
    # Family f1 has three scoreable occurrences but receives the same total mass
    # as f2's single occurrence.  All good-state margins are +1; bad are -1.
    assert report["states"]["C"]["coherence_margin"] == pytest.approx(1.0)
    assert report["states"]["W"]["coherence_margin"] == pytest.approx(-1.0)
    assert report["comparisons"]["C_minus_W"]["mean_effect"] == pytest.approx(2.0)
    assert report["comparisons"]["C_minus_S"]["positive_family_count"] == 8
    assert report["coverage"]["hierarchical_weight_fraction"] == pytest.approx(15 / 16)


def test_candidate_coherence_reports_missing_family_instead_of_pooling_coverage() -> (
    None
):
    _, centroids = _two_state_centroids()
    records = [
        _candidate_record("f1", "r1", "t1", 0, (1.0, 0.0), partition="audit"),
        _candidate_record("f2", "r2", "t2", 1, (0.0, 0.0), partition="audit"),
        *[
            _candidate_record(
                f"f{index}",
                f"r{index}",
                f"t{index}",
                0,
                (1.0, 0.0),
                partition="audit",
            )
            for index in range(3, 9)
        ],
    ]
    assignments = {state: np.asarray([0, 1]) for state in ("W", "C", "F", "S")}

    report = evaluate_candidate_coherence(
        records,
        assignments,
        dict.fromkeys(assignments, centroids["C"]),  # type: ignore[arg-type]
        partition="audit",
        expected_family_ids={f"f{index}" for index in range(1, 9)},
    )

    assert not report["all_families_scoreable"]
    assert report["missing_family_ids"] == ["f2"]
    assert report["coverage"]["family_count"] == 7


def test_family_bootstrap_uses_frozen_seed_and_linear_quantile() -> None:
    protocol_hash = "1e24d333fcf9b595bceea9ef42c12bbc0726af22c66ce2a161fd9a1ca45d7983"
    effects = {f"f{index}": float(index - 3) for index in range(8)}
    report = candidate_coherence_bootstrap(
        effects,
        expected_family_ids=set(effects),
        protocol_sha256=protocol_hash,
        partition="selection_scoring",
    )

    expected_seed = int.from_bytes(
        hashlib.sha256(
            protocol_hash.encode()
            + b"\0selection_scoring\0candidate-coherence-bootstrap-v1"
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(expected_seed)
    values = np.asarray([effects[key] for key in sorted(effects)])
    expected = values[rng.integers(0, 8, size=(10_000, 8))].mean(axis=1)
    assert report["seed"] == expected_seed
    assert report["ci_95_lower"] == pytest.approx(
        np.quantile(expected, 0.025, method="linear")
    )
    assert report["ci_95_upper"] == pytest.approx(
        np.quantile(expected, 0.975, method="linear")
    )


def test_held_out_and_bootstrap_settings_cannot_be_relaxed() -> None:
    protocol_hash = "1e24d333fcf9b595bceea9ef42c12bbc0726af22c66ce2a161fd9a1ca45d7983"
    effects = {f"f{index}": float(index) for index in range(8)}
    with pytest.raises(ValueError, match="10,000"):
        candidate_coherence_bootstrap(
            effects,
            expected_family_ids=set(effects),
            protocol_sha256=protocol_hash,
            partition="audit",
            replicates=100,
        )
    with pytest.raises(ValueError, match="selection_scoring or audit"):
        candidate_coherence_bootstrap(
            effects,
            expected_family_ids=set(effects),
            protocol_sha256=protocol_hash,
            partition="generation",
        )
    with pytest.raises(ValueError, match="exactly 8"):
        evaluate_candidate_coherence(
            [],
            {},
            {},
            partition="audit",
            expected_family_ids={"f"},
        )
    with pytest.raises(ValueError, match="exactly 8"):
        evaluate_width_one_coherence(
            [],
            {},
            partition="selection_scoring",
            expected_family_ids={"f"},
        )
    with pytest.raises(ValueError, match="W, C, F, S"):
        evaluate_candidate_coherence(
            [],
            {},
            {},
            partition="audit",
            expected_family_ids=set(effects),
            required_states=("W", "C"),
        )
    with pytest.raises(ValueError, match="W and F"):
        evaluate_width_one_coherence(
            [],
            {},
            partition="audit",
            expected_family_ids=set(effects),
            required_states=("W",),
        )
    with pytest.raises(ValueError, match="frozen protocol"):
        candidate_coherence_bootstrap(
            effects,
            expected_family_ids=set(effects),
            protocol_sha256="0" * 64,
            partition="audit",
        )


def test_missing_aware_cosine_restricts_to_shared_coordinates() -> None:
    assert missing_aware_cosine(
        [1.0, 100.0, 0.0],
        [True, False, True],
        [1.0, -100.0, 0.0],
        [True, False, False],
    ) == pytest.approx(1.0)
    assert missing_aware_cosine([0.0], [True], [1.0], [True]) is None
    assert missing_aware_cosine([1.0], [False], [1.0], [True]) is None


@pytest.mark.parametrize("invalid", [[1, 0], [True, 1], [np.nan, 0.0]])
def test_missing_aware_cosine_rejects_non_boolean_masks(invalid: list[object]) -> None:
    with pytest.raises(TypeError, match="actual boolean"):
        missing_aware_cosine([1.0, 2.0], invalid, [1.0, 2.0], [True, True])


def _width_record(
    family: str,
    response: str,
    target: str,
    basis: int,
    values: tuple[float, ...],
    support: tuple[bool, ...] | None = None,
    *,
    partition: str = "audit",
) -> dict[str, object]:
    return {
        "partition": partition,
        "family_id": family,
        "response_id": response,
        "target_id": target,
        "basis_index": basis,
        "values": values,
        "support": support if support is not None else tuple(True for _ in values),
    }


def _width_target(
    family: str,
    target: str,
    *,
    dimension: int = 2,
    partition: str = "audit",
) -> list[dict[str, object]]:
    x = (1.0, *([0.0] * (dimension - 1)))
    y = (0.0, 1.0, *([0.0] * (dimension - 2)))
    return [
        _width_record(family, f"{family}-r", target, 0, x, partition=partition),
        _width_record(family, f"{family}-r", target, 1, x, partition=partition),
        _width_record(family, f"{family}-r", target, 2, y, partition=partition),
    ]


def test_width_coherence_uses_common_pairs_and_equal_target_hierarchy() -> None:
    records: list[dict[str, object]] = []
    # Each target has two naturally similar x bases and one y basis.  W places
    # x together; F places one x with y, reversing same-minus-different.
    records.extend(_width_target("f1", "t1"))
    records.extend(_width_target("f1", "t2"))
    for index in range(2, 9):
        records.extend(_width_target(f"f{index}", f"t{index + 1}"))
    report = evaluate_width_one_coherence(
        records,
        {"W": [0, 0, 1], "F": [0, 1, 1]},
        partition="audit",
        expected_family_ids={f"f{index}" for index in range(1, 9)},
    )

    assert report["coverage"]["valid_pair_count"] == 27
    assert report["coverage"]["scoreable_target_count"] == 9
    assert report["all_families_scoreable"]
    assert report["states"]["W"]["coherence"] == pytest.approx(1.0)
    assert report["states"]["F"]["coherence"] == pytest.approx(-0.5)
    assert report["comparisons"]["F_minus_W"]["mean_effect"] == pytest.approx(-1.5)


def test_width_coherence_omits_target_without_both_pair_classes_for_every_state() -> (
    None
):
    records = [
        record
        for index in range(1, 9)
        for record in _width_target(
            f"f{index}", f"t{index}", partition="selection_scoring"
        )
    ]
    report = evaluate_width_one_coherence(
        records,
        {"W": [0, 0, 1], "F": [0, 0, 0]},
        partition="selection_scoring",
        expected_family_ids={f"f{index}" for index in range(1, 9)},
    )
    assert report["coverage"]["common_assigned_pair_count"] == 24
    assert report["coverage"]["scoreable_target_count"] == 0
    assert report["coverage"]["scoreable_pair_count"] == 0
    assert report["coverage"]["scoreable_pair_fraction"] == 0.0
    assert report["states"]["W"]["coherence"] is None
    assert not report["all_families_scoreable"]
    assert report["missing_family_ids"] == [f"f{index}" for index in range(1, 9)]


def test_width_coherence_allows_different_source_lengths_across_targets() -> None:
    records = _width_target("f1", "short")
    records.extend(_width_target("f2", "long", dimension=4))
    for index in range(3, 9):
        records.extend(_width_target(f"f{index}", f"other-{index}", dimension=3))
    report = evaluate_width_one_coherence(
        records,
        {"W": [0, 0, 1], "F": [0, 1, 1]},
        partition="audit",
        expected_family_ids={f"f{index}" for index in range(1, 9)},
    )
    assert report["coverage"]["scoreable_target_count"] == 8
    assert report["coverage"]["scoreable_pair_fraction"] == 1.0


def _readiness_family_ids() -> dict[str, set[str]]:
    return {
        partition: {f"{partition}-f{index}" for index in range(count)}
        for partition, count in (
            ("generation", 18),
            ("selection_scoring", 8),
            ("audit", 8),
        )
    }


def test_readiness_requires_all_three_partition_thresholds() -> None:
    records = []
    configuration = {"generation": 18, "selection_scoring": 8, "audit": 8}
    basis = 0
    for partition, family_count in configuration.items():
        for index in range(family_count):
            family = f"{partition}-f{index}"
            records.append(
                {
                    "partition": partition,
                    "family_id": family,
                    "response_id": f"{family}-r",
                    "target_id": f"{partition}-t{index}",
                    "basis_index": basis,
                }
            )
            basis += 1
    assignments = np.zeros(basis, dtype=np.int64)
    report = cluster_support_readiness(
        records,
        assignments,
        expected_family_ids_by_partition=_readiness_family_ids(),
        n_clusters=2,
    )

    assert report["labeling_ready_cluster_count"] == 1
    assert report["clusters"][0]["labeling_ready"]
    assert not report["clusters"][1]["labeling_ready"]
    assert (
        report["clusters"][0]["partitions"]["generation"]["target_witness_count"] == 18
    )


def test_readiness_rejects_relaxed_thresholds_and_overlapping_partitions() -> None:
    records = []
    basis = 0
    for partition, count in (
        ("generation", 18),
        ("selection_scoring", 8),
        ("audit", 8),
    ):
        for index in range(count):
            family = f"{partition}-f{index}"
            records.append(
                {
                    "partition": partition,
                    "family_id": family,
                    "response_id": f"{family}-r",
                    "target_id": f"{family}-t",
                    "basis_index": basis,
                }
            )
            basis += 1
    assignments = np.zeros(basis, dtype=np.int64)
    with pytest.raises(ValueError, match="frozen"):
        cluster_support_readiness(
            records,
            assignments,
            expected_family_ids_by_partition=_readiness_family_ids(),
            thresholds={"generation": (1, 1)},
        )

    records[-1]["family_id"] = "selection_scoring-f0"
    records[-1]["response_id"] = "overlap-r"
    with pytest.raises(ValueError, match="do not match"):
        cluster_support_readiness(
            records,
            assignments,
            expected_family_ids_by_partition=_readiness_family_ids(),
        )

    overlapping = _readiness_family_ids()
    overlapping["audit"].remove("audit-f7")
    overlapping["audit"].add("selection_scoring-f0")
    with pytest.raises(ValueError, match="not disjoint"):
        cluster_support_readiness(
            records,
            assignments,
            expected_family_ids_by_partition=overlapping,
        )


def test_fail_closed_on_duplicate_reduced_occurrence_and_bad_assignment() -> None:
    record = _candidate_record("f", "r", "t", 0, (1.0, 0.0))
    with pytest.raises(ValueError, match="duplicate"):
        generation_candidate_centroids([record, record], [0])
    with pytest.raises(ValueError, match="cover"):
        generation_candidate_centroids([record], [])


def test_partition_and_bound_family_identity_firewalls() -> None:
    with pytest.raises(ValueError, match="partition firewall"):
        generation_candidate_centroids(
            [_candidate_record("f", "r", "t", 0, (1.0, 0.0), partition="audit")],
            [0],
        )

    family_ids = {f"f{index}" for index in range(8)}
    selection_records = [
        _candidate_record(
            family,
            f"{family}-r",
            f"{family}-t",
            0,
            (1.0, 0.0),
            partition="selection_scoring",
        )
        for family in sorted(family_ids)
    ]
    with pytest.raises(ValueError, match="partition firewall"):
        evaluate_candidate_coherence(
            selection_records,
            {},
            {},
            partition="audit",
            expected_family_ids=family_ids,
        )
    wrong_ids = set(family_ids)
    wrong_ids.remove("f7")
    wrong_ids.add("other")
    audit_records = [dict(record, partition="audit") for record in selection_records]
    with pytest.raises(ValueError, match="bound manifest"):
        evaluate_candidate_coherence(
            audit_records,
            {},
            {},
            partition="audit",
            expected_family_ids=wrong_ids,
        )
    with pytest.raises(ValueError, match="partition firewall"):
        evaluate_width_one_coherence(
            [
                record
                for family in sorted(family_ids)
                for record in _width_target(
                    family, f"{family}-t", partition="selection_scoring"
                )
            ],
            {},
            partition="audit",
            expected_family_ids=family_ids,
        )
    with pytest.raises(ValueError, match="bound manifest"):
        evaluate_width_one_coherence(
            [
                record
                for family in sorted(family_ids)
                for record in _width_target(family, f"{family}-t")
            ],
            {},
            partition="audit",
            expected_family_ids=wrong_ids,
        )

    effects = dict.fromkeys(family_ids, 1.0)
    with pytest.raises(ValueError, match="do not match"):
        candidate_coherence_bootstrap(
            effects,
            expected_family_ids=wrong_ids,
            partition="audit",
        )


def test_candidate_coherence_rejects_assignment_centroid_state_mismatch() -> None:
    _, centroids = _two_state_centroids()
    records = [
        _candidate_record(
            f"f{index}",
            f"r{index}",
            f"t{index}",
            0,
            (1.0, 0.0),
            partition="audit",
        )
        for index in range(8)
    ]
    assignments = {state: [2] for state in ("W", "C", "F", "S")}
    with pytest.raises(ValueError, match="absent"):
        evaluate_candidate_coherence(
            records,
            assignments,
            dict.fromkeys(assignments, centroids["C"]),  # type: ignore[arg-type]
            partition="audit",
            expected_family_ids={f"f{index}" for index in range(8)},
        )
