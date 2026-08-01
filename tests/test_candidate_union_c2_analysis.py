from __future__ import annotations

import runpy
import sys

import numpy as np
import pytest

from scripts.bonafide.candidate_union_c2_analysis import (
    RetrievalRecord,
    TargetProfile,
    _canonical_sha256,
    _family_mrr,
    _hierarchical_mean,
    _validate_identity,
    directional_similarity,
    effective_rank,
    family_block_bootstrap,
    family_block_bootstrap_views,
    leave_one_family_out_views,
    next_bin_retrieval,
    rank_aligned_contrasts,
    summarize_retrieval,
)


def _basis(index: int):
    return ("model", "revision", index // 4, index, 1)


def _profile(
    case_id: str,
    family_id: str,
    response_id: str,
    phase_bin: int,
    values: list[float],
) -> TargetProfile:
    width1 = {_basis(index): value for index, value in enumerate(values)}
    candidate = {
        key: np.asarray([value, value / 2, -value, 2 * value, -value / 3])
        for key, value in width1.items()
    }
    raw = {
        key: np.asarray([value, -value, value / 2, -value / 2, value])
        for key, value in width1.items()
    }
    return TargetProfile(
        case_id=case_id,
        source_artifact_id=f"source-{case_id}",
        family_id=family_id,
        response_id=response_id,
        phase_bin=phase_bin,
        width1=width1,
        candidate=candidate,
        raw_candidate=raw,
        effective_rank=2.0,
        bidirectional_raw_fraction=1.0,
    )


def _trajectory(
    prefix: str,
    family_id: str,
    response_id: str,
    values: list[float],
) -> list[TargetProfile]:
    return [
        _profile(f"{prefix}-p{phase}", family_id, response_id, phase, values)
        for phase in range(7)
    ]


def test_rank_alignment_handles_observed_outside_top_five() -> None:
    ranks = [7, 1, 2, 3, 4, 5]
    contributions = [10.0, 11.0, 8.0, 15.0, 10.0, 4.0]

    result = rank_aligned_contrasts(ranks, 0, contributions)

    np.testing.assert_array_equal(result, [1.0, -2.0, 5.0, 0.0, -6.0])


def test_rank_alignment_makes_observed_top_five_axis_exactly_zero() -> None:
    ranks = [3, 1, 2, 4, 5]
    result = rank_aligned_contrasts(ranks, 0, [7.0, 2.0, 5.0, 11.0, 13.0])
    np.testing.assert_array_equal(result, [-5.0, -2.0, 0.0, 4.0, 6.0])


def test_directional_similarity_treats_absent_basis_as_missing() -> None:
    left = {_basis(index): float(index + 1) for index in range(16)}
    right = {_basis(index): float(index + 1) for index in range(1, 17)}

    assert directional_similarity(left, right) is None
    score, support = directional_similarity(left, right, min_common_bases=15)
    assert support == 15
    assert score == 1.0


def test_next_bin_retrieval_excludes_same_family_alternative_response() -> None:
    positive = [float(index + 1) for index in range(16)]
    negative = [-value for value in positive]
    profiles = [
        *_trajectory("r1", "family-a", "response-1", positive),
        # Same-family response would outrank the truth by stable tie-break if it
        # were incorrectly admitted, but the protocol excludes it.
        *_trajectory("r2", "family-a", "response-0", positive),
        *_trajectory("r3", "family-b", "response-3", negative),
    ]

    records, eligible = next_bin_retrieval(profiles, "width1")

    assert eligible == 18
    by_anchor = {record.anchor_id: record for record in records}
    assert by_anchor["r1-p0"].candidate_count == 2
    assert by_anchor["r1-p0"].true_rank == 1


def test_next_bin_retrieval_omits_invalid_distractor_but_requires_true_score() -> None:
    values = [float(index + 1) for index in range(16)]
    trajectory_a = _trajectory("a", "family-a", "response-a", values)
    trajectory_b = _trajectory("b", "family-b", "response-b", values)
    distractor1 = trajectory_b[1]
    # Keep only 15 common bases with the anchor.
    distractor1 = TargetProfile(
        **{
            **distractor1.__dict__,
            "width1": {
                **{_basis(index): values[index] for index in range(1, 16)},
                _basis(99): 1.0,
            },
        }
    )
    trajectory_b[1] = distractor1

    records, eligible = next_bin_retrieval([*trajectory_a, *trajectory_b], "width1")

    assert eligible == 12
    by_anchor = {record.anchor_id: record for record in records}
    assert by_anchor["a-p0"].candidate_count == 1
    assert by_anchor["a-p0"].planned_candidate_count == 2
    assert "b-p0" not in by_anchor


def test_family_block_bootstrap_is_deterministic() -> None:
    effects = {"family-a": 0.10, "family-b": -0.02, "family-c": 0.04}

    first = family_block_bootstrap(effects, replicates=100, seed=17)
    second = family_block_bootstrap(effects, replicates=100, seed=17)
    different = family_block_bootstrap(effects, replicates=100, seed=18)

    assert first == second
    assert first["lower_95"] != different["lower_95"]


def _record(
    anchor_id: str,
    family_id: str,
    response_id: str,
    reciprocal_rank: float,
) -> RetrievalRecord:
    return RetrievalRecord(
        anchor_id=anchor_id,
        family_id=family_id,
        response_id=response_id,
        reciprocal_rank=reciprocal_rank,
        top_one=float(reciprocal_rank == 1.0),
        true_rank=max(1, round(1 / reciprocal_rank)),
        candidate_count=2,
        planned_candidate_count=2,
        common_basis_support=(16, 17),
    )


def test_hierarchical_mrr_and_utility_use_separate_view_coverage() -> None:
    width = [
        _record("a1", "family-a", "response-1", 1.0),
        _record("a2", "family-a", "response-1", 0.5),
        _record("a3", "family-a", "response-2", 1.0),
        _record("b1", "family-b", "response-3", 0.5),
    ]
    multiview = [
        _record("a1", "family-a", "response-1", 1.0),
        _record("a3", "family-a", "response-2", 0.5),
        # family-b has different anchor coverage from width-one.
        _record("b2", "family-b", "response-3", 1.0),
    ]

    width_mrr = _hierarchical_mean(width, "reciprocal_rank")
    multiview_mrr = _hierarchical_mean(multiview, "reciprocal_rank")
    width_families = _family_mrr(width)
    multiview_families = _family_mrr(multiview)

    assert width_mrr == np.mean(list(width_families.values()))
    assert multiview_mrr == np.mean(list(multiview_families.values()))
    assert multiview_mrr - width_mrr == (
        np.mean(list(multiview_families.values()))
        - np.mean(list(width_families.values()))
    )


def test_scored_coverage_uses_equal_family_then_response_weights() -> None:
    values = [float(index + 1) for index in range(16)]
    profiles = [
        *_trajectory("r1", "family-a", "response-1", values),
        *_trajectory("r2", "family-a", "response-2", values),
        *_trajectory("r3", "family-b", "response-3", values),
    ]
    scored = [
        _record(f"r1-p{phase}", "family-a", "response-1", 1.0) for phase in range(6)
    ]

    summary = summarize_retrieval(scored, 18, eligible_profiles=profiles)

    assert summary["raw_scored_anchor_coverage"] == 1 / 3
    assert summary["scored_anchor_coverage"] == 0.25


def test_entropy_effective_rank_known_matrices() -> None:
    assert effective_rank(np.zeros((8, 5))) == 0.0
    assert effective_rank(np.ones((8, 5))) == pytest.approx(1.0)
    assert effective_rank(np.eye(5)) == pytest.approx(5.0)


def test_view_specific_family_bootstrap_is_deterministic_with_missing_family() -> None:
    width = {"family-a": 0.4, "family-b": 0.6}
    multiview = {"family-a": 0.7, "family-c": 0.8}
    families = ["family-a", "family-b", "family-c"]

    first = family_block_bootstrap_views(
        width, multiview, families, replicates=100, seed=9
    )
    second = family_block_bootstrap_views(
        width, multiview, families, replicates=100, seed=9
    )

    assert first == second
    assert first["family_universe_count"] == 3
    lofo = leave_one_family_out_views(width, multiview, families)
    assert lofo["family_count"] == 3
    assert set(lofo["estimates"]) == set(families)


def test_module_mode_help_smoke(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts.bonafide.candidate_union_c2_analysis", "--help"],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(
            "scripts.bonafide.candidate_union_c2_analysis", run_name="__main__"
        )
    assert exit_info.value.code == 0
    assert "--candidate-union-root" in capsys.readouterr().out


def test_artifact_identity_digest_drift_fails_closed() -> None:
    value = {"source_width1_artifact_id": "source-a", "plan": "abc"}
    manifest = {"artifact_identity": {**value, "sha256": _canonical_sha256(value)}}
    assert _validate_identity(manifest, "synthetic") == manifest["artifact_identity"]

    manifest["artifact_identity"]["plan"] = "drifted"
    with pytest.raises(ValueError, match="identity digest"):
        _validate_identity(manifest, "synthetic")
