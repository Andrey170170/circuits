from __future__ import annotations

import runpy
import sys

import numpy as np
import pytest

from scripts.bonafide.candidate_union_c2_salvage_analysis import (
    Anchor,
    PairRow,
    PairScore,
    _observed_metrics,
    _profile_score,
    analytic_chance,
    constrained_response_permutations,
    hierarchical_mean,
    holm_adjust,
    midrank_reciprocal_and_top_one,
    pair_rows_to_tables,
    permutation_inference,
    width_preferred_backoff,
)


def _basis(index: int):
    return ("model", "revision", index // 4, index, 1)


def test_flattened_candidate_score_matches_manual_cosine() -> None:
    left = {
        _basis(index): np.asarray([index + 1, 2.0, -1.0, 0.5, 3.0])
        for index in range(16)
    }
    right = {
        _basis(index): np.asarray([2 * index + 1, -1.0, 4.0, 0.25, 2.0])
        for index in range(16)
    }
    observed = _profile_score(left, right)
    left_flat = np.concatenate([left[key] for key in sorted(left)])
    right_flat = np.concatenate([right[key] for key in sorted(right)])
    expected = np.dot(left_flat, right_flat) / (
        np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    )

    assert observed.valid
    assert observed.support == 16
    assert observed.score == pytest.approx(expected)


def test_structurally_zero_rank_channel_is_invalid_not_zero_evidence() -> None:
    left = {
        _basis(index): np.asarray([1.0, 0.0, index + 1.0, -1.0, 2.0])
        for index in range(16)
    }
    right = {
        _basis(index): np.asarray([2.0, 0.0, 2 * index + 1.0, 1.0, -3.0])
        for index in range(16)
    }

    score = _profile_score(left, right, channel=1)

    assert not score.valid
    assert score.score is None
    assert score.support == 16
    assert score.invalid_reason == "zero_norm"


def test_percentile_backoff_prefers_width_and_fills_only_missing_pairs() -> None:
    width = np.asarray([[0.1, np.nan, 0.9, np.nan]])
    candidate = np.asarray([[0.9, 0.8, 0.1, np.nan]])

    result = width_preferred_backoff(width, candidate)

    # Width percentiles are 0.25 and 0.75; candidate percentiles are 5/6,
    # 1/2, and 1/6.  Width remains authoritative where it exists.
    np.testing.assert_allclose(result[0, :3], [0.25, 0.5, 0.75])
    assert np.isnan(result[0, 3])


def test_midrank_and_missing_truth_policy() -> None:
    scores = np.asarray([[0.8, 0.8, 0.2, np.nan]])

    reciprocal, top = midrank_reciprocal_and_top_one(scores)

    np.testing.assert_allclose(reciprocal, [[2 / 3, 2 / 3, 1 / 3, 0.0]])
    np.testing.assert_array_equal(top, [[0.5, 0.5, 0.0, 0.0]])


def test_analytic_chance_includes_invalid_truths_as_zero() -> None:
    anchors = (Anchor("a", "family-a", "response-a", 0),)
    scores = np.asarray([[0.9, 0.2, np.nan, np.nan]])
    planned = np.ones((1, 4), dtype=bool)

    chance = analytic_chance(scores, planned, anchors)

    assert chance["conditional_scored_mrr"] == pytest.approx(0.75)
    assert chance["conditional_scored_top_one"] == pytest.approx(0.5)
    assert chance["planned_pool_zero_filled_mrr"] == pytest.approx(0.375)
    assert chance["planned_pool_zero_filled_top_one"] == pytest.approx(0.25)


def test_constrained_whole_response_permutations_are_deterministic() -> None:
    responses = ["a1", "a2", "b", "c"]
    families = ["a", "a", "b", "c"]

    first, rejected_first = constrained_response_permutations(
        responses, families, replicates=100, seed=17
    )
    second, rejected_second = constrained_response_permutations(
        responses, families, replicates=100, seed=17
    )

    np.testing.assert_array_equal(first, second)
    assert rejected_first == rejected_second
    identity = np.arange(4)
    family_array = np.asarray(families)
    for permutation in first:
        assert sorted(permutation.tolist()) == identity.tolist()
        assert not np.any(
            (permutation != identity) & (family_array[permutation] == family_array)
        )


def test_hierarchical_weighting_is_equal_family_then_response_then_anchor() -> None:
    anchors = (
        Anchor("a1", "family-a", "response-a1", 0),
        Anchor("a2", "family-a", "response-a1", 1),
        Anchor("a3", "family-a", "response-a2", 0),
        Anchor("b1", "family-b", "response-b", 0),
    )
    values = np.asarray([0.0, 0.0, 1.0, 1.0])

    # family-a = mean(response-a1=0, response-a2=1) = .5;
    # family-b = 1; global = .75.
    assert hierarchical_mean(values, anchors) == pytest.approx(0.75)


def test_holm_adjustment_preserves_original_keys_and_monotonicity() -> None:
    adjusted = holm_adjust({"large": 0.04, "small": 0.01, "middle": 0.03})

    assert list(adjusted) == ["large", "small", "middle"]
    assert adjusted == {"large": 0.06, "small": 0.03, "middle": 0.06}


def test_pair_tables_preserve_missing_scores() -> None:
    score_names = {
        "width1": PairScore(0.5, 16, None),
        "candidate": PairScore(None, 12, "fewer_than_minimum_common_bases"),
        **{f"rank{rank}": PairScore(0.1, 16, None) for rank in range(1, 6)},
    }
    rows = [
        PairRow(
            anchor_id="anchor",
            anchor_family_id="family",
            anchor_response_id="response",
            phase_bin=0,
            target_id="target",
            target_family_id="family",
            target_response_id="response",
            true_continuation=True,
            scores=score_names,
        )
    ]

    tables = pair_rows_to_tables(rows)

    assert tables.planned[0, 0]
    assert tables.scores["width1"][0, 0] == 0.5
    assert np.isnan(tables.scores["candidate"][0, 0])


def test_cached_table_observed_and_permutation_endpoints_end_to_end() -> None:
    width = [[0.9, None], [0.8, None]]
    candidate = [[0.8, 0.1], [0.2, 0.9]]
    rows = []
    responses = [("response-a", "family-a"), ("response-b", "family-b")]
    for anchor_index, (anchor_response, anchor_family) in enumerate(responses):
        for target_index, (target_response, target_family) in enumerate(responses):
            scores = {
                "width1": PairScore(
                    width[anchor_index][target_index],
                    16,
                    (
                        None
                        if width[anchor_index][target_index] is not None
                        else "fewer_than_minimum_common_bases"
                    ),
                ),
                "candidate": PairScore(candidate[anchor_index][target_index], 16, None),
                **{
                    f"rank{rank}": PairScore(
                        candidate[anchor_index][target_index], 16, None
                    )
                    for rank in range(1, 6)
                },
            }
            rows.append(
                PairRow(
                    anchor_id=f"anchor-{anchor_response}",
                    anchor_family_id=anchor_family,
                    anchor_response_id=anchor_response,
                    phase_bin=0,
                    target_id=f"target-{target_response}",
                    target_family_id=target_family,
                    target_response_id=target_response,
                    true_continuation=target_response == anchor_response,
                    scores=scores,
                )
            )
    tables = pair_rows_to_tables(rows)

    observed, _ = _observed_metrics(tables, rows)
    assert observed["candidate_all"]["zero_filled_mrr"] == 1.0
    assert observed["candidate_rescue"]["subset_anchor_count"] == 1
    assert observed["candidate_rescue"]["zero_filled_mrr"] == 1.0
    assert observed["candidate_rescue_contribution"]["zero_filled_mrr"] == 0.5
    assert observed["width1"]["zero_filled_mrr"] == 0.5
    assert observed["backoff"]["zero_filled_mrr_minus_width1"] == 0.5

    permutations = np.asarray([[0, 1], [1, 0]], dtype=np.int16)
    first, first_families = permutation_inference(tables, permutations, chunk_size=1)
    second, second_families = permutation_inference(tables, permutations, chunk_size=2)
    for endpoint in first:
        np.testing.assert_array_equal(first[endpoint], second[endpoint])
    assert first_families == second_families
    np.testing.assert_allclose(first["candidate_all"], [1.0, 0.5])
    np.testing.assert_allclose(first["candidate_rescue"], [0.5, 0.25])
    np.testing.assert_allclose(first["backoff_delta"], [0.5, 0.0])


def test_module_mode_help_smoke(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts.bonafide.candidate_union_c2_salvage_analysis", "--help"],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(
            "scripts.bonafide.candidate_union_c2_salvage_analysis",
            run_name="__main__",
        )
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--audited-c2-report" in output
    assert "--candidate-union-root" in output
    assert "--protocol" in output
