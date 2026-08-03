from __future__ import annotations

import hashlib
from types import SimpleNamespace

import circuits.analysis.bonafide.candidate_profiles as candidate_profiles_module
import pandas as pd
import pytest
from circuits.analysis.bonafide.candidate_profiles import (
    PARTITION_CAPACITIES,
    CandidateBasisProfile,
    ValidatedTargetProfiles,
    WidthBasisProfile,
    _build_basis_index,
    _crosswalk_summary,
    _partition_hierarchical_weights,
    _redacted_example,
    _validate_audited_artifact_payloads,
    build_family_partitions,
    extract_candidate_profiles,
    extract_width_one_profiles,
    rank_aligned_candidate_contrasts,
)
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.identity import SignedBasisKey
from circuits.tracing.candidates import CandidateLogit, CandidateSelection


def _selection(ranks: list[int], *, observed_rank: int) -> CandidateSelection:
    candidates = []
    for index, rank in enumerate(ranks):
        candidates.append(
            CandidateLogit(
                candidate_index=index,
                full_distribution_rank=rank,
                token_id=100 + index,
                token_text=f"token-{index}",
                logit=10.0 - index,
                probability=0.5 / (index + 1),
                is_observed=index == 0,
            )
        )
    return CandidateSelection(
        policy_id="model_top5_plus_observed",
        policy_version="1",
        ordering_rule="observed_first_then_model_top5",
        observed_token_id=100,
        observed_token_text="token-0",
        observed_token_rank=observed_rank,
        candidates=tuple(candidates),
    )


def test_rank_aligned_contrasts_support_width_five_and_six() -> None:
    assert rank_aligned_candidate_contrasts(
        full_distribution_ranks=[3, 1, 2, 4, 5],
        observed_index=0,
        contribution_values=[10.0, 11.0, 8.0, 15.0, 10.0],
    ) == (1.0, -2.0, 0.0, 5.0, 0.0)
    assert rank_aligned_candidate_contrasts(
        full_distribution_ranks=[7, 1, 2, 3, 4, 5],
        observed_index=0,
        contribution_values=[10.0, 11.0, 8.0, 15.0, 10.0, 4.0],
    ) == (1.0, -2.0, 5.0, 0.0, -6.0)

    with pytest.raises(ValueError, match="ranks one through five"):
        rank_aligned_candidate_contrasts(
            full_distribution_ranks=[6, 1, 2, 3, 4],
            observed_index=0,
            contribution_values=[0.0] * 5,
        )


def test_width_profiles_use_activation_sign_and_sum_occurrences() -> None:
    frame = pd.DataFrame(
        [
            {
                "layer": 1,
                "token": 3,
                "neuron": 7,
                "activation": 2.0,
                "attribution": 0.4,
                "attr_map": [1.0, None, -2.0],
            },
            {
                "layer": 1,
                "token": 4,
                "neuron": 7,
                "activation": 1.0,
                "attribution": -0.1,
                "attr_map": [0.5, 3.0, 2.0],
            },
            {
                "layer": 1,
                "token": 5,
                "neuron": 7,
                "activation": -1.0,
                "attribution": 0.2,
                "attr_map": [-1.0, 1.0, 0.0],
            },
            {
                "layer": 1,
                "token": 6,
                "neuron": 8,
                "activation": 0.0,
                "attribution": 0.9,
                "attr_map": [9.0, 9.0, 9.0],
            },
            {
                "layer": 2,
                "token": 0,
                "neuron": 100,
                "activation": 1.0,
                "attribution": 1.0,
                "attr_map": [1.0, 1.0, 1.0],
            },
        ]
    )

    profiles, zero_count, crosswalk = extract_width_one_profiles(
        frame, model_id="model", model_revision="revision"
    )

    by_polarity = {basis.polarity: value for basis, value in profiles.items()}
    assert set(by_polarity) == {"+", "-"}
    assert by_polarity["+"].values == (1.5, 3.0, 0.0)
    assert by_polarity["+"].support == (True, True, True)
    assert by_polarity["+"].signed_attribution == pytest.approx(0.3)
    assert by_polarity["+"].occurrence_count == 2
    assert by_polarity["-"].values == (-1.0, 1.0, 0.0)
    assert zero_count == 1
    assert crosswalk == {
        "activation_+__attribution_+": 1,
        "activation_+__attribution_-": 1,
        "activation_-__attribution_+": 1,
        "activation_zero__attribution_+": 1,
    }


@pytest.mark.parametrize(
    ("ranks", "contributions", "expected"),
    [
        ([3, 1, 2, 4, 5], [10.0, 11.0, 8.0, 15.0, 10.0], (2.0, -4.0, 0.0, 10.0, 0.0)),
        (
            [7, 1, 2, 3, 4, 5],
            [10.0, 11.0, 8.0, 15.0, 10.0, 4.0],
            (2.0, -4.0, 10.0, 0.0, -12.0),
        ),
    ],
)
def test_candidate_profiles_use_observed_activation_sign_and_sum_occurrences(
    ranks: list[int], contributions: list[float], expected: tuple[float, ...]
) -> None:
    width = len(ranks)
    rows = [
        {
            "layer": 1,
            "token": token,
            "neuron": 9,
            "candidate_activation": [2.0] * width,
            "candidate_attribution": [0.5] * width,
            "candidate_contribution": contributions,
            "applicable_by_candidate": [True] * width,
        }
        for token in (2, 3)
    ]
    rows.extend(
        [
            {
                "layer": 1,
                "token": 4,
                "neuron": 9,
                "candidate_activation": [-1.0] * width,
                "candidate_attribution": [0.5] * width,
                "candidate_contribution": contributions,
                "applicable_by_candidate": [True] * width,
            },
            {
                "layer": 1,
                "token": 5,
                "neuron": 10,
                "candidate_activation": [0.0] * width,
                "candidate_attribution": [-0.5] * width,
                "candidate_contribution": contributions,
                "applicable_by_candidate": [True] * width,
            },
            {
                "layer": 2,
                "token": 0,
                "neuron": 100,
                "candidate_activation": [1.0] * width,
                "candidate_attribution": [0.5] * width,
                "candidate_contribution": contributions,
                "applicable_by_candidate": [True] * width,
            },
        ]
    )
    artifact = SimpleNamespace(
        trace=SimpleNamespace(
            candidate_selection=_selection(ranks, observed_rank=ranks[0]),
            df_node=pd.DataFrame(rows),
        )
    )

    profiles, zero_count, diagnostics = extract_candidate_profiles(
        artifact, model_id="model", model_revision="revision"
    )

    by_polarity = {basis.polarity: value for basis, value in profiles.items()}
    assert set(by_polarity) == {"+", "-"}
    assert by_polarity["+"].values == expected
    assert by_polarity["+"].occurrence_count == 2
    assert by_polarity["-"].values == tuple(value / 2 for value in expected)
    assert zero_count == 1
    assert diagnostics["activation_invariance"]["violation_count"] == 0
    assert diagnostics["activation_invariance"]["max_abs_deviation"] == 0.0
    assert diagnostics[
        "c2_attribution_sign_to_production_activation_sign_crosswalk"
    ] == {
        "activation_+__attribution_+": 2,
        "activation_-__attribution_+": 1,
        "activation_zero__attribution_-": 1,
    }


def test_candidate_profiles_fail_on_hidden_activation_drift() -> None:
    ranks = [1, 2, 3, 4, 5]
    artifact = SimpleNamespace(
        trace=SimpleNamespace(
            candidate_selection=_selection(ranks, observed_rank=1),
            df_node=pd.DataFrame(
                [
                    {
                        "layer": 1,
                        "token": 2,
                        "neuron": 9,
                        "candidate_activation": [1.0, 1.0, 1.0, 1.001, 1.0],
                        "candidate_attribution": [0.5] * 5,
                        "candidate_contribution": [5.0, 4.0, 3.0, 2.0, 1.0],
                        "applicable_by_candidate": [True] * 5,
                    },
                    {
                        "layer": 2,
                        "token": 0,
                        "neuron": 100,
                        "candidate_activation": [1.0] * 5,
                        "candidate_attribution": [1.0] * 5,
                        "candidate_contribution": [1.0] * 5,
                        "applicable_by_candidate": [True] * 5,
                    },
                ]
            ),
        )
    )

    with pytest.raises(ValueError, match="activation invariance failed"):
        extract_candidate_profiles(
            artifact, model_id="model", model_revision="revision"
        )


def test_candidate_invariance_uses_candidate_zero_as_numpy_reference() -> None:
    ranks = [1, 2, 3, 4, 5]
    artifact = SimpleNamespace(
        trace=SimpleNamespace(
            candidate_selection=_selection(ranks, observed_rank=1),
            df_node=pd.DataFrame(
                [
                    {
                        "layer": 1,
                        "token": 2,
                        "neuron": 9,
                        "candidate_activation": [1e6, 1e6 + 1.0000006] + [1e6] * 3,
                        "candidate_attribution": [0.5] * 5,
                        "candidate_contribution": [5.0, 4.0, 3.0, 2.0, 1.0],
                        "applicable_by_candidate": [True] * 5,
                    },
                    {
                        "layer": 2,
                        "token": 0,
                        "neuron": 100,
                        "candidate_activation": [1.0] * 5,
                        "candidate_attribution": [1.0] * 5,
                        "candidate_contribution": [1.0] * 5,
                        "applicable_by_candidate": [True] * 5,
                    },
                ]
            ),
        )
    )

    with pytest.raises(ValueError, match="activation invariance failed"):
        extract_candidate_profiles(
            artifact, model_id="model", model_revision="revision"
        )


def test_candidate_invariance_uses_frozen_near_zero_relative_denominator() -> None:
    ranks = [1, 2, 3, 4, 5]
    artifact = SimpleNamespace(
        trace=SimpleNamespace(
            candidate_selection=_selection(ranks, observed_rank=1),
            df_node=pd.DataFrame(
                [
                    {
                        "layer": 1,
                        "token": 2,
                        "neuron": 9,
                        "candidate_activation": [0.0, 5e-8, 0.0, 0.0, 0.0],
                        "candidate_attribution": [0.0] * 5,
                        "candidate_contribution": [5.0, 4.0, 3.0, 2.0, 1.0],
                        "applicable_by_candidate": [True] * 5,
                    },
                    {
                        "layer": 1,
                        "token": 3,
                        "neuron": 10,
                        "candidate_activation": [1.0] * 5,
                        "candidate_attribution": [1.0] * 5,
                        "candidate_contribution": [5.0, 4.0, 3.0, 2.0, 1.0],
                        "applicable_by_candidate": [True] * 5,
                    },
                    {
                        "layer": 2,
                        "token": 0,
                        "neuron": 100,
                        "candidate_activation": [1.0] * 5,
                        "candidate_attribution": [1.0] * 5,
                        "candidate_contribution": [1.0] * 5,
                        "applicable_by_candidate": [True] * 5,
                    },
                ]
            ),
        )
    )

    _, zero_count, diagnostics = extract_candidate_profiles(
        artifact, model_id="model", model_revision="revision"
    )

    assert zero_count == 1
    assert diagnostics["activation_invariance"]["comparison_count"] == 8
    assert diagnostics["activation_invariance"]["max_relative_deviation"] == (
        pytest.approx(5e4)
    )


def test_family_partition_is_exact_deterministic_and_self_hashed() -> None:
    cases = []
    examples = {}
    for family_index in range(34):
        family_id = f"family-{family_index:02d}"
        response_count = 2 if family_index == 0 else 1
        for response_index in range(response_count):
            response_id = f"response-{family_index:02d}-{response_index}"
            examples[response_id] = {
                "diversity": {
                    "condition": f"condition-{family_index % 4}",
                    "markers": [
                        f"marker-{response_index}",
                        f"group-{family_index % 3}",
                    ],
                }
            }
            cases.extend(
                {
                    "base_question_id": family_id,
                    "example_id": response_id,
                    "phase_bin": phase_bin,
                }
                for phase_bin in range(7)
            )

    first = build_family_partitions(cases, examples_by_response=examples)
    second = build_family_partitions(
        list(reversed(cases)), examples_by_response=examples
    )

    assert first == second
    assert {
        role: len(first["partitions"][role]) for role in PARTITION_CAPACITIES
    } == PARTITION_CAPACITIES
    assert len(first["family_to_partition"]) == 34
    assert first["family_to_partition"]["family-00"] in PARTITION_CAPACITIES
    unhashed = dict(first)
    recorded = unhashed.pop("partitions_sha256")
    assert recorded == canonical_sha256(unhashed)
    expected_order = sorted(
        first["family_to_partition"],
        key=lambda family_id: (
            hashlib.sha256(
                b"candidate-aware-labelability-v1\0" + family_id.encode("utf-8")
            ).hexdigest(),
            family_id,
        ),
    )
    assert first["ordered_families"] == expected_order
    assert first["partitions"]["generation"] == sorted(expected_order[:18])
    assert first["partitions"]["selection_scoring"] == sorted(expected_order[18:26])
    assert first["partitions"]["audit"] == sorted(expected_order[26:])
    assert first["outcome_fields_used"] == []


def test_partition_weights_and_crosswalk_are_hierarchical() -> None:
    rows = [
        {
            "case_id": "a-1",
            "family_partition": "generation",
            "base_question_id": "family-a",
            "response_id": "response-a",
            "candidate_polarity_crosswalk_json": '{"activation_+__attribution_+":2}',
        },
        {
            "case_id": "a-2",
            "family_partition": "generation",
            "base_question_id": "family-a",
            "response_id": "response-a",
            "candidate_polarity_crosswalk_json": '{"activation_+__attribution_-":1}',
        },
    ]
    rows.extend(
        {
            "case_id": f"g-{family_index}",
            "family_partition": "generation",
            "base_question_id": f"family-g-{family_index}",
            "response_id": f"response-g-{family_index}",
            "candidate_polarity_crosswalk_json": (
                '{"activation_zero__attribution_+":1}'
            ),
        }
        for family_index in range(1, 18)
    )
    for partition, count in (("selection_scoring", 8), ("audit", 8)):
        rows.extend(
            {
                "case_id": f"{partition}-{family_index}",
                "family_partition": partition,
                "base_question_id": f"{partition}-family-{family_index}",
                "response_id": f"{partition}-response-{family_index}",
                "candidate_polarity_crosswalk_json": (
                    '{"activation_-__attribution_-":1}'
                ),
            }
            for family_index in range(count)
        )

    weights, diagnostics = _partition_hierarchical_weights(rows)
    weighted_rows = [
        {**row, "partition_hierarchical_weight": weights[row["case_id"]]}
        for row in rows
    ]
    crosswalk = _crosswalk_summary(
        weighted_rows, field="candidate_polarity_crosswalk_json"
    )

    assert weights["a-1"] == pytest.approx(1 / 18 / 2)
    assert weights["a-2"] == pytest.approx(1 / 18 / 2)
    assert diagnostics["generation"]["weight_sum"] == pytest.approx(1.0)
    generation = crosswalk["by_partition"]["generation"]
    assert generation["agreement_summary_counts"]["agreement"] == 2
    assert generation["agreement_summary_counts"]["disagreement"] == 1
    assert generation["agreement_summary_counts"]["zero_or_unsupported"] == 17
    assert generation["agreement_summary_weighted_mass"]["agreement"] == (
        pytest.approx(2 / 18 / 2)
    )


def test_example_redaction_excludes_bonafide_outcomes() -> None:
    redacted = _redacted_example(
        {
            "example_id": "example",
            "prompt": "prompt",
            "response": "response",
            "hint_types": ["hint"],
            "label_types": ["UNFAITHFUL_COT"],
            "labeling_reasons": ["outcome"],
            "annotation_row_ids": ["row"],
            "diversity": {
                "cot_phenotype": "commission",
                "response_length_bin": "129-224",
            },
        }
    )

    assert redacted == {
        "example_id": "example",
        "hint_types": ["hint"],
        "prompt": "prompt",
        "response": "response",
        "diversity": {"response_length_bin": "129-224"},
    }


def test_basis_index_is_canonical_and_view_explicit() -> None:
    basis_a = SignedBasisKey("model", "revision", 1, 9, "+")
    basis_b = SignedBasisKey("model", "revision", 2, 3, "-")
    profiles = [
        ValidatedTargetProfiles(
            target={
                "case_id": "case-a",
                "base_question_id": "family-a",
                "response_id": "response-a",
            },
            width={basis_b: WidthBasisProfile((1.0,), (True,), 1.0, 1)},
            candidate={
                basis_a: CandidateBasisProfile((1.0, 0.0, 0.0, 0.0, 0.0), 1),
                basis_b: CandidateBasisProfile((0.0, 0.0, 0.0, 0.0, 0.0), 1),
            },
        ),
        ValidatedTargetProfiles(
            target={
                "case_id": "case-b",
                "base_question_id": "family-b",
                "response_id": "response-b",
            },
            width={basis_a: WidthBasisProfile((2.0,), (True,), 2.0, 1)},
            candidate={basis_a: CandidateBasisProfile((0.0, 1.0, 0.0, 0.0, 0.0), 1)},
        ),
    ]

    rows, mapping = _build_basis_index(
        profiles, {"family-a": "generation", "family-b": "audit"}
    )

    assert mapping == {basis_a: 0, basis_b: 1}
    assert [row["signed_basis_index"] for row in rows] == [0, 1]
    assert rows[0]["in_width_view"] is True
    assert rows[0]["in_candidate_view"] is True
    assert rows[0]["candidate_generation_target_count"] == 1
    assert rows[1]["in_candidate_support"] is True
    assert rows[1]["in_candidate_view"] is False


def test_loaded_payloads_must_match_audited_target_and_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = []
    diagnostics = []
    payload_records = []
    for index in range(2):
        target = {
            "case_id": f"case-{index}",
            "source_width1_artifact_id": f"source-{index}",
            "base_question_id": f"family-{index}",
            "response_id": f"response-{index}",
            "phase_bin": index,
            "width1_artifact_id": f"width-{index}",
            "width1_payload_sha256": f"width-payload-{index}",
            "candidate_union_artifact_id": f"candidate-{index}",
            "candidate_union_payload_sha256": f"candidate-payload-{index}",
            "candidate_union_topology_sha256": f"topology-{index}",
        }
        targets.append(ValidatedTargetProfiles(target, {}, {}))
        diagnostics.append(
            {
                **target,
                "example_id": target["response_id"],
            }
        )
        diagnostics[-1].pop("response_id")
        payload_records.append(
            {
                "source_width1_artifact_id": target["source_width1_artifact_id"],
                "width1_payload_sha256": target["width1_payload_sha256"],
                "candidate_union_payload_sha256": target[
                    "candidate_union_payload_sha256"
                ],
            }
        )
    expected_hash = canonical_sha256(payload_records)
    monkeypatch.setattr(
        candidate_profiles_module,
        "FROZEN_ARTIFACT_PAYLOAD_SET_SHA256",
        expected_hash,
    )

    assert (
        _validate_audited_artifact_payloads(
            targets, {"target_diagnostics": diagnostics}
        )
        == expected_hash
    )
    diagnostics[0]["width1_payload_sha256"] = "drifted"
    with pytest.raises(ValueError, match="differs from audited"):
        _validate_audited_artifact_payloads(
            targets, {"target_diagnostics": diagnostics}
        )
