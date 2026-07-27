from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from circuits.tracing.parity import compare_observed_token_k1
from tests.test_teacher_forced_trace import _topk_trace


def _legacy_and_candidate():
    candidate = _topk_trace()
    first = candidate.candidate_selection.candidates[0]
    selection = replace(
        candidate.candidate_selection,
        policy_id="observed_token",
        candidates=(first,),
    )
    objective = replace(
        candidate.joint_objective,
        formula="logit[candidate_0]",
        candidate_weights=(1.0,),
    )
    contribution_schema = {
        **candidate.candidate_contribution_schema,
        "width": 1,
    }
    candidate = replace(
        candidate,
        circuit_data=deepcopy(candidate.circuit_data),
        candidate_selection=selection,
        joint_objective=objective,
        candidate_contribution_schema=contribution_schema,
    )
    candidate.circuit_data.df_node["contrib_map"] = [[0.1]]
    candidate.circuit_data.trace_metadata["candidate_trace_contract"] = (
        candidate.contract_dict()
    )
    legacy = deepcopy(candidate.circuit_data)
    legacy.trace_metadata = {"trace_mode": "teacher_forced_response"}
    return legacy, candidate


def test_observed_token_k1_parity_passes_after_canonical_row_sort() -> None:
    legacy, candidate = _legacy_and_candidate()
    legacy.df_node = legacy.df_node.iloc[::-1]
    legacy.df_edge = legacy.df_edge.iloc[::-1]

    report = compare_observed_token_k1(legacy, candidate)

    assert report.passed is True
    assert report.mismatches == ()
    report.require_pass()


def test_observed_token_k1_parity_reports_numerical_drift() -> None:
    legacy, candidate = _legacy_and_candidate()
    candidate.circuit_data.df_node.loc[0, "attribution"] += 0.1

    report = compare_observed_token_k1(legacy, candidate)

    assert report.passed is False
    assert any("node table mismatch" in mismatch for mismatch in report.mismatches)
    with pytest.raises(AssertionError, match="k=1 parity failed"):
        report.require_pass()


def test_observed_token_k1_parity_rejects_wrong_policy() -> None:
    legacy, candidate = _legacy_and_candidate()
    object.__setattr__(candidate.candidate_selection, "policy_id", "model_top5")

    with pytest.raises(ValueError, match="observed_token"):
        compare_observed_token_k1(legacy, candidate)
