from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest
from circuits.tracing.candidates import CandidateSelection, build_joint_objective
from circuits.tracing.topology_comparison import (
    compare_joint_to_independent_candidates,
)
from tests.test_teacher_forced_trace import _topk_trace


def _node_rows(candidate_ids, contribution):
    rows = [
        {
            "layer": -1,
            "token": 0,
            "neuron": 11,
            "attribution": 0.2,
            "activation": 1.0,
            "attr_map": [1.0],
            "contrib_map": contribution,
            "label": "row___0",
        },
        {
            "layer": 0,
            "token": 0,
            "neuron": 10,
            "attribution": 0.5,
            "activation": 2.0,
            "attr_map": [0.5],
            "contrib_map": contribution,
            "label": "row___0",
        },
    ]
    rows.extend(
        {
            "layer": 1,
            "token": 4,
            "neuron": token_id,
            "attribution": 0.3,
            "activation": 3.0,
            "attr_map": [0.3],
            "contrib_map": [
                1.0 if index == candidate_index else 0.0
                for index in range(len(candidate_ids))
            ],
            "label": "row___0",
        }
        for candidate_index, token_id in enumerate(candidate_ids)
    )
    return pd.DataFrame(rows)


def _edge(source, target, attribution=0.2):
    return {
        "layer": f"{source[0]}->{target[0]}",
        "token": f"{source[1]}->{target[1]}",
        "neuron": f"{source[2]}->{target[2]}",
        "attribution": attribution,
        "weight": 0.5,
        "label": "row___0",
    }


def _joint_and_references():
    joint = deepcopy(_topk_trace())
    candidate_ids = [
        candidate.token_id for candidate in joint.candidate_selection.candidates
    ]
    joint.circuit_data.df_node = _node_rows(
        candidate_ids, [1.0, -1.0, 0.5, 0.25, -0.25]
    )
    embed = (-1, 0, 11)
    mlp = (0, 0, 10)
    joint_edges = [_edge(embed, mlp)]
    joint_edges.extend(
        _edge(mlp, (1, 4, token_id))
        for token_id in candidate_ids
        if token_id != candidate_ids[1]
    )
    joint.circuit_data.df_edge = pd.DataFrame(joint_edges)

    references = []
    for candidate in joint.candidate_selection.candidates:
        data = deepcopy(joint.circuit_data)
        selected = replace(candidate, candidate_index=0)
        policy_id = "observed_token" if candidate.is_observed else "specified_token"
        selection = CandidateSelection(
            policy_id=policy_id,
            policy_version="1",
            ordering_rule="descending_logit_then_ascending_token_id",
            observed_token_id=joint.candidate_selection.observed_token_id,
            observed_token_text=joint.candidate_selection.observed_token_text,
            observed_token_rank=joint.candidate_selection.observed_token_rank,
            candidates=(selected,),
        )
        objective = build_joint_objective("raw_logit_sum", (selected,))
        contribution_schema = {
            **joint.candidate_contribution_schema,
            "width": 1,
        }
        data.df_node = _node_rows([candidate.token_id], [candidate.logit])
        data.df_edge = pd.DataFrame(
            [_edge(embed, mlp), _edge(mlp, (1, 4, candidate.token_id))]
        )
        reference = replace(
            joint,
            circuit_data=data,
            candidate_selection=selection,
            joint_objective=objective,
            candidate_contribution_schema=contribution_schema,
        )
        data.trace_metadata["candidate_trace_contract"] = reference.contract_dict()
        references.append(reference)
    joint.circuit_data.trace_metadata["candidate_trace_contract"] = (
        joint.contract_dict()
    )
    return joint, references, candidate_ids


def test_c0_comparison_reports_recall_sign_conflict_and_omitted_path() -> None:
    joint, references, candidate_ids = _joint_and_references()

    report = compare_joint_to_independent_candidates(joint, references)

    assert report["union_node_recall"] == 1.0
    assert report["union_edge_recall"] == 5 / 6
    diagnostics = report["candidate_profile_diagnostics"]
    assert diagnostics["sign_conflict_row_count"] == 1
    assert diagnostics["sign_conflict_rate"] == 1.0
    omitted = next(
        candidate
        for candidate in report["candidates"]
        if candidate["token_id"] == candidate_ids[1]
    )
    assert omitted["joint_edge_recall"] == 0.5
    assert omitted["path_recall"] == 0.0
    assert omitted["omitted_path_count"] == 1
    assert omitted["omitted_path_witnesses"] == [
        [[-1, 0, 11], [0, 0, 10], [1, 4, candidate_ids[1]]]
    ]


def test_c0_comparison_rejects_negative_witness_limit() -> None:
    joint, references, _candidate_ids = _joint_and_references()

    with pytest.raises(ValueError, match="non-negative integer"):
        compare_joint_to_independent_candidates(
            joint,
            references,
            max_omitted_path_witnesses_per_candidate=-1,
        )
