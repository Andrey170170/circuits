from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from circuits.analysis.bonafide.candidate_multiplex_assessment import (
    TARGET_BASIS_ASSESSMENT_SCHEMA,
    TARGET_CROSSWALK_SCHEMA,
    LoadedCandidateMultiplexAssessment,
)
from circuits.analysis.bonafide.candidate_multiplex_diagnostics import (
    compute_candidate_multiplex_diagnostics,
)


def _row(
    index: int,
    *,
    partition: str,
    case: str,
    family: str,
    response: str,
    basis: int,
    cluster: int,
    vector: list[float],
    phase: int = 0,
) -> dict[str, object]:
    return {
        "assessment_row_index": index,
        "case_id": case,
        "signed_basis_index": basis,
        "model_id": "model",
        "model_revision": "revision",
        "layer": 1,
        "neuron_index": basis,
        "polarity": "positive",
        "c2_w64_assigned": True,
        "c2_w64_cluster_id": cluster,
        "family_partition": partition,
        "base_question_id": family,
        "response_id": response,
        "phase_bin": phase,
        "response_position": phase,
        "partition_hierarchical_weight": 0.125,
        "width_profile_available": False,
        "width_signed_attribution": None,
        "width_attribution_profile": None,
        "width_attribution_support": None,
        "width_occurrence_count": None,
        "candidate_profile_available": True,
        "candidate_contrast_vector": vector,
        "candidate_profile_l2_norm": 1.0,
        "candidate_occurrence_count": 1,
        "candidate_measurement_scope": "target_basis_signed_sum",
        "dense_target_match": False,
        "dense_basis_match": False,
        "dense_target_basis_occurrence_match": False,
        "dense_atlas_trace_index": None,
        "dense_signed_basis_index": None,
        "dense_occurrence_count": 0,
        "missing_reasons": ["dense_target_unmatched"],
    }


def _assessment(
    *, reverse_selection: bool = False
) -> LoadedCandidateMultiplexAssessment:
    rows: list[dict[str, object]] = []
    index = 0
    directions = ([1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0])
    for target in range(8):
        family = f"g{target // 2}"
        response = f"gr{target // 2}"
        for basis in range(10):
            rows.append(
                _row(
                    index,
                    partition="generation",
                    case=f"gcase{target}",
                    family=family,
                    response=response,
                    basis=basis,
                    cluster=0,
                    vector=list(directions[basis % 2]),
                    phase=target % 7,
                )
            )
            index += 1
        for basis in (10, 11):
            rows.append(
                _row(
                    index,
                    partition="generation",
                    case=f"gcase{target}",
                    family=family,
                    response=response,
                    basis=basis,
                    cluster=1,
                    vector=[-1.0, 0.0, 0.0, 0.0, 0.0],
                    phase=target % 7,
                )
            )
            index += 1
    for target in range(8):
        vectors = directions[::-1] if reverse_selection else directions
        for basis in range(4):
            rows.append(
                _row(
                    index,
                    partition="selection_scoring",
                    case=f"scase{target}",
                    family=f"s{target}",
                    response=f"sr{target}",
                    basis=basis,
                    cluster=0,
                    vector=list(vectors[basis % 2]),
                )
            )
            index += 1
        for basis in (10, 11):
            rows.append(
                _row(
                    index,
                    partition="selection_scoring",
                    case=f"scase{target}",
                    family=f"s{target}",
                    response=f"sr{target}",
                    basis=basis,
                    cluster=1,
                    vector=[-1.0, 0.0, 0.0, 0.0, 0.0],
                )
            )
            index += 1
    # Deliberately malformed rank width: audit content must remain uninspected.
    rows.append(
        _row(
            index,
            partition="audit",
            case="audit",
            family="audit",
            response="audit",
            basis=0,
            cluster=0,
            vector=[1.0],
        )
    )
    table = pa.Table.from_pylist(rows, schema=TARGET_BASIS_ASSESSMENT_SCHEMA)
    return LoadedCandidateMultiplexAssessment(
        root=Path("/assessment"),
        manifest={
            "manifest_sha256": "a" * 64,
            "overlap": {
                "c2_target_count": 17,
                "dense_matched_target_count": 0,
                "dense_unmatched_target_count": 17,
            },
            "coverage_metrics": {
                "by_family_partition": {
                    "generation": {"target_count": 8},
                    "selection_scoring": {"target_count": 8},
                    "audit": {"target_count": 1},
                }
            },
        },
        target_crosswalk=pa.Table.from_pylist([], schema=TARGET_CROSSWALK_SCHEMA),
        target_basis_assessment=table,
        occurrence_projection=None,
    )


def test_diagnostics_preserve_firewalls_and_report_all_requested_views() -> None:
    report = compute_candidate_multiplex_diagnostics(_assessment())

    assert report["policy"] == {
        "fit_partition": "generation",
        "decision_partition": "selection_scoring",
        "audit_rows_used": False,
        "dense_overlap_used_for_fitting_or_decision": False,
        "primary_state": "c2_w64",
        "candidate_measurement_scope": "target_basis_signed_sum",
    }
    assert set(report["partitions"]) == {"generation", "selection_scoring"}
    assert report["coverage"] == {
        "source_overlap_provenance_only": {
            "c2_target_count": 17,
            "dense_matched_target_count": 0,
            "dense_unmatched_target_count": 17,
        },
        "by_family_partition": {
            "generation": {"target_count": 8},
            "selection_scoring": {"target_count": 8},
        },
        "audit_partition_excluded": True,
    }
    generation = report["partitions"]["generation"]
    assert generation["basis_direction_consistency"]["median"] == 1.0
    assert len(generation["rank_absolute_mass"]["probabilities"]) == 5
    assert len(generation["clusters"][0]["phase_target_mass"]["probabilities"]) == 7


def test_eligibility_gate_does_not_promote_an_alternative() -> None:
    report = compute_candidate_multiplex_diagnostics(_assessment())
    decision = report["decision"]

    assert decision["eligible_for_refinement_fit"] is True
    assert decision["status"] == "eligible_for_constrained_refinement_fit"
    assert decision["primary_state"] == "c2_w64"
    assert decision["alternative_state"] is None
    assert decision["promotion_claimed"] is False
    assert decision["theoretically_splittable_parent_ids"] == [0]


def test_selection_separation_is_a_required_gate() -> None:
    assessment = _assessment()
    rows = assessment.target_basis_assessment.to_pylist()
    for row in rows:
        if row["family_partition"] == "selection_scoring":
            row["candidate_contrast_vector"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    failed = LoadedCandidateMultiplexAssessment(
        root=assessment.root,
        manifest=assessment.manifest,
        target_crosswalk=assessment.target_crosswalk,
        target_basis_assessment=pa.Table.from_pylist(
            rows, schema=TARGET_BASIS_ASSESSMENT_SCHEMA
        ),
        occurrence_projection=None,
    )
    decision = compute_candidate_multiplex_diagnostics(failed)["decision"]

    assert decision["eligible_for_refinement_fit"] is False
    assert decision["status"] == "no_justified_candidate_refinement"
