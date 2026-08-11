from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pytest
from circuits.analysis.bonafide import candidate_identity_assessment as assessment
from circuits.analysis.bonafide.candidate_identity_assessment import (
    CandidateEvent as Event,
)
from circuits.analysis.bonafide.candidate_identity_assessment import (
    ProjectedRow,
    SourceRow,
    SparseCentroids,
    _hierarchical_scores,
    _null_blocks,
    _parse_target_events,
    _seed,
    _validate_target_phase_grid,
    assemble_local_view,
    assemble_motif_view,
    average_tie_reciprocal_rank,
    build_candidate_identity_assessment,
    evaluate_projected_views,
    surface_relation_key,
)
from circuits.analysis.bonafide.candidate_identity_source import (
    EXPOSURE_CONTRACT,
    PROFILE_SCHEMA,
    SOURCE_SCHEMA_VERSION,
    TARGET_SCHEMA,
)


def _source(
    *,
    family: str = "family",
    response: str = "response",
    target: str = "target",
    basis: int = 0,
    cluster: int = 0,
    phase: int = 0,
    events: tuple[Event, ...] = (Event(1, 11, " alpha", 1.0),),
) -> SourceRow:
    return SourceRow(
        family_id=family,
        response_id=response,
        target_id=target,
        basis_index=basis,
        cluster_id=cluster,
        layer=2,
        polarity="+",
        phase=phase,
        observed_token_id=10,
        observed_token_text="Alpha",
        events=events,
    )


def _projected(
    family: str,
    response: str,
    target: str,
    basis: int,
    vector: dict[int, float] | None,
    *,
    cluster: int = 0,
    phase: int = 0,
) -> ProjectedRow:
    return ProjectedRow(
        family_id=family,
        response_id=response,
        target_id=target,
        basis_index=basis,
        cluster_id=cluster,
        layer=2,
        polarity="+",
        phase=phase,
        vector=vector,
        support={} if vector is None else dict.fromkeys(vector, 1.0),
    )


def _selection(observed_rank: int = 2) -> tuple[dict[str, object], list[float]]:
    candidates = []
    vector = []
    for rank in range(1, 6):
        observed = rank == observed_rank
        candidates.append(
            {
                "candidate_index": rank - 1,
                "full_distribution_rank": rank,
                "token_id": 100 + rank,
                "token_text": f"token-{rank}",
                "logit": 1.0 / rank,
                "probability": 0.1,
                "is_observed": observed,
            }
        )
        vector.append(0.0 if observed else float(rank))
    selection = {
        "policy_id": "model_top5_plus_observed",
        "policy_version": "1",
        "ordering_rule": "rank",
        "observed_token_id": 100 + observed_rank,
        "observed_token_text": f"token-{observed_rank}",
        "observed_token_rank": observed_rank,
        "candidates": candidates,
    }
    target = {
        "observed_token_id": 100 + observed_rank,
        "observed_token_text": f"token-{observed_rank}",
        "candidate_selection_json": json.dumps(selection),
    }
    return target, vector


def test_unicode_surface_key_is_deterministic_and_prefix_precedes_suffix() -> None:
    # NFKC/casefold preserves a Unicode-aware letter class and leading-space relation.
    assert surface_relation_key(" ÅBÅ ", "åbå") == (
        "surface",
        "letters",
        "letters",
        "equal",
        "different",
    )
    # "aba" is both a prefix and suffix of "ababa"; prefix must win.
    assert surface_relation_key("aba", "ababa")[3] == "prefix"
    assert surface_relation_key("１２", "12")[3] == "equal"


def test_observed_top_five_slot_is_omitted_from_values_and_support() -> None:
    target, vector = _selection(observed_rank=2)
    events = _parse_target_events(target, vector)
    assert [event.rank for event in events] == [1, 3, 4, 5]
    row = _source(events=events)
    dictionary, generation, selection = assemble_local_view([row], [row], "R")
    assert len(dictionary) == 4
    assert sum(generation[0].support.values()) == 4.0
    assert generation[0].vector is not None
    assert selection[0].vector == generation[0].vector

    vector[1] = 0.25
    with pytest.raises(ValueError, match="structural zero"):
        _parse_target_events(target, vector)


def test_duplicate_surface_keys_sum_values_and_count_support_events() -> None:
    row = _source(
        events=(
            Event(1, 11, "beta", 2.0),
            Event(2, 12, "gamma", -2.0),
        )
    )
    dictionary, generation, _ = assemble_local_view([row], [row], "SR")
    assert len(dictionary) == 1
    assert generation[0].vector is None
    assert generation[0].support == {0: 2.0}


def test_generation_dictionary_drops_selection_only_keys() -> None:
    generation = _source(events=(Event(1, 11, "a", 1.0),))
    selection = _source(events=(Event(1, 99, "z", 3.0),))
    dictionary, _, projected = assemble_local_view([generation], [selection], "T")
    assert len(dictionary) == 1
    assert projected[0].vector is None
    assert projected[0].support == {}


def _phase_rows(*, recurrent: bool = True) -> list[SourceRow]:
    rows = []
    for phase in range(7):
        token = 77 if recurrent or phase != 1 else 88
        rows.append(
            _source(
                target=f"case-{phase}",
                phase=phase,
                events=(Event(1, token, f"token-{token}", float(phase + 1)),),
            )
        )
    return rows


def test_motif_is_anchored_only_to_left_endpoint_and_phase_six_is_missing() -> None:
    rows = _phase_rows()
    dictionary, generation, selection = assemble_motif_view(rows, rows)
    assert len(dictionary) == 12  # six adjacent pairs, left and right coordinate
    for phase in range(6):
        row = generation[phase]
        assert row.target_id == f"case-{phase}"
        assert row.right_target_id == f"case-{phase + 1}"
        assert row.vector is not None
        assert len(row.vector) == 2
    assert generation[6].target_id == "case-6"
    assert generation[6].right_target_id is None
    assert generation[6].vector is None
    assert selection[6].support == {}


def test_motif_requires_exact_phase_grid_and_exact_recurrence() -> None:
    targets = [
        {"response_id": "response", "case_id": f"case-{phase}", "phase_bin": phase}
        for phase in range(7)
    ]
    _validate_target_phase_grid(targets)
    with pytest.raises(ValueError, match="exactly one target per phase"):
        _validate_target_phase_grid(targets[:-1])
    _, generation, _ = assemble_motif_view(_phase_rows(recurrent=False), _phase_rows())
    assert generation[0].vector is None


def test_zero_filled_hierarchy_weights_basis_inside_target_equally() -> None:
    rows = [
        _projected("a", "r1", "t1", 0, {0: 1.0}),
        _projected("a", "r1", "t1", 1, {0: 1.0}),
        _projected("a", "r1", "t2", 0, {0: 1.0}),
        _projected("b", "r2", "t3", 0, {0: 1.0}),
    ]
    overall, families = _hierarchical_scores(rows, [1.0, 0.0, 1.0, 0.0])
    assert families == pytest.approx({"a": 0.75, "b": 0.0})
    assert overall == pytest.approx(0.375)


def test_average_tie_rank_and_missing_cases() -> None:
    centroids = SparseCentroids(
        values={0: {0: 1.0}, 1: {0: 1.0}, 2: {0: -1.0}},
        available=frozenset({0, 1, 2}),
        cluster_reports=(),
    )
    # True cluster ties one competitor at first: average rank 1.5.
    assert average_tie_reciprocal_rank({0: 2.0}, 0, centroids) == pytest.approx(2 / 3)
    assert average_tie_reciprocal_rank(None, 0, centroids) == 0.0
    assert average_tie_reciprocal_rank({0: 1.0}, 9, centroids) == 0.0
    lone = SparseCentroids({0: {0: 1.0}}, frozenset({0}), ())
    assert average_tie_reciprocal_rank({0: 1.0}, 0, lone) == 0.0


def test_null_effectiveness_fails_closed_for_small_strata() -> None:
    rows = [
        _projected("family", "response", "target", basis, {0: 1.0, 1: float(basis + 1)})
        for basis in range(3)
    ]
    blocks, report = _null_blocks(rows, "T")
    assert blocks == []
    assert report["eligible_row_fraction"] == 0.0
    assert not report["effective"]


def test_null_stratifies_on_full_support_including_zero_scientific_coordinates() -> (
    None
):
    rows = [
        _projected(
            "family",
            "response",
            "target",
            basis,
            {0: float(basis + 1)},
        )
        for basis in range(4)
    ]
    rows[0].support[1] = 1.0
    rows[1].support[1] = 1.0
    rows[2].support[2] = 1.0
    rows[3].support[2] = 1.0
    blocks, report = _null_blocks(rows, "T")
    assert blocks == []
    assert not report["effective"]


def test_null_mass_blocks_include_left_endpoint_phase_for_m() -> None:
    rows = [
        _projected(
            "family",
            "response",
            "target",
            basis,
            {0: 1.0, 1: float(basis + 1)},
            phase=0,
        )
        for basis in range(4)
    ]
    blocks, report = _null_blocks(rows, "M")
    assert blocks == [[0, 1, 2, 3]]
    assert report["eligible_hierarchical_weight"] == pytest.approx(1.0)
    assert report["effective"]


def test_seed_byte_recipe_and_joint_null_gate_fail_closed() -> None:
    protocol = "a" * 64
    replicate = 17
    expected = int.from_bytes(
        hashlib.sha256(
            protocol.encode()
            + b"\0candidate-identity-direction-null-v1\0T\0"
            + replicate.to_bytes(8, "big")
        ).digest()[:8],
        "big",
    )
    assert (
        _seed(
            protocol,
            "candidate-identity-direction-null-v1",
            "T",
            replicate,
        )
        == expected
    )

    generation = []
    for family_index in range(18):
        for basis in range(4):
            cluster = basis % 2
            generation.append(
                _projected(
                    f"g{family_index}",
                    f"gr{family_index}",
                    f"gt{family_index}",
                    basis,
                    {
                        0: 1.0 if cluster == 0 else -1.0,
                        1: 0.1 + basis / 100,
                    },
                    cluster=cluster,
                )
            )
    selection = []
    for family_index in range(8):
        for basis in range(4):
            cluster = basis % 2
            selection.append(
                _projected(
                    f"s{family_index}",
                    f"sr{family_index}",
                    f"st{family_index}",
                    basis,
                    {0: 1.0 if cluster == 0 else -1.0, 1: 0.1},
                    cluster=cluster,
                )
            )
    views = {
        view: (list(generation), list(selection)) for view in ("R", "T", "P", "SR", "M")
    }
    report = evaluate_projected_views(
        views,
        protocol_sha256=protocol,
        provenance_valid=False,
    )
    assert report["direction_null"]["all_variants_effective"]
    assert report["direction_null"]["replicates"] == 100
    assert report["offline_winner"] is None
    assert report["local_labeling_winner"] is None
    assert not report["labeling_authorized"]
    assert all(not gate["passed"] for gate in report["gates"].values())
    assert all(
        not gate["conditions"]["provenance_valid"] for gate in report["gates"].values()
    )


def test_no_overwrite_fails_before_source_or_revision_access(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        build_candidate_identity_assessment(
            source_root=tmp_path / "missing-source",
            output_root=output,
            repo_root=tmp_path / "missing-repo",
        )


def test_evaluator_loads_only_published_source_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_tables = {}
    profile_tables = {}
    for partition in assessment.PARTITIONS:
        selection, vector = _selection()
        target_rows = []
        profile_rows = []
        for phase in range(7):
            case_id = f"{partition}-{phase}"
            target_rows.append(
                {
                    "case_id": case_id,
                    "source_width1_artifact_id": f"source-{case_id}",
                    "candidate_union_artifact_id": f"union-{case_id}",
                    "candidate_union_payload_sha256": "1" * 64,
                    "candidate_union_topology_sha256": "2" * 64,
                    "base_question_id": f"family-{partition}",
                    "response_id": f"response-{partition}",
                    "phase_bin": phase,
                    "response_position": phase,
                    "family_partition": partition,
                    "partition_hierarchical_weight": 1.0,
                    **selection,
                }
            )
            profile_rows.append(
                {
                    "case_id": case_id,
                    "signed_basis_index": phase,
                    "model_id": "model",
                    "model_revision": "revision",
                    "layer": 2,
                    "neuron_index": phase,
                    "polarity": "+",
                    "c2_w64_assigned": True,
                    "c2_w64_cluster_id": phase % 2,
                    "candidate_contrast_vector": vector,
                    "candidate_occurrence_count": 1,
                    "family_partition": partition,
                }
            )
        target_tables[partition] = pa.Table.from_pylist(
            target_rows, schema=TARGET_SCHEMA
        )
        profile_tables[partition] = pa.Table.from_pylist(
            profile_rows, schema=PROFILE_SCHEMA
        )
    manifest = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "exposure_contract": EXPOSURE_CONTRACT,
        "cluster_state": {
            "identifier": "c2_w64",
            "view": "W",
            "n_clusters": 64,
            "assignment": "medoid_seed",
        },
    }
    loaded_roots = []
    monkeypatch.setattr(
        assessment,
        "load_candidate_identity_source",
        lambda root: (
            loaded_roots.append(root) or manifest,
            target_tables,
            profile_tables,
        ),
    )
    monkeypatch.setattr(
        assessment,
        "EXPECTED_TARGET_COUNTS",
        {"generation": 7, "selection_scoring": 7},
    )
    monkeypatch.setattr(assessment, "GENERATION_FAMILY_COUNT", 1)
    monkeypatch.setattr(assessment, "SELECTION_FAMILY_COUNT", 1)

    loaded, generation, selection = assessment._load_source_rows(tmp_path)

    assert loaded is manifest
    assert loaded_roots == [tmp_path]
    assert len(generation) == len(selection) == 7
    assert all(len(row.events) == 4 for row in generation + selection)
