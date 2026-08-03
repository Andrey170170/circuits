"""Focused tests for signed identities, partitioning, and target multiplexes."""

from __future__ import annotations

import math

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.features import (
    build_profile_observations,
    cluster_fully_supported_profiles,
    pairwise_profile_similarity,
)
from circuits.analysis.bonafide.identity import (
    OccurrenceKey,
    SignedBasisKey,
    basis_key_from_raw_node,
    build_circuit_input_refs,
    occurrence_key_from_raw_node,
)
from circuits.analysis.bonafide.index import build_atlas_index
from circuits.analysis.bonafide.multiplex import (
    OccurrenceEdge,
    ResponseTimeMultiplex,
    build_target_slice,
    validate_causal_path,
    validate_target_slice_round_trip,
)
from circuits.analysis.bonafide.partition import (
    AnalysisTarget,
    CorpusRole,
    hierarchical_fit_weights,
)

MODEL_ID = "fake/model"
MODEL_REVISION = "exact-revision"


def _node(
    *,
    layer: int,
    token: int,
    neuron: int,
    activation: float,
    attribution: float = 0.2,
    attr_map: list[float | None] | None = None,
    label: str = "response___0",
) -> dict:
    return {
        "layer": layer,
        "token": token,
        "neuron": neuron,
        "activation": activation,
        "attribution": attribution,
        "attr_map": attr_map or [0.1, -0.2],
        "contrib_map": [0.3],
        "label": label,
    }


def _edge(
    source: tuple[int, int, int],
    target: tuple[int, int, int],
    *,
    label: str = "response___0",
) -> dict:
    return {
        "layer": f"{source[0]}->{target[0]}",
        "token": f"{source[1]}->{target[1]}",
        "neuron": f"{source[2]}->{target[2]}",
        "attribution": 0.4,
        "weight": 0.5,
        "label": label,
    }


def test_signed_identity_preserves_polarity_position_trace_and_boundaries() -> None:
    positive = _node(layer=3, token=4, neuron=5, activation=1.0)
    negative = _node(layer=3, token=4, neuron=5, activation=-1.0)
    positive_basis = basis_key_from_raw_node(
        positive,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )
    negative_basis = basis_key_from_raw_node(
        negative,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )
    assert positive_basis != negative_basis
    assert SignedBasisKey.from_record(positive_basis.to_record()) == positive_basis

    first = occurrence_key_from_raw_node(positive, trace_unit_id="trace-a")
    moved = occurrence_key_from_raw_node(
        {**positive, "token": 9},
        trace_unit_id="trace-a",
    )
    other_trace = occurrence_key_from_raw_node(positive, trace_unit_id="trace-b")
    assert len({first, moved, other_trace}) == 3
    assert OccurrenceKey.from_record(first.to_record()) == first

    embed = basis_key_from_raw_node(
        _node(layer=-1, token=0, neuron=0, activation=0.0),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )
    unembed = basis_key_from_raw_node(
        _node(layer=36, token=8, neuron=77, activation=2.0),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )
    assert (embed.layer, embed.polarity) == (-1, "+")
    assert unembed.layer == 36

    with pytest.raises(ValueError, match="conflicts"):
        basis_key_from_raw_node(
            {**positive, "polarity": "-"},
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
        )


def test_circuit_input_reindex_is_explicit_and_deterministic() -> None:
    refs = build_circuit_input_refs(
        [
            ("trace-b", 0, "response-b"),
            ("trace-a", 1, "response-a-1"),
            ("trace-a", 0, "response-a-0"),
        ]
    )
    assert [
        (
            ref.trace_unit_id,
            ref.local_ci_index,
            ref.local_label,
            ref.global_atlas_ci_index,
        )
        for ref in refs
    ] == [
        ("trace-a", 0, "response-a-0", 0),
        ("trace-a", 1, "response-a-1", 1),
        ("trace-b", 0, "response-b", 2),
    ]


def test_hierarchical_weights_and_holdout_firewall() -> None:
    records = [
        AnalysisTarget(
            source_artifact_id="a0",
            base_question_id="family-a",
            response_id="response-a0",
            response_position=0,
            corpus_role=CorpusRole.DENSE_DISCOVERY,
            cluster_fit_eligible=True,
        ),
        AnalysisTarget(
            source_artifact_id="a1",
            base_question_id="family-a",
            response_id="response-a0",
            response_position=1,
            corpus_role=CorpusRole.DENSE_DISCOVERY,
            cluster_fit_eligible=True,
        ),
        AnalysisTarget(
            source_artifact_id="a2",
            base_question_id="family-a",
            response_id="response-a1",
            response_position=0,
            corpus_role=CorpusRole.BROAD_DISCOVERY,
            cluster_fit_eligible=True,
        ),
        AnalysisTarget(
            source_artifact_id="b0",
            base_question_id="family-b",
            response_id="response-b0",
            response_position=0,
            corpus_role=CorpusRole.DENSE_DISCOVERY,
            cluster_fit_eligible=True,
        ),
    ]
    weights = hierarchical_fit_weights(records)
    assert weights == {
        "a0": 0.125,
        "a1": 0.125,
        "a2": 0.25,
        "b0": 0.5,
    }
    assert math.isclose(sum(weights.values()), 1.0)

    holdout = AnalysisTarget(
        source_artifact_id="h0",
        base_question_id="family-h",
        response_id="response-h0",
        response_position=0,
        corpus_role=CorpusRole.BROAD_CONFIRMATORY_HOLDOUT,
        cluster_fit_eligible=False,
    )
    with pytest.raises(ValueError, match="holdout firewall"):
        hierarchical_fit_weights([*records, holdout])
    with pytest.raises(ValueError, match="partition contract mismatch"):
        AnalysisTarget(
            source_artifact_id="bad",
            base_question_id="family-h",
            response_id="response-h0",
            response_position=1,
            corpus_role=CorpusRole.BROAD_CONFIRMATORY_HOLDOUT,
            cluster_fit_eligible=True,
        )


def test_multiplex_keeps_target_paths_separate_and_missing_support_explicit() -> None:
    shared_positive = _node(
        layer=1,
        token=2,
        neuron=10,
        activation=1.0,
        attribution=0.2,
        attr_map=[0.1, 0.2],
    )
    same_basis_second_position = _node(
        layer=1,
        token=3,
        neuron=10,
        activation=2.0,
        attribution=-0.1,
        attr_map=[0.3, None],
    )
    negative_same_raw_neuron = _node(
        layer=1,
        token=4,
        neuron=10,
        activation=-1.0,
        attribution=0.7,
        attr_map=[0.5, 0.5],
    )
    output = _node(layer=36, token=5, neuron=77, activation=3.0)
    first = build_target_slice(
        response_id="response-a",
        target_response_position=0,
        trace_unit_id="trace-0",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        node_rows=[
            shared_positive,
            same_basis_second_position,
            negative_same_raw_neuron,
            output,
        ],
        edge_rows=[
            _edge((1, 2, 10), (36, 5, 77)),
            _edge((1, 4, 10), (36, 5, 77)),
        ],
    )
    second_output = _node(layer=36, token=6, neuron=78, activation=2.0)
    second = build_target_slice(
        response_id="response-a",
        target_response_position=1,
        trace_unit_id="trace-1",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        node_rows=[{**shared_positive, "token": 4}, second_output],
        edge_rows=[_edge((1, 4, 10), (36, 6, 78))],
    )
    multiplex = ResponseTimeMultiplex([second, first])
    validate_target_slice_round_trip(
        first,
        source_node_rows=[
            shared_positive,
            same_basis_second_position,
            negative_same_raw_neuron,
            output,
        ],
        source_edge_rows=[
            _edge((1, 2, 10), (36, 5, 77)),
            _edge((1, 4, 10), (36, 5, 77)),
        ],
    )

    positive_basis = basis_key_from_raw_node(
        shared_positive,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )
    negative_basis = basis_key_from_raw_node(
        negative_same_raw_neuron,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )
    first_positive = first.basis_index[positive_basis]
    assert first_positive.occurrence_count == 2
    assert first_positive.signed_attribution == pytest.approx(0.1)
    assert first_positive.absolute_attribution_mass == pytest.approx(0.3)
    assert first_positive.attribution_map == pytest.approx((0.4, 0.2))
    assert first_positive.attribution_support == (True, True)
    assert negative_basis in first.basis_index

    correspondences = multiplex.longitudinal_correspondences("response-a")
    assert [item.basis for item in correspondences] == [positive_basis]
    assert correspondences[0].causal is False

    negative_trajectory = multiplex.trajectory(
        response_id="response-a",
        basis=negative_basis,
    )
    assert [point.supported for point in negative_trajectory] == [True, False]
    assert negative_trajectory[1].signed_attribution is None
    assert negative_trajectory[1].attribution_map is None
    assert negative_trajectory[1].attribution_support is None

    mixed_target_edges = [first.edges[0], second.edges[0]]
    with pytest.raises(ValueError, match="cannot cross"):
        validate_causal_path(mixed_target_edges)
    with pytest.raises(ValueError, match="one trace unit"):
        OccurrenceEdge(
            "trace-0",
            first.edges[0].source,
            second.edges[0].target,
            0.1,
            0.2,
            "response___0",
        )

    positive_output_basis = first.nodes[-1].basis
    witnesses = multiplex.witnessed_projected_path(
        [positive_basis, positive_output_basis],
        response_id="response-a",
    )
    assert len(witnesses) == 1
    assert witnesses[0].trace_unit_id == "trace-0"

    atlas_index = build_atlas_index(
        [second, first],
        local_circuit_inputs=[
            ("trace-1", 0, "response-a"),
            ("trace-0", 0, "response-a"),
        ],
        source_inventory_sha256="frozen-inventory",
    )
    assert atlas_index["counts"] == {
        "targets": 2,
        "circuit_inputs": 2,
        "signed_bases": 4,
        "occurrences": 6,
        "edges": 3,
    }
    assert [target["trace_unit_id"] for target in atlas_index["targets"]] == [
        "trace-0",
        "trace-1",
    ]
    unhashed = dict(atlas_index)
    recorded_hash = unhashed.pop("atlas_index_sha256")
    assert recorded_hash == canonical_sha256(unhashed)


def test_causal_path_rejects_same_trace_but_discontinuous_occurrences() -> None:
    a = OccurrenceKey("trace", 0, 0, 1, "+")
    b = OccurrenceKey("trace", 1, 1, 2, "+")
    c = OccurrenceKey("trace", 2, 1, 2, "+")
    d = OccurrenceKey("trace", 3, 2, 3, "+")
    edges = [
        OccurrenceEdge("trace", a, b, 0.1, 0.2, "label"),
        OccurrenceEdge("trace", c, d, 0.1, 0.2, "label"),
    ]
    with pytest.raises(ValueError, match="continuity"):
        validate_causal_path(edges)


def test_two_response_profile_smoke_is_weighted_missing_aware_and_unlabeled() -> None:
    slices = [
        build_target_slice(
            response_id=response_id,
            target_response_position=0,
            trace_unit_id=trace_id,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            node_rows=[
                _node(
                    layer=1,
                    token=0,
                    neuron=1,
                    activation=1.0,
                    attr_map=[1.0, 0.0],
                    label=f"{response_id}___0",
                ),
                _node(
                    layer=1,
                    token=1,
                    neuron=2,
                    activation=1.0,
                    attr_map=[0.0, 1.0],
                    label=f"{response_id}___0",
                ),
            ],
            edge_rows=[],
        )
        for response_id, trace_id in (
            ("response-a", "trace-a"),
            ("response-b", "trace-b"),
        )
    ]
    targets = {
        trace_id: AnalysisTarget(
            source_artifact_id=f"source-{trace_id}",
            base_question_id=f"family-{trace_id}",
            response_id=response_id,
            response_position=0,
            corpus_role=CorpusRole.DENSE_DISCOVERY,
            cluster_fit_eligible=True,
        )
        for response_id, trace_id in (
            ("response-a", "trace-a"),
            ("response-b", "trace-b"),
        )
    }
    observations = build_profile_observations(
        slices,
        fit_target_by_trace=targets,
    )
    bases = sorted({observation.basis for observation in observations})
    similarity, witnesses = pairwise_profile_similarity(
        bases[0],
        bases[1],
        observations,
    )
    assert similarity == pytest.approx(0.0)
    assert witnesses == ("trace-a", "trace-b")

    state = cluster_fully_supported_profiles(
        observations,
        expected_trace_ids=["trace-b", "trace-a"],
        n_clusters=2,
    )
    assert state["eligible_signed_basis_count"] == 2
    assert state["descriptions_generated"] is False
    assert state["scientific_cluster_state"] is False
    unhashed = dict(state)
    state_hash = unhashed.pop("cluster_state_sha256")
    assert state_hash == canonical_sha256(unhashed)

    holdout_targets = dict(targets)
    holdout_targets["trace-b"] = AnalysisTarget(
        source_artifact_id="source-trace-b",
        base_question_id="family-trace-b",
        response_id="response-b",
        response_position=0,
        corpus_role=CorpusRole.BROAD_CONFIRMATORY_HOLDOUT,
        cluster_fit_eligible=False,
    )
    with pytest.raises(ValueError, match="holdout firewall"):
        build_profile_observations(
            slices,
            fit_target_by_trace=holdout_targets,
        )
