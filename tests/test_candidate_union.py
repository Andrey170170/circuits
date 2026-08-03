from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
from circuits.tracing.artifact import TopKCompactTraceArtifact
from circuits.tracing.candidate_union import (
    assemble_candidate_union,
    frozen_union_topologies,
    load_candidate_union_artifact,
    save_candidate_union_artifact,
)
from tests.test_topk_topology_comparison import (
    _edge,
    _joint_and_references,
    _node_rows,
)


def _artifact(trace, index: int) -> TopKCompactTraceArtifact:
    return TopKCompactTraceArtifact(
        path=Path(f"/reference-{index}"),
        topk_trace=trace,
        manifest={
            "artifact_id": f"reference-{index}",
            "data_sha256": f"{index + 1:064x}",
        },
        metrics={"status": "complete"},
    )


def _references_and_refinements():
    _joint, reference_traces, candidate_ids = _joint_and_references()
    references = [
        _artifact(trace, index) for index, trace in enumerate(reference_traces)
    ]
    topology_sha256, topologies = frozen_union_topologies(references)
    refinements = []
    embed = (-1, 0, 11)
    mlp = (0, 0, 10)
    for index, (reference, candidate_id) in enumerate(
        zip(references, candidate_ids, strict=True)
    ):
        trace = deepcopy(reference.topk_trace)
        trace.circuit_data.df_node = _node_rows([candidate_id], [float(index + 1)])
        trace.circuit_data.df_edge = pd.DataFrame(
            [
                _edge(embed, mlp, attribution=0.1 * (index + 1)),
                _edge(
                    mlp,
                    (1, 4, candidate_id),
                    attribution=0.2 * (index + 1),
                ),
            ]
        )
        refinements.append(_artifact(trace, index + 10))
    return references, refinements, topology_sha256, topologies, candidate_ids


def test_frozen_union_topologies_keep_exact_internal_and_terminal_edges() -> None:
    references, _refinements, topology_sha256, topologies, candidate_ids = (
        _references_and_refinements()
    )

    assert len(topology_sha256) == 64
    assert len(topologies) == len(references) == 5
    assert topologies[0].mlp_nodes == topologies[4].mlp_nodes
    for index, topology in enumerate(topologies):
        terminal_targets = {
            edge[1].neuron for edge in topology.edges if edge[1].layer == 1
        }
        assert terminal_targets == {candidate_ids[index]}


def test_candidate_union_assembles_dense_node_and_edge_measurements(
    tmp_path: Path,
) -> None:
    references, refinements, topology_sha256, _topologies, candidate_ids = (
        _references_and_refinements()
    )
    # Candidate one did not independently retain the shared internal edge, but
    # pass two still measures it on the exact union.
    references[1].topk_trace.circuit_data.df_edge = (
        references[1].topk_trace.circuit_data.df_edge.iloc[1:].copy()
    )

    trace = assemble_candidate_union(
        references,
        refinements,
        topology_sha256=topology_sha256,
        source_width1_artifact_id="source-1",
    )

    shared = trace.df_edge[trace.df_edge["layer"] == "-1->0"].iloc[0]
    assert shared["applicable_by_candidate"] == [True] * 5
    assert shared["selected_by_candidate"] == [True, False, True, True, True]
    assert shared["candidate_attribution"] == [
        0.1,
        0.2,
        0.30000000000000004,
        0.4,
        0.5,
    ]
    terminal = trace.df_edge[
        trace.df_edge["neuron"].str.endswith(f"->{candidate_ids[0]}")
    ].iloc[0]
    assert terminal["applicable_by_candidate"] == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert terminal["candidate_attribution"][1:] == [None] * 4

    output = tmp_path / "union"
    save_candidate_union_artifact(
        output,
        trace,
        manifest={"artifact_id": "union-1"},
        metrics={"status": "complete"},
    )
    loaded = load_candidate_union_artifact(output)
    assert loaded.trace.contract_dict() == trace.contract_dict()
    assert loaded.metrics["status"] == "complete"
    assert loaded.manifest["artifact_id"] == "union-1"
