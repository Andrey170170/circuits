"""C0 comparison of candidate-joint and independent k=1 graph topologies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from circuits.tracing.artifact import validate_topk_trace_data
from circuits.tracing.trace import TopKPositionTrace

NodeKey = tuple[int, int, int]
EdgeKey = tuple[NodeKey, NodeKey]


def _node_key(row: pd.Series) -> NodeKey:
    return int(row["layer"]), int(row["token"]), int(row["neuron"])


def _edge_key(row: pd.Series) -> EdgeKey:
    values = []
    for column in ("layer", "token", "neuron"):
        parts = str(row[column]).split("->")
        if len(parts) != 2:
            raise ValueError(f"invalid edge {column}: {row[column]!r}")
        values.append((int(parts[0]), int(parts[1])))
    return (
        (values[0][0], values[1][0], values[2][0]),
        (values[0][1], values[1][1], values[2][1]),
    )


def _topology(trace: TopKPositionTrace) -> tuple[set[NodeKey], set[EdgeKey]]:
    data = trace.circuit_data
    return (
        {_node_key(row) for _, row in data.df_node.iterrows()},
        {_edge_key(row) for _, row in data.df_edge.iterrows()},
    )


def _mass_coverage(
    independent: TopKPositionTrace, joint_nodes: set[NodeKey]
) -> float | None:
    masses: dict[NodeKey, float] = defaultdict(float)
    for _, row in independent.circuit_data.df_node.iterrows():
        masses[_node_key(row)] += abs(float(row["attribution"]))
    total = sum(masses.values())
    if total == 0:
        return None
    return sum(value for key, value in masses.items() if key in joint_nodes) / total


def _path_evidence(
    independent_edges: set[EdgeKey],
    retained_edges: set[EdgeKey],
    *,
    target: NodeKey,
    max_witnesses: int,
) -> dict[str, Any]:
    adjacency: dict[NodeKey, list[NodeKey]] = defaultdict(list)
    nodes: set[NodeKey] = {target}
    for source, destination in independent_edges:
        adjacency[source].append(destination)
        nodes.update((source, destination))
    for destinations in adjacency.values():
        destinations.sort()
    sources = sorted(node for node in nodes if node[0] == -1)

    @cache
    def count_paths(node: NodeKey, retained_only: bool) -> int:
        if node == target:
            return 1
        total = 0
        for destination in adjacency.get(node, []):
            edge = (node, destination)
            if retained_only and edge not in retained_edges:
                continue
            if destination[0] <= node[0] and destination != target:
                raise ValueError("candidate topology path graph is not layer-acyclic")
            total += count_paths(destination, retained_only)
        return total

    total_paths = sum(count_paths(source, False) for source in sources)
    retained_paths = sum(count_paths(source, True) for source in sources)
    witnesses: list[list[list[int]]] = []

    def visit(node: NodeKey, path: list[NodeKey], omitted: bool) -> None:
        if len(witnesses) >= max_witnesses:
            return
        if node == target:
            if omitted:
                witnesses.append([list(item) for item in path])
            return
        for destination in adjacency.get(node, []):
            visit(
                destination,
                [*path, destination],
                omitted or (node, destination) not in retained_edges,
            )

    for source in sources:
        visit(source, [source], False)
        if len(witnesses) >= max_witnesses:
            break
    return {
        "independent_source_to_logit_path_count": total_paths,
        "joint_retained_path_count": retained_paths,
        "path_recall": (retained_paths / total_paths if total_paths > 0 else None),
        "omitted_path_count": total_paths - retained_paths,
        "omitted_path_witnesses": witnesses,
        "omitted_path_witnesses_truncated": (
            total_paths - retained_paths > len(witnesses)
        ),
    }


def _candidate_profile_diagnostics(trace: TopKPositionTrace) -> dict[str, Any]:
    frame = trace.circuit_data.df_node
    if frame.empty:
        return {
            "mlp_profile_row_count": 0,
            "candidate_profile_effective_rank": 0,
            "sign_conflict_row_count": 0,
            "sign_conflict_rate": None,
        }
    max_layer = int(frame["layer"].max())
    mlp = frame[(frame["layer"] >= 0) & (frame["layer"] < max_layer)]
    rows = [
        [float(value) for value in contribution]
        for contribution in mlp["contrib_map"]
        if contribution is not None
    ]
    if not rows:
        return {
            "mlp_profile_row_count": 0,
            "candidate_profile_effective_rank": 0,
            "sign_conflict_row_count": 0,
            "sign_conflict_rate": None,
        }
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape[1] != trace.candidate_count or not np.isfinite(matrix).all():
        raise ValueError("joint candidate profile matrix is invalid")
    conflicts = (matrix > 0).any(axis=1) & (matrix < 0).any(axis=1)
    return {
        "mlp_profile_row_count": int(matrix.shape[0]),
        "candidate_profile_effective_rank": int(np.linalg.matrix_rank(matrix)),
        "sign_conflict_row_count": int(conflicts.sum()),
        "sign_conflict_rate": float(conflicts.mean()),
    }


def compare_joint_to_independent_candidates(
    joint: TopKPositionTrace,
    independent: Iterable[TopKPositionTrace],
    *,
    max_omitted_path_witnesses_per_candidate: int = 100,
) -> dict[str, Any]:
    """Compare one candidate-joint graph with exact independent k=1 references."""

    if (
        isinstance(max_omitted_path_witnesses_per_candidate, bool)
        or not isinstance(max_omitted_path_witnesses_per_candidate, int)
        or max_omitted_path_witnesses_per_candidate < 0
    ):
        raise ValueError(
            "max_omitted_path_witnesses_per_candidate must be a non-negative integer"
        )
    validate_topk_trace_data(joint)
    references = list(independent)
    if not references:
        raise ValueError("C0 topology comparison requires independent references")
    joint_nodes, joint_edges = _topology(joint)
    joint_candidate_ids = {
        candidate.token_id for candidate in joint.candidate_selection.candidates
    }
    by_token_id: dict[int, TopKPositionTrace] = {}
    for reference in references:
        validate_topk_trace_data(reference)
        if reference.candidate_count != 1:
            raise ValueError("C0 independent references must have candidate_count=1")
        if reference.joint_objective.objective_id != "raw_logit_sum":
            raise ValueError("C0 independent references require raw_logit_sum")
        if (
            reference.shared_response_position != joint.shared_response_position
            or reference.shared_prediction_position != joint.shared_prediction_position
        ):
            raise ValueError("C0 references must share the joint target position")
        token_id = reference.candidate_selection.candidates[0].token_id
        if token_id in by_token_id:
            raise ValueError(f"duplicate C0 independent candidate token: {token_id}")
        by_token_id[token_id] = reference
    if set(by_token_id) != joint_candidate_ids:
        raise ValueError("C0 independent candidate set does not match joint candidates")

    union_nodes: set[NodeKey] = set()
    union_edges: set[EdgeKey] = set()
    candidate_results: list[dict[str, Any]] = []
    final_layer = max(node[0] for node in joint_nodes)
    for candidate in joint.candidate_selection.candidates:
        reference = by_token_id[candidate.token_id]
        nodes, edges = _topology(reference)
        union_nodes.update(nodes)
        union_edges.update(edges)
        retained_edges = edges & joint_edges
        target = (
            final_layer,
            joint.shared_prediction_position,
            candidate.token_id,
        )
        candidate_results.append(
            {
                "candidate_index": candidate.candidate_index,
                "token_id": candidate.token_id,
                "token_text": candidate.token_text,
                "is_observed": candidate.is_observed,
                "independent_node_count": len(nodes),
                "independent_edge_count": len(edges),
                "joint_node_recall": (
                    len(nodes & joint_nodes) / len(nodes) if nodes else None
                ),
                "joint_edge_recall": (
                    len(retained_edges) / len(edges) if edges else None
                ),
                "retained_absolute_node_attribution_mass": _mass_coverage(
                    reference, joint_nodes
                ),
                **_path_evidence(
                    edges,
                    retained_edges,
                    target=target,
                    max_witnesses=max_omitted_path_witnesses_per_candidate,
                ),
            }
        )
    return {
        "schema_version": "adag.candidate-topology-comparison.v1",
        "trace_family_id": joint.trace_family_id,
        "joint_objective": joint.joint_objective.to_dict(),
        "candidate_policy": {
            "policy_id": joint.candidate_selection.policy_id,
            "policy_version": joint.candidate_selection.policy_version,
        },
        "candidate_count": joint.candidate_count,
        "joint_node_count": len(joint_nodes),
        "joint_edge_count": len(joint_edges),
        "independent_union_node_count": len(union_nodes),
        "independent_union_edge_count": len(union_edges),
        "union_node_recall": (
            len(union_nodes & joint_nodes) / len(union_nodes) if union_nodes else None
        ),
        "union_edge_recall": (
            len(union_edges & joint_edges) / len(union_edges) if union_edges else None
        ),
        "candidate_profile_diagnostics": _candidate_profile_diagnostics(joint),
        "candidates": candidate_results,
    }
