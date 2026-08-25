"""Deterministic evidence packets for graph-local occurrence roles."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.graph_labeling.schema import (
    EvidencePacket,
    EvidenceSpec,
    ExplicitOccurrenceSelection,
    OccurrenceSubject,
)


def _evidence_id(category: str, payload: Mapping[str, Any]) -> str:
    return f"ev-{category}-{canonical_sha256(dict(payload))[:20]}"


def _fact(category: str, value: Mapping[str, Any]) -> dict[str, Any]:
    core = {"category": category, "value": dict(value)}
    return {"evidence_id": _evidence_id("fact", core), **core}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _node_summary(
    node: Mapping[str, Any], tokens: list[dict[str, Any]]
) -> dict[str, Any]:
    occurrence = node.get("occurrence", {})
    position = occurrence.get("token_position")
    token = (
        tokens[position]
        if isinstance(position, int) and 0 <= position < len(tokens)
        else None
    )
    return {
        "occurrence_id": node.get("id"),
        "basis_id": node.get("basis_id"),
        "kind": node.get("kind"),
        "layer": occurrence.get("layer"),
        "neuron_index": occurrence.get("neuron_index"),
        "polarity": occurrence.get("polarity"),
        "token_position": position,
        "token": token,
        "attribution": node.get("attribution"),
        "activation": node.get("activation"),
    }


def _source_item(
    *, occurrence_id: str, position: int, value: float, token: dict[str, Any]
) -> dict[str, Any]:
    core = {
        "occurrence_id": occurrence_id,
        "position": position,
        "value": value,
        "token_id": token.get("token_id"),
        "token_text": token.get("text"),
    }
    return {
        "evidence_id": _evidence_id("source", core),
        "category": "source_attribution",
        "sign": "+" if value >= 0 else "-",
        **core,
        "token": token,
    }


def _edge_item(
    edge: Mapping[str, Any],
    *,
    relation: str,
    other: Mapping[str, Any],
    tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    core = {
        "edge_id": edge.get("id"),
        "relation": relation,
        "source": edge.get("source"),
        "target": edge.get("target"),
        "attribution": _finite_number(edge.get("attribution"), "edge attribution"),
        "weight": _finite_number(edge.get("weight"), "edge weight"),
    }
    return {
        "evidence_id": _evidence_id("edge", core),
        "category": "graph_edge",
        **core,
        "other_endpoint": _node_summary(other, tokens),
    }


def _mass_summary(
    items: list[Mapping[str, Any]], retained: list[Mapping[str, Any]]
) -> dict[str, Any]:
    total = sum(
        abs(_finite_number(item.get("attribution"), "edge attribution"))
        for item in items
    )
    kept = sum(
        abs(_finite_number(item.get("attribution"), "edge attribution"))
        for item in retained
    )
    return {
        "total_count": len(items),
        "retained_count": len(retained),
        "total_absolute_attribution_mass": total,
        "retained_absolute_attribution_mass": kept,
        "retained_mass_fraction": kept / total if total else 1.0,
        "truncated": len(retained) < len(items),
    }


def allowed_evidence_ids(packet: EvidencePacket) -> set[str]:
    groups = (
        packet.facts,
        packet.top_positive_sources,
        packet.top_negative_sources,
        packet.top_incoming_edges,
        packet.top_outgoing_edges,
        packet.direct_target_edges,
        packet.target_connected_paths,
    )
    return {
        str(item["evidence_id"])
        for group in groups
        for item in group
        if isinstance(item.get("evidence_id"), str)
    }


def reads_from_evidence_ids(packet: EvidencePacket) -> set[str]:
    """Return evidence IDs that directly support a reads-from claim."""

    groups = (
        packet.top_positive_sources,
        packet.top_negative_sources,
        packet.top_incoming_edges,
    )
    return {
        str(item["evidence_id"])
        for group in groups
        for item in group
        if isinstance(item.get("evidence_id"), str)
    }


def _target_connected_paths(
    occurrence_id: str,
    target_id: str,
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_hops = 4
    max_paths = 8
    max_expansions = 5000
    branch_limit = 16
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        adjacency.setdefault(str(edge.get("source")), []).append(edge)
    branch_truncated = False
    for source, values in adjacency.items():
        values.sort(
            key=lambda edge: (
                -abs(_finite_number(edge.get("attribution"), "edge attribution")),
                str(edge.get("id")),
            )
        )
        if len(values) > branch_limit:
            branch_truncated = True
        adjacency[source] = values[:branch_limit]

    frontier: list[tuple[str, list[dict[str, Any]], frozenset[str]]] = [
        (occurrence_id, [], frozenset({occurrence_id}))
    ]
    found: list[list[dict[str, Any]]] = []
    expansions = 0
    exhausted = True
    while frontier and expansions < max_expansions and len(found) < max_paths * 4:
        current, path, seen = frontier.pop(0)
        if len(path) >= max_hops:
            continue
        for edge in adjacency.get(current, []):
            expansions += 1
            next_id = str(edge.get("target"))
            if next_id in seen:
                continue
            next_path = [*path, edge]
            if next_id == target_id:
                found.append(next_path)
            else:
                frontier.append((next_id, next_path, seen | {next_id}))
            if expansions >= max_expansions:
                exhausted = False
                break
    if frontier:
        exhausted = False

    def rank(path: list[dict[str, Any]]) -> tuple[Any, ...]:
        values = [abs(float(edge["attribution"])) for edge in path]
        return (
            -min(values),
            -sum(values),
            len(path),
            tuple(str(edge["id"]) for edge in path),
        )

    found.sort(key=rank)
    selected = found[:max_paths]
    records: list[dict[str, Any]] = []
    for path in selected:
        core = {
            "edge_ids": [str(edge["id"]) for edge in path],
            "node_ids": [occurrence_id, *[str(edge["target"]) for edge in path]],
            "edge_attributions": [float(edge["attribution"]) for edge in path],
            "hop_count": len(path),
            "minimum_absolute_edge_attribution": min(
                abs(float(edge["attribution"])) for edge in path
            ),
            "sum_absolute_edge_attribution": sum(
                abs(float(edge["attribution"])) for edge in path
            ),
        }
        records.append(
            {
                "evidence_id": _evidence_id("target-path", core),
                "category": "target_connected_path",
                **core,
            }
        )
    search = {
        "algorithm": "bounded_directed_breadth_first_v1",
        "ranking": (
            "descending minimum absolute edge attribution, then descending sum "
            "absolute edge attribution, ascending hop count, lexicographic edge IDs"
        ),
        "semantic_interpretation": "none; path ranking is a display heuristic",
        "max_hops": max_hops,
        "max_paths": max_paths,
        "max_expansions": max_expansions,
        "per_node_branch_limit": branch_limit,
        "expansions": expansions,
        "candidate_path_count": len(found),
        "retained_path_count": len(records),
        "truncated": (branch_truncated or not exhausted or len(found) > len(records)),
    }
    return records, search


def build_evidence_packets(
    document: dict[str, Any],
    selection: ExplicitOccurrenceSelection,
    spec: EvidenceSpec,
) -> list[EvidencePacket]:
    nodes = document.get("nodes")
    edges = document.get("edges")
    context = document.get("context")
    target = document.get("target")
    artifact = document.get("artifact")
    if not all(isinstance(value, dict) for value in (context, target, artifact)):
        raise ValueError("trace lacks context, target, or artifact identity")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("trace nodes and edges must be lists")
    tokens = context.get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(token, dict) for token in tokens
    ):
        raise ValueError("trace context tokens must be objects")
    by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}
    if len(by_id) != len(nodes):
        raise ValueError("trace contains invalid or duplicate node IDs")
    missing = sorted(set(selection.occurrence_ids) - set(by_id))
    if missing:
        raise ValueError(f"selected occurrence IDs are absent: {missing}")
    for occurrence_id in selection.occurrence_ids:
        if by_id[occurrence_id].get("kind") != "raw_mlp_neuron":
            raise ValueError(
                f"selected occurrence is not a raw MLP neuron: {occurrence_id}"
            )
    target_nodes = [node for node in nodes if node.get("kind") == "target_logit"]
    if len(target_nodes) != 1:
        raise ValueError("trace must contain exactly one target-logit node")
    target_node = target_nodes[0]

    selection_group = {
        occurrence_id: group
        for group, occurrence_ids in selection.groups.items()
        for occurrence_id in occurrence_ids
    }
    packets: list[EvidencePacket] = []
    for occurrence_id in selection.occurrence_ids:
        node = by_id[occurrence_id]
        occurrence = node.get("occurrence", {})
        profile = node.get("attribution_map")
        if not isinstance(profile, list):
            raise ValueError(
                f"occurrence lacks an attribution profile: {occurrence_id}"
            )
        values = [
            _finite_number(value, f"{occurrence_id} attribution_map[{index}]")
            for index, value in enumerate(profile)
        ]
        if len(values) > len(tokens):
            raise ValueError("attribution profile is longer than observed context")

        positives = sorted(
            ((index, value) for index, value in enumerate(values) if value > 0),
            key=lambda item: (-item[1], item[0]),
        )[: spec.top_positive_sources]
        negatives = sorted(
            ((index, value) for index, value in enumerate(values) if value < 0),
            key=lambda item: (item[1], item[0]),
        )[: spec.top_negative_sources]
        positive_items = [
            _source_item(
                occurrence_id=occurrence_id,
                position=position,
                value=value,
                token=tokens[position],
            )
            for position, value in positives
        ]
        negative_items = [
            _source_item(
                occurrence_id=occurrence_id,
                position=position,
                value=value,
                token=tokens[position],
            )
            for position, value in negatives
        ]

        incoming_raw = [edge for edge in edges if edge.get("target") == occurrence_id]
        outgoing_raw = [edge for edge in edges if edge.get("source") == occurrence_id]
        incoming_sorted = sorted(
            incoming_raw,
            key=lambda edge: (
                -abs(_finite_number(edge.get("attribution"), "edge attribution")),
                str(edge.get("id")),
            ),
        )
        outgoing_sorted = sorted(
            outgoing_raw,
            key=lambda edge: (
                -abs(_finite_number(edge.get("attribution"), "edge attribution")),
                str(edge.get("id")),
            ),
        )
        incoming_kept = incoming_sorted[: spec.top_incoming_edges]
        outgoing_kept = outgoing_sorted[: spec.top_outgoing_edges]
        incoming_items = [
            _edge_item(
                edge,
                relation="incoming",
                other=by_id[edge["source"]],
                tokens=tokens,
            )
            for edge in incoming_kept
        ]
        outgoing_items = [
            _edge_item(
                edge,
                relation="outgoing",
                other=by_id[edge["target"]],
                tokens=tokens,
            )
            for edge in outgoing_kept
        ]
        direct_raw = [
            edge for edge in outgoing_raw if edge.get("target") == target_node["id"]
        ]
        direct_items = [
            _edge_item(
                edge,
                relation="direct_target",
                other=target_node,
                tokens=tokens,
            )
            for edge in sorted(direct_raw, key=lambda edge: str(edge.get("id")))
        ]

        observed_position = target.get("observed_absolute_position")
        observed_token = (
            tokens[observed_position]
            if isinstance(observed_position, int)
            and 0 <= observed_position < len(tokens)
            else None
        )
        if not isinstance(observed_position, int) or observed_position < 0:
            raise ValueError("target observed position must be a nonnegative integer")
        observed_tokens = tokens[: observed_position + 1]
        if len(observed_tokens) != observed_position + 1:
            raise ValueError(
                "observed target position is outside the serialized context"
            )
        subject = OccurrenceSubject(
            trace_unit_id=str(artifact["artifact_id"]),
            source_trace_sha256=str(artifact["source_hash"]),
            occurrence_id=occurrence_id,
            basis_id=str(node["basis_id"]),
            layer=int(occurrence["layer"]),
            neuron_index=int(occurrence["neuron_index"]),
            polarity=str(occurrence["polarity"]),  # type: ignore[arg-type]
            token_position=int(occurrence["token_position"]),
            selection_group=selection_group[occurrence_id],
            target=dict(target),
        )
        coverage = {
            "serialized_context_token_count": len(tokens),
            "observed_context_token_count": len(observed_tokens),
            "future_token_count_excluded": len(tokens) - len(observed_tokens),
            "causal_profile_entry_count": len(values),
            "causal_profile_absolute_position_range": [0, len(values) - 1],
            "observed_target_absolute_position": observed_position,
            "observed_target_excluded_from_causal_profile": observed_position
            >= len(values),
            "positive_source": {
                "total_count": sum(value > 0 for value in values),
                "retained_count": len(positive_items),
                "total_absolute_attribution_mass": sum(
                    value for value in values if value > 0
                ),
                "retained_absolute_attribution_mass": sum(
                    item[1] for item in positives
                ),
                "truncated": sum(value > 0 for value in values) > len(positive_items),
            },
            "negative_source": {
                "total_count": sum(value < 0 for value in values),
                "retained_count": len(negative_items),
                "total_absolute_attribution_mass": sum(
                    -value for value in values if value < 0
                ),
                "retained_absolute_attribution_mass": sum(
                    -item[1] for item in negatives
                ),
                "truncated": sum(value < 0 for value in values) > len(negative_items),
            },
            "incoming_edges": _mass_summary(incoming_raw, incoming_kept),
            "outgoing_edges": _mass_summary(outgoing_raw, outgoing_kept),
            "direct_target_edge_count": len(direct_items),
        }
        target_paths, path_search = _target_connected_paths(
            occurrence_id, str(target_node["id"]), edges
        )
        node_summary = {
            **_node_summary(node, tokens),
            "contribution_map": node.get("contribution_map"),
        }
        facts = [
            _fact("subject_identity", subject.model_dump(mode="json")),
            _fact("node_measurement", node_summary),
            _fact("target_identity", dict(target)),
            _fact(
                "target_contribution",
                {"contribution_map": node.get("contribution_map")},
            ),
            _fact("coverage_and_truncation", coverage),
            _fact("target_path_search", path_search),
        ]
        core = {
            "schema_version": "adag.graph-labeling.evidence.v1",
            "evidence_policy": spec.policy,
            "subject": subject.model_dump(mode="json"),
            "claim_boundary": str(document.get("claim_boundary", "")),
            "context": {
                "system_prompt": context.get("system_prompt"),
                "prompt": context.get("prompt"),
                "observed_tokens": observed_tokens,
                "causal_profile_tokens": tokens[: len(values)],
                "observed_target_token": observed_token,
                "observed_text_through_target": "".join(
                    str(token.get("text", "")) for token in observed_tokens
                ),
                "causal_prefix_text": "".join(
                    str(token.get("text", "")) for token in tokens[: len(values)]
                ),
            },
            "node": node_summary,
            "facts": facts,
            "top_positive_sources": positive_items,
            "top_negative_sources": negative_items,
            "top_incoming_edges": incoming_items,
            "top_outgoing_edges": outgoing_items,
            "direct_target_edges": direct_items,
            "target_connected_paths": target_paths,
            "path_search": path_search,
            "coverage": coverage,
        }
        packet = EvidencePacket.model_validate(
            {**core, "evidence_sha256": canonical_sha256(core)}
        )
        packets.append(packet)
    return packets
