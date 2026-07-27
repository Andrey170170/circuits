"""Target-indexed response-time multiplex primitives.

Longitudinal correspondences are deliberately separate from causal edges. A
causal path is valid only inside one independently traced target slice.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from circuits.analysis.bonafide.identity import (
    OccurrenceKey,
    SignedBasisKey,
    basis_from_occurrence,
    basis_key_from_raw_node,
    occurrence_key_from_raw_node,
    polarity_from_raw_node,
)

TARGET_SLICE_SCHEMA = "adag.bonafide.target-slice.v1"
MULTIPLEX_SCHEMA = "adag.bonafide.response-time-multiplex.v1"


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_vector(value: object, field: str) -> tuple[float | None, ...]:
    if not isinstance(value, (list, tuple)):
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(f"{field} must be a one-dimensional sequence") from error
    else:
        values = list(value)
    return tuple(
        None if item is None else _finite_float(item, field) for item in values
    )


def _parse_endpoint_pair(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ValueError(f"{field} must use 'source->target' syntax")
    parts = value.split("->")
    if len(parts) != 2:
        raise ValueError(f"{field} must use 'source->target' syntax")
    try:
        source, target = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"{field} endpoints must be integers") from error
    return source, target


@dataclass(frozen=True)
class OccurrenceNode:
    occurrence: OccurrenceKey
    basis: SignedBasisKey
    attribution: float
    activation: float
    attribution_map: tuple[float | None, ...]
    contribution_map: tuple[float | None, ...]
    local_label: str

    def to_raw_record(self) -> dict[str, object]:
        return {
            "layer": self.occurrence.layer,
            "token": self.occurrence.token_position,
            "neuron": self.occurrence.neuron_index,
            "attribution": self.attribution,
            "activation": self.activation,
            "attr_map": list(self.attribution_map),
            "contrib_map": list(self.contribution_map),
            "label": self.local_label,
            "polarity": self.occurrence.polarity,
        }


@dataclass(frozen=True)
class OccurrenceEdge:
    trace_unit_id: str
    source: OccurrenceKey
    target: OccurrenceKey
    attribution: float
    weight: float
    local_label: str

    def __post_init__(self) -> None:
        if (
            self.source.trace_unit_id != self.trace_unit_id
            or self.target.trace_unit_id != self.trace_unit_id
        ):
            raise ValueError("occurrence edge endpoints must remain in one trace unit")

    def to_raw_record(self) -> dict[str, object]:
        return {
            "layer": f"{self.source.layer}->{self.target.layer}",
            "token": (f"{self.source.token_position}->{self.target.token_position}"),
            "neuron": (f"{self.source.neuron_index}->{self.target.neuron_index}"),
            "attribution": self.attribution,
            "weight": self.weight,
            "label": self.local_label,
        }


@dataclass(frozen=True)
class BasisTargetSummary:
    basis: SignedBasisKey
    occurrences: tuple[OccurrenceKey, ...]
    signed_attribution: float
    absolute_attribution_mass: float
    occurrence_count: int
    mean_activation: float
    attribution_map: tuple[float | None, ...]
    attribution_support: tuple[bool, ...]
    contribution_map: tuple[float | None, ...]
    contribution_support: tuple[bool, ...]
    in_degree: int
    out_degree: int


@dataclass(frozen=True)
class TargetSlice:
    response_id: str
    target_response_position: int
    trace_unit_id: str
    model_id: str
    model_revision: str
    nodes: tuple[OccurrenceNode, ...]
    edges: tuple[OccurrenceEdge, ...]
    basis_summaries: tuple[BasisTargetSummary, ...]

    def __post_init__(self) -> None:
        if not self.response_id or not self.trace_unit_id:
            raise ValueError("response_id and trace_unit_id must be non-empty")
        if self.target_response_position < 0:
            raise ValueError("target_response_position must be nonnegative")
        if any(
            node.occurrence.trace_unit_id != self.trace_unit_id for node in self.nodes
        ):
            raise ValueError("target slice contains a node from another trace")
        if any(edge.trace_unit_id != self.trace_unit_id for edge in self.edges):
            raise ValueError("target slice contains an edge from another trace")

    @property
    def basis_index(self) -> dict[SignedBasisKey, BasisTargetSummary]:
        return {summary.basis: summary for summary in self.basis_summaries}

    def raw_tables(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Reconstruct the ingested scientific node/edge fields per target."""

        return (
            [node.to_raw_record() for node in self.nodes],
            [edge.to_raw_record() for edge in self.edges],
        )


def _aggregate_basis_nodes(
    nodes: Sequence[OccurrenceNode],
    edges: Sequence[OccurrenceEdge],
) -> tuple[BasisTargetSummary, ...]:
    grouped: dict[SignedBasisKey, list[OccurrenceNode]] = defaultdict(list)
    for node in nodes:
        grouped[node.basis].append(node)

    summaries: list[BasisTargetSummary] = []
    in_degree_by_occurrence: dict[OccurrenceKey, int] = defaultdict(int)
    out_degree_by_occurrence: dict[OccurrenceKey, int] = defaultdict(int)
    for edge in edges:
        out_degree_by_occurrence[edge.source] += 1
        in_degree_by_occurrence[edge.target] += 1
    for basis, members in sorted(grouped.items()):
        widths = {len(member.attribution_map) for member in members}
        if len(widths) != 1:
            raise ValueError("one target/basis has inconsistent attribution-map widths")
        width = widths.pop()
        supported_values = tuple(
            tuple(
                value
                for member in members
                if (value := member.attribution_map[index]) is not None
            )
            for index in range(width)
        )
        summed_map = tuple(
            sum(values) if values else None for values in supported_values
        )
        contribution_widths = {len(member.contribution_map) for member in members}
        if len(contribution_widths) != 1:
            raise ValueError(
                "one target/basis has inconsistent contribution-map widths"
            )
        contribution_width = contribution_widths.pop()
        supported_contributions = tuple(
            tuple(
                value
                for member in members
                if (value := member.contribution_map[index]) is not None
            )
            for index in range(contribution_width)
        )
        summed_contribution = tuple(
            sum(values) if values else None for values in supported_contributions
        )
        summaries.append(
            BasisTargetSummary(
                basis=basis,
                occurrences=tuple(sorted(member.occurrence for member in members)),
                signed_attribution=sum(member.attribution for member in members),
                absolute_attribution_mass=sum(
                    abs(member.attribution) for member in members
                ),
                occurrence_count=len(members),
                mean_activation=sum(member.activation for member in members)
                / len(members),
                attribution_map=summed_map,
                attribution_support=tuple(bool(values) for values in supported_values),
                contribution_map=summed_contribution,
                contribution_support=tuple(
                    bool(values) for values in supported_contributions
                ),
                in_degree=sum(
                    in_degree_by_occurrence[member.occurrence] for member in members
                ),
                out_degree=sum(
                    out_degree_by_occurrence[member.occurrence] for member in members
                ),
            )
        )
    return tuple(summaries)


def build_target_slice(
    *,
    response_id: str,
    target_response_position: int,
    trace_unit_id: str,
    model_id: str,
    model_revision: str,
    node_rows: Iterable[Mapping[str, Any]],
    edge_rows: Iterable[Mapping[str, Any]],
) -> TargetSlice:
    nodes: list[OccurrenceNode] = []
    endpoint_index: dict[tuple[int, int, int], OccurrenceKey] = {}
    for row in node_rows:
        occurrence = occurrence_key_from_raw_node(
            row,
            trace_unit_id=trace_unit_id,
        )
        basis = basis_key_from_raw_node(
            row,
            model_id=model_id,
            model_revision=model_revision,
        )
        endpoint = (
            occurrence.layer,
            occurrence.token_position,
            occurrence.neuron_index,
        )
        if endpoint in endpoint_index:
            raise ValueError(f"duplicate raw occurrence endpoint: {endpoint}")
        endpoint_index[endpoint] = occurrence
        label = row.get("label")
        if not isinstance(label, str):
            raise ValueError("raw node label must be a string")
        nodes.append(
            OccurrenceNode(
                occurrence=occurrence,
                basis=basis,
                attribution=_finite_float(
                    row.get("attribution"),
                    "node attribution",
                ),
                activation=_finite_float(row.get("activation"), "node activation"),
                attribution_map=_optional_vector(row.get("attr_map"), "node attr_map"),
                contribution_map=_optional_vector(
                    row.get("contrib_map"),
                    "node contrib_map",
                ),
                local_label=label,
            )
        )

    edges: list[OccurrenceEdge] = []
    for row in edge_rows:
        source_layer, target_layer = _parse_endpoint_pair(
            row.get("layer"), "edge layer"
        )
        source_token, target_token = _parse_endpoint_pair(
            row.get("token"), "edge token"
        )
        source_neuron, target_neuron = _parse_endpoint_pair(
            row.get("neuron"), "edge neuron"
        )
        source_endpoint = (source_layer, source_token, source_neuron)
        target_endpoint = (target_layer, target_token, target_neuron)
        try:
            source = endpoint_index[source_endpoint]
            target = endpoint_index[target_endpoint]
        except KeyError as error:
            raise ValueError(
                f"edge endpoint is absent from target slice: {error.args[0]}"
            ) from error
        label = row.get("label")
        if not isinstance(label, str):
            raise ValueError("raw edge label must be a string")
        edges.append(
            OccurrenceEdge(
                trace_unit_id=trace_unit_id,
                source=source,
                target=target,
                attribution=_finite_float(
                    row.get("attribution"),
                    "edge attribution",
                ),
                weight=_finite_float(row.get("weight"), "edge weight"),
                local_label=label,
            )
        )

    ordered_nodes = tuple(sorted(nodes, key=lambda node: node.occurrence))
    ordered_edges = tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.source,
                edge.target,
                edge.attribution,
                edge.weight,
            ),
        )
    )
    return TargetSlice(
        response_id=response_id,
        target_response_position=target_response_position,
        trace_unit_id=trace_unit_id,
        model_id=model_id,
        model_revision=model_revision,
        nodes=ordered_nodes,
        edges=ordered_edges,
        basis_summaries=_aggregate_basis_nodes(ordered_nodes, ordered_edges),
    )


def validate_target_slice_round_trip(
    target_slice: TargetSlice,
    *,
    source_node_rows: Iterable[Mapping[str, Any]],
    source_edge_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Verify exact reconstruction of the ingested scientific table fields."""

    expected_nodes = sorted(
        (
            int(row["layer"]),
            int(row["token"]),
            int(row["neuron"]),
            polarity_from_raw_node(row),
            _finite_float(row.get("attribution"), "node attribution"),
            _finite_float(row.get("activation"), "node activation"),
            _optional_vector(row.get("attr_map"), "node attr_map"),
            _optional_vector(row.get("contrib_map"), "node contrib_map"),
            row.get("label"),
        )
        for row in source_node_rows
    )
    reconstructed_nodes = sorted(
        (
            node.occurrence.layer,
            node.occurrence.token_position,
            node.occurrence.neuron_index,
            node.occurrence.polarity,
            node.attribution,
            node.activation,
            node.attribution_map,
            node.contribution_map,
            node.local_label,
        )
        for node in target_slice.nodes
    )
    if expected_nodes != reconstructed_nodes:
        raise ValueError("target-slice node round trip mismatch")

    expected_edges = sorted(
        (
            _parse_endpoint_pair(row.get("layer"), "edge layer"),
            _parse_endpoint_pair(row.get("token"), "edge token"),
            _parse_endpoint_pair(row.get("neuron"), "edge neuron"),
            _finite_float(row.get("attribution"), "edge attribution"),
            _finite_float(row.get("weight"), "edge weight"),
            row.get("label"),
        )
        for row in source_edge_rows
    )
    reconstructed_edges = sorted(
        (
            (edge.source.layer, edge.target.layer),
            (edge.source.token_position, edge.target.token_position),
            (edge.source.neuron_index, edge.target.neuron_index),
            edge.attribution,
            edge.weight,
            edge.local_label,
        )
        for edge in target_slice.edges
    )
    if expected_edges != reconstructed_edges:
        raise ValueError("target-slice edge round trip mismatch")


@dataclass(frozen=True)
class LongitudinalCorrespondence:
    response_id: str
    left_target_position: int
    right_target_position: int
    left_trace_unit_id: str
    right_trace_unit_id: str
    basis: SignedBasisKey
    left_occurrences: tuple[OccurrenceKey, ...]
    right_occurrences: tuple[OccurrenceKey, ...]
    relation: str = "same_basis_at_next_target"
    causal: bool = False


@dataclass(frozen=True)
class TrajectoryPoint:
    response_id: str
    target_response_position: int
    trace_unit_id: str
    basis: SignedBasisKey
    supported: bool
    signed_attribution: float | None
    absolute_attribution_mass: float | None
    occurrence_count: int | None
    mean_activation: float | None
    attribution_map: tuple[float | None, ...] | None
    attribution_support: tuple[bool, ...] | None


@dataclass(frozen=True)
class PathWitness:
    trace_unit_id: str
    target_response_position: int
    occurrences: tuple[OccurrenceKey, ...]


def validate_causal_path(edges: Sequence[OccurrenceEdge]) -> None:
    if not edges:
        raise ValueError("causal path must contain at least one edge")
    trace_unit_id = edges[0].trace_unit_id
    for edge in edges:
        if edge.trace_unit_id != trace_unit_id:
            raise ValueError("causal path cannot cross independently traced targets")
    for left, right in zip(edges, edges[1:], strict=False):
        if left.target != right.source:
            raise ValueError(
                "causal path does not preserve exact occurrence continuity"
            )


class ResponseTimeMultiplex:
    """Read-only collection of independent target slices."""

    def __init__(self, slices: Iterable[TargetSlice]) -> None:
        ordered = tuple(
            sorted(
                slices,
                key=lambda item: (
                    item.response_id,
                    item.target_response_position,
                    item.trace_unit_id,
                ),
            )
        )
        trace_ids = [item.trace_unit_id for item in ordered]
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("multiplex contains duplicate trace_unit_id values")
        positions = [
            (item.response_id, item.target_response_position) for item in ordered
        ]
        if len(positions) != len(set(positions)):
            raise ValueError("multiplex contains duplicate response target positions")
        self._slices = ordered

    @property
    def slices(self) -> tuple[TargetSlice, ...]:
        return self._slices

    def _response_slices(self, response_id: str) -> tuple[TargetSlice, ...]:
        return tuple(
            target_slice
            for target_slice in self._slices
            if target_slice.response_id == response_id
        )

    def longitudinal_correspondences(
        self,
        response_id: str,
    ) -> tuple[LongitudinalCorrespondence, ...]:
        slices = self._response_slices(response_id)
        correspondences: list[LongitudinalCorrespondence] = []
        for left, right in zip(slices, slices[1:], strict=False):
            shared = sorted(left.basis_index.keys() & right.basis_index.keys())
            for basis in shared:
                correspondences.append(
                    LongitudinalCorrespondence(
                        response_id=response_id,
                        left_target_position=left.target_response_position,
                        right_target_position=right.target_response_position,
                        left_trace_unit_id=left.trace_unit_id,
                        right_trace_unit_id=right.trace_unit_id,
                        basis=basis,
                        left_occurrences=left.basis_index[basis].occurrences,
                        right_occurrences=right.basis_index[basis].occurrences,
                    )
                )
        return tuple(correspondences)

    def trajectory(
        self,
        *,
        response_id: str,
        basis: SignedBasisKey,
    ) -> tuple[TrajectoryPoint, ...]:
        points: list[TrajectoryPoint] = []
        for target_slice in self._response_slices(response_id):
            summary = target_slice.basis_index.get(basis)
            points.append(
                TrajectoryPoint(
                    response_id=response_id,
                    target_response_position=target_slice.target_response_position,
                    trace_unit_id=target_slice.trace_unit_id,
                    basis=basis,
                    supported=summary is not None,
                    signed_attribution=(
                        summary.signed_attribution if summary else None
                    ),
                    absolute_attribution_mass=(
                        summary.absolute_attribution_mass if summary else None
                    ),
                    occurrence_count=summary.occurrence_count if summary else None,
                    mean_activation=summary.mean_activation if summary else None,
                    attribution_map=summary.attribution_map if summary else None,
                    attribution_support=(
                        summary.attribution_support if summary else None
                    ),
                )
            )
        return tuple(points)

    def witnessed_projected_path(
        self,
        basis_path: Sequence[SignedBasisKey],
        *,
        response_id: str | None = None,
    ) -> tuple[PathWitness, ...]:
        """Return exact per-target occurrence witnesses for a basis-level path."""

        if len(basis_path) < 2:
            raise ValueError("projected path requires at least two signed bases")
        witnesses: list[PathWitness] = []
        for target_slice in self._slices:
            if response_id is not None and target_slice.response_id != response_id:
                continue
            outgoing: dict[OccurrenceKey, list[OccurrenceKey]] = defaultdict(list)
            for edge in target_slice.edges:
                outgoing[edge.source].append(edge.target)

            starts = [
                node.occurrence
                for node in target_slice.nodes
                if node.basis == basis_path[0]
            ]
            for start in sorted(starts):
                frontier: list[tuple[OccurrenceKey, ...]] = [(start,)]
                for expected_basis in basis_path[1:]:
                    next_frontier: list[tuple[OccurrenceKey, ...]] = []
                    for path in frontier:
                        for candidate in sorted(outgoing.get(path[-1], [])):
                            if (
                                basis_from_occurrence(
                                    candidate,
                                    model_id=target_slice.model_id,
                                    model_revision=target_slice.model_revision,
                                )
                                == expected_basis
                            ):
                                next_frontier.append((*path, candidate))
                    frontier = next_frontier
                    if not frontier:
                        break
                if frontier:
                    witnesses.extend(
                        PathWitness(
                            trace_unit_id=target_slice.trace_unit_id,
                            target_response_position=(
                                target_slice.target_response_position
                            ),
                            occurrences=path,
                        )
                        for path in frontier
                    )
        return tuple(
            sorted(
                witnesses,
                key=lambda witness: (
                    witness.target_response_position,
                    witness.trace_unit_id,
                    witness.occurrences,
                ),
            )
        )
