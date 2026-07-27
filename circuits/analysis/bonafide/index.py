"""Deterministic atlas sidecar indexes over target slices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from circuits.analysis.bonafide.canonical import canonical_sha256, write_hashed_json
from circuits.analysis.bonafide.identity import build_circuit_input_refs
from circuits.analysis.bonafide.multiplex import TargetSlice

ATLAS_INDEX_SCHEMA = "adag.bonafide.atlas-index.v1"


def _target_order(target_slice: TargetSlice) -> tuple[str, int, str]:
    return (
        target_slice.response_id,
        target_slice.target_response_position,
        target_slice.trace_unit_id,
    )


def build_atlas_index(
    slices: Iterable[TargetSlice],
    *,
    local_circuit_inputs: Sequence[tuple[str, int, str]],
    source_inventory_sha256: str,
) -> dict[str, Any]:
    """Assign stable target, circuit-input, signed-basis, and occurrence indexes."""

    if not source_inventory_sha256:
        raise ValueError("source_inventory_sha256 must be non-empty")
    ordered_slices = tuple(sorted(slices, key=_target_order))
    trace_ids = [target_slice.trace_unit_id for target_slice in ordered_slices]
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("atlas index received duplicate trace_unit_id values")
    known_trace_ids = set(trace_ids)

    circuit_input_refs = build_circuit_input_refs(list(local_circuit_inputs))
    unknown_ci_traces = {
        ref.trace_unit_id for ref in circuit_input_refs
    } - known_trace_ids
    if unknown_ci_traces:
        raise ValueError(
            f"circuit inputs reference unknown traces: {sorted(unknown_ci_traces)}"
        )
    traces_without_inputs = known_trace_ids - {
        ref.trace_unit_id for ref in circuit_input_refs
    }
    if traces_without_inputs:
        raise ValueError(
            f"target slices lack circuit-input mappings: {sorted(traces_without_inputs)}"
        )

    all_bases = sorted(
        {node.basis for target_slice in ordered_slices for node in target_slice.nodes}
    )
    basis_to_index = {basis: index for index, basis in enumerate(all_bases)}
    all_occurrences = sorted(
        {
            node.occurrence
            for target_slice in ordered_slices
            for node in target_slice.nodes
        }
    )
    occurrence_to_index = {
        occurrence: index for index, occurrence in enumerate(all_occurrences)
    }
    trace_to_index = {
        target_slice.trace_unit_id: index
        for index, target_slice in enumerate(ordered_slices)
    }
    node_by_occurrence = {
        node.occurrence: node
        for target_slice in ordered_slices
        for node in target_slice.nodes
    }
    if len(node_by_occurrence) != sum(
        len(target_slice.nodes) for target_slice in ordered_slices
    ):
        raise ValueError("duplicate occurrence key across target slices")

    value: dict[str, Any] = {
        "schema_version": ATLAS_INDEX_SCHEMA,
        "source_inventory_sha256": source_inventory_sha256,
        "counts": {
            "targets": len(ordered_slices),
            "circuit_inputs": len(circuit_input_refs),
            "signed_bases": len(all_bases),
            "occurrences": len(all_occurrences),
            "edges": sum(len(target_slice.edges) for target_slice in ordered_slices),
        },
        "targets": [
            {
                "atlas_trace_index": index,
                "trace_unit_id": target_slice.trace_unit_id,
                "response_id": target_slice.response_id,
                "target_response_position": target_slice.target_response_position,
                "model_id": target_slice.model_id,
                "model_revision": target_slice.model_revision,
                "occurrence_indices": [
                    occurrence_to_index[node.occurrence] for node in target_slice.nodes
                ],
                "edge_count": len(target_slice.edges),
            }
            for index, target_slice in enumerate(ordered_slices)
        ],
        "circuit_inputs": [
            {
                "trace_unit_id": ref.trace_unit_id,
                "atlas_trace_index": trace_to_index[ref.trace_unit_id],
                "local_ci_index": ref.local_ci_index,
                "local_label": ref.local_label,
                "global_atlas_ci_index": ref.global_atlas_ci_index,
            }
            for ref in circuit_input_refs
        ],
        "signed_bases": [
            {
                "signed_basis_index": index,
                "basis_key": basis.to_record(),
            }
            for index, basis in enumerate(all_bases)
        ],
        "occurrences": [
            {
                "occurrence_index": index,
                "occurrence_key": occurrence.to_record(),
                "signed_basis_index": basis_to_index[
                    node_by_occurrence[occurrence].basis
                ],
                "atlas_trace_index": trace_to_index[occurrence.trace_unit_id],
            }
            for index, occurrence in enumerate(all_occurrences)
        ],
    }
    value["atlas_index_sha256"] = canonical_sha256(value)
    return value


def write_atlas_index(path: Path, atlas_index: Mapping[str, Any]) -> None:
    write_hashed_json(
        path,
        atlas_index,
        hash_field="atlas_index_sha256",
    )
