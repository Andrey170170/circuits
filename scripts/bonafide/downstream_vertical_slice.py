"""Run a bounded two-response BonaFide downstream vertical-slice smoke."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    load_json_object,
)
from circuits.analysis.bonafide.features import (
    build_profile_observations,
    cluster_fully_supported_profiles,
    fully_supported_profile_bases,
)
from circuits.analysis.bonafide.index import build_atlas_index
from circuits.analysis.bonafide.inventory import INVENTORY_SCHEMA
from circuits.analysis.bonafide.multiplex import (
    ResponseTimeMultiplex,
    TargetSlice,
    build_target_slice,
    validate_target_slice_round_trip,
)
from circuits.analysis.bonafide.partition import AnalysisTarget, CorpusRole
from circuits.tracing.artifact import load_compact_trace

VERTICAL_SLICE_SCHEMA = "adag.bonafide.vertical-slice-smoke.v1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _validated_inventory(path: Path) -> dict[str, Any]:
    inventory = load_json_object(path)
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("unsupported inventory schema")
    recorded_hash = inventory.get("inventory_sha256")
    core = dict(inventory)
    core.pop("inventory_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("inventory canonical hash mismatch")
    return inventory


def _select_records(
    inventory: Mapping[str, Any],
    *,
    response_ids: Sequence[str],
    targets_per_response: int,
) -> list[Mapping[str, Any]]:
    records_value = inventory.get("records")
    if not isinstance(records_value, list):
        raise ValueError("inventory records must be a list")
    dense = [
        cast(Mapping[str, Any], record)
        for record in records_value
        if isinstance(record, Mapping)
        and record.get("status") == "discovery"
        and record.get("corpus_role") == CorpusRole.DENSE_DISCOVERY.value
    ]
    by_response: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in dense:
        response_id = record.get("response_id")
        if not isinstance(response_id, str):
            raise ValueError("dense inventory record is missing response_id")
        by_response[response_id].append(record)
    eligible = {
        response_id: sorted(
            records,
            key=lambda record: (
                int(record["response_position"]),
                str(record["source_artifact_id"]),
            ),
        )
        for response_id, records in by_response.items()
        if len(records) >= targets_per_response
    }

    if response_ids:
        chosen = list(response_ids)
        if len(chosen) != 2 or len(set(chosen)) != 2:
            raise ValueError("pass exactly two distinct --response-id values")
        missing = set(chosen) - eligible.keys()
        if missing:
            raise ValueError(
                f"requested dense responses lack enough targets: {sorted(missing)}"
            )
    else:
        chosen = []
        used_families: set[str] = set()
        for response_id in sorted(eligible):
            family_id = str(eligible[response_id][0]["base_question_id"])
            if family_id in used_families:
                continue
            chosen.append(response_id)
            used_families.add(family_id)
            if len(chosen) == 2:
                break
        if len(chosen) != 2:
            raise ValueError("inventory lacks two eligible dense response families")

    return [
        record
        for response_id in chosen
        for record in eligible[response_id][:targets_per_response]
    ]


def run_vertical_slice(
    *,
    inventory_path: Path,
    output_path: Path,
    response_ids: Sequence[str] = (),
    targets_per_response: int = 2,
    requested_clusters: int = 8,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"vertical-slice output already exists: {output_path}")
    if targets_per_response < 2:
        raise ValueError(
            "targets_per_response must be >= 2 to exercise longitudinal recurrence"
        )
    if requested_clusters < 1:
        raise ValueError("requested_clusters must be positive")

    inventory = _validated_inventory(inventory_path)
    selected = _select_records(
        inventory,
        response_ids=response_ids,
        targets_per_response=targets_per_response,
    )
    slices: list[TargetSlice] = []
    target_by_trace: dict[str, AnalysisTarget] = {}
    local_circuit_inputs: list[tuple[str, int, str]] = []
    selected_records: list[dict[str, Any]] = []
    for record in selected:
        artifact_path_value = record.get("artifact_path")
        if not isinstance(artifact_path_value, str):
            raise ValueError("selected inventory record lacks artifact_path")
        loaded = load_compact_trace(artifact_path_value)
        manifest = loaded.manifest
        trace_unit_id = str(record["trace_unit_id"])
        if manifest.get("artifact_id") != trace_unit_id:
            raise ValueError("inventory/payload trace_unit_id mismatch")
        data = loaded.circuit_data
        node_rows = data.df_node.to_dict(orient="records")
        edge_rows = data.df_edge.to_dict(orient="records")
        target_slice = build_target_slice(
            response_id=str(record["response_id"]),
            target_response_position=int(record["response_position"]),
            trace_unit_id=trace_unit_id,
            model_id=str(record["model_id"]),
            model_revision=str(record["model_revision"]),
            node_rows=node_rows,
            edge_rows=edge_rows,
        )
        validate_target_slice_round_trip(
            target_slice,
            source_node_rows=node_rows,
            source_edge_rows=edge_rows,
        )
        slices.append(target_slice)
        target_by_trace[trace_unit_id] = AnalysisTarget(
            source_artifact_id=str(record["source_artifact_id"]),
            base_question_id=str(record["base_question_id"]),
            response_id=str(record["response_id"]),
            response_position=int(record["response_position"]),
            corpus_role=CorpusRole(str(record["corpus_role"])),
            cluster_fit_eligible=bool(record["cluster_fit_eligible"]),
        )
        local_circuit_inputs.extend(
            (trace_unit_id, index, label) for index, label in enumerate(data.labels)
        )
        selected_records.append(
            {
                "source_artifact_id": record["source_artifact_id"],
                "trace_unit_id": trace_unit_id,
                "response_id": record["response_id"],
                "base_question_id": record["base_question_id"],
                "response_position": record["response_position"],
                "node_count": len(target_slice.nodes),
                "edge_count": len(target_slice.edges),
                "signed_basis_count": len(target_slice.basis_summaries),
                "round_trip_valid": True,
            }
        )

    multiplex = ResponseTimeMultiplex(slices)
    inventory_sha256 = str(inventory["inventory_sha256"])
    atlas_index = build_atlas_index(
        slices,
        local_circuit_inputs=local_circuit_inputs,
        source_inventory_sha256=inventory_sha256,
    )
    observations = build_profile_observations(
        slices,
        fit_target_by_trace=target_by_trace,
    )
    trace_ids = sorted(target_by_trace)
    eligible_bases = fully_supported_profile_bases(
        observations,
        expected_trace_ids=trace_ids,
    )
    if len(eligible_bases) < 2:
        raise ValueError(
            "vertical slice has fewer than two fully supported nonzero bases"
        )
    n_clusters = min(requested_clusters, len(eligible_bases))
    cluster_state = cluster_fully_supported_profiles(
        observations,
        expected_trace_ids=trace_ids,
        n_clusters=n_clusters,
    )
    chosen_response_ids = sorted({target_slice.response_id for target_slice in slices})
    longitudinal_counts = {
        response_id: len(multiplex.longitudinal_correspondences(response_id))
        for response_id in chosen_response_ids
    }
    summary: dict[str, Any] = {
        "schema_version": VERTICAL_SLICE_SCHEMA,
        "purpose": "two_dense_response_engineering_smoke",
        "source_inventory_sha256": inventory_sha256,
        "response_ids": chosen_response_ids,
        "targets_per_response": targets_per_response,
        "target_count": len(slices),
        "selected_targets": sorted(
            selected_records,
            key=lambda record: (
                str(record["response_id"]),
                int(record["response_position"]),
            ),
        ),
        "atlas_index_sha256": atlas_index["atlas_index_sha256"],
        "cluster_state_sha256": cluster_state["cluster_state_sha256"],
        "fully_supported_signed_basis_count": len(eligible_bases),
        "longitudinal_correspondence_counts": longitudinal_counts,
        "descriptions_generated": False,
        "scientific_cluster_state": False,
    }
    summary["vertical_slice_sha256"] = canonical_sha256(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _write_json(temporary / "atlas-index.json", atlas_index)
        _write_json(temporary / "smoke-cluster-state.json", cluster_state)
        _write_json(temporary / "summary.json", summary)
        os.replace(temporary, output_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response-id", action="append", default=[])
    parser.add_argument("--targets-per-response", type=int, default=2)
    parser.add_argument("--requested-clusters", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_vertical_slice(
        inventory_path=args.inventory,
        output_path=args.output,
        response_ids=args.response_id,
        targets_per_response=args.targets_per_response,
        requested_clusters=args.requested_clusters,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
