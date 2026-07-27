"""Deterministic virtual compaction of immutable per-response shards."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from circuits.analysis.bonafide.build_plan import (
    task_records,
    validate_downstream_plan,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.streaming import _validate_existing_shard

COMPACTED_FEATURE_SCHEMA = "adag.bonafide.dense-feature-store.v1"
COMPACTED_MULTIPLEX_SCHEMA = "adag.bonafide.response-time-multiplex.v1"

BASIS_INDEX_SCHEMA = pa.schema(
    [
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
    ]
)

CIRCUIT_INPUT_INDEX_SCHEMA = pa.schema(
    [
        pa.field("global_atlas_ci_index", pa.int64(), nullable=False),
        pa.field("atlas_trace_index", pa.int64(), nullable=False),
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("local_ci_index", pa.int32(), nullable=False),
        pa.field("local_label", pa.string(), nullable=False),
    ]
)

OCCURRENCE_INDEX_SCHEMA = pa.schema(
    [
        pa.field("occurrence_index", pa.int64(), nullable=False),
        pa.field("atlas_trace_index", pa.int64(), nullable=False),
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("token_position", pa.int32(), nullable=False),
        pa.field("signed_basis_index", pa.int64(), nullable=False),
    ]
)


def _basis_tuple(record: Mapping[str, Any], prefix: str = "") -> tuple[Any, ...]:
    return (
        str(record[f"{prefix}model_id"]),
        str(record[f"{prefix}model_revision"]),
        int(record[f"{prefix}layer"]),
        int(record[f"{prefix}neuron_index"]),
        str(record[f"{prefix}polarity"]),
    )


def _expected_shard_path(
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
) -> Path:
    return (
        Path(str(plan["output_root"]))
        / "shards"
        / f"task-{int(task['task_index']):03d}-{str(task['response_id'])}"
    )


def _validate_existing_compaction(
    path: Path,
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"compacted output lacks manifest: {path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("compacted manifest must be an object")
    core = dict(manifest)
    recorded_hash = core.pop("manifest_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("compacted manifest hash mismatch")
    if manifest.get("plan_sha256") != plan_sha256:
        raise ValueError("compacted output belongs to another plan")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("compacted index file inventory is invalid")
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("compacted index file record is invalid")
        file_path = path / str(item["path"])
        if file_path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"compacted index size mismatch: {file_path}")
        if file_sha256(file_path) != item["sha256"]:
            raise ValueError(f"compacted index hash mismatch: {file_path}")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("compacted shard inventory is invalid")
    for item in shards:
        if not isinstance(item, Mapping):
            raise ValueError("compacted shard record is invalid")
        shard_path = path.parent / str(item["path"])
        shard_manifest = _validate_existing_shard(
            shard_path,
            plan_sha256=plan_sha256,
            task_index=int(item["task_index"]),
        )
        if shard_manifest.get("manifest_sha256") != item.get("manifest_sha256"):
            raise ValueError("compacted source shard manifest drift")
    return manifest


def _write_parquet(
    path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema
) -> None:
    pq.write_table(
        pa.Table.from_pylist(list(rows), schema=schema),
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def compact_downstream_lane(plan: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_downstream_plan(
        plan,
        verify_inputs=True,
        verify_code=True,
    )
    output_root = Path(str(validated["output_root"]))
    compacted_path = output_root / "compacted"
    if compacted_path.exists():
        manifest = _validate_existing_compaction(
            compacted_path,
            plan_sha256=str(validated["plan_sha256"]),
        )
        return {"status": "skipped_complete", "manifest": manifest}

    tasks = validated["tasks"]
    if not isinstance(tasks, list):
        raise ValueError("downstream build-plan tasks are invalid")
    shard_manifests: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    expected_target_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("downstream build task must be an object")
        task_index = int(task["task_index"])
        _, records = task_records(validated, task_index=task_index)
        expected_target_ids.update(str(record["trace_unit_id"]) for record in records)
        shard_path = _expected_shard_path(validated, task)
        manifest = _validate_existing_shard(
            shard_path,
            plan_sha256=str(validated["plan_sha256"]),
            task_index=task_index,
        )
        shard_manifests.append(manifest)
        shard_paths.append(shard_path)

    target_tables = [
        pq.read_table(shard_path / "targets.parquet") for shard_path in shard_paths
    ]
    targets = pa.concat_tables(target_tables)
    target_rows = targets.to_pylist()
    observed_target_ids = {str(row["trace_unit_id"]) for row in target_rows}
    if observed_target_ids != expected_target_ids:
        raise ValueError("compaction target membership mismatch")
    if len(observed_target_ids) != len(target_rows):
        raise ValueError("compaction contains duplicate target rows")
    target_rows.sort(key=lambda row: int(row["atlas_trace_index"]))
    if [int(row["atlas_trace_index"]) for row in target_rows] != sorted(
        int(row["atlas_trace_index"]) for row in target_rows
    ):
        raise ValueError("target index ordering is invalid")

    lane = str(validated["lane"])
    if lane == "dense_features":
        basis_source_name = "basis-observations.parquet"
        compacted_schema = COMPACTED_FEATURE_SCHEMA
    elif lane == "dense_multiplex":
        basis_source_name = "basis-nodes.parquet"
        compacted_schema = COMPACTED_MULTIPLEX_SCHEMA
    else:
        raise ValueError(f"unsupported downstream lane: {lane}")

    basis_set: set[tuple[Any, ...]] = set()
    basis_columns = [
        "model_id",
        "model_revision",
        "layer",
        "neuron_index",
        "polarity",
    ]
    for shard_path in shard_paths:
        table = pq.read_table(
            shard_path / basis_source_name,
            columns=basis_columns,
        )
        basis_set.update(_basis_tuple(row) for row in table.to_pylist())
    ordered_bases = sorted(basis_set)
    basis_to_index = {basis: index for index, basis in enumerate(ordered_bases)}

    circuit_input_rows: list[dict[str, Any]] = []
    global_ci_index = 0
    for target in target_rows:
        labels = target["local_labels"]
        if len(labels) != int(target["local_ci_count"]):
            raise ValueError("target local circuit-input count mismatch")
        for local_index, label in enumerate(labels):
            circuit_input_rows.append(
                {
                    "global_atlas_ci_index": global_ci_index,
                    "atlas_trace_index": int(target["atlas_trace_index"]),
                    "trace_unit_id": target["trace_unit_id"],
                    "local_ci_index": local_index,
                    "local_label": label,
                }
            )
            global_ci_index += 1

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".compacted.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        pq.write_table(
            pa.Table.from_pylist(target_rows, schema=targets.schema),
            temporary / "target-index.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        _write_parquet(
            temporary / "basis-index.parquet",
            [
                {
                    "signed_basis_index": index,
                    "model_id": basis[0],
                    "model_revision": basis[1],
                    "layer": basis[2],
                    "neuron_index": basis[3],
                    "polarity": basis[4],
                }
                for index, basis in enumerate(ordered_bases)
            ],
            BASIS_INDEX_SCHEMA,
        )
        _write_parquet(
            temporary / "circuit-input-index.parquet",
            circuit_input_rows,
            CIRCUIT_INPUT_INDEX_SCHEMA,
        )

        occurrence_count = 0
        if lane == "dense_multiplex":
            occurrence_rows: list[dict[str, Any]] = []
            target_index_by_trace = {
                str(row["trace_unit_id"]): int(row["atlas_trace_index"])
                for row in target_rows
            }
            occurrence_identity: set[tuple[Any, ...]] = set()
            for shard_path in shard_paths:
                table = pq.read_table(
                    shard_path / "node-occurrences.parquet",
                    columns=[
                        "trace_unit_id",
                        "token_position",
                        *basis_columns,
                    ],
                )
                for row in table.to_pylist():
                    basis = _basis_tuple(row)
                    identity = (
                        str(row["trace_unit_id"]),
                        int(row["token_position"]),
                        *basis[2:],
                    )
                    if identity in occurrence_identity:
                        raise ValueError(
                            "duplicate occurrence identity during compaction"
                        )
                    occurrence_identity.add(identity)
                    occurrence_rows.append(
                        {
                            "atlas_trace_index": target_index_by_trace[
                                str(row["trace_unit_id"])
                            ],
                            "trace_unit_id": row["trace_unit_id"],
                            "token_position": row["token_position"],
                            "signed_basis_index": basis_to_index[basis],
                        }
                    )
            occurrence_rows.sort(
                key=lambda row: (
                    int(row["atlas_trace_index"]),
                    int(row["token_position"]),
                    int(row["signed_basis_index"]),
                )
            )
            for index, row in enumerate(occurrence_rows):
                row["occurrence_index"] = index
            occurrence_count = len(occurrence_rows)
            _write_parquet(
                temporary / "occurrence-index.parquet",
                occurrence_rows,
                OCCURRENCE_INDEX_SCHEMA,
            )

        totals: dict[str, int] = {}
        for shard_manifest in shard_manifests:
            for name, value in shard_manifest.get("totals", {}).items():
                totals[str(name)] = totals.get(str(name), 0) + int(value)
        profile_columns = totals.get("attribution_profile_column_count", 0)
        supported_cells = totals.get("attribution_supported_cell_count", 0)
        nonzero_cells = totals.get("attribution_nonzero_count", 0)
        possible_profile_cells = len(ordered_bases) * profile_columns
        float32_profile_bytes = possible_profile_cells * 4
        float32_similarity_bytes = len(ordered_bases) ** 2 * 4
        resource_estimate = {
            "fitted_signed_basis_count": len(ordered_bases),
            "profile_column_count": profile_columns,
            "supported_profile_cell_count": supported_cells,
            "nonzero_profile_cell_count": nonzero_cells,
            "profile_density_over_global_basis_columns": (
                supported_cells / possible_profile_cells
                if possible_profile_cells
                else 0.0
            ),
            "float32_dense_profile_bytes": float32_profile_bytes,
            "float32_dense_pairwise_similarity_bytes": (float32_similarity_bytes),
            "conservative_pairwise_peak_bytes": float32_similarity_bytes * 3,
            "candidate_sparse_or_chunked_alternative": (
                "retain response Parquet blocks; compute support-aware "
                "similarities blockwise without global dense profile materialization"
            ),
        }

        index_files = sorted(temporary.glob("*.parquet"))
        files = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "row_count": pq.read_metadata(path).num_rows,
            }
            for path in index_files
        ]
        manifest: dict[str, Any] = {
            "schema_version": compacted_schema,
            "plan_sha256": validated["plan_sha256"],
            "lane": lane,
            "source_inventory": validated["source_inventory"],
            "code_revision": validated["code_revision"],
            "target_count": len(target_rows),
            "response_count": len(shard_paths),
            "circuit_input_count": len(circuit_input_rows),
            "signed_basis_count": len(ordered_bases),
            "occurrence_count": occurrence_count,
            "totals": dict(sorted(totals.items())),
            "resource_estimate": resource_estimate,
            "shards": [
                {
                    "task_index": shard_manifest["task_index"],
                    "path": str(shard_path.relative_to(output_root)),
                    "manifest_sha256": shard_manifest["manifest_sha256"],
                    "shard_identity_sha256": shard_manifest["shard_identity_sha256"],
                    "files": shard_manifest["files"],
                }
                for shard_path, shard_manifest in zip(
                    shard_paths,
                    shard_manifests,
                    strict=True,
                )
            ],
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, compacted_path)
        return {"status": "complete", "manifest": manifest}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
