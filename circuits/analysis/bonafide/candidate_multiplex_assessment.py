"""Provenance-bound C2 candidate and width-one multiplex assessment inputs.

This module joins measurements only at their valid scope.  Candidate contrasts
remain target/basis signed sums; occurrence projection rows contain pointers and
cluster identities, never an invented per-token candidate value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from circuits.analysis.bonafide.candidate_clustering import (
    load_candidate_cluster_input_bundle,
)
from circuits.analysis.bonafide.candidate_clustering_execution import (
    ASSIGNMENT_SCHEMA,
    ASSIGNMENTS_FILE,
    CANDIDATE_CLUSTER_BASELINE_SCHEMA,
    COMMON_ELIGIBILITY_FILE,
    COMMON_ELIGIBILITY_SCHEMA,
    _publish_directory_no_replace,
    load_candidate_clustering_baseline,
)
from circuits.analysis.bonafide.candidate_profiles import (
    BASIS_INDEX_SCHEMA as C2_BASIS_INDEX_SCHEMA,
)
from circuits.analysis.bonafide.candidate_profiles import (
    CANDIDATE_CLUSTER_INPUT_SCHEMA,
    CANDIDATE_PROFILE_SCHEMA,
    WIDTH_PROFILE_SCHEMA,
)
from circuits.analysis.bonafide.candidate_profiles import (
    TARGET_SCHEMA as C2_TARGET_SCHEMA,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.analysis.bonafide.compaction import (
    BASIS_INDEX_SCHEMA as DENSE_BASIS_INDEX_SCHEMA,
)
from circuits.analysis.bonafide.compaction import (
    CIRCUIT_INPUT_INDEX_SCHEMA,
    COMPACTED_MULTIPLEX_SCHEMA,
    OCCURRENCE_INDEX_SCHEMA,
    _validate_existing_compaction,
)
from circuits.analysis.bonafide.streaming import TARGET_SCHEMA as DENSE_TARGET_SCHEMA

ASSESSMENT_SCHEMA_VERSION = "adag.bonafide.candidate-multiplex-assessment.v1"
CANDIDATE_MEASUREMENT_SCOPE = "target_basis_signed_sum"

TARGET_CROSSWALK_FILE = "target-crosswalk.parquet"
TARGET_BASIS_FILE = "target-basis-assessment.parquet"
OCCURRENCE_PROJECTION_FILE = "occurrence-projection.parquet"

_BASIS_FIELDS = (
    "model_id",
    "model_revision",
    "layer",
    "neuron_index",
    "polarity",
)
_PRODUCING_SOURCE_PATHS = (
    "circuits/analysis/bonafide/candidate_multiplex_assessment.py",
    "scripts/bonafide/candidate_multiplex_assess.py",
)

TARGET_CROSSWALK_SCHEMA = pa.schema(
    [
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("source_width1_artifact_id", pa.string(), nullable=False),
        pa.field("width1_artifact_id", pa.string(), nullable=False),
        pa.field("width1_payload_sha256", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("base_question_id", pa.string(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("phase_bin", pa.int8(), nullable=False),
        pa.field("family_partition", pa.string(), nullable=False),
        pa.field("partition_hierarchical_weight", pa.float64(), nullable=False),
        pa.field("dense_target_match", pa.bool_(), nullable=False),
        pa.field("dense_atlas_trace_index", pa.int64(), nullable=True),
        pa.field("dense_trace_unit_id", pa.string(), nullable=True),
        pa.field("dense_source_artifact_id", pa.string(), nullable=True),
        pa.field("dense_artifact_payload_sha256", pa.string(), nullable=True),
        pa.field("join_validation", pa.string(), nullable=False),
        pa.field("missing_reason", pa.string(), nullable=True),
    ]
)

TARGET_BASIS_ASSESSMENT_SCHEMA = pa.schema(
    [
        pa.field("assessment_row_index", pa.int64(), nullable=False),
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
        pa.field("c2_w64_assigned", pa.bool_(), nullable=False),
        pa.field("c2_w64_cluster_id", pa.int32(), nullable=True),
        pa.field("family_partition", pa.string(), nullable=False),
        pa.field("base_question_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("phase_bin", pa.int8(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("partition_hierarchical_weight", pa.float64(), nullable=False),
        pa.field("width_profile_available", pa.bool_(), nullable=False),
        pa.field("width_signed_attribution", pa.float64(), nullable=True),
        pa.field("width_attribution_profile", pa.list_(pa.float64()), nullable=True),
        pa.field("width_attribution_support", pa.list_(pa.bool_()), nullable=True),
        pa.field("width_occurrence_count", pa.int32(), nullable=True),
        pa.field("candidate_profile_available", pa.bool_(), nullable=False),
        pa.field("candidate_contrast_vector", pa.list_(pa.float64()), nullable=True),
        pa.field("candidate_profile_l2_norm", pa.float64(), nullable=True),
        pa.field("candidate_occurrence_count", pa.int32(), nullable=True),
        pa.field("candidate_measurement_scope", pa.string(), nullable=False),
        pa.field("dense_target_match", pa.bool_(), nullable=False),
        pa.field("dense_basis_match", pa.bool_(), nullable=False),
        pa.field("dense_target_basis_occurrence_match", pa.bool_(), nullable=False),
        pa.field("dense_atlas_trace_index", pa.int64(), nullable=True),
        pa.field("dense_signed_basis_index", pa.int64(), nullable=True),
        pa.field("dense_occurrence_count", pa.int64(), nullable=False),
        pa.field("missing_reasons", pa.list_(pa.string()), nullable=False),
    ]
)

OCCURRENCE_PROJECTION_SCHEMA = pa.schema(
    [
        pa.field("occurrence_index", pa.int64(), nullable=False),
        pa.field("atlas_trace_index", pa.int64(), nullable=False),
        pa.field("trace_unit_id", pa.string(), nullable=False),
        pa.field("token_position", pa.int32(), nullable=False),
        pa.field("dense_signed_basis_index", pa.int64(), nullable=False),
        pa.field("target_basis_assessment_row_index", pa.int64(), nullable=False),
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("c2_signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
        pa.field("c2_w64_assigned", pa.bool_(), nullable=False),
        pa.field("c2_w64_cluster_id", pa.int32(), nullable=True),
        pa.field("candidate_profile_available", pa.bool_(), nullable=False),
        pa.field("candidate_measurement_scope", pa.string(), nullable=False),
    ]
)

_C2_INPUT_SCHEMAS = {
    "basis-index.parquet": C2_BASIS_INDEX_SCHEMA,
    "targets.parquet": C2_TARGET_SCHEMA,
    "width-profiles.parquet": WIDTH_PROFILE_SCHEMA,
    "candidate-profiles.parquet": CANDIDATE_PROFILE_SCHEMA,
}
_BASELINE_SCHEMAS = {
    ASSIGNMENTS_FILE: ASSIGNMENT_SCHEMA,
    COMMON_ELIGIBILITY_FILE: COMMON_ELIGIBILITY_SCHEMA,
}
_DENSE_SCHEMAS = {
    "basis-index.parquet": DENSE_BASIS_INDEX_SCHEMA,
    "circuit-input-index.parquet": CIRCUIT_INPUT_INDEX_SCHEMA,
    "occurrence-index.parquet": OCCURRENCE_INDEX_SCHEMA,
    "target-index.parquet": DENSE_TARGET_SCHEMA,
}
_OUTPUT_SCHEMAS = {
    TARGET_CROSSWALK_FILE: TARGET_CROSSWALK_SCHEMA,
    TARGET_BASIS_FILE: TARGET_BASIS_ASSESSMENT_SCHEMA,
    OCCURRENCE_PROJECTION_FILE: OCCURRENCE_PROJECTION_SCHEMA,
}


@dataclass(frozen=True)
class LoadedCandidateMultiplexAssessment:
    """Fully validated derived tables and their provenance manifest."""

    root: Path
    manifest: Mapping[str, Any]
    target_crosswalk: pa.Table
    target_basis_assessment: pa.Table
    occurrence_projection: pa.Table | None


@dataclass(frozen=True)
class _DerivedAssessment:
    crosswalk_rows: tuple[Mapping[str, Any], ...]
    assessment_rows: tuple[Mapping[str, Any], ...]
    projection_rows: tuple[Mapping[str, Any], ...]
    c2_basis_count: int
    width_profile_count: int
    candidate_profile_count: int


def _git(repo_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=not binary,
    )
    if binary:
        return completed.stdout
    return completed.stdout.strip()


def _collect_producing_revision(repo_root: Path) -> dict[str, Any]:
    """Bind the clean committed module and CLI that produced the artifact."""

    repo_root = repo_root.resolve()
    if (
        Path(str(_git(repo_root, "rev-parse", "--show-toplevel"))).resolve()
        != repo_root
    ):
        raise ValueError("assessment must run from the repository root")
    status = str(_git(repo_root, "status", "--porcelain=v1", "--untracked-files=no"))
    if status:
        raise ValueError("assessment requires a clean tracked worktree")
    commit = str(_git(repo_root, "rev-parse", "HEAD"))
    tree = str(_git(repo_root, "rev-parse", "HEAD^{tree}"))
    files: list[dict[str, Any]] = []
    for relative in _PRODUCING_SOURCE_PATHS:
        if _git(repo_root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"assessment producing source is not tracked: {relative}")
        path = repo_root / relative
        blob = str(_git(repo_root, "rev-parse", f"HEAD:{relative}"))
        if _git(repo_root, "hash-object", relative) != blob:
            raise ValueError(
                f"assessment producing source differs from HEAD: {relative}"
            )
        files.append(
            {
                "path": relative,
                "git_blob": blob,
                "sha256": file_sha256(path),
            }
        )
    return {
        "repo_root": str(repo_root),
        "git_commit": commit,
        "git_tree": tree,
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "files": files,
    }


def _validate_producing_revision(revision: Mapping[str, Any]) -> None:
    """Validate recorded Git objects, without requiring the producing checkout HEAD."""

    recorded_root = Path(str(revision.get("repo_root"))).resolve()
    current_root = Path(__file__).resolve().parents[3]
    repo_root = recorded_root if (recorded_root / ".git").exists() else current_root
    commit = str(revision.get("git_commit"))
    tree = str(revision.get("git_tree"))
    if (
        revision.get("tracked_worktree_clean") is not True
        or revision.get("tracked_status_sha256") != hashlib.sha256(b"").hexdigest()
        or str(_git(repo_root, "rev-parse", f"{commit}^{{tree}}")) != tree
    ):
        raise ValueError("assessment producing revision drift")
    records = revision.get("files")
    if not isinstance(records, list) or len(records) != len(_PRODUCING_SOURCE_PATHS):
        raise ValueError("assessment producing source inventory drift")
    by_path = {
        str(record.get("path")): record
        for record in records
        if isinstance(record, Mapping)
    }
    if set(by_path) != set(_PRODUCING_SOURCE_PATHS):
        raise ValueError("assessment producing source inventory drift")
    for relative in _PRODUCING_SOURCE_PATHS:
        record = by_path[relative]
        blob = str(_git(repo_root, "rev-parse", f"{commit}:{relative}"))
        content = _git(repo_root, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(content, bytes)
        if (
            record.get("git_blob") != blob
            or record.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"assessment producing source object drift: {relative}")


def _exact_table(path: Path, schema: pa.Schema, *, label: str) -> pa.Table:
    table = pq.read_table(path)
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"{label} parquet schema drift: {path.name}")
    return table


def _validated_manifest(
    root: Path,
    *,
    schema_version: str,
    label: str,
    parquet_schemas: Mapping[str, pa.Schema],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = load_json_object(manifest_path)
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core):
        raise ValueError(f"{label} manifest self-hash mismatch")
    if manifest.get("schema_version") != schema_version:
        raise ValueError(f"unsupported {label} schema")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TypeError(f"{label} file inventory is invalid")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(f"{label} file record is invalid")
        name = str(record.get("path"))
        if Path(name).name != name or name in seen:
            raise ValueError(f"{label} file path is unsafe or duplicated")
        seen.add(name)
        path = root / name
        if not path.is_file():
            raise ValueError(f"{label} file is missing: {name}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"{label} file size drift: {name}")
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"{label} file hash drift: {name}")
        if path.suffix == ".parquet" and pq.read_metadata(path).num_rows != int(
            record.get("row_count", -1)
        ):
            raise ValueError(f"{label} parquet row-count drift: {name}")
    missing = set(parquet_schemas) - seen
    if missing:
        raise ValueError(
            f"{label} required file inventory is incomplete: {sorted(missing)}"
        )
    for name, schema in parquet_schemas.items():
        _exact_table(root / name, schema, label=label)
    return manifest


def _source_record(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    return {
        "path": str(root),
        "manifest_path": str(manifest_path),
        "schema_version": manifest["schema_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": file_sha256(manifest_path),
    }


def _validate_source_record(
    record: Mapping[str, Any],
    *,
    schema_version: str,
    label: str,
    parquet_schemas: Mapping[str, pa.Schema],
) -> tuple[Path, dict[str, Any]]:
    root = Path(str(record.get("path"))).resolve()
    if Path(str(record.get("manifest_path"))).resolve() != root / "manifest.json":
        raise ValueError(f"{label} manifest path binding drift")
    manifest = _validated_manifest(
        root,
        schema_version=schema_version,
        label=label,
        parquet_schemas=parquet_schemas,
    )
    if (
        record.get("schema_version") != schema_version
        or record.get("manifest_sha256") != manifest["manifest_sha256"]
        or record.get("manifest_file_sha256") != file_sha256(root / "manifest.json")
    ):
        raise ValueError(f"{label} source binding drift")
    return root, manifest


def _validate_source_firewalls(
    input_manifest: Mapping[str, Any], baseline_manifest: Mapping[str, Any]
) -> None:
    for field in (
        "outcomes_inspected",
        "model_calls_made",
        "cluster_fit_performed",
        "confirmatory_holdout_opened",
    ):
        if input_manifest.get(field) is not False:
            raise ValueError(f"C2 input violates {field} firewall")
    for field in (
        "outcomes_inspected",
        "descriptions_generated",
        "model_calls_made",
        "confirmatory_holdout_opened",
    ):
        if baseline_manifest.get(field) is not False:
            raise ValueError(f"C2 baseline violates {field} firewall")


def _validate_baseline_input_binding(
    *,
    input_root: Path,
    input_manifest: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
) -> None:
    source = baseline_manifest.get("source_input_bundle")
    if not isinstance(source, Mapping):
        raise TypeError("C2 baseline source input binding is invalid")
    if (
        Path(str(source.get("path"))).resolve() != input_root
        or Path(str(source.get("manifest_path"))).resolve()
        != input_root / "manifest.json"
        or source.get("schema_version") != input_manifest["schema_version"]
        or source.get("manifest_sha256") != input_manifest["manifest_sha256"]
        or source.get("manifest_file_sha256")
        != file_sha256(input_root / "manifest.json")
    ):
        raise ValueError("C2 baseline source input binding drift")


def _validate_dense_relations(root: Path, manifest: Mapping[str, Any]) -> None:
    """Validate compacted index primary keys, foreign keys, and recorded counts."""

    targets = _exact_table(
        root / "target-index.parquet", DENSE_TARGET_SCHEMA, label="dense multiplex"
    ).to_pylist()
    bases = _exact_table(
        root / "basis-index.parquet", DENSE_BASIS_INDEX_SCHEMA, label="dense multiplex"
    ).to_pylist()
    occurrences = _exact_table(
        root / "occurrence-index.parquet",
        OCCURRENCE_INDEX_SCHEMA,
        label="dense multiplex",
    ).to_pylist()
    circuit_inputs = _exact_table(
        root / "circuit-input-index.parquet",
        CIRCUIT_INPUT_INDEX_SCHEMA,
        label="dense multiplex",
    ).to_pylist()
    target_indices = [int(row["atlas_trace_index"]) for row in targets]
    if target_indices != sorted(target_indices) or len(set(target_indices)) != len(
        target_indices
    ):
        raise ValueError("dense target primary-key ordering or uniqueness drift")
    if [int(row["signed_basis_index"]) for row in bases] != list(range(len(bases))):
        raise ValueError("dense basis primary-key ordering drift")
    if [int(row["occurrence_index"]) for row in occurrences] != list(
        range(len(occurrences))
    ):
        raise ValueError("dense occurrence primary-key ordering drift")
    if [int(row["global_atlas_ci_index"]) for row in circuit_inputs] != list(
        range(len(circuit_inputs))
    ):
        raise ValueError("dense circuit-input primary-key ordering drift")
    targets_by_index = {int(row["atlas_trace_index"]): row for row in targets}
    basis_indices = {int(row["signed_basis_index"]) for row in bases}
    for occurrence in occurrences:
        target = targets_by_index.get(int(occurrence["atlas_trace_index"]))
        if (
            target is None
            or occurrence["trace_unit_id"] != target["trace_unit_id"]
            or int(occurrence["signed_basis_index"]) not in basis_indices
        ):
            raise ValueError("dense occurrence foreign-key drift")
    for circuit_input in circuit_inputs:
        target = targets_by_index.get(int(circuit_input["atlas_trace_index"]))
        local_index = int(circuit_input["local_ci_index"])
        if (
            target is None
            or circuit_input["trace_unit_id"] != target["trace_unit_id"]
            or not 0 <= local_index < int(target["local_ci_count"])
            or circuit_input["local_label"] != target["local_labels"][local_index]
        ):
            raise ValueError("dense circuit-input foreign-key drift")
    expected_counts = {
        "target_count": len(targets),
        "signed_basis_count": len(bases),
        "occurrence_count": len(occurrences),
        "circuit_input_count": len(circuit_inputs),
    }
    for field, expected in expected_counts.items():
        if int(manifest.get(field, -1)) != expected:
            raise ValueError(f"dense compacted count drift: {field}")


def _deep_validate_source_artifacts(
    *,
    input_root: Path,
    input_manifest: Mapping[str, Any],
    baseline_root: Path,
    baseline_manifest: Mapping[str, Any],
    dense_root: Path,
    dense_manifest: Mapping[str, Any],
) -> None:
    """Run canonical deep loaders plus compacted relational validation."""

    bundle = load_candidate_cluster_input_bundle(input_root)
    if bundle.root != input_root or bundle.manifest.get(
        "manifest_sha256"
    ) != input_manifest.get("manifest_sha256"):
        raise ValueError("deep C2 input validation returned a different artifact")
    baseline = load_candidate_clustering_baseline(baseline_root, verify_source=True)
    if baseline.root != baseline_root or baseline.manifest.get(
        "manifest_sha256"
    ) != baseline_manifest.get("manifest_sha256"):
        raise ValueError("deep C2 baseline validation returned a different artifact")
    plan_sha256 = dense_manifest.get("plan_sha256")
    if not isinstance(plan_sha256, str) or not plan_sha256:
        raise ValueError("dense compacted artifact lacks its source plan identity")
    reloaded_dense = _validate_existing_compaction(dense_root, plan_sha256=plan_sha256)
    if reloaded_dense.get("manifest_sha256") != dense_manifest.get("manifest_sha256"):
        raise ValueError("deep dense compaction validation returned another artifact")
    _validate_dense_relations(dense_root, dense_manifest)


def _basis_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in _BASIS_FIELDS)


def _unique_index(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    label: str,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"duplicate {label} identity: {key!r}")
        result[key] = row
    return result


def _target_crosswalk(
    c2_targets: Sequence[Mapping[str, Any]],
    dense_targets: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    dense_indexes = {
        "source": _unique_index(
            dense_targets, ("source_artifact_id",), label="dense source artifact"
        ),
        "trace": _unique_index(dense_targets, ("trace_unit_id",), label="dense trace"),
        "logical": _unique_index(
            dense_targets,
            ("response_id", "base_question_id", "response_position"),
            label="dense response/base-question/position",
        ),
        "payload": _unique_index(
            dense_targets, ("artifact_payload_sha256",), label="dense payload"
        ),
    }
    rows: list[dict[str, Any]] = []
    matches: dict[str, Mapping[str, Any]] = {}
    seen_cases: set[str] = set()
    for c2 in sorted(c2_targets, key=lambda row: str(row["case_id"])):
        case_id = str(c2["case_id"])
        if case_id in seen_cases:
            raise ValueError(f"duplicate C2 case identity: {case_id}")
        seen_cases.add(case_id)
        candidates = {
            "source": dense_indexes["source"].get((c2["source_width1_artifact_id"],)),
            "trace": dense_indexes["trace"].get((c2["width1_artifact_id"],)),
            "logical": dense_indexes["logical"].get(
                (
                    c2["response_id"],
                    c2["base_question_id"],
                    c2["response_position"],
                )
            ),
            "payload": dense_indexes["payload"].get((c2["width1_payload_sha256"],)),
        }
        present = [row for row in candidates.values() if row is not None]
        if not present:
            dense = None
            validation = "no_dense_counterpart_expected_partial_overlap"
            reason = "dense_target_unmatched"
        else:
            if len(present) != len(candidates) or any(
                row is not present[0] for row in present
            ):
                raise ValueError(
                    f"C2/dense target identity conflict for {case_id}: "
                    "source, trace, logical, and payload joins must agree exactly"
                )
            dense = present[0]
            exact_fields = {
                "source_artifact_id": c2["source_width1_artifact_id"],
                "trace_unit_id": c2["width1_artifact_id"],
                "response_id": c2["response_id"],
                "base_question_id": c2["base_question_id"],
                "response_position": c2["response_position"],
                "artifact_payload_sha256": c2["width1_payload_sha256"],
            }
            if any(
                dense[field] != expected for field, expected in exact_fields.items()
            ):
                raise ValueError(f"C2/dense exact target identity drift for {case_id}")
            matches[case_id] = dense
            validation = "exact_redundant_source_trace_logical_payload_match"
            reason = None
        rows.append(
            {
                "case_id": case_id,
                "source_width1_artifact_id": c2["source_width1_artifact_id"],
                "width1_artifact_id": c2["width1_artifact_id"],
                "width1_payload_sha256": c2["width1_payload_sha256"],
                "response_id": c2["response_id"],
                "base_question_id": c2["base_question_id"],
                "response_position": c2["response_position"],
                "phase_bin": c2["phase_bin"],
                "family_partition": c2["family_partition"],
                "partition_hierarchical_weight": c2["partition_hierarchical_weight"],
                "dense_target_match": dense is not None,
                "dense_atlas_trace_index": (
                    int(dense["atlas_trace_index"]) if dense is not None else None
                ),
                "dense_trace_unit_id": (
                    str(dense["trace_unit_id"]) if dense is not None else None
                ),
                "dense_source_artifact_id": (
                    str(dense["source_artifact_id"]) if dense is not None else None
                ),
                "dense_artifact_payload_sha256": (
                    str(dense["artifact_payload_sha256"]) if dense is not None else None
                ),
                "join_validation": validation,
                "missing_reason": reason,
            }
        )
    return rows, matches


def _validate_profile_row(
    row: Mapping[str, Any],
    *,
    basis_by_index: Mapping[int, Mapping[str, Any]],
    candidate: bool,
) -> None:
    index = int(row["signed_basis_index"])
    basis = basis_by_index.get(index)
    if basis is None or _basis_identity(row) != _basis_identity(basis):
        raise ValueError("C2 profile signed-basis identity drift")
    count = row.get("occurrence_count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("C2 profile occurrence count is invalid")
    if candidate:
        vector = row["candidate_contrast_profile"]
        if len(vector) != 5 or any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("C2 candidate contrast vector must be finite width five")
        norm = math.sqrt(sum(float(value) ** 2 for value in vector))
        if not math.isclose(
            norm,
            float(row["candidate_profile_l2_norm"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("C2 candidate profile norm drift")
    else:
        profile = row["attribution_profile"]
        support = row["attribution_support"]
        if not profile or len(profile) != len(support):
            raise ValueError("C2 width profile/support shape drift")
        for value, supported in zip(profile, support, strict=True):
            if supported and (value is None or not math.isfinite(float(value))):
                raise ValueError("supported C2 width attribution is nonfinite")
            if not supported and value is not None:
                raise ValueError("unsupported C2 width attribution must be null")


def _c2_w64_assignments(
    assignment_rows: Sequence[Mapping[str, Any]],
    *,
    basis_by_index: Mapping[int, Mapping[str, Any]],
    baseline_manifest: Mapping[str, Any],
) -> dict[int, int | None]:
    configuration = baseline_manifest.get("configuration")
    if (
        baseline_manifest.get("chosen_cluster_count") != 64
        or not isinstance(configuration, Mapping)
        or "W" not in configuration.get("views", [])
        or 64 not in configuration.get("directional_cluster_counts", [])
    ):
        raise ValueError("C2 baseline does not select the configured W64 resolution")
    states = baseline_manifest.get("states")
    if not isinstance(states, list):
        raise TypeError("C2 baseline state inventory is invalid")
    medoid_states = [
        state
        for state in states
        if isinstance(state, Mapping)
        and state.get("view") == "W"
        and state.get("n_clusters") == 64
        and state.get("is_medoid") is True
    ]
    if len(medoid_states) != 1:
        raise ValueError("C2 baseline must select exactly one W64 medoid state")
    state = medoid_states[0]
    if (
        state.get("fit_valid") is not True
        or state.get("seed_valid") is not True
        or state.get("fit_error") is not None
        or not isinstance(state.get("seed"), int)
    ):
        raise ValueError("C2 W64 medoid fit/seed is invalid")
    assignment_fraction = float(state.get("assignment_fraction", float("nan")))
    if not math.isfinite(assignment_fraction) or not 0.0 <= assignment_fraction <= 1.0:
        raise ValueError("C2 W64 medoid assignment fraction is invalid")
    w64 = [row for row in assignment_rows if row["state_index"] == state["state_index"]]
    if len(w64) != len(basis_by_index):
        raise ValueError("C2 baseline must contain one complete medoid W64 basis block")
    result: dict[int, int | None] = {}
    for row in w64:
        for field in (
            "view",
            "n_clusters",
            "seed",
            "fit_valid",
            "seed_valid",
            "is_medoid",
            "assignment_fraction",
            "fit_error",
        ):
            if row[field] != state[field]:
                raise ValueError(f"C2 W64 assignment state drift: {field}")
        index = int(row["signed_basis_index"])
        basis = basis_by_index.get(index)
        if basis is None or _basis_identity(row) != _basis_identity(basis):
            raise ValueError("C2 W64 assignment basis identity drift")
        if index in result:
            raise ValueError("duplicate C2 W64 basis assignment")
        cluster = row["cluster_id"]
        assigned = cluster is not None
        if bool(row["assigned"]) != assigned:
            raise ValueError("C2 W64 assignment nullability drift")
        if cluster is not None and not 0 <= int(cluster) < 64:
            raise ValueError("C2 W64 cluster ID is invalid")
        result[index] = None if cluster is None else int(cluster)
    return result


def _profile_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    basis_by_index: Mapping[int, Mapping[str, Any]],
    target_ids: set[str],
    candidate: bool,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in target_ids:
            raise ValueError("C2 profile refers to an unknown target")
        _validate_profile_row(row, basis_by_index=basis_by_index, candidate=candidate)
        key = (case_id, int(row["signed_basis_index"]))
        if key in result:
            raise ValueError("duplicate C2 target/basis profile row")
        result[key] = row
    if {case_id for case_id, _ in result} != target_ids:
        raise ValueError("C2 profiles do not retain every target")
    return result


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
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _coverage_metrics(
    crosswalk_rows: Sequence[Mapping[str, Any]],
    assessment_rows: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return deterministic label-free coverage and missingness diagnostics."""

    targets_by_case = {str(row["case_id"]): row for row in crosswalk_rows}

    def summarize(case_ids: set[str]) -> dict[str, Any]:
        target_rows = [targets_by_case[case_id] for case_id in sorted(case_ids)]
        basis_rows = [row for row in assessment_rows if str(row["case_id"]) in case_ids]
        projected = [row for row in projection_rows if str(row["case_id"]) in case_ids]
        missing: dict[str, int] = defaultdict(int)
        for row in basis_rows:
            for reason in row["missing_reasons"]:
                missing[str(reason)] += 1
        return {
            "target_count": len(target_rows),
            "dense_matched_target_count": sum(
                bool(row["dense_target_match"]) for row in target_rows
            ),
            "target_basis_union_row_count": len(basis_rows),
            "width_profile_available_row_count": sum(
                bool(row["width_profile_available"]) for row in basis_rows
            ),
            "candidate_profile_available_row_count": sum(
                bool(row["candidate_profile_available"]) for row in basis_rows
            ),
            "both_profiles_available_row_count": sum(
                bool(row["width_profile_available"])
                and bool(row["candidate_profile_available"])
                for row in basis_rows
            ),
            "c2_w64_assigned_row_count": sum(
                bool(row["c2_w64_assigned"]) for row in basis_rows
            ),
            "dense_basis_match_row_count": sum(
                bool(row["dense_basis_match"]) for row in basis_rows
            ),
            "dense_target_basis_occurrence_match_row_count": sum(
                bool(row["dense_target_basis_occurrence_match"]) for row in basis_rows
            ),
            "dense_occurrence_count": sum(
                int(row["dense_occurrence_count"]) for row in basis_rows
            ),
            "occurrence_projection_row_count": len(projected),
            "missing_reason_row_counts": dict(sorted(missing.items())),
        }

    all_cases = set(targets_by_case)
    partitions = sorted({str(row["family_partition"]) for row in crosswalk_rows})
    phases = sorted({int(row["phase_bin"]) for row in crosswalk_rows})
    return {
        "overall": summarize(all_cases),
        "by_family_partition": {
            partition: summarize(
                {
                    str(row["case_id"])
                    for row in crosswalk_rows
                    if str(row["family_partition"]) == partition
                }
            )
            for partition in partitions
        },
        "by_c2_phase_bin": {
            str(phase): summarize(
                {
                    str(row["case_id"])
                    for row in crosswalk_rows
                    if int(row["phase_bin"]) == phase
                }
            )
            for phase in phases
        },
    }


def _derive_assessment_tables(
    *,
    input_root: Path,
    input_manifest: Mapping[str, Any],
    baseline_root: Path,
    baseline_manifest: Mapping[str, Any],
    dense_root: Path,
    dense_manifest: Mapping[str, Any],
    include_occurrence_projection: bool,
) -> _DerivedAssessment:
    """Pure deterministic derivation from already validated source artifacts."""

    c2_basis = _exact_table(
        input_root / "basis-index.parquet", C2_BASIS_INDEX_SCHEMA, label="C2 input"
    ).to_pylist()
    basis_by_index = {int(row["signed_basis_index"]): row for row in c2_basis}
    if len(basis_by_index) != len(c2_basis) or sorted(basis_by_index) != list(
        range(len(c2_basis))
    ):
        raise ValueError("C2 signed-basis index is not canonical and contiguous")

    c2_targets = _exact_table(
        input_root / "targets.parquet", C2_TARGET_SCHEMA, label="C2 input"
    ).to_pylist()
    expected_target_count = int(
        input_manifest.get("cohort", {}).get("target_count", -1)
    )
    if expected_target_count < 1 or len(c2_targets) != expected_target_count:
        raise ValueError("C2 target count disagrees with its immutable manifest")
    target_ids = {str(row["case_id"]) for row in c2_targets}
    if len(target_ids) != len(c2_targets):
        raise ValueError("C2 target identities are not unique")

    dense_targets = _exact_table(
        dense_root / "target-index.parquet",
        DENSE_TARGET_SCHEMA,
        label="dense multiplex",
    ).to_pylist()
    if len(dense_targets) != int(dense_manifest.get("target_count", -1)):
        raise ValueError("dense target count disagrees with its compacted manifest")
    crosswalk_rows, dense_matches = _target_crosswalk(c2_targets, dense_targets)

    width_rows = _exact_table(
        input_root / "width-profiles.parquet", WIDTH_PROFILE_SCHEMA, label="C2 input"
    ).to_pylist()
    candidate_rows = _exact_table(
        input_root / "candidate-profiles.parquet",
        CANDIDATE_PROFILE_SCHEMA,
        label="C2 input",
    ).to_pylist()
    width_by_key = _profile_index(
        width_rows,
        basis_by_index=basis_by_index,
        target_ids=target_ids,
        candidate=False,
    )
    candidate_by_key = _profile_index(
        candidate_rows,
        basis_by_index=basis_by_index,
        target_ids=target_ids,
        candidate=True,
    )
    assignment_rows = _exact_table(
        baseline_root / ASSIGNMENTS_FILE, ASSIGNMENT_SCHEMA, label="C2 baseline"
    ).to_pylist()
    w64_by_basis = _c2_w64_assignments(
        assignment_rows,
        basis_by_index=basis_by_index,
        baseline_manifest=baseline_manifest,
    )

    dense_basis_rows = _exact_table(
        dense_root / "basis-index.parquet",
        DENSE_BASIS_INDEX_SCHEMA,
        label="dense multiplex",
    ).to_pylist()
    dense_basis_by_identity = {
        _basis_identity(row): int(row["signed_basis_index"]) for row in dense_basis_rows
    }
    if len(dense_basis_by_identity) != len(dense_basis_rows):
        raise ValueError("dense multiplex contains duplicate signed-basis identities")

    occurrences = _exact_table(
        dense_root / "occurrence-index.parquet",
        OCCURRENCE_INDEX_SCHEMA,
        label="dense multiplex",
    ).to_pylist()
    occurrence_by_target_basis: dict[tuple[int, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for occurrence in occurrences:
        occurrence_by_target_basis[
            (
                int(occurrence["atlas_trace_index"]),
                int(occurrence["signed_basis_index"]),
            )
        ].append(occurrence)

    target_by_case = {str(row["case_id"]): row for row in c2_targets}
    union_keys = sorted(set(width_by_key) | set(candidate_by_key))
    if {case_id for case_id, _ in union_keys} != target_ids:
        raise ValueError("target/basis profile union does not retain every C2 target")

    assessment_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for row_index, (case_id, basis_index) in enumerate(union_keys):
        target = target_by_case[case_id]
        basis = basis_by_index[basis_index]
        width = width_by_key.get((case_id, basis_index))
        candidate = candidate_by_key.get((case_id, basis_index))
        cluster_id = w64_by_basis[basis_index]
        dense_target = dense_matches.get(case_id)
        dense_basis_index = dense_basis_by_identity.get(_basis_identity(basis))
        dense_occurrences: list[Mapping[str, Any]] = []
        if dense_target is not None and dense_basis_index is not None:
            dense_occurrences = occurrence_by_target_basis.get(
                (int(dense_target["atlas_trace_index"]), dense_basis_index), []
            )
        missing: list[str] = []
        if width is None:
            missing.append("width_profile_missing")
        if candidate is None:
            missing.append("candidate_profile_missing")
        if cluster_id is None:
            missing.append("c2_w64_assignment_missing")
        if dense_target is None:
            missing.append("dense_target_unmatched")
        elif dense_basis_index is None:
            missing.append("dense_basis_unmatched")
        elif not dense_occurrences:
            missing.append("dense_target_basis_occurrence_missing")

        assessment = {
            "assessment_row_index": row_index,
            "case_id": case_id,
            "signed_basis_index": basis_index,
            **{field: basis[field] for field in _BASIS_FIELDS},
            "c2_w64_assigned": cluster_id is not None,
            "c2_w64_cluster_id": cluster_id,
            "family_partition": target["family_partition"],
            "base_question_id": target["base_question_id"],
            "response_id": target["response_id"],
            "phase_bin": target["phase_bin"],
            "response_position": target["response_position"],
            "partition_hierarchical_weight": target["partition_hierarchical_weight"],
            "width_profile_available": width is not None,
            "width_signed_attribution": (
                float(width["signed_attribution"]) if width is not None else None
            ),
            "width_attribution_profile": (
                width["attribution_profile"] if width is not None else None
            ),
            "width_attribution_support": (
                width["attribution_support"] if width is not None else None
            ),
            "width_occurrence_count": (
                int(width["occurrence_count"]) if width is not None else None
            ),
            "candidate_profile_available": candidate is not None,
            "candidate_contrast_vector": (
                candidate["candidate_contrast_profile"]
                if candidate is not None
                else None
            ),
            "candidate_profile_l2_norm": (
                float(candidate["candidate_profile_l2_norm"])
                if candidate is not None
                else None
            ),
            "candidate_occurrence_count": (
                int(candidate["occurrence_count"]) if candidate is not None else None
            ),
            "candidate_measurement_scope": CANDIDATE_MEASUREMENT_SCOPE,
            "dense_target_match": dense_target is not None,
            "dense_basis_match": dense_basis_index is not None,
            "dense_target_basis_occurrence_match": bool(dense_occurrences),
            "dense_atlas_trace_index": (
                int(dense_target["atlas_trace_index"])
                if dense_target is not None
                else None
            ),
            "dense_signed_basis_index": dense_basis_index,
            "dense_occurrence_count": len(dense_occurrences),
            "missing_reasons": missing,
        }
        assessment_rows.append(assessment)
        if include_occurrence_projection:
            for occurrence in dense_occurrences:
                projection_rows.append(
                    {
                        "occurrence_index": occurrence["occurrence_index"],
                        "atlas_trace_index": occurrence["atlas_trace_index"],
                        "trace_unit_id": occurrence["trace_unit_id"],
                        "token_position": occurrence["token_position"],
                        "dense_signed_basis_index": occurrence["signed_basis_index"],
                        "target_basis_assessment_row_index": row_index,
                        "case_id": case_id,
                        "c2_signed_basis_index": basis_index,
                        **{field: basis[field] for field in _BASIS_FIELDS},
                        "c2_w64_assigned": cluster_id is not None,
                        "c2_w64_cluster_id": cluster_id,
                        "candidate_profile_available": candidate is not None,
                        "candidate_measurement_scope": CANDIDATE_MEASUREMENT_SCOPE,
                    }
                )
    projection_rows.sort(key=lambda row: int(row["occurrence_index"]))
    return _DerivedAssessment(
        crosswalk_rows=tuple(crosswalk_rows),
        assessment_rows=tuple(assessment_rows),
        projection_rows=tuple(projection_rows),
        c2_basis_count=len(c2_basis),
        width_profile_count=len(width_rows),
        candidate_profile_count=len(candidate_rows),
    )


def build_candidate_multiplex_assessment(
    *,
    c2_input_root: Path,
    c2_baseline_root: Path,
    dense_multiplex_root: Path,
    output_root: Path,
    include_occurrence_projection: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish the unified label-free assessment tables."""

    input_root = c2_input_root.resolve()
    baseline_root = c2_baseline_root.resolve()
    dense_root = dense_multiplex_root.resolve()
    output_root = output_root.resolve()
    resolved_repo_root = (
        Path(__file__).resolve().parents[3]
        if repo_root is None
        else repo_root.resolve()
    )
    if output_root.exists():
        raise FileExistsError(f"assessment output already exists: {output_root}")

    input_manifest = _validated_manifest(
        input_root,
        schema_version=CANDIDATE_CLUSTER_INPUT_SCHEMA,
        label="C2 input",
        parquet_schemas=_C2_INPUT_SCHEMAS,
    )
    baseline_manifest = _validated_manifest(
        baseline_root,
        schema_version=CANDIDATE_CLUSTER_BASELINE_SCHEMA,
        label="C2 baseline",
        parquet_schemas=_BASELINE_SCHEMAS,
    )
    dense_manifest = _validated_manifest(
        dense_root,
        schema_version=COMPACTED_MULTIPLEX_SCHEMA,
        label="dense multiplex",
        parquet_schemas=_DENSE_SCHEMAS,
    )
    _validate_source_firewalls(input_manifest, baseline_manifest)
    _validate_baseline_input_binding(
        input_root=input_root,
        input_manifest=input_manifest,
        baseline_manifest=baseline_manifest,
    )
    _deep_validate_source_artifacts(
        input_root=input_root,
        input_manifest=input_manifest,
        baseline_root=baseline_root,
        baseline_manifest=baseline_manifest,
        dense_root=dense_root,
        dense_manifest=dense_manifest,
    )
    producing_revision = _collect_producing_revision(resolved_repo_root)
    derived = _derive_assessment_tables(
        input_root=input_root,
        input_manifest=input_manifest,
        baseline_root=baseline_root,
        baseline_manifest=baseline_manifest,
        dense_root=dense_root,
        dense_manifest=dense_manifest,
        include_occurrence_projection=include_occurrence_projection,
    )
    crosswalk_rows = list(derived.crosswalk_rows)
    assessment_rows = list(derived.assessment_rows)
    projection_rows = list(derived.projection_rows)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _write_parquet(
            temporary / TARGET_CROSSWALK_FILE,
            crosswalk_rows,
            TARGET_CROSSWALK_SCHEMA,
        )
        _write_parquet(
            temporary / TARGET_BASIS_FILE,
            assessment_rows,
            TARGET_BASIS_ASSESSMENT_SCHEMA,
        )
        output_files = [TARGET_CROSSWALK_FILE, TARGET_BASIS_FILE]
        if include_occurrence_projection:
            _write_parquet(
                temporary / OCCURRENCE_PROJECTION_FILE,
                projection_rows,
                OCCURRENCE_PROJECTION_SCHEMA,
            )
            output_files.append(OCCURRENCE_PROJECTION_FILE)
        files = []
        for name in output_files:
            path = temporary / name
            files.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "row_count": pq.read_metadata(path).num_rows,
                }
            )
        matched_count = sum(bool(row["dense_target_match"]) for row in crosswalk_rows)
        metrics = _coverage_metrics(crosswalk_rows, assessment_rows, projection_rows)
        manifest: dict[str, Any] = {
            "schema_version": ASSESSMENT_SCHEMA_VERSION,
            "purpose": "label_free_candidate_and_width_multiplex_assessment_inputs",
            "sources": {
                "c2_input": _source_record(input_root, input_manifest),
                "c2_clustering_baseline": _source_record(
                    baseline_root, baseline_manifest
                ),
                "dense_multiplex": _source_record(dense_root, dense_manifest),
            },
            "producing_revision": producing_revision,
            "cluster_state": {
                "identifier": "c2_w64",
                "view": "W",
                "n_clusters": 64,
                "assignment": "medoid_seed",
            },
            "candidate_measurement_scope": CANDIDATE_MEASUREMENT_SCOPE,
            "occurrence_projection_contains_candidate_values": False,
            "overlap": {
                "policy": "partial_overlap_allowed_explicit_no_imputation",
                "c2_target_count": len(crosswalk_rows),
                "dense_matched_target_count": matched_count,
                "dense_unmatched_target_count": len(crosswalk_rows) - matched_count,
                "target_crosswalk_retains_all_c2_targets": True,
                "redundant_exact_join_fields": [
                    "source_artifact_id",
                    "trace_unit_id",
                    "response_id+base_question_id+response_position",
                    "artifact_payload_sha256",
                ],
            },
            "counts": {
                "c2_signed_basis_count": derived.c2_basis_count,
                "target_basis_union_row_count": len(assessment_rows),
                "width_profile_row_count": derived.width_profile_count,
                "candidate_profile_row_count": derived.candidate_profile_count,
                "occurrence_projection_row_count": len(projection_rows),
            },
            "coverage_metrics": metrics,
            "occurrence_projection_included": include_occurrence_projection,
            "outcomes_inspected": False,
            "labels_used": False,
            "descriptions_generated": False,
            "model_calls_made": False,
            "confirmatory_holdout_opened": False,
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Close the read-to-publication window: every immutable source must
        # still be byte-identical and the baseline must still bind this input.
        reloaded_input = _validated_manifest(
            input_root,
            schema_version=CANDIDATE_CLUSTER_INPUT_SCHEMA,
            label="C2 input",
            parquet_schemas=_C2_INPUT_SCHEMAS,
        )
        reloaded_baseline = _validated_manifest(
            baseline_root,
            schema_version=CANDIDATE_CLUSTER_BASELINE_SCHEMA,
            label="C2 baseline",
            parquet_schemas=_BASELINE_SCHEMAS,
        )
        reloaded_dense = _validated_manifest(
            dense_root,
            schema_version=COMPACTED_MULTIPLEX_SCHEMA,
            label="dense multiplex",
            parquet_schemas=_DENSE_SCHEMAS,
        )
        if (
            reloaded_input != input_manifest
            or reloaded_baseline != baseline_manifest
            or reloaded_dense != dense_manifest
        ):
            raise ValueError("assessment source changed during construction")
        _validate_baseline_input_binding(
            input_root=input_root,
            input_manifest=reloaded_input,
            baseline_manifest=reloaded_baseline,
        )
        _deep_validate_source_artifacts(
            input_root=input_root,
            input_manifest=reloaded_input,
            baseline_root=baseline_root,
            baseline_manifest=reloaded_baseline,
            dense_root=dense_root,
            dense_manifest=reloaded_dense,
        )
        if _collect_producing_revision(resolved_repo_root) != producing_revision:
            raise ValueError(
                "assessment producing revision changed during construction"
            )
        _publish_directory_no_replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_candidate_multiplex_assessment(
    root: Path, *, verify_sources: bool = True
) -> LoadedCandidateMultiplexAssessment:
    """Load a derived artifact, validating content, schemas, and source bindings."""

    root = root.resolve()
    manifest = _validated_manifest(
        root,
        schema_version=ASSESSMENT_SCHEMA_VERSION,
        label="candidate multiplex assessment",
        parquet_schemas={
            name: schema
            for name, schema in _OUTPUT_SCHEMAS.items()
            if name != OCCURRENCE_PROJECTION_FILE
            or bool(
                load_json_object(root / "manifest.json").get(
                    "occurrence_projection_included"
                )
            )
        },
    )
    for field in (
        "outcomes_inspected",
        "labels_used",
        "descriptions_generated",
        "model_calls_made",
        "confirmatory_holdout_opened",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"assessment violates {field} firewall")
    if manifest.get("candidate_measurement_scope") != CANDIDATE_MEASUREMENT_SCOPE:
        raise ValueError("assessment candidate measurement scope drift")
    if manifest.get("occurrence_projection_contains_candidate_values") is not False:
        raise ValueError("assessment occurrence projection scope drift")
    if manifest.get("cluster_state") != {
        "identifier": "c2_w64",
        "view": "W",
        "n_clusters": 64,
        "assignment": "medoid_seed",
    }:
        raise ValueError("assessment C2 W64 state binding drift")
    revision = manifest.get("producing_revision")
    if not isinstance(revision, Mapping):
        raise TypeError("assessment producing revision is invalid")
    _validate_producing_revision(revision)

    crosswalk = _exact_table(
        root / TARGET_CROSSWALK_FILE,
        TARGET_CROSSWALK_SCHEMA,
        label="candidate multiplex assessment",
    )
    assessment = _exact_table(
        root / TARGET_BASIS_FILE,
        TARGET_BASIS_ASSESSMENT_SCHEMA,
        label="candidate multiplex assessment",
    )
    projection = None
    if bool(manifest.get("occurrence_projection_included")):
        projection = _exact_table(
            root / OCCURRENCE_PROJECTION_FILE,
            OCCURRENCE_PROJECTION_SCHEMA,
            label="candidate multiplex assessment",
        )
        if set(projection.column("candidate_measurement_scope").to_pylist()) - {
            CANDIDATE_MEASUREMENT_SCOPE
        }:
            raise ValueError("occurrence projection candidate scope drift")

    overlap = manifest.get("overlap")
    counts = manifest.get("counts")
    if not isinstance(overlap, Mapping) or not isinstance(counts, Mapping):
        raise TypeError("assessment count summaries are invalid")
    crosswalk_rows = crosswalk.to_pylist()
    if (
        len(crosswalk_rows) != int(overlap.get("c2_target_count", -1))
        or len({row["case_id"] for row in crosswalk_rows}) != len(crosswalk_rows)
        or sum(bool(row["dense_target_match"]) for row in crosswalk_rows)
        != int(overlap.get("dense_matched_target_count", -1))
        or sum(not bool(row["dense_target_match"]) for row in crosswalk_rows)
        != int(overlap.get("dense_unmatched_target_count", -1))
    ):
        raise ValueError("assessment target crosswalk summary drift")
    assessment_rows = assessment.to_pylist()
    if (
        len(assessment_rows) != int(counts.get("target_basis_union_row_count", -1))
        or [int(row["assessment_row_index"]) for row in assessment_rows]
        != list(range(len(assessment_rows)))
        or any(
            row["candidate_measurement_scope"] != CANDIDATE_MEASUREMENT_SCOPE
            for row in assessment_rows
        )
    ):
        raise ValueError("assessment target/basis table drift")
    projection_count = 0 if projection is None else projection.num_rows
    if projection_count != int(counts.get("occurrence_projection_row_count", -1)):
        raise ValueError("assessment occurrence projection count drift")
    expected_metrics = _coverage_metrics(
        crosswalk_rows,
        assessment_rows,
        [] if projection is None else projection.to_pylist(),
    )
    if manifest.get("coverage_metrics") != expected_metrics:
        raise ValueError("assessment coverage metrics drift")

    if verify_sources:
        sources = manifest.get("sources")
        if not isinstance(sources, Mapping):
            raise TypeError("assessment source bindings are invalid")
        input_root, input_manifest = _validate_source_record(
            sources["c2_input"],
            schema_version=CANDIDATE_CLUSTER_INPUT_SCHEMA,
            label="C2 input",
            parquet_schemas=_C2_INPUT_SCHEMAS,
        )
        baseline_root, baseline_manifest = _validate_source_record(
            sources["c2_clustering_baseline"],
            schema_version=CANDIDATE_CLUSTER_BASELINE_SCHEMA,
            label="C2 baseline",
            parquet_schemas=_BASELINE_SCHEMAS,
        )
        dense_root, dense_manifest = _validate_source_record(
            sources["dense_multiplex"],
            schema_version=COMPACTED_MULTIPLEX_SCHEMA,
            label="dense multiplex",
            parquet_schemas=_DENSE_SCHEMAS,
        )
        _validate_source_firewalls(input_manifest, baseline_manifest)
        _validate_baseline_input_binding(
            input_root=input_root,
            input_manifest=input_manifest,
            baseline_manifest=baseline_manifest,
        )
        _deep_validate_source_artifacts(
            input_root=input_root,
            input_manifest=input_manifest,
            baseline_root=baseline_root,
            baseline_manifest=baseline_manifest,
            dense_root=dense_root,
            dense_manifest=dense_manifest,
        )
        expected = _derive_assessment_tables(
            input_root=input_root,
            input_manifest=input_manifest,
            baseline_root=baseline_root,
            baseline_manifest=baseline_manifest,
            dense_root=dense_root,
            dense_manifest=dense_manifest,
            include_occurrence_projection=bool(
                manifest.get("occurrence_projection_included")
            ),
        )
        expected_crosswalk = pa.Table.from_pylist(
            list(expected.crosswalk_rows), schema=TARGET_CROSSWALK_SCHEMA
        )
        expected_assessment = pa.Table.from_pylist(
            list(expected.assessment_rows), schema=TARGET_BASIS_ASSESSMENT_SCHEMA
        )
        expected_projection = (
            pa.Table.from_pylist(
                list(expected.projection_rows), schema=OCCURRENCE_PROJECTION_SCHEMA
            )
            if projection is not None
            else None
        )
        if not crosswalk.equals(expected_crosswalk, check_metadata=False):
            raise ValueError(
                "assessment crosswalk differs from bound source derivation"
            )
        if not assessment.equals(expected_assessment, check_metadata=False):
            raise ValueError(
                "assessment target/basis rows differ from bound source derivation"
            )
        if projection is not None and not projection.equals(
            expected_projection, check_metadata=False
        ):
            raise ValueError(
                "assessment occurrence projection differs from bound source derivation"
            )

    return LoadedCandidateMultiplexAssessment(
        root=root,
        manifest=manifest,
        target_crosswalk=crosswalk,
        target_basis_assessment=assessment,
        occurrence_projection=projection,
    )
