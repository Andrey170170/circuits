"""Metrics-free executable-only source for candidate-identity evaluation.

Only the authorization's structural target columns are projected from the mixed
C2 target table.  Audit rows are filtered before conversion to Python, and only
generation/selection candidate-union payloads are ever resolved or opened.
Raw artifact manifests, metrics, compact candidate profiles, and multiplex
target-basis values are outside this executable path.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from circuits.analysis.bonafide.candidate_clustering_execution import (
    _publish_directory_no_replace,
    load_candidate_clustering_baseline,
)
from circuits.analysis.bonafide.candidate_labelability_evaluation import (
    extract_chosen_medoid_assignments,
)
from circuits.analysis.bonafide.candidate_profiles import (
    BASIS_INDEX_SCHEMA,
    CANDIDATE_CLUSTER_INPUT_SCHEMA,
    extract_candidate_profiles,
)
from circuits.analysis.bonafide.candidate_profiles import (
    TARGET_SCHEMA as C2_TARGET_SCHEMA,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.analysis.bonafide.identity import Polarity, SignedBasisKey
from circuits.tracing.candidate_union import (
    CANDIDATE_UNION_TRACE_FAMILY_ID,
    DATA_FILENAME,
    CandidateUnionArtifact,
    CandidateUnionTrace,
    validate_candidate_union_trace,
)

SOURCE_SCHEMA_VERSION = "adag.bonafide.candidate-identity-source.v1"
AUTHORIZATION_SCHEMA_VERSION = (
    "adag.bonafide.candidate-identity-source-authorization.v1"
)
AUTHORIZATION_SHA256 = (
    "ff5aafeec23c31c419b95338c18e3af883eb4ce90e7c410a0218ad0700c8627f"
)
AUTHORIZATION_RELATIVE = "plans/candidate-identity-executable-source-v1.json"
PROTOCOL_RELATIVE = "docs/CANDIDATE_IDENTITY_ALIGNMENT_PROTOCOL.md"

PARTITIONS = ("generation", "selection_scoring")
EXPECTED_TARGET_COUNTS = {"generation": 133, "selection_scoring": 56}
EXPECTED_AUDIT_TARGET_COUNT = 56
PROJECTED_TARGET_COLUMNS = (
    "case_id",
    "source_width1_artifact_id",
    "candidate_union_artifact_id",
    "candidate_union_payload_sha256",
    "candidate_union_topology_sha256",
    "base_question_id",
    "response_id",
    "phase_bin",
    "response_position",
    "family_partition",
    "partition_hierarchical_weight",
)
FORBIDDEN_TARGET_COLUMNS = (
    "candidate_selection_json",
    "observed_token_id",
    "observed_token_text",
    "example_json",
)

TARGET_FILES = {
    "generation": "generation-targets.parquet",
    "selection_scoring": "selection-targets.parquet",
}
PROFILE_FILES = {
    "generation": "generation-profiles.parquet",
    "selection_scoring": "selection-profiles.parquet",
}

TARGET_SCHEMA = pa.schema(
    [
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("source_width1_artifact_id", pa.string(), nullable=False),
        pa.field("candidate_union_artifact_id", pa.string(), nullable=False),
        pa.field("candidate_union_payload_sha256", pa.string(), nullable=False),
        pa.field("candidate_union_topology_sha256", pa.string(), nullable=False),
        pa.field("base_question_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("phase_bin", pa.int8(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("family_partition", pa.string(), nullable=False),
        pa.field("partition_hierarchical_weight", pa.float64(), nullable=False),
        pa.field("observed_token_id", pa.int64(), nullable=False),
        pa.field("observed_token_text", pa.string(), nullable=False),
        pa.field("candidate_selection_json", pa.string(), nullable=False),
    ]
)

PROFILE_SCHEMA = pa.schema(
    [
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
        pa.field("c2_w64_assigned", pa.bool_(), nullable=False),
        pa.field("c2_w64_cluster_id", pa.int32(), nullable=True),
        pa.field("candidate_contrast_vector", pa.list_(pa.float64()), nullable=False),
        pa.field("candidate_occurrence_count", pa.int32(), nullable=False),
        pa.field("family_partition", pa.string(), nullable=False),
    ]
)

_SOURCE_PATHS = (
    "circuits/analysis/bonafide/candidate_identity_source.py",
    "scripts/bonafide/candidate_identity_source.py",
    PROTOCOL_RELATIVE,
    AUTHORIZATION_RELATIVE,
    "circuits/analysis/bonafide/candidate_profiles.py",
    "circuits/tracing/artifact.py",
    "circuits/tracing/candidates.py",
    "circuits/tracing/candidate_union.py",
)

EXPOSURE_CONTRACT = {
    "assessment_target_basis_table_opened": False,
    "audit_candidate_artifacts_opened": False,
    "audit_candidate_metadata_loaded": False,
    "audit_candidate_values_loaded": False,
    "audit_metrics_computed": False,
    "audit_structural_target_index_fields_loaded": True,
    "candidate_profile_table_opened": False,
    "confirmatory_holdout_opened": False,
    "labels_or_prior_model_outputs_loaded": False,
    "outcomes_loaded": False,
    "raw_artifact_manifest_or_metrics_files_opened": False,
}

CLUSTER_STATE = {
    "identifier": "c2_w64",
    "view": "W",
    "n_clusters": 64,
    "assignment": "medoid_seed",
}

IDENTITY_CHECKS = {
    "artifact_id": "exact_selected_directory",
    "payload_sha256": "verified_before_deserialization",
    "source_width1_artifact_id": "trace_equals_projected_target",
    "topology_sha256": "trace_equals_projected_target",
    "response_position": "trace_equals_projected_target",
    "candidate_union_plan": "transitively_bound_by_c2_input_manifest",
    "signed_basis": "exact_model_revision_layer_neuron_polarity_join_to_c2_w64",
}


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def _collect_revision(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    if Path(str(_git(root, "rev-parse", "--show-toplevel"))).resolve() != root:
        raise ValueError("candidate identity source must run from repository root")
    status = str(_git(root, "status", "--porcelain=v1", "--untracked-files=no"))
    if status:
        raise ValueError("candidate identity source requires a clean tracked worktree")
    files = []
    for relative in _SOURCE_PATHS:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"candidate identity source is not tracked: {relative}")
        blob = str(_git(root, "rev-parse", f"HEAD:{relative}"))
        if _git(root, "hash-object", relative) != blob:
            raise ValueError(f"candidate identity source differs from HEAD: {relative}")
        files.append(
            {"path": relative, "git_blob": blob, "sha256": file_sha256(root / relative)}
        )
    return {
        "repo_root": str(root),
        "git_commit": str(_git(root, "rev-parse", "HEAD")),
        "git_tree": str(_git(root, "rev-parse", "HEAD^{tree}")),
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(b"").hexdigest(),
        "files": files,
    }


def _validate_revision(revision: Mapping[str, Any]) -> None:
    root = Path(str(revision.get("repo_root"))).resolve()
    if not (root / ".git").exists():
        root = Path(__file__).resolve().parents[3]
    commit = str(revision.get("git_commit"))
    if (
        revision.get("tracked_worktree_clean") is not True
        or revision.get("tracked_status_sha256") != hashlib.sha256(b"").hexdigest()
        or _git(root, "rev-parse", f"{commit}^{{tree}}") != revision.get("git_tree")
    ):
        raise ValueError("candidate identity source producing revision drift")
    raw = revision.get("files")
    if not isinstance(raw, list):
        raise TypeError("candidate identity source revision inventory is invalid")
    records = {str(item.get("path")): item for item in raw if isinstance(item, Mapping)}
    if set(records) != set(_SOURCE_PATHS):
        raise ValueError("candidate identity source revision inventory drift")
    for relative in _SOURCE_PATHS:
        content = _git(root, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(content, bytes)
        if (
            records[relative].get("git_blob")
            != _git(root, "rev-parse", f"{commit}:{relative}")
            or records[relative].get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"candidate identity source object drift: {relative}")


def _load_authorization(repo_root: Path) -> dict[str, Any]:
    path = repo_root / AUTHORIZATION_RELATIVE
    authorization = load_json_object(path)
    core = dict(authorization)
    claimed = core.pop("authorization_sha256", None)
    if claimed != AUTHORIZATION_SHA256 or canonical_sha256(core) != claimed:
        raise ValueError("candidate identity source authorization self-hash mismatch")
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported candidate identity source authorization")
    if authorization.get("projected_target_columns") != list(PROJECTED_TARGET_COLUMNS):
        raise ValueError("candidate identity structural projection authorization drift")
    if authorization.get("forbidden_target_columns") != list(FORBIDDEN_TARGET_COLUMNS):
        raise ValueError("candidate identity forbidden target columns drift")
    if authorization.get("allowed_partitions") != EXPECTED_TARGET_COUNTS:
        raise ValueError("candidate identity executable partition authorization drift")
    if authorization.get("exposure_contract") != EXPOSURE_CONTRACT:
        raise ValueError("candidate identity exposure authorization drift")
    by_path = {
        str(record.get("path")): record
        for record in authorization.get("original_rederivation_code", [])
        if isinstance(record, Mapping)
    }
    expected_paths = set(_SOURCE_PATHS[-4:])
    if set(by_path) != expected_paths:
        raise ValueError("candidate identity rederivation source inventory drift")
    for relative in expected_paths:
        if by_path[relative].get("sha256") != file_sha256(repo_root / relative):
            raise ValueError(
                f"candidate identity frozen rederivation code drift: {relative}"
            )
    return authorization


def _load_self_hashed_manifest(root: Path, *, expected_schema: str) -> dict[str, Any]:
    manifest = load_json_object(root / "manifest.json")
    core = dict(manifest)
    if core.pop("manifest_sha256", None) != canonical_sha256(core):
        raise ValueError("source manifest self-hash mismatch")
    if manifest.get("schema_version") != expected_schema:
        raise ValueError("source manifest schema drift")
    return manifest


def _file_record(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TypeError("source file inventory is invalid")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("path") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"source file inventory does not bind {name} exactly once")
    return matches[0]


def _validate_authorized_sources(
    authorization: Mapping[str, Any], repo_root: Path
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path]:
    sources = authorization.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("candidate identity authorized sources are invalid")
    input_record = sources.get("c2_input")
    baseline_record = sources.get("c2_w64_baseline")
    assessment_record = sources.get("multiplex_assessment_provenance_only")
    plan_record = sources.get("candidate_union_plan")
    if not all(
        isinstance(record, Mapping)
        for record in (input_record, baseline_record, assessment_record, plan_record)
    ):
        raise TypeError("candidate identity authorized source record is invalid")
    assert isinstance(input_record, Mapping)
    assert isinstance(baseline_record, Mapping)
    assert isinstance(assessment_record, Mapping)
    assert isinstance(plan_record, Mapping)

    input_root = Path(str(input_record["path"])).resolve()
    if file_sha256(input_root / "manifest.json") != input_record.get(
        "manifest_file_sha256"
    ):
        raise ValueError("candidate identity C2 input manifest file drift")
    input_manifest = _load_self_hashed_manifest(
        input_root, expected_schema=CANDIDATE_CLUSTER_INPUT_SCHEMA
    )
    if input_manifest.get("manifest_sha256") != input_record.get("manifest_sha256"):
        raise ValueError("candidate identity C2 input binding drift")
    for field in (
        "outcomes_inspected",
        "model_calls_made",
        "cluster_fit_performed",
        "confirmatory_holdout_opened",
    ):
        if input_manifest.get(field) is not False:
            raise ValueError(f"candidate identity C2 input violates {field}")

    baseline_root = Path(str(baseline_record["path"])).resolve()
    baseline_manifest_path = baseline_root / "manifest.json"
    if file_sha256(baseline_manifest_path) != baseline_record.get(
        "manifest_file_sha256"
    ):
        raise ValueError("candidate identity baseline manifest file drift")
    baseline_manifest = load_json_object(baseline_manifest_path)
    baseline_core = dict(baseline_manifest)
    if baseline_core.pop("manifest_sha256", None) != canonical_sha256(
        baseline_core
    ) or baseline_manifest.get("manifest_sha256") != baseline_record.get(
        "manifest_sha256"
    ):
        raise ValueError("candidate identity baseline manifest binding drift")
    source_input = baseline_manifest.get("source_input_bundle")
    if not isinstance(source_input, Mapping) or source_input.get(
        "manifest_sha256"
    ) != input_manifest.get("manifest_sha256"):
        raise ValueError("candidate identity baseline/input binding drift")

    assessment_root = Path(str(assessment_record["path"])).resolve()
    assessment_manifest_path = assessment_root / "manifest.json"
    if file_sha256(assessment_manifest_path) != assessment_record.get(
        "manifest_file_sha256"
    ):
        raise ValueError("candidate identity assessment manifest file drift")
    assessment_manifest = load_json_object(assessment_manifest_path)
    assessment_core = dict(assessment_manifest)
    if assessment_core.pop("manifest_sha256", None) != canonical_sha256(
        assessment_core
    ) or assessment_manifest.get("manifest_sha256") != assessment_record.get(
        "manifest_sha256"
    ):
        raise ValueError("candidate identity assessment manifest binding drift")

    input_bindings = input_manifest.get("inputs")
    if not isinstance(input_bindings, Mapping):
        raise TypeError("candidate identity C2 raw bindings are invalid")
    frozen_plan = input_bindings.get("candidate_union_plan")
    if (
        not isinstance(frozen_plan, Mapping)
        or frozen_plan.get("canonical_sha256") != plan_record.get("canonical_sha256")
        or frozen_plan.get("file_sha256") != plan_record.get("file_sha256")
    ):
        raise ValueError("candidate identity candidate-union plan binding drift")
    local_plan = (
        repo_root
        / "scripts/bonafide/manifests/qwen3_4b_instruct_candidate_union_c2_plan_v1.json"
    )
    if file_sha256(local_plan) != plan_record.get("file_sha256"):
        raise ValueError("candidate identity opaque plan file drift")
    candidate_union_root = Path(str(sources.get("candidate_union_root"))).resolve()
    if (
        candidate_union_root
        != Path(str(input_bindings.get("candidate_union_root"))).resolve()
    ):
        raise ValueError("candidate identity candidate-union root binding drift")
    return (
        input_root,
        input_manifest,
        baseline_root,
        baseline_manifest,
        assessment_root,
        assessment_manifest,
        candidate_union_root,
    )


def _project_executable_targets(
    input_root: Path, input_manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    path = input_root / "targets.parquet"
    record = _file_record(input_manifest, path.name)
    if file_sha256(path) != record.get("sha256"):
        raise ValueError("candidate identity structural target index hash drift")
    table = pq.read_table(path, columns=list(PROJECTED_TARGET_COLUMNS))
    expected_schema = pa.schema(
        [C2_TARGET_SCHEMA.field(name) for name in PROJECTED_TARGET_COLUMNS]
    )
    if not table.schema.equals(expected_schema, check_metadata=False):
        raise ValueError("candidate identity structural target projection schema drift")
    compute = cast(Any, pc)
    counts = {
        str(item["values"]): int(item["counts"])
        for item in compute.value_counts(table["family_partition"]).to_pylist()
    }
    expected_all = {**EXPECTED_TARGET_COUNTS, "audit": EXPECTED_AUDIT_TARGET_COUNT}
    if counts != expected_all:
        raise ValueError("candidate identity structural partition counts drift")
    mask = compute.is_in(table["family_partition"], value_set=pa.array(PARTITIONS))
    executable = table.filter(mask)
    rows = executable.to_pylist()
    if len(rows) != sum(EXPECTED_TARGET_COUNTS.values()) or any(
        row["family_partition"] not in PARTITIONS for row in rows
    ):
        raise AssertionError("candidate identity audit filter failed")
    case_ids = [str(row["case_id"]) for row in rows]
    artifact_ids = [str(row["candidate_union_artifact_id"]) for row in rows]
    if len(set(case_ids)) != len(rows) or len(set(artifact_ids)) != len(rows):
        raise ValueError("candidate identity executable targets are not unique")
    return rows, counts


def _load_basis_index(
    input_root: Path, input_manifest: Mapping[str, Any]
) -> tuple[dict[SignedBasisKey, int], str, str]:
    path = input_root / "basis-index.parquet"
    record = _file_record(input_manifest, path.name)
    if file_sha256(path) != record.get("sha256"):
        raise ValueError("candidate identity basis index hash drift")
    columns = (
        "signed_basis_index",
        "model_id",
        "model_revision",
        "layer",
        "neuron_index",
        "polarity",
    )
    table = pq.read_table(path, columns=list(columns))
    expected_schema = pa.schema([BASIS_INDEX_SCHEMA.field(name) for name in columns])
    if not table.schema.equals(expected_schema, check_metadata=False):
        raise ValueError("candidate identity basis projection schema drift")
    rows = table.to_pylist()
    result: dict[SignedBasisKey, int] = {}
    models: set[tuple[str, str]] = set()
    for row in rows:
        key = SignedBasisKey(
            model_id=str(row["model_id"]),
            model_revision=str(row["model_revision"]),
            layer=int(row["layer"]),
            neuron_index=int(row["neuron_index"]),
            polarity=cast(Polarity, str(row["polarity"])),
        )
        if key in result:
            raise ValueError("candidate identity basis index is duplicated")
        result[key] = int(row["signed_basis_index"])
        models.add((key.model_id, key.model_revision))
    if sorted(result.values()) != list(range(len(result))) or len(models) != 1:
        raise ValueError("candidate identity basis index is noncanonical")
    model_id, model_revision = next(iter(models))
    return result, model_id, model_revision


def _resolve_payload(root: Path, artifact_id: str) -> Path:
    family_root = root / CANDIDATE_UNION_TRACE_FAMILY_ID
    matches = list(family_root.glob(f"*/{artifact_id}/{DATA_FILENAME}"))
    if len(matches) != 1:
        raise ValueError(
            f"candidate identity expected one selected payload for {artifact_id}, found {len(matches)}"
        )
    path = matches[0].resolve()
    if family_root.resolve() not in path.parents:
        raise ValueError("candidate identity selected payload escaped its raw root")
    return path


def load_hash_bound_candidate_union_trace(
    payload_path: Path, *, expected_sha256: str
) -> CandidateUnionTrace:
    """Deserialize one trusted payload after hashing, without adjacent metadata."""

    if file_sha256(payload_path) != expected_sha256:
        raise ValueError("candidate identity selected payload hash drift")
    with gzip.open(payload_path, "rb") as handle:
        trace = pickle.load(handle)
    validate_candidate_union_trace(trace)
    return trace


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


def _validate_selected_trace_binding(
    trace: CandidateUnionTrace,
    target: Mapping[str, Any],
    *,
    artifact_id: str,
) -> None:
    if (
        trace.source_width1_artifact_id != target["source_width1_artifact_id"]
        or trace.shared_response_position != int(target["response_position"])
        or trace.topology_sha256 != target["candidate_union_topology_sha256"]
    ):
        raise ValueError(
            f"candidate identity selected trace binding drift: {artifact_id}"
        )


def _derive_rows(
    *,
    targets: Sequence[Mapping[str, Any]],
    candidate_union_root: Path,
    basis_by_key: Mapping[SignedBasisKey, int],
    assignments: NDArray[np.int64],
    model_id: str,
    model_revision: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    target_rows = {partition: [] for partition in PARTITIONS}
    profile_rows = {partition: [] for partition in PARTITIONS}
    payload_records = []
    for target in sorted(targets, key=lambda row: str(row["case_id"])):
        partition = str(target["family_partition"])
        artifact_id = str(target["candidate_union_artifact_id"])
        payload_path = _resolve_payload(candidate_union_root, artifact_id)
        payload_sha256 = str(target["candidate_union_payload_sha256"])
        trace = load_hash_bound_candidate_union_trace(
            payload_path, expected_sha256=payload_sha256
        )
        _validate_selected_trace_binding(trace, target, artifact_id=artifact_id)
        selection = trace.candidate_selection
        profiles, _, diagnostics = extract_candidate_profiles(
            # Empty metadata makes the no-manifest/no-metrics boundary explicit. The
            # frozen extractor consumes only ``trace`` from this carrier.
            CandidateUnionArtifact(
                path=payload_path.parent,
                trace=trace,
                manifest={},
                metrics={},
            ),
            model_id=model_id,
            model_revision=model_revision,
        )
        if diagnostics["activation_invariance"]["violation_count"] != 0:
            raise ValueError("candidate identity selected trace activation drift")
        target_rows[partition].append(
            {
                **{field: target[field] for field in PROJECTED_TARGET_COLUMNS},
                "observed_token_id": selection.observed_token_id,
                "observed_token_text": selection.observed_token_text,
                "candidate_selection_json": json.dumps(
                    selection.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            }
        )
        for key, profile in sorted(profiles.items()):
            basis_index = basis_by_key.get(key)
            if basis_index is None:
                raise ValueError(
                    "candidate identity reconstructed basis is absent from C2 index"
                )
            cluster = int(assignments[basis_index])
            profile_rows[partition].append(
                {
                    "case_id": target["case_id"],
                    "signed_basis_index": basis_index,
                    "model_id": key.model_id,
                    "model_revision": key.model_revision,
                    "layer": key.layer,
                    "neuron_index": key.neuron_index,
                    "polarity": key.polarity,
                    "c2_w64_assigned": cluster >= 0,
                    "c2_w64_cluster_id": cluster if cluster >= 0 else None,
                    "candidate_contrast_vector": list(profile.values),
                    "candidate_occurrence_count": profile.occurrence_count,
                    "family_partition": partition,
                }
            )
        payload_records.append(
            {
                "case_id": str(target["case_id"]),
                "artifact_id": artifact_id,
                "partition": partition,
                "payload_sha256": payload_sha256,
                "source_width1_artifact_id": str(target["source_width1_artifact_id"]),
                "topology_sha256": str(target["candidate_union_topology_sha256"]),
                "response_position": int(target["response_position"]),
                "relative_payload_path": str(
                    payload_path.relative_to(candidate_union_root)
                ),
            }
        )
    for partition, expected in EXPECTED_TARGET_COUNTS.items():
        if len(target_rows[partition]) != expected or not profile_rows[partition]:
            raise ValueError(
                f"candidate identity reconstructed {partition} coverage drift"
            )
        profile_rows[partition].sort(
            key=lambda row: (str(row["case_id"]), int(row["signed_basis_index"]))
        )
    return target_rows, profile_rows, payload_records


def _extract_w64_assignments(baseline: Any, *, basis_count: int) -> NDArray[np.int64]:
    if baseline.manifest.get("chosen_cluster_count") != 64:
        raise ValueError("candidate identity authorized baseline is not W64")
    assignments = extract_chosen_medoid_assignments(baseline, basis_count=basis_count)[
        "W"
    ]
    if assignments.shape != (basis_count,) or np.any(
        (assignments < -1) | (assignments >= 64)
    ):
        raise ValueError("candidate identity W64 assignment domain drift")
    return assignments


def build_candidate_identity_source(
    *, output_root: Path, repo_root: Path
) -> dict[str, Any]:
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to replace candidate identity source: {output}"
        )
    revision = _collect_revision(repo_root)
    authorization = _load_authorization(repo_root)
    (
        input_root,
        input_manifest,
        baseline_root,
        baseline_manifest,
        assessment_root,
        assessment_manifest,
        candidate_union_root,
    ) = _validate_authorized_sources(authorization, repo_root)
    targets, structural_counts = _project_executable_targets(input_root, input_manifest)
    basis_by_key, model_id, model_revision = _load_basis_index(
        input_root, input_manifest
    )
    baseline = load_candidate_clustering_baseline(baseline_root, verify_source=False)
    if baseline.manifest != baseline_manifest:
        raise ValueError("candidate identity baseline deep load drift")
    assignments = _extract_w64_assignments(baseline, basis_count=len(basis_by_key))
    target_rows, profile_rows, payload_records = _derive_rows(
        targets=targets,
        candidate_union_root=candidate_union_root,
        basis_by_key=basis_by_key,
        assignments=assignments,
        model_id=model_id,
        model_revision=model_revision,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        files = []
        for partition in PARTITIONS:
            for name, rows, schema in (
                (TARGET_FILES[partition], target_rows[partition], TARGET_SCHEMA),
                (PROFILE_FILES[partition], profile_rows[partition], PROFILE_SCHEMA),
            ):
                path = temporary / name
                _write_parquet(path, rows, schema)
                files.append(
                    {
                        "path": name,
                        "sha256": file_sha256(path),
                        "size_bytes": path.stat().st_size,
                        "row_count": len(rows),
                    }
                )
        payload_set_sha256 = canonical_sha256(payload_records)
        core: dict[str, Any] = {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "purpose": "metrics_free_executable_only_candidate_identity_source",
            "authorization": {
                "path": AUTHORIZATION_RELATIVE,
                "authorization_sha256": AUTHORIZATION_SHA256,
                "file_sha256": file_sha256(repo_root / AUTHORIZATION_RELATIVE),
            },
            "protocol": {
                "path": PROTOCOL_RELATIVE,
                "sha256": file_sha256(repo_root / PROTOCOL_RELATIVE),
            },
            "producing_revision": revision,
            "sources": {
                "c2_input_manifest_sha256": input_manifest["manifest_sha256"],
                "c2_input_manifest_file_sha256": file_sha256(
                    input_root / "manifest.json"
                ),
                "c2_w64_baseline_manifest_sha256": baseline_manifest["manifest_sha256"],
                "c2_w64_baseline_manifest_file_sha256": file_sha256(
                    baseline_root / "manifest.json"
                ),
                "multiplex_assessment_manifest_sha256": assessment_manifest[
                    "manifest_sha256"
                ],
                "multiplex_assessment_manifest_file_sha256": file_sha256(
                    assessment_root / "manifest.json"
                ),
                "candidate_union_root": str(candidate_union_root),
                "candidate_union_plan_canonical_sha256": authorization["sources"][
                    "candidate_union_plan"
                ]["canonical_sha256"],
            },
            "model": {"model_id": model_id, "model_revision": model_revision},
            "cluster_state": CLUSTER_STATE,
            "structural_target_projection": {
                "columns": list(PROJECTED_TARGET_COLUMNS),
                "all_partition_counts": structural_counts,
                "audit_rows_converted_to_python": False,
            },
            "identity_checks": IDENTITY_CHECKS,
            "audit_sentinels": {
                "structural_rows_projected": EXPECTED_AUDIT_TARGET_COUNT,
                "rows_converted_to_python": 0,
                "artifact_ids_resolved": 0,
                "payloads_opened": 0,
                "metadata_loaded": False,
                "values_loaded": False,
            },
            "selected_payloads": {
                "count": len(payload_records),
                "record_set_sha256": payload_set_sha256,
                "records": payload_records,
            },
            "counts": {
                "targets_by_partition": {
                    partition: len(target_rows[partition]) for partition in PARTITIONS
                },
                "profiles_by_partition": {
                    partition: len(profile_rows[partition]) for partition in PARTITIONS
                },
            },
            "exposure_contract": EXPOSURE_CONTRACT,
            "metrics_computed": False,
            "files": files,
        }
        manifest = {**core, "manifest_sha256": canonical_sha256(core)}
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if _collect_revision(repo_root) != revision:
            raise ValueError(
                "candidate identity source revision changed during construction"
            )
        # Close the selected-payload read window without resolving any audit artifact.
        for record in payload_records:
            path = candidate_union_root / str(record["relative_payload_path"])
            if file_sha256(path) != record["payload_sha256"]:
                raise ValueError(
                    "candidate identity selected payload changed during construction"
                )
        _publish_directory_no_replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _exact_table(path: Path, schema: pa.Schema, record: Mapping[str, Any]) -> pa.Table:
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("size_bytes", -1))
        or file_sha256(path) != record.get("sha256")
    ):
        raise ValueError(f"candidate identity source file drift: {path.name}")
    table = pq.read_table(path)
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"candidate identity source parquet schema drift: {path.name}")
    if table.num_rows != int(record.get("row_count", -1)):
        raise ValueError(f"candidate identity source row count drift: {path.name}")
    return table


def load_candidate_identity_source(
    root: Path,
) -> tuple[dict[str, Any], dict[str, pa.Table], dict[str, pa.Table]]:
    source_root = root.resolve()
    manifest = load_json_object(source_root / "manifest.json")
    core = dict(manifest)
    if core.pop("manifest_sha256", None) != canonical_sha256(core):
        raise ValueError("candidate identity source manifest self-hash mismatch")
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported candidate identity source schema")
    if (
        manifest.get("exposure_contract") != EXPOSURE_CONTRACT
        or manifest.get("metrics_computed") is not False
    ):
        raise ValueError("candidate identity source exposure contract drift")
    if manifest.get("audit_sentinels") != {
        "structural_rows_projected": EXPECTED_AUDIT_TARGET_COUNT,
        "rows_converted_to_python": 0,
        "artifact_ids_resolved": 0,
        "payloads_opened": 0,
        "metadata_loaded": False,
        "values_loaded": False,
    }:
        raise ValueError("candidate identity source audit sentinel drift")
    revision = manifest.get("producing_revision")
    if not isinstance(revision, Mapping):
        raise TypeError("candidate identity source producing revision is invalid")
    _validate_revision(revision)
    revision_files = revision.get("files")
    if not isinstance(revision_files, list):
        raise TypeError("candidate identity source revision inventory is invalid")
    revision_by_path = {
        str(record.get("path")): record
        for record in revision_files
        if isinstance(record, Mapping)
    }
    authorization = manifest.get("authorization")
    protocol = manifest.get("protocol")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("path") != AUTHORIZATION_RELATIVE
        or authorization.get("authorization_sha256") != AUTHORIZATION_SHA256
        or authorization.get("file_sha256")
        != revision_by_path.get(AUTHORIZATION_RELATIVE, {}).get("sha256")
        or not isinstance(protocol, Mapping)
        or protocol.get("path") != PROTOCOL_RELATIVE
        or protocol.get("sha256")
        != revision_by_path.get(PROTOCOL_RELATIVE, {}).get("sha256")
    ):
        raise ValueError("candidate identity source authorization binding drift")
    revision_root = Path(str(revision.get("repo_root"))).resolve()
    if not (revision_root / ".git").exists():
        revision_root = Path(__file__).resolve().parents[3]
    authorization_bytes = _git(
        revision_root,
        "show",
        f"{revision.get('git_commit')}:{AUTHORIZATION_RELATIVE}",
        binary=True,
    )
    assert isinstance(authorization_bytes, bytes)
    frozen_authorization = json.loads(authorization_bytes)
    if not isinstance(frozen_authorization, dict):
        raise TypeError("candidate identity frozen authorization is invalid")
    authorization_core = dict(frozen_authorization)
    if (
        authorization_core.pop("authorization_sha256", None) != AUTHORIZATION_SHA256
        or canonical_sha256(authorization_core) != AUTHORIZATION_SHA256
    ):
        raise ValueError("candidate identity frozen authorization drift")
    authorized_sources = frozen_authorization.get("sources")
    if not isinstance(authorized_sources, Mapping):
        raise TypeError("candidate identity frozen source bindings are invalid")
    c2_input = authorized_sources.get("c2_input")
    baseline = authorized_sources.get("c2_w64_baseline")
    assessment = authorized_sources.get("multiplex_assessment_provenance_only")
    plan = authorized_sources.get("candidate_union_plan")
    if not all(
        isinstance(record, Mapping) for record in (c2_input, baseline, assessment, plan)
    ):
        raise TypeError("candidate identity frozen source record is invalid")
    assert isinstance(c2_input, Mapping)
    assert isinstance(baseline, Mapping)
    assert isinstance(assessment, Mapping)
    assert isinstance(plan, Mapping)
    expected_sources = {
        "c2_input_manifest_sha256": c2_input.get("manifest_sha256"),
        "c2_input_manifest_file_sha256": c2_input.get("manifest_file_sha256"),
        "c2_w64_baseline_manifest_sha256": baseline.get("manifest_sha256"),
        "c2_w64_baseline_manifest_file_sha256": baseline.get("manifest_file_sha256"),
        "multiplex_assessment_manifest_sha256": assessment.get("manifest_sha256"),
        "multiplex_assessment_manifest_file_sha256": assessment.get(
            "manifest_file_sha256"
        ),
        "candidate_union_root": str(authorized_sources.get("candidate_union_root")),
        "candidate_union_plan_canonical_sha256": plan.get("canonical_sha256"),
    }
    if (
        manifest.get("sources") != expected_sources
        or manifest.get("cluster_state") != CLUSTER_STATE
        or manifest.get("identity_checks") != IDENTITY_CHECKS
    ):
        raise ValueError("candidate identity source provenance claim drift")
    records = manifest.get("files")
    if (
        not isinstance(records, list)
        or any(not isinstance(record, Mapping) for record in records)
        or len(records) != len(set(TARGET_FILES.values()) | set(PROFILE_FILES.values()))
    ):
        raise TypeError("candidate identity source file inventory is invalid")
    expected_files = set(TARGET_FILES.values()) | set(PROFILE_FILES.values())
    by_name = {
        str(record.get("path")): record
        for record in records
        if isinstance(record, Mapping)
    }
    if set(by_name) != expected_files:
        raise ValueError("candidate identity source file inventory drift")
    targets = {}
    profiles = {}
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("candidate identity source model binding is invalid")
    expected_model = (str(model.get("model_id")), str(model.get("model_revision")))
    basis_identity: dict[int, tuple[str, str, int, int, str]] = {}
    seen_target_basis: set[tuple[str, int]] = set()
    for partition in PARTITIONS:
        target_path = source_root / TARGET_FILES[partition]
        profile_path = source_root / PROFILE_FILES[partition]
        targets[partition] = _exact_table(
            target_path, TARGET_SCHEMA, by_name[target_path.name]
        )
        profiles[partition] = _exact_table(
            profile_path, PROFILE_SCHEMA, by_name[profile_path.name]
        )
        if set(targets[partition]["family_partition"].to_pylist()) != {partition}:
            raise ValueError("candidate identity source target partition drift")
        if set(profiles[partition]["family_partition"].to_pylist()) != {partition}:
            raise ValueError("candidate identity source profile partition drift")
        target_ids = set(targets[partition]["case_id"].to_pylist())
        if set(profiles[partition]["case_id"].to_pylist()) - target_ids:
            raise ValueError("candidate identity source profile target binding drift")
        for row in profiles[partition].to_pylist():
            basis_index = int(row["signed_basis_index"])
            identity = (
                str(row["model_id"]),
                str(row["model_revision"]),
                int(row["layer"]),
                int(row["neuron_index"]),
                str(row["polarity"]),
            )
            if identity[:2] != expected_model:
                raise ValueError("candidate identity source model identity drift")
            previous = basis_identity.setdefault(basis_index, identity)
            if previous != identity:
                raise ValueError("candidate identity source signed-basis join drift")
            target_basis = (str(row["case_id"]), basis_index)
            if target_basis in seen_target_basis:
                raise ValueError("candidate identity source target-basis duplication")
            seen_target_basis.add(target_basis)
            assigned = bool(row["c2_w64_assigned"])
            cluster = row["c2_w64_cluster_id"]
            if assigned != (cluster is not None) or (
                cluster is not None and not 0 <= int(cluster) < 64
            ):
                raise ValueError("candidate identity source W64 assignment drift")
            vector = np.asarray(row["candidate_contrast_vector"], dtype=np.float64)
            if (
                vector.shape != (5,)
                or not np.isfinite(vector).all()
                or int(row["candidate_occurrence_count"]) <= 0
            ):
                raise ValueError(
                    "candidate identity source reconstructed profile drift"
                )
    counts = manifest.get("counts")
    if (
        not isinstance(counts, Mapping)
        or counts.get("targets_by_partition")
        != {partition: targets[partition].num_rows for partition in PARTITIONS}
        or counts.get("profiles_by_partition")
        != {partition: profiles[partition].num_rows for partition in PARTITIONS}
    ):
        raise ValueError("candidate identity source count summary drift")
    if counts.get("targets_by_partition") != EXPECTED_TARGET_COUNTS:
        raise ValueError("candidate identity source frozen target count drift")
    structural = manifest.get("structural_target_projection")
    if structural != {
        "columns": list(PROJECTED_TARGET_COLUMNS),
        "all_partition_counts": {
            **EXPECTED_TARGET_COUNTS,
            "audit": EXPECTED_AUDIT_TARGET_COUNT,
        },
        "audit_rows_converted_to_python": False,
    }:
        raise ValueError("candidate identity source structural projection drift")
    selected = manifest.get("selected_payloads")
    if not isinstance(selected, Mapping) or not isinstance(
        selected.get("records"), list
    ):
        raise TypeError("candidate identity selected payload inventory is invalid")
    payload_records = selected["records"]
    target_records = {
        str(row["candidate_union_artifact_id"]): (
            str(row["case_id"]),
            partition,
            str(row["candidate_union_payload_sha256"]),
            str(row["source_width1_artifact_id"]),
            str(row["candidate_union_topology_sha256"]),
            int(row["response_position"]),
        )
        for partition in PARTITIONS
        for row in targets[partition].to_pylist()
    }
    payload_bindings = {
        str(record.get("artifact_id")): (
            str(record.get("case_id")),
            str(record.get("partition")),
            str(record.get("payload_sha256")),
            str(record.get("source_width1_artifact_id")),
            str(record.get("topology_sha256")),
            int(record.get("response_position", -1)),
        )
        for record in payload_records
        if isinstance(record, Mapping)
    }
    payload_paths_valid = all(
        isinstance(record, Mapping)
        and Path(str(record.get("relative_payload_path"))).parts
        == (
            CANDIDATE_UNION_TRACE_FAMILY_ID,
            Path(str(record.get("relative_payload_path"))).parts[1],
            str(record.get("artifact_id")),
            DATA_FILENAME,
        )
        for record in payload_records
        if isinstance(record, Mapping)
        and len(Path(str(record.get("relative_payload_path"))).parts) == 4
    ) and all(
        isinstance(record, Mapping)
        and len(Path(str(record.get("relative_payload_path"))).parts) == 4
        for record in payload_records
    )
    if (
        len(payload_records) != sum(EXPECTED_TARGET_COUNTS.values())
        or len(target_records) != sum(EXPECTED_TARGET_COUNTS.values())
        or len(payload_bindings) != sum(EXPECTED_TARGET_COUNTS.values())
        or int(selected.get("count", -1)) != len(payload_records)
        or selected.get("record_set_sha256") != canonical_sha256(payload_records)
        or payload_bindings != target_records
        or not payload_paths_valid
    ):
        raise ValueError("candidate identity selected payload binding drift")
    return manifest, targets, profiles
