"""Fail-closed persistence for the frozen candidate-aware clustering baseline.

This module is intentionally label-free.  It persists only graph affinities,
cluster assignments, and structural diagnostics produced from the generation
partition of a validated candidate-cluster input bundle.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.csgraph import connected_components

import circuits.analysis.bonafide.candidate_clustering as candidate_clustering_module
import circuits.analysis.bonafide.candidate_profiles as candidate_profiles_module
import circuits.analysis.bonafide.canonical as canonical_module
import circuits.analysis.bonafide.clustering as clustering_module
import circuits.analysis.bonafide.clustering_evaluation as clustering_evaluation_module
import circuits.analysis.bonafide.clustering_store as clustering_store_module
from circuits.analysis.bonafide.candidate_clustering import (
    CLUSTER_COUNTS,
    NEIGHBORS,
    RANDOM_SEEDS,
    CandidateClusterInputBundle,
    CandidateViewEvidence,
    GenerationClusterFit,
    ResolutionFit,
    build_generation_evidence,
    choose_medoid_seed,
    fit_generation_grid,
    load_candidate_cluster_input_bundle,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.analysis.bonafide.clustering import knn_affinity
from circuits.analysis.bonafide.clustering_evaluation import (
    assignment_ari,
    cluster_size_metrics,
    sparse_graph_partition_metrics,
)
from circuits.analysis.bonafide.clustering_store import csr_content_sha256

CANDIDATE_CLUSTER_BASELINE_SCHEMA = "adag.bonafide.candidate-clustering-baseline.v1"

AFFINITY_FILES = {
    "W": "affinity-W.npz",
    "C": "affinity-C.npz",
    "F": "affinity-F.npz",
    "S": "affinity-S.npz",
}
ASSIGNMENTS_FILE = "assignments.parquet"
COMMON_ELIGIBILITY_FILE = "common-eligibility.parquet"

_SOURCE_BINDINGS = {
    "canonical": "circuits/analysis/bonafide/canonical.py",
    "clustering": "circuits/analysis/bonafide/clustering.py",
    "clustering_store": "circuits/analysis/bonafide/clustering_store.py",
    "candidate_profiles": "circuits/analysis/bonafide/candidate_profiles.py",
    "candidate_clustering": ("circuits/analysis/bonafide/candidate_clustering.py"),
    "candidate_coherence": "circuits/analysis/bonafide/candidate_coherence.py",
    "candidate_nulls": "circuits/analysis/bonafide/candidate_nulls.py",
    "frozen_protocol": "docs/CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md",
    "frozen_clustering_evaluation": (
        "circuits/analysis/bonafide/clustering_evaluation.py"
    ),
    "candidate_clustering_execution": (
        "circuits/analysis/bonafide/candidate_clustering_execution.py"
    ),
    "candidate_clustering_fit_cli": ("scripts/bonafide/candidate_clustering_fit.py"),
}


def _runtime_source_paths() -> dict[str, Path]:
    return {
        "canonical": Path(str(canonical_module.__file__)),
        "clustering": Path(str(clustering_module.__file__)),
        "clustering_store": Path(str(clustering_store_module.__file__)),
        "candidate_profiles": Path(str(candidate_profiles_module.__file__)),
        "candidate_clustering": Path(str(candidate_clustering_module.__file__)),
        "frozen_clustering_evaluation": Path(
            str(clustering_evaluation_module.__file__)
        ),
        "candidate_clustering_execution": Path(__file__),
    }


_BASIS_FIELDS = (
    "signed_basis_index",
    "model_id",
    "model_revision",
    "layer",
    "neuron_index",
    "polarity",
)

COMMON_ELIGIBILITY_SCHEMA = pa.schema(
    [
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
        pa.field("common_eligible", pa.bool_(), nullable=False),
    ]
)

ASSIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("state_index", pa.int32(), nullable=False),
        pa.field("view", pa.string(), nullable=False),
        pa.field("n_clusters", pa.int32(), nullable=True),
        pa.field("seed", pa.int32(), nullable=False),
        pa.field("fit_valid", pa.bool_(), nullable=False),
        pa.field("seed_valid", pa.bool_(), nullable=False),
        pa.field("is_medoid", pa.bool_(), nullable=False),
        pa.field("assignment_fraction", pa.float64(), nullable=False),
        pa.field("fit_error", pa.string(), nullable=True),
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
        pa.field("eligible", pa.bool_(), nullable=False),
        pa.field("assigned", pa.bool_(), nullable=False),
        pa.field("cluster_id", pa.int32(), nullable=True),
        pa.field("assignment_status", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class LoadedCandidateClusteringBaseline:
    """Fully content-validated baseline artifacts."""

    root: Path
    manifest: Mapping[str, Any]
    affinities: Mapping[str, csr_matrix]
    assignments: pa.Table
    common_eligibility: pa.Table


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise ValueError(
            f"unable to collect candidate-clustering revision: {message}"
        ) from error
    return completed.stdout.strip()


def collect_candidate_clustering_revision(repo_root: Path) -> dict[str, Any]:
    """Bind a clean full tracked tree and every frozen implementation source."""

    repo_root = repo_root.resolve()
    if Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve() != repo_root:
        raise ValueError("candidate clustering must run from the repository root")
    for role, observed in _runtime_source_paths().items():
        expected = (repo_root / _SOURCE_BINDINGS[role]).resolve()
        if observed.resolve() != expected:
            raise ValueError(
                "candidate clustering runtime source path mismatch: "
                f"{role} loaded from {observed.resolve()}, expected {expected}"
            )
    tracked_status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if tracked_status:
        raise ValueError("candidate clustering requires a clean full tracked worktree")

    files: list[dict[str, str]] = []
    for role, relative in _SOURCE_BINDINGS.items():
        try:
            tracked = _git(
                repo_root,
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            )
        except ValueError as error:
            raise ValueError(
                f"candidate clustering source is not tracked: {relative}"
            ) from error
        if tracked != relative:
            raise ValueError(f"candidate clustering source tracking drift: {relative}")
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"candidate clustering source is missing: {relative}")
        files.append(
            {
                "role": role,
                "path": relative,
                "sha256": file_sha256(path),
            }
        )

    commit = _git(repo_root, "rev-parse", "HEAD")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    return {
        "repo_root": str(repo_root),
        "git_commit": commit,
        "git_tree": tree,
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(
            tracked_status.encode("utf-8")
        ).hexdigest(),
        "files": files,
    }


def _canonical_csr(matrix: csr_matrix) -> csr_matrix:
    result = matrix.copy().tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _validate_affinity(matrix: csr_matrix, *, basis_count: int, view: str) -> None:
    if matrix.shape != (basis_count, basis_count):
        raise ValueError(f"{view} affinity shape disagrees with the canonical basis")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{view} affinity contains nonfinite values")
    if np.any(matrix.data < 0):
        raise ValueError(f"{view} affinity contains negative values")
    if (matrix - matrix.T).nnz:
        raise ValueError(f"{view} affinity is not exactly symmetric")


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("candidate clustering diagnostics contain nonfinite values")
    return value


def _resolution_record(fit: ResolutionFit) -> dict[str, Any]:
    seed_records: list[dict[str, Any]] = []
    for seed in RANDOM_SEEDS:
        seed_fit = fit.seeds[seed]
        result = seed_fit.result
        seed_records.append(
            {
                "seed": seed,
                "valid": seed_fit.valid,
                "assignment_fraction": seed_fit.assignment_fraction,
                "error": seed_fit.error,
                "result_available": result is not None,
                "assigned_basis_count": (
                    int(np.sum(result.labels >= 0)) if result is not None else 0
                ),
                "active_basis_count": (
                    int(np.sum(result.active_mask)) if result is not None else 0
                ),
                "connected_component_count": (
                    result.connected_component_count if result is not None else None
                ),
                "cluster_sizes": (
                    {str(key): value for key, value in result.cluster_sizes.items()}
                    if result is not None
                    else {}
                ),
            }
        )
    return _safe_json(
        {
            "view": fit.view,
            "n_clusters": fit.n_clusters,
            "valid": fit.valid,
            "medoid_seed": fit.medoid_seed,
            "pairwise_seed_ari": [
                {"left_seed": left, "right_seed": right, "ari": score}
                for (left, right), score in sorted(fit.pairwise_seed_ari.items())
            ],
            "mean_seed_ari": fit.mean_seed_ari,
            "minimum_seed_ari": fit.minimum_seed_ari,
            "size_metrics": fit.size_metrics,
            "graph_metrics": fit.graph_metrics,
            "seeds": seed_records,
        }
    )


def _fit_states(
    fit: GenerationClusterFit,
) -> list[tuple[str, int | None, ResolutionFit | None]]:
    states = [
        (view, count, fit.directional[view][count])
        for view in ("W", "C", "F")
        for count in CLUSTER_COUNTS
    ]
    states.append(("S", fit.chosen_cluster_count, fit.support))
    return states


def _baseline_numerically_valid(fit: GenerationClusterFit) -> bool:
    chosen = fit.chosen_cluster_count
    return bool(
        chosen is not None
        and all(fit.directional[view][chosen].valid for view in ("W", "C", "F"))
        and fit.support is not None
        and fit.support.valid
    )


def _cross_view_ari(fit: GenerationClusterFit) -> dict[str, Any]:
    """Compare chosen medoids on each pair's common assigned eligible bases."""

    if not _baseline_numerically_valid(fit):
        return {
            "available": False,
            "reason": (
                "no_common_chosen_cluster_count"
                if fit.chosen_cluster_count is None
                else "invalid_required_chosen_resolution"
            ),
            "pairs": [],
        }
    chosen = fit.chosen_cluster_count
    assert chosen is not None
    assert fit.support is not None
    resolutions = {view: fit.directional[view][chosen] for view in ("W", "C", "F")}
    resolutions["S"] = fit.support
    labels: dict[str, np.ndarray] = {}
    for view, resolution in resolutions.items():
        value = resolution.labels
        if value is None:
            raise ValueError(f"valid {view} chosen resolution has no medoid assignment")
        labels[view] = value
    eligible = fit.evidence.common_eligible_mask
    pairs = []
    for left, right in itertools.combinations(("W", "C", "F", "S"), 2):
        common = eligible & (labels[left] >= 0) & (labels[right] >= 0)
        count = int(common.sum())
        if count < 2:
            raise ValueError(
                f"chosen {left}/{right} states have insufficient common assignments"
            )
        pairs.append(
            {
                "left_view": left,
                "right_view": right,
                "common_assigned_basis_count": count,
                "ari": assignment_ari(labels[left][common], labels[right][common]),
            }
        )
    return {
        "available": True,
        "reason": None,
        "chosen_cluster_count": chosen,
        "pairs": pairs,
    }


def _fit_affinities(fit: GenerationClusterFit) -> dict[str, csr_matrix]:
    affinities: dict[str, csr_matrix] = {}
    for view in ("W", "C", "F"):
        matrices = [
            _canonical_csr(fit.directional[view][count].affinity)
            for count in CLUSTER_COUNTS
        ]
        hashes = {csr_content_sha256(matrix) for matrix in matrices}
        if len(hashes) != 1:
            raise ValueError(f"{view} affinity drifted across cluster resolutions")
        affinities[view] = matrices[0]
    support = (
        fit.support.affinity
        if fit.support is not None
        else knn_affinity(
            fit.evidence.support_similarity,
            neighbors=NEIGHBORS,
            symmetrization="union_max",
        )
    )
    affinities["S"] = _canonical_csr(support)
    return affinities


def _basis_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in _BASIS_FIELDS}


def _common_eligibility_rows(
    bundle: CandidateClusterInputBundle,
    evidence: CandidateViewEvidence,
) -> list[dict[str, Any]]:
    if evidence.common_eligible_mask.shape != (bundle.basis_count,):
        raise ValueError("common eligibility shape disagrees with canonical basis")
    if len(bundle.basis_rows) != bundle.basis_count:
        raise ValueError("candidate clustering bundle basis rows are incomplete")
    rows: list[dict[str, Any]] = []
    for basis_index, basis in enumerate(bundle.basis_rows):
        identity = _basis_identity(basis)
        if int(identity["signed_basis_index"]) != basis_index:
            raise ValueError("candidate clustering basis rows are not canonical")
        rows.append(
            {
                **identity,
                "common_eligible": bool(evidence.common_eligible_mask[basis_index]),
            }
        )
    return rows


def _assignment_rows(
    bundle: CandidateClusterInputBundle,
    fit: GenerationClusterFit,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = fit.evidence.common_eligible_mask
    rows: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    state_index = 0
    for view, n_clusters, resolution in _fit_states(fit):
        for seed in RANDOM_SEEDS:
            seed_fit = None if resolution is None else resolution.seeds[seed]
            result = None if seed_fit is None else seed_fit.result
            state_error = (
                "no_common_chosen_cluster_count"
                if resolution is None
                else seed_fit.error
            )
            state_record = {
                "state_index": state_index,
                "view": view,
                "n_clusters": n_clusters,
                "seed": seed,
                "fit_valid": bool(resolution is not None and resolution.valid),
                "seed_valid": bool(seed_fit is not None and seed_fit.valid),
                "is_medoid": bool(
                    resolution is not None and resolution.medoid_seed == seed
                ),
                "assignment_fraction": (
                    seed_fit.assignment_fraction if seed_fit is not None else 0.0
                ),
                "fit_error": state_error,
            }
            state_records.append(dict(state_record))
            labels = None if result is None else result.labels
            if labels is not None and labels.shape != (bundle.basis_count,):
                raise ValueError(
                    "cluster assignment shape disagrees with canonical basis"
                )
            for basis_index, basis in enumerate(bundle.basis_rows):
                assigned = bool(labels is not None and labels[basis_index] >= 0)
                if assigned and not bool(eligible[basis_index]):
                    raise ValueError(
                        "cluster result assigned a basis outside common eligibility"
                    )
                if assigned:
                    status = "assigned"
                elif result is None:
                    status = "unavailable_fit_error"
                elif not bool(eligible[basis_index]):
                    status = "unassigned_ineligible"
                else:
                    status = "unassigned_no_recurring_positive_neighbor"
                rows.append(
                    {
                        **state_record,
                        **_basis_identity(basis),
                        "eligible": bool(eligible[basis_index]),
                        "assigned": assigned,
                        "cluster_id": (int(labels[basis_index]) if assigned else None),
                        "assignment_status": status,
                    }
                )
            state_index += 1
    return rows, state_records


def _source_input_record(bundle: CandidateClusterInputBundle) -> dict[str, Any]:
    manifest_path = bundle.root / "manifest.json"
    manifest = load_json_object(manifest_path)
    if manifest != dict(bundle.manifest):
        raise ValueError("loaded candidate-cluster input manifest changed during fit")
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core):
        raise ValueError("candidate-cluster input manifest hash drift")
    return {
        "path": str(bundle.root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "manifest_file_sha256": file_sha256(manifest_path),
        "schema_version": str(manifest["schema_version"]),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS, "atomic no-replace directory publication is unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            "candidate clustering output already exists",
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
    finally:
        os.close(descriptor)


def _move_staged_file(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Publish without overwrite, using a manifest-last fallback when needed."""

    try:
        _rename_directory_no_replace(source, destination)
        return
    except OSError as error:
        if error.errno not in {
            errno.EINVAL,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }:
            raise

    created_destination = False
    try:
        try:
            destination.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                error.errno,
                "candidate clustering output already exists",
                str(destination),
            ) from error
        created_destination = True
        manifest = source / "manifest.json"
        if not manifest.is_file():
            raise ValueError("staged candidate clustering manifest is missing")
        staged = sorted(path for path in source.iterdir() if path.name != manifest.name)
        if any(not path.is_file() for path in staged):
            raise ValueError("staged candidate clustering output contains a directory")
        for path in staged:
            _move_staged_file(path, destination / path.name)
        _fsync_directory(destination)
        _move_staged_file(manifest, destination / manifest.name)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        source.rmdir()
    except BaseException:
        if created_destination:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def run_candidate_clustering_baseline(
    *, input_root: Path, output_root: Path, repo_root: Path
) -> dict[str, Any]:
    """Run the frozen generation grid and publish one immutable baseline root."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"candidate clustering output already exists: {output_root}"
        )
    revision = collect_candidate_clustering_revision(repo_root)
    bundle = load_candidate_cluster_input_bundle(input_root)
    if bundle.root != input_root.resolve():
        raise ValueError("candidate-cluster loader returned a different input root")
    evidence = build_generation_evidence(bundle)
    fit = fit_generation_grid(evidence)
    if collect_candidate_clustering_revision(repo_root) != revision:
        raise ValueError("candidate clustering source changed during execution")

    affinities = _fit_affinities(fit)
    for view, affinity in affinities.items():
        _validate_affinity(affinity, basis_count=bundle.basis_count, view=view)
    eligibility_rows = _common_eligibility_rows(bundle, evidence)
    assignment_rows, state_records = _assignment_rows(bundle, fit)
    input_record = _source_input_record(bundle)
    numerically_valid = _baseline_numerically_valid(fit)
    cross_view_ari = _cross_view_ari(fit)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        matrix_records: dict[str, dict[str, Any]] = {}
        for view, filename in AFFINITY_FILES.items():
            matrix = affinities[view]
            path = temporary / filename
            save_npz(path, matrix, compressed=True)
            matrix_records[view] = {
                "path": filename,
                "shape": list(matrix.shape),
                "dtype": matrix.dtype.str,
                "nnz": int(matrix.nnz),
                "content_sha256": csr_content_sha256(matrix),
            }

        common_path = temporary / COMMON_ELIGIBILITY_FILE
        pq.write_table(
            pa.Table.from_pylist(eligibility_rows, schema=COMMON_ELIGIBILITY_SCHEMA),
            common_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        assignment_path = temporary / ASSIGNMENTS_FILE
        pq.write_table(
            pa.Table.from_pylist(assignment_rows, schema=ASSIGNMENT_SCHEMA),
            assignment_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )

        files = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                **(
                    {"row_count": pq.read_metadata(path).num_rows}
                    if path.suffix == ".parquet"
                    else {}
                ),
            }
            for path in sorted(temporary.iterdir())
        ]
        resolution_records = [
            _resolution_record(fit.directional[view][count])
            for view in ("W", "C", "F")
            for count in CLUSTER_COUNTS
        ]
        if fit.support is not None:
            resolution_records.append(_resolution_record(fit.support))
        manifest: dict[str, Any] = {
            "schema_version": CANDIDATE_CLUSTER_BASELINE_SCHEMA,
            "purpose": "label_free_generation_only_candidate_clustering_baseline",
            "source_input_bundle": input_record,
            "code_revision": revision,
            "configuration": {
                "views": ["W", "C", "F", "S"],
                "directional_cluster_counts": list(CLUSTER_COUNTS),
                "random_seeds": list(RANDOM_SEEDS),
                "neighbors": NEIGHBORS,
                "knn_symmetrization": "union_max",
            },
            "basis_count": bundle.basis_count,
            "common_eligible_basis_count": int(evidence.common_eligible_mask.sum()),
            "chosen_cluster_count": fit.chosen_cluster_count,
            "support_fit_performed": fit.support is not None,
            "matrix_records": matrix_records,
            "states": state_records,
            "resolution_diagnostics": resolution_records,
            "cross_view_common_basis_ari": cross_view_ari,
            "numerically_valid": numerically_valid,
            "diagnostic_only": not numerically_valid,
            "outcomes_inspected": False,
            "descriptions_generated": False,
            "model_calls_made": False,
            "confirmatory_holdout_opened": False,
            "files": files,
        }
        manifest = _safe_json(manifest)
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _write_json(temporary / "manifest.json", manifest)
        _publish_directory_no_replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _exact_table(path: Path, schema: pa.Schema) -> pa.Table:
    table = pq.read_table(path)
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"candidate baseline parquet schema drift: {path.name}")
    return table


def _validated_baseline_manifest(root: Path) -> dict[str, Any]:
    manifest = load_json_object(root / "manifest.json")
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core):
        raise ValueError("candidate clustering baseline manifest hash mismatch")
    if manifest.get("schema_version") != CANDIDATE_CLUSTER_BASELINE_SCHEMA:
        raise ValueError("unsupported candidate clustering baseline schema")
    for flag in (
        "outcomes_inspected",
        "descriptions_generated",
        "model_calls_made",
        "confirmatory_holdout_opened",
    ):
        if manifest.get(flag) is not False:
            raise ValueError(f"candidate clustering baseline violates {flag} firewall")
    numerically_valid = manifest.get("numerically_valid")
    if not isinstance(numerically_valid, bool):
        raise TypeError("candidate clustering baseline numerical validity is invalid")
    if manifest.get("diagnostic_only") is not (not numerically_valid):
        raise ValueError("candidate clustering baseline diagnostic status drift")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TypeError("candidate clustering baseline file inventory is invalid")
    expected = set(AFFINITY_FILES.values()) | {
        ASSIGNMENTS_FILE,
        COMMON_ELIGIBILITY_FILE,
    }
    by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("candidate clustering baseline file record is invalid")
        name = str(record.get("path"))
        if Path(name).name != name or name in by_name:
            raise ValueError("candidate clustering baseline file path is unsafe")
        by_name[name] = record
    if set(by_name) != expected:
        raise ValueError("candidate clustering baseline file inventory is incomplete")
    for name, record in by_name.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"candidate clustering baseline file is missing: {name}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"candidate clustering baseline file size drift: {name}")
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"candidate clustering baseline file hash drift: {name}")
        if path.suffix == ".parquet" and pq.read_metadata(path).num_rows != int(
            record.get("row_count", -1)
        ):
            raise ValueError(f"candidate clustering baseline row count drift: {name}")
    return manifest


def _validate_common_rows(table: pa.Table, *, basis_count: int) -> list[dict[str, Any]]:
    rows = table.to_pylist()
    if len(rows) != basis_count:
        raise ValueError("common eligibility row count disagrees with basis count")
    if [int(row["signed_basis_index"]) for row in rows] != list(range(basis_count)):
        raise ValueError("common eligibility basis index is not canonical")
    return rows


def _validate_assignment_rows(
    table: pa.Table,
    *,
    manifest: Mapping[str, Any],
    common_rows: Sequence[Mapping[str, Any]],
) -> None:
    states = manifest.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("candidate clustering baseline state inventory is invalid")
    chosen = manifest.get("chosen_cluster_count")
    if chosen is not None and chosen not in CLUSTER_COUNTS:
        raise ValueError("candidate clustering chosen cluster count is invalid")
    expected_keys = [
        (view, count, seed)
        for view in ("W", "C", "F")
        for count in CLUSTER_COUNTS
        for seed in RANDOM_SEEDS
    ] + [("S", chosen, seed) for seed in RANDOM_SEEDS]
    observed_keys = [
        (
            (state.get("view"), state.get("n_clusters"), state.get("seed"))
            if isinstance(state, Mapping)
            else None
        )
        for state in states
    ]
    if observed_keys != expected_keys:
        raise ValueError("candidate clustering baseline state grid drift")
    grouped_states: dict[tuple[str, int | None], list[Mapping[str, Any]]] = {}
    for state in states:
        assert isinstance(state, Mapping)
        grouped_states.setdefault((str(state["view"]), state["n_clusters"]), []).append(
            state
        )
    resolution_validity: dict[tuple[str, int | None], bool] = {}
    for key, group in grouped_states.items():
        fit_values = {bool(state["fit_valid"]) for state in group}
        if len(fit_values) != 1:
            raise ValueError("candidate clustering fit validity differs across seeds")
        fit_valid = fit_values.pop()
        if fit_valid != all(bool(state["seed_valid"]) for state in group):
            raise ValueError("candidate clustering fit/seed validity disagrees")
        medoid_count = sum(bool(state["is_medoid"]) for state in group)
        if medoid_count != (1 if fit_valid else 0):
            raise ValueError("candidate clustering medoid marker count is invalid")
        resolution_validity[key] = fit_valid
    common_counts = [
        count
        for count in CLUSTER_COUNTS
        if all(resolution_validity[(view, count)] for view in ("W", "C", "F"))
    ]
    expected_chosen = (
        64 if 64 in common_counts else (min(common_counts) if common_counts else None)
    )
    if chosen != expected_chosen:
        raise ValueError("candidate clustering chosen cluster count drift")
    if bool(manifest.get("support_fit_performed")) is not (chosen is not None):
        raise ValueError("candidate clustering support-fit status drift")
    expected_numerical_validity = bool(
        chosen is not None and resolution_validity[("S", chosen)]
    )
    if manifest.get("numerically_valid") is not expected_numerical_validity:
        raise ValueError("candidate clustering numerical validity drift")
    basis_count = len(common_rows)
    rows = table.to_pylist()
    if len(rows) != len(states) * basis_count:
        raise ValueError("candidate clustering assignment row count is incomplete")
    for expected_index, state in enumerate(states):
        if not isinstance(state, Mapping) or state.get("state_index") != expected_index:
            raise ValueError("candidate clustering state index is invalid")
        fraction = float(state.get("assignment_fraction", float("nan")))
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("candidate clustering assignment fraction is invalid")
        block = rows[expected_index * basis_count : (expected_index + 1) * basis_count]
        for basis_index, (row, common) in enumerate(
            zip(block, common_rows, strict=True)
        ):
            if int(row["state_index"]) != expected_index:
                raise ValueError("candidate clustering assignment state order drift")
            if int(row["signed_basis_index"]) != basis_index:
                raise ValueError("candidate clustering assignment basis order drift")
            for field in _BASIS_FIELDS:
                if row[field] != common[field]:
                    raise ValueError(
                        "candidate clustering assignment basis identity drift"
                    )
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
                    raise ValueError(
                        "candidate clustering assignment state metadata drift"
                    )
            if bool(row["eligible"]) != bool(common["common_eligible"]):
                raise ValueError("candidate clustering assignment eligibility drift")
            assigned = bool(row["assigned"])
            cluster_id = row["cluster_id"]
            if assigned != (cluster_id is not None):
                raise ValueError("candidate clustering assignment nullability drift")
            if assigned and not bool(row["eligible"]):
                raise ValueError("candidate clustering assigned an ineligible basis")
            if cluster_id is not None:
                n_clusters = row["n_clusters"]
                if n_clusters is None or not 0 <= int(cluster_id) < int(n_clusters):
                    raise ValueError(
                        "candidate clustering assignment cluster is invalid"
                    )

    recorded_cross_view = manifest.get("cross_view_common_basis_ari")
    if not isinstance(recorded_cross_view, Mapping):
        raise TypeError("candidate clustering cross-view ARI report is invalid")
    if not expected_numerical_validity:
        expected_cross_view = {
            "available": False,
            "reason": (
                "no_common_chosen_cluster_count"
                if chosen is None
                else "invalid_required_chosen_resolution"
            ),
            "pairs": [],
        }
    else:
        assert chosen is not None
        eligible = np.asarray(
            [bool(row["common_eligible"]) for row in common_rows], dtype=np.bool_
        )
        labels: dict[str, np.ndarray] = {}
        for view in ("W", "C", "F", "S"):
            medoid = next(
                state
                for state in grouped_states[(view, chosen)]
                if bool(state["is_medoid"])
            )
            state_index = int(medoid["state_index"])
            block = rows[state_index * basis_count : (state_index + 1) * basis_count]
            labels[view] = np.asarray(
                [
                    -1 if row["cluster_id"] is None else int(row["cluster_id"])
                    for row in block
                ],
                dtype=np.int64,
            )
        pairs = []
        for left, right in itertools.combinations(("W", "C", "F", "S"), 2):
            common = eligible & (labels[left] >= 0) & (labels[right] >= 0)
            if int(common.sum()) < 2:
                raise ValueError(
                    "candidate clustering cross-view ARI has insufficient support"
                )
            pairs.append(
                {
                    "left_view": left,
                    "right_view": right,
                    "common_assigned_basis_count": int(common.sum()),
                    "ari": assignment_ari(labels[left][common], labels[right][common]),
                }
            )
        expected_cross_view = {
            "available": True,
            "reason": None,
            "chosen_cluster_count": chosen,
            "pairs": pairs,
        }
    if recorded_cross_view != expected_cross_view:
        raise ValueError("candidate clustering cross-view ARI report drift")


def _validate_resolution_diagnostics(
    *,
    manifest: Mapping[str, Any],
    assignments: pa.Table,
    common_rows: Sequence[Mapping[str, Any]],
    affinities: Mapping[str, csr_matrix],
) -> None:
    diagnostics = manifest.get("resolution_diagnostics")
    if not isinstance(diagnostics, list):
        raise TypeError("candidate clustering resolution diagnostics are invalid")
    chosen = manifest.get("chosen_cluster_count")
    expected_keys = [
        (view, count) for view in ("W", "C", "F") for count in CLUSTER_COUNTS
    ]
    if chosen is not None:
        expected_keys.append(("S", chosen))
    observed_keys = [
        (
            (record.get("view"), record.get("n_clusters"))
            if isinstance(record, Mapping)
            else None
        )
        for record in diagnostics
    ]
    if observed_keys != expected_keys:
        raise ValueError("candidate clustering resolution diagnostic grid drift")

    states = manifest["states"]
    assert isinstance(states, list)
    rows = assignments.to_pylist()
    basis_count = len(common_rows)
    eligible = np.asarray(
        [bool(row["common_eligible"]) for row in common_rows], dtype=np.bool_
    )
    eligible_count = int(eligible.sum())
    for record in diagnostics:
        assert isinstance(record, Mapping)
        view = str(record["view"])
        count = int(record["n_clusters"])
        state_group = [
            state
            for state in states
            if state["view"] == view and state["n_clusters"] == count
        ]
        seed_diagnostics = record.get("seeds")
        if not isinstance(seed_diagnostics, list) or [
            item.get("seed") if isinstance(item, Mapping) else None
            for item in seed_diagnostics
        ] != list(RANDOM_SEEDS):
            raise ValueError("candidate clustering seed diagnostic grid drift")

        labels_by_seed: dict[int, np.ndarray] = {}
        affinity_active = np.asarray(affinities[view].sum(axis=1)).ravel() > 0
        actual_component_count = (
            int(
                connected_components(
                    affinities[view][affinity_active][:, affinity_active],
                    directed=False,
                    return_labels=False,
                )
            )
            if np.any(affinity_active)
            else 0
        )
        for state, seed_record in zip(state_group, seed_diagnostics, strict=True):
            assert isinstance(state, Mapping)
            assert isinstance(seed_record, Mapping)
            state_index = int(state["state_index"])
            block = rows[state_index * basis_count : (state_index + 1) * basis_count]
            labels = np.asarray(
                [
                    -1 if row["cluster_id"] is None else int(row["cluster_id"])
                    for row in block
                ],
                dtype=np.int64,
            )
            assigned = labels >= 0
            statuses = [str(row["assignment_status"]) for row in block]
            result_available = any(
                status != "unavailable_fit_error" for status in statuses
            )
            if result_available and not np.array_equal(assigned, affinity_active):
                raise ValueError("candidate clustering active assignment drift")
            expected_statuses = []
            for basis_index in range(basis_count):
                if assigned[basis_index]:
                    expected_statuses.append("assigned")
                elif not result_available:
                    expected_statuses.append("unavailable_fit_error")
                elif not eligible[basis_index]:
                    expected_statuses.append("unassigned_ineligible")
                else:
                    expected_statuses.append(
                        "unassigned_no_recurring_positive_neighbor"
                    )
            if statuses != expected_statuses:
                raise ValueError("candidate clustering assignment status drift")
            expected_fraction = (
                float(np.sum(assigned & eligible) / eligible_count)
                if eligible_count
                else 0.0
            )
            if not np.isclose(
                float(state["assignment_fraction"]),
                expected_fraction,
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError("candidate clustering assignment fraction drift")
            unique = np.unique(labels[assigned])
            exact_clusters = np.array_equal(unique, np.arange(count, dtype=np.int64))
            expected_seed_valid = bool(
                result_available
                and int(affinity_active.sum()) >= count + 1
                and actual_component_count <= count
                and expected_fraction >= 0.95
                and exact_clusters
            )
            if bool(state["seed_valid"]) != expected_seed_valid:
                raise ValueError("candidate clustering seed validity drift")
            if expected_seed_valid:
                expected_error = None
            elif result_available and expected_fraction < 0.95:
                expected_error = "assignment_coverage"
            elif result_available:
                expected_error = "assigned_cluster_count"
            else:
                expected_error = state["fit_error"]
                if not isinstance(expected_error, str) or not expected_error:
                    raise ValueError("candidate clustering missing fit error")
            if state["fit_error"] != expected_error:
                raise ValueError("candidate clustering fit error drift")
            cluster_sizes = {
                str(cluster): int(size)
                for cluster, size in zip(
                    *np.unique(labels[assigned], return_counts=True), strict=True
                )
            }
            expected_seed_record = {
                "seed": int(state["seed"]),
                "valid": bool(state["seed_valid"]),
                "assignment_fraction": float(state["assignment_fraction"]),
                "error": state["fit_error"],
                "result_available": result_available,
                "assigned_basis_count": int(assigned.sum()),
                "active_basis_count": int(assigned.sum()),
                "connected_component_count": (
                    actual_component_count if result_available else None
                ),
                "cluster_sizes": cluster_sizes if result_available else {},
            }
            if dict(seed_record) != expected_seed_record:
                raise ValueError("candidate clustering seed diagnostics drift")
            if result_available:
                component_count = seed_record.get("connected_component_count")
                if (
                    isinstance(component_count, bool)
                    or not isinstance(component_count, int)
                    or component_count < 1
                ):
                    raise ValueError("candidate clustering component diagnostics drift")
                labels_by_seed[int(state["seed"])] = labels
            elif seed_record.get("connected_component_count") is not None:
                raise ValueError(
                    "candidate clustering unavailable-result diagnostics drift"
                )

        fit_valid = all(bool(state["seed_valid"]) for state in state_group)
        if record.get("valid") is not fit_valid:
            raise ValueError("candidate clustering resolution validity drift")
        if fit_valid:
            medoid_seed, pairwise = choose_medoid_seed(labels_by_seed)
            if {
                int(state["seed"]) for state in state_group if bool(state["is_medoid"])
            } != {medoid_seed}:
                raise ValueError("candidate clustering medoid assignment drift")
            pairwise_records = [
                {"left_seed": left, "right_seed": right, "ari": score}
                for (left, right), score in sorted(pairwise.items())
            ]
            values = list(pairwise.values())
            medoid_labels = labels_by_seed[medoid_seed]
            expected_size = cluster_size_metrics(medoid_labels)
            expected_graph = sparse_graph_partition_metrics(
                medoid_labels, affinities[view]
            )
            if (
                record.get("medoid_seed") != medoid_seed
                or record.get("pairwise_seed_ari") != pairwise_records
                or record.get("mean_seed_ari") != mean(values)
                or record.get("minimum_seed_ari") != min(values)
                or record.get("size_metrics") != expected_size
                or record.get("graph_metrics") != expected_graph
            ):
                raise ValueError("candidate clustering resolution metrics drift")
        elif (
            any(
                record.get(field) is not None
                for field in (
                    "medoid_seed",
                    "mean_seed_ari",
                    "minimum_seed_ari",
                    "size_metrics",
                    "graph_metrics",
                )
            )
            or record.get("pairwise_seed_ari") != []
        ):
            raise ValueError("candidate clustering invalid-resolution metrics drift")


def _validate_source_input(manifest: Mapping[str, Any]) -> None:
    source = manifest.get("source_input_bundle")
    if not isinstance(source, Mapping):
        raise TypeError("candidate clustering baseline source input is invalid")
    path = Path(str(source.get("manifest_path")))
    if not path.is_file():
        raise ValueError("candidate clustering baseline source manifest is missing")
    if file_sha256(path) != source.get("manifest_file_sha256"):
        raise ValueError("candidate clustering baseline source manifest file drift")
    source_manifest = load_json_object(path)
    core = dict(source_manifest)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core) or recorded != source.get("manifest_sha256"):
        raise ValueError("candidate clustering baseline source manifest hash drift")
    bundle = load_candidate_cluster_input_bundle(Path(str(source.get("path"))))
    if (
        bundle.root != Path(str(source.get("path"))).resolve()
        or bundle.manifest.get("manifest_sha256") != source.get("manifest_sha256")
        or bundle.manifest.get("schema_version") != source.get("schema_version")
    ):
        raise ValueError("candidate clustering baseline source bundle drift")


def _validate_revision(manifest: Mapping[str, Any]) -> None:
    revision = manifest.get("code_revision")
    if not isinstance(revision, Mapping):
        raise TypeError("candidate clustering baseline code revision is invalid")
    if revision.get("tracked_worktree_clean") is not True:
        raise ValueError("candidate clustering baseline was not fit from a clean tree")
    for field, length in (("git_commit", 40), ("git_tree", 40)):
        value = revision.get(field)
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"candidate clustering baseline {field} is invalid")
    records = revision.get("files")
    if not isinstance(records, list):
        raise TypeError("candidate clustering baseline source hashes are invalid")
    by_role = {
        str(record.get("role")): record
        for record in records
        if isinstance(record, Mapping)
    }
    if set(by_role) != set(_SOURCE_BINDINGS):
        raise ValueError("candidate clustering baseline source hash inventory drift")
    for role, relative in _SOURCE_BINDINGS.items():
        record = by_role[role]
        digest = record.get("sha256")
        if record.get("path") != relative or (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"candidate clustering baseline source hash drift: {role}")


def load_candidate_clustering_baseline(
    root: Path, *, verify_source: bool = True
) -> LoadedCandidateClusteringBaseline:
    """Load and content-validate one persisted candidate baseline."""

    root = root.resolve()
    manifest = _validated_baseline_manifest(root)
    _validate_revision(manifest)
    configuration = manifest.get("configuration")
    expected_configuration = {
        "views": ["W", "C", "F", "S"],
        "directional_cluster_counts": list(CLUSTER_COUNTS),
        "random_seeds": list(RANDOM_SEEDS),
        "neighbors": NEIGHBORS,
        "knn_symmetrization": "union_max",
    }
    if configuration != expected_configuration:
        raise ValueError("candidate clustering baseline configuration drift")
    basis_count = int(manifest.get("basis_count", -1))
    if basis_count < 1:
        raise ValueError("candidate clustering baseline basis count is invalid")
    matrix_records = manifest.get("matrix_records")
    if not isinstance(matrix_records, Mapping) or set(matrix_records) != set(
        AFFINITY_FILES
    ):
        raise ValueError("candidate clustering baseline matrix inventory is invalid")
    affinities: dict[str, csr_matrix] = {}
    for view, filename in AFFINITY_FILES.items():
        record = matrix_records[view]
        if not isinstance(record, Mapping) or record.get("path") != filename:
            raise ValueError(f"candidate clustering {view} matrix record is invalid")
        matrix = _canonical_csr(load_npz(root / filename).tocsr())
        _validate_affinity(matrix, basis_count=basis_count, view=view)
        if list(matrix.shape) != record.get("shape"):
            raise ValueError(f"candidate clustering {view} affinity shape drift")
        if matrix.dtype.str != record.get("dtype"):
            raise ValueError(f"candidate clustering {view} affinity dtype drift")
        if int(matrix.nnz) != int(record.get("nnz", -1)):
            raise ValueError(f"candidate clustering {view} affinity nnz drift")
        if csr_content_sha256(matrix) != record.get("content_sha256"):
            raise ValueError(f"candidate clustering {view} affinity content drift")
        affinities[view] = matrix

    common = _exact_table(root / COMMON_ELIGIBILITY_FILE, COMMON_ELIGIBILITY_SCHEMA)
    common_rows = _validate_common_rows(common, basis_count=basis_count)
    if sum(bool(row["common_eligible"]) for row in common_rows) != int(
        manifest.get("common_eligible_basis_count", -1)
    ):
        raise ValueError("candidate clustering common eligibility count drift")
    assignments = _exact_table(root / ASSIGNMENTS_FILE, ASSIGNMENT_SCHEMA)
    _validate_assignment_rows(assignments, manifest=manifest, common_rows=common_rows)
    _validate_resolution_diagnostics(
        manifest=manifest,
        assignments=assignments,
        common_rows=common_rows,
        affinities=affinities,
    )
    if verify_source:
        _validate_source_input(manifest)
    return LoadedCandidateClusteringBaseline(
        root=root,
        manifest=manifest,
        affinities=affinities,
        assignments=assignments,
        common_eligibility=common,
    )
