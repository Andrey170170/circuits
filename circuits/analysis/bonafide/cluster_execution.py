"""Frozen execution plans and atomic states for sparse BonaFide clustering."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
    write_hashed_json,
)
from circuits.analysis.bonafide.clustering import (
    SPARSE_CLUSTER_SCHEMA,
    KnnSymmetrization,
    SparseSpectralResult,
    knn_affinity,
    mean_similarity_matrix,
    sparse_spectral_cluster,
)
from circuits.analysis.bonafide.clustering_store import (
    BasisSupport,
    FeatureStoreReader,
    PairEvidence,
    load_pair_evidence,
)

CLUSTER_PLAN_SCHEMA = "adag.bonafide.sparse-clustering-plan.v1"

ASSIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
        pa.field("target_count", pa.int64(), nullable=False),
        pa.field("response_count", pa.int64(), nullable=False),
        pa.field("family_count", pa.int64(), nullable=False),
        pa.field("eligible", pa.bool_(), nullable=False),
        pa.field("assigned", pa.bool_(), nullable=False),
        pa.field("cluster_id", pa.int32(), nullable=True),
        pa.field("assignment_status", pa.string(), nullable=False),
    ]
)


def _knn_symmetrization(value: object) -> KnnSymmetrization:
    if value == "union_max":
        return "union_max"
    if value == "mutual_min":
        return "mutual_min"
    raise ValueError("unsupported kNN symmetrization")


def collect_clustering_code_revision(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    scoped_paths = (
        "circuits/analysis/bonafide",
        "scripts/bonafide/clustering_build_evidence.py",
        "scripts/bonafide/clustering_build_plan.py",
        "scripts/bonafide/clustering_evaluate.py",
        "scripts/bonafide/clustering_finalize.py",
        "scripts/bonafide/clustering_fit.py",
        "scripts/bonafide/clustering_project.py",
        "scripts/bonafide/clustering_resample_build_plan.py",
        "scripts/bonafide/clustering_resample_evidence.py",
        "scripts/bonafide/clustering_resample_fit.py",
        "scripts/bonafide/clustering_resample_report.py",
        "scripts/bonafide/clustering_evidence.sbatch",
        "scripts/bonafide/clustering_evaluate.sbatch",
        "scripts/bonafide/clustering_finalize.sbatch",
        "scripts/bonafide/clustering_project.sbatch",
        "scripts/bonafide/clustering_resample_build_plan.sbatch",
        "scripts/bonafide/clustering_resample_evidence.sbatch",
        "scripts/bonafide/clustering_resample_fit.sbatch",
        "scripts/bonafide/clustering_resample_report.sbatch",
        "scripts/bonafide/clustering_sweep.sbatch",
        "pyproject.toml",
        "uv.lock",
    )
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *scoped_paths,
    )
    source_paths: list[Path] = []
    for relative in scoped_paths:
        path = repo_root / relative
        if path.is_dir():
            source_paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and not candidate.name.endswith(".pyc")
            )
        elif path.is_file():
            source_paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(source_paths)):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_tree_sha256": digest.hexdigest(),
    }


def collect_clustering_environment() -> dict[str, Any]:
    distributions = ("circuits", "numpy", "pyarrow", "scikit-learn", "scipy")
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _cluster_configurations() -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []

    def add(
        *,
        weighting: str,
        basis_target_support: int,
        pair_target_overlap: int,
        neighbors: int,
        n_clusters: int,
        random_seed: int,
        sensitivity: str,
    ) -> None:
        config: dict[str, Any] = {
            "feature_view": "input_attribution_only",
            "weighting": weighting,
            "basis_min_target_count": basis_target_support,
            "basis_min_response_count": 2,
            "basis_min_family_count": 2,
            "exclude_boundary_layers": True,
            "pair_min_target_overlap": pair_target_overlap,
            "pair_min_response_overlap": 2,
            "pair_min_family_overlap": 2,
            "positive_affinity_only": True,
            "neighbors": neighbors,
            "knn_symmetrization": "union_max",
            "minimum_affinity": 0.0,
            "algorithm": "normalized_sparse_spectral_kmeans",
            "n_clusters": n_clusters,
            "random_seed": random_seed,
            "self_loop_weight": 1.0,
            "eigen_tolerance": 1e-6,
            "sensitivity": sensitivity,
            "descriptions_generated": False,
        }
        config["config_sha256"] = canonical_sha256(config)
        configurations.append(config)

    cluster_counts = (32, 64, 96, 128)
    seeds = (17, 29, 43)
    for weighting in ("hierarchical", "unweighted"):
        for n_clusters in cluster_counts:
            for random_seed in seeds:
                add(
                    weighting=weighting,
                    basis_target_support=3,
                    pair_target_overlap=2,
                    neighbors=32,
                    n_clusters=n_clusters,
                    random_seed=random_seed,
                    sensitivity=(
                        "primary" if weighting == "hierarchical" else "unweighted"
                    ),
                )
    for threshold in (2, 5):
        for n_clusters in cluster_counts:
            add(
                weighting="hierarchical",
                basis_target_support=threshold,
                pair_target_overlap=2,
                neighbors=32,
                n_clusters=n_clusters,
                random_seed=17,
                sensitivity=f"basis_target_support_{threshold}",
            )
    for neighbors in (16, 64):
        for n_clusters in cluster_counts:
            add(
                weighting="hierarchical",
                basis_target_support=3,
                pair_target_overlap=2,
                neighbors=neighbors,
                n_clusters=n_clusters,
                random_seed=17,
                sensitivity=f"neighbors_{neighbors}",
            )
    for n_clusters in cluster_counts:
        add(
            weighting="hierarchical",
            basis_target_support=3,
            pair_target_overlap=3,
            neighbors=32,
            n_clusters=n_clusters,
            random_seed=17,
            sensitivity="pair_target_overlap_3",
        )
    for task_index, config in enumerate(configurations):
        config["task_index"] = task_index
    return configurations


def build_clustering_plan(
    *,
    repo_root: Path,
    feature_store_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    reader = FeatureStoreReader(feature_store_root)
    revision = collect_clustering_code_revision(repo_root)
    if revision["git_dirty"]:
        raise ValueError("refuse to freeze clustering plan from a dirty source tree")
    plan: dict[str, Any] = {
        "schema_version": CLUSTER_PLAN_SCHEMA,
        "repo_root": str(repo_root.resolve()),
        "feature_store": {
            "path": str(reader.compacted_root),
            "manifest_sha256": reader.manifest["manifest_sha256"],
            "plan_sha256": reader.manifest["plan_sha256"],
        },
        "output_root": str(output_root.resolve()),
        "pair_evidence": [
            {
                "task_index": index,
                "weighting": weighting,
                "output_path": str(output_root.resolve() / "pair-evidence" / weighting),
            }
            for index, weighting in enumerate(("hierarchical", "unweighted"))
        ],
        "configurations": _cluster_configurations(),
        "code_revision": revision,
        "environment": collect_clustering_environment(),
        "descriptions_generated": False,
        "scientific_cluster_state": False,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def write_clustering_plan(path: Path, plan: Mapping[str, Any]) -> None:
    write_hashed_json(path, plan, hash_field="plan_sha256")


def validate_clustering_plan(
    plan: Mapping[str, Any],
    *,
    repo_root: Path,
    verify_code: bool = True,
) -> dict[str, Any]:
    validated = dict(plan)
    core = dict(validated)
    recorded_hash = core.pop("plan_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("clustering plan hash mismatch")
    if validated.get("schema_version") != CLUSTER_PLAN_SCHEMA:
        raise ValueError("unsupported clustering plan schema")
    reader = FeatureStoreReader(Path(str(validated["feature_store"]["path"])))
    if (
        reader.manifest["manifest_sha256"]
        != validated["feature_store"]["manifest_sha256"]
    ):
        raise ValueError("clustering plan feature-store drift")
    if Path(str(validated.get("repo_root"))).resolve() != repo_root.resolve():
        raise ValueError("clustering plan belongs to another executable worktree")
    configurations = validated.get("configurations")
    if not isinstance(configurations, list) or len(configurations) != 44:
        raise ValueError("clustering plan configuration grid is invalid")
    for task_index, config in enumerate(configurations):
        if not isinstance(config, Mapping) or config.get("task_index") != task_index:
            raise ValueError("clustering plan task index is invalid")
        core_config = dict(config)
        config_hash = core_config.pop("config_sha256", None)
        core_config.pop("task_index", None)
        if config_hash != canonical_sha256(core_config):
            raise ValueError("clustering configuration hash mismatch")
    if verify_code:
        current = collect_clustering_code_revision(repo_root)
        if current != validated.get("code_revision"):
            raise ValueError("clustering executable source has drifted")
    return validated


def _validate_existing_cluster_state(output_path: Path) -> dict[str, Any]:
    manifest = load_json_object(output_path / "manifest.json")
    core = dict(manifest)
    recorded_hash = core.pop("manifest_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("cluster-state manifest hash mismatch")
    assignment_path = output_path / "assignments.parquet"
    file_record = manifest.get("assignment_file")
    if not isinstance(file_record, Mapping):
        raise ValueError("cluster-state assignment inventory is invalid")
    if assignment_path.stat().st_size != int(file_record["size_bytes"]):
        raise ValueError("cluster-state assignment size drift")
    if file_sha256(assignment_path) != file_record["sha256"]:
        raise ValueError("cluster-state assignment hash drift")
    return manifest


def fit_sparse_cluster_config(
    *,
    evidence: PairEvidence,
    support: BasisSupport,
    config: Mapping[str, Any],
) -> tuple[SparseSpectralResult, NDArray[np.bool_]]:
    eligible = (
        (support.target_counts >= int(config["basis_min_target_count"]))
        & (support.response_counts >= int(config["basis_min_response_count"]))
        & (support.family_counts >= int(config["basis_min_family_count"]))
    )
    if bool(config["exclude_boundary_layers"]):
        eligible &= ~support.boundary_mask
    similarity = mean_similarity_matrix(
        evidence,
        min_pair_target_overlap=int(config["pair_min_target_overlap"]),
        min_pair_response_overlap=int(config["pair_min_response_overlap"]),
        min_pair_family_overlap=int(config["pair_min_family_overlap"]),
        eligible_mask=eligible,
    )
    affinity = knn_affinity(
        similarity,
        neighbors=int(config["neighbors"]),
        symmetrization=_knn_symmetrization(config["knn_symmetrization"]),
        minimum_affinity=float(config["minimum_affinity"]),
    )
    return (
        sparse_spectral_cluster(
            affinity,
            n_clusters=int(config["n_clusters"]),
            random_seed=int(config["random_seed"]),
            self_loop_weight=float(config["self_loop_weight"]),
            eigen_tolerance=float(config["eigen_tolerance"]),
        ),
        eligible,
    )


def fit_clustering_task(
    plan: Mapping[str, Any],
    *,
    repo_root: Path,
    task_index: int,
) -> dict[str, Any]:
    validated = validate_clustering_plan(
        plan,
        repo_root=repo_root,
        verify_code=True,
    )
    configurations = validated["configurations"]
    if task_index < 0 or task_index >= len(configurations):
        raise ValueError("clustering task index is out of range")
    config = configurations[task_index]
    output_path = (
        Path(str(validated["output_root"]))
        / "cluster-states"
        / f"task-{task_index:03d}-{config['config_sha256'][:12]}"
    )
    if output_path.exists():
        return _validate_existing_cluster_state(output_path)

    evidence_path = (
        Path(str(validated["output_root"])) / "pair-evidence" / str(config["weighting"])
    )
    evidence, support = load_pair_evidence(evidence_path)
    result, eligible = fit_sparse_cluster_config(
        evidence=evidence,
        support=support,
        config=config,
    )
    reader = FeatureStoreReader(Path(str(validated["feature_store"]["path"])))
    assignment_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for basis_index, basis in enumerate(reader.basis_rows):
        assigned = int(result.labels[basis_index]) >= 0
        if assigned:
            status = "assigned"
        elif support.boundary_mask[basis_index]:
            status = "unassigned_boundary"
        elif not eligible[basis_index]:
            status = "unassigned_insufficient_basis_support"
        else:
            status = "unassigned_no_recurring_positive_neighbor"
        status_counts[status] = status_counts.get(status, 0) + 1
        assignment_rows.append(
            {
                **dict(basis),
                "target_count": int(support.target_counts[basis_index]),
                "response_count": int(support.response_counts[basis_index]),
                "family_count": int(support.family_counts[basis_index]),
                "eligible": bool(eligible[basis_index]),
                "assigned": assigned,
                "cluster_id": (int(result.labels[basis_index]) if assigned else None),
                "assignment_status": status,
            }
        )

    temporary = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        assignment_path = temporary / "assignments.parquet"
        pq.write_table(
            pa.Table.from_pylist(assignment_rows, schema=ASSIGNMENT_SCHEMA),
            assignment_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        pair_manifest = load_json_object(evidence_path / "manifest.json")
        state: dict[str, Any] = {
            "schema_version": SPARSE_CLUSTER_SCHEMA,
            "plan_sha256": validated["plan_sha256"],
            "task_index": task_index,
            "config": dict(config),
            "source_pair_evidence": {
                "path": str(evidence_path),
                "manifest_sha256": pair_manifest["manifest_sha256"],
            },
            "source_feature_store": dict(validated["feature_store"]),
            "eligible_basis_count": int(eligible.sum()),
            "assigned_basis_count": int(result.active_mask.sum()),
            "unassigned_basis_count": int((~result.active_mask).sum()),
            "assignment_status_counts": dict(sorted(status_counts.items())),
            "cluster_sizes": {
                str(key): value for key, value in result.cluster_sizes.items()
            },
            "connected_component_count": result.connected_component_count,
            "eigenvalues": result.eigenvalues.tolist(),
            "normalization": "symmetric_degree_normalized_affinity",
            "prototypes_persisted": False,
            "descriptions_generated": False,
            "scientific_cluster_state": False,
            "code_revision": validated["code_revision"],
            "environment": validated["environment"],
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
            "assignment_file": {
                "path": assignment_path.name,
                "size_bytes": assignment_path.stat().st_size,
                "sha256": file_sha256(assignment_path),
                "row_count": len(assignment_rows),
            },
        }
        state["manifest_sha256"] = canonical_sha256(state)
        with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_path)
        return state
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
