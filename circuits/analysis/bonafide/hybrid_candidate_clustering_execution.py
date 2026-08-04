"""Atomic persistence for exploratory hybrid candidate cluster states."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import load_npz, save_npz

from circuits.analysis.bonafide.candidate_clustering import (
    CLUSTER_COUNTS,
    MIN_BASIS_FAMILIES,
    MIN_BASIS_RESPONSES,
    MIN_BASIS_TARGETS,
    MIN_PAIR_FAMILIES,
    MIN_PAIR_RESPONSES,
    MIN_PAIR_TARGETS,
    RANDOM_SEEDS,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.clustering_evaluation import (
    cluster_size_metrics,
    sparse_graph_partition_metrics,
)
from circuits.analysis.bonafide.hybrid_candidate_clustering import (
    FUSION_ID,
    REPRESENTATION_IDS,
    HybridInvalidFit,
    HybridResolutionFit,
    fit_hybrid_grid,
)
from circuits.analysis.bonafide.hybrid_candidate_inputs import (
    HYBRID_INPUT_SCHEMA,
    HYBRID_PROTOCOL_PATH,
    _validate_code_revision,
    collect_hybrid_code_revision,
    load_hybrid_input_bundle,
)

HYBRID_CLUSTER_SCHEMA = "adag.bonafide.hybrid-candidate-clustering.v3"
ASSIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("representation", pa.string(), nullable=False),
        pa.field("affinity_mode", pa.string(), nullable=False),
        pa.field("n_clusters", pa.int32(), nullable=False),
        pa.field("seed", pa.int32(), nullable=False),
        pa.field("is_medoid", pa.bool_(), nullable=False),
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        pa.field("assigned", pa.bool_(), nullable=False),
        pa.field("cluster_id", pa.int32(), nullable=True),
    ]
)


def _fit_summary(fit: HybridResolutionFit | HybridInvalidFit) -> dict[str, Any]:
    if isinstance(fit, HybridInvalidFit):
        return {
            "representation": REPRESENTATION_IDS[fit.representation],
            "affinity_mode": fit.affinity_mode,
            "n_clusters": fit.n_clusters,
            "status": "invalid",
            "error_type": fit.error_type,
            "error_message": fit.error_message,
        }
    medoid_labels = fit.seeds[fit.medoid_seed].result.labels
    return {
        "representation": REPRESENTATION_IDS[fit.representation],
        "affinity_mode": fit.affinity_mode,
        "n_clusters": fit.n_clusters,
        "status": "valid",
        "medoid_seed": fit.medoid_seed,
        "mean_seed_ari": fit.mean_seed_ari,
        "minimum_seed_ari": fit.minimum_seed_ari,
        "pairwise_seed_ari": {
            f"{left}:{right}": value
            for (left, right), value in sorted(fit.pairwise_seed_ari.items())
        },
        "active_basis_count": int(
            next(iter(fit.seeds.values())).result.active_mask.sum()
        ),
        "connected_component_count": int(
            next(iter(fit.seeds.values())).result.connected_component_count
        ),
        "assignment_fraction_by_seed": {
            str(seed): seed_fit.assignment_fraction
            for seed, seed_fit in sorted(fit.seeds.items())
        },
        "cluster_size_metrics": cluster_size_metrics(medoid_labels),
        "partition_metrics": sparse_graph_partition_metrics(
            medoid_labels, fit.affinity
        ),
    }


def run_hybrid_candidate_clustering(
    *, input_root: Path, output_root: Path, repo_root: Path
) -> dict[str, Any]:
    """Fit all predeclared fresh states and publish one immutable artifact."""

    if output_root.exists():
        raise FileExistsError(
            f"hybrid clustering destination already exists: {output_root}"
        )
    code_revision = collect_hybrid_code_revision(repo_root)
    if code_revision["git_dirty"]:
        raise ValueError("refuse to publish hybrid clustering from dirty tracked source")
    bundle = load_hybrid_input_bundle(input_root)
    fits = fit_hybrid_grid(bundle)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        affinity_files: dict[tuple[str, str], str] = {}
        for representation in REPRESENTATION_IDS:
            for affinity_mode in ("full_positive", "knn32"):
                valid = [
                    fit
                    for (rep, mode, _), fit in fits.items()
                    if rep == representation
                    and mode == affinity_mode
                    and isinstance(fit, HybridResolutionFit)
                ]
                if not valid:
                    continue
                fit = valid[0]
                if any((candidate.affinity != fit.affinity).nnz for candidate in valid[1:]):
                    raise ValueError("hybrid affinity varies across cluster-count cells")
                name = f"affinity-{representation}-{affinity_mode}.npz"
                save_npz(temporary / name, fit.affinity, compressed=True)
                affinity_files[(representation, affinity_mode)] = name

        assignment_rows: list[dict[str, Any]] = []
        for key in sorted(fits):
            fit = fits[key]
            if isinstance(fit, HybridInvalidFit):
                continue
            for seed, seed_fit in sorted(fit.seeds.items()):
                for basis_index, label in enumerate(seed_fit.result.labels):
                    assignment_rows.append(
                        {
                            "representation": REPRESENTATION_IDS[fit.representation],
                            "affinity_mode": fit.affinity_mode,
                            "n_clusters": fit.n_clusters,
                            "seed": seed,
                            "is_medoid": seed == fit.medoid_seed,
                            "signed_basis_index": basis_index,
                            "assigned": int(label) >= 0,
                            "cluster_id": None if int(label) < 0 else int(label),
                        }
                    )
        pq.write_table(
            pa.Table.from_pylist(assignment_rows, schema=ASSIGNMENT_SCHEMA),
            temporary / "assignments.parquet",
            compression="zstd",
        )
        files = [
            {"path": path.name, "sha256": file_sha256(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        input_manifest = bundle.manifest
        protocol_hash = next(
            record["sha256"]
            for record in code_revision["files"]
            if record["path"] == HYBRID_PROTOCOL_PATH
        )
        fit_partition = input_manifest["fit_partition"]
        manifest: dict[str, Any] = {
            "schema_version": HYBRID_CLUSTER_SCHEMA,
            "input_binding": {
                "schema_version": HYBRID_INPUT_SCHEMA,
                "path": str(input_root.resolve()),
                "manifest_sha256": input_manifest["manifest_sha256"],
                "source_candidate_cluster_input": input_manifest[
                    "source_candidate_cluster_input"
                ],
                "artifact_payloads": input_manifest["artifact_payloads"],
                "fit_partition": fit_partition,
            },
            "code_revision": code_revision,
            "protocol": {"path": HYBRID_PROTOCOL_PATH, "sha256": protocol_hash},
            "method": {
                "representations": dict(REPRESENTATION_IDS),
                "primary_representation": REPRESENTATION_IDS["raw"],
                "fusion": FUSION_ID,
                "candidate_ordering": "target_local_explicit_model_rank_indices.v1",
                "hierarchical_weighting": "family_then_response_then_target.v1",
                "recurrence": {
                    "basis_min_targets": MIN_BASIS_TARGETS,
                    "basis_min_responses": MIN_BASIS_RESPONSES,
                    "basis_min_families": MIN_BASIS_FAMILIES,
                    "pair_min_targets": MIN_PAIR_TARGETS,
                    "pair_min_responses": MIN_PAIR_RESPONSES,
                    "pair_min_families": MIN_PAIR_FAMILIES,
                },
                "affinities": {
                    "primary": "full_positive",
                    "sensitivity": "knn32_union_max",
                },
                "cluster_counts": list(CLUSTER_COUNTS),
                "random_seeds": list(RANDOM_SEEDS),
                "cluster_source": "fresh_spectral_fit_no_width64_assignments.v1",
                "fit_partition": "generation_only.v1",
                "fitted_case_set_sha256": fit_partition["case_set_sha256"],
                "fitted_family_set_sha256": fit_partition["family_set_sha256"],
            },
            "fits": [_fit_summary(fits[key]) for key in sorted(fits)],
            "affinity_files": {
                f"{representation}:{mode}": name
                for (representation, mode), name in sorted(affinity_files.items())
            },
            "files": files,
            "counts": {
                "basis_count": len(bundle.basis_rows),
                "target_count": len(bundle.target_rows),
                "fit_count": len(fits),
                "valid_fit_count": sum(
                    isinstance(fit, HybridResolutionFit) for fit in fits.values()
                ),
                "invalid_fit_count": sum(
                    isinstance(fit, HybridInvalidFit) for fit in fits.values()
                ),
                "assignment_row_count": len(assignment_rows),
                "affinity_file_count": len(affinity_files),
            },
            "exploratory": True,
            "labeling_authorized": False,
            "confirmatory_holdout_opened": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_hybrid_clustering_manifest(
    root: Path, *, repo_root: Path | None = None
) -> Mapping[str, Any]:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("hybrid clustering manifest must be an object")
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if manifest.get(
        "schema_version"
    ) != HYBRID_CLUSTER_SCHEMA or recorded != canonical_sha256(core):
        raise ValueError("hybrid clustering manifest is invalid")
    if (
        manifest.get("exploratory") is not True
        or manifest.get("labeling_authorized") is not False
        or manifest.get("confirmatory_holdout_opened") is not False
    ):
        raise ValueError("hybrid clustering scientific status drift")
    _validate_code_revision(manifest.get("code_revision"))
    source_hashes = {
        record["path"]: record["sha256"]
        for record in manifest["code_revision"]["files"]
    }
    if manifest.get("protocol") != {
        "path": HYBRID_PROTOCOL_PATH,
        "sha256": source_hashes[HYBRID_PROTOCOL_PATH],
    }:
        raise ValueError("hybrid clustering protocol provenance drift")
    if repo_root is not None and collect_hybrid_code_revision(repo_root) != manifest.get(
        "code_revision"
    ):
        raise ValueError("hybrid clustering executable source revision drift")
    input_binding = manifest.get("input_binding")
    if not isinstance(input_binding, Mapping) or set(input_binding) != {
        "schema_version",
        "path",
        "manifest_sha256",
        "source_candidate_cluster_input",
        "artifact_payloads",
        "fit_partition",
    }:
        raise TypeError("hybrid clustering lacks input binding")
    bundle = load_hybrid_input_bundle(Path(str(input_binding["path"])))
    expected_binding = {
        "schema_version": HYBRID_INPUT_SCHEMA,
        "path": str(bundle.root),
        "manifest_sha256": bundle.manifest["manifest_sha256"],
        "source_candidate_cluster_input": bundle.manifest[
            "source_candidate_cluster_input"
        ],
        "artifact_payloads": bundle.manifest["artifact_payloads"],
        "fit_partition": bundle.manifest["fit_partition"],
    }
    if input_binding != expected_binding:
        raise ValueError("hybrid clustering input hash drift")
    method = manifest.get("method")
    if not isinstance(method, Mapping) or (
        method.get("representations") != dict(REPRESENTATION_IDS)
        or method.get("primary_representation") != REPRESENTATION_IDS["raw"]
        or method.get("fusion") != FUSION_ID
        or method.get("cluster_counts") != list(CLUSTER_COUNTS)
        or method.get("random_seeds") != list(RANDOM_SEEDS)
        or method.get("fit_partition") != "generation_only.v1"
        or method.get("fitted_case_set_sha256")
        != bundle.manifest["fit_partition"]["case_set_sha256"]
        or method.get("fitted_family_set_sha256")
        != bundle.manifest["fit_partition"]["family_set_sha256"]
    ):
        raise ValueError("hybrid clustering method contract drift")
    fits = manifest.get("fits")
    if not isinstance(fits, list):
        raise TypeError("hybrid clustering fit inventory is invalid")
    expected_fit_keys = {
        (representation_id, mode, count)
        for representation_id in REPRESENTATION_IDS.values()
        for mode in ("full_positive", "knn32")
        for count in CLUSTER_COUNTS
    }
    fit_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for fit in fits:
        if not isinstance(fit, Mapping):
            raise TypeError("hybrid clustering fit record is invalid")
        key = (
            str(fit.get("representation")),
            str(fit.get("affinity_mode")),
            int(fit.get("n_clusters", -1)),
        )
        if key in fit_by_key or fit.get("status") not in {"valid", "invalid"}:
            raise ValueError("hybrid clustering fit key/status is invalid")
        fit_by_key[key] = fit
    if set(fit_by_key) != expected_fit_keys:
        raise ValueError("hybrid clustering fit grid is incomplete")
    affinity_files = manifest.get("affinity_files")
    if not isinstance(affinity_files, Mapping):
        raise TypeError("hybrid clustering affinity inventory is invalid")
    expected_affinity_keys = {
        f"{representation}:{mode}"
        for representation in REPRESENTATION_IDS
        for mode in ("full_positive", "knn32")
        if any(
            fit["status"] == "valid"
            for (representation_id, fit_mode, _), fit in fit_by_key.items()
            if representation_id == REPRESENTATION_IDS[representation]
            and fit_mode == mode
        )
    }
    if set(affinity_files) != expected_affinity_keys:
        raise ValueError("hybrid clustering affinity grid is incomplete")
    expected_names = {"assignments.parquet", *map(str, affinity_files.values())}
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TypeError("hybrid clustering file inventory is invalid")
    by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise TypeError("hybrid clustering file record is invalid")
        name = record.get("path")
        if not isinstance(name, str) or Path(name).name != name or name in by_name:
            raise ValueError("hybrid clustering file path is unsafe or duplicate")
        by_name[name] = record
    if set(by_name) != expected_names:
        raise ValueError("hybrid clustering file inventory is incomplete")
    for name, record in by_name.items():
        path = root / name
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise ValueError(f"hybrid clustering file hash mismatch: {name}")
    basis_count = len(bundle.basis_rows)
    for name in affinity_files.values():
        affinity = load_npz(root / str(name)).tocsr()
        if (
            affinity.shape != (basis_count, basis_count)
            or not np.isfinite(affinity.data).all()
            or (affinity != affinity.T).nnz
        ):
            raise ValueError("hybrid clustering affinity content is invalid")
    assignment_table = pq.read_table(root / "assignments.parquet")
    if not assignment_table.schema.equals(ASSIGNMENT_SCHEMA, check_metadata=False):
        raise ValueError("hybrid clustering assignment schema drift")
    assignment_rows = assignment_table.to_pylist()
    seen: set[tuple[str, str, int, int, int]] = set()
    for row in assignment_rows:
        fit_key = (
            row["representation"],
            row["affinity_mode"],
            row["n_clusters"],
        )
        if fit_by_key.get(fit_key, {}).get("status") != "valid":
            raise ValueError("hybrid assignment references an invalid fit")
        key = (*fit_key, row["seed"], row["signed_basis_index"])
        if (
            key in seen
            or row["seed"] not in RANDOM_SEEDS
            or not 0 <= row["signed_basis_index"] < basis_count
            or row["assigned"] != (row["cluster_id"] is not None)
            or (
                row["cluster_id"] is not None
                and not 0 <= row["cluster_id"] < row["n_clusters"]
            )
        ):
            raise ValueError("hybrid assignment row is invalid or duplicate")
        seen.add(key)
    expected_assignment_count = (
        sum(fit["status"] == "valid" for fit in fits)
        * len(RANDOM_SEEDS)
        * basis_count
    )
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or (
        len(assignment_rows) != expected_assignment_count
        or counts.get("assignment_row_count") != len(assignment_rows)
        or counts.get("fit_count") != len(expected_fit_keys)
        or counts.get("valid_fit_count")
        != sum(fit["status"] == "valid" for fit in fits)
        or counts.get("invalid_fit_count")
        != sum(fit["status"] == "invalid" for fit in fits)
        or counts.get("basis_count") != basis_count
        or counts.get("target_count") != len(bundle.target_rows)
        or counts.get("affinity_file_count") != len(affinity_files)
    ):
        raise ValueError("hybrid clustering count summary drift")
    return manifest
