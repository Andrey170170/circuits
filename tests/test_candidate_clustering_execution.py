from __future__ import annotations

import errno
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, load_npz, save_npz

from circuits.analysis.bonafide.candidate_clustering import (
    CLUSTER_COUNTS,
    RANDOM_SEEDS,
    GenerationClusterFit,
    ResolutionFit,
    SeedFit,
)
from circuits.analysis.bonafide.candidate_clustering_execution import (
    _SOURCE_BINDINGS,
    AFFINITY_FILES,
    ASSIGNMENTS_FILE,
    COMMON_ELIGIBILITY_FILE,
    collect_candidate_clustering_revision,
    load_candidate_clustering_baseline,
    run_candidate_clustering_baseline,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.clustering import SparseSpectralResult
from circuits.analysis.bonafide.clustering_evaluation import (
    cluster_size_metrics,
    sparse_graph_partition_metrics,
)


def _write_self_hashed_input(root: Path) -> dict[str, object]:
    root.mkdir(parents=True)
    manifest: dict[str, object] = {
        "schema_version": "synthetic.candidate-input.v1",
        "fixture": True,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _basis_rows(count: int) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "signed_basis_index": index,
            "model_id": "synthetic/model",
            "model_revision": "revision",
            "layer": index // 2,
            "neuron_index": index,
            "polarity": "positive" if index % 2 == 0 else "negative",
        }
        for index in range(count)
    )


def _resolution(view: str, count: int, affinity: csr_matrix) -> ResolutionFit:
    seeds: dict[int, SeedFit] = {}
    basis_count = affinity.shape[0]
    base_labels = np.arange(basis_count, dtype=np.int64) % count
    labels_by_seed = {
        17: base_labels,
        29: base_labels.copy(),
        43: (base_labels + 1) % count,
    }
    for seed in RANDOM_SEEDS:
        labels = labels_by_seed[seed]
        result = SparseSpectralResult(
            labels=labels,
            active_mask=np.ones(basis_count, dtype=np.bool_),
            eigenvalues=np.linspace(1.0, 0.1, count, dtype=np.float64),
            connected_component_count=1,
            cluster_sizes={
                int(cluster): int(size)
                for cluster, size in zip(
                    *np.unique(labels, return_counts=True), strict=True
                )
            },
        )
        seeds[seed] = SeedFit(
            seed=seed,
            result=result,
            valid=True,
            assignment_fraction=1.0,
            error=None,
        )
    return ResolutionFit(
        view=view,
        n_clusters=count,
        affinity=affinity,
        seeds=seeds,
        valid=True,
        medoid_seed=17,
        pairwise_seed_ari={(17, 29): 1.0, (17, 43): 1.0, (29, 43): 1.0},
        mean_seed_ari=1.0,
        minimum_seed_ari=1.0,
        size_metrics=cluster_size_metrics(base_labels),
        graph_metrics=sparse_graph_partition_metrics(base_labels, affinity),
    )


def _synthetic_pipeline(input_root: Path):
    input_manifest = _write_self_hashed_input(input_root)
    basis_count = 100
    basis_rows = _basis_rows(basis_count)
    bundle = SimpleNamespace(
        root=input_root.resolve(),
        manifest=input_manifest,
        basis_count=basis_count,
        basis_rows=basis_rows,
    )
    dense = np.zeros((basis_count, basis_count), dtype=np.float64)
    for index in range(basis_count):
        dense[index, (index + 1) % basis_count] = 1.0
        dense[(index + 1) % basis_count, index] = 1.0
    affinity = csr_matrix(dense)
    evidence = SimpleNamespace(
        common_eligible_mask=np.ones(basis_count, dtype=np.bool_),
        support_similarity=affinity,
    )
    directional = {
        view: {count: _resolution(view, count, affinity) for count in CLUSTER_COUNTS}
        for view in ("W", "C", "F")
    }
    fit = GenerationClusterFit(
        evidence=evidence,
        directional=directional,
        chosen_cluster_count=64,
        support=_resolution("S", 64, affinity),
    )
    return bundle, evidence, fit


def _patch_pipeline(monkeypatch, bundle, evidence, fit) -> None:
    revision = {
        "repo_root": "/synthetic/repo",
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "tracked_worktree_clean": True,
        "tracked_status_sha256": "c" * 64,
        "files": [
            {"role": role, "path": path, "sha256": "d" * 64}
            for role, path in _SOURCE_BINDINGS.items()
        ],
    }
    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_clustering_execution."
        "collect_candidate_clustering_revision",
        lambda repo_root: revision,
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_clustering_execution."
        "load_candidate_cluster_input_bundle",
        lambda root: bundle,
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_clustering_execution."
        "build_generation_evidence",
        lambda loaded: evidence,
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_clustering_execution."
        "fit_generation_grid",
        lambda built: fit,
    )


def _run_synthetic(tmp_path: Path, monkeypatch):
    bundle, evidence, fit = _synthetic_pipeline(tmp_path / "input")
    _patch_pipeline(monkeypatch, bundle, evidence, fit)
    output = tmp_path / "baseline"
    manifest = run_candidate_clustering_baseline(
        input_root=bundle.root,
        output_root=output,
        repo_root=tmp_path,
    )
    return output, manifest


def test_baseline_round_trip_persists_complete_label_free_state(
    tmp_path: Path, monkeypatch
) -> None:
    output, manifest = _run_synthetic(tmp_path, monkeypatch)

    assert manifest["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    assert manifest["chosen_cluster_count"] == 64
    assert manifest["common_eligible_basis_count"] == 100
    assert len(manifest["states"]) == 30
    assert len(manifest["resolution_diagnostics"]) == 10
    assert sum(state["is_medoid"] for state in manifest["states"]) == 10
    assert manifest["numerically_valid"] is True
    assert manifest["diagnostic_only"] is False
    assert manifest["cross_view_common_basis_ari"]["available"] is True
    assert len(manifest["cross_view_common_basis_ari"]["pairs"]) == 6
    assert manifest["outcomes_inspected"] is False
    assert manifest["descriptions_generated"] is False
    assert manifest["model_calls_made"] is False
    assert {path.name for path in output.iterdir()} == {
        *AFFINITY_FILES.values(),
        ASSIGNMENTS_FILE,
        COMMON_ELIGIBILITY_FILE,
        "manifest.json",
    }

    loaded = load_candidate_clustering_baseline(output)
    assert set(loaded.affinities) == {"W", "C", "F", "S"}
    assert loaded.assignments.num_rows == 30 * 100
    assert loaded.common_eligibility.num_rows == 100


def test_no_common_count_persists_explicit_diagnostic_only_state(
    tmp_path: Path, monkeypatch
) -> None:
    bundle, evidence, fit = _synthetic_pipeline(tmp_path / "input")
    invalid_candidate = {
        count: replace(
            resolution,
            seeds={
                seed: replace(
                    seed_fit,
                    result=replace(
                        seed_fit.result,
                        labels=np.zeros(bundle.basis_count, dtype=np.int64),
                        cluster_sizes={0: bundle.basis_count},
                    ),
                    valid=False,
                    error="assigned_cluster_count",
                )
                for seed, seed_fit in resolution.seeds.items()
            },
            valid=False,
            medoid_seed=None,
            pairwise_seed_ari={},
            mean_seed_ari=None,
            minimum_seed_ari=None,
            size_metrics=None,
            graph_metrics=None,
        )
        for count, resolution in fit.directional["C"].items()
    }
    diagnostic_fit = replace(
        fit,
        directional={**fit.directional, "C": invalid_candidate},
        chosen_cluster_count=None,
        support=None,
    )
    _patch_pipeline(monkeypatch, bundle, evidence, diagnostic_fit)
    output = tmp_path / "baseline"
    manifest = run_candidate_clustering_baseline(
        input_root=bundle.root,
        output_root=output,
        repo_root=tmp_path,
    )

    assert manifest["chosen_cluster_count"] is None
    assert manifest["numerically_valid"] is False
    assert manifest["diagnostic_only"] is True
    assert manifest["cross_view_common_basis_ari"] == {
        "available": False,
        "reason": "no_common_chosen_cluster_count",
        "pairs": [],
    }
    loaded = load_candidate_clustering_baseline(output)
    assert loaded.manifest["numerically_valid"] is False


def test_loader_rejects_file_and_csr_content_hash_drift(
    tmp_path: Path, monkeypatch
) -> None:
    output, _ = _run_synthetic(tmp_path, monkeypatch)
    assignment_path = output / ASSIGNMENTS_FILE
    content = bytearray(assignment_path.read_bytes())
    content[-1] ^= 1
    assignment_path.write_bytes(content)
    with pytest.raises(ValueError, match="file hash drift: assignments.parquet"):
        load_candidate_clustering_baseline(output)

    output2, _ = _run_synthetic(tmp_path / "second", monkeypatch)
    matrix_path = output2 / AFFINITY_FILES["W"]
    matrix = load_npz(matrix_path).tocsr()
    matrix.data *= 0.5
    save_npz(matrix_path, matrix, compressed=True)
    manifest_path = output2 / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        if record["path"] == AFFINITY_FILES["W"]:
            record["size_bytes"] = matrix_path.stat().st_size
            record["sha256"] = file_sha256(matrix_path)
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_sha256(core)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="W affinity content drift"):
        load_candidate_clustering_baseline(output2)

    output3, _ = _run_synthetic(tmp_path / "third", monkeypatch)
    manifest_path = output3 / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resolution_diagnostics"][0]["size_metrics"]["cluster_count"] = 999
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_sha256(core)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="resolution metrics drift"):
        load_candidate_clustering_baseline(output3)


def test_loader_full_validates_source_bundle(tmp_path: Path, monkeypatch) -> None:
    output, _ = _run_synthetic(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_clustering_execution."
        "load_candidate_cluster_input_bundle",
        lambda root: (_ for _ in ()).throw(ValueError("synthetic input file drift")),
    )
    with pytest.raises(ValueError, match="synthetic input file drift"):
        load_candidate_clustering_baseline(output)
    # Relocated/offline inspection remains available only when explicitly requested.
    assert load_candidate_clustering_baseline(output, verify_source=False).manifest[
        "numerically_valid"
    ]


def test_writer_never_overwrites_or_leaves_partial_state(
    tmp_path: Path, monkeypatch
) -> None:
    output, _ = _run_synthetic(tmp_path, monkeypatch)
    marker = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        run_candidate_clustering_baseline(
            input_root=tmp_path / "input",
            output_root=output,
            repo_root=tmp_path,
        )
    assert (output / "manifest.json").read_bytes() == marker

    failure_root = tmp_path / "failure-case"
    bundle, evidence, fit = _synthetic_pipeline(failure_root / "input")
    _patch_pipeline(monkeypatch, bundle, evidence, fit)
    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_clustering_execution.save_npz",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic")),
    )
    output_failure = failure_root / "baseline"
    with pytest.raises(RuntimeError, match="synthetic"):
        run_candidate_clustering_baseline(
            input_root=bundle.root,
            output_root=output_failure,
            repo_root=failure_root,
        )
    assert not output_failure.exists()
    assert not list(failure_root.glob(".baseline.tmp-*"))


def test_publish_race_does_not_replace_empty_destination(
    tmp_path: Path, monkeypatch
) -> None:
    bundle, evidence, fit = _synthetic_pipeline(tmp_path / "input")
    _patch_pipeline(monkeypatch, bundle, evidence, fit)
    output = tmp_path / "baseline"
    from circuits.analysis.bonafide import candidate_clustering_execution as execution

    original_write = execution._write_json

    def write_then_race(path, value) -> None:
        original_write(path, value)
        output.mkdir()

    monkeypatch.setattr(execution, "_write_json", write_then_race)
    monkeypatch.setattr(
        execution,
        "_rename_directory_no_replace",
        lambda source, destination: (_ for _ in ()).throw(
            OSError(errno.EINVAL, "synthetic unsupported rename")
        ),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        run_candidate_clustering_baseline(
            input_root=bundle.root,
            output_root=output,
            repo_root=tmp_path,
        )
    assert output.is_dir()
    assert not any(output.iterdir())
    assert not list(tmp_path.glob(".baseline.tmp-*"))


def test_manifest_last_fallback_succeeds_and_cleans_caught_failure(
    tmp_path: Path, monkeypatch
) -> None:
    from circuits.analysis.bonafide import candidate_clustering_execution as execution

    bundle, evidence, fit = _synthetic_pipeline(tmp_path / "success" / "input")
    _patch_pipeline(monkeypatch, bundle, evidence, fit)
    monkeypatch.setattr(
        execution,
        "_rename_directory_no_replace",
        lambda source, destination: (_ for _ in ()).throw(
            OSError(errno.EINVAL, "synthetic unsupported rename")
        ),
    )
    output = tmp_path / "success" / "baseline"
    run_candidate_clustering_baseline(
        input_root=bundle.root, output_root=output, repo_root=tmp_path
    )
    assert load_candidate_clustering_baseline(output).manifest["numerically_valid"]

    failure = tmp_path / "failure"
    bundle, evidence, fit = _synthetic_pipeline(failure / "input")
    _patch_pipeline(monkeypatch, bundle, evidence, fit)
    moved: list[str] = []
    original_move = execution._move_staged_file

    def fail_on_manifest(source: Path, destination: Path) -> None:
        moved.append(source.name)
        if source.name == "manifest.json":
            raise RuntimeError("synthetic manifest publication failure")
        original_move(source, destination)

    monkeypatch.setattr(execution, "_move_staged_file", fail_on_manifest)
    output = failure / "baseline"
    with pytest.raises(RuntimeError, match="manifest publication failure"):
        run_candidate_clustering_baseline(
            input_root=bundle.root, output_root=output, repo_root=failure
        )
    assert moved[-1] == "manifest.json"
    assert not output.exists()
    assert not list(failure.glob(".baseline.tmp-*"))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _synthetic_repo(root: Path, *, untracked_role: str | None = None) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Tests")
    for role, relative in _SOURCE_BINDINGS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{role}\n", encoding="utf-8")
        if role != untracked_role:
            _git(root, "add", "--", relative)
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-qm", "fixture")
    return root


def _patch_runtime_paths(monkeypatch, repo: Path) -> None:
    from circuits.analysis.bonafide import candidate_clustering_execution as execution

    monkeypatch.setattr(
        execution,
        "_runtime_source_paths",
        lambda: {
            role: repo / _SOURCE_BINDINGS[role]
            for role in (
                "canonical",
                "clustering",
                "clustering_store",
                "candidate_profiles",
                "candidate_clustering",
                "frozen_clustering_evaluation",
                "candidate_clustering_execution",
            )
        },
    )


def test_revision_binds_clean_full_tree_and_rejects_tracked_dirt(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _synthetic_repo(tmp_path / "repo")
    _patch_runtime_paths(monkeypatch, repo)
    revision = collect_candidate_clustering_revision(repo)
    assert revision["git_commit"] == _git(repo, "rev-parse", "HEAD")
    assert revision["git_tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    assert {record["role"] for record in revision["files"]} == set(_SOURCE_BINDINGS)

    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean full tracked worktree"):
        collect_candidate_clustering_revision(repo)


def test_revision_rejects_untracked_source_module(tmp_path: Path, monkeypatch) -> None:
    repo = _synthetic_repo(tmp_path / "repo", untracked_role="candidate_nulls")
    _patch_runtime_paths(monkeypatch, repo)
    with pytest.raises(ValueError, match="source is not tracked:.*candidate_nulls"):
        collect_candidate_clustering_revision(repo)


def test_revision_rejects_runtime_module_from_another_worktree(
    tmp_path: Path,
) -> None:
    repo = _synthetic_repo(tmp_path / "repo")
    with pytest.raises(ValueError, match="runtime source path mismatch"):
        collect_candidate_clustering_revision(repo)
