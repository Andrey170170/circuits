"""Focused fixtures for the unified candidate/multiplex assessment artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import circuits.analysis.bonafide.candidate_multiplex_assessment as assessment_module
from circuits.analysis.bonafide.candidate_clustering_execution import (
    ASSIGNMENT_SCHEMA,
    ASSIGNMENTS_FILE,
    CANDIDATE_CLUSTER_BASELINE_SCHEMA,
    COMMON_ELIGIBILITY_FILE,
    COMMON_ELIGIBILITY_SCHEMA,
)
from circuits.analysis.bonafide.candidate_multiplex_assessment import (
    CANDIDATE_MEASUREMENT_SCOPE,
    OCCURRENCE_PROJECTION_FILE,
    TARGET_BASIS_FILE,
    TARGET_CROSSWALK_FILE,
    build_candidate_multiplex_assessment,
    load_candidate_multiplex_assessment,
)
from circuits.analysis.bonafide.candidate_profiles import (
    BASIS_INDEX_SCHEMA as C2_BASIS_SCHEMA,
)
from circuits.analysis.bonafide.candidate_profiles import (
    CANDIDATE_CLUSTER_INPUT_SCHEMA,
    CANDIDATE_PROFILE_SCHEMA,
    WIDTH_PROFILE_SCHEMA,
)
from circuits.analysis.bonafide.candidate_profiles import (
    TARGET_SCHEMA as C2_TARGET_SCHEMA,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.compaction import (
    BASIS_INDEX_SCHEMA as DENSE_BASIS_SCHEMA,
)
from circuits.analysis.bonafide.compaction import (
    CIRCUIT_INPUT_INDEX_SCHEMA,
    COMPACTED_MULTIPLEX_SCHEMA,
    OCCURRENCE_INDEX_SCHEMA,
)
from circuits.analysis.bonafide.streaming import TARGET_SCHEMA as DENSE_TARGET_SCHEMA


@pytest.fixture(autouse=True)
def _fixture_provenance_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    revision = {
        "repo_root": "/fixture/repo",
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
        "tracked_worktree_clean": True,
        "tracked_status_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "files": [
            {"path": path, "git_blob": "3" * 40, "sha256": "4" * 64}
            for path in assessment_module._PRODUCING_SOURCE_PATHS
        ],
    }
    monkeypatch.setattr(
        assessment_module,
        "_collect_producing_revision",
        lambda _repo_root: revision,
    )
    monkeypatch.setattr(
        assessment_module, "_validate_producing_revision", lambda _revision: None
    )
    monkeypatch.setattr(
        assessment_module, "_deep_validate_source_artifacts", lambda **_kwargs: None
    )


def _write_table(path: Path, schema: pa.Schema, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _file_record(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if path.suffix == ".parquet":
        result["row_count"] = pq.read_metadata(path).num_rows
    return result


def _write_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(manifest)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
    return manifest


def _rewrite_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    core = dict(manifest)
    core.pop("manifest_sha256", None)
    return _write_manifest(root, core)


def _basis(index: int, polarity: str) -> dict[str, Any]:
    return {
        "signed_basis_index": index,
        "model_id": "model",
        "model_revision": "revision",
        "layer": 7,
        "neuron_index": 19,
        "polarity": polarity,
        "in_width_support": True,
        "in_width_view": True,
        "in_candidate_support": True,
        "in_candidate_view": True,
        "width_support_target_count": 1,
        "width_target_count": 1,
        "candidate_support_target_count": 1,
        "candidate_target_count": 1,
        "width_support_generation_target_count": 1,
        "width_generation_target_count": 1,
        "candidate_support_generation_target_count": 1,
        "candidate_generation_target_count": 1,
        "width_support_generation_response_count": 1,
        "width_generation_response_count": 1,
        "candidate_support_generation_response_count": 1,
        "candidate_generation_response_count": 1,
        "width_support_generation_family_count": 1,
        "width_generation_family_count": 1,
        "candidate_support_generation_family_count": 1,
        "candidate_generation_family_count": 1,
    }


def _target(case_id: str, suffix: str, *, partition: str, phase: int) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "source_width1_artifact_id": f"source-{suffix}",
        "width1_artifact_id": f"trace-{suffix}",
        "width1_payload_sha256": suffix * 64,
        "candidate_union_artifact_id": f"candidate-{suffix}",
        "candidate_union_payload_sha256": ("c" if suffix != "c" else "d") * 64,
        "candidate_union_topology_sha256": ("e" if suffix != "e" else "f") * 64,
        "base_question_id": f"family-{suffix}",
        "response_id": f"response-{suffix}",
        "phase_bin": phase,
        "response_position": phase + 3,
        "family_partition": partition,
        "partition_hierarchical_weight": 0.5,
        "candidate_count": 5,
        "observed_token_id": 101,
        "observed_token_text": " token",
        "candidate_selection_json": "{}",
        "example_json": "{}",
        "width_signed_basis_count": 1,
        "candidate_signed_basis_count": 1,
        "zero_activation_width_occurrence_count": 0,
        "zero_activation_candidate_occurrence_count": 0,
        "candidate_activation_invariance_max_abs_deviation": 0.0,
        "candidate_activation_invariance_max_relative_deviation": 0.0,
        "candidate_activation_invariance_violation_count": 0,
        "candidate_activation_invariance_comparison_count": 1,
        "width_polarity_crosswalk_json": "{}",
        "candidate_polarity_crosswalk_json": "{}",
    }


def _profile_identity(basis: dict[str, Any]) -> dict[str, Any]:
    return {
        field: basis[field]
        for field in ("model_id", "model_revision", "layer", "neuron_index", "polarity")
    }


def _source_record(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(root.resolve()),
        "manifest_path": str((root / "manifest.json").resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": file_sha256(root / "manifest.json"),
        "schema_version": manifest["schema_version"],
    }


def _make_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_root = tmp_path / "inputs"
    baseline_root = tmp_path / "baseline"
    dense_root = tmp_path / "dense"
    input_root.mkdir()
    baseline_root.mkdir()
    dense_root.mkdir()

    positive = _basis(0, "+")
    negative = _basis(1, "-")
    targets = [
        _target("case-a", "a", partition="generation", phase=0),
        _target("case-b", "b", partition="audit", phase=1),
    ]
    _write_table(
        input_root / "basis-index.parquet", C2_BASIS_SCHEMA, [positive, negative]
    )
    _write_table(input_root / "targets.parquet", C2_TARGET_SCHEMA, targets)
    width_rows = [
        {
            "case_id": "case-a",
            "signed_basis_index": 0,
            **_profile_identity(positive),
            "attribution_profile": [1.0, None],
            "attribution_support": [True, False],
            "signed_attribution": 1.0,
            "occurrence_count": 2,
        },
        {
            "case_id": "case-b",
            "signed_basis_index": 1,
            **_profile_identity(negative),
            "attribution_profile": [0.5],
            "attribution_support": [True],
            "signed_attribution": 0.5,
            "occurrence_count": 1,
        },
    ]
    _write_table(
        input_root / "width-profiles.parquet", WIDTH_PROFILE_SCHEMA, width_rows
    )
    candidate_rows = []
    for case_id, basis, values in (
        ("case-a", positive, [1.0, 0.0, 0.0, 0.0, 0.0]),
        ("case-a", negative, [0.0, 2.0, 0.0, 0.0, 0.0]),
        ("case-b", positive, [0.0, 0.0, 3.0, 0.0, 0.0]),
    ):
        candidate_rows.append(
            {
                "case_id": case_id,
                "signed_basis_index": basis["signed_basis_index"],
                **_profile_identity(basis),
                "candidate_contrast_profile": values,
                "candidate_profile_l2_norm": math.sqrt(
                    sum(value**2 for value in values)
                ),
                "occurrence_count": 1,
            }
        )
    _write_table(
        input_root / "candidate-profiles.parquet",
        CANDIDATE_PROFILE_SCHEMA,
        candidate_rows,
    )
    partitions = input_root / "family-partitions.json"
    partitions.write_text("{}\n", encoding="utf-8")
    input_files = sorted(
        path for path in input_root.iterdir() if path.name != "manifest.json"
    )
    input_manifest = _write_manifest(
        input_root,
        {
            "schema_version": CANDIDATE_CLUSTER_INPUT_SCHEMA,
            "cohort": {"target_count": 2},
            "outcomes_inspected": False,
            "model_calls_made": False,
            "cluster_fit_performed": False,
            "confirmatory_holdout_opened": False,
            "files": [_file_record(path) for path in input_files],
        },
    )

    assignment_rows = []
    common_rows = []
    for basis, cluster in ((positive, 4), (negative, 9)):
        identity = {
            "signed_basis_index": basis["signed_basis_index"],
            **_profile_identity(basis),
        }
        assignment_rows.append(
            {
                "state_index": 0,
                "view": "W",
                "n_clusters": 64,
                "seed": 17,
                "fit_valid": True,
                "seed_valid": True,
                "is_medoid": True,
                "assignment_fraction": 1.0,
                "fit_error": None,
                **identity,
                "eligible": True,
                "assigned": True,
                "cluster_id": cluster,
                "assignment_status": "assigned",
            }
        )
        common_rows.append({**identity, "common_eligible": True})
    _write_table(baseline_root / ASSIGNMENTS_FILE, ASSIGNMENT_SCHEMA, assignment_rows)
    _write_table(
        baseline_root / COMMON_ELIGIBILITY_FILE,
        COMMON_ELIGIBILITY_SCHEMA,
        common_rows,
    )
    baseline_manifest = _write_manifest(
        baseline_root,
        {
            "schema_version": CANDIDATE_CLUSTER_BASELINE_SCHEMA,
            "source_input_bundle": _source_record(input_root, input_manifest),
            "chosen_cluster_count": 64,
            "configuration": {
                "views": ["W", "C", "F", "S"],
                "directional_cluster_counts": [32, 64, 96],
            },
            "states": [
                {
                    "state_index": 0,
                    "view": "W",
                    "n_clusters": 64,
                    "seed": 17,
                    "fit_valid": True,
                    "seed_valid": True,
                    "is_medoid": True,
                    "assignment_fraction": 1.0,
                    "fit_error": None,
                }
            ],
            "outcomes_inspected": False,
            "descriptions_generated": False,
            "model_calls_made": False,
            "confirmatory_holdout_opened": False,
            "files": [
                _file_record(baseline_root / ASSIGNMENTS_FILE),
                _file_record(baseline_root / COMMON_ELIGIBILITY_FILE),
            ],
        },
    )
    assert baseline_manifest

    dense_basis = [
        {
            "signed_basis_index": basis["signed_basis_index"],
            **_profile_identity(basis),
        }
        for basis in (positive, negative)
    ]
    _write_table(dense_root / "basis-index.parquet", DENSE_BASIS_SCHEMA, dense_basis)
    _write_table(
        dense_root / "circuit-input-index.parquet",
        CIRCUIT_INPUT_INDEX_SCHEMA,
        [
            {
                "global_atlas_ci_index": 0,
                "atlas_trace_index": 2,
                "trace_unit_id": "trace-a",
                "local_ci_index": 0,
                "local_label": "response-a",
            }
        ],
    )
    dense_target = {
        # Atlas trace indices retain the source inventory identity and may have
        # gaps when traces are excluded; they are ordered and unique, not dense.
        "atlas_trace_index": 2,
        "trace_unit_id": "trace-a",
        "source_artifact_id": "source-a",
        "response_id": "response-a",
        "base_question_id": "family-a",
        "response_position": 3,
        "prediction_position": 3,
        "target_token_id": 101,
        "target_token_text": " token",
        "target_logit": 1.0,
        "target_probability": 0.5,
        "corpus_role": "dense_discovery",
        "cluster_fit_eligible": True,
        "fit_weight": 1.0,
        "artifact_manifest_sha256": "f" * 64,
        "artifact_payload_sha256": "a" * 64,
        "condition_json": "{}",
        "selection_reasons_json": "[]",
        "local_ci_count": 1,
        "local_labels": ["response-a"],
    }
    _write_table(
        dense_root / "target-index.parquet", DENSE_TARGET_SCHEMA, [dense_target]
    )
    occurrences = [
        {
            "occurrence_index": index,
            "atlas_trace_index": 2,
            "trace_unit_id": "trace-a",
            "token_position": index + 10,
            "signed_basis_index": 0,
        }
        for index in range(2)
    ]
    _write_table(
        dense_root / "occurrence-index.parquet", OCCURRENCE_INDEX_SCHEMA, occurrences
    )
    dense_files = sorted(
        path for path in dense_root.iterdir() if path.name != "manifest.json"
    )
    _write_manifest(
        dense_root,
        {
            "schema_version": COMPACTED_MULTIPLEX_SCHEMA,
            "target_count": 1,
            "files": [_file_record(path) for path in dense_files],
        },
    )
    return input_root, baseline_root, dense_root


def test_build_retains_partial_crosswalk_union_polarity_and_scope(
    tmp_path: Path,
) -> None:
    input_root, baseline_root, dense_root = _make_sources(tmp_path)
    output = tmp_path / "assessment"
    manifest = build_candidate_multiplex_assessment(
        c2_input_root=input_root,
        c2_baseline_root=baseline_root,
        dense_multiplex_root=dense_root,
        output_root=output,
    )
    loaded = load_candidate_multiplex_assessment(output)

    crosswalk = loaded.target_crosswalk.to_pylist()
    assert len(crosswalk) == 2
    assert [row["dense_target_match"] for row in crosswalk] == [True, False]
    assert manifest["overlap"]["dense_matched_target_count"] == 1
    assert manifest["overlap"]["dense_unmatched_target_count"] == 1

    rows = loaded.target_basis_assessment.to_pylist()
    assert [(row["case_id"], row["polarity"]) for row in rows] == [
        ("case-a", "+"),
        ("case-a", "-"),
        ("case-b", "+"),
        ("case-b", "-"),
    ]
    by_key = {(row["case_id"], row["polarity"]): row for row in rows}
    assert by_key[("case-a", "+")]["c2_w64_cluster_id"] == 4
    assert by_key[("case-a", "-")]["c2_w64_cluster_id"] == 9
    assert by_key[("case-a", "+")]["dense_occurrence_count"] == 2
    assert "width_profile_missing" in by_key[("case-a", "-")]["missing_reasons"]
    assert "candidate_profile_missing" in by_key[("case-b", "-")]["missing_reasons"]
    assert "dense_target_unmatched" in by_key[("case-b", "+")]["missing_reasons"]

    projection = loaded.occurrence_projection
    assert projection is not None and projection.num_rows == 2
    assert projection.column("candidate_measurement_scope").to_pylist() == [
        CANDIDATE_MEASUREMENT_SCOPE,
        CANDIDATE_MEASUREMENT_SCOPE,
    ]
    assert "candidate_contrast_vector" not in projection.schema.names
    metrics = manifest["coverage_metrics"]
    assert metrics["overall"]["target_count"] == 2
    assert metrics["overall"]["occurrence_projection_row_count"] == 2
    assert (
        metrics["by_family_partition"]["generation"]["dense_matched_target_count"] == 1
    )
    assert metrics["by_c2_phase_bin"]["1"]["dense_matched_target_count"] == 0


def test_target_identity_conflict_fails_closed(tmp_path: Path) -> None:
    input_root, baseline_root, dense_root = _make_sources(tmp_path)
    target_path = dense_root / "target-index.parquet"
    rows = pq.read_table(target_path).to_pylist()
    rows[0]["response_id"] = "conflicting-response"
    _write_table(target_path, DENSE_TARGET_SCHEMA, rows)
    manifest = json.loads((dense_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = [
        _file_record(dense_root / record["path"]) for record in manifest["files"]
    ]
    manifest.pop("manifest_sha256")
    _write_manifest(dense_root, manifest)

    with pytest.raises(ValueError, match="identity conflict"):
        build_candidate_multiplex_assessment(
            c2_input_root=input_root,
            c2_baseline_root=baseline_root,
            dense_multiplex_root=dense_root,
            output_root=tmp_path / "assessment",
        )


def test_output_is_no_overwrite_and_tamper_evident(tmp_path: Path) -> None:
    input_root, baseline_root, dense_root = _make_sources(tmp_path)
    output = tmp_path / "assessment"
    arguments = {
        "c2_input_root": input_root,
        "c2_baseline_root": baseline_root,
        "dense_multiplex_root": dense_root,
        "output_root": output,
    }
    build_candidate_multiplex_assessment(**arguments)
    with pytest.raises(FileExistsError, match="already exists"):
        build_candidate_multiplex_assessment(**arguments)

    target_basis_path = output / TARGET_BASIS_FILE
    with target_basis_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="size drift"):
        load_candidate_multiplex_assessment(output)

    assert (output / TARGET_CROSSWALK_FILE).is_file()
    assert (output / OCCURRENCE_PROJECTION_FILE).is_file()


def test_invalid_w64_medoid_state_fails_closed(tmp_path: Path) -> None:
    input_root, baseline_root, dense_root = _make_sources(tmp_path)
    manifest = json.loads((baseline_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["states"][0]["seed_valid"] = False
    _rewrite_manifest(baseline_root, manifest)
    with pytest.raises(ValueError, match="fit/seed"):
        build_candidate_multiplex_assessment(
            c2_input_root=input_root,
            c2_baseline_root=baseline_root,
            dense_multiplex_root=dense_root,
            output_root=tmp_path / "assessment",
        )


def test_coherent_rehash_cluster_mutation_fails_source_rederivation(
    tmp_path: Path,
) -> None:
    input_root, baseline_root, dense_root = _make_sources(tmp_path)
    output = tmp_path / "assessment"
    build_candidate_multiplex_assessment(
        c2_input_root=input_root,
        c2_baseline_root=baseline_root,
        dense_multiplex_root=dense_root,
        output_root=output,
    )

    target_basis_path = output / TARGET_BASIS_FILE
    rows = pq.read_table(target_basis_path).to_pylist()
    rows[0]["c2_w64_cluster_id"] = 63
    _write_table(
        target_basis_path,
        assessment_module.TARGET_BASIS_ASSESSMENT_SCHEMA,
        rows,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = [
        _file_record(output / record["path"]) for record in manifest["files"]
    ]
    _rewrite_manifest(output, manifest)

    with pytest.raises(ValueError, match="bound source derivation"):
        load_candidate_multiplex_assessment(output, verify_sources=True)
