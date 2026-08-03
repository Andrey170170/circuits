from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from circuits.analysis.bonafide import (
    candidate_labelability_evaluation as evaluation_module,
)
from circuits.analysis.bonafide.candidate_clustering import (
    CandidateClusterInputBundle,
)
from circuits.analysis.bonafide.candidate_clustering_execution import (
    ASSIGNMENT_SCHEMA,
    LoadedCandidateClusteringBaseline,
)
from circuits.analysis.bonafide.candidate_labelability_evaluation import (
    CANDIDATE_LABELABILITY_EVALUATION_SCHEMA,
    evaluate_loaded_candidate_labelability,
    extract_chosen_medoid_assignments,
    load_candidate_labelability_evaluation,
    normalized_profile_records,
    validate_candidate_labelability_runtime_paths,
)
from circuits.analysis.bonafide.candidate_profiles import (
    CANDIDATE_PROFILE_SCHEMA,
    WIDTH_PROFILE_SCHEMA,
)
from circuits.analysis.bonafide.canonical import canonical_sha256
from scipy.sparse import csr_matrix

STATES = ("W", "C", "F", "S")


def _family_partitions() -> dict[str, object]:
    partitions = {
        "generation": [f"generation-{index:02d}" for index in range(18)],
        "selection_scoring": [f"selection-{index:02d}" for index in range(8)],
        "audit": [f"audit-{index:02d}" for index in range(8)],
    }
    return {
        "partitions": partitions,
        "family_to_partition": {
            family: partition
            for partition, families in partitions.items()
            for family in families
        },
    }


def _basis_row(index: int) -> dict[str, object]:
    return {
        "signed_basis_index": index,
        "model_id": "synthetic/model",
        "model_revision": "revision",
        "layer": index,
        "neuron_index": 100 + index,
        "polarity": "positive",
    }


def _profile_identity(case_id: str, index: int) -> dict[str, object]:
    return {
        "case_id": case_id,
        **_basis_row(index),
    }


def _write_profiles(root: Path, families: dict[str, object]) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    width_rows: list[dict[str, object]] = []
    for partition, family_ids in families["partitions"].items():
        for family in family_ids:
            case_id = f"target-{family}"
            targets.append(
                {
                    "case_id": case_id,
                    "family_partition": partition,
                    "base_question_id": family,
                    "response_id": f"response-{family}",
                }
            )
            vectors = ([1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9])
            profiles = ([1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9])
            for index in range(4):
                vector = [*vectors[index], 0.0, 0.0, 0.0]
                candidate_rows.append(
                    {
                        **_profile_identity(case_id, index),
                        "candidate_contrast_profile": vector,
                        "candidate_profile_l2_norm": float(np.linalg.norm(vector)),
                        "occurrence_count": 1,
                    }
                )
                width_rows.append(
                    {
                        **_profile_identity(case_id, index),
                        "attribution_profile": profiles[index],
                        "attribution_support": [True, True],
                        "signed_attribution": 1.0,
                        "occurrence_count": 1,
                    }
                )
    pq.write_table(
        pa.Table.from_pylist(candidate_rows, schema=CANDIDATE_PROFILE_SCHEMA),
        root / "candidate-profiles.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(width_rows, schema=WIDTH_PROFILE_SCHEMA),
        root / "width-profiles.parquet",
    )
    return targets


def _resolution(view: str) -> dict[str, object]:
    return {
        "view": view,
        "n_clusters": 2,
        "valid": True,
        "medoid_seed": 17,
        "mean_seed_ari": 0.90,
        "minimum_seed_ari": 0.85,
        "size_metrics": {"maximum_cluster_fraction": 0.50},
        "graph_metrics": {
            "modularity": 0.30,
            "internal_affinity_enrichment": 1.50,
        },
        "seeds": [
            {"seed": 17, "assignment_fraction": 1.0},
            {"seed": 29, "assignment_fraction": 1.0},
            {"seed": 43, "assignment_fraction": 1.0},
        ],
    }


def _baseline(
    input_hash: str, *, duplicate_medoid: bool = False
) -> LoadedCandidateClusteringBaseline:
    rows: list[dict[str, object]] = []
    state_index = 0
    for view in STATES:
        seeds = (17, 29) if duplicate_medoid and view == "W" else (17,)
        for seed in seeds:
            rows.extend(
                {
                    "state_index": state_index,
                    "view": view,
                    "n_clusters": 2,
                    "seed": seed,
                    "fit_valid": True,
                    "seed_valid": True,
                    "is_medoid": True,
                    "assignment_fraction": 1.0,
                    "fit_error": None,
                    **_basis_row(basis_index),
                    "eligible": True,
                    "assigned": True,
                    "cluster_id": 0 if basis_index < 2 else 1,
                    "assignment_status": "assigned",
                }
                for basis_index in range(4)
            )
            state_index += 1
    return LoadedCandidateClusteringBaseline(
        root=Path("/baseline"),
        manifest={
            "manifest_sha256": "b" * 64,
            "schema_version": "baseline",
            "numerically_valid": True,
            "chosen_cluster_count": 2,
            "basis_count": 4,
            "source_input_bundle": {"manifest_sha256": input_hash},
            "resolution_diagnostics": [_resolution(state) for state in STATES],
        },
        affinities={state: csr_matrix((4, 4)) for state in STATES},
        assignments=pa.Table.from_pylist(rows, schema=ASSIGNMENT_SCHEMA),
        common_eligibility=pa.table({"common_eligible": [True] * 4}),
    )


@pytest.fixture
def synthetic_artifacts(
    tmp_path: Path,
) -> tuple[CandidateClusterInputBundle, LoadedCandidateClusteringBaseline]:
    families = _family_partitions()
    targets = _write_profiles(tmp_path, families)
    input_hash = "a" * 64
    bundle = CandidateClusterInputBundle(
        root=tmp_path,
        manifest={"manifest_sha256": input_hash, "schema_version": "inputs"},
        basis_count=4,
        basis_rows=tuple(_basis_row(index) for index in range(4)),
        target_rows=tuple(targets),
        family_partitions=families,
        generation_case_ids=tuple(
            row["case_id"] for row in targets if row["family_partition"] == "generation"
        ),
        width_blocks=(),
        candidate_blocks=(),
        candidate_support_blocks=(),
    )
    return bundle, _baseline(input_hash)


def test_extracts_one_complete_chosen_medoid(
    synthetic_artifacts: tuple[
        CandidateClusterInputBundle, LoadedCandidateClusteringBaseline
    ],
) -> None:
    bundle, baseline = synthetic_artifacts
    result = extract_chosen_medoid_assignments(baseline, basis_count=bundle.basis_count)
    assert set(result) == set(STATES)
    np.testing.assert_array_equal(result["F"], [0, 0, 1, 1])


def test_assignment_extraction_rejects_duplicate_medoid() -> None:
    baseline = _baseline("a" * 64, duplicate_medoid=True)
    with pytest.raises(ValueError, match="complete chosen medoid"):
        extract_chosen_medoid_assignments(baseline, basis_count=4)


def test_normalized_records_keep_explicit_identity_and_boolean_support(
    synthetic_artifacts: tuple[
        CandidateClusterInputBundle, LoadedCandidateClusteringBaseline
    ],
) -> None:
    bundle, _ = synthetic_artifacts
    candidate, width = normalized_profile_records(bundle)
    assert len(candidate) == len(width) == 34 * 4
    assert set(candidate[0]) == {
        "partition",
        "family_id",
        "response_id",
        "target_id",
        "basis_index",
        "vector",
    }
    assert all(type(value) is bool for value in width[0]["support"])


def test_full_adapter_is_label_free_and_derives_only_preliminary_gates(
    synthetic_artifacts: tuple[
        CandidateClusterInputBundle, LoadedCandidateClusteringBaseline
    ],
) -> None:
    bundle, baseline = synthetic_artifacts
    report = evaluate_loaded_candidate_labelability(bundle, baseline)
    assert report["frozen_family_ids_by_partition"].keys() == {
        "generation",
        "selection_scoring",
        "audit",
    }
    assert report["generation_centroids"]["W"]["family_count"] == 18
    assert (
        report["paired_family_bootstraps"]["audit"]["C_minus_W"]["replicates"] == 10_000
    )
    readiness = report["cluster_labeling_readiness"]["W"]
    assert readiness["support_policy"] == "candidate_and_width_frozen_thresholds"
    assert "candidate_support" in readiness
    assert "width_support" in readiness
    gates = report["pre_null_pre_jackknife_gates"]
    assert gates["final_pass_claimed"] is False
    assert gates["status"] == "pending_direction_null_and_generation_family_jackknife"
    assert gates["functional"]["C"]["final_functional_gate"] is None
    assert gates["structural"]["W"]["final_structural_gate"] is None


def test_adapter_rejects_partition_family_leakage(
    synthetic_artifacts: tuple[
        CandidateClusterInputBundle, LoadedCandidateClusteringBaseline
    ],
) -> None:
    bundle, baseline = synthetic_artifacts
    partitions = dict(bundle.family_partitions)
    partition_ids = {
        key: list(value) for key, value in partitions["partitions"].items()
    }
    partition_ids["selection_scoring"][0] = "not-in-profile-data"
    leaked = CandidateClusterInputBundle(
        **{
            **bundle.__dict__,
            "family_partitions": {**partitions, "partitions": partition_ids},
        }
    )
    with pytest.raises(ValueError, match="profile family coverage"):
        evaluate_loaded_candidate_labelability(leaked, baseline)


def _minimal_persisted_report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": CANDIDATE_LABELABILITY_EVALUATION_SCHEMA,
        "purpose": "label_free_pre_null_pre_jackknife_candidate_labelability_evaluation",
        "source_input_bundle": {},
        "source_clustering_baseline": {},
        "code_revision": {},
        "firewall": {
            "outcomes_inspected": False,
            "labels_inspected": False,
            "descriptions_generated": False,
            "model_calls_made": False,
            "confirmatory_holdout_opened": False,
            "generation_only_centroids": True,
            "selection_and_audit_fit_influence": False,
            "final_pass_claimed": False,
        },
        "evaluation": {
            "pre_null_pre_jackknife_gates": {
                "final_pass_claimed": False,
                "status": "pending_direction_null_and_generation_family_jackknife",
            }
        },
    }
    report["manifest_sha256"] = canonical_sha256(report)
    return report


def test_report_loader_checks_self_hash_and_firewall(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = _minimal_persisted_report()
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_candidate_labelability_evaluation(path, verify_sources=False) == report

    report["firewall"]["labels_inspected"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        load_candidate_labelability_evaluation(path, verify_sources=False)


def test_report_loader_rejects_rehashed_firewall_tamper(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = _minimal_persisted_report()
    report["firewall"]["labels_inspected"] = True
    report.pop("manifest_sha256")
    report["manifest_sha256"] = canonical_sha256(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="firewall"):
        load_candidate_labelability_evaluation(path, verify_sources=False)


def test_runtime_path_guard_rejects_editable_install_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    validate_candidate_labelability_runtime_paths(repo_root)
    monkeypatch.setattr(
        evaluation_module.candidate_coherence_module,
        "__file__",
        "/different/worktree/candidate_coherence.py",
    )
    with pytest.raises(ValueError, match="another worktree: candidate_coherence"):
        validate_candidate_labelability_runtime_paths(repo_root)


def test_exclusive_report_writer_never_replaces(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    evaluation_module._write_exclusive(path, b"first")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        evaluation_module._write_exclusive(path, b"second")
    assert path.read_bytes() == b"first"


def test_missing_scoreable_family_persists_unavailable_bootstrap() -> None:
    families = [f"family-{index}" for index in range(8)]
    effects = dict.fromkeys(families[:-1], 0.1)
    coherence = {
        "partition": "audit",
        "expected_family_ids": families,
        "comparisons": {
            comparison: {
                "mean_effect": 0.1,
                "per_family_effect": effects,
            }
            for comparison in ("C_minus_W", "F_minus_W", "C_minus_S", "F_minus_S")
        },
    }
    reports = evaluation_module._bootstrap_reports(coherence)
    assert reports["C_minus_W"] == {
        "available": False,
        "reason": "not_all_frozen_families_scoreable",
        "family_ids": families[:-1],
        "family_count": 7,
        "missing_family_ids": [families[-1]],
        "unexpected_family_ids": [],
        "replicates": 10_000,
        "mean_effect": 0.1,
        "ci_95_lower": None,
        "ci_95_upper": None,
    }


def test_deep_loader_recomputes_and_rejects_rehashed_numeric_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifacts: tuple[
        CandidateClusterInputBundle, LoadedCandidateClusteringBaseline
    ],
) -> None:
    bundle, baseline = synthetic_artifacts
    evaluation = evaluate_loaded_candidate_labelability(bundle, baseline)
    report = _minimal_persisted_report()
    report["source_input_bundle"] = {
        "manifest_path": str(tmp_path / "inputs" / "manifest.json"),
        "manifest_sha256": bundle.manifest["manifest_sha256"],
    }
    report["source_clustering_baseline"] = {
        "manifest_path": str(tmp_path / "baseline" / "manifest.json"),
        "manifest_sha256": baseline.manifest["manifest_sha256"],
    }
    evaluation["record_counts"]["candidate_basis_target_occurrences"] += 1
    report["evaluation"] = evaluation
    report.pop("manifest_sha256")
    report["manifest_sha256"] = canonical_sha256(report)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(evaluation_module, "_validate_revision", lambda value: None)
    monkeypatch.setattr(
        evaluation_module,
        "_validate_artifact_binding",
        lambda value, *, label: None,
    )
    monkeypatch.setattr(
        evaluation_module,
        "load_candidate_cluster_input_bundle",
        lambda root: bundle,
    )
    monkeypatch.setattr(
        evaluation_module,
        "load_candidate_clustering_baseline",
        lambda root, *, verify_source: baseline,
    )
    with pytest.raises(ValueError, match="recomputed evaluation drift"):
        load_candidate_labelability_evaluation(path, verify_sources=True)
