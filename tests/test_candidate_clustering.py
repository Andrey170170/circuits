"""Focused tests for the frozen candidate-aware clustering core."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from circuits.analysis.bonafide.candidate_clustering import (
    _validate_profile_rows,
    _validated_manifest,
    basis_eligibility,
    choose_common_cluster_count,
    choose_medoid_seed,
    empirical_positive_midranks,
    fit_resolution,
    fuse_calibrated_similarities,
    weighted_support_jaccard,
)
from circuits.analysis.bonafide.candidate_profiles import (
    CANDIDATE_CLUSTER_INPUT_SCHEMA,
    FROZEN_ARTIFACT_PAYLOAD_SET_SHA256,
    FROZEN_C2_REPORT_SHA256,
    FROZEN_CLUSTERING_EVALUATION_REPORT_SCHEMA,
    FROZEN_CLUSTERING_EVALUATION_SHA256,
    FROZEN_PLAN_CANONICAL_SHA256,
    FROZEN_PLAN_FILE_SHA256,
    FROZEN_PROTOCOL_SHA256,
    FROZEN_SALVAGE_REPORT_SHA256,
    FROZEN_SELECTION_SHA256,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.clustering import (
    SparseSpectralResult,
    TargetProfileBlock,
    accumulate_pair_evidence,
)
from scipy.sparse import csr_matrix


def _block(
    case_id: str,
    family: str,
    response: str,
    indices: list[int],
    values: list[list[float]],
    *,
    weight: float = 1.0,
    support: list[list[bool]] | None = None,
) -> TargetProfileBlock:
    array = np.asarray(values, dtype=np.float32)
    return TargetProfileBlock(
        trace_unit_id=case_id,
        response_id=response,
        base_question_id=family,
        basis_indices=np.asarray(indices, dtype=np.int64),
        values=array,
        support=(
            np.ones(array.shape, dtype=np.bool_)
            if support is None
            else np.asarray(support, dtype=np.bool_)
        ),
        fit_weight=weight,
    )


def test_directional_diagonal_treats_zero_norm_as_missing() -> None:
    evidence = accumulate_pair_evidence(
        [
            _block(
                "t1",
                "f1",
                "r1",
                [0, 1, 2],
                [[1.0, 0.0], [0.0, 0.0], [9.0, 2.0]],
                support=[[True, False], [True, True], [False, True]],
            )
        ],
        basis_count=3,
    )

    assert evidence.overlap_count.diagonal().tolist() == [1, 0, 1]
    # Disjoint coordinate support is also missing, never scientific zero.
    assert evidence.overlap_count[0, 2] == 0


def test_common_eligibility_requires_three_targets_two_responses_two_families() -> None:
    evidence = accumulate_pair_evidence(
        [
            _block("t1", "f1", "r1", [0, 1, 2], [[1.0], [1.0], [1.0]]),
            _block("t2", "f1", "r2", [0, 1], [[1.0], [1.0]]),
            _block("t3", "f2", "r3", [0, 2], [[1.0], [1.0]]),
        ],
        basis_count=3,
    )

    assert basis_eligibility(evidence).tolist() == [True, False, False]


def test_empirical_midrank_is_ascending_and_ties_share_exact_midrank() -> None:
    similarity = csr_matrix(
        np.asarray(
            [
                [1.0, 0.2, 0.2, 0.9],
                [0.2, 1.0, -0.1, 0.0],
                [0.2, -0.1, 1.0, 0.9],
                [0.9, 0.0, 0.9, 1.0],
            ]
        )
    )

    ranked = empirical_positive_midranks(similarity)

    # Four positive off-diagonal pairs: the .2 tie occupies ranks 1 and 2,
    # while the .9 tie occupies ranks 3 and 4. Diagonals are excluded.
    assert ranked[0, 1] == pytest.approx(1.5 / 4)
    assert ranked[0, 2] == pytest.approx(1.5 / 4)
    assert ranked[0, 3] == pytest.approx(3.5 / 4)
    assert ranked[2, 3] == pytest.approx(3.5 / 4)
    assert ranked.diagonal().tolist() == [0.0] * 4
    assert (ranked - ranked.T).nnz == 0


def test_fusion_uses_only_positive_recurring_pair_intersection() -> None:
    width = csr_matrix(
        np.asarray(
            [
                [1.0, 0.2, 0.8, 0.0],
                [0.2, 1.0, 0.0, 0.7],
                [0.8, 0.0, 1.0, 0.0],
                [0.0, 0.7, 0.0, 1.0],
            ]
        )
    )
    candidate = csr_matrix(
        np.asarray(
            [
                [1.0, 0.4, 0.0, 0.0],
                [0.4, 1.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.9],
                [0.0, -0.2, 0.9, 1.0],
            ]
        )
    )

    fused = fuse_calibrated_similarities(width, candidate)

    assert fused.nnz == 2
    assert fused[0, 1] > 0
    assert fused[0, 2] == 0
    assert fused[1, 3] == 0  # candidate evidence is nonpositive
    assert fused[2, 3] == 0  # missing from W


def test_weighted_support_jaccard_uses_union_mass_and_recurrence_gates() -> None:
    evidence = accumulate_pair_evidence(
        [
            _block("t1", "f1", "r1", [0, 1], [[1.0], [1.0]], weight=0.25),
            _block("t2", "f2", "r2", [0, 1], [[1.0], [1.0]], weight=0.25),
            _block("t3", "f3", "r3", [1, 2], [[1.0], [1.0]], weight=0.5),
        ],
        basis_count=3,
    )

    jaccard = weighted_support_jaccard(
        evidence, eligible_mask=np.ones(3, dtype=np.bool_)
    )

    assert jaccard[0, 1] == pytest.approx(0.5 / 1.0)
    assert jaccard[1, 0] == pytest.approx(0.5)
    # Only one co-supported target/response/family, so recurrence removes it.
    assert jaccard[1, 2] == 0


def test_seed_medoid_uses_mean_ari_and_smaller_seed_tie() -> None:
    labels = {
        17: np.asarray([0, 0, 1, 1, -1], dtype=np.int64),
        29: np.asarray([0, 0, 1, 1, -1], dtype=np.int64),
        43: np.asarray([0, 1, 0, 1, -1], dtype=np.int64),
    }

    medoid, pairwise = choose_medoid_seed(labels)

    assert medoid == 17
    assert pairwise[(17, 29)] == pytest.approx(1.0)
    assert pairwise[(17, 43)] == pairwise[(29, 43)]


def test_common_count_prefers_64_then_smallest_valid_common() -> None:
    def fit(valid: bool) -> SimpleNamespace:
        return SimpleNamespace(valid=valid)

    states = {
        view: {32: fit(True), 64: fit(True), 96: fit(True)} for view in ("W", "C", "F")
    }
    assert choose_common_cluster_count(states) == 64
    states["F"][64] = fit(False)
    assert choose_common_cluster_count(states) == 32
    states["C"][32] = fit(False)
    assert choose_common_cluster_count(states) == 96
    states["W"][96] = fit(False)
    assert choose_common_cluster_count(states) is None


def test_resolution_validity_requires_exact_requested_cluster_count(
    monkeypatch,
) -> None:
    similarity = csr_matrix(np.ones((4, 4), dtype=np.float64) - np.eye(4))

    def collapsed(*args, **kwargs) -> SparseSpectralResult:
        del args, kwargs
        return SparseSpectralResult(
            labels=np.zeros(4, dtype=np.int64),
            active_mask=np.ones(4, dtype=np.bool_),
            eigenvalues=np.asarray([1.0, 0.5]),
            connected_component_count=1,
            cluster_sizes={0: 4},
        )

    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_clustering.sparse_spectral_cluster",
        collapsed,
    )
    fitted = fit_resolution(
        "W",
        similarity,
        eligible_mask=np.ones(4, dtype=np.bool_),
        n_clusters=2,
    )

    assert not fitted.valid
    assert {seed.error for seed in fitted.seeds.values()} == {"assigned_cluster_count"}


def _minimal_manifest(tmp_path) -> dict[str, object]:
    names = [
        "basis-index.parquet",
        "targets.parquet",
        "width-profiles.parquet",
        "candidate-profiles.parquet",
        "family-partitions.json",
    ]
    for name in names:
        (tmp_path / name).write_bytes(f"fixture:{name}".encode())
    manifest: dict[str, object] = {
        "schema_version": CANDIDATE_CLUSTER_INPUT_SCHEMA,
        "purpose": "frozen_inputs_only_no_cluster_fit_or_description_generation",
        "inputs": {
            "selection": {"path": "/selection.json", "sha256": FROZEN_SELECTION_SHA256},
            "candidate_union_plan": {
                "path": "/plan.json",
                "file_sha256": FROZEN_PLAN_FILE_SHA256,
                "canonical_sha256": FROZEN_PLAN_CANONICAL_SHA256,
            },
            "audited_c2_report": {
                "path": "/c2.json",
                "sha256": FROZEN_C2_REPORT_SHA256,
            },
            "posthoc_salvage_report": {
                "path": "/salvage.json",
                "sha256": FROZEN_SALVAGE_REPORT_SHA256,
            },
            "artifact_payload_set_sha256": FROZEN_ARTIFACT_PAYLOAD_SET_SHA256,
            "width1_root": "/width",
            "candidate_union_root": "/candidate",
        },
        "protocol": {"path": "/frozen/protocol.md", "sha256": FROZEN_PROTOCOL_SHA256},
        "structural_evaluation_contract": {
            "path": "circuits/analysis/bonafide/clustering_evaluation.py",
            "sha256": FROZEN_CLUSTERING_EVALUATION_SHA256,
            "report_schema": FROZEN_CLUSTERING_EVALUATION_REPORT_SCHEMA,
        },
        "partition_manifest_sha256": "a" * 64,
        "cohort": {
            "target_count": 245,
            "response_count": 35,
            "family_count": 34,
            "phase_bin_counts": {str(index): 35 for index in range(7)},
            "candidate_width_counts": {"5": 235, "6": 10},
            "candidate_activation_invariance": {
                "rtol": 1e-6,
                "atol": 1e-7,
                "violation_count": 0,
                "comparison_count": 1,
                "max_abs_deviation": 0.0,
                "max_relative_deviation": 0.0,
            },
        },
        "code_revision": {
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "git_dirty": False,
            "git_status_sha256": hashlib.sha256(b"").hexdigest(),
            "source_tree_sha256": "c" * 64,
            "files": [
                {
                    "path": "docs/CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md",
                    "sha256": FROZEN_PROTOCOL_SHA256,
                },
                {
                    "path": "circuits/analysis/bonafide/clustering_evaluation.py",
                    "sha256": FROZEN_CLUSTERING_EVALUATION_SHA256,
                },
            ],
        },
        "files": [
            {"path": name, "sha256": file_sha256(tmp_path / name)} for name in names
        ],
        "outcomes_inspected": False,
        "model_calls_made": False,
        "cluster_fit_performed": False,
        "confirmatory_holdout_opened": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def test_bundle_manifest_rejects_manifest_and_file_hash_drift(tmp_path) -> None:
    manifest = _minimal_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        _validated_manifest(tmp_path)

    import json

    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "targets.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"file hash mismatch: targets\.parquet"):
        _validated_manifest(tmp_path)


def test_bundle_manifest_rejects_frozen_source_and_publication_flag_drift(
    tmp_path,
) -> None:
    import json

    manifest = _minimal_manifest(tmp_path)
    manifest["inputs"]["selection"]["sha256"] = "0" * 64
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="selection"):
        _validated_manifest(tmp_path)

    manifest = _minimal_manifest(tmp_path)
    manifest["model_calls_made"] = True
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model_calls_made"):
        _validated_manifest(tmp_path)


def test_full_cohort_profile_validation_rejects_width_and_finiteness_drift() -> None:
    identity = {
        "model_id": "model",
        "model_revision": "revision",
        "layer": 1,
        "neuron_index": 2,
        "polarity": "+",
    }
    basis_rows = [{"signed_basis_index": 0, **identity}]
    width_row = {
        "case_id": "case",
        "signed_basis_index": 0,
        **identity,
        "attribution_profile": [1.0, None],
        "attribution_support": [True, False],
        "signed_attribution": 0.5,
        "occurrence_count": 1,
    }
    _validate_profile_rows(
        [width_row],
        all_case_ids={"case"},
        basis_rows=basis_rows,
        candidate=False,
    )
    with pytest.raises(ValueError, match="supported width profile value"):
        _validate_profile_rows(
            [
                {
                    **width_row,
                    "attribution_profile": [1.0, float("nan")],
                    "attribution_support": [True, True],
                }
            ],
            all_case_ids={"case"},
            basis_rows=basis_rows,
            candidate=False,
        )

    candidate_row = {
        "case_id": "case",
        "signed_basis_index": 0,
        **identity,
        "candidate_contrast_profile": [1.0, 2.0, 3.0, 4.0],
        "candidate_profile_l2_norm": np.sqrt(30.0),
        "occurrence_count": 1,
    }
    with pytest.raises(ValueError, match="finite width five"):
        _validate_profile_rows(
            [candidate_row],
            all_case_ids={"case"},
            basis_rows=basis_rows,
            candidate=True,
        )
