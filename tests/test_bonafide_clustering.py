"""Tests for exact missing-aware sparse clustering primitives."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from circuits.analysis.bonafide.clustering import (
    TargetProfileBlock,
    accumulate_pair_evidence,
    knn_affinity,
    mean_similarity_matrix,
    sparse_spectral_cluster,
    target_pairwise_profile_similarity,
)
from circuits.analysis.bonafide.clustering_store import (
    BasisSupport,
    PairEvidenceBuild,
    load_pair_evidence,
    write_pair_evidence_build,
)


def _block(
    trace_unit_id: str,
    basis_indices: list[int],
    values: list[list[float]],
    support: list[list[bool]],
    *,
    fit_weight: float = 1.0,
    response_id: str | None = None,
    base_question_id: str | None = None,
) -> TargetProfileBlock:
    return TargetProfileBlock(
        trace_unit_id=trace_unit_id,
        response_id=response_id or f"response-{trace_unit_id}",
        base_question_id=base_question_id or f"family-{trace_unit_id}",
        basis_indices=np.asarray(basis_indices, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
        support=np.asarray(support, dtype=np.bool_),
        fit_weight=fit_weight,
    )


def test_target_similarity_uses_only_pair_intersection_support() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 99.0],
            [2.0, 99.0, 7.0],
            [99.0, -2.0, 3.0],
            [0.5, 4.0, 99.0],
        ],
        dtype=np.float32,
    )
    support = np.asarray(
        [
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [True, True, False],
        ],
        dtype=np.bool_,
    )

    similarity, valid = target_pairwise_profile_similarity(values, support)

    assert valid.all()
    assert similarity[0, 1] == pytest.approx(1.0)
    assert similarity[0, 2] == pytest.approx(-1.0)
    assert similarity[1, 2] == pytest.approx(1.0)
    assert similarity[0, 3] == pytest.approx(8.5 / np.sqrt(81.25))
    assert similarity == pytest.approx(similarity.T)


def test_zero_norm_or_disjoint_support_is_missing_not_zero() -> None:
    values = np.asarray(
        [[1.0, 0.0], [0.0, 2.0], [0.0, 7.0]],
        dtype=np.float32,
    )
    support = np.asarray(
        [[True, False], [False, True], [True, False]],
        dtype=np.bool_,
    )

    similarity, valid = target_pairwise_profile_similarity(values, support)

    assert valid[0, 0]
    assert valid[1, 1]
    assert not valid[0, 1]
    assert not valid[0, 2]
    assert similarity[0, 1] == 0.0
    assert similarity[0, 2] == 0.0


def test_pair_evidence_uses_hierarchical_weights_and_overlap_counts() -> None:
    blocks = [
        _block(
            "trace-a",
            [0, 1],
            [[1.0, 0.0], [1.0, 0.0]],
            [[True, True], [True, True]],
            fit_weight=0.25,
            response_id="response-a",
            base_question_id="family-a",
        ),
        _block(
            "trace-b",
            [0, 1],
            [[1.0, 0.0], [-1.0, 0.0]],
            [[True, True], [True, True]],
            fit_weight=0.75,
            response_id="response-b",
            base_question_id="family-b",
        ),
    ]

    evidence = accumulate_pair_evidence(
        blocks,
        basis_count=3,
        weighting="hierarchical",
    )
    mean = mean_similarity_matrix(
        evidence,
        min_pair_target_overlap=2,
    )

    assert evidence.target_count == 2
    assert evidence.overlap_count[0, 1] == 2
    assert evidence.response_overlap_count[0, 1] == 2
    assert evidence.family_overlap_count[0, 1] == 2
    assert evidence.support_weight_sum[0, 1] == pytest.approx(1.0)
    assert mean[0, 1] == pytest.approx(-0.5)
    assert evidence.valid_profile_target_counts.tolist() == [2, 2, 0]


def test_overlap_and_eligibility_filters_are_explicit() -> None:
    evidence = accumulate_pair_evidence(
        [
            _block(
                "trace-a",
                [0, 1, 2],
                [[1.0], [1.0], [1.0]],
                [[True], [True], [True]],
                response_id="response-a",
                base_question_id="family-a",
            ),
            _block(
                "trace-b",
                [0, 1],
                [[1.0], [1.0]],
                [[True], [True]],
                response_id="response-a",
                base_question_id="family-a",
            ),
        ],
        basis_count=3,
    )
    eligible = np.asarray([True, True, False], dtype=np.bool_)

    similarity = mean_similarity_matrix(
        evidence,
        min_pair_target_overlap=2,
        eligible_mask=eligible,
    )

    assert similarity[0, 1] == pytest.approx(1.0)
    assert similarity[0, 2] == 0.0
    assert similarity[1, 2] == 0.0
    assert similarity[2, 2] == 0.0

    recurring = mean_similarity_matrix(
        evidence,
        min_pair_target_overlap=2,
        min_pair_response_overlap=2,
        min_pair_family_overlap=2,
        eligible_mask=eligible,
    )
    assert recurring.nnz == 0


def test_duplicate_target_or_basis_fails_closed() -> None:
    duplicate_basis = _block(
        "trace-a",
        [0, 0],
        [[1.0], [1.0]],
        [[True], [True]],
    )
    with pytest.raises(ValueError, match="duplicate basis"):
        accumulate_pair_evidence([duplicate_basis], basis_count=2)

    block = _block(
        "trace-a",
        [0],
        [[1.0]],
        [[True]],
    )
    with pytest.raises(ValueError, match="duplicate target"):
        accumulate_pair_evidence([block, block], basis_count=2)


def test_knn_ties_are_deterministic_and_symmetrization_is_declared() -> None:
    similarity = csr_matrix(
        np.asarray(
            [
                [1.0, 0.8, 0.8, 0.0],
                [0.8, 1.0, 0.2, 0.0],
                [0.8, 0.2, 1.0, 0.7],
                [0.0, 0.0, 0.7, 1.0],
            ]
        )
    )

    union = knn_affinity(similarity, neighbors=1, symmetrization="union_max")
    mutual = knn_affinity(similarity, neighbors=1, symmetrization="mutual_min")

    assert union[0, 1] == pytest.approx(0.8)
    assert union[0, 2] == pytest.approx(0.8)
    assert union[2, 3] == pytest.approx(0.7)
    assert mutual[0, 1] == pytest.approx(0.8)
    assert mutual[0, 2] == 0.0
    assert mutual[2, 3] == 0.0
    assert (union - union.T).nnz == 0
    assert (mutual - mutual.T).nnz == 0


def test_sparse_spectral_clustering_is_deterministic_and_leaves_isolates_out() -> None:
    affinity = np.zeros((7, 7), dtype=np.float64)
    for group in ([0, 1, 2], [3, 4, 5]):
        for left in group:
            for right in group:
                if left != right:
                    affinity[left, right] = 1.0
    sparse_affinity = csr_matrix(affinity)

    first = sparse_spectral_cluster(
        sparse_affinity,
        n_clusters=2,
        random_seed=17,
    )
    second = sparse_spectral_cluster(
        sparse_affinity,
        n_clusters=2,
        random_seed=17,
    )

    assert first.labels.tolist() == second.labels.tolist()
    assert first.labels[0] == first.labels[1] == first.labels[2] == 0
    assert first.labels[3] == first.labels[4] == first.labels[5] == 1
    assert first.labels[6] == -1
    assert first.connected_component_count == 2
    assert first.cluster_sizes == {0: 3, 1: 3}


def test_pair_evidence_persistence_round_trip(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = accumulate_pair_evidence(
        [
            _block(
                "trace-a",
                [0, 1],
                [[1.0], [1.0]],
                [[True], [True]],
            )
        ],
        basis_count=2,
    )
    feature_manifest = {
        "manifest_sha256": "feature-manifest",
        "plan_sha256": "feature-plan",
        "schema_version": "adag.bonafide.dense-feature-store.v1",
    }
    build = PairEvidenceBuild(
        evidence=evidence,
        basis_support=BasisSupport(
            target_counts=np.asarray([1, 1], dtype=np.int64),
            response_counts=np.asarray([1, 1], dtype=np.int64),
            family_counts=np.asarray([1, 1], dtype=np.int64),
            boundary_mask=np.asarray([False, True], dtype=np.bool_),
        ),
        feature_manifest=feature_manifest,
        feature_store_root=tmp_path / "source",
        basis_rows=(),
    )
    output = tmp_path / "pair-evidence"
    manifest = write_pair_evidence_build(
        output,
        build,
        code_revision={"git_commit": "test"},
        environment={"python": "test"},
    )

    class FakeFeatureStoreReader:
        def __init__(self, _path) -> None:
            self.manifest = feature_manifest

    monkeypatch.setattr(
        "circuits.analysis.bonafide.clustering_store.FeatureStoreReader",
        FakeFeatureStoreReader,
    )
    restored, support = load_pair_evidence(output)

    assert manifest["descriptions_generated"] is False
    assert restored.overlap_count.toarray().tolist() == [[1, 1], [0, 1]]
    assert restored.response_overlap_count.toarray().tolist() == [
        [1, 1],
        [0, 1],
    ]
    assert support.target_counts.tolist() == [1, 1]
    assert support.boundary_mask.tolist() == [False, True]
