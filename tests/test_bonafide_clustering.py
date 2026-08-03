"""Tests for exact missing-aware sparse clustering primitives."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from circuits.analysis.bonafide.clustering import (
    TargetProfileBlock,
    accumulate_pair_evidence,
    knn_affinity,
    mean_similarity_matrix,
    sparse_spectral_cluster,
    target_pairwise_profile_similarity,
)
from circuits.analysis.bonafide.clustering_evaluation import (
    LoadedClusterState,
    assignment_ari,
    cluster_size_metrics,
    seed_stability,
    sparse_graph_partition_metrics,
)
from circuits.analysis.bonafide.clustering_projection import (
    _cluster_summaries,
    _decode_trace_mask,
    _phase_bin,
)
from circuits.analysis.bonafide.clustering_resampling import (
    _checkpoint_family_sets,
)
from circuits.analysis.bonafide.clustering_selection import (
    _balanced_exemplars,
    _family_partitions,
    _percentile_ranks,
)
from circuits.analysis.bonafide.clustering_store import (
    BasisSupport,
    PairEvidenceBuild,
    build_pair_evidence_from_feature_store,
    load_pair_evidence,
    write_pair_evidence_build,
)
from scipy.sparse import csr_matrix


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


def test_sparse_graph_partition_metrics_recovers_two_disconnected_cliques() -> None:
    affinity = csr_matrix(
        np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    graph = sparse_graph_partition_metrics(labels, affinity)
    sizes = cluster_size_metrics(labels)

    assert graph["observed_internal_affinity_fraction"] == pytest.approx(1.0)
    assert graph["degree_volume_null_internal_fraction"] == pytest.approx(0.5)
    assert graph["internal_affinity_enrichment"] == pytest.approx(2.0)
    assert graph["modularity"] == pytest.approx(0.5)
    assert graph["maximum_conductance"] == pytest.approx(0.0)
    assert sizes["cluster_sizes"] == [2, 2]
    assert sizes["normalized_size_entropy"] == pytest.approx(1.0)
    assert sizes["size_gini"] == pytest.approx(0.0)


def test_seed_stability_selects_assignment_medoid_and_ignores_unassigned() -> None:
    state_path = Path("/tmp/state")
    states = [
        LoadedClusterState(
            task_index=0,
            path=state_path,
            manifest={"config": {}},
            labels=np.asarray([0, 0, 1, 1, -1], dtype=np.int64),
        ),
        LoadedClusterState(
            task_index=1,
            path=state_path,
            manifest={"config": {}},
            labels=np.asarray([0, 0, 1, 1, 0], dtype=np.int64),
        ),
        LoadedClusterState(
            task_index=2,
            path=state_path,
            manifest={"config": {}},
            labels=np.asarray([0, 1, 0, 1, 1], dtype=np.int64),
        ),
    ]

    stability = seed_stability(states)

    assert assignment_ari(states[0].labels, states[1].labels) == pytest.approx(1.0)
    assert stability["medoid_task_index"] in {0, 1}
    assert stability["maximum_ari"] == pytest.approx(1.0)


def test_projection_summary_enforces_labelability_and_temporal_contract() -> None:
    rows = []
    for response_index in range(3):
        rows.extend(
            {
                "cluster_id": 0,
                "response_id": f"response-{response_index}",
                "base_question_id": f"family-{response_index}",
                "response_target_ordinal": ordinal,
                "response_phase_bin": _phase_bin(ordinal, 10),
                "absolute_attribution_mass": float(ordinal + 1),
            }
            for ordinal in range(7)
        )
    rows.append(
        {
            "cluster_id": 1,
            "response_id": "response-0",
            "base_question_id": "family-0",
            "response_target_ordinal": 0,
            "response_phase_bin": 0,
            "absolute_attribution_mass": 1.0,
        }
    )
    summaries = _cluster_summaries(
        labels=np.asarray([0] * 8 + [1] * 2, dtype=np.int64),
        target_cluster_rows=rows,
        response_target_counts={
            "response-0": 10,
            "response-1": 10,
            "response-2": 10,
        },
    )

    assert summaries[0]["labelable"] is True
    assert summaries[0]["support_target_count"] == 21
    assert summaries[0]["support_family_count"] == 3
    assert summaries[0]["median_persistence_density"] == pytest.approx(1.0)
    assert summaries[1]["labelable"] is False
    assert summaries[1]["labeling_status"] == "insufficient_labeling_support"
    witnesses, witness_hash = _decode_trace_mask(
        (1 << 0) | (1 << 2),
        ["trace-a", "trace-b", "trace-c"],
    )
    assert witnesses == ["trace-a", "trace-c"]
    assert len(witness_hash) == 64


def test_feature_store_evidence_family_selection_is_explicit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = [
        (
            _block(
                "trace-a",
                [0, 1],
                [[1.0], [1.0]],
                [[True], [True]],
                response_id="response-a",
                base_question_id="family-a",
            ),
            {
                "response_id": "response-a",
                "base_question_id": "family-a",
            },
        ),
        (
            _block(
                "trace-b",
                [0],
                [[1.0]],
                [[True]],
                response_id="response-b",
                base_question_id="family-b",
            ),
            {
                "response_id": "response-b",
                "base_question_id": "family-b",
            },
        ),
    ]

    class FakeFeatureStoreReader:
        basis_count = 2
        compacted_root = tmp_path
        manifest: ClassVar[dict[str, str]] = {
            "manifest_sha256": "feature",
            "plan_sha256": "plan",
            "schema_version": "feature",
        }
        basis_rows = (
            {"layer": 0},
            {"layer": 1},
        )

        def __init__(self, _path) -> None:
            pass

        def iter_blocks(self):
            yield from blocks

    monkeypatch.setattr(
        "circuits.analysis.bonafide.clustering_store.FeatureStoreReader",
        FakeFeatureStoreReader,
    )
    build = build_pair_evidence_from_feature_store(
        tmp_path,
        included_family_ids=frozenset({"family-a"}),
    )

    assert build.evidence.target_count == 1
    assert build.basis_support.target_counts.tolist() == [1, 1]
    assert build.target_selection["mode"] == "include_families"
    assert build.target_selection["selected_family_count"] == 1
    assert build.target_selection["selected_target_count"] == 1


def test_checkpoint_family_sets_use_deterministic_whole_family_prefixes() -> None:
    rows = [
        {
            "base_question_id": f"family-{family}",
            "trace_unit_id": f"trace-{family}-{target}",
        }
        for family, target_count in enumerate([100, 150, 200, 250, 300, 350, 400, 450])
        for target in range(target_count)
    ]

    checkpoints = _checkpoint_family_sets(rows)

    assert [item["requested_target_count"] for item in checkpoints] == [
        500,
        1000,
        1500,
    ]
    assert [item["selected_target_count"] for item in checkpoints] == [
        450,
        1000,
        1350,
    ]
    for item in checkpoints:
        assert item["included_family_ids"] == sorted(item["included_family_ids"])


def test_selection_percentile_ranks_handle_ties_and_direction() -> None:
    values = {4: 1.0, 6: 2.0, 9: 2.0}

    higher = _percentile_ranks(values, higher_is_better=True)
    lower = _percentile_ranks(values, higher_is_better=False)

    assert higher == pytest.approx({4: 0.0, 6: 0.75, 9: 0.75})
    assert lower == pytest.approx({4: 1.0, 6: 0.25, 9: 0.25})


def test_label_family_partitions_are_deterministic_and_disjoint() -> None:
    family_ids = [f"family-{index}" for index in range(10)]

    first = _family_partitions(family_ids, state_identity="selection-source")
    second = _family_partitions(
        list(reversed(family_ids)),
        state_identity="selection-source",
    )

    assert first == second
    assert set(first) == set(family_ids)
    assert set(first.values()) == {
        "generation",
        "selection_scoring",
        "audit",
    }


def test_label_exemplars_prefer_phase_condition_and_token_diversity() -> None:
    rows = [
        {
            "trace_unit_id": "trace-a",
            "base_question_id": "family-a",
            "response_id": "response-a",
            "response_phase_bin": 0,
            "absolute_attribution_mass": 10.0,
        },
        {
            "trace_unit_id": "trace-b",
            "base_question_id": "family-b",
            "response_id": "response-b",
            "response_phase_bin": 0,
            "absolute_attribution_mass": 9.0,
        },
        {
            "trace_unit_id": "trace-c",
            "base_question_id": "family-c",
            "response_id": "response-c",
            "response_phase_bin": 4,
            "absolute_attribution_mass": 8.0,
        },
    ]
    inventory = {
        "trace-a": {"condition": {"kind": "same"}, "target_token_text": "same"},
        "trace-b": {"condition": {"kind": "same"}, "target_token_text": "same"},
        "trace-c": {
            "condition": {"kind": "different"},
            "target_token_text": "different",
        },
    }

    selected = _balanced_exemplars(
        rows,
        family_partitions={
            "family-a": "generation",
            "family-b": "generation",
            "family-c": "generation",
        },
        inventory_by_trace=inventory,
    )

    assert [row["trace_unit_id"] for row in selected] == ["trace-a", "trace-c"]
