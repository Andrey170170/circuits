from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import circuits.analysis.bonafide.hybrid_candidate_clustering as hybrid_module
from circuits.analysis.bonafide.hybrid_candidate_clustering import (
    HybridInvalidFit,
    HybridTargetBlock,
    accumulate_fused_evidence,
    blocks_from_bundle,
    candidate_view,
    fit_hybrid_grid,
    target_fused_similarity,
)
from circuits.analysis.bonafide.hybrid_candidate_clustering_execution import (
    run_hybrid_candidate_clustering,
)
from circuits.analysis.bonafide.hybrid_candidate_inputs import (
    _artifact_payload_hashes,
    build_hybrid_input_bundle,
    collect_hybrid_code_revision,
    extract_hybrid_target,
    load_hybrid_input_bundle,
    paper_normalize_occurrence,
)


def _target(
    case_id: str,
    attr: list[list[float]],
    candidate: list[list[float]],
    *,
    response: str = "r",
    family: str = "f",
) -> HybridTargetBlock:
    attr_values = np.asarray(attr, dtype=np.float32)
    return HybridTargetBlock(
        case_id=case_id,
        response_id=response,
        base_question_id=family,
        basis_indices=np.arange(len(attr), dtype=np.int64),
        attr_values=attr_values,
        attr_support=np.ones(attr_values.shape, dtype=np.bool_),
        candidate_values=np.asarray(candidate, dtype=np.float32),
        fit_weight=1.0,
    )


def test_raw_view_preserves_opposing_candidate_contributions() -> None:
    raw = np.asarray([[2.0, -2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = candidate_view(
        raw,
        model_top5_indices=[0, 1, 2, 3, 4],
        observed_candidate_index=0,
        representation="raw",
    )

    assert result.sum() == 0.0
    assert result[0, 0] == 2.0
    assert result[0, 1] == -2.0


def test_common_mode_survives_raw_but_not_observed_contrast() -> None:
    raw = np.asarray([[3.0, 3.0, 3.0, 3.0, 3.0]], dtype=np.float32)

    direct = candidate_view(
        raw,
        model_top5_indices=[0, 1, 2, 3, 4],
        observed_candidate_index=0,
        representation="raw",
    )
    contrast = candidate_view(
        raw,
        model_top5_indices=[0, 1, 2, 3, 4],
        observed_candidate_index=0,
        representation="contrast",
    )

    assert np.linalg.norm(direct) > 0
    assert np.array_equal(contrast, np.zeros((1, 5), dtype=np.float32))


def test_width_five_and_six_keep_explicit_model_rank_order() -> None:
    width5 = np.asarray([[10, 20, 30, 40, 50]], dtype=np.float32)
    assert candidate_view(
        width5,
        model_top5_indices=[2, 0, 4, 1, 3],
        observed_candidate_index=0,
        representation="top5",
    ).tolist() == [[30, 10, 50, 20, 40]]

    width6 = np.asarray([[99, 10, 20, 30, 40, 50]], dtype=np.float32)
    assert candidate_view(
        width6,
        model_top5_indices=[1, 2, 3, 4, 5],
        observed_candidate_index=0,
        representation="top5",
    ).tolist() == [[10, 20, 30, 40, 50]]


def test_paper_normalization_uses_upstream_epsilon_fallback() -> None:
    attr, contribution = paper_normalize_occurrence(
        [4.0, None, -2.0],
        activation=2.0,
        candidate_contribution=[4.0, 5.0, -6.0],
        candidate_logits=[2.0, 1e-12, -3.0],
    )
    assert attr == [2.0, None, -1.0]
    assert contribution == [2.0, 5.0, 2.0]

    fallback_attr, _ = paper_normalize_occurrence(
        [4.0],
        activation=1e-12,
        candidate_contribution=[1.0],
        candidate_logits=[1.0],
    )
    assert fallback_attr == [4.0]


def test_paper_normalized_view_uses_only_ranked_top_five() -> None:
    raw = np.asarray([[99, 10, 20, 30, 40, 50]], dtype=np.float32)
    result = candidate_view(
        raw,
        model_top5_indices=[1, 2, 3, 4, 5],
        observed_candidate_index=0,
        representation="paper_normalized",
        paper_normalized=raw / 10,
    )
    assert result.tolist() == [[1, 2, 3, 4, 5]]


def test_valid_zero_is_evidence_but_zero_norm_is_missing() -> None:
    values = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    support = np.ones(values.shape, dtype=np.bool_)
    fused, valid = target_fused_similarity(values, support, values)

    assert valid[0, 1]
    assert fused[0, 1] == 0.0

    missing_values = values.copy()
    missing_values[1] = 0.0
    _, missing_valid = target_fused_similarity(missing_values, support, values)
    assert not missing_valid[0, 1]


def test_harmonic_fusion_happens_before_target_average() -> None:
    first = _target(
        "a",
        [[1, 0], [1, 0]],
        [[1, 0], [0, 1]],
        response="r1",
        family="f1",
    )
    second = _target(
        "b",
        [[1, 0], [0, 1]],
        [[1, 0], [1, 0]],
        response="r2",
        family="f2",
    )

    evidence = accumulate_fused_evidence([first, second], basis_count=2)

    # Each target has one zero view, so mean(target harmonic) is zero.  The
    # incorrect harmonic(mean views) operation would produce 0.5.
    assert evidence.support_weight_sum[0, 1] == 2.0
    assert evidence.weighted_similarity_sum[0, 1] == 0.0


def test_evidence_is_deterministic_under_target_ordering() -> None:
    blocks = [
        _target("b", [[1, 0], [1, 1]], [[1, 0], [1, 1]], response="r2"),
        _target("a", [[0, 1], [1, 1]], [[0, 1], [1, 1]], response="r1"),
    ]

    forward = accumulate_fused_evidence(blocks, basis_count=2)
    reverse = accumulate_fused_evidence(list(reversed(blocks)), basis_count=2)

    assert (forward.weighted_similarity_sum != reverse.weighted_similarity_sum).nnz == 0
    assert (forward.support_weight_sum != reverse.support_weight_sum).nnz == 0


def test_fresh_cluster_path_has_no_width_assignment_input() -> None:
    signature = inspect.signature(fit_hybrid_grid)
    assert set(signature.parameters) == {
        "bundle",
        "representations",
        "affinity_modes",
        "cluster_counts",
    }


def test_input_builder_refuses_overwrite_before_reading_source(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_hybrid_input_bundle(
            source_root=tmp_path / "missing",
            output_root=output,
            repo_root=tmp_path / "missing-repo",
        )


def test_clustering_refuses_overwrite_before_reading_input(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_hybrid_candidate_clustering(
            input_root=tmp_path / "missing",
            output_root=output,
            repo_root=tmp_path / "missing-repo",
        )


def test_target_extraction_rejects_union_payload_hash_mismatch() -> None:
    union = SimpleNamespace(
        trace=SimpleNamespace(),
        manifest={"artifact_id": "union-1", "data_sha256": "actual"},
    )
    with pytest.raises(ValueError, match="payload hash"):
        extract_hybrid_target(
            union,
            target={
                "candidate_union_artifact_id": "union-1",
                "candidate_union_payload_sha256": "expected",
            },
        )


def test_input_loader_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"schema_version":"adag.bonafide.hybrid-candidate-inputs.v3",'
        '"manifest_sha256":"wrong"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest is invalid"):
        load_hybrid_input_bundle(tmp_path)


def test_artifact_payload_hash_binds_case_and_artifact_identity() -> None:
    row = {
        "case_id": "case-1",
        "candidate_union_artifact_id": "union-1",
        "candidate_union_payload_sha256": "a" * 64,
        "candidate_union_topology_sha256": "b" * 64,
        "refinement_artifact_id": "refinement-1",
        "refinement_payload_sha256": "c" * 64,
    }
    original = _artifact_payload_hashes([row])
    changed = _artifact_payload_hashes(
        [{**row, "candidate_union_artifact_id": "union-2"}]
    )
    assert original["candidate_union_set_sha256"] != changed[
        "candidate_union_set_sha256"
    ]


def test_grid_persists_invalid_cell_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = SimpleNamespace(representation="raw")
    monkeypatch.setattr(
        hybrid_module, "build_hybrid_evidence", lambda *args, **kwargs: evidence
    )

    def fit_one(_evidence, *, affinity_mode, n_clusters):
        if n_clusters == 32:
            raise ValueError("too few active bases")
        return SimpleNamespace(n_clusters=n_clusters)

    monkeypatch.setattr(hybrid_module, "_fit_one", fit_one)
    result = fit_hybrid_grid(
        SimpleNamespace(),
        representations=("raw",),
        affinity_modes=("full_positive",),
        cluster_counts=(32, 64),
    )
    assert isinstance(result[("raw", "full_positive", 32)], HybridInvalidFit)
    assert result[("raw", "full_positive", 64)].n_clusters == 64


def test_blocks_reject_mixed_partition_input() -> None:
    bundle = SimpleNamespace(
        target_rows=({"case_id": "a", "family_partition": "audit"},),
        profile_rows=(),
    )
    with pytest.raises(ValueError, match="non-generation"):
        blocks_from_bundle(bundle, representation="raw")


def test_code_revision_rejects_dirty_and_untracked_required_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    monkeypatch.setattr(
        "circuits.analysis.bonafide.hybrid_candidate_inputs.HYBRID_SOURCE_PATHS",
        ("source.py",),
    )
    assert collect_hybrid_code_revision(tmp_path)["git_dirty"] is False

    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from HEAD"):
        collect_hybrid_code_revision(tmp_path)

    source.write_text("value = 1\n", encoding="utf-8")
    untracked = tmp_path / "untracked.py"
    untracked.write_text("value = 3\n", encoding="utf-8")
    monkeypatch.setattr(
        "circuits.analysis.bonafide.hybrid_candidate_inputs.HYBRID_SOURCE_PATHS",
        ("untracked.py",),
    )
    with pytest.raises(ValueError, match="not tracked"):
        collect_hybrid_code_revision(tmp_path)
