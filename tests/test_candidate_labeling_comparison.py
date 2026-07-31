from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from circuits.analysis.bonafide.candidate_labeling_comparison import (
    ANCHOR_SCHEMA,
    ANCHORS_FILE,
    ARM_HANDOFF_SCHEMA,
    CANDIDATE_LABELING_COMPARISON_SCHEMA,
    EXPECTED_ELIGIBLE_ARMS,
    EXPECTED_FIREWALL,
    EXPECTED_PROMPT_CONTRACT,
    EXPECTED_W_ANCHORS,
    GENERATION_EVIDENCE_SCHEMA,
    GENERATION_FILE,
    HANDOFF_FILE,
    MANIFEST_FILE,
    SCORING_EVIDENCE_SCHEMA,
    SCORING_FILE,
    TOKENIZER_CHAT_TEMPLATE_SHA256,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    _aggregate_candidate,
    _aggregate_width,
    _candidate_evidence,
    _candidate_slots,
    _target_context,
    load_candidate_labeling_comparison,
    select_w_anchors,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256


def test_anchor_assignment_uses_lexicographically_smallest_global_optimum() -> None:
    result = select_w_anchors(
        member_counts={cluster: 5 for cluster in range(13)},
        generation_target_counts={cluster: 8 for cluster in range(13)},
    )

    assert result["anchors_in_target_point_order"] == list(range(12))
    ready = result["ready_clusters"]
    assert {item["member_midrank_percentile"] for item in ready} == {0.5}
    assert {item["support_midrank_percentile"] for item in ready} == {0.5}


def test_anchor_assignment_rejects_sparse_fallback() -> None:
    with pytest.raises(ValueError, match="fewer than 12"):
        select_w_anchors(
            member_counts={cluster: cluster + 1 for cluster in range(11)},
            generation_target_counts={cluster: cluster + 8 for cluster in range(11)},
        )


def test_candidate_signature_uses_occurrence_count_for_m_and_mean() -> None:
    result = _aggregate_candidate(
        [
            {
                "signed_basis_index": 2,
                "candidate_contrast_profile": [2.0, 4.0, 0.0, -2.0, -4.0],
                "occurrence_count": 2,
            },
            {
                "signed_basis_index": 7,
                "candidate_contrast_profile": [1.0, 2.0, 0.0, -1.0, -2.0],
                "occurrence_count": 1,
            },
        ]
    )

    assert result["member_basis_count"] == 2
    assert result["member_occurrence_count_m"] == 3
    assert result["signed_sum"] == [3.0, 6.0, 0.0, -3.0, -6.0]
    assert result["elementwise_mean"] == [1.0, 2.0, 0.0, -1.0, -2.0]
    assert result["clipped"] is False


def test_width_highlights_are_missing_aware_and_occurrence_weighted() -> None:
    result = _aggregate_width(
        [
            {
                "signed_basis_index": 2,
                "attribution_profile": [4.0, None, -2.0],
                "attribution_support": [True, False, True],
                "occurrence_count": 2,
            },
            {
                "signed_basis_index": 7,
                "attribution_profile": [2.0, 3.0, None],
                "attribution_support": [True, True, False],
                "occurrence_count": 1,
            },
        ]
    )

    assert result["member_occurrence_count"] == 3
    assert result["signed_sum_by_source_token"] == [6.0, 3.0, -2.0]
    assert result["support_occurrence_count_by_source_token"] == [3, 1, 2]
    assert result["mean_by_member_occurrence"] == [2.0, 3.0, -1.0]
    assert result["highlight_token_indices"] == [1, 0, 2]


def test_candidate_slots_keep_model_rank_order_and_observed_outside_top_five() -> None:
    candidates = [
        {
            "full_distribution_rank": rank,
            "token_id": 100 + rank,
            "token_text": f"t{rank}",
            "logit": 10.0 - rank,
            "probability": 0.1 / rank,
            "is_observed": False,
        }
        for rank in range(1, 6)
    ]
    candidates.append(
        {
            "full_distribution_rank": 9,
            "token_id": 999,
            "token_text": "observed",
            "logit": 0.0,
            "probability": 0.001,
            "is_observed": True,
        }
    )
    result = _candidate_slots(
        {
            "candidate_selection_json": json.dumps({"candidates": candidates}),
            "candidate_count": 6,
            "observed_token_id": 999,
        }
    )

    assert [item["rank"] for item in result["model_rank_slots"]] == [1, 2, 3, 4, 5]
    assert result["observed_token_full_distribution_rank"] == 9
    assert result["distinct_competitor_count"] == 5

    invalid_width_six = [dict(item) for item in candidates]
    invalid_width_six[-1]["full_distribution_rank"] = 2
    with pytest.raises(ValueError, match="width/observed-rank"):
        _candidate_slots(
            {
                "candidate_selection_json": json.dumps(
                    {"candidates": invalid_width_six}
                ),
                "candidate_count": 6,
                "observed_token_id": 999,
            }
        )


def test_candidate_evidence_requires_width_five_structural_zero() -> None:
    candidates = [
        {
            "full_distribution_rank": rank,
            "token_id": 100 + rank,
            "token_text": f"t{rank}",
            "logit": 10.0 - rank,
            "probability": 0.1 / rank,
            "is_observed": rank == 2,
        }
        for rank in range(1, 6)
    ]
    target = {
        "candidate_selection_json": json.dumps({"candidates": candidates}),
        "candidate_count": 5,
        "observed_token_id": 102,
    }
    rows = [
        {
            "signed_basis_index": 1,
            "candidate_contrast_profile": [1.0, 0.0, 2.0, 3.0, 4.0],
            "occurrence_count": 1,
        }
    ]
    _, signature = _candidate_evidence(target, rows)
    assert signature["signed_sum"][1] == 0.0

    rows[0]["candidate_contrast_profile"][1] = 0.25
    with pytest.raises(ValueError, match="structural zero"):
        _candidate_evidence(target, rows)


def test_target_context_is_exact_prefix_excluding_observed(monkeypatch) -> None:
    tokenized = SimpleNamespace(assistant_prefix_ids=[1, 2], response_ids=[10, 11, 12])
    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_labeling_comparison."
        "tokenize_teacher_forced_response",
        lambda tokenizer, prompt, response: tokenized,
    )

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return {1: "<u>", 2: "prompt", 10: " a", 11: " b", 12: " c"}[ids[0]]

    result = _target_context(
        {
            "example_json": json.dumps({"prompt": "question", "response": "answer"}),
            "response_position": 1,
            "observed_token_id": 11,
            "observed_token_text": " b",
        },
        Tokenizer(),
    )

    assert result["token_ids"] == [1, 2, 10]
    assert result["text"] == "<u>prompt a"
    assert result["source_attribution_token_count"] == 3
    assert result["observed_token"] == {
        "response_position": 1,
        "token_id": 11,
        "token_text": " b",
    }


def _write_artifact(root: Path) -> None:
    root.mkdir()
    anchors = {
        "schema_version": ANCHOR_SCHEMA,
        "selection": {
            "anchors_in_target_point_order": list(EXPECTED_W_ANCHORS),
        },
        "anchors": [
            {"anchor_index": index, "cluster_id": cluster}
            for index, cluster in enumerate(EXPECTED_W_ANCHORS)
        ],
    }
    generation = {
        "schema_version": GENERATION_EVIDENCE_SCHEMA,
        "family_partition": "generation",
        "prompt_eligible": True,
    }
    scoring = {
        "schema_version": SCORING_EVIDENCE_SCHEMA,
        "family_partition": "audit",
        "prompt_eligible": False,
    }
    handoff = {"schema_version": ARM_HANDOFF_SCHEMA, "arm_id": "arm_1_width_only"}
    (root / ANCHORS_FILE).write_text(json.dumps(anchors) + "\n")
    (root / GENERATION_FILE).write_text(json.dumps(generation) + "\n")
    (root / SCORING_FILE).write_text(json.dumps(scoring) + "\n")
    (root / HANDOFF_FILE).write_text(json.dumps(handoff) + "\n")
    files = []
    for name, count in (
        (ANCHORS_FILE, 1),
        (GENERATION_FILE, 1),
        (SCORING_FILE, 1),
        (HANDOFF_FILE, 1),
    ):
        path = root / name
        files.append(
            {
                "path": name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "row_count": count,
            }
        )
    manifest = {
        "schema_version": CANDIDATE_LABELING_COMPARISON_SCHEMA,
        "purpose": "provider_neutral_pre_model_call_evidence_only_labeling_comparison",
        "eligible_arms": EXPECTED_ELIGIBLE_ARMS,
        "anchor_cluster_ids_in_target_point_order": list(EXPECTED_W_ANCHORS),
        "firewall": EXPECTED_FIREWALL,
        "prompt_contract": EXPECTED_PROMPT_CONTRACT,
        "tokenizer": {
            "model_id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "name_or_path": TOKENIZER_ID,
            "resolved_commit_hash": TOKENIZER_REVISION,
            "class": "FakeTokenizer",
            "chat_template_sha256": TOKENIZER_CHAT_TEMPLATE_SHA256,
            "local_files_only": True,
            "reconstruction": "tokenize_teacher_forced_response",
        },
        "files": sorted(files, key=lambda item: item["path"]),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / MANIFEST_FILE).write_text(json.dumps(manifest) + "\n")


def test_loader_enforces_partition_prompt_firewall_and_file_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "comparison"
    _write_artifact(root)
    loaded = load_candidate_labeling_comparison(root, verify_sources=False)
    assert len(loaded.generation_evidence) == 1
    assert len(loaded.scoring_evidence) == 1

    scoring_path = root / SCORING_FILE
    scoring_path.write_text(
        json.dumps(
            {
                "schema_version": SCORING_EVIDENCE_SCHEMA,
                "family_partition": "audit",
                "prompt_eligible": True,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="file drift"):
        load_candidate_labeling_comparison(root, verify_sources=False)


def test_loader_rejects_semantically_rehashed_firewall_drift(tmp_path: Path) -> None:
    root = tmp_path / "comparison"
    _write_artifact(root)
    manifest_path = root / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["firewall"]["model_calls_made"] = True
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="firewall drift"):
        load_candidate_labeling_comparison(root, verify_sources=False)
