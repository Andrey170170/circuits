from __future__ import annotations

from pathlib import Path

from scripts.bonafide.build_topk_c2_bundle import (
    C2_CASE_COUNT,
    build_c2_manifests,
    select_c2_cases,
)
from scripts.bonafide.build_topk_c2_screen_pool import build_c2_screen_pool
from tests.test_build_topk_c2_screen_pool import _source


def _inputs() -> tuple[dict, dict, dict]:
    source = _source()
    source["tokenizer"]["chat_template_sha256"] = "d" * 64
    pool = build_c2_screen_pool(
        source,
        source_manifest_path=Path("/source.json"),
        source_manifest_sha256="a" * 64,
    )
    results = []
    for case in pool["cases"]:
        width = 6 if case["screen_slot"] == "minimum_probability" else 5
        source_item = next(
            item
            for wave in source["waves"]
            for item in wave["items"]
            if item["artifact_id"] == case["source_width1_artifact_id"]
        )
        observed = source_item["target_selection"]["final_target_token_id"]
        results.append(
            {
                "source_width1_artifact_id": case["source_width1_artifact_id"],
                "corpus_role": case["corpus_role"],
                "input_token_count": 100 + case["target_response_position"],
                "candidate_count": width,
                "candidate_selection": {
                    "policy_id": "model_top5_plus_observed",
                    "observed_token_rank": 7 if width == 6 else 1,
                    "candidates": [
                        {"token_id": observed},
                        *({"token_id": 1000 + index} for index in range(width - 1)),
                    ],
                },
            }
        )
    rank = {
        "schema_version": "bonafide-topk-rank-screen/v1",
        "results": results,
    }
    return pool, source, rank


def test_c2_selection_applies_frozen_width_preference() -> None:
    pool, source, rank = _inputs()

    cases = select_c2_cases(pool, source, rank)

    assert len(cases) == C2_CASE_COUNT == 245
    assert len({(case["example_id"], case["phase_bin"]) for case in cases}) == 245
    assert {case["candidate_count"] for case in cases} == {5, 6}
    assert all(
        case["candidate_count"] == case["desired_candidate_count"] for case in cases
    )


def test_c2_manifests_cover_every_candidate_with_bounded_waves() -> None:
    pool, source, rank = _inputs()
    cases = select_c2_cases(pool, source, rank)

    manifests = build_c2_manifests(
        cases,
        source,
        selection_path=Path("/selection.json"),
        selection_sha256="b" * 64,
        source_manifest_path=Path("/source.json"),
        source_manifest_sha256="a" * 64,
        rank_screen_path=Path("/rank.json"),
        rank_screen_sha256="c" * 64,
    )

    assert set(manifests) == {*(f"independent-candidate-{index}" for index in range(6))}
    for candidate_index in range(5):
        manifest = manifests[f"independent-candidate-{candidate_index}"]
        assert manifest["phase"] == "c2_scientific_utility"
        assert sum(len(wave["items"]) for wave in manifest["waves"]) == 245
        assert max(len(wave["items"]) for wave in manifest["waves"]) <= 28
    assert sum(
        len(wave["items"]) for wave in manifests["independent-candidate-5"]["waves"]
    ) == sum(case["candidate_count"] == 6 for case in cases)
