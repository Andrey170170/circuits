from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from scripts.bonafide.build_topk_c2_screen_pool import (
    C2_SCREEN_ITEM_COUNT,
    build_c2_screen_pool,
)
from tests.test_bonafide_benchmark import _single_item_manifest


def _source() -> dict:
    source = _single_item_manifest()
    template = source["waves"][0]["items"][0]
    waves = []
    for response_index in range(35):
        role = "dense_discovery" if response_index < 11 else "broad_discovery"
        wave = {
            "wave_id": f"wave-{response_index}",
            "corpus_role": role,
            "items": [],
        }
        for position in range(14):
            item = deepcopy(template)
            item["artifact_id"] = f"source-{response_index:02d}-{position:02d}"
            item["example"]["example_id"] = f"response-{response_index:02d}"
            item["example"]["base_question_id"] = (
                "family-shared"
                if response_index in {0, 1}
                else f"family-{response_index:02d}"
            )
            item["example"]["response"] = "response"
            item["response_token_count"] = 14
            item["target_selection"]["response_token_positions"] = [position]
            item["target_selection"].pop("sampling", None)
            item["target_selection"]["final_selection"] = {
                "corpus_role": role,
                "refinement_diagnostics": {"probability": (position + 1) / 100.0},
            }
            wave["items"].append(item)
        waves.append(wave)
    holdout = deepcopy(waves[-1])
    holdout["wave_id"] = "holdout"
    holdout["corpus_role"] = "broad_confirmatory_holdout"
    for item in holdout["items"]:
        item["artifact_id"] = "holdout-" + item["artifact_id"]
    extreme = deepcopy(waves[0])
    extreme["wave_id"] = "extreme"
    extreme["extreme_workload_isolation"] = True
    for item in extreme["items"]:
        item["artifact_id"] = "extreme-" + item["artifact_id"]
    source["waves"] = [*waves, holdout, extreme]
    return source


def test_c2_screen_pool_is_response_and_phase_balanced() -> None:
    pool = build_c2_screen_pool(
        _source(),
        source_manifest_path=Path("/source.json"),
        source_manifest_sha256="a" * 64,
    )

    assert len(pool["cases"]) == C2_SCREEN_ITEM_COUNT == 490
    assert pool["selection_contract"]["response_count"] == 35
    assert pool["selection_contract"]["base_question_family_count"] == 34
    assert pool["selection_contract"]["final_trace_count"] == 245
    assert {(case["example_id"], case["phase_bin"]) for case in pool["cases"]} == {
        (f"response-{response_index:02d}", phase_bin)
        for response_index in range(35)
        for phase_bin in range(7)
    }
    assert {case["screen_slot"] for case in pool["cases"]} == {
        "minimum_probability",
        "temporal_center",
    }
    assert all(
        not case["source_width1_artifact_id"].startswith(("holdout-", "extreme-"))
        for case in pool["cases"]
    )


def test_c2_screen_pool_fails_if_a_response_has_too_few_targets() -> None:
    source = _source()
    source["waves"][0]["items"] = source["waves"][0]["items"][:13]

    with pytest.raises(ValueError, match="two regular targets"):
        build_c2_screen_pool(
            source,
            source_manifest_path=Path("/source.json"),
            source_manifest_sha256="a" * 64,
        )
