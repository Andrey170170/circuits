from __future__ import annotations

from copy import deepcopy

from scripts.bonafide.build_t5_corpus_bundle import (
    build_pass1_manifests,
    resolve_cases,
)
from tests.test_bonafide_benchmark import _single_item_manifest


def _source_and_rank():
    source = _single_item_manifest()
    source["tokenizer"]["chat_template_sha256"] = "d" * 64
    source["waves"][0]["corpus_role"] = "primary_discovery"
    item = source["waves"][0]["items"][0]
    item["example"]["base_question_id"] = "family-1"
    item["target_selection"].pop("sampling", None)
    item["target_selection"]["final_selection"] = {"corpus_role": "primary_discovery"}
    second = deepcopy(item)
    second["artifact_id"] = "source-trace-2"
    second["example"] = {**second["example"], "example_id": "example-2"}
    source["waves"][0]["items"].append(second)

    def result(source_item, width):
        observed = source_item["target_selection"]["final_target_token_id"]
        tokens = [observed, 101, 102, 103, 104, 105][:width]
        return {
            "source_width1_artifact_id": source_item["artifact_id"],
            "example_id": source_item["example"]["example_id"],
            "corpus_role": "primary_discovery",
            "target_response_position": source_item["target_selection"][
                "response_token_positions"
            ][0],
            "input_token_count": 128,
            "candidate_count": width,
            "candidate_selection": {
                "policy_id": "model_top5_plus_observed",
                "observed_token_rank": 1 if width == 5 else 9,
                "candidates": [{"token_id": token} for token in tokens],
            },
        }

    rank = {
        "schema_version": "bonafide-topk-rank-screen/v1",
        "results": [result(item, 5), result(second, 6)],
    }
    return source, rank


def test_t5_bundle_resolves_every_source_and_realized_width() -> None:
    source, rank = _source_and_rank()

    cases = resolve_cases(source, rank)

    assert [case["candidate_count"] for case in cases] == [5, 6]
    assert (
        cases[0]["candidate_token_ids"][0]
        == source["waves"][0]["items"][0]["target_selection"]["final_target_token_id"]
    )


def test_t5_bundle_builds_six_independent_candidate_manifests(tmp_path) -> None:
    source, rank = _source_and_rank()
    cases = resolve_cases(source, rank)

    manifests = build_pass1_manifests(
        cases,
        source,
        source_manifest_path=tmp_path / "source.json",
        source_manifest_sha256="a" * 64,
        rank_screen_path=tmp_path / "rank.json",
        rank_screen_sha256="b" * 64,
        selection_path=tmp_path / "selection.json",
        selection_sha256="c" * 64,
        cohort_id="t5-test",
        max_items_per_wave=1,
    )

    assert len(manifests) == 6
    assert (
        sum(
            len(wave["items"]) for wave in manifests["independent-candidate-0"]["waves"]
        )
        == 2
    )
    assert (
        sum(
            len(wave["items"]) for wave in manifests["independent-candidate-5"]["waves"]
        )
        == 1
    )
    for manifest in manifests.values():
        assert manifest["phase"] == "matched_corpus"
