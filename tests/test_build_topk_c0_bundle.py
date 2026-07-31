from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from scripts.bonafide.build_topk_c0_bundle import build_c0_manifests
from tests.test_bonafide_benchmark import _single_item_manifest


def _inputs() -> tuple[dict, dict, dict]:
    source = _single_item_manifest()
    template = source["waves"][0]["items"][0]
    source["waves"] = []
    rank_results = []
    cases = []
    for role, offset in (("dense_discovery", 0), ("broad_discovery", 4)):
        wave = {"corpus_role": role, "items": []}
        for local_index in range(4):
            index = offset + local_index
            item = deepcopy(template)
            item["artifact_id"] = f"source-{index}"
            item["example"]["example_id"] = f"example-{index}"
            wave["items"].append(item)
            width = 5 if index % 2 == 0 else 6
            observed_rank = 1 if width == 5 else 7
            token_ids = [
                item["target_selection"]["final_target_token_id"],
                *range(100 + index * 10, 100 + index * 10 + width - 1),
            ]
            rank_results.append(
                {
                    "source_width1_artifact_id": item["artifact_id"],
                    "corpus_role": role,
                    "candidate_selection": {
                        "policy_id": "model_top5_plus_observed",
                        "observed_token_rank": observed_rank,
                        "candidates": [
                            {"token_id": token_id} for token_id in token_ids
                        ],
                    },
                }
            )
            cases.append(
                {
                    "case_id": f"case-{index}",
                    "source_width1_artifact_id": item["artifact_id"],
                    "corpus_role": role,
                    "candidate_count": width,
                    "observed_token_rank": observed_rank,
                    "selection_reasons": ["test coverage"],
                }
            )
        source["waves"].append(wave)
    source["tokenizer"]["chat_template_sha256"] = "b" * 64
    rank = {
        "schema_version": "bonafide-topk-rank-screen/v1",
        "source_manifest_sha256": "a" * 64,
        "results": rank_results,
    }
    selection = {
        "schema_version": "bonafide-topk-c0-cohort-selection/v1",
        "cohort_id": "test-c0-v1",
        "cases": cases,
    }
    return selection, source, rank


def test_c0_builder_creates_joint_and_candidate_reference_manifests() -> None:
    selection, source, rank = _inputs()

    manifests, cases = build_c0_manifests(
        selection,
        source,
        rank,
        selection_path=Path("/selection.json"),
        selection_sha256="c" * 64,
        source_manifest_path=Path("/source.json"),
        source_manifest_sha256="a" * 64,
        rank_screen_path=Path("/rank.json"),
        rank_screen_sha256="d" * 64,
    )

    assert set(manifests) == {
        "joint-raw",
        "joint-contrastive",
        *(f"independent-candidate-{index}" for index in range(6)),
    }
    assert len(cases) == 8
    assert (
        sum(len(wave["items"]) for wave in manifests["joint-contrastive"]["waves"]) == 8
    )
    candidate_five = manifests["independent-candidate-5"]
    reference_items = [
        item for wave in candidate_five["waves"] for item in wave["items"]
    ]
    assert len(reference_items) == 4
    assert all("specified_candidate_token_id" in item for item in reference_items)
    assert candidate_five["trace_family"]["candidate_count"] == 1
