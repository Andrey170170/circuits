from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from scripts.bonafide.build_topk_c1_bundle import build_c1_manifests
from tests.test_bonafide_benchmark import _single_item_manifest


def _inputs() -> tuple[dict, dict, dict]:
    source = _single_item_manifest()
    template = source["waves"][0]["items"][0]
    source["waves"] = []
    rank_results = []
    cases = []
    for role, offset in (("dense_discovery", 0), ("broad_discovery", 16)):
        wave = {"corpus_role": role, "items": []}
        for local_index in range(16):
            index = offset + local_index
            item = deepcopy(template)
            item["artifact_id"] = f"source-{index}"
            item["example"]["example_id"] = f"example-{index}"
            item["example"]["base_question_id"] = f"family-{index}"
            item["example"]["diversity"] = {
                "cot_phenotype": ("faithful", "omission", "commission", "both")[
                    index % 4
                ]
            }
            item["example"]["hint_types"] = ["test"]
            wave["items"].append(item)
            # This gives 17 width-five and 15 width-six cases overall, with
            # at least seven in every role/width cell.
            width = 6 if index % 2 == 1 and index != 31 else 5
            observed_rank = 2 if width == 5 else 7
            token_ids = [
                item["target_selection"]["final_target_token_id"],
                *range(100 + index * 10, 100 + index * 10 + width - 1),
            ]
            rank_results.append(
                {
                    "source_width1_artifact_id": item["artifact_id"],
                    "corpus_role": role,
                    "input_token_count": 100 + index,
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
                    "case_id": f"c1-case-{index:02d}",
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
        "schema_version": "bonafide-topk-c1-cohort-selection/v1",
        "cohort_id": "test-c1-v1",
        "cases": cases,
    }
    return selection, source, rank


def test_c1_builder_creates_six_balanced_independent_manifests() -> None:
    selection, source, rank = _inputs()

    manifests, cases, balance = build_c1_manifests(
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

    assert set(manifests) == {*(f"independent-candidate-{index}" for index in range(6))}
    assert len(cases) == 32
    assert balance["role_counts"] == {
        "broad_discovery": 16,
        "dense_discovery": 16,
    }
    assert balance["width_counts"] == {"5": 17, "6": 15}
    assert (
        sum(
            len(wave["items"]) for wave in manifests["independent-candidate-0"]["waves"]
        )
        == 32
    )
    candidate_five = manifests["independent-candidate-5"]
    reference_items = [
        item for wave in candidate_five["waves"] for item in wave["items"]
    ]
    assert len(reference_items) == 15
    assert all("specified_candidate_token_id" in item for item in reference_items)
    assert all(
        manifest["phase"] == "c1_policy_resource" for manifest in manifests.values()
    )
