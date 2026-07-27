from __future__ import annotations

from copy import deepcopy

from scripts.bonafide.topk_rank_screen import (
    screen_candidate_ranks,
    select_rank_screen_items,
)
from tests.test_bonafide_benchmark import _single_item_manifest
from tests.test_teacher_forced_trace import FakeChatTokenizer, FakeModel


def _rank_source_manifest() -> dict:
    source = _single_item_manifest()
    source["waves"][0]["corpus_role"] = "dense_discovery"
    item = source["waves"][0]["items"][0]
    item["example"]["response"] = "abcd"
    item["response_token_count"] = 4
    item["target_selection"]["response_token_positions"] = [2]
    item["target_selection"]["final_target_token_id"] = 79
    item["target_selection"]["width"] = 1
    item["target_selection"].pop("sampling", None)
    item["target_selection"]["final_selection"] = {
        "corpus_role": "dense_discovery",
        "refinement_diagnostics": {"probability": 0.25},
    }
    return source


def test_rank_screen_selection_excludes_holdout_and_orders_probability() -> None:
    source = _rank_source_manifest()
    low = deepcopy(source["waves"][0]["items"][0])
    low["artifact_id"] = "low-probability"
    low["target_selection"]["final_selection"]["refinement_diagnostics"][
        "probability"
    ] = 0.01
    source["waves"][0]["items"].append(low)
    holdout = deepcopy(source["waves"][0])
    holdout["corpus_role"] = "broad_confirmatory_holdout"
    holdout["items"][0]["artifact_id"] = "holdout"
    source["waves"].append(holdout)

    selected = select_rank_screen_items(source, max_items=2)

    assert [item["artifact_id"] for item in selected] == [
        "low-probability",
        "source-trace-1",
    ]


def test_rank_screen_measures_union_width_without_graph_tracing() -> None:
    item = _rank_source_manifest()["waves"][0]["items"][0]

    results = screen_candidate_ranks(FakeModel(), FakeChatTokenizer(), [item])

    assert len(results) == 1
    result = results[0]
    assert result["candidate_count"] == 6
    selection = result["candidate_selection"]
    assert selection["observed_token_rank"] == 49
    assert [candidate["token_id"] for candidate in selection["candidates"]] == [
        79,
        127,
        126,
        125,
        124,
        123,
    ]
