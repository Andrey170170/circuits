"""Static contract checks for the first human-selected Qwen observatory wave."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.bonafide.runner import (
    validate_frozen_serialization_contract,
    validate_run_config,
    validate_target_selection,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / (
    "scripts/bonafide/manifests/qwen3_4b_thinking_raw_graph_observatory_v1.json"
)
CONFIG_PATH = REPO_ROOT / (
    "scripts/bonafide/configs/qwen3_4b_thinking_raw_graph_observatory_v1.json"
)
SELECTION_PATH = REPO_ROOT / (
    "scripts/bonafide/selections/qwen3_4b_thinking_raw_graph_observatory_v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_qwen_observatory_wave_has_exact_independent_targets() -> None:
    manifest = _load(MANIFEST_PATH)
    config = _load(CONFIG_PATH)
    selection = _load(SELECTION_PATH)
    validate_run_config(config)

    assert manifest["tokenizer"]["model_id"] == config["model"]["model_id"]
    assert manifest["tokenizer"]["revision"] == config["model"]["revision"]
    assert manifest["teacher_forcing_contract"]["serialization_mode"] == (
        "historical_thinking_continuation"
    )
    assert manifest["execution_contract"] == {
        "claim_boundary": (
            "Exploratory pruned local attribution graphs for selected observed "
            "tokens; not causal or faithfulness verdicts."
        ),
        "merge_graphs": False,
        "objective": "observed_token_logit",
        "target_width": 1,
        "trace_units_are_independent": True,
    }

    items = manifest["waves"][0]["items"]
    positions = [
        item["target_selection"]["response_token_positions"][0] for item in items
    ]
    assert positions == [65, 88, 120, 135, 162, 181, 184]
    assert positions == selection["selection_policy"]["approved_response_positions"]
    assert len({item["artifact_id"] for item in items}) == 7
    assert {item["example"]["example_id"] for item in items} == {
        selection["selection_policy"]["completion_id"]
    }
    for item in items:
        validate_target_selection(item)
        validate_frozen_serialization_contract(item, manifest)
        assert item["target_selection"]["width"] == 1
        assert item["objective"] == {
            "benchmark_only_multi_target": False,
            "name": "sum_selected_logits",
        }


def test_position_120_is_frozen_as_first_subtoken_of_displayed_45() -> None:
    manifest = _load(MANIFEST_PATH)
    item = next(
        item
        for item in manifest["waves"][0]["items"]
        if item["target_selection"]["response_token_positions"] == [120]
    )
    selection = item["target_selection"]
    assert selection["final_target_token_id"] == 19
    assert selection["human_selection"]["surface_text"] == "4"
    assert selection["human_selection"]["response_tokens_before"] == 120
