from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.coarse_sampling_production_v1 import (
    assign_response_shards,
    broad_family,
    load_production_config,
    openai_batch_line,
    production_request,
    production_units,
    proposal_from_votes,
    response_windows,
    sampling_groups,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/bonafide/configs/process_witness_coarse_production_v1.json"


def _document(text: str) -> dict[str, Any]:
    tokens = [
        [index, match.start(), match.end()]
        for index, match in enumerate(re.finditer(r"\S+", text))
    ]
    prompt = "Solve this public task."
    return {
        "response_id": f"response-{hashlib.sha256(text.encode()).hexdigest()[:8]}",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "task_context": {"prompt": prompt},
        "response_source": "test",
        "trace_scope": "thinking",
        "source_annotation_record_sha256": "c" * 64,
        "tokenization": {
            "token_count": len(tokens),
            "tokens": tokens,
            "input_ids_sha256": "a" * 64,
            "offset_mapping_sha256": "b" * 64,
        },
    }


def test_config_freezes_v4_request_shape_and_broad_projection(tmp_path: Path) -> None:
    config = load_production_config(CONFIG)
    assert config["segmentation"]["maximum_semantic_unit_tokens"] == 96
    assert config["request_protocol"]["maximum_focal_units_per_window"] == 6
    assert config["request_protocol"]["replicas_per_window"] == 3
    assert config["sharding"]["maximum_batch_input_bytes"] == 180_000_000
    assert config["launch_gates"] == {
        "fresh_run_specific_spend_authorization_required": True,
        "provider_batch_queued_input_token_limit_must_be_recorded": True,
        "maximum_failed_only_recovery_waves": 1,
        "fresh_recovery_authorization_required": True,
    }
    assert broad_family("active_task_work") == "process_bearing"
    assert broad_family("other_semantic_text") == "contextual"
    assert broad_family("uncertain") == "unresolved"
    drifted = {**config, "launch_gates": {**config["launch_gates"]}}
    drifted["launch_gates"]["fresh_recovery_authorization_required"] = False
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="launch-gate contract drift"):
        load_production_config(path)


def test_windows_are_response_local_consecutive_and_replicas_are_body_identical() -> (
    None
):
    config = load_production_config(CONFIG)
    document = _document("One. Two. Three. Four. Five. Six. Seven.")
    units = production_units(document)
    windows = response_windows(document, units, window_start=14)
    assert [len(window["focal_unit_ids"]) for window in windows] == [6, 1]
    assert windows[0]["focal_sequence_indices"] == list(range(6))
    focal_by_id = {unit["unit_id"]: unit for unit in units}
    focal = [focal_by_id[unit_id] for unit_id in windows[0]["focal_unit_ids"]]
    primary = production_request(
        physical_index=0,
        replica_index=0,
        window=windows[0],
        document=document,
        focal=focal,
        all_units=units,
        config=config,
        primary_request_id=None,
    )
    repeat = production_request(
        physical_index=1,
        replica_index=1,
        window=windows[0],
        document=document,
        focal=focal,
        all_units=units,
        config=config,
        primary_request_id=primary["request_id"],
    )
    assert primary["provider_body"] == repeat["provider_body"]
    assert primary["request_id"] != repeat["request_id"]
    assert repeat["repeat_of_request_id"] == primary["request_id"]
    assert openai_batch_line(primary)["body"] == primary["provider_body"]
    assert primary["provider_body"]["reasoning"] == {"effort": "medium"}
    assert primary["provider_body"]["max_output_tokens"] == 16384
    user = primary["provider_body"]["input"][1]["content"]
    assert user.count("{{TARGET ") == 6
    assert "COMPLETE MODEL RESPONSE" in user


def test_response_affinity_ffd_preserves_blocks_and_byte_guard() -> None:
    blocks = [
        {"response_index": 0, "response_id": "a", "bytes": 70},
        {"response_index": 1, "response_id": "b", "bytes": 60},
        {"response_index": 2, "response_id": "c", "bytes": 40},
        {"response_index": 3, "response_id": "d", "bytes": 30},
    ]
    shards = assign_response_shards(blocks, 101)
    assert [[b["response_id"] for b in shard] for shard in shards] == [
        ["a", "d"],
        ["b", "c"],
    ]
    assert sorted(b["response_id"] for shard in shards for b in shard) == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert all(sum(b["bytes"] for b in shard) < 101 for shard in shards)
    with pytest.raises(ValueError, match="one response block"):
        assign_response_shards(
            [{"response_index": 0, "response_id": "x", "bytes": 101}], 101
        )


def test_proposal_preserves_all_votes_and_broad_tie_is_unresolved() -> None:
    unit = production_units(_document("Compute it."))[0]
    votes = [
        {
            "replica_index": 0,
            "tag": "active_task_work",
            "confidence": "high",
            "boundary_concerns": [],
            "boundary_note": "",
        },
        {
            "replica_index": 1,
            "tag": "evaluation_or_revision",
            "confidence": "medium",
            "boundary_concerns": [],
            "boundary_note": "",
        },
        {
            "replica_index": 2,
            "tag": "other_semantic_text",
            "confidence": "low",
            "boundary_concerns": ["meaning_unclear"],
            "boundary_note": "ambiguous",
        },
    ]
    proposal = proposal_from_votes(unit, votes)
    assert proposal["fine_votes"] == [
        "active_task_work",
        "evaluation_or_revision",
        "other_semantic_text",
    ]
    assert proposal["fine_one_one_one"] is True
    assert proposal["broad_majority"] == "process_bearing"
    assert proposal["physical_votes"] == votes


def test_sampling_groups_fragment_first_then_equal_broad_across_surface() -> None:
    document = _document("Alpha.   Beta.")
    units = production_units(document)
    proposals = []
    for unit in units:
        if unit["assignment_route"] == "openai_pending":
            votes = [
                {
                    "replica_index": index,
                    "tag": "active_task_work",
                    "confidence": "high",
                    "boundary_concerns": [],
                    "boundary_note": "",
                }
                for index in range(3)
            ]
        else:
            votes = []
        proposals.append(proposal_from_votes(unit, votes))
    groups = sampling_groups(units, proposals)
    assert len(groups) == 1
    assert groups[0]["broad_family"] == "process_bearing"
    assert groups[0]["member_unit_ids"] == [unit["unit_id"] for unit in units]
    assert canonical_sha256(json.loads(json.dumps(groups[0])))
