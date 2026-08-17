from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import circuits.analysis.bonafide.coarse_sampling_annotation_v3 as module
import pytest
from circuits.analysis.bonafide.coarse_sampling_annotation_v3 import (
    ARM_FEW_SHOT,
    ARM_ZERO_SHOT,
    DESIRED_STRATA,
    _focal_indices_are_consecutive_eligible,
    _fresh_holdout_windows,
    _reconstruct_exact_target_only_user_prompt,
    _request,
    cost_plan_v3,
    forbidden_provider_input_leaks_v3,
    load_coarse_v3_config,
)
from circuits.analysis.bonafide.coarse_sampling_comparison_v3 import (
    _vote_pattern,
    apply_human_gate,
)
from circuits.analysis.bonafide.coarse_sampling_review_v3 import (
    BOUNDARY_DEFINITIONS,
    EXPORT_SCHEMA,
    UI_VERSION,
    build_review_payload,
    render_review_html,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/bonafide/configs/process_witness_coarse_openai_v3.json"


def _seal_human_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    items = [
        {"item_id": f"item-{index}", "unit_id": row["unit_id"]}
        for index, row in enumerate(rows)
    ]
    packet = {
        "packet_id": "packet-v3",
        "packet_binding_sha256": "b" * 64,
        "qualification_manifest_sha256": "q" * 64,
    }
    sealed = []
    for item, row in zip(items, rows, strict=True):
        sealed.append(
            {
                "schema_version": EXPORT_SCHEMA,
                "packet_id": packet["packet_id"],
                "packet_binding_sha256": packet["packet_binding_sha256"],
                "item_id": item["item_id"],
                "unit_id": row["unit_id"],
                "primary_label": row["primary_label"],
                "defensible_alternatives": row["defensible_alternatives"],
                "boundary_concerns": [],
                "note": "",
                "globally_sealed": True,
                "global_seal_id": "00000000-0000-4000-8000-000000000003",
                "global_sealed_at": "2026-08-17T00:00:00Z",
            }
        )
    return sealed, {"packet": packet, "items": items}


def test_config_freezes_factorial_three_replica_shape_and_examples() -> None:
    config = load_coarse_v3_config(CONFIG)
    assert len(DESIRED_STRATA) == 24
    assert len(set(DESIRED_STRATA)) == 24
    assert config["qualification"]["unique_window_count"] == 24
    assert config["qualification"]["replicas_per_arm"] == 3
    assert len(config["few_shot_demonstrations"]) == 11
    assert config["comparison_plan"]["decision_gate"]["three_way_votes"].startswith(
        "a 1-1-1"
    )


def test_global_holdout_matching_handles_scarce_shared_prompt(monkeypatch) -> None:
    documents = []
    units_by_response = {}
    for index, (source, position, hint) in enumerate(DESIRED_STRATA):
        response_id = f"r-{index}"
        prompt_hash = f"p-{index}"
        document = {
            "response_id": response_id,
            "prompt_sha256": prompt_hash,
            "task_context": {"source_types": [source]},
            "tokenization": {"token_count": 100},
        }
        documents.append(document)
        units_by_response[response_id] = [
            {
                "unit_id": f"u-{index}-{offset}",
                "sequence_index": offset,
                "assignment_route": "openai_pending",
                "hint": hint,
                "position": position,
                "core_character_span": [offset * 2, offset * 2 + 1],
            }
            for offset in range(6)
        ]

    # Cell 0 can use either its own prompt or cell 1's prompt. Cell 1 can only
    # use its own prompt. A greedy cell-0-first selector can strand cell 1.
    shared = documents[1]
    shared["task_context"]["source_types"] = ["complex"]
    units_by_response[shared["response_id"]] = [
        {
            "unit_id": f"shared-process-{index}",
            "sequence_index": index,
            "assignment_route": "openai_pending",
            "hint": "process",
            "position": "early",
            "core_character_span": [index * 2, index * 2 + 1],
        }
        for index in range(6)
    ] + [
        {
            "unit_id": f"shared-evaluation-{index}",
            "sequence_index": 6 + index,
            "assignment_route": "openai_pending",
            "hint": "evaluation",
            "position": "early",
            "core_character_span": [12 + index * 2, 13 + index * 2],
        }
        for index in range(6)
    ]
    monkeypatch.setattr(
        module,
        "_hinted_units",
        lambda _document, units, hint: [unit for unit in units if unit["hint"] == hint],
    )
    monkeypatch.setattr(
        module, "_position_bucket", lambda unit, _count: unit["position"]
    )
    config = load_coarse_v3_config(CONFIG)
    windows = _fresh_holdout_windows(
        documents=documents,
        units_by_response=units_by_response,
        excluded_response_ids=set(),
        excluded_prompt_sha256=set(),
        excluded_unit_ids=set(),
        config=config,
    )
    assert len(windows) == 24
    assert len({window["prompt_sha256"] for window in windows}) == 24
    assert windows[1]["prompt_sha256"] == "p-1"
    assert windows[0]["prompt_sha256"] == "p-0"


def _document_and_units() -> tuple[dict, list[dict]]:
    pieces = [f"unit-{index}" for index in range(8)]
    text = "|".join(pieces)
    prompt = "Solve the task."
    units = []
    cursor = 0
    for index, piece in enumerate(pieces):
        start = text.index(piece, cursor)
        end = start + len(piece)
        cursor = end
        units.append(
            {
                "unit_id": f"u-{index}",
                "response_id": "response",
                "sequence_index": index,
                "token_span": [index, index + 1],
                "core_character_span": [start, end],
                "covering_character_span": [start, end],
                "text": piece,
            }
        )
    return (
        {
            "response_id": "response",
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "task_context": {"prompt": prompt},
        },
        units,
    )


def test_requests_are_target_only_full_response_and_exact_replicas() -> None:
    config = load_coarse_v3_config(CONFIG)
    document, units = _document_and_units()
    window = {
        "window_index": 0,
        "response_id": "response",
        "prompt_sha256": document["prompt_sha256"],
        "focal_unit_ids": [unit["unit_id"] for unit in units[1:7]],
    }
    primary = _request(
        physical_index=0,
        arm_id=ARM_FEW_SHOT,
        replica_index=0,
        repeat_of_request_id=None,
        window=window,
        document=document,
        units_by_id={unit["unit_id"]: unit for unit in units},
        all_response_units=units,
        config=config,
    )
    repeat = _request(
        physical_index=1,
        arm_id=ARM_FEW_SHOT,
        replica_index=1,
        repeat_of_request_id=primary["request_id"],
        window=window,
        document=document,
        units_by_id={unit["unit_id"]: unit for unit in units},
        all_response_units=units,
        config=config,
    )
    zero = _request(
        physical_index=2,
        arm_id=ARM_ZERO_SHOT,
        replica_index=0,
        repeat_of_request_id=None,
        window=window,
        document=document,
        units_by_id={unit["unit_id"]: unit for unit in units},
        all_response_units=units,
        config=config,
    )
    assert primary["provider_body"] == repeat["provider_body"]
    assert primary["request_id"] != repeat["request_id"]
    assert primary["markup_audit"]["target_markup_count"] == 6
    assert primary["markup_audit"]["context_markup_count"] == 0
    assert "Example 11" in primary["provider_body"]["input"][0]["content"]
    assert "No labeled demonstrations" in zero["provider_body"]["input"][0]["content"]
    assert not forbidden_provider_input_leaks_v3(primary)
    assert (
        _reconstruct_exact_target_only_user_prompt(primary, units[1:7])
        == primary["provider_body"]["input"][1]["content"]
    )
    tampered = json.loads(json.dumps(primary))
    tampered["provider_body"]["input"][1]["content"] = tampered["provider_body"][
        "input"
    ][1]["content"].replace("Solve the task.", "Solve another task.", 1)
    with pytest.raises(ValueError, match="binding drift"):
        _reconstruct_exact_target_only_user_prompt(tampered, units[1:7])


def test_focal_indices_must_be_consecutive_within_full_eligible_sequence() -> None:
    assert _focal_indices_are_consecutive_eligible(
        [3, 8, 12, 20, 30, 41],
        [1, 3, 8, 12, 20, 30, 41, 50],
    )
    assert not _focal_indices_are_consecutive_eligible(
        [1, 8, 12, 20, 30, 41],
        [1, 3, 8, 12, 20, 30, 41, 50],
    )


def test_cost_plan_covers_144_physical_requests() -> None:
    config = load_coarse_v3_config(CONFIG)
    prices = json.loads(
        (
            ROOT / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
        ).read_text()
    )
    requests = [
        {
            "request_id": f"request-{index}",
            "provider_body": {"input": [{"role": "user", "content": "x" * 100}]},
        }
        for index in range(144)
    ]
    plan = cost_plan_v3(requests, config, prices)
    assert plan["request_count"] == 144
    assert plan["campaign_shape"]["unique_units"] == 144
    assert plan["output_token_upper_bound"] == 144 * 16384


def test_three_way_split_stays_disputed_without_precedence() -> None:
    assert _vote_pattern(
        {"active_task_work": 1, "final_answer": 1, "uncertain": 1}
    ) == (
        "one_one_one_disputed",
        None,
    )


def test_predeclared_human_gate_selects_few_shot_only_after_five_net_wins() -> None:
    vote_rows = []
    human = []
    for index in range(144):
        unit_id = f"unit-{index}"
        human.append(
            {
                "unit_id": unit_id,
                "primary_label": "other_semantic_text",
                "defensible_alternatives": [],
                "globally_sealed": True,
            }
        )
        vote_rows.extend(
            [
                {
                    "arm_id": ARM_ZERO_SHOT,
                    "unit_id": unit_id,
                    "response_id": f"response-{index // 6}",
                    "majority_label": (
                        "active_task_work" if index < 5 else "other_semantic_text"
                    ),
                    "stable_high_confidence": False,
                    "vote_pattern": "two_one_mixed",
                    "majority_boundary_concerns": [],
                },
                {
                    "arm_id": ARM_FEW_SHOT,
                    "unit_id": unit_id,
                    "response_id": f"response-{index // 6}",
                    "majority_label": "other_semantic_text",
                    "stable_high_confidence": False,
                    "vote_pattern": "three_zero_stable",
                    "majority_boundary_concerns": [],
                },
            ]
        )
    human, packet = _seal_human_rows(human)
    result = apply_human_gate(
        {
            "vote_rows": vote_rows,
            "qualification_manifest_sha256": "q" * 64,
        },
        human,
        packet,
    )
    assert result["human_gate"]["net_paired_few_shot_wins"] == 5
    assert result["human_gate"]["few_shot_improved"] is True
    assert result["human_gate"]["selected_arm"] == ARM_FEW_SHOT


def test_human_gate_uses_zero_shot_parsimony_on_four_net_wins() -> None:
    vote_rows = []
    human = []
    for index in range(144):
        unit_id = f"unit-{index}"
        human.append(
            {
                "unit_id": unit_id,
                "primary_label": "other_semantic_text",
                "defensible_alternatives": [],
                "globally_sealed": True,
            }
        )
        vote_rows.extend(
            [
                {
                    "arm_id": ARM_ZERO_SHOT,
                    "unit_id": unit_id,
                    "response_id": f"response-{index // 6}",
                    "majority_label": (
                        "active_task_work" if index < 4 else "other_semantic_text"
                    ),
                    "stable_high_confidence": False,
                    "vote_pattern": "two_one_mixed",
                    "majority_boundary_concerns": [],
                },
                {
                    "arm_id": ARM_FEW_SHOT,
                    "unit_id": unit_id,
                    "response_id": f"response-{index // 6}",
                    "majority_label": "other_semantic_text",
                    "stable_high_confidence": False,
                    "vote_pattern": "three_zero_stable",
                    "majority_boundary_concerns": [],
                },
            ]
        )
    human, packet = _seal_human_rows(human)
    result = apply_human_gate(
        {
            "vote_rows": vote_rows,
            "qualification_manifest_sha256": "q" * 64,
        },
        human,
        packet,
    )
    assert result["human_gate"]["net_paired_few_shot_wins"] == 4
    assert result["human_gate"]["selected_arm"] == ARM_ZERO_SHOT


def test_review_packet_randomizes_response_blocks_and_globally_seals_reveal() -> None:
    config = load_coarse_v3_config(CONFIG)
    windows = []
    units = []
    documents = []
    vote_rows = []
    for block in range(24):
        response_id = f"response-{block}"
        prompt = f"prompt-{block}"
        text = " ".join(f"unit-{block}-{index}" for index in range(6))
        focal = []
        cursor = 0
        for index in range(6):
            value = f"unit-{block}-{index}"
            start = text.index(value, cursor)
            end = start + len(value)
            cursor = end
            unit_id = f"unit-id-{block}-{index}"
            focal.append(unit_id)
            units.append(
                {
                    "unit_id": unit_id,
                    "response_id": response_id,
                    "sequence_index": index,
                    "text": value,
                    "core_character_span": [start, end],
                }
            )
            vote_rows.extend(
                [
                    {
                        "arm_id": arm,
                        "unit_id": unit_id,
                        "vote_pattern": "three_zero_stable",
                        "majority_label": "active_task_work",
                        "votes": [{"tag": "active_task_work"}] * 3,
                    }
                    for arm in (ARM_ZERO_SHOT, ARM_FEW_SHOT)
                ]
            )
        windows.append(
            {
                "window_index": block,
                "response_id": response_id,
                "prompt_sha256": f"p-{block}",
                "focal_unit_ids": focal,
            }
        )
        documents.append(
            {
                "response_id": response_id,
                "text": text,
                "text_sha256": f"t-{block}",
                "prompt_sha256": f"p-{block}",
                "task_context": {"prompt": prompt},
            }
        )
    qualification = {
        "config": config,
        "manifest": {"manifest_sha256": "q" * 64},
        "windows": windows,
        "focal_units": units,
    }
    payload = build_review_payload(
        qualification=qualification,
        workstation_bundle={"documents": documents},
    )
    assert len(payload["items"]) == 144
    assert len(payload["documents"]) == 24
    assert [item["response_id"] for item in payload["items"][::6]] != [
        f"response-{index}" for index in range(24)
    ]
    assert payload["packet"]["ui_version"] == UI_VERSION
    assert payload["packet"]["tag_definitions"] == config["tags"]
    assert payload["packet"]["boundary_definitions"] == BOUNDARY_DEFINITIONS
    assert payload["packet"]["decision_precedence"] == config["decision_precedence"]
    html = render_review_html(payload)
    assert "globally_sealed:true" in html
    assert "Seal all 144 blind decisions" in html
    assert "Complete exact response" in html
    assert "Label reference" in html
    assert "Import progress" in html
    assert "rows.join('\\n')+'\\n'" in html
    script = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert script is not None
    node = shutil.which("node")
    if node is not None:
        subprocess.run(
            [node, "--check"],
            input=script.group(1),
            text=True,
            check=True,
            capture_output=True,
        )
    assert "reveal" not in payload
