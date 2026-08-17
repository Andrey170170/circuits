from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import circuits.analysis.bonafide.coarse_sampling_annotation_v2 as module
import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation_v2 import (
    ARM_FULL_UNIT,
    ARM_TARGET_ONLY,
    BUNDLE_SCHEMA,
    COMPARISON_PLAN,
    build_v2_qualification,
    cost_plan_v2,
    load_coarse_v2_config,
    load_v2_qualification,
    openai_batch_line,
    render_full_response_user_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/bonafide/configs/process_witness_coarse_openai_v2.json"


def _document(response_id: str) -> tuple[dict, list[dict]]:
    pieces = [f"unit-{index}" for index in range(8)]
    text = "|".join(pieces)
    prompt = f"Solve {response_id} exactly."
    units = []
    cursor = 0
    for index, piece in enumerate(pieces):
        start = text.index(piece, cursor)
        end = start + len(piece)
        cursor = end
        units.append(
            {
                "unit_id": f"{response_id}-u{index}",
                "response_id": response_id,
                "sequence_index": index,
                "token_span": [index, index + 1],
                "core_character_span": [start, end],
                "covering_character_span": [start, end],
                "text": piece,
            }
        )
    document = {
        "response_id": response_id,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "task_context": {"prompt": prompt},
    }
    return document, units


def _marked_response(prompt: str) -> str:
    match = re.search(
        r"<<<BEGIN_FULL_RESPONSE_SHA256:[0-9a-f]{64}>>>\n(.*?)\n"
        r"<<<END_FULL_RESPONSE_SHA256:[0-9a-f]{64}>>>",
        prompt,
        flags=re.DOTALL,
    )
    assert match
    return match.group(1)


def _strip_unit_markup(value: str) -> str:
    return re.sub(r"\{\{/?(?:TARGET|CONTEXT)\s+[^{}]+\}\}", "", value)


def test_both_markup_arms_are_inline_and_losslessly_reconstruct_full_response() -> None:
    document, units = _document("response-a")
    focal = units[1:7]
    target_prompt, target_audit = render_full_response_user_prompt(
        document, focal, units, arm_id=ARM_TARGET_ONLY
    )
    full_prompt, full_audit = render_full_response_user_prompt(
        document, focal, units, arm_id=ARM_FULL_UNIT
    )

    target_marked = _marked_response(target_prompt)
    full_marked = _marked_response(full_prompt)
    assert target_marked.count("{{TARGET ") == 6
    assert "{{CONTEXT " not in target_marked
    assert full_marked.count("{{TARGET ") == 6
    assert full_marked.count("{{CONTEXT ") == 2
    assert _strip_unit_markup(target_marked) == document["text"]
    assert _strip_unit_markup(full_marked) == document["text"]
    assert target_audit["lossless_reconstruction_verified"] is True
    assert full_audit["response_unit_count"] == 8


def test_full_unit_markup_fails_closed_on_overlapping_or_empty_core_spans() -> None:
    document, units = _document("response-overlap")
    units[7]["core_character_span"] = units[0]["core_character_span"]
    units[7]["text"] = units[0]["text"]
    with pytest.raises(ValueError, match="overlapping or empty"):
        render_full_response_user_prompt(
            document, units[1:7], units, arm_id=ARM_FULL_UNIT
        )


def _v1_fixture(tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
    config = load_coarse_v2_config(CONFIG)
    documents = []
    all_units = []
    windows = []
    for index in range(12):
        document, units = _document(f"response-{index}")
        documents.append(document)
        all_units.extend(units)
        windows.append(
            {
                "window_index": index,
                "response_id": document["response_id"],
                "prompt_sha256": document["prompt_sha256"],
                "focal_unit_ids": [unit["unit_id"] for unit in units[1:7]],
            }
        )
    v1_root = tmp_path / "v1"
    v1_root.mkdir()
    (v1_root / "windows.json").write_text(json.dumps(windows))
    (v1_root / "units.jsonl").write_text(
        "".join(json.dumps(unit) + "\n" for unit in all_units)
    )
    repeat_windows = {0, 5, 7, 9}
    requests = []
    for window in windows:
        primary_id = f"v1-{window['window_index']}-primary"
        body = {"window": window["window_index"]}
        primary = {
            "request_id": primary_id,
            "window_index": window["window_index"],
            "focal_unit_ids": window["focal_unit_ids"],
            "body_sha256": canonical_sha256(body),
            "repeat_of_request_id": None,
        }
        requests.append(primary)
        if window["window_index"] in repeat_windows:
            requests.append(
                {
                    **primary,
                    "request_id": f"v1-{window['window_index']}-repeat",
                    "repeat_of_request_id": primary_id,
                }
            )
    monkeypatch.setattr(
        module,
        "load_offline_qualification",
        lambda _: {
            "manifest": {
                "manifest_sha256": config["source"]["v1_qualification_manifest_sha256"],
                "source_workstation_bundle_sha256": config["source"][
                    "workstation_bundle_sha256"
                ],
            },
            "requests": requests,
        },
    )
    return config, {
        "v1_root": v1_root,
        "workstation": {"documents": documents},
        "windows": windows,
    }


def test_build_reuses_exact_v1_census_for_two_matched_arms(
    tmp_path: Path, monkeypatch
) -> None:
    config, fixture = _v1_fixture(tmp_path, monkeypatch)
    result = build_v2_qualification(
        v1_root=fixture["v1_root"],
        workstation_bundle=fixture["workstation"],
        config=config,
    )
    assert len(result["requests"]) == 32
    assert len(result["batch_lines"]) == 32
    assert len(result["matched_arm_bindings"]) == 32
    assert {request["arm_id"] for request in result["requests"]} == {
        ARM_TARGET_ONLY,
        ARM_FULL_UNIT,
    }
    assert (
        sum(
            request["repeat_of_request_id"] is not None
            for request in result["requests"]
        )
        == 8
    )
    by_id = {request["request_id"]: request for request in result["requests"]}
    for request, line in zip(result["requests"], result["batch_lines"], strict=True):
        assert line == openai_batch_line(request)
        assert line["url"] == "/v1/responses"
        assert line["body"]["reasoning"] == {"effort": "medium"}
        assert line["body"]["max_output_tokens"] == 16384
        assert line["body"]["store"] is False
        assert line["body"]["prompt_cache_key"].startswith("pwcv2-")
        if request["repeat_of_request_id"]:
            assert (
                request["provider_body"]
                == by_id[request["repeat_of_request_id"]]["provider_body"]
            )

    for source_id in {
        item["source_v1_request_id"] for item in result["matched_arm_bindings"]
    }:
        matched = [
            item
            for item in result["matched_arm_bindings"]
            if item["source_v1_request_id"] == source_id
        ]
        assert {item["arm_id"] for item in matched} == {
            ARM_TARGET_ONLY,
            ARM_FULL_UNIT,
        }
        assert matched[0]["focal_unit_ids"] == matched[1]["focal_unit_ids"]


def test_comparison_plan_is_predeclared_and_exactly_frozen() -> None:
    config = load_coarse_v2_config(CONFIG)
    plan = config["comparison_plan"]
    assert plan == COMPARISON_PLAN
    assert plan["status"] == "predeclared_before_submission_not_executed"
    comparisons = {item["comparison_id"]: item for item in plan["comparisons"]}
    assert comparisons["within_arm_repeat"]["unit_pairs_per_arm"] == 24
    assert comparisons["cross_arm_primary"]["unit_pairs"] == 72
    assert comparisons["each_arm_vs_v1_primary"]["unit_pairs_per_arm"] == 72
    assert comparisons["each_arm_vs_v1_repeat"]["unit_pairs_per_arm"] == 24
    assert plan["interpretation"]["formal_pass_threshold"] is None
    assert "do not attribute" in plan["interpretation"]["causal_attribution"]
    assert "cache-read" in plan["required_reporting"]["usage_cost_by_arm"]


def test_config_rejects_comparison_plan_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["comparison_plan"]["comparisons"][1]["unit_pairs"] = 71
    drifted = tmp_path / "drifted-v2-config.json"
    drifted.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="comparison plan drift"):
        load_coarse_v2_config(drifted)


def test_native_batch_cost_plan_uses_worst_input_lane_and_full_output_ceiling() -> None:
    config = load_coarse_v2_config(CONFIG)
    prices = json.loads(
        (
            ROOT / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
        ).read_text()
    )
    requests = [
        {"provider_body": {"input": [{"role": "user", "content": "x" * 100}]}}
        for _ in range(32)
    ]
    plan = cost_plan_v2(requests, config, prices)
    assert plan["transport"] == "native_batch"
    assert plan["request_count"] == 32
    assert plan["output_token_upper_bound"] == 32 * 16384
    assert plan["ordinary_worst_case_input_rate_per_million"] == 0.125
    assert "no prompt-cache savings are assumed" in plan["assumptions"]

    long_plan = cost_plan_v2(
        [
            {
                "request_id": "long-request",
                "provider_body": {
                    "input": [{"role": "user", "content": "x" * 272_001}]
                },
            }
        ],
        config,
        prices,
    )
    assert long_plan["long_context_request_ids_by_byte_bound"] == ["long-request"]
    assert long_plan["long_context_authorization_input_rate_per_million"] == 0.5
    assert long_plan[
        "long_context_authorization_output_rate_per_million"
    ] == pytest.approx(1.8)


def test_frozen_v2_bundle_loader_binds_rows_arms_price_and_exact_file_membership(
    tmp_path: Path, monkeypatch
) -> None:
    config, fixture = _v1_fixture(tmp_path, monkeypatch)
    result = build_v2_qualification(
        v1_root=fixture["v1_root"],
        workstation_bundle=fixture["workstation"],
        config=config,
    )
    bundle = tmp_path / "v2-bundle"
    bundle.mkdir()

    def write_json(name: str, value: object) -> None:
        (bundle / name).write_text(json.dumps(value, sort_keys=True) + "\n")

    def write_jsonl(name: str, values: list[object]) -> None:
        (bundle / name).write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in values)
        )

    write_jsonl("focal-units.jsonl", result["focal_units"])
    write_json("windows.json", result["windows"])
    write_jsonl("requests.jsonl", result["requests"])
    write_jsonl("batch-input.jsonl", result["batch_lines"])
    write_json("matched-arm-bindings.json", result["matched_arm_bindings"])
    price_path = (
        ROOT / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
    )
    prices = json.loads(price_path.read_text())
    plan = cost_plan_v2(result["requests"], config, prices)
    plan.update(
        {
            "price_snapshot_path": str(price_path),
            "price_snapshot_sha256": file_sha256(price_path),
        }
    )
    plan["cost_plan_sha256"] = canonical_sha256(plan)
    write_json("cost-plan.json", plan)
    v1_events = []
    for request in [
        item for item in result["requests"] if item["arm_id"] == ARM_TARGET_ONLY
    ]:
        event = {
            "request_id": request["source_v1_request_id"],
            "status": "success",
            "decisions": [
                {
                    "unit_id": unit_id,
                    "tag": "active_task_work",
                    "confidence": "high",
                    "boundary_concerns": [],
                    "boundary_note": "",
                }
                for unit_id in request["focal_unit_ids"]
            ],
        }
        event["event_sha256"] = canonical_sha256(event)
        v1_events.append(event)
    write_jsonl("v1-baseline-events.jsonl", v1_events)
    v1_events_sha256 = file_sha256(bundle / "v1-baseline-events.jsonl")
    v1_manifest = {
        "schema_version": "adag.process-witness.coarse-openai-run.v1",
        "status": "complete",
        "qualification_manifest_sha256": config["source"][
            "v1_qualification_manifest_sha256"
        ],
        "events_jsonl_sha256": v1_events_sha256,
        "event_count": 16,
        "success_count": 16,
        "record_bindings_in_order": [
            {
                "request_id": event["request_id"],
                "event_sha256": event["event_sha256"],
            }
            for event in v1_events
        ],
    }
    v1_manifest["run_manifest_sha256"] = canonical_sha256(v1_manifest)
    write_json("v1-baseline-run-manifest.json", v1_manifest)
    v1_cost_audit = {
        "schema_version": "adag.process-witness.coarse-cost-correction-audit.v1",
        "status": "offline_correction_preserving_original_run",
        "run_manifest_sha256": v1_manifest["run_manifest_sha256"],
        "run_events_jsonl_sha256": v1_events_sha256,
        "request_count": 16,
        "original_run_mutated": False,
        "requests": [{"request_id": event["request_id"]} for event in v1_events],
    }
    v1_cost_audit["cost_correction_audit_sha256"] = canonical_sha256(v1_cost_audit)
    write_json("v1-baseline-cost-correction-audit.json", v1_cost_audit)
    synthetic_config = json.loads(json.dumps(config))
    synthetic_config["source"]["v1_completed_run_manifest_file_sha256"] = file_sha256(
        bundle / "v1-baseline-run-manifest.json"
    )
    synthetic_config["source"]["v1_completed_run_manifest_sha256"] = v1_manifest[
        "run_manifest_sha256"
    ]
    synthetic_config["source"]["v1_completed_events_sha256"] = v1_events_sha256
    synthetic_config["source"]["v1_cost_correction_audit_file_sha256"] = file_sha256(
        bundle / "v1-baseline-cost-correction-audit.json"
    )
    synthetic_config["source"]["v1_cost_correction_audit_sha256"] = v1_cost_audit[
        "cost_correction_audit_sha256"
    ]
    synthetic_config["comparison_plan"]["baselines"]["v1"].update(
        {
            "completed_run_manifest_sha256": v1_manifest["run_manifest_sha256"],
            "completed_events_sha256": v1_events_sha256,
            "cost_correction_audit_sha256": v1_cost_audit[
                "cost_correction_audit_sha256"
            ],
        }
    )
    monkeypatch.setattr(module, "load_coarse_v2_config", lambda _: synthetic_config)
    files = [
        {
            "path": path.name,
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(bundle.iterdir())
    ]
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "status": "prepared_offline_no_provider_calls",
        "network_calls_made": 0,
        "comparison_plan": synthetic_config["comparison_plan"],
        "comparison_plan_sha256": canonical_sha256(synthetic_config["comparison_plan"]),
        "source_v1_completed_run_manifest_file_sha256": synthetic_config["source"][
            "v1_completed_run_manifest_file_sha256"
        ],
        "source_v1_completed_run_manifest_sha256": v1_manifest["run_manifest_sha256"],
        "source_v1_completed_events_sha256": v1_events_sha256,
        "source_v1_cost_correction_audit_file_sha256": synthetic_config["source"][
            "v1_cost_correction_audit_file_sha256"
        ],
        "source_v1_cost_correction_audit_sha256": v1_cost_audit[
            "cost_correction_audit_sha256"
        ],
        "config_path": str(CONFIG),
        "config_sha256": file_sha256(CONFIG),
        "files": files,
        "request_bindings_in_order": [
            {
                "request_id": request["request_id"],
                "arm_id": request["arm_id"],
                "body_sha256": request["body_sha256"],
                "source_v1_request_id": request["source_v1_request_id"],
                "repeat_of_request_id": request["repeat_of_request_id"],
            }
            for request in result["requests"]
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json("manifest.json", manifest)

    loaded = load_v2_qualification(bundle)
    assert len(loaded["requests"]) == 32
    assert len(loaded["matched_arm_bindings"]) == 32
    assert loaded["cost_plan"]["transport"] == "native_batch"
    assert loaded["manifest"]["comparison_plan"] == synthetic_config["comparison_plan"]
    assert loaded["manifest"]["comparison_plan_sha256"] == canonical_sha256(
        synthetic_config["comparison_plan"]
    )
    assert len(loaded["v1_comparison_baseline"]["events"]) == 16
    assert loaded["v1_cost_correction_audit"]["request_count"] == 16
