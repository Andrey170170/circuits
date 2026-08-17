from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    COARSE_TAGS,
    build_qualification,
    cost_plan,
    decision_json_schema,
    forbidden_provider_input_leaks,
    load_coarse_config,
    segment_document,
    validate_decisions,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/bonafide/configs/process_witness_coarse_openai_v1.json"


def _sha(value: object) -> str:
    return canonical_sha256(value)


def _document(
    response_id: str,
    *,
    source_type: str = "complex",
    text: str = 'First work.\nSecond result.\n</think>\n{"final_answer":"42"}',
    trace_scope: str = "full_assistant_serialization",
) -> dict[str, object]:
    tokens = [
        [ord(character), index, index + 1] for index, character in enumerate(text)
    ]
    token_ids = [token[0] for token in tokens]
    offsets = [[token[1], token[2]] for token in tokens]
    return {
        "response_id": response_id,
        "response_source": "test",
        "trace_scope": trace_scope,
        "prompt_sha256": hashlib.sha256(response_id.encode()).hexdigest(),
        "task_context": {
            "prompt": f"Solve {response_id}",
            "prompt_sha256": hashlib.sha256(response_id.encode()).hexdigest(),
            "source_types": [source_type],
            "question_ids": [],
            "task_family": None,
            "historical_process_role": None,
        },
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "source_annotation_record_sha256": "a" * 64,
        "tokenization": {
            "identity_status": "captured",
            "token_count": len(tokens),
            "input_ids_sha256": _sha(token_ids),
            "offset_mapping_sha256": _sha(offsets),
            "tokens": tokens,
        },
        "machine_layers": {},
    }


def test_segment_document_exactly_partitions_and_routes_terminal() -> None:
    document = _document("response-a")
    units = segment_document(document, maximum_semantic_unit_tokens=96)
    observed = [index for unit in units for index in range(*unit["token_span"])]
    assert observed == list(range(document["tokenization"]["token_count"]))  # type: ignore[index]
    assert any(unit["assignment_route"] == "deterministic_surface" for unit in units)
    final = [
        unit
        for unit in units
        if unit["assignment_route"] == "deterministic_terminal_serialization"
    ]
    assert len(final) == 1
    assert final[0]["deterministic_tag"] == "final_answer"


def test_reasoning_only_does_not_fabricate_final_answer() -> None:
    document = _document(
        "response-b", text="Work.\nNo serialized answer.", trace_scope="reasoning_only"
    )
    units = segment_document(document, maximum_semantic_unit_tokens=4)
    assert not any(unit["deterministic_tag"] == "final_answer" for unit in units)
    assert any(unit["fragment_of"] is not None for unit in units)


def test_partition_preserves_overlapping_and_zero_width_token_identities() -> None:
    document = _document("response-overlap", text="ab")
    tokens = [[1, 0, 1], [2, 0, 1], [3, 1, 1], [4, 1, 2]]
    document["tokenization"] = {
        "identity_status": "captured",
        "token_count": 4,
        "input_ids_sha256": _sha([1, 2, 3, 4]),
        "offset_mapping_sha256": _sha([[0, 1], [0, 1], [1, 1], [1, 2]]),
        "tokens": tokens,
    }
    units = segment_document(document, maximum_semantic_unit_tokens=96)
    assert [index for unit in units for index in range(*unit["token_span"])] == [
        0,
        1,
        2,
        3,
    ]


def test_decision_validator_requires_exact_unique_coverage() -> None:
    focal = ["u1", "u2"]
    value = {
        "decisions": [
            {
                "unit_id": unit_id,
                "tag": "active_task_work",
                "confidence": "high",
                "boundary_concerns": [],
                "boundary_note": "",
            }
            for unit_id in reversed(focal)
        ]
    }
    assert [
        item["unit_id"] for item in validate_decisions(value, focal_unit_ids=focal)
    ] == focal
    invalid = copy.deepcopy(value)
    invalid["decisions"][0]["unit_id"] = "u1"
    with pytest.raises(ValueError, match="duplicate"):
        validate_decisions(invalid, focal_unit_ids=focal)
    schema = decision_json_schema(focal)
    assert schema["properties"]["decisions"]["items"]["properties"]["tag"][
        "enum"
    ] == list(COARSE_TAGS)


def _qualification_document(index: int) -> dict[str, object]:
    desired = [
        ("complex", "early", "process"),
        ("graph", "middle", "process"),
        ("complex", "late", "evaluation"),
        ("graph", "early", "evaluation"),
        ("complex", "middle", "commitment"),
        ("graph", "late", "commitment"),
        ("complex", "early", "other"),
        ("graph", "middle", "other"),
        ("graph", "late", "process"),
        ("complex", "middle", "evaluation"),
        ("graph", "early", "commitment"),
        ("complex", "late", "other"),
    ]
    source_type, position, hint = desired[index]
    text = "\n".join(f"Unit {number} does work." for number in range(30))
    document = _document(f"qualification-{index}", source_type=source_type, text=text)
    tokens = document["tokenization"]["tokens"]  # type: ignore[index]
    token_count = len(tokens)
    if position == "early":
        run = [0, token_count // 3, "candidate"]
    elif position == "middle":
        run = [token_count // 3, 2 * token_count // 3, "candidate"]
    else:
        run = [2 * token_count // 3, token_count, "candidate"]
    layers: dict[str, list[list[object]]] = {
        "process_span": [],
        "discourse_phase": [],
    }
    if hint == "process":
        layers["process_span"] = [run]
    elif hint == "evaluation":
        run[2] = "verification"
        layers["discourse_phase"] = [run]
    elif hint == "commitment":
        run[2] = "conclusion"
        layers["discourse_phase"] = [run]
    else:
        run[2] = "planning"
        layers["discourse_phase"] = [run]
    document["machine_layers"] = layers
    return document


def test_qualification_is_balanced_blind_and_repeats_exact_bodies() -> None:
    config = load_coarse_config(CONFIG)
    bundle = {"documents": [_qualification_document(index) for index in range(12)]}
    result = build_qualification(bundle, config)
    assert len(result["windows"]) == 12
    assert len(result["review_units"]) == 72
    assert len({unit["unit_id"] for unit in result["review_units"]}) == 72
    assert len(result["requests"]) == 16
    strata = {
        (window["source_type_stratum"], window["v9_hint_stratum_hidden_from_provider"])
        for window in result["windows"]
    }
    assert strata == {
        (source, hint)
        for source in ("complex", "graph")
        for hint in ("process", "evaluation", "commitment", "other")
    }
    by_id = {request["request_id"]: request for request in result["requests"]}
    repeated = [
        request for request in result["requests"] if request["repeat_of_request_id"]
    ]
    assert {
        request["selection_strata"]["v9_hint_hidden_from_provider"]
        for request in repeated
    } == {"process", "evaluation", "commitment", "other"}
    assert {
        request["selection_strata"]["source_type"] for request in repeated
    } == {"complex", "graph"}
    for request in result["requests"]:
        assert forbidden_provider_input_leaks(request) == []
        if request["repeat_of_request_id"]:
            original = by_id[request["repeat_of_request_id"]]
            assert request["provider_body"] == original["provider_body"]
    assert any(request["repeat_of_request_id"] for request in result["requests"][:8])


def test_cost_plan_is_conservative_and_binds_live_price() -> None:
    config = load_coarse_config(CONFIG)
    request = {
        "provider_body": {
            "input": [{"role": "user", "content": "x" * 100}],
        }
    }
    prices = {
        "snapshot_id": "test",
        "rates": {
            "openai": {
                "gpt-5.6-luna": {
                    "live": {
                        "input_per_million": 0.2,
                        "output_per_million": 1.2,
                    }
                }
            }
        },
    }
    plan = cost_plan([request], config, prices)
    assert plan["input_token_upper_bound"] > plan["provider_body_utf8_bytes"]
    assert plan["output_token_upper_bound"] == 3200
    assert plan["projected_upper_bound_usd"] > 0


def test_config_tag_contract_is_exact() -> None:
    config = load_coarse_config(CONFIG)
    assert tuple(config["tags"]) == COARSE_TAGS
    assert json.loads(CONFIG.read_text())["provider"]["store"] is False
