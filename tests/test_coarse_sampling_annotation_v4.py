from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from circuits.analysis.bonafide.coarse_sampling_annotation import segment_document
from circuits.analysis.bonafide.coarse_sampling_annotation_v4 import (
    SEGMENTATION_POLICY_ID,
    _request,
    _select_windows,
    _text_units_quote_aware,
    build_v4_qualification,
    decision_json_schema_v4,
    load_coarse_v4_config,
    segment_document_v4,
)
from circuits.analysis.bonafide.coarse_sampling_review_v4 import (
    build_review_payload,
    render_review_html,
)
from circuits.analysis.bonafide.process_annotation import _text_units

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/bonafide/configs/process_witness_coarse_openai_v4.json"
WORKSTATION = Path(
    "/scratch/general/vast/u1653998/circuits/results/process_witness/annotations/process-witness-graph-blind-auto-v9/workstation-bundle.json"
)
V3_REVIEW = Path(
    "/scratch/general/vast/u1653998/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-v3/qualification-refined-zero-vs-few-shot-v1-human-review-v2"
)
V3_LEDGER = Path(
    "/scratch/general/vast/u1653998/circuits/results/process_witness/coarse_annotation/process-witness-coarse-openai-v4/qualification-96-token-compatibility-v1/v3-human-ledger.jsonl"
)
CORRECTIONS = (
    ROOT / "experiments/process_witness/v3_human_review_post_seal_corrections_v1.jsonl"
)


def _document(text: str) -> dict:
    tokens = []
    for index, match in enumerate(__import__("re").finditer(r"\S+", text)):
        tokens.append([index, match.start(), match.end()])
    prompt = "Solve this public task."
    return {
        "response_id": "response-v4",
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


def test_quote_aware_split_fixes_exact_failure_without_changing_legacy() -> None:
    text = (
        'Wait, the problem says: "Stop when you reach a node with no outgoing '
        'edges." So if we reach H, and there are no edges from H, then we stop.'
    )
    assert len(_text_units(text)) == 1
    spans = _text_units_quote_aware(text)
    assert [text[start:end] for start, end in spans] == [
        'Wait, the problem says: "Stop when you reach a node with no outgoing edges."',
        "So if we reach H, and there are no edges from H, then we stop.",
    ]


def test_quote_aware_split_is_conservative_for_inline_quotes() -> None:
    examples = (
        'The "Decision: weight=..." value is 7.',
        'Sentence: "The map..." → yes.',
    )
    for text in examples:
        assert _text_units_quote_aware(text) == [(0, len(text))]
    curly = 'She said "Done.” Then we continued.'
    assert [curly[a:b] for a, b in _text_units_quote_aware(curly)] == [
        'She said "Done.”',
        "Then we continued.",
    ]


def test_v4_segments_exactly_and_uses_a_new_96_token_policy() -> None:
    text = 'Value 3.14 stays here. She said "Done." Then we continued.'
    document = _document(text)
    legacy = segment_document(document, maximum_semantic_unit_tokens=96)
    units = segment_document_v4(document)
    assert units[0]["token_span"][0] == 0
    assert units[-1]["token_span"][1] == document["tokenization"]["token_count"]
    assert [unit["token_span"][0] for unit in units[1:]] == [
        unit["token_span"][1] for unit in units[:-1]
    ]
    assert all(
        unit["segmentation_policy"]["policy_id"] == SEGMENTATION_POLICY_ID
        and unit["segmentation_policy"]["maximum_semantic_unit_tokens"] == 96
        for unit in units
    )
    assert [u["core_character_span"] for u in units] != [
        u["core_character_span"] for u in legacy
    ]
    assert len({u["unit_id"] for u in units}) == len(units)


def test_v4_config_and_request_freeze_two_targets_three_replicas() -> None:
    config = load_coarse_v4_config(CONFIG)
    assert config["qualification"]["maximum_semantic_unit_tokens"] == 96
    assert config["qualification"]["maximum_focal_units_per_window"] == 6
    assert decision_json_schema_v4(2)["properties"]["decisions"]["minItems"] == 2
    document = _document("First sentence. Second sentence. Third sentence.")
    units = [
        u
        for u in segment_document_v4(document)
        if u["assignment_route"] == "openai_pending"
    ]
    assert len(units) == 3
    window = {
        "window_index": 0,
        "response_id": document["response_id"],
        "prompt_sha256": document["prompt_sha256"],
        "focal_unit_ids": [u["unit_id"] for u in units[:2]],
    }
    primary = _request(
        physical_index=0,
        replica_index=0,
        repeat_of_request_id=None,
        window=window,
        document=document,
        focal=units[:2],
        all_units=units,
        config=config,
    )
    repeat = _request(
        physical_index=1,
        replica_index=1,
        repeat_of_request_id=primary["request_id"],
        window=window,
        document=document,
        focal=units[:2],
        all_units=units,
        config=config,
    )
    assert primary["provider_body"] == repeat["provider_body"]
    assert primary["request_id"] != repeat["request_id"]
    assert primary["markup_audit"]["target_markup_count"] == 2
    assert (
        "No labeled demonstrations" in primary["provider_body"]["input"][0]["content"]
    )
    schema = primary["provider_body"]["text"]["format"]["schema"]
    assert schema == decision_json_schema_v4(2)
    json.dumps(primary)


def test_v4_blind_review_has_24_items_and_no_model_output() -> None:
    config = load_coarse_v4_config(CONFIG)
    documents = []
    windows = []
    focal_units = []
    for index in range(15):
        text = (
            f"Repair sentence {index}. Control sentence {index}."
            if index < 9
            else f"Repair sentence {index}."
        )
        document = _document(text)
        document["response_id"] = f"response-{index}"
        units = segment_document_v4(document)
        for unit in units:
            unit["response_id"] = document["response_id"]
        documents.append(document)
        focal = [u for u in units if u["assignment_route"] == "openai_pending"]
        assert len(focal) == (2 if index < 9 else 1)
        focal_units.extend(focal)
        windows.append(
            {
                "window_index": index,
                "response_id": document["response_id"],
                "focal_unit_ids": [u["unit_id"] for u in focal],
            }
        )
    qualification = {
        "manifest": {"manifest_sha256": "d" * 64},
        "config": config,
        "windows": windows,
        "focal_units": focal_units,
    }
    payload = build_review_payload(
        qualification=qualification,
        workstation_bundle={"documents": documents},
    )
    assert payload["packet"]["counts"] == {"response_blocks": 15, "items": 24}
    serialized = json.dumps(payload)
    assert "provider_body" not in serialized
    assert "provider_response" not in serialized
    assert '"decisions"' not in serialized
    html = render_review_html(payload)
    assert "Save all 24 judgments" in html
    assert "Seal all 24 blind decisions" in html


def test_real_v4_panel_builds_exact_predeclared_shape() -> None:
    if not all(path.exists() for path in (WORKSTATION, V3_REVIEW, V3_LEDGER)):
        pytest.skip("frozen CHPC v4 source artifacts unavailable")
    qualification = build_v4_qualification(
        workstation_bundle=json.loads(WORKSTATION.read_text(encoding="utf-8")),
        review_root=V3_REVIEW,
        human_ledger_path=V3_LEDGER,
        correction_path=CORRECTIONS,
        config=load_coarse_v4_config(CONFIG),
    )
    assert len(qualification["windows"]) == 15
    assert len(qualification["focal_units"]) == 24
    assert len(qualification["requests"]) == 45
    roles = [
        role
        for window in qualification["windows"]
        for role in window["target_roles"].values()
    ]
    assert roles.count("repair") == 14
    assert roles.count("unchanged_short") == 6
    assert roles.count("long_diagnostic") == 4
    audit = qualification["full_corpus_segmentation_audit"]
    assert (
        audit["changed_response_count"],
        audit["added_quote_boundary_count"],
        audit["removed_legacy_boundary_count"],
        audit["all_added_boundaries_token_aligned"],
    ) == (77, 162, 0, True)


def test_v4_selection_rejects_tampered_v3_review_packet(tmp_path: Path) -> None:
    if not V3_REVIEW.exists():
        pytest.skip("frozen CHPC v3 review packet unavailable")
    tampered = tmp_path / "review"
    shutil.copytree(V3_REVIEW, tampered)
    items = tampered / "items.jsonl"
    items.chmod(0o644)
    items.write_text(items.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="review file drift"):
        _select_windows(
            workstation_bundle={"documents": []},
            review_root=tampered,
            ledger_rows=[],
            correction_rows=[],
            config=load_coarse_v4_config(CONFIG),
        )
