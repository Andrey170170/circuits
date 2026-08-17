from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import circuits.analysis.bonafide.coarse_sampling_review_v2 as module
import pytest
from circuits.analysis.bonafide.canonical import file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation_v2 import (
    ARM_FULL_UNIT,
    ARM_TARGET_ONLY,
)
from circuits.analysis.bonafide.coarse_sampling_review_v2 import (
    assemble_review_payload,
    build_review_packet_v2,
    render_review_html,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        )
    )


def _decision(unit_id: str, tag: str = "active_task_work") -> dict:
    return {
        "unit_id": unit_id,
        "tag": tag,
        "confidence": "high",
        "boundary_concerns": [],
        "boundary_note": "",
    }


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    qualification_root = tmp_path / "qualification"
    run_root = tmp_path / "run"
    comparison_root = tmp_path / "comparison"
    v1_root = tmp_path / "v1"
    for path in (qualification_root, run_root, comparison_root, v1_root):
        path.mkdir()
    documents = []
    units = []
    windows = []
    requests = []
    v2_events = []
    v1_events = []
    repeat_windows = {0, 5, 7, 9}

    for window in range(12):
        pieces = [
            f"A😀{window}",
            f"beta{window}",
            f"gamma{window}",
            f"delta{window}",
            f"eps{window}",
            f"zeta{window}",
        ]
        response_text = " | ".join(pieces)
        prompt = f"Complete prompt {window} </script>"
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        text_hash = hashlib.sha256(response_text.encode()).hexdigest()
        token_hash = hashlib.sha256(f"tokens-{window}".encode()).hexdigest()
        offset_hash = hashlib.sha256(f"offsets-{window}".encode()).hexdigest()
        response_id = f"response-{window}"
        response_units = []
        cursor = 0
        for index, piece in enumerate(pieces):
            start = response_text.index(piece, cursor)
            end = start + len(piece)
            cursor = end
            unit = {
                "schema_version": "adag.process-witness.coarse-unit.v1",
                "unit_id": f"unit-{window}-{index}",
                "response_id": response_id,
                "prompt_sha256": prompt_hash,
                "input_ids_sha256": token_hash,
                "offset_mapping_sha256": offset_hash,
                "sequence_index": index,
                "unit_kind": "semantic_text",
                "token_span": [start, end],
                "core_character_span": [start, end],
                "covering_character_span": [start, end],
                "text": piece,
                "text_sha256": text_hash,
            }
            response_units.append(unit)
            units.append(unit)
        documents.append(
            {
                "response_id": response_id,
                "prompt_sha256": prompt_hash,
                "text_sha256": text_hash,
                "text": response_text,
                "task_context": {"prompt": prompt},
                "tokenization": {
                    "identity_status": "verified",
                    "token_count": len(response_text),
                    "input_ids_sha256": token_hash,
                    "offset_mapping_sha256": offset_hash,
                    "tokens": [
                        [index, index, index + 1] for index in range(len(response_text))
                    ],
                },
            }
        )
        focal_ids = [value["unit_id"] for value in response_units]
        windows.append(
            {
                "window_index": window,
                "response_id": response_id,
                "prompt_sha256": prompt_hash,
                "focal_unit_ids": focal_ids,
                "source_type_stratum": "complex",
                "position_stratum": "early",
            }
        )
        v1_primary_id = f"v1-{window}"
        v1_events.append(
            {
                "request_id": v1_primary_id,
                "repeat_of_request_id": None,
                "model_resolved": "gpt-5.6-luna",
                "decisions": [_decision(unit_id) for unit_id in focal_ids],
            }
        )
        if window in repeat_windows:
            v1_events.append(
                {
                    "request_id": f"{v1_primary_id}-repeat",
                    "repeat_of_request_id": v1_primary_id,
                    "model_resolved": "gpt-5.6-luna",
                    "decisions": [_decision(unit_id) for unit_id in focal_ids],
                }
            )
        for arm in (ARM_TARGET_ONLY, ARM_FULL_UNIT):
            primary_id = f"v2-{arm}-{window}"
            request = {
                "request_id": primary_id,
                "arm_id": arm,
                "window_index": window,
                "response_id": response_id,
                "source_v1_request_id": v1_primary_id,
                "repeat_of_request_id": None,
                "focal_unit_ids": focal_ids,
            }
            requests.append(request)
            v2_events.append(
                {
                    **{key: request[key] for key in ("request_id", "arm_id")},
                    "model_resolved": "gpt-5.6-luna",
                    "decisions": [_decision(unit_id) for unit_id in focal_ids],
                }
            )
            if window in repeat_windows:
                repeat = {
                    **request,
                    "request_id": f"{primary_id}-repeat",
                    "source_v1_request_id": f"{v1_primary_id}-repeat",
                    "repeat_of_request_id": primary_id,
                }
                requests.append(repeat)
                v2_events.append(
                    {
                        **{key: repeat[key] for key in ("request_id", "arm_id")},
                        "model_resolved": "gpt-5.6-luna",
                        "decisions": [_decision(unit_id) for unit_id in focal_ids],
                    }
                )

    workstation = tmp_path / "workstation.json"
    _write_json(workstation, {"documents": documents})
    units_path = v1_root / "units.jsonl"
    _write_jsonl(units_path, units)
    q_manifest = {
        "manifest_sha256": "q" * 64,
        "source_workstation_bundle": str(workstation),
        "source_workstation_bundle_sha256": file_sha256(workstation),
        "source_v1_qualification_root": str(v1_root),
    }
    qualification = {
        "manifest": q_manifest,
        "config": {"source": {"v1_units_sha256": file_sha256(units_path)}},
        "windows": windows,
        "focal_units": units,
        "requests": requests,
        "v1_comparison_baseline": {"events": v1_events},
    }
    collection = {
        "collection_manifest_sha256": "c" * 64,
        "events_jsonl_sha256": "e" * 64,
    }
    inputs = {
        "qualification": qualification,
        "run_intent": {"run_intent_sha256": "r" * 64},
        "collection": collection,
        "events": v2_events,
    }
    source_bindings = {
        "qualification_manifest_sha256": "q" * 64,
        "collection_manifest_sha256": "c" * 64,
    }
    comparison = {
        "manifest": {
            "manifest_sha256": "m" * 64,
            "source_run_root": str(run_root),
            "source_qualification_root": str(qualification_root),
            "source_bindings": source_bindings,
        },
        "report": {"report_sha256": "p" * 64, "source_bindings": source_bindings},
    }
    monkeypatch.setattr(module, "load_completed_comparison_inputs", lambda **_: inputs)
    monkeypatch.setattr(module, "load_comparison_bundle", lambda _: comparison)
    _write_json(qualification_root / "manifest.json", q_manifest)
    _write_json(run_root / "collection-manifest.json", collection)
    _write_json(comparison_root / "manifest.json", comparison["manifest"])
    return qualification_root, run_root, comparison_root


def test_payload_deduplicates_docs_and_reveals_all_protocol_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification, run, comparison = _fixture(tmp_path, monkeypatch)
    documents, items, sources = assemble_review_payload(
        qualification_root=qualification,
        run_root=run,
        comparison_root=comparison,
    )
    assert len(documents) == 12
    assert len(items) == 72
    assert len({item["document_id"] for item in items}) == 12
    assert all("response_text" not in item for item in items)
    assert documents[0]["response_text"].startswith("A😀0")
    assert documents[0]["response_character_count"] == len(
        documents[0]["response_text"]
    )
    assert items[0]["group_position"] == 1
    assert items[0]["group_size"] == 6
    assert len(items[0]["group_focal_unit_ids"]) == 6
    assert [value["decision_key"] for value in items[0]["revealed_decisions"]] == [
        "v1_primary",
        "v1_repeat",
        "target_only_primary",
        "target_only_repeat",
        "full_unit_primary",
        "full_unit_repeat",
    ]
    assert all(value["available"] for value in items[0]["revealed_decisions"])
    assert (
        sum(
            all(value["available"] for value in item["revealed_decisions"])
            for item in items
        )
        == 24
    )
    assert sources["packet_id"].startswith("process-witness-coarse-review-v2-")


def test_html_is_blind_first_codepoint_safe_searchable_and_importable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification, run, comparison = _fixture(tmp_path, monkeypatch)
    documents, items, sources = assemble_review_payload(
        qualification_root=qualification,
        run_root=run,
        comparison_root=comparison,
    )
    html = render_review_html(documents, items, sources["packet_id"]).decode()
    assert "Complete exact response" in html
    assert "Show all coarse-unit boundaries" in html
    assert 'id="boundaries" type="checkbox"' in html
    assert "Array.from(textValue)" in html
    assert "sliceCodePoints" in html
    assert "textContent" in html
    assert "scrollIntoView" in html
    assert "Lock judgment and reveal all decisions" in html
    assert (
        "Model decisions, repeat availability, and disagreements remain hidden" in html
    )
    assert "Import JSONL" in html
    assert "Export JSONL" in html
    assert "post_reveal_correction_recorded" in html
    assert "Complete prompt 0 </script>" not in html
    encoded = re.search(r'atob\("([A-Za-z0-9+/=]+)"\)', html).group(1)  # type: ignore[union-attr]
    payload = json.loads(base64.b64decode(encoded))
    assert len(payload["documents"]) == 12
    assert len(payload["items"]) == 72
    assert "response_text" not in payload["items"][0]


def test_payload_rejects_character_span_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification, run, comparison = _fixture(tmp_path, monkeypatch)
    v1_root = Path(
        json.loads((qualification / "manifest.json").read_text())[
            "source_v1_qualification_root"
        ]
    )
    units = [
        json.loads(line) for line in (v1_root / "units.jsonl").read_text().splitlines()
    ]
    units[0]["core_character_span"] = [0, 2]
    _write_jsonl(v1_root / "units.jsonl", units)
    inputs = module.load_completed_comparison_inputs()
    inputs["qualification"]["config"]["source"]["v1_units_sha256"] = file_sha256(
        v1_root / "units.jsonl"
    )
    with pytest.raises(ValueError, match="coarse-unit identity drift"):
        assemble_review_payload(
            qualification_root=qualification,
            run_root=run,
            comparison_root=comparison,
        )


def test_build_writes_fresh_hash_bound_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification, run, comparison = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module,
        "_source_revision",
        lambda: {
            "git_commit": "a" * 40,
            "tracked_worktree_clean": True,
            "files": [{"path": "builder.py", "sha256": "b" * 64}],
        },
    )
    monkeypatch.setattr(module, "_readonly_tree", lambda _: None)
    output = tmp_path / "review-v2"
    manifest = build_review_packet_v2(
        qualification_root=qualification,
        run_root=run,
        comparison_root=comparison,
        destination=output,
    )
    assert manifest["status"] == "frozen_offline_full_response_blind_review_packet"
    assert manifest["network_calls_made"] == 0
    assert manifest["counts"] == {
        "documents": 12,
        "review_items": 72,
        "focal_units_per_group": 6,
        "items_with_six_available_decisions": 24,
    }
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "review-documents.jsonl",
        "review-items.jsonl",
        "review.html",
    }
    assert manifest["ui_sha256"] == file_sha256(output / "review.html")
    assert len(manifest["document_bindings_in_order"]) == 12
    assert len(manifest["item_bindings_in_order"]) == 72
    with pytest.raises(FileExistsError):
        build_review_packet_v2(
            qualification_root=qualification,
            run_root=run,
            comparison_root=comparison,
            destination=output,
        )
