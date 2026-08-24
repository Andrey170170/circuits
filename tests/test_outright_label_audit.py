from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_json, canonical_sha256
from circuits.analysis.bonafide.outright_label_audit import (
    DEFAULT_QWEN_MODEL,
    SCHEMA_VERSION,
    build_label_audit_packet,
    load_target_review_packet,
    project_label_audit_payload,
    render_label_audit_html,
)
from circuits.analysis.bonafide.outright_target_review import (
    SCHEMA_VERSION as TARGET_SCHEMA_VERSION,
)
from circuits.analysis.bonafide.outright_target_review import (
    render_target_review_html,
)


def _completion(
    *, completion_id: str, model: str, unfaithful: bool
) -> dict[str, object]:
    annotations = [
        {
            "sourceRowIndex": 4,
            "sourceRowId": "faith-row",
            "questionId": "q",
            "labelType": "FAITHFUL_STEP",
            "sentenceText": "The answer is four.",
            "sentenceSpan": [0, 19],
            "extract": "answer is four",
            "extractSpan": [4, 18],
            "labelingReason": "matches the expected step",
        }
    ]
    if unfaithful:
        annotations.append(
            {
                "sourceRowIndex": 5,
                "sourceRowId": "unfaith-row",
                "questionId": "q",
                "labelType": "UNFAITHFUL_COT",
                "sentenceText": "",
                "sentenceSpan": [0, -1],
                "extract": "",
                "extractSpan": [0, -1],
                "labelingReason": "missing the answer-is-four step",
            }
        )
    return {
        "completionId": completion_id,
        "taskId": "task",
        "model": model,
        "question": "What is two plus two?",
        "prompt": "Solve it.",
        "reasoning": "The answer is four.",
        "modelAnswer": "4",
        "correctAnswer": "4",
        "sourceType": "complex",
        "broadLabel": "mixed" if unfaithful else "faithful-only",
        "exactLabelTypes": (
            ["FAITHFUL_STEP", "UNFAITHFUL_COT"] if unfaithful else ["FAITHFUL_STEP"]
        ),
        "hasUnfaithful": unfaithful,
        "statistics": {
            "responseTokens": 5,
            "assistantPrefixTokens": 7,
            "causalContextAtResponseEndTokens": 12,
            "serializedConversationTokens": 13,
            "characters": 19,
            "words": 4,
            "lines": 1,
        },
        "annotations": annotations,
        "tokens": [[1, "x", "x", 0, 1, "", [], []]],
        "tokenization": {"tokenizerRevision": "a" * 40},
    }


def _source_payload() -> dict[str, object]:
    return {
        "schemaVersion": TARGET_SCHEMA_VERSION,
        "meta": {
            "sourceName": "BonaFide.csv",
            "sourceSha256": "e" * 64,
        },
        "completions": [
            _completion(
                completion_id="qwen-mixed",
                model=DEFAULT_QWEN_MODEL,
                unfaithful=True,
            ),
            _completion(
                completion_id="other-mixed",
                model="example/Other",
                unfaithful=True,
            ),
            _completion(
                completion_id="qwen-faithful",
                model=DEFAULT_QWEN_MODEL,
                unfaithful=False,
            ),
        ],
    }


def _write_source_packet(path: Path) -> None:
    payload = _source_payload()
    payload_bytes = canonical_json(payload)
    page = render_target_review_html(payload).encode()
    manifest = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "page_sha256": hashlib.sha256(page).hexdigest(),
        "manifest_sha256": "f" * 64,
        "files": {
            "review.html": {
                "bytes": len(page),
                "sha256": hashlib.sha256(page).hexdigest(),
            }
        },
    }
    path.mkdir()
    (path / "review.html").write_bytes(page)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_projection_defaults_to_disputed_qwen_and_removes_trace_fields() -> None:
    source = _source_payload()
    projected = project_label_audit_payload(source)
    assert projected["schemaVersion"] == SCHEMA_VERSION
    assert projected["defaults"] == {
        "model": DEFAULT_QWEN_MODEL,
        "scope": "contains-unfaithful",
    }
    assert projected["counts"] == {
        "completions": 3,
        "containsUnfaithful": 2,
        "defaultModelContainsUnfaithful": 1,
    }
    encoded = json.dumps(projected)
    for forbidden in (
        "tokens",
        "tokenization",
        "tokenizerRevision",
        "assistantPrefixTokens",
        "causalContextAtResponseEndTokens",
    ):
        assert forbidden not in encoded


def test_html_is_safe_focused_and_defaults_are_in_payload() -> None:
    payload = project_label_audit_payload(_source_payload())
    payload["completions"][0]["reasoning"] = "</script><script>alert(1)</script>"
    html = render_label_audit_html(payload)
    assert "</script><script>alert(1)</script>" not in html
    assert ".innerHTML" not in html
    for marker in (
        "Claim to audit: why the source says unfaithful",
        "Evidence the source separately marked faithful",
        "Full model reasoning",
        'data-filter="model"',
        'data-filter="scope"',
        "Mixed / contains UNFAITHFUL",
        "contains-unfaithful",
    ):
        assert marker in html
    for removed in (
        "target-comment",
        "Save target",
        "saved-only",
        "Export selection",
        "localStorage",
        "responsePosition",
    ):
        assert removed not in html
    match = re.search(r'atob\("([A-Za-z0-9+/=]+)"\)', html)
    assert match is not None
    decoded = json.loads(gzip.decompress(base64.b64decode(match.group(1))))
    assert decoded == payload


def test_build_verifies_source_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source_packet(source)
    loaded_manifest, loaded_payload = load_target_review_packet(source)
    assert loaded_manifest["schema_version"] == TARGET_SCHEMA_VERSION
    assert loaded_payload == _source_payload()
    first, second = tmp_path / "first", tmp_path / "second"
    manifest_a = build_label_audit_packet(source=source, destination=first)
    manifest_b = build_label_audit_packet(source=source, destination=second)
    assert (first / "review.html").read_bytes() == (second / "review.html").read_bytes()
    assert manifest_a == manifest_b
    assert manifest_a["scope"]["default_model"] == DEFAULT_QWEN_MODEL
    assert manifest_a["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in manifest_a.items() if key != "manifest_sha256"}
    )


def test_loader_rejects_source_page_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source_packet(source)
    (source / "review.html").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="page hash or size drift"):
        load_target_review_packet(source)
