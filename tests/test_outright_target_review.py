from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.outright_review import EXPECTED_COLUMNS
from circuits.analysis.bonafide.outright_target_review import (
    DEFAULT_REGISTRY_PATH,
    EXPORT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _default_tokenizer_loader,
    assemble_target_review_payload,
    build_target_review_packet,
    load_tokenizer_registry,
    render_target_review_html,
    snapshot_manifest,
)

AUTHORITATIVE_SOURCE = Path(
    "/uufs/chpc.utah.edu/common/home/u1653998/projects/circuits/BonaFide.csv"
)


class FakeFastTokenizer:
    """Character tokenizer with a stable toy chat template and exact offsets."""

    is_fast = True
    name_or_path = "fixture/Fake"
    chat_template = "fixture-chat-template-v1"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_attention_mask: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        assert not add_special_tokens
        assert not return_attention_mask
        result: dict[str, object] = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        chat_template: str,
        tokenize: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]:
        assert chat_template == self.chat_template
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        elif messages and messages[-1]["role"] == "assistant":
            # The assistant content was already emitted above; make the empty
            # and non-empty turn suffix identical.
            rendered += "<turn-end>"
        if enable_thinking:
            rendered += "<think>"
        return [ord(char) for char in rendered] if tokenize else rendered

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(value) for value in ids)


def _row(
    *,
    row_id: str,
    cot: str,
    extract: str,
    label: str = "FAITHFUL_STEP",
) -> dict[str, str]:
    start = cot.index(extract) if extract else 0
    end = start + len(extract) if extract else -1
    row = dict.fromkeys(EXPECTED_COLUMNS, "")
    row.update(
        {
            "id": row_id,
            "question_id": "fixture-question",
            "label_type": label,
            "sentence_text": extract,
            "sentence_span_start": str(start),
            "sentence_span_end": str(end),
            "extract": extract,
            "extract_span_start": str(start),
            "extract_span_end": str(end),
            "labeling_reason": "fixture reason",
            "target_model": "fixture/Fake",
            "question": "What is two plus two?",
            "prompt": "Solve it.",
            "cot": cot,
            "model_answer": "4",
            "correct_answer": "4",
            "src_type": "complex",
        }
    )
    return row


def _registry() -> dict[str, object]:
    prompt = "System prompt."
    return {
        "schema_version": "adag.raw-graph-observatory.tokenizer-profiles.v2",
        "prompt_provenance": {
            "cot": {
                "source": "fixture",
                "source_revision": "f" * 40,
                "value": prompt,
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        },
        "profiles": {
            "fixture/Fake": {
                "tokenizer_model_id": "fixture/Fake",
                "tokenizer_revision": "a" * 40,
                "local_snapshot": "/fixture/" + "a" * 40,
                "snapshot_manifest_sha256": "b" * 64,
                "serialization_mode": "assistant_turn",
                "system_prompt": "cot",
                "reconstruction_status": "reconstructed_at_pinned_revision",
            }
        },
    }


def _payload() -> dict[str, object]:
    rows = [
        _row(row_id="faith", cot="abc", extract="a"),
        _row(
            row_id="unfaith",
            cot="abc",
            extract="b",
            label="UNFAITHFUL_STEP",
        ),
    ]
    return assemble_target_review_payload(
        rows,
        source_sha256="e" * 64,
        registry=_registry(),
        registry_sha256="d" * 64,
        tokenizer_loader=lambda _profile: FakeFastTokenizer(),
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authoritative_scope_has_all_nine_models_qwen_and_expected_counts() -> None:
    registry, _ = load_tokenizer_registry(DEFAULT_REGISTRY_PATH)
    with AUTHORITATIVE_SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["src_type"].strip().lower() in {"complex", "graph"}
        ]
    completion_keys = {
        (row["target_model"], row["prompt"], row["cot"]) for row in rows
    }
    models = Counter(model for model, _prompt, _cot in completion_keys)
    assert len(rows) == 496
    assert len(completion_keys) == 272
    assert len(models) == 9
    assert set(models) == set(registry["profiles"])
    assert any("qwen" in model.lower() for model in models)
    for profile in registry["profiles"].values():
        assert canonical_sha256(
            snapshot_manifest(Path(profile["local_snapshot"]))
        ) == profile["snapshot_manifest_sha256"]


def test_exact_positions_offsets_overlap_and_hashes() -> None:
    payload = _payload()
    assert payload["schemaVersion"] == SCHEMA_VERSION
    item = payload["completions"][0]
    assert item["statistics"] == {
        "responseTokens": 3,
        "assistantPrefixTokens": item["tokenization"]["assistantPrefixTokenCount"],
        "causalContextAtResponseEndTokens": item["tokenization"][
            "causalContextAtResponseEndTokenCount"
        ],
        "serializedConversationTokens": item["tokenization"][
            "serializedConversationTokenCount"
        ],
        "characters": 3,
        "words": 1,
        "lines": 1,
    }
    first, second = item["tokens"][:2]
    assert item["tokenization"]["causalContextAtResponseEndTokenCount"] == (
        item["tokenization"]["assistantPrefixTokenCount"] + 3
    )
    assert item["tokenization"]["serializedConversationTokenCount"] == (
        item["tokenization"]["causalContextAtResponseEndTokenCount"]
        + item["tokenization"]["assistantSuffixTokenCount"]
    )
    assert first == [ord("a"), "a", "a", 0, 1, "f", ["faith"], ["FAITHFUL_STEP"]]
    assert second == [
        ord("b"),
        "b",
        "b",
        1,
        2,
        "u",
        ["unfaith"],
        ["UNFAITHFUL_STEP"],
    ]
    assert item["tokenization"]["responseIdsSha256"] == hashlib.sha256(
        json.dumps([97, 98, 99], separators=(",", ":")).encode()
    ).hexdigest()


def test_html_has_safe_payload_target_controls_filters_and_export() -> None:
    payload = _payload()
    payload["completions"][0]["reasoning"] = "</script><script>alert('no')</script>"
    html = render_target_review_html(payload)
    assert "</script><script>alert('no')</script>" not in html
    assert ".innerHTML" not in html
    assert "textContent" in html
    for marker in (
        'data-filter="task"',
        'data-filter="source-type"',
        'data-filter="broad-label"',
        'data-filter="exact-label-type"',
        'data-filter="max-response-tokens"',
        'data-filter="sort"',
        'id="saved-only"',
        'id="target-comment"',
        "Save target",
        "Update saved target",
        "Clear draft",
        "Remove saved target",
        "localContext",
        "responsePosition",
        "tokenizerRevision",
        "snapshotManifestSha256",
        "sourceAnnotations",
        "exactLabelTypes",
        "sourceRowIds",
        "renderedEnd",
        "component-overlap",
        "token-whitespace",
        "whitespaceGlyphs",
        "Localized unfaithful overlap",
        "Causal context at response end",
        "crypto.subtle.digest",
        'return "target_"',
        "localStorage",
    ):
        assert marker in html
    match = re.search(r'atob\("([A-Za-z0-9+/=]+)"\)', html)
    assert match is not None
    decoded = json.loads(gzip.decompress(base64.b64decode(match.group(1))))
    assert decoded == payload
    assert decoded["exportSchemaVersion"] == EXPORT_SCHEMA_VERSION


def test_default_loader_fails_closed_on_snapshot_manifest_drift(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    profile = {
        "local_snapshot": str(snapshot),
        "snapshot_manifest_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="snapshot manifest drift"):
        _default_tokenizer_loader(profile)


def test_build_is_deterministic_and_manifest_binds_registry(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    digest = _write_csv(source, [_row(row_id="a", cot="abc", extract="a")])
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(_registry(), sort_keys=True) + "\n", encoding="utf-8"
    )
    first, second = tmp_path / "first", tmp_path / "second"
    kwargs = {
        "source_path": source,
        "registry_path": registry_path,
        "expected_source_sha256": digest,
        "tokenizer_loader": lambda _profile: FakeFastTokenizer(),
    }
    manifest_a = build_target_review_packet(destination=first, **kwargs)
    manifest_b = build_target_review_packet(destination=second, **kwargs)
    assert (first / "review.html").read_bytes() == (second / "review.html").read_bytes()
    assert manifest_a == manifest_b
    assert manifest_a["page_sha256"] == hashlib.sha256(
        (first / "review.html").read_bytes()
    ).hexdigest()
    assert manifest_a["tokenization"]["registry_sha256"] == hashlib.sha256(
        registry_path.read_bytes()
    ).hexdigest()
