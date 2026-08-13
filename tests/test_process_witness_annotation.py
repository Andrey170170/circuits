from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
from circuits.analysis.bonafide.process_annotation import (
    annotate_response,
    audit_documents,
    canonical_sha256,
    captured_byte_level_token_offsets,
    continuation_token_offsets,
    load_ontology,
    suggest_matches,
    text_sha256,
)

ONTOLOGY_PATH = (
    Path(__file__).parents[1]
    / "scripts/bonafide/configs/process_witness_annotation_ontology_v1.json"
)


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
        return_attention_mask: bool,
    ) -> dict[str, object]:
        assert not add_special_tokens
        result: dict[str, object] = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class ByteLevelFixtureTokenizer:
    pieces: ClassVar[dict[int, str]] = {
        1: "ĠH",
        2: "VK",
        3: "PU",
        4: "FL",
        5: "I",
    }

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return " HVKPUFLI"

    def convert_ids_to_tokens(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return [self.pieces[value] for value in ids]


def test_continuation_offsets_use_prefix_boundary_and_authoritative_ids() -> None:
    tokenizer = CharacterTokenizer()
    prefix = "prefix:"
    response = "64 / 2 = 32"
    ids, offsets = continuation_token_offsets(
        tokenizer,
        prefix_text=prefix,
        response_text=response,
        expected_prefix_ids=[ord(char) for char in prefix],
        expected_response_ids=[ord(char) for char in response],
    )
    assert ids == [ord(char) for char in response]
    assert offsets == [[index, index + 1] for index in range(len(response))]

    with pytest.raises(ValueError, match="captured generation IDs"):
        continuation_token_offsets(
            tokenizer,
            prefix_text=prefix,
            response_text=response,
            expected_prefix_ids=[ord(char) for char in prefix],
            expected_response_ids=[0],
        )


def test_captured_bpe_segmentation_is_aligned_without_retokenization() -> None:
    offsets = captured_byte_level_token_offsets(
        ByteLevelFixtureTokenizer(),
        text=" HVKPUFLI",
        token_ids=[1, 2, 3, 4, 5],
    )
    assert offsets == [[0, 2], [2, 4], [4, 6], [6, 8], [8, 9]]


def test_rules_avoid_decimal_dot_backtick_and_non_arithmetic_hyphen() -> None:
    ontology = load_ontology(ONTOLOGY_PATH)
    text = "- Digital Ethics\nVersion 2.5 uses `code-block`, **bold**, 64 / 2 = 32, 5-2=3, 8÷2, 4×2, x², and 41mod79 ≡ 4 ≈ 4."
    matches = suggest_matches(text, ontology)
    decimal_dot = text.index(".")
    assert not any(
        match.value == "sentence_terminal" and match.start == decimal_dot
        for match in matches
    )
    assert not any(
        match.value == "quote" and text[match.start : match.end] == "`"
        for match in matches
    )
    subtraction_positions = {
        match.start for match in matches if match.value == "subtraction"
    }
    assert text.index("-", text.index("code")) not in subtraction_positions
    assert text.index("-") not in subtraction_positions
    assert text.index("*", text.index("**bold")) not in {
        match.start for match in matches if match.value == "multiplication"
    }
    assert text.index("-", text.index("5-2")) in subtraction_positions
    assert any(match.value == "division" for match in matches)
    assert any(
        match.value == "division" and text[match.start : match.end] == "÷"
        for match in matches
    )
    assert any(
        match.value == "multiplication" and text[match.start : match.end] == "×"
        for match in matches
    )
    assert any(
        match.value == "exponentiation" and text[match.start : match.end] == "²"
        for match in matches
    )
    assert any(
        match.value == "modulo" and text[match.start : match.end] == "mod"
        for match in matches
    )
    assert sum(match.value == "equality_symbol" for match in matches) >= 3


def test_apostrophe_percent_and_terminal_json_rules() -> None:
    ontology = load_ontology(ONTOLOGY_PATH)
    text = 'I\'m 100% sure.\n{"final_answer": 7}\ntrailing prose'
    matches = suggest_matches(text, ontology, accepted_answer_keys={"final_answer"})
    assert any(
        match.value == "apostrophe" and text[match.start : match.end] == "'"
        for match in matches
    )
    assert not any(
        match.value in {"modulo", "operator_symbol"}
        and text[match.start : match.end] == "%"
        for match in matches
    )
    assert not any(match.value == "answer_key" for match in matches)

    terminal = 'reasoning\n{"final_answer": 7}\n</think>\n'
    terminal_matches = suggest_matches(
        terminal, ontology, accepted_answer_keys={"final_answer"}
    )
    assert any(match.value == "answer_key" for match in terminal_matches)


def test_terminal_task_native_json_keys_are_discovered() -> None:
    ontology = load_ontology(ONTOLOGY_PATH)
    text = (
        'An illustrative object is {"final_answer": "wrong"}.\n'
        '{"final_node": "Depot", "cargo_value": 17}'
    )
    matches = suggest_matches(
        text,
        ontology,
        accepted_answer_keys={"final_node", "cargo_value"},
    )
    answer_keys = {
        text[match.start : match.end]
        for match in matches
        if match.value == "answer_key"
    }
    assert answer_keys == {"final_node", "cargo_value"}
    assert "final_answer" not in answer_keys


def test_annotation_retains_unknown_status_and_exact_token_alignment() -> None:
    ontology = load_ontology(ONTOLOGY_PATH)
    text = '64 / 2 = 32.\n{"cargo_value": 32}'
    ids = [ord(char) for char in text]
    offsets = [[index, index + 1] for index in range(len(text))]
    response = {
        "response_id": "response-1",
        "source": "fixture",
        "trace_scope": "full_assistant_serialization",
        "prompt_sha256": text_sha256("fixture prompt"),
        "generation_row": {
            "prompt": "fixture prompt",
            "src_types_json": '["complex"]',
            "question_ids_json": '["q1"]',
            "accepted_answer_schemas_json": json.dumps(
                [{"exact_keys": ["cargo_value"]}]
            ),
        },
    }
    document = annotate_response(
        response=response,
        text=text,
        ids=ids,
        offsets=offsets,
        token_identity={
            "kind": "fixture",
            "response_ids_sha256": canonical_sha256(ids),
        },
        ontology=ontology,
        ontology_sha256="b" * 64,
        cohort_id="cohort-fixture",
        annotation_set_id="annotation-fixture",
    )
    assert document["annotation_status"] == "automatic_suggestions_unreviewed"
    assert all(
        suggestion["status"] == "suggested_unreviewed"
        for suggestion in document["suggestions"]
    )
    assert all(
        suggestion["token_span"]["boundary_alignment"] == "exact"
        for suggestion in document["suggestions"]
    )
    assert audit_documents([document])["status"] == "passed"
    assert document["task_context"]["prompt"] == "fixture prompt"
    assert document["task_context"]["source_types"] == ["complex"]
    assert document["process_events"]["events"] == []
    assert {"usage", "token_position", "event_status"} <= set(
        document["ontology"]["axes"]
    )


def test_review_ui_guards_stale_context_and_paginates() -> None:
    html = (
        Path(__file__).parents[1]
        / "scripts/bonafide/process_witness_annotation_review.html"
    ).read_text(encoding="utf-8")
    assert "box.dataset.responseId!==doc.response_id" in html
    assert "const PAGE_SIZE = 200" in html
    assert 'coordinate_unit:"Unicode_code_point"' in html
    assert 'id="reviewImport"' in html
