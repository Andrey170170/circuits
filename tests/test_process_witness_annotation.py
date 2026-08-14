from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import ClassVar

import pytest
from circuits.analysis.bonafide.process_annotation import (
    annotate_response,
    audit_documents,
    build_workstation_bundle,
    canonical_sha256,
    captured_byte_level_token_offsets,
    continuation_token_offsets,
    load_ontology,
    suggest_matches,
    text_sha256,
)

ONTOLOGY_PATH = (
    Path(__file__).parents[1]
    / "scripts/bonafide/configs/process_witness_annotation_ontology_v2.json"
)
ONTOLOGY_V3_PATH = (
    Path(__file__).parents[1]
    / "scripts/bonafide/configs/process_witness_annotation_ontology_v3.json"
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


def test_v3_rejects_base_hash_drift_and_ignores_unaccepted_terminal_json(
    tmp_path: Path,
) -> None:
    extension = json.loads(ONTOLOGY_V3_PATH.read_text(encoding="utf-8"))
    extension["base_ontology_sha256"] = "0" * 64
    (tmp_path / extension["extends"]).write_bytes(ONTOLOGY_PATH.read_bytes())
    drifted = tmp_path / ONTOLOGY_V3_PATH.name
    drifted.write_text(json.dumps(extension), encoding="utf-8")
    with pytest.raises(ValueError, match="base ontology hash drift"):
        load_ontology(drifted)

    ontology = load_ontology(ONTOLOGY_V3_PATH)
    text = '</think>\n{"diagnostic": 7}'
    matches = suggest_matches(
        text,
        ontology,
        accepted_answer_keys={"final_answer"},
    )
    assert not any(
        match.value in {"answer_key", "answer_commitment", "final_result"}
        for match in matches
    )


def test_v3_process_events_cover_full_units_and_apply_role_precedence() -> None:
    ontology = load_ontology(ONTOLOGY_V3_PATH)
    assert ontology["schema_version"].endswith(".v3")
    assert "token_position" not in ontology["axes"]
    assert {"serialization_segment", "event_token_position"} <= set(ontology["axes"])

    fixtures = {
        "Step 3: 21 → 64 (odd, 21*3+1=64)": "state_update_with_arithmetic",
        "14^2 = 14*14 = 196": "arithmetic_event_candidate",
        "divide 84 by 2 to get 42": "arithmetic_event_candidate",
        "151 mod 41 = 28": "arithmetic_event_candidate",
    }
    for text, expected_span in fixtures.items():
        matches = suggest_matches(text, ontology)
        process_spans = [match for match in matches if match.axis == "process_span"]
        assert [(match.start, match.end, match.value) for match in process_spans] == [
            (0, len(text), expected_span)
        ]
        assert any(
            match.axis == "event_operation"
            and match.start == 0
            and match.end == len(text)
            for match in matches
        )

    state = "A -> B"
    state_matches = suggest_matches(state, ontology)
    assert any(
        match.axis == "process_span"
        and match.value == "state_transition_event_candidate"
        and (match.start, match.end) == (0, len(state))
        for match in state_matches
    )
    assert {
        (match.value, state[match.start : match.end])
        for match in state_matches
        if match.axis == "process_role"
    } >= {("state_value_candidate", "A"), ("state_update", "B")}

    chain = "14^2 = 14*14 = 196"
    chain_matches = suggest_matches(chain, ontology)
    middle = chain.index("14", chain.index("=") + 1)
    middle_roles = {
        match.value
        for match in chain_matches
        if match.axis == "process_role" and match.start == middle
    }
    assert {"operand_candidate", "intermediate_result_candidate"} <= middle_roles

    verbal = "divide 84 by 2 to get 42"
    verbal_matches = suggest_matches(verbal, ontology)
    result_start = verbal.rindex("42")
    assert any(
        match.axis == "process_role"
        and match.value == "intermediate_result_candidate"
        and match.start == result_start
        for match in verbal_matches
    )


def test_v3_extension_fails_closed_on_base_hash_drift(tmp_path: Path) -> None:
    base = tmp_path / ONTOLOGY_PATH.name
    extension = tmp_path / ONTOLOGY_V3_PATH.name
    base.write_bytes(ONTOLOGY_PATH.read_bytes())
    extension.write_bytes(ONTOLOGY_V3_PATH.read_bytes())
    assert load_ontology(extension)["ontology_id"] == "process-witness-graph-blind-v3"
    base.write_bytes(base.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="base ontology hash drift"):
        load_ontology(extension)


def test_v3_discourse_phase_is_exclusive_and_final_serialization_is_exact() -> None:
    ontology = load_ontology(ONTOLOGY_V3_PATH)
    text = (
        "The user asks me to transform the values.\n"
        "Okay, first I need to plan this.\n"
        "Compute 14*3 = 42.\n"
        "Maybe that is enough.\n"
        "This sentence has no positive phase evidence.\n"
        "Wait, I made a mistake.\n"
        "Let me verify 42 / 2 = 21.\n"
        "Thus the final answer is 21.\n"
        "</think>\n\n"
        '{"final_answer": 21}'
    )
    matches = suggest_matches(text, ontology, accepted_answer_keys={"final_answer"})
    phases = [match for match in matches if match.axis == "discourse_phase"]
    assert [match.value for match in phases] == [
        "orientation_or_restating",
        "planning",
        "working_or_derivation",
        "uncertainty_or_deliberation",
        "unclassified_or_other",
        "correction_or_reconsideration",
        "verification",
        "conclusion",
        "answer_serialization",
    ]
    assert all(left.end <= right.start for left, right in pairwise(phases))
    assert not any(text[match.start : match.end] == "</think>" for match in phases)

    final_segments = [
        match
        for match in matches
        if match.axis == "serialization_segment"
        and match.value == "final_answer_segment"
    ]
    assert [text[match.start : match.end] for match in final_segments] == [
        '{"final_answer": 21}'
    ]
    assert any(
        match.axis == "process_role"
        and match.value == "final_result"
        and text[match.start : match.end] == "21"
        for match in matches
    )
    assert not any(match.axis in {"usage", "event_status"} for match in matches)

    string_answer = '</think>\n{"final_answer": "Geopolitics"}'
    string_matches = suggest_matches(
        string_answer, ontology, accepted_answer_keys={"final_answer"}
    )
    assert any(
        match.axis == "process_role"
        and match.value == "final_result"
        and string_answer[match.start : match.end] == '"Geopolitics"'
        for match in string_matches
    )

    non_answer = '</think>\n{"debug": 21}'
    non_answer_matches = suggest_matches(
        non_answer, ontology, accepted_answer_keys={"final_answer"}
    )
    assert not any(
        match.value
        in {"answer_serialization", "answer_event_candidate", "final_answer_segment"}
        for match in non_answer_matches
    )

    truncated = "Working through 8 / 2 = 4, but I need more time"
    truncated_matches = suggest_matches(truncated, ontology)
    assert not any(
        match.value in {"answer_serialization", "final_answer_segment"}
        for match in truncated_matches
    )


def test_v3_false_positive_guards_and_compact_projection() -> None:
    ontology = load_ontology(ONTOLOGY_V3_PATH)
    text = (
        "Step 3: Version 2.5 is 100% sure.\n"
        "- `code-block` and **bold**.\n"
        "14^2 = 14*14 = 196\n"
        "A -> B"
    )
    matches = suggest_matches(text, ontology)
    first_line_end = text.index("\n")
    assert not any(
        match.axis == "process_span" and match.start < first_line_end
        for match in matches
    )

    response = {
        "response_id": "response-v3",
        "source": "fixture",
        "trace_scope": "full_assistant_serialization",
        "prompt_sha256": text_sha256("fixture prompt"),
        "generation_row": {
            "prompt": "fixture prompt",
            "src_types_json": "[]",
            "question_ids_json": "[]",
            "accepted_answer_schemas_json": "[]",
        },
    }
    document = annotate_response(
        response=response,
        text=text,
        ids=[ord(character) for character in text],
        offsets=[[index, index + 1] for index in range(len(text))],
        token_identity={"kind": "fixture"},
        ontology=ontology,
        ontology_sha256="f" * 64,
        cohort_id="cohort-fixture",
        annotation_set_id="annotation-v3",
    )
    bundle = build_workstation_bundle(
        [document],
        source_record_sha256s=["d" * 64],
        review_ui_version="process-witness-token-painter.v6",
        review_ui_sha256="e" * 64,
    )
    compact = bundle["documents"][0]
    arithmetic_start = text.index("14^2")
    operation_runs = compact["machine_layers"]["event_operation"]
    assert any(
        start <= arithmetic_start < end and value == "mixed_arithmetic"
        for start, end, value in operation_runs
    )
    localized_operations = compact["machine_layers"]["operation"]
    exponent = text.index("^")
    multiply = text.index("*", text.index("14^2"))
    assert any(
        start <= exponent < end and value == "exponentiation"
        for start, end, value in localized_operations
    )
    assert any(
        start <= multiply < end and value == "multiplication"
        for start, end, value in localized_operations
    )
    role_runs = compact["machine_layers"]["process_role"]
    middle = text.index("14", text.index("=") + 1)
    assert any(
        start <= middle < end and value == "intermediate_result_candidate"
        for start, end, value in role_runs
    )
    destination = text.rindex("B")
    assert any(
        start <= destination < end and value == "state_update"
        for start, end, value in role_runs
    )
    assert "usage" not in compact["machine_layers"]
    assert "event_status" not in compact["machine_layers"]

    event_position_suggestions = [
        suggestion
        for suggestion in document["suggestions"]
        if suggestion["axis"] == "event_token_position"
    ]
    process_spans = [
        suggestion
        for suggestion in document["suggestions"]
        if suggestion["axis"] == "process_span"
    ]
    assert len(event_position_suggestions) <= 4 * len(process_spans)
    assert any(
        suggestion["value"] == "span_interior"
        and suggestion["token_span"]["end"] - suggestion["token_span"]["start"] > 1
        for suggestion in event_position_suggestions
    )
    assert all(
        suggestion["interpretation"] == "semantic_hypothesis"
        and suggestion["confidence"] == "medium"
        for suggestion in event_position_suggestions
    )


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


def test_workstation_bundle_compacts_surface_conflicts_but_preserves_semantic_ones() -> (
    None
):
    ontology = load_ontology(ONTOLOGY_PATH)
    text = " 12 plus minus"
    response = {
        "response_id": "response-conflict",
        "source": "fixture",
        "trace_scope": "full_assistant_serialization",
        "prompt_sha256": text_sha256("fixture prompt"),
        "generation_row": {
            "prompt": "fixture prompt",
            "src_types_json": "[]",
            "question_ids_json": "[]",
            "accepted_answer_schemas_json": "[]",
        },
    }
    document = annotate_response(
        response=response,
        text=text,
        ids=[123],
        offsets=[[0, len(text)]],
        token_identity={"kind": "fixture"},
        ontology=ontology,
        ontology_sha256="c" * 64,
        cohort_id="cohort-fixture",
        annotation_set_id="annotation-fixture",
    )
    bundle = build_workstation_bundle(
        [document],
        source_record_sha256s=["d" * 64],
        review_ui_version="process-witness-token-painter.v6",
        review_ui_sha256="e" * 64,
    )
    compact = bundle["documents"][0]
    assert bundle["schema_version"].endswith("workstation-bundle.v1")
    assert bundle["review_ui"] == {
        "version": "process-witness-token-painter.v6",
        "sha256": "e" * 64,
    }
    assert compact["tokenization"]["tokens"] == [[123, 0, len(text)]]
    assert compact["machine_layers"]["surface_form"] == [[0, 1, "compound_surface"]]
    assert compact["machine_layers"]["operation"] == [
        [0, 1, ["addition", "subtraction"]]
    ]
    assert ontology["token_assignment_contract"]["within_axis"].startswith(
        "zero_or_one"
    )


def test_review_ui_is_token_painter_with_bound_provenance() -> None:
    html = (
        Path(__file__).parents[1]
        / "scripts/bonafide/process_witness_annotation_review.html"
    ).read_text(encoding="utf-8")
    assert 'id="tokenCanvas"' in html
    assert 'id="axisSelect"' in html
    assert 'id="directoryInput"' in html
    assert 'id="reviewInput"' in html
    assert "source_workstation_bundle_sha256" in html
    assert "source_annotation_record_sha256" in html
    assert "source_annotation_text_sha256" in html
    assert 'operation === "revert_to_machine"' in html
    assert "resolved_state" in html
    assert "state.gesture.lastPosition" in html
    assert "compound_surface" in html
    assert 'id="toggleAxisReview"' in html
    assert "applyEventsTransactional" in html
    assert "event.schema_version !== EVENT_SCHEMA" in html
    assert 'event.coordinate_unit !== "authoritative_response_token_index"' in html
    assert 'file.name.toLowerCase().endsWith(".jsonl")' in html
    assert "bundles.length > 1" in html
    assert "bundles.length === 1" in html
    assert "review_coverage" in html
    assert "span.tabIndex = 0" in html
    assert 'span.setAttribute("aria-label"' in html
    assert 'node.classList.contains("overlap-fragment")' in html
    assert ".document-shell { min-width: 0; min-height: 0;" in html
    assert ".document-scroll { flex: 1; min-height: 0; overflow: auto;" in html
    assert 'const UI_VERSION = "process-witness-token-painter.v6"' in html
    assert 'axes.includes("discourse_phase") ? "discourse_phase"' in html
    builder = (
        Path(__file__).parents[1]
        / "scripts/bonafide/build_process_witness_annotations.py"
    ).read_text(encoding="utf-8")
    assert 'REVIEW_UI_VERSION = "process-witness-token-painter.v6"' in builder
    assert "const documentCodepoints" in html
    assert "cpSlice(document.text" not in html
    assert '"--annotation-set-id"' in builder
    assert "required=True" in builder
