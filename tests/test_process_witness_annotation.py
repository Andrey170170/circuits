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
    inspection_examples,
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
ONTOLOGY_V4_PATH = (
    Path(__file__).parents[1]
    / "scripts/bonafide/configs/process_witness_annotation_ontology_v4.json"
)
ONTOLOGY_V5_PATH = (
    Path(__file__).parents[1]
    / "scripts/bonafide/configs/process_witness_annotation_ontology_v5.json"
)
ONTOLOGY_V6_PATH = (
    Path(__file__).parents[1]
    / "scripts/bonafide/configs/process_witness_annotation_ontology_v6.json"
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


def test_v4_process_events_cover_full_units_and_apply_role_precedence() -> None:
    ontology = load_ontology(ONTOLOGY_V4_PATH)
    assert ontology["axes"]["usage"] == {
        "kind": "human_selection",
        "values": [
            "process_atlas_fit",
            "surface_reference",
            "trajectory_only",
            "unknown",
        ],
    }
    assert "usage" in ontology["token_assignment_contract"]["human_only_axes"]
    assert ontology["schema_version"].endswith(".v4")
    assert "token_position" not in ontology["axes"]
    assert {"serialization_segment", "event_token_position"} <= set(ontology["axes"])

    fixtures = {
        "Step 3: 21 → 64 (odd, 21*3+1=64)": "arithmetic_event_candidate",
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

    state = "Digital Ethics → Quantum Physics"
    state_matches = suggest_matches(state, ontology)
    assert any(
        match.axis == "process_span"
        and match.value == "state_relation_or_schema_candidate"
        and (match.start, match.end) == (0, len(state))
        for match in state_matches
    )
    assert {
        (match.value, state[match.start : match.end])
        for match in state_matches
        if match.axis == "process_role"
    } >= {
        ("state_value_candidate", "Digital Ethics"),
        ("state_value_candidate", "Quantum Physics"),
    }

    chain = "14^2 = 14*14 = 196"
    chain_matches = suggest_matches(chain, ontology)
    middle = chain.index("14", chain.index("=") + 1)
    middle_roles = {
        match.value
        for match in chain_matches
        if match.axis == "process_role" and match.start == middle
    }
    assert "operand_candidate" in middle_roles
    assert "intermediate_result_candidate" not in middle_roles
    final_start = chain.rindex("196")
    assert any(
        match.axis == "process_role"
        and match.value == "intermediate_result_candidate"
        and match.start == final_start
        for match in chain_matches
    )

    verbal = "divide 84 by 2 to get 42"
    verbal_matches = suggest_matches(verbal, ontology)
    result_start = verbal.rindex("42")
    assert any(
        match.axis == "process_role"
        and match.value == "intermediate_result_candidate"
        and match.start == result_start
        for match in verbal_matches
    )


def test_v4_role_and_modality_hard_negatives() -> None:
    ontology = load_ontology(ONTOLOGY_V4_PATH)

    conclusion = "The answer is likely a small prime like 2, 3, 5, 7, 11."
    conclusion_matches = suggest_matches(conclusion, ontology)
    assert not any(match.value == "final_result" for match in conclusion_matches)
    correct_conclusion = "Thus, this is the correct answer."
    correct_conclusion_matches = suggest_matches(correct_conclusion, ontology)
    assert any(match.value == "conclusion" for match in correct_conclusion_matches)
    assert not any(
        match.value in {"verification", "verification_event_candidate"}
        for match in correct_conclusion_matches
    )
    correction = "Wait, I need to correct this step."
    assert any(
        match.value == "correction_or_reconsideration"
        for match in suggest_matches(correction, ontology)
    )

    formula = (
        "Start with initial citation_index=0. "
        "The rule is citation_index = (citation_index * 3 + 39) mod 75. "
        "Then 14*3 = 42."
    )
    formula_matches = suggest_matches(formula, ontology)
    intermediate_texts = {
        formula[match.start : match.end]
        for match in formula_matches
        if match.value == "intermediate_result_candidate"
    }
    assert intermediate_texts == {"42"}

    described = (
        "The rules say each reference is listed as Digital Ethics → Quantum Physics. "
        '"Move to the referenced section." "Update citation_index."'
    )
    described_matches = suggest_matches(described, ontology)
    assert not any(
        match.value in {"state_transition_event_candidate", "working_or_derivation"}
        for match in described_matches
    )
    assert any(
        match.value == "instruction_or_task_description" for match in described_matches
    )
    assert any(
        match.value == "state_relation_or_schema_candidate"
        for match in described_matches
    )

    quoted_list = (
        'The problem says: "For each reference you follow, you must: 1. '
        'Move to the referenced section. 2. Update citation_index."'
    )
    quoted_matches = suggest_matches(quoted_list, ontology)
    move_start = quoted_list.index("Move")
    assert any(
        match.axis == "discourse_phase"
        and match.value == "instruction_or_task_description"
        and match.start <= move_start < match.end
        for match in quoted_matches
    )
    assert not any(
        match.value == "state_transition_event_candidate" for match in quoted_matches
    )
    assert not any(
        match.axis == "operation" and match.value == "state_transition"
        for match in quoted_matches
    )

    lookup = "Looking at the outgoing references, I see three edges."
    lookup_matches = suggest_matches(lookup, ontology)
    assert any(
        match.value == "reference_lookup_event_candidate" for match in lookup_matches
    )
    assert not any(
        match.value == "verification_event_candidate" for match in lookup_matches
    )
    assert not any(match.value == "verification_cue" for match in lookup_matches)
    assert not any(
        match.axis == "operation" and match.value == "verification"
        for match in lookup_matches
    )
    lookup_all = "Check all references where the source is Spectroscopy."
    lookup_all_matches = suggest_matches(lookup_all, ontology)
    assert not any(
        match.value
        in {"reference_lookup_event_candidate", "verification_event_candidate"}
        for match in lookup_all_matches
    )
    assert any(
        match.value in {"instruction_or_task_description", "planning"}
        for match in lookup_all_matches
    )

    encoding_mention = "The encoded string is 8 characters; look at the encoded result."
    assert not any(
        match.value == "encoding_or_decoding_event_candidate"
        for match in suggest_matches(encoding_mention, ontology)
    )
    assert not any(
        match.axis == "operation" and match.value == "encoding_or_decoding"
        for match in suggest_matches(encoding_mention, ontology)
    )
    encoding_action = "We decode ABC by shifting each letter, yielding XYZ."
    assert any(
        match.value == "encoding_or_decoding_event_candidate"
        for match in suggest_matches(encoding_action, ontology)
    )
    encoding_plan = "So to decode, we need to reverse step2 first."
    assert not any(
        match.value == "encoding_or_decoding_event_candidate"
        for match in suggest_matches(encoding_plan, ontology)
    )

    executed = "We move to Quantum Physics: Digital Ethics → Quantum Physics."
    executed_matches = suggest_matches(executed, ontology)
    assert any(
        match.value == "state_transition_event_candidate" for match in executed_matches
    )
    assert {
        executed[match.start : match.end]
        for match in executed_matches
        if match.axis == "operation" and match.value == "state_transition"
    } == {"move"}
    assert {
        (match.value, executed[match.start : match.end])
        for match in executed_matches
        if match.axis == "process_role"
    } >= {
        ("state_value_candidate", "Digital Ethics"),
        ("state_update", "Quantum Physics"),
    }

    for ambiguous_motion in (
        "We choose Digital Ethics → Quantum Physics.",
        "Take Digital Ethics → Quantum Physics.",
        "Go Digital Ethics → Quantum Physics.",
    ):
        assert not any(
            match.value == "state_transition_event_candidate"
            for match in suggest_matches(ambiguous_motion, ontology)
        )

    for invalid_arrow in (
        'No "the" -> correct',
        "-> Quantum Physics",
        "Digital Ethics -> multiplier=2",
        "Digital Ethics -> 2385",
        "14 -> 28",
    ):
        invalid_matches = suggest_matches(invalid_arrow, ontology)
        assert not any(
            match.value == "state_relation_or_schema_candidate"
            for match in invalid_matches
        )
        assert not any(
            match.value in {"state_value_candidate", "state_update"}
            for match in invalid_matches
        )

    for task_description in (
        "The task is to move to the next node.",
        "The goal is to decode the value.",
        "Objective: follow the edge.",
        "The task involves checking the table.",
    ):
        assert any(
            match.value == "instruction_or_task_description"
            for match in suggest_matches(task_description, ontology)
        )

    for planned_encoding in (
        "We will decode ABC to XYZ.",
        "If we reverse ABC, it yields CBA.",
        "The goal is to encode ABC as XYZ.",
        "For example, sometimes Caesar shifts are done with A=1, B=2.",
    ):
        assert not any(
            match.value == "encoding_or_decoding_event_candidate"
            for match in suggest_matches(planned_encoding, ontology)
        )

    negated_sense = "Wait, this does not make sense."
    negated_matches = suggest_matches(negated_sense, ontology)
    assert any(
        match.value in {"uncertainty_or_deliberation", "correction_or_reconsideration"}
        for match in negated_matches
    )

    parameter_assignment = "multiplier=2, addend=10, mod=905"
    assert not any(
        match.value == "intermediate_result_candidate"
        for match in suggest_matches(parameter_assignment, ontology)
    )
    assignment_after_arrow = "This gives x mod 905 → multiplier=2, addend=10, mod=905."
    assert not any(
        match.value == "intermediate_result_candidate"
        for match in suggest_matches(assignment_after_arrow, ontology)
    )
    assert not any(
        match.value in {"verification", "verification_event_candidate"}
        for match in negated_matches
    )


def test_v5_derived_results_and_active_listed_relations() -> None:
    ontology = load_ontology(ONTOLOGY_V5_PATH)
    assert ontology["schema_version"].endswith(".v5")
    assert ontology["ontology_id"] == "process-witness-graph-blind-v5"

    manual_meta = "But given that I'm doing this manually, I'll proceed carefully."
    manual_matches = suggest_matches(manual_meta, ontology)
    assert any(match.value == "planning" for match in manual_matches)
    assert not any(
        match.value == "orientation_or_restating" for match in manual_matches
    )

    intended_path = (
        "But given that the problem expects a unique answer, "
        "this must be the intended path."
    )
    assert not any(
        match.value == "orientation_or_restating"
        for match in suggest_matches(intended_path, ontology)
    )

    explicit_restatement = "We have given three values."
    assert any(
        match.value == "orientation_or_restating"
        for match in suggest_matches(explicit_restatement, ontology)
    )

    for text in (
        "Given that, the decoded plaintext is K V H F U P I L.",
        "Given that, the winner is Basilisk.",
    ):
        matches = suggest_matches(text, ontology)
        assert any(match.value == "conclusion" for match in matches)
        assert any(match.value == "answer_event_candidate" for match in matches)
        assert not any(match.value == "orientation_or_restating" for match in matches)

    active_scan = (
        'Looking through the list.\n"Pollux → Spica" is listed as:\n'
        "Decision: fuel_cost=66"
    )
    active_matches = suggest_matches(active_scan, ontology)
    relation_start = active_scan.index('"Pollux')
    assert any(
        match.axis == "discourse_phase"
        and match.value == "reference_lookup_or_reading"
        and active_scan[match.start : match.end] == "Looking through the list."
        for match in active_matches
    )
    assert any(
        match.axis == "discourse_phase"
        and match.value == "reference_lookup_or_reading"
        and match.start == relation_start
        for match in active_matches
    )
    assert any(
        match.value == "reference_lookup_event_candidate"
        and match.start == relation_start
        for match in active_matches
    )
    assert not any(
        match.value == "instruction_or_task_description"
        and match.start == relation_start
        for match in active_matches
    )

    bare_schema = '"Pollux → Spica" is listed as:'
    assert any(
        match.value == "instruction_or_task_description"
        for match in suggest_matches(bare_schema, ontology)
    )


def test_v6_surrounding_unit_execution_context() -> None:
    ontology = load_ontology(ONTOLOGY_V6_PATH)
    assert ontology["schema_version"].endswith(".v6")
    assert ontology["ontology_id"] == "process-witness-graph-blind-v6"

    compact = "Step 2: H → J.\nNext:4."
    compact_matches = suggest_matches(compact, ontology)
    assert any(
        match.value == "state_transition_event_candidate"
        and compact[match.start : match.end] == "Step 2: H → J."
        for match in compact_matches
    )
    assert {
        compact[match.start : match.end]: match.value
        for match in compact_matches
        if match.axis == "discourse_phase"
    } == {
        "Step 2: H → J.": "working_or_derivation",
        "Next:4.": "working_or_derivation",
    }

    for instruction_text in (
        "1.\nReverse the resulting string.",
        '"Reverse the resulting string."',
    ):
        matches = suggest_matches(instruction_text, ontology)
        reverse_start = instruction_text.index("Reverse")
        assert any(
            match.value == "instruction_or_task_description"
            and match.start <= reverse_start < match.end
            for match in matches
        )
        assert not any(
            match.value == "working_or_derivation"
            and match.start <= reverse_start < match.end
            for match in matches
        )

    transition = "So Den → Laboratory."
    transition_matches = suggest_matches(transition, ontology)
    assert any(
        match.value == "state_transition_event_candidate"
        for match in transition_matches
    )
    assert not any(
        match.value == "state_relation_or_schema_candidate"
        for match in transition_matches
    )

    outcome = "Manticore → Wyvern wins."
    outcome_matches = suggest_matches(outcome, ontology)
    assert any(
        match.value == "comparison_or_selection_event_candidate"
        for match in outcome_matches
    )
    assert not any(
        match.value == "state_relation_or_schema_candidate" for match in outcome_matches
    )

    so_outcome = "So Wyvern beats Manticore → Wyvern wins."
    so_outcome_matches = suggest_matches(so_outcome, ontology)
    assert any(
        match.value == "comparison_or_selection_event_candidate"
        for match in so_outcome_matches
    )
    assert not any(
        match.value == "state_transition_event_candidate"
        for match in so_outcome_matches
    )

    compact_match_outcome = "Match5: Harpy vs Gargoyle → Gargoyle"
    compact_match_matches = suggest_matches(compact_match_outcome, ontology)
    assert any(
        match.value == "comparison_or_selection_event_candidate"
        for match in compact_match_matches
    )
    assert not any(
        match.value == "state_relation_or_schema_candidate"
        for match in compact_match_matches
    )

    final_match_outcome = "Final: Gargoyle vs Gargoyle → Gargoyle."
    final_match_matches = suggest_matches(final_match_outcome, ontology)
    assert any(
        match.value == "comparison_or_selection_event_candidate"
        for match in final_match_matches
    )
    assert not any(
        match.value == "state_relation_or_schema_candidate"
        for match in final_match_matches
    )

    policy = "At each step, we choose the smallest fuel_cost outgoing edge."
    policy_matches = suggest_matches(policy, ontology)
    assert any(
        match.value == "instruction_or_task_description" for match in policy_matches
    )
    assert not any(
        match.value == "comparison_or_selection_event_candidate"
        for match in policy_matches
    )

    for check in (
        "Check outgoing edges from J.",
        "Check outgoing lanes.",
        "Check if Isaac has any outgoing edges.",
    ):
        active = f"Step 2: H → J.\n{check}"
        active_matches = suggest_matches(active, ontology)
        check_start = active.index(check)
        assert any(
            match.value == "reference_lookup_event_candidate"
            and match.start == check_start
            for match in active_matches
        )
        assert not any(
            match.value == "instruction_or_task_description"
            and match.start == check_start
            for match in active_matches
        )

        bare_matches = suggest_matches(check, ontology)
        assert any(
            match.value == "instruction_or_task_description" for match in bare_matches
        )
        assert not any(
            match.value == "reference_lookup_event_candidate" for match in bare_matches
        )

    next_context = (
        "Squares: 1, 1, 36. Sum: 1 + 1 + 36 = 38. So next is 38.\n"
        "Sequence: [8640, 116, 38]\n"
        "Next: 38.\nDigits: 3, 8.\nSquares: 9, 64. Sum: 9 + 64 = 73.\n"
        "Sequence: [8640, 116, 38, 73]\n"
        "Next: 73.\nDigits:7,3. Squares:49,9. Sum:49+9=58.\n"
        "Sequence: [8640, 116, 38, 73, 58]\n"
        "Next:58.\nDigits:5,8. Squares:25,64. Sum:25+64=89.\n"
        "Sequence: [8640, 116, 38, 73, 58, 89]\n"
        "Next:89.\nDigits:8,9. Squares:64,81. Sum:64+81=145.\n"
        "Sequence: [8640, 116, 38, 73, 58, 89, 145]\n"
        "Next:145.\nDigits:1,4,5. Squares:1,16,25. Sum:1+16=17; 17+25=42.\n"
        "Sequence: [8640, 116, 38, 73, 58, 89, 145, 42]\n"
        "Next:42.\nDigits:4,2. Squares:16,4. Sum:20.\n"
        "Sequence: [8640, 116, 38, 73, 58, 89, 145, 42, 20]\n"
        "Next:20.\nDigits:2,0. Squares:4,0. Sum:4.\n"
        "Sequence: [8640, 116, 38, 73, 58, 89, 145, 42, 20, 4]\n"
        "Next:4.\nDigits:4. Square:16. Sum:16."
    )
    next_start = next_context.index("\nNext:4.\n") + 1
    assert any(
        match.axis == "discourse_phase"
        and match.value == "working_or_derivation"
        and match.start == next_start
        for match in suggest_matches(next_context, ontology)
    )

    exact_lookup_contexts = {
        "Check outgoing edges from J.": (
            "2*3 =6, 6+34=40. 40 mod 308 is 40.\n"
            "So digest becomes 40.\nNow at J.\nCheck outgoing edges from J."
        ),
        "Check outgoing lanes.": (
            "143*11 = 1573; 1573 +19 = 1592.\n"
            "1592 mod 985: 985*1=985, 1592-985=607. So 607.\n"
            "5. Now at Saiph. Check outgoing lanes."
        ),
        "Check if Isaac has any outgoing edges.": (
            "219*3 = 657. 657 +18 = 675. 675 mod 470: 470*1=470, "
            "675-470=205. Correct.\nNow, at Isaac. "
            "Check if Isaac has any outgoing edges."
        ),
    }
    for target, context in exact_lookup_contexts.items():
        target_start = context.index(target)
        assert any(
            match.value == "reference_lookup_event_candidate"
            and match.start == target_start
            for match in suggest_matches(context, ontology)
        )

    recap_context = (
        "Let's recount the steps:\nStart: 8640\nStep 1: 116\nStep 2: 38\n"
        "Step3:73\nStep4:58"
    )
    recap_start = recap_context.index("Step 2: 38")
    assert any(
        match.axis == "discourse_phase"
        and match.value == "working_or_derivation"
        and match.start == recap_start
        for match in suggest_matches(recap_context, ontology)
    )

    inventory_context = (
        "Looking at the list:\n- Aldebaran → Saiph: fuel 95.\n"
        "Wait, is there any other outgoing lane from Aldebaran? Let me check all "
        "the entries.\nThe list:\nWezen → Aldebaran (so Aldebaran is a "
        "destination here, not a source)\n"
        "Aldebaran → Saiph is listed as a lane."
    )
    inventory_start = inventory_context.index("Aldebaran → Saiph is listed")
    inventory_matches = suggest_matches(inventory_context, ontology)
    assert any(
        match.axis == "discourse_phase"
        and match.value == "reference_lookup_or_reading"
        and match.start == inventory_start
        for match in inventory_matches
    )
    assert not any(
        match.value == "instruction_or_task_description"
        and match.start == inventory_start
        for match in inventory_matches
    )


def test_rule_inspection_is_stratified_across_response_support() -> None:
    documents = []
    for index in range(10):
        text = f"match-{index}"
        documents.append(
            {
                "response_id": f"response-{index}",
                "text": text,
                "suggestions": [
                    {
                        "suggestion_id": f"suggestion-{index}",
                        "character_span": [0, len(text)],
                        "text": text,
                        "axis": "operation",
                        "value": "lookup",
                        "provenance": {"rule_id": "fixture.rule"},
                    }
                ],
            }
        )
    examples = inspection_examples(documents)
    selected = examples[0]["examples"]
    assert len(selected) == 7
    assert len({example["response_id"] for example in selected}) == 7
    assert selected[0]["response_id"] == "response-0"
    assert selected[-1]["response_id"] == "response-9"

    bounded = inspection_examples(
        [
            {
                "response_id": "response-many",
                "text": "longer tiny",
                "suggestions": [
                    {
                        "suggestion_id": "suggestion-long",
                        "character_span": [0, 6],
                        "text": "longer",
                        "axis": "operation",
                        "value": "lookup",
                        "provenance": {"rule_id": "fixture.rule"},
                    },
                    {
                        "suggestion_id": "suggestion-small",
                        "character_span": [7, 11],
                        "text": "tiny",
                        "axis": "operation",
                        "value": "lookup",
                        "provenance": {"rule_id": "fixture.rule"},
                    },
                ],
            }
        ]
    )
    assert bounded[0]["examples"] == [
        {
            "response_id": "response-many",
            "suggestion_id": "suggestion-small",
            "axis": "operation",
            "value": "lookup",
            "match": "tiny",
            "context": "longer tiny",
        }
    ]


def test_v3_extension_fails_closed_on_base_hash_drift(tmp_path: Path) -> None:
    base = tmp_path / ONTOLOGY_PATH.name
    extension = tmp_path / ONTOLOGY_V3_PATH.name
    base.write_bytes(ONTOLOGY_PATH.read_bytes())
    extension.write_bytes(ONTOLOGY_V3_PATH.read_bytes())
    assert load_ontology(extension)["ontology_id"] == "process-witness-graph-blind-v3"
    base.write_bytes(base.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="base ontology hash drift"):
        load_ontology(extension)


def test_v4_discourse_phase_is_exclusive_and_final_serialization_is_exact() -> None:
    ontology = load_ontology(ONTOLOGY_V4_PATH)
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
        "instruction_or_task_description",
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


def test_v4_false_positive_guards_and_compact_projection() -> None:
    ontology = load_ontology(ONTOLOGY_V4_PATH)
    text = (
        "Step 3: Version 2.5 is 100% sure.\n"
        "- `code-block` and **bold**.\n"
        "14^2 = 14*14 = 196\n"
        "We move to Quantum Physics: Digital Ethics → Quantum Physics."
    )
    matches = suggest_matches(text, ontology)
    first_line_end = text.index("\n")
    assert not any(
        match.axis == "process_span" and match.start < first_line_end
        for match in matches
    )

    response = {
        "response_id": "response-v4",
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
        annotation_set_id="annotation-v4",
    )
    bundle = build_workstation_bundle(
        [document],
        source_record_sha256s=["d" * 64],
        review_ui_version="process-witness-token-painter.v7",
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
    final_result = text.index("196")
    assert any(
        start <= final_result < end and value == "intermediate_result_candidate"
        for start, end, value in role_runs
    )
    destination = text.rindex("Quantum Physics")
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
    assert 'const UI_VERSION = "process-witness-token-painter.v9"' in html
    assert 'axes.includes("discourse_phase") ? "discourse_phase"' in html
    builder = (
        Path(__file__).parents[1]
        / "scripts/bonafide/build_process_witness_annotations.py"
    ).read_text(encoding="utf-8")
    assert 'REVIEW_UI_VERSION = "process-witness-token-painter.v9"' in builder
    assert "const documentCodepoints" in html
    assert "cpSlice(document.text" not in html
    assert '"--annotation-set-id"' in builder
    assert "required=True" in builder
