"""Focused CPU-only tests for deterministic BonaFide prompt selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.bonafide.corpus_selection import (
    DEFAULT_RECOMMENDED_DENSE_IDS,
    SCHEMA_VERSION,
    build_corpus_selection,
    load_prompt_candidates,
    write_corpus_selection,
)


class FakeChatTokenizer:
    name_or_path = "fake/model"
    chat_template = "fake-chat-template"
    eos_token_id = 999

    def apply_chat_template(
        self, messages, *, add_generation_prompt: bool, chat_template: str
    ) -> list[int]:
        del chat_template
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        prefix = [1, *[100 + ord(char) for char in prompt], 2]
        if add_generation_prompt:
            return prefix
        response = next(
            message["content"] for message in messages if message["role"] == "assistant"
        )
        return [*prefix, *[1000 + ord(char) for char in response], self.eos_token_id]


FIELDNAMES = [
    "id",
    "question_id",
    "label_type",
    "sentence_text",
    "sentence_span_start",
    "sentence_span_end",
    "extract",
    "extract_span_start",
    "extract_span_end",
    "labeling_reason",
    "target_model",
    "question",
    "prompt",
    "cot",
    "model_answer",
    "correct_answer",
    "hinted_answer",
    "src_type",
    "hint_dataset",
    "hint_type",
    "prompted_hint",
]

HINT_TYPES = [
    "unauthorized_access",
    "validator",
    "metadata",
    "sycophancy",
    "error_message",
    "security_audit",
]
DATASETS = ["google_simpleqa-verified", "cais_hle", "aai530-group6_ddxplus"]


def _row(
    index: int,
    *,
    prompt: str,
    response: str,
    question: str | None = None,
    annotation_id: str | None = None,
    label_type: str | None = None,
    reason: str | None = None,
) -> dict[str, str]:
    return {
        "id": annotation_id or f"annotation-{index:04d}",
        "question_id": f"question-{question or index}",
        "label_type": label_type or ("UNFAITHFUL_STEP" if index % 3 == 0 else "FAITHFUL_STEP"),
        "sentence_text": "annotated sentence",
        "sentence_span_start": "2",
        "sentence_span_end": "20",
        "extract": "annotated",
        "extract_span_start": "2",
        "extract_span_end": "11",
        "labeling_reason": reason or (
            "unfaithful attribution (incorrect)"
            if index % 3 == 0
            else "faithful commitment to answer"
        ),
        "target_model": "fake/model",
        "question": question or f"Unique base question {index}",
        "prompt": prompt,
        "cot": response,
        "model_answer": "hint answer",
        "correct_answer": "correct answer",
        "hinted_answer": "hint answer",
        "src_type": "complex_hints" if index % 11 == 0 else "hinting",
        "hint_dataset": DATASETS[index % len(DATASETS)],
        "hint_type": HINT_TYPES[index % len(HINT_TYPES)],
        "prompted_hint": f"hint-{index}",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _large_fixture(path: Path) -> None:
    rows = [
        _row(index, prompt=f"dense-prompt-{index}", response="d" * (30 + index))
        for index in range(9)
    ]
    # Sixty unique base questions are enough to exercise the primary uniqueness rule.
    for index in range(90):
        question_index = index if index < 60 else index - 60
        rows.append(
            _row(
                100 + index,
                prompt=f"broad-prompt-{index}",
                response="b" * (230 + index % 50),
                question=f"Broad base question {question_index}",
            )
        )
    _write_csv(path, rows)


def _recommended_dense_ids(csv_path: Path) -> list[str]:
    examples, _ = load_prompt_candidates(
        csv_path=csv_path,
        tokenizer=FakeChatTokenizer(),
        target_model="fake/model",
    )
    return [
        example["example_id"]
        for example in examples
        if example["eligibility"]["dense_inventory"]
    ]


def test_dense_cap_uses_total_context_not_only_short_response(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _write_csv(csv_path, [_row(1, prompt="p" * 500, response="short response")])

    examples, _ = load_prompt_candidates(
        csv_path=csv_path,
        tokenizer=FakeChatTokenizer(),
        target_model="fake/model",
    )

    example = examples[0]
    assert example["token_counts"] == {
        "assistant_prefix": 502,
        "response": 14,
        "assistant_suffix": 1,
        "maximum_teacher_forced_input": 516,
        "full_conversation_with_assistant_suffix": 517,
    }
    assert example["eligibility"]["dense_inventory"] is False
    assert example["eligibility"]["dense_reasons"] == {
        "response_within_cap": True,
        "total_context_within_cap": False,
    }


def test_selection_is_deterministic_disjoint_and_retains_all_broad_inventory(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _large_fixture(csv_path)
    dense_ids = _recommended_dense_ids(csv_path)
    kwargs = {
        "csv_path": csv_path,
        "tokenizer": FakeChatTokenizer(),
        "target_model": "fake/model",
        "model_revision": "exact-revision",
        "recommended_dense_ids": dense_ids,
    }

    first = build_corpus_selection(**kwargs)
    second = build_corpus_selection(**kwargs)

    assert first == second
    selected = first["selections"]
    assert len(selected["dense_inventory"]) == 9
    assert len(selected["recommended_dense_core"]) == 9
    assert len(selected["broad_eligible_inventory"]) == 90
    assert len(selected["broad_primary"]) == 48
    assert len(selected["broad_alternates"]) == 24
    assert len(selected["broad_remaining_eligible"]) == 18
    dense = set(selected["dense_inventory"])
    primary = set(selected["broad_primary"])
    alternates = set(selected["broad_alternates"])
    remaining = set(selected["broad_remaining_eligible"])
    broad = set(selected["broad_eligible_inventory"])
    assert not dense & broad
    assert not primary & alternates
    assert not primary & remaining
    assert not alternates & remaining
    assert primary | alternates | remaining == broad


def test_primary_prefers_one_response_per_base_question(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _large_fixture(csv_path)
    selection = build_corpus_selection(
        csv_path=csv_path,
        tokenizer=FakeChatTokenizer(),
        target_model="fake/model",
        model_revision="exact-revision",
        recommended_dense_ids=_recommended_dense_ids(csv_path),
    )
    by_id = {example["example_id"]: example for example in selection["examples"]}
    base_questions = [
        by_id[example_id]["base_question_id"]
        for example_id in selection["selections"]["broad_primary"]
    ]
    assert len(base_questions) == len(set(base_questions)) == 48


def test_metadata_answers_and_annotation_spans_survive_deduplication(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    first = _row(
        1,
        prompt="prompt",
        response="response text",
        annotation_id="annotation-b",
        label_type="UNFAITHFUL_STEP",
    )
    second = {
        **first,
        "id": "annotation-a",
        "label_type": "UNFAITHFUL_COT",
        "labeling_reason": "no acknowledgements of hint and no faithful steps",
        "sentence_text": "other sentence",
        "extract": "other",
        "extract_span_start": "3",
        "extract_span_end": "8",
    }
    _write_csv(csv_path, [first, second])

    examples, counts = load_prompt_candidates(
        csv_path=csv_path,
        tokenizer=FakeChatTokenizer(),
        target_model="fake/model",
    )

    assert counts["target_model_annotation_row_count"] == 2
    assert counts["target_model_deduplicated_example_count"] == 1
    example = examples[0]
    assert example["annotation_row_ids"] == ["annotation-a", "annotation-b"]
    assert example["label_types"] == ["UNFAITHFUL_COT", "UNFAITHFUL_STEP"]
    assert example["diversity"]["cot_phenotype"] == "both"
    assert example["diversity"]["answer_relation"] == "model_matches_hint_only"
    assert [span["annotation_row_id"] for span in example["annotation_spans"]] == [
        "annotation-a",
        "annotation-b",
    ]
    assert example["annotation_spans"][0]["extract_span_start"] == 3
    assert example["annotation_spans"][0]["extract_span_valid"] is True
    assert example["source_annotations"][0]["sentence_text"] == "other sentence"
    assert len(example["provenance"]["source_annotations_sha256"]) == 64


def test_cot_placeholder_span_does_not_mask_real_annotation_position(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "bonafide.csv"
    response = "r" * 100
    placeholder = _row(
        1,
        prompt="prompt",
        response=response,
        annotation_id="cot-level",
        label_type="UNFAITHFUL_COT",
    )
    placeholder.update(
        {
            "sentence_text": "",
            "sentence_span_start": "0",
            "sentence_span_end": "-1",
            "extract": "",
            "extract_span_start": "0",
            "extract_span_end": "-1",
        }
    )
    real_step = {
        **placeholder,
        "id": "real-step",
        "label_type": "UNFAITHFUL_STEP",
        "sentence_text": "late sentence",
        "sentence_span_start": "60",
        "sentence_span_end": "73",
        "extract": "late",
        "extract_span_start": "60",
        "extract_span_end": "64",
    }
    _write_csv(csv_path, [placeholder, real_step])

    examples, _ = load_prompt_candidates(
        csv_path=csv_path,
        tokenizer=FakeChatTokenizer(),
        target_model="fake/model",
    )

    example = examples[0]
    assert example["diversity"]["annotation_position_bin"] == "third_quarter"
    spans = {span["annotation_row_id"]: span for span in example["annotation_spans"]}
    assert spans["cot-level"]["sentence_span_valid"] is False
    assert spans["cot-level"]["extract_span_valid"] is False
    assert spans["real-step"]["sentence_span_valid"] is True
    assert spans["real-step"]["extract_span_valid"] is True


def test_recommended_dense_ids_are_validated(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _write_csv(csv_path, [_row(1, prompt="short", response="short")])

    with pytest.raises(ValueError, match="missing="):
        build_corpus_selection(
            csv_path=csv_path,
            tokenizer=FakeChatTokenizer(),
            target_model="fake/model",
            model_revision="revision",
            recommended_dense_ids=DEFAULT_RECOMMENDED_DENSE_IDS,
        )

    long_csv_path = tmp_path / "long.csv"
    _write_csv(long_csv_path, [_row(2, prompt="p", response="r" * 300)])
    long_examples, _ = load_prompt_candidates(
        csv_path=long_csv_path,
        tokenizer=FakeChatTokenizer(),
        target_model="fake/model",
    )
    with pytest.raises(ValueError, match="not_dense_eligible"):
        build_corpus_selection(
            csv_path=long_csv_path,
            tokenizer=FakeChatTokenizer(),
            target_model="fake/model",
            model_revision="revision",
            recommended_dense_ids=[long_examples[0]["example_id"]],
        )


def test_broad_requested_counts_fail_closed_when_pool_is_too_small(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _write_csv(csv_path, [_row(1, prompt="prompt", response="r" * 300)])

    with pytest.raises(ValueError, match="exceed the eligible pool"):
        build_corpus_selection(
            csv_path=csv_path,
            tokenizer=FakeChatTokenizer(),
            target_model="fake/model",
            model_revision="revision",
            recommended_dense_ids=[],
            broad_primary_count=1,
            broad_alternate_count=1,
        )


def test_schema_provenance_and_atomic_writer(tmp_path: Path) -> None:
    csv_path = tmp_path / "bonafide.csv"
    _write_csv(csv_path, [_row(1, prompt="short", response="short")])
    dense_ids = _recommended_dense_ids(csv_path)
    selection = build_corpus_selection(
        csv_path=csv_path,
        tokenizer=FakeChatTokenizer(),
        target_model="fake/model",
        model_revision="revision-sha",
        tokenizer_path=Path("/cached/tokenizer"),
        recommended_dense_ids=dense_ids,
        broad_primary_count=0,
        broad_alternate_count=0,
    )

    assert selection["schema_version"] == SCHEMA_VERSION
    assert selection["artifact_kind"] == "bonafide_prompt_candidates"
    assert selection["candidate_contract"] == {
        "selection_unit": "deduplicated_prompt_response_example",
        "prompt_candidates_selected": True,
        "target_spans_selected": False,
        "target_spans_frozen": False,
        "trace_work_items_created": False,
    }
    assert selection["tokenizer"]["revision"] == "revision-sha"
    assert selection["tokenizer"]["resolved_path"] == "/cached/tokenizer"
    assert len(selection["dataset"]["sha256"]) == 64
    assert selection["selection_policy"]["diversity_axes"] == [
        "nonexclusive_label_types",
        "hint_types",
        "hint_datasets",
        "src_types",
        "cot_phenotype",
        "answer_relation",
        "annotation_position_bin",
        "response_length_bin",
        "total_length_bin",
        "question_novelty_control_family_marker",
    ]
    example = selection["examples"][0]
    assert example["selection_membership"] == {
        "dense_inventory": True,
        "recommended_dense_core": True,
        "broad_eligible_inventory": False,
        "broad_role": None,
    }
    assert "hint_types=validator" in example["coverage_features"]

    output_path = tmp_path / "nested" / "selection.json"
    write_corpus_selection(selection, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == selection
    assert not list(output_path.parent.glob(".*.tmp-*"))
