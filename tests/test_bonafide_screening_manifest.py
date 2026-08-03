"""Focused CPU-only tests for the consolidated prompt-screening manifest."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from circuits.tracing.trace import tokenize_teacher_forced_response
from scripts.bonafide.corpus_selection import (
    SCHEMA_VERSION as CANDIDATE_SCHEMA_VERSION,
)
from scripts.bonafide.corpus_selection import (
    _tokenizer_file_manifest,
)
from scripts.bonafide.manifest import SCHEMA_VERSION
from scripts.bonafide.runner import (
    select_wave,
    validate_target_selection,
    validate_wave_sampling_design,
)
from scripts.bonafide.screening_manifest import (
    DEFAULT_SCREENING_SEED,
    EXPECTED_BROAD_EXAMPLES,
    EXPECTED_DENSE_EXAMPLES,
    WAVE_ID,
    build_screening_manifest,
    load_selection,
    write_screening_manifest,
)


class FakeChatTokenizer:
    name_or_path = "fake/model"
    chat_template = "fake-screening-template"
    eos_token_id = 999

    def apply_chat_template(
        self, messages, *, add_generation_prompt: bool, chat_template: str
    ) -> list[int]:
        del chat_template
        prompt = next(
            message["content"] for message in messages if message["role"] == "user"
        )
        prefix = [1, *[100 + ord(char) for char in prompt], 2]
        if add_generation_prompt:
            return prefix
        response = next(
            message["content"] for message in messages if message["role"] == "assistant"
        )
        return [*prefix, *[1000 + ord(char) for char in response], self.eos_token_id]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_counts(tokenizer: FakeChatTokenizer, prompt: str, response: str) -> dict:
    tokenized = tokenize_teacher_forced_response(tokenizer, prompt, response)
    prefix = len(tokenized.assistant_prefix_ids)
    response_count = len(tokenized.response_ids)
    suffix = len(tokenized.assistant_suffix_ids)
    return {
        "assistant_prefix": prefix,
        "response": response_count,
        "assistant_suffix": suffix,
        "maximum_teacher_forced_input": prefix + response_count,
        "full_conversation_with_assistant_suffix": prefix + response_count + suffix,
    }


def _example(tokenizer: FakeChatTokenizer, index: int, *, inventory: str) -> dict:
    if inventory == "dense_inventory":
        prompt = f"dense-{index}"
        response = "abcdefghijklmnopqrst"
    else:
        # The response remains broad-cap eligible while the long prefix makes
        # the example ineligible for the synthetic dense total-context cap.
        prompt = f"broad-{index}-" + ("p" * 110)
        response = "zyxwvutsrqponmlkjihgfedcba"
    example_id = f"bf-test-{index:04d}"
    dense = inventory == "dense_inventory"
    return {
        "example_id": example_id,
        "target_model": "fake/model",
        "question": f"Question {index}",
        "base_question_id": f"bfq-test-{index:04d}",
        "prompt": prompt,
        "response": response,
        "annotation_row_ids": [f"annotation-{index}"],
        "question_ids": [f"question-{index}"],
        "label_types": ["FAITHFUL_STEP" if index % 2 else "UNFAITHFUL_STEP"],
        "labeling_reasons": ["synthetic screening reason"],
        "hint_types": ["synthetic_hint"],
        "hint_datasets": ["synthetic_dataset"],
        "src_types": ["hinting"],
        "diversity": {"cot_phenotype": "faithful" if index % 2 else "omission"},
        "token_counts": _token_counts(tokenizer, prompt, response),
        "selection_membership": {
            "dense_inventory": dense,
            "recommended_dense_core": False,
            "broad_eligible_inventory": not dense,
            "broad_role": None if dense else "remaining_eligible",
        },
    }


def _selection_fixture(tmp_path: Path) -> tuple[Path, Path, FakeChatTokenizer]:
    tokenizer = FakeChatTokenizer()
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "config.json").write_text('{"model_type":"fake"}\n')
    (tokenizer_path / "tokenizer_config.json").write_text(
        '{"chat_template":"fake-screening-template"}\n'
    )

    dense = [
        _example(tokenizer, index, inventory="dense_inventory")
        for index in range(EXPECTED_DENSE_EXAMPLES)
    ]
    broad = [
        _example(
            tokenizer,
            EXPECTED_DENSE_EXAMPLES + index,
            inventory="broad_eligible_inventory",
        )
        for index in range(EXPECTED_BROAD_EXAMPLES)
    ]
    selection = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "artifact_kind": "bonafide_prompt_candidates",
        "candidate_contract": {
            "selection_unit": "deduplicated_prompt_response_example",
            "prompt_candidates_selected": True,
            "target_spans_selected": False,
            "target_spans_frozen": False,
            "trace_work_items_created": False,
        },
        "dataset": {
            "path": "BonaFide.csv",
            "sha256": "d" * 64,
            "dedupe_key": ["target_model", "prompt", "cot"],
        },
        "tokenizer": {
            "model_id": "fake/model",
            "revision": "revision-1",
            "class": type(tokenizer).__name__,
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode()
            ).hexdigest(),
            "file_manifest": _tokenizer_file_manifest(tokenizer_path),
            "length_semantics": {
                "helper": "circuits.tracing.trace.tokenize_teacher_forced_response"
            },
        },
        "selection_policy": {
            "version": "test-policy",
            "dense": {
                "response_token_cap": 30,
                "total_context_token_cap": 80,
            },
            "broad": {
                "disjoint_from": "dense_inventory",
                "response_token_cap": 40,
                "total_context_token_cap": 200,
            },
        },
        "selections": {
            "dense_inventory": [example["example_id"] for example in dense],
            "recommended_dense_core": [],
            "broad_eligible_inventory": [example["example_id"] for example in broad],
            "broad_primary": [],
            "broad_alternates": [],
            "broad_remaining_eligible": [example["example_id"] for example in broad],
        },
        "examples": [*dense, *broad],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection, indent=2) + "\n")
    return path, tokenizer_path, tokenizer


def _build(tmp_path: Path) -> tuple[dict, Path, Path, FakeChatTokenizer]:
    selection_path, tokenizer_path, tokenizer = _selection_fixture(tmp_path)
    manifest = build_screening_manifest(
        selection=load_selection(selection_path),
        selection_path=selection_path,
        expected_selection_sha256=_sha256(selection_path),
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        model_id="fake/model",
        model_revision="revision-1",
    )
    return manifest, selection_path, tokenizer_path, tokenizer


def test_builds_one_consolidated_multi_prompt_wave_accepted_by_runner(
    tmp_path: Path,
) -> None:
    manifest, _, _, _ = _build(tmp_path)

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["screening_contract"]["purpose"] == "prompt_screening_estimation"
    assert (
        manifest["screening_contract"]["final_trace_prompt_membership_frozen"] is False
    )
    assert (
        manifest["screening_contract"]["final_trace_target_membership_frozen"] is False
    )
    wave = select_wave(manifest, WAVE_ID)
    assert len(wave["items"]) == 2_128
    assert len({item["example"]["example_id"] for item in wave["items"]}) == 133
    assert "sampling_design" not in wave
    validate_wave_sampling_design(wave, manifest)
    for item in wave["items"]:
        validate_target_selection(item)
        assert "sampling" not in item["target_selection"]
        assert "screening_selection" in item["target_selection"]
        assert item["example"]["base_question_id"].startswith("bfq-test-")
        assert item["example"]["hint_types"] == ["synthetic_hint"]
        assert item["example"]["diversity"]["cot_phenotype"] in {
            "faithful",
            "omission",
        }


def test_each_example_has_16_unique_deterministic_stratified_positions(
    tmp_path: Path,
) -> None:
    manifest, selection_path, tokenizer_path, tokenizer = _build(tmp_path)
    repeated = build_screening_manifest(
        selection=load_selection(selection_path),
        selection_path=selection_path,
        expected_selection_sha256=_sha256(selection_path),
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        model_id="fake/model",
        model_revision="revision-1",
        seed=DEFAULT_SCREENING_SEED,
    )
    assert manifest == repeated

    grouped: dict[str, list[dict]] = {}
    for item in manifest["waves"][0]["items"]:
        grouped.setdefault(item["example"]["example_id"], []).append(item)
    for items in grouped.values():
        positions = [
            item["target_selection"]["response_token_positions"][0] for item in items
        ]
        strata = [
            item["target_selection"]["screening_selection"]["position_selection"][
                "stratum_index"
            ]
            for item in items
        ]
        assert len(positions) == len(set(positions)) == 16
        assert sorted(strata) == list(range(16))


def test_target_token_identity_and_inventory_disjointness(tmp_path: Path) -> None:
    manifest, selection_path, _, tokenizer = _build(tmp_path)
    selection = load_selection(selection_path)
    dense = set(selection["selections"]["dense_inventory"])
    broad = set(selection["selections"]["broad_eligible_inventory"])
    assert len(dense) == 25
    assert len(broad) == 108
    assert dense.isdisjoint(broad)

    for item in manifest["waves"][0]["items"]:
        response_ids = tokenize_teacher_forced_response(
            tokenizer, item["example"]["prompt"], item["example"]["response"]
        ).response_ids
        position = item["target_selection"]["response_token_positions"][0]
        assert (
            item["target_selection"]["final_target_token_id"] == response_ids[position]
        )
        provenance = item["target_selection"]["screening_selection"]
        expected_inventory = (
            "dense_inventory"
            if item["example"]["example_id"] in dense
            else "broad_eligible_inventory"
        )
        assert provenance["candidate_inventory"] == expected_inventory
        assert provenance["final_trace_target_membership_frozen"] is False


def test_validates_hash_caps_model_and_tokenizer_provenance(tmp_path: Path) -> None:
    selection_path, tokenizer_path, tokenizer = _selection_fixture(tmp_path)
    selection = load_selection(selection_path)
    expected_hash = _sha256(selection_path)

    with pytest.raises(ValueError, match="model_id"):
        build_screening_manifest(
            selection=selection,
            selection_path=selection_path,
            expected_selection_sha256=expected_hash,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            model_id="different/model",
            model_revision="revision-1",
        )
    with pytest.raises(ValueError, match="revision"):
        build_screening_manifest(
            selection=selection,
            selection_path=selection_path,
            expected_selection_sha256=expected_hash,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            model_id="fake/model",
            model_revision="changed-revision",
        )

    broken = copy.deepcopy(selection)
    first_dense = broken["selections"]["dense_inventory"][0]
    example = next(
        item for item in broken["examples"] if item["example_id"] == first_dense
    )
    example["token_counts"]["response"] = 31
    selection_path.write_text(json.dumps(broken) + "\n")
    with pytest.raises(ValueError, match="exceeds dense_inventory caps"):
        build_screening_manifest(
            selection=load_selection(selection_path),
            selection_path=selection_path,
            expected_selection_sha256=_sha256(selection_path),
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            model_id="fake/model",
            model_revision="revision-1",
        )


def test_fails_closed_when_selection_or_tokenizer_files_change(tmp_path: Path) -> None:
    selection_path, tokenizer_path, tokenizer = _selection_fixture(tmp_path)
    original_hash = _sha256(selection_path)
    selection = load_selection(selection_path)
    selection["dataset"]["path"] = "changed.csv"
    selection_path.write_text(json.dumps(selection) + "\n")
    with pytest.raises(ValueError, match="selection SHA-256 changed"):
        build_screening_manifest(
            selection=load_selection(selection_path),
            selection_path=selection_path,
            expected_selection_sha256=original_hash,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            model_id="fake/model",
            model_revision="revision-1",
        )

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    selection_path, tokenizer_path, tokenizer = _selection_fixture(fresh)
    (tokenizer_path / "config.json").write_text('{"model_type":"changed"}\n')
    with pytest.raises(ValueError, match="tokenizer files"):
        build_screening_manifest(
            selection=load_selection(selection_path),
            selection_path=selection_path,
            expected_selection_sha256=_sha256(selection_path),
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            model_id="fake/model",
            model_revision="revision-1",
        )


def test_atomic_write_and_compact_source_provenance(tmp_path: Path) -> None:
    manifest, selection_path, _, _ = _build(tmp_path)
    output = tmp_path / "nested" / "screening.json"
    write_screening_manifest(manifest, output)

    assert json.loads(output.read_text()) == manifest
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))
    assert manifest["source_selection"]["sha256"] == _sha256(selection_path)
    assert manifest["tokenizer"]["file_manifest"]["state"] == "file_backed"
