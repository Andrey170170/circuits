"""CPU-only tests for probe-informed BonaFide refinement selection."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.bonafide import refinement_selection as selection
from scripts.bonafide.runner import validate_target_selection


class CharTokenizer:
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _example(
    example_id: str,
    *,
    phenotype: str,
    hint: str,
    length_bin: str,
    dataset: str,
    family: str,
    recommended_dense: bool = False,
) -> dict:
    response = "Given the provided hint. Final answer: Nairobi, Kenya.\n"
    return {
        "example_id": example_id,
        "target_model": "fake/model",
        "question": f"Question {example_id}",
        "base_question_id": family,
        "prompt": f"Prompt {example_id}",
        "response": response,
        "annotation_row_ids": [f"annotation-{example_id}"],
        "question_ids": [f"question-{example_id}"],
        "label_types": ["UNFAITHFUL_STEP"],
        "labeling_reasons": ["synthetic"],
        "hint_types": [hint],
        "hint_datasets": [dataset],
        "src_types": ["hinting"],
        "answer_records": [
            {
                "model_answer": "Nairobi, Kenya",
                "hinted_answer": "Nairobi, Kenya",
                "correct_answer": "Karachi, Pakistan",
            }
        ],
        "annotation_spans": [
            {
                "annotation_row_id": f"annotation-{example_id}",
                "label_type": "UNFAITHFUL_STEP",
                "sentence_span_start": 0,
                "sentence_span_end": 23,
                "sentence_span_valid": True,
                "extract_span_start": 0,
                "extract_span_end": 23,
                "extract_span_valid": True,
            }
        ],
        "diversity": {
            "cot_phenotype": phenotype,
            "answer_relation": "model_matches_hint_only",
            "annotation_position_bin": "first_quarter",
            "response_length_bin": length_bin,
            "total_length_bin": "513-768",
            "question_novelty_control_family_marker": "novel_singleton",
        },
        "selection_membership": {
            "dense_inventory": recommended_dense,
            "recommended_dense_core": recommended_dense,
            "broad_eligible_inventory": not recommended_dense,
            "broad_role": "primary",
        },
        "token_counts": {
            "assistant_prefix": 5,
            "response": len(response),
            "assistant_suffix": 1,
            "maximum_teacher_forced_input": len(response) + 5,
            "full_conversation_with_assistant_suffix": len(response) + 6,
        },
    }


def _aggregate(
    example: dict, inventory: str, workload_bin: str
) -> selection.PromptAggregate:
    return selection.PromptAggregate(
        example=example,
        inventory=inventory,
        targets=(),
        workload={
            "candidate_mlp_edge_count": {
                "p90": 100.0,
                "min": 10.0,
                "p50": 50.0,
                "max": 120.0,
                "mean": 60.0,
                "coefficient_of_variation": 0.2,
            }
        },
        feature_ids=frozenset({(0, len(example["example_id"]))}),
        feature_stability=0.5,
        target_sensitivity=0.2,
        workload_bin=workload_bin,
    )


def _frozen_broad() -> list[selection.PromptAggregate]:
    phenotypes = ["faithful", "omission", "commission", "both"]
    hints = [
        "error_message",
        "metadata",
        "security_audit",
        "sycophancy",
        "unauthorized_access",
        "validator",
    ]
    lengths = ["225-384", "385-512", "513-768"]
    workloads = ["low", "middle", "high"]
    aggregates: list[selection.PromptAggregate] = []
    for index, example_id in enumerate(selection.DEFAULT_BROAD_HOLDOUT_IDS):
        phenotype = phenotypes[index // 2]
        aggregates.append(
            _aggregate(
                _example(
                    example_id,
                    phenotype=phenotype,
                    hint=hints[index % 6],
                    length_bin=lengths[index % 3],
                    dataset="google_simpleqa-verified",
                    family=f"holdout-family-{index}",
                ),
                "broad_eligible_inventory",
                workloads[index % 3],
            )
        )
    datasets = (
        ["aai530-group6_ddxplus"] * 7
        + ["cais_hle"] * 5
        + ["google_simpleqa-verified"] * 12
    )
    for index, example_id in enumerate(selection.DEFAULT_BROAD_DISCOVERY_IDS):
        aggregates.append(
            _aggregate(
                _example(
                    example_id,
                    phenotype=phenotypes[index // 6],
                    hint=hints[index // 4],
                    length_bin=lengths[index // 8],
                    dataset=datasets[index],
                    family=f"discovery-family-{index}",
                ),
                "broad_eligible_inventory",
                workloads[index // 8],
            )
        )
    return aggregates


def _dense() -> list[selection.PromptAggregate]:
    ids = [f"dense-{index}" for index in range(10)]
    ids.append(selection.DEFAULT_DENSE_AUGMENTATION_IDS[0])
    result = []
    for index, example_id in enumerate(ids):
        example = _example(
            example_id,
            phenotype=("faithful", "omission", "commission", "both")[index % 4],
            hint=("validator", "metadata")[index % 2],
            length_bin="225-384",
            dataset="google_simpleqa-verified",
            family=f"dense-family-{index}",
            recommended_dense=index < 10,
        )
        if index == 10:
            example["selection_membership"].update(
                {"dense_inventory": True, "broad_eligible_inventory": False}
            )
        result.append(
            _aggregate(example, "dense_inventory", ("low", "middle", "high")[index % 3])
        )
    return result


def test_frozen_broad_membership_validates_every_exact_balance() -> None:
    selected = selection.validate_frozen_broad_prompts(_frozen_broad())

    assert len(selected) == 32
    assert [reason["selection_partition"] for _, reason in selected[:8]] == [
        "confirmatory_holdout"
    ] * 8
    assert [reason["selection_partition"] for _, reason in selected[8:]] == [
        "discovery"
    ] * 24


def test_frozen_broad_membership_fails_closed_on_workload_drift() -> None:
    aggregates = _frozen_broad()
    changed = aggregates[8]
    aggregates[8] = selection.PromptAggregate(
        example=changed.example,
        inventory=changed.inventory,
        targets=changed.targets,
        workload=changed.workload,
        feature_ids=changed.feature_ids,
        feature_stability=changed.feature_stability,
        target_sensitivity=changed.target_sensitivity,
        workload_bin="middle",
    )
    with pytest.raises(ValueError, match="workload_bin balance changed"):
        selection.validate_frozen_broad_prompts(aggregates)


def test_broad_refinement_targets_preserve_semantic_windows_and_phase_controls() -> (
    None
):
    tokenizer = CharTokenizer()
    example = _frozen_broad()[0].example
    response = example["response"]
    targets = selection._broad_refinement_targets(
        tokenizer=tokenizer,
        response=response,
        response_ids=[ord(character) for character in response],
        example=example,
        cap=32,
    )

    assert 16 <= len(targets) <= 32
    reasons = [reason for _, target in targets for reason in target["reasons"]]
    assert any(
        reason["reason_type"] == "bonafide_annotation_anchor" for reason in reasons
    )
    assert any(reason["reason_type"] == "answer_or_source_anchor" for reason in reasons)
    phase_indices = {
        reason["phase_index"]
        for reason in reasons
        if reason["reason_type"] == "phase_control"
    }
    assert phase_indices == set(range(16))


def test_screening_aggregation_fails_before_loading_incomplete_artifacts(
    tmp_path: Path,
) -> None:
    example = _example(
        "screened",
        phenotype="faithful",
        hint="metadata",
        length_bin="225-384",
        dataset="google_simpleqa-verified",
        family="screened-family",
    )
    candidate = {
        "schema_version": "bonafide-prompt-candidates/v1",
        "artifact_kind": "bonafide_prompt_candidates",
        "candidate_contract": {
            "prompt_candidates_selected": True,
            "target_spans_selected": False,
            "target_spans_frozen": False,
            "trace_work_items_created": False,
        },
        "examples": [example],
    }
    items = []
    summaries = []
    for position in range(16):
        source_id = f"probe-source-{position}"
        item = {
            "artifact_id": source_id,
            "example": example,
            "response_token_count": len(example["response"]),
            "target_selection": {
                "response_token_positions": [position],
                "final_target_token_id": ord(example["response"][position]),
                "screening_selection": {"candidate_inventory": "dense_inventory"},
            },
        }
        items.append(item)
        summaries.append({"source_artifact_id": source_id})
    screening = {
        "artifact_kind": "bonafide_prompt_screening_manifest",
        "screening_contract": {
            "probe_targets_frozen_for_this_estimation": True,
            "final_trace_prompt_membership_frozen": False,
            "final_trace_target_membership_frozen": False,
            "may_not_be_interpreted_as_final_trace_selection": True,
        },
        "source_selection": {"sha256": "a" * 64},
        "waves": [
            {
                "screening_design": {"targets_per_example": 16},
                "items": items,
            }
        ],
    }
    with pytest.raises(ValueError, match="incomplete; missing 1 records"):
        selection.aggregate_screening(
            candidate_selection=candidate,
            candidate_sha256="a" * 64,
            screening_manifest=screening,
            summary_records=summaries[:-1],
            artifact_root=tmp_path,
        )

    screening["source_selection"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="not bound to the candidate selection"):
        selection.aggregate_screening(
            candidate_selection=candidate,
            candidate_sha256="a" * 64,
            screening_manifest=screening,
            summary_records=summaries,
            artifact_root=tmp_path,
        )


def test_builds_one_resident_refinement_wave_and_leaves_final_targets_unfrozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregates = [*_dense(), *_frozen_broad()]
    monkeypatch.setattr(selection, "_validate_tokenizer_provenance", lambda **_: None)
    monkeypatch.setattr(
        selection,
        "_runtime_response_ids",
        lambda tokenizer, example: [
            ord(character) for character in example["response"]
        ],
    )
    candidate = {"dataset": {"sha256": "d" * 64}}
    screening = {"tokenizer": {"model_id": "fake/model", "revision": "r1"}}
    manifest = selection.build_refinement_probe_manifest(
        candidate_selection=candidate,
        candidate_path=Path("candidate.json"),
        candidate_sha256="a" * 64,
        screening_manifest=screening,
        screening_manifest_path=Path("screening.json"),
        screening_manifest_sha256="b" * 64,
        summary_path=Path("summary.jsonl"),
        summary_sha256="c" * 64,
        aggregates=aggregates,
        tokenizer=CharTokenizer(),
        tokenizer_path=Path("tokenizer"),
        broad_candidate_cap=32,
        expected_dense_probe_count=None,
    )

    assert manifest["artifact_kind"] == "bonafide_refinement_probe_manifest"
    assert manifest["selection_contract"]["prompt_membership_frozen"] is True
    assert (
        manifest["selection_contract"]["final_trace_target_membership_frozen"] is False
    )
    assert len(manifest["waves"]) == 1
    wave = manifest["waves"][0]
    assert wave["refinement_design"]["dense_probe_count"] == sum(
        aggregate.example["token_counts"]["response"] for aggregate in _dense()
    )
    assert 32 * 16 <= wave["refinement_design"]["broad_probe_count"] <= 32 * 32
    for item in wave["items"]:
        validate_target_selection(item)
        assert item["target_selection"]["width"] == 1
        assert "refinement_selection" in item["target_selection"]
