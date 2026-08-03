from __future__ import annotations

import json
from pathlib import Path

import pytest

import circuits.analysis.bonafide.candidate_labeling_execution as execution_module
from circuits.analysis.bonafide.candidate_labeling_execution import (
    _validate_output,
    execute_candidate_labeling_fake_evaluation,
    load_candidate_labeling_fake_evaluation,
)
from circuits.analysis.bonafide.candidate_labeling_renderer import (
    STATUS_ENUM,
    TYPED_OUTPUT_FIELDS,
    LoadedCandidateLabelingRenderer,
)
from circuits.analysis.bonafide.candidate_labeling_runtime import (
    LoadedCandidateLabelingExecutionCohort,
    PreparedCandidateLabelingRequest,
    RewriteDependency,
    build_candidate_labeling_execution_cohort,
    load_candidate_labeling_runtime_config,
)
from circuits.analysis.bonafide.canonical import canonical_sha256

CONFIG = Path("scripts/bonafide/configs/labeling/openai-c2-evidence-v1.json")


def _output_schema(include_candidate: bool) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(TYPED_OUTPUT_FIELDS),
        "properties": {
            "input_localization_hypothesis": {"type": "string", "minLength": 1},
            "exploratory_candidate_description": (
                {"type": "string", "minLength": 1}
                if include_candidate
                else {"const": "not_available"}
            ),
            "background_or_confound": {"type": "string", "minLength": 1},
            "limitations": {"type": "string", "minLength": 1},
            "status": {"type": "string", "enum": list(STATUS_ENUM)},
        },
    }


def _cohort(tmp_path: Path) -> LoadedCandidateLabelingExecutionCohort:
    renderer_root = tmp_path / "renderer"
    renderer_root.mkdir()
    prompts = []
    for arm_id, include_candidate in (
        ("arm_1_width_only", False),
        ("arm_2_width_plus_candidate", True),
    ):
        for index in range(12):
            message_payload = {
                "messages": [
                    {"role": "system", "content": "bounded generation evidence"},
                    {"role": "user", "content": f"arm {arm_id} cluster {index}"},
                ],
                "expected_output_json_schema": _output_schema(include_candidate),
            }
            prompt = {
                "logical_prompt_id": f"{arm_id}:w64:{index:02d}",
                "arm_id": arm_id,
                "anchor_index": index,
                "cluster_id": index,
                "candidate_evidence_included": include_candidate,
                "message_payload": message_payload,
                "message_payload_sha256": canonical_sha256(message_payload),
                "width_evidence_sha256": f"{index:064x}",
                "rendered_candidate_witness_sha256_in_order": (
                    [f"{index + offset + 1:064x}" for offset in range(8)]
                    if include_candidate
                    else None
                ),
            }
            prompt["prompt_sha256"] = canonical_sha256(prompt)
            prompts.append(prompt)
    renderer = LoadedCandidateLabelingRenderer(
        root=renderer_root,
        manifest={"manifest_sha256": "a" * 64},
        witness_selection={},
        generation_prompts=tuple(prompts),
        stage_plan={},
    )
    config = load_candidate_labeling_runtime_config(CONFIG)
    rows, dependencies, _ = build_candidate_labeling_execution_cohort(renderer, config)
    root = tmp_path / "cohort"
    root.mkdir()
    manifest = {"manifest_sha256": "b" * 64}
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return LoadedCandidateLabelingExecutionCohort(
        root=root,
        manifest=manifest,
        config=config,
        initial_requests=tuple(
            PreparedCandidateLabelingRequest.model_validate(row) for row in rows
        ),
        rewrite_dependencies=tuple(
            RewriteDependency.model_validate(row) for row in dependencies
        ),
    )


def _patch_sources(
    monkeypatch: pytest.MonkeyPatch,
    cohort: LoadedCandidateLabelingExecutionCohort,
) -> None:
    monkeypatch.setattr(
        execution_module,
        "load_candidate_labeling_execution_cohort",
        lambda *args, **kwargs: cohort,
    )
    monkeypatch.setattr(
        execution_module,
        "collect_candidate_labeling_execution_revision",
        lambda: {
            "repo_root": str(cohort.root.parent),
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "tracked_worktree_clean": True,
            "tracked_status_sha256": "3" * 64,
            "files": [
                {
                    "role": role,
                    "path": path,
                    "git_blob": "4" * 40,
                    "sha256": "5" * 64,
                }
                for role, path in execution_module._SOURCE_BINDINGS.items()
            ],
        },
    )


def test_fake_execution_is_complete_resumable_and_paired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = _cohort(tmp_path)
    _patch_sources(monkeypatch, cohort)
    root = tmp_path / "evaluation"

    first = execute_candidate_labeling_fake_evaluation(
        cohort_root=cohort.root, output_root=root
    )
    resumed = execute_candidate_labeling_fake_evaluation(
        cohort_root=cohort.root, output_root=root
    )

    assert first == resumed
    assert len(first.events) == 168
    assert len(first.rewrite_requests) == 24
    assert first.paired_summary["counts"]["by_stage"] == {
        "conservative_control": 24,
        "semantic_generation": 120,
        "semantic_rewrite": 24,
    }
    assert len(first.paired_summary["paired_arm_summaries"]) == 12
    assert first.paired_summary["telemetry"] == {
        "executor": "deterministic_fake",
        "network_call_count": 0,
        "api_call_count": 0,
        "billable_event_count": 0,
        "known_cost_usd": 0.0,
    }
    assert all(
        len(request.required_semantic_request_ids) == 5
        and len(request.validated_semantic_output_sha256_in_order) == 5
        for request in first.rewrite_requests
    )


def test_loader_rejects_tampered_fake_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = _cohort(tmp_path)
    _patch_sources(monkeypatch, cohort)
    root = tmp_path / "evaluation"
    evaluation = execute_candidate_labeling_fake_evaluation(
        cohort_root=cohort.root, output_root=root
    )
    event = evaluation.events[0]
    event_path = root / "events" / event.stage_id / f"{event.request_sha256}.json"
    payload = json.loads(event_path.read_text())
    payload["parsed"]["limitations"] = "tampered"
    event_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="self-hash drift"):
        load_candidate_labeling_fake_evaluation(root, cohort_root=cohort.root)


def test_loader_rejects_tampered_execution_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = _cohort(tmp_path)
    _patch_sources(monkeypatch, cohort)
    root = tmp_path / "evaluation"
    execute_candidate_labeling_fake_evaluation(
        cohort_root=cohort.root, output_root=root
    )
    manifest_path = root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["execution_revision"]["git_tree"] = "9" * 40
    unhashed = dict(manifest)
    unhashed.pop("run_manifest_sha256")
    manifest["run_manifest_sha256"] = canonical_sha256(unhashed)
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="no longer matches source HEAD"):
        load_candidate_labeling_fake_evaluation(root, cohort_root=cohort.root)


def test_exact_output_validation_rejects_const_status_and_extra_fields() -> None:
    schema = _output_schema(False)
    valid = {
        "input_localization_hypothesis": "local",
        "exploratory_candidate_description": "not_available",
        "background_or_confound": "background",
        "limitations": "bounded",
        "status": "provisional_description",
    }
    assert (
        _validate_output(
            valid,
            expected_schema=schema,
            typed_fields=TYPED_OUTPUT_FIELDS,
            status_enum=STATUS_ENUM,
        )
        == valid
    )

    for changed in (
        {**valid, "exploratory_candidate_description": "invented"},
        {**valid, "status": "accepted"},
        {**valid, "extra": "forbidden"},
    ):
        with pytest.raises(ValueError):
            _validate_output(
                changed,
                expected_schema=schema,
                typed_fields=TYPED_OUTPUT_FIELDS,
                status_enum=STATUS_ENUM,
            )
