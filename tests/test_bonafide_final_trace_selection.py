"""CPU-only tests for the post-refinement BonaFide target freeze."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from circuits.tracing.probe_artifact import ProbeArtifact
from scripts.bonafide import final_trace_selection as final
from scripts.bonafide.runner import _sha256, validate_target_selection


def _example(example_id: str, response_count: int, phenotype: str = "commission") -> dict:
    return {
        "example_id": example_id,
        "target_model": "fake/model",
        "question": "Question",
        "base_question_id": f"family-{example_id}",
        "prompt": "Prompt",
        "response": "x" * response_count,
        "annotation_row_ids": ["annotation"],
        "question_ids": ["question"],
        "label_types": ["UNFAITHFUL_STEP"],
        "labeling_reasons": ["synthetic"],
        "hint_types": ["metadata"],
        "hint_datasets": ["google_simpleqa-verified"],
        "src_types": ["hinting"],
        "diversity": {"cot_phenotype": phenotype},
        "selection_membership": {},
        "token_counts": {"response": response_count},
    }


def _item(
    example: dict,
    position: int,
    *,
    role: str,
    reasons: list[dict] | None = None,
) -> dict:
    target_reason = (
        {"policy": "all_teacher_forced_response_positions", "full_response_coverage": True}
        if role == "dense_full_response_refinement"
        else {"policy": "semantic_boundary_refinement_candidates", "reasons": reasons or []}
    )
    return {
        "artifact_id": f"probe-source-{example['example_id']}-{position}",
        "example": example,
        "response_token_count": len(example["response"]),
        "target_selection": {
            "kind": "explicit_response_positions",
            "response_token_positions": [position],
            "width": 1,
            "final_target_token_id": 1000 + position,
            "refinement_selection": {
                "corpus_role": role,
                "target_selection_reason": target_reason,
            },
        },
        "objective": {"name": "single_selected_logit", "benchmark_only_multi_target": False},
    }


def _probe(item: dict, *, probability: float | None = None) -> final.RefinementProbe:
    position = item["target_selection"]["response_token_positions"][0]
    # Deliberately create a sharp feature transition at position 40.
    feature_group = 0 if position < 40 else 1
    cohort_identity = {
        "mode": "teacher_forced_probe",
        "batch_size": 1,
        "model": {"model_id": "fake/model", "revision": "r1"},
        "adag_config": {"edge_threshold": 0.01},
        "code_revision": {"git_commit": "a" * 40},
    }
    return final.RefinementProbe(
        item=item,
        artifact_path=Path(f"/artifact/{position}"),
        artifact_id=f"runtime-{item['artifact_id']}",
        probability=float(probability if probability is not None else 0.5 + position / 1000),
        candidate_edge_count=100 + position * 10,
        selected_occurrence_count=20 + position,
        feature_ids=frozenset({(feature_group, feature_group * 100 + position % 3)}),
        probe_sha256=f"{position:064x}",
        metrics_sha256=f"{position + 1000:064x}",
        cohort_identity=cohort_identity,
        cohort_identity_sha256=_sha256(cohort_identity),
    )


def _broad_probes(example_id: str = "broad", phenotype: str = "commission") -> list[final.RefinementProbe]:
    example = _example(example_id, 64, phenotype)
    probes = []
    for position in range(64):
        reasons: list[dict] = [
            {
                "reason_type": "semantic_boundary_control",
                "micro_window_center": position,
            }
        ]
        if position == 10:
            reasons.append(
                {
                    "reason_type": "answer_or_source_anchor",
                    "phrase_type": "hinted_answer",
                    "phrase": "Nairobi, Kenya",
                    "phrase_boundary": "end",
                    "character_span": [100, 110],
                    "micro_window_center": 10,
                }
            )
        if position == 30:
            reasons.append(
                {
                    "reason_type": "bonafide_annotation_anchor",
                    "label_type": "UNFAITHFUL_STEP",
                    "micro_window_center": 30,
                }
            )
        if position == 55:
            reasons.append(
                {
                    "reason_type": "answer_or_source_anchor",
                    "phrase_type": "model_answer",
                    "phrase": "Nairobi, Kenya",
                    "phrase_boundary": "end",
                    "character_span": [500, 510],
                    "micro_window_center": 55,
                }
            )
        phase_by_position = {0: 0, 20: 5, 40: 10, 63: 15}
        if position in phase_by_position:
            reasons.append(
                {
                    "reason_type": "phase_control",
                    "phase_index": phase_by_position[position],
                    "phase_count": 16,
                }
            )
        item = _item(
            example,
            position,
            role="broad_semantic_boundary_refinement",
            reasons=reasons,
        )
        probes.append(_probe(item, probability=0.01 if position == 25 else None))
    return probes


def _refinement_manifest(items: list[dict], prompt_analysis: list[dict] | None = None) -> dict:
    return {
        "schema_version": "bonafide-trace-benchmark/v1",
        "artifact_kind": "bonafide_refinement_probe_manifest",
        "selection_contract": {
            "prompt_membership_frozen": True,
            "refinement_probe_membership_frozen": True,
            "final_trace_target_membership_frozen": False,
        },
        "dataset": {"sha256": "d" * 64},
        "tokenizer": {"model_id": "fake/model", "revision": "r1"},
        "prompt_analysis": prompt_analysis or [],
        "waves": [{"wave_id": "prompt-refinement-probes", "items": items}],
    }


def test_broad_freeze_selects_16_and_preserves_reviewed_bucket_reasons() -> None:
    chosen = final.select_broad_targets(_broad_probes(), phenotype="commission")

    assert len(chosen) == 16
    assert len({probe.position for probe, _ in chosen}) == 16
    buckets = {reason["bucket"] for _, reasons in chosen for reason in reasons}
    assert {
        "first_hinted_answer_commitment_window",
        "source_or_unsupported_evidence_window",
        "final_answer_commitment_window",
        "phase_control",
        "low_probability_semantic",
        "large_adjacent_feature_change",
        "median_workload_control",
    }.issubset(buckets)
    assert any(probe.position == 25 for probe, _ in chosen)
    assert any(probe.position == 40 for probe, _ in chosen)


def test_deduplication_refills_deterministically_and_keeps_all_reasons() -> None:
    probes = _broad_probes()
    # Collapse the final-answer anchor onto the first commitment occurrence.
    item = json.loads(json.dumps(probes[55].item))
    item["target_selection"]["refinement_selection"]["target_selection_reason"]["reasons"] = []
    probes[55] = _probe(item)
    first_item = json.loads(json.dumps(probes[10].item))
    first_item["target_selection"]["refinement_selection"]["target_selection_reason"][
        "reasons"
    ].append(
        {
            "reason_type": "answer_or_source_anchor",
            "phrase_type": "model_answer",
            "phrase": "Nairobi, Kenya",
            "phrase_boundary": "end",
            "character_span": [100, 110],
            "micro_window_center": 10,
        }
    )
    probes[10] = _probe(first_item)

    first = final.select_broad_targets(probes, phenotype="commission")
    second = final.select_broad_targets(probes, phenotype="commission")
    assert [(probe.position, reasons) for probe, reasons in first] == [
        (probe.position, reasons) for probe, reasons in second
    ]
    assert len(first) == 16
    buckets_at_ten = {
        reason["bucket"]
        for probe, reasons in first
        if probe.position == 10
        for reason in reasons
    }
    assert {
        "first_hinted_answer_commitment_window",
        "final_answer_commitment_window",
    }.issubset(buckets_at_ten)
    assert any(
        reason["bucket"] == "deterministic_refill_after_deduplication"
        for _, reasons in first
        for reason in reasons
    )


def test_curated_annotation_beats_earlier_generic_source_marker_for_faithful() -> None:
    probes = _broad_probes(phenotype="faithful")
    item = json.loads(json.dumps(probes[5].item))
    item["target_selection"]["refinement_selection"]["target_selection_reason"][
        "reasons"
    ].append(
        {
            "reason_type": "answer_or_source_anchor",
            "phrase_type": "source_marker",
            "phrase": "information",
            "micro_window_center": 5,
        }
    )
    probes[5] = _probe(item)

    chosen = final.select_broad_targets(probes, phenotype="faithful")
    source_reasons = [
        reason
        for _, reasons in chosen
        for reason in reasons
        if reason["bucket"] == "source_or_unsupported_evidence_window"
    ]
    assert source_reasons
    assert {reason["window_center"] for reason in source_reasons} == {30}
    assert {
        reason["anchor"]["reason_type"] for reason in source_reasons
    } == {"bonafide_annotation_anchor"}


def test_numeric_answer_anchor_falls_back_to_annotation_and_final_phase() -> None:
    probes = _broad_probes()
    for position in (10, 55):
        item = json.loads(json.dumps(probes[position].item))
        reasons = item["target_selection"]["refinement_selection"][
            "target_selection_reason"
        ]["reasons"]
        for reason in reasons:
            if reason.get("phrase_type") in {"hinted_answer", "model_answer"}:
                reason["phrase"] = "3"
                reason["character_span"] = [3, 4]
        probes[position] = _probe(item)

    chosen = final.select_broad_targets(probes, phenotype="commission")
    first_reasons = [
        reason
        for _, reasons in chosen
        for reason in reasons
        if reason["bucket"] == "first_hinted_answer_commitment_window"
    ]
    final_reasons = [
        reason
        for _, reasons in chosen
        for reason in reasons
        if reason["bucket"] == "final_answer_commitment_window"
    ]
    assert {reason["window_center"] for reason in first_reasons} == {30}
    assert {
        reason["anchor"]["fallback"] for reason in first_reasons
    } == {"curated_source_or_annotation_center"}
    assert {reason["window_center"] for reason in final_reasons} == {63}
    assert {reason["anchor"]["fallback"] for reason in final_reasons} == {
        "final_phase_control"
    }


def test_states_substring_is_not_treated_as_source_attribution() -> None:
    probes = _broad_probes(phenotype="omission")
    annotation_item = json.loads(json.dumps(probes[30].item))
    annotation_item["target_selection"]["refinement_selection"][
        "target_selection_reason"
    ]["reasons"] = [
        reason
        for reason in annotation_item["target_selection"]["refinement_selection"][
            "target_selection_reason"
        ]["reasons"]
        if reason.get("reason_type") != "bonafide_annotation_anchor"
    ]
    probes[30] = _probe(annotation_item)
    states_item = json.loads(json.dumps(probes[5].item))
    states_item["target_selection"]["refinement_selection"][
        "target_selection_reason"
    ]["reasons"].append(
        {
            "reason_type": "answer_or_source_anchor",
            "phrase_type": "source_marker",
            "phrase": "states",
            "character_span": [10, 16],
            "micro_window_center": 5,
        }
    )
    probes[5] = _probe(states_item)

    candidates = [
        final.BroadCandidate(
            probe=probe, candidate_reasons=final._candidate_reasons(probe.item)
        )
        for probe in probes
    ]
    center, reason = final._source_anchor_center(candidates, phenotype="omission")
    assert center == 25
    assert reason["reason_type"] == "unsupported_evidence_fallback"


def test_append_only_summary_accepts_complete_then_skipped_duplicate() -> None:
    records = [
        {
            "wave_id": "wave",
            "source_artifact_id": "source",
            "artifact_id": "runtime",
            "status": "complete",
        },
        {
            "wave_id": "wave",
            "source_artifact_id": "source",
            "artifact_id": "runtime",
            "status": "skipped_complete",
        },
    ]
    audit = final.audit_append_only_summary(
        records, expected_source_ids={"source"}, expected_wave_id="wave"
    )
    assert audit["sources_with_repeated_records"] == 1
    assert audit["completed_runtime_ids"] == {"source": ["runtime"]}


def test_probe_cohort_rejects_mixed_model_config_or_code_identity() -> None:
    probes = _broad_probes()[:2]
    mixed_identity = {**probes[1].cohort_identity, "code_revision": {"git_commit": "b" * 40}}
    probes[1] = replace(
        probes[1],
        cohort_identity=mixed_identity,
        cohort_identity_sha256=_sha256(mixed_identity),
    )
    with pytest.raises(ValueError, match="mix model, ADAG config, or code-revision"):
        final._audit_probe_cohort(probes)


def test_authoritative_artifact_overrides_append_only_summary_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = _example("dense", 1)
    item = _item(example, 0, role="dense_full_response_refinement")
    manifest = _refinement_manifest([item])
    artifact_dir = tmp_path / "prompt-refinement-probes" / "runtime"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"source_artifact_id": item["artifact_id"]})
    )
    cohort_identity = {
        "mode": "teacher_forced_probe",
        "batch_size": 1,
        "model": {"model_id": "fake/model", "revision": "r1"},
        "adag_config": {"edge_threshold": 0.01},
        "code_revision": {"git_commit": "a" * 40},
    }
    loaded = ProbeArtifact(
        path=artifact_dir,
        manifest={
            "source_artifact_id": item["artifact_id"],
            "artifact_id": "runtime",
            "artifact_identity": {
                "source_work_item_sha256": _sha256(item),
                "source_target_selection": item["target_selection"],
                **cohort_identity,
            },
            "model_revision": "r1",
            "code_revision": cohort_identity["code_revision"],
            "probe_sha256": "a" * 64,
            "metrics_sha256": "b" * 64,
        },
        probe={
            "target_provenance": {
                "response_token_position": 0,
                "token_id": 1000,
                "probability": 0.5,
            },
            "feature_basis_signature": {"feature_ids": [[0, 1]]},
        },
        metrics={
            "selected_occurrence_count": 1,
            "instrumentation": {"counters": {"candidate_mlp_edge_count": 2}},
        },
    )
    monkeypatch.setattr(final, "load_probe_artifact", lambda path: loaded)
    records = [
        {
            "wave_id": "prompt-refinement-probes",
            "source_artifact_id": item["artifact_id"],
            "artifact_id": "runtime",
            "status": "skipped_complete",
        },
        {
            "wave_id": "prompt-refinement-probes",
            "source_artifact_id": item["artifact_id"],
            "artifact_id": "runtime",
            "status": "complete",
        },
    ]
    probes, audit = final.load_authoritative_refinement_probes(
        manifest=manifest, artifact_root=tmp_path, summary_records=records
    )
    assert probes[item["artifact_id"]].artifact_id == "runtime"
    assert audit["authoritative_artifact_count"] == 1
    assert audit["sources_with_repeated_records"] == 1
    assert audit["homogeneous_probe_cohort"] is True


def test_builds_per_prompt_waves_and_excludes_holdout_from_cluster_fit() -> None:
    dense_example = _example("dense", 4)
    dense_probes = [
        _probe(_item(dense_example, position, role="dense_full_response_refinement"))
        for position in range(4)
    ]
    discovery = _broad_probes("discovery", "faithful")
    holdout = _broad_probes("holdout", "omission")
    all_probes = [*dense_probes, *discovery, *holdout]
    manifest = _refinement_manifest(
        [probe.item for probe in all_probes],
        prompt_analysis=[
            {
                "example_id": "discovery",
                "selection_reason": {"selection_partition": "discovery"},
            },
            {
                "example_id": "holdout",
                "selection_reason": {"selection_partition": "confirmatory_holdout"},
            },
        ],
    )
    result = final.build_final_trace_manifest(
        refinement_manifest=manifest,
        refinement_manifest_path=Path("refinement.json"),
        refinement_manifest_sha256="a" * 64,
        refinement_summary_path=Path("summary.jsonl"),
        refinement_summary_sha256="b" * 64,
        refinement_artifact_root=Path("artifacts"),
        probes_by_source={probe.item["artifact_id"]: probe for probe in all_probes},
        summary_audit={"authoritative_artifact_count": len(all_probes)},
    )

    assert len(result["waves"]) == 3
    assert result["source_artifacts"]["refinement_probe_artifact_root"] == "artifacts"
    by_role = {wave["corpus_role"]: wave for wave in result["waves"]}
    assert len(by_role["dense_discovery"]["items"]) == 4
    assert len(by_role["broad_discovery"]["items"]) == 16
    assert len(by_role["broad_confirmatory_holdout"]["items"]) == 16
    assert by_role["broad_confirmatory_holdout"]["cluster_fit_eligible"] is False
    assert by_role["broad_confirmatory_holdout"][
        "holdout_excluded_from_cluster_fitting"
    ] is True
    for wave in result["waves"]:
        for item in wave["items"]:
            validate_target_selection(item)
