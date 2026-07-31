"""Deterministic, conservative post-audit assessment of width-one labels."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.config import LabelingRecipe
from circuits.labeling.io import atomic_write_json
from circuits.labeling.provenance import validate_local_score_artifact
from circuits.labeling.runtime import (
    collect_code_revision,
    load_run_manifest,
    load_stage_requests,
)
from circuits.labeling.schema import GenerationResult

QUALITY_SCHEMA = "adag.labeling.width-one-quality.v1"
QUALITY_MANIFEST_SCHEMA = "adag.labeling.width-one-quality-manifest.v1"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def conservative_quality_status(
    best_candidate_correlation: Any,
    *,
    model_status: Any,
    model_label: Any,
) -> tuple[str, list[str]]:
    """Return a selection-only decision; audit results are intentionally absent."""

    best = _finite_number(best_candidate_correlation)
    reasons: list[str] = []
    if best is None:
        reasons.append("best_candidate_correlation_nonfinite")
    elif best <= 0:
        reasons.append("best_candidate_correlation_not_positive")
    if (
        model_status == "insufficient_evidence"
        or model_label == "insufficient_evidence"
    ):
        reasons.append("model_reported_insufficient_evidence")
    elif model_status != "provisional_label":
        reasons.append("missing_valid_provisional_model_status")
    return ("insufficient_evidence" if reasons else "review_required", reasons)


def best_finite_score(scores: list[dict[str, Any]]) -> dict[str, Any] | None:
    finite = [
        (_finite_number(score.get("correlation")), score)
        for score in scores
        if _finite_number(score.get("correlation")) is not None
    ]
    if not finite:
        return None
    return max(finite, key=lambda item: (item[0], str(item[1].get("request_id"))))[1]


def assess_width_one_quality(*, run_root: Path) -> dict[str, Any]:
    """Assess v2 outputs without promoting simulator scores to acceptance gates."""

    run_manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(run_manifest["recipe"])
    if recipe.prompt_policy != "width_one_v2":
        raise ValueError("quality assessment is defined only for width_one_v2 runs")
    assessment_root = run_root / "assessments" / "label_quality"
    if assessment_root.exists():
        raise FileExistsError(f"quality assessment already exists: {assessment_root}")
    staging_root = assessment_root.parent / f".label_quality.tmp-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True)
    try:
        value = _write_quality_assessment(
            run_root=run_root,
            assessment_root=staging_root,
            run_manifest=run_manifest,
            recipe=recipe,
        )
        assessment_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root, assessment_root)
        return value
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def _write_quality_assessment(
    *,
    run_root: Path,
    assessment_root: Path,
    run_manifest: dict[str, Any],
    recipe: LabelingRecipe,
) -> dict[str, Any]:
    stage_manifest_path = run_root / "stages" / "cluster_summary" / "manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    expected_stage_hash = stage_manifest.pop("manifest_sha256", None)
    if expected_stage_hash != canonical_sha256(stage_manifest):
        raise ValueError("cluster summary stage manifest hash mismatch")
    evidence_inputs = {
        (item["state"], int(item["cluster_id"])): item
        for item in stage_manifest.get("evidence_inputs", [])
    }
    summary_requests = {
        (request.state, request.cluster_id): request
        for request in load_stage_requests(run_root, "cluster_summary")
    }
    candidate_requests = load_stage_requests(run_root, "candidate_generation")

    files: list[dict[str, Any]] = []
    counts = {"insufficient_evidence": 0, "review_required": 0}
    for state, raw_cluster_ids in run_manifest["selected_clusters"].items():
        for raw_cluster_id in raw_cluster_ids:
            cluster_id = int(raw_cluster_id)
            request = summary_requests.get((state, cluster_id))
            if request is None:
                raise ValueError(
                    f"summary request is missing: {state} cluster {cluster_id}"
                )

            candidate_path = (
                run_root
                / "scores"
                / "candidate_selection"
                / state
                / f"cluster-{cluster_id:04d}.json"
            )
            summary_path = (
                run_root / "results" / "cluster_summary" / f"{request.request_id}.json"
            )
            audit_path = (
                run_root
                / "scores"
                / "summary_audit"
                / state
                / f"cluster-{cluster_id:04d}.json"
            )
            for label, path in (
                ("candidate selection", candidate_path),
                ("summary result", summary_path),
                ("summary audit", audit_path),
            ):
                if not path.is_file():
                    raise ValueError(f"{label} artifact is missing: {path}")

            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            evidence_input = evidence_inputs.get((state, cluster_id))
            if (
                evidence_input is None
                or evidence_input.get("candidate_score_path")
                != candidate_path.relative_to(run_root).as_posix()
                or evidence_input.get("candidate_score_sha256")
                != file_sha256(candidate_path)
            ):
                raise ValueError(
                    f"candidate score is not hash-bound by summary stage: "
                    f"{state} cluster {cluster_id}"
                )
            summary = GenerationResult.model_validate_json(
                summary_path.read_text(encoding="utf-8")
            )
            if summary.request_id != request.request_id:
                raise ValueError(f"summary result request ID mismatch: {summary_path}")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            expected_candidate_ids: set[str] = set()
            for candidate_request in candidate_requests:
                if (
                    candidate_request.state != state
                    or candidate_request.cluster_id != cluster_id
                ):
                    continue
                result_path = (
                    run_root
                    / "results"
                    / "candidate_generation"
                    / f"{candidate_request.request_id}.json"
                )
                if not result_path.is_file():
                    raise ValueError(f"candidate result is missing: {result_path}")
                candidate_result = GenerationResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
                if candidate_result.request_id != candidate_request.request_id:
                    raise ValueError(
                        f"candidate result request ID mismatch: {result_path}"
                    )
                if (
                    candidate_result.parse_status == "success"
                    and isinstance(candidate_result.parsed, dict)
                    and isinstance(candidate_result.parsed.get("description"), str)
                    and candidate_result.parsed["description"].strip()
                ):
                    expected_candidate_ids.add(candidate_request.request_id)
            candidate_scores = validate_local_score_artifact(
                candidate,
                recipe=recipe,
                run_id=run_manifest["run_id"],
                phase="candidate_selection",
                state=state,
                cluster_id=cluster_id,
                expected_request_ids=expected_candidate_ids,
            )
            audit_scores = validate_local_score_artifact(
                audit,
                recipe=recipe,
                run_id=run_manifest["run_id"],
                phase="summary_audit",
                state=state,
                cluster_id=cluster_id,
                expected_request_ids={request.request_id},
            )

            best_score = best_finite_score(candidate_scores)
            best_raw = best_score.get("correlation") if best_score else None
            best_correlation = _finite_number(best_raw)
            parsed = summary.parsed if summary.parse_status == "success" else None
            model_status = parsed.get("status") if isinstance(parsed, dict) else None
            model_label = parsed.get("label") if isinstance(parsed, dict) else None
            status, reasons = conservative_quality_status(
                best_raw, model_status=model_status, model_label=model_label
            )
            counts[status] += 1
            audit_score = audit_scores[0]
            audit_correlation = _finite_number(audit_score.get("correlation"))
            relative = Path(state) / f"cluster-{cluster_id:04d}.json"
            output = assessment_root / relative
            value: dict[str, Any] = {
                "schema_version": QUALITY_SCHEMA,
                "run_id": run_manifest["run_id"],
                "recipe_id": recipe.recipe_id,
                "prompt_policy": recipe.prompt_policy,
                "state": state,
                "cluster_id": cluster_id,
                "status": status,
                "decision_reasons": reasons or ["requires_manual_semantic_review"],
                "model_output": {
                    "request_id": request.request_id,
                    "status": model_status,
                    "label": model_label,
                    "prompt_template_version": request.prompt_template_version,
                    "prompt_sha256": request.prompt_sha256,
                },
                "selection_decision_input": {
                    "best_candidate_request_id": (
                        best_score.get("request_id") if best_score else None
                    ),
                    "best_candidate_correlation": best_raw,
                    "finite": best_correlation is not None,
                },
                "audit_evaluation": {
                    "request_id": audit_score.get("request_id"),
                    "correlation": audit_score.get("correlation"),
                    "rsquared": audit_score.get("rsquared"),
                    "finite": audit_correlation is not None,
                    "gates_label_decision": False,
                    "interpretation": "separate held-out evaluation; never acceptance or rewrite evidence",
                },
                "evidence_limitations": {
                    "trace_scope": "single_target_width_one",
                    "contribution_evidence": "shallow",
                    "non_degenerate_contribution_comparison": False,
                    "top_k_target_comparison": False,
                },
                "provenance": {
                    "run_manifest_sha256": run_manifest["manifest_sha256"],
                    "source_manifest_sha256": run_manifest["source_manifest_sha256"],
                    "evidence_sha256": request.evidence_sha256,
                    "candidate_score_path": candidate_path.relative_to(
                        run_root
                    ).as_posix(),
                    "candidate_score_sha256": file_sha256(candidate_path),
                    "summary_result_path": summary_path.relative_to(
                        run_root
                    ).as_posix(),
                    "summary_result_sha256": file_sha256(summary_path),
                    "audit_score_path": audit_path.relative_to(run_root).as_posix(),
                    "audit_score_sha256": file_sha256(audit_path),
                },
            }
            value["assessment_sha256"] = canonical_sha256(value)
            atomic_write_json(output, value)
            files.append(
                {
                    "path": relative.as_posix(),
                    "sha256": file_sha256(output),
                    "status": status,
                }
            )

    quality_manifest: dict[str, Any] = {
        "schema_version": QUALITY_MANIFEST_SCHEMA,
        "run_id": run_manifest["run_id"],
        "recipe_id": recipe.recipe_id,
        "prompt_policy": recipe.prompt_policy,
        "source_run_manifest_sha256": run_manifest["manifest_sha256"],
        "assessor_code_revision": collect_code_revision(),
        "decision_policy": {
            "best_candidate_nonfinite_or_not_positive": "insufficient_evidence",
            "model_status_insufficient": "insufficient_evidence",
            "otherwise": "review_required",
            "automatic_acceptance": False,
            "audit_gates_decision": False,
        },
        "counts": counts,
        "files": files,
    }
    quality_manifest["manifest_sha256"] = canonical_sha256(quality_manifest)
    atomic_write_json(assessment_root / "manifest.json", quality_manifest)
    return quality_manifest
