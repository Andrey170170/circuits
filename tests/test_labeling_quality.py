from __future__ import annotations

import math
from pathlib import Path

import pytest

from circuits.labeling.quality import (
    assess_width_one_quality,
    best_finite_score,
    conservative_quality_status,
)


@pytest.mark.parametrize("correlation", [-0.1, 0.0, None, math.nan, math.inf])
def test_nonpositive_or_nonfinite_final_label_is_insufficient(correlation) -> None:
    status, _ = conservative_quality_status(
        correlation, model_status="provisional_label", model_label="localized feature"
    )
    assert status == "insufficient_evidence"


def test_model_can_report_insufficient_evidence() -> None:
    status, reasons = conservative_quality_status(
        0.2,
        model_status="insufficient_evidence",
        model_label="insufficient_evidence",
    )
    assert status == "insufficient_evidence"
    assert "model_reported_insufficient_evidence" in reasons


def test_positive_selection_is_review_required_never_autoaccepted() -> None:
    status, reasons = conservative_quality_status(
        0.2, model_status="provisional_label", model_label="localized feature"
    )
    assert status == "review_required"
    assert reasons == []


def test_negative_candidate_does_not_override_positive_final_label() -> None:
    best_candidate_correlation = -0.9
    status, reasons = conservative_quality_status(
        0.2, model_status="provisional_label", model_label="rewritten label"
    )
    assert best_candidate_correlation < 0
    assert status == "review_required"
    assert reasons == []


def test_positive_candidate_does_not_rescue_negative_final_label() -> None:
    best_candidate_correlation = 0.9
    status, reasons = conservative_quality_status(
        -0.2, model_status="provisional_label", model_label="rewritten label"
    )
    assert best_candidate_correlation > 0
    assert status == "insufficient_evidence"
    assert reasons == ["final_label_correlation_not_positive"]


@pytest.mark.parametrize("audit_correlation", [-0.9, 0.9])
def test_audit_correlation_is_not_a_quality_input(audit_correlation: float) -> None:
    status, reasons = conservative_quality_status(
        0.2, model_status="provisional_label", model_label="localized feature"
    )
    assert math.isfinite(audit_correlation)
    assert status == "review_required"
    assert reasons == []


def test_best_score_ignores_nonfinite_values() -> None:
    best = best_finite_score(
        [
            {"request_id": "nan", "correlation": math.nan},
            {"request_id": "finite", "correlation": 0.2},
            {"request_id": "inf", "correlation": math.inf},
        ]
    )
    assert best == {"request_id": "finite", "correlation": 0.2}


def test_quality_transaction_cleans_staging_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recipe = {
        "schema_version": "adag.labeling.recipe.v1",
        "recipe_id": "transaction-width-one-v2",
        "description": "test",
        "prompt_policy": "width_one_v2",
        "candidate_samples": 1,
        "candidate_generator": {
            "provider": "fake",
            "model": "fake",
            "max_output_tokens": 100,
        },
        "scorer": {},
        "cluster_summarizer": {
            "provider": "fake",
            "model": "fake",
            "max_output_tokens": 1200,
        },
        "price_snapshot": "prices.json",
    }
    monkeypatch.setattr(
        "circuits.labeling.quality.load_run_manifest",
        lambda root: {"recipe": recipe},
    )

    def fail(**kwargs):
        (kwargs["assessment_root"] / "partial").write_text("partial")
        raise RuntimeError("injected")

    monkeypatch.setattr("circuits.labeling.quality._write_quality_assessment", fail)
    with pytest.raises(RuntimeError, match="injected"):
        assess_width_one_quality(run_root=tmp_path)
    assert not (tmp_path / "assessments" / "label_quality_v2").exists()
    assert not list((tmp_path / "assessments").glob(".label_quality_v2.tmp-*"))
