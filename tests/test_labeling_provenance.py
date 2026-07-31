from __future__ import annotations

import pytest

from circuits.labeling.config import LabelingRecipe
from circuits.labeling.provenance import (
    validate_local_score_artifact,
    validate_summary_score_binding,
)


def _recipe() -> LabelingRecipe:
    return LabelingRecipe.model_validate(
        {
            "recipe_id": "test-width-one-v2",
            "description": "test",
            "prompt_policy": "width_one_v2",
            "candidate_generator": {
                "provider": "fake",
                "model": "fake",
                "max_output_tokens": 100,
            },
            "cluster_summarizer": {
                "provider": "fake",
                "model": "fake",
                "max_output_tokens": 1200,
            },
            "price_snapshot": "prices.json",
        }
    )


def _score() -> dict:
    recipe = _recipe()
    return {
        "schema_version": "adag.labeling.local-scores.v1",
        "run_id": "run-1",
        "phase": "candidate_selection",
        "state": "primary",
        "cluster_id": 4,
        "partition": "selection_scoring",
        "simulator": recipe.scorer.model_dump(mode="json"),
        "scores": [{"request_id": "req-1", "correlation": 0.2}],
    }


def test_score_provenance_validates_all_identities() -> None:
    scores = validate_local_score_artifact(
        _score(),
        recipe=_recipe(),
        run_id="run-1",
        phase="candidate_selection",
        state="primary",
        cluster_id=4,
        expected_request_ids={"req-1"},
    )
    assert scores[0]["request_id"] == "req-1"


def test_score_provenance_rejects_request_id_drift() -> None:
    with pytest.raises(ValueError, match="request IDs mismatch"):
        validate_local_score_artifact(
            _score(),
            recipe=_recipe(),
            run_id="run-1",
            phase="candidate_selection",
            state="primary",
            cluster_id=4,
            expected_request_ids={"req-other"},
        )


def _summary_score(*, phase: str = "summary_selection") -> dict:
    recipe = _recipe()
    return {
        "schema_version": "adag.labeling.local-scores.v1",
        "run_id": "run-1",
        "phase": phase,
        "state": "primary",
        "cluster_id": 4,
        "partition": "selection_scoring" if phase == "summary_selection" else "audit",
        "simulator": recipe.scorer.model_dump(mode="json"),
        "scores": [
            {
                "request_id": "summary-1",
                "text": "rewritten final label",
                "source_result_path": "results/cluster_summary/summary-1.json",
                "source_result_sha256": "abc123",
                "correlation": 0.2,
            }
        ],
        "skipped": [],
    }


def test_summary_score_is_bound_to_exact_rewritten_label_and_result_hash() -> None:
    score = validate_summary_score_binding(
        _summary_score(),
        recipe=_recipe(),
        run_id="run-1",
        phase="summary_selection",
        state="primary",
        cluster_id=4,
        request_id="summary-1",
        expected_text="rewritten final label",
        source_result_path="results/cluster_summary/summary-1.json",
        source_result_sha256="abc123",
    )
    assert score is not None
    assert score["correlation"] == 0.2


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("text", "original candidate label", "text mismatch"),
        ("source_result_sha256", "rewritten-result-hash", "sha256 mismatch"),
    ],
)
def test_summary_score_rejects_rewritten_or_replaced_result(
    field: str, value: str, match: str
) -> None:
    artifact = _summary_score()
    artifact["scores"][0][field] = value
    with pytest.raises(ValueError, match=match):
        validate_summary_score_binding(
            artifact,
            recipe=_recipe(),
            run_id="run-1",
            phase="summary_selection",
            state="primary",
            cluster_id=4,
            request_id="summary-1",
            expected_text="rewritten final label",
            source_result_path="results/cluster_summary/summary-1.json",
            source_result_sha256="abc123",
        )


def test_insufficient_evidence_is_validated_as_unscored_control_flow() -> None:
    artifact = _summary_score()
    artifact["scores"] = []
    artifact["skipped"] = [
        {
            "request_id": "summary-1",
            "text": "insufficient_evidence",
            "source_result_path": "results/cluster_summary/summary-1.json",
            "source_result_sha256": "abc123",
            "reason": "model_reported_insufficient_evidence",
        }
    ]
    assert (
        validate_summary_score_binding(
            artifact,
            recipe=_recipe(),
            run_id="run-1",
            phase="summary_selection",
            state="primary",
            cluster_id=4,
            request_id="summary-1",
            expected_text="insufficient_evidence",
            source_result_path="results/cluster_summary/summary-1.json",
            source_result_sha256="abc123",
            skip_reason="model_reported_insufficient_evidence",
        )
        is None
    )
