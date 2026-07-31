from __future__ import annotations

import pytest

from circuits.labeling.config import LabelingRecipe
from circuits.labeling.provenance import validate_local_score_artifact


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
