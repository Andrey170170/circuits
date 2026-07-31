"""Identity validation for local scorer artifacts consumed downstream."""

from __future__ import annotations

from typing import Any, Literal

from circuits.labeling.config import LabelingRecipe


def validate_local_score_artifact(
    value: dict[str, Any],
    *,
    recipe: LabelingRecipe,
    run_id: str,
    phase: Literal["candidate_selection", "summary_audit"],
    state: str,
    cluster_id: int,
    expected_request_ids: set[str],
) -> list[dict[str, Any]]:
    partition = "selection_scoring" if phase == "candidate_selection" else "audit"
    expected = {
        "schema_version": "adag.labeling.local-scores.v1",
        "run_id": run_id,
        "phase": phase,
        "state": state,
        "cluster_id": cluster_id,
        "partition": partition,
        "simulator": recipe.scorer.model_dump(mode="json"),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(
                f"local score {field} mismatch: "
                f"expected={expected_value!r}, actual={value.get(field)!r}"
            )
    scores = value.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError("local score artifact contains no scores")
    request_ids = [score.get("request_id") for score in scores]
    if any(not isinstance(request_id, str) for request_id in request_ids):
        raise ValueError("local score artifact contains an invalid request ID")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("local score artifact repeats request IDs")
    if set(request_ids) != expected_request_ids:
        raise ValueError(
            "local score request IDs mismatch: "
            f"expected={sorted(expected_request_ids)!r}, actual={sorted(request_ids)!r}"
        )
    return scores
