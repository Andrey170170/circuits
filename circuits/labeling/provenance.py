"""Identity validation for local scorer artifacts consumed downstream."""

from __future__ import annotations

from typing import Any, Literal

from circuits.labeling.config import LabelingRecipe


def validate_local_score_artifact(
    value: dict[str, Any],
    *,
    recipe: LabelingRecipe,
    run_id: str,
    phase: Literal["candidate_selection", "summary_selection", "summary_audit"],
    state: str,
    cluster_id: int,
    expected_request_ids: set[str],
    expected_skipped_request_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    partition = (
        "selection_scoring"
        if phase in {"candidate_selection", "summary_selection"}
        else "audit"
    )
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
    if not isinstance(scores, list):
        raise ValueError("local score artifact scores must be a list")
    skipped = value.get("skipped", [])
    if not isinstance(skipped, list):
        raise ValueError("local score artifact skipped inputs must be a list")
    expected_skipped = expected_skipped_request_ids or set()
    if not scores and not skipped:
        raise ValueError("local score artifact contains no scores or skipped inputs")
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
    skipped_ids = [item.get("request_id") for item in skipped]
    if any(not isinstance(request_id, str) for request_id in skipped_ids):
        raise ValueError("local score artifact contains an invalid skipped request ID")
    if len(skipped_ids) != len(set(skipped_ids)):
        raise ValueError("local score artifact repeats skipped request IDs")
    if set(skipped_ids) != expected_skipped:
        raise ValueError(
            "local score skipped request IDs mismatch: "
            f"expected={sorted(expected_skipped)!r}, actual={sorted(skipped_ids)!r}"
        )
    if set(request_ids) & set(skipped_ids):
        raise ValueError("local score artifact both scores and skips a request ID")
    return scores


def validate_summary_score_binding(
    value: dict[str, Any],
    *,
    recipe: LabelingRecipe,
    run_id: str,
    phase: Literal["summary_selection", "summary_audit"],
    state: str,
    cluster_id: int,
    request_id: str,
    expected_text: str,
    source_result_path: str,
    source_result_sha256: str,
    skip_reason: str | None = None,
) -> dict[str, Any] | None:
    """Validate that a final-label score is bound to the exact summary result."""

    scores = validate_local_score_artifact(
        value,
        recipe=recipe,
        run_id=run_id,
        phase=phase,
        state=state,
        cluster_id=cluster_id,
        expected_request_ids=set() if skip_reason else {request_id},
        expected_skipped_request_ids={request_id} if skip_reason else set(),
    )
    entries = value.get("skipped", []) if skip_reason else scores
    entry = entries[0]
    expected = {
        "request_id": request_id,
        "text": expected_text,
        "source_result_path": source_result_path,
        "source_result_sha256": source_result_sha256,
    }
    if skip_reason:
        expected["reason"] = skip_reason
    for field, expected_value in expected.items():
        if entry.get(field) != expected_value:
            raise ValueError(
                f"summary score {field} mismatch: "
                f"expected={expected_value!r}, actual={entry.get(field)!r}"
            )
    return None if skip_reason else entry
