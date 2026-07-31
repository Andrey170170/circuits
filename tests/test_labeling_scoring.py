from __future__ import annotations

import math

from circuits.labeling.scoring import _is_insufficient_evidence, correlation_sort_key


def test_correlation_sort_places_nonfinite_values_after_finite_values() -> None:
    values = [
        {"request_id": "nan", "correlation": math.nan},
        {"request_id": "positive", "correlation": 0.2},
        {"request_id": "inf", "correlation": math.inf},
        {"request_id": "negative", "correlation": -0.1},
    ]
    values.sort(key=correlation_sort_key, reverse=True)
    assert [value["request_id"] for value in values[:2]] == ["positive", "negative"]


def test_insufficient_evidence_control_flow_detection() -> None:
    assert _is_insufficient_evidence(text="insufficient_evidence")
    assert _is_insufficient_evidence(text="some label", status="insufficient_evidence")
    assert not _is_insufficient_evidence(
        text="localized feature", status="provisional_label"
    )
