from __future__ import annotations

import math

from circuits.labeling.scoring import correlation_sort_key


def test_correlation_sort_places_nonfinite_values_after_finite_values() -> None:
    values = [
        {"request_id": "nan", "correlation": math.nan},
        {"request_id": "positive", "correlation": 0.2},
        {"request_id": "inf", "correlation": math.inf},
        {"request_id": "negative", "correlation": -0.1},
    ]
    values.sort(key=correlation_sort_key, reverse=True)
    assert [value["request_id"] for value in values[:2]] == ["positive", "negative"]
