"""Focused tests for the important-neuron selection zero guard."""

from __future__ import annotations

import pytest
import torch
from circuits.tracing.attribution import _get_global_important_neurons_mask
from circuits.tracing.important_neuron_selection import (
    has_any_strictly_positive_value,
)
from torch.profiler import ProfilerActivity, profile


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], False),
        ([0.0, -0.0], False),
        ([float("nan"), 0.0], False),
        ([-1.0, 0.0], False),
        ([float("inf")], True),
        ([float("nan"), 0.25], True),
    ],
)
def test_positive_guard_preserves_strict_comparison_semantics(
    values: list[float],
    expected: bool,
) -> None:
    assert has_any_strictly_positive_value(torch.tensor(values)) is expected


def test_tracing_positive_guard_does_not_compact_positive_values() -> None:
    values = torch.arange(16, dtype=torch.float32)

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        assert has_any_strictly_positive_value(values)

    operator_names = {event.key for event in profiler.key_averages()}
    assert "aten::any" in operator_names
    assert "aten::index" not in operator_names
    assert "aten::nonzero" not in operator_names


def test_all_zero_selection_keeps_existing_mask_shape_and_dtype() -> None:
    attributions = torch.zeros((2, 1, 3, 4, 1), dtype=torch.bfloat16)

    mask = _get_global_important_neurons_mask(
        keep_tokens=[0, 1, 2],
        start_layer=0,
        end_layer=2,
        mlp_final_attributions=attributions,
        topk_neurons=1,
        node_attribution_threshold=None,
    )

    assert mask.shape == (2, 3, 4)
    assert mask.dtype == torch.bfloat16
    assert torch.count_nonzero(mask).item() == 0


def test_nan_only_selection_keeps_existing_all_zero_behavior() -> None:
    attributions = torch.full(
        (2, 1, 3, 4, 1),
        float("nan"),
        dtype=torch.float32,
    )

    mask = _get_global_important_neurons_mask(
        keep_tokens=[0, 1, 2],
        start_layer=0,
        end_layer=2,
        mlp_final_attributions=attributions,
        topk_neurons=1,
        node_attribution_threshold=None,
    )

    assert torch.equal(mask, torch.zeros_like(mask))
