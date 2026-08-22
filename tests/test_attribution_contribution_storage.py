"""Focused tests for compact selected-neuron contribution storage."""

from __future__ import annotations

import pytest
import torch
from circuits.tracing.attribution import (
    _copy_selected_neuron_contributions,
    _nonempty_neuron_layers,
)


def test_selected_contribution_copy_preserves_pair_order_shape_and_values() -> None:
    dense = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    pairs = [[3, 4], [0, 1], [2, 0], [0, 3]]
    expected = torch.stack(
        [dense[:, :, position, neuron] for position, neuron in pairs]
    ).permute(0, 2, 1)

    actual = _copy_selected_neuron_contributions(dense, pairs)

    assert actual.shape == (len(pairs), 3, 2)
    assert torch.equal(actual, expected)


def test_selected_contribution_copy_does_not_retain_dense_storage() -> None:
    dense = torch.arange(2 * 2 * 8 * 16, dtype=torch.float32).reshape(2, 2, 8, 16)
    selected = _copy_selected_neuron_contributions(dense, [[7, 15], [0, 0]])
    before = selected.clone()

    assert selected.untyped_storage().data_ptr() != dense.untyped_storage().data_ptr()
    assert (
        selected.untyped_storage().nbytes()
        == selected.numel() * selected.element_size()
    )

    dense.fill_(-1)
    assert torch.equal(selected, before)


def test_nonempty_neuron_layers_skips_empty_layers_and_preserves_order() -> None:
    config = {
        9: [],
        4: [[3, 7], [1, 2]],
        12: [],
        2: [[0, 5]],
    }

    assert list(_nonempty_neuron_layers(config)) == [
        (4, [[3, 7], [1, 2]]),
        (2, [[0, 5]]),
    ]


def test_selected_contribution_copy_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="empty neuron selection"):
        _copy_selected_neuron_contributions(torch.zeros(1, 1, 1, 1), [])
