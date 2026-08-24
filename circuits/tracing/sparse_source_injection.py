"""Equal-valued sparse differentiation at a linear projection.

The module hides the hook lifecycle and projection algebra needed to begin an
autograd graph at an ordered collection of ``(token, feature)`` coordinates.
It is intentionally independent of contribution and cross-layer scheduling so
the same seam can be reused by later bounded-memory execution adapters.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SparseSourceCapture:
    """One completed sparse injection and its compact differentiation leaf."""

    source_leaf: torch.Tensor
    selected_values: torch.Tensor
    dense_source_shape: tuple[int, ...]
    coordinates: tuple[tuple[int, int], ...]


def _validated_coordinates(
    coordinates: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    if not coordinates:
        raise ValueError("sparse source injection requires selected coordinates")
    normalized: list[tuple[int, int]] = []
    for coordinate in coordinates:
        if len(coordinate) != 2:
            raise ValueError(
                "each sparse source coordinate must contain token and feature"
            )
        token, feature = coordinate
        if isinstance(token, bool) or isinstance(feature, bool):
            raise TypeError("sparse source coordinates must be integer indices")
        if not isinstance(token, int) or not isinstance(feature, int):
            raise TypeError("sparse source coordinates must be integer indices")
        if token < 0 or feature < 0:
            raise ValueError("sparse source coordinates must be non-negative")
        normalized.append((token, feature))
    return tuple(normalized)


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"sparse source {name} must contain only finite values")


class SparseSourceInjection:
    """Install one sparse source leaf at a rank-3 linear input.

    Coordinates retain caller order and duplicates. A duplicate therefore
    creates a distinct leaf coordinate with the same selected value and
    derivative. The original linear output (including bias and any earlier
    hooks) is detached, then an exactly zero selected-column correction begins
    the downstream graph.
    """

    def __init__(
        self,
        projection: nn.Module,
        coordinates: Sequence[Sequence[int]],
    ) -> None:
        weight = getattr(projection, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError("sparse source projection must have a rank-2 weight")
        self._projection = projection
        self._coordinates = _validated_coordinates(coordinates)
        self._capture: SparseSourceCapture | None = None
        self._execution_count = 0

    def _hook(
        self,
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        self._execution_count += 1
        if self._execution_count != 1:
            raise RuntimeError("sparse source projection executed more than once")
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("sparse source projection received no tensor input")
        source = inputs[0]
        if source.ndim != 3:
            raise ValueError(
                "sparse source input must have shape (batch, token, feature)"
            )
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError(
                "sparse source output must have shape (batch, token, feature)"
            )
        weight = self._projection.weight
        if weight.ndim != 2:
            raise ValueError("sparse source projection must have a rank-2 weight")
        if source.shape[-1] != weight.shape[1]:
            raise ValueError(
                "sparse source input width does not match projection weight"
            )
        if output.shape[:2] != source.shape[:2] or output.shape[-1] != weight.shape[0]:
            raise ValueError("sparse source output shape does not match projection")

        tokens = tuple(token for token, _feature in self._coordinates)
        features = tuple(feature for _token, feature in self._coordinates)
        if max(tokens) >= source.shape[1] or max(features) >= source.shape[2]:
            raise IndexError("sparse source coordinate is out of bounds")

        token_indices = torch.tensor(tokens, device=source.device, dtype=torch.long)
        feature_indices = torch.tensor(features, device=source.device, dtype=torch.long)
        selected_values = source[:, token_indices, feature_indices]
        _require_finite("selected values", selected_values)
        source_leaf = selected_values.detach().requires_grad_(True)

        selected_columns = (
            weight.detach().index_select(1, feature_indices).transpose(0, 1)
        )
        _require_finite("selected weight columns", selected_columns)
        correction = (source_leaf - selected_values.detach()).unsqueeze(-1)
        correction = correction * selected_columns.unsqueeze(0)
        corrected_output = output.detach().index_add(1, token_indices, correction)
        self._capture = SparseSourceCapture(
            source_leaf=source_leaf,
            selected_values=selected_values.detach(),
            dense_source_shape=tuple(source.shape),
            coordinates=self._coordinates,
        )
        return corrected_output

    def capture(self) -> SparseSourceCapture:
        if self._execution_count != 1 or self._capture is None:
            raise RuntimeError("sparse source projection did not execute exactly once")
        return self._capture


@contextmanager
def sparse_source_injection(
    projection: nn.Module,
    coordinates: Sequence[Sequence[int]],
) -> Iterator[SparseSourceInjection]:
    """Install one injection hook and remove only that hook on every exit."""

    injection = SparseSourceInjection(projection, coordinates)
    # Run before existing output hooks so they remain downstream of both the
    # detached base output and sparse correction, preserving their derivatives.
    handle = projection.register_forward_hook(injection._hook, prepend=True)
    try:
        yield injection
    finally:
        handle.remove()


__all__ = [
    "SparseSourceCapture",
    "SparseSourceInjection",
    "sparse_source_injection",
]
