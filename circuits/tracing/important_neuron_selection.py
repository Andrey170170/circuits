"""Memory-conscious primitives for selecting important neurons."""

from __future__ import annotations

import torch


def has_any_strictly_positive_value(values: torch.Tensor) -> bool:
    """Return whether ``values`` contains a value greater than zero.

    Keep the comparison explicit so NaNs retain the existing selection semantics:
    they do not count as positive. Reducing the boolean comparison avoids boolean
    indexing, whose compacted output and index workspace scale with every positive
    attribution even though the caller only needs this scalar predicate.
    """

    return bool(torch.any(values > 0).item())
