"""Storage adapters for terminal stop-gradient selected attributions.

Projection math and source-token selection happen before this seam.  The
adapter controls only whether the compact terminal result retains its autograd
graph while later selected layers execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch

from circuits.tracing.instrumentation import TraceInstrumentation

StopGradientSelectedAttributionStorage = Literal[
    "graph_retaining_v1",
    "terminal_detached_v1",
]
DEFAULT_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE: StopGradientSelectedAttributionStorage = "graph_retaining_v1"


@dataclass(frozen=True)
class StoredStopGradientSelectedAttribution:
    """Compact projection plus observational graph-lifetime receipts."""

    tensor: torch.Tensor
    strategy: StopGradientSelectedAttributionStorage
    input_requires_grad: bool
    input_grad_fn_retained: bool
    stored_requires_grad: bool
    stored_grad_fn_retained: bool
    terminal_detached: bool
    shares_projection_storage: bool


def resolve_stop_gradient_selected_attribution_storage(
    strategy: str,
) -> StopGradientSelectedAttributionStorage:
    """Validate and return a provenance-bearing terminal storage strategy."""

    if strategy not in {"graph_retaining_v1", "terminal_detached_v1"}:
        raise ValueError(
            "invalid stop-gradient selected-attribution storage "
            f"{strategy!r}; expected one of "
            "['graph_retaining_v1', 'terminal_detached_v1']"
        )
    return cast(StopGradientSelectedAttributionStorage, strategy)


def store_stop_gradient_selected_attribution(
    projection: torch.Tensor,
    *,
    strategy: StopGradientSelectedAttributionStorage = (
        DEFAULT_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE
    ),
    layer: int,
    chunk_start: int,
    instrumentation: TraceInstrumentation | None = None,
) -> StoredStopGradientSelectedAttribution:
    """Apply only the requested lifetime policy to an already compact result."""

    resolved = resolve_stop_gradient_selected_attribution_storage(strategy)
    if projection.ndim != 3:
        raise ValueError(
            "stop-gradient selected-attribution projection must have shape "
            "(neuron, batch, source)"
        )
    input_requires_grad = projection.requires_grad
    input_grad_fn_retained = projection.grad_fn is not None
    stored = projection if resolved == "graph_retaining_v1" else projection.detach()
    result = StoredStopGradientSelectedAttribution(
        tensor=stored,
        strategy=resolved,
        input_requires_grad=input_requires_grad,
        input_grad_fn_retained=input_grad_fn_retained,
        stored_requires_grad=stored.requires_grad,
        stored_grad_fn_retained=stored.grad_fn is not None,
        terminal_detached=resolved == "terminal_detached_v1",
        shares_projection_storage=(
            stored.untyped_storage().data_ptr()
            == projection.untyped_storage().data_ptr()
        ),
    )
    if instrumentation is not None:
        instrumentation.increment_counter(
            "stop_gradient_selected_attribution_storage_execution_count"
        )
        instrumentation.increment_counter(
            f"stop_gradient_selected_attribution_{resolved}_storage_count"
        )
        instrumentation.increment_counter(
            "stop_gradient_selected_attribution_projection_graph_retained_count",
            int(result.input_grad_fn_retained),
        )
        instrumentation.increment_counter(
            "stop_gradient_selected_attribution_stored_graph_retained_count",
            int(result.stored_grad_fn_retained),
        )
        instrumentation.increment_counter(
            "stop_gradient_selected_attribution_terminal_detached_count",
            int(result.terminal_detached),
        )
        instrumentation.append_execution_record(
            "stop_gradient_selected_attribution_storage",
            layer=layer,
            chunk_start=chunk_start,
            strategy=result.strategy,
            input_requires_grad=result.input_requires_grad,
            input_grad_fn_retained=result.input_grad_fn_retained,
            stored_requires_grad=result.stored_requires_grad,
            stored_grad_fn_retained=result.stored_grad_fn_retained,
            terminal_detached=result.terminal_detached,
            shares_projection_storage=result.shares_projection_storage,
        )
    return result
