"""Execution strategies for per-layer stop-gradient contribution VJPs.

The public seam in this module owns four details that attribution callers
should not need to coordinate themselves: selecting the layer's real
``down_proj`` beneath any stop-gradient wrapper, installing and removing the
appropriate activation hook, deciding where the autograd graph begins, and
projecting a batched VJP into canonical selected-coordinate order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch

from circuits.tracing.instrumentation import (
    TraceInstrumentation,
    cuda_memory_instrumentation_stage,
)
from circuits.tracing.sparse_source_injection import sparse_source_injection

StopGradientContributionExecution = Literal[
    "full_graph_v1",
    "source_leaf_v1",
    "sparse_source_leaf_v1",
]
DEFAULT_STOP_GRADIENT_CONTRIBUTION_EXECUTION: StopGradientContributionExecution = (
    "full_graph_v1"
)


@dataclass(frozen=True)
class StopGradientContributionForward:
    """Forward result and VJP lifetime contract for one selected MLP layer."""

    logits: torch.Tensor
    source_activation: torch.Tensor
    retain_graph_for_vjp: bool
    execution: StopGradientContributionExecution
    source_representation: Literal["dense", "selected_coordinates"]
    dense_vjp_result_materialized: bool
    dense_source_shape: tuple[int, ...]
    selected_coordinates: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _ExecutionAdapter:
    source_kind: Literal["full_graph", "dense_leaf", "sparse_leaf"]

    @property
    def differentiable_embeddings(self) -> bool:
        return self.source_kind == "full_graph"

    @property
    def source_is_leaf(self) -> bool:
        return self.source_kind != "full_graph"

    @property
    def uses_sparse_injection(self) -> bool:
        return self.source_kind == "sparse_leaf"

    @property
    def source_representation(self) -> Literal["dense", "selected_coordinates"]:
        return "selected_coordinates" if self.uses_sparse_injection else "dense"

    @property
    def dense_vjp_result_materialized(self) -> bool:
        return not self.uses_sparse_injection

    @property
    def retain_graph_for_vjp(self) -> bool:
        return self.source_kind == "full_graph"


_EXECUTION_ADAPTERS: dict[StopGradientContributionExecution, _ExecutionAdapter] = {
    "full_graph_v1": _ExecutionAdapter(
        source_kind="full_graph",
    ),
    "source_leaf_v1": _ExecutionAdapter(
        source_kind="dense_leaf",
    ),
    "sparse_source_leaf_v1": _ExecutionAdapter(
        source_kind="sparse_leaf",
    ),
}


def resolve_stop_gradient_contribution_execution(
    execution: StopGradientContributionExecution | str,
) -> StopGradientContributionExecution:
    """Validate and normalize a provenance-bearing execution strategy."""

    if execution not in _EXECUTION_ADAPTERS:
        choices = ", ".join(sorted(_EXECUTION_ADAPTERS))
        raise ValueError(
            "invalid stop-gradient contribution execution "
            f"{execution!r}; expected one of: {choices}"
        )
    return cast(StopGradientContributionExecution, execution)


def _selected_down_projection(model, layer: int):
    mlp = model.model.layers[layer].mlp
    return mlp.mlp.down_proj if hasattr(mlp, "mlp") else mlp.down_proj


def run_stop_gradient_contribution_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask,
    *,
    layer: int,
    execution: StopGradientContributionExecution,
    selected_coordinates: Sequence[Sequence[int]] = (),
    instrumentation: TraceInstrumentation | None = None,
) -> StopGradientContributionForward:
    """Run one contribution forward and expose only its selected source.

    ``full_graph_v1`` reproduces the historical differentiable-embedding
    execution. ``source_leaf_v1`` computes the prefix without an autograd
    source, then replaces the selected ``down_proj`` input with an equal-valued
    detached leaf. The latter keeps the scientific source tensor unchanged
    while retaining only the downstream graph needed by the sole VJP.

    ``sparse_source_leaf_v1`` also computes a graph-free prefix, but begins the
    graph at only the ordered selected coordinates through an equal-valued zero
    correction at the linear projection. Duplicate coordinates remain distinct
    leaf entries.

    The temporary hook is always removed, including when the model forward
    raises. Source-leaf execution temporarily freezes every model parameter so
    no upstream graph can begin at a trainable weight, then restores every
    original ``requires_grad`` flag before returning.
    """

    execution = resolve_stop_gradient_contribution_execution(execution)
    adapter = _EXECUTION_ADAPTERS[execution]
    down_projection = _selected_down_projection(model, layer)
    source: torch.Tensor | None = None
    dense_source_shape: tuple[int, ...] | None = None
    normalized_coordinates = tuple(
        (int(position), int(neuron)) for position, neuron in selected_coordinates
    )
    parameter_states = [
        (parameter, bool(parameter.requires_grad)) for parameter in model.parameters()
    ]
    if adapter.source_is_leaf:
        for parameter, _requires_grad in parameter_states:
            parameter.requires_grad_(False)

    try:
        embeddings = model.model.embed_tokens(input_ids).detach()
        if adapter.differentiable_embeddings:
            embeddings.requires_grad_(True)

        if adapter.uses_sparse_injection:
            with sparse_source_injection(
                down_projection, selected_coordinates
            ) as injection:
                output = model(inputs_embeds=embeddings, attention_mask=attention_mask)
                capture = injection.capture()
            source = capture.source_leaf
            dense_source_shape = capture.dense_source_shape
            normalized_coordinates = capture.coordinates
        elif adapter.source_is_leaf:

            def capture_source(_module, args):
                nonlocal source
                if not args:
                    raise RuntimeError(
                        "down_proj forward did not receive a positional input"
                    )
                if args[0].requires_grad or args[0].grad_fn is not None:
                    raise RuntimeError(
                        "source-leaf prefix unexpectedly retained an autograd graph"
                    )
                source = args[0].detach().requires_grad_(True)
                return (source, *args[1:])

            handle = down_projection.register_forward_pre_hook(capture_source)
        else:

            def capture_source(_module, args, _output) -> None:
                nonlocal source
                if not args:
                    raise RuntimeError(
                        "down_proj forward did not receive a positional input"
                    )
                source = args[0]

            handle = down_projection.register_forward_hook(capture_source)

        if not adapter.uses_sparse_injection:
            try:
                output = model(inputs_embeds=embeddings, attention_mask=attention_mask)
            finally:
                handle.remove()
    finally:
        if adapter.source_is_leaf:
            for parameter, requires_grad in parameter_states:
                parameter.requires_grad_(requires_grad)

    if source is None:
        raise RuntimeError(f"selected layer {layer} down_proj did not execute")
    if adapter.source_is_leaf and (
        not source.is_leaf or not source.requires_grad or source.grad_fn is not None
    ):
        raise RuntimeError("source-leaf execution did not produce an autograd leaf")
    if dense_source_shape is None:
        dense_source_shape = tuple(source.shape)

    if instrumentation is not None:
        dense_source_numel = 1
        for extent in dense_source_shape:
            dense_source_numel *= extent
        instrumentation.record_layer(
            layer,
            stop_gradient_contribution_execution=execution,
            stop_gradient_contribution_source_representation=(
                adapter.source_representation
            ),
            stop_gradient_contribution_dense_source_shape=list(dense_source_shape),
            stop_gradient_contribution_differentiated_source_shape=list(source.shape),
            stop_gradient_contribution_selected_coordinate_count=len(
                normalized_coordinates
            ),
            stop_gradient_contribution_dense_source_numel=dense_source_numel,
            stop_gradient_contribution_differentiated_source_numel=source.numel(),
        )
        if adapter.uses_sparse_injection:
            instrumentation.increment_counter(
                "stop_gradient_sparse_source_forward_count"
            )
            instrumentation.increment_counter(
                "stop_gradient_sparse_source_coordinate_count",
                len(normalized_coordinates),
            )
            instrumentation.increment_counter(
                "stop_gradient_sparse_source_dense_numel_avoided",
                dense_source_numel - source.numel(),
            )

    return StopGradientContributionForward(
        logits=output.logits,
        source_activation=source,
        retain_graph_for_vjp=adapter.retain_graph_for_vjp,
        execution=execution,
        source_representation=adapter.source_representation,
        dense_vjp_result_materialized=adapter.dense_vjp_result_materialized,
        dense_source_shape=dense_source_shape,
        selected_coordinates=normalized_coordinates,
    )


def _copy_dense_selected_vjp(
    dense_vjp: torch.Tensor,
    coordinates: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    positions = torch.tensor(
        [position for position, _neuron in coordinates],
        device=dense_vjp.device,
        dtype=torch.long,
    )
    neurons = torch.tensor(
        [neuron for _position, neuron in coordinates],
        device=dense_vjp.device,
        dtype=torch.long,
    )
    return dense_vjp[:, :, positions, neurons].permute(2, 1, 0)


def run_stop_gradient_contribution_vjp(
    contribution_forward: StopGradientContributionForward,
    target_values: torch.Tensor,
    *,
    layer: int,
    instrumentation: TraceInstrumentation | None = None,
) -> torch.Tensor:
    """Return selected gradients in canonical ``(coordinate, batch, target)`` order."""

    if target_values.ndim != 2:
        raise ValueError("contribution targets must have shape (target, batch)")
    target_count, batch = target_values.shape
    source = contribution_forward.source_activation
    if source.shape[0] != batch:
        raise ValueError("contribution source and target batch dimensions differ")
    coordinates = contribution_forward.selected_coordinates
    if not coordinates:
        raise ValueError("contribution VJP requires selected coordinates")

    grad_outputs = torch.eye(target_count * batch, device=source.device)
    sparse = not contribution_forward.dense_vjp_result_materialized
    dense_vjp_shape = (
        target_count * batch,
        *contribution_forward.dense_source_shape,
    )
    metadata = {
        "operation_kind": "batched_vjp",
        "layer": layer,
        "source_representation": contribution_forward.source_representation,
        "selected_neuron_count": len(coordinates),
        "lane_count": target_count * batch,
        "differentiated_output_shape": list(target_values.shape),
        "dense_differentiated_input_shape": list(
            contribution_forward.dense_source_shape
        ),
        "differentiated_input_shape": list(source.shape),
        "grad_outputs_shape": list(grad_outputs.shape),
        "dense_vjp_result_shape": list(dense_vjp_shape),
        "dense_vjp_result_materialized": (
            contribution_forward.dense_vjp_result_materialized
        ),
    }
    with cuda_memory_instrumentation_stage(
        instrumentation,
        "stop_grad_neuron_contribution_vjp",
        metadata=metadata,
    ) as vjp_measurement:
        raw_vjp = torch.autograd.grad(
            target_values.flatten(),
            source,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=contribution_forward.retain_graph_for_vjp,
        )[0]
        if sparse:
            # (target * batch, batch, coordinate) -> (coordinate, batch, target)
            projected = (
                raw_vjp.reshape(target_count, batch, batch, len(coordinates))
                .diagonal(dim1=1, dim2=2)
                .permute(1, 2, 0)
                .contiguous()
            )
        else:
            dense_vjp = (
                raw_vjp.reshape(
                    target_count,
                    batch,
                    batch,
                    raw_vjp.shape[-2],
                    raw_vjp.shape[-1],
                )
                .diagonal(dim1=1, dim2=2)
                .permute(0, 3, 1, 2)
            )
            projected = _copy_dense_selected_vjp(dense_vjp, coordinates)
        if vjp_measurement is not None:
            # Preserve the historical generic key while adding explicit raw
            # and projected shapes for sparse-source qualification.
            vjp_measurement.metadata["vjp_result_shape"] = list(raw_vjp.shape)
            vjp_measurement.metadata["raw_vjp_result_shape"] = list(raw_vjp.shape)
            vjp_measurement.metadata["projected_vjp_result_shape"] = list(
                projected.shape
            )

    if instrumentation is not None:
        instrumentation.record_layer(
            layer,
            stop_gradient_contribution_raw_vjp_shape=list(raw_vjp.shape),
            stop_gradient_contribution_projected_vjp_shape=list(projected.shape),
            stop_gradient_contribution_dense_vjp_result_materialized=(
                contribution_forward.dense_vjp_result_materialized
            ),
        )
        if sparse:
            dense_numel = 1
            for extent in dense_vjp_shape:
                dense_numel *= extent
            instrumentation.increment_counter("stop_gradient_sparse_vjp_count")
            instrumentation.increment_counter(
                "stop_gradient_sparse_dense_vjp_result_numel_avoided",
                dense_numel - raw_vjp.numel(),
            )
            instrumentation.increment_counter(
                "stop_gradient_sparse_vjp_result_numel", raw_vjp.numel()
            )
    return projected


__all__ = [
    "DEFAULT_STOP_GRADIENT_CONTRIBUTION_EXECUTION",
    "StopGradientContributionExecution",
    "StopGradientContributionForward",
    "resolve_stop_gradient_contribution_execution",
    "run_stop_gradient_contribution_forward",
    "run_stop_gradient_contribution_vjp",
]
