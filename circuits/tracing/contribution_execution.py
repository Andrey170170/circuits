"""Execution strategies for selected-neuron contribution VJPs.

The interfaces here own the autograd details attribution callers should not
coordinate themselves: source selection and graph lifetime, target-lane
chunking, and projection into canonical selected-coordinate order. The
stop-gradient interface additionally owns temporary activation hooks and its
choice of dense or sparse differentiated source.
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
from circuits.tracing.tensor_receipts import raw_tensor_sha256

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
class SelectedNeuronContributionSource:
    """One ordinary contribution source in stable selected-layer order."""

    layer: int
    source_activation: torch.Tensor
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


def resolve_stop_gradient_contribution_target_lane_chunk_size(
    chunk_size: int | None,
) -> int | None:
    """Validate a provenance-bearing target-axis VJP chunk width."""

    if chunk_size is None:
        return None
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError(
            "stop-gradient contribution target-lane chunk size must be a "
            "positive integer or None"
        )
    return chunk_size


def resolve_selected_neuron_contribution_target_lane_chunk_size(
    chunk_size: int | None,
) -> int | None:
    """Validate the ordinary selected-neuron target-axis VJP chunk width."""

    if chunk_size is None:
        return None
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError(
            "selected-neuron contribution target-lane chunk size must be a "
            "positive integer or None"
        )
    return chunk_size


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


def run_selected_neuron_contribution_vjps(
    sources: Sequence[SelectedNeuronContributionSource],
    target_values: torch.Tensor,
    *,
    target_lane_chunk_size: int | None = None,
    full_grad_outputs: torch.Tensor | None = None,
    instrumentation: TraceInstrumentation | None = None,
    execution_index: int | None = None,
) -> list[torch.Tensor]:
    """Return compact ordinary VJPs in selected-layer order.

    Results use canonical ``(coordinate, batch, target)`` order. Target chunks
    are contiguous and retain every batch lane, while each dense raw VJP is
    projected into selected-coordinate storage before the next traversal. The
    shared forward graph is retained for every traversal because ordinary
    contribution sources all belong to the same graph and later tracing work
    may still consume it.

    ``full_grad_outputs`` lets the compatibility path reuse the identity matrix
    already materialized for the embedding VJP. It is accepted only when the
    resolved execution has a single target chunk. ``execution_index`` records
    repeated IG steps as an ordered receipt series instead of overwriting a
    singular per-layer receipt.
    """

    if target_values.ndim != 2:
        raise ValueError("selected-neuron targets must have shape (target, batch)")
    target_count, batch = target_values.shape
    if target_count <= 0 or batch <= 0:
        raise ValueError(
            "selected-neuron targets must have non-empty target and batch axes"
        )
    if not sources:
        raise ValueError("selected-neuron contribution VJP requires a source")

    requested_chunk_size = resolve_selected_neuron_contribution_target_lane_chunk_size(
        target_lane_chunk_size
    )
    resolved_chunk_size = min(requested_chunk_size or target_count, target_count)
    target_chunks = tuple(
        (start, min(start + resolved_chunk_size, target_count))
        for start in range(0, target_count, resolved_chunk_size)
    )
    if full_grad_outputs is not None:
        expected_shape = (target_count * batch, target_count * batch)
        if len(target_chunks) != 1:
            raise ValueError(
                "full grad outputs can only be reused by an unchunked "
                "selected-neuron contribution VJP"
            )
        if tuple(full_grad_outputs.shape) != expected_shape:
            raise ValueError(
                "full grad outputs shape does not match selected-neuron targets"
            )
        if full_grad_outputs.device != target_values.device:
            raise ValueError(
                "full grad outputs and selected-neuron targets must share a device"
            )

    # Reuse one identity per distinct chunk width across every selected layer.
    # This preserves the historical single-eye compatibility path and avoids
    # trading bounded raw VJPs for repeated identity allocations.
    grad_outputs_by_lane_count: dict[int, torch.Tensor] = {}
    if full_grad_outputs is not None:
        grad_outputs_by_lane_count[target_count * batch] = full_grad_outputs

    results: list[torch.Tensor] = []
    for source_spec in sources:
        source = source_spec.source_activation
        coordinates = source_spec.selected_coordinates
        if source.ndim != 3:
            raise ValueError(
                "selected-neuron source must have shape (batch, sequence, neuron)"
            )
        if source.shape[0] != batch:
            raise ValueError(
                "selected-neuron source and target batch dimensions differ"
            )
        if not coordinates:
            raise ValueError("selected-neuron contribution VJP requires coordinates")

        source.grad = None
        dense_vjp_shape = (target_count * batch, *source.shape)
        metadata = {
            "operation_kind": "batched_vjp",
            "layer": source_spec.layer,
            "source_representation": "dense",
            "selected_neuron_count": len(coordinates),
            "lane_count": target_count * batch,
            "target_lane_count": target_count,
            "target_lane_chunk_size_requested": requested_chunk_size,
            "target_lane_chunk_size_resolved": resolved_chunk_size,
            "target_lane_chunk_count": len(target_chunks),
            "max_materialized_target_lanes": resolved_chunk_size,
            "max_materialized_autograd_lanes": resolved_chunk_size * batch,
            "differentiated_output_shape": list(target_values.shape),
            "differentiated_input_shape": list(source.shape),
            "grad_outputs_shape": (
                [target_count * batch] * 2 if len(target_chunks) == 1 else None
            ),
            "max_grad_outputs_shape": [resolved_chunk_size * batch] * 2,
            "dense_vjp_result_shape": list(dense_vjp_shape),
            "dense_vjp_result_materialized": True,
            "retain_graph": True,
            "execution_index": execution_index,
        }
        with cuda_memory_instrumentation_stage(
            instrumentation,
            "selected_neuron_contribution_vjp",
            metadata=metadata,
        ) as vjp_measurement:
            projected_chunks: list[torch.Tensor] = []
            raw_vjp_chunk_shapes: list[list[int]] = []
            grad_outputs_chunk_shapes: list[list[int]] = []
            for target_start, target_end in target_chunks:
                chunk_target_count = target_end - target_start
                chunk_autograd_lanes = chunk_target_count * batch
                grad_outputs = grad_outputs_by_lane_count.get(chunk_autograd_lanes)
                if grad_outputs is None:
                    grad_outputs = torch.eye(
                        chunk_autograd_lanes, device=target_values.device
                    )
                    grad_outputs_by_lane_count[chunk_autograd_lanes] = grad_outputs
                target_chunk = target_values[target_start:target_end]
                raw_vjp = torch.autograd.grad(
                    target_chunk.flatten(),
                    source,
                    grad_outputs=grad_outputs,
                    is_grads_batched=True,
                    retain_graph=True,
                )[0]
                raw_vjp_chunk_shapes.append(list(raw_vjp.shape))
                grad_outputs_chunk_shapes.append(list(grad_outputs.shape))
                dense_vjp = (
                    raw_vjp.reshape(
                        chunk_target_count,
                        batch,
                        batch,
                        raw_vjp.shape[-2],
                        raw_vjp.shape[-1],
                    )
                    .diagonal(dim1=1, dim2=2)
                    .permute(0, 3, 1, 2)
                )
                projected_chunks.append(
                    _copy_dense_selected_vjp(dense_vjp, coordinates)
                )
                del dense_vjp, raw_vjp, target_chunk

            projected = (
                projected_chunks[0]
                if len(projected_chunks) == 1
                else torch.cat(projected_chunks, dim=2)
            )
            if vjp_measurement is not None:
                single_raw_shape = (
                    raw_vjp_chunk_shapes[0] if len(raw_vjp_chunk_shapes) == 1 else None
                )
                vjp_measurement.metadata["vjp_result_shape"] = single_raw_shape
                vjp_measurement.metadata["raw_vjp_result_shape"] = single_raw_shape
                vjp_measurement.metadata["raw_vjp_chunk_shapes"] = raw_vjp_chunk_shapes
                vjp_measurement.metadata["grad_outputs_chunk_shapes"] = (
                    grad_outputs_chunk_shapes
                )
                vjp_measurement.metadata["projected_vjp_result_shape"] = list(
                    projected.shape
                )

        if instrumentation is not None:
            single_raw_shape = (
                raw_vjp_chunk_shapes[0] if len(raw_vjp_chunk_shapes) == 1 else None
            )
            projected_receipt = raw_tensor_sha256(projected)
            layer_values = {
                "selected_neuron_contribution_raw_vjp_shape": single_raw_shape,
                "selected_neuron_contribution_raw_vjp_chunk_shapes": (
                    raw_vjp_chunk_shapes
                ),
                "selected_neuron_contribution_grad_outputs_shape": (
                    [target_count * batch] * 2 if len(target_chunks) == 1 else None
                ),
                "selected_neuron_contribution_grad_outputs_chunk_shapes": (
                    grad_outputs_chunk_shapes
                ),
                "selected_neuron_contribution_max_grad_outputs_shape": (
                    [resolved_chunk_size * batch] * 2
                ),
                "selected_neuron_contribution_projected_vjp_shape": list(
                    projected.shape
                ),
                "selected_neuron_contribution_target_lane_count": target_count,
                "selected_neuron_contribution_target_lane_chunk_size_requested": (
                    requested_chunk_size
                ),
                "selected_neuron_contribution_target_lane_chunk_size_resolved": (
                    resolved_chunk_size
                ),
                "selected_neuron_contribution_target_lane_chunk_count": len(
                    target_chunks
                ),
                "selected_neuron_contribution_max_materialized_target_lanes": (
                    resolved_chunk_size
                ),
                "selected_neuron_contribution_max_materialized_autograd_lanes": (
                    resolved_chunk_size * batch
                ),
                "selected_neuron_contribution_dense_vjp_result_materialized": True,
                "selected_neuron_contribution_retain_graph": True,
            }
            if execution_index is None:
                instrumentation.record_layer(
                    source_spec.layer,
                    **layer_values,
                    selected_neuron_contribution_receipt_mode="singular",
                    selected_neuron_contribution_projected_vjp_sha256=(
                        projected_receipt
                    ),
                )
            else:
                instrumentation.record_layer(
                    source_spec.layer,
                    **layer_values,
                    selected_neuron_contribution_receipt_mode="execution_indexed",
                )
                instrumentation.append_layer_record(
                    source_spec.layer,
                    "selected_neuron_contribution_execution_receipts",
                    execution_index=execution_index,
                    projected_vjp_shape=list(projected.shape),
                    projected_vjp_sha256=projected_receipt,
                    target_lane_count=target_count,
                )
            instrumentation.increment_counter(
                "selected_neuron_contribution_vjp_chunk_executions",
                len(target_chunks),
            )
            instrumentation.set_counter(
                "selected_neuron_contribution_max_materialized_target_lanes",
                resolved_chunk_size,
            )
            instrumentation.set_counter(
                "selected_neuron_contribution_max_materialized_autograd_lanes",
                resolved_chunk_size * batch,
            )
        results.append(projected)

    return results


def run_stop_gradient_contribution_vjp(
    contribution_forward: StopGradientContributionForward,
    target_values: torch.Tensor,
    *,
    layer: int,
    target_lane_chunk_size: int | None = None,
    instrumentation: TraceInstrumentation | None = None,
) -> torch.Tensor:
    """Return selected gradients in canonical ``(coordinate, batch, target)`` order.

    Chunking slices the contiguous target axis and always keeps every batch
    lane for those targets in the same batched VJP. Each raw chunk is projected
    before the next backward traversal, bounding its materialized target-lane
    width without changing target-major result ordering.
    """

    if target_values.ndim != 2:
        raise ValueError("contribution targets must have shape (target, batch)")
    target_count, batch = target_values.shape
    if target_count <= 0 or batch <= 0:
        raise ValueError(
            "contribution targets must have non-empty target and batch axes"
        )
    requested_chunk_size = resolve_stop_gradient_contribution_target_lane_chunk_size(
        target_lane_chunk_size
    )
    resolved_chunk_size = min(requested_chunk_size or target_count, target_count)
    target_chunks = tuple(
        (start, min(start + resolved_chunk_size, target_count))
        for start in range(0, target_count, resolved_chunk_size)
    )
    source = contribution_forward.source_activation
    if source.shape[0] != batch:
        raise ValueError("contribution source and target batch dimensions differ")
    coordinates = contribution_forward.selected_coordinates
    if not coordinates:
        raise ValueError("contribution VJP requires selected coordinates")

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
        "target_lane_count": target_count,
        "target_lane_chunk_size_requested": requested_chunk_size,
        "target_lane_chunk_size_resolved": resolved_chunk_size,
        "target_lane_chunk_count": len(target_chunks),
        "max_materialized_target_lanes": resolved_chunk_size,
        "max_materialized_autograd_lanes": resolved_chunk_size * batch,
        "differentiated_output_shape": list(target_values.shape),
        "dense_differentiated_input_shape": list(
            contribution_forward.dense_source_shape
        ),
        "differentiated_input_shape": list(source.shape),
        "grad_outputs_shape": (
            [target_count * batch] * 2 if len(target_chunks) == 1 else None
        ),
        "max_grad_outputs_shape": [resolved_chunk_size * batch] * 2,
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
        projected_chunks: list[torch.Tensor] = []
        raw_vjp_chunk_shapes: list[list[int]] = []
        grad_outputs_chunk_shapes: list[list[int]] = []
        raw_vjp_numel = 0
        for chunk_index, (target_start, target_end) in enumerate(target_chunks):
            chunk_target_count = target_end - target_start
            chunk_autograd_lanes = chunk_target_count * batch
            grad_outputs = torch.eye(chunk_autograd_lanes, device=source.device)
            target_chunk = target_values[target_start:target_end]
            final_chunk = chunk_index == len(target_chunks) - 1
            raw_vjp = torch.autograd.grad(
                target_chunk.flatten(),
                source,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                retain_graph=(
                    not final_chunk or contribution_forward.retain_graph_for_vjp
                ),
            )[0]
            raw_vjp_chunk_shapes.append(list(raw_vjp.shape))
            grad_outputs_chunk_shapes.append(list(grad_outputs.shape))
            raw_vjp_numel += raw_vjp.numel()
            if sparse:
                # (target * batch, batch, coordinate)
                # -> (coordinate, batch, target)
                projected_chunk = (
                    raw_vjp.reshape(chunk_target_count, batch, batch, len(coordinates))
                    .diagonal(dim1=1, dim2=2)
                    .permute(1, 2, 0)
                    .contiguous()
                )
            else:
                dense_vjp = (
                    raw_vjp.reshape(
                        chunk_target_count,
                        batch,
                        batch,
                        raw_vjp.shape[-2],
                        raw_vjp.shape[-1],
                    )
                    .diagonal(dim1=1, dim2=2)
                    .permute(0, 3, 1, 2)
                )
                projected_chunk = _copy_dense_selected_vjp(dense_vjp, coordinates)
                del dense_vjp
            projected_chunks.append(projected_chunk)
            del grad_outputs, raw_vjp, target_chunk

        projected = (
            projected_chunks[0]
            if len(projected_chunks) == 1
            else torch.cat(projected_chunks, dim=2)
        )
        if vjp_measurement is not None:
            # Preserve the historical generic key while adding explicit raw
            # and projected shapes for sparse-source qualification.
            single_raw_shape = (
                raw_vjp_chunk_shapes[0] if len(raw_vjp_chunk_shapes) == 1 else None
            )
            vjp_measurement.metadata["vjp_result_shape"] = single_raw_shape
            vjp_measurement.metadata["raw_vjp_result_shape"] = single_raw_shape
            vjp_measurement.metadata["raw_vjp_chunk_shapes"] = raw_vjp_chunk_shapes
            vjp_measurement.metadata["grad_outputs_chunk_shapes"] = (
                grad_outputs_chunk_shapes
            )
            vjp_measurement.metadata["projected_vjp_result_shape"] = list(
                projected.shape
            )

    if instrumentation is not None:
        single_raw_shape = (
            raw_vjp_chunk_shapes[0] if len(raw_vjp_chunk_shapes) == 1 else None
        )
        instrumentation.record_layer(
            layer,
            stop_gradient_contribution_raw_vjp_shape=single_raw_shape,
            stop_gradient_contribution_raw_vjp_chunk_shapes=raw_vjp_chunk_shapes,
            stop_gradient_contribution_grad_outputs_shape=(
                [target_count * batch] * 2 if len(target_chunks) == 1 else None
            ),
            stop_gradient_contribution_max_grad_outputs_shape=(
                [resolved_chunk_size * batch] * 2
            ),
            stop_gradient_contribution_projected_vjp_shape=list(projected.shape),
            stop_gradient_contribution_projected_vjp_sha256=raw_tensor_sha256(
                projected
            ),
            stop_gradient_contribution_target_lane_count=target_count,
            stop_gradient_contribution_target_lane_chunk_size_requested=(
                requested_chunk_size
            ),
            stop_gradient_contribution_target_lane_chunk_size_resolved=(
                resolved_chunk_size
            ),
            stop_gradient_contribution_target_lane_chunk_count=len(target_chunks),
            stop_gradient_contribution_max_materialized_target_lanes=(
                resolved_chunk_size
            ),
            stop_gradient_contribution_max_materialized_autograd_lanes=(
                resolved_chunk_size * batch
            ),
            stop_gradient_contribution_dense_vjp_result_materialized=(
                contribution_forward.dense_vjp_result_materialized
            ),
        )
        instrumentation.increment_counter(
            "stop_gradient_contribution_vjp_chunk_executions", len(target_chunks)
        )
        instrumentation.set_counter(
            "stop_gradient_contribution_max_materialized_target_lanes",
            resolved_chunk_size,
        )
        instrumentation.set_counter(
            "stop_gradient_contribution_max_materialized_autograd_lanes",
            resolved_chunk_size * batch,
        )
        if sparse:
            dense_numel = 1
            for extent in dense_vjp_shape:
                dense_numel *= extent
            instrumentation.increment_counter("stop_gradient_sparse_vjp_count")
            instrumentation.increment_counter(
                "stop_gradient_sparse_dense_vjp_result_numel_avoided",
                dense_numel - raw_vjp_numel,
            )
            instrumentation.increment_counter(
                "stop_gradient_sparse_vjp_result_numel", raw_vjp_numel
            )
    return projected


__all__ = [
    "DEFAULT_STOP_GRADIENT_CONTRIBUTION_EXECUTION",
    "SelectedNeuronContributionSource",
    "StopGradientContributionExecution",
    "StopGradientContributionForward",
    "resolve_selected_neuron_contribution_target_lane_chunk_size",
    "resolve_stop_gradient_contribution_execution",
    "resolve_stop_gradient_contribution_target_lane_chunk_size",
    "run_selected_neuron_contribution_vjps",
    "run_stop_gradient_contribution_forward",
    "run_stop_gradient_contribution_vjp",
]
