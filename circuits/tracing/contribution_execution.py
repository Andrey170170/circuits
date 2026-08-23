"""Execution strategies for per-layer stop-gradient contribution VJPs.

The public seam in this module owns three details that attribution callers
should not need to coordinate themselves: selecting the layer's real
``down_proj`` beneath any stop-gradient wrapper, installing and removing the
appropriate activation hook, and deciding where the autograd graph begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch

StopGradientContributionExecution = Literal["full_graph_v1", "source_leaf_v1"]
DEFAULT_STOP_GRADIENT_CONTRIBUTION_EXECUTION: StopGradientContributionExecution = (
    "full_graph_v1"
)


@dataclass(frozen=True)
class StopGradientContributionForward:
    """Forward result and VJP lifetime contract for one selected MLP layer."""

    logits: torch.Tensor
    source_activation: torch.Tensor
    retain_graph_for_vjp: bool


@dataclass(frozen=True)
class _ExecutionAdapter:
    differentiable_embeddings: bool
    source_is_leaf: bool
    retain_graph_for_vjp: bool


_EXECUTION_ADAPTERS: dict[StopGradientContributionExecution, _ExecutionAdapter] = {
    "full_graph_v1": _ExecutionAdapter(
        differentiable_embeddings=True,
        source_is_leaf=False,
        retain_graph_for_vjp=True,
    ),
    "source_leaf_v1": _ExecutionAdapter(
        differentiable_embeddings=False,
        source_is_leaf=True,
        retain_graph_for_vjp=False,
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
) -> StopGradientContributionForward:
    """Run one contribution forward and expose only its selected source.

    ``full_graph_v1`` reproduces the historical differentiable-embedding
    execution. ``source_leaf_v1`` computes the prefix without an autograd
    source, then replaces the selected ``down_proj`` input with an equal-valued
    detached leaf. The latter keeps the scientific source tensor unchanged
    while retaining only the downstream graph needed by the sole VJP.

    The temporary hook is always removed, including when the model forward
    raises. Source-leaf execution temporarily freezes every model parameter so
    no upstream graph can begin at a trainable weight, then restores every
    original ``requires_grad`` flag before returning.
    """

    execution = resolve_stop_gradient_contribution_execution(execution)
    adapter = _EXECUTION_ADAPTERS[execution]
    down_projection = _selected_down_projection(model, layer)
    source: torch.Tensor | None = None
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

        if adapter.source_is_leaf:

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

    return StopGradientContributionForward(
        logits=output.logits,
        source_activation=source,
        retain_graph_for_vjp=adapter.retain_graph_for_vjp,
    )
