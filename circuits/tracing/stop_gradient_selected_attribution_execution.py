"""Forward execution adapters for stop-gradient selected attribution.

The interface owns temporary activation hooks and the control-flow sentinel
used to stop a model forward at one selected MLP down projection.  Callers get
the same selected activation and differentiable embedding graph without
coordinating hook lifetime or exception handling themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn

from circuits.tracing.instrumentation import TraceInstrumentation

StopGradientSelectedAttributionForwardExecution = Literal[
    "full_model_v1",
    "prefix_stop_v1",
]
DEFAULT_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION: StopGradientSelectedAttributionForwardExecution = "full_model_v1"


@dataclass(frozen=True)
class StopGradientSelectedAttributionForward:
    """Captured activation and prefix autograd graph for one selected layer."""

    embeddings: torch.Tensor
    activation: torch.Tensor
    execution: StopGradientSelectedAttributionForwardExecution
    layer: int
    decoder_layer_entries: tuple[int, ...]
    selected_down_projection_completed: bool
    lm_head_completed: bool
    logits_completed: bool
    model_forward_completed: bool
    stopped_before_down_projection: bool
    down_projection_materialized: bool
    decoder_suffix_materialized: bool
    logits_materialized: bool
    logit_shape: tuple[int, ...] | None


class _PrefixStop(BaseException):
    """Private control-flow sentinel raised only by our capture pre-hook."""


class _ForwardObservation:
    """Own temporary hooks that observe what the model actually executes."""

    def __init__(self, model: nn.Module, layer: int) -> None:
        self.model = model
        self.layer = layer
        self.decoder_layer_entries: list[int] = []
        self.selected_input: torch.Tensor | None = None
        self.selected_down_projection_completed = False
        self.lm_head_completed = False
        self._handles: list[Any] = []

    def __enter__(self) -> _ForwardObservation:
        def record_layer_entry(layer: int):
            def hook(_module, _inputs) -> None:
                self.decoder_layer_entries.append(layer)

            return hook

        def capture_selected_input(_module, inputs) -> None:
            self.selected_input = inputs[0]

        def record_selected_completion(_module, _inputs, _output) -> None:
            self.selected_down_projection_completed = True

        def record_lm_head_completion(_module, _inputs, _output) -> None:
            self.lm_head_completed = True

        try:
            for layer, decoder_layer in enumerate(self.model.model.layers):
                self._handles.append(
                    decoder_layer.register_forward_pre_hook(record_layer_entry(layer))
                )
            selected_down_projection = _down_projection(self.model, self.layer)
            self._handles.append(
                selected_down_projection.register_forward_pre_hook(
                    capture_selected_input
                )
            )
            self._handles.append(
                selected_down_projection.register_forward_hook(
                    record_selected_completion
                )
            )
            lm_head = getattr(self.model, "lm_head", None)
            if not isinstance(lm_head, nn.Module):
                raise RuntimeError(
                    "selected-attribution execution requires model.lm_head"
                )
            self._handles.append(
                lm_head.register_forward_hook(record_lm_head_completion)
            )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()


def resolve_stop_gradient_selected_attribution_forward_execution(
    execution: str,
) -> StopGradientSelectedAttributionForwardExecution:
    """Validate and return a provenance-bearing selected-forward adapter."""

    if execution not in {"full_model_v1", "prefix_stop_v1"}:
        raise ValueError(
            "invalid stop-gradient selected-attribution forward execution "
            f"{execution!r}; expected one of ['full_model_v1', 'prefix_stop_v1']"
        )
    return cast(StopGradientSelectedAttributionForwardExecution, execution)


def _down_projection(model: nn.Module, layer: int) -> nn.Module:
    mlp = model.model.layers[layer].mlp
    return mlp.mlp.down_proj if hasattr(mlp, "mlp") else mlp.down_proj


def _increment(
    instrumentation: TraceInstrumentation | None, name: str, value: int = 1
) -> None:
    if instrumentation is not None:
        instrumentation.increment_counter(name, value)


def _differentiable_embeddings(
    model: nn.Module, input_ids: torch.Tensor
) -> torch.Tensor:
    return model.model.embed_tokens(input_ids).detach().requires_grad_()


def _run_full_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layer: int,
    center_logits: bool,
) -> StopGradientSelectedAttributionForward:
    embeddings = _differentiable_embeddings(model, input_ids)
    with _ForwardObservation(model, layer) as observation:
        output = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = output.logits
        logits_completed = True
        if center_logits:
            logits -= logits.mean(dim=-1)
        logit_shape = tuple(logits.shape)
        del output, logits
    if observation.selected_input is None:
        raise RuntimeError(
            f"selected layer {layer} down projection did not execute during full forward"
        )
    expected_entries = tuple(range(len(model.model.layers)))
    observed_entries = tuple(observation.decoder_layer_entries)
    if (
        observed_entries != expected_entries
        or not observation.selected_down_projection_completed
        or not observation.lm_head_completed
        or not logits_completed
    ):
        raise RuntimeError(
            "full-model selected-attribution forward execution receipts are incomplete"
        )
    return StopGradientSelectedAttributionForward(
        embeddings=embeddings,
        activation=observation.selected_input,
        execution="full_model_v1",
        layer=layer,
        decoder_layer_entries=observed_entries,
        selected_down_projection_completed=(
            observation.selected_down_projection_completed
        ),
        lm_head_completed=observation.lm_head_completed,
        logits_completed=logits_completed,
        model_forward_completed=True,
        stopped_before_down_projection=False,
        down_projection_materialized=(observation.selected_down_projection_completed),
        decoder_suffix_materialized=any(entry > layer for entry in observed_entries),
        logits_materialized=logits_completed,
        logit_shape=logit_shape,
    )


def _run_prefix_stop(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layer: int,
) -> StopGradientSelectedAttributionForward:
    def capture_and_stop(_module, inputs) -> None:
        del inputs
        raise _PrefixStop

    embeddings = _differentiable_embeddings(model, input_ids)
    with _ForwardObservation(model, layer) as observation:
        hook_handle = _down_projection(model, layer).register_forward_pre_hook(
            capture_and_stop
        )
        try:
            try:
                model(inputs_embeds=embeddings, attention_mask=attention_mask)
            except _PrefixStop:
                pass
            else:
                raise RuntimeError(
                    f"selected layer {layer} down projection did not execute during prefix forward"
                )
        finally:
            hook_handle.remove()
    if observation.selected_input is None:
        raise RuntimeError(
            f"selected layer {layer} prefix stop did not capture an activation"
        )
    observed_entries = tuple(observation.decoder_layer_entries)
    expected_entries = tuple(range(layer + 1))
    if (
        observed_entries != expected_entries
        or observation.selected_down_projection_completed
        or observation.lm_head_completed
    ):
        raise RuntimeError(
            "prefix-stop selected-attribution forward execution receipts are invalid"
        )
    return StopGradientSelectedAttributionForward(
        embeddings=embeddings,
        activation=observation.selected_input,
        execution="prefix_stop_v1",
        layer=layer,
        decoder_layer_entries=observed_entries,
        selected_down_projection_completed=False,
        lm_head_completed=False,
        logits_completed=False,
        model_forward_completed=False,
        stopped_before_down_projection=True,
        down_projection_materialized=False,
        decoder_suffix_materialized=False,
        logits_materialized=False,
        logit_shape=None,
    )


def run_stop_gradient_selected_attribution_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layer: int,
    execution: StopGradientSelectedAttributionForwardExecution = (
        DEFAULT_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION
    ),
    center_logits: bool = False,
    instrumentation: TraceInstrumentation | None = None,
) -> StopGradientSelectedAttributionForward:
    """Run one selected-layer forward behind the configured execution seam."""

    resolved = resolve_stop_gradient_selected_attribution_forward_execution(execution)
    if not 0 <= layer < len(model.model.layers):
        raise ValueError(f"selected layer {layer} is outside the model layer range")
    if resolved == "full_model_v1":
        result = _run_full_model(
            model,
            input_ids,
            attention_mask,
            layer=layer,
            center_logits=center_logits,
        )
    else:
        result = _run_prefix_stop(
            model,
            input_ids,
            attention_mask,
            layer=layer,
        )
    _increment(
        instrumentation,
        "stop_gradient_selected_attribution_forward_execution_count",
    )
    _increment(
        instrumentation,
        f"stop_gradient_selected_attribution_{resolved}_execution_count",
    )
    for name, materialized in (
        ("down_projection", result.down_projection_materialized),
        ("decoder_suffix", result.decoder_suffix_materialized),
        ("logits", result.logits_materialized),
    ):
        _increment(
            instrumentation,
            f"stop_gradient_selected_attribution_{name}_materialized_count",
            int(materialized),
        )
    if instrumentation is not None:
        instrumentation.increment_counter(
            "stop_gradient_selected_attribution_decoder_layer_entry_count",
            len(result.decoder_layer_entries),
        )
        instrumentation.increment_counter(
            "stop_gradient_selected_attribution_selected_down_projection_completed_count",
            int(result.selected_down_projection_completed),
        )
        instrumentation.increment_counter(
            "stop_gradient_selected_attribution_lm_head_completed_count",
            int(result.lm_head_completed),
        )
        instrumentation.increment_counter(
            "stop_gradient_selected_attribution_logits_completed_count",
            int(result.logits_completed),
        )
        instrumentation.append_execution_record(
            "stop_gradient_selected_attribution_forward",
            execution=result.execution,
            layer=result.layer,
            decoder_layer_entries=list(result.decoder_layer_entries),
            selected_down_projection_completed=(
                result.selected_down_projection_completed
            ),
            lm_head_completed=result.lm_head_completed,
            logits_completed=result.logits_completed,
            down_projection_materialized=result.down_projection_materialized,
            decoder_suffix_materialized=result.decoder_suffix_materialized,
            logits_materialized=result.logits_materialized,
        )
    return result
