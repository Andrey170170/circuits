"""Execution strategies for cross-layer Jacobian evaluation.

The caller owns layer-pair planning and graph materialization.  This module owns
only the expensive forward/VJP execution behind one prepared-executor seam.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from circuits.tracing.grad import (
    DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND,
    StopGradientAttentionBackend,
    layerwise_revert_stop_nonlinear_grad,
    layerwise_stop_nonlinear_grad,
    resolve_stop_gradient_attention_backend,
)
from circuits.tracing.instrumentation import TraceInstrumentation

CrossLayerJacobianExecution = Literal["full_model_v1", "cached_range_v1"]
DEFAULT_CROSS_LAYER_JACOBIAN_EXECUTION: CrossLayerJacobianExecution = "full_model_v1"


@dataclass(frozen=True)
class CrossLayerJacobianPreparation:
    """Inputs shared by every layer-pair execution in one trace."""

    model: nn.Module
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    source_layers: tuple[int, ...]
    use_relp_grad: bool
    disable_stop_grad: bool
    use_stop_grad_on_mlps: bool
    device: str | torch.device
    attention_backend: StopGradientAttentionBackend = (
        DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND
    )
    instrumentation: TraceInstrumentation | None = None


@dataclass(frozen=True)
class CrossLayerJacobianPair:
    """Ordered selected coordinates for one source/target layer pair."""

    src_layer: int
    tgt_layer: int
    src_neurons: tuple[tuple[int, int], ...]
    tgt_neurons: tuple[tuple[int, int], ...]
    tgt_chunk_size: int
    alpha: float | None = None


_RECEIPT_NAMES = (
    "selected_source_activations",
    "selected_target_activations",
    "selected_raw_jacobian",
)


@dataclass(frozen=True)
class CrossLayerJacobianReceipts:
    """Ordered exact receipts over compact, raw-dtype pair intermediates."""

    selected_source_activations_sha256: str
    selected_target_activations_sha256: str
    selected_raw_jacobian_sha256: str

    def ordered(self) -> list[dict[str, str]]:
        return [
            {"name": name, "sha256": digest}
            for name, digest in zip(
                _RECEIPT_NAMES,
                (
                    self.selected_source_activations_sha256,
                    self.selected_target_activations_sha256,
                    self.selected_raw_jacobian_sha256,
                ),
                strict=True,
            )
        ]


@dataclass(frozen=True)
class CrossLayerJacobianPairResult:
    """Pair attribution plus receipts from before attribution normalization."""

    relative_attribution: torch.Tensor
    receipts: CrossLayerJacobianReceipts


CrossLayerJacobianResult = (
    CrossLayerJacobianPairResult | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
)


class CrossLayerJacobianExecutor(Protocol):
    """Prepared adapter at the cross-layer Jacobian execution seam."""

    def compute_pair(self, pair: CrossLayerJacobianPair) -> CrossLayerJacobianResult:
        """Compute one pair while preserving selected-coordinate order."""


def resolve_cross_layer_jacobian_execution(
    execution: str,
) -> CrossLayerJacobianExecution:
    """Validate and return a provenance-bearing execution strategy."""

    if execution not in {"full_model_v1", "cached_range_v1"}:
        raise ValueError(
            f"invalid cross-layer Jacobian execution {execution!r}; expected one of "
            "['cached_range_v1', 'full_model_v1']"
        )
    return cast(CrossLayerJacobianExecution, execution)


def _down_projection(model: nn.Module, layer: int) -> nn.Module:
    mlp = model.model.layers[layer].mlp
    return mlp.mlp.down_proj if hasattr(mlp, "mlp") else mlp.down_proj


def _increment(
    instrumentation: TraceInstrumentation | None, name: str, value: int = 1
) -> None:
    if instrumentation is not None:
        instrumentation.increment_counter(name, value)


def _set(instrumentation: TraceInstrumentation | None, name: str, value: Any) -> None:
    if instrumentation is not None:
        instrumentation.set_counter(name, value)


def _selected_activations(
    activation: torch.Tensor, coordinates: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    return torch.stack(
        [activation[:, position, neuron] for position, neuron in coordinates],
        dim=-1,
    )


def _raw_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash dtype, shape, and exact compact tensor bytes without value casting."""

    value = tensor.detach().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _receipts(
    source_values: torch.Tensor,
    target_values: torch.Tensor,
    raw_jacobian: torch.Tensor,
) -> CrossLayerJacobianReceipts:
    return CrossLayerJacobianReceipts(
        selected_source_activations_sha256=_raw_tensor_sha256(source_values),
        selected_target_activations_sha256=_raw_tensor_sha256(target_values),
        selected_raw_jacobian_sha256=_raw_tensor_sha256(raw_jacobian),
    )


def _full_model_selected_vjp(
    source_activation: torch.Tensor,
    target_activation: torch.Tensor,
    pair: CrossLayerJacobianPair,
    *,
    device: str | torch.device,
    instrumentation: TraceInstrumentation | None,
) -> CrossLayerJacobianResult:
    """Historical full-gradient materialization, kept byte-for-byte in math order."""

    batch = source_activation.shape[0]
    src_acts = _selected_activations(source_activation, pair.src_neurons)
    tgt_activations = _selected_activations(
        target_activation, pair.tgt_neurons
    ).permute(1, 0)
    source_jacobian_chunks: list[torch.Tensor] = []
    for chunk_start in range(0, len(pair.tgt_neurons), pair.tgt_chunk_size):
        chunk_end = min(chunk_start + pair.tgt_chunk_size, len(pair.tgt_neurons))
        target_chunk = tgt_activations[chunk_start:chunk_end]
        chunk_size = target_chunk.shape[0]
        grad_outputs = torch.eye(batch * chunk_size, device=device)
        source_activation.grad = None
        chunk_source_jacobian = torch.autograd.grad(
            target_chunk.flatten(),
            source_activation,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=chunk_end < len(pair.tgt_neurons),
        )[0]
        chunk_source_jacobian = chunk_source_jacobian.reshape(
            chunk_size,
            batch,
            batch,
            chunk_source_jacobian.shape[-2],
            chunk_source_jacobian.shape[-1],
        )
        chunk_source_jacobian = chunk_source_jacobian.diagonal(dim1=1, dim2=2).permute(
            0, 3, 1, 2
        )
        source_jacobian_chunks.append(chunk_source_jacobian)
        _increment(instrumentation, "cross_layer_vjp_chunk_executions")
        del grad_outputs

    source_jacobian = torch.cat(source_jacobian_chunks, dim=0)
    if pair.alpha is not None:
        grads = torch.stack(
            [
                source_jacobian[:, :, position, neuron]
                for position, neuron in pair.src_neurons
            ],
            dim=-1,
        )
        grads = grads.permute(1, 2, 0).contiguous()
        return grads, src_acts, tgt_activations.detach().permute(1, 0)

    raw_selected_jacobian = (
        torch.stack(
            [
                source_jacobian[:, :, position, neuron]
                for position, neuron in pair.src_neurons
            ],
            dim=-1,
        )
        .permute(1, 2, 0)
        .contiguous()
    )
    jvps = torch.stack(
        [
            source_jacobian[:, :, position, neuron]
            * source_activation[:, position, neuron][None, :].detach()
            for position, neuron in pair.src_neurons
        ],
        dim=-1,
    )
    jvps = jvps.permute(1, 2, 0).contiguous()
    target_values = tgt_activations.detach().permute(1, 0)
    eps = target_values.abs().mean() * 1e-6
    return CrossLayerJacobianPairResult(
        relative_attribution=jvps / (target_values[:, None, :] + eps),
        receipts=_receipts(src_acts, target_values, raw_selected_jacobian),
    )


def _streamed_selected_vjp(
    source_activation: torch.Tensor,
    target_activation: torch.Tensor,
    pair: CrossLayerJacobianPair,
    *,
    device: str | torch.device,
    instrumentation: TraceInstrumentation | None,
) -> CrossLayerJacobianPairResult:
    """Project each target VJP chunk before retaining the next one."""

    if pair.alpha is not None:
        raise ValueError("cached_range_v1 does not support integrated gradients")
    batch = source_activation.shape[0]
    source_values = _selected_activations(source_activation, pair.src_neurons)
    target_values_t_b = _selected_activations(
        target_activation, pair.tgt_neurons
    ).permute(1, 0)
    target_values = target_values_t_b.detach().permute(1, 0)
    eps = target_values.abs().mean() * 1e-6
    relative_chunks: list[torch.Tensor] = []
    raw_selected_chunks: list[torch.Tensor] = []
    for chunk_start in range(0, len(pair.tgt_neurons), pair.tgt_chunk_size):
        chunk_end = min(chunk_start + pair.tgt_chunk_size, len(pair.tgt_neurons))
        target_chunk = target_values_t_b[chunk_start:chunk_end]
        chunk_size = target_chunk.shape[0]
        grad_outputs = torch.eye(batch * chunk_size, device=device)
        source_activation.grad = None
        chunk_source_jacobian = torch.autograd.grad(
            target_chunk.flatten(),
            source_activation,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=chunk_end < len(pair.tgt_neurons),
        )[0]
        chunk_source_jacobian = chunk_source_jacobian.reshape(
            chunk_size,
            batch,
            batch,
            chunk_source_jacobian.shape[-2],
            chunk_source_jacobian.shape[-1],
        )
        chunk_source_jacobian = chunk_source_jacobian.diagonal(dim1=1, dim2=2).permute(
            0, 3, 1, 2
        )
        selected_chunk = torch.stack(
            [
                chunk_source_jacobian[:, :, position, neuron]
                for position, neuron in pair.src_neurons
            ],
            dim=-1,
        )
        raw_selected_chunks.append(selected_chunk.permute(1, 2, 0).contiguous())
        chunk_jvps = (selected_chunk * source_values.detach()[None, :, :]).permute(
            1, 2, 0
        )
        relative_chunks.append(
            chunk_jvps / (target_values[:, None, chunk_start:chunk_end] + eps)
        )
        _increment(instrumentation, "cross_layer_vjp_chunk_executions")
        del grad_outputs, chunk_source_jacobian, selected_chunk, chunk_jvps
    raw_selected_jacobian = torch.cat(raw_selected_chunks, dim=-1).contiguous()
    return CrossLayerJacobianPairResult(
        relative_attribution=torch.cat(relative_chunks, dim=-1).contiguous(),
        receipts=_receipts(source_values, target_values, raw_selected_jacobian),
    )


class _FullModelExecutor:
    def __init__(self, preparation: CrossLayerJacobianPreparation) -> None:
        self.preparation = preparation

    def compute_pair(self, pair: CrossLayerJacobianPair) -> CrossLayerJacobianResult:
        request = self.preparation
        model = request.model
        handles: list[torch.utils.hooks.RemovableHandle] = []
        transaction_active = False
        activation_cache: dict[int, torch.Tensor] = {}

        def make_hook(layer: int):
            def hook(_module, inputs, _output):
                activation_cache[layer] = inputs[0]

            return hook

        try:
            if not request.disable_stop_grad:
                layerwise_stop_nonlinear_grad(
                    model,
                    pair.src_layer,
                    pair.tgt_layer,
                    use_relp_grad=request.use_relp_grad,
                    use_stop_grad_on_mlps=request.use_stop_grad_on_mlps,
                    attention_backend=request.attention_backend,
                )
                transaction_active = True
            model.zero_grad()
            embeds = (
                model.model.embed_tokens(request.input_ids).detach().requires_grad_()
            )
            if pair.alpha is not None:
                embeds = embeds * pair.alpha
            handles.extend(
                [
                    _down_projection(model, pair.src_layer).register_forward_hook(
                        make_hook(pair.src_layer)
                    ),
                    _down_projection(model, pair.tgt_layer).register_forward_hook(
                        make_hook(pair.tgt_layer)
                    ),
                ]
            )
            _ = model(inputs_embeds=embeds, attention_mask=request.attention_mask)
            _increment(
                request.instrumentation,
                "cross_layer_full_decoder_layer_executions",
                len(model.model.layers),
            )
            return _full_model_selected_vjp(
                activation_cache[pair.src_layer],
                activation_cache[pair.tgt_layer],
                pair,
                device=request.device,
                instrumentation=request.instrumentation,
            )
        finally:
            for handle in handles:
                handle.remove()
            if transaction_active:
                layerwise_revert_stop_nonlinear_grad(
                    model, pair.src_layer, pair.tgt_layer
                )


@dataclass(frozen=True)
class _LayerFrame:
    kwargs: dict[str, Any]
    uses_dynamic_cache: bool


class _TargetActivationCaptured(Exception):
    pass


def _tensor_bytes(value: Any, seen: set[int]) -> int:
    if isinstance(value, torch.Tensor):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item, seen) for item in value)
    return 0


class _CachedRangeExecutor:
    def __init__(self, preparation: CrossLayerJacobianPreparation) -> None:
        if preparation.disable_stop_grad:
            raise ValueError(
                "cached_range_v1 requires stop-gradient tracing to be enabled"
            )
        self.preparation = preparation
        self.source_entries: dict[int, torch.Tensor] = {}
        self.layer_frames: dict[int, _LayerFrame] = {}
        if preparation.source_layers:
            self._prepare()

    def _prepare(self) -> None:
        request = self.preparation
        model = request.model
        handles: list[torch.utils.hooks.RemovableHandle] = []
        attention_configs: list[tuple[Any, bool, Any]] = []
        source_layers = set(request.source_layers)

        def make_hook(layer: int):
            def hook(_module, args, kwargs):
                hidden_states = args[0] if args else kwargs["hidden_states"]
                if layer in source_layers:
                    self.source_entries[layer] = hidden_states.detach()
                frame_kwargs = dict(kwargs)
                past_key_values = frame_kwargs.pop("past_key_values", None)
                if past_key_values is not None and not isinstance(
                    past_key_values, DynamicCache
                ):
                    raise TypeError(
                        "cached_range_v1 currently requires DynamicCache semantics"
                    )
                self.layer_frames[layer] = _LayerFrame(
                    kwargs=frame_kwargs,
                    uses_dynamic_cache=past_key_values is not None,
                )

            return hook

        try:
            implementation = resolve_stop_gradient_attention_backend(
                request.attention_backend
            )
            seen_configs: set[int] = set()
            for layer in model.model.layers:
                config = layer.self_attn.config
                if id(config) in seen_configs:
                    continue
                seen_configs.add(id(config))
                attention_configs.append(
                    (
                        config,
                        hasattr(config, "_attn_implementation"),
                        getattr(config, "_attn_implementation", None),
                    )
                )
                config._attn_implementation = implementation
            for layer_index, layer in enumerate(model.model.layers):
                handles.append(
                    layer.register_forward_pre_hook(
                        make_hook(layer_index), with_kwargs=True
                    )
                )
            model.zero_grad()
            embeds = model.model.embed_tokens(request.input_ids).detach()
            with torch.no_grad():
                _ = model(inputs_embeds=embeds, attention_mask=request.attention_mask)
            missing = source_layers.difference(self.source_entries)
            if missing:
                raise RuntimeError(
                    f"preparation did not capture source layers {sorted(missing)}"
                )
            _increment(request.instrumentation, "cross_layer_preparation_forward_count")
            _increment(
                request.instrumentation,
                "cross_layer_full_decoder_layer_executions",
                len(model.model.layers),
            )
            seen: set[int] = set()
            cache_bytes = _tensor_bytes(self.source_entries, seen) + sum(
                _tensor_bytes(frame.kwargs, seen)
                for frame in self.layer_frames.values()
            )
            _set(
                request.instrumentation,
                "cross_layer_preparation_cache_bytes",
                cache_bytes,
            )
        finally:
            for handle in handles:
                handle.remove()
            for config, had_attribute, original_value in attention_configs:
                if had_attribute:
                    config._attn_implementation = original_value
                elif hasattr(config, "_attn_implementation"):
                    delattr(config, "_attn_implementation")

    def compute_pair(
        self, pair: CrossLayerJacobianPair
    ) -> CrossLayerJacobianPairResult:
        if pair.alpha is not None:
            raise ValueError("cached_range_v1 does not support integrated gradients")
        request = self.preparation
        model = request.model
        if pair.src_layer not in self.source_entries:
            raise ValueError(
                f"source layer {pair.src_layer} was not included in preparation"
            )
        handles: list[torch.utils.hooks.RemovableHandle] = []
        transaction_active = False
        source_leaf: torch.Tensor | None = None
        target_activation: torch.Tensor | None = None

        def replace_source(_module, inputs):
            nonlocal source_leaf
            source_leaf = inputs[0].detach().requires_grad_()
            return (source_leaf, *inputs[1:])

        def capture_target(_module, inputs):
            nonlocal target_activation
            target_activation = inputs[0]
            raise _TargetActivationCaptured

        try:
            layerwise_stop_nonlinear_grad(
                model,
                pair.src_layer,
                pair.tgt_layer,
                use_relp_grad=request.use_relp_grad,
                use_stop_grad_on_mlps=request.use_stop_grad_on_mlps,
                attention_backend=request.attention_backend,
            )
            transaction_active = True
            handles.extend(
                [
                    _down_projection(model, pair.src_layer).register_forward_pre_hook(
                        replace_source
                    ),
                    _down_projection(model, pair.tgt_layer).register_forward_pre_hook(
                        capture_target
                    ),
                ]
            )
            model.zero_grad()
            hidden_states = self.source_entries[pair.src_layer]
            replay_cache = (
                DynamicCache(config=model.config)
                if self.layer_frames[pair.src_layer].uses_dynamic_cache
                else None
            )
            try:
                for layer_index in range(pair.src_layer, pair.tgt_layer + 1):
                    frame = self.layer_frames[layer_index]
                    hidden_states = model.model.layers[layer_index](
                        hidden_states,
                        past_key_values=replay_cache,
                        **frame.kwargs,
                    )
            except _TargetActivationCaptured:
                pass
            else:
                raise RuntimeError(
                    "target down projection was not reached during replay"
                )
            if source_leaf is None or target_activation is None:
                raise RuntimeError(
                    "replay did not capture source and target activations"
                )
            _increment(
                request.instrumentation,
                "cross_layer_replay_decoder_layer_entries",
                pair.tgt_layer - pair.src_layer + 1,
            )
            return _streamed_selected_vjp(
                source_leaf,
                target_activation,
                pair,
                device=request.device,
                instrumentation=request.instrumentation,
            )
        finally:
            for handle in handles:
                handle.remove()
            if transaction_active:
                layerwise_revert_stop_nonlinear_grad(
                    model, pair.src_layer, pair.tgt_layer
                )


def prepare_cross_layer_jacobian_execution(
    preparation: CrossLayerJacobianPreparation,
    *,
    execution: CrossLayerJacobianExecution = DEFAULT_CROSS_LAYER_JACOBIAN_EXECUTION,
) -> CrossLayerJacobianExecutor:
    """Prepare the selected adapter once for a trace's ordered pair stream."""

    execution = resolve_cross_layer_jacobian_execution(execution)
    _set(
        preparation.instrumentation,
        "cross_layer_jacobian_execution",
        execution,
    )
    for counter in (
        "cross_layer_preparation_forward_count",
        "cross_layer_preparation_cache_bytes",
        "cross_layer_full_decoder_layer_executions",
        "cross_layer_replay_decoder_layer_entries",
        "cross_layer_vjp_chunk_executions",
    ):
        _set(preparation.instrumentation, counter, 0)
    if execution == "full_model_v1":
        return _FullModelExecutor(preparation)
    return _CachedRangeExecutor(preparation)


__all__ = [
    "DEFAULT_CROSS_LAYER_JACOBIAN_EXECUTION",
    "CrossLayerJacobianExecution",
    "CrossLayerJacobianExecutor",
    "CrossLayerJacobianPair",
    "CrossLayerJacobianPairResult",
    "CrossLayerJacobianPreparation",
    "CrossLayerJacobianReceipts",
    "CrossLayerJacobianResult",
    "prepare_cross_layer_jacobian_execution",
    "resolve_cross_layer_jacobian_execution",
]
