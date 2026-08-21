"""Selectable OV-only attention adapters for stop-gradient tracing.

The corrected eager and Flash-SDPA adapters preserve ordinary causal attention
values while treating the QK-derived attention map as a constant during
backward. A separate legacy mode preserves the historical unmasked behavior for
forensic reproduction. The Flash adapter does not return or retain the full
attention map and refuses to fall back to a quadratic SDPA backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv

StopGradientAttentionBackend = Literal[
    "legacy_eager_unmasked_v1",
    "eager_causal_v1",
    "flash_sdpa_causal_v1",
]
DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND: StopGradientAttentionBackend = (
    "legacy_eager_unmasked_v1"
)

_EAGER_IMPLEMENTATION = "noqk"
_EAGER_CAUSAL_IMPLEMENTATION = "noqk_eager_causal_v1"
_FLASH_SDPA_IMPLEMENTATION = "noqk_flash_sdpa_causal_v1"


@dataclass(frozen=True)
class _AttentionBackendAdapter:
    """Internal adapter selected by the public backend name."""

    implementation: str
    attention_forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    mask_forward: Callable[..., Any] | None


def noqk_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialized reference attention with gradients restricted to the OV path."""

    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_scores = attn_scores + causal_mask

    attn_weights = (
        nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        .to(query.dtype)
        .detach()
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def noqk_flash_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Flash-only SDPA with Q/K detached and gradients preserved through V/O.

    Restricting the SDPA context to ``FLASH_ATTENTION`` is part of the adapter's
    scientific and memory contract.  Unsupported devices, dtypes, masks, or
    autograd transforms raise instead of silently selecting a quadratic backend.
    """

    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        raise ValueError(
            "stop-gradient attention backend 'flash_sdpa_causal_v1' does not "
            "support output_attentions or head_mask"
        )

    tensors = (query, key, value)
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError(
            "stop-gradient attention backend 'flash_sdpa' requires CUDA tensors; "
            "fallback is disabled"
        )
    if len({tensor.device for tensor in tensors}) != 1:
        raise RuntimeError("Flash-SDPA Q/K/V tensors must be on the same CUDA device")

    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_output, _ = sdpa_attention_forward(
                module,
                query.detach(),
                key.detach(),
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                **kwargs,
            )
    except RuntimeError as error:
        raise RuntimeError(
            "Flash-SDPA could not execute the OV-only attention operation; "
            "fallback to another SDPA backend is disabled"
        ) from error
    return attn_output, None


def _registered_mask(name: str) -> Callable[..., Any]:
    return ALL_MASK_ATTENTION_FUNCTIONS[name]


_BACKENDS: dict[StopGradientAttentionBackend, _AttentionBackendAdapter] = {
    # Frozen historical traces used a custom attention key without a mask
    # registry entry. Transformers 4.57 therefore supplied no causal/padding
    # mask. Keep that behavior selectable for forensic reproduction only.
    "legacy_eager_unmasked_v1": _AttentionBackendAdapter(
        implementation=_EAGER_IMPLEMENTATION,
        attention_forward=noqk_attention_forward,
        mask_forward=None,
    ),
    "eager_causal_v1": _AttentionBackendAdapter(
        implementation=_EAGER_CAUSAL_IMPLEMENTATION,
        attention_forward=noqk_attention_forward,
        mask_forward=_registered_mask("eager"),
    ),
    "flash_sdpa_causal_v1": _AttentionBackendAdapter(
        implementation=_FLASH_SDPA_IMPLEMENTATION,
        attention_forward=noqk_flash_sdpa_attention_forward,
        mask_forward=_registered_mask("sdpa"),
    ),
}


def resolve_stop_gradient_attention_backend(backend: str) -> str:
    """Resolve a public backend name to its registered Transformers implementation."""

    if backend not in _BACKENDS:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"invalid stop-gradient attention backend {backend!r}; "
            f"expected one of: {supported}"
        )
    return _BACKENDS[cast(StopGradientAttentionBackend, backend)].implementation


def _register_once(registry, name: str, function: Callable[..., Any]) -> None:
    """Register globally without overwriting an unrelated installed adapter."""

    try:
        installed = registry[name]
    except KeyError:
        type(registry).register(name, function)
        return
    if installed is not function:
        raise RuntimeError(
            f"refusing to overwrite existing Transformers attention adapter {name!r}"
        )


def _register_backends() -> None:
    for adapter in _BACKENDS.values():
        _register_once(
            ALL_ATTENTION_FUNCTIONS,
            adapter.implementation,
            adapter.attention_forward,
        )
        if adapter.mask_forward is not None:
            _register_once(
                ALL_MASK_ATTENTION_FUNCTIONS,
                adapter.implementation,
                adapter.mask_forward,
            )


_register_backends()


__all__ = [
    "DEFAULT_STOP_GRADIENT_ATTENTION_BACKEND",
    "StopGradientAttentionBackend",
    "noqk_attention_forward",
    "noqk_flash_sdpa_attention_forward",
    "resolve_stop_gradient_attention_backend",
]
