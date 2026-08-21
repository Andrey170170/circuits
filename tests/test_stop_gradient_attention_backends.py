"""Focused qualification of selectable OV-only attention backends."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.grad.attention import (
    StopGradientAttentionBackend,
    noqk_attention_forward,
    noqk_flash_sdpa_attention_forward,
    resolve_stop_gradient_attention_backend,
)
from torch import nn
from transformers import Qwen3Config
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    create_causal_mask,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class _AttentionModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_key_value_groups = 1
        self.is_causal = True


def _additive_causal_mask(
    length: int, dtype: torch.dtype, device: torch.device | str = "cpu"
) -> torch.Tensor:
    allowed = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    return torch.where(
        allowed,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device),
    )[None, None]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_materialized_ov_only_forward_and_gradients_match_reference(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(3)
    batch, tokens, heads, head_dim = 1, 4, 2, 3
    hidden = heads * head_dim
    x = torch.randn(batch, tokens, hidden, dtype=dtype, requires_grad=True)
    q_weight = torch.randn(hidden, hidden, dtype=dtype, requires_grad=True)
    k_weight = torch.randn(hidden, hidden, dtype=dtype, requires_grad=True)
    v_weight = torch.randn(hidden, hidden, dtype=dtype, requires_grad=True)
    o_weight = torch.randn(hidden, hidden, dtype=dtype, requires_grad=True)
    cotangent = torch.randn(batch, tokens, hidden, dtype=dtype)

    def project(weight: torch.Tensor) -> torch.Tensor:
        return (x @ weight).view(batch, tokens, heads, head_dim).transpose(1, 2)

    query, key, value = project(q_weight), project(k_weight), project(v_weight)
    mask = _additive_causal_mask(tokens, dtype)
    module = _AttentionModule()
    actual_attention, actual_weights = noqk_attention_forward(
        module,
        query,
        key,
        value,
        mask,
        scaling=head_dim**-0.5,
    )
    actual = actual_attention.reshape(batch, tokens, hidden) @ o_weight

    reference_weights = torch.softmax(
        (query.detach() @ key.detach().transpose(2, 3)) * (head_dim**-0.5) + mask,
        dim=-1,
        dtype=torch.float32,
    ).to(dtype)
    reference_attention = (reference_weights @ value).transpose(1, 2).contiguous()
    reference = reference_attention.reshape(batch, tokens, hidden) @ o_weight

    variables = (x, q_weight, k_weight, v_weight, o_weight)
    actual_gradients = torch.autograd.grad(
        (actual * cotangent).sum(), variables, allow_unused=True, retain_graph=True
    )
    reference_gradients = torch.autograd.grad(
        (reference * cotangent).sum(), variables, allow_unused=True
    )
    tolerance = {"atol": 1e-5, "rtol": 1e-5}
    if dtype is torch.bfloat16:
        tolerance = {"atol": 2e-2, "rtol": 2e-2}

    torch.testing.assert_close(actual, reference, **tolerance)
    torch.testing.assert_close(actual_weights, reference_weights, **tolerance)
    assert actual_gradients[1] is None
    assert actual_gradients[2] is None
    for actual_gradient, reference_gradient in zip(
        (actual_gradients[0], actual_gradients[3], actual_gradients[4]),
        (reference_gradients[0], reference_gradients[3], reference_gradients[4]),
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, reference_gradient, **tolerance)


def _qwen_mask(implementation: str, attention_mask: torch.Tensor):
    config = Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
    )
    config._attn_implementation = implementation
    tokens = attention_mask.shape[1]
    return create_causal_mask(
        config=config,
        input_embeds=torch.zeros(1, tokens, 8),
        attention_mask=attention_mask,
        cache_position=torch.arange(tokens),
        past_key_values=None,
        position_ids=torch.arange(tokens)[None],
    )


def test_versioned_backends_register_distinct_qwen_mask_contracts() -> None:
    legacy = resolve_stop_gradient_attention_backend("legacy_eager_unmasked_v1")
    eager = resolve_stop_gradient_attention_backend("eager_causal_v1")
    flash = resolve_stop_gradient_attention_backend("flash_sdpa_causal_v1")

    assert legacy == "noqk"
    assert eager == "noqk_eager_causal_v1"
    assert flash == "noqk_flash_sdpa_causal_v1"
    assert legacy in ALL_ATTENTION_FUNCTIONS._global_mapping
    assert legacy not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    assert eager in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    assert flash in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping

    padded = torch.tensor([[1, 1, 0]])
    assert _qwen_mask(legacy, padded) is None

    eager_mask = _qwen_mask(eager, padded)
    assert eager_mask.shape == (1, 1, 3, 3)
    assert eager_mask.dtype == torch.float32
    assert eager_mask[0, 0, 1, 0] == 0
    assert eager_mask[0, 0, 0, 1] < -1e20
    assert eager_mask[0, 0, 1, 2] < -1e20

    flash_mask = _qwen_mask(flash, padded)
    assert flash_mask.shape == (1, 1, 3, 3)
    assert flash_mask.dtype == torch.bool
    assert flash_mask[0, 0, 1, 0]
    assert not flash_mask[0, 0, 0, 1]
    assert not flash_mask[0, 0, 1, 2]


def test_flash_backend_rejects_cpu_instead_of_falling_back() -> None:
    query = torch.randn(1, 1, 2, 4, requires_grad=True)
    key = torch.randn(1, 1, 2, 4, requires_grad=True)
    value = torch.randn(1, 1, 2, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match=r"requires CUDA.*fallback is disabled"):
        noqk_flash_sdpa_attention_forward(
            _AttentionModule(),
            query,
            key,
            value,
            None,
            scaling=0.5,
        )

    with pytest.raises(ValueError, match="does not support output_attentions"):
        noqk_flash_sdpa_attention_forward(
            _AttentionModule(),
            query,
            key,
            value,
            None,
            scaling=0.5,
            output_attentions=True,
        )

    with pytest.raises(ValueError, match="does not support output_attentions"):
        noqk_flash_sdpa_attention_forward(
            _AttentionModule(),
            query,
            key,
            value,
            None,
            scaling=0.5,
            head_mask=torch.ones(1),
        )


def test_invalid_backend_and_old_config_state_fail_or_restore_explicitly() -> None:
    with pytest.raises(ValueError, match="invalid stop-gradient attention backend"):
        resolve_stop_gradient_attention_backend("auto")
    with pytest.raises(ValueError, match="invalid stop-gradient attention backend"):
        ADAGConfig(
            stop_gradient_attention_backend=cast(StopGradientAttentionBackend, "auto")
        )

    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.stop_gradient_attention_backend == "legacy_eager_unmasked_v1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_flash_backend_cuda_gqa_batched_vjp_matches_eager_and_uses_flash() -> None:
    torch.manual_seed(9)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query = torch.randn(1, 4, 5, 8, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(1, 2, 5, 8, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(1, 2, 5, 8, device=device, dtype=dtype, requires_grad=True)
    module = _AttentionModule().to(device)
    module.num_key_value_groups = 2
    scaling = 8**-0.5

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        flash_output, flash_weights = noqk_flash_sdpa_attention_forward(
            module, query, key, value, None, scaling=scaling
        )
        batched_cotangents = torch.randn(
            2, *flash_output.shape, device=device, dtype=dtype
        )
        flash_gradient = torch.autograd.grad(
            flash_output,
            (query, key, value),
            grad_outputs=batched_cotangents,
            is_grads_batched=True,
            allow_unused=True,
            retain_graph=True,
        )
    eager_output, _ = noqk_attention_forward(
        module,
        query,
        key,
        value,
        _additive_causal_mask(5, dtype, device),
        scaling=scaling,
    )
    eager_gradient = torch.autograd.grad(
        eager_output,
        (query, key, value),
        grad_outputs=batched_cotangents,
        is_grads_batched=True,
        allow_unused=True,
    )

    assert flash_weights is None
    assert flash_gradient[0] is None
    assert flash_gradient[1] is None
    torch.testing.assert_close(flash_output, eager_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        flash_gradient[2], eager_gradient[2], atol=2e-2, rtol=2e-2
    )
    operator_names = {event.key for event in profile.key_averages()}
    assert any("_scaled_dot_product_flash_attention" in name for name in operator_names)
    assert not any(
        "_scaled_dot_product_attention_math" in name for name in operator_names
    )
