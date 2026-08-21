"""State-restoration tests for reversible ADAG gradient wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from circuits.tracing.grad import (
    layerwise_revert_stop_nonlinear_grad,
    layerwise_stop_nonlinear_grad,
    revert_active_stop_nonlinear_grad,
    revert_stop_nonlinear_grad,
    stop_nonlinear_grad,
)
from torch import nn


class FakeNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.variance_epsilon = 1e-6


class FakeAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(2, 2)
        self.k_proj = nn.Linear(2, 2)
        self.v_proj = nn.Linear(2, 2)
        self.o_proj = nn.Linear(2, 2)


class FakeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(2, 2)
        self.up_proj = nn.Linear(2, 2)
        self.down_proj = nn.Linear(2, 2)
        self.act_fn = nn.SiLU()


class FakeLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_layernorm = FakeNorm()
        self.post_attention_layernorm = FakeNorm()
        self.self_attn = FakeAttention(config)
        self.mlp = FakeMLP()


class FakeBackbone(nn.Module):
    def __init__(self, config, layer_count: int) -> None:
        super().__init__()
        self.norm = FakeNorm()
        self.layers = nn.ModuleList([FakeLayer(config) for _ in range(layer_count)])


class FakeModel(nn.Module):
    def __init__(self, attention_implementation: str | None, layer_count: int = 4):
        super().__init__()
        self.config = SimpleNamespace()
        if attention_implementation is not None:
            self.config._attn_implementation = attention_implementation
        self.model = FakeBackbone(self.config, layer_count)


def _requires_grad_state(model: nn.Module) -> dict[int, bool]:
    return {id(parameter): parameter.requires_grad for parameter in model.parameters()}


def test_global_stop_and_revert_restore_shared_attention_backend_and_flags() -> None:
    model = FakeModel("flash_attention_2")
    # Preserve non-default flags exactly, rather than assuming every parameter
    # originally required gradients.
    next(model.model.layers[2].mlp.up_proj.parameters()).requires_grad_(False)
    original_flags = _requires_grad_state(model)
    original_attentions = [layer.self_attn for layer in model.model.layers]

    stop_nonlinear_grad(model)
    assert model.config._attn_implementation == "noqk"
    assert all(
        layer.self_attn.attn.config is model.config for layer in model.model.layers
    )

    revert_stop_nonlinear_grad(model)
    assert model.config._attn_implementation == "flash_attention_2"
    assert [layer.self_attn for layer in model.model.layers] == original_attentions
    assert _requires_grad_state(model) == original_flags


def test_layerwise_stop_and_revert_restore_shared_backend_and_flags() -> None:
    model = FakeModel("sdpa")
    next(model.model.layers[1].mlp.down_proj.parameters()).requires_grad_(False)
    original_flags = _requires_grad_state(model)
    original_attentions = [layer.self_attn for layer in model.model.layers]

    layerwise_stop_nonlinear_grad(model, 1, 2)
    assert model.config._attn_implementation == "noqk"
    assert model.model.layers[0].self_attn is original_attentions[0]
    assert model.model.layers[3].self_attn is original_attentions[3]

    layerwise_revert_stop_nonlinear_grad(model, 1, 2)
    assert model.config._attn_implementation == "sdpa"
    assert [layer.self_attn for layer in model.model.layers] == original_attentions
    assert _requires_grad_state(model) == original_flags


@pytest.mark.parametrize(
    ("backend", "implementation"),
    [
        ("eager_causal_v1", "noqk_eager_causal_v1"),
        ("flash_sdpa_causal_v1", "noqk_flash_sdpa_causal_v1"),
    ],
)
def test_selectable_backend_is_applied_and_restored(
    backend: str, implementation: str
) -> None:
    model = FakeModel("sdpa")
    original_flags = _requires_grad_state(model)

    stop_nonlinear_grad(model, attention_backend=backend)
    assert model.config._attn_implementation == implementation
    revert_stop_nonlinear_grad(model)

    assert model.config._attn_implementation == "sdpa"
    assert _requires_grad_state(model) == original_flags
    assert not hasattr(model, "_adag_stop_gradient_model_state")


def test_global_then_layerwise_use_restores_backend_sequentially() -> None:
    model = FakeModel("eager")
    original_flags = _requires_grad_state(model)

    stop_nonlinear_grad(model)
    revert_stop_nonlinear_grad(model)
    assert model.config._attn_implementation == "eager"

    layerwise_stop_nonlinear_grad(model, 0, 3)
    assert model.config._attn_implementation == "noqk"
    layerwise_revert_stop_nonlinear_grad(model, 0, 3)
    assert model.config._attn_implementation == "eager"
    assert _requires_grad_state(model) == original_flags


def test_stop_and_revert_restore_absent_attention_backend_attribute() -> None:
    model = FakeModel(None)
    assert not hasattr(model.config, "_attn_implementation")

    stop_nonlinear_grad(model)
    assert model.config._attn_implementation == "noqk"
    revert_stop_nonlinear_grad(model)

    assert not hasattr(model.config, "_attn_implementation")


def test_partial_global_construction_rolls_back_exact_modules(monkeypatch) -> None:
    import circuits.tracing.grad as grad_module

    model = FakeModel("sdpa")
    original_norm = model.model.norm
    original_layers = [
        (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.self_attn,
            layer.mlp,
        )
        for layer in model.model.layers
    ]
    original_flags = _requires_grad_state(model)
    real_wrapper = grad_module.NoQKGradAttention
    calls = 0

    def failing_wrapper(attention):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("partial wrapper construction")
        return real_wrapper(attention)

    monkeypatch.setattr(grad_module, "NoQKGradAttention", failing_wrapper)
    with pytest.raises(RuntimeError, match="partial wrapper construction"):
        stop_nonlinear_grad(model, attention_backend="flash_sdpa_causal_v1")

    assert model.model.norm is original_norm
    assert [
        (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.self_attn,
            layer.mlp,
        )
        for layer in model.model.layers
    ] == original_layers
    assert model.config._attn_implementation == "sdpa"
    assert _requires_grad_state(model) == original_flags
    assert not hasattr(model, "_adag_stop_gradient_model_state")


def test_invalid_backend_does_not_begin_a_transaction() -> None:
    model = FakeModel("sdpa")
    original_modules = [layer.self_attn for layer in model.model.layers]
    original_flags = _requires_grad_state(model)

    with pytest.raises(ValueError, match="invalid stop-gradient attention backend"):
        stop_nonlinear_grad(model, attention_backend="auto")

    assert model.config._attn_implementation == "sdpa"
    assert [layer.self_attn for layer in model.model.layers] == original_modules
    assert _requires_grad_state(model) == original_flags
    assert not hasattr(model, "_adag_stop_gradient_model_state")


def test_revert_active_stop_gradient_restores_layerwise_transaction() -> None:
    model = FakeModel("sdpa")
    original_modules = [layer.self_attn for layer in model.model.layers]
    original_flags = _requires_grad_state(model)

    layerwise_stop_nonlinear_grad(
        model,
        0,
        1,
        attention_backend="flash_sdpa_causal_v1",
    )
    revert_active_stop_nonlinear_grad(model)

    assert model.config._attn_implementation == "sdpa"
    assert [layer.self_attn for layer in model.model.layers] == original_modules
    assert _requires_grad_state(model) == original_flags
    assert not hasattr(model, "_adag_stop_gradient_model_state")


def test_equal_layerwise_boundary_is_wrapped_once_and_restored() -> None:
    model = FakeModel("sdpa")
    original_norm = model.model.norm
    original_layer = model.model.layers[2]
    original_modules = (
        original_layer.input_layernorm,
        original_layer.post_attention_layernorm,
        original_layer.self_attn,
        original_layer.mlp,
    )

    layerwise_stop_nonlinear_grad(model, 2, 2)
    # A second wrapping attempt would fail because these wrapper types do not
    # expose the original constructor interface.
    assert model.config._attn_implementation == "noqk"
    layerwise_revert_stop_nonlinear_grad(model, 2, 2)

    assert model.model.norm is original_norm
    assert (
        original_layer.input_layernorm,
        original_layer.post_attention_layernorm,
        original_layer.self_attn,
        original_layer.mlp,
    ) == original_modules


def test_mismatched_revert_preserves_active_transaction() -> None:
    model = FakeModel("sdpa")
    layerwise_stop_nonlinear_grad(model, 1, 2)

    with pytest.raises(RuntimeError, match="does not match active"):
        revert_stop_nonlinear_grad(model)
    assert model.config._attn_implementation == "noqk"
    assert hasattr(model, "_adag_stop_gradient_model_state")

    with pytest.raises(RuntimeError, match="does not match active"):
        layerwise_revert_stop_nonlinear_grad(model, 0, 2)
    assert hasattr(model, "_adag_stop_gradient_model_state")

    layerwise_revert_stop_nonlinear_grad(model, 1, 2)
    assert model.config._attn_implementation == "sdpa"
    assert not hasattr(model, "_adag_stop_gradient_model_state")


def test_clean_revert_is_noop_and_unaffected_parameter_flags_are_not_captured() -> None:
    model = FakeModel("sdpa")
    original_norm = model.model.norm
    revert_stop_nonlinear_grad(model)
    layerwise_revert_stop_nonlinear_grad(model, 1, 2)
    assert model.model.norm is original_norm

    unaffected_parameter = next(model.model.layers[0].mlp.parameters())
    assert unaffected_parameter.requires_grad
    layerwise_stop_nonlinear_grad(model, 1, 2)
    unaffected_parameter.requires_grad_(False)
    layerwise_revert_stop_nonlinear_grad(model, 1, 2)
    assert unaffected_parameter.requires_grad is False
