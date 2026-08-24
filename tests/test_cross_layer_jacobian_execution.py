"""Focused qualification of cross-layer Jacobian execution strategies."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import cast

import pytest
import torch
from circuits.tracing import cross_layer_jacobian_execution as execution_module
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.cross_layer_jacobian_execution import (
    CrossLayerJacobianExecution,
    CrossLayerJacobianPair,
    CrossLayerJacobianPairResult,
    CrossLayerJacobianPreparation,
    prepare_cross_layer_jacobian_execution,
    resolve_cross_layer_jacobian_execution,
)
from circuits.tracing.instrumentation import TraceInstrumentation
from transformers import Qwen3Config, Qwen3ForCausalLM


def _model(dtype: torch.dtype = torch.float32) -> Qwen3ForCausalLM:
    torch.manual_seed(19)
    config = Qwen3Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).to(dtype=dtype).eval()


def _preparation(
    model: Qwen3ForCausalLM,
    *,
    instrumentation: TraceInstrumentation | None = None,
    use_relp_grad: bool = True,
    use_stop_grad_on_mlps: bool = True,
) -> CrossLayerJacobianPreparation:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    return CrossLayerJacobianPreparation(
        model=model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        source_layers=(0, 1, 2, 3),
        use_relp_grad=use_relp_grad,
        disable_stop_grad=False,
        use_stop_grad_on_mlps=use_stop_grad_on_mlps,
        device="cpu",
        attention_backend="eager_causal_v1",
        instrumentation=instrumentation,
    )


def _pair(src_layer: int = 0, tgt_layer: int = 4) -> CrossLayerJacobianPair:
    return CrossLayerJacobianPair(
        src_layer=src_layer,
        tgt_layer=tgt_layer,
        # Deliberately non-sorted, with an exact duplicate.
        src_neurons=((2, 3), (0, 1), (2, 3)),
        tgt_neurons=((3, 4), (1, 2), (2, 5)),
        tgt_chunk_size=2,
    )


def _mutable_state(model: Qwen3ForCausalLM) -> tuple:
    parameter_flags = tuple(parameter.requires_grad for parameter in model.parameters())
    hook_counts = tuple(
        (
            len(module._forward_pre_hooks),
            len(module._forward_hooks),
        )
        for module in model.modules()
    )
    return (
        model.config._attn_implementation,
        parameter_flags,
        hook_counts,
        hasattr(model, "_adag_stop_gradient_model_state"),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_range_matches_full_model_exactly_with_streamed_chunks(
    dtype: torch.dtype,
) -> None:
    model = _model(dtype)
    preparation = _preparation(model)
    pair = _pair()
    state_before = _mutable_state(model)

    reference = prepare_cross_layer_jacobian_execution(
        preparation, execution="full_model_v1"
    ).compute_pair(pair)
    candidate = prepare_cross_layer_jacobian_execution(
        preparation, execution="cached_range_v1"
    ).compute_pair(pair)

    assert isinstance(reference, CrossLayerJacobianPairResult)
    assert isinstance(candidate, CrossLayerJacobianPairResult)
    # Receipt equality is an independent gate over pre-normalization values.
    assert candidate.receipts.ordered() == reference.receipts.ordered()
    torch.testing.assert_close(
        candidate.relative_attribution,
        reference.relative_attribution,
        atol=0,
        rtol=0,
    )
    assert candidate.relative_attribution.shape == (1, 3, 3)
    torch.testing.assert_close(
        candidate.relative_attribution[:, 0],
        candidate.relative_attribution[:, 2],
        atol=0,
        rtol=0,
    )
    assert _mutable_state(model) == state_before


def test_preparation_uses_resolved_stop_gradient_backend_and_restores_ordinary_one() -> (
    None
):
    model = _model()
    model.config._attn_implementation = "sdpa"
    observed_implementations: list[str] = []
    handle = model.model.layers[0].self_attn.register_forward_pre_hook(
        lambda module, _inputs: observed_implementations.append(
            module.config._attn_implementation
        )
    )
    try:
        prepare_cross_layer_jacobian_execution(
            _preparation(model), execution="cached_range_v1"
        )
    finally:
        handle.remove()

    assert observed_implementations == ["noqk_eager_causal_v1"]
    assert model.config._attn_implementation == "sdpa"


def test_range_replay_skips_prefix_and_suffix_decoder_layers() -> None:
    model = _model()
    executor = prepare_cross_layer_jacobian_execution(
        _preparation(model), execution="cached_range_v1"
    )
    calls = [0] * len(model.model.layers)
    handles = []
    for layer_index, layer_module in enumerate(model.model.layers):
        handles.append(
            layer_module.register_forward_pre_hook(
                lambda _module, _inputs, layer_index=layer_index: calls.__setitem__(
                    layer_index, calls[layer_index] + 1
                )
            )
        )
    try:
        executor.compute_pair(_pair(src_layer=1, tgt_layer=3))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == [0, 1, 1, 1, 0]


def test_telemetry_records_preparation_ranges_and_vjp_chunks() -> None:
    model = _model()
    instrumentation = TraceInstrumentation(device="cpu")
    executor = prepare_cross_layer_jacobian_execution(
        _preparation(model, instrumentation=instrumentation),
        execution="cached_range_v1",
    )
    executor.compute_pair(_pair(src_layer=1, tgt_layer=3))

    counters = instrumentation.snapshot()["counters"]
    assert counters["cross_layer_jacobian_execution"] == "cached_range_v1"
    assert counters["cross_layer_preparation_forward_count"] == 1
    assert counters["cross_layer_preparation_cache_bytes"] > 0
    assert counters["cross_layer_full_decoder_layer_executions"] == 5
    assert counters["cross_layer_replay_decoder_layer_entries"] == 3
    assert counters["cross_layer_vjp_chunk_executions"] == 2


def test_preparation_failure_removes_owned_hooks_and_restores_state() -> None:
    model = _model()
    state_before = _mutable_state(model)
    handle = model.model.layers[2].register_forward_pre_hook(
        lambda _module, _inputs: (_ for _ in ()).throw(
            RuntimeError("synthetic preparation failure")
        )
    )
    try:
        with pytest.raises(RuntimeError, match="synthetic preparation failure"):
            prepare_cross_layer_jacobian_execution(
                _preparation(model), execution="cached_range_v1"
            )
    finally:
        handle.remove()

    assert _mutable_state(model) == state_before


def test_replay_failure_removes_owned_hooks_and_restores_state(monkeypatch) -> None:
    model = _model()
    executor = prepare_cross_layer_jacobian_execution(
        _preparation(model), execution="cached_range_v1"
    )
    state_before = _mutable_state(model)

    def fail_replay(*_args, **_kwargs):
        raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(model.model.layers[2], "forward", fail_replay)
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        executor.compute_pair(_pair(src_layer=1, tgt_layer=3))

    assert _mutable_state(model) == state_before


def test_vjp_failure_removes_owned_hooks_and_restores_state(monkeypatch) -> None:
    model = _model()
    executor = prepare_cross_layer_jacobian_execution(
        _preparation(model), execution="cached_range_v1"
    )
    state_before = _mutable_state(model)

    def fail_vjp(*_args, **_kwargs):
        raise RuntimeError("synthetic VJP failure")

    monkeypatch.setattr(execution_module.torch.autograd, "grad", fail_vjp)
    with pytest.raises(RuntimeError, match="synthetic VJP failure"):
        executor.compute_pair(_pair(src_layer=1, tgt_layer=3))

    assert _mutable_state(model) == state_before


@pytest.mark.parametrize("execution", ["full_model_v1", "cached_range_v1"])
def test_execution_preserves_unrelated_down_projection_hooks(
    execution: CrossLayerJacobianExecution,
) -> None:
    model = _model()
    calls = 0

    def unrelated_hook(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.model.layers[1].mlp.down_proj.register_forward_hook(unrelated_hook)
    state_with_hook = _mutable_state(model)
    try:
        executor = prepare_cross_layer_jacobian_execution(
            _preparation(model), execution=execution
        )
        executor.compute_pair(_pair(src_layer=1, tgt_layer=3))
        assert calls > 0
        assert _mutable_state(model) == state_with_hook
    finally:
        handle.remove()


def test_cached_range_rejects_ig_disabled_stop_grad_and_invalid_strategy() -> None:
    model = _model()
    with pytest.raises(ValueError, match="stop-gradient tracing"):
        prepare_cross_layer_jacobian_execution(
            replace(_preparation(model), disable_stop_grad=True),
            execution="cached_range_v1",
        )
    executor = prepare_cross_layer_jacobian_execution(
        _preparation(model), execution="cached_range_v1"
    )
    with pytest.raises(ValueError, match="integrated gradients"):
        executor.compute_pair(replace(_pair(), alpha=0.5))
    with pytest.raises(ValueError, match="invalid cross-layer Jacobian execution"):
        resolve_cross_layer_jacobian_execution("auto")
    with pytest.raises(ValueError, match="invalid cross-layer Jacobian execution"):
        ADAGConfig(
            cross_layer_jacobian_execution=cast(CrossLayerJacobianExecution, "auto")
        )


def test_adag_config_serializes_strategy_and_restores_legacy_default() -> None:
    config = ADAGConfig(device="cpu", cross_layer_jacobian_execution="cached_range_v1")
    assert asdict(config)["cross_layer_jacobian_execution"] == "cached_range_v1"

    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.cross_layer_jacobian_execution == "full_model_v1"
