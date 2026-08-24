"""Focused tests for stop-gradient contribution execution strategies."""

from __future__ import annotations

import gc
import weakref
from dataclasses import asdict
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.contribution_execution import (
    StopGradientContributionExecution,
    resolve_stop_gradient_contribution_execution,
    resolve_stop_gradient_contribution_target_lane_chunk_size,
    run_stop_gradient_contribution_forward,
    run_stop_gradient_contribution_vjp,
)
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.sparse_source_injection import sparse_source_injection
from circuits.tracing.tensor_receipts import raw_tensor_sha256
from torch import nn


class _FakeMLP(nn.Module):
    def __init__(self, hidden: int, *, down_bias: bool) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.down_proj = nn.Linear(hidden * 2, hidden, bias=down_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.tanh(self.up_proj(inputs)))


class _NestedMLP(nn.Module):
    """Match the ``mlp.mlp.down_proj`` shape of stop-gradient wrappers."""

    def __init__(self, mlp: _FakeMLP) -> None:
        super().__init__()
        self.mlp = mlp
        self.down_proj = mlp.down_proj

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class _FakeLayer(nn.Module):
    def __init__(self, hidden: int, *, nested: bool, down_bias: bool) -> None:
        super().__init__()
        mlp = _FakeMLP(hidden, down_bias=down_bias)
        self.mlp = _NestedMLP(mlp) if nested else mlp

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.mlp(inputs)


class _FakeBackbone(nn.Module):
    def __init__(self, *, nested: bool, down_bias: bool) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(13, 4)
        self.layers = nn.ModuleList(
            [
                _FakeLayer(4, nested=nested, down_bias=down_bias),
                _FakeLayer(4, nested=nested, down_bias=down_bias),
            ]
        )


class _FakeModel(nn.Module):
    def __init__(self, *, nested: bool = False, down_bias: bool = True) -> None:
        super().__init__()
        self.model = _FakeBackbone(nested=nested, down_bias=down_bias)
        self.lm_head = nn.Linear(4, 7, bias=False)
        self.fail_after_layers = False

    def forward(self, *, inputs_embeds, attention_mask=None):
        del attention_mask
        hidden = inputs_embeds
        for layer in self.model.layers:
            hidden = layer(hidden)
        if self.fail_after_layers:
            raise RuntimeError("synthetic forward failure")
        return SimpleNamespace(logits=self.lm_head(hidden))


class _FailingEmbedding(nn.Module):
    def forward(self, _input_ids):
        raise RuntimeError("synthetic embedding failure")


def _down_projection(model: _FakeModel, layer: int = 0) -> nn.Module:
    mlp = model.model.layers[layer].mlp
    return mlp.mlp.down_proj if hasattr(mlp, "mlp") else mlp.down_proj


def _five_targets(logits: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            logits[:, -1, 0],
            logits[:, -2, 1],
            logits[:, 0, 2],
            logits[:, 1, 3],
            logits[:, -1, 4],
        ]
    )


@pytest.mark.parametrize("nested", [False, True])
def test_source_leaf_matches_full_graph_values_and_source_gradients(
    nested: bool,
) -> None:
    torch.manual_seed(17)
    model = _FakeModel(nested=nested)
    input_ids = torch.tensor([[1, 2, 3]])

    full = run_stop_gradient_contribution_forward(
        model,
        input_ids,
        None,
        layer=0,
        execution="full_graph_v1",
    )
    full_gradient = torch.autograd.grad(
        full.logits[:, -1, :].sum(),
        full.source_activation,
        retain_graph=full.retain_graph_for_vjp,
    )[0]
    source_leaf = run_stop_gradient_contribution_forward(
        model,
        input_ids,
        None,
        layer=0,
        execution="source_leaf_v1",
    )
    source_leaf_gradient = torch.autograd.grad(
        source_leaf.logits[:, -1, :].sum(),
        source_leaf.source_activation,
        retain_graph=source_leaf.retain_graph_for_vjp,
    )[0]

    torch.testing.assert_close(source_leaf.logits, full.logits, atol=0, rtol=0)
    torch.testing.assert_close(
        source_leaf.source_activation,
        full.source_activation,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(source_leaf_gradient, full_gradient, atol=0, rtol=0)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert full.source_activation.grad_fn is not None
    assert full.retain_graph_for_vjp is True
    assert source_leaf.source_activation.is_leaf
    assert source_leaf.source_activation.requires_grad
    assert source_leaf.source_activation.grad_fn is None
    assert source_leaf.retain_graph_for_vjp is False


@pytest.mark.parametrize(
    "execution", ["full_graph_v1", "source_leaf_v1", "sparse_source_leaf_v1"]
)
def test_execution_hook_is_removed_after_success(execution: str) -> None:
    model = _FakeModel(nested=True)
    down_projection = _down_projection(model)
    hooks_before = (
        len(down_projection._forward_pre_hooks),
        len(down_projection._forward_hooks),
    )

    run_stop_gradient_contribution_forward(
        model,
        torch.tensor([[1, 2]]),
        None,
        layer=0,
        execution=cast(StopGradientContributionExecution, execution),
        selected_coordinates=[[0, 0]],
    )

    assert (
        len(down_projection._forward_pre_hooks),
        len(down_projection._forward_hooks),
    ) == hooks_before


def test_source_leaf_restores_mixed_parameter_gradient_flags() -> None:
    model = _FakeModel(nested=True)
    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    flags_before = [parameter.requires_grad for parameter in parameters]

    run_stop_gradient_contribution_forward(
        model,
        torch.tensor([[1, 2]]),
        None,
        layer=0,
        execution="source_leaf_v1",
    )

    assert [parameter.requires_grad for parameter in parameters] == flags_before


@pytest.mark.parametrize(
    "execution", ["full_graph_v1", "source_leaf_v1", "sparse_source_leaf_v1"]
)
def test_execution_hook_is_removed_when_forward_fails(execution: str) -> None:
    model = _FakeModel(nested=True)
    model.fail_after_layers = True
    down_projection = _down_projection(model)
    hooks_before = (
        len(down_projection._forward_pre_hooks),
        len(down_projection._forward_hooks),
    )

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        run_stop_gradient_contribution_forward(
            model,
            torch.tensor([[1, 2]]),
            None,
            layer=0,
            execution=cast(StopGradientContributionExecution, execution),
            selected_coordinates=[[0, 0]],
        )

    assert (
        len(down_projection._forward_pre_hooks),
        len(down_projection._forward_hooks),
    ) == hooks_before
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(
    "execution", ["full_graph_v1", "source_leaf_v1", "sparse_source_leaf_v1"]
)
def test_execution_hook_is_removed_when_embedding_fails(execution: str) -> None:
    model = _FakeModel(nested=True)
    model.model.embed_tokens = _FailingEmbedding()
    down_projection = _down_projection(model)
    hooks_before = (
        len(down_projection._forward_pre_hooks),
        len(down_projection._forward_hooks),
    )

    with pytest.raises(RuntimeError, match="synthetic embedding failure"):
        run_stop_gradient_contribution_forward(
            model,
            torch.tensor([[1, 2]]),
            None,
            layer=0,
            execution=cast(StopGradientContributionExecution, execution),
            selected_coordinates=[[0, 0]],
        )

    assert (
        len(down_projection._forward_pre_hooks),
        len(down_projection._forward_hooks),
    ) == hooks_before
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_execution_is_validated_and_restored_in_adag_config() -> None:
    config = ADAGConfig(
        device="cpu",
        stop_gradient_contribution_execution="source_leaf_v1",
    )
    assert asdict(config)["stop_gradient_contribution_execution"] == "source_leaf_v1"
    assert (
        resolve_stop_gradient_contribution_execution("full_graph_v1") == "full_graph_v1"
    )
    sparse_config = ADAGConfig(
        device="cpu",
        stop_gradient_contribution_execution="sparse_source_leaf_v1",
    )
    assert (
        asdict(sparse_config)["stop_gradient_contribution_execution"]
        == "sparse_source_leaf_v1"
    )

    with pytest.raises(
        ValueError, match="invalid stop-gradient contribution execution"
    ):
        resolve_stop_gradient_contribution_execution("auto")
    with pytest.raises(
        ValueError, match="invalid stop-gradient contribution execution"
    ):
        ADAGConfig(
            stop_gradient_contribution_execution=cast(
                StopGradientContributionExecution, "auto"
            )
        )

    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.stop_gradient_contribution_execution == "full_graph_v1"
    assert restored.stop_gradient_contribution_target_lane_chunk_size is None


def test_target_lane_chunk_size_is_validated_and_round_trips_in_config() -> None:
    config = ADAGConfig(
        device="cpu",
        stop_gradient_contribution_target_lane_chunk_size=2,
    )
    assert asdict(config)["stop_gradient_contribution_target_lane_chunk_size"] == 2
    assert resolve_stop_gradient_contribution_target_lane_chunk_size(None) is None
    assert resolve_stop_gradient_contribution_target_lane_chunk_size(3) == 3

    for invalid in (0, -1, True, False, 1.5, "2"):
        with pytest.raises(ValueError, match="positive integer or None"):
            resolve_stop_gradient_contribution_target_lane_chunk_size(  # type: ignore[arg-type]
                invalid
            )
        with pytest.raises(ValueError, match="positive integer or None"):
            ADAGConfig(
                stop_gradient_contribution_target_lane_chunk_size=invalid  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    "execution", ["full_graph_v1", "source_leaf_v1", "sparse_source_leaf_v1"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [1, 2, 99])
def test_target_lane_chunking_is_exact_and_preserves_canonical_order(
    execution: StopGradientContributionExecution,
    dtype: torch.dtype,
    chunk_size: int,
) -> None:
    torch.manual_seed(101)
    model = _FakeModel(nested=True).to(dtype=dtype)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    coordinates = [[2, 6], [0, 1], [2, 6], [1, 4]]

    reference = run_stop_gradient_contribution_forward(
        model,
        input_ids,
        None,
        layer=0,
        execution=execution,
        selected_coordinates=coordinates,
    )
    reference_vjp = run_stop_gradient_contribution_vjp(
        reference, _five_targets(reference.logits), layer=0
    )
    chunked = run_stop_gradient_contribution_forward(
        model,
        input_ids,
        None,
        layer=0,
        execution=execution,
        selected_coordinates=coordinates,
    )
    chunked_vjp = run_stop_gradient_contribution_vjp(
        chunked,
        _five_targets(chunked.logits),
        layer=0,
        target_lane_chunk_size=chunk_size,
    )

    torch.testing.assert_close(chunked.logits, reference.logits, atol=0, rtol=0)
    torch.testing.assert_close(chunked_vjp, reference_vjp, atol=0, rtol=0)
    assert chunked_vjp.shape == (len(coordinates), 2, 5)
    torch.testing.assert_close(chunked_vjp[0], chunked_vjp[2], atol=0, rtol=0)


def test_target_lane_chunking_telemetry_keeps_all_batch_lanes_together() -> None:
    model = _FakeModel(nested=True)
    instrumentation = TraceInstrumentation(device="cpu")
    coordinates = [[0, 1], [2, 6], [0, 1]]
    forward = run_stop_gradient_contribution_forward(
        model,
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        None,
        layer=0,
        execution="sparse_source_leaf_v1",
        selected_coordinates=coordinates,
        instrumentation=instrumentation,
    )
    result = run_stop_gradient_contribution_vjp(
        forward,
        _five_targets(forward.logits),
        layer=0,
        target_lane_chunk_size=2,
        instrumentation=instrumentation,
    )

    snapshot = instrumentation.snapshot()
    layer = snapshot["layers"][0]
    counters = snapshot["counters"]
    assert layer["stop_gradient_contribution_raw_vjp_shape"] is None
    assert layer["stop_gradient_contribution_raw_vjp_chunk_shapes"] == [
        [4, 2, 3],
        [4, 2, 3],
        [2, 2, 3],
    ]
    assert layer["stop_gradient_contribution_grad_outputs_shape"] is None
    assert layer["stop_gradient_contribution_max_grad_outputs_shape"] == [4, 4]
    assert layer["stop_gradient_contribution_target_lane_count"] == 5
    assert layer["stop_gradient_contribution_target_lane_chunk_size_requested"] == 2
    assert layer["stop_gradient_contribution_target_lane_chunk_size_resolved"] == 2
    assert layer["stop_gradient_contribution_target_lane_chunk_count"] == 3
    assert layer["stop_gradient_contribution_max_materialized_target_lanes"] == 2
    assert layer["stop_gradient_contribution_max_materialized_autograd_lanes"] == 4
    assert counters["stop_gradient_contribution_vjp_chunk_executions"] == 3
    assert counters["stop_gradient_contribution_max_materialized_target_lanes"] == 2
    assert counters["stop_gradient_contribution_max_materialized_autograd_lanes"] == 4
    assert result.shape == (3, 2, 5)
    assert layer["stop_gradient_contribution_projected_vjp_sha256"] == (
        raw_tensor_sha256(result)
    )


def test_dense_raw_vjp_chunk_dies_before_next_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel(nested=True)
    forward = run_stop_gradient_contribution_forward(
        model,
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        None,
        layer=0,
        execution="source_leaf_v1",
        selected_coordinates=[[0, 1], [2, 6]],
    )
    original_grad = torch.autograd.grad
    raw_chunk_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def observe_chunk_lifetime(*args, **kwargs):
        if raw_chunk_refs:
            gc.collect()
            assert raw_chunk_refs[-1]() is None
        result = original_grad(*args, **kwargs)
        raw_chunk_refs.append(weakref.ref(result[0]))
        return result

    monkeypatch.setattr(torch.autograd, "grad", observe_chunk_lifetime)
    result = run_stop_gradient_contribution_vjp(
        forward,
        torch.stack(
            [
                forward.logits[:, -1, 0],
                forward.logits[:, -2, 1],
                forward.logits[:, 0, 2],
            ]
        ),
        layer=0,
        target_lane_chunk_size=1,
    )

    gc.collect()
    assert len(raw_chunk_refs) == 3
    assert raw_chunk_refs[-1]() is None
    assert result.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("execution", "expected_retain_graph"),
    [
        ("full_graph_v1", [True, True, True]),
        ("source_leaf_v1", [True, True, False]),
        ("sparse_source_leaf_v1", [True, True, False]),
    ],
)
def test_target_lane_chunks_preserve_final_graph_lifetime_contract(
    monkeypatch: pytest.MonkeyPatch,
    execution: StopGradientContributionExecution,
    expected_retain_graph: list[bool],
) -> None:
    model = _FakeModel(nested=True)
    coordinates = [[0, 1], [2, 6]]
    forward = run_stop_gradient_contribution_forward(
        model,
        torch.tensor([[1, 2, 3]]),
        None,
        layer=0,
        execution=execution,
        selected_coordinates=coordinates,
    )
    original_grad = torch.autograd.grad
    observed_retain_graph: list[bool] = []

    def record_grad(*args, **kwargs):
        observed_retain_graph.append(bool(kwargs["retain_graph"]))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    result = run_stop_gradient_contribution_vjp(
        forward,
        torch.stack(
            [
                forward.logits[:, -1, 0],
                forward.logits[:, -2, 1],
                forward.logits[:, 0, 2],
            ]
        ),
        layer=0,
        target_lane_chunk_size=1,
    )

    assert result.shape == (2, 1, 3)
    assert observed_retain_graph == expected_retain_graph


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("down_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_source_leaf_matches_dense_selected_vjp_with_duplicates_and_batches(
    nested: bool,
    down_bias: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(41)
    model = _FakeModel(nested=nested, down_bias=down_bias).to(dtype=dtype)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    coordinates = [[0, 1], [2, 6], [0, 1], [1, 4]]

    dense = run_stop_gradient_contribution_forward(
        model,
        input_ids,
        None,
        layer=0,
        execution="source_leaf_v1",
        selected_coordinates=coordinates,
    )
    dense_targets = torch.stack([dense.logits[:, -1, 2], dense.logits[:, -2, 5]])
    dense_selected_vjp = run_stop_gradient_contribution_vjp(
        dense, dense_targets, layer=0
    )

    sparse = run_stop_gradient_contribution_forward(
        model,
        input_ids,
        None,
        layer=0,
        execution="sparse_source_leaf_v1",
        selected_coordinates=coordinates,
    )
    sparse_targets = torch.stack([sparse.logits[:, -1, 2], sparse.logits[:, -2, 5]])
    sparse_selected_vjp = run_stop_gradient_contribution_vjp(
        sparse, sparse_targets, layer=0
    )

    absolute_tolerance = 2e-8 if dtype == torch.float32 else 2e-2
    relative_tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(sparse.logits, dense.logits, atol=0, rtol=0)
    torch.testing.assert_close(
        sparse_selected_vjp,
        dense_selected_vjp,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    )
    positions = torch.tensor([position for position, _neuron in coordinates])
    neurons = torch.tensor([neuron for _position, neuron in coordinates])
    torch.testing.assert_close(
        sparse.source_activation,
        dense.source_activation[:, positions, neurons],
        atol=0,
        rtol=0,
    )
    assert sparse.source_activation.shape == (2, len(coordinates))
    assert sparse.source_activation.is_leaf
    assert sparse.source_activation.untyped_storage().nbytes() == (
        sparse.source_activation.numel() * sparse.source_activation.element_size()
    )
    assert sparse_selected_vjp.shape == (len(coordinates), 2, 2)
    assert sparse_selected_vjp.untyped_storage().nbytes() == (
        sparse_selected_vjp.numel() * sparse_selected_vjp.element_size()
    )
    torch.testing.assert_close(
        sparse_selected_vjp[0], sparse_selected_vjp[2], atol=0, rtol=0
    )
    assert sparse.selected_coordinates == tuple(map(tuple, coordinates))
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_sparse_source_leaf_records_compact_vjp_evidence() -> None:
    model = _FakeModel(nested=True)
    instrumentation = TraceInstrumentation(device="cpu")
    coordinates = [[0, 1], [2, 6], [0, 1]]
    forward = run_stop_gradient_contribution_forward(
        model,
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        None,
        layer=0,
        execution="sparse_source_leaf_v1",
        selected_coordinates=coordinates,
        instrumentation=instrumentation,
    )
    targets = torch.stack([forward.logits[:, -1, 0], forward.logits[:, -2, 1]])
    result = run_stop_gradient_contribution_vjp(
        forward, targets, layer=0, instrumentation=instrumentation
    )

    snapshot = instrumentation.snapshot()
    layer = snapshot["layers"][0]
    counters = snapshot["counters"]
    assert layer["stop_gradient_contribution_source_representation"] == (
        "selected_coordinates"
    )
    assert layer["stop_gradient_contribution_dense_source_shape"] == [2, 3, 8]
    assert layer["stop_gradient_contribution_differentiated_source_shape"] == [2, 3]
    assert layer["stop_gradient_contribution_raw_vjp_shape"] == [4, 2, 3]
    assert layer["stop_gradient_contribution_projected_vjp_shape"] == [3, 2, 2]
    assert layer["stop_gradient_contribution_dense_vjp_result_materialized"] is False
    assert counters["stop_gradient_sparse_source_coordinate_count"] == 3
    assert counters["stop_gradient_sparse_vjp_result_numel"] == result.numel() * 2
    assert counters["stop_gradient_sparse_dense_vjp_result_numel_avoided"] == (
        (4 * 2 * 3 * 8) - (result.numel() * 2)
    )


def test_sparse_source_leaf_preserves_unrelated_hooks() -> None:
    model = _FakeModel(nested=True)
    projection = _down_projection(model)
    calls: list[torch.Size] = []

    def observe(_module, _inputs, output):
        calls.append(output.shape)

    unrelated = projection.register_forward_hook(observe)
    hooks_before = tuple(projection._forward_hooks)
    try:
        run_stop_gradient_contribution_forward(
            model,
            torch.tensor([[1, 2]]),
            None,
            layer=0,
            execution="sparse_source_leaf_v1",
            selected_coordinates=[[0, 1]],
        )
        assert tuple(projection._forward_hooks) == hooks_before
        assert calls == [torch.Size([1, 2, 4])]
    finally:
        unrelated.remove()


def test_sparse_source_leaf_keeps_output_transforming_hook_downstream() -> None:
    torch.manual_seed(73)
    model = _FakeModel(nested=True)
    projection = _down_projection(model)
    calls = 0

    def transform(_module, _inputs, output):
        nonlocal calls
        calls += 1
        return output * 1.75 + 0.25

    unrelated = projection.register_forward_hook(transform)
    coordinates = [[0, 1], [2, 6], [0, 1]]
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    try:
        dense = run_stop_gradient_contribution_forward(
            model,
            input_ids,
            None,
            layer=0,
            execution="source_leaf_v1",
            selected_coordinates=coordinates,
        )
        dense_targets = torch.stack([dense.logits[:, -1, 2]])
        dense_vjp = run_stop_gradient_contribution_vjp(dense, dense_targets, layer=0)

        sparse = run_stop_gradient_contribution_forward(
            model,
            input_ids,
            None,
            layer=0,
            execution="sparse_source_leaf_v1",
            selected_coordinates=coordinates,
        )
        sparse_targets = torch.stack([sparse.logits[:, -1, 2]])
        sparse_vjp = run_stop_gradient_contribution_vjp(sparse, sparse_targets, layer=0)

        torch.testing.assert_close(sparse.logits, dense.logits, atol=0, rtol=0)
        torch.testing.assert_close(sparse_vjp, dense_vjp, atol=2e-8, rtol=1e-5)
        assert calls == 2
        assert unrelated.id in projection._forward_hooks
    finally:
        unrelated.remove()


def test_sparse_source_leaf_restores_flags_and_hook_after_injection_failure() -> None:
    model = _FakeModel(nested=True)
    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    flags_before = [parameter.requires_grad for parameter in parameters]
    projection = _down_projection(model)
    hooks_before = tuple(projection._forward_hooks)

    with pytest.raises(IndexError, match="out of bounds"):
        run_stop_gradient_contribution_forward(
            model,
            torch.tensor([[1, 2]]),
            None,
            layer=0,
            execution="sparse_source_leaf_v1",
            selected_coordinates=[[99, 1]],
        )

    assert tuple(projection._forward_hooks) == hooks_before
    assert [parameter.requires_grad for parameter in parameters] == flags_before


def test_sparse_source_leaf_rejects_empty_and_nonfinite_selected_algebra() -> None:
    model = _FakeModel()
    with pytest.raises(ValueError, match="requires selected coordinates"):
        run_stop_gradient_contribution_forward(
            model,
            torch.tensor([[1, 2]]),
            None,
            layer=0,
            execution="sparse_source_leaf_v1",
        )

    projection = _down_projection(model)
    with torch.no_grad():
        projection.weight[:, 1] = torch.inf
    with pytest.raises(ValueError, match="selected weight columns"):
        run_stop_gradient_contribution_forward(
            model,
            torch.tensor([[1, 2]]),
            None,
            layer=0,
            execution="sparse_source_leaf_v1",
            selected_coordinates=[[0, 1]],
        )

    finite_projection = nn.Linear(3, 2)
    nonfinite_source = torch.ones(1, 2, 3)
    nonfinite_source[:, 0, 1] = torch.nan
    with (
        pytest.raises(ValueError, match="selected values"),
        sparse_source_injection(finite_projection, [[0, 1]]),
    ):
        finite_projection(nonfinite_source)


def test_sparse_injection_rejects_a_second_projection_execution_and_cleans_up() -> None:
    projection = nn.Linear(3, 2)
    hooks_before = tuple(projection._forward_hooks)

    def execute_twice() -> None:
        with sparse_source_injection(projection, [[0, 1]]):
            projection(torch.ones(1, 2, 3))
            projection(torch.ones(1, 2, 3))

    with pytest.raises(RuntimeError, match="more than once"):
        execute_twice()
    assert tuple(projection._forward_hooks) == hooks_before


def test_sparse_vjp_failure_leaves_no_execution_hook_or_parameter_mutation() -> None:
    model = _FakeModel(nested=True)
    projection = _down_projection(model)
    hooks_before = tuple(projection._forward_hooks)
    flags_before = [parameter.requires_grad for parameter in model.parameters()]
    forward = run_stop_gradient_contribution_forward(
        model,
        torch.tensor([[1, 2]]),
        None,
        layer=0,
        execution="sparse_source_leaf_v1",
        selected_coordinates=[[0, 1]],
    )
    unrelated_targets = torch.ones(1, 1, requires_grad=True)

    with pytest.raises(RuntimeError, match="not have been used"):
        run_stop_gradient_contribution_vjp(forward, unrelated_targets, layer=0)

    assert tuple(projection._forward_hooks) == hooks_before
    assert [parameter.requires_grad for parameter in model.parameters()] == flags_before


def test_target_lane_chunk_failure_leaves_no_hook_or_parameter_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel(nested=True)
    projection = _down_projection(model)
    hooks_before = tuple(projection._forward_hooks)
    flags_before = [parameter.requires_grad for parameter in model.parameters()]
    forward = run_stop_gradient_contribution_forward(
        model,
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        None,
        layer=0,
        execution="sparse_source_leaf_v1",
        selected_coordinates=[[0, 1], [2, 6]],
    )
    original_grad = torch.autograd.grad
    calls = 0

    def fail_second_chunk(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic chunk VJP failure")
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", fail_second_chunk)
    with pytest.raises(RuntimeError, match="synthetic chunk VJP failure"):
        run_stop_gradient_contribution_vjp(
            forward,
            torch.stack(
                [
                    forward.logits[:, -1, 0],
                    forward.logits[:, -2, 1],
                    forward.logits[:, 0, 2],
                ]
            ),
            layer=0,
            target_lane_chunk_size=1,
        )

    assert calls == 2
    assert tuple(projection._forward_hooks) == hooks_before
    assert [parameter.requires_grad for parameter in model.parameters()] == flags_before
