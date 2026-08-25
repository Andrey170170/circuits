"""Focused tests for ordinary selected-neuron contribution VJP chunking."""

from __future__ import annotations

import gc
import weakref
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from circuits.tracing.attribution import (
    _get_neuron_attr_and_contrib,
    _get_neuron_attr_and_contrib_ig,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.contribution_execution import (
    SelectedNeuronContributionSource,
    resolve_selected_neuron_contribution_target_lane_chunk_size,
    run_selected_neuron_contribution_vjps,
)
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.tensor_receipts import raw_tensor_sha256
from torch import nn


def _ordinary_graph() -> tuple[list[SelectedNeuronContributionSource], torch.Tensor]:
    base = torch.linspace(-0.8, 0.9, 2 * 3 * 4).reshape(2, 3, 4)
    base.requires_grad_(True)
    first = base * 1.25
    second = torch.tanh(first) + first.square() * 0.1
    target_values = torch.stack(
        [
            (second * (index + 1)).sum(dim=(1, 2))
            + (first * (index - 2)).sum(dim=(1, 2))
            for index in range(5)
        ]
    )
    return (
        [
            SelectedNeuronContributionSource(
                layer=0,
                source_activation=first,
                selected_coordinates=((2, 3), (0, 1), (2, 3), (1, 0)),
            ),
            SelectedNeuronContributionSource(
                layer=1,
                source_activation=second,
                selected_coordinates=((1, 2), (0, 0), (1, 2)),
            ),
        ],
        target_values,
    )


def test_config_validates_round_trips_and_loads_legacy_state() -> None:
    config = ADAGConfig(
        device="cpu",
        selected_neuron_contribution_target_lane_chunk_size=2,
    )
    assert asdict(config)["selected_neuron_contribution_target_lane_chunk_size"] == 2
    assert resolve_selected_neuron_contribution_target_lane_chunk_size(None) is None
    assert resolve_selected_neuron_contribution_target_lane_chunk_size(3) == 3

    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.selected_neuron_contribution_target_lane_chunk_size is None

    for invalid in (0, -1, True, False, 1.5, "2"):
        with pytest.raises(ValueError, match="positive integer or None"):
            resolve_selected_neuron_contribution_target_lane_chunk_size(  # type: ignore[arg-type]
                invalid
            )
        with pytest.raises(ValueError, match="positive integer or None"):
            ADAGConfig(
                selected_neuron_contribution_target_lane_chunk_size=invalid  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 99])
def test_chunking_is_exact_and_preserves_order_batches_and_duplicates(
    chunk_size: int | None,
) -> None:
    sources, targets = _ordinary_graph()
    reference = run_selected_neuron_contribution_vjps(sources, targets)
    actual = run_selected_neuron_contribution_vjps(
        sources,
        targets,
        target_lane_chunk_size=chunk_size,
    )

    assert [tuple(result.shape) for result in actual] == [(4, 2, 5), (3, 2, 5)]
    for expected_layer, actual_layer in zip(reference, actual, strict=True):
        torch.testing.assert_close(actual_layer, expected_layer, atol=0, rtol=0)
    torch.testing.assert_close(actual[0][0], actual[0][2], atol=0, rtol=0)
    torch.testing.assert_close(actual[1][0], actual[1][2], atol=0, rtol=0)


def test_chunking_telemetry_and_receipts_keep_batch_lanes_together() -> None:
    sources, targets = _ordinary_graph()
    instrumentation = TraceInstrumentation(device="cpu")
    result = run_selected_neuron_contribution_vjps(
        sources,
        targets,
        target_lane_chunk_size=2,
        instrumentation=instrumentation,
    )

    snapshot = instrumentation.snapshot()
    first_layer = snapshot["layers"][0]
    second_layer = snapshot["layers"][1]
    counters = snapshot["counters"]
    assert first_layer["selected_neuron_contribution_raw_vjp_shape"] is None
    assert first_layer["selected_neuron_contribution_raw_vjp_chunk_shapes"] == [
        [4, 2, 3, 4],
        [4, 2, 3, 4],
        [2, 2, 3, 4],
    ]
    assert first_layer["selected_neuron_contribution_grad_outputs_chunk_shapes"] == [
        [4, 4],
        [4, 4],
        [2, 2],
    ]
    assert (
        first_layer["selected_neuron_contribution_target_lane_chunk_size_requested"]
        == 2
    )
    assert (
        first_layer["selected_neuron_contribution_target_lane_chunk_size_resolved"] == 2
    )
    assert first_layer["selected_neuron_contribution_target_lane_chunk_count"] == 3
    assert (
        first_layer["selected_neuron_contribution_max_materialized_target_lanes"] == 2
    )
    assert (
        first_layer["selected_neuron_contribution_max_materialized_autograd_lanes"] == 4
    )
    assert first_layer[
        "selected_neuron_contribution_projected_vjp_sha256"
    ] == raw_tensor_sha256(result[0])
    assert second_layer[
        "selected_neuron_contribution_projected_vjp_sha256"
    ] == raw_tensor_sha256(result[1])
    assert counters["selected_neuron_contribution_vjp_chunk_executions"] == 6
    assert counters["selected_neuron_contribution_max_materialized_target_lanes"] == 2
    assert counters["selected_neuron_contribution_max_materialized_autograd_lanes"] == 4


def test_repeated_execution_receipts_are_indexed_instead_of_overwritten() -> None:
    instrumentation = TraceInstrumentation(device="cpu")
    for execution_index in (0, 1):
        sources, targets = _ordinary_graph()
        run_selected_neuron_contribution_vjps(
            sources,
            targets,
            target_lane_chunk_size=2,
            instrumentation=instrumentation,
            execution_index=execution_index,
        )

    layer = instrumentation.snapshot()["layers"][0]
    assert layer["selected_neuron_contribution_receipt_mode"] == "execution_indexed"
    assert "selected_neuron_contribution_projected_vjp_sha256" not in layer
    receipts = layer["selected_neuron_contribution_execution_receipts"]
    assert [receipt["execution_index"] for receipt in receipts] == [0, 1]
    assert all(receipt["projected_vjp_shape"] == [4, 2, 5] for receipt in receipts)
    assert all(len(receipt["projected_vjp_sha256"]) == 64 for receipt in receipts)


def test_dense_raw_chunk_dies_before_next_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, targets = _ordinary_graph()
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
    result = run_selected_neuron_contribution_vjps(
        sources,
        targets,
        target_lane_chunk_size=1,
    )

    gc.collect()
    assert len(raw_chunk_refs) == 10
    assert raw_chunk_refs[-1]() is None
    assert [tuple(layer.shape) for layer in result] == [(4, 2, 5), (3, 2, 5)]


def test_every_chunk_retains_the_shared_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    sources, targets = _ordinary_graph()
    original_grad = torch.autograd.grad
    retain_graph_calls: list[bool] = []

    def record_grad(*args, **kwargs):
        retain_graph_calls.append(bool(kwargs["retain_graph"]))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    run_selected_neuron_contribution_vjps(
        sources,
        targets,
        target_lane_chunk_size=2,
    )

    assert retain_graph_calls == [True] * 6


def test_default_path_reuses_embedding_identity_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, targets = _ordinary_graph()
    full_grad_outputs = torch.eye(10)

    def unexpected_eye(*_args, **_kwargs):
        raise AssertionError("ordinary default path allocated a second identity")

    monkeypatch.setattr(torch, "eye", unexpected_eye)
    result = run_selected_neuron_contribution_vjps(
        sources,
        targets,
        full_grad_outputs=full_grad_outputs,
    )
    assert [tuple(layer.shape) for layer in result] == [(4, 2, 5), (3, 2, 5)]


class _ToyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(4, 4, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _ToyMlp()


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(12, 4)
        self.layers = nn.ModuleList([_ToyLayer()])


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyBackbone()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def forward(self, *, inputs_embeds, attention_mask):
        del attention_mask
        hidden = torch.tanh(self.model.layers[0].mlp.down_proj(inputs_embeds))
        return SimpleNamespace(logits=self.lm_head(hidden))


def _run_toy(model: _ToyModel, *, chunk_size: int | None, ig: bool):
    kwargs = {
        "model": model,
        "neuron_cfg": {0: [[2, 3], [0, 1], [2, 3]]},
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "src_tokens": [0, 2],
        "tgt_tokens": [1, 2],
        "focus_positions": [1, 2],
        "focus_logits": [[2, 3], [2, 3]],
        "attention_masks": torch.ones(2, 3),
        "neuron_chunk_size": 2,
        "contribution_target_lane_chunk_size": chunk_size,
    }
    if ig:
        return _get_neuron_attr_and_contrib_ig(**kwargs, ig_steps=2)
    return _get_neuron_attr_and_contrib(**kwargs)


@pytest.mark.parametrize("ig", [False, True])
def test_direct_and_integrated_gradient_paths_forward_chunking_exactly(
    ig: bool,
) -> None:
    torch.manual_seed(73)
    reference_model = _ToyModel()
    candidate_model = _ToyModel()
    candidate_model.load_state_dict(reference_model.state_dict())

    reference = _run_toy(reference_model, chunk_size=None, ig=ig)
    candidate = _run_toy(candidate_model, chunk_size=1, ig=ig)

    for expected, actual in zip(reference[:3], candidate[:3], strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert candidate[3] == reference[3]
