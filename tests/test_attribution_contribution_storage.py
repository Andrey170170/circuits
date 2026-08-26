"""Focused tests for compact selected-neuron contribution storage."""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import circuits.tracing.attribution as attribution_module
import circuits.tracing.contribution_execution as contribution_execution_module
import pytest
import torch
from circuits.tracing.attribution import (
    _copy_selected_neuron_contributions,
    _get_neuron_attr_and_contrib_with_stop_grad_on_mlps,
    _nonempty_neuron_layers,
    _project_selected_attribution_vjp,
)
from torch import nn


def test_selected_contribution_copy_preserves_pair_order_shape_and_values() -> None:
    dense = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    pairs = [[3, 4], [0, 1], [2, 0], [0, 3]]
    expected = torch.stack(
        [dense[:, :, position, neuron] for position, neuron in pairs]
    ).permute(0, 2, 1)

    actual = _copy_selected_neuron_contributions(dense, pairs)

    assert actual.shape == (len(pairs), 3, 2)
    assert torch.equal(actual, expected)


def test_selected_contribution_copy_does_not_retain_dense_storage() -> None:
    dense = torch.arange(2 * 2 * 8 * 16, dtype=torch.float32).reshape(2, 2, 8, 16)
    selected = _copy_selected_neuron_contributions(dense, [[7, 15], [0, 0]])
    before = selected.clone()

    assert selected.untyped_storage().data_ptr() != dense.untyped_storage().data_ptr()
    assert (
        selected.untyped_storage().nbytes()
        == selected.numel() * selected.element_size()
    )

    dense.fill_(-1)
    assert torch.equal(selected, before)


def test_nonempty_neuron_layers_skips_empty_layers_and_preserves_order() -> None:
    config = {
        9: [],
        4: [[3, 7], [1, 2]],
        12: [],
        2: [[0, 5]],
    }

    assert list(_nonempty_neuron_layers(config)) == [
        (4, [[3, 7], [1, 2]]),
        (2, [[0, 5]]),
    ]


def test_selected_contribution_copy_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="empty neuron selection"):
        _copy_selected_neuron_contributions(torch.zeros(1, 1, 1, 1), [])


@pytest.mark.parametrize("return_gradient_only", [False, True])
def test_selected_attribution_projection_is_exact_compact_and_terminal(
    return_gradient_only: bool,
) -> None:
    embeddings = torch.linspace(-0.7, 0.8, 2 * 4 * 3).reshape(2, 4, 3)
    embeddings.requires_grad_(True)
    raw_vjp = torch.linspace(-1.1, 1.2, 3 * 2 * 2 * 4 * 3).reshape(3 * 2, 2, 4, 3)
    raw_vjp.requires_grad_(True)
    raw_ref = weakref.ref(raw_vjp)
    raw_data_ptr = raw_vjp.untyped_storage().data_ptr()
    source_tokens = [3, 0, 3, 1]

    dense_vjp = (
        raw_vjp.reshape(3, 2, 2, 4, 3).diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
    )
    if return_gradient_only:
        expected = dense_vjp[:, :, source_tokens, :].detach().clone()
    else:
        expected = (dense_vjp * embeddings.detach()[None, ...]).sum(-1)
        expected = expected[:, :, source_tokens].detach().clone()

    actual = _project_selected_attribution_vjp(
        raw_vjp,
        embeddings,
        source_tokens,
        return_gradient_only=return_gradient_only,
    )
    del dense_vjp, raw_vjp

    assert actual.requires_grad is False
    assert actual.grad_fn is None
    assert actual.untyped_storage().data_ptr() != raw_data_ptr
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(actual[:, :, 0], actual[:, :, 2])
    del expected
    assert raw_ref() is None


@pytest.mark.parametrize(
    ("raw_shape", "embedding_shape", "message"),
    [
        ((2, 3, 4), (2, 3, 4), "must have shape"),
        ((4, 2, 3, 4), (1, 3, 4), "does not match"),
        ((5, 2, 3, 4), (2, 3, 4), "not divisible"),
    ],
)
def test_selected_attribution_projection_rejects_malformed_shapes(
    raw_shape: tuple[int, ...],
    embedding_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _project_selected_attribution_vjp(
            torch.zeros(raw_shape),
            torch.zeros(embedding_shape),
            [0],
            return_gradient_only=False,
        )


class _ToyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(3, 3, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _ToyMlp()


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 3)
        self.layers = nn.ModuleList([_ToyLayer()])


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyBackbone()
        self.lm_head = nn.Linear(3, 5, bias=False)
        self.logit_refs = []

    def forward(self, *, inputs_embeds, attention_mask):
        del attention_mask
        hidden = self.model.layers[0].mlp.down_proj(inputs_embeds)
        logits = self.lm_head(hidden)
        self.logit_refs.append(weakref.ref(logits))
        return SimpleNamespace(logits=logits)


def test_selected_capture_hook_is_removed_when_forward_fails(monkeypatch) -> None:
    model = _ToyModel()
    monkeypatch.setattr(
        attribution_module, "revert_stop_nonlinear_grad", lambda current: current
    )
    monkeypatch.setattr(
        attribution_module,
        "layerwise_stop_nonlinear_grad",
        lambda current, *_args, **_kwargs: current,
    )

    def fail_after_capture(*, inputs_embeds, attention_mask):
        del attention_mask
        model.model.layers[0].mlp.down_proj(inputs_embeds)
        raise RuntimeError("forced selected forward failure")

    monkeypatch.setattr(model, "forward", fail_after_capture)
    with pytest.raises(RuntimeError, match="forced selected forward failure"):
        _get_neuron_attr_and_contrib_with_stop_grad_on_mlps(
            model,
            neuron_cfg={0: [[0, 0]]},
            input_ids=torch.tensor([[1, 2]]),
            src_tokens=[0],
            tgt_tokens=[1],
            focus_positions=[1],
            focus_logits=[[2]],
            attention_masks=torch.ones(1, 2),
        )

    assert len(model.model.layers[0].mlp.down_proj._forward_hooks) == 0


def test_stop_gradient_cuda_stage_partition_preserves_outputs_and_order(
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict]] = []
    retain_graph_calls: list[bool | None] = []
    selected_hook_counts: list[int] = []
    output_liveness: list[bool] = []
    original_autograd_grad = torch.autograd.grad
    active_model = None

    @contextmanager
    def capture_stage(_instrumentation, name, *, metadata):
        if _instrumentation is None:
            yield None
            return
        copied_metadata = dict(metadata)
        calls.append((name, copied_metadata))
        yield SimpleNamespace(metadata=copied_metadata)

    monkeypatch.setattr(
        attribution_module, "cuda_memory_instrumentation_stage", capture_stage
    )
    monkeypatch.setattr(
        attribution_module, "cuda_memory_observation_stage", capture_stage
    )
    monkeypatch.setattr(
        contribution_execution_module,
        "cuda_memory_instrumentation_stage",
        capture_stage,
    )
    monkeypatch.setattr(
        attribution_module, "revert_stop_nonlinear_grad", lambda model: model
    )
    monkeypatch.setattr(
        attribution_module,
        "layerwise_stop_nonlinear_grad",
        lambda model, *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        attribution_module,
        "layerwise_revert_stop_nonlinear_grad",
        lambda model, *_args, **_kwargs: model,
    )

    def fake_contribution_forward(
        model,
        input_ids,
        attention_mask,
        *,
        layer,
        execution,
        selected_coordinates,
        instrumentation,
    ):
        del layer, execution, instrumentation
        embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()
        output = model(inputs_embeds=embeds, attention_mask=attention_mask)
        return SimpleNamespace(
            logits=output.logits,
            source_activation=embeds,
            source_representation="dense",
            selected_coordinates=tuple(map(tuple, selected_coordinates)),
        )

    def fake_contribution_vjp(
        contribution_forward,
        target_values,
        *,
        layer,
        target_lane_chunk_size,
        instrumentation,
    ):
        del layer, target_lane_chunk_size, instrumentation
        return torch.zeros(
            len(contribution_forward.selected_coordinates),
            target_values.shape[1],
            target_values.shape[0],
        )

    monkeypatch.setattr(
        attribution_module,
        "run_stop_gradient_contribution_forward",
        fake_contribution_forward,
    )
    monkeypatch.setattr(
        attribution_module,
        "run_stop_gradient_contribution_vjp",
        fake_contribution_vjp,
    )

    def capture_autograd_grad(*args, **kwargs):
        retain_graph_calls.append(kwargs.get("retain_graph"))
        selected_hook_counts.append(
            len(active_model.model.layers[0].mlp.down_proj._forward_hooks)
        )
        output_liveness.append(active_model.logit_refs[-1]() is not None)
        return original_autograd_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", capture_autograd_grad)

    def run(model, instrumentation):
        nonlocal active_model
        active_model = model
        return _get_neuron_attr_and_contrib_with_stop_grad_on_mlps(
            model,
            neuron_cfg={0: [[0, 0], [1, 1]]},
            input_ids=torch.tensor([[1, 2]]),
            src_tokens=[0, 1],
            tgt_tokens=[1],
            focus_positions=[1],
            focus_logits=[[2]],
            attention_masks=torch.ones(1, 2),
            neuron_chunk_size=1,
            instrumentation=instrumentation,
        )

    torch.manual_seed(7)
    baseline_model = _ToyModel()
    candidate_model = _ToyModel()
    candidate_model.load_state_dict(baseline_model.state_dict())
    baseline = run(baseline_model, None)
    assert calls == []
    instrumentation = SimpleNamespace(
        cuda_memory_telemetry=True,
        append_execution_record=lambda *_args, **_kwargs: None,
        increment_counter=lambda *_args, **_kwargs: None,
        set_counter=lambda *_args, **_kwargs: None,
    )
    result = run(candidate_model, instrumentation)

    attr, contrib, embed_contrib, tags = result
    assert attr.shape == (2, 1, 2)
    assert contrib.shape == (2, 1, 1)
    assert embed_contrib.shape == (2, 1, 1)
    assert len(tags) == 2
    for baseline_tensor, candidate_tensor in zip(baseline[:3], result[:3], strict=True):
        assert torch.equal(baseline_tensor, candidate_tensor)
    assert baseline[3] == tags

    names = [name for name, _metadata in calls]
    assert names == [
        "stop_grad_selected_layer_forward",
        "stop_grad_selected_chunk_prepare",
        "stop_grad_selected_attribution_vjp",
        "stop_grad_selected_chunk_projection",
        "stop_grad_selected_chunk_prepare",
        "stop_grad_selected_attribution_vjp",
        "stop_grad_selected_chunk_projection",
        "stop_grad_selected_layer_release",
        "stop_grad_selected_phase_finalize",
        "stop_grad_embed_forward",
        "stop_grad_embed_contribution_vjp",
        "stop_grad_embed_projection_release",
        "stop_grad_neuron_layer_forward",
        "stop_grad_neuron_layer_release",
        "stop_grad_neuron_phase_finalize",
    ]
    selected_forward = calls[0][1]
    assert selected_forward["layer"] == 0
    assert selected_forward["selected_neuron_count"] == 2
    assert selected_forward["planned_chunk_count"] == 2
    assert selected_forward["activation_shape"] == [1, 2, 3]
    assert calls[-1][1]["contribution_shape"] == [2, 1, 1]
    assert retain_graph_calls == [True, False, False, True, False, False]
    assert selected_hook_counts == [0, 0, 0, 0, 0, 0]
    assert output_liveness == [False, False, True, False, False, True]
