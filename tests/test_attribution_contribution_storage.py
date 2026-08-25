"""Focused tests for compact selected-neuron contribution storage."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import circuits.tracing.attribution as attribution_module
import pytest
import torch
from circuits.tracing.attribution import (
    _copy_selected_neuron_contributions,
    _get_neuron_attr_and_contrib_with_stop_grad_on_mlps,
    _nonempty_neuron_layers,
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

    def forward(self, *, inputs_embeds, attention_mask):
        del attention_mask
        hidden = self.model.layers[0].mlp.down_proj(inputs_embeds)
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_stop_gradient_cuda_stage_partition_preserves_outputs_and_order(
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict]] = []

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

    def run(model, instrumentation):
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
    result = run(candidate_model, object())

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
