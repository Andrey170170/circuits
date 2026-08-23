"""Focused tests for stop-gradient contribution execution strategies."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.contribution_execution import (
    StopGradientContributionExecution,
    resolve_stop_gradient_contribution_execution,
    run_stop_gradient_contribution_forward,
)
from torch import nn


class _FakeMLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.down_proj = nn.Linear(hidden * 2, hidden, bias=False)

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
    def __init__(self, hidden: int, *, nested: bool) -> None:
        super().__init__()
        mlp = _FakeMLP(hidden)
        self.mlp = _NestedMLP(mlp) if nested else mlp

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.mlp(inputs)


class _FakeBackbone(nn.Module):
    def __init__(self, *, nested: bool) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(13, 4)
        self.layers = nn.ModuleList(
            [_FakeLayer(4, nested=nested), _FakeLayer(4, nested=nested)]
        )


class _FakeModel(nn.Module):
    def __init__(self, *, nested: bool = False) -> None:
        super().__init__()
        self.model = _FakeBackbone(nested=nested)
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


@pytest.mark.parametrize("execution", ["full_graph_v1", "source_leaf_v1"])
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


@pytest.mark.parametrize("execution", ["full_graph_v1", "source_leaf_v1"])
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
        )

    assert (
        len(down_projection._forward_pre_hooks),
        len(down_projection._forward_hooks),
    ) == hooks_before
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("execution", ["full_graph_v1", "source_leaf_v1"])
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
