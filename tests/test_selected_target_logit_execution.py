"""Focused tests for ordinary selected target-logit execution."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.selected_target_logit_execution import (
    resolve_selected_target_logit_execution,
    run_selected_target_logits,
)
from torch import nn


class _ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(17, 5)
        self.layers = nn.ModuleList([nn.Linear(5, 5, bias=False)])
        self.norm = nn.LayerNorm(5)

    def forward(self, *, inputs_embeds, attention_mask):
        del attention_mask
        hidden = torch.tanh(self.layers[0](inputs_embeds))
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyDecoder()
        self.lm_head = nn.Linear(5, 11, bias=False)
        self.received_logits_to_keep: list[list[int] | None] = []

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        logits_to_keep=None,
    ):
        decoder_output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden = decoder_output.last_hidden_state
        if logits_to_keep is None:
            self.received_logits_to_keep.append(None)
        else:
            self.received_logits_to_keep.append(logits_to_keep.tolist())
            hidden = hidden[:, logits_to_keep, :]
        return SimpleNamespace(logits=self.lm_head(hidden))


def _execute(
    model: _ToyCausalLM,
    embeddings: torch.Tensor,
    strategy: str,
    *,
    center_logits: bool,
    instrumentation: TraceInstrumentation | None = None,
):
    return run_selected_target_logits(
        model,
        embeddings,
        torch.ones(2, 4),
        [3, 1, 3],
        [[2, 8, 4], [7, 1, 7]],
        execution=strategy,  # type: ignore[arg-type]
        center_logits=center_logits,
        instrumentation=instrumentation,
    )


def _historical_target_logits(
    model: _ToyCausalLM,
    embeddings: torch.Tensor,
    *,
    center_logits: bool,
) -> torch.Tensor:
    output = model(inputs_embeds=embeddings, attention_mask=torch.ones(2, 4))
    logits = output.logits
    if center_logits:
        logits -= logits.mean(dim=-1)
    focus_positions = [3, 1, 3]
    focus_logits = [[2, 8, 4], [7, 1, 7]]
    target_nodes = []
    for target_index, position in enumerate(focus_positions):
        token_ids = [row[target_index] for row in focus_logits]
        target_nodes.append(logits[torch.arange(logits.shape[0]), position, token_ids])
    return torch.stack(target_nodes)


def test_adapters_match_independent_historical_values_gradients_order_and_duplicates() -> (
    None
):
    torch.manual_seed(109)
    historical_model = _ToyCausalLM()
    reference_model = _ToyCausalLM()
    candidate_model = _ToyCausalLM()
    reference_model.load_state_dict(historical_model.state_dict())
    candidate_model.load_state_dict(historical_model.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    historical_embeddings = (
        historical_model.model.embed_tokens(input_ids).detach().requires_grad_()
    )
    reference_embeddings = (
        reference_model.model.embed_tokens(input_ids).detach().requires_grad_()
    )
    candidate_embeddings = (
        candidate_model.model.embed_tokens(input_ids).detach().requires_grad_()
    )

    historical = _historical_target_logits(
        historical_model,
        historical_embeddings,
        center_logits=False,
    )
    reference = _execute(
        reference_model,
        reference_embeddings,
        "full_logits_v1",
        center_logits=False,
    )
    candidate = _execute(
        candidate_model,
        candidate_embeddings,
        "selected_position_logits_v1",
        center_logits=False,
    )
    historical_grad = torch.autograd.grad(historical.sum(), historical_embeddings)[0]
    reference_grad = torch.autograd.grad(
        reference.target_logits.sum(), reference_embeddings
    )[0]
    candidate_grad = torch.autograd.grad(
        candidate.target_logits.sum(), candidate_embeddings
    )[0]

    assert historical_model.received_logits_to_keep == [None]
    assert reference_model.received_logits_to_keep == [None]
    assert candidate_model.received_logits_to_keep == [[3, 1, 3]]
    assert reference.target_logit_shape == candidate.target_logit_shape == (3, 2)
    assert reference.unique_selected_position_count == 2
    assert candidate.unique_selected_position_count == 2
    torch.testing.assert_close(reference.target_logits, historical, atol=0, rtol=0)
    torch.testing.assert_close(candidate.target_logits, historical, atol=0, rtol=0)
    torch.testing.assert_close(reference_grad, historical_grad, atol=0, rtol=0)
    torch.testing.assert_close(candidate_grad, historical_grad, atol=0, rtol=0)
    torch.testing.assert_close(
        candidate.target_logits[0, 1],
        candidate.target_logits[2, 1],
        atol=0,
        rtol=0,
    )


def test_full_logits_preserves_historical_centered_shape_failure() -> None:
    torch.manual_seed(127)
    historical_model = _ToyCausalLM()
    adapter_model = _ToyCausalLM()
    adapter_model.load_state_dict(historical_model.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    historical_embeddings = historical_model.model.embed_tokens(input_ids).detach()
    adapter_embeddings = adapter_model.model.embed_tokens(input_ids).detach()

    with pytest.raises(RuntimeError, match="size of tensor"):
        _historical_target_logits(
            historical_model,
            historical_embeddings.requires_grad_(),
            center_logits=True,
        )
    with pytest.raises(RuntimeError, match="size of tensor"):
        _execute(
            adapter_model,
            adapter_embeddings.requires_grad_(),
            "full_logits_v1",
            center_logits=True,
        )


def test_selected_position_centering_fails_closed_in_config_and_direct_execution() -> (
    None
):
    with pytest.raises(ValueError, match="do not support center_logits=true"):
        ADAGConfig(
            center_logits=True,
            selected_target_logit_execution="selected_position_logits_v1",
        )
    model = _ToyCausalLM()
    embeddings = model.model.embed_tokens(
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    ).detach()
    with pytest.raises(ValueError, match="do not support center_logits=true"):
        _execute(
            model,
            embeddings.requires_grad_(),
            "selected_position_logits_v1",
            center_logits=True,
        )


def test_receipts_prove_full_vs_selected_lm_head_position_rows() -> None:
    torch.manual_seed(113)
    model = _ToyCausalLM()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    instrumentation = TraceInstrumentation(device="cpu")

    for execution_index, strategy in enumerate(
        ("full_logits_v1", "selected_position_logits_v1")
    ):
        embeddings = model.model.embed_tokens(input_ids).detach().requires_grad_()
        result = _execute(
            model,
            embeddings,
            strategy,
            center_logits=False,
            instrumentation=instrumentation,
        )
        assert result.lm_head_input_shape[1] == (4 if execution_index == 0 else 3)

    snapshot = instrumentation.snapshot()
    records = snapshot["execution_records"]["selected_target_logit_execution"]
    assert [record["execution"] for record in records] == [
        "full_logits_v1",
        "selected_position_logits_v1",
    ]
    assert [record["lm_head_input_shape"] for record in records] == [
        [2, 4, 5],
        [2, 3, 5],
    ]
    assert [record["lm_head_output_shape"] for record in records] == [
        [2, 4, 11],
        [2, 3, 11],
    ]
    assert [record["full_sequence_logits_materialized"] for record in records] == [
        True,
        False,
    ]
    counters = snapshot["counters"]
    assert counters["selected_target_logit_execution_count"] == 2
    assert counters["selected_target_logit_full_logits_v1_execution_count"] == 1
    assert (
        counters["selected_target_logit_selected_position_logits_v1_execution_count"]
        == 1
    )
    assert counters["selected_target_logit_lm_head_position_rows"] == 14


def test_config_validates_round_trips_and_loads_legacy_state() -> None:
    config = ADAGConfig(
        device="cpu",
        selected_target_logit_execution="selected_position_logits_v1",
    )
    assert asdict(config)["selected_target_logit_execution"] == (
        "selected_position_logits_v1"
    )
    assert resolve_selected_target_logit_execution("full_logits_v1") == (
        "full_logits_v1"
    )
    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.selected_target_logit_execution == "full_logits_v1"
    with pytest.raises(ValueError, match="invalid selected target-logit execution"):
        ADAGConfig(
            selected_target_logit_execution="selected"  # type: ignore[arg-type]
        )


def test_candidate_fails_closed_when_tensor_logits_to_keep_is_unsupported() -> None:
    class Unsupported(_ToyCausalLM):
        def forward(self, *, inputs_embeds, attention_mask):
            return super().forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

    model = Unsupported()
    embeddings = model.model.embed_tokens(torch.tensor([[1, 2], [3, 4]])).detach()
    with pytest.raises(RuntimeError, match="supports tensor logits_to_keep"):
        run_selected_target_logits(
            model,
            embeddings.requires_grad_(),
            torch.ones(2, 2),
            [1],
            [[2], [3]],
            execution="selected_position_logits_v1",
        )
