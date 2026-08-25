"""Focused tests for stop-gradient embedding contribution VJP chunking."""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace

import circuits.tracing.attribution as attribution_module
import circuits.tracing.contribution_execution as contribution_execution_module
import pytest
import torch
from circuits.tracing.attribution import (
    _get_neuron_attr_and_contrib_with_stop_grad_on_mlps,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.contribution_execution import (
    resolve_stop_gradient_embed_contribution_target_lane_chunk_size,
    run_stop_gradient_embed_contribution_vjp,
)
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.tensor_receipts import raw_tensor_sha256
from torch import nn


def _embedding_graph() -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = torch.linspace(-0.8, 0.9, 2 * 4 * 3).reshape(2, 4, 3)
    embeddings.requires_grad_(True)
    hidden = torch.tanh(embeddings) + embeddings.square() * 0.2
    targets = torch.stack(
        [
            (hidden * (index + 1)).sum(dim=(1, 2))
            + (embeddings * (index - 2)).sum(dim=(1, 2))
            for index in range(5)
        ]
    )
    return embeddings, targets


def _legacy_direct_contribution(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    source_tokens: list[int],
) -> torch.Tensor:
    target_count, batch = targets.shape
    raw_vjp = torch.autograd.grad(
        targets.flatten(),
        embeddings,
        grad_outputs=torch.eye(target_count * batch),
        is_grads_batched=True,
        retain_graph=False,
    )[0]
    dense_vjp = (
        raw_vjp.reshape(
            target_count,
            batch,
            batch,
            raw_vjp.shape[-2],
            raw_vjp.shape[-1],
        )
        .diagonal(dim1=1, dim2=2)
        .permute(0, 3, 1, 2)
    )
    selected = dense_vjp[:, :, source_tokens, :]
    return (selected * embeddings[None, :, source_tokens, :]).sum(-1).permute(2, 1, 0)


def test_config_validates_round_trips_and_loads_legacy_state() -> None:
    config = ADAGConfig(
        device="cpu",
        stop_gradient_embed_contribution_target_lane_chunk_size=2,
    )
    assert (
        asdict(config)["stop_gradient_embed_contribution_target_lane_chunk_size"] == 2
    )
    assert resolve_stop_gradient_embed_contribution_target_lane_chunk_size(None) is None
    assert resolve_stop_gradient_embed_contribution_target_lane_chunk_size(3) == 3

    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.stop_gradient_embed_contribution_target_lane_chunk_size is None

    for invalid in (0, -1, True, False, 1.5, "2"):
        with pytest.raises(ValueError, match="positive integer or None"):
            resolve_stop_gradient_embed_contribution_target_lane_chunk_size(  # type: ignore[arg-type]
                invalid
            )
        with pytest.raises(ValueError, match="positive integer or None"):
            ADAGConfig(
                stop_gradient_embed_contribution_target_lane_chunk_size=invalid  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 99])
def test_chunking_matches_legacy_and_preserves_order_batches_and_duplicates(
    chunk_size: int | None,
) -> None:
    source_tokens = [3, 0, 3, 1]
    expected_embeddings, expected_targets = _embedding_graph()
    expected = _legacy_direct_contribution(
        expected_embeddings, expected_targets, source_tokens
    )
    embeddings, targets = _embedding_graph()
    actual = run_stop_gradient_embed_contribution_vjp(
        embeddings,
        targets,
        source_tokens,
        target_lane_chunk_size=chunk_size,
    )

    assert actual.shape == (4, 2, 5)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[0], actual[2], atol=0, rtol=0)


def test_raw_chunks_die_and_only_the_final_chunk_releases_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings, targets = _embedding_graph()
    original_grad = torch.autograd.grad
    raw_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    retain_graph_calls: list[bool] = []

    def observe(*args, **kwargs):
        if raw_refs:
            gc.collect()
            assert raw_refs[-1]() is None
        retain_graph_calls.append(bool(kwargs["retain_graph"]))
        result = original_grad(*args, **kwargs)
        raw_refs.append(weakref.ref(result[0]))
        return result

    monkeypatch.setattr(torch.autograd, "grad", observe)
    run_stop_gradient_embed_contribution_vjp(
        embeddings,
        targets,
        [3, 0, 3],
        target_lane_chunk_size=1,
    )

    gc.collect()
    assert len(raw_refs) == 5
    assert raw_refs[-1]() is None
    assert retain_graph_calls == [True, True, True, True, False]


def test_telemetry_has_independent_stage_receipt_and_counter_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings, targets = _embedding_graph()
    instrumentation = TraceInstrumentation(device="cpu")
    stage_metadata: dict[str, object] = {}

    @contextmanager
    def capture_stage(_instrumentation, name, *, metadata):
        assert _instrumentation is instrumentation
        assert name == "stop_grad_embed_contribution_vjp"
        stage_metadata.update(metadata)
        yield SimpleNamespace(metadata=stage_metadata)

    monkeypatch.setattr(
        contribution_execution_module,
        "cuda_memory_instrumentation_stage",
        capture_stage,
    )
    projected = run_stop_gradient_embed_contribution_vjp(
        embeddings,
        targets,
        [3, 0, 3],
        target_lane_chunk_size=2,
        instrumentation=instrumentation,
    )

    snapshot = instrumentation.snapshot()
    records = snapshot["execution_records"]["stop_gradient_embed_contribution_vjp"]
    assert len(records) == 1
    record = records[0]
    assert record["canonical_result_order"] == "source_batch_target"
    assert record["source_tokens"] == [3, 0, 3]
    assert record["raw_vjp_shape"] is None
    assert record["raw_vjp_chunk_shapes"] == [
        [4, 2, 4, 3],
        [4, 2, 4, 3],
        [2, 2, 4, 3],
    ]
    assert record["grad_outputs_chunk_shapes"] == [[4, 4], [4, 4], [2, 2]]
    assert record["projected_vjp_shape"] == [3, 2, 5]
    assert record["projected_vjp_sha256"] == raw_tensor_sha256(projected)
    assert record["target_lane_chunk_size_requested"] == 2
    assert record["target_lane_chunk_size_resolved"] == 2
    assert record["target_lane_chunk_count"] == 3
    assert record["retain_graph_after_execution"] is False
    assert (
        snapshot["counters"]["stop_gradient_embed_contribution_vjp_chunk_executions"]
        == 3
    )
    assert stage_metadata["retain_graph"] is False
    assert stage_metadata["retain_graph_after_execution"] is False
    assert stage_metadata["raw_vjp_chunk_shapes"] == record["raw_vjp_chunk_shapes"]
    assert stage_metadata["projected_vjp_result_shape"] == [3, 2, 5]


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


def test_stop_gradient_attribution_plumbs_embed_width_and_preserves_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        embeddings = model.model.embed_tokens(input_ids).detach().requires_grad_()
        output = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        return SimpleNamespace(
            logits=output.logits,
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
    original_embed = attribution_module.run_stop_gradient_embed_contribution_vjp
    observed_widths: list[int | None] = []

    def capture_embed(*args, **kwargs):
        observed_widths.append(kwargs["target_lane_chunk_size"])
        return original_embed(*args, **kwargs)

    monkeypatch.setattr(
        attribution_module,
        "run_stop_gradient_embed_contribution_vjp",
        capture_embed,
    )

    torch.manual_seed(31)
    reference_model = _ToyModel()
    candidate_model = _ToyModel()
    candidate_model.load_state_dict(reference_model.state_dict())
    common = {
        "neuron_cfg": {0: [[0, 0], [1, 1]]},
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "src_tokens": [1, 0, 1],
        "tgt_tokens": [0, 1],
        "focus_positions": [0, 1],
        "focus_logits": [[2, 3], [2, 3]],
        "attention_masks": torch.ones(2, 2),
        "neuron_chunk_size": 1,
    }
    reference = _get_neuron_attr_and_contrib_with_stop_grad_on_mlps(
        reference_model,
        **common,
    )
    candidate = _get_neuron_attr_and_contrib_with_stop_grad_on_mlps(
        candidate_model,
        **common,
        embed_contribution_target_lane_chunk_size=1,
    )

    assert observed_widths == [None, 1]
    for expected, actual in zip(reference[:3], candidate[:3], strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert candidate[3] == reference[3]
