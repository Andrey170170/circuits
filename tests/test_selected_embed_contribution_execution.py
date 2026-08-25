"""Focused tests for ordinary selected-embedding contribution VJP chunking."""

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
    _get_neuron_attr_and_contrib,
    _get_neuron_attr_and_contrib_ig,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.contribution_execution import (
    resolve_selected_embed_contribution_target_lane_chunk_size,
    run_selected_embed_contribution_vjp,
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


def test_config_validates_round_trips_and_loads_legacy_state() -> None:
    config = ADAGConfig(
        device="cpu",
        selected_embed_contribution_target_lane_chunk_size=2,
    )
    assert asdict(config)["selected_embed_contribution_target_lane_chunk_size"] == 2
    assert resolve_selected_embed_contribution_target_lane_chunk_size(None) is None
    assert resolve_selected_embed_contribution_target_lane_chunk_size(3) == 3

    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.selected_embed_contribution_target_lane_chunk_size is None

    for invalid in (0, -1, True, False, 1.5, "2"):
        with pytest.raises(ValueError, match="positive integer or None"):
            resolve_selected_embed_contribution_target_lane_chunk_size(  # type: ignore[arg-type]
                invalid
            )
        with pytest.raises(ValueError, match="positive integer or None"):
            ADAGConfig(
                selected_embed_contribution_target_lane_chunk_size=invalid  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("return_gradient_only", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 1, 2, 99])
def test_chunking_is_exact_for_batches_reordered_and_duplicate_sources(
    return_gradient_only: bool,
    chunk_size: int | None,
) -> None:
    embeddings, targets = _embedding_graph()
    source_tokens = [3, 0, 3, 1]
    reference = run_selected_embed_contribution_vjp(
        embeddings,
        targets,
        source_tokens,
        return_gradient_only=return_gradient_only,
    )
    actual = run_selected_embed_contribution_vjp(
        embeddings,
        targets,
        source_tokens,
        return_gradient_only=return_gradient_only,
        target_lane_chunk_size=chunk_size,
    )

    expected_shape = (5, 2, 4, 3) if return_gradient_only else (4, 2, 5)
    assert tuple(actual.shape) == expected_shape
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    duplicate_axis = 2 if return_gradient_only else 0
    torch.testing.assert_close(
        actual.select(duplicate_axis, 0),
        actual.select(duplicate_axis, 2),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("return_gradient_only", [False, True])
def test_raw_chunk_dies_and_every_chunk_retains_graph(
    monkeypatch: pytest.MonkeyPatch,
    return_gradient_only: bool,
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
    run_selected_embed_contribution_vjp(
        embeddings,
        targets,
        [3, 0, 3],
        return_gradient_only=return_gradient_only,
        target_lane_chunk_size=1,
    )

    gc.collect()
    assert len(raw_refs) == 5
    assert raw_refs[-1]() is None
    assert retain_graph_calls == [True] * 5


def test_full_identity_is_reused_and_rejected_for_real_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings, targets = _embedding_graph()
    full_grad_outputs = torch.eye(10)

    def unexpected_eye(*_args, **_kwargs):
        raise AssertionError("embedding compatibility path allocated another identity")

    monkeypatch.setattr(torch, "eye", unexpected_eye)
    result = run_selected_embed_contribution_vjp(
        embeddings,
        targets,
        [0, 2],
        return_gradient_only=False,
        full_grad_outputs=full_grad_outputs,
    )
    assert result.shape == (2, 2, 5)

    with pytest.raises(ValueError, match="only be reused by an unchunked"):
        run_selected_embed_contribution_vjp(
            embeddings,
            targets,
            [0],
            return_gradient_only=False,
            target_lane_chunk_size=1,
            full_grad_outputs=full_grad_outputs,
        )


def test_telemetry_receipts_are_top_level_and_execution_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumentation = TraceInstrumentation(device="cpu")
    expected_receipts = []
    stage_metadata: list[dict[str, object]] = []

    @contextmanager
    def capture_stage(_instrumentation, _name, *, metadata):
        stage_metadata.append(metadata)
        yield SimpleNamespace(metadata=metadata)

    monkeypatch.setattr(
        contribution_execution_module,
        "cuda_memory_instrumentation_stage",
        capture_stage,
    )
    for execution_index in (0, 1):
        embeddings, targets = _embedding_graph()
        projected = run_selected_embed_contribution_vjp(
            embeddings,
            targets,
            [3, 0, 3],
            return_gradient_only=True,
            target_lane_chunk_size=2,
            instrumentation=instrumentation,
            execution_index=execution_index,
        )
        expected_receipts.append(raw_tensor_sha256(projected))

    snapshot = instrumentation.snapshot()
    records = snapshot["execution_records"]["selected_embed_contribution_vjp"]
    assert [record["execution_index"] for record in records] == [0, 1]
    assert [record["projected_vjp_sha256"] for record in records] == expected_receipts
    assert all(record["receipt_mode"] == "execution_indexed" for record in records)
    assert all(record["return_gradient_only"] is True for record in records)
    assert all(record["projected_vjp_shape"] == [5, 2, 3, 3] for record in records)
    assert all(record["raw_vjp_shape"] is None for record in records)
    assert all(
        record["raw_vjp_chunk_shapes"] == [[4, 2, 4, 3], [4, 2, 4, 3], [2, 2, 4, 3]]
        for record in records
    )
    assert all(record["target_lane_chunk_size_requested"] == 2 for record in records)
    assert snapshot["counters"]["selected_embed_contribution_vjp_chunk_executions"] == 6
    assert len(stage_metadata) == 2
    assert all(metadata["retain_graph"] is True for metadata in stage_metadata)
    assert all(
        metadata["retain_graph_after_execution"] is True for metadata in stage_metadata
    )


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


def _run_toy(
    model: _ToyModel,
    *,
    embed_width: int | None,
    neuron_width: int | None,
    ig: bool,
):
    kwargs = {
        "model": model,
        "neuron_cfg": {0: [[2, 3], [0, 1], [2, 3]]},
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "src_tokens": [2, 0, 2],
        "tgt_tokens": [1, 2],
        "focus_positions": [1, 2],
        "focus_logits": [[2, 3], [2, 3]],
        "attention_masks": torch.ones(2, 3),
        "neuron_chunk_size": 2,
        "embed_contribution_target_lane_chunk_size": embed_width,
        "contribution_target_lane_chunk_size": neuron_width,
    }
    if ig:
        return _get_neuron_attr_and_contrib_ig(**kwargs, ig_steps=2)
    return _get_neuron_attr_and_contrib(**kwargs)


@pytest.mark.parametrize("ig", [False, True])
@pytest.mark.parametrize(
    ("embed_width", "neuron_width"), [(1, None), (None, 1), (1, 1)]
)
def test_end_to_end_widths_are_independent_and_exact(
    ig: bool,
    embed_width: int | None,
    neuron_width: int | None,
) -> None:
    torch.manual_seed(73)
    reference_model = _ToyModel()
    candidate_model = _ToyModel()
    candidate_model.load_state_dict(reference_model.state_dict())
    reference = _run_toy(reference_model, embed_width=None, neuron_width=None, ig=ig)
    candidate = _run_toy(
        candidate_model,
        embed_width=embed_width,
        neuron_width=neuron_width,
        ig=ig,
    )
    for expected, actual in zip(reference[:3], candidate[:3], strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert candidate[3] == reference[3]


@pytest.mark.parametrize(
    ("embed_width", "neuron_width", "embed_has_full", "neuron_has_full"),
    [
        (None, None, True, True),
        (None, 1, True, False),
        (1, None, False, True),
        (1, 1, False, False),
    ],
)
def test_caller_materializes_and_shares_only_required_full_identity(
    monkeypatch: pytest.MonkeyPatch,
    embed_width: int | None,
    neuron_width: int | None,
    embed_has_full: bool,
    neuron_has_full: bool,
) -> None:
    original_embed = attribution_module.run_selected_embed_contribution_vjp
    original_neuron = attribution_module.run_selected_neuron_contribution_vjps
    observed: dict[str, torch.Tensor | None] = {}

    def capture_embed(*args, **kwargs):
        observed["embed"] = kwargs["full_grad_outputs"]
        return original_embed(*args, **kwargs)

    def capture_neuron(*args, **kwargs):
        observed["neuron"] = kwargs["full_grad_outputs"]
        return original_neuron(*args, **kwargs)

    monkeypatch.setattr(
        attribution_module, "run_selected_embed_contribution_vjp", capture_embed
    )
    monkeypatch.setattr(
        attribution_module, "run_selected_neuron_contribution_vjps", capture_neuron
    )
    _run_toy(
        _ToyModel(),
        embed_width=embed_width,
        neuron_width=neuron_width,
        ig=False,
    )

    assert (observed["embed"] is not None) is embed_has_full
    assert (observed["neuron"] is not None) is neuron_has_full
    if embed_has_full and neuron_has_full:
        assert observed["embed"] is observed["neuron"]


def test_embed_only_full_identity_is_released_before_chunked_neuron_vjp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_embed = attribution_module.run_selected_embed_contribution_vjp
    original_neuron = attribution_module.run_selected_neuron_contribution_vjps
    identity_ref: weakref.ReferenceType[torch.Tensor] | None = None

    def capture_embed(*args, **kwargs):
        nonlocal identity_ref
        full_identity = kwargs["full_grad_outputs"]
        assert full_identity is not None
        identity_ref = weakref.ref(full_identity)
        return original_embed(*args, **kwargs)

    def assert_released_before_neuron(*args, **kwargs):
        assert kwargs["full_grad_outputs"] is None
        gc.collect()
        assert identity_ref is not None
        assert identity_ref() is None
        return original_neuron(*args, **kwargs)

    monkeypatch.setattr(
        attribution_module, "run_selected_embed_contribution_vjp", capture_embed
    )
    monkeypatch.setattr(
        attribution_module,
        "run_selected_neuron_contribution_vjps",
        assert_released_before_neuron,
    )

    _run_toy(_ToyModel(), embed_width=None, neuron_width=1, ig=False)
