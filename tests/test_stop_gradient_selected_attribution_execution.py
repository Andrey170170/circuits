from __future__ import annotations

import gc
import weakref
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.stop_gradient_selected_attribution_execution import (
    DEFAULT_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION,
    resolve_stop_gradient_selected_attribution_forward_execution,
    run_stop_gradient_selected_attribution_forward,
)
from torch import nn


class _ToyMlp(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_projection_calls = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        expanded = self.up_proj(hidden).tanh()
        projected = self.down_proj(expanded)
        self.down_projection_calls += 1
        return projected


class _ToyLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.mlp = _ToyMlp(hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(hidden)


class _ToyBackbone(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden)
        self.layers = nn.ModuleList([_ToyLayer(hidden) for _ in range(3)])


class _ToyModel(nn.Module):
    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.model = _ToyBackbone(hidden)
        self.lm_head = nn.Linear(hidden, 7, bias=False)
        self.lm_head_calls = 0

    def forward(self, *, inputs_embeds, attention_mask):
        del attention_mask
        hidden = inputs_embeds
        for layer in self.model.layers:
            hidden = layer(hidden)
        self.lm_head_calls += 1
        return SimpleNamespace(logits=self.lm_head(hidden))


class _Instrumentation:
    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.execution_records: dict[str, list[dict]] = {}

    def increment_counter(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def append_execution_record(self, name: str, **values) -> None:
        self.execution_records.setdefault(name, []).append(values)


def _run(model: _ToyModel, execution: str, *, layer: int = 1):
    return run_stop_gradient_selected_attribution_forward(
        model,
        torch.tensor([[1, 2, 3]]),
        torch.ones(1, 3),
        layer=layer,
        execution=execution,  # type: ignore[arg-type]
    )


def test_default_and_resolver_are_provenance_bearing_and_fail_closed() -> None:
    assert DEFAULT_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION == (
        "full_model_v1"
    )
    assert (
        resolve_stop_gradient_selected_attribution_forward_execution("prefix_stop_v1")
        == "prefix_stop_v1"
    )
    with pytest.raises(ValueError, match="invalid stop-gradient"):
        resolve_stop_gradient_selected_attribution_forward_execution("prefix")


def test_adag_config_validates_execution_and_loads_legacy_state() -> None:
    config = ADAGConfig(
        device="cpu",
        stop_gradient_selected_attribution_forward_execution="prefix_stop_v1",
    )
    assert (
        asdict(config)["stop_gradient_selected_attribution_forward_execution"]
        == "prefix_stop_v1"
    )

    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.stop_gradient_selected_attribution_forward_execution == (
        "full_model_v1"
    )

    with pytest.raises(ValueError, match="invalid stop-gradient"):
        ADAGConfig(
            stop_gradient_selected_attribution_forward_execution="prefix"  # type: ignore[arg-type]
        )


def test_prefix_stop_preserves_activation_and_prefix_gradient_exactly() -> None:
    torch.manual_seed(11)
    full_model = _ToyModel()
    prefix_model = _ToyModel()
    prefix_model.load_state_dict(full_model.state_dict())

    full = _run(full_model, "full_model_v1")
    prefix = _run(prefix_model, "prefix_stop_v1")
    torch.testing.assert_close(prefix.activation, full.activation, atol=0, rtol=0)

    full_gradient = torch.autograd.grad(full.activation.sum(), full.embeddings)[0]
    prefix_gradient = torch.autograd.grad(prefix.activation.sum(), prefix.embeddings)[0]
    torch.testing.assert_close(prefix_gradient, full_gradient, atol=0, rtol=0)

    assert full.model_forward_completed is True
    assert full.decoder_layer_entries == (0, 1, 2)
    assert full.selected_down_projection_completed is True
    assert full.lm_head_completed is True
    assert full.logits_completed is True
    assert full.stopped_before_down_projection is False
    assert full.down_projection_materialized is True
    assert full.decoder_suffix_materialized is True
    assert full.logits_materialized is True
    assert full.logit_shape == (1, 3, 7)
    assert prefix.model_forward_completed is False
    assert prefix.decoder_layer_entries == (0, 1)
    assert prefix.selected_down_projection_completed is False
    assert prefix.lm_head_completed is False
    assert prefix.logits_completed is False
    assert prefix.stopped_before_down_projection is True
    assert prefix.down_projection_materialized is False
    assert prefix.decoder_suffix_materialized is False
    assert prefix.logits_materialized is False
    assert prefix.logit_shape is None
    assert full_model.lm_head_calls == 1
    assert prefix_model.lm_head_calls == 0
    assert prefix_model.model.layers[0].mlp.down_projection_calls == 1
    assert prefix_model.model.layers[1].mlp.down_projection_calls == 0
    assert prefix_model.model.layers[2].mlp.down_projection_calls == 0


def test_full_model_reports_no_decoder_suffix_for_final_layer() -> None:
    result = _run(_ToyModel(), "full_model_v1", layer=2)

    assert result.down_projection_materialized is True
    assert result.decoder_suffix_materialized is False
    assert result.logits_materialized is True


@pytest.mark.parametrize("execution", ["full_model_v1", "prefix_stop_v1"])
def test_capture_hook_is_removed_after_success(execution: str) -> None:
    model = _ToyModel()
    _run(model, execution)

    down_projection = model.model.layers[1].mlp.down_proj
    assert len(down_projection._forward_hooks) == 0
    assert len(down_projection._forward_pre_hooks) == 0
    assert all(
        len(layer._forward_hooks) == len(layer._forward_pre_hooks) == 0
        for layer in model.model.layers
    )
    assert len(model.lm_head._forward_hooks) == 0


def test_prefix_stop_does_not_swallow_unrelated_model_failure() -> None:
    model = _ToyModel()

    def fail_before_target(*, inputs_embeds, attention_mask):
        del inputs_embeds, attention_mask
        raise RuntimeError("unrelated model failure")

    model.forward = fail_before_target  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="unrelated model failure"):
        _run(model, "prefix_stop_v1")
    assert len(model.model.layers[1].mlp.down_proj._forward_pre_hooks) == 0
    assert all(
        len(layer._forward_hooks) == len(layer._forward_pre_hooks) == 0
        for layer in model.model.layers
    )
    assert len(model.lm_head._forward_hooks) == 0


def test_prefix_stop_graph_releases_without_cyclic_gc() -> None:
    model = _ToyModel()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        result = _run(model, "prefix_stop_v1")
        embeddings_ref = weakref.ref(result.embeddings)
        activation_ref = weakref.ref(result.activation)
        assert embeddings_ref() is not None
        assert activation_ref() is not None

        del result

        assert embeddings_ref() is None
        assert activation_ref() is None
    finally:
        if was_enabled:
            gc.enable()


def test_full_model_hook_is_removed_after_forward_failure() -> None:
    model = _ToyModel()

    def fail_after_target(*, inputs_embeds, attention_mask):
        del attention_mask
        model.model.layers[0](inputs_embeds)
        model.model.layers[1](inputs_embeds)
        raise RuntimeError("full forward failure")

    model.forward = fail_after_target  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="full forward failure"):
        _run(model, "full_model_v1")
    assert len(model.model.layers[1].mlp.down_proj._forward_hooks) == 0


def test_missing_target_fails_closed_and_removes_hook() -> None:
    model = _ToyModel()

    def omit_target(*, inputs_embeds, attention_mask):
        del attention_mask
        return SimpleNamespace(logits=model.lm_head(inputs_embeds))

    model.forward = omit_target  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="did not execute"):
        _run(model, "prefix_stop_v1")
    assert len(model.model.layers[1].mlp.down_proj._forward_pre_hooks) == 0


@pytest.mark.parametrize(
    ("execution", "strategy_counter"),
    [
        (
            "full_model_v1",
            "stop_gradient_selected_attribution_full_model_v1_execution_count",
        ),
        (
            "prefix_stop_v1",
            "stop_gradient_selected_attribution_prefix_stop_v1_execution_count",
        ),
    ],
)
def test_execution_counter_proves_requested_adapter(
    execution: str, strategy_counter: str
) -> None:
    instrumentation = _Instrumentation()
    model = _ToyModel()
    run_stop_gradient_selected_attribution_forward(
        model,
        torch.tensor([[1, 2]]),
        torch.ones(1, 2),
        layer=0,
        execution=execution,  # type: ignore[arg-type]
        instrumentation=instrumentation,  # type: ignore[arg-type]
    )

    assert instrumentation.counters == {
        "stop_gradient_selected_attribution_forward_execution_count": 1,
        strategy_counter: 1,
        "stop_gradient_selected_attribution_down_projection_materialized_count": (
            int(execution == "full_model_v1")
        ),
        "stop_gradient_selected_attribution_decoder_suffix_materialized_count": (
            int(execution == "full_model_v1")
        ),
        "stop_gradient_selected_attribution_logits_materialized_count": int(
            execution == "full_model_v1"
        ),
        "stop_gradient_selected_attribution_decoder_layer_entry_count": (
            3 if execution == "full_model_v1" else 1
        ),
        "stop_gradient_selected_attribution_selected_down_projection_completed_count": int(
            execution == "full_model_v1"
        ),
        "stop_gradient_selected_attribution_lm_head_completed_count": int(
            execution == "full_model_v1"
        ),
        "stop_gradient_selected_attribution_logits_completed_count": int(
            execution == "full_model_v1"
        ),
    }
    assert instrumentation.execution_records == {
        "stop_gradient_selected_attribution_forward": [
            {
                "execution": execution,
                "layer": 0,
                "decoder_layer_entries": (
                    [0, 1, 2] if execution == "full_model_v1" else [0]
                ),
                "selected_down_projection_completed": execution == "full_model_v1",
                "lm_head_completed": execution == "full_model_v1",
                "logits_completed": execution == "full_model_v1",
                "down_projection_materialized": execution == "full_model_v1",
                "decoder_suffix_materialized": execution == "full_model_v1",
                "logits_materialized": execution == "full_model_v1",
            }
        ]
    }


def test_execution_records_preserve_selected_layer_call_order() -> None:
    instrumentation = _Instrumentation()
    model = _ToyModel()
    for execution, layer in (("full_model_v1", 2), ("prefix_stop_v1", 1)):
        run_stop_gradient_selected_attribution_forward(
            model,
            torch.tensor([[1, 2]]),
            torch.ones(1, 2),
            layer=layer,
            execution=execution,  # type: ignore[arg-type]
            instrumentation=instrumentation,  # type: ignore[arg-type]
        )

    records = instrumentation.execution_records[
        "stop_gradient_selected_attribution_forward"
    ]
    assert [(record["execution"], record["layer"]) for record in records] == [
        ("full_model_v1", 2),
        ("prefix_stop_v1", 1),
    ]
    assert records[0]["decoder_layer_entries"] == [0, 1, 2]
    assert records[1]["decoder_layer_entries"] == [0, 1]
