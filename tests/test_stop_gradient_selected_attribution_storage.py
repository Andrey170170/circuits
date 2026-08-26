from __future__ import annotations

import gc
import weakref
from dataclasses import asdict

import pytest
import torch
from circuits.tracing.attribution import (
    _project_stop_gradient_selected_attribution_vjp,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.stop_gradient_selected_attribution_storage import (
    DEFAULT_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE,
    resolve_stop_gradient_selected_attribution_storage,
    store_stop_gradient_selected_attribution,
)


class _Instrumentation:
    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.execution_records: dict[str, list[dict]] = {}

    def increment_counter(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def append_execution_record(self, name: str, **values) -> None:
        self.execution_records.setdefault(name, []).append(values)


def _graph_projection():
    raw_vjp = torch.linspace(-1.0, 1.0, 2 * 1 * 3 * 4).reshape(2, 1, 3, 4)
    raw_vjp.requires_grad_(True)
    embeddings = torch.linspace(-0.5, 0.7, 1 * 3 * 4).reshape(1, 3, 4)
    embeddings.requires_grad_(True)
    projection = _project_stop_gradient_selected_attribution_vjp(
        raw_vjp,
        embeddings,
        [2, 0, 2],
    )
    return raw_vjp, embeddings, projection


def test_storage_default_validation_and_legacy_config_are_provenance_bearing() -> None:
    assert DEFAULT_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE == ("graph_retaining_v1")
    assert (
        resolve_stop_gradient_selected_attribution_storage("terminal_detached_v1")
        == "terminal_detached_v1"
    )
    with pytest.raises(ValueError, match="invalid stop-gradient"):
        resolve_stop_gradient_selected_attribution_storage("detached")

    config = ADAGConfig(
        stop_gradient_selected_attribution_storage="terminal_detached_v1"
    )
    assert asdict(config)["stop_gradient_selected_attribution_storage"] == (
        "terminal_detached_v1"
    )
    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.stop_gradient_selected_attribution_storage == ("graph_retaining_v1")
    with pytest.raises(ValueError, match="invalid stop-gradient"):
        ADAGConfig(
            stop_gradient_selected_attribution_storage="detached"  # type: ignore[arg-type]
        )


def test_terminal_detach_preserves_exact_values_dtype_order_and_storage() -> None:
    _raw_vjp, _embeddings, projection = _graph_projection()
    historical = store_stop_gradient_selected_attribution(
        projection,
        strategy="graph_retaining_v1",
        layer=3,
        chunk_start=4,
    )
    candidate = store_stop_gradient_selected_attribution(
        projection,
        strategy="terminal_detached_v1",
        layer=3,
        chunk_start=4,
    )

    torch.testing.assert_close(candidate.tensor, historical.tensor, atol=0, rtol=0)
    assert candidate.tensor.dtype == historical.tensor.dtype == projection.dtype
    assert torch.equal(candidate.tensor[:, :, 0], candidate.tensor[:, :, 2])
    assert historical.tensor is projection
    assert historical.stored_grad_fn_retained is True
    assert candidate.stored_requires_grad is False
    assert candidate.stored_grad_fn_retained is False
    assert candidate.terminal_detached is True
    assert candidate.shares_projection_storage is True


def test_terminal_detach_releases_projection_graph_without_cyclic_gc() -> None:
    raw_vjp, embeddings, projection = _graph_projection()
    raw_ref = weakref.ref(raw_vjp)
    embeddings_ref = weakref.ref(embeddings)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        stored = store_stop_gradient_selected_attribution(
            projection,
            strategy="terminal_detached_v1",
            layer=0,
            chunk_start=0,
        )
        del projection, raw_vjp, embeddings

        assert raw_ref() is None
        assert embeddings_ref() is None
        assert stored.tensor.grad_fn is None
        assert stored.tensor.requires_grad is False
    finally:
        if was_enabled:
            gc.enable()


def test_historical_storage_retains_graph_until_result_release() -> None:
    raw_vjp, embeddings, projection = _graph_projection()
    raw_ref = weakref.ref(raw_vjp)
    embeddings_ref = weakref.ref(embeddings)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        stored = store_stop_gradient_selected_attribution(
            projection,
            strategy="graph_retaining_v1",
            layer=0,
            chunk_start=0,
        )
        del projection, raw_vjp, embeddings
        assert raw_ref() is not None
        assert embeddings_ref() is not None

        del stored

        assert raw_ref() is None
        assert embeddings_ref() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize(
    ("strategy", "stored_graph", "detached"),
    [
        ("graph_retaining_v1", 1, 0),
        ("terminal_detached_v1", 0, 1),
    ],
)
def test_storage_telemetry_observes_graph_lifetime(
    strategy: str, stored_graph: int, detached: int
) -> None:
    _raw_vjp, _embeddings, projection = _graph_projection()
    instrumentation = _Instrumentation()
    result = store_stop_gradient_selected_attribution(
        projection,
        strategy=strategy,  # type: ignore[arg-type]
        layer=5,
        chunk_start=7,
        instrumentation=instrumentation,  # type: ignore[arg-type]
    )

    assert instrumentation.counters == {
        "stop_gradient_selected_attribution_storage_execution_count": 1,
        f"stop_gradient_selected_attribution_{strategy}_storage_count": 1,
        "stop_gradient_selected_attribution_projection_graph_retained_count": 1,
        "stop_gradient_selected_attribution_stored_graph_retained_count": stored_graph,
        "stop_gradient_selected_attribution_terminal_detached_count": detached,
    }
    assert instrumentation.execution_records == {
        "stop_gradient_selected_attribution_storage": [
            {
                "layer": 5,
                "chunk_start": 7,
                "strategy": strategy,
                "input_requires_grad": True,
                "input_grad_fn_retained": True,
                "stored_requires_grad": bool(stored_graph),
                "stored_grad_fn_retained": bool(stored_graph),
                "terminal_detached": bool(detached),
                "shares_projection_storage": True,
            }
        ]
    }
    assert result.stored_grad_fn_retained is bool(stored_graph)
