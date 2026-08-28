from __future__ import annotations

import gc
import weakref
from dataclasses import asdict

import pytest
import torch
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.selected_discovery_state import (
    DEFAULT_POST_SELECTION_STATE_STORAGE,
    resolve_post_selection_state_storage,
    store_selected_discovery_state,
)


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[int, list[list[int]]],
]:
    attributions = torch.arange(2 * 2 * 3 * 4, dtype=torch.bfloat16).reshape(
        2, 2, 3, 4, 1
    )
    mask = torch.zeros((2, 3, 4), dtype=torch.bool)
    mask[0, 2, 3] = True
    mask[1, 0, 1] = True
    mask[1, 2, 0] = True
    mlp_acts = torch.ones((2, 2, 3, 4), dtype=torch.bfloat16)
    embed_acts = torch.ones((2, 3, 5), dtype=torch.bfloat16)
    neuron_cfg = {0: [[2, 3]], 1: [[0, 1], [2, 0]]}
    return attributions, mask, mlp_acts, embed_acts, neuron_cfg


def _occurrence_payload(state) -> list[tuple[int, int, int, torch.Tensor]]:
    return [
        (
            occurrence.layer,
            occurrence.token,
            occurrence.neuron,
            occurrence.final_attribution,
        )
        for occurrence in state.ordered_occurrences
    ]


def test_storage_resolver_is_explicit_and_fail_closed() -> None:
    assert DEFAULT_POST_SELECTION_STATE_STORAGE == "dense_v1"
    assert resolve_post_selection_state_storage("dense_v1") == "dense_v1"
    assert resolve_post_selection_state_storage("compact_cpu_v1") == "compact_cpu_v1"
    with pytest.raises(ValueError, match="invalid post-selection state storage"):
        resolve_post_selection_state_storage("compact")


def test_adag_config_binds_storage_and_restores_legacy_default() -> None:
    config = ADAGConfig(post_selection_state_storage="compact_cpu_v1")
    assert asdict(config)["post_selection_state_storage"] == "compact_cpu_v1"

    restored = ADAGConfig.__new__(ADAGConfig)
    restored.__setstate__({"device": "cpu"})
    assert restored.post_selection_state_storage == "dense_v1"

    with pytest.raises(ValueError, match="invalid post-selection state storage"):
        ADAGConfig(post_selection_state_storage="compact")  # type: ignore[arg-type]


def test_dense_and_compact_states_preserve_occurrence_order_values_and_dtype() -> None:
    inputs = _inputs()
    dense = store_selected_discovery_state(
        *inputs[:4], neuron_cfg=inputs[4], strategy="dense_v1"
    )
    compact = store_selected_discovery_state(
        *inputs[:4], neuron_cfg=inputs[4], strategy="compact_cpu_v1"
    )

    assert dense.active_layers == compact.active_layers == (0, 1)
    dense_payload = _occurrence_payload(dense)
    compact_payload = _occurrence_payload(compact)
    assert [item[:3] for item in dense_payload] == [
        (0, 2, 3),
        (1, 0, 1),
        (1, 2, 0),
    ]
    assert [item[:3] for item in compact_payload] == [
        item[:3] for item in dense_payload
    ]
    for dense_item, compact_item in zip(dense_payload, compact_payload, strict=True):
        torch.testing.assert_close(dense_item[3], compact_item[3], atol=0, rtol=0)
        assert dense_item[3].dtype == compact_item[3].dtype == torch.bfloat16
        assert compact_item[3].device.type == "cpu"


@pytest.mark.parametrize(
    ("strategy", "retained"),
    [("dense_v1", True), ("compact_cpu_v1", False)],
)
def test_state_storage_has_declared_dense_buffer_lifecycle(
    strategy: str, retained: bool
) -> None:
    attributions, mask, mlp_acts, embed_acts, neuron_cfg = _inputs()
    references = [
        weakref.ref(attributions),
        weakref.ref(mask),
        weakref.ref(mlp_acts),
        weakref.ref(embed_acts),
    ]
    state = store_selected_discovery_state(
        attributions,
        mask,
        mlp_acts,
        embed_acts,
        neuron_cfg=neuron_cfg,
        strategy=strategy,  # type: ignore[arg-type]
    )
    del attributions, mask, mlp_acts, embed_acts
    gc.collect()

    assert [reference() is not None for reference in references] == [retained] * 4
    if retained:
        del state
        gc.collect()
        assert [reference() for reference in references] == [None] * 4


def test_storage_records_exact_receipts_bytes_and_lifecycle() -> None:
    attributions, mask, mlp_acts, embed_acts, neuron_cfg = _inputs()
    recorder = TraceInstrumentation(device="cpu")
    compact = store_selected_discovery_state(
        attributions,
        mask,
        mlp_acts,
        embed_acts,
        neuron_cfg=neuron_cfg,
        strategy="compact_cpu_v1",
        instrumentation=recorder,
    )
    snapshot = recorder.snapshot()
    counters = snapshot["counters"]
    records = snapshot["execution_records"]["post_selection_state_storage"]

    assert counters["post_selection_state_storage"] == "compact_cpu_v1"
    assert counters["post_selection_state_storage_execution_count"] == 1
    assert counters["post_selection_state_selected_occurrence_count"] == 3
    assert len(records) == 1
    record = records[0]
    assert record["strategy"] == "compact_cpu_v1"
    assert record["selected_values_shape"] == [3, 2, 1]
    assert record["selected_values_dtype"] == "torch.bfloat16"
    assert record["selected_values_bytes"] == 12
    assert len(record["selected_values_raw_sha256"]) == 64
    assert record["selected_coordinates_shape"] == [3, 3]
    assert record["selected_coordinates_dtype"] == "torch.int64"
    assert record["selected_coordinates_bytes"] == 72
    assert len(record["selected_coordinates_raw_sha256"]) == 64
    assert record["logical_input_bytes"] == sum(
        tensor.numel() * tensor.element_size()
        for tensor in (attributions, mask, mlp_acts, embed_acts)
    )
    assert record["logical_retained_bytes"] == (
        3 * 2 * 1 * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    assert record["logical_released_bytes"] == (
        record["logical_input_bytes"] - record["logical_retained_bytes"]
    )
    assert record["retains_dense_mlp_final_attributions"] is False
    assert record["retains_dense_important_neuron_mask"] is False
    assert record["retains_unused_mlp_final_acts"] is False
    assert record["retains_unused_embed_final_acts"] is False
    assert record["state_values_device"] == "cpu"
    assert compact.active_layers == (0, 1)


def test_storage_fails_closed_when_mask_and_neuron_cfg_differ() -> None:
    attributions, mask, mlp_acts, embed_acts, neuron_cfg = _inputs()
    mask[0, 1, 1] = True
    with pytest.raises(ValueError, match="exactly match"):
        store_selected_discovery_state(
            attributions,
            mask,
            mlp_acts,
            embed_acts,
            neuron_cfg=neuron_cfg,
            strategy="compact_cpu_v1",
        )


def test_storage_ignores_mask_entries_outside_configured_layers() -> None:
    attributions, mask, mlp_acts, embed_acts, _ = _inputs()
    state = store_selected_discovery_state(
        attributions,
        mask,
        mlp_acts,
        embed_acts,
        neuron_cfg={1: [[0, 1], [2, 0]]},
        strategy="compact_cpu_v1",
    )

    assert [item[:3] for item in _occurrence_payload(state)] == [
        (1, 0, 1),
        (1, 2, 0),
    ]


@pytest.mark.parametrize(
    ("neuron_cfg", "message"),
    [
        ({2: [[0, 0]]}, "layer coordinate is out of bounds"),
        ({0: [[3, 0]]}, "token coordinate is out of bounds"),
        ({0: [[0, 4]]}, "neuron coordinate is out of bounds"),
        ({0.0: [[0, 0]]}, "layer coordinate must be an integer"),
        ({0: [[0.0, 0]]}, "token coordinate must be an integer"),
        ({0: [[0, "0"]]}, "neuron coordinate must be an integer"),
    ],
)
def test_storage_rejects_malformed_or_out_of_bounds_coordinates(
    neuron_cfg: dict, message: str
) -> None:
    attributions, mask, mlp_acts, embed_acts, _ = _inputs()
    with pytest.raises(ValueError, match=message):
        store_selected_discovery_state(
            attributions,
            mask,
            mlp_acts,
            embed_acts,
            neuron_cfg=neuron_cfg,
            strategy="compact_cpu_v1",
        )


def test_zero_selection_has_empty_canonical_state() -> None:
    attributions = torch.zeros((2, 1, 3, 4, 1), dtype=torch.bfloat16)
    mask = torch.zeros((2, 3, 4), dtype=torch.bool)
    state = store_selected_discovery_state(
        attributions,
        mask,
        torch.zeros((2, 1, 3, 4), dtype=torch.bfloat16),
        torch.zeros((1, 3, 5), dtype=torch.bfloat16),
        neuron_cfg={0: [], 1: []},
        strategy="compact_cpu_v1",
    )

    assert state.ordered_occurrences == ()
    assert state.active_layers == ()
    assert state.logical_retained_bytes == 0
