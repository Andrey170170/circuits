"""Focused CPU-only tests for observational ADAG trace instrumentation."""

from __future__ import annotations

import json

import pytest

from circuits.tracing.instrumentation import (
    SCHEMA_VERSION,
    TraceInstrumentation,
    record_selection_predictors,
)


def test_recorder_accumulates_json_safe_stage_data() -> None:
    recorder = TraceInstrumentation(device="cpu")
    with recorder.stage("work"):
        recorder.increment_counter("items", 2)
    recorder.record_stage("work", 0.25)
    recorder.set_counter("non_finite_is_sanitized", float("inf"))
    recorder.record_layer(3, selected_neuron_count=7)
    recorder.record_layer_pair(
        1,
        3,
        candidate_edges=14,
        jacobian_seconds=0.5,
        retained_edges=4,
    )

    snapshot = recorder.snapshot()

    assert snapshot["schema_version"] == SCHEMA_VERSION
    assert snapshot["timing_semantics"] == "host_wall_v1"
    assert snapshot["stages"]["work"]["calls"] == 2
    assert snapshot["stages"]["work"]["wall_seconds"] >= 0.25
    assert snapshot["counters"]["items"] == 2
    assert snapshot["counters"]["non_finite_is_sanitized"] is None
    assert snapshot["layers"] == [{"layer": 3, "selected_neuron_count": 7}]
    assert snapshot["layer_pairs"][0]["retained_edges"] == 4
    json.dumps(snapshot, allow_nan=False)


def test_selection_predictors_match_planned_pair_math() -> None:
    recorder = TraceInstrumentation(device="cpu")
    record_selection_predictors(
        recorder,
        {
            0: [[0, 10], [1, 11], [9, 99]],
            1: [[1, 20]],
            2: [[0, 30], [1, 31], [2, 32]],
            3: [],
        },
        keep_tokens=[0, 1],
        start_layer=-1,
        end_layer=4,
        selected_attribution_chunk_size=2,
        use_stop_grad_on_mlps=False,
        ig_steps=None,
    )

    snapshot = recorder.snapshot()
    predictors = snapshot["early_predictors"]
    assert predictors["selected_neuron_count"] == 5
    assert predictors["selected_neuron_counts_by_layer"] == [
        {"layer": 0, "count": 2},
        {"layer": 1, "count": 1},
        {"layer": 2, "count": 2},
        {"layer": 3, "count": 0},
    ]
    assert predictors["selected_neuron_counts_by_token"] == [
        {"token_position": 0, "count": 2},
        {"token_position": 1, "count": 3},
    ]
    assert predictors["active_layer_count"] == 3
    assert predictors["active_layer_span"] == 3
    assert predictors["planned_active_layer_pair_count"] == 3
    assert predictors["candidate_mlp_edge_count"] == 8
    assert predictors["jacobian_target_chunks_per_pass"] == 3
    assert predictors["planned_jacobian_target_chunk_executions"] == 3
    assert predictors["selected_attribution_chunks_per_pass"] == 3
    assert predictors["selected_attribution_chunk_executions"] == 3
    assert snapshot["counters"]["early_predictors_ready_seconds"] >= 0


def test_selection_predictors_count_ig_and_stop_grad_passes() -> None:
    recorder = TraceInstrumentation(device="cpu")
    record_selection_predictors(
        recorder,
        {
            0: [[0, neuron] for neuron in range(21)],
            1: [[0, neuron] for neuron in range(41)],
        },
        keep_tokens=[0],
        start_layer=-1,
        end_layer=2,
        selected_attribution_chunk_size=20,
        use_stop_grad_on_mlps=True,
        ig_steps=2,
    )

    predictors = recorder.snapshot()["early_predictors"]
    assert predictors["ig_execution_count"] == 3
    assert predictors["selected_attribution_chunks_per_pass"] == 5
    assert predictors["selected_attribution_chunk_executions"] == 15
    assert predictors["stop_grad_mlp_attribution_chunks_per_pass"] == 8
    assert predictors["stop_grad_mlp_attribution_chunk_executions"] == 8
    assert predictors["total_selected_attribution_chunk_executions"] == 23
    assert predictors["jacobian_target_chunks_per_pass"] == 3
    assert predictors["planned_jacobian_target_chunk_executions"] == 9


def test_selection_predictors_match_non_default_layer_pair_bounds() -> None:
    recorder = TraceInstrumentation(device="cpu")
    record_selection_predictors(
        recorder,
        {1: [[0, 1], [0, 2]], 2: [[0, 3]], 3: [[0, 4], [0, 5], [0, 6]]},
        keep_tokens=[0],
        start_layer=1,
        end_layer=4,
        selected_attribution_chunk_size=50,
        use_stop_grad_on_mlps=False,
        ig_steps=None,
    )

    predictors = recorder.snapshot()["early_predictors"]
    assert predictors["pair_eligible_layers"] == [2, 3]
    assert predictors["planned_active_layer_pairs"] == [
        {"src_layer": 2, "tgt_layer": 3}
    ]
    assert predictors["planned_active_layer_pair_count"] == 1
    assert predictors["candidate_mlp_edge_count"] == 3
    assert predictors["jacobian_target_chunks_per_pass"] == 1


def test_stage_does_not_synchronize_while_unwinding_original_error(monkeypatch) -> None:
    recorder = TraceInstrumentation(device="cuda:0", synchronize_cuda=True)
    sync_calls = 0

    def fake_synchronize() -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls > 1:
            raise RuntimeError("synchronization failure")

    monkeypatch.setattr(recorder, "_synchronize", fake_synchronize)

    with pytest.raises(ValueError, match="original trace failure"):
        with recorder.stage("failing_stage"):
            raise ValueError("original trace failure")

    assert sync_calls == 1
    assert recorder.snapshot()["stages"]["failing_stage"]["calls"] == 1


def test_layer_pair_aggregates_equal_sum_of_pair_records() -> None:
    recorder = TraceInstrumentation(device="cpu")
    recorder.record_layer_pair(
        0,
        1,
        candidate_edges=6,
        target_chunks_per_pass=1,
        target_chunk_executions=3,
        retained_edges=2,
    )
    recorder.record_layer_pair(
        0,
        2,
        candidate_edges=10,
        target_chunks_per_pass=2,
        target_chunk_executions=6,
        retained_edges=4,
    )

    snapshot = recorder.snapshot()
    counters = snapshot["counters"]
    pairs = snapshot["layer_pairs"]
    assert counters["cross_layer_pair_count"] == len(pairs) == 2
    assert counters["cross_layer_candidate_edge_count"] == sum(
        pair["candidate_edges"] for pair in pairs
    )
    assert counters["cross_layer_target_chunks_per_pass"] == sum(
        pair["target_chunks_per_pass"] for pair in pairs
    )
    assert counters["cross_layer_target_chunk_executions"] == sum(
        pair["target_chunk_executions"] for pair in pairs
    )
    assert counters["cross_layer_retained_edge_count"] == sum(
        pair["retained_edges"] for pair in pairs
    )
