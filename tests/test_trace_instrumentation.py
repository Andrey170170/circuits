"""Focused CPU-only tests for observational ADAG trace instrumentation."""

from __future__ import annotations

import json

import pytest
from circuits.tracing.instrumentation import (
    CUDA_ALLOCATOR_SNAPSHOT_SCHEMA_VERSION,
    CUDA_MEMORY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TraceInstrumentation,
    _summarize_cuda_allocator_snapshot,
    cuda_memory_instrumentation_stage,
    cuda_memory_observation_stage,
    record_cuda_allocator_snapshot,
    record_selection_predictors,
)


class _FakeCudaAllocator:
    def __init__(self) -> None:
        self.allocated = 100
        self.reserved = 200
        self.active = 120
        self.inactive_split = 20
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved
        self.peak_active = self.active
        self.peak_inactive_split = self.inactive_split
        self.retries = 0
        self.ooms = 0

    def reset(self, _device=None) -> None:
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved
        self.peak_active = self.active
        self.peak_inactive_split = self.inactive_split

    def set(self, *, allocated: int, reserved: int, inactive_split: int) -> None:
        self.allocated = allocated
        self.reserved = reserved
        self.active = allocated + 10
        self.inactive_split = inactive_split
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        self.peak_active = max(self.peak_active, self.active)
        self.peak_inactive_split = max(self.peak_inactive_split, self.inactive_split)

    def stats(self, _device=None) -> dict[str, int]:
        return {
            "active_bytes.all.current": self.active,
            "active_bytes.all.peak": self.peak_active,
            "inactive_split_bytes.all.current": self.inactive_split,
            "inactive_split_bytes.all.peak": self.peak_inactive_split,
            "num_alloc_retries": self.retries,
            "num_ooms": self.ooms,
        }


def _install_fake_cuda(monkeypatch) -> _FakeCudaAllocator:
    allocator = _FakeCudaAllocator()
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", allocator.reset)
    monkeypatch.setattr("torch.cuda.memory_stats", allocator.stats)
    monkeypatch.setattr(
        "torch.cuda.memory_allocated", lambda _device=None: allocator.allocated
    )
    monkeypatch.setattr(
        "torch.cuda.memory_reserved", lambda _device=None: allocator.reserved
    )
    monkeypatch.setattr(
        "torch.cuda.max_memory_allocated",
        lambda _device=None: allocator.peak_allocated,
    )
    monkeypatch.setattr(
        "torch.cuda.max_memory_reserved",
        lambda _device=None: allocator.peak_reserved,
    )
    return allocator


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


def test_cuda_telemetry_is_opt_in(monkeypatch) -> None:
    def unexpected_cuda_call(*_args, **_kwargs):
        raise AssertionError("disabled telemetry touched the CUDA allocator")

    monkeypatch.setattr("torch.cuda.memory_stats", unexpected_cuda_call)
    recorder = TraceInstrumentation(device="cuda:0", synchronize_cuda=False)
    with recorder.stage("work"):
        pass

    snapshot = recorder.snapshot()
    assert "cuda_memory" not in snapshot
    assert "call_measurements" not in snapshot["stages"]["work"]


def test_allocator_snapshot_summary_math_and_device_filtering() -> None:
    mib = 1 << 20
    gib = 1 << 30
    summary = _summarize_cuda_allocator_snapshot(
        [
            {
                "device": 0,
                "total_size": 80 * mib,
                "blocks": [
                    {
                        "state": "active_allocated",
                        "size": 64 * mib,
                        "requested_size": 60 * mib,
                    },
                    {"state": "active_allocated", "size": 4 * mib},
                    {"state": "inactive", "size": 12 * mib},
                ],
            },
            {
                "device": 0,
                "total_size": 2 * gib + 8 * mib,
                "blocks": [
                    {"state": "inactive", "size": 2 * gib},
                    {"state": "inactive", "size": 8 * mib},
                ],
            },
            {
                "device": 0,
                "total_size": 48 * mib,
                "blocks": [
                    {"state": "active_pending_free", "size": 32 * mib},
                    {"state": "active_awaiting_free", "size": 16 * mib},
                ],
            },
            {
                "device": 1,
                "total_size": 999,
                "blocks": [{"state": "inactive", "size": 999}],
            },
        ],
        device_index=0,
    )

    assert summary["segment_count"] == 3
    assert summary["block_count"] == 7
    assert summary["total_segment_bytes"] == 2 * gib + 136 * mib
    assert summary["block_states"] == {
        "active_allocated": {"block_count": 2, "bytes": 68 * mib},
        "active_pending_free": {"block_count": 2, "bytes": 48 * mib},
        "inactive": {"block_count": 3, "bytes": 2 * gib + 20 * mib},
    }
    assert summary["largest_inactive_block_bytes"] == 2 * gib
    assert summary["fully_inactive_segment_count"] == 1
    assert summary["fully_inactive_segment_bytes"] == 2 * gib + 8 * mib
    assert summary["mixed_segment_count"] == 1
    assert summary["inactive_blocks_in_mixed_segments"] == 1
    assert summary["inactive_bytes_in_mixed_segments"] == 12 * mib
    assert summary["active_allocated_requested_bytes"] == 60 * mib
    assert summary["active_allocated_size_minus_requested_bytes"] == 4 * mib
    assert summary["active_allocated_missing_requested_size_count"] == 1
    buckets = {
        record["bucket"]: record for record in summary["inactive_block_size_buckets"]
    }
    assert buckets["gt_1_mib_le_16_mib"] == {
        "bucket": "gt_1_mib_le_16_mib",
        "block_count": 2,
        "bytes": 20 * mib,
    }
    assert buckets["gt_1_gib"] == {
        "bucket": "gt_1_gib",
        "block_count": 1,
        "bytes": 2 * gib,
    }


def test_allocator_snapshot_is_opt_in_and_helper_is_noop(monkeypatch) -> None:
    def unexpected_snapshot():
        raise AssertionError("disabled allocator snapshot telemetry captured")

    monkeypatch.setattr("torch.cuda.memory_snapshot", unexpected_snapshot)
    recorder = TraceInstrumentation(device="cuda:0")

    assert record_cuda_allocator_snapshot(None, "disabled") is None
    assert record_cuda_allocator_snapshot(recorder, "disabled") is None
    assert "cuda_allocator_snapshots" not in recorder.snapshot()


def test_allocator_snapshot_requires_cuda_memory_telemetry() -> None:
    with pytest.raises(ValueError, match="requires CUDA memory telemetry"):
        TraceInstrumentation(device="cuda:0", cuda_allocator_snapshot_telemetry=True)


def test_allocator_snapshots_are_compact_ordered_and_do_not_synchronize(
    monkeypatch,
) -> None:
    _install_fake_cuda(monkeypatch)
    raw_segments = [
        {
            "device": 0,
            "address": 987654321,
            "total_size": 100,
            "frames": [{"filename": "secret_frame.py"}],
            "blocks": [
                {
                    "address": 987654321,
                    "state": "active_allocated",
                    "size": 60,
                    "requested_size": 55,
                    "frames": [{"filename": "secret_block_frame.py"}],
                },
                {"state": "inactive", "size": 40},
            ],
        }
    ]
    snapshot_calls = 0

    def fake_memory_snapshot():
        nonlocal snapshot_calls
        snapshot_calls += 1
        return raw_segments

    monkeypatch.setattr("torch.cuda.memory_snapshot", fake_memory_snapshot)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda _device=None: (300, 400))
    recorder = TraceInstrumentation(
        device="cuda:0",
        synchronize_cuda=True,
        cuda_memory_telemetry=True,
        cuda_allocator_snapshot_telemetry=True,
    )

    def unexpected_synchronize() -> None:
        raise AssertionError("allocator snapshot explicitly synchronized CUDA")

    monkeypatch.setattr(recorder, "_synchronize", unexpected_synchronize)
    first = record_cuda_allocator_snapshot(
        recorder,
        "before_first_vjp",
        metadata={"execution_index": 0, "shape": (1, 2)},
        once=True,
        once_key=0,
    )
    duplicate = record_cuda_allocator_snapshot(
        recorder,
        "before_first_vjp",
        metadata={"execution_index": 0},
        once=True,
        once_key=0,
    )
    second = record_cuda_allocator_snapshot(
        recorder,
        "before_first_vjp",
        metadata={"execution_index": 1},
        once=True,
        once_key=1,
    )

    assert first is not None
    assert duplicate is None
    assert second is not None
    assert snapshot_calls == 2
    snapshot = recorder.snapshot()
    allocator_snapshots = snapshot["cuda_allocator_snapshots"]
    assert allocator_snapshots["schema_version"] == (
        CUDA_ALLOCATOR_SNAPSHOT_SCHEMA_VERSION
    )
    assert allocator_snapshots["capture_semantics"] == {
        "allocator_history_enabled_by_instrumentation": False,
        "raw_snapshot_retained": False,
        "explicit_cuda_synchronization": False,
        "active_pending_free_raw_state_aliases": [
            "active_pending_free",
            "active_awaiting_free",
        ],
        "mixed_segment_inactive_bytes_semantics": (
            "fragmentation_risk_diagnostic_reuse_and_release_are_allocator_"
            "policy_dependent_v1"
        ),
        "largest_inactive_block_semantics": (
            "diagnostic_not_allocation_success_guarantee_v1"
        ),
    }
    captures = allocator_snapshots["captures"]
    assert [capture["capture_index"] for capture in captures] == [0, 1]
    assert [capture["metadata"]["execution_index"] for capture in captures] == [
        0,
        1,
    ]
    assert captures[0]["metadata"]["shape"] == [1, 2]
    assert captures[0]["device_free_bytes"] == 300
    assert captures[0]["device_total_bytes"] == 400
    assert captures[0]["capture_wall_seconds"] >= 0
    assert captures[0]["elapsed_since_trace_start_seconds"] >= 0
    serialized = json.dumps(snapshot, allow_nan=False)
    for raw_only_value in (
        "address",
        "frames",
        "secret_frame.py",
        "secret_block_frame.py",
        "987654321",
    ):
        assert raw_only_value not in serialized


def test_allocator_snapshot_malformed_input_fails_closed(monkeypatch) -> None:
    _install_fake_cuda(monkeypatch)
    monkeypatch.setattr(
        "torch.cuda.memory_snapshot",
        lambda: [{"total_size": 100, "blocks": []}],
    )
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda _device=None: (300, 400))
    recorder = TraceInstrumentation(
        device="cuda:0",
        cuda_memory_telemetry=True,
        cuda_allocator_snapshot_telemetry=True,
    )

    with pytest.raises(ValueError, match="lacks device identity"):
        recorder.record_cuda_allocator_snapshot("malformed")

    assert recorder.snapshot()["cuda_allocator_snapshots"]["captures"] == []


def test_cuda_telemetry_preserves_nested_and_repeated_stage_peaks(monkeypatch) -> None:
    allocator = _install_fake_cuda(monkeypatch)
    recorder = TraceInstrumentation(device="cuda:0", cuda_memory_telemetry=True)

    with recorder.measure_stage("outer") as outer:
        allocator.set(allocated=300, reserved=500, inactive_split=80)
        with recorder.measure_stage("repeated") as first:
            allocator.set(allocated=450, reserved=700, inactive_split=140)
        allocator.set(allocated=250, reserved=650, inactive_split=120)
    with recorder.measure_stage("repeated") as second:
        allocator.set(allocated=500, reserved=800, inactive_split=180)

    snapshot = recorder.snapshot()
    assert snapshot["cuda_memory"]["schema_version"] == CUDA_MEMORY_SCHEMA_VERSION
    assert snapshot["cuda_memory"]["overall"]["peak"] == {
        "peak_allocated_bytes": 500,
        "peak_reserved_bytes": 800,
        "peak_active_bytes": 510,
        "peak_inactive_split_bytes": 180,
    }
    assert outer.cuda_memory["peak"]["peak_reserved_bytes"] == 700
    assert first.cuda_memory["peak"]["peak_allocated_bytes"] == 450
    assert second.cuda_memory["peak"]["peak_allocated_bytes"] == 500
    repeated = snapshot["stages"]["repeated"]
    assert repeated["calls"] == 2
    assert [call["call_index"] for call in repeated["call_measurements"]] == [0, 1]
    assert repeated["cuda_memory_peak"]["peak_reserved_bytes"] == 800
    json.dumps(snapshot, allow_nan=False)


def test_nested_manual_stage_failure_unwinds_without_masking_original(
    monkeypatch,
) -> None:
    _install_fake_cuda(monkeypatch)
    recorder = TraceInstrumentation(device="cuda:0", cuda_memory_telemetry=True)

    def fail_nested_stage() -> None:
        with recorder.measure_stage("graph_expansion"):
            recorder.measurement_start("layer_pair_jacobian")

            def fail_reset() -> None:
                raise RuntimeError("allocator reset failed during unwind")

            monkeypatch.setattr(recorder, "_reset_cuda_peaks", fail_reset)
            raise ValueError("original tracing failure")

    with pytest.raises(ValueError, match="original tracing failure"):
        fail_nested_stage()

    snapshot = recorder.snapshot()
    assert snapshot["stages"]["layer_pair_jacobian"]["failed_calls"] == 1
    assert snapshot["stages"]["graph_expansion"]["failed_calls"] == 1
    assert recorder._measurement_stack == []


def test_cuda_memory_stage_records_json_safe_vjp_metadata(monkeypatch) -> None:
    _install_fake_cuda(monkeypatch)
    recorder = TraceInstrumentation(device="cuda:0", cuda_memory_telemetry=True)

    with cuda_memory_instrumentation_stage(
        recorder,
        "stop_grad_selected_attribution_vjp",
        metadata={
            "operation_kind": "batched_vjp",
            "layer": 7,
            "chunk_start": 10,
            "chunk_neuron_count": 4,
            "lane_count": 4,
            "differentiated_output_shape": [4, 1],
            "differentiated_input_shape": [1, 2951, 2560],
            "grad_outputs_shape": [4, 4],
        },
    ) as measurement:
        assert measurement is not None
        measurement.metadata["vjp_result_shape"] = [4, 1, 2951, 2560]

    snapshot = recorder.snapshot()
    calls = snapshot["stages"]["stop_grad_selected_attribution_vjp"][
        "call_measurements"
    ]
    assert calls[0]["metadata"] == {
        "operation_kind": "batched_vjp",
        "layer": 7,
        "chunk_start": 10,
        "chunk_neuron_count": 4,
        "lane_count": 4,
        "differentiated_output_shape": [4, 1],
        "differentiated_input_shape": [1, 2951, 2560],
        "grad_outputs_shape": [4, 4],
        "vjp_result_shape": [4, 1, 2951, 2560],
    }
    json.dumps(snapshot, allow_nan=False)


def test_cuda_memory_stage_is_noop_without_telemetry(monkeypatch) -> None:
    recorder = TraceInstrumentation(device="cuda:0", synchronize_cuda=True)

    def unexpected_synchronize() -> None:
        raise AssertionError("disabled fine-grained telemetry synchronized CUDA")

    monkeypatch.setattr(recorder, "_synchronize", unexpected_synchronize)
    with cuda_memory_instrumentation_stage(
        recorder,
        "selected_attribution_vjp",
        metadata={"operation_kind": "batched_vjp"},
    ) as measurement:
        assert measurement is None

    assert "selected_attribution_vjp" not in recorder.snapshot()["stages"]


def test_cuda_memory_observation_stage_does_not_synchronize(monkeypatch) -> None:
    _install_fake_cuda(monkeypatch)
    recorder = TraceInstrumentation(
        device="cuda:0",
        synchronize_cuda=True,
        cuda_memory_telemetry=True,
    )

    def unexpected_synchronize() -> None:
        raise AssertionError("memory-only observation synchronized CUDA")

    monkeypatch.setattr(recorder, "_synchronize", unexpected_synchronize)
    with cuda_memory_observation_stage(
        recorder,
        "stop_grad_selected_layer_forward",
        metadata={"operation_kind": "model_forward", "layer": 3},
    ):
        pass

    call = recorder.snapshot()["stages"]["stop_grad_selected_layer_forward"][
        "call_measurements"
    ][0]
    assert call["metadata"] == {
        "timing_semantics": "host_enqueue_wall_v1",
        "synchronizes_cuda": False,
        "operation_kind": "model_forward",
        "layer": 3,
    }


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
        jacobian_target_chunk_size=2,
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
        jacobian_target_chunk_size=20,
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
        jacobian_target_chunk_size=50,
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


def test_selection_predictors_keep_attribution_and_jacobian_widths_independent() -> (
    None
):
    recorder = TraceInstrumentation(device="cpu")
    record_selection_predictors(
        recorder,
        {
            1: [[0, neuron] for neuron in range(11)],
            2: [[0, neuron] for neuron in range(11)],
        },
        keep_tokens=[0],
        start_layer=0,
        end_layer=3,
        selected_attribution_chunk_size=3,
        jacobian_target_chunk_size=50,
        use_stop_grad_on_mlps=True,
        ig_steps=None,
    )

    predictors = recorder.snapshot()["early_predictors"]
    assert predictors["selected_attribution_chunk_size"] == 3
    assert predictors["selected_attribution_chunks_per_pass"] == 8
    assert predictors["jacobian_target_chunk_size"] == 50
    assert predictors["jacobian_target_chunks_per_pass"] == 1
    assert predictors["stop_grad_mlp_attribution_chunk_size"] == 10
    assert predictors["stop_grad_mlp_attribution_chunks_per_pass"] == 4


def test_stage_does_not_synchronize_while_unwinding_original_error(monkeypatch) -> None:
    recorder = TraceInstrumentation(device="cuda:0", synchronize_cuda=True)
    sync_calls = 0

    def fake_synchronize() -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls > 1:
            raise RuntimeError("synchronization failure")

    monkeypatch.setattr(recorder, "_synchronize", fake_synchronize)

    with (
        pytest.raises(ValueError, match="original trace failure"),
        recorder.stage("failing_stage"),
    ):
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
