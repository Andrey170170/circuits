from __future__ import annotations

import json
from copy import deepcopy

import pytest
from scripts.bonafide.cuda_headroom import (
    ALLOCATOR_DENSE_JOINT_POLICY,
    CUDA_ALLOCATOR_SNAPSHOT_SCHEMA_VERSION,
    CUDA_DENSE_JOINT_PRESSURE_SCHEMA_VERSION,
    CUDA_DENSE_JOINT_SAMPLING_VERSION,
    CUDA_HEADROOM_GATE_SCHEMA_VERSION,
    CUDA_MEMORY_SCHEMA_VERSION,
    PEAK_RESERVED_POLICY,
    assess_cuda_headroom,
    cuda_headroom_identity_contract,
    headroom_action_requires_stop,
    headroom_gate_passed,
    normalized_cuda_headroom_gate_contract,
    normalized_cuda_headroom_policy,
)


def _dense_sample(
    *, index: int, active: int, inactive: int, reserved: int, free: int
) -> dict:
    external = max(0, 1_000 - free - reserved)
    pressure = active + inactive + external
    return {
        "sample_index": index,
        "point": f"point-{index}",
        "measurement_stack": ["trace", f"stage-{index}"],
        "elapsed_since_trace_start_seconds": float(index),
        "sampling_wall_seconds": 0.001,
        "active_bytes": active,
        "inactive_split_bytes": inactive,
        "external_device_bytes": external,
        "reserved_bytes": reserved,
        "device_free_bytes": free,
        "device_total_bytes": 1_000,
        "joint_pressure_bytes": pressure,
        "joint_headroom_bytes": max(0, 1_000 - pressure),
    }


def _capture(
    *,
    index: int,
    point: str,
    free: int,
    segments: int,
    active: int,
    mixed_inactive: int,
    execution_index: int | None = None,
    final_metadata: bool = False,
) -> dict:
    metadata = (
        {
            "integrated_gradients_enabled": False,
            "integrated_gradients_execution_count": 1,
        }
        if final_metadata
        else {"execution_index": execution_index}
    )
    return {
        "capture_index": index,
        "point": point,
        "metadata": metadata,
        "device_free_bytes": free,
        "device_total_bytes": 1_000,
        "total_segment_bytes": segments,
        "inactive_bytes_in_mixed_segments": mixed_inactive,
        "block_states": {
            "active_allocated": {"bytes": active},
            "active_pending_free": {"bytes": 0},
            "inactive": {"bytes": segments - active},
        },
    }


def _instrumentation() -> dict:
    limiting = _dense_sample(index=0, active=500, inactive=100, reserved=700, free=100)
    secondary = _dense_sample(index=1, active=520, inactive=80, reserved=700, free=150)
    return {
        "cuda_memory": {
            "schema_version": CUDA_MEMORY_SCHEMA_VERSION,
            "overall": {
                "peak": {
                    "peak_active_bytes": 550,
                    "peak_inactive_split_bytes": 120,
                },
                "allocator_activity_delta": {
                    "num_alloc_retries": 0,
                    "num_ooms": 0,
                },
            },
            "dense_joint_pressure": {
                "schema_version": CUDA_DENSE_JOINT_PRESSURE_SCHEMA_VERSION,
                "sampling_version": CUDA_DENSE_JOINT_SAMPLING_VERSION,
                "sampling_semantics": (
                    "boundary_sampled_not_continuous_no_failure_prediction_v1"
                ),
                "sample_count": 2,
                "cumulative_sampling_wall_seconds": 0.002,
                "max_sampled_external_device_bytes": 200,
                "limiting_sample": limiting,
                "top_pressure_samples": [limiting, secondary],
            },
        },
        "cuda_allocator_snapshots": {
            "schema_version": CUDA_ALLOCATOR_SNAPSHOT_SCHEMA_VERSION,
            "captures": [
                _capture(
                    index=0,
                    point="after_important_mask_selection",
                    free=150,
                    segments=700,
                    active=400,
                    mixed_inactive=80,
                ),
                _capture(
                    index=1,
                    point="before_first_selected_attribution_vjp",
                    free=100,
                    segments=700,
                    active=500,
                    mixed_inactive=100,
                ),
                _capture(
                    index=2,
                    point="after_first_selected_attribution_vjp_raw_result",
                    free=150,
                    segments=700,
                    active=520,
                    mixed_inactive=80,
                ),
                _capture(
                    index=3,
                    point="after_selected_attribution_contribution",
                    free=200,
                    segments=700,
                    active=450,
                    mixed_inactive=100,
                    final_metadata=True,
                ),
            ],
        },
    }


def test_peak_reserved_policy_preserves_legacy_headroom() -> None:
    receipt = assess_cuda_headroom(
        policy=PEAK_RESERVED_POLICY,
        threshold_bytes=200,
        action="stop",
        device_total_bytes=1_000,
        peak_reserved_bytes=750,
        instrumentation={},
    )

    assert receipt["schema_version"] == CUDA_HEADROOM_GATE_SCHEMA_VERSION
    assert receipt["classification"] == "comfortable"
    assert receipt["should_stop"] is False
    assert receipt["warning"] is None
    assert receipt["estimates"]["legacy_peak_reserved"] == {
        "pressure_bytes": 750,
        "headroom_bytes": 250,
    }
    assert headroom_gate_passed(
        receipt,
        expected_threshold_bytes=200,
        expected_policy=PEAK_RESERVED_POLICY,
    )


def test_allocator_policy_counts_live_mixed_and_external_pressure() -> None:
    receipt = assess_cuda_headroom(
        policy=ALLOCATOR_DENSE_JOINT_POLICY,
        threshold_bytes=180,
        action="warn",
        device_total_bytes=1_000,
        peak_reserved_bytes=900,
        instrumentation=_instrumentation(),
    )

    # Independent conservative maxima: peak active 550 + peak inactive-split
    # 120 + max sampled external 200 = 870 pressure and 130 headroom.
    assert receipt["headroom_bytes"] == 130
    assert receipt["effective_pressure_bytes"] == 870
    assert receipt["passed"] is False
    assert receipt["classification"] == "watch"
    assert receipt["should_stop"] is False
    assert receipt["warning"]["classification"] == "watch"
    assert receipt["evidence"]["overall_peak_inactive_split_bytes"] == 120
    assert receipt["evidence"]["max_sampled_external_device_bytes"] == 200
    assert receipt["estimates"]["dense_observed_joint"]["headroom_bytes"] == 200
    assert receipt["estimates"]["conservative_independent_max"]["headroom_bytes"] == 130
    assert receipt["evidence"]["checkpoint_contract"] == {
        "integrated_gradients_enabled": False,
        "integrated_gradients_execution_count": 1,
        "expected_execution_indexes": [None],
        "before_execution_indexes": [None],
        "after_execution_indexes": [None],
    }
    json.dumps(receipt, allow_nan=False)
    assert not headroom_gate_passed(
        receipt,
        expected_threshold_bytes=180,
        expected_policy=ALLOCATOR_DENSE_JOINT_POLICY,
        expected_action="warn",
    )
    assert not headroom_action_requires_stop(
        receipt,
        expected_threshold_bytes=180,
        expected_policy=ALLOCATOR_DENSE_JOINT_POLICY,
        expected_action="warn",
    )


def test_allocator_retry_is_critical_but_warn_action_continues() -> None:
    instrumentation = _instrumentation()
    instrumentation["cuda_memory"]["overall"]["allocator_activity_delta"][
        "num_alloc_retries"
    ] = 1
    receipt = assess_cuda_headroom(
        policy=ALLOCATOR_DENSE_JOINT_POLICY,
        threshold_bytes=100,
        action="warn",
        device_total_bytes=1_000,
        peak_reserved_bytes=900,
        instrumentation=instrumentation,
    )

    assert receipt["classification"] == "critical"
    assert receipt["should_stop"] is False
    assert (
        "allocator retries or OOM counters were observed"
        in receipt["warning"]["message"]
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["cuda_allocator_snapshots"].update(
                {"schema_version": "wrong"}
            ),
            "unsupported snapshot schema",
        ),
        (
            lambda value: value["cuda_allocator_snapshots"].update({"captures": []}),
            "requires non-empty captures",
        ),
        (
            lambda value: value["cuda_allocator_snapshots"]["captures"][0].update(
                {"device_total_bytes": 999}
            ),
            "disagrees with GPU",
        ),
        (
            lambda value: value["cuda_allocator_snapshots"]["captures"][0][
                "block_states"
            ]["active_allocated"].update({"bytes": True}),
            "active_allocated bytes must be a non-negative integer",
        ),
        (
            lambda value: value["cuda_allocator_snapshots"]["captures"][0][
                "block_states"
            ]["active_allocated"].update({"bytes": 401}),
            "block-state bytes do not partition segments",
        ),
        (
            lambda value: value["cuda_allocator_snapshots"]["captures"][0].update(
                {"inactive_bytes_in_mixed_segments": 301}
            ),
            "mixed inactive bytes exceed inactive blocks",
        ),
    ],
)
def test_allocator_policy_fails_closed_on_malformed_evidence(
    mutate, match: str
) -> None:
    instrumentation = _instrumentation()
    mutate(instrumentation)

    with pytest.raises(ValueError, match=match):
        assess_cuda_headroom(
            policy=ALLOCATOR_DENSE_JOINT_POLICY,
            threshold_bytes=180,
            action="warn",
            device_total_bytes=1_000,
            peak_reserved_bytes=900,
            instrumentation=instrumentation,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda dense: dense["top_pressure_samples"][0].update(
                {"reserved_bytes": 1_001}
            ),
            "reserved bytes exceed total",
        ),
        (
            lambda dense: dense["top_pressure_samples"][0].update(
                {"sample_index": dense["sample_count"]}
            ),
            "sample index exceeds sample count",
        ),
        (
            lambda dense: dense.update({"max_sampled_external_device_bytes": 1_001}),
            "max external bytes exceed device total",
        ),
    ],
)
def test_dense_policy_rejects_impossible_bounded_evidence(mutate, match: str) -> None:
    instrumentation = _instrumentation()
    mutate(instrumentation["cuda_memory"]["dense_joint_pressure"])

    with pytest.raises(ValueError, match=match):
        assess_cuda_headroom(
            policy=ALLOCATOR_DENSE_JOINT_POLICY,
            threshold_bytes=180,
            action="warn",
            device_total_bytes=1_000,
            peak_reserved_bytes=900,
            instrumentation=instrumentation,
        )


def test_allocator_policy_is_explicit_and_requires_both_telemetry_streams() -> None:
    assert normalized_cuda_headroom_policy({}) == PEAK_RESERVED_POLICY
    config = {"wave_limits": {"cuda_headroom_policy": ALLOCATOR_DENSE_JOINT_POLICY}}
    with pytest.raises(ValueError, match="requires explicit"):
        normalized_cuda_headroom_policy(config)

    config["wave_limits"]["cuda_headroom_action"] = "warn"
    with pytest.raises(ValueError, match="requires an instrumentation object"):
        normalized_cuda_headroom_policy(config)

    config["instrumentation"] = {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": False,
        "cuda_dense_joint_pressure_telemetry": False,
    }
    with pytest.raises(ValueError, match="cuda_allocator_snapshot_telemetry=true"):
        normalized_cuda_headroom_policy(config)

    config["instrumentation"]["cuda_allocator_snapshot_telemetry"] = True
    with pytest.raises(ValueError, match="cuda_dense_joint_pressure_telemetry=true"):
        normalized_cuda_headroom_policy(config)
    config["instrumentation"]["cuda_dense_joint_pressure_telemetry"] = True
    assert normalized_cuda_headroom_policy(config) == ALLOCATOR_DENSE_JOINT_POLICY
    config["wave_limits"]["min_cuda_headroom_bytes"] = 180
    assert normalized_cuda_headroom_gate_contract(config) == {
        "policy": ALLOCATOR_DENSE_JOINT_POLICY,
        "min_cuda_headroom_bytes": 180,
        "action": "warn",
        "sampling_version": CUDA_DENSE_JOINT_SAMPLING_VERSION,
    }
    assert cuda_headroom_identity_contract(config) == (
        normalized_cuda_headroom_gate_contract(config)
    )
    assert cuda_headroom_identity_contract({}) is None


def test_allocator_policy_requires_complete_checkpoint_contract() -> None:
    instrumentation = _instrumentation()
    instrumentation["cuda_allocator_snapshots"]["captures"].pop()
    with pytest.raises(ValueError, match="exactly one after_selected"):
        assess_cuda_headroom(
            policy=ALLOCATOR_DENSE_JOINT_POLICY,
            threshold_bytes=180,
            action="warn",
            device_total_bytes=1_000,
            peak_reserved_bytes=900,
            instrumentation=instrumentation,
        )


def test_allocator_policy_rejects_mismatched_integrated_gradient_pairs() -> None:
    instrumentation = _instrumentation()
    captures = instrumentation["cuda_allocator_snapshots"]["captures"]
    captures[-1]["metadata"] = {
        "integrated_gradients_enabled": True,
        "integrated_gradients_execution_count": 2,
    }
    with pytest.raises(ValueError, match="execution indexes disagree"):
        assess_cuda_headroom(
            policy=ALLOCATOR_DENSE_JOINT_POLICY,
            threshold_bytes=180,
            action="warn",
            device_total_bytes=1_000,
            peak_reserved_bytes=900,
            instrumentation=instrumentation,
        )


def test_headroom_receipt_validation_fails_closed() -> None:
    receipt = assess_cuda_headroom(
        policy=PEAK_RESERVED_POLICY,
        threshold_bytes=200,
        action="stop",
        device_total_bytes=1_000,
        peak_reserved_bytes=750,
        instrumentation={},
    )
    with pytest.raises(ValueError, match="threshold disagrees"):
        headroom_gate_passed(
            receipt,
            expected_threshold_bytes=201,
            expected_policy=PEAK_RESERVED_POLICY,
        )

    with pytest.raises(ValueError, match="policy disagrees"):
        headroom_gate_passed(
            receipt,
            expected_threshold_bytes=200,
            expected_policy=ALLOCATOR_DENSE_JOINT_POLICY,
        )

    receipt["passed"] = False
    with pytest.raises(ValueError, match="pass decision is inconsistent"):
        headroom_gate_passed(
            receipt,
            expected_threshold_bytes=200,
            expected_policy=PEAK_RESERVED_POLICY,
        )

    receipt["passed"] = True
    receipt["effective_pressure_bytes"] = 749
    with pytest.raises(ValueError, match="does not reconcile to device total"):
        headroom_gate_passed(
            receipt,
            expected_threshold_bytes=200,
            expected_policy=PEAK_RESERVED_POLICY,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt["estimates"]["dense_observed_joint"].update(
            {"headroom_bytes": 999}
        ),
        lambda receipt: receipt["warning"].update({"message": "fabricated"}),
        lambda receipt: receipt["evidence"].pop("checkpoint_contract"),
    ],
)
def test_dense_receipt_validation_recomputes_complete_semantics(mutate) -> None:
    receipt = assess_cuda_headroom(
        policy=ALLOCATOR_DENSE_JOINT_POLICY,
        threshold_bytes=180,
        action="warn",
        device_total_bytes=1_000,
        peak_reserved_bytes=900,
        instrumentation=_instrumentation(),
    )
    tampered = deepcopy(receipt)
    mutate(tampered)

    with pytest.raises(ValueError, match="disagrees with its retained evidence"):
        headroom_gate_passed(
            tampered,
            expected_threshold_bytes=180,
            expected_policy=ALLOCATOR_DENSE_JOINT_POLICY,
            expected_action="warn",
        )
