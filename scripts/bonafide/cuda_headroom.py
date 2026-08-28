"""Post-unit CUDA pressure diagnostics and optional continuation gates.

The dense allocator policy reports both contemporaneous boundary samples and
an intentionally pessimistic composite of independent allocator maxima.  The
result describes a completed work unit; it is not a prediction that the next
unit will or will not fail.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from circuits.tracing.instrumentation import (
    CUDA_ALLOCATOR_SNAPSHOT_SCHEMA_VERSION,
    CUDA_DENSE_JOINT_PRESSURE_SCHEMA_VERSION,
    CUDA_DENSE_JOINT_SAMPLING_SEMANTICS,
    CUDA_DENSE_JOINT_SAMPLING_VERSION,
    CUDA_DENSE_JOINT_TOP_SAMPLE_LIMIT,
    CUDA_MEMORY_SCHEMA_VERSION,
)

CUDA_HEADROOM_GATE_SCHEMA_VERSION = "bonafide.cuda-headroom-gate.v1"

PEAK_RESERVED_POLICY = "peak_reserved_v1"
ALLOCATOR_DENSE_JOINT_POLICY = "allocator_dense_joint_v1"
CUDA_HEADROOM_POLICIES = {
    PEAK_RESERVED_POLICY,
    ALLOCATOR_DENSE_JOINT_POLICY,
}
CUDA_HEADROOM_ACTIONS = {"warn", "stop"}


def normalized_cuda_headroom_policy(config: Mapping[str, Any]) -> str:
    """Return and validate the explicitly selected operational policy."""

    limits = config.get("wave_limits", {})
    if not isinstance(limits, Mapping):
        raise ValueError("run config wave_limits must be an object")
    policy = limits.get("cuda_headroom_policy", PEAK_RESERVED_POLICY)
    if not isinstance(policy, str) or policy not in CUDA_HEADROOM_POLICIES:
        raise ValueError(
            "run config wave_limits.cuda_headroom_policy must be one of "
            f"{sorted(CUDA_HEADROOM_POLICIES)}"
        )
    action = limits.get("cuda_headroom_action", "stop")
    if not isinstance(action, str) or action not in CUDA_HEADROOM_ACTIONS:
        raise ValueError(
            "run config wave_limits.cuda_headroom_action must be one of "
            f"{sorted(CUDA_HEADROOM_ACTIONS)}"
        )
    if policy == ALLOCATOR_DENSE_JOINT_POLICY:
        if "cuda_headroom_action" not in limits:
            raise ValueError(
                f"{ALLOCATOR_DENSE_JOINT_POLICY} requires explicit "
                "wave_limits.cuda_headroom_action"
            )
        instrumentation = config.get("instrumentation")
        if not isinstance(instrumentation, Mapping):
            raise ValueError(
                f"{ALLOCATOR_DENSE_JOINT_POLICY} requires an instrumentation object"
            )
        for field in (
            "cuda_memory_telemetry",
            "cuda_allocator_snapshot_telemetry",
            "cuda_dense_joint_pressure_telemetry",
        ):
            if instrumentation.get(field) is not True:
                raise ValueError(
                    f"{ALLOCATOR_DENSE_JOINT_POLICY} requires "
                    f"instrumentation.{field}=true"
                )
    return policy


def normalized_cuda_headroom_gate_contract(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the validated post-unit headroom decision contract."""

    policy = normalized_cuda_headroom_policy(config)
    limits = config.get("wave_limits", {})
    if not isinstance(limits, Mapping):  # Kept local for type narrowing.
        raise ValueError("run config wave_limits must be an object")
    threshold_bytes = _require_nonnegative_int(
        limits.get("min_cuda_headroom_bytes", 0),
        "run config wave_limits.min_cuda_headroom_bytes",
    )
    return {
        "policy": policy,
        "min_cuda_headroom_bytes": threshold_bytes,
        "action": limits.get("cuda_headroom_action", "stop"),
        **(
            {"sampling_version": CUDA_DENSE_JOINT_SAMPLING_VERSION}
            if policy == ALLOCATOR_DENSE_JOINT_POLICY
            else {}
        ),
    }


def cuda_headroom_identity_contract(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind only the explicit dense policy; frozen legacy identities stay stable."""

    contract = normalized_cuda_headroom_gate_contract(config)
    if contract["policy"] != ALLOCATOR_DENSE_JOINT_POLICY:
        return None
    return contract


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _allocator_capture_assessment(
    capture: Mapping[str, Any],
    *,
    expected_device_total_bytes: int,
) -> dict[str, Any]:
    capture_index = _require_nonnegative_int(
        capture.get("capture_index"), "allocator capture capture_index"
    )
    point = capture.get("point")
    if not isinstance(point, str) or not point:
        raise ValueError("allocator capture point must be a non-empty string")
    device_total_bytes = _require_nonnegative_int(
        capture.get("device_total_bytes"),
        f"allocator capture {point} device_total_bytes",
    )
    if device_total_bytes != expected_device_total_bytes:
        raise ValueError(
            f"allocator capture {point} device_total_bytes disagrees with GPU"
        )
    device_free_bytes = _require_nonnegative_int(
        capture.get("device_free_bytes"),
        f"allocator capture {point} device_free_bytes",
    )
    if device_free_bytes > device_total_bytes:
        raise ValueError(f"allocator capture {point} device_free_bytes exceeds total")
    total_segment_bytes = _require_nonnegative_int(
        capture.get("total_segment_bytes"),
        f"allocator capture {point} total_segment_bytes",
    )
    if total_segment_bytes > device_total_bytes:
        raise ValueError(f"allocator capture {point} total_segment_bytes exceeds total")
    inactive_mixed_bytes = _require_nonnegative_int(
        capture.get("inactive_bytes_in_mixed_segments"),
        f"allocator capture {point} inactive_bytes_in_mixed_segments",
    )
    block_states = capture.get("block_states")
    if not isinstance(block_states, Mapping):
        raise ValueError(f"allocator capture {point} block_states must be an object")
    expected_states = {
        "active_allocated",
        "active_pending_free",
        "inactive",
    }
    if set(block_states) != expected_states:
        raise ValueError(f"allocator capture {point} block_states are incomplete")
    state_bytes: dict[str, int] = {}
    for state in sorted(expected_states):
        state_record = block_states[state]
        if not isinstance(state_record, Mapping):
            raise ValueError(
                f"allocator capture {point} block state {state} must be an object"
            )
        state_bytes[state] = _require_nonnegative_int(
            state_record.get("bytes"),
            f"allocator capture {point} block state {state} bytes",
        )
    if sum(state_bytes.values()) != total_segment_bytes:
        raise ValueError(
            f"allocator capture {point} block-state bytes do not partition segments"
        )
    if inactive_mixed_bytes > state_bytes["inactive"]:
        raise ValueError(
            f"allocator capture {point} mixed inactive bytes exceed inactive blocks"
        )
    active_bytes = state_bytes["active_allocated"] + state_bytes["active_pending_free"]
    if active_bytes + inactive_mixed_bytes > total_segment_bytes:
        raise ValueError(
            f"allocator capture {point} active plus mixed inactive bytes "
            "exceed segments"
        )

    physical_used_bytes = device_total_bytes - device_free_bytes
    external_device_bytes = max(0, physical_used_bytes - total_segment_bytes)
    effective_pressure_bytes = (
        active_bytes + inactive_mixed_bytes + external_device_bytes
    )
    observed_headroom_bytes = max(0, device_total_bytes - effective_pressure_bytes)
    return {
        "capture_index": capture_index,
        "point": point,
        "device_total_bytes": device_total_bytes,
        "device_free_bytes": device_free_bytes,
        "total_segment_bytes": total_segment_bytes,
        "active_bytes": active_bytes,
        "active_allocated_bytes": state_bytes["active_allocated"],
        "active_pending_free_bytes": state_bytes["active_pending_free"],
        "inactive_bytes": state_bytes["inactive"],
        "inactive_bytes_in_mixed_segments": inactive_mixed_bytes,
        "external_device_bytes": external_device_bytes,
        "effective_pressure_bytes": effective_pressure_bytes,
        "observed_headroom_bytes": observed_headroom_bytes,
    }


def _validate_checkpoint_contract(
    captures: list[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_points = {
        "after_important_mask_selection",
        "before_first_selected_attribution_vjp",
        "after_first_selected_attribution_vjp_raw_result",
        "after_selected_attribution_contribution",
    }
    point_groups = {point: [] for point in expected_points}
    for capture in captures:
        point = capture.get("point")
        if point not in expected_points:
            raise ValueError(
                f"allocator headroom has unsupported checkpoint: {point!r}"
            )
        point_groups[point].append(capture)

    for point in (
        "after_important_mask_selection",
        "after_selected_attribution_contribution",
    ):
        if len(point_groups[point]) != 1:
            raise ValueError(
                f"allocator headroom requires exactly one {point} checkpoint"
            )

    final_capture = point_groups["after_selected_attribution_contribution"][0]
    final_metadata = final_capture.get("metadata")
    if not isinstance(final_metadata, Mapping):
        raise ValueError("allocator headroom final checkpoint metadata is invalid")
    ig_enabled = final_metadata.get("integrated_gradients_enabled")
    if not isinstance(ig_enabled, bool):
        raise ValueError(
            "allocator headroom final integrated_gradients_enabled is invalid"
        )
    execution_count = _require_nonnegative_int(
        final_metadata.get("integrated_gradients_execution_count"),
        "allocator headroom final integrated_gradients_execution_count",
    )
    if execution_count < 1:
        raise ValueError(
            "allocator headroom integrated-gradients execution count must be positive"
        )
    expected_execution_indexes: list[int | None] = (
        list(range(execution_count)) if ig_enabled else [None]
    )
    if not ig_enabled and execution_count != 1:
        raise ValueError("allocator headroom direct attribution requires one execution")

    paired_points = (
        "before_first_selected_attribution_vjp",
        "after_first_selected_attribution_vjp_raw_result",
    )
    observed: dict[str, list[int | None]] = {}
    for point in paired_points:
        point_captures = point_groups[point]
        indexes: list[int | None] = []
        for capture in point_captures:
            metadata = capture.get("metadata")
            if not isinstance(metadata, Mapping) or "execution_index" not in metadata:
                raise ValueError(
                    f"allocator headroom {point} lacks execution_index metadata"
                )
            execution_index = metadata["execution_index"]
            if execution_index is not None and (
                isinstance(execution_index, bool)
                or not isinstance(execution_index, int)
                or execution_index < 0
            ):
                raise ValueError(
                    f"allocator headroom {point} execution_index is invalid"
                )
            indexes.append(execution_index)
        if indexes != expected_execution_indexes:
            raise ValueError(
                f"allocator headroom {point} execution indexes disagree with final "
                "integrated-gradients metadata"
            )
        observed[point] = indexes
    if len(captures) != 2 + 2 * execution_count:
        raise ValueError("allocator headroom checkpoint count is inconsistent")

    return {
        "integrated_gradients_enabled": ig_enabled,
        "integrated_gradients_execution_count": execution_count,
        "expected_execution_indexes": expected_execution_indexes,
        "before_execution_indexes": observed[paired_points[0]],
        "after_execution_indexes": observed[paired_points[1]],
    }


def assess_cuda_headroom(
    *,
    policy: str,
    threshold_bytes: int,
    action: str,
    device_total_bytes: int,
    peak_reserved_bytes: int,
    instrumentation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a JSON-safe, fail-closed diagnostic receipt for one completed trace."""

    if policy not in CUDA_HEADROOM_POLICIES:
        raise ValueError(f"unsupported CUDA headroom policy: {policy!r}")
    if action not in CUDA_HEADROOM_ACTIONS:
        raise ValueError(f"unsupported CUDA headroom action: {action!r}")
    threshold_bytes = _require_nonnegative_int(
        threshold_bytes, "CUDA headroom threshold_bytes"
    )
    device_total_bytes = _require_nonnegative_int(
        device_total_bytes, "CUDA device_total_bytes"
    )
    peak_reserved_bytes = _require_nonnegative_int(
        peak_reserved_bytes, "CUDA peak_reserved_bytes"
    )
    if peak_reserved_bytes > device_total_bytes:
        raise ValueError("CUDA peak_reserved_bytes exceeds device total")

    if policy == PEAK_RESERVED_POLICY:
        headroom_bytes = device_total_bytes - peak_reserved_bytes
        classification = "comfortable" if headroom_bytes >= threshold_bytes else "watch"
        return _receipt(
            policy=policy,
            action=action,
            classification=classification,
            threshold_bytes=threshold_bytes,
            device_total_bytes=device_total_bytes,
            decision_headroom_bytes=headroom_bytes,
            estimates={
                "legacy_peak_reserved": {
                    "pressure_bytes": peak_reserved_bytes,
                    "headroom_bytes": headroom_bytes,
                },
                "dense_observed_joint": None,
                "conservative_independent_max": None,
            },
            allocator_activity_delta=None,
            evidence={"peak_reserved_bytes": peak_reserved_bytes},
        )

    cuda_memory = instrumentation.get("cuda_memory")
    if not isinstance(cuda_memory, Mapping):
        raise ValueError("allocator headroom policy lacks CUDA memory telemetry")
    if cuda_memory.get("schema_version") != CUDA_MEMORY_SCHEMA_VERSION:
        raise ValueError("allocator headroom policy has unsupported CUDA memory schema")
    overall = cuda_memory.get("overall")
    peak = overall.get("peak") if isinstance(overall, Mapping) else None
    if not isinstance(peak, Mapping):
        raise ValueError("allocator headroom policy lacks overall CUDA memory peaks")
    overall_peak_active_bytes = _require_nonnegative_int(
        peak.get("peak_active_bytes"), "overall CUDA peak_active_bytes"
    )
    overall_peak_inactive_split_bytes = _require_nonnegative_int(
        peak.get("peak_inactive_split_bytes"),
        "overall CUDA peak_inactive_split_bytes",
    )
    allocator_activity_delta = overall.get("allocator_activity_delta")
    if not isinstance(allocator_activity_delta, Mapping):
        raise ValueError("allocator headroom policy lacks allocator activity delta")
    retries_delta = _require_nonnegative_int(
        allocator_activity_delta.get("num_alloc_retries"),
        "overall CUDA num_alloc_retries delta",
    )
    ooms_delta = _require_nonnegative_int(
        allocator_activity_delta.get("num_ooms"), "overall CUDA num_ooms delta"
    )

    dense = cuda_memory.get("dense_joint_pressure")
    dense_evidence = _validate_dense_joint_pressure(
        dense, expected_device_total_bytes=device_total_bytes
    )

    allocator = instrumentation.get("cuda_allocator_snapshots")
    if not isinstance(allocator, Mapping):
        raise ValueError("allocator headroom policy lacks allocator snapshots")
    if allocator.get("schema_version") != CUDA_ALLOCATOR_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("allocator headroom policy has unsupported snapshot schema")
    captures = allocator.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("allocator headroom policy requires non-empty captures")

    assessments = []
    for capture in captures:
        if not isinstance(capture, Mapping):
            raise ValueError("allocator headroom capture must be an object")
        assessments.append(
            _allocator_capture_assessment(
                capture, expected_device_total_bytes=device_total_bytes
            )
        )
    checkpoint_contract = _validate_checkpoint_contract(captures)
    capture_indices = [item["capture_index"] for item in assessments]
    if len(set(capture_indices)) != len(capture_indices):
        raise ValueError("allocator headroom capture_index values must be unique")

    max_sampled_external_device_bytes = dense_evidence[
        "max_sampled_external_device_bytes"
    ]
    independent_max_pressure_bytes_unclamped = (
        overall_peak_active_bytes
        + overall_peak_inactive_split_bytes
        + max_sampled_external_device_bytes
    )
    independent_max_pressure_bytes = min(
        device_total_bytes, independent_max_pressure_bytes_unclamped
    )
    independent_max_headroom_bytes = device_total_bytes - independent_max_pressure_bytes
    limiting_sample = dense_evidence["limiting_sample"]
    dense_headroom_bytes = limiting_sample["joint_headroom_bytes"]
    if retries_delta > 0 or ooms_delta > 0:
        classification = "critical"
    elif (
        dense_headroom_bytes < threshold_bytes
        or independent_max_headroom_bytes < threshold_bytes
    ):
        classification = "watch"
    else:
        classification = "comfortable"
    legacy_headroom_bytes = device_total_bytes - peak_reserved_bytes
    return _receipt(
        policy=policy,
        action=action,
        classification=classification,
        threshold_bytes=threshold_bytes,
        device_total_bytes=device_total_bytes,
        decision_headroom_bytes=min(
            dense_headroom_bytes, independent_max_headroom_bytes
        ),
        estimates={
            "dense_observed_joint": {
                "pressure_bytes": limiting_sample["joint_pressure_bytes"],
                "headroom_bytes": dense_headroom_bytes,
                "sampling_version": CUDA_DENSE_JOINT_SAMPLING_VERSION,
                "sample_index": limiting_sample["sample_index"],
                "point": limiting_sample["point"],
            },
            "conservative_independent_max": {
                "pressure_bytes": independent_max_pressure_bytes,
                "pressure_bytes_unclamped": independent_max_pressure_bytes_unclamped,
                "headroom_bytes": independent_max_headroom_bytes,
            },
            "legacy_peak_reserved": {
                "pressure_bytes": peak_reserved_bytes,
                "headroom_bytes": legacy_headroom_bytes,
            },
        },
        allocator_activity_delta={
            "num_alloc_retries": retries_delta,
            "num_ooms": ooms_delta,
        },
        evidence={
            "peak_reserved_bytes": peak_reserved_bytes,
            "capture_count": len(assessments),
            "checkpoint_contract": checkpoint_contract,
            "decision_semantics": "post_unit_diagnostic_not_failure_prediction_v1",
            "overall_peak_active_bytes": overall_peak_active_bytes,
            "overall_peak_inactive_split_bytes": (overall_peak_inactive_split_bytes),
            "max_sampled_external_device_bytes": (max_sampled_external_device_bytes),
            "independent_max_pressure_bytes_unclamped": (
                independent_max_pressure_bytes_unclamped
            ),
            "dense_joint_pressure": dense_evidence,
            "allocator_snapshot_captures": [dict(capture) for capture in captures],
            "capture_assessments": assessments,
        },
    )


def _validate_dense_joint_sample(
    sample: Any,
    *,
    expected_device_total_bytes: int,
) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise ValueError("dense joint pressure sample must be an object")
    _require_nonnegative_int(
        sample.get("sample_index"), "dense joint pressure sample_index"
    )
    point = sample.get("point")
    if not isinstance(point, str) or not point:
        raise ValueError("dense joint pressure point must be a non-empty string")
    stack = sample.get("measurement_stack")
    if not isinstance(stack, list) or not all(
        isinstance(item, str) and item for item in stack
    ):
        raise ValueError("dense joint pressure measurement_stack is invalid")
    for field in (
        "elapsed_since_trace_start_seconds",
        "sampling_wall_seconds",
    ):
        value = sample.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"dense joint pressure {field} is invalid")
    values = {
        field: _require_nonnegative_int(
            sample.get(field), f"dense joint pressure {field}"
        )
        for field in (
            "active_bytes",
            "inactive_split_bytes",
            "external_device_bytes",
            "reserved_bytes",
            "device_free_bytes",
            "device_total_bytes",
            "joint_pressure_bytes",
            "joint_headroom_bytes",
        )
    }
    if values["device_total_bytes"] != expected_device_total_bytes:
        raise ValueError("dense joint pressure device total disagrees with GPU")
    if values["device_free_bytes"] > expected_device_total_bytes:
        raise ValueError("dense joint pressure free bytes exceed total")
    if values["reserved_bytes"] > expected_device_total_bytes:
        raise ValueError("dense joint pressure reserved bytes exceed total")
    if (
        values["active_bytes"] + values["inactive_split_bytes"]
        > values["reserved_bytes"]
    ):
        raise ValueError("dense joint pressure active plus inactive exceeds reserved")
    expected_external = max(
        0,
        expected_device_total_bytes
        - values["device_free_bytes"]
        - values["reserved_bytes"],
    )
    if values["external_device_bytes"] != expected_external:
        raise ValueError("dense joint pressure external bytes are inconsistent")
    expected_pressure = (
        values["active_bytes"]
        + values["inactive_split_bytes"]
        + values["external_device_bytes"]
    )
    if values["joint_pressure_bytes"] != expected_pressure:
        raise ValueError("dense joint pressure components are inconsistent")
    if values["joint_headroom_bytes"] != max(
        0, expected_device_total_bytes - expected_pressure
    ):
        raise ValueError("dense joint pressure headroom is inconsistent")
    return dict(sample)


def _validate_dense_joint_pressure(
    dense: Any,
    *,
    expected_device_total_bytes: int,
) -> dict[str, Any]:
    if not isinstance(dense, Mapping):
        raise ValueError("allocator headroom policy lacks dense joint telemetry")
    if dense.get("schema_version") != CUDA_DENSE_JOINT_PRESSURE_SCHEMA_VERSION:
        raise ValueError("allocator headroom policy has unsupported dense schema")
    if dense.get("sampling_version") != CUDA_DENSE_JOINT_SAMPLING_VERSION:
        raise ValueError("allocator headroom policy has unsupported sampling version")
    if dense.get("sampling_semantics") != CUDA_DENSE_JOINT_SAMPLING_SEMANTICS:
        raise ValueError("allocator headroom policy has unsupported sampling semantics")
    sample_count = _require_nonnegative_int(
        dense.get("sample_count"), "dense joint pressure sample_count"
    )
    if sample_count < 1:
        raise ValueError("allocator headroom policy requires dense joint samples")
    overhead = dense.get("cumulative_sampling_wall_seconds")
    if (
        isinstance(overhead, bool)
        or not isinstance(overhead, (int, float))
        or overhead < 0
    ):
        raise ValueError("dense joint cumulative sampling overhead is invalid")
    top = dense.get("top_pressure_samples")
    if (
        not isinstance(top, list)
        or not 1 <= len(top) <= CUDA_DENSE_JOINT_TOP_SAMPLE_LIMIT
    ):
        raise ValueError("dense joint top pressure samples are invalid")
    if len(top) > sample_count:
        raise ValueError("dense joint retained samples exceed sample count")
    validated_top = [
        _validate_dense_joint_sample(
            sample, expected_device_total_bytes=expected_device_total_bytes
        )
        for sample in top
    ]
    expected_order = sorted(
        validated_top,
        key=lambda item: (-item["joint_pressure_bytes"], item["sample_index"]),
    )
    if validated_top != expected_order:
        raise ValueError("dense joint top pressure samples are not ordered")
    if any(item["sample_index"] >= sample_count for item in validated_top):
        raise ValueError("dense joint retained sample index exceeds sample count")
    if len({item["sample_index"] for item in validated_top}) != len(validated_top):
        raise ValueError("dense joint retained sample indexes are not unique")
    limiting = _validate_dense_joint_sample(
        dense.get("limiting_sample"),
        expected_device_total_bytes=expected_device_total_bytes,
    )
    if limiting != validated_top[0]:
        raise ValueError("dense joint limiting sample disagrees with retained top")
    max_external = _require_nonnegative_int(
        dense.get("max_sampled_external_device_bytes"),
        "dense joint max sampled external bytes",
    )
    if max_external > expected_device_total_bytes:
        raise ValueError("dense joint max external bytes exceed device total")
    if max_external < max(item["external_device_bytes"] for item in validated_top):
        raise ValueError("dense joint max external bytes are inconsistent")
    return {
        "schema_version": CUDA_DENSE_JOINT_PRESSURE_SCHEMA_VERSION,
        "sampling_version": CUDA_DENSE_JOINT_SAMPLING_VERSION,
        "sampling_semantics": dense["sampling_semantics"],
        "sample_count": sample_count,
        "cumulative_sampling_wall_seconds": overhead,
        "max_sampled_external_device_bytes": max_external,
        "limiting_sample": limiting,
        "top_pressure_samples": validated_top,
    }


def _receipt(
    *,
    policy: str,
    action: str,
    classification: str,
    threshold_bytes: int,
    device_total_bytes: int,
    decision_headroom_bytes: int,
    estimates: Mapping[str, Any],
    allocator_activity_delta: Mapping[str, int] | None,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if classification not in {"comfortable", "watch", "critical"}:
        raise ValueError("unsupported CUDA headroom classification")
    should_stop = action == "stop" and classification != "comfortable"
    return {
        "schema_version": CUDA_HEADROOM_GATE_SCHEMA_VERSION,
        "policy": policy,
        "action": action,
        "threshold_bytes": threshold_bytes,
        "classification": classification,
        "passed": classification == "comfortable",
        "should_stop": should_stop,
        "warning": _warning_for_classification(classification),
        "headroom_bytes": decision_headroom_bytes,
        "effective_pressure_bytes": device_total_bytes - decision_headroom_bytes,
        "device_total_bytes": device_total_bytes,
        "estimates": dict(estimates),
        "allocator_activity_delta": (
            dict(allocator_activity_delta)
            if allocator_activity_delta is not None
            else None
        ),
        "evidence": dict(evidence),
    }


def _warning_for_classification(classification: str) -> dict[str, str] | None:
    if classification == "comfortable":
        return None
    if classification == "watch":
        message = (
            "completed trace is near the configured diagnostic boundary; "
            "this is not a prediction that a later trace will fail"
        )
    elif classification == "critical":
        message = (
            "allocator retries or OOM counters were observed during the completed "
            "trace; inspect allocator pressure before continuing"
        )
    else:
        raise ValueError("unsupported CUDA headroom classification")
    return {
        "kind": "cuda_headroom_post_unit_diagnostic",
        "classification": classification,
        "message": message,
    }


def _recompute_stored_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the complete decision from the receipt's retained evidence."""

    policy = receipt.get("policy")
    action = receipt.get("action")
    threshold_bytes = receipt.get("threshold_bytes")
    device_total_bytes = receipt.get("device_total_bytes")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("CUDA headroom gate receipt evidence must be an object")
    peak_reserved_bytes = evidence.get("peak_reserved_bytes")
    instrumentation: dict[str, Any] = {}
    if policy == ALLOCATOR_DENSE_JOINT_POLICY:
        allocator_activity_delta = receipt.get("allocator_activity_delta")
        if not isinstance(allocator_activity_delta, Mapping):
            raise ValueError(
                "dense CUDA headroom receipt allocator activity must be an object"
            )
        captures = evidence.get("allocator_snapshot_captures")
        if not isinstance(captures, list):
            raise ValueError(
                "dense CUDA headroom receipt lacks allocator snapshot evidence"
            )
        instrumentation = {
            "cuda_memory": {
                "schema_version": CUDA_MEMORY_SCHEMA_VERSION,
                "overall": {
                    "peak": {
                        "peak_active_bytes": evidence.get("overall_peak_active_bytes"),
                        "peak_inactive_split_bytes": evidence.get(
                            "overall_peak_inactive_split_bytes"
                        ),
                    },
                    "allocator_activity_delta": dict(allocator_activity_delta),
                },
                "dense_joint_pressure": evidence.get("dense_joint_pressure"),
            },
            "cuda_allocator_snapshots": {
                "schema_version": CUDA_ALLOCATOR_SNAPSHOT_SCHEMA_VERSION,
                "captures": captures,
            },
        }
    return assess_cuda_headroom(
        policy=policy,
        action=action,
        threshold_bytes=threshold_bytes,
        device_total_bytes=device_total_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        instrumentation=instrumentation,
    )


def headroom_gate_passed(
    receipt: Mapping[str, Any],
    *,
    expected_threshold_bytes: int,
    expected_policy: str,
    expected_action: str = "stop",
) -> bool:
    """Validate a receipt and report whether its classification is comfortable."""

    if receipt.get("schema_version") != CUDA_HEADROOM_GATE_SCHEMA_VERSION:
        raise ValueError("CUDA headroom gate receipt has unsupported schema")
    policy = receipt.get("policy")
    if policy not in CUDA_HEADROOM_POLICIES:
        raise ValueError("CUDA headroom gate receipt has unsupported policy")
    if expected_policy not in CUDA_HEADROOM_POLICIES:
        raise ValueError("expected CUDA headroom policy is unsupported")
    if policy != expected_policy:
        raise ValueError("CUDA headroom gate receipt policy disagrees with config")
    if expected_action not in CUDA_HEADROOM_ACTIONS:
        raise ValueError("expected CUDA headroom action is unsupported")
    if receipt.get("action") != expected_action:
        raise ValueError("CUDA headroom gate receipt action disagrees with config")
    expected_threshold_bytes = _require_nonnegative_int(
        expected_threshold_bytes, "expected CUDA headroom threshold"
    )
    threshold_bytes = _require_nonnegative_int(
        receipt.get("threshold_bytes"), "CUDA headroom receipt threshold"
    )
    if threshold_bytes != expected_threshold_bytes:
        raise ValueError("CUDA headroom gate receipt threshold disagrees with config")
    passed = receipt.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("CUDA headroom gate receipt passed must be boolean")
    headroom_bytes = _require_nonnegative_int(
        receipt.get("headroom_bytes"), "CUDA headroom receipt headroom_bytes"
    )
    effective_pressure_bytes = _require_nonnegative_int(
        receipt.get("effective_pressure_bytes"),
        "CUDA headroom receipt effective_pressure_bytes",
    )
    device_total_bytes = _require_nonnegative_int(
        receipt.get("device_total_bytes"),
        "CUDA headroom receipt device_total_bytes",
    )
    if effective_pressure_bytes + headroom_bytes != device_total_bytes:
        raise ValueError(
            "CUDA headroom gate receipt does not reconcile to device total"
        )
    if policy == PEAK_RESERVED_POLICY and passed != (headroom_bytes >= threshold_bytes):
        raise ValueError("CUDA headroom gate receipt pass decision is inconsistent")
    classification = receipt.get("classification")
    if classification not in {"comfortable", "watch", "critical"}:
        raise ValueError("CUDA headroom gate receipt classification is invalid")
    if passed != (classification == "comfortable"):
        raise ValueError("CUDA headroom gate receipt classification is inconsistent")
    should_stop = receipt.get("should_stop")
    if not isinstance(should_stop, bool):
        raise ValueError("CUDA headroom gate receipt should_stop must be boolean")
    if should_stop != (expected_action == "stop" and classification != "comfortable"):
        raise ValueError("CUDA headroom gate receipt action decision is inconsistent")
    recomputed = _recompute_stored_receipt(receipt)
    if dict(receipt) != recomputed:
        raise ValueError(
            "CUDA headroom gate receipt disagrees with its retained evidence"
        )
    return passed


def headroom_action_requires_stop(
    receipt: Mapping[str, Any],
    *,
    expected_threshold_bytes: int,
    expected_policy: str,
    expected_action: str,
) -> bool:
    """Validate the receipt and return its configured continuation decision."""

    headroom_gate_passed(
        receipt,
        expected_threshold_bytes=expected_threshold_bytes,
        expected_policy=expected_policy,
        expected_action=expected_action,
    )
    return bool(receipt["should_stop"])
