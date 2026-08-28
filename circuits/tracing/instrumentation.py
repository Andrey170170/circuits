"""JSON-safe, observational instrumentation for ADAG tracing.

The recorder is deliberately independent of the scientific trace data.  Callers
opt in by passing an instance; a ``None`` recorder leaves tracing behavior and
return values unchanged.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch

SCHEMA_VERSION = "adag.trace-instrumentation.v1"
CUDA_MEMORY_SCHEMA_VERSION = "adag.cuda-memory-telemetry.v1"
CUDA_ALLOCATOR_SNAPSHOT_SCHEMA_VERSION = "adag.cuda-allocator-fragmentation-snapshot.v1"
CUDA_DENSE_JOINT_PRESSURE_SCHEMA_VERSION = "adag.cuda-dense-joint-pressure.v1"
CUDA_DENSE_JOINT_SAMPLING_VERSION = "boundary_cuda_metrics_v1"
CUDA_DENSE_JOINT_SAMPLING_SEMANTICS = (
    "boundary_sampled_not_continuous_no_failure_prediction_v1"
)
CUDA_DENSE_JOINT_TOP_SAMPLE_LIMIT = 16

_INACTIVE_BLOCK_SIZE_BUCKETS = (
    (1 << 20, "le_1_mib"),
    (16 << 20, "gt_1_mib_le_16_mib"),
    (64 << 20, "gt_16_mib_le_64_mib"),
    (256 << 20, "gt_64_mib_le_256_mib"),
    (1 << 30, "gt_256_mib_le_1_gib"),
    (None, "gt_1_gib"),
)


def _json_safe(value: Any) -> Any:
    """Return a JSON-only value, replacing non-finite measurements with null."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"instrumentation value is not JSON serializable: {type(value)!r}")


def _summarize_cuda_allocator_snapshot(
    segments: list[dict[str, Any]], *, device_index: int
) -> dict[str, Any]:
    """Reduce one allocator snapshot to compact, device-local diagnostics.

    Inactive bytes in mixed segments are a fragmentation-risk signal, not a
    claim that the allocator cannot reuse those blocks for a fitting request.
    PyTorch has emitted both ``active_pending_free`` and
    ``active_awaiting_free`` for the same allocator state; both are reported
    canonically as ``active_pending_free``.
    """

    device_segments: list[dict[str, Any]] = []
    for segment in segments:
        if "device" not in segment:
            raise ValueError("CUDA allocator snapshot segment lacks device identity")
        if int(segment["device"]) == device_index:
            device_segments.append(segment)

    state_totals = {
        "active_allocated": {"block_count": 0, "bytes": 0},
        "active_pending_free": {"block_count": 0, "bytes": 0},
        "inactive": {"block_count": 0, "bytes": 0},
    }
    inactive_buckets = {
        name: {"block_count": 0, "bytes": 0}
        for _upper_bound, name in _INACTIVE_BLOCK_SIZE_BUCKETS
    }
    block_count = 0
    largest_inactive_block_bytes = 0
    fully_inactive_segment_count = 0
    fully_inactive_segment_bytes = 0
    mixed_segment_count = 0
    inactive_blocks_in_mixed_segments = 0
    inactive_bytes_in_mixed_segments = 0
    active_allocated_requested_bytes = 0
    active_allocated_size_minus_requested_bytes = 0
    active_allocated_missing_requested_size_count = 0
    active_allocated_requested_size_exceeds_block_size_count = 0

    for segment in device_segments:
        blocks = segment.get("blocks")
        if not isinstance(blocks, list):
            raise ValueError("CUDA allocator snapshot segment blocks are invalid")
        block_count += len(blocks)
        segment_states: set[str] = set()
        segment_inactive_count = 0
        segment_inactive_bytes = 0
        for block in blocks:
            raw_state = block.get("state")
            state = (
                "active_pending_free"
                if raw_state in {"active_pending_free", "active_awaiting_free"}
                else raw_state
            )
            if state not in state_totals:
                raise ValueError(
                    f"unsupported CUDA allocator block state: {raw_state!r}"
                )
            size = block.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("CUDA allocator snapshot block size is invalid")
            state_totals[state]["block_count"] += 1
            state_totals[state]["bytes"] += size
            segment_states.add(state)
            if state == "inactive":
                segment_inactive_count += 1
                segment_inactive_bytes += size
                largest_inactive_block_bytes = max(largest_inactive_block_bytes, size)
                for upper_bound, name in _INACTIVE_BLOCK_SIZE_BUCKETS:
                    if upper_bound is None or size <= upper_bound:
                        inactive_buckets[name]["block_count"] += 1
                        inactive_buckets[name]["bytes"] += size
                        break
            elif state == "active_allocated":
                requested_size = block.get("requested_size")
                if isinstance(requested_size, bool) or not isinstance(
                    requested_size, int
                ):
                    active_allocated_missing_requested_size_count += 1
                else:
                    active_allocated_requested_bytes += requested_size
                    if requested_size > size:
                        active_allocated_requested_size_exceeds_block_size_count += 1
                    else:
                        active_allocated_size_minus_requested_bytes += (
                            size - requested_size
                        )

        total_size = segment.get("total_size")
        if (
            isinstance(total_size, bool)
            or not isinstance(total_size, int)
            or total_size < 0
        ):
            raise ValueError("CUDA allocator snapshot segment total_size is invalid")
        if blocks and segment_states == {"inactive"}:
            fully_inactive_segment_count += 1
            fully_inactive_segment_bytes += total_size
        elif segment_inactive_count and segment_states - {"inactive"}:
            mixed_segment_count += 1
            inactive_blocks_in_mixed_segments += segment_inactive_count
            inactive_bytes_in_mixed_segments += segment_inactive_bytes

    return {
        "device_index": device_index,
        "segment_count": len(device_segments),
        "block_count": block_count,
        "total_segment_bytes": sum(
            int(segment["total_size"]) for segment in device_segments
        ),
        "block_states": state_totals,
        "largest_inactive_block_bytes": largest_inactive_block_bytes,
        "fully_inactive_segment_count": fully_inactive_segment_count,
        "fully_inactive_segment_bytes": fully_inactive_segment_bytes,
        "mixed_segment_count": mixed_segment_count,
        "inactive_blocks_in_mixed_segments": inactive_blocks_in_mixed_segments,
        "inactive_bytes_in_mixed_segments": inactive_bytes_in_mixed_segments,
        "active_allocated_requested_bytes": active_allocated_requested_bytes,
        "active_allocated_size_minus_requested_bytes": (
            active_allocated_size_minus_requested_bytes
        ),
        "active_allocated_missing_requested_size_count": (
            active_allocated_missing_requested_size_count
        ),
        "active_allocated_requested_size_exceeds_block_size_count": (
            active_allocated_requested_size_exceeds_block_size_count
        ),
        "inactive_block_size_buckets": [
            {"bucket": name, **inactive_buckets[name]}
            for _upper_bound, name in _INACTIVE_BLOCK_SIZE_BUCKETS
        ],
    }


@dataclass
class StageMeasurement:
    """One completed stage call, populated when its context exits."""

    name: str
    wall_seconds: float | None = None
    failed: bool = False
    cuda_memory: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _started: float = field(default=0.0, repr=False)
    _cuda_start: dict[str, int | float] | None = field(default=None, repr=False)
    _cuda_peak: dict[str, int] = field(default_factory=dict, repr=False)


class TraceInstrumentation:
    """Accumulate stage timings and workload counters for one trace.

    When ``synchronize_cuda`` is true, CUDA is synchronized at every timing
    boundary.  This makes GPU stage timings meaningful, at the cost of a small
    change to execution scheduling and total runtime.
    """

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        synchronize_cuda: bool = False,
        cuda_memory_telemetry: bool = False,
        cuda_allocator_snapshot_telemetry: bool = False,
        cuda_dense_joint_pressure_telemetry: bool = False,
    ) -> None:
        self.device = str(device) if device is not None else None
        self.synchronize_cuda = bool(synchronize_cuda)
        self.cuda_memory_telemetry = bool(cuda_memory_telemetry)
        self.cuda_allocator_snapshot_telemetry = bool(cuda_allocator_snapshot_telemetry)
        self.cuda_dense_joint_pressure_telemetry = bool(
            cuda_dense_joint_pressure_telemetry
        )
        if self.cuda_allocator_snapshot_telemetry and not self.cuda_memory_telemetry:
            raise ValueError(
                "CUDA allocator snapshot telemetry requires CUDA memory telemetry"
            )
        if self.cuda_dense_joint_pressure_telemetry and not self.cuda_memory_telemetry:
            raise ValueError(
                "CUDA dense joint pressure telemetry requires CUDA memory telemetry"
            )
        self._started = time.perf_counter()
        self._stages: dict[str, dict[str, Any]] = {}
        self._counters: dict[str, Any] = {}
        self._early_predictors: dict[str, Any] = {}
        self._layers: dict[int, dict[str, Any]] = {}
        self._layer_pairs: list[dict[str, Any]] = []
        self._execution_records: dict[str, list[dict[str, Any]]] = {}
        self._measurement_stack: list[StageMeasurement] = []
        self._cuda_device: torch.device | None = None
        self._cuda_overall_start: dict[str, int | float] | None = None
        self._cuda_overall_peak: dict[str, int] = {}
        self._cuda_joint_pressure_sample_count = 0
        self._cuda_joint_pressure_sampling_wall_seconds = 0.0
        self._cuda_joint_pressure_limiting_sample: dict[str, Any] | None = None
        self._cuda_joint_pressure_max_external_bytes = 0
        self._cuda_joint_pressure_top_samples: list[dict[str, Any]] = []
        self._cuda_allocator_snapshots: list[dict[str, Any]] = []
        self._cuda_allocator_snapshot_once_keys: set[str] = set()
        if self.cuda_memory_telemetry:
            device_value = (
                torch.device(self.device) if self.device is not None else None
            )
            if device_value is None or device_value.type != "cuda":
                raise ValueError(
                    "CUDA memory telemetry requires an explicit CUDA device"
                )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA memory telemetry requested without CUDA")
            self._cuda_device = device_value
            torch.cuda.reset_peak_memory_stats(self._cuda_device)
            self._cuda_overall_start = self._cuda_metrics(point="trace_start")
            self._update_overall_peak(self._cuda_overall_start)

    def record_cuda_allocator_snapshot(
        self,
        point: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        once: bool = False,
        once_key: Any = None,
    ) -> dict[str, Any] | None:
        """Capture and immediately summarize allocator structure without a sync.

        ``once_key`` scopes recorder-owned de-duplication. This lets an
        integrated-gradients run capture the first VJP of each execution while
        suppressing later chunks within that execution.
        """

        if not self.cuda_allocator_snapshot_telemetry:
            return None
        if self._cuda_device is None:
            raise RuntimeError("allocator snapshot telemetry lacks a CUDA device")
        if not isinstance(point, str) or not point:
            raise ValueError("allocator snapshot point must be a non-empty string")
        safe_once_key = _json_safe(once_key)
        deduplication_key = json.dumps(
            [point, safe_once_key], sort_keys=True, separators=(",", ":")
        )
        if once and deduplication_key in self._cuda_allocator_snapshot_once_keys:
            return None

        device_index = self._cuda_device.index
        if device_index is None:
            device_index = int(torch.cuda.current_device())
        started = time.perf_counter()
        raw_segments = torch.cuda.memory_snapshot()
        if not isinstance(raw_segments, list):
            raise ValueError("CUDA allocator snapshot is not a segment list")
        summary = _summarize_cuda_allocator_snapshot(
            raw_segments, device_index=device_index
        )
        device_free_bytes, device_total_bytes = torch.cuda.mem_get_info(
            self._cuda_device
        )
        current_allocator_stats = self._cuda_metrics(
            point=f"allocator_snapshot:{point}"
        )
        capture_wall_seconds = time.perf_counter() - started
        safe_metadata = _json_safe(dict(metadata or {}))
        if not isinstance(safe_metadata, dict):
            raise RuntimeError("allocator snapshot metadata is not an object")
        record = {
            "capture_index": len(self._cuda_allocator_snapshots),
            "point": point,
            "metadata": safe_metadata,
            "elapsed_since_trace_start_seconds": started - self._started,
            "capture_wall_seconds": capture_wall_seconds,
            "device_free_bytes": int(device_free_bytes),
            "device_total_bytes": int(device_total_bytes),
            "current_allocator_stats": current_allocator_stats,
            **summary,
        }
        # Retain only the compact summary. The raw segment/block/address graph
        # is deliberately allowed to fall out of scope here.
        self._cuda_allocator_snapshots.append(record)
        if once:
            self._cuda_allocator_snapshot_once_keys.add(deduplication_key)
        return record

    def _synchronize(self) -> None:
        if not self.synchronize_cuda:
            return
        device = torch.device(self.device) if self.device is not None else None
        if device is not None and device.type != "cuda":
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def timer_start(self) -> float:
        self._synchronize()
        return time.perf_counter()

    def timer_finish(self, started: float) -> float:
        self._synchronize()
        return time.perf_counter() - started

    def _cuda_metrics(
        self,
        *,
        point: str,
        measurement_stack: list[str] | None = None,
    ) -> dict[str, int | float]:
        if self._cuda_device is None:
            raise RuntimeError("CUDA metrics requested when telemetry is disabled")
        if not isinstance(point, str) or not point:
            raise ValueError("CUDA metrics point must be a non-empty string")
        started = (
            time.perf_counter() if self.cuda_dense_joint_pressure_telemetry else 0.0
        )
        stats = torch.cuda.memory_stats(self._cuda_device)
        allocated = int(torch.cuda.memory_allocated(self._cuda_device))
        reserved = int(torch.cuda.memory_reserved(self._cuda_device))
        active = int(stats.get("active_bytes.all.current", allocated))
        inactive_split = int(stats.get("inactive_split_bytes.all.current", 0))
        metrics: dict[str, int | float] = {
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "active_bytes": active,
            "inactive_split_bytes": inactive_split,
            "reserved_minus_allocated_bytes": max(0, reserved - allocated),
            "inactive_split_fraction_of_reserved": (
                inactive_split / reserved if reserved else 0.0
            ),
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(self._cuda_device)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(self._cuda_device)
            ),
            "peak_active_bytes": int(stats.get("active_bytes.all.peak", active)),
            "peak_inactive_split_bytes": int(
                stats.get("inactive_split_bytes.all.peak", inactive_split)
            ),
        }
        if not self.cuda_dense_joint_pressure_telemetry:
            return metrics
        if active < 0 or inactive_split < 0 or reserved < 0:
            raise ValueError("CUDA allocator metrics contain negative byte counts")
        if active + inactive_split > reserved:
            raise ValueError(
                "CUDA active plus inactive-split bytes exceed reserved bytes"
            )
        device_free, device_total = torch.cuda.mem_get_info(self._cuda_device)
        device_free = int(device_free)
        device_total = int(device_total)
        if device_total <= 0 or not 0 <= device_free <= device_total:
            raise ValueError("CUDA mem_get_info returned invalid device bytes")
        physical_used = device_total - device_free
        external = max(0, physical_used - reserved)
        joint_pressure = active + inactive_split + external
        joint_headroom = max(0, device_total - joint_pressure)
        sampling_wall_seconds = time.perf_counter() - started
        stack = (
            list(measurement_stack)
            if measurement_stack is not None
            else [measurement.name for measurement in self._measurement_stack]
        )
        sample = {
            "sample_index": self._cuda_joint_pressure_sample_count,
            "point": point,
            "measurement_stack": stack,
            "elapsed_since_trace_start_seconds": started - self._started,
            "sampling_wall_seconds": sampling_wall_seconds,
            "active_bytes": active,
            "inactive_split_bytes": inactive_split,
            "external_device_bytes": external,
            "reserved_bytes": reserved,
            "device_free_bytes": device_free,
            "device_total_bytes": device_total,
            "joint_pressure_bytes": joint_pressure,
            "joint_headroom_bytes": joint_headroom,
        }
        self._cuda_joint_pressure_sample_count += 1
        self._cuda_joint_pressure_sampling_wall_seconds += sampling_wall_seconds
        self._cuda_joint_pressure_max_external_bytes = max(
            self._cuda_joint_pressure_max_external_bytes, external
        )
        limiting = self._cuda_joint_pressure_limiting_sample
        if limiting is None or joint_pressure > int(limiting["joint_pressure_bytes"]):
            self._cuda_joint_pressure_limiting_sample = sample
        self._cuda_joint_pressure_top_samples.append(sample)
        self._cuda_joint_pressure_top_samples.sort(
            key=lambda item: (
                -int(item["joint_pressure_bytes"]),
                int(item["sample_index"]),
            )
        )
        del self._cuda_joint_pressure_top_samples[CUDA_DENSE_JOINT_TOP_SAMPLE_LIMIT:]
        return metrics

    @staticmethod
    def _peak_values(metrics: Mapping[str, int | float]) -> dict[str, int]:
        return {
            name: int(metrics[name])
            for name in (
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "peak_active_bytes",
                "peak_inactive_split_bytes",
            )
        }

    @staticmethod
    def _merge_peak(target: dict[str, int], values: Mapping[str, int]) -> None:
        for name, value in values.items():
            target[name] = max(target.get(name, 0), int(value))

    def _update_overall_peak(self, metrics: Mapping[str, int | float]) -> None:
        self._merge_peak(self._cuda_overall_peak, self._peak_values(metrics))

    def _checkpoint_cuda(self, *, point: str) -> dict[str, int | float]:
        """Capture peaks before a reset, including them in all nested calls."""

        metrics = self._cuda_metrics(point=point)
        peaks = self._peak_values(metrics)
        self._update_overall_peak(metrics)
        for measurement in self._measurement_stack:
            self._merge_peak(measurement._cuda_peak, peaks)
        return metrics

    def _reset_cuda_peaks(self) -> None:
        if self._cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(self._cuda_device)

    @staticmethod
    def _allocator_delta(
        start: Mapping[str, int | float], end: Mapping[str, int | float]
    ) -> dict[str, int]:
        return {
            name: int(end[name]) - int(start[name])
            for name in ("num_alloc_retries", "num_ooms")
        }

    def _finish_cuda_measurement(
        self,
        measurement: StageMeasurement,
        *,
        failed: bool,
    ) -> None:
        if self._cuda_device is None or measurement._cuda_start is None:
            return
        try:
            end = self._checkpoint_cuda(point=f"measurement_finish:{measurement.name}")
            measurement.cuda_memory = {
                "schema_version": CUDA_MEMORY_SCHEMA_VERSION,
                "start": measurement._cuda_start,
                "end": end,
                "peak": measurement._cuda_peak,
                "allocator_activity_delta": self._allocator_delta(
                    measurement._cuda_start, end
                ),
            }
        except Exception as error:
            if not failed:
                raise
            # Preserve the original CUDA/tracing exception while retaining an
            # explicit diagnostic that telemetry could not be finalized.
            measurement.cuda_memory = {
                "schema_version": CUDA_MEMORY_SCHEMA_VERSION,
                "measurement_error": f"{type(error).__name__}: {error}",
            }

    def measurement_start(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        synchronize: bool = True,
    ) -> StageMeasurement:
        """Start one stage call for code that cannot naturally use ``with``."""

        if synchronize:
            self._synchronize()
        if self.cuda_memory_telemetry:
            self._checkpoint_cuda(point=f"before_measurement_start:{name}")
            self._reset_cuda_peaks()
        measurement = StageMeasurement(
            name=name,
            metadata=dict(metadata or {}),
            _started=time.perf_counter(),
        )
        if self.cuda_memory_telemetry:
            measurement._cuda_start = self._cuda_metrics(
                point=f"measurement_start:{name}",
                measurement_stack=[
                    *(item.name for item in self._measurement_stack),
                    name,
                ],
            )
            measurement._cuda_peak = self._peak_values(measurement._cuda_start)
        self._measurement_stack.append(measurement)
        return measurement

    def measurement_finish(
        self,
        measurement: StageMeasurement,
        *,
        failed: bool = False,
        synchronize: bool = True,
    ) -> StageMeasurement:
        """Finish a stage call and return its wall/memory measurement."""

        if measurement.wall_seconds is not None:
            raise RuntimeError("stage measurement was already finished")
        if synchronize:
            self._synchronize()
        measurement.failed = bool(failed)
        measurement.wall_seconds = time.perf_counter() - measurement._started
        self._finish_cuda_measurement(measurement, failed=failed)
        if (
            not self._measurement_stack
            or self._measurement_stack[-1] is not measurement
        ):
            raise RuntimeError("stage measurements must finish in nested order")
        self._measurement_stack.pop()
        if self.cuda_memory_telemetry:
            try:
                self._reset_cuda_peaks()
            except Exception:
                if not failed:
                    raise
        self._record_measurement(measurement)
        return measurement

    def _fail_measurements_through(self, measurement: StageMeasurement) -> None:
        """Best-effort LIFO unwind that never replaces a tracing exception."""

        while self._measurement_stack:
            current = self._measurement_stack[-1]
            try:
                self.measurement_finish(current, failed=True, synchronize=False)
            except Exception:
                # measurement_finish normally preserves failures itself. If an
                # allocator or recorder failure still escapes, force progress
                # through the stack so the original tracing error wins.
                if self._measurement_stack and self._measurement_stack[-1] is current:
                    self._measurement_stack.pop()
            if current is measurement:
                return

    def record_stage(
        self, name: str, wall_seconds: float, *, failed: bool = False
    ) -> None:
        stage = self._stages.setdefault(
            name, {"wall_seconds": 0.0, "calls": 0, "failed_calls": 0}
        )
        stage["wall_seconds"] = float(stage["wall_seconds"]) + float(wall_seconds)
        stage["calls"] = int(stage["calls"]) + 1
        stage["failed_calls"] = int(stage["failed_calls"]) + int(failed)

    @contextmanager
    def measure_stage(
        self,
        name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        synchronize: bool = True,
    ) -> Iterator[StageMeasurement]:
        """Measure one possibly nested call and return its completed record."""

        measurement = self.measurement_start(
            name, metadata=metadata, synchronize=synchronize
        )
        try:
            yield measurement
        except BaseException:
            # Never synchronize while unwinding: a CUDA error here could mask
            # the original tracing failure. This partial duration is host wall
            # time and is intentionally not promoted to synchronized timing.
            self._fail_measurements_through(measurement)
            raise
        else:
            self.measurement_finish(measurement, synchronize=synchronize)

    def _record_measurement(self, measurement: StageMeasurement) -> None:
        if measurement.wall_seconds is None:
            raise RuntimeError("cannot record an unfinished stage measurement")
        self.record_stage(
            measurement.name,
            measurement.wall_seconds,
            failed=measurement.failed,
        )
        if measurement.cuda_memory is not None:
            stage = self._stages[measurement.name]
            calls = stage.setdefault("call_measurements", [])
            if not isinstance(calls, list):
                raise RuntimeError("stage call_measurements is not a list")
            calls.append(
                {
                    "call_index": len(calls),
                    "failed": measurement.failed,
                    "wall_seconds": measurement.wall_seconds,
                    "metadata": measurement.metadata,
                    "cuda_memory": measurement.cuda_memory,
                }
            )
            peak = measurement.cuda_memory.get("peak")
            if isinstance(peak, Mapping):
                aggregate = stage.setdefault("cuda_memory_peak", {})
                if not isinstance(aggregate, dict):
                    raise RuntimeError("stage cuda_memory_peak is not an object")
                self._merge_peak(aggregate, peak)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        with self.measure_stage(name):
            yield

    def set_counter(self, name: str, value: Any) -> None:
        self._counters[name] = value

    def increment_counter(self, name: str, value: int | float = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def set_early_predictors(self, values: Mapping[str, Any]) -> None:
        self._early_predictors.update(values)

    def record_layer(self, layer: int, **values: Any) -> None:
        record = self._layers.setdefault(int(layer), {"layer": int(layer)})
        record.update(values)

    def append_layer_record(self, layer: int, name: str, **values: Any) -> None:
        """Append one ordered execution record without overwriting earlier calls."""

        record = self._layers.setdefault(int(layer), {"layer": int(layer)})
        series = record.setdefault(name, [])
        if not isinstance(series, list):
            raise RuntimeError(f"layer field {name!r} is not an appendable record list")
        series.append(values)

    def append_execution_record(self, name: str, **values: Any) -> None:
        """Append one ordered run-level execution record."""

        self._execution_records.setdefault(name, []).append(values)

    def record_layer_pair(self, src_layer: int, tgt_layer: int, **values: Any) -> None:
        record = {"src_layer": int(src_layer), "tgt_layer": int(tgt_layer), **values}
        self._layer_pairs.append(record)
        self.increment_counter("cross_layer_pair_count")
        aggregate_fields = {
            "candidate_edges": "cross_layer_candidate_edge_count",
            "target_chunks_per_pass": "cross_layer_target_chunks_per_pass",
            "target_chunk_executions": "cross_layer_target_chunk_executions",
            "retained_edges": "cross_layer_retained_edge_count",
        }
        for record_field, counter_name in aggregate_fields.items():
            if record_field in record:
                self.increment_counter(counter_name, record[record_field])

    def elapsed_seconds(self) -> float:
        self._synchronize()
        return time.perf_counter() - self._started

    def snapshot(self) -> dict[str, Any]:
        cuda_memory = None
        if self.cuda_memory_telemetry:
            current = self._checkpoint_cuda(point="instrumentation_snapshot")
            if self._cuda_overall_start is None:
                raise RuntimeError("CUDA telemetry lacks its initial measurement")
            cuda_memory = {
                "schema_version": CUDA_MEMORY_SCHEMA_VERSION,
                "reset_safe_peak_semantics": "max_across_public_allocator_peak_windows_v1",
                "overall": {
                    "start": self._cuda_overall_start,
                    "current": current,
                    "peak": self._cuda_overall_peak,
                    "allocator_activity_delta": self._allocator_delta(
                        self._cuda_overall_start, current
                    ),
                },
                **(
                    {
                        "dense_joint_pressure": {
                            "schema_version": (
                                CUDA_DENSE_JOINT_PRESSURE_SCHEMA_VERSION
                            ),
                            "sampling_version": CUDA_DENSE_JOINT_SAMPLING_VERSION,
                            "sampling_semantics": (CUDA_DENSE_JOINT_SAMPLING_SEMANTICS),
                            "sample_count": self._cuda_joint_pressure_sample_count,
                            "cumulative_sampling_wall_seconds": (
                                self._cuda_joint_pressure_sampling_wall_seconds
                            ),
                            "max_sampled_external_device_bytes": (
                                self._cuda_joint_pressure_max_external_bytes
                            ),
                            "limiting_sample": (
                                self._cuda_joint_pressure_limiting_sample
                            ),
                            "top_pressure_samples": (
                                self._cuda_joint_pressure_top_samples
                            ),
                        }
                    }
                    if self.cuda_dense_joint_pressure_telemetry
                    else {}
                ),
            }
        snapshot = _json_safe(
            {
                "schema_version": SCHEMA_VERSION,
                "timing_semantics": (
                    "cuda_synchronized_wall_v1"
                    if self.synchronize_cuda
                    else "host_wall_v1"
                ),
                "failed_stage_timing_semantics": "unsynchronized_host_wall_v1",
                "device": self.device,
                # Snapshotting must remain safe in an OOM/error handler. Stage
                # boundaries already synchronize when requested.
                "elapsed_seconds": time.perf_counter() - self._started,
                "stages": self._stages,
                "counters": self._counters,
                "early_predictors": self._early_predictors,
                "layers": [self._layers[layer] for layer in sorted(self._layers)],
                "layer_pairs": self._layer_pairs,
                "execution_records": self._execution_records,
                **({"cuda_memory": cuda_memory} if cuda_memory is not None else {}),
                **(
                    {
                        "cuda_allocator_snapshots": {
                            "schema_version": (CUDA_ALLOCATOR_SNAPSHOT_SCHEMA_VERSION),
                            "capture_semantics": {
                                "allocator_history_enabled_by_instrumentation": False,
                                "raw_snapshot_retained": False,
                                "explicit_cuda_synchronization": False,
                                "active_pending_free_raw_state_aliases": [
                                    "active_pending_free",
                                    "active_awaiting_free",
                                ],
                                "mixed_segment_inactive_bytes_semantics": (
                                    "fragmentation_risk_diagnostic_reuse_and_release_"
                                    "are_allocator_policy_dependent_v1"
                                ),
                                "largest_inactive_block_semantics": (
                                    "diagnostic_not_allocation_success_guarantee_v1"
                                ),
                            },
                            "captures": self._cuda_allocator_snapshots,
                        }
                    }
                    if self.cuda_allocator_snapshot_telemetry
                    else {}
                ),
            }
        )
        # Keep the guarantee local rather than relying on the artifact writer.
        json.dumps(snapshot, allow_nan=False)
        return snapshot


def instrumentation_stage(instrumentation: TraceInstrumentation | None, name: str):
    """Return a measured context that becomes a no-op without a recorder."""

    if instrumentation is None:
        return nullcontext()
    return instrumentation.measure_stage(name)


def cuda_memory_instrumentation_stage(
    instrumentation: TraceInstrumentation | None,
    name: str,
    *,
    metadata: Mapping[str, Any],
):
    """Return a measured stage only when fine-grained CUDA telemetry is enabled."""

    if instrumentation is None or not instrumentation.cuda_memory_telemetry:
        return nullcontext()
    return instrumentation.measure_stage(name, metadata=metadata)


def cuda_memory_observation_stage(
    instrumentation: TraceInstrumentation | None,
    name: str,
    *,
    metadata: Mapping[str, Any],
):
    """Observe a CUDA allocation lifetime without adding synchronization.

    These stages preserve the runner's existing CUDA scheduling while recording
    allocator start/end/peak state. Their host wall time is enqueue time, not a
    synchronized GPU duration; callers that need GPU timing should use
    :func:`cuda_memory_instrumentation_stage`.
    """

    if instrumentation is None or not instrumentation.cuda_memory_telemetry:
        return nullcontext()
    observation_metadata = {
        **metadata,
        "timing_semantics": "host_enqueue_wall_v1",
        "synchronizes_cuda": False,
    }
    return instrumentation.measure_stage(
        name,
        metadata=observation_metadata,
        synchronize=False,
    )


def record_cuda_allocator_snapshot(
    instrumentation: TraceInstrumentation | None,
    point: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    once: bool = False,
    once_key: Any = None,
) -> dict[str, Any] | None:
    """Record a compact allocator snapshot or no-op when it is not enabled."""

    if instrumentation is None:
        return None
    return instrumentation.record_cuda_allocator_snapshot(
        point,
        metadata=metadata,
        once=once,
        once_key=once_key,
    )


def record_selection_predictors(
    instrumentation: TraceInstrumentation | None,
    neuron_cfg: Mapping[int, list[list[int]]],
    *,
    keep_tokens: list[int],
    start_layer: int,
    end_layer: int,
    selected_attribution_chunk_size: int,
    jacobian_target_chunk_size: int,
    use_stop_grad_on_mlps: bool,
    ig_steps: int | None,
) -> None:
    """Record pre-Jacobian workload predictors from the selected neuron mask."""

    if instrumentation is None:
        return
    if selected_attribution_chunk_size <= 0 or jacobian_target_chunk_size <= 0:
        raise ValueError("attribution and Jacobian chunk sizes must be positive")
    keep = set(keep_tokens)
    selected_by_layer: dict[int, list[list[int]]] = {
        int(layer): [pair for pair in pairs if int(pair[0]) in keep]
        for layer, pairs in neuron_cfg.items()
    }
    selected_counts = {layer: len(pairs) for layer, pairs in selected_by_layer.items()}
    # Match _get_cl_ja_based_edges exactly: range(tgt - 1, start_layer,
    # -1) excludes start_layer itself as a source.
    pair_eligible_layers = sorted(
        layer
        for layer, count in selected_counts.items()
        if count > 0 and start_layer < layer < end_layer
    )
    by_token: dict[int, int] = {}
    for layer, pairs in selected_by_layer.items():
        token_positions = set()
        for token, _neuron in pairs:
            token = int(token)
            token_positions.add(token)
            by_token[token] = by_token.get(token, 0) + 1
        count = selected_counts[layer]
        instrumentation.record_layer(
            layer,
            selected_neuron_count=count,
            selected_token_count=len(token_positions),
            selected_attribution_chunks_per_pass=(
                math.ceil(count / selected_attribution_chunk_size) if count else 0
            ),
        )

    candidate_edges = 0
    jacobian_target_chunks_per_pass = 0
    eligible_pairs = []
    for tgt_index, tgt_layer in enumerate(pair_eligible_layers):
        tgt_count = selected_counts[tgt_layer]
        for src_layer in pair_eligible_layers[:tgt_index]:
            eligible_pairs.append({"src_layer": src_layer, "tgt_layer": tgt_layer})
            candidate_edges += selected_counts[src_layer] * tgt_count
            jacobian_target_chunks_per_pass += math.ceil(
                tgt_count / jacobian_target_chunk_size
            )

    selected_total = sum(selected_counts.values())
    selected_chunks_per_pass = sum(
        math.ceil(count / selected_attribution_chunk_size)
        for count in selected_counts.values()
        if count
    )
    ig_execution_count = ig_steps + 1 if ig_steps is not None else 1
    stop_grad_chunk_size = 10
    stop_grad_chunks = (
        sum(
            math.ceil(count / stop_grad_chunk_size)
            for count in selected_counts.values()
            if count
        )
        if use_stop_grad_on_mlps
        else 0
    )
    predictors = {
        "selected_neuron_count": selected_total,
        "selected_neuron_counts_by_layer": [
            {"layer": layer, "count": selected_counts[layer]}
            for layer in sorted(selected_counts)
        ],
        "selected_neuron_counts_by_token": [
            {"token_position": token, "count": by_token[token]}
            for token in sorted(by_token)
        ],
        "active_layer_count": len(pair_eligible_layers),
        "active_layer_min": pair_eligible_layers[0] if pair_eligible_layers else None,
        "active_layer_max": pair_eligible_layers[-1] if pair_eligible_layers else None,
        "active_layer_span": (
            pair_eligible_layers[-1] - pair_eligible_layers[0] + 1
            if pair_eligible_layers
            else 0
        ),
        "pair_eligible_layers": pair_eligible_layers,
        "planned_active_layer_pairs": eligible_pairs,
        "planned_active_layer_pair_count": len(eligible_pairs),
        "candidate_mlp_edge_count": candidate_edges,
        "jacobian_target_chunk_size": jacobian_target_chunk_size,
        "jacobian_target_chunks_per_pass": jacobian_target_chunks_per_pass,
        "jacobian_pass_count": ig_execution_count,
        "planned_jacobian_target_chunk_executions": (
            jacobian_target_chunks_per_pass * ig_execution_count
        ),
        "selected_attribution_chunk_size": selected_attribution_chunk_size,
        "selected_attribution_chunks_per_pass": selected_chunks_per_pass,
        "selected_attribution_pass_count": ig_execution_count,
        "selected_attribution_chunk_executions": (
            selected_chunks_per_pass * ig_execution_count
        ),
        "stop_grad_mlp_attribution_enabled": use_stop_grad_on_mlps,
        "stop_grad_mlp_attribution_chunk_size": stop_grad_chunk_size,
        "stop_grad_mlp_attribution_chunks_per_pass": stop_grad_chunks,
        "stop_grad_mlp_attribution_pass_count": 1 if use_stop_grad_on_mlps else 0,
        "stop_grad_mlp_attribution_chunk_executions": stop_grad_chunks,
        "total_selected_attribution_chunk_executions": (
            selected_chunks_per_pass * ig_execution_count + stop_grad_chunks
        ),
        "ig_steps": ig_steps,
        "ig_execution_count": ig_execution_count,
        "early_predictors_ready_seconds": instrumentation.elapsed_seconds(),
    }
    instrumentation.set_early_predictors(predictors)
    for name, value in predictors.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            instrumentation.set_counter(name, value)
