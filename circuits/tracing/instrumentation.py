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
from typing import Any

import torch

SCHEMA_VERSION = "adag.trace-instrumentation.v1"


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
    ) -> None:
        self.device = str(device) if device is not None else None
        self.synchronize_cuda = bool(synchronize_cuda)
        self._started = time.perf_counter()
        self._stages: dict[str, dict[str, float | int]] = {}
        self._counters: dict[str, Any] = {}
        self._early_predictors: dict[str, Any] = {}
        self._layers: dict[int, dict[str, Any]] = {}
        self._layer_pairs: list[dict[str, Any]] = []

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
    def stage(self, name: str) -> Iterator[None]:
        started = self.timer_start()
        try:
            yield
        except BaseException:
            # Never synchronize while unwinding: a CUDA error here could mask
            # the original tracing failure. This partial duration is host wall
            # time and is intentionally not promoted to synchronized timing.
            self.record_stage(name, time.perf_counter() - started, failed=True)
            raise
        else:
            self.record_stage(name, self.timer_finish(started))

    def set_counter(self, name: str, value: Any) -> None:
        self._counters[name] = value

    def increment_counter(self, name: str, value: int | float = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def set_early_predictors(self, values: Mapping[str, Any]) -> None:
        self._early_predictors.update(values)

    def record_layer(self, layer: int, **values: Any) -> None:
        record = self._layers.setdefault(int(layer), {"layer": int(layer)})
        record.update(values)

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
            }
        )
        # Keep the guarantee local rather than relying on the artifact writer.
        json.dumps(snapshot, allow_nan=False)
        return snapshot


def instrumentation_stage(instrumentation: TraceInstrumentation | None, name: str):
    """Return a timing context that becomes a no-op without a recorder."""

    if instrumentation is None:
        return nullcontext()
    return instrumentation.stage(name)


def record_selection_predictors(
    instrumentation: TraceInstrumentation | None,
    neuron_cfg: Mapping[int, list[list[int]]],
    *,
    keep_tokens: list[int],
    start_layer: int,
    end_layer: int,
    selected_attribution_chunk_size: int,
    use_stop_grad_on_mlps: bool,
    ig_steps: int | None,
) -> None:
    """Record pre-Jacobian workload predictors from the selected neuron mask."""

    if instrumentation is None:
        return
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
                tgt_count / selected_attribution_chunk_size
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
        "jacobian_target_chunk_size": selected_attribution_chunk_size,
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
