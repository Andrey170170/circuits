"""Atomic, compact persistence for reusable teacher-forced trace units."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from circuits.tracing.trace import CircuitData, TopKPositionTrace

SCHEMA_VERSION = "adag.compact-trace.v1"
TOPK_SCHEMA_VERSION = "adag.compact-trace.topk-position.v1"
DATA_FILENAME = "circuit_data.pkl.gz"
MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True)
class CompactTraceArtifact:
    """Loaded compact trace and its lightweight metadata."""

    path: Path
    circuit_data: CircuitData
    manifest: dict[str, Any]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TopKCompactTraceArtifact:
    """Loaded candidate-axis trace and its lightweight metadata."""

    path: Path
    topk_trace: TopKPositionTrace
    manifest: dict[str, Any]
    metrics: dict[str, Any]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_count(data: CircuitData) -> int:
    if len(data.target_logits) != 1:
        raise ValueError("compact trace units require exactly one batch item")
    return len(data.target_logits[0])


def _validate_scalar_columns(
    frame: pd.DataFrame,
    *,
    frame_name: str,
    required_columns: tuple[str, ...],
) -> None:
    """Validate scalar scientific outputs without rejecting an empty graph.

    A fully pruned graph is a meaningful result, so zero rows are allowed.  Its
    dataframe must still carry the expected schema.  List-valued attribution
    maps are intentionally not checked here because they may contain ``None``
    for unavailable source positions.
    """

    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{frame_name} must be a pandas DataFrame")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{frame_name} is missing required numeric columns: {missing}"
        )
    for column in required_columns:
        if frame.empty:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.map(math.isfinite)
        if invalid.any():
            bad_rows = frame.index[invalid].tolist()[:5]
            raise ValueError(
                f"{frame_name}.{column} contains non-finite or non-numeric "
                f"values at rows {bad_rows}"
            )


def _validate_nested_finite(
    values: object, *, field_name: str, target_count: int
) -> None:
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"{field_name} must contain exactly one batch item")
    row = values[0]
    if not isinstance(row, list) or len(row) != target_count:
        raise ValueError(
            f"{field_name} must contain exactly one value per traced target"
        )
    for index, value in enumerate(row):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name}[0][{index}] is not numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name}[0][{index}] is not finite")


def _validate_candidate_contribution_maps(
    frame: pd.DataFrame, *, candidate_count: int
) -> None:
    if "contrib_map" not in frame.columns:
        raise ValueError("top-k df_node is missing contrib_map")
    for row_index, value in frame["contrib_map"].items():
        if value is None:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != candidate_count:
            raise ValueError(
                "top-k df_node.contrib_map must match candidate_count "
                f"at row {row_index}"
            )
        for candidate_index, contribution in enumerate(value):
            if (
                isinstance(contribution, bool)
                or not isinstance(contribution, (int, float))
                or not math.isfinite(float(contribution))
            ):
                raise ValueError(
                    "top-k df_node.contrib_map contains an invalid value at "
                    f"row {row_index}, candidate {candidate_index}"
                )


def validate_compact_trace_data(data: CircuitData) -> int:
    """Fail closed if a trace cannot be persisted as numerical evidence."""

    target_count = _validate_reuse_scope(data)
    _validate_scalar_columns(
        data.df_node,
        frame_name="df_node",
        required_columns=("attribution", "activation"),
    )
    _validate_scalar_columns(
        data.df_edge,
        frame_name="df_edge",
        required_columns=("attribution", "weight"),
    )
    _validate_nested_finite(
        data.target_logit_probs,
        field_name="target_logit_probs",
        target_count=target_count,
    )
    _validate_nested_finite(
        data.target_logit_values,
        field_name="target_logit_values",
        target_count=target_count,
    )
    return target_count


def validate_topk_trace_data(trace: TopKPositionTrace) -> int:
    """Fail closed if a same-position candidate trace violates its schema."""

    if not isinstance(trace, TopKPositionTrace):
        raise TypeError("top-k trace payload is not TopKPositionTrace")
    data = trace.circuit_data
    target_count = _target_count(data)
    if target_count != 1 or data.benchmark_only:
        raise ValueError("top-k artifacts require one reusable response target")
    if len(data.target_provenance) != 1:
        raise ValueError("top-k artifacts require exactly one target provenance entry")
    provenance = data.target_provenance[0]
    if provenance.get("response_token_position") != trace.shared_response_position:
        raise ValueError("top-k shared response position disagrees with target provenance")
    if provenance.get("prediction_token_position") != trace.shared_prediction_position:
        raise ValueError(
            "top-k shared prediction position disagrees with target provenance"
        )

    candidates = trace.candidate_selection.candidates
    candidate_count = len(candidates)
    if candidate_count < 1:
        raise ValueError("top-k artifacts require at least one candidate")
    if [candidate.candidate_index for candidate in candidates] != list(
        range(candidate_count)
    ):
        raise ValueError("top-k candidate indices must be contiguous and zero-based")
    token_ids = [candidate.token_id for candidate in candidates]
    if len(set(token_ids)) != candidate_count:
        raise ValueError("top-k candidate token IDs must be unique")
    if any(candidate.full_distribution_rank < 1 for candidate in candidates):
        raise ValueError("top-k candidate ranks must be one-based positive integers")
    for candidate in candidates:
        if not math.isfinite(candidate.logit) or not math.isfinite(
            candidate.probability
        ):
            raise ValueError("top-k candidate scores must be finite")
        if not 0.0 <= candidate.probability <= 1.0:
            raise ValueError("top-k candidate probability is outside [0, 1]")
        if candidate.is_observed != (
            candidate.token_id == trace.candidate_selection.observed_token_id
        ):
            raise ValueError("top-k observed-token membership is inconsistent")

    policy_id = trace.candidate_selection.policy_id
    if policy_id == "observed_token":
        if candidate_count != 1 or not candidates[0].is_observed:
            raise ValueError("observed_token artifacts require one observed candidate")
    elif policy_id == "observed_plus_top4_alternatives":
        if candidate_count != 5 or not candidates[0].is_observed:
            raise ValueError(
                "observed_plus_top4_alternatives requires observed candidate zero"
            )
        alternatives = list(candidates[1:])
        if alternatives != sorted(
            alternatives, key=lambda candidate: (-candidate.logit, candidate.token_id)
        ):
            raise ValueError("top-k alternatives are not deterministically ordered")
    elif policy_id == "model_top5":
        if candidate_count != 5 or list(candidates) != sorted(
            candidates, key=lambda candidate: (-candidate.logit, candidate.token_id)
        ):
            raise ValueError("model_top5 candidates are not deterministically ordered")
    else:
        raise ValueError(f"unsupported top-k candidate policy: {policy_id!r}")

    observed_token_id = trace.candidate_selection.observed_token_id
    if data.target_logits != [[observed_token_id]]:
        raise ValueError("top-k response target must remain the observed token")
    if provenance.get("token_id") != observed_token_id:
        raise ValueError("top-k observed token disagrees with target provenance")
    if len(trace.joint_objective.candidate_weights) != candidate_count:
        raise ValueError("top-k joint objective width does not match candidates")
    if any(
        not math.isfinite(weight)
        for weight in trace.joint_objective.candidate_weights
    ):
        raise ValueError("top-k joint objective weights must be finite")
    if trace.candidate_contribution_schema.get("width") != candidate_count:
        raise ValueError("top-k contribution schema width does not match candidates")
    if data.trace_metadata.get("candidate_trace_contract") != trace.contract_dict():
        raise ValueError("top-k payload contract disagrees with trace metadata")

    _validate_scalar_columns(
        data.df_node,
        frame_name="df_node",
        required_columns=("attribution", "activation"),
    )
    _validate_scalar_columns(
        data.df_edge,
        frame_name="df_edge",
        required_columns=("attribution", "weight"),
    )
    _validate_candidate_contribution_maps(
        data.df_node, candidate_count=candidate_count
    )
    _validate_nested_finite(
        data.target_logit_probs,
        field_name="target_logit_probs",
        target_count=1,
    )
    _validate_nested_finite(
        data.target_logit_values,
        field_name="target_logit_values",
        target_count=1,
    )
    return candidate_count


def _validate_reuse_scope(data: CircuitData) -> int:
    if data.trace_metadata.get("candidate_trace_contract") is not None:
        raise ValueError(
            "candidate-axis traces require save_topk_compact_trace"
        )
    target_count = _target_count(data)
    if target_count < 1:
        raise ValueError("compact trace units require at least one target")
    if len(data.target_provenance) != target_count:
        raise ValueError(
            "target_provenance must contain exactly one entry per traced target"
        )
    if target_count > 1 and not data.benchmark_only:
        raise ValueError(
            "multi-target CircuitData must be explicitly marked benchmark_only"
        )
    return target_count


def save_compact_trace(
    path: str | os.PathLike[str],
    circuit_data: CircuitData,
    *,
    metrics: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically create a compact trace directory.

    The destination must not already exist. ``pickle`` is intentionally used to
    preserve pandas list-valued columns and ADAG configuration exactly; only
    load artifacts from trusted sources.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"compact trace destination already exists: {target}")
    target_count = validate_compact_trace_data(circuit_data)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        data_path = temporary / DATA_FILENAME
        with gzip.open(data_path, "wb", compresslevel=6) as handle:
            pickle.dump(circuit_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        data_sha256 = _sha256_file(data_path)
        data_size_bytes = data_path.stat().st_size
        canonical_manifest: dict[str, Any] = dict(manifest or {})
        canonical_manifest.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data_file": DATA_FILENAME,
                "data_sha256": data_sha256,
                "data_size_bytes": data_size_bytes,
                "model_id": circuit_data.model_id,
                "target_count": target_count,
                "target_provenance": circuit_data.target_provenance,
                "trace_metadata": circuit_data.trace_metadata,
                "benchmark_only": circuit_data.benchmark_only,
                "numerically_valid": True,
                "node_count": len(circuit_data.df_node),
                "edge_count": len(circuit_data.df_edge),
                "scientifically_reusable": target_count == 1
                and not circuit_data.benchmark_only,
            }
        )
        _write_json(temporary / MANIFEST_FILENAME, canonical_manifest)
        _write_json(temporary / METRICS_FILENAME, dict(metrics or {}))
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def save_topk_compact_trace(
    path: str | os.PathLike[str],
    trace: TopKPositionTrace,
    *,
    metrics: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically create a same-position candidate-axis trace artifact."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"compact trace destination already exists: {target}")
    candidate_count = validate_topk_trace_data(trace)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        data_path = temporary / DATA_FILENAME
        with gzip.open(data_path, "wb", compresslevel=6) as handle:
            pickle.dump(trace, handle, protocol=pickle.HIGHEST_PROTOCOL)

        data_sha256 = _sha256_file(data_path)
        data_size_bytes = data_path.stat().st_size
        canonical_manifest: dict[str, Any] = dict(manifest or {})
        canonical_manifest.update(
            {
                "schema_version": TOPK_SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data_file": DATA_FILENAME,
                "data_sha256": data_sha256,
                "data_size_bytes": data_size_bytes,
                "model_id": trace.circuit_data.model_id,
                "target_count": 1,
                "candidate_count": candidate_count,
                "target_provenance": trace.circuit_data.target_provenance,
                "candidate_trace_contract": trace.contract_dict(),
                "trace_metadata": trace.circuit_data.trace_metadata,
                "benchmark_only": False,
                "numerically_valid": True,
                "node_count": len(trace.circuit_data.df_node),
                "edge_count": len(trace.circuit_data.df_edge),
                "scientifically_reusable": True,
            }
        )
        _write_json(temporary / MANIFEST_FILENAME, canonical_manifest)
        _write_json(temporary / METRICS_FILENAME, dict(metrics or {}))
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def validate_compact_trace_integrity(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Lightweight schema and checksum validation without unpickling payloads."""

    artifact_path = Path(path)
    if not artifact_path.is_dir():
        raise ValueError(f"compact trace path is not a directory: {artifact_path}")
    manifest_path = artifact_path / MANIFEST_FILENAME
    metrics_path = artifact_path / METRICS_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"compact trace manifest is missing: {manifest_path}")
    if not metrics_path.is_file():
        raise ValueError(f"compact trace metrics are missing: {metrics_path}")
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"compact trace metadata is unreadable: {artifact_path}") from error
    if not isinstance(manifest, dict) or not isinstance(metrics, dict):
        raise ValueError("compact trace manifest and metrics must be JSON objects")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported compact trace schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("data_file") != DATA_FILENAME:
        raise ValueError(f"compact trace data_file must be {DATA_FILENAME!r}")
    data_path = artifact_path / DATA_FILENAME
    if not data_path.is_file():
        raise ValueError(f"compact trace data file is missing: {data_path}")
    expected_size = manifest.get("data_size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise ValueError("compact trace data_size_bytes is invalid")
    actual_size = data_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError("compact trace data size mismatch")
    expected_hash = manifest.get("data_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("compact trace data_sha256 is invalid")
    if _sha256_file(data_path) != expected_hash:
        raise ValueError("compact trace data checksum mismatch")
    return manifest


def validate_topk_compact_trace_integrity(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate the top-k schema and checksum without unpickling the payload."""

    artifact_path = Path(path)
    if not artifact_path.is_dir():
        raise ValueError(f"compact trace path is not a directory: {artifact_path}")
    manifest_path = artifact_path / MANIFEST_FILENAME
    metrics_path = artifact_path / METRICS_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"compact trace manifest is missing: {manifest_path}")
    if not metrics_path.is_file():
        raise ValueError(f"compact trace metrics are missing: {metrics_path}")
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"compact trace metadata is unreadable: {artifact_path}") from error
    if not isinstance(manifest, dict) or not isinstance(metrics, dict):
        raise ValueError("compact trace manifest and metrics must be JSON objects")
    if manifest.get("schema_version") != TOPK_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported top-k compact trace schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("data_file") != DATA_FILENAME:
        raise ValueError(f"compact trace data_file must be {DATA_FILENAME!r}")
    data_path = artifact_path / DATA_FILENAME
    if not data_path.is_file():
        raise ValueError(f"compact trace data file is missing: {data_path}")
    expected_size = manifest.get("data_size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise ValueError("compact trace data_size_bytes is invalid")
    if data_path.stat().st_size != expected_size:
        raise ValueError("compact trace data size mismatch")
    expected_hash = manifest.get("data_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("compact trace data_sha256 is invalid")
    if _sha256_file(data_path) != expected_hash:
        raise ValueError("compact trace data checksum mismatch")
    return manifest


def load_compact_trace(path: str | os.PathLike[str]) -> CompactTraceArtifact:
    """Load and integrity-check a trusted compact trace directory."""
    artifact_path = Path(path)
    manifest = validate_compact_trace_integrity(artifact_path)
    with (artifact_path / METRICS_FILENAME).open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    data_path = artifact_path / DATA_FILENAME
    with gzip.open(data_path, "rb") as handle:
        circuit_data = pickle.load(handle)
    if not isinstance(circuit_data, CircuitData):
        raise TypeError("compact trace payload is not CircuitData")
    validate_compact_trace_data(circuit_data)
    return CompactTraceArtifact(
        path=artifact_path,
        circuit_data=circuit_data,
        manifest=manifest,
        metrics=metrics,
    )


def load_topk_compact_trace(
    path: str | os.PathLike[str],
) -> TopKCompactTraceArtifact:
    """Load and integrity-check a trusted same-position candidate trace."""

    artifact_path = Path(path)
    manifest = validate_topk_compact_trace_integrity(artifact_path)
    with (artifact_path / METRICS_FILENAME).open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    with gzip.open(artifact_path / DATA_FILENAME, "rb") as handle:
        trace = pickle.load(handle)
    if not isinstance(trace, TopKPositionTrace):
        raise TypeError("top-k compact trace payload is not TopKPositionTrace")
    validate_topk_trace_data(trace)
    return TopKCompactTraceArtifact(
        path=artifact_path,
        topk_trace=trace,
        manifest=manifest,
        metrics=metrics,
    )
