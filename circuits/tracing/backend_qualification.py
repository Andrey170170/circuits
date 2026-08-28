"""Numerical qualification for two saved top-k execution traces.

This module compares trusted compact artifacts from the same frozen work item.
It deliberately reports bounded implementation drift rather than asserting
scientific parity. Scientific identity fields are never configurable; only
named execution strategies, runtime, and code-revision identity differences
may be explicitly allow-listed by callers. Historical attention-backend
reports keep their original schema; contribution-execution comparisons use a
broader execution-qualification schema.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from circuits.tracing.artifact import (
    TopKCompactTraceArtifact,
    load_topk_compact_trace,
)

REPORT_SCHEMA = "bonafide-attention-backend-qualification/v1"
EXECUTION_REPORT_SCHEMA = "bonafide-execution-qualification/v1"
TOLERANCE_GROUPS = ("target", "node", "edge", "candidate_profile")
CUDA_ALLOCATOR_SNAPSHOT_TELEMETRY_IDENTITY_PATH = (
    "artifact_identity.instrumentation.cuda_allocator_snapshot_telemetry"
)
CUDA_ALLOCATOR_AB_IDENTITY_PATHS = (
    "artifact_identity.cuda_allocator_policy",
    "artifact_identity.runtime_environment.cuda_allocator_policy.intended_policy_id",
    "artifact_identity.runtime_environment.cuda_allocator_policy."
    "observed_environment.value",
    "artifact_identity.runtime_environment.cuda_allocator_policy."
    "observed_environment.is_set",
)
CUDA_ALLOCATOR_AB_POLICIES = ("default_v1", "expandable_segments_v1")
EMBEDDING_EDGE_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config.embedding_edge_materialization",
)
EMBEDDING_EDGE_AB_STRATEGIES = ("scalar_v1", "vectorized_v1")
CROSS_LAYER_JACOBIAN_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config.cross_layer_jacobian_execution",
)
CROSS_LAYER_JACOBIAN_AB_STRATEGIES = ("full_model_v1", "cached_range_v1")
CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config.stop_gradient_contribution_target_lane_chunk_size",
)
SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config.selected_neuron_contribution_target_lane_chunk_size",
)
SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_STRATEGIES = (None, 1)
SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config.selected_attribution_neuron_lane_chunk_size",
)
SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_STRATEGIES = (None, 1)
SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_RESOLVED_WIDTHS = (50, 1)
SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config.selected_embed_contribution_target_lane_chunk_size",
)
SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_PROFILES = {
    "full_width_exact_v1": (None, 5),
    "width_one_bf16_v1": (5, 1),
}
STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config."
    "stop_gradient_embed_contribution_target_lane_chunk_size",
)
STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_PROFILES = {
    "full_width_exact_v1": (None, 5),
    "width_one_exact_v1": (5, 1),
}
STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config."
    "stop_gradient_selected_attribution_forward_execution",
)
STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_STRATEGIES = (
    "full_model_v1",
    "prefix_stop_v1",
)
STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config.stop_gradient_selected_attribution_storage",
)
STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_STRATEGIES = (
    "graph_retaining_v1",
    "terminal_detached_v1",
)
SELECTED_TARGET_LOGIT_EXECUTION_AB_IDENTITY_PATHS = (
    "artifact_identity.adag_config.selected_target_logit_execution",
)
SELECTED_TARGET_LOGIT_EXECUTION_AB_STRATEGIES = (
    "full_logits_v1",
    "selected_position_logits_v1",
)
CROSS_LAYER_JACOBIAN_RECEIPT_NAMES = (
    "selected_source_activations",
    "selected_target_activations",
    "selected_raw_jacobian",
)

_ALLOWABLE_SCALAR_IDENTITY_RULES = (
    CUDA_ALLOCATOR_SNAPSHOT_TELEMETRY_IDENTITY_PATH,
    "artifact_identity.adag_config.stop_gradient_attention_backend",
    "artifact_identity.adag_config.stop_gradient_contribution_execution",
    "artifact_identity.adag_config.stop_gradient_contribution_target_lane_chunk_size",
    "artifact_identity.adag_config.selected_neuron_contribution_target_lane_chunk_size",
    "artifact_identity.adag_config.selected_embed_contribution_target_lane_chunk_size",
    "artifact_identity.adag_config."
    "stop_gradient_embed_contribution_target_lane_chunk_size",
    "artifact_identity.adag_config.selected_attribution_neuron_lane_chunk_size",
    "artifact_identity.adag_config."
    "stop_gradient_selected_attribution_forward_execution",
    "artifact_identity.adag_config.stop_gradient_selected_attribution_storage",
    "artifact_identity.adag_config.selected_target_logit_execution",
    "artifact_identity.cuda_allocator_policy",
    "artifact_identity.adag_config.embedding_edge_materialization",
    "artifact_identity.adag_config.cross_layer_jacobian_execution",
)
_ALLOWABLE_SUBTREE_IDENTITY_RULES = (
    "artifact_identity.code_revision.",
    "artifact_identity.runtime_environment.",
)
_TRACE_METADATA_IDENTITY_FIELDS = (
    "trace_mode",
    "prompt",
    "prompt_sha256",
    "response",
    "response_sha256",
    "system_prompt",
    "system_prompt_sha256",
    "teacher_forced_serialization_mode",
    "teacher_forced_token_identity",
    "assistant_prefix_token_count",
    "response_token_count",
    "included_response_token_count",
    "input_token_count",
    "chat_template_sha256",
    "frozen_topology",
)


@dataclass(frozen=True)
class NumericTolerance:
    """An explicit allclose-style qualification threshold."""

    absolute: float
    relative: float

    def __post_init__(self) -> None:
        for name, value in (("absolute", self.absolute), ("relative", self.relative)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} tolerance must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        return {"absolute": self.absolute, "relative": self.relative}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return dict(left) == dict(right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=False)
    try:
        value = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(value, (np.ndarray, pd.Series)):
        return bool(np.asarray(value).all())
    return bool(value)


def _recursive_differences(
    reference: Any,
    candidate: Any,
    *,
    path: str,
) -> list[dict[str, Any]]:
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(reference) | set(candidate), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in reference:
                differences.append(
                    {
                        "path": child_path,
                        "kind": "candidate_only",
                        "reference": None,
                        "candidate": _json_value(candidate[key]),
                    }
                )
            elif key not in candidate:
                differences.append(
                    {
                        "path": child_path,
                        "kind": "reference_only",
                        "reference": _json_value(reference[key]),
                        "candidate": None,
                    }
                )
            else:
                differences.extend(
                    _recursive_differences(
                        reference[key], candidate[key], path=child_path
                    )
                )
        return differences
    if (
        isinstance(reference, Sequence)
        and not isinstance(reference, (str, bytes))
        and isinstance(candidate, Sequence)
        and not isinstance(candidate, (str, bytes))
        and len(reference) == len(candidate)
    ):
        differences = []
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            differences.extend(
                _recursive_differences(left, right, path=f"{path}[{index}]")
            )
        return differences
    if _values_equal(reference, candidate):
        return []
    return [
        {
            "path": path,
            "kind": "value",
            "reference": _json_value(reference),
            "candidate": _json_value(candidate),
        }
    ]


def _validate_allowed_identity_paths(paths: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for path in paths:
        if not isinstance(path, str) or not path:
            raise ValueError(
                "allowed identity difference paths must be non-empty strings"
            )
        prefix = path[:-2] if path.endswith(".*") else path
        if path.endswith(".*") and prefix in _ALLOWABLE_SCALAR_IDENTITY_RULES:
            raise ValueError(
                "scalar execution-strategy identity differences must be "
                f"allow-listed by exact path: {path!r}"
            )
        if not (
            prefix in _ALLOWABLE_SCALAR_IDENTITY_RULES
            or any(
                prefix == allowed_prefix.removesuffix(".")
                or prefix.startswith(allowed_prefix)
                for allowed_prefix in _ALLOWABLE_SUBTREE_IDENTITY_RULES
            )
        ):
            raise ValueError(
                "identity difference may only allow the stop-gradient attention "
                "backend, contribution execution, selected-attribution forward "
                "execution or storage, lane chunk size, CUDA "
                "allocator policy or snapshot telemetry, embedding-edge "
                "materialization, cross-layer Jacobian execution, or fields under "
                "code_revision/runtime_environment: "
                f"{path!r}"
            )
        normalized.append(path)
    if len(normalized) != len(set(normalized)):
        raise ValueError("allowed identity difference paths must be unique")
    return tuple(normalized)


def _path_is_allowed(path: str, rules: Sequence[str]) -> bool:
    return any(
        path == rule or (rule.endswith(".*") and path.startswith(rule[:-1]))
        for rule in rules
    )


def _identity_value_checks(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
) -> list[dict[str, Any]]:
    reference_trace = reference.topk_trace
    candidate_trace = candidate.topk_trace
    reference_data = reference_trace.circuit_data
    candidate_data = candidate_trace.circuit_data

    def structural_contract(artifact: TopKCompactTraceArtifact) -> dict[str, Any]:
        contract = artifact.topk_trace.contract_dict()
        candidates = contract["candidate_selection"]["candidates"]
        contract["candidate_selection"]["candidates"] = [
            {
                key: value
                for key, value in item.items()
                if key not in {"logit", "probability"}
            }
            for item in candidates
        ]
        return contract

    def structural_provenance(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                key: item
                for key, item in provenance.items()
                if key not in {"logit", "probability"}
            }
            for provenance in value
        ]

    manifest_fields = (
        "source_width1_artifact_id",
        "source_width1_manifest_sha256",
        "source_target_selection",
        "bonafide_example",
        "model_id",
        "model_revision",
    )
    checks: list[tuple[str, Any, Any, bool]] = [
        *(
            (
                f"manifest.{field}",
                reference.manifest.get(field),
                candidate.manifest.get(field),
                field in reference.manifest and field in candidate.manifest,
            )
            for field in manifest_fields
        ),
        (
            "payload.model_id",
            reference_data.model_id,
            candidate_data.model_id,
            True,
        ),
        (
            "payload.input_token_ids",
            reference_data.cis,
            candidate_data.cis,
            True,
        ),
        (
            "payload.attention_masks",
            reference_data.attention_masks,
            candidate_data.attention_masks,
            True,
        ),
        ("payload.labels", reference_data.labels, candidate_data.labels, True),
        (
            "payload.target_token_ids",
            reference_data.target_logits,
            candidate_data.target_logits,
            True,
        ),
        (
            "payload.target_provenance",
            structural_provenance(reference_data.target_provenance),
            structural_provenance(candidate_data.target_provenance),
            True,
        ),
        (
            "payload.candidate_axis",
            structural_contract(reference),
            structural_contract(candidate),
            True,
        ),
    ]
    for field in _TRACE_METADATA_IDENTITY_FIELDS:
        present = (
            field in reference_data.trace_metadata
            and field in candidate_data.trace_metadata
        ) or (
            field == "frozen_topology"
            and field not in reference_data.trace_metadata
            and field not in candidate_data.trace_metadata
        )
        checks.append(
            (
                f"payload.trace_metadata.{field}",
                reference_data.trace_metadata.get(field),
                candidate_data.trace_metadata.get(field),
                present,
            )
        )
    results = []
    for name, left, right, present in checks:
        exact = present and _values_equal(left, right)
        results.append(
            {
                "field": name,
                "passed": exact,
                "reason": "exact"
                if exact
                else "missing"
                if not present
                else "mismatch",
            }
        )
    return results


def _artifact_identity_comparison(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
    *,
    allowed_paths: Sequence[str],
) -> dict[str, Any]:
    left = reference.manifest.get("artifact_identity")
    right = candidate.manifest.get("artifact_identity")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {
            "passed": False,
            "reason": "artifact_identity missing or not an object",
            "allowed_difference_paths": list(allowed_paths),
            "allowed_differences": [],
            "unallowed_differences": [],
        }
    left = {key: value for key, value in left.items() if key != "sha256"}
    right = {key: value for key, value in right.items() if key != "sha256"}
    differences = _recursive_differences(left, right, path="artifact_identity")
    allowed = [
        difference
        for difference in differences
        if _path_is_allowed(difference["path"], allowed_paths)
    ]
    unallowed = [difference for difference in differences if difference not in allowed]
    return {
        "passed": not unallowed,
        "reason": "all differences explicitly allowed" if not unallowed else "mismatch",
        "allowed_difference_paths": list(allowed_paths),
        "allowed_differences": allowed,
        "unallowed_differences": unallowed,
    }


def _artifact_identity_integrity(
    artifact: TopKCompactTraceArtifact,
) -> dict[str, Any]:
    identity = artifact.manifest.get("artifact_identity")
    if not isinstance(identity, Mapping):
        return {"passed": False, "reason": "artifact_identity missing or not an object"}
    claimed_sha256 = identity.get("sha256")
    identity_without_hash = {
        key: value for key, value in identity.items() if key != "sha256"
    }
    canonical = json.dumps(
        identity_without_hash,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    computed_sha256 = hashlib.sha256(canonical).hexdigest()
    expected_artifact_id = f"topk-trace-{computed_sha256[:24]}"
    artifact_id = artifact.manifest.get("artifact_id")
    passed = claimed_sha256 == computed_sha256 and artifact_id == expected_artifact_id
    return {
        "passed": passed,
        "reason": "exact" if passed else "identity hash or artifact ID mismatch",
        "claimed_sha256": claimed_sha256,
        "computed_sha256": computed_sha256,
        "artifact_id": artifact_id,
        "expected_artifact_id": expected_artifact_id,
    }


def _gpu_identity(artifact: TopKCompactTraceArtifact) -> dict[str, Any]:
    raw = artifact.manifest.get("gpu")
    if not isinstance(raw, Mapping):
        raw = {}
    name = raw.get("name")
    if not isinstance(name, str):
        runtime = artifact.manifest.get("runtime_environment")
        gpu_runtime = (
            runtime.get("gpu_runtime") if isinstance(runtime, Mapping) else None
        )
        devices = (
            gpu_runtime.get("devices") if isinstance(gpu_runtime, Mapping) else None
        )
        first = devices[0] if isinstance(devices, list) and devices else None
        name = first.get("name") if isinstance(first, Mapping) else None
    family = None
    if isinstance(name, str):
        upper = name.upper()
        for known in ("A100", "H200", "H100", "L40S", "V100"):
            if known in upper:
                family = known
                break
        if family is None:
            family = name
    return {
        "model": name,
        "family": family,
        "total_memory_bytes": raw.get("total_memory_bytes"),
        "compute_capability": raw.get("compute_capability"),
    }


def _node_key(row: pd.Series) -> tuple[int, int, int]:
    return int(row["layer"]), int(row["token"]), int(row["neuron"])


def _edge_key(row: pd.Series) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    values: list[tuple[int, int]] = []
    for column in ("layer", "token", "neuron"):
        parts = str(row[column]).split("->")
        if len(parts) != 2:
            raise ValueError(f"invalid edge {column}: {row[column]!r}")
        values.append((int(parts[0]), int(parts[1])))
    return (
        (values[0][0], values[1][0], values[2][0]),
        (values[0][1], values[1][1], values[2][1]),
    )


def _keyed_rows(
    frame: pd.DataFrame,
    key_function,
    *,
    name: str,
) -> dict[Any, pd.Series]:
    rows: dict[Any, pd.Series] = {}
    for _, row in frame.iterrows():
        key = key_function(row)
        if key in rows:
            raise ValueError(f"{name} contains duplicate key: {key!r}")
        rows[key] = row
    return rows


def _key_json(key: Any) -> Any:
    if isinstance(key, tuple):
        return [_key_json(item) for item in key]
    return key


def _topology_summary(reference: set[Any], candidate: set[Any]) -> dict[str, Any]:
    intersection = reference & candidate
    union = reference | candidate
    reference_only = sorted(reference - candidate)
    candidate_only = sorted(candidate - reference)
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "reference_only_count": len(reference_only),
        "candidate_only_count": len(candidate_only),
        "exact": reference == candidate,
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "reference_only_examples": [_key_json(key) for key in reference_only[:20]],
        "candidate_only_examples": [_key_json(key) for key in candidate_only[:20]],
        "examples_truncated": len(reference_only) > 20 or len(candidate_only) > 20,
    }


def _numeric_summary(
    reference: Sequence[float],
    candidate: Sequence[float],
    tolerance: NumericTolerance | None,
) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("numeric comparison arrays must have identical shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("numeric comparison arrays must be finite")
    if left.size == 0:
        return {
            "value_count": 0,
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "root_mean_squared_error": None,
            "max_symmetric_relative_error": None,
            "reference_l2_norm": 0.0,
            "candidate_l2_norm": 0.0,
            "cosine_similarity": None,
            "tolerance": tolerance.to_dict() if tolerance else None,
            "within_tolerance": None,
        }
    difference = np.abs(right - left)
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-300)
    relative = difference / denominator
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = (
        float(np.dot(left, right) / (left_norm * right_norm))
        if left_norm > 0 and right_norm > 0
        else None
    )
    return {
        "value_count": int(left.size),
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "root_mean_squared_error": float(np.sqrt(np.mean((right - left) ** 2))),
        "max_symmetric_relative_error": float(relative.max()),
        "reference_l2_norm": left_norm,
        "candidate_l2_norm": right_norm,
        "cosine_similarity": cosine,
        "tolerance": tolerance.to_dict() if tolerance else None,
        "within_tolerance": (
            bool(
                np.all(
                    difference <= tolerance.absolute + tolerance.relative * np.abs(left)
                )
            )
            if tolerance
            else None
        ),
    }


def _scalar_field_summary(
    reference: Mapping[Any, pd.Series],
    candidate: Mapping[Any, pd.Series],
    field: str,
    tolerance: NumericTolerance | None,
) -> dict[str, Any]:
    keys = sorted(set(reference) & set(candidate))
    left = [float(reference[key][field]) for key in keys]
    right = [float(candidate[key][field]) for key in keys]
    return _numeric_summary(left, right, tolerance)


def _vector_field_summary(
    reference: Mapping[Any, pd.Series],
    candidate: Mapping[Any, pd.Series],
    field: str,
    tolerance: NumericTolerance | None,
    *,
    expected_width: int | None = None,
) -> dict[str, Any]:
    left_values: list[float] = []
    right_values: list[float] = []
    comparable_row_count = 0
    presence_mismatch_count = 0
    width_mismatch_count = 0
    for key in sorted(set(reference) & set(candidate)):
        left = reference[key].get(field)
        right = candidate[key].get(field)
        if left is None or right is None:
            if (left is None) != (right is None):
                presence_mismatch_count += 1
            continue
        left_row = list(left)
        right_row = list(right)
        if len(left_row) != len(right_row) or (
            expected_width is not None and len(left_row) != expected_width
        ):
            width_mismatch_count += 1
            continue
        if any(value is None for value in [*left_row, *right_row]):
            if left_row != right_row:
                presence_mismatch_count += 1
            continue
        comparable_row_count += 1
        left_values.extend(float(value) for value in left_row)
        right_values.extend(float(value) for value in right_row)
    summary = _numeric_summary(left_values, right_values, tolerance)
    summary.update(
        {
            "comparable_row_count": comparable_row_count,
            "presence_mismatch_count": presence_mismatch_count,
            "width_mismatch_count": width_mismatch_count,
        }
    )
    if tolerance and (presence_mismatch_count or width_mismatch_count):
        summary["within_tolerance"] = False
    return summary


def _candidate_profile_summary(
    reference: Mapping[Any, pd.Series],
    candidate: Mapping[Any, pd.Series],
    *,
    candidate_count: int,
    tolerance: NumericTolerance | None,
) -> dict[str, Any]:
    overall = _vector_field_summary(
        reference,
        candidate,
        "contrib_map",
        tolerance,
        expected_width=candidate_count,
    )
    per_candidate: list[dict[str, Any]] = []
    for index in range(candidate_count):
        left_values: list[float] = []
        right_values: list[float] = []
        for key in sorted(set(reference) & set(candidate)):
            left = reference[key].get("contrib_map")
            right = candidate[key].get("contrib_map")
            if left is None or right is None:
                continue
            if len(left) == candidate_count and len(right) == candidate_count:
                left_values.append(float(left[index]))
                right_values.append(float(right[index]))
        per_candidate.append(
            {
                "candidate_index": index,
                **_numeric_summary(left_values, right_values, tolerance),
            }
        )
    return {"overall": overall, "per_candidate": per_candidate}


def _metric_pair(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *keys: str
) -> dict[str, Any]:
    key = next(
        (
            candidate_key
            for candidate_key in keys
            if candidate_key in reference or candidate_key in candidate
        ),
        keys[0],
    )
    left = reference.get(key)
    right = candidate.get(key)
    delta = None
    ratio = None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        delta = right - left
        ratio = right / left if left != 0 else None
    return {
        "metric_key": key,
        "reference": left,
        "candidate": right,
        "candidate_minus_reference": delta,
        "candidate_over_reference": ratio,
    }


def _group_within_tolerance(value: Any) -> bool:
    if isinstance(value, Mapping):
        direct = value.get("within_tolerance")
        if direct is False:
            return False
        return all(_group_within_tolerance(item) for item in value.values())
    if isinstance(value, list):
        return all(_group_within_tolerance(item) for item in value)
    return True


def _cuda_allocator_contract(
    artifact: TopKCompactTraceArtifact,
    *,
    expected_policy: str,
) -> dict[str, Any]:
    expected_value = (
        None if expected_policy == "default_v1" else "expandable_segments:True"
    )
    expected_receipt = {
        "intended_policy_id": expected_policy,
        "observed_environment": {
            "name": "PYTORCH_CUDA_ALLOC_CONF",
            "value": expected_value,
            "is_set": expected_value is not None,
        },
        "observed_allocator_backend": "native",
    }
    identity = artifact.manifest.get("artifact_identity")
    identity_policy = (
        identity.get("cuda_allocator_policy") if isinstance(identity, Mapping) else None
    )
    identity_runtime = (
        identity.get("runtime_environment") if isinstance(identity, Mapping) else None
    )
    identity_receipt = (
        identity_runtime.get("cuda_allocator_policy")
        if isinstance(identity_runtime, Mapping)
        else None
    )
    manifest_runtime = artifact.manifest.get("runtime_environment")
    manifest_receipt = (
        manifest_runtime.get("cuda_allocator_policy")
        if isinstance(manifest_runtime, Mapping)
        else None
    )
    checks = {
        "identity_policy": identity_policy == expected_policy,
        "identity_runtime_receipt": identity_receipt == expected_receipt,
        "manifest_runtime_receipt": manifest_receipt == expected_receipt,
    }
    return {
        "expected_policy": expected_policy,
        "expected_receipt": expected_receipt,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _embedding_edge_materialization_contract(
    artifact: TopKCompactTraceArtifact,
    *,
    expected_strategy: str,
) -> dict[str, Any]:
    identity = artifact.manifest.get("artifact_identity")
    adag_config = identity.get("adag_config") if isinstance(identity, Mapping) else None
    observed_strategy = (
        adag_config.get("embedding_edge_materialization")
        if isinstance(adag_config, Mapping)
        else None
    )
    return {
        "expected_strategy": expected_strategy,
        "observed_strategy": observed_strategy,
        "passed": observed_strategy == expected_strategy,
    }


def _cross_layer_jacobian_execution_contract(
    artifact: TopKCompactTraceArtifact,
    *,
    expected_strategy: str,
) -> dict[str, Any]:
    identity = artifact.manifest.get("artifact_identity")
    adag_config = identity.get("adag_config") if isinstance(identity, Mapping) else None
    observed_strategy = (
        adag_config.get("cross_layer_jacobian_execution")
        if isinstance(adag_config, Mapping)
        else None
    )
    return {
        "expected_strategy": expected_strategy,
        "observed_strategy": observed_strategy,
        "passed": observed_strategy == expected_strategy,
    }


def _cross_layer_jacobian_receipt_contract(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
) -> dict[str, Any]:
    """Require canonical ordered pair receipts to match exactly across lanes."""

    def layer_pairs(artifact: TopKCompactTraceArtifact) -> Any:
        instrumentation = artifact.topk_trace.circuit_data.trace_metadata.get(
            "instrumentation"
        )
        return (
            instrumentation.get("layer_pairs")
            if isinstance(instrumentation, Mapping)
            else None
        )

    def validate(pairs: Any) -> tuple[bool, list[dict[str, Any]]]:
        summaries: list[dict[str, Any]] = []
        if not isinstance(pairs, list) or not pairs:
            return False, summaries
        passed = True
        coordinates: list[tuple[int, int]] = []
        for pair in pairs:
            if not isinstance(pair, Mapping):
                passed = False
                continue
            src_layer = pair.get("src_layer")
            tgt_layer = pair.get("tgt_layer")
            coordinates_valid = (
                type(src_layer) is int
                and type(tgt_layer) is int
                and 0 <= src_layer < tgt_layer
            )
            if coordinates_valid:
                coordinates.append((src_layer, tgt_layer))
            receipts = pair.get("exact_receipts")
            receipt_names = (
                [item.get("name") for item in receipts]
                if isinstance(receipts, list)
                and all(isinstance(item, Mapping) for item in receipts)
                else None
            )
            receipt_hashes = (
                [item.get("sha256") for item in receipts]
                if receipt_names is not None
                else None
            )
            receipt_shape_valid = (
                receipt_names == list(CROSS_LAYER_JACOBIAN_RECEIPT_NAMES)
                and receipt_hashes is not None
                and all(set(item) == {"name", "sha256"} for item in receipts)
                and all(
                    isinstance(value, str)
                    and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value)
                    for value in receipt_hashes
                )
            )
            passed = passed and coordinates_valid and receipt_shape_valid
            summaries.append(
                {
                    "src_layer": src_layer,
                    "tgt_layer": tgt_layer,
                    "exact_receipts": receipts,
                    "valid": coordinates_valid and receipt_shape_valid,
                }
            )
        canonical_coordinates = sorted(
            coordinates, key=lambda coordinate: (-coordinate[1], -coordinate[0])
        )
        passed = (
            passed
            and len(coordinates) == len(pairs)
            and len(set(coordinates)) == len(coordinates)
            and coordinates == canonical_coordinates
        )
        return passed, summaries

    reference_valid, reference_pairs = validate(layer_pairs(reference))
    candidate_valid, candidate_pairs = validate(layer_pairs(candidate))
    pair_order_equal = [
        (pair.get("src_layer"), pair.get("tgt_layer")) for pair in reference_pairs
    ] == [(pair.get("src_layer"), pair.get("tgt_layer")) for pair in candidate_pairs]
    receipt_hashes_equal = (
        reference_valid
        and candidate_valid
        and len(reference_pairs) == len(candidate_pairs)
        and all(
            reference_pair["exact_receipts"] == candidate_pair["exact_receipts"]
            for reference_pair, candidate_pair in zip(
                reference_pairs, candidate_pairs, strict=True
            )
        )
    )
    checks = {
        "reference_presence_and_order": reference_valid,
        "candidate_presence_and_order": candidate_valid,
        "pair_order_equal": pair_order_equal,
        "receipt_hashes_exact": receipt_hashes_equal,
    }
    return {
        "receipt_names": list(CROSS_LAYER_JACOBIAN_RECEIPT_NAMES),
        "reference_pairs": reference_pairs,
        "candidate_pairs": candidate_pairs,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _selected_neuron_contribution_receipt_contract(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
) -> dict[str, Any]:
    """Require canonical per-layer projected ordinary-VJP receipts to match."""

    receipt_field = "selected_neuron_contribution_projected_vjp_sha256"
    shape_field = "selected_neuron_contribution_projected_vjp_shape"
    target_count_field = "selected_neuron_contribution_target_lane_count"

    def instrumentation(artifact: TopKCompactTraceArtifact) -> Any:
        return artifact.topk_trace.circuit_data.trace_metadata.get("instrumentation")

    def validate(raw_instrumentation: Any) -> tuple[bool, list[dict[str, Any]]]:
        summaries: list[dict[str, Any]] = []
        if not isinstance(raw_instrumentation, Mapping):
            return False, summaries
        raw_layers = raw_instrumentation.get("layers")
        early_predictors = raw_instrumentation.get("early_predictors")
        selected_counts_by_layer = (
            early_predictors.get("selected_neuron_counts_by_layer")
            if isinstance(early_predictors, Mapping)
            else None
        )
        if not isinstance(raw_layers, list) or not raw_layers:
            return False, summaries
        selected_counts_valid = (
            isinstance(selected_counts_by_layer, list)
            and bool(selected_counts_by_layer)
            and all(
                isinstance(item, Mapping)
                and type(item.get("layer")) is int
                and item["layer"] >= 0
                and type(item.get("count")) is int
                and item["count"] >= 0
                for item in selected_counts_by_layer
            )
        )
        if not selected_counts_valid:
            return False, summaries
        expected_counts = {
            item["layer"]: item["count"] for item in selected_counts_by_layer
        }
        selected_count_layers = [item["layer"] for item in selected_counts_by_layer]
        selected_counts_valid = len(expected_counts) == len(
            selected_counts_by_layer
        ) and selected_count_layers == sorted(selected_count_layers)
        if not selected_counts_valid:
            return False, summaries
        expected_layers = [
            layer for layer, count in expected_counts.items() if count > 0
        ]
        if not expected_layers:
            return False, summaries
        raw_layer_ids = [
            raw_layer.get("layer")
            for raw_layer in raw_layers
            if isinstance(raw_layer, Mapping)
        ]
        raw_layers_valid = (
            len(raw_layer_ids) == len(raw_layers)
            and all(type(layer) is int and layer >= 0 for layer in raw_layer_ids)
            and len(set(raw_layer_ids)) == len(raw_layer_ids)
            and raw_layer_ids == sorted(raw_layer_ids)
        )
        records_by_layer = {
            raw_layer["layer"]: raw_layer
            for raw_layer in raw_layers
            if isinstance(raw_layer, Mapping) and type(raw_layer.get("layer")) is int
        }
        passed = True
        for layer in expected_layers:
            raw_layer = records_by_layer.get(layer, {})
            receipt = raw_layer.get(receipt_field)
            shape = raw_layer.get(shape_field)
            target_count = raw_layer.get(target_count_field)
            selected_neuron_count = raw_layer.get("selected_neuron_count")
            layer_valid = selected_neuron_count == expected_counts[layer]
            receipt_valid = (
                isinstance(receipt, str)
                and len(receipt) == 64
                and all(character in "0123456789abcdef" for character in receipt)
            )
            shape_valid = (
                isinstance(shape, list)
                and len(shape) == 3
                and all(type(extent) is int and extent > 0 for extent in shape)
            )
            target_count_valid = (
                type(target_count) is int
                and target_count > 0
                and shape_valid
                and shape[2] == target_count
            )
            valid = layer_valid and receipt_valid and shape_valid and target_count_valid
            summaries.append(
                {
                    "layer": layer,
                    "projected_vjp_shape": shape,
                    "target_lane_count": target_count,
                    "projected_vjp_sha256": receipt,
                    "valid": valid,
                }
            )
            passed = passed and valid
        return passed and raw_layers_valid, summaries

    reference_valid, reference_layers = validate(instrumentation(reference))
    candidate_valid, candidate_layers = validate(instrumentation(candidate))
    layer_order_equal = [item.get("layer") for item in reference_layers] == [
        item.get("layer") for item in candidate_layers
    ]
    receipts_exact = (
        reference_valid
        and candidate_valid
        and len(reference_layers) == len(candidate_layers)
        and all(
            reference_layer["projected_vjp_shape"]
            == candidate_layer["projected_vjp_shape"]
            and reference_layer["target_lane_count"]
            == candidate_layer["target_lane_count"]
            and reference_layer["projected_vjp_sha256"]
            == candidate_layer["projected_vjp_sha256"]
            for reference_layer, candidate_layer in zip(
                reference_layers, candidate_layers, strict=True
            )
        )
    )
    checks = {
        "reference_presence_and_order": reference_valid,
        "candidate_presence_and_order": candidate_valid,
        "layer_order_equal": layer_order_equal,
        "receipt_hashes_exact": receipts_exact,
    }
    return {
        "reference_layers": reference_layers,
        "candidate_layers": candidate_layers,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _selected_attribution_neuron_lane_runtime_contract(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
) -> dict[str, Any]:
    """Prove the selected-attribution VJPs executed at None/50 versus width one."""

    def instrumentation(artifact: TopKCompactTraceArtifact) -> Any:
        return artifact.topk_trace.circuit_data.trace_metadata.get("instrumentation")

    def validate(
        raw: Any,
        *,
        expected_requested: int | None,
    ) -> tuple[bool, dict[str, Any]]:
        if not isinstance(raw, Mapping):
            return False, {}
        counters = raw.get("counters")
        early = raw.get("early_predictors")
        stages = raw.get("stages")
        if not all(isinstance(value, Mapping) for value in (counters, early, stages)):
            return False, {}
        strategy_index = SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_STRATEGIES.index(
            expected_requested
        )
        expected_resolved = SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_RESOLVED_WIDTHS[
            strategy_index
        ]
        selected_counts = early.get("selected_neuron_counts_by_layer")
        counts_valid = (
            isinstance(selected_counts, list)
            and bool(selected_counts)
            and all(
                isinstance(item, Mapping)
                and type(item.get("layer")) is int
                and item["layer"] >= 0
                and type(item.get("count")) is int
                and item["count"] >= 0
                for item in selected_counts
            )
        )
        if not counts_valid:
            return False, {}
        layer_ids = [item["layer"] for item in selected_counts]
        counts_valid = len(set(layer_ids)) == len(layer_ids) and layer_ids == sorted(
            layer_ids
        )
        expected_chunks: list[dict[str, int]] = []
        for item in selected_counts:
            layer = item["layer"]
            count = item["count"]
            chunk_count = (
                (count + expected_resolved - 1) // expected_resolved if count else 0
            )
            for chunk_index, chunk_start in enumerate(
                range(0, count, expected_resolved)
            ):
                expected_chunks.append(
                    {
                        "layer": layer,
                        "chunk_start": chunk_start,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "chunk_neuron_count": min(
                            expected_resolved, count - chunk_start
                        ),
                    }
                )
        if not counts_valid or not expected_chunks:
            return False, {}

        def calls(stage_name: str) -> list[Any] | None:
            stage = stages.get(stage_name)
            call_measurements = (
                stage.get("call_measurements") if isinstance(stage, Mapping) else None
            )
            return call_measurements if isinstance(call_measurements, list) else None

        vjp_calls = calls("selected_attribution_vjp")
        projection_calls = calls("selected_attribution_chunk_projection")
        if not isinstance(vjp_calls, list) or not isinstance(projection_calls, list):
            return False, {}
        calls_valid = len(vjp_calls) == len(projection_calls) == len(expected_chunks)
        batch: int | None = None
        differentiated_input_shape: list[int] | None = None
        source_token_count: int | None = None
        call_summaries: list[dict[str, Any]] = []
        for call_index, expected in enumerate(expected_chunks):
            vjp_call = vjp_calls[call_index] if call_index < len(vjp_calls) else None
            projection_call = (
                projection_calls[call_index]
                if call_index < len(projection_calls)
                else None
            )
            vjp = vjp_call.get("metadata") if isinstance(vjp_call, Mapping) else None
            projection = (
                projection_call.get("metadata")
                if isinstance(projection_call, Mapping)
                else None
            )
            if not isinstance(vjp, Mapping) or not isinstance(projection, Mapping):
                calls_valid = False
                continue
            output_shape = vjp.get("differentiated_output_shape")
            input_shape = vjp.get("differentiated_input_shape")
            current_batch = (
                output_shape[1]
                if isinstance(output_shape, list)
                and len(output_shape) == 2
                and type(output_shape[1]) is int
                else None
            )
            if batch is None:
                batch = current_batch
            if differentiated_input_shape is None and isinstance(input_shape, list):
                differentiated_input_shape = input_shape
            current_source_count = projection.get("source_token_count")
            if source_token_count is None and type(current_source_count) is int:
                source_token_count = current_source_count
            lanes = (
                expected["chunk_neuron_count"] * current_batch
                if type(current_batch) is int
                else None
            )
            raw_shape = (
                [lanes, current_batch, *input_shape[1:]]
                if type(lanes) is int
                and isinstance(input_shape, list)
                and len(input_shape) == 3
                else None
            )
            common_valid = all(
                vjp.get(field) == value and projection.get(field) == value
                for field, value in {
                    **expected,
                    "neuron_lane_chunk_size_resolved": expected_resolved,
                }.items()
            )
            call_valid = (
                isinstance(vjp_call, Mapping)
                and isinstance(projection_call, Mapping)
                and vjp_call.get("call_index") == call_index
                and projection_call.get("call_index") == call_index
                and vjp_call.get("failed") is False
                and projection_call.get("failed") is False
                and common_valid
                and vjp.get("operation_kind") == "batched_vjp"
                and type(current_batch) is int
                and current_batch > 0
                and output_shape == [expected["chunk_neuron_count"], current_batch]
                and input_shape == differentiated_input_shape
                and isinstance(input_shape, list)
                and len(input_shape) == 3
                and all(type(extent) is int and extent > 0 for extent in input_shape)
                and input_shape[0] == current_batch
                and vjp.get("lane_count") == lanes
                and vjp.get("grad_outputs_shape") == [lanes, lanes]
                and vjp.get("vjp_result_shape") == raw_shape
                and projection.get("operation_kind") == "terminal_projection"
                and projection.get("raw_vjp_result_shape") == raw_shape
                and type(current_source_count) is int
                and current_source_count > 0
                and current_source_count == source_token_count
                and projection.get("return_gradient_only") is False
                and projection.get("terminal_projection_detached") is True
                and projection.get("retained_chunk_count_before")
                == expected["chunk_index"]
                and projection.get("retained_chunk_count_after")
                == expected["chunk_index"] + 1
                and projection.get("projected_shape")
                == [expected["chunk_neuron_count"], current_batch, current_source_count]
                and projection.get("projected_requires_grad") is False
            )
            calls_valid = calls_valid and call_valid
            call_summaries.append(
                {
                    **expected,
                    "lane_count": lanes,
                    "valid": call_valid,
                }
            )

        predictor_chunk_count = len(expected_chunks)
        counters_valid = (
            counters.get("selected_attribution_neuron_lane_chunk_size_requested")
            == expected_requested
            and counters.get("selected_attribution_neuron_lane_chunk_size_resolved")
            == expected_resolved
            and counters.get("selected_attribution_chunk_size") == expected_resolved
            and counters.get("selected_attribution_chunks_per_pass")
            == predictor_chunk_count
            and counters.get("selected_attribution_pass_count") == 1
            and counters.get("selected_attribution_chunk_executions")
            == predictor_chunk_count
        )
        predictors_valid = (
            early.get("selected_attribution_chunk_size") == expected_resolved
            and early.get("selected_attribution_chunks_per_pass")
            == predictor_chunk_count
            and early.get("selected_attribution_pass_count") == 1
            and early.get("selected_attribution_chunk_executions")
            == predictor_chunk_count
            and early.get("ig_steps") is None
            and early.get("ig_execution_count") == 1
        )
        summary = {
            "expected_requested_width": expected_requested,
            "expected_resolved_width": expected_resolved,
            "selected_neuron_counts_by_layer": selected_counts,
            "batch": batch,
            "differentiated_input_shape": differentiated_input_shape,
            "source_token_count": source_token_count,
            "calls": call_summaries,
            "checks": {
                "counters": counters_valid,
                "early_predictors": predictors_valid,
                "ordered_vjp_and_projection_calls": calls_valid,
            },
        }
        return all(summary["checks"].values()), summary

    reference_valid, reference_runtime = validate(
        instrumentation(reference),
        expected_requested=SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_STRATEGIES[0],
    )
    candidate_valid, candidate_runtime = validate(
        instrumentation(candidate),
        expected_requested=SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_STRATEGIES[1],
    )
    cross_side_workload_equal = (
        reference_valid
        and candidate_valid
        and reference_runtime["selected_neuron_counts_by_layer"]
        == candidate_runtime["selected_neuron_counts_by_layer"]
        and reference_runtime["batch"] == candidate_runtime["batch"]
        and reference_runtime["differentiated_input_shape"]
        == candidate_runtime["differentiated_input_shape"]
        and reference_runtime["source_token_count"]
        == candidate_runtime["source_token_count"]
    )
    checks = {
        "reference_runtime_width_proven": reference_valid,
        "candidate_runtime_width_proven": candidate_valid,
        "cross_side_workload_equal": cross_side_workload_equal,
    }
    return {
        "reference_runtime": reference_runtime,
        "candidate_runtime": candidate_runtime,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _stop_gradient_selected_attribution_forward_contract(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
) -> dict[str, Any]:
    """Prove full-model versus prefix-stop execution from ordered observations."""

    namespace = "stop_gradient_selected_attribution_forward"
    stage_name = "stop_grad_selected_layer_forward"
    receipt_fields = (
        "execution",
        "layer",
        "decoder_layer_entries",
        "selected_down_projection_completed",
        "lm_head_completed",
        "logits_completed",
        "down_projection_materialized",
        "decoder_suffix_materialized",
        "logits_materialized",
    )

    def instrumentation(artifact: TopKCompactTraceArtifact) -> Any:
        return artifact.topk_trace.circuit_data.trace_metadata.get("instrumentation")

    def identity_strategy(artifact: TopKCompactTraceArtifact) -> Any:
        identity = artifact.manifest.get("artifact_identity")
        adag = identity.get("adag_config") if isinstance(identity, Mapping) else None
        return (
            adag.get("stop_gradient_selected_attribution_forward_execution")
            if isinstance(adag, Mapping)
            else None
        )

    def validate(
        raw: Any,
        *,
        expected_execution: str,
        expected_full_layer_entries: list[int] | None,
    ) -> tuple[bool, dict[str, Any]]:
        if not isinstance(raw, Mapping):
            return False, {}
        execution_records = raw.get("execution_records")
        counters = raw.get("counters")
        stages = raw.get("stages")
        records = (
            execution_records.get(namespace)
            if isinstance(execution_records, Mapping)
            else None
        )
        stage = stages.get(stage_name) if isinstance(stages, Mapping) else None
        calls = stage.get("call_measurements") if isinstance(stage, Mapping) else None
        if (
            not isinstance(records, list)
            or not records
            or not isinstance(counters, Mapping)
            or not isinstance(calls, list)
        ):
            return False, {}

        records_valid = len(records) == len(calls)
        summaries: list[dict[str, Any]] = []
        layers: list[int] = []
        inferred_full_entries = expected_full_layer_entries
        for index, record in enumerate(records):
            call = calls[index] if index < len(calls) else None
            metadata = call.get("metadata") if isinstance(call, Mapping) else None
            required_shape = isinstance(record, Mapping) and all(
                field in record for field in receipt_fields
            )
            layer = record.get("layer") if isinstance(record, Mapping) else None
            entries = (
                record.get("decoder_layer_entries")
                if isinstance(record, Mapping)
                else None
            )
            layer_valid = type(layer) is int and layer >= 0
            entries_valid = (
                isinstance(entries, list)
                and bool(entries)
                and all(type(value) is int and value >= 0 for value in entries)
            )
            if layer_valid:
                layers.append(layer)
            if expected_execution == "full_model_v1" and entries_valid:
                if inferred_full_entries is None:
                    inferred_full_entries = entries
                expected_entries = inferred_full_entries
                expected_materialization = {
                    "selected_down_projection_completed": True,
                    "lm_head_completed": True,
                    "logits_completed": True,
                    "down_projection_materialized": True,
                    "decoder_suffix_materialized": (
                        layer + 1 < len(expected_entries) if layer_valid else None
                    ),
                    "logits_materialized": True,
                }
            else:
                expected_entries = list(range(layer + 1)) if layer_valid else None
                expected_materialization = {
                    "selected_down_projection_completed": False,
                    "lm_head_completed": False,
                    "logits_completed": False,
                    "down_projection_materialized": False,
                    "decoder_suffix_materialized": False,
                    "logits_materialized": False,
                }
            materialization_valid = required_shape and all(
                record.get(field) is expected
                for field, expected in expected_materialization.items()
            )
            stage_valid = (
                isinstance(call, Mapping)
                and call.get("call_index") == index
                and call.get("failed") is False
                and isinstance(metadata, Mapping)
                and isinstance(record, Mapping)
                and all(
                    metadata.get(field) == record.get(field) for field in receipt_fields
                )
            )
            record_valid = (
                required_shape
                and record.get("execution") == expected_execution
                and layer_valid
                and entries_valid
                and isinstance(expected_entries, list)
                and layer < len(expected_entries)
                and entries == expected_entries
                and materialization_valid
                and stage_valid
            )
            records_valid = records_valid and record_valid
            summaries.append(
                {
                    "execution": (
                        record.get("execution") if isinstance(record, Mapping) else None
                    ),
                    "layer": layer,
                    "decoder_layer_entries": entries,
                    "stage_receipt_matches": stage_valid,
                    "valid": record_valid,
                }
            )

        layer_order_valid = len(set(layers)) == len(records) and layers == sorted(
            layers
        )
        stage_aggregate_valid = (
            isinstance(stage, Mapping)
            and stage.get("calls") == len(records)
            and stage.get("failed_calls") == 0
        )
        full_entries_valid = (
            isinstance(inferred_full_entries, list)
            and bool(inferred_full_entries)
            and inferred_full_entries == list(range(len(inferred_full_entries)))
        )
        expected_materialized_counts = {
            "down_projection": sum(
                int(bool(record.get("down_projection_materialized")))
                for record in records
                if isinstance(record, Mapping)
            ),
            "decoder_suffix": sum(
                int(bool(record.get("decoder_suffix_materialized")))
                for record in records
                if isinstance(record, Mapping)
            ),
            "logits": sum(
                int(bool(record.get("logits_materialized")))
                for record in records
                if isinstance(record, Mapping)
            ),
        }
        expected_completion_counts = {
            "selected_down_projection": sum(
                int(bool(record.get("selected_down_projection_completed")))
                for record in records
                if isinstance(record, Mapping)
            ),
            "lm_head": sum(
                int(bool(record.get("lm_head_completed")))
                for record in records
                if isinstance(record, Mapping)
            ),
            "logits": sum(
                int(bool(record.get("logits_completed")))
                for record in records
                if isinstance(record, Mapping)
            ),
        }
        counters_valid = (
            counters.get("stop_gradient_selected_attribution_forward_execution")
            == expected_execution
            and counters.get(
                "stop_gradient_selected_attribution_forward_execution_count"
            )
            == len(records)
            and counters.get(
                f"stop_gradient_selected_attribution_{expected_execution}_execution_count"
            )
            == len(records)
            and all(
                counters.get(
                    f"stop_gradient_selected_attribution_{name}_materialized_count"
                )
                == count
                for name, count in expected_materialized_counts.items()
            )
            and counters.get(
                "stop_gradient_selected_attribution_decoder_layer_entry_count"
            )
            == sum(
                (
                    len(record.get("decoder_layer_entries"))
                    if isinstance(record.get("decoder_layer_entries"), list)
                    else 0
                )
                for record in records
                if isinstance(record, Mapping)
            )
            and all(
                counters.get(
                    f"stop_gradient_selected_attribution_{name}_completed_count"
                )
                == count
                for name, count in expected_completion_counts.items()
            )
        )
        checks = {
            "ordered_execution_records": (
                records_valid and layer_order_valid and stage_aggregate_valid
            ),
            "canonical_full_layer_range": full_entries_valid,
            "aggregate_counters": counters_valid,
        }
        return all(checks.values()), {
            "execution": expected_execution,
            "selected_layers": layers,
            "full_decoder_layer_entries": inferred_full_entries,
            "records": summaries,
            "checks": checks,
        }

    reference_strategy = identity_strategy(reference)
    candidate_strategy = identity_strategy(candidate)
    reference_valid, reference_runtime = validate(
        instrumentation(reference),
        expected_execution=STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_STRATEGIES[0],
        expected_full_layer_entries=None,
    )
    full_entries = reference_runtime.get("full_decoder_layer_entries")
    candidate_valid, candidate_runtime = validate(
        instrumentation(candidate),
        expected_execution=STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_STRATEGIES[1],
        expected_full_layer_entries=(
            full_entries if isinstance(full_entries, list) else None
        ),
    )
    strategy_valid = (
        reference_strategy
        == STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_STRATEGIES[0]
        and candidate_strategy
        == STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_STRATEGIES[1]
    )
    workload_equal = (
        reference_valid
        and candidate_valid
        and reference_runtime["selected_layers"] == candidate_runtime["selected_layers"]
        and reference_runtime["full_decoder_layer_entries"]
        == candidate_runtime["full_decoder_layer_entries"]
    )
    checks = {
        "canonical_identity_strategies": strategy_valid,
        "reference_full_model_receipts": reference_valid,
        "candidate_prefix_stop_receipts": candidate_valid,
        "cross_side_workload_equal": workload_equal,
    }
    return {
        "reference_strategy": reference_strategy,
        "candidate_strategy": candidate_strategy,
        "reference_runtime": reference_runtime,
        "candidate_runtime": candidate_runtime,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _stop_gradient_selected_attribution_storage_contract(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
) -> dict[str, Any]:
    """Prove graph-retaining versus terminal-detached compact storage."""

    namespace = "stop_gradient_selected_attribution_storage"
    stage_name = "stop_grad_selected_chunk_projection"
    receipt_fields = (
        "layer",
        "chunk_start",
        "strategy",
        "input_requires_grad",
        "input_grad_fn_retained",
        "stored_requires_grad",
        "stored_grad_fn_retained",
        "terminal_detached",
        "shares_projection_storage",
    )
    stage_workload_fields = (
        "chunk_neuron_count",
        "source_token_count",
        "raw_vjp_result_shape",
        "projected_shape",
        "retained_chunk_count_before",
        "retained_chunk_count_after",
    )

    def instrumentation(artifact: TopKCompactTraceArtifact) -> Any:
        return artifact.topk_trace.circuit_data.trace_metadata.get("instrumentation")

    def identity_strategy(artifact: TopKCompactTraceArtifact) -> Any:
        identity = artifact.manifest.get("artifact_identity")
        adag = identity.get("adag_config") if isinstance(identity, Mapping) else None
        return (
            adag.get("stop_gradient_selected_attribution_storage")
            if isinstance(adag, Mapping)
            else None
        )

    def validate(
        raw: Any,
        *,
        expected_strategy: str,
    ) -> tuple[bool, dict[str, Any]]:
        if not isinstance(raw, Mapping):
            return False, {}
        execution_records = raw.get("execution_records")
        counters = raw.get("counters")
        stages = raw.get("stages")
        records = (
            execution_records.get(namespace)
            if isinstance(execution_records, Mapping)
            else None
        )
        stage = stages.get(stage_name) if isinstance(stages, Mapping) else None
        calls = stage.get("call_measurements") if isinstance(stage, Mapping) else None
        if (
            not isinstance(records, list)
            or not records
            or not isinstance(counters, Mapping)
            or not isinstance(calls, list)
        ):
            return False, {}

        graph_retaining = expected_strategy == "graph_retaining_v1"
        expected_graph_receipts = {
            "input_requires_grad": True,
            "input_grad_fn_retained": True,
            "stored_requires_grad": graph_retaining,
            "stored_grad_fn_retained": graph_retaining,
            "terminal_detached": not graph_retaining,
            "shares_projection_storage": True,
        }
        records_valid = len(records) == len(calls)
        coordinates: list[tuple[int, int]] = []
        workloads: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        expected_retained_index_by_layer: dict[int, int] = {}
        for index, record in enumerate(records):
            call = calls[index] if index < len(calls) else None
            metadata = call.get("metadata") if isinstance(call, Mapping) else None
            required_shape = isinstance(record, Mapping) and all(
                field in record for field in receipt_fields
            )
            layer = record.get("layer") if isinstance(record, Mapping) else None
            chunk_start = (
                record.get("chunk_start") if isinstance(record, Mapping) else None
            )
            coordinate_valid = (
                type(layer) is int
                and layer >= 0
                and type(chunk_start) is int
                and chunk_start >= 0
            )
            if coordinate_valid:
                coordinates.append((layer, chunk_start))
            graph_receipts_valid = required_shape and all(
                record.get(field) is expected
                for field, expected in expected_graph_receipts.items()
            )
            workload = (
                {field: metadata.get(field) for field in stage_workload_fields}
                if isinstance(metadata, Mapping)
                else {}
            )
            chunk_neuron_count = workload.get("chunk_neuron_count")
            source_token_count = workload.get("source_token_count")
            raw_shape = workload.get("raw_vjp_result_shape")
            projected_shape = workload.get("projected_shape")
            retained_before = workload.get("retained_chunk_count_before")
            retained_after = workload.get("retained_chunk_count_after")
            expected_retained_before = (
                expected_retained_index_by_layer.get(layer, 0)
                if coordinate_valid
                else None
            )
            workload_valid = (
                type(chunk_neuron_count) is int
                and chunk_neuron_count > 0
                and type(source_token_count) is int
                and source_token_count > 0
                and isinstance(raw_shape, list)
                and len(raw_shape) == 4
                and all(type(extent) is int and extent > 0 for extent in raw_shape)
                and isinstance(projected_shape, list)
                and len(projected_shape) == 3
                and all(
                    type(extent) is int and extent > 0 for extent in projected_shape
                )
                and projected_shape[0] == chunk_neuron_count
                and projected_shape[2] == source_token_count
                and raw_shape[0] == chunk_neuron_count * projected_shape[1]
                and raw_shape[1] == projected_shape[1]
                and type(retained_before) is int
                and type(retained_after) is int
                and retained_before == expected_retained_before
                and retained_after == retained_before + 1
            )
            if coordinate_valid and workload_valid:
                expected_retained_index_by_layer[layer] = retained_after
            stage_valid = (
                isinstance(call, Mapping)
                and call.get("call_index") == index
                and call.get("failed") is False
                and isinstance(metadata, Mapping)
                and isinstance(record, Mapping)
                and metadata.get("operation_kind") == "vjp_projection"
                and metadata.get("layer") == layer
                and metadata.get("chunk_start") == chunk_start
                and metadata.get("selected_attribution_storage") == expected_strategy
                and all(
                    metadata.get(field) == record.get(field)
                    for field in receipt_fields
                    if field not in {"layer", "chunk_start", "strategy"}
                )
                and workload_valid
            )
            record_valid = (
                required_shape
                and record.get("strategy") == expected_strategy
                and coordinate_valid
                and graph_receipts_valid
                and stage_valid
            )
            records_valid = records_valid and record_valid
            workloads.append({"layer": layer, "chunk_start": chunk_start, **workload})
            summaries.append(
                {
                    "layer": layer,
                    "chunk_start": chunk_start,
                    "stage_receipt_matches": stage_valid,
                    "valid": record_valid,
                }
            )

        ordered_workload_valid = (
            len(coordinates) == len(records)
            and len(set(coordinates)) == len(coordinates)
            and coordinates == sorted(coordinates)
        )
        stage_aggregate_valid = (
            isinstance(stage, Mapping)
            and stage.get("calls") == len(records)
            and stage.get("failed_calls") == 0
        )
        counters_valid = (
            counters.get("stop_gradient_selected_attribution_storage")
            == expected_strategy
            and counters.get(
                "stop_gradient_selected_attribution_storage_execution_count"
            )
            == len(records)
            and counters.get(
                f"stop_gradient_selected_attribution_{expected_strategy}_storage_count"
            )
            == len(records)
            and counters.get(
                "stop_gradient_selected_attribution_projection_graph_retained_count"
            )
            == len(records)
            and counters.get(
                "stop_gradient_selected_attribution_stored_graph_retained_count"
            )
            == (len(records) if graph_retaining else 0)
            and counters.get(
                "stop_gradient_selected_attribution_terminal_detached_count"
            )
            == (0 if graph_retaining else len(records))
        )
        checks = {
            "ordered_execution_and_stage_receipts": (
                records_valid and ordered_workload_valid and stage_aggregate_valid
            ),
            "aggregate_counters": counters_valid,
        }
        return all(checks.values()), {
            "strategy": expected_strategy,
            "coordinates": [list(coordinate) for coordinate in coordinates],
            "workloads": workloads,
            "records": summaries,
            "checks": checks,
        }

    reference_strategy = identity_strategy(reference)
    candidate_strategy = identity_strategy(candidate)
    reference_valid, reference_runtime = validate(
        instrumentation(reference),
        expected_strategy=STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_STRATEGIES[0],
    )
    candidate_valid, candidate_runtime = validate(
        instrumentation(candidate),
        expected_strategy=STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_STRATEGIES[1],
    )
    strategy_valid = (
        reference_strategy
        == STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_STRATEGIES[0]
        and candidate_strategy
        == STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_STRATEGIES[1]
    )
    workload_equal = (
        reference_valid
        and candidate_valid
        and reference_runtime["coordinates"] == candidate_runtime["coordinates"]
        and reference_runtime["workloads"] == candidate_runtime["workloads"]
    )
    checks = {
        "canonical_identity_strategies": strategy_valid,
        "reference_graph_retaining_receipts": reference_valid,
        "candidate_terminal_detached_receipts": candidate_valid,
        "cross_side_workload_equal": workload_equal,
    }
    return {
        "reference_strategy": reference_strategy,
        "candidate_strategy": candidate_strategy,
        "reference_runtime": reference_runtime,
        "candidate_runtime": candidate_runtime,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _selected_target_logit_execution_contract(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
) -> dict[str, Any]:
    """Prove full-sequence versus selected-position LM-head execution."""

    namespace = "selected_target_logit_execution"
    missing = object()

    def instrumentation(artifact: TopKCompactTraceArtifact) -> Any:
        return artifact.topk_trace.circuit_data.trace_metadata.get("instrumentation")

    def identity_adag(artifact: TopKCompactTraceArtifact) -> Mapping[str, Any] | None:
        identity = artifact.manifest.get("artifact_identity")
        adag = identity.get("adag_config") if isinstance(identity, Mapping) else None
        return adag if isinstance(adag, Mapping) else None

    def validate(
        raw: Any,
        *,
        expected_strategy: str,
        adag: Mapping[str, Any] | None,
    ) -> tuple[bool, dict[str, Any]]:
        ig_steps = (
            adag.get("ig_steps", missing) if isinstance(adag, Mapping) else missing
        )
        ig_steps_valid = ig_steps is None or (type(ig_steps) is int and ig_steps > 0)
        config_valid = (
            isinstance(adag, Mapping)
            and adag.get("selected_target_logit_execution") == expected_strategy
            and "ig_steps" in adag
            and ig_steps_valid
            and adag.get("center_logits") is False
        )
        expected_execution_indexes = (
            ([None] if ig_steps is None else list(range(ig_steps + 1)))
            if ig_steps_valid
            else []
        )
        if not isinstance(raw, Mapping):
            return False, {
                "config_valid": config_valid,
                "ig_steps": None if ig_steps is missing else ig_steps,
                "expected_execution_indexes": expected_execution_indexes,
            }
        execution_records = raw.get("execution_records")
        counters = raw.get("counters")
        records = (
            execution_records.get(namespace)
            if isinstance(execution_records, Mapping)
            else None
        )
        if (
            not isinstance(records, list)
            or not records
            or not isinstance(counters, Mapping)
        ):
            return False, {
                "config_valid": config_valid,
                "ig_steps": None if ig_steps is missing else ig_steps,
                "expected_execution_indexes": expected_execution_indexes,
            }

        full_strategy = expected_strategy == "full_logits_v1"
        records_valid = True
        workloads: list[dict[str, Any]] = []
        lm_head_position_rows = 0
        observed_execution_indexes: list[Any] = []
        for record in records:
            if not isinstance(record, Mapping):
                records_valid = False
                continue
            batch = record.get("batch_size")
            sequence_count = record.get("sequence_position_count")
            selected_count = record.get("selected_position_count")
            unique_count = record.get("unique_selected_position_count")
            vocab_size = record.get("vocab_size")
            lm_input = record.get("lm_head_input_shape")
            lm_output = record.get("lm_head_output_shape")
            selected_shape = record.get("selected_position_logit_shape")
            target_shape = record.get("target_logit_shape")
            expected_head_positions = (
                sequence_count if full_strategy else selected_count
            )
            shape_valid = (
                (
                    type(batch) is int
                    and batch > 0
                    and type(sequence_count) is int
                    and sequence_count > 0
                    and type(selected_count) is int
                    and selected_count > 0
                    and (full_strategy or selected_count < sequence_count)
                    and type(unique_count) is int
                    and 0 < unique_count <= min(selected_count, sequence_count)
                    and type(vocab_size) is int
                    and vocab_size > 0
                    and lm_input == [batch, expected_head_positions, lm_input[2]]
                    and type(lm_input[2]) is int
                    and lm_input[2] > 0
                    and lm_output == [batch, expected_head_positions, vocab_size]
                    and selected_shape == [batch, selected_count, vocab_size]
                    and target_shape == [selected_count, batch]
                )
                if isinstance(lm_input, list) and len(lm_input) == 3
                else False
            )
            receipt_valid = (
                record.get("execution") == expected_strategy
                and record.get("causal_lm_forward_completed") is True
                and record.get("selected_position_request_forwarded")
                is (not full_strategy)
                and record.get("full_sequence_logits_materialized") is full_strategy
                and record.get("selected_position_logits_materialized") is True
                and record.get("center_logits") is False
            )
            observed_execution_indexes.append(record.get("execution_index"))
            records_valid = records_valid and shape_valid and receipt_valid
            if shape_valid:
                lm_head_position_rows += batch * expected_head_positions
            workloads.append(
                {
                    "execution_index": record.get("execution_index"),
                    "batch_size": batch,
                    "sequence_position_count": sequence_count,
                    "selected_position_count": selected_count,
                    "unique_selected_position_count": unique_count,
                    "vocab_size": vocab_size,
                    "selected_position_logit_shape": selected_shape,
                    "target_logit_shape": target_shape,
                    "center_logits": record.get("center_logits"),
                }
            )
        counters_valid = (
            counters.get("selected_target_logit_execution") == expected_strategy
            and counters.get("selected_target_logit_execution_count") == len(records)
            and counters.get(
                f"selected_target_logit_{expected_strategy}_execution_count"
            )
            == len(records)
            and counters.get(
                "selected_target_logit_full_sequence_logits_materialized_count"
            )
            == (len(records) if full_strategy else 0)
            and counters.get(
                "selected_target_logit_selected_position_logits_materialized_count"
            )
            == len(records)
            and counters.get("selected_target_logit_lm_head_position_rows")
            == lm_head_position_rows
        )
        schedule_valid = (
            config_valid
            and len(records) == len(expected_execution_indexes)
            and observed_execution_indexes == expected_execution_indexes
        )
        checks = {
            "artifact_config_and_ig_schedule": schedule_valid,
            "ordered_execution_receipts": records_valid,
            "aggregate_counters": counters_valid,
        }
        return all(checks.values()), {
            "strategy": expected_strategy,
            "config_valid": config_valid,
            "ig_steps": None if ig_steps is missing else ig_steps,
            "expected_execution_indexes": expected_execution_indexes,
            "observed_execution_indexes": observed_execution_indexes,
            "workloads": workloads,
            "lm_head_position_rows": lm_head_position_rows,
            "checks": checks,
        }

    reference_adag = identity_adag(reference)
    candidate_adag = identity_adag(candidate)
    reference_strategy = (
        reference_adag.get("selected_target_logit_execution")
        if reference_adag is not None
        else None
    )
    candidate_strategy = (
        candidate_adag.get("selected_target_logit_execution")
        if candidate_adag is not None
        else None
    )
    reference_valid, reference_runtime = validate(
        instrumentation(reference),
        expected_strategy=SELECTED_TARGET_LOGIT_EXECUTION_AB_STRATEGIES[0],
        adag=reference_adag,
    )
    candidate_valid, candidate_runtime = validate(
        instrumentation(candidate),
        expected_strategy=SELECTED_TARGET_LOGIT_EXECUTION_AB_STRATEGIES[1],
        adag=candidate_adag,
    )
    strategy_valid = (
        reference_strategy == SELECTED_TARGET_LOGIT_EXECUTION_AB_STRATEGIES[0]
        and candidate_strategy == SELECTED_TARGET_LOGIT_EXECUTION_AB_STRATEGIES[1]
    )
    workload_equal = (
        reference_valid
        and candidate_valid
        and reference_runtime["workloads"] == candidate_runtime["workloads"]
    )
    aggregate_row_reduction = (
        reference_valid
        and candidate_valid
        and candidate_runtime["lm_head_position_rows"]
        < reference_runtime["lm_head_position_rows"]
    )
    checks = {
        "canonical_identity_strategies": strategy_valid,
        "reference_full_logits_receipts": reference_valid,
        "candidate_selected_position_logits_receipts": candidate_valid,
        "cross_side_workload_equal": workload_equal,
        "aggregate_lm_head_row_reduction": aggregate_row_reduction,
    }
    return {
        "reference_strategy": reference_strategy,
        "candidate_strategy": candidate_strategy,
        "reference_runtime": reference_runtime,
        "candidate_runtime": candidate_runtime,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _embed_contribution_receipt_contract(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
    *,
    execution_record_namespace: str,
    execution_contract: str,
    expected_reference_width: int | None,
    expected_candidate_width: int | None,
    require_exact_hashes: bool,
) -> dict[str, Any]:
    """Prove each side executed its claimed dense embedding-VJP width."""

    def instrumentation(artifact: TopKCompactTraceArtifact) -> Any:
        return artifact.topk_trace.circuit_data.trace_metadata.get("instrumentation")

    def validate(raw: Any, expected_width: int | None) -> tuple[bool, dict[str, Any]]:
        if not isinstance(raw, Mapping):
            return False, {}
        execution_records = raw.get("execution_records")
        records = (
            execution_records.get(execution_record_namespace)
            if isinstance(execution_records, Mapping)
            else None
        )
        if not isinstance(records, list) or len(records) != 1:
            return False, {}
        record = records[0]
        if not isinstance(record, Mapping):
            return False, {}
        receipt = record.get("projected_vjp_sha256")
        projected_shape = record.get("projected_vjp_shape")
        source_tokens = record.get("source_tokens")
        target_count = record.get("target_lane_count")
        requested_width = record.get("target_lane_chunk_size_requested")
        resolved_width = record.get("target_lane_chunk_size_resolved")
        chunk_count = record.get("target_lane_chunk_count")
        max_target_lanes = record.get("max_materialized_target_lanes")
        max_autograd_lanes = record.get("max_materialized_autograd_lanes")
        raw_shapes = record.get("raw_vjp_chunk_shapes")
        grad_shapes = record.get("grad_outputs_chunk_shapes")
        batch = (
            projected_shape[1]
            if isinstance(projected_shape, list) and len(projected_shape) == 3
            else None
        )
        expected_resolved = (
            min(expected_width or target_count, target_count)
            if type(target_count) is int and target_count > 0
            else None
        )
        expected_chunk_widths = (
            [
                min(expected_resolved, target_count - start)
                for start in range(0, target_count, expected_resolved)
            ]
            if type(expected_resolved) is int and expected_resolved > 0
            else []
        )
        expected_autograd_lanes = (
            expected_resolved * batch
            if type(expected_resolved) is int and type(batch) is int
            else None
        )
        shape_consistent = (
            type(batch) is int
            and batch > 0
            and isinstance(raw_shapes, list)
            and isinstance(grad_shapes, list)
            and len(raw_shapes) == len(grad_shapes) == len(expected_chunk_widths)
            and all(
                isinstance(raw_shape, list)
                and len(raw_shape) == 4
                and all(type(extent) is int and extent > 0 for extent in raw_shape)
                and raw_shape[0] == width * batch
                and raw_shape[1] == batch
                and isinstance(grad_shape, list)
                and grad_shape == [width * batch, width * batch]
                for raw_shape, grad_shape, width in zip(
                    raw_shapes, grad_shapes, expected_chunk_widths, strict=True
                )
            )
            and len({tuple(shape[2:]) for shape in raw_shapes}) == 1
            and record.get("raw_vjp_shape")
            == (raw_shapes[0] if len(raw_shapes) == 1 else None)
            and record.get("grad_outputs_shape")
            == (grad_shapes[0] if len(grad_shapes) == 1 else None)
        )
        if execution_contract == "ordinary_direct_v1":
            execution_specific_valid = (
                record.get("execution_index") is None
                and record.get("receipt_mode") == "singular"
                and record.get("return_gradient_only") is False
                and record.get("retain_graph") is True
                and "retain_graph_after_execution" not in record
            )
        elif execution_contract == "stop_gradient_direct_v1":
            execution_specific_valid = (
                "execution_index" not in record
                and "receipt_mode" not in record
                and "return_gradient_only" not in record
                and "retain_graph" not in record
                and record.get("retain_graph_after_execution") is False
            )
        else:
            raise ValueError("unknown embedding contribution receipt contract")
        valid = (
            isinstance(receipt, str)
            and len(receipt) == 64
            and all(character in "0123456789abcdef" for character in receipt)
            and execution_specific_valid
            and record.get("canonical_result_order") == "source_batch_target"
            and isinstance(source_tokens, list)
            and bool(source_tokens)
            and all(type(token) is int for token in source_tokens)
            and projected_shape == [len(source_tokens), batch, target_count]
            and requested_width == expected_width
            and resolved_width == expected_resolved
            and chunk_count == len(expected_chunk_widths)
            and max_target_lanes == expected_resolved
            and max_autograd_lanes == expected_autograd_lanes
            and record.get("max_grad_outputs_shape")
            == [expected_autograd_lanes, expected_autograd_lanes]
            and record.get("dense_vjp_result_materialized") is True
            and shape_consistent
        )
        return valid, dict(record)

    reference_valid, reference_record = validate(
        instrumentation(reference), expected_reference_width
    )
    candidate_valid, candidate_record = validate(
        instrumentation(candidate), expected_candidate_width
    )
    cross_side_structure_equal = (
        reference_valid
        and candidate_valid
        and reference_record["source_tokens"] == candidate_record["source_tokens"]
        and reference_record["projected_vjp_shape"]
        == candidate_record["projected_vjp_shape"]
        and reference_record["target_lane_count"]
        == candidate_record["target_lane_count"]
        and reference_record["raw_vjp_chunk_shapes"][0][1:]
        == candidate_record["raw_vjp_chunk_shapes"][0][1:]
    )
    hashes_exact = (
        reference_valid
        and candidate_valid
        and reference_record["projected_vjp_sha256"]
        == candidate_record["projected_vjp_sha256"]
    )
    checks = {
        "reference_runtime_width_proven": reference_valid,
        "candidate_runtime_width_proven": candidate_valid,
        "cross_side_source_and_dense_shape_equal": cross_side_structure_equal,
        "receipt_hashes_exact": hashes_exact,
    }
    required_checks = [
        reference_valid,
        candidate_valid,
        cross_side_structure_equal,
    ]
    if require_exact_hashes:
        required_checks.append(hashes_exact)
    return {
        "execution_record_namespace": execution_record_namespace,
        "execution_contract": execution_contract,
        "reference_execution": reference_record,
        "candidate_execution": candidate_record,
        "require_exact_hashes": require_exact_hashes,
        "checks": checks,
        "passed": all(required_checks),
    }


def _selected_embed_bf16_scope_contract(
    reference: TopKCompactTraceArtifact,
    candidate: TopKCompactTraceArtifact,
    *,
    reference_nodes: Mapping[Any, pd.Series],
    candidate_nodes: Mapping[Any, pd.Series],
    reference_edges: Mapping[Any, pd.Series],
    candidate_edges: Mapping[Any, pd.Series],
    candidate_count: int,
    source_attribution_tolerance: NumericTolerance,
    candidate_profile_tolerance: NumericTolerance,
    edge_tolerance: NumericTolerance,
) -> dict[str, Any]:
    """Confine BF16 drift to values fed by the selected embedding VJP."""

    def identity_and_adag(
        artifact: TopKCompactTraceArtifact,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
        identity = artifact.manifest.get("artifact_identity")
        if not isinstance(identity, Mapping):
            return None, None
        adag = identity.get("adag_config")
        return identity, adag if isinstance(adag, Mapping) else None

    def dtype(identity: Mapping[str, Any] | None) -> Any:
        model = identity.get("model") if isinstance(identity, Mapping) else None
        return model.get("dtype") if isinstance(model, Mapping) else None

    reference_identity, reference_adag = identity_and_adag(reference)
    candidate_identity, candidate_adag = identity_and_adag(candidate)
    reference_trace = reference.topk_trace
    candidate_trace = candidate.topk_trace
    reference_data = reference_trace.circuit_data
    candidate_data = candidate_trace.circuit_data
    reference_dtype = dtype(reference_identity)
    candidate_dtype = dtype(candidate_identity)
    dtype_exact_bf16 = reference_dtype == candidate_dtype == "bfloat16"
    node_topology_exact = set(reference_nodes) == set(candidate_nodes)
    node_layers = {key[0] for key in reference_nodes}
    logit_layer = max(node_layers) if node_layers else None
    expected_logit_neurons = {
        int(candidate.token_id)
        for candidate in reference_trace.candidate_selection.candidates
    }
    reference_logit_keys = {key for key in reference_nodes if key[0] == logit_layer}
    candidate_logit_keys = {key for key in candidate_nodes if key[0] == logit_layer}

    logit_scope_classified = (
        node_topology_exact
        and logit_layer is not None
        and logit_layer >= 0
        and len(expected_logit_neurons) == candidate_count
        and len(reference_logit_keys) == candidate_count
        and reference_logit_keys == candidate_logit_keys
        and {key[2] for key in reference_logit_keys} == expected_logit_neurons
        and len({key[1] for key in reference_logit_keys}) == 1
    )

    embedding_reference_nodes = {
        key: row for key, row in reference_nodes.items() if key[0] == -1
    }
    embedding_candidate_nodes = {
        key: row for key, row in candidate_nodes.items() if key[0] == -1
    }
    non_embedding_reference_nodes = {
        key: row for key, row in reference_nodes.items() if key[0] != -1
    }
    non_embedding_candidate_nodes = {
        key: row for key, row in candidate_nodes.items() if key[0] != -1
    }
    logit_reference_nodes = {
        key: row for key, row in reference_nodes.items() if key[0] == logit_layer
    }
    logit_candidate_nodes = {
        key: row for key, row in candidate_nodes.items() if key[0] == logit_layer
    }
    non_logit_reference_nodes = {
        key: row for key, row in reference_nodes.items() if key[0] != logit_layer
    }
    non_logit_candidate_nodes = {
        key: row for key, row in candidate_nodes.items() if key[0] != logit_layer
    }
    exact = NumericTolerance(absolute=0.0, relative=0.0)
    target_base_values = {
        "observed_logit": _numeric_summary(
            reference_data.target_logit_values[0],
            candidate_data.target_logit_values[0],
            exact,
        ),
        "observed_probability": _numeric_summary(
            reference_data.target_logit_probs[0],
            candidate_data.target_logit_probs[0],
            exact,
        ),
        "candidate_logits": _numeric_summary(
            [item.logit for item in reference_trace.candidate_selection.candidates],
            [item.logit for item in candidate_trace.candidate_selection.candidates],
            exact,
        ),
        "candidate_probabilities": _numeric_summary(
            [
                item.probability
                for item in reference_trace.candidate_selection.candidates
            ],
            [
                item.probability
                for item in candidate_trace.candidate_selection.candidates
            ],
            exact,
        ),
    }
    node_base_values = {
        "attribution": _scalar_field_summary(
            reference_nodes, candidate_nodes, "attribution", exact
        ),
        "activation": _scalar_field_summary(
            reference_nodes, candidate_nodes, "activation", exact
        ),
    }
    source_attribution_profiles = {
        "target_logit": _vector_field_summary(
            logit_reference_nodes,
            logit_candidate_nodes,
            "attr_map",
            source_attribution_tolerance,
        ),
        "non_logit": _vector_field_summary(
            non_logit_reference_nodes,
            non_logit_candidate_nodes,
            "attr_map",
            exact,
        ),
    }
    embedding_profiles = _candidate_profile_summary(
        embedding_reference_nodes,
        embedding_candidate_nodes,
        candidate_count=candidate_count,
        tolerance=candidate_profile_tolerance,
    )
    non_embedding_profiles = _candidate_profile_summary(
        non_embedding_reference_nodes,
        non_embedding_candidate_nodes,
        candidate_count=candidate_count,
        tolerance=exact,
    )
    profile_scope_classified = (
        bool(embedding_reference_nodes)
        and set(embedding_reference_nodes) == set(embedding_candidate_nodes)
        and set(non_embedding_reference_nodes) == set(non_embedding_candidate_nodes)
    )

    flag_names = ("disable_stop_grad", "use_stop_grad_on_mlps")
    reference_flags = (
        {name: reference_adag.get(name) for name in flag_names}
        if isinstance(reference_adag, Mapping)
        else None
    )
    candidate_flags = (
        {name: candidate_adag.get(name) for name in flag_names}
        if isinstance(candidate_adag, Mapping)
        else None
    )
    flags_proven = (
        reference_flags == candidate_flags
        and isinstance(reference_flags, Mapping)
        and all(type(reference_flags[name]) is bool for name in flag_names)
    )
    affected_edge_keys: set[Any] = set()
    edge_scope_reason = "configuration flags missing or malformed"
    if flags_proven:
        stop_gradient_logit_edges = (
            reference_flags["use_stop_grad_on_mlps"]
            and not reference_flags["disable_stop_grad"]
        )
        if stop_gradient_logit_edges:
            edge_scope_reason = "stop-gradient logit edges isolate ordinary embed VJP"
        else:
            if logit_scope_classified:
                affected_edge_keys = {
                    key
                    for key in reference_edges
                    if key[0][0] == -1 and key[1][0] == logit_layer
                }
                edge_scope_reason = "embedding-source to logit edges"
            else:
                flags_proven = False
                edge_scope_reason = "cannot prove logit layer from topology and config"

    unaffected_edge_keys = set(reference_edges) - affected_edge_keys
    edge_scope_classified = (
        flags_proven
        and set(reference_edges) == set(candidate_edges)
        and affected_edge_keys | unaffected_edge_keys == set(reference_edges)
        and not (affected_edge_keys & unaffected_edge_keys)
    )

    def subset(rows: Mapping[Any, pd.Series], keys: set[Any]) -> dict[Any, pd.Series]:
        return {key: rows[key] for key in keys if key in rows}

    def edge_summary(
        left: Mapping[Any, pd.Series],
        right: Mapping[Any, pd.Series],
        tolerance: NumericTolerance,
    ) -> dict[str, Any]:
        return {
            "attribution": _scalar_field_summary(left, right, "attribution", tolerance),
            "weight": _scalar_field_summary(left, right, "weight", tolerance),
        }

    affected_edges = edge_summary(
        subset(reference_edges, affected_edge_keys),
        subset(candidate_edges, affected_edge_keys),
        edge_tolerance,
    )
    unaffected_edges = edge_summary(
        subset(reference_edges, unaffected_edge_keys),
        subset(candidate_edges, unaffected_edge_keys),
        exact,
    )
    checks = {
        "exact_bf16_dtype_identity": dtype_exact_bf16,
        "node_topology_exact": node_topology_exact,
        "edge_topology_exact": set(reference_edges) == set(candidate_edges),
        "target_base_values_exact": _group_within_tolerance(target_base_values),
        "node_base_values_exact": _group_within_tolerance(node_base_values),
        "logit_node_scope_classified": logit_scope_classified,
        "target_logit_source_attribution_profiles_within_bf16_tolerance": (
            logit_scope_classified
            and _group_within_tolerance(source_attribution_profiles["target_logit"])
        ),
        "non_logit_source_attribution_profiles_exact": (
            logit_scope_classified
            and _group_within_tolerance(source_attribution_profiles["non_logit"])
        ),
        "candidate_profile_scope_classified": profile_scope_classified,
        "embedding_source_profiles_within_bf16_tolerance": (
            profile_scope_classified and _group_within_tolerance(embedding_profiles)
        ),
        "non_embedding_profiles_exact": (
            profile_scope_classified and _group_within_tolerance(non_embedding_profiles)
        ),
        "edge_scope_classified": edge_scope_classified,
        "embedding_derived_edges_within_bf16_tolerance": (
            edge_scope_classified and _group_within_tolerance(affected_edges)
        ),
        "unaffected_edges_exact": (
            edge_scope_classified and _group_within_tolerance(unaffected_edges)
        ),
    }
    return {
        "dtype": {"reference": reference_dtype, "candidate": candidate_dtype},
        "logit_layer": logit_layer,
        "target_base_values": target_base_values,
        "node_base_values": node_base_values,
        "source_attribution_profiles": source_attribution_profiles,
        "candidate_profiles": {
            "embedding_source": embedding_profiles,
            "non_embedding": non_embedding_profiles,
        },
        "edges": {
            "classification_reason": edge_scope_reason,
            "affected_count": len(affected_edge_keys),
            "unaffected_count": len(unaffected_edge_keys),
            "embedding_derived": affected_edges,
            "unaffected": unaffected_edges,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_execution_artifacts(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    allowed_identity_difference_paths: Sequence[str] = (),
    tolerances: Mapping[str, NumericTolerance] | None = None,
    require_same_gpu_model: bool = False,
    require_same_gpu_family: bool = False,
    require_exact_node_topology: bool = False,
    require_exact_edge_topology: bool = False,
    require_canonical_cuda_allocator_ab: bool = False,
    require_canonical_embedding_edge_ab: bool = False,
    require_canonical_cross_layer_jacobian_ab: bool = False,
    require_canonical_stop_gradient_selected_attribution_forward_ab: bool = False,
    require_canonical_stop_gradient_selected_attribution_storage_ab: bool = False,
    require_canonical_selected_target_logit_execution_ab: bool = False,
    require_canonical_selected_attribution_neuron_lane_chunk_ab: bool = False,
    require_canonical_selected_neuron_contribution_target_lane_chunk_ab: bool = False,
    selected_embed_contribution_target_lane_chunk_ab_profile: str | None = None,
    stop_gradient_embed_contribution_target_lane_chunk_ab_profile: str | None = None,
) -> dict[str, Any]:
    """Compare two saved execution traces under explicit qualification gates.

    The historical public name is retained for compatibility. Callers may
    explicitly qualify a named attention, attribution, contribution, allocator,
    embedding-edge, or cross-layer Jacobian execution strategy, but no other
    scientific configuration field.
    """

    allowed_paths = _validate_allowed_identity_paths(allowed_identity_difference_paths)
    tolerance_map = dict(tolerances or {})
    unknown_groups = sorted(set(tolerance_map) - set(TOLERANCE_GROUPS))
    if unknown_groups:
        raise ValueError(f"unknown tolerance groups: {unknown_groups}")
    if require_same_gpu_model and not require_same_gpu_family:
        require_same_gpu_family = True
    if (
        selected_embed_contribution_target_lane_chunk_ab_profile is not None
        and selected_embed_contribution_target_lane_chunk_ab_profile
        not in SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_PROFILES
    ):
        raise ValueError("unknown selected-embed contribution chunk A/B profile")
    if (
        stop_gradient_embed_contribution_target_lane_chunk_ab_profile is not None
        and stop_gradient_embed_contribution_target_lane_chunk_ab_profile
        not in STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_PROFILES
    ):
        raise ValueError("unknown stop-gradient-embed contribution chunk A/B profile")

    reference = load_topk_compact_trace(reference_path)
    candidate = load_topk_compact_trace(candidate_path)
    hard_identity_checks = _identity_value_checks(reference, candidate)
    reference_identity_integrity = _artifact_identity_integrity(reference)
    candidate_identity_integrity = _artifact_identity_integrity(candidate)
    artifact_identity = _artifact_identity_comparison(
        reference, candidate, allowed_paths=allowed_paths
    )

    reference_gpu = _gpu_identity(reference)
    candidate_gpu = _gpu_identity(candidate)
    same_gpu_model = (
        reference_gpu["model"] is not None
        and reference_gpu["model"] == candidate_gpu["model"]
    )
    same_gpu_family = (
        reference_gpu["family"] is not None
        and reference_gpu["family"] == candidate_gpu["family"]
    )

    reference_trace = reference.topk_trace
    candidate_trace = candidate.topk_trace
    reference_data = reference_trace.circuit_data
    candidate_data = candidate_trace.circuit_data
    reference_nodes = _keyed_rows(reference_data.df_node, _node_key, name="df_node")
    candidate_nodes = _keyed_rows(candidate_data.df_node, _node_key, name="df_node")
    reference_edges = _keyed_rows(reference_data.df_edge, _edge_key, name="df_edge")
    candidate_edges = _keyed_rows(candidate_data.df_edge, _edge_key, name="df_edge")
    node_topology = _topology_summary(set(reference_nodes), set(candidate_nodes))
    edge_topology = _topology_summary(set(reference_edges), set(candidate_edges))

    target_tolerance = tolerance_map.get("target")
    target_values = {
        "observed_logit": _numeric_summary(
            reference_data.target_logit_values[0],
            candidate_data.target_logit_values[0],
            target_tolerance,
        ),
        "observed_probability": _numeric_summary(
            reference_data.target_logit_probs[0],
            candidate_data.target_logit_probs[0],
            target_tolerance,
        ),
        "candidate_logits": _numeric_summary(
            [item.logit for item in reference_trace.candidate_selection.candidates],
            [item.logit for item in candidate_trace.candidate_selection.candidates],
            target_tolerance,
        ),
        "candidate_probabilities": _numeric_summary(
            [
                item.probability
                for item in reference_trace.candidate_selection.candidates
            ],
            [
                item.probability
                for item in candidate_trace.candidate_selection.candidates
            ],
            target_tolerance,
        ),
    }
    node_tolerance = tolerance_map.get("node")
    node_values = {
        "attribution": _scalar_field_summary(
            reference_nodes, candidate_nodes, "attribution", node_tolerance
        ),
        "activation": _scalar_field_summary(
            reference_nodes, candidate_nodes, "activation", node_tolerance
        ),
        "source_attribution_profile": _vector_field_summary(
            reference_nodes, candidate_nodes, "attr_map", node_tolerance
        ),
    }
    edge_tolerance = tolerance_map.get("edge")
    edge_values = {
        "attribution": _scalar_field_summary(
            reference_edges, candidate_edges, "attribution", edge_tolerance
        ),
        "weight": _scalar_field_summary(
            reference_edges, candidate_edges, "weight", edge_tolerance
        ),
    }
    candidate_profiles = _candidate_profile_summary(
        reference_nodes,
        candidate_nodes,
        candidate_count=reference_trace.candidate_count,
        tolerance=tolerance_map.get("candidate_profile"),
    )

    gates: list[dict[str, Any]] = [
        {
            "gate": "reference_artifact_identity_integrity",
            "required": True,
            "passed": reference_identity_integrity["passed"],
        },
        {
            "gate": "candidate_artifact_identity_integrity",
            "required": True,
            "passed": candidate_identity_integrity["passed"],
        },
        {
            "gate": "frozen_scientific_identity",
            "required": True,
            "passed": all(check["passed"] for check in hard_identity_checks),
        },
        {
            "gate": "artifact_identity_allowlist",
            "required": True,
            "passed": artifact_identity["passed"],
        },
    ]
    allocator_contract = None
    if require_canonical_cuda_allocator_ab:
        reference_allocator_contract = _cuda_allocator_contract(
            reference, expected_policy=CUDA_ALLOCATOR_AB_POLICIES[0]
        )
        candidate_allocator_contract = _cuda_allocator_contract(
            candidate, expected_policy=CUDA_ALLOCATOR_AB_POLICIES[1]
        )
        allocator_contract = {
            "reference": reference_allocator_contract,
            "candidate": candidate_allocator_contract,
            "passed": (
                reference_allocator_contract["passed"]
                and candidate_allocator_contract["passed"]
            ),
        }
        gates.append(
            {
                "gate": "canonical_cuda_allocator_ab_pair",
                "required": True,
                "passed": allocator_contract["passed"],
            }
        )
    embedding_edge_contract = None
    if require_canonical_embedding_edge_ab:
        reference_embedding_edge_contract = _embedding_edge_materialization_contract(
            reference, expected_strategy=EMBEDDING_EDGE_AB_STRATEGIES[0]
        )
        candidate_embedding_edge_contract = _embedding_edge_materialization_contract(
            candidate, expected_strategy=EMBEDDING_EDGE_AB_STRATEGIES[1]
        )
        embedding_edge_contract = {
            "reference": reference_embedding_edge_contract,
            "candidate": candidate_embedding_edge_contract,
            "passed": (
                reference_embedding_edge_contract["passed"]
                and candidate_embedding_edge_contract["passed"]
            ),
        }
        gates.append(
            {
                "gate": "canonical_embedding_edge_ab_pair",
                "required": True,
                "passed": embedding_edge_contract["passed"],
            }
        )
    cross_layer_jacobian_contract = None
    if require_canonical_cross_layer_jacobian_ab:
        reference_cross_layer_contract = _cross_layer_jacobian_execution_contract(
            reference, expected_strategy=CROSS_LAYER_JACOBIAN_AB_STRATEGIES[0]
        )
        candidate_cross_layer_contract = _cross_layer_jacobian_execution_contract(
            candidate, expected_strategy=CROSS_LAYER_JACOBIAN_AB_STRATEGIES[1]
        )
        receipt_contract = _cross_layer_jacobian_receipt_contract(reference, candidate)
        cross_layer_jacobian_contract = {
            "reference": reference_cross_layer_contract,
            "candidate": candidate_cross_layer_contract,
            "exact_receipts": receipt_contract,
            "passed": (
                reference_cross_layer_contract["passed"]
                and candidate_cross_layer_contract["passed"]
                and receipt_contract["passed"]
            ),
        }
        gates.append(
            {
                "gate": "canonical_cross_layer_jacobian_ab_pair",
                "required": True,
                "passed": cross_layer_jacobian_contract["passed"],
            }
        )
    stop_gradient_selected_attribution_forward_contract = None
    if require_canonical_stop_gradient_selected_attribution_forward_ab:
        stop_gradient_selected_attribution_forward_contract = (
            _stop_gradient_selected_attribution_forward_contract(
                reference,
                candidate,
            )
        )
        gates.append(
            {
                "gate": (
                    "canonical_stop_gradient_selected_attribution_forward_ab_pair"
                ),
                "required": True,
                "passed": stop_gradient_selected_attribution_forward_contract["passed"],
            }
        )
    stop_gradient_selected_attribution_storage_contract = None
    if require_canonical_stop_gradient_selected_attribution_storage_ab:
        stop_gradient_selected_attribution_storage_contract = (
            _stop_gradient_selected_attribution_storage_contract(
                reference,
                candidate,
            )
        )
        gates.append(
            {
                "gate": (
                    "canonical_stop_gradient_selected_attribution_storage_ab_pair"
                ),
                "required": True,
                "passed": stop_gradient_selected_attribution_storage_contract["passed"],
            }
        )
    selected_target_logit_execution_contract = None
    if require_canonical_selected_target_logit_execution_ab:
        selected_target_logit_execution_contract = (
            _selected_target_logit_execution_contract(reference, candidate)
        )
        gates.append(
            {
                "gate": "canonical_selected_target_logit_execution_ab_pair",
                "required": True,
                "passed": selected_target_logit_execution_contract["passed"],
            }
        )
    selected_neuron_contribution_chunk_contract = None
    selected_attribution_neuron_lane_chunk_contract = None
    if require_canonical_selected_attribution_neuron_lane_chunk_ab:
        reference_identity = reference.manifest.get("artifact_identity")
        candidate_identity = candidate.manifest.get("artifact_identity")
        reference_adag = (
            reference_identity.get("adag_config")
            if isinstance(reference_identity, Mapping)
            else None
        )
        candidate_adag = (
            candidate_identity.get("adag_config")
            if isinstance(candidate_identity, Mapping)
            else None
        )
        field = "selected_attribution_neuron_lane_chunk_size"
        reference_strategy_valid = (
            isinstance(reference_adag, Mapping)
            and field in reference_adag
            and reference_adag[field]
            == SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_STRATEGIES[0]
        )
        candidate_strategy_valid = (
            isinstance(candidate_adag, Mapping)
            and field in candidate_adag
            and candidate_adag[field]
            == SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_STRATEGIES[1]
        )
        runtime_contract = _selected_attribution_neuron_lane_runtime_contract(
            reference, candidate
        )
        selected_attribution_neuron_lane_chunk_contract = {
            "reference_strategy": {
                "expected": SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_STRATEGIES[0],
                "observed": (
                    reference_adag.get(field)
                    if isinstance(reference_adag, Mapping)
                    else None
                ),
                "field_present": (
                    isinstance(reference_adag, Mapping) and field in reference_adag
                ),
                "passed": reference_strategy_valid,
            },
            "candidate_strategy": {
                "expected": SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_STRATEGIES[1],
                "observed": (
                    candidate_adag.get(field)
                    if isinstance(candidate_adag, Mapping)
                    else None
                ),
                "field_present": (
                    isinstance(candidate_adag, Mapping) and field in candidate_adag
                ),
                "passed": candidate_strategy_valid,
            },
            "runtime_width_receipts": runtime_contract,
            "passed": (
                reference_strategy_valid
                and candidate_strategy_valid
                and runtime_contract["passed"]
            ),
        }
        gates.append(
            {
                "gate": "canonical_selected_attribution_neuron_lane_chunk_ab_pair",
                "required": True,
                "passed": selected_attribution_neuron_lane_chunk_contract["passed"],
            }
        )
    if require_canonical_selected_neuron_contribution_target_lane_chunk_ab:
        reference_identity = reference.manifest.get("artifact_identity")
        candidate_identity = candidate.manifest.get("artifact_identity")
        reference_adag = (
            reference_identity.get("adag_config")
            if isinstance(reference_identity, Mapping)
            else None
        )
        candidate_adag = (
            candidate_identity.get("adag_config")
            if isinstance(candidate_identity, Mapping)
            else None
        )
        field = "selected_neuron_contribution_target_lane_chunk_size"
        reference_strategy_valid = (
            isinstance(reference_adag, Mapping)
            and field in reference_adag
            and reference_adag[field]
            == SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_STRATEGIES[0]
        )
        candidate_strategy_valid = (
            isinstance(candidate_adag, Mapping)
            and field in candidate_adag
            and candidate_adag[field]
            == SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_STRATEGIES[1]
        )
        receipt_contract = _selected_neuron_contribution_receipt_contract(
            reference, candidate
        )
        selected_neuron_contribution_chunk_contract = {
            "reference_strategy": {
                "expected": (
                    SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_STRATEGIES[0]
                ),
                "observed": (
                    reference_adag.get(field)
                    if isinstance(reference_adag, Mapping)
                    else None
                ),
                "field_present": (
                    isinstance(reference_adag, Mapping) and field in reference_adag
                ),
                "passed": reference_strategy_valid,
            },
            "candidate_strategy": {
                "expected": (
                    SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_STRATEGIES[1]
                ),
                "observed": (
                    candidate_adag.get(field)
                    if isinstance(candidate_adag, Mapping)
                    else None
                ),
                "field_present": (
                    isinstance(candidate_adag, Mapping) and field in candidate_adag
                ),
                "passed": candidate_strategy_valid,
            },
            "exact_receipts": receipt_contract,
            "passed": (
                reference_strategy_valid
                and candidate_strategy_valid
                and receipt_contract["passed"]
            ),
        }
        gates.append(
            {
                "gate": "canonical_selected_neuron_contribution_target_lane_chunk_ab_pair",
                "required": True,
                "passed": selected_neuron_contribution_chunk_contract["passed"],
            }
        )
    selected_embed_contribution_chunk_contract = None
    if selected_embed_contribution_target_lane_chunk_ab_profile is not None:
        reference_identity = reference.manifest.get("artifact_identity")
        candidate_identity = candidate.manifest.get("artifact_identity")
        reference_adag = (
            reference_identity.get("adag_config")
            if isinstance(reference_identity, Mapping)
            else None
        )
        candidate_adag = (
            candidate_identity.get("adag_config")
            if isinstance(candidate_identity, Mapping)
            else None
        )
        field = "selected_embed_contribution_target_lane_chunk_size"
        expected_reference, expected_candidate = (
            SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_PROFILES[
                selected_embed_contribution_target_lane_chunk_ab_profile
            ]
        )
        reference_strategy_valid = (
            isinstance(reference_adag, Mapping)
            and field in reference_adag
            and reference_adag[field] == expected_reference
        )
        candidate_strategy_valid = (
            isinstance(candidate_adag, Mapping)
            and field in candidate_adag
            and candidate_adag[field] == expected_candidate
        )
        require_exact_hashes = (
            selected_embed_contribution_target_lane_chunk_ab_profile
            == "full_width_exact_v1"
        )
        receipt_contract = _embed_contribution_receipt_contract(
            reference,
            candidate,
            execution_record_namespace="selected_embed_contribution_vjp",
            execution_contract="ordinary_direct_v1",
            expected_reference_width=expected_reference,
            expected_candidate_width=expected_candidate,
            require_exact_hashes=require_exact_hashes,
        )
        bf16_scope_contract = (
            _selected_embed_bf16_scope_contract(
                reference,
                candidate,
                reference_nodes=reference_nodes,
                candidate_nodes=candidate_nodes,
                reference_edges=reference_edges,
                candidate_edges=candidate_edges,
                candidate_count=reference_trace.candidate_count,
                source_attribution_tolerance=NumericTolerance(
                    absolute=0.125, relative=1e-2
                ),
                candidate_profile_tolerance=NumericTolerance(
                    absolute=0.125, relative=1e-2
                ),
                edge_tolerance=NumericTolerance(absolute=5e-4, relative=1e-2),
            )
            if selected_embed_contribution_target_lane_chunk_ab_profile
            == "width_one_bf16_v1"
            else None
        )
        selected_embed_contribution_chunk_contract = {
            "profile": selected_embed_contribution_target_lane_chunk_ab_profile,
            "reference_strategy": {
                "expected": expected_reference,
                "observed": (
                    reference_adag.get(field)
                    if isinstance(reference_adag, Mapping)
                    else None
                ),
                "field_present": (
                    isinstance(reference_adag, Mapping) and field in reference_adag
                ),
                "passed": reference_strategy_valid,
            },
            "candidate_strategy": {
                "expected": expected_candidate,
                "observed": (
                    candidate_adag.get(field)
                    if isinstance(candidate_adag, Mapping)
                    else None
                ),
                "field_present": (
                    isinstance(candidate_adag, Mapping) and field in candidate_adag
                ),
                "passed": candidate_strategy_valid,
            },
            "projected_receipts": receipt_contract,
            "bf16_scope": bf16_scope_contract,
            "passed": (
                reference_strategy_valid
                and candidate_strategy_valid
                and receipt_contract["passed"]
                and (bf16_scope_contract is None or bf16_scope_contract["passed"])
            ),
        }
        gates.append(
            {
                "gate": "canonical_selected_embed_contribution_target_lane_chunk_ab_pair",
                "required": True,
                "passed": selected_embed_contribution_chunk_contract["passed"],
            }
        )
    stop_gradient_embed_contribution_chunk_contract = None
    if stop_gradient_embed_contribution_target_lane_chunk_ab_profile is not None:
        reference_identity = reference.manifest.get("artifact_identity")
        candidate_identity = candidate.manifest.get("artifact_identity")
        reference_adag = (
            reference_identity.get("adag_config")
            if isinstance(reference_identity, Mapping)
            else None
        )
        candidate_adag = (
            candidate_identity.get("adag_config")
            if isinstance(candidate_identity, Mapping)
            else None
        )
        field = "stop_gradient_embed_contribution_target_lane_chunk_size"
        expected_reference, expected_candidate = (
            STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_PROFILES[
                stop_gradient_embed_contribution_target_lane_chunk_ab_profile
            ]
        )
        reference_strategy_valid = (
            isinstance(reference_adag, Mapping)
            and field in reference_adag
            and reference_adag[field] == expected_reference
        )
        candidate_strategy_valid = (
            isinstance(candidate_adag, Mapping)
            and field in candidate_adag
            and candidate_adag[field] == expected_candidate
        )
        receipt_contract = _embed_contribution_receipt_contract(
            reference,
            candidate,
            execution_record_namespace="stop_gradient_embed_contribution_vjp",
            execution_contract="stop_gradient_direct_v1",
            expected_reference_width=expected_reference,
            expected_candidate_width=expected_candidate,
            require_exact_hashes=True,
        )
        stop_gradient_embed_contribution_chunk_contract = {
            "profile": stop_gradient_embed_contribution_target_lane_chunk_ab_profile,
            "reference_strategy": {
                "expected": expected_reference,
                "observed": (
                    reference_adag.get(field)
                    if isinstance(reference_adag, Mapping)
                    else None
                ),
                "field_present": (
                    isinstance(reference_adag, Mapping) and field in reference_adag
                ),
                "passed": reference_strategy_valid,
            },
            "candidate_strategy": {
                "expected": expected_candidate,
                "observed": (
                    candidate_adag.get(field)
                    if isinstance(candidate_adag, Mapping)
                    else None
                ),
                "field_present": (
                    isinstance(candidate_adag, Mapping) and field in candidate_adag
                ),
                "passed": candidate_strategy_valid,
            },
            "projected_receipts": receipt_contract,
            "passed": (
                reference_strategy_valid
                and candidate_strategy_valid
                and receipt_contract["passed"]
            ),
        }
        gates.append(
            {
                "gate": (
                    "canonical_stop_gradient_embed_contribution_"
                    "target_lane_chunk_ab_pair"
                ),
                "required": True,
                "passed": stop_gradient_embed_contribution_chunk_contract["passed"],
            }
        )
    if require_same_gpu_family:
        gates.append(
            {
                "gate": "same_gpu_family",
                "required": True,
                "passed": same_gpu_family,
            }
        )
    if require_same_gpu_model:
        gates.append(
            {
                "gate": "same_gpu_model",
                "required": True,
                "passed": same_gpu_model,
            }
        )
    if require_exact_node_topology:
        gates.append(
            {
                "gate": "exact_node_topology",
                "required": True,
                "passed": node_topology["exact"],
            }
        )
    if require_exact_edge_topology:
        gates.append(
            {
                "gate": "exact_edge_topology",
                "required": True,
                "passed": edge_topology["exact"],
            }
        )
    numerical_groups = {
        "target": target_values,
        "node": node_values,
        "edge": edge_values,
        "candidate_profile": candidate_profiles,
    }
    gates.extend(
        {
            "gate": f"{group}_numeric_tolerance",
            "required": True,
            "passed": _group_within_tolerance(numerical_groups[group]),
            "tolerance": tolerance_map[group].to_dict(),
        }
        for group in TOLERANCE_GROUPS
        if group in tolerance_map
    )
    result_gate_names = {
        "exact_node_topology",
        "exact_edge_topology",
        *(f"{group}_numeric_tolerance" for group in TOLERANCE_GROUPS),
    }
    result_gate_count = sum(gate["gate"] in result_gate_names for gate in gates)
    validation_passed = all(gate["passed"] for gate in gates)

    report_schema = (
        EXECUTION_REPORT_SCHEMA
        if {
            "artifact_identity.adag_config.stop_gradient_contribution_execution",
            "artifact_identity.adag_config."
            "stop_gradient_contribution_target_lane_chunk_size",
            "artifact_identity.adag_config."
            "selected_neuron_contribution_target_lane_chunk_size",
            "artifact_identity.adag_config."
            "selected_embed_contribution_target_lane_chunk_size",
            "artifact_identity.adag_config."
            "stop_gradient_embed_contribution_target_lane_chunk_size",
            "artifact_identity.adag_config.selected_attribution_neuron_lane_chunk_size",
            "artifact_identity.adag_config."
            "stop_gradient_selected_attribution_forward_execution",
            "artifact_identity.adag_config.stop_gradient_selected_attribution_storage",
            "artifact_identity.adag_config.selected_target_logit_execution",
            "artifact_identity.cuda_allocator_policy",
            "artifact_identity.adag_config.embedding_edge_materialization",
            "artifact_identity.adag_config.cross_layer_jacobian_execution",
        }
        & set(allowed_paths)
        else REPORT_SCHEMA
    )

    return {
        "schema_version": report_schema,
        "validation_passed": validation_passed,
        "qualification_passed": validation_passed if result_gate_count else None,
        "diagnostic_only": result_gate_count == 0,
        "result_gate_count": result_gate_count,
        "scientific_parity_claimed": False,
        "interpretation": (
            "validation_passed covers identity and every explicitly requested "
            "gate. qualification_passed is null for diagnostic-only reports; "
            "otherwise it covers the requested result gates. Neither is a "
            "scientific parity claim."
        ),
        "artifacts": {
            "reference": {
                "path": str(reference.path),
                "artifact_id": reference.manifest.get("artifact_id"),
                "payload_sha256": reference.manifest.get("data_sha256"),
            },
            "candidate": {
                "path": str(candidate.path),
                "artifact_id": candidate.manifest.get("artifact_id"),
                "payload_sha256": candidate.manifest.get("data_sha256"),
            },
        },
        "identity": {
            "hard_checks": hard_identity_checks,
            "reference_integrity": reference_identity_integrity,
            "candidate_integrity": candidate_identity_integrity,
            "artifact_identity": artifact_identity,
        },
        "hardware": {
            "reference": reference_gpu,
            "candidate": candidate_gpu,
            "same_family": same_gpu_family,
            "same_model": same_gpu_model,
            "require_same_family": require_same_gpu_family,
            "require_same_model": require_same_gpu_model,
        },
        "cuda_allocator_ab_contract": allocator_contract,
        "embedding_edge_ab_contract": embedding_edge_contract,
        "cross_layer_jacobian_ab_contract": cross_layer_jacobian_contract,
        "stop_gradient_selected_attribution_forward_ab_contract": (
            stop_gradient_selected_attribution_forward_contract
        ),
        "stop_gradient_selected_attribution_storage_ab_contract": (
            stop_gradient_selected_attribution_storage_contract
        ),
        "selected_target_logit_execution_ab_contract": (
            selected_target_logit_execution_contract
        ),
        "selected_attribution_neuron_lane_chunk_ab_contract": (
            selected_attribution_neuron_lane_chunk_contract
        ),
        "selected_neuron_contribution_target_lane_chunk_ab_contract": (
            selected_neuron_contribution_chunk_contract
        ),
        "selected_embed_contribution_target_lane_chunk_ab_contract": (
            selected_embed_contribution_chunk_contract
        ),
        "stop_gradient_embed_contribution_target_lane_chunk_ab_contract": (
            stop_gradient_embed_contribution_chunk_contract
        ),
        "counts": {
            "reference": {
                "nodes": len(reference_nodes),
                "edges": len(reference_edges),
                "candidates": reference_trace.candidate_count,
            },
            "candidate": {
                "nodes": len(candidate_nodes),
                "edges": len(candidate_edges),
                "candidates": candidate_trace.candidate_count,
            },
        },
        "topology": {"nodes": node_topology, "edges": edge_topology},
        "target_values": target_values,
        "node_values_on_intersection": node_values,
        "edge_values_on_intersection": edge_values,
        "candidate_profiles_on_node_intersection": candidate_profiles,
        "resources": {
            "trace_wall_seconds": _metric_pair(
                reference.metrics,
                candidate.metrics,
                "trace_wall_seconds",
                "elapsed_seconds",
            ),
            "cuda_peak_allocated_bytes": _metric_pair(
                reference.metrics, candidate.metrics, "cuda_peak_allocated_bytes"
            ),
            "cuda_peak_reserved_bytes": _metric_pair(
                reference.metrics, candidate.metrics, "cuda_peak_reserved_bytes"
            ),
            "rss_peak_after_bytes": _metric_pair(
                reference.metrics, candidate.metrics, "rss_peak_after_bytes"
            ),
        },
        "gates": gates,
    }


# Historical compatibility name. New callers should use the execution-generic
# name so reports are not described as attention-only qualifications.
compare_attention_backend_artifacts = compare_execution_artifacts
