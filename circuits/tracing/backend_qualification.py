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
CROSS_LAYER_JACOBIAN_RECEIPT_NAMES = (
    "selected_source_activations",
    "selected_target_activations",
    "selected_raw_jacobian",
)

_ALLOWABLE_SCALAR_IDENTITY_RULES = (
    "artifact_identity.adag_config.stop_gradient_attention_backend",
    "artifact_identity.adag_config.stop_gradient_contribution_execution",
    "artifact_identity.adag_config.stop_gradient_contribution_target_lane_chunk_size",
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
                "backend, stop-gradient contribution execution or target-lane "
                "chunk size, CUDA allocator policy, embedding-edge "
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
) -> dict[str, Any]:
    """Compare two saved execution traces under explicit qualification gates.

    The historical public name is retained for compatibility. Callers may
    explicitly qualify a named attention, contribution, allocator, or
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
