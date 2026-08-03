"""Frozen C2 inputs for candidate-aware signed-basis clustering.

This module only validates and packages already completed discovery artifacts.  It
does not fit clusters, inspect labels, open the confirmatory holdout, or call a
model.  Width-one and candidate profiles retain separate semantics so downstream
code cannot accidentally concatenate unlike feature axes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.analysis.bonafide.identity import Polarity, SignedBasisKey
from circuits.tracing.artifact import load_compact_trace
from circuits.tracing.candidate_union import (
    CandidateUnionArtifact,
    load_candidate_union_artifact,
)

CANDIDATE_CLUSTER_INPUT_SCHEMA = "adag.bonafide.candidate-cluster-inputs.v1"
CANDIDATE_CLUSTER_PARTITION_SCHEMA = (
    "adag.bonafide.candidate-cluster-family-partitions.v1"
)

FROZEN_SELECTION_SHA256 = (
    "67cd4b2d5bf2fe2558b2abe55060b75d098856ccdb71be1df771b2278ca869b1"
)
FROZEN_PLAN_FILE_SHA256 = (
    "7d8a79144a6616a30ac67802d2975ef8a7fe5f92bfc1f879ed1a980fbe0a7724"
)
FROZEN_PLAN_CANONICAL_SHA256 = (
    "0390409e9488fa822be50c0114242369bfa50ff27559d6da40ba182e6d91ae68"
)
FROZEN_C2_REPORT_SHA256 = (
    "9ea1123685e73bc45f8c93490429a2a309ed62953406d61509d0014730ef6530"
)
FROZEN_SALVAGE_REPORT_SHA256 = (
    "3ce0f45c1d05e97310481ec4bc42462b09c7490108d9e921a3490fa3620fa0b3"
)
FROZEN_ARTIFACT_PAYLOAD_SET_SHA256 = (
    "9e73332c93c3dcf3b8ea9c8f0f5c107f5e51fe3a0bba21114b1372cbfa64ef94"
)
FROZEN_PROTOCOL_SHA256 = (
    "1e24d333fcf9b595bceea9ef42c12bbc0726af22c66ce2a161fd9a1ca45d7983"
)
FROZEN_CLUSTERING_EVALUATION_SHA256 = (
    "f4c215486239e9fc46fb2232869eeff6633cf080d4ea15f314d37ed04a185555"
)
FROZEN_CLUSTERING_EVALUATION_REPORT_SCHEMA = (
    "adag.bonafide.clustering-structural-report.v1"
)

PARTITION_CAPACITIES = {
    "generation": 18,
    "selection_scoring": 8,
    "audit": 8,
}
PARTITION_ORDER = tuple(PARTITION_CAPACITIES)
SAFE_EXAMPLE_FIELDS = (
    "base_question_id",
    "example_id",
    "hint_datasets",
    "hint_types",
    "prompt",
    "question",
    "response",
    "src_types",
    "target_model",
    "token_counts",
)
SAFE_DIVERSITY_FIELDS = (
    "hint_datasets",
    "hint_types",
    "question_novelty_control_family_marker",
    "response_length_bin",
    "src_types",
    "total_length_bin",
)


def _basis_fields() -> list[pa.Field]:
    return [
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("layer", pa.int32(), nullable=False),
        pa.field("neuron_index", pa.int64(), nullable=False),
        pa.field("polarity", pa.string(), nullable=False),
    ]


TARGET_SCHEMA = pa.schema(
    [
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("source_width1_artifact_id", pa.string(), nullable=False),
        pa.field("width1_artifact_id", pa.string(), nullable=False),
        pa.field("width1_payload_sha256", pa.string(), nullable=False),
        pa.field("candidate_union_artifact_id", pa.string(), nullable=False),
        pa.field("candidate_union_payload_sha256", pa.string(), nullable=False),
        pa.field("candidate_union_topology_sha256", pa.string(), nullable=False),
        pa.field("base_question_id", pa.string(), nullable=False),
        pa.field("response_id", pa.string(), nullable=False),
        pa.field("phase_bin", pa.int8(), nullable=False),
        pa.field("response_position", pa.int32(), nullable=False),
        pa.field("family_partition", pa.string(), nullable=False),
        pa.field("partition_hierarchical_weight", pa.float64(), nullable=False),
        pa.field("candidate_count", pa.int8(), nullable=False),
        pa.field("observed_token_id", pa.int64(), nullable=False),
        pa.field("observed_token_text", pa.string(), nullable=False),
        pa.field("candidate_selection_json", pa.string(), nullable=False),
        pa.field("example_json", pa.string(), nullable=False),
        pa.field("width_signed_basis_count", pa.int32(), nullable=False),
        pa.field("candidate_signed_basis_count", pa.int32(), nullable=False),
        pa.field("zero_activation_width_occurrence_count", pa.int32(), nullable=False),
        pa.field(
            "zero_activation_candidate_occurrence_count", pa.int32(), nullable=False
        ),
        pa.field(
            "candidate_activation_invariance_max_abs_deviation",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "candidate_activation_invariance_max_relative_deviation",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "candidate_activation_invariance_violation_count",
            pa.int32(),
            nullable=False,
        ),
        pa.field(
            "candidate_activation_invariance_comparison_count",
            pa.int32(),
            nullable=False,
        ),
        pa.field("width_polarity_crosswalk_json", pa.string(), nullable=False),
        pa.field("candidate_polarity_crosswalk_json", pa.string(), nullable=False),
    ]
)

BASIS_INDEX_SCHEMA = pa.schema(
    [
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        *_basis_fields(),
        pa.field("in_width_support", pa.bool_(), nullable=False),
        pa.field("in_width_view", pa.bool_(), nullable=False),
        pa.field("in_candidate_support", pa.bool_(), nullable=False),
        pa.field("in_candidate_view", pa.bool_(), nullable=False),
        pa.field("width_support_target_count", pa.int32(), nullable=False),
        pa.field("width_target_count", pa.int32(), nullable=False),
        pa.field("candidate_support_target_count", pa.int32(), nullable=False),
        pa.field("candidate_target_count", pa.int32(), nullable=False),
        pa.field("width_support_generation_target_count", pa.int32(), nullable=False),
        pa.field("width_generation_target_count", pa.int32(), nullable=False),
        pa.field(
            "candidate_support_generation_target_count", pa.int32(), nullable=False
        ),
        pa.field("candidate_generation_target_count", pa.int32(), nullable=False),
        pa.field("width_support_generation_response_count", pa.int32(), nullable=False),
        pa.field("width_generation_response_count", pa.int32(), nullable=False),
        pa.field(
            "candidate_support_generation_response_count", pa.int32(), nullable=False
        ),
        pa.field("candidate_generation_response_count", pa.int32(), nullable=False),
        pa.field("width_support_generation_family_count", pa.int32(), nullable=False),
        pa.field("width_generation_family_count", pa.int32(), nullable=False),
        pa.field(
            "candidate_support_generation_family_count", pa.int32(), nullable=False
        ),
        pa.field("candidate_generation_family_count", pa.int32(), nullable=False),
    ]
)

WIDTH_PROFILE_SCHEMA = pa.schema(
    [
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        *_basis_fields(),
        pa.field("attribution_profile", pa.list_(pa.float64()), nullable=False),
        pa.field("attribution_support", pa.list_(pa.bool_()), nullable=False),
        pa.field("signed_attribution", pa.float64(), nullable=False),
        pa.field("occurrence_count", pa.int32(), nullable=False),
    ]
)

CANDIDATE_PROFILE_SCHEMA = pa.schema(
    [
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("signed_basis_index", pa.int64(), nullable=False),
        *_basis_fields(),
        pa.field("candidate_contrast_profile", pa.list_(pa.float64()), nullable=False),
        pa.field("candidate_profile_l2_norm", pa.float64(), nullable=False),
        pa.field("occurrence_count", pa.int32(), nullable=False),
    ]
)


@dataclass(frozen=True)
class WidthBasisProfile:
    values: tuple[float | None, ...]
    support: tuple[bool, ...]
    signed_attribution: float
    occurrence_count: int


@dataclass(frozen=True)
class CandidateBasisProfile:
    values: tuple[float, float, float, float, float]
    occurrence_count: int


@dataclass(frozen=True)
class ValidatedTargetProfiles:
    target: dict[str, Any]
    width: Mapping[SignedBasisKey, WidthBasisProfile]
    candidate: Mapping[SignedBasisKey, CandidateBasisProfile]


class _CandidateNodeRow(Protocol):
    """Columns required from a pandas candidate-node named tuple."""

    layer: Any
    neuron: Any
    applicable_by_candidate: Any
    candidate_activation: Any
    candidate_attribution: Any
    candidate_contribution: Any


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _sequence(value: object, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a one-dimensional sequence")
    try:
        result = list(cast(Iterable[Any], value))
    except TypeError as error:
        raise ValueError(f"{field} must be a one-dimensional sequence") from error
    return result


def _activation_polarity(value: float) -> Polarity | None:
    if value > 0:
        return "+"
    if value < 0:
        return "-"
    return None


def rank_aligned_candidate_contrasts(
    *,
    full_distribution_ranks: Sequence[int],
    observed_index: int,
    contribution_values: Sequence[float],
) -> tuple[float, float, float, float, float]:
    """Return rank-one-through-five contribution minus observed contribution."""

    if len(full_distribution_ranks) not in {5, 6}:
        raise ValueError("candidate measurements must have width five or six")
    if len(full_distribution_ranks) != len(contribution_values):
        raise ValueError("candidate ranks and contribution values disagree in width")
    if observed_index < 0 or observed_index >= len(contribution_values):
        raise ValueError("observed candidate index is outside the candidate axis")
    if len(set(full_distribution_ranks)) != len(full_distribution_ranks):
        raise ValueError("candidate full-distribution ranks are not unique")
    by_rank = {int(rank): index for index, rank in enumerate(full_distribution_ranks)}
    if any(rank not in by_rank for rank in range(1, 6)):
        raise ValueError("candidate axis must contain model ranks one through five")
    values = [_finite(value, "candidate contribution") for value in contribution_values]
    observed = values[observed_index]
    return (
        values[by_rank[1]] - observed,
        values[by_rank[2]] - observed,
        values[by_rank[3]] - observed,
        values[by_rank[4]] - observed,
        values[by_rank[5]] - observed,
    )


def extract_width_one_profiles(
    frame: Any,
    *,
    model_id: str,
    model_revision: str,
) -> tuple[dict[SignedBasisKey, WidthBasisProfile], int, dict[str, int]]:
    """Aggregate source-token attribution maps using production activation polarity."""

    if frame.empty:
        raise ValueError("width-one node frame cannot be empty")
    final_layer = int(frame["layer"].max())
    accumulators: dict[SignedBasisKey, dict[str, Any]] = {}
    zero_activation_count = 0
    polarity_crosswalk: Counter[str] = Counter()
    expected_width: int | None = None
    for row in frame.itertuples(index=False):
        layer = int(row.layer)
        if not 0 <= layer < final_layer:
            continue
        activation = _finite(row.activation, "width-one activation")
        polarity = _activation_polarity(activation)
        attribution_polarity = _activation_polarity(
            _finite(row.attribution, "width-one attribution")
        )
        polarity_crosswalk[
            f"activation_{polarity or 'zero'}__attribution_{attribution_polarity or 'zero'}"
        ] += 1
        if polarity is None:
            zero_activation_count += 1
            continue
        raw_profile = _sequence(row.attr_map, "width-one attr_map")
        if expected_width is None:
            expected_width = len(raw_profile)
        elif len(raw_profile) != expected_width:
            raise ValueError("width-one attribution-map widths disagree")
        values = [
            None if value is None else _finite(value, "width-one attr_map")
            for value in raw_profile
        ]
        basis = SignedBasisKey(
            model_id=model_id,
            model_revision=model_revision,
            layer=layer,
            neuron_index=int(row.neuron),
            polarity=polarity,
        )
        accumulator = accumulators.setdefault(
            basis,
            {
                "values": [0.0] * len(values),
                "support": [False] * len(values),
                "signed_attribution": 0.0,
                "occurrence_count": 0,
            },
        )
        for index, value in enumerate(values):
            if value is not None:
                accumulator["values"][index] += value
                accumulator["support"][index] = True
        accumulator["signed_attribution"] += float(row.attribution)
        accumulator["occurrence_count"] += 1
    profiles = {
        basis: WidthBasisProfile(
            values=tuple(
                value if supported else None
                for value, supported in zip(
                    accumulator["values"], accumulator["support"], strict=True
                )
            ),
            support=tuple(accumulator["support"]),
            signed_attribution=float(accumulator["signed_attribution"]),
            occurrence_count=int(accumulator["occurrence_count"]),
        )
        for basis, accumulator in sorted(accumulators.items())
    }
    if not profiles:
        raise ValueError("width-one target has no supported non-boundary MLP basis")
    return profiles, zero_activation_count, dict(sorted(polarity_crosswalk.items()))


def extract_candidate_profiles(
    artifact: CandidateUnionArtifact,
    *,
    model_id: str,
    model_revision: str,
) -> tuple[dict[SignedBasisKey, CandidateBasisProfile], int, dict[str, Any]]:
    """Aggregate five rank-aligned candidate contrasts by activation-signed basis."""

    trace = artifact.trace
    candidates = tuple(trace.candidate_selection.candidates)
    observed = [
        index for index, candidate in enumerate(candidates) if candidate.is_observed
    ]
    if observed != [0]:
        raise ValueError("candidate union must place exactly one observed token first")
    ranks = [int(candidate.full_distribution_rank) for candidate in candidates]
    final_layer = int(trace.df_node["layer"].max())
    sums: dict[SignedBasisKey, NDArray[np.float64]] = {}
    counts: Counter[SignedBasisKey] = Counter()
    zero_activation_count = 0
    invariance_rtol = 1e-6
    invariance_atol = 1e-7
    invariance_rows = 0
    invariance_comparisons = 0
    invariance_violations = 0
    maximum_absolute_deviation = 0.0
    maximum_relative_deviation = 0.0
    polarity_crosswalk: Counter[str] = Counter()
    for raw_row in trace.df_node.itertuples(index=False):
        row = cast(_CandidateNodeRow, raw_row)
        layer = int(row.layer)
        if not 0 <= layer < final_layer:
            continue
        applicable = [
            bool(value)
            for value in _sequence(
                row.applicable_by_candidate, "candidate applicability"
            )
        ]
        if len(applicable) != len(candidates) or not all(applicable):
            raise ValueError("candidate MLP row lacks complete candidate applicability")
        activations = [
            _finite(value, "candidate activation")
            for value in _sequence(row.candidate_activation, "candidate activation")
        ]
        if len(activations) != len(candidates):
            raise ValueError("candidate activation width disagrees with selection")
        invariance_rows += 1
        observed_activation = activations[0]
        for value in activations[1:]:
            invariance_comparisons += 1
            absolute_deviation = abs(value - observed_activation)
            relative_deviation = absolute_deviation / max(
                abs(observed_activation), 1e-12
            )
            maximum_absolute_deviation = max(
                maximum_absolute_deviation, absolute_deviation
            )
            maximum_relative_deviation = max(
                maximum_relative_deviation, relative_deviation
            )
            if not bool(
                np.isclose(
                    value,
                    observed_activation,
                    rtol=invariance_rtol,
                    atol=invariance_atol,
                )
            ):
                invariance_violations += 1
        attribution_values = [
            _finite(value, "candidate attribution")
            for value in _sequence(row.candidate_attribution, "candidate attribution")
        ]
        if len(attribution_values) != len(candidates):
            raise ValueError("candidate attribution width disagrees with selection")
        polarity = _activation_polarity(activations[0])
        attribution_polarity = _activation_polarity(attribution_values[0])
        polarity_crosswalk[
            f"activation_{polarity or 'zero'}__attribution_{attribution_polarity or 'zero'}"
        ] += 1
        if polarity is None:
            zero_activation_count += 1
            continue
        contributions = [
            _finite(value, "candidate contribution")
            for value in _sequence(row.candidate_contribution, "candidate contribution")
        ]
        contrast = np.asarray(
            rank_aligned_candidate_contrasts(
                full_distribution_ranks=ranks,
                observed_index=0,
                contribution_values=contributions,
            ),
            dtype=np.float64,
        )
        basis = SignedBasisKey(
            model_id=model_id,
            model_revision=model_revision,
            layer=layer,
            neuron_index=int(row.neuron),
            polarity=polarity,
        )
        if basis in sums:
            sums[basis] += contrast
        else:
            sums[basis] = contrast.copy()
        counts[basis] += 1
    profiles = {
        basis: CandidateBasisProfile(
            values=(
                float(sums[basis][0]),
                float(sums[basis][1]),
                float(sums[basis][2]),
                float(sums[basis][3]),
                float(sums[basis][4]),
            ),
            occurrence_count=counts[basis],
        )
        for basis in sorted(sums)
    }
    if not profiles:
        raise ValueError("candidate target has no supported non-boundary MLP basis")
    if invariance_violations:
        raise ValueError(
            "candidate activation invariance failed: "
            f"violations={invariance_violations}, "
            f"max_abs={maximum_absolute_deviation}, "
            f"max_relative={maximum_relative_deviation}"
        )
    return (
        profiles,
        zero_activation_count,
        {
            "activation_invariance": {
                "rtol": invariance_rtol,
                "atol": invariance_atol,
                "internal_node_row_count": invariance_rows,
                "comparison_count": invariance_comparisons,
                "max_abs_deviation": maximum_absolute_deviation,
                "max_relative_deviation": maximum_relative_deviation,
                "violation_count": 0,
            },
            "c2_attribution_sign_to_production_activation_sign_crosswalk": dict(
                sorted(polarity_crosswalk.items())
            ),
        },
    )


def _condition_markers(example: Mapping[str, Any]) -> frozenset[str]:
    diversity = example.get("diversity")
    if not isinstance(diversity, Mapping):
        diversity = {}
    markers: set[str] = set()
    safe_fields: dict[str, Any] = {
        key: example[key] for key in SAFE_DIVERSITY_FIELDS if key in example
    }
    for key in SAFE_DIVERSITY_FIELDS:
        if key in diversity:
            safe_fields[key] = diversity[key]
    for key, raw_value in sorted(safe_fields.items()):
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            markers.add(
                f"{key}="
                + json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
    return frozenset(markers)


def _redacted_example(example: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only prompt/witness and non-outcome condition fields."""

    result = {key: example[key] for key in SAFE_EXAMPLE_FIELDS if key in example}
    diversity = example.get("diversity")
    if isinstance(diversity, Mapping):
        safe_diversity = {
            key: diversity[key] for key in SAFE_DIVERSITY_FIELDS if key in diversity
        }
        if safe_diversity:
            result["diversity"] = safe_diversity
    return result


def build_family_partitions(
    cases: Sequence[Mapping[str, Any]],
    *,
    examples_by_response: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an exact deterministic 18/8/8 family split without outcome data."""

    responses_by_family: dict[str, set[str]] = defaultdict(set)
    phase_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for case in cases:
        family = str(case["base_question_id"])
        response = str(case["example_id"])
        responses_by_family[family].add(response)
        phase_counts[family][int(case["phase_bin"])] += 1
    if len(responses_by_family) != 34:
        raise ValueError("candidate clustering requires exactly 34 discovery families")
    if set(examples_by_response) != set().union(*responses_by_family.values()):
        raise ValueError("partition examples do not match the C2 response universe")

    families: dict[str, dict[str, Any]] = {}
    for family in sorted(responses_by_family):
        marker_counts: Counter[str] = Counter()
        for response in sorted(responses_by_family[family]):
            marker_counts.update(_condition_markers(examples_by_response[response]))
        families[family] = {
            "response_count": len(responses_by_family[family]),
            "phase_counts": dict(sorted(phase_counts[family].items())),
            "marker_counts": dict(sorted(marker_counts.items())),
        }

    salt = b"candidate-aware-labelability-v1\0"
    ordered_families = sorted(
        families,
        key=lambda family_id: (
            hashlib.sha256(salt + family_id.encode("utf-8")).hexdigest(),
            family_id,
        ),
    )
    slots = [
        role for role in PARTITION_ORDER for _ in range(PARTITION_CAPACITIES[role])
    ]
    assignments = {
        family: slots[index] for index, family in enumerate(ordered_families)
    }

    role_to_families = {
        role: sorted(
            family for family, assigned in assignments.items() if assigned == role
        )
        for role in PARTITION_ORDER
    }
    if {
        role: len(values) for role, values in role_to_families.items()
    } != PARTITION_CAPACITIES:
        raise ValueError("family partition capacities drifted")
    if set().union(*(set(values) for values in role_to_families.values())) != set(
        families
    ):
        raise ValueError("family partitions do not exactly cover the cohort")
    balance_diagnostics: dict[str, Any] = {}
    for role, family_ids in role_to_families.items():
        role_phases: Counter[int] = Counter()
        role_markers: Counter[str] = Counter()
        for family_id in family_ids:
            role_phases.update(families[family_id]["phase_counts"])
            role_markers.update(families[family_id]["marker_counts"])
        balance_diagnostics[role] = {
            "family_count": len(family_ids),
            "response_count": sum(
                int(families[family_id]["response_count"]) for family_id in family_ids
            ),
            "phase_bin_target_counts": dict(sorted(role_phases.items())),
            "condition_marker_response_counts": dict(sorted(role_markers.items())),
        }
    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_CLUSTER_PARTITION_SCHEMA,
        "algorithm": "sha256_salted_family_order_exact_slices_v1",
        "namespace_utf8": "candidate-aware-labelability-v1",
        "hash_preimage": "namespace_utf8_then_nul_byte_then_family_id_utf8",
        "ordering": "ascending_sha256_then_canonical_family_id",
        "slice_definition": "first_18_generation_next_8_selection_scoring_final_8_audit",
        "capacities": dict(PARTITION_CAPACITIES),
        "partitions": role_to_families,
        "family_to_partition": dict(sorted(assignments.items())),
        "ordered_families": ordered_families,
        "family_statistics": families,
        "balance_diagnostics": balance_diagnostics,
        "outcome_fields_used": [],
    }
    payload["partitions_sha256"] = canonical_sha256(payload)
    return payload


def _verify_artifact_identity(
    manifest: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    identity = manifest.get("artifact_identity")
    if not isinstance(identity, Mapping):
        raise TypeError(f"{label} manifest lacks artifact_identity")
    value = dict(identity)
    recorded = value.pop("sha256", None)
    if recorded != canonical_sha256(value):
        raise ValueError(f"{label} artifact_identity hash mismatch")
    return identity


def _index_width_artifacts(root: Path, source_ids: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for manifest_path in root.rglob("manifest.json"):
        try:
            manifest = load_json_object(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        source_id = manifest.get("source_artifact_id")
        if source_id not in source_ids:
            continue
        if source_id in result:
            raise ValueError(f"duplicate width-one artifact for {source_id}")
        result[str(source_id)] = manifest_path.parent
    missing = source_ids - set(result)
    if missing:
        raise ValueError(f"missing {len(missing)} width-one C2 artifacts")
    return result


def _index_union_artifacts(
    root: Path, source_ids: set[str], *, plan_sha256: str
) -> dict[str, Path]:
    family_root = root / "bonafide.candidate-union.v1"
    if not family_root.is_dir():
        raise ValueError(f"candidate-union artifact family is missing: {family_root}")
    result: dict[str, Path] = {}
    for manifest_path in family_root.rglob("manifest.json"):
        manifest = load_json_object(manifest_path)
        if manifest.get("candidate_union_plan_sha256") != plan_sha256:
            raise ValueError(
                f"candidate-union plan provenance drift: {manifest_path.parent}"
            )
        contract = manifest.get("candidate_union_contract")
        if not isinstance(contract, Mapping):
            raise TypeError("candidate-union manifest lacks trace contract")
        source_id = contract.get("source_width1_artifact_id")
        if source_id not in source_ids:
            raise ValueError(
                f"candidate-union root contains unplanned source {source_id}"
            )
        if source_id in result:
            raise ValueError(f"duplicate candidate-union artifact for {source_id}")
        result[str(source_id)] = manifest_path.parent
    missing = source_ids - set(result)
    if missing:
        raise ValueError(f"missing {len(missing)} candidate-union C2 artifacts")
    return result


def _validated_frozen_inputs(
    *,
    selection_path: Path,
    plan_path: Path,
    c2_report_path: Path,
    salvage_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if file_sha256(selection_path) != FROZEN_SELECTION_SHA256:
        raise ValueError("frozen C2 selection file hash drift")
    if file_sha256(plan_path) != FROZEN_PLAN_FILE_SHA256:
        raise ValueError("frozen candidate-union plan file hash drift")
    if file_sha256(c2_report_path) != FROZEN_C2_REPORT_SHA256:
        raise ValueError("audited C2 report file hash drift")
    if file_sha256(salvage_report_path) != FROZEN_SALVAGE_REPORT_SHA256:
        raise ValueError("post-hoc C2 salvage report file hash drift")
    selection = load_json_object(selection_path)
    plan = load_json_object(plan_path)
    report = load_json_object(c2_report_path)
    if canonical_sha256(plan) != FROZEN_PLAN_CANONICAL_SHA256:
        raise ValueError("frozen candidate-union plan canonical hash drift")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("audited C2 report lacks inputs")
    expected = {
        "selection_file_sha256": FROZEN_SELECTION_SHA256,
        "plan_file_sha256": FROZEN_PLAN_FILE_SHA256,
        "plan_canonical_sha256": FROZEN_PLAN_CANONICAL_SHA256,
        "artifact_payload_set_sha256": FROZEN_ARTIFACT_PAYLOAD_SET_SHA256,
    }
    for field, value in expected.items():
        if inputs.get(field) != value:
            raise ValueError(f"audited C2 report provenance drift: {field}")
    cases = selection.get("cases")
    if not isinstance(cases, list) or len(cases) != 245:
        raise ValueError("frozen C2 selection must contain exactly 245 cases")
    plan_cases = [
        case for wave in plan.get("waves", []) for case in wave.get("cases", [])
    ]
    if len(plan_cases) != 245:
        raise ValueError("candidate-union plan must contain exactly 245 cases")
    return selection, plan, report


def _validate_audited_artifact_payloads(
    profiles: Sequence[ValidatedTargetProfiles], report: Mapping[str, Any]
) -> str:
    """Match loaded artifact identities to the audited C2 report and payload set."""

    raw_diagnostics = report.get("target_diagnostics")
    if (
        not isinstance(raw_diagnostics, list)
        or not raw_diagnostics
        or len(raw_diagnostics) != len(profiles)
    ):
        raise ValueError("audited C2 report target diagnostics are incomplete")
    audited_by_case: dict[str, Mapping[str, Any]] = {}
    for raw in raw_diagnostics:
        if not isinstance(raw, Mapping):
            raise TypeError("audited C2 target diagnostic must be an object")
        case_id = str(raw.get("case_id"))
        if case_id in audited_by_case:
            raise ValueError(f"duplicate audited C2 target diagnostic: {case_id}")
        audited_by_case[case_id] = raw
    current_by_case = {str(profile.target["case_id"]): profile for profile in profiles}
    if len(current_by_case) != len(profiles) or set(current_by_case) != set(
        audited_by_case
    ):
        raise ValueError("loaded target coverage differs from audited C2 diagnostics")
    field_pairs = {
        "source_width1_artifact_id": "source_width1_artifact_id",
        "base_question_id": "base_question_id",
        "response_id": "example_id",
        "phase_bin": "phase_bin",
        "width1_artifact_id": "width1_artifact_id",
        "width1_payload_sha256": "width1_payload_sha256",
        "candidate_union_artifact_id": "candidate_union_artifact_id",
        "candidate_union_payload_sha256": "candidate_union_payload_sha256",
        "candidate_union_topology_sha256": "candidate_union_topology_sha256",
    }
    for case_id, audited in audited_by_case.items():
        target = current_by_case[case_id].target
        for current_field, audited_field in field_pairs.items():
            if target[current_field] != audited.get(audited_field):
                raise ValueError(
                    f"loaded artifact differs from audited C2 target {case_id}: "
                    f"{current_field}"
                )
    payload_records = [
        {
            "source_width1_artifact_id": current_by_case[str(raw["case_id"])].target[
                "source_width1_artifact_id"
            ],
            "width1_payload_sha256": current_by_case[str(raw["case_id"])].target[
                "width1_payload_sha256"
            ],
            "candidate_union_payload_sha256": current_by_case[
                str(raw["case_id"])
            ].target["candidate_union_payload_sha256"],
        }
        for raw in raw_diagnostics
    ]
    payload_set_sha256 = canonical_sha256(payload_records)
    if payload_set_sha256 != FROZEN_ARTIFACT_PAYLOAD_SET_SHA256:
        raise ValueError("loaded artifact payload-set hash differs from frozen C2")
    return payload_set_sha256


def load_validated_target_profiles(
    *,
    selection_path: Path,
    plan_path: Path,
    c2_report_path: Path,
    salvage_report_path: Path,
    width1_root: Path,
    candidate_union_root: Path,
) -> tuple[
    list[ValidatedTargetProfiles], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Validate the frozen C2 cohort and reconstruct both clustering views."""

    selection, plan, report = _validated_frozen_inputs(
        selection_path=selection_path,
        plan_path=plan_path,
        c2_report_path=c2_report_path,
        salvage_report_path=salvage_report_path,
    )
    cases = [dict(case) for case in selection["cases"]]
    selected_by_source = {
        str(case["source_width1_artifact_id"]): case for case in cases
    }
    plan_by_source = {
        str(case["source_width1_artifact_id"]): case
        for wave in plan["waves"]
        for case in wave["cases"]
    }
    if len(selected_by_source) != 245 or set(selected_by_source) != set(plan_by_source):
        raise ValueError("C2 selection and plan source coverage disagree")
    source_ids = set(selected_by_source)
    width_paths = _index_width_artifacts(width1_root, source_ids)
    union_paths = _index_union_artifacts(
        candidate_union_root,
        source_ids,
        plan_sha256=FROZEN_PLAN_CANONICAL_SHA256,
    )
    model_id = str(plan["source"]["model_id"])
    model_revision = str(plan["source"]["model_revision"])
    chat_template_sha256 = str(plan["source"]["chat_template_sha256"])
    profiles: list[ValidatedTargetProfiles] = []
    for source_id in sorted(
        source_ids, key=lambda value: str(selected_by_source[value]["case_id"])
    ):
        case = selected_by_source[source_id]
        planned = plan_by_source[source_id]
        width = load_compact_trace(width_paths[source_id])
        union = load_candidate_union_artifact(union_paths[source_id])
        expected_position = int(case["target_response_position"])
        expected_example = planned["source_item"]["example"]
        expected_selection = planned["source_item"]["target_selection"]
        expected_source_item_sha256 = canonical_sha256(planned["source_item"])
        for label, manifest in (
            ("width-one", width.manifest),
            ("candidate-union", union.manifest),
        ):
            if (
                manifest.get("numerically_valid") is not True
                or manifest.get("scientifically_reusable") is not True
            ):
                raise ValueError(f"{label} artifact is not reusable numerical evidence")
            if (
                manifest.get("model_id") != model_id
                or manifest.get("model_revision") != model_revision
            ):
                raise ValueError(f"{label} model identity drift for {source_id}")
            if manifest.get("bonafide_example") != expected_example:
                raise ValueError(f"{label} example provenance drift for {source_id}")
            if manifest.get("source_target_selection") != expected_selection:
                raise ValueError(f"{label} target-selection drift for {source_id}")
        width_identity = _verify_artifact_identity(width.manifest, "width-one")
        union_identity = _verify_artifact_identity(union.manifest, "candidate-union")
        if width.manifest.get("source_artifact_id") != source_id:
            raise ValueError("width-one source artifact ID drift")
        if (
            width_identity.get("source_artifact_id") != source_id
            or width_identity.get("source_work_item_sha256")
            != expected_source_item_sha256
        ):
            raise ValueError("width-one source-work-item identity drift")
        if (
            width.manifest.get("trace_metadata", {}).get("chat_template_sha256")
            != chat_template_sha256
        ):
            raise ValueError("width-one chat-template provenance drift")
        if (
            union_identity.get("candidate_union_plan_sha256")
            != FROZEN_PLAN_CANONICAL_SHA256
            or union_identity.get("source_width1_artifact_id") != source_id
            or union_identity.get("source_work_item_sha256")
            != expected_source_item_sha256
            or union_identity.get("reference_artifacts")
            != planned["reference_artifacts"]
            or union_identity.get("trace_family_id") != "bonafide.candidate-union.v1"
        ):
            raise ValueError("candidate-union frozen artifact identity drift")
        if union_identity.get("adag_config") != width_identity.get("adag_config"):
            raise ValueError("width-one and candidate-union ADAG configs disagree")
        if union_identity.get("model") != width_identity.get("model"):
            raise ValueError("width-one and candidate-union model configs disagree")
        if union.trace.source_width1_artifact_id != source_id:
            raise ValueError("candidate-union source artifact ID drift")
        if union.trace.topology_sha256 != planned["frozen_union_topology_sha256"]:
            raise ValueError("candidate-union topology drift")
        if width.circuit_data.target_logits != [
            [int(expected_selection["final_target_token_id"])]
        ]:
            raise ValueError("width-one observed target token drift")
        if union.trace.shared_response_position != expected_position:
            raise ValueError("candidate-union response position drift")
        candidate_ids = [
            candidate.token_id
            for candidate in union.trace.candidate_selection.candidates
        ]
        if candidate_ids != list(case["candidate_token_ids"]):
            raise ValueError("candidate token selection drift")
        planned_references = [
            (str(item["artifact_id"]), str(item["payload_sha256"]))
            for item in planned["reference_artifacts"]
        ]
        trace_references = [
            (str(item["artifact_id"]), str(item["payload_sha256"]))
            for item in union.trace.reference_artifacts
        ]
        if trace_references != planned_references:
            raise ValueError("candidate reference payload provenance drift")
        width_profiles, width_zero, width_crosswalk = extract_width_one_profiles(
            width.circuit_data.df_node,
            model_id=model_id,
            model_revision=model_revision,
        )
        candidate_profiles, candidate_zero, candidate_diagnostics = (
            extract_candidate_profiles(
                union,
                model_id=model_id,
                model_revision=model_revision,
            )
        )
        selection_dict = union.trace.candidate_selection.to_dict()
        profiles.append(
            ValidatedTargetProfiles(
                target={
                    "case_id": str(case["case_id"]),
                    "source_width1_artifact_id": source_id,
                    "width1_artifact_id": str(width.manifest["artifact_id"]),
                    "width1_payload_sha256": str(width.manifest["data_sha256"]),
                    "candidate_union_artifact_id": str(union.manifest["artifact_id"]),
                    "candidate_union_payload_sha256": str(
                        union.manifest["data_sha256"]
                    ),
                    "candidate_union_topology_sha256": union.trace.topology_sha256,
                    "base_question_id": str(case["base_question_id"]),
                    "response_id": str(case["example_id"]),
                    "phase_bin": int(case["phase_bin"]),
                    "response_position": expected_position,
                    "candidate_count": len(candidate_ids),
                    "observed_token_id": union.trace.candidate_selection.observed_token_id,
                    "observed_token_text": union.trace.candidate_selection.observed_token_text,
                    "candidate_selection_json": json.dumps(
                        selection_dict,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    "example_json": json.dumps(
                        _redacted_example(expected_example),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    "width_signed_basis_count": len(width_profiles),
                    "candidate_signed_basis_count": len(candidate_profiles),
                    "zero_activation_width_occurrence_count": width_zero,
                    "zero_activation_candidate_occurrence_count": candidate_zero,
                    "candidate_activation_invariance_max_abs_deviation": (
                        candidate_diagnostics["activation_invariance"][
                            "max_abs_deviation"
                        ]
                    ),
                    "candidate_activation_invariance_max_relative_deviation": (
                        candidate_diagnostics["activation_invariance"][
                            "max_relative_deviation"
                        ]
                    ),
                    "candidate_activation_invariance_violation_count": (
                        candidate_diagnostics["activation_invariance"][
                            "violation_count"
                        ]
                    ),
                    "candidate_activation_invariance_comparison_count": (
                        candidate_diagnostics["activation_invariance"][
                            "comparison_count"
                        ]
                    ),
                    "width_polarity_crosswalk_json": json.dumps(
                        width_crosswalk,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "candidate_polarity_crosswalk_json": json.dumps(
                        candidate_diagnostics[
                            "c2_attribution_sign_to_production_activation_sign_crosswalk"
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                },
                width=width_profiles,
                candidate=candidate_profiles,
            )
        )
    _validate_audited_artifact_payloads(profiles, report)
    return profiles, selection, plan, report


def _basis_record(basis: SignedBasisKey) -> dict[str, Any]:
    return {
        "model_id": basis.model_id,
        "model_revision": basis.model_revision,
        "layer": basis.layer,
        "neuron_index": basis.neuron_index,
        "polarity": basis.polarity,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _partition_hierarchical_weights(
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return equal-family/response/target weights within each frozen partition."""

    response_targets: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in target_rows:
        response_targets[str(row["family_partition"])][str(row["base_question_id"])][
            str(row["response_id"])
        ].append(str(row["case_id"]))
    weights: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    for partition, families in sorted(response_targets.items()):
        family_count = len(families)
        if family_count != PARTITION_CAPACITIES[partition]:
            raise ValueError(f"unexpected family count in {partition}")
        for responses in families.values():
            response_count = len(responses)
            for case_ids in responses.values():
                target_weight = 1.0 / family_count / response_count / len(case_ids)
                for case_id in case_ids:
                    if case_id in weights:
                        raise ValueError(f"duplicate target identity: {case_id}")
                    weights[case_id] = target_weight
        partition_total = math.fsum(
            weights[case_id]
            for families_by_id in families.values()
            for case_ids in families_by_id.values()
            for case_id in case_ids
        )
        if not math.isclose(partition_total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"hierarchical weights do not sum to one in {partition}")
        diagnostics[partition] = {
            "family_count": family_count,
            "target_count": sum(
                len(case_ids)
                for responses in families.values()
                for case_ids in responses.values()
            ),
            "weight_sum": partition_total,
        }
    if set(weights) != {str(row["case_id"]) for row in target_rows}:
        raise ValueError("hierarchical weights do not cover every target")
    return weights, diagnostics


def _build_basis_index(
    profiles: Sequence[ValidatedTargetProfiles],
    family_to_partition: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[SignedBasisKey, int]]:
    """Freeze one canonical index for the union of the W and C signed bases."""

    view_names = (
        "width_support",
        "width_directional",
        "candidate_support",
        "candidate_directional",
    )
    view_targets: dict[str, dict[SignedBasisKey, set[str]]] = {
        view: defaultdict(set) for view in view_names
    }
    generation_responses: dict[str, dict[SignedBasisKey, set[str]]] = {
        view: defaultdict(set) for view in view_names
    }
    generation_families: dict[str, dict[SignedBasisKey, set[str]]] = {
        view: defaultdict(set) for view in view_names
    }
    generation_targets: dict[str, dict[SignedBasisKey, set[str]]] = {
        view: defaultdict(set) for view in view_names
    }
    all_bases: set[SignedBasisKey] = set()
    for profile in profiles:
        family_id = str(profile.target["base_question_id"])
        response_id = str(profile.target["response_id"])
        case_id = str(profile.target["case_id"])
        is_generation = family_to_partition[family_id] == "generation"
        values_by_view: dict[str, Mapping[SignedBasisKey, object]] = {
            "width_support": profile.width,
            "width_directional": {
                basis: value
                for basis, value in profile.width.items()
                if float(
                    np.linalg.norm([item for item in value.values if item is not None])
                )
                > 0.0
            },
            "candidate_support": profile.candidate,
            "candidate_directional": {
                basis: value
                for basis, value in profile.candidate.items()
                if float(np.linalg.norm(value.values)) > 0.0
            },
        }
        for view, values in values_by_view.items():
            for basis in values:
                all_bases.add(basis)
                view_targets[view][basis].add(case_id)
                if is_generation:
                    generation_targets[view][basis].add(case_id)
                    generation_responses[view][basis].add(response_id)
                    generation_families[view][basis].add(family_id)
    ordered = sorted(all_bases)
    index_by_basis = {basis: index for index, basis in enumerate(ordered)}
    rows = [
        {
            "signed_basis_index": index_by_basis[basis],
            **_basis_record(basis),
            "in_width_support": bool(view_targets["width_support"][basis]),
            "in_width_view": bool(view_targets["width_directional"][basis]),
            "in_candidate_support": bool(view_targets["candidate_support"][basis]),
            "in_candidate_view": bool(view_targets["candidate_directional"][basis]),
            "width_support_target_count": len(view_targets["width_support"][basis]),
            "width_target_count": len(view_targets["width_directional"][basis]),
            "candidate_support_target_count": len(
                view_targets["candidate_support"][basis]
            ),
            "candidate_target_count": len(view_targets["candidate_directional"][basis]),
            "width_support_generation_target_count": len(
                generation_targets["width_support"][basis]
            ),
            "width_generation_target_count": len(
                generation_targets["width_directional"][basis]
            ),
            "candidate_support_generation_target_count": len(
                generation_targets["candidate_support"][basis]
            ),
            "candidate_generation_target_count": len(
                generation_targets["candidate_directional"][basis]
            ),
            "width_support_generation_response_count": len(
                generation_responses["width_support"][basis]
            ),
            "width_generation_response_count": len(
                generation_responses["width_directional"][basis]
            ),
            "candidate_support_generation_response_count": len(
                generation_responses["candidate_support"][basis]
            ),
            "candidate_generation_response_count": len(
                generation_responses["candidate_directional"][basis]
            ),
            "width_support_generation_family_count": len(
                generation_families["width_support"][basis]
            ),
            "width_generation_family_count": len(
                generation_families["width_directional"][basis]
            ),
            "candidate_support_generation_family_count": len(
                generation_families["candidate_support"][basis]
            ),
            "candidate_generation_family_count": len(
                generation_families["candidate_directional"][basis]
            ),
        }
        for basis in ordered
    ]
    if [row["signed_basis_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("canonical signed-basis index is not contiguous")
    return rows, index_by_basis


def _crosswalk_summary(
    target_rows: Sequence[Mapping[str, Any]], *, field: str
) -> dict[str, Any]:
    """Aggregate polarity agreement with occurrence counts and frozen target weights."""

    overall: Counter[str] = Counter()
    by_partition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_partition_mass: dict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    summary_counts: dict[str, Counter[str]] = defaultdict(Counter)
    summary_mass: dict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in target_rows:
        partition = str(row["family_partition"])
        weight = float(row["partition_hierarchical_weight"])
        crosswalk = json.loads(str(row[field]))
        for key, raw_count in crosswalk.items():
            count = int(raw_count)
            overall[key] += count
            by_partition_counts[partition][key] += count
            by_partition_mass[partition][key] += weight * count
            activation, attribution = key.split("__", maxsplit=1)
            activation_sign = activation.removeprefix("activation_")
            attribution_sign = attribution.removeprefix("attribution_")
            if "zero" in {activation_sign, attribution_sign}:
                category = "zero_or_unsupported"
            elif activation_sign == attribution_sign:
                category = "agreement"
            else:
                category = "disagreement"
            summary_counts[partition][category] += count
            summary_mass[partition][category] += weight * count
    return {
        "all_unweighted_occurrence_counts": dict(sorted(overall.items())),
        "by_partition": {
            partition: {
                "unweighted_occurrence_counts": dict(
                    sorted(by_partition_counts[partition].items())
                ),
                "hierarchical_weighted_occurrence_mass": dict(
                    sorted(by_partition_mass[partition].items())
                ),
                "agreement_summary_counts": dict(
                    sorted(summary_counts[partition].items())
                ),
                "agreement_summary_weighted_mass": dict(
                    sorted(summary_mass[partition].items())
                ),
            }
            for partition in PARTITION_ORDER
        },
    }


def build_candidate_cluster_input_bundle(
    *,
    selection_path: Path,
    plan_path: Path,
    c2_report_path: Path,
    salvage_report_path: Path,
    width1_root: Path,
    candidate_union_root: Path,
    output_root: Path,
    code_revision: Mapping[str, Any],
    protocol_path: Path,
) -> dict[str, Any]:
    """Atomically persist validated profiles plus an outcome-free family split."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"candidate cluster input bundle already exists: {output_root}"
        )
    if file_sha256(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("frozen candidate-clustering protocol hash drift")
    if code_revision.get("git_dirty") is not False:
        raise ValueError(
            "candidate clustering inputs require a clean scoped source tree"
        )
    source_hashes = {
        str(record["path"]): str(record["sha256"])
        for record in code_revision.get("files", [])
    }
    expected_source_hashes = {
        "docs/CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md": (
            FROZEN_PROTOCOL_SHA256
        ),
        "circuits/analysis/bonafide/clustering_evaluation.py": (
            FROZEN_CLUSTERING_EVALUATION_SHA256
        ),
    }
    for source_path, expected_hash in expected_source_hashes.items():
        if source_hashes.get(source_path) != expected_hash:
            raise ValueError(f"bound source hash drift: {source_path}")
    profiles, selection, plan, report = load_validated_target_profiles(
        selection_path=selection_path,
        plan_path=plan_path,
        c2_report_path=c2_report_path,
        salvage_report_path=salvage_report_path,
        width1_root=width1_root,
        candidate_union_root=candidate_union_root,
    )
    examples_by_response = {
        str(profile.target["response_id"]): json.loads(profile.target["example_json"])
        for profile in profiles
    }
    partitions = build_family_partitions(
        selection["cases"], examples_by_response=examples_by_response
    )
    family_to_partition = partitions["family_to_partition"]
    target_rows = [
        {
            **profile.target,
            "family_partition": family_to_partition[profile.target["base_question_id"]],
        }
        for profile in profiles
    ]
    target_weights, weight_diagnostics = _partition_hierarchical_weights(target_rows)
    target_rows = [
        {
            **row,
            "partition_hierarchical_weight": target_weights[str(row["case_id"])],
        }
        for row in target_rows
    ]
    basis_rows, index_by_basis = _build_basis_index(
        profiles, family_to_partition=family_to_partition
    )
    width_rows = [
        {
            "case_id": profile.target["case_id"],
            "signed_basis_index": index_by_basis[basis],
            **_basis_record(basis),
            "attribution_profile": list(value.values),
            "attribution_support": list(value.support),
            "signed_attribution": value.signed_attribution,
            "occurrence_count": value.occurrence_count,
        }
        for profile in profiles
        for basis, value in sorted(profile.width.items())
    ]
    candidate_rows = [
        {
            "case_id": profile.target["case_id"],
            "signed_basis_index": index_by_basis[basis],
            **_basis_record(basis),
            "candidate_contrast_profile": list(value.values),
            "candidate_profile_l2_norm": float(np.linalg.norm(value.values)),
            "occurrence_count": value.occurrence_count,
        }
        for profile in profiles
        for basis, value in sorted(profile.candidate.items())
    ]
    width_crosswalk = _crosswalk_summary(
        target_rows, field="width_polarity_crosswalk_json"
    )
    candidate_crosswalk = _crosswalk_summary(
        target_rows, field="candidate_polarity_crosswalk_json"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        pq.write_table(
            pa.Table.from_pylist(basis_rows, schema=BASIS_INDEX_SCHEMA),
            temporary / "basis-index.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        pq.write_table(
            pa.Table.from_pylist(target_rows, schema=TARGET_SCHEMA),
            temporary / "targets.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        pq.write_table(
            pa.Table.from_pylist(width_rows, schema=WIDTH_PROFILE_SCHEMA),
            temporary / "width-profiles.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        pq.write_table(
            pa.Table.from_pylist(candidate_rows, schema=CANDIDATE_PROFILE_SCHEMA),
            temporary / "candidate-profiles.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        _write_json(temporary / "family-partitions.json", partitions)
        files = []
        for path in sorted(temporary.iterdir()):
            if path.name == "manifest.json":
                continue
            record: dict[str, Any] = {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            if path.suffix == ".parquet":
                record["row_count"] = pq.read_metadata(path).num_rows
            files.append(record)
        manifest: dict[str, Any] = {
            "schema_version": CANDIDATE_CLUSTER_INPUT_SCHEMA,
            "purpose": "frozen_inputs_only_no_cluster_fit_or_description_generation",
            "inputs": {
                "selection": {
                    "path": str(selection_path.resolve()),
                    "sha256": FROZEN_SELECTION_SHA256,
                },
                "candidate_union_plan": {
                    "path": str(plan_path.resolve()),
                    "file_sha256": FROZEN_PLAN_FILE_SHA256,
                    "canonical_sha256": FROZEN_PLAN_CANONICAL_SHA256,
                },
                "audited_c2_report": {
                    "path": str(c2_report_path.resolve()),
                    "sha256": FROZEN_C2_REPORT_SHA256,
                },
                "posthoc_salvage_report": {
                    "path": str(salvage_report_path.resolve()),
                    "sha256": FROZEN_SALVAGE_REPORT_SHA256,
                },
                "artifact_payload_set_sha256": FROZEN_ARTIFACT_PAYLOAD_SET_SHA256,
                "width1_root": str(width1_root.resolve()),
                "candidate_union_root": str(candidate_union_root.resolve()),
            },
            "protocol": {
                "path": str(protocol_path.resolve()),
                "sha256": FROZEN_PROTOCOL_SHA256,
            },
            "structural_evaluation_contract": {
                "path": "circuits/analysis/bonafide/clustering_evaluation.py",
                "sha256": FROZEN_CLUSTERING_EVALUATION_SHA256,
                "report_schema": FROZEN_CLUSTERING_EVALUATION_REPORT_SCHEMA,
            },
            "cohort": {
                "target_count": len(profiles),
                "response_count": len({row["response_id"] for row in target_rows}),
                "family_count": len({row["base_question_id"] for row in target_rows}),
                "phase_bin_counts": dict(
                    sorted(Counter(row["phase_bin"] for row in target_rows).items())
                ),
                "candidate_width_counts": dict(
                    sorted(
                        Counter(row["candidate_count"] for row in target_rows).items()
                    )
                ),
                "width_profile_row_count": len(width_rows),
                "candidate_profile_row_count": len(candidate_rows),
                "signed_basis_count": len(basis_rows),
                "shared_view_signed_basis_count": sum(
                    row["in_width_view"] and row["in_candidate_view"]
                    for row in basis_rows
                ),
                "candidate_activation_invariance": {
                    "rtol": 1e-6,
                    "atol": 1e-7,
                    "max_abs_deviation": max(
                        row["candidate_activation_invariance_max_abs_deviation"]
                        for row in target_rows
                    ),
                    "max_relative_deviation": max(
                        row["candidate_activation_invariance_max_relative_deviation"]
                        for row in target_rows
                    ),
                    "violation_count": sum(
                        row["candidate_activation_invariance_violation_count"]
                        for row in target_rows
                    ),
                    "comparison_count": sum(
                        row["candidate_activation_invariance_comparison_count"]
                        for row in target_rows
                    ),
                },
                "hierarchical_weight_diagnostics": weight_diagnostics,
                "width_attribution_sign_to_production_activation_sign_crosswalk": (
                    width_crosswalk
                ),
                "candidate_attribution_sign_to_production_activation_sign_crosswalk": (
                    candidate_crosswalk
                ),
            },
            "feature_contract": {
                "basis_identity": "model_revision_layer_neuron_activation_sign",
                "zero_activation": "unsupported",
                "boundary_layers": "excluded",
                "occurrence_reducer": "signed_sum_within_target_basis",
                "width_view": "source_token_attr_map_missing_aware",
                "candidate_view": "rank_1_through_5_contribution_minus_observed",
                "candidate_widths": [5, 6],
                "candidate_activation_invariance": {
                    "reference": "candidate_index_0_observed",
                    "rtol": 1e-6,
                    "atol": 1e-7,
                    "violations": "fatal",
                },
                "polarity_crosswalk": (
                    "separate_width_and_candidate_attribution_sign_to_activation_sign"
                ),
                "example_fields_retained": list(SAFE_EXAMPLE_FIELDS),
                "outcome_fields_excluded": [
                    "annotation_row_ids",
                    "label_types",
                    "labeling_reasons",
                    "selection_membership",
                    "diversity.answer_relation",
                    "diversity.annotation_position_bin",
                    "diversity.cot_phenotype",
                    "diversity.label_types",
                ],
                "views_kept_separate": True,
                "canonical_basis_index": (
                    "ascending_signed_basis_key_union_of_width_and_candidate_views"
                ),
            },
            "partition_manifest_sha256": partitions["partitions_sha256"],
            "files": files,
            "source_report_schema": report.get("schema_version"),
            "source_plan_schema": plan.get("schema_version"),
            "code_revision": dict(code_revision),
            "outcomes_inspected": False,
            "model_calls_made": False,
            "cluster_fit_performed": False,
            "confirmatory_holdout_opened": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
