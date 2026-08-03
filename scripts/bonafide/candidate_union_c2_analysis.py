"""Evaluate the frozen C2 candidate-union scientific-utility protocol.

The input trace trees are treated as immutable evidence.  This module validates
their coverage and provenance, constructs sparse signed-MLP profiles, and writes
one deterministic JSON report.  It does not perform tracing or mutate an input
artifact.

Run from the repository root as a module, for example
``python -m scripts.bonafide.candidate_union_c2_analysis --help``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from circuits.tracing.artifact import load_compact_trace
from circuits.tracing.candidate_union import load_candidate_union_artifact
from numpy.typing import NDArray

from scripts.bonafide.candidate_union_runner import validate_candidate_union_plan
from scripts.bonafide.runner import collect_code_revision, collect_runtime_environment

REPORT_SCHEMA_VERSION = "bonafide-candidate-union-c2-analysis/v1"
SELECTION_SCHEMA_VERSION = "bonafide-topk-c2-cohort-selection/v1"
MIN_COMMON_BASES = 16
UTILITY_MRR_THRESHOLD = 0.03
NONDEGENERACY_MEDIAN_EFFECTIVE_RANK = 2.0
NONDEGENERACY_TARGET_EFFECTIVE_RANK = 1.5
NONDEGENERACY_TARGET_FRACTION = 0.90
LOFO_POSITIVE_FRACTION = 0.80
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260730

# Model and revision are included deliberately: profiles from different neuron
# identity spaces can never intersect accidentally.
Basis = tuple[str, str, int, int, int]


@dataclass(frozen=True)
class TargetProfile:
    case_id: str
    source_artifact_id: str
    family_id: str
    response_id: str
    phase_bin: int
    width1: Mapping[Basis, float]
    candidate: Mapping[Basis, NDArray[np.float64]]
    raw_candidate: Mapping[Basis, NDArray[np.float64]]
    effective_rank: float
    bidirectional_raw_fraction: float
    boundary_width1_occurrences: int = 0
    boundary_candidate_occurrences: int = 0
    width1_artifact_id: str = ""
    width1_payload_sha256: str = ""
    candidate_union_artifact_id: str = ""
    candidate_union_payload_sha256: str = ""
    topology_sha256: str = ""


@dataclass(frozen=True)
class RetrievalRecord:
    anchor_id: str
    family_id: str
    response_id: str
    reciprocal_rank: float
    top_one: float
    true_rank: int
    candidate_count: int
    planned_candidate_count: int
    common_basis_support: tuple[int, ...]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _validate_identity(
    manifest: Mapping[str, Any], artifact_name: str
) -> Mapping[str, Any]:
    identity = manifest.get("artifact_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{artifact_name} manifest lacks an artifact identity")
    claimed = identity.get("sha256")
    identity_value = {key: value for key, value in identity.items() if key != "sha256"}
    if not isinstance(claimed, str) or claimed != _canonical_sha256(identity_value):
        raise ValueError(f"{artifact_name} artifact identity digest is invalid")
    return identity


def _validate_bound_json_file(
    owner: Mapping[str, Any], path_field: str, sha256_field: str
) -> dict[str, Any]:
    path_value = owner.get(path_field)
    expected_sha256 = owner.get(sha256_field)
    if not isinstance(path_value, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"frozen file binding lacks {path_field}/{sha256_field}")
    path = Path(path_value)
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen file binding drift: {path}")
    return _load_json(path)


def rank_aligned_contrasts(
    full_distribution_ranks: Sequence[int],
    observed_index: int,
    contribution_values: Sequence[float],
) -> NDArray[np.float64]:
    """Return rank-1..5 contribution-minus-observed contrasts."""

    if len(full_distribution_ranks) != len(contribution_values):
        raise ValueError("candidate ranks and contributions have different widths")
    if observed_index < 0 or observed_index >= len(contribution_values):
        raise ValueError("observed candidate index is outside the candidate axis")
    if len(set(full_distribution_ranks)) != len(full_distribution_ranks):
        raise ValueError("candidate full-distribution ranks are not unique")
    by_rank = {rank: index for index, rank in enumerate(full_distribution_ranks)}
    if any(rank not in by_rank for rank in range(1, 6)):
        raise ValueError("candidate axis does not contain every model rank 1 through 5")
    values = np.asarray(contribution_values, dtype=np.float64)
    if values.shape != (len(contribution_values),) or not np.isfinite(values).all():
        raise ValueError("candidate contributions must be a finite vector")
    observed = values[observed_index]
    result = np.asarray([values[by_rank[rank]] - observed for rank in range(1, 6)])
    if not np.isfinite(result).all():
        raise ValueError("candidate contrasts are not finite")
    return result


def effective_rank(matrix: NDArray[Any]) -> float:
    """Entropy effective rank using normalized singular values."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 5 or not np.isfinite(value).all():
        raise ValueError("effective-rank input must be a finite bases-by-five matrix")
    if value.shape[0] == 0:
        return 0.0
    singular = np.linalg.svd(value, compute_uv=False)
    total = float(singular.sum())
    if total == 0.0:
        return 0.0
    probabilities = singular[singular > 0.0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def directional_similarity(
    left: Mapping[Basis, float | NDArray[Any]],
    right: Mapping[Basis, float | NDArray[Any]],
    *,
    min_common_bases: int = MIN_COMMON_BASES,
) -> tuple[float, int] | None:
    """Cosine similarity on the exact intersection of supported signed bases."""

    common = sorted(set(left).intersection(right))
    if len(common) < min_common_bases:
        return None
    left_vector = np.concatenate(
        [np.asarray(left[key], dtype=np.float64).reshape(-1) for key in common]
    )
    right_vector = np.concatenate(
        [np.asarray(right[key], dtype=np.float64).reshape(-1) for key in common]
    )
    if left_vector.shape != right_vector.shape:
        raise ValueError("directional profiles disagree on channel width")
    if not np.isfinite(left_vector).all() or not np.isfinite(right_vector).all():
        raise ValueError("directional profiles contain non-finite values")
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    score = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
    if not math.isfinite(score):
        raise ValueError("directional similarity is not finite")
    return score, len(common)


def _polarity(value: float) -> int | None:
    if not math.isfinite(value):
        raise ValueError("attribution polarity requires a finite value")
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return None


def _extract_width1_profile(
    artifact: Any,
    *,
    model_id: str,
    model_revision: str,
) -> tuple[dict[Basis, float], int]:
    frame = artifact.circuit_data.df_node
    final_layer = int(frame["layer"].max())
    result: dict[Basis, float] = defaultdict(float)
    boundary = 0
    for _, row in frame.iterrows():
        layer = int(row["layer"])
        if not 0 <= layer < final_layer:
            continue
        attribution = float(row["attribution"])
        polarity = _polarity(attribution)
        if polarity is None:
            boundary += 1
            continue
        basis = (model_id, model_revision, layer, int(row["neuron"]), polarity)
        result[basis] += attribution
    if not result or not all(math.isfinite(value) for value in result.values()):
        raise ValueError("width-one target has no finite non-boundary MLP profile")
    return dict(result), boundary


def _extract_candidate_profile(
    artifact: Any,
    *,
    model_id: str,
    model_revision: str,
) -> tuple[
    dict[Basis, NDArray[np.float64]],
    dict[Basis, NDArray[np.float64]],
    int,
]:
    trace = artifact.trace
    candidates = tuple(trace.candidate_selection.candidates)
    observed_indices = [
        index for index, value in enumerate(candidates) if value.is_observed
    ]
    if observed_indices != [0]:
        raise ValueError(
            "C2 candidate union must place exactly one observed token first"
        )
    ranks = [int(candidate.full_distribution_rank) for candidate in candidates]
    frame = trace.df_node
    final_layer = int(frame["layer"].max())
    contrasts: dict[Basis, NDArray[np.float64]] = {}
    raw: dict[Basis, NDArray[np.float64]] = {}
    boundary = 0
    for _, row in frame.iterrows():
        layer = int(row["layer"])
        if not 0 <= layer < final_layer:
            continue
        applicable = list(row["applicable_by_candidate"])
        if len(applicable) != len(candidates) or not all(applicable):
            raise ValueError(
                "retained C2 MLP node lacks complete candidate applicability"
            )
        attributions = np.asarray(row["candidate_attribution"], dtype=np.float64)
        contributions = np.asarray(row["candidate_contribution"], dtype=np.float64)
        activations = np.asarray(row["candidate_activation"], dtype=np.float64)
        if any(
            vector.shape != (len(candidates),)
            for vector in (attributions, contributions, activations)
        ):
            raise ValueError(
                "retained C2 MLP measurement has the wrong candidate width"
            )
        if not all(
            np.isfinite(vector).all()
            for vector in (attributions, contributions, activations)
        ):
            raise ValueError("retained C2 MLP measurement is not finite")
        polarity = _polarity(float(attributions[0]))
        if polarity is None:
            boundary += 1
            continue
        basis = (model_id, model_revision, layer, int(row["neuron"]), polarity)
        contrast = rank_aligned_contrasts(ranks, 0, contributions.tolist())
        if basis in contrasts:
            contrasts[basis] = contrasts[basis] + contrast
            raw[basis] = raw[basis] + contributions
        else:
            contrasts[basis] = contrast.copy()
            raw[basis] = contributions.copy()
    if not contrasts:
        raise ValueError("candidate target has no finite non-boundary MLP profile")
    return contrasts, raw, boundary


def _target_nondegeneracy(
    candidate: Mapping[Basis, NDArray[np.float64]],
    raw_candidate: Mapping[Basis, NDArray[np.float64]],
) -> tuple[float, float]:
    matrix = np.stack([candidate[key] for key in sorted(candidate)])
    rank = effective_rank(matrix)
    bidirectional = [
        float(np.min(raw_candidate[key])) < 0.0 < float(np.max(raw_candidate[key]))
        for key in sorted(raw_candidate)
    ]
    fraction = float(np.mean(bidirectional)) if bidirectional else 0.0
    return rank, fraction


def _candidate_score(
    left: TargetProfile,
    right: TargetProfile,
    view: str,
    *,
    min_common_bases: int,
) -> tuple[float, int] | None:
    if view == "width1":
        return directional_similarity(
            left.width1, right.width1, min_common_bases=min_common_bases
        )
    if view == "candidate":
        return directional_similarity(
            left.candidate, right.candidate, min_common_bases=min_common_bases
        )
    if view != "multiview":
        raise ValueError(f"unsupported retrieval view: {view}")
    values = [
        directional_similarity(
            left.width1, right.width1, min_common_bases=min_common_bases
        ),
        directional_similarity(
            left.candidate, right.candidate, min_common_bases=min_common_bases
        ),
    ]
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(np.mean([value[0] for value in valid])), min(
        value[1] for value in valid
    )


def next_bin_retrieval(
    profiles: Sequence[TargetProfile],
    view: str,
    *,
    min_common_bases: int = MIN_COMMON_BASES,
) -> tuple[list[RetrievalRecord], int]:
    """Run pairwise-applicable next-bin retrieval for one profile view.

    A pair without enough common support has no score, as required by the
    protocol.  Invalid distractor pairs are omitted and an anchor is scored if
    and only if its true-continuation pair has a valid score.
    """

    by_response_bin: dict[tuple[str, int], TargetProfile] = {}
    responses: dict[str, str] = {}
    for profile in profiles:
        key = (profile.response_id, profile.phase_bin)
        if key in by_response_bin:
            raise ValueError(f"duplicate response/bin profile: {key}")
        by_response_bin[key] = profile
        previous = responses.setdefault(profile.response_id, profile.family_id)
        if previous != profile.family_id:
            raise ValueError("one response appears in multiple base-question families")
    records = []
    eligible = 0
    for anchor in sorted(profiles, key=lambda item: item.case_id):
        if anchor.phase_bin == 6:
            continue
        eligible += 1
        true_key = (anchor.response_id, anchor.phase_bin + 1)
        if true_key not in by_response_bin:
            raise ValueError(f"anchor lacks its true next-bin target: {anchor.case_id}")
        candidates = []
        for response_id, family_id in sorted(responses.items()):
            if response_id != anchor.response_id and family_id == anchor.family_id:
                continue
            target = by_response_bin.get((response_id, anchor.phase_bin + 1))
            if target is None:
                raise ValueError(f"response lacks next-bin target: {response_id}")
            candidates.append(target)
        scored = []
        for target in candidates:
            value = _candidate_score(
                anchor, target, view, min_common_bases=min_common_bases
            )
            if value is None:
                continue
            scored.append((target.response_id, value[0], value[1]))
        if not any(item[0] == anchor.response_id for item in scored):
            continue
        # Stable response ID breaks exact score ties without inspecting labels.
        ordered = sorted(scored, key=lambda item: (-item[1], item[0]))
        true_rank = next(
            index + 1
            for index, item in enumerate(ordered)
            if item[0] == anchor.response_id
        )
        records.append(
            RetrievalRecord(
                anchor_id=anchor.case_id,
                family_id=anchor.family_id,
                response_id=anchor.response_id,
                reciprocal_rank=1.0 / true_rank,
                top_one=float(true_rank == 1),
                true_rank=true_rank,
                candidate_count=len(scored),
                planned_candidate_count=len(candidates),
                common_basis_support=tuple(item[2] for item in scored),
            )
        )
    return records, eligible


def _hierarchical_mean(records: Sequence[RetrievalRecord], field: str) -> float:
    by_family: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_family[record.family_id][record.response_id].append(
            float(getattr(record, field))
        )
    if not by_family:
        raise ValueError("hierarchical estimate has no scored records")
    family_values = [
        (float(np.mean([np.mean(values) for values in responses.values()])))
        for responses in by_family.values()
    ]
    return float(np.mean(family_values))


def summarize_retrieval(
    records: Sequence[RetrievalRecord],
    eligible: int,
    *,
    eligible_profiles: Sequence[TargetProfile] | None = None,
) -> dict[str, Any]:
    supports = [value for record in records for value in record.common_basis_support]
    raw_coverage = len(records) / eligible if eligible else 0.0
    weighted_coverage = raw_coverage
    if eligible_profiles is not None:
        scored_ids = {record.anchor_id for record in records}
        coverage_records = [
            RetrievalRecord(
                anchor_id=profile.case_id,
                family_id=profile.family_id,
                response_id=profile.response_id,
                reciprocal_rank=float(profile.case_id in scored_ids),
                top_one=0.0,
                true_rank=0,
                candidate_count=0,
                planned_candidate_count=0,
                common_basis_support=(),
            )
            for profile in eligible_profiles
            if profile.phase_bin < 6
        ]
        if len(coverage_records) != eligible:
            raise ValueError("eligible retrieval profile count is inconsistent")
        weighted_coverage = _hierarchical_mean(coverage_records, "reciprocal_rank")
    return {
        "eligible_anchor_count": eligible,
        "scored_anchor_count": len(records),
        "scored_anchor_coverage": weighted_coverage,
        "raw_scored_anchor_coverage": raw_coverage,
        "valid_candidate_pair_count": sum(record.candidate_count for record in records),
        "planned_candidate_pair_count_for_scored_anchors": sum(
            record.planned_candidate_count for record in records
        ),
        "valid_candidate_pair_coverage_for_scored_anchors": (
            sum(record.candidate_count for record in records)
            / sum(record.planned_candidate_count for record in records)
            if records
            else 0.0
        ),
        "mrr": _hierarchical_mean(records, "reciprocal_rank") if records else None,
        "top_one_accuracy": _hierarchical_mean(records, "top_one") if records else None,
        "common_basis_support": {
            "pair_count": len(supports),
            "minimum": min(supports) if supports else None,
            "median": float(np.median(supports)) if supports else None,
            "maximum": max(supports) if supports else None,
        },
    }


def _family_mrr(records: Sequence[RetrievalRecord]) -> dict[str, float]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.family_id][record.response_id].append(record.reciprocal_rank)
    return {
        family: float(np.mean([np.mean(values) for values in responses.values()]))
        for family, responses in grouped.items()
    }


def family_block_bootstrap(
    family_effects: Mapping[str, float],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    families = sorted(family_effects)
    values = np.asarray(
        [family_effects[family] for family in families], dtype=np.float64
    )
    if not families or not np.isfinite(values).all():
        raise ValueError("family bootstrap requires finite family effects")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.integers(0, len(values), size=len(values))
        estimates[index] = float(np.mean(values[draw]))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "method": "percentile_family_block_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def family_block_bootstrap_views(
    width_family_mrr: Mapping[str, float],
    multiview_family_mrr: Mapping[str, float],
    all_families: Sequence[str],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap the literal difference of view-specific weighted MRRs."""

    if replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    families = sorted(set(all_families))
    if len(families) != len(all_families):
        raise ValueError("family bootstrap universe must be unique")
    if not families or not width_family_mrr or not multiview_family_mrr:
        raise ValueError("family bootstrap requires two non-empty view estimates")
    if not set(width_family_mrr).issubset(families) or not set(
        multiview_family_mrr
    ).issubset(families):
        raise ValueError("view estimate contains a family outside the frozen universe")
    if not all(
        math.isfinite(float(value))
        for value in (*width_family_mrr.values(), *multiview_family_mrr.values())
    ):
        raise ValueError("family bootstrap requires finite view estimates")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        for _attempt in range(10_000):
            draw = [
                families[item] for item in rng.integers(0, len(families), len(families))
            ]
            width = [
                width_family_mrr[family]
                for family in draw
                if family in width_family_mrr
            ]
            multiview = [
                multiview_family_mrr[family]
                for family in draw
                if family in multiview_family_mrr
            ]
            if width and multiview:
                break
        else:
            raise ValueError("bootstrap could not draw both scored views")
        estimates[index] = float(np.mean(multiview) - np.mean(width))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "method": "percentile_family_block_bootstrap_of_view_specific_mrrs",
        "family_universe_count": len(families),
        "width1_scored_family_count": len(width_family_mrr),
        "multiview_scored_family_count": len(multiview_family_mrr),
        "replicates": replicates,
        "seed": seed,
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def leave_one_family_out(family_effects: Mapping[str, float]) -> dict[str, Any]:
    families = sorted(family_effects)
    if len(families) < 2:
        raise ValueError("LOFO requires at least two families")
    estimates = {}
    for omitted in families:
        retained = [
            value for family, value in family_effects.items() if family != omitted
        ]
        estimates[omitted] = float(np.mean(retained))
    positive = sum(value > 0.0 for value in estimates.values())
    return {
        "family_count": len(families),
        "positive_count": positive,
        "positive_fraction": positive / len(families),
        "estimates": estimates,
    }


def leave_one_family_out_views(
    width_family_mrr: Mapping[str, float],
    multiview_family_mrr: Mapping[str, float],
    all_families: Sequence[str],
) -> dict[str, Any]:
    families = sorted(set(all_families))
    if len(families) < 2 or len(families) != len(all_families):
        raise ValueError("view-specific LOFO requires a unique family universe")
    estimates = {}
    for omitted in families:
        width = [
            value for family, value in width_family_mrr.items() if family != omitted
        ]
        multiview = [
            value for family, value in multiview_family_mrr.items() if family != omitted
        ]
        if not width or not multiview:
            raise ValueError("LOFO leaves one view without a scored family")
        estimates[omitted] = float(np.mean(multiview) - np.mean(width))
    positive = sum(value > 0.0 for value in estimates.values())
    return {
        "family_count": len(families),
        "width1_scored_family_count": len(width_family_mrr),
        "multiview_scored_family_count": len(multiview_family_mrr),
        "positive_count": positive,
        "positive_fraction": positive / len(families),
        "estimates": estimates,
    }


def _validate_selection_and_plan(
    selection: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported frozen C2 selection schema")
    validate_candidate_union_plan(plan)
    cases = selection.get("cases")
    if not isinstance(cases, list) or len(cases) != 245:
        raise ValueError("frozen C2 selection must contain exactly 245 cases")
    plan_cases = {
        case["source_width1_artifact_id"]: case
        for wave in plan["waves"]
        for case in wave["cases"]
    }
    if len(plan_cases) != 245:
        raise ValueError("candidate-union plan must contain 245 unique source cases")
    selected_by_source = {case["source_width1_artifact_id"]: case for case in cases}
    if len(selected_by_source) != 245 or set(selected_by_source) != set(plan_cases):
        raise ValueError("C2 selection and candidate-union plan coverage disagree")
    response_bins: dict[str, set[int]] = defaultdict(set)
    response_families = {}
    for case in cases:
        source_id = case["source_width1_artifact_id"]
        planned = plan_cases[source_id]
        expected = {
            "case_id": planned["case_id"],
            "candidate_count": len(planned["reference_artifacts"]),
            "candidate_token_ids": [
                item["token_id"] for item in planned["reference_artifacts"]
            ],
            "example_id": planned["source_item"]["example"]["example_id"],
            "base_question_id": planned["source_item"]["example"]["base_question_id"],
            "target_response_position": planned["source_item"]["target_selection"][
                "response_token_positions"
            ][0],
        }
        for field, value in expected.items():
            if case.get(field) != value:
                raise ValueError(f"C2 selection/plan drift for {source_id}: {field}")
        phase_bin = case.get("phase_bin")
        if (
            isinstance(phase_bin, bool)
            or not isinstance(phase_bin, int)
            or not 0 <= phase_bin <= 6
        ):
            raise ValueError("C2 phase bin must be an integer from zero through six")
        response = case["example_id"]
        family = case["base_question_id"]
        response_bins[response].add(phase_bin)
        if response_families.setdefault(response, family) != family:
            raise ValueError("one C2 response appears in multiple families")
    if len(response_bins) != 35 or len(set(response_families.values())) != 34:
        raise ValueError("C2 selection must contain 35 responses from 34 families")
    if any(bins != set(range(7)) for bins in response_bins.values()):
        raise ValueError("every C2 response must cover all seven phase bins")
    return [dict(case) for case in cases], plan_cases


def _index_width1(root: Path, source_ids: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("manifest.json"):
        try:
            manifest = _load_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        source_id = manifest.get("source_artifact_id")
        if source_id not in source_ids:
            continue
        if source_id in result:
            raise ValueError(f"duplicate width-one artifact for {source_id}")
        result[source_id] = path.parent
    missing = source_ids - set(result)
    if missing:
        raise ValueError(f"missing {len(missing)} planned width-one artifacts")
    return result


def _index_candidate_union(
    root: Path, source_ids: set[str], plan_sha256: str
) -> dict[str, Path]:
    family_root = root / "bonafide.candidate-union.v1"
    if not family_root.is_dir():
        raise ValueError(f"candidate-union artifact family is missing: {family_root}")
    result = {}
    for path in family_root.rglob("manifest.json"):
        manifest = _load_json(path)
        if manifest.get("candidate_union_plan_sha256") != plan_sha256:
            raise ValueError(f"candidate-union plan provenance drift: {path.parent}")
        contract = manifest.get("candidate_union_contract")
        if not isinstance(contract, Mapping):
            raise ValueError("candidate-union manifest lacks its trace contract")
        source_id = contract.get("source_width1_artifact_id")
        if source_id not in source_ids:
            raise ValueError(
                f"candidate-union root contains an unplanned source: {source_id}"
            )
        if source_id in result:
            raise ValueError(f"duplicate candidate-union artifact for {source_id}")
        result[source_id] = path.parent
    missing = source_ids - set(result)
    if missing:
        raise ValueError(f"missing {len(missing)} planned candidate-union artifacts")
    return result


def _load_profiles(
    cases: Sequence[Mapping[str, Any]],
    plan_cases: Mapping[str, Mapping[str, Any]],
    width_paths: Mapping[str, Path],
    union_paths: Mapping[str, Path],
    *,
    model_id: str,
    model_revision: str,
    chat_template_sha256: str,
    plan_sha256: str,
) -> list[TargetProfile]:
    profiles = []
    for case in sorted(cases, key=lambda value: value["case_id"]):
        source_id = case["source_width1_artifact_id"]
        planned = plan_cases[source_id]
        width = load_compact_trace(width_paths[source_id])
        union = load_candidate_union_artifact(union_paths[source_id])
        expected_source_item_sha256 = _canonical_sha256(planned["source_item"])
        for artifact_name, manifest in (
            ("width-one", width.manifest),
            ("candidate-union", union.manifest),
        ):
            if (
                manifest.get("numerically_valid") is not True
                or manifest.get("scientifically_reusable") is not True
            ):
                raise ValueError(
                    f"{artifact_name} is not marked reusable numerical evidence"
                )
            if (
                manifest.get("model_id") != model_id
                or manifest.get("model_revision") != model_revision
            ):
                raise ValueError(
                    f"{artifact_name} model provenance drift for {source_id}"
                )
            if (
                manifest.get("bonafide_example", {}).get("example_id")
                != case["example_id"]
            ):
                raise ValueError(
                    f"{artifact_name} response provenance drift for {source_id}"
                )
            positions = manifest.get("source_target_selection", {}).get(
                "response_token_positions"
            )
            if positions != [case["target_response_position"]]:
                raise ValueError(
                    f"{artifact_name} target-position drift for {source_id}"
                )
            if manifest.get("bonafide_example") != planned["source_item"]["example"]:
                raise ValueError(
                    f"{artifact_name} frozen example drift for {source_id}"
                )
            if (
                manifest.get("source_target_selection")
                != planned["source_item"]["target_selection"]
            ):
                raise ValueError(
                    f"{artifact_name} frozen target selection drift for {source_id}"
                )
        width_identity = _validate_identity(width.manifest, "width-one")
        union_identity = _validate_identity(union.manifest, "candidate-union")
        if width.manifest.get("source_artifact_id") != source_id:
            raise ValueError("width-one source artifact ID drift")
        if (
            width_identity.get("source_artifact_id") != source_id
            or width_identity.get("source_work_item_sha256")
            != expected_source_item_sha256
        ):
            raise ValueError("width-one frozen source-work-item identity drift")
        target_token_id = planned["source_item"]["target_selection"][
            "final_target_token_id"
        ]
        if width.circuit_data.target_logits != [[target_token_id]]:
            raise ValueError("width-one observed target token drift")
        if (
            width.manifest.get("trace_metadata", {}).get("chat_template_sha256")
            != chat_template_sha256
        ):
            raise ValueError("width-one chat-template provenance drift")
        if union.manifest.get("candidate_union_plan_sha256") != plan_sha256:
            raise ValueError("candidate-union plan hash drift")
        if (
            union_identity.get("candidate_union_plan_sha256") != plan_sha256
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
        trace = union.trace
        if trace.source_width1_artifact_id != source_id:
            raise ValueError("candidate-union source artifact ID drift")
        if trace.topology_sha256 != planned.get("frozen_union_topology_sha256"):
            raise ValueError("candidate-union frozen topology hash drift")
        token_ids = [
            candidate.token_id for candidate in trace.candidate_selection.candidates
        ]
        if token_ids != case["candidate_token_ids"]:
            raise ValueError("candidate-union token selection drift")
        trace_references = [
            {
                "artifact_id": reference["artifact_id"],
                "payload_sha256": reference["payload_sha256"],
            }
            for reference in trace.reference_artifacts
        ]
        planned_references = [
            {
                "artifact_id": reference["artifact_id"],
                "payload_sha256": reference["payload_sha256"],
            }
            for reference in planned["reference_artifacts"]
        ]
        if trace_references != planned_references:
            raise ValueError("candidate-union payload reference provenance drift")
        width_profile, width_boundary = _extract_width1_profile(
            width, model_id=model_id, model_revision=model_revision
        )
        candidate_profile, raw_profile, candidate_boundary = _extract_candidate_profile(
            union, model_id=model_id, model_revision=model_revision
        )
        rank, bidirectional = _target_nondegeneracy(candidate_profile, raw_profile)
        profiles.append(
            TargetProfile(
                case_id=case["case_id"],
                source_artifact_id=source_id,
                family_id=case["base_question_id"],
                response_id=case["example_id"],
                phase_bin=case["phase_bin"],
                width1=width_profile,
                candidate=candidate_profile,
                raw_candidate=raw_profile,
                effective_rank=rank,
                bidirectional_raw_fraction=bidirectional,
                boundary_width1_occurrences=width_boundary,
                boundary_candidate_occurrences=candidate_boundary,
                width1_artifact_id=width.manifest["artifact_id"],
                width1_payload_sha256=width.manifest["data_sha256"],
                candidate_union_artifact_id=union.manifest["artifact_id"],
                candidate_union_payload_sha256=union.manifest["data_sha256"],
                topology_sha256=trace.topology_sha256,
            )
        )
    return profiles


def analyze_c2(
    *,
    selection_path: Path,
    plan_path: Path,
    width1_root: Path,
    candidate_union_root: Path,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    min_common_bases: int = MIN_COMMON_BASES,
) -> dict[str, Any]:
    selection = _load_json(selection_path)
    plan = _load_json(plan_path)
    selection_file_sha256 = _sha256_file(selection_path)
    plan_file_sha256 = _sha256_file(plan_path)
    if plan.get("cohort", {}).get("selection_sha256") != selection_file_sha256:
        raise ValueError(
            "candidate-union plan does not bind the supplied selection file"
        )
    selection_source_manifest = _validate_bound_json_file(
        selection, "source_manifest_path", "source_manifest_sha256"
    )
    plan_source_manifest = _validate_bound_json_file(
        plan["source"], "width1_manifest_path", "width1_manifest_sha256"
    )
    if selection_source_manifest != plan_source_manifest:
        raise ValueError("selection and plan bind different width-one manifests")
    config = _validate_bound_json_file(
        plan["execution"], "config_path", "config_sha256"
    )
    if _canonical_sha256(config) != plan["execution"].get("config_canonical_sha256"):
        raise ValueError("candidate-union execution config canonical hash drift")
    cases, plan_cases = _validate_selection_and_plan(selection, plan)
    source_ids = {case["source_width1_artifact_id"] for case in cases}
    plan_sha256 = _canonical_sha256(plan)
    width_paths = _index_width1(width1_root, source_ids)
    union_paths = _index_candidate_union(candidate_union_root, source_ids, plan_sha256)
    model_id = plan["source"]["model_id"]
    model_revision = plan["source"]["model_revision"]
    profiles = _load_profiles(
        cases,
        plan_cases,
        width_paths,
        union_paths,
        model_id=model_id,
        model_revision=model_revision,
        chat_template_sha256=plan["source"]["chat_template_sha256"],
        plan_sha256=plan_sha256,
    )

    ranks = np.asarray([profile.effective_rank for profile in profiles])
    bidirectional = np.asarray(
        [profile.bidirectional_raw_fraction for profile in profiles]
    )
    nondegeneracy = {
        "target_count": len(profiles),
        "complete_candidate_applicability_and_finite": True,
        "median_effective_rank": float(np.median(ranks)),
        "effective_rank_at_least_1_5_count": int(
            np.sum(ranks >= NONDEGENERACY_TARGET_EFFECTIVE_RANK)
        ),
        "effective_rank_at_least_1_5_fraction": float(
            np.mean(ranks >= NONDEGENERACY_TARGET_EFFECTIVE_RANK)
        ),
        "median_bidirectional_raw_contribution_fraction": float(
            np.median(bidirectional)
        ),
    }
    nondegeneracy["gate"] = {
        "median_effective_rank_at_least_2": nondegeneracy["median_effective_rank"]
        >= NONDEGENERACY_MEDIAN_EFFECTIVE_RANK,
        "at_least_90_percent_targets_effective_rank_1_5": nondegeneracy[
            "effective_rank_at_least_1_5_fraction"
        ]
        >= NONDEGENERACY_TARGET_FRACTION,
        "positive_median_bidirectional_fraction": nondegeneracy[
            "median_bidirectional_raw_contribution_fraction"
        ]
        > 0.0,
    }
    nondegeneracy["gate"]["passes"] = all(nondegeneracy["gate"].values())

    retrieval_records = {}
    retrieval = {}
    for view in ("width1", "candidate", "multiview"):
        records, eligible = next_bin_retrieval(
            profiles, view, min_common_bases=min_common_bases
        )
        retrieval_records[view] = records
        retrieval[view] = summarize_retrieval(
            records, eligible, eligible_profiles=profiles
        )

    width_family_mrr = _family_mrr(retrieval_records["width1"])
    multiview_family_mrr = _family_mrr(retrieval_records["multiview"])
    all_families = sorted({profile.family_id for profile in profiles})
    common_scored_anchor_ids = sorted(
        {record.anchor_id for record in retrieval_records["width1"]}.intersection(
            record.anchor_id for record in retrieval_records["multiview"]
        )
    )
    improvement = float(retrieval["multiview"]["mrr"] - retrieval["width1"]["mrr"])
    direct_family_improvement = float(
        np.mean(list(multiview_family_mrr.values()))
        - np.mean(list(width_family_mrr.values()))
    )
    if not math.isclose(improvement, direct_family_improvement, abs_tol=1e-12):
        raise ValueError(
            "family summaries disagree with the reported weighted MRR contrast"
        )
    bootstrap = family_block_bootstrap_views(
        width_family_mrr,
        multiview_family_mrr,
        all_families,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    lofo = leave_one_family_out_views(
        width_family_mrr, multiview_family_mrr, all_families
    )
    utility_gate = {
        "mrr_improvement_at_least_0_03": improvement >= UTILITY_MRR_THRESHOLD,
        "bootstrap_lower_bound_above_zero": bootstrap["lower_95"] > 0.0,
        "lofo_positive_fraction_at_least_0_80": lofo["positive_fraction"]
        >= LOFO_POSITIVE_FRACTION,
    }
    utility_gate["passes"] = all(utility_gate.values())

    target_diagnostics = [
        {
            "case_id": profile.case_id,
            "source_width1_artifact_id": profile.source_artifact_id,
            "base_question_id": profile.family_id,
            "example_id": profile.response_id,
            "phase_bin": profile.phase_bin,
            "width1_signed_basis_count": len(profile.width1),
            "candidate_signed_basis_count": len(profile.candidate),
            "effective_rank": profile.effective_rank,
            "bidirectional_raw_contribution_fraction": profile.bidirectional_raw_fraction,
            "width1_boundary_occurrence_count": profile.boundary_width1_occurrences,
            "candidate_boundary_occurrence_count": profile.boundary_candidate_occurrences,
            "width1_artifact_id": profile.width1_artifact_id,
            "width1_payload_sha256": profile.width1_payload_sha256,
            "candidate_union_artifact_id": profile.candidate_union_artifact_id,
            "candidate_union_payload_sha256": profile.candidate_union_payload_sha256,
            "candidate_union_topology_sha256": profile.topology_sha256,
        }
        for profile in profiles
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "inputs": {
            "selection_path": str(selection_path.resolve()),
            "selection_file_sha256": selection_file_sha256,
            "plan_path": str(plan_path.resolve()),
            "plan_file_sha256": plan_file_sha256,
            "plan_canonical_sha256": plan_sha256,
            "width1_root": str(width1_root.resolve()),
            "candidate_union_root": str(candidate_union_root.resolve()),
            "model_id": model_id,
            "model_revision": model_revision,
            "analysis_module_path": str(Path(__file__).resolve()),
            "analysis_module_sha256": _sha256_file(Path(__file__).resolve()),
            "analysis_code_revision": collect_code_revision(
                Path(__file__).resolve().parents[2]
            ),
            "analysis_runtime_environment": collect_runtime_environment(),
            "artifact_payload_set_sha256": _canonical_sha256(
                [
                    {
                        "source_width1_artifact_id": profile.source_artifact_id,
                        "width1_payload_sha256": profile.width1_payload_sha256,
                        "candidate_union_payload_sha256": profile.candidate_union_payload_sha256,
                    }
                    for profile in profiles
                ]
            ),
        },
        "coverage": {
            "case_count": len(profiles),
            "response_count": len({profile.response_id for profile in profiles}),
            "family_count": len({profile.family_id for profile in profiles}),
            "phase_bin_counts": {
                str(phase): sum(profile.phase_bin == phase for profile in profiles)
                for phase in range(7)
            },
        },
        "method": {
            "minimum_common_non_boundary_mlp_bases": min_common_bases,
            "effective_rank_definition": "exp_entropy_of_normalized_singular_values",
            "retrieval_missing_score_policy": "omit invalid distractor pairs; score anchor if and only if its true-continuation pair is valid",
            "scored_anchor_coverage_weighting": "equal_family_then_equal_response_then_equal_target; raw count coverage also reported",
            "multiview_policy": "mean_of_valid_width1_and_candidate_pair_similarities",
            "tie_break": "descending_similarity_then_ascending_response_id",
            "primary_utility_contrast": "reported_multiview_mrr_minus_reported_width1_mrr_using_each_views_scored_coverage",
            "artifact_numeric_validation": "candidate-union loader validates finite/null applicability semantics for every node and edge vector; primary profiles retain non-boundary MLP nodes only",
            "bootstrap": {
                "unit": "base_question_family",
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "interval": "percentile_2_5_to_97_5",
                "view_missingness": "sample the frozen 34-family universe; compute each view mean over scored families in the draw and deterministically redraw only if a view is empty",
            },
            "frozen_protocol_gaps_resolved_by_implementation": [
                "effective-rank formula was unspecified; entropy effective rank over singular values is used",
                "bootstrap seed and replicate count were unspecified; explicit deterministic CLI values are recorded",
                "invalid-pair ranking policy was unspecified; invalid distractors are omitted and true-pair validity determines anchor coverage",
                "exact-score ties were unspecified; ascending response ID is used",
                "view-specific invalid anchors remain absent from that view; utility subtracts the separately weighted view MRRs",
            ],
        },
        "nondegeneracy": nondegeneracy,
        "retrieval": retrieval,
        "utility": {
            "common_scored_anchor_count": len(common_scored_anchor_ids),
            "family_universe_count": len(all_families),
            "width1_scored_family_count": len(width_family_mrr),
            "multiview_scored_family_count": len(multiview_family_mrr),
            "multiview_mrr_minus_width1_mrr": improvement,
            "width1_family_mrr": width_family_mrr,
            "multiview_family_mrr": multiview_family_mrr,
            "bootstrap": bootstrap,
            "leave_one_family_out": lofo,
            "gate": utility_gate,
        },
        "overall_gate_passes": bool(
            nondegeneracy["gate"]["passes"] and utility_gate["passes"]
        ),
        "target_diagnostics": target_diagnostics,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--width1-root", type=Path, required=True)
    parser.add_argument("--candidate-union-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--min-common-bases", type=int, default=MIN_COMMON_BASES)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(
            f"refusing to overwrite analysis output: {args.output_json}"
        )
    if args.min_common_bases < 1:
        raise ValueError("minimum common bases must be positive")
    report = analyze_c2(
        selection_path=args.selection,
        plan_path=args.plan,
        width1_root=args.width1_root,
        candidate_union_root=args.candidate_union_root,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        min_common_bases=args.min_common_bases,
    )
    _write_json_atomic(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "overall_gate_passes": report["overall_gate_passes"],
                "utility_improvement": report["utility"][
                    "multiview_mrr_minus_width1_mrr"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
