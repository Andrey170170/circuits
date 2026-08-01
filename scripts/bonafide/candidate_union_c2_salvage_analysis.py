"""Evaluate the frozen post-hoc C2 candidate-union salvage protocol.

This module is deliberately separate from ``candidate_union_c2_analysis``.
The latter is checksum-bound evidence; its validation and profile-construction
helpers are reused without changing it.  This program reads immutable trace
artifacts and writes one new, no-overwrite JSON report.

Run from the repository root, for example::

    python -m scripts.bonafide.candidate_union_c2_salvage_analysis --help
"""

from __future__ import annotations

import argparse
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

from scripts.bonafide import candidate_union_c2_analysis as c2
from scripts.bonafide.runner import collect_code_revision, collect_runtime_environment


REPORT_SCHEMA_VERSION = "bonafide-candidate-union-c2-salvage-analysis/v2"
EXPECTED_PROTOCOL_SHA256 = (
    "f8eb538f956ac8f3ce69b8bbe98bf46e96ab7435a47bf7b3644aaef25a27bd49"
)
EXPECTED_AUDITED_C2_REPORT_SHA256 = (
    "9ea1123685e73bc45f8c93490429a2a309ed62953406d61509d0014730ef6530"
)
EXPECTED_PAIR_COUNT = 7_338
EXPECTED_ANCHOR_COUNT = 210
PERMUTATION_REPLICATES = 100_000
BOOTSTRAP_REPLICATES = 10_000
MIN_COMMON_BASES = 16
PRIMARY_ENDPOINTS = ("candidate_all", "candidate_rescue", "backoff_delta")


@dataclass(frozen=True)
class PairScore:
    score: float | None
    support: int
    invalid_reason: str | None

    @property
    def valid(self) -> bool:
        return self.score is not None


@dataclass(frozen=True)
class PairRow:
    anchor_id: str
    anchor_family_id: str
    anchor_response_id: str
    phase_bin: int
    target_id: str
    target_family_id: str
    target_response_id: str
    true_continuation: bool
    scores: Mapping[str, PairScore]


@dataclass(frozen=True)
class Anchor:
    case_id: str
    family_id: str
    response_id: str
    phase_bin: int


@dataclass(frozen=True)
class ScoreTables:
    anchors: tuple[Anchor, ...]
    responses: tuple[str, ...]
    response_families: tuple[str, ...]
    planned: np.ndarray
    scores: Mapping[str, np.ndarray]


def _profile_score(
    left: Mapping[c2.Basis, float | np.ndarray],
    right: Mapping[c2.Basis, float | np.ndarray],
    *,
    channel: int | None = None,
    min_common_bases: int = MIN_COMMON_BASES,
) -> PairScore:
    """Cosine plus an explicit invalidity reason on common signed bases."""

    common = sorted(set(left).intersection(right))
    support = len(common)
    if support < min_common_bases:
        return PairScore(None, support, "fewer_than_minimum_common_bases")

    def vector(profile: Mapping[c2.Basis, float | np.ndarray]) -> np.ndarray:
        values = []
        for key in common:
            value = np.asarray(profile[key], dtype=np.float64).reshape(-1)
            if channel is not None:
                if value.shape != (5,):
                    raise ValueError("rank-channel profile does not have width five")
                value = value[channel : channel + 1]
            values.append(value)
        return np.concatenate(values)

    left_vector = vector(left)
    right_vector = vector(right)
    if left_vector.shape != right_vector.shape:
        raise ValueError("profile channel widths disagree")
    if not np.isfinite(left_vector).all() or not np.isfinite(right_vector).all():
        raise ValueError("profile contains non-finite values")
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return PairScore(None, support, "zero_norm")
    score = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
    if not math.isfinite(score):
        raise ValueError("profile cosine is non-finite")
    return PairScore(score, support, None)


def construct_pair_rows(
    profiles: Sequence[c2.TargetProfile],
    *,
    min_common_bases: int = MIN_COMMON_BASES,
) -> list[PairRow]:
    """Construct all planned anchor/next-bin-target rows exactly once."""

    by_response_bin: dict[tuple[str, int], c2.TargetProfile] = {}
    response_families: dict[str, str] = {}
    for profile in profiles:
        key = (profile.response_id, profile.phase_bin)
        if key in by_response_bin:
            raise ValueError(f"duplicate response/bin profile: {key}")
        by_response_bin[key] = profile
        previous = response_families.setdefault(profile.response_id, profile.family_id)
        if previous != profile.family_id:
            raise ValueError("one response occurs in multiple families")

    rows = []
    for anchor in sorted(profiles, key=lambda item: item.case_id):
        if anchor.phase_bin == 6:
            continue
        for response_id, family_id in sorted(response_families.items()):
            if response_id != anchor.response_id and family_id == anchor.family_id:
                continue
            target = by_response_bin.get((response_id, anchor.phase_bin + 1))
            if target is None:
                raise ValueError(f"response lacks next-bin target: {response_id}")
            scores: dict[str, PairScore] = {
                "width1": _profile_score(
                    anchor.width1,
                    target.width1,
                    min_common_bases=min_common_bases,
                ),
                "candidate": _profile_score(
                    anchor.candidate,
                    target.candidate,
                    min_common_bases=min_common_bases,
                ),
            }
            for channel in range(5):
                scores[f"rank{channel + 1}"] = _profile_score(
                    anchor.candidate,
                    target.candidate,
                    channel=channel,
                    min_common_bases=min_common_bases,
                )
            rows.append(
                PairRow(
                    anchor_id=anchor.case_id,
                    anchor_family_id=anchor.family_id,
                    anchor_response_id=anchor.response_id,
                    phase_bin=anchor.phase_bin,
                    target_id=target.case_id,
                    target_family_id=target.family_id,
                    target_response_id=target.response_id,
                    true_continuation=target.response_id == anchor.response_id,
                    scores=scores,
                )
            )
    return rows


def pair_rows_to_tables(rows: Sequence[PairRow]) -> ScoreTables:
    if not rows:
        raise ValueError("pair table is empty")
    anchor_values = {
        row.anchor_id: Anchor(
            row.anchor_id,
            row.anchor_family_id,
            row.anchor_response_id,
            row.phase_bin,
        )
        for row in rows
    }
    anchors = tuple(sorted(anchor_values.values(), key=lambda item: item.case_id))
    responses = tuple(sorted({row.target_response_id for row in rows}))
    response_family_map = {}
    for row in rows:
        previous = response_family_map.setdefault(
            row.target_response_id, row.target_family_id
        )
        if previous != row.target_family_id:
            raise ValueError("target response occurs in multiple families")
    response_families = tuple(response_family_map[value] for value in responses)
    anchor_index = {value.case_id: index for index, value in enumerate(anchors)}
    response_index = {value: index for index, value in enumerate(responses)}
    shape = (len(anchors), len(responses))
    planned = np.zeros(shape, dtype=bool)
    names = tuple(sorted(rows[0].scores))
    scores = {name: np.full(shape, np.nan, dtype=np.float64) for name in names}
    seen = set()
    for row in rows:
        if tuple(sorted(row.scores)) != names:
            raise ValueError("pair rows disagree on score views")
        index = (anchor_index[row.anchor_id], response_index[row.target_response_id])
        if index in seen:
            raise ValueError("duplicate anchor/target pair")
        seen.add(index)
        planned[index] = True
        for name, value in row.scores.items():
            if value.valid:
                scores[name][index] = value.score
    return ScoreTables(anchors, responses, response_families, planned, scores)


def midrank_reciprocal_and_top_one(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank every valid cell descending; missing truths receive structural zero."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("score table must be two-dimensional")
    reciprocal = np.zeros(values.shape, dtype=np.float64)
    top_one = np.zeros(values.shape, dtype=np.float64)
    for row_index, row in enumerate(values):
        valid_indices = np.flatnonzero(np.isfinite(row))
        valid = row[valid_indices]
        for local_index, column_index in enumerate(valid_indices):
            score = valid[local_index]
            greater = int(np.sum(valid > score))
            tied = int(np.sum(valid == score))
            midrank = greater + (tied + 1.0) / 2.0
            reciprocal[row_index, column_index] = 1.0 / midrank
            top_one[row_index, column_index] = 1.0 / tied if greater == 0 else 0.0
    return reciprocal, top_one


def mid_ecdf_percentiles(scores: np.ndarray) -> np.ndarray:
    """Convert valid scores to within-row mid empirical-CDF percentiles."""

    values = np.asarray(scores, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for row_index, row in enumerate(values):
        valid_indices = np.flatnonzero(np.isfinite(row))
        valid = row[valid_indices]
        count = len(valid)
        for local_index, column_index in enumerate(valid_indices):
            score = valid[local_index]
            result[row_index, column_index] = (
                np.sum(valid < score) + 0.5 * np.sum(valid == score)
            ) / count
    return result


def width_preferred_backoff(
    width_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    percentile_calibrated: bool = True,
) -> np.ndarray:
    width = np.asarray(width_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    if width.shape != candidate.shape:
        raise ValueError("backoff views have different shapes")
    if percentile_calibrated:
        width = mid_ecdf_percentiles(width)
        candidate = mid_ecdf_percentiles(candidate)
    return np.where(np.isfinite(width), width, candidate)


def _group_indices(anchors: Sequence[Anchor]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, anchor in enumerate(anchors):
        grouped[anchor.family_id][anchor.response_id].append(index)
    return {
        family: {
            response: np.asarray(indices, dtype=np.int64)
            for response, indices in sorted(responses.items())
        }
        for family, responses in sorted(grouped.items())
    }


def hierarchical_mean(
    values: np.ndarray,
    anchors: Sequence[Anchor],
    subset: np.ndarray | None = None,
) -> float:
    """Equal-family, then response, then retained-anchor mean."""

    value = np.asarray(values, dtype=np.float64)
    if value.shape != (len(anchors),):
        raise ValueError("hierarchical values have the wrong length")
    mask = np.ones(len(anchors), dtype=bool) if subset is None else np.asarray(subset)
    if mask.shape != value.shape or mask.dtype != bool:
        raise ValueError("hierarchical subset must be a matching Boolean vector")
    family_values = []
    for responses in _group_indices(anchors).values():
        response_values = []
        for indices in responses.values():
            kept = indices[mask[indices]]
            if len(kept):
                response_values.append(float(np.mean(value[kept])))
        if response_values:
            family_values.append(float(np.mean(response_values)))
    if not family_values:
        raise ValueError("hierarchical subset is empty")
    return float(np.mean(family_values))


def family_means(
    values: np.ndarray,
    anchors: Sequence[Anchor],
    subset: np.ndarray | None = None,
) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    mask = np.ones(len(anchors), dtype=bool) if subset is None else np.asarray(subset)
    result = {}
    for family, responses in _group_indices(anchors).items():
        response_values = []
        for indices in responses.values():
            kept = indices[mask[indices]]
            if len(kept):
                response_values.append(float(np.mean(value[kept])))
        if response_values:
            result[family] = float(np.mean(response_values))
    return result


def analytic_chance(
    scores: np.ndarray,
    planned: np.ndarray,
    anchors: Sequence[Anchor],
    subset: np.ndarray | None = None,
) -> dict[str, float]:
    """Uniform-truth chance using the actual midranks, including exact ties."""

    reciprocal, top_one = midrank_reciprocal_and_top_one(scores)
    valid = np.isfinite(scores)
    valid_counts = np.sum(valid, axis=1)
    planned_counts = np.sum(planned, axis=1)
    conditional_mrr = np.divide(
        np.sum(reciprocal, axis=1),
        valid_counts,
        out=np.zeros(len(anchors), dtype=np.float64),
        where=valid_counts > 0,
    )
    conditional_top = np.divide(
        np.sum(top_one, axis=1),
        valid_counts,
        out=np.zeros(len(anchors), dtype=np.float64),
        where=valid_counts > 0,
    )
    zero_mrr = np.divide(
        np.sum(reciprocal, axis=1),
        planned_counts,
        out=np.zeros(len(anchors), dtype=np.float64),
        where=planned_counts > 0,
    )
    zero_top = np.divide(
        np.sum(top_one, axis=1),
        planned_counts,
        out=np.zeros(len(anchors), dtype=np.float64),
        where=planned_counts > 0,
    )
    return {
        "conditional_scored_mrr": hierarchical_mean(conditional_mrr, anchors, subset),
        "conditional_scored_top_one": hierarchical_mean(
            conditional_top, anchors, subset
        ),
        "planned_pool_zero_filled_mrr": hierarchical_mean(zero_mrr, anchors, subset),
        "planned_pool_zero_filled_top_one": hierarchical_mean(
            zero_top, anchors, subset
        ),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("Holm adjustment requires p-values")
    ordered = sorted((float(value), name) for name, value in p_values.items())
    if any(not 0.0 <= value <= 1.0 for value, _ in ordered):
        raise ValueError("p-values must lie in [0, 1]")
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for index, (value, name) in enumerate(ordered):
        running = max(running, (total - index) * value)
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in p_values}


def constrained_response_permutations(
    responses: Sequence[str],
    families: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Uniform permutations conditional on avoiding same-family alternatives."""

    if len(responses) != len(families) or len(set(responses)) != len(responses):
        raise ValueError("response/family universe is inconsistent")
    if replicates < 1:
        raise ValueError("permutation replicate count must be positive")
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, len(responses)), dtype=np.int16)
    family_values = np.asarray(families, dtype=object)
    identity = np.arange(len(responses))
    accepted = 0
    rejected = 0
    while accepted < replicates:
        proposal = rng.permutation(len(responses))
        forbidden = (proposal != identity) & (family_values[proposal] == family_values)
        if np.any(forbidden):
            rejected += 1
            continue
        result[accepted] = proposal
        accepted += 1
    return result, rejected


def _truth_columns(tables: ScoreTables, mapping: np.ndarray) -> np.ndarray:
    response_index = {value: index for index, value in enumerate(tables.responses)}
    source = np.asarray(
        [response_index[anchor.response_id] for anchor in tables.anchors],
        dtype=np.int64,
    )
    return mapping[..., source]


def _gather(table: np.ndarray, columns: np.ndarray) -> np.ndarray:
    anchor_index = np.arange(table.shape[0])
    if columns.ndim == 1:
        return table[anchor_index, columns]
    return table[anchor_index[None, :], columns]


def _batch_hierarchical(
    values: np.ndarray,
    anchors: Sequence[Anchor],
    subset: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Vectorized hierarchy; empty subset families/replicates remain NaN."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(anchors):
        raise ValueError("batch hierarchy has the wrong shape")
    masks = np.ones(data.shape, dtype=bool) if subset is None else np.asarray(subset)
    family_outputs = {}
    for family, responses in _group_indices(anchors).items():
        response_outputs = []
        for indices in responses.values():
            count = np.sum(masks[:, indices], axis=1)
            total = np.sum(np.where(masks[:, indices], data[:, indices], 0.0), axis=1)
            response_outputs.append(
                np.divide(
                    total,
                    count,
                    out=np.full(data.shape[0], np.nan),
                    where=count > 0,
                )
            )
        response_matrix = np.stack(response_outputs, axis=1)
        count = np.sum(np.isfinite(response_matrix), axis=1)
        family_outputs[family] = np.divide(
            np.nansum(response_matrix, axis=1),
            count,
            out=np.full(data.shape[0], np.nan),
            where=count > 0,
        )
    family_matrix = np.stack(list(family_outputs.values()), axis=1)
    count = np.sum(np.isfinite(family_matrix), axis=1)
    overall = np.divide(
        np.nansum(family_matrix, axis=1),
        count,
        out=np.full(data.shape[0], np.nan),
        where=count > 0,
    )
    return overall, family_outputs


def permutation_inference(
    tables: ScoreTables,
    permutations: np.ndarray,
    *,
    chunk_size: int = 5_000,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Evaluate the three frozen endpoints without reconstructing profiles."""

    candidate_rr, _ = midrank_reciprocal_and_top_one(tables.scores["candidate"])
    width_rr, _ = midrank_reciprocal_and_top_one(tables.scores["width1"])
    backoff = width_preferred_backoff(
        tables.scores["width1"], tables.scores["candidate"]
    )
    backoff_rr, _ = midrank_reciprocal_and_top_one(backoff)
    endpoint_parts = {name: [] for name in PRIMARY_ENDPOINTS}
    family_sums: dict[str, dict[str, float]] = {
        name: defaultdict(float) for name in PRIMARY_ENDPOINTS
    }
    family_counts: dict[str, dict[str, int]] = {
        name: defaultdict(int) for name in PRIMARY_ENDPOINTS
    }
    for start in range(0, len(permutations), chunk_size):
        mapping = permutations[start : start + chunk_size]
        columns = _truth_columns(tables, mapping)
        cand = _gather(candidate_rr, columns)
        width = _gather(width_rr, columns)
        backed = _gather(backoff_rr, columns)
        width_valid = np.isfinite(_gather(tables.scores["width1"], columns))
        all_candidate, candidate_families = _batch_hierarchical(cand, tables.anchors)
        rescue, rescue_families = _batch_hierarchical(
            cand * (~width_valid), tables.anchors
        )
        delta, delta_families = _batch_hierarchical(backed - width, tables.anchors)
        values = {
            "candidate_all": (all_candidate, candidate_families),
            "candidate_rescue": (rescue, rescue_families),
            "backoff_delta": (delta, delta_families),
        }
        for endpoint, (overall, by_family) in values.items():
            endpoint_parts[endpoint].append(overall)
            for family, family_values in by_family.items():
                finite = np.isfinite(family_values)
                family_sums[endpoint][family] += float(np.sum(family_values[finite]))
                family_counts[endpoint][family] += int(np.sum(finite))
    endpoints = {name: np.concatenate(parts) for name, parts in endpoint_parts.items()}
    family_null = {
        endpoint: {
            family: family_sums[endpoint][family] / count
            for family, count in family_counts[endpoint].items()
            if count
        }
        for endpoint in PRIMARY_ENDPOINTS
    }
    return endpoints, family_null


def _support_summary(
    rows: Sequence[PairRow],
    name: str,
    *,
    anchor_ids: set[str] | None = None,
) -> dict[str, Any]:
    supports = [
        row.scores[name].support
        for row in rows
        if row.scores[name].valid
        and (anchor_ids is None or row.anchor_id in anchor_ids)
    ]
    return {
        "valid_pair_count": len(supports),
        "minimum": min(supports) if supports else None,
        "median": float(np.median(supports)) if supports else None,
        "maximum": max(supports) if supports else None,
    }


def _observed_metrics(
    tables: ScoreTables, rows: Sequence[PairRow]
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    identity = np.arange(len(tables.responses), dtype=np.int64)
    columns = _truth_columns(tables, identity)
    methods = {
        name: midrank_reciprocal_and_top_one(scores)
        for name, scores in tables.scores.items()
    }
    backoff_scores = width_preferred_backoff(
        tables.scores["width1"], tables.scores["candidate"]
    )
    raw_backoff_scores = width_preferred_backoff(
        tables.scores["width1"],
        tables.scores["candidate"],
        percentile_calibrated=False,
    )
    methods["backoff"] = midrank_reciprocal_and_top_one(backoff_scores)
    methods["raw_backoff"] = midrank_reciprocal_and_top_one(raw_backoff_scores)
    gathered = {
        name: (_gather(rr, columns), _gather(top, columns))
        for name, (rr, top) in methods.items()
    }
    valid = {
        name: np.isfinite(_gather(scores, columns))
        for name, scores in {
            **tables.scores,
            "backoff": backoff_scores,
            "raw_backoff": raw_backoff_scores,
        }.items()
    }

    def summary(name: str, subset: np.ndarray | None = None) -> dict[str, Any]:
        rr, top = gathered[name]
        base = np.ones(len(rr), dtype=bool) if subset is None else subset
        scored = base & valid[name]
        return {
            "subset_anchor_count": int(np.sum(base)),
            "valid_true_score_count": int(np.sum(scored)),
            "weighted_true_score_coverage": hierarchical_mean(
                valid[name].astype(float), tables.anchors, base
            ),
            "zero_filled_mrr": hierarchical_mean(rr, tables.anchors, base),
            "zero_filled_top_one": hierarchical_mean(top, tables.anchors, base),
            "scored_only_mrr": (
                hierarchical_mean(rr, tables.anchors, scored)
                if np.any(scored)
                else None
            ),
            "scored_only_top_one": (
                hierarchical_mean(top, tables.anchors, scored)
                if np.any(scored)
                else None
            ),
        }

    width_missing = ~valid["width1"]
    rescue_rr = gathered["candidate"][0] * width_missing
    rescue_top = gathered["candidate"][1] * width_missing
    results = {
        "candidate_all": summary("candidate"),
        "candidate_rescue": summary("candidate", width_missing),
        "candidate_rescue_contribution": {
            "anchor_count": len(tables.anchors),
            "width1_invalid_true_count": int(np.sum(width_missing)),
            "zero_filled_mrr": hierarchical_mean(rescue_rr, tables.anchors),
            "zero_filled_top_one": hierarchical_mean(rescue_top, tables.anchors),
        },
        "width1": summary("width1"),
        "backoff": summary("backoff"),
        "raw_cosine_backoff_sensitivity": summary("raw_backoff"),
        "rank_channels": {
            f"rank{rank}": summary(f"rank{rank}") for rank in range(1, 6)
        },
        "candidate_analytic_chance": analytic_chance(
            tables.scores["candidate"], tables.planned, tables.anchors
        ),
        "candidate_rescue_analytic_chance": analytic_chance(
            tables.scores["candidate"],
            tables.planned,
            tables.anchors,
            width_missing,
        ),
    }
    results["candidate_all"]["common_signed_basis_support"] = _support_summary(
        rows, "candidate"
    )
    results["candidate_rescue"]["common_signed_basis_support"] = _support_summary(
        rows,
        "candidate",
        anchor_ids={
            tables.anchors[index].case_id for index in np.flatnonzero(width_missing)
        },
    )
    results["width1"]["common_signed_basis_support"] = _support_summary(rows, "width1")
    results["backoff"]["zero_filled_mrr_minus_width1"] = (
        results["backoff"]["zero_filled_mrr"] - results["width1"]["zero_filled_mrr"]
    )
    results["raw_cosine_backoff_sensitivity"]["zero_filled_mrr_minus_width1"] = (
        results["raw_cosine_backoff_sensitivity"]["zero_filled_mrr"]
        - results["width1"]["zero_filled_mrr"]
    )
    for rank in range(1, 6):
        name = f"rank{rank}"
        results["rank_channels"][name]["analytic_chance"] = analytic_chance(
            tables.scores[name], tables.planned, tables.anchors
        )
        results["rank_channels"][name]["common_signed_basis_support"] = (
            _support_summary(rows, name)
        )
    family_observed = {
        "candidate_all": family_means(gathered["candidate"][0], tables.anchors),
        "candidate_rescue": family_means(rescue_rr, tables.anchors),
        "backoff_delta": family_means(
            gathered["backoff"][0] - gathered["width1"][0], tables.anchors
        ),
    }
    return results, family_observed


def _bootstrap_effects(
    effects: Mapping[str, float], *, replicates: int, seed: int
) -> dict[str, Any]:
    families = sorted(effects)
    values = np.asarray([effects[family] for family in families], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap family effects must be finite and non-empty")
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(values), size=(replicates, len(values)))
    estimates = np.mean(values[draw], axis=1)
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "method": "percentile_family_block_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "family_count": len(families),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _lofo(effects: Mapping[str, float]) -> dict[str, Any]:
    families = sorted(effects)
    if len(families) < 2:
        raise ValueError("LOFO requires at least two families")
    estimates = {
        omitted: float(
            np.mean([value for family, value in effects.items() if family != omitted])
        )
        for omitted in families
    }
    positive = sum(value > 0.0 for value in estimates.values())
    return {
        "family_count": len(families),
        "positive_count": positive,
        "positive_fraction": positive / len(families),
        "estimates": estimates,
    }


def _derived_seed(audited_report_sha256: str, protocol_sha256: str) -> int:
    digest = hashlib.sha256(
        (
            f"{audited_report_sha256}:{protocol_sha256}:" f"{REPORT_SCHEMA_VERSION}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _pair_row_json(row: PairRow) -> dict[str, Any]:
    return {
        "anchor_id": row.anchor_id,
        "anchor_family_id": row.anchor_family_id,
        "anchor_response_id": row.anchor_response_id,
        "phase_bin": row.phase_bin,
        "target_id": row.target_id,
        "target_family_id": row.target_family_id,
        "target_response_id": row.target_response_id,
        "true_continuation": row.true_continuation,
        "scores": {
            name: {
                "score": value.score,
                "common_signed_basis_support": value.support,
                "valid": value.valid,
                "invalid_reason": value.invalid_reason,
            }
            for name, value in sorted(row.scores.items())
        },
    }


def _load_validated_profiles(
    *,
    selection_path: Path,
    plan_path: Path,
    width1_root: Path,
    candidate_union_root: Path,
    audited_c2_report_path: Path,
    protocol_path: Path,
) -> tuple[list[c2.TargetProfile], dict[str, Any]]:
    """Mirror C2 validation while loading profiles exactly once."""

    selection = c2._load_json(selection_path)
    plan = c2._load_json(plan_path)
    selection_sha = c2._sha256_file(selection_path)
    plan_sha = c2._sha256_file(plan_path)
    if plan.get("cohort", {}).get("selection_sha256") != selection_sha:
        raise ValueError("candidate-union plan does not bind the selection")
    selection_manifest = c2._validate_bound_json_file(
        selection, "source_manifest_path", "source_manifest_sha256"
    )
    plan_manifest = c2._validate_bound_json_file(
        plan["source"], "width1_manifest_path", "width1_manifest_sha256"
    )
    if selection_manifest != plan_manifest:
        raise ValueError("selection and plan bind different source manifests")
    config = c2._validate_bound_json_file(
        plan["execution"], "config_path", "config_sha256"
    )
    if c2._canonical_sha256(config) != plan["execution"].get("config_canonical_sha256"):
        raise ValueError("execution config canonical hash drift")
    cases, plan_cases = c2._validate_selection_and_plan(selection, plan)
    source_ids = {case["source_width1_artifact_id"] for case in cases}
    plan_canonical_sha = c2._canonical_sha256(plan)
    width_paths = c2._index_width1(width1_root, source_ids)
    union_paths = c2._index_candidate_union(
        candidate_union_root, source_ids, plan_canonical_sha
    )
    profiles = c2._load_profiles(
        cases,
        plan_cases,
        width_paths,
        union_paths,
        model_id=plan["source"]["model_id"],
        model_revision=plan["source"]["model_revision"],
        chat_template_sha256=plan["source"]["chat_template_sha256"],
        plan_sha256=plan_canonical_sha,
    )

    audited_sha = c2._sha256_file(audited_c2_report_path)
    if audited_sha != EXPECTED_AUDITED_C2_REPORT_SHA256:
        raise ValueError("audited C2 report SHA-256 is not the frozen report")
    protocol_sha = c2._sha256_file(protocol_path)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("salvage protocol SHA-256 drift")
    audited = c2._load_json(audited_c2_report_path)
    if audited.get("schema_version") != c2.REPORT_SCHEMA_VERSION:
        raise ValueError("audited C2 report has the wrong schema")
    if audited.get("overall_gate_passes") is not False:
        raise ValueError("salvage protocol requires the audited failed C2 decision")
    audited_inputs = audited.get("inputs", {})
    if (
        audited_inputs.get("selection_file_sha256") != selection_sha
        or audited_inputs.get("plan_file_sha256") != plan_sha
        or audited_inputs.get("plan_canonical_sha256") != plan_canonical_sha
    ):
        raise ValueError("audited C2 report input provenance disagrees")
    payload_set_sha = c2._canonical_sha256(
        [
            {
                "source_width1_artifact_id": profile.source_artifact_id,
                "width1_payload_sha256": profile.width1_payload_sha256,
                "candidate_union_payload_sha256": profile.candidate_union_payload_sha256,
            }
            for profile in profiles
        ]
    )
    if audited_inputs.get("artifact_payload_set_sha256") != payload_set_sha:
        raise ValueError("audited C2 report artifact payload set disagrees")
    c2_module_path = Path(c2.__file__).resolve()
    if audited_inputs.get("analysis_module_sha256") != c2._sha256_file(c2_module_path):
        raise ValueError(
            "checksum-bound C2 analyzer has changed since the audited report"
        )
    provenance = {
        "selection_path": str(selection_path.resolve()),
        "selection_file_sha256": selection_sha,
        "plan_path": str(plan_path.resolve()),
        "plan_file_sha256": plan_sha,
        "plan_canonical_sha256": plan_canonical_sha,
        "width1_root": str(width1_root.resolve()),
        "candidate_union_root": str(candidate_union_root.resolve()),
        "audited_c2_report_path": str(audited_c2_report_path.resolve()),
        "audited_c2_report_sha256": audited_sha,
        "salvage_protocol_path": str(protocol_path.resolve()),
        "salvage_protocol_sha256": protocol_sha,
        "audited_c2_decision_passes": audited.get("overall_gate_passes"),
        "c2_analysis_module_path": str(c2_module_path),
        "c2_analysis_module_sha256": c2._sha256_file(c2_module_path),
        "artifact_payload_set_sha256": payload_set_sha,
        "model_id": plan["source"]["model_id"],
        "model_revision": plan["source"]["model_revision"],
    }
    return profiles, provenance


def analyze_salvage(
    *,
    selection_path: Path,
    plan_path: Path,
    width1_root: Path,
    candidate_union_root: Path,
    audited_c2_report_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    profiles, input_provenance = _load_validated_profiles(
        selection_path=selection_path,
        plan_path=plan_path,
        width1_root=width1_root,
        candidate_union_root=candidate_union_root,
        audited_c2_report_path=audited_c2_report_path,
        protocol_path=protocol_path,
    )
    rows = construct_pair_rows(profiles)
    tables = pair_rows_to_tables(rows)
    if len(rows) != EXPECTED_PAIR_COUNT or len(tables.anchors) != EXPECTED_ANCHOR_COUNT:
        raise ValueError(
            f"frozen salvage pair shape drift: {len(rows)} rows, "
            f"{len(tables.anchors)} anchors"
        )
    audited_sha = input_provenance["audited_c2_report_sha256"]
    permutation_seed = _derived_seed(
        audited_sha, input_provenance["salvage_protocol_sha256"]
    )
    permutations, rejection_count = constrained_response_permutations(
        tables.responses,
        tables.response_families,
        replicates=PERMUTATION_REPLICATES,
        seed=permutation_seed,
    )
    observed, family_observed = _observed_metrics(tables, rows)
    null, family_null = permutation_inference(tables, permutations)
    observed_endpoints = {
        "candidate_all": observed["candidate_all"]["zero_filled_mrr"],
        "candidate_rescue": observed["candidate_rescue_contribution"][
            "zero_filled_mrr"
        ],
        "backoff_delta": observed["backoff"]["zero_filled_mrr_minus_width1"],
    }
    raw_p = {
        endpoint: (1 + int(np.sum(null[endpoint] >= observed_endpoints[endpoint])))
        / (PERMUTATION_REPLICATES + 1)
        for endpoint in PRIMARY_ENDPOINTS
    }
    adjusted_p = holm_adjust(raw_p)
    endpoint_reports = {}
    for endpoint_index, endpoint in enumerate(PRIMARY_ENDPOINTS):
        common_families = sorted(
            set(family_observed[endpoint]).intersection(family_null[endpoint])
        )
        effects = {
            family: family_observed[endpoint][family] - family_null[endpoint][family]
            for family in common_families
        }
        if not math.isclose(
            float(np.mean(list(effects.values()))),
            float(observed_endpoints[endpoint] - np.mean(null[endpoint])),
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{endpoint} centered family effects disagree with global effect"
            )
        bootstrap_seed = permutation_seed ^ (0x9E3779B97F4A7C15 + endpoint_index)
        bootstrap_seed &= (1 << 64) - 1
        endpoint_reports[endpoint] = {
            "observed": observed_endpoints[endpoint],
            "permutation_null_mean": float(np.mean(null[endpoint])),
            "observed_minus_permutation_null_mean": float(
                observed_endpoints[endpoint] - np.mean(null[endpoint])
            ),
            "permutation_null_percentiles": {
                "2.5": float(np.percentile(null[endpoint], 2.5)),
                "50": float(np.percentile(null[endpoint], 50)),
                "97.5": float(np.percentile(null[endpoint], 97.5)),
            },
            "one_sided_empirical_p": raw_p[endpoint],
            "holm_adjusted_one_sided_p": adjusted_p[endpoint],
            "family_effects": effects,
            "family_block_bootstrap": _bootstrap_effects(
                effects, replicates=BOOTSTRAP_REPLICATES, seed=bootstrap_seed
            ),
            "leave_one_family_out": _lofo(effects),
        }
    endpoint_reports["candidate_all"]["promising"] = bool(
        endpoint_reports["candidate_all"]["observed_minus_permutation_null_mean"]
        >= 0.03
        and adjusted_p["candidate_all"] < 0.05
        and endpoint_reports["candidate_all"]["family_block_bootstrap"]["lower_95"]
        > 0.0
    )
    endpoint_reports["candidate_rescue"]["promising"] = bool(
        endpoint_reports["candidate_rescue"]["observed_minus_permutation_null_mean"]
        >= 0.03
        and adjusted_p["candidate_rescue"] < 0.05
        and endpoint_reports["candidate_rescue"]["family_block_bootstrap"]["lower_95"]
        > 0.0
    )
    endpoint_reports["backoff_delta"]["promising"] = bool(
        observed_endpoints["backoff_delta"] >= 0.02
        and adjusted_p["backoff_delta"] < 0.05
        and endpoint_reports["backoff_delta"]["family_block_bootstrap"]["lower_95"]
        > 0.0
    )
    row_json = [_pair_row_json(row) for row in rows]
    module_path = Path(__file__).resolve()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "claim_boundary": {
            "exploratory_post_hoc": True,
            "failed_c2_decision_retained": input_provenance[
                "audited_c2_decision_passes"
            ]
            is False,
            "retroactively_passes_c2": False,
            "authorizes_full_matched_run": False,
            "holdout_touched": False,
        },
        "inputs": {
            **input_provenance,
            "analysis_module_path": str(module_path),
            "analysis_module_sha256": c2._sha256_file(module_path),
            "analysis_code_revision": collect_code_revision(module_path.parents[2]),
            "analysis_runtime_environment": collect_runtime_environment(),
            "pair_table_sha256": c2._canonical_sha256(row_json),
        },
        "method": {
            "minimum_common_non_boundary_mlp_bases": MIN_COMMON_BASES,
            "pair_count": len(rows),
            "anchor_count": len(tables.anchors),
            "response_count": len(tables.responses),
            "family_count": len(set(tables.response_families)),
            "ranking": "descending cosine with average midranks for exact ties; a first-place tie of size t gives top-one credit 1/t",
            "missing_truth_policy": "zero reciprocal rank and zero top-one",
            "weighting": "equal family then response then retained anchor",
            "backoff": "within-anchor mid-empirical-CDF width1 when valid, otherwise candidate when valid",
            "raw_cosine_backoff": "prespecified descriptive sensitivity only",
            "permutation": {
                "unit": "whole response trajectory",
                "replicates": PERMUTATION_REPLICATES,
                "seed_derivation": "first 64 bits of SHA256(audited C2 report SHA256 + frozen protocol SHA256 + schema ID)",
                "numeric_seed": permutation_seed,
                "rejected_proposal_count": rejection_count,
                "fixed_points_permitted": True,
                "same_family_alternative_mappings_rejected": True,
                "numpy_version": np.__version__,
            },
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "multiplicity": "Holm correction over E1, E2, and E3",
        },
        "observed": observed,
        "primary_endpoints": endpoint_reports,
        "pair_table": row_json,
    }


def _write_json_atomic_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {path}")
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
        # Recheck after analysis so an independently created result is not replaced.
        if path.exists():
            raise FileExistsError(f"refusing to overwrite analysis output: {path}")
        os.link(temporary_name, path)
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--width1-root", type=Path, required=True)
    parser.add_argument("--candidate-union-root", type=Path, required=True)
    parser.add_argument("--audited-c2-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(
            f"refusing to overwrite analysis output: {args.output_json}"
        )
    report = analyze_salvage(
        selection_path=args.selection,
        plan_path=args.plan,
        width1_root=args.width1_root,
        candidate_union_root=args.candidate_union_root,
        audited_c2_report_path=args.audited_c2_report,
        protocol_path=args.protocol,
    )
    _write_json_atomic_no_overwrite(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "candidate_all_promising": report["primary_endpoints"]["candidate_all"][
                    "promising"
                ],
                "candidate_rescue_promising": report["primary_endpoints"][
                    "candidate_rescue"
                ]["promising"],
                "backoff_promising": report["primary_endpoints"]["backoff_delta"][
                    "promising"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
