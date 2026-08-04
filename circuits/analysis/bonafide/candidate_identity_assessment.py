"""Frozen candidate-identity assembly and label-free C2-W64 evaluation.

The executable path deliberately consumes only generation and selection rows.
Audit candidate metadata and candidate-profile values are never materialized.
Sparse dictionaries are fit from generation events and then reused unchanged for
selection, support controls, and direction nulls.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from circuits.analysis.bonafide.candidate_clustering_execution import (
    _publish_directory_no_replace,
)
from circuits.analysis.bonafide.candidate_identity_source import (
    EXPECTED_TARGET_COUNTS,
    PARTITIONS,
    SOURCE_SCHEMA_VERSION,
    load_candidate_identity_source,
)
from circuits.analysis.bonafide.candidate_identity_source import (
    EXPOSURE_CONTRACT as SOURCE_EXPOSURE_CONTRACT,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)

SCHEMA_VERSION = "adag.bonafide.candidate-identity-assessment.v1"
REPORT_FILE = "assessment.json"
VARIANTS = ("T", "P", "SR", "M")
ALL_VIEWS = ("R", "T", "P", "SR", "M")
LOCAL_VARIANTS = ("T", "P", "SR")
GENERATION_FAMILY_COUNT = 18
SELECTION_FAMILY_COUNT = 8
NULL_REPLICATES = 100
BOOTSTRAP_REPLICATES = 10_000
NULL_EFFECTIVENESS_THRESHOLD = 0.8

_PROTOCOL_RELATIVE = "docs/CANDIDATE_IDENTITY_ALIGNMENT_PROTOCOL.md"
_SOURCE_PATHS = (
    "circuits/analysis/bonafide/candidate_identity_assessment.py",
    "scripts/bonafide/candidate_identity_assess.py",
    _PROTOCOL_RELATIVE,
)

Sparse = dict[int, float]
RawKey = tuple[Any, ...]


@dataclass(frozen=True)
class CandidateEvent:
    rank: int
    token_id: int
    token_text: str
    value: float


@dataclass(frozen=True)
class SourceRow:
    family_id: str
    response_id: str
    target_id: str
    basis_index: int
    cluster_id: int
    layer: int
    polarity: str
    phase: int
    observed_token_id: int
    observed_token_text: str
    events: tuple[CandidateEvent, ...]


@dataclass(frozen=True)
class ProjectedRow:
    family_id: str
    response_id: str
    target_id: str
    basis_index: int
    cluster_id: int
    layer: int
    polarity: str
    phase: int
    vector: Sparse | None
    support: Sparse
    right_target_id: str | None = None


@dataclass(frozen=True)
class SparseCentroids:
    values: Mapping[int, Sparse]
    available: frozenset[int]
    cluster_reports: tuple[Mapping[str, Any], ...]


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def _collect_revision(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    if Path(str(_git(root, "rev-parse", "--show-toplevel"))).resolve() != root:
        raise ValueError("candidate identity assessment must run from repository root")
    status = str(_git(root, "status", "--porcelain=v1", "--untracked-files=no"))
    if status:
        raise ValueError(
            "candidate identity assessment requires a clean tracked worktree"
        )
    records = []
    for relative in _SOURCE_PATHS:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"candidate identity source is not tracked: {relative}")
        blob = str(_git(root, "rev-parse", f"HEAD:{relative}"))
        if _git(root, "hash-object", relative) != blob:
            raise ValueError(f"candidate identity source differs from HEAD: {relative}")
        records.append(
            {"path": relative, "git_blob": blob, "sha256": file_sha256(root / relative)}
        )
    return {
        "repo_root": str(root),
        "git_commit": str(_git(root, "rev-parse", "HEAD")),
        "git_tree": str(_git(root, "rev-parse", "HEAD^{tree}")),
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(b"").hexdigest(),
        "files": records,
    }


def _validate_revision(revision: Mapping[str, Any]) -> None:
    root = Path(str(revision.get("repo_root"))).resolve()
    if not (root / ".git").exists():
        root = Path(__file__).resolve().parents[3]
    commit = str(revision.get("git_commit"))
    if (
        revision.get("tracked_worktree_clean") is not True
        or revision.get("tracked_status_sha256") != hashlib.sha256(b"").hexdigest()
        or _git(root, "rev-parse", f"{commit}^{{tree}}") != revision.get("git_tree")
    ):
        raise ValueError("candidate identity producing revision drift")
    raw = revision.get("files")
    if not isinstance(raw, list):
        raise TypeError("candidate identity source inventory is invalid")
    records = {str(item.get("path")): item for item in raw if isinstance(item, Mapping)}
    if set(records) != set(_SOURCE_PATHS):
        raise ValueError("candidate identity source inventory drift")
    for relative in _SOURCE_PATHS:
        content = _git(root, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(content, bytes)
        if (
            records[relative].get("git_blob")
            != _git(root, "rev-parse", f"{commit}:{relative}")
            or records[relative].get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"candidate identity source object drift: {relative}")


def _canonical_key(key: RawKey) -> str:
    """Serialize a sparse key without losing Unicode identity."""

    return json.dumps(key, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _surface_class(text: str) -> tuple[str, str, bool]:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    categories = tuple(unicodedata.category(character) for character in normalized)
    if not normalized:
        kind = "empty"
    elif all(category.startswith("L") for category in categories):
        kind = "letters"
    elif all(category == "Nd" for category in categories):
        kind = "digits"
    elif (
        all(category.startswith("L") or category == "Nd" for category in categories)
        and any(category.startswith("L") for category in categories)
        and any(category == "Nd" for category in categories)
    ):
        kind = "alphanumeric"
    elif all(category.startswith(("P", "S")) for category in categories):
        kind = "punctuation_symbol"
    else:
        kind = "mixed"
    return kind, normalized, bool(text) and text[0].isspace()


def surface_relation_key(observed_text: str, competitor_text: str) -> RawKey:
    observed_class, observed, observed_space = _surface_class(observed_text)
    competitor_class, competitor, competitor_space = _surface_class(competitor_text)
    if observed == competitor:
        relation = "equal"
    elif (
        observed
        and competitor
        and min(len(observed), len(competitor)) >= 2
        and (observed.startswith(competitor) or competitor.startswith(observed))
    ):
        relation = "prefix"
    elif (
        observed
        and competitor
        and min(len(observed), len(competitor)) >= 2
        and (observed.endswith(competitor) or competitor.endswith(observed))
    ):
        relation = "suffix"
    else:
        relation = "none"
    return (
        "surface",
        observed_class,
        competitor_class,
        relation,
        "same" if observed_space == competitor_space else "different",
    )


def _event_key(row: SourceRow, event: CandidateEvent, view: str) -> RawKey:
    if view == "R":
        return ("rank", event.rank)
    if view == "T":
        return ("token", event.token_id)
    if view == "P":
        return ("ordered-token-pair", row.observed_token_id, event.token_id)
    if view == "SR":
        return surface_relation_key(row.observed_token_text, event.token_text)
    raise ValueError(f"unsupported local candidate view: {view}")


def _dictionary(
    raw_rows: Sequence[
        tuple[SourceRow, Mapping[RawKey, float], Mapping[RawKey, float]]
    ],
) -> dict[str, int]:
    keys = {_canonical_key(key) for _, values, _ in raw_rows for key in values}
    return {key: index for index, key in enumerate(sorted(keys))}


def _project_raw(
    row: SourceRow,
    values: Mapping[RawKey, float],
    counts: Mapping[RawKey, float],
    dictionary: Mapping[str, int],
    *,
    right_target_id: str | None = None,
) -> ProjectedRow:
    vector: Sparse = defaultdict(float)
    support: Sparse = defaultdict(float)
    if set(values) != set(counts):
        raise ValueError("scientific and support event keys differ")
    for key, value in values.items():
        coordinate = dictionary.get(_canonical_key(key))
        if coordinate is None:
            continue
        vector[coordinate] += float(value)
        support[coordinate] += float(counts[key])
    finite_nonzero = {
        key: value
        for key, value in vector.items()
        if math.isfinite(value) and value != 0.0
    }
    if any(not math.isfinite(value) for value in vector.values()):
        raise ValueError("candidate identity projection contains a nonfinite value")
    return ProjectedRow(
        family_id=row.family_id,
        response_id=row.response_id,
        target_id=row.target_id,
        basis_index=row.basis_index,
        cluster_id=row.cluster_id,
        layer=row.layer,
        polarity=row.polarity,
        phase=row.phase,
        vector=finite_nonzero or None,
        support=dict(support),
        right_target_id=right_target_id,
    )


def assemble_local_view(
    generation: Sequence[SourceRow], selection: Sequence[SourceRow], view: str
) -> tuple[dict[str, int], list[ProjectedRow], list[ProjectedRow]]:
    """Fit one R/T/P/SR dictionary on generation and project both partitions."""

    if view not in {"R", "T", "P", "SR"}:
        raise ValueError("local view must be R, T, P, or SR")

    def raw(row: SourceRow) -> tuple[dict[RawKey, float], dict[RawKey, float]]:
        result: dict[RawKey, float] = defaultdict(float)
        counts: dict[RawKey, float] = defaultdict(float)
        for event in row.events:
            key = _event_key(row, event, view)
            result[key] += event.value
            counts[key] += 1.0
        return dict(result), dict(counts)

    generation_raw = [(row, *raw(row)) for row in generation]
    dictionary = _dictionary(generation_raw)
    return (
        dictionary,
        [
            _project_raw(row, values, counts, dictionary)
            for row, values, counts in generation_raw
        ],
        [
            _project_raw(row, values, counts, dictionary)
            for row in selection
            for values, counts in (raw(row),)
        ],
    )


def _validate_phase_grid(rows: Sequence[SourceRow]) -> None:
    per_response: dict[str, dict[int, str]] = defaultdict(dict)
    for row in rows:
        if not 0 <= row.phase <= 6:
            raise ValueError("candidate row phase lies outside zero through six")
        previous = per_response[row.response_id].setdefault(row.phase, row.target_id)
        if previous != row.target_id:
            raise ValueError("response has multiple target identities in one phase")


def _motif_raw(
    rows: Sequence[SourceRow],
) -> list[tuple[SourceRow, dict[RawKey, float], dict[RawKey, float], str | None]]:
    _validate_phase_grid(rows)
    by_response_phase_basis: dict[tuple[str, int, int], SourceRow] = {
        (row.response_id, row.phase, row.basis_index): row for row in rows
    }
    if len(by_response_phase_basis) != len(rows):
        raise ValueError("duplicate response/phase/signed-basis row")
    result: list[
        tuple[SourceRow, dict[RawKey, float], dict[RawKey, float], str | None]
    ] = []
    for row in rows:
        values: dict[RawKey, float] = defaultdict(float)
        counts: dict[RawKey, float] = defaultdict(float)
        right_target: str | None = None
        if row.phase < 6:
            right = by_response_phase_basis.get(
                (row.response_id, row.phase + 1, row.basis_index)
            )
            if right is not None:
                left_events = {event.token_id: event for event in row.events}
                right_events = {event.token_id: event for event in right.events}
                for token_id in sorted(set(left_events) & set(right_events)):
                    left_key = ("motif", token_id, row.phase, row.phase + 1, "left")
                    right_key = ("motif", token_id, row.phase, row.phase + 1, "right")
                    values[left_key] += left_events[token_id].value
                    values[right_key] += right_events[token_id].value
                    counts[left_key] += 1.0
                    counts[right_key] += 1.0
                    right_target = right.target_id
        result.append((row, dict(values), dict(counts), right_target))
    return result


def assemble_motif_view(
    generation: Sequence[SourceRow], selection: Sequence[SourceRow]
) -> tuple[dict[str, int], list[ProjectedRow], list[ProjectedRow]]:
    generation_raw = _motif_raw(generation)
    dictionary = _dictionary(
        [(row, values, counts) for row, values, counts, _ in generation_raw]
    )
    selection_raw = _motif_raw(selection)
    return (
        dictionary,
        [
            _project_raw(row, values, counts, dictionary, right_target_id=right)
            for row, values, counts, right in generation_raw
        ],
        [
            _project_raw(row, values, counts, dictionary, right_target_id=right)
            for row, values, counts, right in selection_raw
        ],
    )


def _sparse_unit(vector: Sparse | None) -> Sparse | None:
    if not vector:
        return None
    norm = math.sqrt(math.fsum(value * value for value in vector.values()))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return {coordinate: value / norm for coordinate, value in vector.items()}


def _sparse_mean(vectors: Sequence[Sparse]) -> Sparse:
    if not vectors:
        return {}
    total: Sparse = defaultdict(float)
    for vector in vectors:
        for coordinate, value in vector.items():
            total[coordinate] += value
    scale = 1.0 / len(vectors)
    return {coordinate: value * scale for coordinate, value in total.items() if value}


def _hierarchical_cluster_centroids(
    rows: Sequence[ProjectedRow], *, support: bool, n_clusters: int = 64
) -> SparseCentroids:
    families = sorted({row.family_id for row in rows})
    if len(families) != GENERATION_FAMILY_COUNT:
        raise ValueError("generation dictionary requires exactly 18 families")
    grouped: dict[int, dict[tuple[str, str, str], list[Sparse]]] = {
        cluster: defaultdict(list) for cluster in range(n_clusters)
    }
    assigned = [0] * n_clusters
    missing = [0] * n_clusters
    for row in rows:
        if not 0 <= row.cluster_id < n_clusters:
            raise ValueError("W64 cluster ID is out of range")
        assigned[row.cluster_id] += 1
        vector = row.support if support else row.vector
        unit = _sparse_unit(vector)
        if unit is None:
            missing[row.cluster_id] += 1
            continue
        grouped[row.cluster_id][(row.family_id, row.response_id, row.target_id)].append(
            unit
        )
    centroids: dict[int, Sparse] = {}
    reports = []
    for cluster in range(n_clusters):
        target_means = {
            key: _sparse_mean(values) for key, values in grouped[cluster].items()
        }
        responses: dict[tuple[str, str], list[Sparse]] = defaultdict(list)
        for (family, response, _), vector in target_means.items():
            responses[(family, response)].append(vector)
        response_means = {
            key: _sparse_mean(values) for key, values in responses.items()
        }
        family_vectors: dict[str, list[Sparse]] = defaultdict(list)
        for (family, _), vector in response_means.items():
            family_vectors[family].append(vector)
        family_means = {
            family: _sparse_mean(values) for family, values in family_vectors.items()
        }
        final = _sparse_mean(
            [family_means[family] for family in families if family in family_means]
        )
        unit = _sparse_unit(final)
        if unit is not None:
            centroids[cluster] = unit
        reports.append(
            {
                "cluster_id": cluster,
                "available": unit is not None,
                "assigned_row_count": assigned[cluster],
                "missing_scientific_row_count": missing[cluster],
                "target_count": len(target_means),
                "response_count": len(response_means),
                "family_count": len(family_means),
            }
        )
    return SparseCentroids(centroids, frozenset(centroids), tuple(reports))


def _dot(left: Sparse, right: Sparse) -> float:
    if len(left) > len(right):
        left, right = right, left
    return math.fsum(value * right.get(key, 0.0) for key, value in left.items())


def average_tie_reciprocal_rank(
    vector: Sparse | None, true_cluster: int, centroids: SparseCentroids
) -> float:
    unit = _sparse_unit(vector)
    if unit is None or true_cluster not in centroids.available:
        return 0.0
    competitors = centroids.available - {true_cluster}
    if not competitors:
        return 0.0
    scores = {
        cluster: _dot(unit, centroids.values[cluster])
        for cluster in centroids.available
    }
    truth = scores[true_cluster]
    greater = sum(
        value > truth for cluster, value in scores.items() if cluster != true_cluster
    )
    tied = sum(
        value == truth for cluster, value in scores.items() if cluster != true_cluster
    )
    rank = 1.0 + greater + tied / 2.0
    return 1.0 / rank


def _hierarchical_weights(rows: Sequence[ProjectedRow]) -> np.ndarray:
    hierarchy: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for index, row in enumerate(rows):
        hierarchy[row.family_id][row.response_id][row.target_id].append(index)
    weights = np.zeros(len(rows), dtype=np.float64)
    if not hierarchy:
        return weights
    for responses in hierarchy.values():
        for targets in responses.values():
            for indices in targets.values():
                mass = (
                    1.0 / len(hierarchy) / len(responses) / len(targets) / len(indices)
                )
                weights[indices] = mass
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
        raise AssertionError("hierarchical weights do not sum to one")
    return weights


def _hierarchical_scores(
    rows: Sequence[ProjectedRow], values: Sequence[float]
) -> tuple[float, dict[str, float]]:
    if len(rows) != len(values):
        raise ValueError("hierarchical scores are misaligned")
    weights = _hierarchical_weights(rows)
    family_values: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        family_values[row.family_id].append(index)
    per_family = {
        family: float(
            np.dot(weights[indices], np.asarray(values)[indices]) * len(family_values)
        )
        for family, indices in sorted(family_values.items())
    }
    return float(mean(per_family.values())), per_family


def _coverage(
    rows: Sequence[ProjectedRow],
    *,
    support: bool,
    centroids: SparseCentroids,
    require_competitor: bool,
) -> dict[str, Any]:
    weights = _hierarchical_weights(rows)
    scoreable = []
    for row in rows:
        vector = row.support if support else row.vector
        scoreable.append(
            _sparse_unit(vector) is not None
            and row.cluster_id in centroids.available
            and (not require_competitor or bool(centroids.available - {row.cluster_id}))
        )
    indexes = [index for index, value in enumerate(scoreable) if value]
    return {
        "row_count": len(indexes),
        "total_row_count": len(rows),
        "hierarchical_weight": float(weights[indexes].sum()) if indexes else 0.0,
        "target_count": len({rows[index].target_id for index in indexes}),
        "response_count": len({rows[index].response_id for index in indexes}),
        "family_count": len({rows[index].family_id for index in indexes}),
    }


def _score_selection(
    rows: Sequence[ProjectedRow], centroids: SparseCentroids, *, support: bool
) -> dict[str, Any]:
    values = [
        average_tie_reciprocal_rank(
            row.support if support else row.vector, row.cluster_id, centroids
        )
        for row in rows
    ]
    mrr, per_family = _hierarchical_scores(rows, values)
    return {
        "zero_filled_mrr": mrr,
        "per_family_mrr": per_family,
        "coverage": _coverage(
            rows,
            support=support,
            centroids=centroids,
            require_competitor=True,
        ),
    }


def _consistency(
    rows: Sequence[ProjectedRow], *, support: bool = False
) -> dict[str, Any]:
    by_basis: dict[int, list[ProjectedRow]] = defaultdict(list)
    for row in rows:
        by_basis[row.basis_index].append(row)
    values = []
    eligible_rows = 0
    for basis_rows in by_basis.values():
        anchored = [
            row
            for row in basis_rows
            if _sparse_unit(row.support if support else row.vector) is not None
        ]
        if (
            len({row.target_id for row in anchored}) < 3
            or len({row.response_id for row in anchored}) < 2
            or len({row.family_id for row in anchored}) < 2
        ):
            continue
        eligible_rows += len(anchored)
        targets = {
            (row.family_id, row.response_id, row.target_id): _sparse_unit(
                row.support if support else row.vector
            )
            for row in anchored
        }
        responses: dict[tuple[str, str], list[Sparse]] = defaultdict(list)
        for (family, response, _), vector in targets.items():
            assert vector is not None
            responses[(family, response)].append(vector)
        response_means = {key: _sparse_mean(items) for key, items in responses.items()}
        families: dict[str, list[Sparse]] = defaultdict(list)
        for (family, _), vector in response_means.items():
            families[family].append(vector)
        final = _sparse_mean([_sparse_mean(items) for items in families.values()])
        values.append(math.sqrt(math.fsum(value * value for value in final.values())))
    return {
        "eligible_basis_count": len(values),
        "eligible_row_count": eligible_rows,
        "coverage_fraction": eligible_rows / len(rows) if rows else 0.0,
        "distribution": {
            "count": len(values),
            "mean": float(mean(values)) if values else None,
            "median": float(median(values)) if values else None,
            "minimum": float(min(values)) if values else None,
            "maximum": float(max(values)) if values else None,
            "values": sorted(float(value) for value in values),
        },
    }


def _seed(protocol_sha256: str, namespace: str, *parts: str | int) -> int:
    payload = protocol_sha256.encode("ascii") + b"\0" + namespace.encode("ascii")
    for part in parts:
        payload += b"\0"
        payload += (
            part.to_bytes(8, "big") if isinstance(part, int) else part.encode("utf-8")
        )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bootstrap(
    effects: Mapping[str, float], protocol_sha256: str, comparison: str
) -> dict[str, Any]:
    if len(effects) != SELECTION_FAMILY_COUNT:
        raise ValueError("selection bootstrap requires exactly eight family effects")
    values = np.asarray([effects[key] for key in sorted(effects)], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("selection bootstrap effects must be finite")
    seed = _seed(
        protocol_sha256, "candidate-identity-selection-bootstrap-v1", comparison
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    samples = values[draws].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "comparison": comparison,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "mean_effect": float(values.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def _null_blocks(
    rows: Sequence[ProjectedRow], variant: str
) -> tuple[list[list[int]], dict[str, Any]]:
    weights = _hierarchical_weights(rows)
    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not row.vector:
            continue
        support = tuple(sorted(row.support))
        prefix: tuple[Any, ...]
        if variant == "M":
            prefix = (row.response_id, row.phase, row.phase + 1)
        else:
            prefix = (row.target_id,)
        strata[prefix + (row.layer, row.polarity, support)].append(index)
    blocks: list[list[int]] = []
    movable: set[int] = set()
    for key in sorted(strata, key=repr):
        indexes = strata[key]
        ordered = sorted(
            indexes,
            key=lambda index: (
                math.sqrt(
                    math.fsum(
                        value * value for value in (rows[index].vector or {}).values()
                    )
                ),
                rows[index].basis_index,
            ),
        )
        n = len(ordered)
        deciles: dict[int, list[int]] = defaultdict(list)
        for ordinal, index in enumerate(ordered):
            deciles[min(9, (10 * ordinal) // n)].append(index)
        pending: list[int] = []
        local: list[list[int]] = []
        for decile in sorted(deciles):
            pending.extend(deciles[decile])
            if len(pending) >= 4:
                local.append(pending)
                pending = []
        if pending:
            if local:
                local[-1].extend(pending)
            else:
                local.append(pending)
        for block in local:
            if len(block) >= 4:
                blocks.append(block)
                movable.update(block)
    report = {
        "eligible_row_fraction": len(movable) / len(rows) if rows else 0.0,
        "eligible_hierarchical_weight": float(weights[sorted(movable)].sum())
        if movable
        else 0.0,
        "effective": (
            len(movable) / len(rows) >= NULL_EFFECTIVENESS_THRESHOLD
            and float(weights[sorted(movable)].sum()) >= NULL_EFFECTIVENESS_THRESHOLD
        )
        if rows
        else False,
        "movable_block_count": len(blocks),
    }
    return blocks, report


def _permuted_rows(
    rows: Sequence[ProjectedRow], blocks: Sequence[Sequence[int]], *, seed: int
) -> list[ProjectedRow]:
    rng = np.random.default_rng(seed)
    vectors = [row.vector for row in rows]
    for block in blocks:
        sources = rng.permutation(np.asarray(block, dtype=np.int64)).tolist()
        original = [rows[int(source)].vector for source in sources]
        for destination, vector in zip(block, original, strict=True):
            vectors[destination] = vector
    return [replace(row, vector=vectors[index]) for index, row in enumerate(rows)]


def evaluate_projected_views(
    views: Mapping[str, tuple[Sequence[ProjectedRow], Sequence[ProjectedRow]]],
    *,
    protocol_sha256: str,
    provenance_valid: bool,
) -> dict[str, Any]:
    """Evaluate the complete frozen family, controls, nulls, and gates."""

    if (
        len(protocol_sha256) != 64
        or protocol_sha256.lower() != protocol_sha256
        or any(character not in "0123456789abcdef" for character in protocol_sha256)
    ):
        raise ValueError("protocol_sha256 must be 64 lowercase hexadecimal characters")
    if tuple(views) != ALL_VIEWS:
        raise ValueError("evaluation requires the frozen R, T, P, SR, M view order")
    reports: dict[str, Any] = {}
    centroids: dict[str, SparseCentroids] = {}
    support_centroids: dict[str, SparseCentroids] = {}
    for view in ALL_VIEWS:
        generation, selection = views[view]
        centroids[view] = _hierarchical_cluster_centroids(generation, support=False)
        support_centroids[view] = _hierarchical_cluster_centroids(
            generation, support=True
        )
        reports[view] = {
            "generation": {
                "scientific_coverage": _coverage(
                    generation,
                    support=False,
                    centroids=centroids[view],
                    require_competitor=False,
                ),
                "support_coverage": _coverage(
                    generation,
                    support=True,
                    centroids=support_centroids[view],
                    require_competitor=False,
                ),
                "recurrent_basis_consistency": _consistency(generation),
                "centroid_reports": list(centroids[view].cluster_reports),
            },
            "selection": {
                "scientific": _score_selection(
                    selection, centroids[view], support=False
                ),
                "support": _score_selection(
                    selection, support_centroids[view], support=True
                ),
            },
        }
    r_mrr = reports["R"]["selection"]["scientific"]["zero_filled_mrr"]
    comparisons: dict[str, Any] = {}
    for variant in VARIANTS:
        value = reports[variant]["selection"]["scientific"]
        support = reports[variant]["selection"]["support"]
        effect_r = {
            family: value["per_family_mrr"][family]
            - reports["R"]["selection"]["scientific"]["per_family_mrr"][family]
            for family in value["per_family_mrr"]
        }
        effect_support = {
            family: value["per_family_mrr"][family] - support["per_family_mrr"][family]
            for family in value["per_family_mrr"]
        }
        comparisons[variant] = {
            "mrr_minus_R": value["zero_filled_mrr"] - r_mrr,
            "mrr_minus_support": value["zero_filled_mrr"] - support["zero_filled_mrr"],
            "positive_family_count_minus_R": sum(
                item > 0 for item in effect_r.values()
            ),
            "per_family_effect_minus_R": effect_r,
            "per_family_effect_minus_support": effect_support,
            "bootstrap_minus_R": _bootstrap(
                effect_r, protocol_sha256, f"{variant}-minus-R"
            ),
            "bootstrap_minus_support": _bootstrap(
                effect_support, protocol_sha256, f"{variant}-minus-{variant}_support"
            ),
        }

    blocks: dict[str, list[list[int]]] = {}
    effectiveness: dict[str, Any] = {}
    for variant in VARIANTS:
        blocks[variant], effectiveness[variant] = _null_blocks(
            views[variant][0], variant
        )
    null_max_r: list[float] = []
    null_max_support: list[float] = []
    null_valid = all(report["effective"] for report in effectiveness.values())
    for replicate in range(NULL_REPLICATES):
        effects_r = []
        effects_support = []
        for variant in VARIANTS:
            permuted = _permuted_rows(
                views[variant][0],
                blocks[variant],
                seed=_seed(
                    protocol_sha256,
                    "candidate-identity-direction-null-v1",
                    variant,
                    replicate,
                ),
            )
            null_centroids = _hierarchical_cluster_centroids(permuted, support=False)
            null_mrr = _score_selection(
                views[variant][1], null_centroids, support=False
            )["zero_filled_mrr"]
            effects_r.append(null_mrr - r_mrr)
            effects_support.append(
                null_mrr - reports[variant]["selection"]["support"]["zero_filled_mrr"]
            )
        if not all(math.isfinite(value) for value in effects_r + effects_support):
            raise ValueError("invalid direction-null replicate")
        null_max_r.append(max(effects_r))
        null_max_support.append(max(effects_support))
    thresholds = {
        "minus_R": float(np.quantile(null_max_r, 0.95, method="higher")),
        "minus_support": float(np.quantile(null_max_support, 0.95, method="higher")),
    }
    r_generation = reports["R"]["generation"]["scientific_coverage"][
        "hierarchical_weight"
    ]
    r_selection = reports["R"]["selection"]["scientific"]["coverage"][
        "hierarchical_weight"
    ]
    gates: dict[str, Any] = {}
    for variant in VARIANTS:
        comparison = comparisons[variant]
        generation_coverage = reports[variant]["generation"]["scientific_coverage"][
            "hierarchical_weight"
        ]
        selection_coverage = reports[variant]["selection"]["scientific"]["coverage"][
            "hierarchical_weight"
        ]
        recurrence = reports[variant]["generation"]["recurrent_basis_consistency"][
            "distribution"
        ]["median"]
        conditions = {
            "provenance_valid": provenance_valid is True,
            "all_eight_selection_families": len(
                reports[variant]["selection"]["scientific"]["per_family_mrr"]
            )
            == SELECTION_FAMILY_COUNT,
            "generation_scoreable_at_least_80pct_R": generation_coverage
            >= 0.8 * r_generation,
            "selection_scoreable_at_least_80pct_R": selection_coverage
            >= 0.8 * r_selection,
            "median_recurrent_consistency_at_least_055": recurrence is not None
            and recurrence >= 0.55,
            "mrr_lift_at_least_003": comparison["mrr_minus_R"] >= 0.03,
            "at_least_seven_positive_families": comparison[
                "positive_family_count_minus_R"
            ]
            >= 7,
            "bootstrap_minus_R_lower_above_zero": comparison["bootstrap_minus_R"][
                "ci_95_lower"
            ]
            > 0,
            "value_exceeds_support": comparison["mrr_minus_support"] > 0
            and comparison["bootstrap_minus_support"]["ci_95_lower"] > 0,
            "null_effective": null_valid and effectiveness[variant]["effective"],
            "effect_exceeds_joint_max_null": comparison["mrr_minus_R"]
            > thresholds["minus_R"],
            "support_effect_exceeds_joint_max_null": comparison["mrr_minus_support"]
            > thresholds["minus_support"],
        }
        gates[variant] = {"passed": all(conditions.values()), "conditions": conditions}
    passing = [variant for variant in VARIANTS if gates[variant]["passed"]]
    passing_local = [variant for variant in LOCAL_VARIANTS if gates[variant]["passed"]]
    order = {variant: index for index, variant in enumerate(VARIANTS)}

    def winner(candidates: Sequence[str]) -> str | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda variant: (
                comparisons[variant]["mrr_minus_R"],
                comparisons[variant]["mrr_minus_support"],
                -order[variant],
            ),
        )

    return {
        "views": reports,
        "comparisons": comparisons,
        "direction_null": {
            "replicates": NULL_REPLICATES,
            "all_replicates_valid": True,
            "all_variants_effective": null_valid,
            "effectiveness": effectiveness,
            "joint_max_thresholds_95_higher": thresholds,
        },
        "gates": gates,
        "offline_winner": winner(passing),
        "local_labeling_winner": winner(passing_local),
        "labeling_authorized": bool(passing_local),
    }


def _validate_self_hashed_manifest(
    root: Path,
    *,
    expected_schema: str,
    label: str,
    required_files: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    manifest = load_json_object(root / "manifest.json")
    core = dict(manifest)
    if core.pop("manifest_sha256", None) != canonical_sha256(core):
        raise ValueError(f"{label} manifest self-hash mismatch")
    if manifest.get("schema_version") != expected_schema:
        raise ValueError(f"unsupported {label} schema")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TypeError(f"{label} file inventory is invalid")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(f"{label} file inventory is invalid")
        name = str(record.get("path"))
        if Path(name).name != name or name in seen:
            raise ValueError(f"{label} file path is unsafe or duplicated")
        seen.add(name)
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or file_sha256(path) != record.get("sha256")
        ):
            raise ValueError(f"{label} file drift: {name}")
    if not required_files <= seen:
        raise ValueError(f"{label} required file inventory is incomplete")
    return manifest


def _validate_target_phase_grid(targets: Sequence[Mapping[str, Any]]) -> None:
    per_response: dict[str, dict[int, str]] = defaultdict(dict)
    for target in targets:
        response = str(target["response_id"])
        phase = int(target["phase_bin"])
        case_id = str(target["case_id"])
        if not 0 <= phase <= 6:
            raise ValueError("target phase lies outside zero through six")
        previous = per_response[response].setdefault(phase, case_id)
        if previous != case_id:
            raise ValueError("response has multiple targets in one phase")
    for response, phases in per_response.items():
        if sorted(phases) != list(range(7)):
            raise ValueError(
                f"response {response!r} lacks exactly one target per phase 0..6"
            )


def _parse_target_events(
    target: Mapping[str, Any], vector: Sequence[float]
) -> tuple[CandidateEvent, ...]:
    selection = json.loads(str(target["candidate_selection_json"]))
    if not isinstance(selection, Mapping) or not isinstance(
        selection.get("candidates"), list
    ):
        raise TypeError("candidate selection JSON is invalid")
    candidates = selection["candidates"]
    if not candidates or any(
        not isinstance(candidate, Mapping) for candidate in candidates
    ):
        raise TypeError("candidate selection entries are invalid")
    if len(candidates) not in {5, 6}:
        raise ValueError(
            "candidate selection must contain top five plus optional observed"
        )
    candidate_indices = [int(candidate["candidate_index"]) for candidate in candidates]
    if sorted(candidate_indices) != list(range(len(candidates))):
        raise ValueError("candidate selection indices are not contiguous")
    token_ids = [int(candidate["token_id"]) for candidate in candidates]
    ranks = [int(candidate["full_distribution_rank"]) for candidate in candidates]
    if len(set(token_ids)) != len(token_ids) or len(set(ranks)) != len(ranks):
        raise ValueError("candidate selection token IDs and ranks must be unique")
    by_rank = {
        int(candidate["full_distribution_rank"]): candidate
        for candidate in candidates
        if int(candidate["full_distribution_rank"]) in range(1, 6)
    }
    if set(by_rank) != set(range(1, 6)) or len(vector) != 5:
        raise ValueError(
            "candidate rank/value vector does not contain ranks one through five"
        )
    observed_id = int(target["observed_token_id"])
    observed = [candidate for candidate in candidates if bool(candidate["is_observed"])]
    if (
        len(observed) != 1
        or int(observed[0]["token_id"]) != observed_id
        or int(selection.get("observed_token_id", -1)) != observed_id
        or str(selection.get("observed_token_text"))
        != str(target["observed_token_text"])
    ):
        raise ValueError("candidate selection observed-token identity drift")
    observed_rank = int(observed[0]["full_distribution_rank"])
    if (len(candidates) == 5 and observed_rank not in range(1, 6)) or (
        len(candidates) == 6 and observed_rank in range(1, 6)
    ):
        raise ValueError("candidate selection observed-rank width semantics drift")
    events = []
    for rank in range(1, 6):
        candidate = by_rank[rank]
        if bool(candidate["is_observed"]):
            if (
                int(candidate["token_id"]) != observed_id
                or float(vector[rank - 1]) != 0.0
            ):
                raise ValueError("observed candidate slot is not a structural zero")
            continue
        value = float(vector[rank - 1])
        if not math.isfinite(value):
            raise ValueError("candidate contrast must be finite")
        events.append(
            CandidateEvent(
                rank, int(candidate["token_id"]), str(candidate["token_text"]), value
            )
        )
    return tuple(events)


def _load_source_rows(
    source_root: Path,
) -> tuple[dict[str, Any], list[SourceRow], list[SourceRow]]:
    """Load only the immutable executable-only source artifact."""

    manifest, target_tables, profile_tables = load_candidate_identity_source(
        source_root
    )
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("candidate identity source schema drift")
    if manifest.get("exposure_contract") != SOURCE_EXPOSURE_CONTRACT:
        raise ValueError("candidate identity source exposure drift")
    if manifest.get("cluster_state") != {
        "identifier": "c2_w64",
        "view": "W",
        "n_clusters": 64,
        "assignment": "medoid_seed",
    }:
        raise ValueError("candidate identity source C2-W64 binding drift")

    targets = [
        row
        for partition in PARTITIONS
        for row in target_tables[partition].to_pylist()
    ]
    _validate_target_phase_grid(targets)
    target_by_case = {str(row["case_id"]): row for row in targets}
    if len(target_by_case) != len(targets):
        raise ValueError("duplicate published target identity")

    result: dict[str, list[SourceRow]] = {partition: [] for partition in PARTITIONS}
    seen: set[tuple[str, int]] = set()
    for partition in PARTITIONS:
        for row in profile_tables[partition].to_pylist():
            case_id = str(row["case_id"])
            basis_index = int(row["signed_basis_index"])
            identity = (case_id, basis_index)
            if identity in seen:
                raise ValueError("duplicate published target-basis identity")
            seen.add(identity)
            target = target_by_case.get(case_id)
            if target is None or target["family_partition"] != partition:
                raise ValueError("published profile target binding drift")
            cluster = row["c2_w64_cluster_id"]
            assigned = bool(row["c2_w64_assigned"])
            if assigned != (cluster is not None):
                raise ValueError("published C2-W64 assignment nullability drift")
            if not assigned:
                continue
            result[partition].append(
                SourceRow(
                    family_id=str(target["base_question_id"]),
                    response_id=str(target["response_id"]),
                    target_id=case_id,
                    basis_index=basis_index,
                    cluster_id=int(cluster),
                    layer=int(row["layer"]),
                    polarity=str(row["polarity"]),
                    phase=int(target["phase_bin"]),
                    observed_token_id=int(target["observed_token_id"]),
                    observed_token_text=str(target["observed_token_text"]),
                    events=_parse_target_events(
                        target, row["candidate_contrast_vector"]
                    ),
                )
            )
    for partition, expected_families in (
        ("generation", GENERATION_FAMILY_COUNT),
        ("selection_scoring", SELECTION_FAMILY_COUNT),
    ):
        if target_tables[partition].num_rows != EXPECTED_TARGET_COUNTS[partition]:
            raise ValueError(f"{partition} published target count drift")
        if not result[partition]:
            raise ValueError(f"{partition} has no candidate-supported W64 rows")
        if len({row.family_id for row in result[partition]}) != expected_families:
            raise ValueError(f"{partition} does not contain the frozen family count")
    return manifest, result["generation"], result["selection_scoring"]


def compute_candidate_identity_assessment(
    generation: Sequence[SourceRow],
    selection: Sequence[SourceRow],
    *,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dictionaries: dict[str, dict[str, int]] = {}
    views: dict[str, tuple[Sequence[ProjectedRow], Sequence[ProjectedRow]]] = {}
    for view in ("R", "T", "P", "SR"):
        dictionary, generation_rows, selection_rows = assemble_local_view(
            generation, selection, view
        )
        dictionaries[view] = dictionary
        views[view] = (generation_rows, selection_rows)
    dictionary, generation_rows, selection_rows = assemble_motif_view(
        generation, selection
    )
    dictionaries["M"] = dictionary
    views["M"] = (generation_rows, selection_rows)
    report = evaluate_projected_views(
        views,
        protocol_sha256=protocol_sha256,
        provenance_valid=True,
    )
    dictionary_report = {
        view: {
            "dimension": len(values),
            "canonical_keys": sorted(values, key=values.get),
        }
        for view, values in dictionaries.items()
    }
    return report, dictionary_report


def _source_record(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(root.resolve()),
        "manifest_path": str((root / "manifest.json").resolve()),
        "schema_version": manifest["schema_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": file_sha256(root / "manifest.json"),
    }


def build_candidate_identity_assessment(
    *, source_root: Path, output_root: Path, repo_root: Path
) -> dict[str, Any]:
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to replace candidate identity artifact: {output}"
        )
    revision = _collect_revision(repo_root)
    protocol_sha256 = file_sha256(repo_root / _PROTOCOL_RELATIVE)
    source_manifest, generation, selection = _load_source_rows(source_root.resolve())
    report, dictionaries = compute_candidate_identity_assessment(
        generation, selection, protocol_sha256=protocol_sha256
    )
    report["dictionaries"] = dictionaries
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        report_bytes = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        report_path = temporary / REPORT_FILE
        with report_path.open("xb") as handle:
            handle.write(report_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        core: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "purpose": "label_free_candidate_identity_alignment_selection_evaluation",
            "candidate_identity_source": _source_record(source_root, source_manifest),
            "protocol": {
                "path": _PROTOCOL_RELATIVE,
                "sha256": protocol_sha256,
            },
            "producing_revision": revision,
            "decision": {
                "offline_winner": report["offline_winner"],
                "local_labeling_winner": report["local_labeling_winner"],
                "labeling_authorized": report["labeling_authorized"],
            },
            "firewall": {
                "audit_candidate_metadata_loaded": False,
                "audit_candidate_values_loaded": False,
                "audit_metrics_computed": False,
                "mixed_source_value_tables_opened": False,
                "raw_candidate_artifacts_opened": False,
                "labels_used": False,
                "model_calls_made": False,
            },
            "files": [
                {
                    "path": REPORT_FILE,
                    "sha256": hashlib.sha256(report_bytes).hexdigest(),
                    "size_bytes": len(report_bytes),
                }
            ],
        }
        manifest = {**core, "manifest_sha256": canonical_sha256(core)}
        with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if _collect_revision(repo_root) != revision:
            raise ValueError(
                "candidate identity producing revision changed during construction"
            )
        current_source, _, _ = load_candidate_identity_source(source_root)
        if current_source != source_manifest:
            raise ValueError("candidate identity source changed during construction")
        _publish_directory_no_replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_candidate_identity_assessment(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = root.resolve()
    manifest = _validate_self_hashed_manifest(
        artifact,
        expected_schema=SCHEMA_VERSION,
        label="candidate identity assessment",
        required_files=frozenset({REPORT_FILE}),
    )
    firewall = manifest.get("firewall")
    if not isinstance(firewall, Mapping) or any(
        value is not False for value in firewall.values()
    ):
        raise ValueError("candidate identity firewall drift")
    revision = manifest.get("producing_revision")
    if not isinstance(revision, Mapping):
        raise TypeError("candidate identity producing revision is invalid")
    _validate_revision(revision)
    records = manifest.get("files")
    if (
        not isinstance(records, list)
        or len(records) != 1
        or not isinstance(records[0], Mapping)
        or records[0].get("path") != REPORT_FILE
    ):
        raise ValueError("candidate identity report inventory drift")
    source_records = revision.get("files")
    if not isinstance(source_records, list):
        raise TypeError("candidate identity source revision inventory is invalid")
    source_by_path = {
        str(record.get("path")): record
        for record in source_records
        if isinstance(record, Mapping)
    }
    protocol = manifest.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("path") != _PROTOCOL_RELATIVE
        or protocol.get("sha256")
        != source_by_path.get(_PROTOCOL_RELATIVE, {}).get("sha256")
    ):
        raise ValueError("candidate identity protocol binding drift")
    report = load_json_object(artifact / REPORT_FILE)
    if manifest.get("decision") != {
        "offline_winner": report.get("offline_winner"),
        "local_labeling_winner": report.get("local_labeling_winner"),
        "labeling_authorized": report.get("labeling_authorized"),
    }:
        raise ValueError("candidate identity decision summary drift")
    source = manifest.get("candidate_identity_source")
    if not isinstance(source, Mapping):
        raise TypeError("candidate identity source binding is invalid")
    source_root = Path(str(source.get("path"))).resolve()
    source_manifest, _, _ = load_candidate_identity_source(source_root)
    if source.get("manifest_sha256") != source_manifest.get(
        "manifest_sha256"
    ) or source.get("manifest_file_sha256") != file_sha256(
        source_root / "manifest.json"
    ):
        raise ValueError("candidate identity source binding drift")
    return manifest, report
