"""Label-free diagnostics for candidate evidence on the fixed C2-W64 state.

The evaluator never fits clusters.  It uses generation rows for descriptive
direction/heterogeneity summaries and selection-scoring rows for the sole
decision: whether a constrained within-W64 refinement is worth fitting in a
separate, subsequently frozen artifact.  Dense overlap is reported but never
filters or weights that decision.  Audit rows, labels, outcomes, and model
outputs are outside this contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from circuits.analysis.bonafide.candidate_clustering_execution import (
    _publish_directory_no_replace,
)
from circuits.analysis.bonafide.candidate_multiplex_assessment import (
    LoadedCandidateMultiplexAssessment,
    load_candidate_multiplex_assessment,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)

DIAGNOSTICS_SCHEMA_VERSION = "adag.bonafide.candidate-multiplex-diagnostics.v1"
REPORT_FILE = "diagnostics.json"
PARTITIONS = ("generation", "selection_scoring")

# Frozen before the first live diagnostics run.  Passing these conditions does
# not promote an alternative; it only authorizes a separately frozen fit.
MIN_MEDIAN_BASIS_CONSISTENCY = 0.55
MIN_SELECTION_SEPARATION = 0.0
MIN_STABLE_BASES_PER_SPLITTABLE_PARENT = 10
MIN_GENERATION_TARGETS_PER_SPLITTABLE_PARENT = 8
MIN_GENERATION_FAMILIES_PER_SPLITTABLE_PARENT = 4
MIN_PARENT_HETEROGENEITY = 0.20
REQUIRED_SELECTION_FAMILY_COUNT = 8
REQUIRED_POSITIVE_SELECTION_FAMILY_COUNT = 7
SELECTION_BOOTSTRAP_REPLICATES = 10_000

_SOURCE_PATHS = (
    "circuits/analysis/bonafide/candidate_multiplex_diagnostics.py",
    "scripts/bonafide/candidate_multiplex_diagnostics.py",
)


def _git(repo_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def _collect_revision(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    if (
        Path(str(_git(repo_root, "rev-parse", "--show-toplevel"))).resolve()
        != repo_root
    ):
        raise ValueError("diagnostics must run from the repository root")
    status = str(_git(repo_root, "status", "--porcelain=v1", "--untracked-files=no"))
    if status:
        raise ValueError("diagnostics require a clean tracked worktree")
    records: list[dict[str, str]] = []
    for relative in _SOURCE_PATHS:
        if _git(repo_root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"diagnostics source is not tracked: {relative}")
        blob = str(_git(repo_root, "rev-parse", f"HEAD:{relative}"))
        if _git(repo_root, "hash-object", relative) != blob:
            raise ValueError(f"diagnostics source differs from HEAD: {relative}")
        records.append(
            {
                "path": relative,
                "git_blob": blob,
                "sha256": file_sha256(repo_root / relative),
            }
        )
    return {
        "repo_root": str(repo_root),
        "git_commit": str(_git(repo_root, "rev-parse", "HEAD")),
        "git_tree": str(_git(repo_root, "rev-parse", "HEAD^{tree}")),
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
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
        raise ValueError("diagnostics producing revision drift")
    raw = revision.get("files")
    if not isinstance(raw, list):
        raise TypeError("diagnostics source inventory is invalid")
    records = {str(item.get("path")): item for item in raw if isinstance(item, Mapping)}
    if set(records) != set(_SOURCE_PATHS):
        raise ValueError("diagnostics source inventory drift")
    for relative in _SOURCE_PATHS:
        content = _git(root, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(content, bytes)
        if (
            records[relative].get("git_blob")
            != _git(root, "rev-parse", f"{commit}:{relative}")
            or records[relative].get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"diagnostics source object drift: {relative}")


def _unit(value: object) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (5,) or not np.all(np.isfinite(vector)):
        raise ValueError("candidate contrast vectors must be finite rank-five vectors")
    norm = float(np.linalg.norm(vector))
    return None if norm <= 0.0 else vector / norm


def _mean_vectors(vectors: Iterable[np.ndarray]) -> np.ndarray | None:
    values = tuple(vectors)
    if not values:
        return None
    result = np.mean(np.stack(values), axis=0)
    return result if np.all(np.isfinite(result)) else None


def _hierarchical_vector(
    rows: Sequence[Mapping[str, Any]], *, average_bases_within_target: bool
) -> np.ndarray | None:
    """Equal-weight target -> response -> family mean of unit directions."""

    by_target: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        vector = _unit(row["candidate_contrast_vector"])
        if vector is not None:
            by_target[
                (
                    str(row["base_question_id"]),
                    str(row["response_id"]),
                    str(row["case_id"]),
                )
            ].append(vector)
    target_values: dict[tuple[str, str, str], np.ndarray] = {}
    for key, vectors in by_target.items():
        value = _mean_vectors(vectors if average_bases_within_target else vectors[:1])
        if value is not None:
            target_values[key] = value
    by_response: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for (family, response, _), value in target_values.items():
        by_response[(family, response)].append(value)
    response_values = {
        key: value
        for key, values in by_response.items()
        if (value := _mean_vectors(values)) is not None
    }
    by_family: dict[str, list[np.ndarray]] = defaultdict(list)
    for (family, _), value in response_values.items():
        by_family[family].append(value)
    family_values = [
        value
        for values in by_family.values()
        if (value := _mean_vectors(values)) is not None
    ]
    return _mean_vectors(family_values)


def _hierarchical_rank_mass(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Equal-weight absolute unit-vector mass across target/response/family."""

    by_target: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        vector = _unit(row["candidate_contrast_vector"])
        if vector is not None:
            by_target[
                (
                    str(row["base_question_id"]),
                    str(row["response_id"]),
                    str(row["case_id"]),
                )
            ].append(np.abs(vector))
    target_values = {
        key: value
        for key, values in by_target.items()
        if (value := _mean_vectors(values)) is not None
    }
    by_response: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for (family, response, _), value in target_values.items():
        by_response[(family, response)].append(value)
    response_values = {
        key: value
        for key, values in by_response.items()
        if (value := _mean_vectors(values)) is not None
    }
    by_family: dict[str, list[np.ndarray]] = defaultdict(list)
    for (family, _), value in response_values.items():
        by_family[family].append(value)
    family_values = [
        value
        for values in by_family.values()
        if (value := _mean_vectors(values)) is not None
    ]
    result = _mean_vectors(family_values)
    return np.zeros(5, dtype=np.float64) if result is None else result


def _quantile(values: Sequence[float], q: float) -> float | None:
    return None if not values else float(np.quantile(values, q, method="linear"))


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": None if not values else float(mean(values)),
        "median": None if not values else float(median(values)),
        "p10": _quantile(values, 0.10),
        "p90": _quantile(values, 0.90),
        "minimum": None if not values else float(min(values)),
        "maximum": None if not values else float(max(values)),
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    total = float(sum(values))
    probabilities = (
        [0.0 for _ in values] if total <= 0.0 else [float(v / total) for v in values]
    )
    positive = [value for value in probabilities if value > 0.0]
    entropy = -sum(value * math.log(value) for value in positive)
    normalized = 0.0 if len(values) <= 1 else entropy / math.log(len(values))
    return {
        "probabilities": probabilities,
        "normalized_entropy": float(normalized),
        "maximum_share": float(max(probabilities, default=0.0)),
        "effective_count": float(math.exp(entropy)),
    }


def _candidate_rows(
    rows: Sequence[Mapping[str, Any]], partition: str
) -> list[Mapping[str, Any]]:
    if partition not in PARTITIONS:
        raise ValueError(
            "diagnostics partition is outside the generation/selection firewall"
        )
    return [
        row
        for row in rows
        if row["family_partition"] == partition
        and bool(row["candidate_profile_available"])
        and bool(row["c2_w64_assigned"])
        and _unit(row["candidate_contrast_vector"]) is not None
    ]


def _basis_consistency(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, float], dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["signed_basis_index"])].append(row)
    values: dict[int, float] = {}
    ineligible = 0
    for basis, basis_rows in grouped.items():
        if (
            len({str(row["case_id"]) for row in basis_rows}) < 3
            or len({str(row["response_id"]) for row in basis_rows}) < 2
            or len({str(row["base_question_id"]) for row in basis_rows}) < 2
        ):
            ineligible += 1
            continue
        vector = _hierarchical_vector(basis_rows, average_bases_within_target=False)
        if vector is not None:
            values[basis] = float(np.linalg.norm(vector))
    summary = _summary(list(values.values()))
    summary["eligible_recurrent_basis_count"] = len(values)
    summary["ineligible_recurrence_basis_count"] = ineligible
    return values, summary


def _separation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_target: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[
            (str(row["base_question_id"]), str(row["response_id"]), str(row["case_id"]))
        ].append(row)
    target_values: dict[tuple[str, str, str], tuple[float, float, float, int, int]] = {}
    for key, target_rows in by_target.items():
        same: list[float] = []
        different: list[float] = []
        units = [
            (_unit(row["candidate_contrast_vector"]), int(row["c2_w64_cluster_id"]))
            for row in target_rows
        ]
        for left in range(len(units)):
            for right in range(left + 1, len(units)):
                cosine = float(np.dot(units[left][0], units[right][0]))  # type: ignore[arg-type]
                (same if units[left][1] == units[right][1] else different).append(
                    cosine
                )
        if same and different:
            target_values[key] = (
                float(mean(same)),
                float(mean(different)),
                float(mean(same) - mean(different)),
                len(same),
                len(different),
            )
    by_response: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(
        list
    )
    for (family, response, _), values in target_values.items():
        by_response[(family, response)].append(values[:3])
    response_values = {
        key: tuple(float(mean(item[i] for item in values)) for i in range(3))
        for key, values in by_response.items()
    }
    by_family: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (family, _), values in response_values.items():
        by_family[family].append(values)
    family_values = {
        family: tuple(float(mean(item[i] for item in values)) for i in range(3))
        for family, values in by_family.items()
    }
    family_margins = {
        family: value[2] for family, value in sorted(family_values.items())
    }
    bootstrap: dict[str, Any]
    if family_margins:
        ordered = np.asarray(list(family_margins.values()), dtype=np.float64)
        seed = int.from_bytes(
            hashlib.sha256(
                f"{DIAGNOSTICS_SCHEMA_VERSION}\0selection-separation-bootstrap-v1".encode()
            ).digest()[:8],
            "big",
        )
        generator = np.random.default_rng(seed)
        indices = generator.integers(
            0,
            len(ordered),
            size=(SELECTION_BOOTSTRAP_REPLICATES, len(ordered)),
        )
        replicates = ordered[indices].mean(axis=1)
        bootstrap = {
            "replicate_count": SELECTION_BOOTSTRAP_REPLICATES,
            "ci_95_lower": float(np.quantile(replicates, 0.025, method="linear")),
            "ci_95_upper": float(np.quantile(replicates, 0.975, method="linear")),
        }
    else:
        bootstrap = {
            "replicate_count": 0,
            "ci_95_lower": None,
            "ci_95_upper": None,
        }
    return {
        "scoreable_target_count": len(target_values),
        "scoreable_response_count": len(response_values),
        "scoreable_family_count": len(family_values),
        "same_pair_count": sum(value[3] for value in target_values.values()),
        "different_pair_count": sum(value[4] for value in target_values.values()),
        "same_cluster_cosine": None
        if not family_values
        else float(mean(value[0] for value in family_values.values())),
        "different_cluster_cosine": None
        if not family_values
        else float(mean(value[1] for value in family_values.values())),
        "same_minus_different": None
        if not family_values
        else float(mean(value[2] for value in family_values.values())),
        "positive_family_count": sum(value > 0.0 for value in family_margins.values()),
        "per_family_same_minus_different": family_margins,
        "family_block_bootstrap": bootstrap,
    }


def _partition_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    basis_values, basis_summary = _basis_consistency(rows)
    by_cluster: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[int(row["c2_w64_cluster_id"])].append(row)
    clusters: list[dict[str, Any]] = []
    for cluster, cluster_rows in sorted(by_cluster.items()):
        vector = _hierarchical_vector(cluster_rows, average_bases_within_target=True)
        resultant = None if vector is None else float(np.linalg.norm(vector))
        rank_mass = _hierarchical_rank_mass(cluster_rows)
        phases = [0.0] * 7
        seen_targets: set[str] = set()
        for row in cluster_rows:
            case_id = str(row["case_id"])
            if case_id not in seen_targets:
                phases[int(row["phase_bin"])] += float(
                    row["partition_hierarchical_weight"]
                )
                seen_targets.add(case_id)
        bases = {int(row["signed_basis_index"]) for row in cluster_rows}
        stable_rows = [
            row
            for row in cluster_rows
            if basis_values.get(int(row["signed_basis_index"]), -1.0)
            >= MIN_MEDIAN_BASIS_CONSISTENCY
        ]
        stable_vector = _hierarchical_vector(
            stable_rows, average_bases_within_target=True
        )
        stable_resultant = (
            None if stable_vector is None else float(np.linalg.norm(stable_vector))
        )
        stable = sum(
            basis_values.get(basis, -1.0) >= MIN_MEDIAN_BASIS_CONSISTENCY
            for basis in bases
        )
        clusters.append(
            {
                "cluster_id": cluster,
                "basis_count": len(bases),
                "stable_basis_count": stable,
                "stable_target_count": len(
                    {str(row["case_id"]) for row in stable_rows}
                ),
                "stable_response_count": len(
                    {str(row["response_id"]) for row in stable_rows}
                ),
                "stable_family_count": len(
                    {str(row["base_question_id"]) for row in stable_rows}
                ),
                "stable_direction_resultant_norm": stable_resultant,
                "stable_heterogeneity": (
                    None if stable_resultant is None else float(1.0 - stable_resultant)
                ),
                "target_count": len({str(row["case_id"]) for row in cluster_rows}),
                "response_count": len(
                    {str(row["response_id"]) for row in cluster_rows}
                ),
                "family_count": len(
                    {str(row["base_question_id"]) for row in cluster_rows}
                ),
                "candidate_row_count": len(cluster_rows),
                "direction_resultant_norm": resultant,
                "heterogeneity": None if resultant is None else float(1.0 - resultant),
                "rank_absolute_mass": _distribution(rank_mass.tolist()),
                "phase_target_mass": _distribution(phases),
            }
        )
    rank_mass = _hierarchical_rank_mass(rows)
    return {
        "candidate_row_count": len(rows),
        "basis_count": len({int(row["signed_basis_index"]) for row in rows}),
        "target_count": len({str(row["case_id"]) for row in rows}),
        "response_count": len({str(row["response_id"]) for row in rows}),
        "family_count": len({str(row["base_question_id"]) for row in rows}),
        "basis_direction_consistency": basis_summary,
        "within_vs_between_w64": _separation(rows),
        "rank_absolute_mass": _distribution(rank_mass.tolist()),
        "clusters": clusters,
    }


def compute_candidate_multiplex_diagnostics(
    assessment: LoadedCandidateMultiplexAssessment,
) -> dict[str, Any]:
    """Compute deterministic generation/selection metrics and the fit-eligibility gate."""

    rows = assessment.target_basis_assessment.to_pylist()
    partition_metrics = {
        partition: _partition_metrics(_candidate_rows(rows, partition))
        for partition in PARTITIONS
    }
    generation = partition_metrics["generation"]
    selection = partition_metrics["selection_scoring"]
    splittable = [
        cluster
        for cluster in generation["clusters"]
        if cluster["stable_basis_count"] >= MIN_STABLE_BASES_PER_SPLITTABLE_PARENT
        and cluster["stable_target_count"]
        >= MIN_GENERATION_TARGETS_PER_SPLITTABLE_PARENT
        and cluster["stable_family_count"]
        >= MIN_GENERATION_FAMILIES_PER_SPLITTABLE_PARENT
        and cluster["stable_heterogeneity"] is not None
        and cluster["stable_heterogeneity"] >= MIN_PARENT_HETEROGENEITY
    ]
    median_consistency = generation["basis_direction_consistency"]["median"]
    selection_separation = selection["within_vs_between_w64"]["same_minus_different"]
    selection_report = selection["within_vs_between_w64"]
    selection_bootstrap_lower = selection_report["family_block_bootstrap"][
        "ci_95_lower"
    ]
    conditions = [
        {
            "name": "generation_median_basis_consistency",
            "value": median_consistency,
            "requirement": f">= {MIN_MEDIAN_BASIS_CONSISTENCY}",
            "satisfied": median_consistency is not None
            and median_consistency >= MIN_MEDIAN_BASIS_CONSISTENCY,
        },
        {
            "name": "selection_w64_same_minus_different_separation",
            "value": selection_separation,
            "requirement": f"> {MIN_SELECTION_SEPARATION}",
            "satisfied": selection_separation is not None
            and selection_separation > MIN_SELECTION_SEPARATION,
        },
        {
            "name": "selection_all_families_scoreable",
            "value": selection_report["scoreable_family_count"],
            "requirement": f"== {REQUIRED_SELECTION_FAMILY_COUNT}",
            "satisfied": selection_report["scoreable_family_count"]
            == REQUIRED_SELECTION_FAMILY_COUNT,
        },
        {
            "name": "selection_positive_family_count",
            "value": selection_report["positive_family_count"],
            "requirement": f">= {REQUIRED_POSITIVE_SELECTION_FAMILY_COUNT} of 8",
            "satisfied": selection_report["positive_family_count"]
            >= REQUIRED_POSITIVE_SELECTION_FAMILY_COUNT,
        },
        {
            "name": "selection_family_bootstrap_lower_bound",
            "value": selection_bootstrap_lower,
            "requirement": "> 0",
            "satisfied": selection_bootstrap_lower is not None
            and selection_bootstrap_lower > 0.0,
        },
        {
            "name": "theoretically_splittable_parent_count",
            "value": len(splittable),
            "requirement": ">= 1",
            "satisfied": bool(splittable),
        },
    ]
    eligible = all(condition["satisfied"] for condition in conditions)
    return {
        "policy": {
            "fit_partition": "generation",
            "decision_partition": "selection_scoring",
            "audit_rows_used": False,
            "dense_overlap_used_for_fitting_or_decision": False,
            "primary_state": "c2_w64",
            "candidate_measurement_scope": "target_basis_signed_sum",
        },
        "coverage": {
            "source_overlap_provenance_only": assessment.manifest["overlap"],
            "by_family_partition": {
                partition: assessment.manifest["coverage_metrics"][
                    "by_family_partition"
                ][partition]
                for partition in PARTITIONS
            },
            "audit_partition_excluded": True,
        },
        "partitions": partition_metrics,
        "decision": {
            "status": (
                "eligible_for_constrained_refinement_fit"
                if eligible
                else "no_justified_candidate_refinement"
            ),
            "primary_state": "c2_w64",
            "alternative_state": None,
            "eligible_for_refinement_fit": eligible,
            "theoretically_splittable_parent_ids": [
                cluster["cluster_id"] for cluster in splittable
            ],
            "conditions": conditions,
            "promotion_claimed": False,
            "next_if_eligible": (
                "freeze_generation_only_within_w64_fit_then_evaluate_on_selection"
            ),
        },
    }


def _source_record(assessment: LoadedCandidateMultiplexAssessment) -> dict[str, Any]:
    path = assessment.root / "manifest.json"
    return {
        "path": str(assessment.root),
        "manifest_path": str(path),
        "manifest_sha256": str(assessment.manifest["manifest_sha256"]),
        "manifest_file_sha256": file_sha256(path),
        "schema_version": str(assessment.manifest["schema_version"]),
    }


def build_candidate_multiplex_diagnostics(
    *, assessment_root: Path, output_root: Path, repo_root: Path
) -> dict[str, Any]:
    """Deep-validate, derive, and atomically publish a no-overwrite artifact."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"refusing to replace diagnostics artifact: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    revision = _collect_revision(repo_root)
    assessment = load_candidate_multiplex_assessment(
        assessment_root, verify_sources=True
    )
    report = compute_candidate_multiplex_diagnostics(assessment)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        (temporary / REPORT_FILE).write_bytes(report_bytes)
        core: dict[str, Any] = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "purpose": "label_free_candidate_multiplex_diagnostics_and_refinement_eligibility",
            "source_assessment": _source_record(assessment),
            "producing_revision": revision,
            "decision": report["decision"],
            "firewall": {
                "labels_used": False,
                "outcomes_inspected": False,
                "confirmatory_holdout_opened": False,
                "model_calls_made": False,
                "audit_rows_used": False,
                "dense_overlap_used_for_fitting_or_decision": False,
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
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        reloaded = load_candidate_multiplex_assessment(
            assessment_root, verify_sources=True
        )
        if compute_candidate_multiplex_diagnostics(reloaded) != report:
            raise ValueError(
                "diagnostics source re-derivation changed before publication"
            )
        if _collect_revision(repo_root) != revision:
            raise ValueError(
                "diagnostics producing revision changed during construction"
            )
        _publish_directory_no_replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_candidate_multiplex_diagnostics(
    root: Path, *, verify_source: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest = load_json_object(root / "manifest.json")
    core = dict(manifest)
    if core.pop("manifest_sha256", None) != canonical_sha256(core):
        raise ValueError("diagnostics manifest self-hash mismatch")
    if manifest.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("unsupported diagnostics schema")
    firewall = manifest.get("firewall")
    if not isinstance(firewall, Mapping) or any(
        firewall.get(field) is not False
        for field in (
            "labels_used",
            "outcomes_inspected",
            "confirmatory_holdout_opened",
            "model_calls_made",
            "audit_rows_used",
            "dense_overlap_used_for_fitting_or_decision",
        )
    ):
        raise ValueError("diagnostics firewall drift")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("diagnostics file inventory drift")
    record = records[0]
    path = root / REPORT_FILE
    if (
        not isinstance(record, Mapping)
        or record.get("path") != REPORT_FILE
        or path.stat().st_size != int(record.get("size_bytes", -1))
        or file_sha256(path) != record.get("sha256")
    ):
        raise ValueError("diagnostics report file drift")
    report = load_json_object(path)
    if manifest.get("decision") != report.get("decision"):
        raise ValueError("diagnostics decision summary drift")
    revision = manifest.get("producing_revision")
    if not isinstance(revision, Mapping):
        raise TypeError("diagnostics producing revision is invalid")
    _validate_revision(revision)
    if verify_source:
        source = manifest.get("source_assessment")
        if not isinstance(source, Mapping):
            raise TypeError("diagnostics source assessment is invalid")
        assessment_root = Path(str(source.get("path")))
        assessment = load_candidate_multiplex_assessment(
            assessment_root, verify_sources=True
        )
        manifest_path = assessment_root / "manifest.json"
        if (
            assessment.manifest.get("manifest_sha256") != source.get("manifest_sha256")
            or file_sha256(manifest_path) != source.get("manifest_file_sha256")
            or compute_candidate_multiplex_diagnostics(assessment) != report
        ):
            raise ValueError("diagnostics differs from bound assessment derivation")
    return manifest, report
