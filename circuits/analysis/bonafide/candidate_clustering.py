"""Generation-only clustering core for the frozen C2 candidate-aware analysis.

The functions in this module stop at deterministic, label-free cluster fitting.
They deliberately do not persist results, score held-out families, construct
directional nulls, or inspect generated descriptions.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix, triu
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence

import circuits.analysis.bonafide.clustering_evaluation as clustering_evaluation_module
from circuits.analysis.bonafide.candidate_profiles import (
    BASIS_INDEX_SCHEMA,
    CANDIDATE_CLUSTER_INPUT_SCHEMA,
    CANDIDATE_CLUSTER_PARTITION_SCHEMA,
    CANDIDATE_PROFILE_SCHEMA,
    FROZEN_ARTIFACT_PAYLOAD_SET_SHA256,
    FROZEN_C2_REPORT_SHA256,
    FROZEN_CLUSTERING_EVALUATION_REPORT_SCHEMA,
    FROZEN_CLUSTERING_EVALUATION_SHA256,
    FROZEN_PLAN_CANONICAL_SHA256,
    FROZEN_PLAN_FILE_SHA256,
    FROZEN_PROTOCOL_SHA256,
    FROZEN_SALVAGE_REPORT_SHA256,
    FROZEN_SELECTION_SHA256,
    PARTITION_CAPACITIES,
    PARTITION_ORDER,
    TARGET_SCHEMA,
    WIDTH_PROFILE_SCHEMA,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.analysis.bonafide.clustering import (
    PairEvidence,
    SparseSpectralResult,
    TargetProfileBlock,
    accumulate_pair_evidence,
    knn_affinity,
    mean_similarity_matrix,
    sparse_spectral_cluster,
)
from circuits.analysis.bonafide.clustering_evaluation import (
    assignment_ari,
    cluster_size_metrics,
    sparse_graph_partition_metrics,
)

CLUSTER_COUNTS = (32, 64, 96)
RANDOM_SEEDS = (17, 29, 43)
NEIGHBORS = 32
MIN_BASIS_TARGETS = 3
MIN_BASIS_RESPONSES = 2
MIN_BASIS_FAMILIES = 2
MIN_PAIR_TARGETS = 2
MIN_PAIR_RESPONSES = 2
MIN_PAIR_FAMILIES = 2
MIN_ASSIGNMENT_FRACTION = 0.95

_REQUIRED_FILES = {
    "basis-index.parquet": BASIS_INDEX_SCHEMA,
    "targets.parquet": TARGET_SCHEMA,
    "width-profiles.parquet": WIDTH_PROFILE_SCHEMA,
    "candidate-profiles.parquet": CANDIDATE_PROFILE_SCHEMA,
}
_BASIS_IDENTITY_FIELDS = (
    "model_id",
    "model_revision",
    "layer",
    "neuron_index",
    "polarity",
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_frozen_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if manifest.get("purpose") != (
        "frozen_inputs_only_no_cluster_fit_or_description_generation"
    ):
        raise ValueError("candidate-cluster input purpose drift")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("candidate-cluster frozen inputs are invalid")
    expected_records = {
        "selection": {"sha256": FROZEN_SELECTION_SHA256},
        "candidate_union_plan": {
            "file_sha256": FROZEN_PLAN_FILE_SHA256,
            "canonical_sha256": FROZEN_PLAN_CANONICAL_SHA256,
        },
        "audited_c2_report": {"sha256": FROZEN_C2_REPORT_SHA256},
        "posthoc_salvage_report": {"sha256": FROZEN_SALVAGE_REPORT_SHA256},
    }
    for name, expected in expected_records.items():
        record = inputs.get(name)
        if not isinstance(record, Mapping) or any(
            record.get(field) != value for field, value in expected.items()
        ):
            raise ValueError(f"candidate-cluster frozen input provenance drift: {name}")
    if inputs.get("artifact_payload_set_sha256") != FROZEN_ARTIFACT_PAYLOAD_SET_SHA256:
        raise ValueError("candidate-cluster artifact payload provenance drift")
    for root_field in ("width1_root", "candidate_union_root"):
        if not isinstance(inputs.get(root_field), str) or not inputs[root_field]:
            raise ValueError(f"candidate-cluster source root is invalid: {root_field}")

    expected_flags = {
        "outcomes_inspected": False,
        "model_calls_made": False,
        "cluster_fit_performed": False,
        "confirmatory_holdout_opened": False,
    }
    for field, expected in expected_flags.items():
        if manifest.get(field) is not expected:
            raise ValueError(f"candidate-cluster publication flag drift: {field}")

    revision = manifest.get("code_revision")
    if not isinstance(revision, Mapping):
        raise TypeError("candidate-cluster code revision is invalid")
    if revision.get("git_dirty") is not False:
        raise ValueError("candidate-cluster inputs require a clean source revision")
    if revision.get("git_status_sha256") != _EMPTY_SHA256:
        raise ValueError("candidate-cluster source status was not empty")
    for field, length in (
        ("git_commit", 40),
        ("git_tree", 40),
        ("source_tree_sha256", 64),
    ):
        if not _is_lower_hex(revision.get(field), length):
            raise ValueError(
                f"candidate-cluster code revision field is invalid: {field}"
            )
    source_records = revision.get("files")
    if not isinstance(source_records, list):
        raise TypeError("candidate-cluster bound source files are invalid")
    source_by_path: dict[str, str] = {}
    for record in source_records:
        if not isinstance(record, Mapping):
            raise TypeError("candidate-cluster bound source file is invalid")
        path = str(record.get("path"))
        sha256 = record.get("sha256")
        if path in source_by_path or not _is_lower_hex(sha256, 64):
            raise ValueError("candidate-cluster bound source file record is invalid")
        source_by_path[path] = str(sha256)
    expected_source = {
        "docs/CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md": (
            FROZEN_PROTOCOL_SHA256
        ),
        "circuits/analysis/bonafide/clustering_evaluation.py": (
            FROZEN_CLUSTERING_EVALUATION_SHA256
        ),
    }
    if any(
        source_by_path.get(path) != value for path, value in expected_source.items()
    ):
        raise ValueError("candidate-cluster bound source provenance drift")

    cohort = manifest.get("cohort")
    if not isinstance(cohort, Mapping):
        raise TypeError("candidate-cluster cohort summary is invalid")
    expected_counts: dict[str, object] = {
        "target_count": 245,
        "response_count": 35,
        "family_count": 34,
        "phase_bin_counts": {str(index): 35 for index in range(7)},
        "candidate_width_counts": {"5": 235, "6": 10},
    }
    for field, expected in expected_counts.items():
        if cohort.get(field) != expected:
            raise ValueError(f"candidate-cluster cohort boundary drift: {field}")
    invariance = cohort.get("candidate_activation_invariance")
    if not isinstance(invariance, Mapping) or (
        invariance.get("rtol") != 1e-6
        or invariance.get("atol") != 1e-7
        or invariance.get("violation_count") != 0
        or not isinstance(invariance.get("comparison_count"), int)
        or int(invariance["comparison_count"]) <= 0
    ):
        raise ValueError("candidate activation invariance contract drift")
    for field in ("max_abs_deviation", "max_relative_deviation"):
        value = invariance.get(field)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError("candidate activation invariance diagnostic is invalid")


@dataclass(frozen=True)
class CandidateClusterInputBundle:
    """Validated rows and generation-only blocks from one immutable bundle."""

    root: Path
    manifest: Mapping[str, Any]
    basis_count: int
    basis_rows: tuple[Mapping[str, Any], ...]
    target_rows: tuple[Mapping[str, Any], ...]
    family_partitions: Mapping[str, Any]
    generation_case_ids: tuple[str, ...]
    width_blocks: tuple[TargetProfileBlock, ...]
    candidate_blocks: tuple[TargetProfileBlock, ...]
    candidate_support_blocks: tuple[TargetProfileBlock, ...]


@dataclass(frozen=True)
class CandidateViewEvidence:
    """Matched directional evidence and similarities before kNN truncation."""

    width: PairEvidence
    candidate: PairEvidence
    common_eligible_mask: NDArray[np.bool_]
    width_similarity: csr_matrix
    candidate_similarity: csr_matrix
    fusion_similarity: csr_matrix
    support_similarity: csr_matrix


@dataclass(frozen=True)
class SeedFit:
    seed: int
    result: SparseSpectralResult | None
    valid: bool
    assignment_fraction: float
    error: str | None


@dataclass(frozen=True)
class ResolutionFit:
    view: str
    n_clusters: int
    affinity: csr_matrix
    seeds: Mapping[int, SeedFit]
    valid: bool
    medoid_seed: int | None
    pairwise_seed_ari: Mapping[tuple[int, int], float]
    mean_seed_ari: float | None
    minimum_seed_ari: float | None
    size_metrics: Mapping[str, Any] | None
    graph_metrics: Mapping[str, Any] | None

    @property
    def labels(self) -> NDArray[np.int64] | None:
        if self.medoid_seed is None:
            return None
        result = self.seeds[self.medoid_seed].result
        return None if result is None else result.labels


@dataclass(frozen=True)
class GenerationClusterFit:
    """Complete initial W/C/F grid plus S at the common selected count."""

    evidence: CandidateViewEvidence
    directional: Mapping[str, Mapping[int, ResolutionFit]]
    chosen_cluster_count: int | None
    support: ResolutionFit | None


def _validated_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = load_json_object(manifest_path)
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core):
        raise ValueError("candidate-cluster input manifest hash mismatch")
    if manifest.get("schema_version") != CANDIDATE_CLUSTER_INPUT_SCHEMA:
        raise ValueError("unsupported candidate-cluster input schema")
    _validate_frozen_manifest_contract(manifest)
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping) or (
        set(protocol) != {"path", "sha256"}
        or protocol.get("sha256") != FROZEN_PROTOCOL_SHA256
    ):
        raise ValueError("candidate-cluster protocol provenance drift")
    protocol_path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / ("CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md")
    )
    if file_sha256(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("local candidate-clustering protocol hash drift")
    structural = manifest.get("structural_evaluation_contract")
    if not isinstance(structural, Mapping) or (
        structural.get("sha256") != FROZEN_CLUSTERING_EVALUATION_SHA256
        or structural.get("report_schema") != FROZEN_CLUSTERING_EVALUATION_REPORT_SCHEMA
    ):
        raise ValueError("candidate-cluster structural metric provenance drift")
    evaluation_path = Path(str(clustering_evaluation_module.__file__)).resolve()
    if file_sha256(evaluation_path) != FROZEN_CLUSTERING_EVALUATION_SHA256:
        raise ValueError("local structural metric implementation hash drift")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TypeError("candidate-cluster input file inventory is invalid")
    by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("candidate-cluster input file record is invalid")
        name = str(record.get("path"))
        if Path(name).name != name or name in by_name:
            raise ValueError("candidate-cluster input file path is unsafe or duplicate")
        by_name[name] = record
    expected_names = set(_REQUIRED_FILES) | {"family-partitions.json"}
    if set(by_name) != expected_names:
        raise ValueError("candidate-cluster input file inventory is incomplete")
    for name, record in by_name.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"candidate-cluster input file is missing: {name}")
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"candidate-cluster input file hash mismatch: {name}")
    return manifest


def _read_exact_parquet(path: Path, schema: pa.Schema) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"candidate-cluster parquet schema drift: {path.name}")
    return table.to_pylist()


def _validate_targets_and_weights(
    target_rows: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    by_case: dict[str, Mapping[str, Any]] = {}
    family_to_partition = partitions.get("family_to_partition")
    if not isinstance(family_to_partition, Mapping):
        raise TypeError("family-to-partition mapping is invalid")
    for row in target_rows:
        case_id = str(row["case_id"])
        if case_id in by_case:
            raise ValueError("candidate-cluster target case IDs are not unique")
        family = str(row["base_question_id"])
        if row["family_partition"] != family_to_partition.get(family):
            raise ValueError("target partition disagrees with family partition")
        weight = float(row["partition_hierarchical_weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("target hierarchical weight is invalid")
        by_case[case_id] = row

    if len(by_case) != 245:
        raise ValueError("candidate-cluster target count drift")
    response_families: dict[str, set[str]] = {}
    phases_by_response: dict[str, list[int]] = {}
    for row in target_rows:
        response = str(row["response_id"])
        family = str(row["base_question_id"])
        response_families.setdefault(response, set()).add(family)
        phases_by_response.setdefault(response, []).append(int(row["phase_bin"]))
    if len(response_families) != 35:
        raise ValueError("candidate-cluster response count drift")
    if len({str(row["base_question_id"]) for row in target_rows}) != 34:
        raise ValueError("candidate-cluster family count drift")
    if any(len(families) != 1 for families in response_families.values()):
        raise ValueError("response identity is owned by multiple families")
    if any(sorted(phases) != list(range(7)) for phases in phases_by_response.values()):
        raise ValueError("response does not own exactly one target per phase bin")
    if Counter(int(row["phase_bin"]) for row in target_rows) != Counter(
        {phase: 35 for phase in range(7)}
    ):
        raise ValueError("candidate-cluster phase-bin coverage drift")
    if Counter(int(row["candidate_count"]) for row in target_rows) != Counter(
        {5: 235, 6: 10}
    ):
        raise ValueError("candidate-cluster candidate-width coverage drift")

    for partition in ("generation", "selection_scoring", "audit"):
        rows = [row for row in target_rows if row["family_partition"] == partition]
        families = {str(row["base_question_id"]) for row in rows}
        responses_by_family: dict[str, set[str]] = {
            family: set() for family in families
        }
        targets_by_response: dict[str, set[str]] = {}
        for row in rows:
            family = str(row["base_question_id"])
            response = str(row["response_id"])
            responses_by_family[family].add(response)
            targets_by_response.setdefault(response, set()).add(str(row["case_id"]))
        for row in rows:
            family = str(row["base_question_id"])
            response = str(row["response_id"])
            expected = (
                1.0
                / len(families)
                / len(responses_by_family[family])
                / len(targets_by_response[response])
            )
            if not np.isclose(
                float(row["partition_hierarchical_weight"]),
                expected,
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError("target hierarchical weight formula drift")

    violation_count = sum(
        int(row["candidate_activation_invariance_violation_count"])
        for row in target_rows
    )
    comparison_count = sum(
        int(row["candidate_activation_invariance_comparison_count"])
        for row in target_rows
    )
    max_abs = max(
        float(row["candidate_activation_invariance_max_abs_deviation"])
        for row in target_rows
    )
    max_relative = max(
        float(row["candidate_activation_invariance_max_relative_deviation"])
        for row in target_rows
    )
    if violation_count != 0 or comparison_count <= 0:
        raise ValueError("per-target candidate activation invariance failed")
    for value in (max_abs, max_relative):
        if not math.isfinite(value) or value < 0:
            raise ValueError("per-target candidate activation diagnostic is invalid")
    return by_case


def _validate_profile_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    all_case_ids: set[str],
    basis_rows: Sequence[Mapping[str, Any]],
    candidate: bool,
) -> None:
    if {str(row["case_id"]) for row in rows} != all_case_ids:
        raise ValueError("profile rows do not cover the frozen target cohort")
    seen: set[tuple[str, int]] = set()
    width_by_case: dict[str, int] = {}
    for row in rows:
        index = int(row["signed_basis_index"])
        if not 0 <= index < len(basis_rows):
            raise ValueError("profile signed-basis index is out of range")
        identity = basis_rows[index]
        if any(row[field] != identity[field] for field in _BASIS_IDENTITY_FIELDS):
            raise ValueError("profile signed-basis identity disagrees with basis index")
        key = (str(row["case_id"]), index)
        if key in seen:
            raise ValueError("profile contains a duplicate target/basis row")
        seen.add(key)
        occurrence_count = row.get("occurrence_count")
        if not isinstance(occurrence_count, int) or occurrence_count <= 0:
            raise ValueError("profile occurrence count is invalid")
        if candidate:
            values = np.asarray(row["candidate_contrast_profile"], dtype=np.float64)
            recorded_norm = float(row["candidate_profile_l2_norm"])
            if values.shape != (5,) or not np.all(np.isfinite(values)):
                raise ValueError("candidate contrast profile is not finite width five")
            if not math.isfinite(recorded_norm) or recorded_norm < 0:
                raise ValueError("candidate profile norm is invalid")
            if not np.isclose(
                recorded_norm, np.linalg.norm(values), rtol=1e-12, atol=1e-12
            ):
                raise ValueError("candidate profile norm drift")
        else:
            raw_values = row["attribution_profile"]
            support_raw = row["attribution_support"]
            if not isinstance(raw_values, list) or not isinstance(support_raw, list):
                raise TypeError("width profile values/support are not lists")
            if not all(isinstance(value, bool) for value in support_raw):
                raise TypeError("width profile support is not boolean")
            support = np.asarray(support_raw, dtype=np.bool_)
            if not raw_values or len(raw_values) != len(support):
                raise ValueError("width profile shape/support is invalid")
            for value, is_supported in zip(raw_values, support_raw, strict=True):
                if is_supported:
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError("supported width profile value is not finite")
                elif value is not None:
                    raise ValueError("unsupported width profile value is not null")
            signed_attribution = float(row["signed_attribution"])
            if not math.isfinite(signed_attribution):
                raise ValueError("width signed attribution is not finite")
            case_id = str(row["case_id"])
            previous = width_by_case.setdefault(case_id, len(raw_values))
            if previous != len(raw_values):
                raise ValueError("width profile coordinate width varies within target")


def _validate_partition_manifest(
    root: Path, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    partitions = load_json_object(root / "family-partitions.json")
    core = dict(partitions)
    recorded = core.pop("partitions_sha256", None)
    if recorded != canonical_sha256(core):
        raise ValueError("candidate-cluster family partition hash mismatch")
    if recorded != manifest.get("partition_manifest_sha256"):
        raise ValueError("candidate-cluster family partition provenance drift")
    frozen_header = {
        "schema_version": CANDIDATE_CLUSTER_PARTITION_SCHEMA,
        "algorithm": "sha256_salted_family_order_exact_slices_v1",
        "namespace_utf8": "candidate-aware-labelability-v1",
        "hash_preimage": "namespace_utf8_then_nul_byte_then_family_id_utf8",
        "ordering": "ascending_sha256_then_canonical_family_id",
        "slice_definition": (
            "first_18_generation_next_8_selection_scoring_final_8_audit"
        ),
        "capacities": dict(PARTITION_CAPACITIES),
        "outcome_fields_used": [],
    }
    if any(partitions.get(field) != value for field, value in frozen_header.items()):
        raise ValueError("candidate-cluster family partition algorithm drift")
    expected = dict(PARTITION_CAPACITIES)
    raw = partitions.get("partitions")
    if (
        not isinstance(raw, Mapping)
        or {role: len(raw.get(role, [])) for role in expected} != expected
    ):
        raise ValueError("candidate-cluster family partition sizes drifted")
    families = [str(value) for role in expected for value in raw[role]]
    if len(set(families)) != 34:
        raise ValueError("candidate-cluster family partitions overlap")
    salt = b"candidate-aware-labelability-v1\0"
    ordered = sorted(
        families,
        key=lambda family: (
            hashlib.sha256(salt + family.encode("utf-8")).hexdigest(),
            family,
        ),
    )
    slots = [
        role for role in PARTITION_ORDER for _ in range(PARTITION_CAPACITIES[role])
    ]
    expected_mapping = {family: slots[index] for index, family in enumerate(ordered)}
    expected_partitions = {
        role: sorted(
            family for family, assigned in expected_mapping.items() if assigned == role
        )
        for role in PARTITION_ORDER
    }
    if partitions.get("ordered_families") != ordered:
        raise ValueError("candidate-cluster family hash order drift")
    if partitions.get("family_to_partition") != dict(sorted(expected_mapping.items())):
        raise ValueError("candidate-cluster family assignment drift")
    if raw != expected_partitions:
        raise ValueError("candidate-cluster partition membership drift")
    return partitions


def _profile_blocks(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Mapping[str, Mapping[str, Any]],
    value_field: str,
    support_field: str | None,
    width: int | None,
) -> tuple[TargetProfileBlock, ...]:
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in targets:
            raise ValueError("profile row refers to an unknown generation target")
        by_case.setdefault(case_id, []).append(row)
    if set(by_case) != set(targets):
        raise ValueError("generation profile rows do not cover every target")

    blocks: list[TargetProfileBlock] = []
    for case_id in sorted(targets):
        target = targets[case_id]
        ordered = sorted(
            by_case[case_id], key=lambda row: int(row["signed_basis_index"])
        )
        indices = np.asarray(
            [int(row["signed_basis_index"]) for row in ordered], dtype=np.int64
        )
        if len(np.unique(indices)) != len(indices):
            raise ValueError("profile contains a duplicate target/basis row")
        raw_values = [list(row[value_field]) for row in ordered]
        profile_widths = {len(values) for values in raw_values}
        if len(profile_widths) != 1 or (
            width is not None and profile_widths != {width}
        ):
            raise ValueError("profile coordinate width is invalid")
        values = np.asarray(raw_values, dtype=np.float32)
        if support_field is None:
            support = np.ones(values.shape, dtype=np.bool_)
        else:
            support = np.asarray(
                [list(row[support_field]) for row in ordered], dtype=np.bool_
            )
            if support.shape != values.shape:
                raise ValueError("profile support shape is invalid")
        block = TargetProfileBlock(
            trace_unit_id=case_id,
            response_id=str(target["response_id"]),
            base_question_id=str(target["base_question_id"]),
            basis_indices=indices,
            values=values,
            support=support,
            fit_weight=float(target["partition_hierarchical_weight"]),
        )
        block.validate()
        blocks.append(block)
    return tuple(blocks)


def _support_blocks(
    candidate_blocks: Sequence[TargetProfileBlock],
) -> tuple[TargetProfileBlock, ...]:
    return tuple(
        TargetProfileBlock(
            trace_unit_id=block.trace_unit_id,
            response_id=block.response_id,
            base_question_id=block.base_question_id,
            basis_indices=block.basis_indices.copy(),
            values=np.ones((len(block.basis_indices), 1), dtype=np.float32),
            support=np.ones((len(block.basis_indices), 1), dtype=np.bool_),
            fit_weight=block.fit_weight,
        )
        for block in candidate_blocks
    )


def load_candidate_cluster_input_bundle(root: Path) -> CandidateClusterInputBundle:
    """Load and fail-closed validate an immutable candidate input bundle."""

    root = root.resolve()
    manifest = _validated_manifest(root)
    partitions = _validate_partition_manifest(root, manifest)
    basis_rows = _read_exact_parquet(root / "basis-index.parquet", BASIS_INDEX_SCHEMA)
    if [int(row["signed_basis_index"]) for row in basis_rows] != list(
        range(len(basis_rows))
    ):
        raise ValueError(
            "candidate-cluster basis index is not canonical and contiguous"
        )
    basis_count = len(basis_rows)
    if int(manifest.get("cohort", {}).get("signed_basis_count", -1)) != basis_count:
        raise ValueError("candidate-cluster manifest basis count drift")
    targets_rows = _read_exact_parquet(root / "targets.parquet", TARGET_SCHEMA)
    if len(targets_rows) != 245:
        raise ValueError("candidate-cluster bundle must contain exactly 245 targets")
    payload_records = [
        {
            "source_width1_artifact_id": row["source_width1_artifact_id"],
            "width1_payload_sha256": row["width1_payload_sha256"],
            "candidate_union_payload_sha256": row["candidate_union_payload_sha256"],
        }
        for row in targets_rows
    ]
    if canonical_sha256(payload_records) != FROZEN_ARTIFACT_PAYLOAD_SET_SHA256:
        raise ValueError("target artifact payload-set hash differs from frozen C2")
    all_targets = _validate_targets_and_weights(targets_rows, partitions)
    cohort = manifest["cohort"]
    invariance = cohort["candidate_activation_invariance"]
    target_invariance = {
        "comparison_count": sum(
            int(row["candidate_activation_invariance_comparison_count"])
            for row in targets_rows
        ),
        "violation_count": sum(
            int(row["candidate_activation_invariance_violation_count"])
            for row in targets_rows
        ),
        "max_abs_deviation": max(
            float(row["candidate_activation_invariance_max_abs_deviation"])
            for row in targets_rows
        ),
        "max_relative_deviation": max(
            float(row["candidate_activation_invariance_max_relative_deviation"])
            for row in targets_rows
        ),
    }
    if any(
        invariance.get(field) != value for field, value in target_invariance.items()
    ):
        raise ValueError("candidate activation invariance summary drift")
    generation_families = set(partitions["partitions"]["generation"])
    generation_rows = [
        row for row in targets_rows if str(row["family_partition"]) == "generation"
    ]
    if {str(row["base_question_id"]) for row in generation_rows} != generation_families:
        raise ValueError("generation target membership disagrees with family partition")
    targets = {str(row["case_id"]): row for row in generation_rows}

    width_rows = _read_exact_parquet(
        root / "width-profiles.parquet", WIDTH_PROFILE_SCHEMA
    )
    candidate_rows = _read_exact_parquet(
        root / "candidate-profiles.parquet", CANDIDATE_PROFILE_SCHEMA
    )
    if int(cohort.get("width_profile_row_count", -1)) != len(width_rows):
        raise ValueError("candidate-cluster width profile row count drift")
    if int(cohort.get("candidate_profile_row_count", -1)) != len(candidate_rows):
        raise ValueError("candidate-cluster candidate profile row count drift")
    _validate_profile_rows(
        width_rows,
        all_case_ids=set(all_targets),
        basis_rows=basis_rows,
        candidate=False,
    )
    _validate_profile_rows(
        candidate_rows,
        all_case_ids=set(all_targets),
        basis_rows=basis_rows,
        candidate=True,
    )
    shared_count = sum(
        bool(row["in_width_view"]) and bool(row["in_candidate_view"])
        for row in basis_rows
    )
    if int(cohort.get("shared_view_signed_basis_count", -1)) != shared_count:
        raise ValueError("candidate-cluster shared-view basis count drift")
    generation_ids = set(targets)
    width_blocks = _profile_blocks(
        [row for row in width_rows if str(row["case_id"]) in generation_ids],
        targets=targets,
        value_field="attribution_profile",
        support_field="attribution_support",
        width=None,
    )
    candidate_blocks = _profile_blocks(
        [row for row in candidate_rows if str(row["case_id"]) in generation_ids],
        targets=targets,
        value_field="candidate_contrast_profile",
        support_field=None,
        width=5,
    )
    return CandidateClusterInputBundle(
        root=root,
        manifest=manifest,
        basis_count=basis_count,
        basis_rows=tuple(basis_rows),
        target_rows=tuple(sorted(targets_rows, key=lambda row: str(row["case_id"]))),
        family_partitions=partitions,
        generation_case_ids=tuple(sorted(targets)),
        width_blocks=width_blocks,
        candidate_blocks=candidate_blocks,
        candidate_support_blocks=_support_blocks(candidate_blocks),
    )


def basis_eligibility(evidence: PairEvidence) -> NDArray[np.bool_]:
    """Apply the frozen target/response/family gates to directional diagonals."""

    evidence.validate()
    return (
        (np.asarray(evidence.overlap_count.diagonal()).ravel() >= MIN_BASIS_TARGETS)
        & (
            np.asarray(evidence.response_overlap_count.diagonal()).ravel()
            >= MIN_BASIS_RESPONSES
        )
        & (
            np.asarray(evidence.family_overlap_count.diagonal()).ravel()
            >= MIN_BASIS_FAMILIES
        )
    )


def empirical_positive_midranks(similarity: csr_matrix) -> csr_matrix:
    """Calibrate positive off-diagonal pairs by ascending midrank divided by n."""

    if similarity.shape[0] != similarity.shape[1] or (similarity - similarity.T).nnz:
        raise ValueError("similarity must be square and exactly symmetric")
    upper = triu(similarity, k=1, format="coo")
    positive = np.isfinite(upper.data) & (upper.data > 0)
    rows = upper.row[positive]
    columns = upper.col[positive]
    values = upper.data[positive]
    if not len(values):
        return csr_matrix(similarity.shape, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        # One-based ascending rank, averaged across an exact-value tie.
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    calibrated = ranks / len(values)
    upper_result = coo_matrix(
        (calibrated, (rows, columns)), shape=similarity.shape, dtype=np.float64
    ).tocsr()
    result = (upper_result + upper_result.T).tocsr()
    result.sort_indices()
    return result


def fuse_calibrated_similarities(
    width: csr_matrix, candidate: csr_matrix
) -> csr_matrix:
    """Average calibrated W/C values only on their recurring-pair intersection."""

    if width.shape != candidate.shape:
        raise ValueError("fusion view shapes disagree")
    width_rank = empirical_positive_midranks(width)
    candidate_rank = empirical_positive_midranks(candidate)
    shared = width_rank.copy()
    shared.data.fill(1.0)
    candidate_mask = candidate_rank.copy()
    candidate_mask.data.fill(1.0)
    shared = shared.multiply(candidate_mask)
    fused = ((width_rank + candidate_rank) * 0.5).multiply(shared).tocsr()
    fused.setdiag(0.0)
    fused.eliminate_zeros()
    fused.sort_indices()
    return fused


def weighted_support_jaccard(
    support_evidence: PairEvidence,
    *,
    eligible_mask: NDArray[np.bool_],
) -> csr_matrix:
    """Build recurrence-gated hierarchical weighted candidate-support Jaccard."""

    support_evidence.validate()
    if eligible_mask.shape != (support_evidence.basis_count,):
        raise ValueError("support Jaccard eligibility shape is invalid")
    co_support = support_evidence.support_weight_sum.tocoo()
    basis_weight = np.asarray(support_evidence.support_weight_sum.diagonal()).ravel()
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    overlap = support_evidence.overlap_count.tocsr()
    responses = support_evidence.response_overlap_count.tocsr()
    families = support_evidence.family_overlap_count.tocsr()
    for left, right, intersection in zip(
        co_support.row, co_support.col, co_support.data, strict=True
    ):
        if not eligible_mask[left] or not eligible_mask[right]:
            continue
        if (
            overlap[left, right] < MIN_PAIR_TARGETS
            or responses[left, right] < MIN_PAIR_RESPONSES
            or families[left, right] < MIN_PAIR_FAMILIES
        ):
            continue
        denominator = basis_weight[left] + basis_weight[right] - intersection
        if not math.isfinite(denominator) or denominator <= 0:
            continue
        value = float(intersection / denominator)
        if math.isfinite(value) and value > 0:
            rows.append(int(left))
            columns.append(int(right))
            values.append(value)
    upper = coo_matrix(
        (values, (rows, columns)),
        shape=(support_evidence.basis_count, support_evidence.basis_count),
        dtype=np.float64,
    ).tocsr()
    diagonal = csr_matrix(
        (upper.diagonal(), (range(upper.shape[0]), range(upper.shape[0]))),
        shape=upper.shape,
    )
    result = (upper + upper.T - diagonal).tocsr()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def build_generation_evidence(
    bundle: CandidateClusterInputBundle,
) -> CandidateViewEvidence:
    """Construct frozen W/C/F/S similarities from generation families only."""

    width = accumulate_pair_evidence(
        bundle.width_blocks, basis_count=bundle.basis_count
    )
    candidate = accumulate_pair_evidence(
        bundle.candidate_blocks, basis_count=bundle.basis_count
    )
    eligible = basis_eligibility(width) & basis_eligibility(candidate)
    width_similarity = mean_similarity_matrix(
        width,
        min_pair_target_overlap=MIN_PAIR_TARGETS,
        min_pair_response_overlap=MIN_PAIR_RESPONSES,
        min_pair_family_overlap=MIN_PAIR_FAMILIES,
        eligible_mask=eligible,
    )
    candidate_similarity = mean_similarity_matrix(
        candidate,
        min_pair_target_overlap=MIN_PAIR_TARGETS,
        min_pair_response_overlap=MIN_PAIR_RESPONSES,
        min_pair_family_overlap=MIN_PAIR_FAMILIES,
        eligible_mask=eligible,
    )
    support_evidence = accumulate_pair_evidence(
        bundle.candidate_support_blocks, basis_count=bundle.basis_count
    )
    return CandidateViewEvidence(
        width=width,
        candidate=candidate,
        common_eligible_mask=eligible,
        width_similarity=width_similarity,
        candidate_similarity=candidate_similarity,
        fusion_similarity=fuse_calibrated_similarities(
            width_similarity, candidate_similarity
        ),
        support_similarity=weighted_support_jaccard(
            support_evidence, eligible_mask=eligible
        ),
    )


def choose_medoid_seed(
    labels_by_seed: Mapping[int, NDArray[np.int64]],
) -> tuple[int, dict[tuple[int, int], float]]:
    """Choose the maximum-mean-ARI seed medoid with smaller-seed ties."""

    if len(labels_by_seed) < 2:
        raise ValueError("seed medoid requires at least two assignments")
    scores: dict[tuple[int, int], float] = {}
    by_seed: dict[int, list[float]] = {seed: [] for seed in labels_by_seed}
    for left, right in itertools.combinations(sorted(labels_by_seed), 2):
        score = assignment_ari(labels_by_seed[left], labels_by_seed[right])
        scores[(left, right)] = score
        by_seed[left].append(score)
        by_seed[right].append(score)
    medoid = min(by_seed, key=lambda seed: (-mean(by_seed[seed]), seed))
    return medoid, scores


def fit_resolution(
    view: str,
    similarity: csr_matrix,
    *,
    eligible_mask: NDArray[np.bool_],
    n_clusters: int,
) -> ResolutionFit:
    """Fit all frozen seeds and select their medoid for one view/resolution."""

    affinity = knn_affinity(similarity, neighbors=NEIGHBORS, symmetrization="union_max")
    finite_symmetric = bool(
        np.all(np.isfinite(affinity.data)) and (affinity - affinity.T).nnz == 0
    )
    active = np.asarray(affinity.sum(axis=1)).ravel() > 0
    active_count = int(active.sum())
    component_count = (
        int(
            connected_components(
                affinity[active][:, active], directed=False, return_labels=False
            )
        )
        if active_count
        else 0
    )
    preflight_error: str | None = None
    if not finite_symmetric:
        preflight_error = "nonfinite_or_asymmetric_affinity"
    elif active_count < n_clusters + 1:
        preflight_error = "insufficient_active_bases"
    elif component_count > n_clusters:
        preflight_error = "too_many_connected_components"

    eligible_count = int(eligible_mask.sum())
    seed_fits: dict[int, SeedFit] = {}
    for seed in RANDOM_SEEDS:
        if preflight_error is not None:
            seed_fits[seed] = SeedFit(seed, None, False, 0.0, preflight_error)
            continue
        try:
            result = sparse_spectral_cluster(
                affinity,
                n_clusters=n_clusters,
                random_seed=seed,
                self_loop_weight=1.0,
                eigen_tolerance=1e-6,
            )
            fraction = (
                float(np.sum((result.labels >= 0) & eligible_mask) / eligible_count)
                if eligible_count
                else 0.0
            )
            assigned_labels = np.unique(result.labels[result.labels >= 0])
            exact_cluster_count = np.array_equal(
                assigned_labels, np.arange(n_clusters, dtype=np.int64)
            )
            valid = fraction >= MIN_ASSIGNMENT_FRACTION and exact_cluster_count
            if fraction < MIN_ASSIGNMENT_FRACTION:
                error = "assignment_coverage"
            elif not exact_cluster_count:
                error = "assigned_cluster_count"
            else:
                error = None
            seed_fits[seed] = SeedFit(seed, result, valid, fraction, error)
        except (
            ArpackNoConvergence,
            FloatingPointError,
            RuntimeError,
            ValueError,
        ) as error:
            seed_fits[seed] = SeedFit(
                seed, None, False, 0.0, f"{type(error).__name__}: {error}"
            )

    valid = all(seed_fit.valid for seed_fit in seed_fits.values())
    medoid: int | None = None
    pairwise: dict[tuple[int, int], float] = {}
    size: Mapping[str, Any] | None = None
    graph: Mapping[str, Any] | None = None
    mean_ari: float | None = None
    minimum_ari: float | None = None
    if valid:
        labels_by_seed = {
            seed: seed_fit.result.labels  # type: ignore[union-attr]
            for seed, seed_fit in seed_fits.items()
        }
        medoid, pairwise = choose_medoid_seed(labels_by_seed)
        ari_values = list(pairwise.values())
        mean_ari = mean(ari_values)
        minimum_ari = min(ari_values)
        labels = labels_by_seed[medoid]
        size = cluster_size_metrics(labels)
        graph = sparse_graph_partition_metrics(labels, affinity)
    return ResolutionFit(
        view=view,
        n_clusters=n_clusters,
        affinity=affinity,
        seeds=seed_fits,
        valid=valid,
        medoid_seed=medoid,
        pairwise_seed_ari=pairwise,
        mean_seed_ari=mean_ari,
        minimum_seed_ari=minimum_ari,
        size_metrics=size,
        graph_metrics=graph,
    )


def choose_common_cluster_count(
    fits: Mapping[str, Mapping[int, ResolutionFit]],
) -> int | None:
    """Apply the frozen K=64 preference, otherwise smallest common valid K."""

    required = ("W", "C", "F")
    common = [
        count
        for count in CLUSTER_COUNTS
        if all(
            fits.get(view, {}).get(count) is not None and fits[view][count].valid
            for view in required
        )
    ]
    if 64 in common:
        return 64
    return min(common) if common else None


def fit_generation_grid(evidence: CandidateViewEvidence) -> GenerationClusterFit:
    """Fit the complete initial W/C/F grid and S at the selected common K."""

    similarities = {
        "W": evidence.width_similarity,
        "C": evidence.candidate_similarity,
        "F": evidence.fusion_similarity,
    }
    directional = {
        view: {
            count: fit_resolution(
                view,
                similarity,
                eligible_mask=evidence.common_eligible_mask,
                n_clusters=count,
            )
            for count in CLUSTER_COUNTS
        }
        for view, similarity in similarities.items()
    }
    chosen = choose_common_cluster_count(directional)
    support = (
        None
        if chosen is None
        else fit_resolution(
            "S",
            evidence.support_similarity,
            eligible_mask=evidence.common_eligible_mask,
            n_clusters=chosen,
        )
    )
    return GenerationClusterFit(
        evidence=evidence,
        directional=directional,
        chosen_cluster_count=chosen,
        support=support,
    )
