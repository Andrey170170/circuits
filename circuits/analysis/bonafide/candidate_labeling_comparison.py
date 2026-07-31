"""Freeze the evidence-only C2 labeling comparison before model calls.

The artifact prepared here contains only the two protocol-eligible arms: the
same twelve W64 clusters with width-one evidence, and those same clusters with
candidate evidence added.  Generation witnesses and held-out scoring records
are physically separated so selection/audit measurements cannot enter prompts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

from circuits.analysis.bonafide import (
    candidate_clustering as candidate_clustering_module,
)
from circuits.analysis.bonafide import (
    candidate_clustering_execution as candidate_clustering_execution_module,
)
from circuits.analysis.bonafide import (
    candidate_labelability_evaluation as candidate_labelability_module,
)
from circuits.analysis.bonafide import canonical as canonical_module
from circuits.analysis.bonafide.candidate_clustering import (
    CandidateClusterInputBundle,
    load_candidate_cluster_input_bundle,
)
from circuits.analysis.bonafide.candidate_clustering_execution import (
    LoadedCandidateClusteringBaseline,
    load_candidate_clustering_baseline,
)
from circuits.analysis.bonafide.candidate_labelability_evaluation import (
    extract_chosen_medoid_assignments,
    load_candidate_labelability_evaluation,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.tracing import trace as trace_module
from circuits.tracing.trace import get_chat_template, tokenize_teacher_forced_response

CANDIDATE_LABELING_COMPARISON_SCHEMA = "adag.bonafide.candidate-labeling-comparison.v1"
ANCHOR_SCHEMA = "adag.bonafide.candidate-labeling-anchors.v1"
GENERATION_EVIDENCE_SCHEMA = "adag.bonafide.candidate-labeling-generation-witness.v1"
SCORING_EVIDENCE_SCHEMA = "adag.bonafide.candidate-labeling-scoring-witness.v1"
ARM_HANDOFF_SCHEMA = "adag.bonafide.candidate-labeling-arm-handoff.v1"

ANCHORS_FILE = "anchors.json"
GENERATION_FILE = "generation-evidence.jsonl"
SCORING_FILE = "scoring-evidence.jsonl"
HANDOFF_FILE = "arm-handoff.jsonl"
MANIFEST_FILE = "manifest.json"

TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"
TOKENIZER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TOKENIZER_CHAT_TEMPLATE_SHA256 = (
    "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"
)
EXPECTED_W_ANCHORS = (61, 5, 42, 34, 41, 21, 59, 13, 43, 58, 49, 47)
TARGET_POINTS = tuple(
    (member, support)
    for member in (Fraction(1, 6), Fraction(1, 2), Fraction(5, 6))
    for support in (
        Fraction(1, 8),
        Fraction(3, 8),
        Fraction(5, 8),
        Fraction(7, 8),
    )
)
MAX_WIDTH_HIGHLIGHTS = 16

EXPECTED_ELIGIBLE_ARMS = [
    {
        "arm_id": "arm_1_width_only",
        "legacy_request_state": "primary",
        "cluster_state": "W64",
        "candidate_evidence": False,
    },
    {
        "arm_id": "arm_2_width_plus_candidate",
        "legacy_request_state": "alternative",
        "cluster_state": "W64",
        "candidate_evidence": True,
    },
]
EXPECTED_FIREWALL = {
    "model_calls_made": False,
    "descriptions_generated": False,
    "outcomes_inspected": False,
    "confirmatory_holdout_opened": False,
    "generation_prompts_use_generation_partition_only": True,
    "selection_audit_prompt_eligible": False,
    "selection_audit_use": "later_input_localization_scoring_only",
    "candidate_reclustered_arms_included": False,
}
EXPECTED_PROMPT_CONTRACT = {
    "generation_witness_policy": "all_W_width_supported_generation_targets_no_subsampling",
    "local_prefix_definition": "full_teacher_forced_input_excluding_observed_target_token",
    "renderer_frozen": False,
    "renderer_requirement": "freeze_prompt_length_and_rendering_policy_before_model_calls",
    "typed_output_fields": [
        "input_localization_hypothesis",
        "exploratory_candidate_description",
        "background_or_confound",
        "limitations",
        "status",
    ],
    "forbidden_claims": [
        "response_identity",
        "causality",
        "selectivity",
        "generality",
        "faithfulness",
    ],
}

_SOURCE_BINDINGS = {
    "canonical": "circuits/analysis/bonafide/canonical.py",
    "candidate_clustering": "circuits/analysis/bonafide/candidate_clustering.py",
    "candidate_clustering_execution": (
        "circuits/analysis/bonafide/candidate_clustering_execution.py"
    ),
    "candidate_labelability_evaluation": (
        "circuits/analysis/bonafide/candidate_labelability_evaluation.py"
    ),
    "candidate_labeling_comparison": (
        "circuits/analysis/bonafide/candidate_labeling_comparison.py"
    ),
    "candidate_labeling_prepare_cli": (
        "scripts/bonafide/candidate_labeling_prepare.py"
    ),
    "frozen_protocol": "docs/CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md",
    "teacher_forced_tokenization": "circuits/tracing/trace.py",
}
_RUNTIME_MODULE_BINDINGS = {
    "canonical": canonical_module,
    "candidate_clustering": candidate_clustering_module,
    "candidate_clustering_execution": candidate_clustering_execution_module,
    "candidate_labelability_evaluation": candidate_labelability_module,
    "teacher_forced_tokenization": trace_module,
}


@dataclass(frozen=True)
class LoadedCandidateLabelingComparison:
    """A content-validated provider-neutral comparison artifact."""

    root: Path
    manifest: Mapping[str, Any]
    anchors: Mapping[str, Any]
    generation_evidence: tuple[Mapping[str, Any], ...]
    scoring_evidence: tuple[Mapping[str, Any], ...]
    arm_handoff: tuple[Mapping[str, Any], ...]


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise ValueError(
            f"unable to bind labeling comparison revision: {message}"
        ) from error
    return completed.stdout.strip()


def validate_candidate_labeling_runtime_paths(repo_root: Path) -> None:
    """Reject imports from another editable worktree."""

    repo_root = repo_root.resolve()
    expected_self = repo_root / _SOURCE_BINDINGS["candidate_labeling_comparison"]
    if Path(__file__).resolve() != expected_self.resolve():
        raise ValueError(
            "candidate labeling comparison was imported from another worktree"
        )
    for role, module in _RUNTIME_MODULE_BINDINGS.items():
        observed = getattr(module, "__file__", None)
        if not isinstance(observed, str):
            raise TypeError(f"candidate labeling runtime module has no path: {role}")
        expected = repo_root / _SOURCE_BINDINGS[role]
        if Path(observed).resolve() != expected.resolve():
            raise ValueError(
                "candidate labeling runtime module came from another worktree: "
                f"{role}"
            )


def collect_candidate_labeling_revision(repo_root: Path) -> dict[str, Any]:
    """Bind the clean tracked tree and all preparation sources."""

    repo_root = repo_root.resolve()
    if Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve() != repo_root:
        raise ValueError("candidate labeling preparation must run from repository root")
    validate_candidate_labeling_runtime_paths(repo_root)
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError(
            "candidate labeling preparation requires a clean tracked worktree"
        )
    files: list[dict[str, str]] = []
    for role, relative in _SOURCE_BINDINGS.items():
        if _git(repo_root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(
                f"candidate labeling preparation source is not tracked: {relative}"
            )
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(
                f"candidate labeling preparation source is missing: {relative}"
            )
        files.append({"role": role, "path": relative, "sha256": file_sha256(path)})
    return {
        "repo_root": str(repo_root),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "files": files,
    }


def _midrank_percentiles(values: Mapping[int, int]) -> dict[int, Fraction]:
    """Ascending shared midrank percentiles with cluster-ID-only tie ordering."""

    if not values:
        raise ValueError("midrank percentiles require values")
    ordered = sorted(values, key=lambda cluster: (values[cluster], cluster))
    n = len(ordered)
    result: dict[int, Fraction] = {}
    start = 0
    while start < n:
        end = start + 1
        while end < n and values[ordered[end]] == values[ordered[start]]:
            end += 1
        # One-based ranks are start+1 through end; subtracting 1/2 occurs
        # before dividing by n, exactly as frozen in the protocol.
        midrank = Fraction((start + 1) + end, 2)
        percentile = (midrank - Fraction(1, 2)) / n
        for index in range(start, end):
            result[ordered[index]] = percentile
        start = end
    return result


def _integer_cost_matrix(costs: Sequence[Sequence[Fraction]]) -> np.ndarray:
    denominator = 1
    for row in costs:
        for value in row:
            denominator = math.lcm(denominator, value.denominator)
    integers = np.asarray(
        [
            [value.numerator * (denominator // value.denominator) for value in row]
            for row in costs
        ],
        dtype=np.int64,
    )
    if integers.ndim != 2 or integers.shape[0] > integers.shape[1]:
        raise ValueError("anchor cost matrix must be rectangular with enough clusters")
    return integers


def _minimum_assignment_cost(costs: np.ndarray) -> int:
    rows, columns = linear_sum_assignment(costs)
    if len(rows) != costs.shape[0]:
        raise AssertionError("Hungarian assignment did not cover every target point")
    return int(sum(int(costs[row, column]) for row, column in zip(rows, columns)))


def _lexicographic_optimal_assignment(
    costs: np.ndarray, cluster_ids: Sequence[int]
) -> tuple[int, ...]:
    """Choose the lexicographically smallest tuple among exact global optima."""

    remaining = list(range(len(cluster_ids)))
    optimum = _minimum_assignment_cost(costs)
    chosen: list[int] = []
    spent = 0
    for target_index in range(costs.shape[0]):
        suffix_rows = list(range(target_index + 1, costs.shape[0]))
        for column in sorted(remaining, key=lambda item: cluster_ids[item]):
            candidate_spent = spent + int(costs[target_index, column])
            suffix_columns = [item for item in remaining if item != column]
            suffix_cost = 0
            if suffix_rows:
                suffix_cost = _minimum_assignment_cost(
                    costs[np.ix_(suffix_rows, suffix_columns)]
                )
            if candidate_spent + suffix_cost == optimum:
                chosen.append(int(cluster_ids[column]))
                remaining.remove(column)
                spent = candidate_spent
                break
        else:  # pragma: no cover - mathematical invariant
            raise AssertionError(
                "unable to recover lexicographically optimal assignment"
            )
    return tuple(chosen)


def select_w_anchors(
    *, member_counts: Mapping[int, int], generation_target_counts: Mapping[int, int]
) -> dict[str, Any]:
    """Apply the frozen 3x4 quantile/Hungarian anchor design exactly."""

    if set(member_counts) != set(generation_target_counts):
        raise ValueError("anchor member and support cluster inventories differ")
    if len(member_counts) < len(TARGET_POINTS):
        raise ValueError("fewer than 12 labeling-ready W clusters")
    if any(value <= 0 for value in member_counts.values()) or any(
        value <= 0 for value in generation_target_counts.values()
    ):
        raise ValueError("anchor coordinates must be positive")
    cluster_ids = sorted(member_counts)
    member_percentiles = _midrank_percentiles(member_counts)
    support_percentiles = _midrank_percentiles(generation_target_counts)
    fractional_costs = [
        [
            (member_percentiles[cluster] - target_member) ** 2
            + (support_percentiles[cluster] - target_support) ** 2
            for cluster in cluster_ids
        ]
        for target_member, target_support in TARGET_POINTS
    ]
    integer_costs = _integer_cost_matrix(fractional_costs)
    selected = _lexicographic_optimal_assignment(integer_costs, cluster_ids)
    return {
        "method": "fixed_3x4_midrank_hungarian_lexicographic_v1",
        "ready_cluster_count": len(cluster_ids),
        "target_points": [
            {
                "member_coordinate": float(member),
                "support_coordinate": float(support),
            }
            for member, support in TARGET_POINTS
        ],
        "anchors_in_target_point_order": list(selected),
        "ready_clusters": [
            {
                "cluster_id": cluster,
                "member_count": int(member_counts[cluster]),
                "generation_target_witness_count": int(
                    generation_target_counts[cluster]
                ),
                "member_midrank_percentile": float(member_percentiles[cluster]),
                "support_midrank_percentile": float(support_percentiles[cluster]),
            }
            for cluster in cluster_ids
        ],
    }


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return str(
        tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _target_context(target: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    example = json.loads(str(target["example_json"]))
    if not isinstance(example, Mapping):
        raise TypeError("target example JSON is not an object")
    tokenized = tokenize_teacher_forced_response(
        tokenizer, str(example["prompt"]), str(example["response"])
    )
    position = int(target["response_position"])
    if not 0 <= position < len(tokenized.response_ids):
        raise ValueError("target response position is outside exact tokenization")
    observed_id = int(tokenized.response_ids[position])
    if observed_id != int(target["observed_token_id"]):
        raise ValueError("reconstructed observed token ID drift")
    observed_text = _decode_token(tokenizer, observed_id)
    if observed_text != str(target["observed_token_text"]):
        raise ValueError("reconstructed observed token text drift")
    prefix_ids = [
        *map(int, tokenized.assistant_prefix_ids),
        *map(int, tokenized.response_ids[:position]),
    ]
    prefix_tokens = [_decode_token(tokenizer, token_id) for token_id in prefix_ids]
    return {
        "definition": "full_teacher_forced_input_excluding_observed_target_token",
        "token_ids": prefix_ids,
        "tokens": prefix_tokens,
        "text": "".join(prefix_tokens),
        "observed_token": {
            "response_position": position,
            "token_id": observed_id,
            "token_text": observed_text,
        },
        "source_attribution_token_count": len(prefix_ids),
    }


def _validated_tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    name_or_path = str(getattr(tokenizer, "name_or_path", ""))
    if name_or_path != TOKENIZER_ID:
        raise ValueError(
            "candidate labeling tokenizer identity drift: "
            f"observed={name_or_path!r}, expected={TOKENIZER_ID!r}"
        )
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, Mapping):
        raise TypeError("candidate labeling tokenizer init metadata is invalid")
    commit_hash = init_kwargs.get("_commit_hash")
    if commit_hash is not None and str(commit_hash) != TOKENIZER_REVISION:
        raise ValueError("candidate labeling tokenizer revision drift")
    chat_hash = hashlib.sha256(get_chat_template(tokenizer).encode()).hexdigest()
    if chat_hash != TOKENIZER_CHAT_TEMPLATE_SHA256:
        raise ValueError("tokenizer chat-template provenance drift")
    return {
        "model_id": TOKENIZER_ID,
        "revision": TOKENIZER_REVISION,
        "name_or_path": name_or_path,
        "resolved_commit_hash": str(commit_hash) if commit_hash is not None else None,
        "class": type(tokenizer).__name__,
        "chat_template_sha256": chat_hash,
        "local_files_only": True,
        "reconstruction": "tokenize_teacher_forced_response",
    }


def _aggregate_width(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("width aggregation requires assigned supported members")
    width = len(rows[0]["attribution_profile"])
    sums = np.zeros(width, dtype=np.float64)
    occurrence_support = np.zeros(width, dtype=np.int64)
    member_occurrences = 0
    member_bases: set[int] = set()
    for row in rows:
        values = row["attribution_profile"]
        support = row["attribution_support"]
        count = int(row["occurrence_count"])
        if len(values) != width or len(support) != width or count <= 0:
            raise ValueError("invalid width profile row")
        member_occurrences += count
        member_bases.add(int(row["signed_basis_index"]))
        for index, (value, supported) in enumerate(zip(values, support, strict=True)):
            if supported:
                if value is None or not math.isfinite(float(value)):
                    raise ValueError("supported width profile value is not finite")
                sums[index] += float(value)
                occurrence_support[index] += count
            elif value is not None:
                raise ValueError("unsupported width profile coordinate is not null")
    means: list[float | None] = [
        (
            float(sums[index] / occurrence_support[index])
            if occurrence_support[index]
            else None
        )
        for index in range(width)
    ]
    ranked = sorted(
        (index for index, value in enumerate(means) if value is not None),
        key=lambda index: (-abs(float(means[index])), index),
    )[:MAX_WIDTH_HIGHLIGHTS]
    return {
        "member_basis_count": len(member_bases),
        "member_occurrence_count": member_occurrences,
        "signed_sum_by_source_token": [
            float(sums[index]) if occurrence_support[index] else None
            for index in range(width)
        ],
        "mean_by_member_occurrence": means,
        "support_occurrence_count_by_source_token": occurrence_support.tolist(),
        "highlight_token_indices": ranked,
    }


def _aggregate_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate aggregation requires assigned supported members")
    total = np.zeros(5, dtype=np.float64)
    member_occurrences = 0
    member_bases: set[int] = set()
    for row in rows:
        vector = np.asarray(row["candidate_contrast_profile"], dtype=np.float64)
        count = int(row["occurrence_count"])
        if vector.shape != (5,) or not np.all(np.isfinite(vector)) or count <= 0:
            raise ValueError("invalid candidate profile row")
        total += vector
        member_occurrences += count
        member_bases.add(int(row["signed_basis_index"]))
    mean = total / member_occurrences
    norm = float(np.linalg.norm(mean))
    unit = (mean / norm).tolist() if norm > 0.0 else None
    return {
        "rank_order": [1, 2, 3, 4, 5],
        "member_basis_count": len(member_bases),
        "member_occurrence_count_m": member_occurrences,
        "signed_sum": total.tolist(),
        "elementwise_mean": mean.tolist(),
        "mean_l2_norm": norm,
        "mean_unit_direction": unit,
        "clipped": False,
    }


def _candidate_slots(target: Mapping[str, Any]) -> dict[str, Any]:
    selection = json.loads(str(target["candidate_selection_json"]))
    if not isinstance(selection, Mapping) or not isinstance(
        selection.get("candidates"), list
    ):
        raise TypeError("candidate selection JSON is invalid")
    candidates = selection["candidates"]
    by_rank = {
        int(item["full_distribution_rank"]): item
        for item in candidates
        if int(item["full_distribution_rank"]) in range(1, 6)
    }
    if set(by_rank) != set(range(1, 6)):
        raise ValueError("candidate selection lacks exact model ranks one through five")
    slots = []
    for rank in range(1, 6):
        item = by_rank[rank]
        logit = float(item["logit"])
        probability = float(item["probability"])
        if not math.isfinite(logit) or not math.isfinite(probability):
            raise ValueError("candidate slot contains nonfinite values")
        slots.append(
            {
                "rank": rank,
                "token_id": int(item["token_id"]),
                "token_text": str(item["token_text"]),
                "logit": logit,
                "probability": probability,
                "is_observed": bool(item["is_observed"]),
            }
        )
    observed = [item for item in candidates if bool(item["is_observed"])]
    if len(observed) != 1 or int(observed[0]["token_id"]) != int(
        target["observed_token_id"]
    ):
        raise ValueError("candidate selection observed-token drift")
    candidate_count = int(target["candidate_count"])
    if candidate_count not in {5, 6} or len(candidates) != candidate_count:
        raise ValueError("candidate selection width drift")
    observed_rank = int(observed[0]["full_distribution_rank"])
    if (candidate_count == 5 and observed_rank not in range(1, 6)) or (
        candidate_count == 6 and observed_rank in range(1, 6)
    ):
        raise ValueError("candidate width/observed-rank structural semantics drift")
    return {
        "model_rank_slots": slots,
        "observed_token_full_distribution_rank": observed_rank,
        "candidate_axis_width": candidate_count,
        "distinct_competitor_count": candidate_count - 1,
    }


def _candidate_evidence(
    target: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    slots = _candidate_slots(target)
    signature = _aggregate_candidate(rows)
    if slots["candidate_axis_width"] == 5:
        zero_index = slots["observed_token_full_distribution_rank"] - 1
        if (
            signature["signed_sum"][zero_index] != 0.0
            or signature["elementwise_mean"][zero_index] != 0.0
        ):
            raise ValueError("observed-rank candidate signature is not structural zero")
    return slots, signature


def _group_profiles(
    rows: Sequence[Mapping[str, Any]], assignments: np.ndarray
) -> dict[tuple[int, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        basis = int(row["signed_basis_index"])
        if not 0 <= basis < len(assignments):
            raise ValueError("profile basis lies outside W assignment")
        cluster = int(assignments[basis])
        if cluster >= 0:
            grouped[(cluster, str(row["case_id"]))].append(row)
    return grouped


def build_candidate_labeling_comparison(
    *,
    bundle: CandidateClusterInputBundle,
    baseline: LoadedCandidateClusteringBaseline,
    evaluation_report: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Build deterministic anchors, prompt evidence, and held-out scorer records."""

    input_hash = str(bundle.manifest["manifest_sha256"])
    baseline_hash = str(baseline.manifest["manifest_sha256"])
    if evaluation_report["source_input_bundle"]["manifest_sha256"] != input_hash:
        raise ValueError("evaluation/input bundle binding drift")
    if (
        evaluation_report["source_clustering_baseline"]["manifest_sha256"]
        != baseline_hash
    ):
        raise ValueError("evaluation/baseline binding drift")
    if int(evaluation_report["evaluation"]["chosen_cluster_count"]) != 64:
        raise ValueError("labeling comparison requires the frozen W64 resolution")
    _validated_tokenizer_identity(tokenizer)

    assignments = extract_chosen_medoid_assignments(
        baseline, basis_count=bundle.basis_count
    )["W"]
    readiness = evaluation_report["evaluation"]["cluster_labeling_readiness"]["W"]
    if readiness.get("support_policy") != "candidate_and_width_frozen_thresholds":
        raise ValueError("W readiness does not use the conservative frozen policy")
    ready_ids = [
        int(item["cluster_id"])
        for item in readiness["clusters"]
        if item["labeling_ready"] is True
    ]
    member_counts = {
        cluster: int(np.sum(assignments == cluster)) for cluster in ready_ids
    }
    width_reports = {
        int(item["cluster_id"]): item for item in readiness["width_support"]["clusters"]
    }
    candidate_reports = {
        int(item["cluster_id"]): item
        for item in readiness["candidate_support"]["clusters"]
    }
    generation_counts = {
        cluster: int(
            width_reports[cluster]["partitions"]["generation"]["target_witness_count"]
        )
        for cluster in ready_ids
    }
    anchor_selection = select_w_anchors(
        member_counts=member_counts, generation_target_counts=generation_counts
    )
    anchor_ids = tuple(anchor_selection["anchors_in_target_point_order"])
    if anchor_ids != EXPECTED_W_ANCHORS:
        raise ValueError(
            f"frozen C2 anchor drift: observed={anchor_ids}, expected={EXPECTED_W_ANCHORS}"
        )

    targets = {str(row["case_id"]): row for row in bundle.target_rows}
    if len(targets) != len(bundle.target_rows):
        raise ValueError("candidate labeling target identities are not unique")
    width_rows = pq.read_table(bundle.root / "width-profiles.parquet").to_pylist()
    candidate_rows = pq.read_table(
        bundle.root / "candidate-profiles.parquet"
    ).to_pylist()
    width_groups = _group_profiles(width_rows, assignments)
    candidate_groups = _group_profiles(candidate_rows, assignments)
    contexts: dict[str, dict[str, Any]] = {}
    generation_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    witness_ids_by_anchor: dict[int, list[str]] = {}

    for anchor_index, cluster in enumerate(anchor_ids):
        cluster_case_ids = sorted(
            case_id
            for candidate_cluster, case_id in width_groups
            if candidate_cluster == cluster
        )
        generation_case_ids = [
            case_id
            for case_id in cluster_case_ids
            if targets[case_id]["family_partition"] == "generation"
        ]
        if len(generation_case_ids) != generation_counts[cluster]:
            raise ValueError("generation witness count disagrees with readiness report")
        if any(
            (cluster, case_id) not in candidate_groups
            for case_id in generation_case_ids
        ):
            raise ValueError("paired W generation witness lacks candidate evidence")
        witness_ids_by_anchor[cluster] = generation_case_ids
        for case_id in cluster_case_ids:
            target = targets[case_id]
            partition = str(target["family_partition"])
            if case_id not in contexts:
                contexts[case_id] = _target_context(target, tokenizer)
            context = contexts[case_id]
            width = _aggregate_width(width_groups[(cluster, case_id)])
            if (
                len(width["mean_by_member_occurrence"])
                != context["source_attribution_token_count"]
            ):
                raise ValueError(
                    "width profile length differs from exact prediction prefix"
                )
            token_ids = list(context["token_ids"])
            token_texts = list(context["tokens"])
            width["highlights"] = [
                {
                    "token_index": index,
                    "token_id": token_ids[index],
                    "token_text": token_texts[index],
                    "score": width["mean_by_member_occurrence"][index],
                    "signed_sum": width["signed_sum_by_source_token"][index],
                    "support_occurrence_count": width[
                        "support_occurrence_count_by_source_token"
                    ][index],
                }
                for index in width.pop("highlight_token_indices")
            ]
            common = {
                "anchor_index": anchor_index,
                "cluster_id": cluster,
                "case_id": case_id,
                "family_id": str(target["base_question_id"]),
                "response_id": str(target["response_id"]),
                "phase_bin": int(target["phase_bin"]),
                "family_partition": partition,
                "width_one_source_attribution": width,
            }
            if partition == "generation":
                candidate_slots, candidate_signature = _candidate_evidence(
                    target, candidate_groups[(cluster, case_id)]
                )
                generation_rows.append(
                    {
                        "schema_version": GENERATION_EVIDENCE_SCHEMA,
                        **common,
                        "prompt_eligible": True,
                        "local_prefix": context,
                        "candidate_slots": candidate_slots,
                        "candidate_signature": candidate_signature,
                    }
                )
            elif partition in {"selection_scoring", "audit"}:
                candidate_rows = candidate_groups.get((cluster, case_id))
                if candidate_rows is None:
                    candidate_slots = _candidate_slots(target)
                    candidate_signature = None
                else:
                    candidate_slots, candidate_signature = _candidate_evidence(
                        target, candidate_rows
                    )
                scoring_rows.append(
                    {
                        "schema_version": SCORING_EVIDENCE_SCHEMA,
                        **common,
                        "prompt_eligible": False,
                        "scorer_use": "input_localization_only",
                        "source_tokens": {
                            "token_ids": token_ids,
                            "tokens": token_texts,
                        },
                        "observed_token": context["observed_token"],
                        "candidate_slots": candidate_slots,
                        "candidate_signature": candidate_signature,
                    }
                )
            else:
                raise ValueError(f"unexpected family partition: {partition}")

    handoff: list[dict[str, Any]] = []
    for arm_id, legacy_state, include_candidate in (
        ("arm_1_width_only", "primary", False),
        ("arm_2_width_plus_candidate", "alternative", True),
    ):
        for anchor_index, cluster in enumerate(anchor_ids):
            handoff.append(
                {
                    "schema_version": ARM_HANDOFF_SCHEMA,
                    "arm_id": arm_id,
                    "legacy_request_state": legacy_state,
                    "anchor_index": anchor_index,
                    "cluster_id": cluster,
                    "generation_witness_case_ids": witness_ids_by_anchor[cluster],
                    "include_width_one_source_attribution": True,
                    "include_candidate_slots_and_signature": include_candidate,
                    "provider": None,
                    "model": None,
                    "prompt_renderer_status": "requires_separately_frozen_renderer",
                }
            )

    anchors = {
        "schema_version": ANCHOR_SCHEMA,
        "state": "W",
        "cluster_count": 64,
        "support_coordinate": "width_generation_target_witness_count",
        "selection": anchor_selection,
        "anchors": [
            {
                "anchor_index": index,
                "cluster_id": cluster,
                "member_count": member_counts[cluster],
                "generation_target_witness_count": generation_counts[cluster],
                "candidate_generation_target_witness_count": int(
                    candidate_reports[cluster]["partitions"]["generation"][
                        "target_witness_count"
                    ]
                ),
                "generation_witness_case_ids": witness_ids_by_anchor[cluster],
            }
            for index, cluster in enumerate(anchor_ids)
        ],
    }
    return anchors, generation_rows, scoring_rows, handoff


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
        for row in rows
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_binding(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if load_json_object(path) != dict(manifest):
        raise ValueError(f"source manifest changed during preparation: {path}")
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "manifest_file_sha256": file_sha256(path),
        "schema_version": str(manifest["schema_version"]),
    }


def _validate_bound_manifest(record: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} binding is invalid")
    path = Path(str(record.get("manifest_path")))
    if not path.is_file() or file_sha256(path) != record.get("manifest_file_sha256"):
        raise ValueError(f"{label} manifest file drift")
    manifest = load_json_object(path)
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core) or recorded != record.get("manifest_sha256"):
        raise ValueError(f"{label} manifest content drift")
    if manifest.get("schema_version") != record.get("schema_version"):
        raise ValueError(f"{label} schema drift")
    return manifest


def _load_labelability_evaluation_portably(path: Path) -> dict[str, Any]:
    """Deep-validate an upstream report without importing it from two worktrees."""

    report = load_candidate_labelability_evaluation(path, verify_sources=False)
    revision = report.get("code_revision")
    if (
        not isinstance(revision, Mapping)
        or revision.get("tracked_worktree_clean") is not True
    ):
        raise ValueError("candidate labelability source revision is invalid")
    recorded_root = Path(str(revision.get("repo_root")))
    bindings = candidate_labelability_module._SOURCE_BINDINGS
    records = revision.get("files")
    if not isinstance(records, list) or len(records) != len(bindings):
        raise ValueError("candidate labelability source file inventory drift")
    by_role = {item.get("role"): item for item in records if isinstance(item, Mapping)}
    if set(by_role) != set(bindings):
        raise ValueError("candidate labelability source roles drift")
    for role, relative in bindings.items():
        item = by_role[role]
        if item.get("path") != relative or file_sha256(
            recorded_root / relative
        ) != item.get("sha256"):
            raise ValueError(f"candidate labelability source hash drift: {role}")

    input_record = report.get("source_input_bundle")
    baseline_record = report.get("source_clustering_baseline")
    input_manifest = _validate_bound_manifest(
        input_record, label="candidate labelability input bundle"
    )
    baseline_manifest = _validate_bound_manifest(
        baseline_record, label="candidate labelability clustering baseline"
    )
    assert isinstance(input_record, Mapping)
    assert isinstance(baseline_record, Mapping)
    bundle = load_candidate_cluster_input_bundle(
        Path(str(input_record["manifest_path"])).resolve().parent
    )
    baseline = load_candidate_clustering_baseline(
        Path(str(baseline_record["manifest_path"])).resolve().parent,
        verify_source=True,
    )
    if (
        bundle.manifest != input_manifest
        or baseline.manifest != baseline_manifest
        or baseline.manifest["source_input_bundle"]["manifest_sha256"]
        != bundle.manifest["manifest_sha256"]
    ):
        raise ValueError("candidate labelability portable source binding drift")
    recomputed = candidate_labelability_module.evaluate_loaded_candidate_labelability(
        bundle, baseline
    )
    if report.get("evaluation") != recomputed:
        raise ValueError("candidate labelability portable recomputation drift")
    return report


def run_candidate_labeling_preparation(
    *,
    input_root: Path,
    baseline_root: Path,
    evaluation_path: Path,
    output_root: Path,
    repo_root: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    """Deep-validate sources and atomically publish a no-overwrite artifact."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace labeling artifact: {output_root}")
    revision = collect_candidate_labeling_revision(repo_root)
    bundle = load_candidate_cluster_input_bundle(input_root)
    baseline = load_candidate_clustering_baseline(baseline_root, verify_source=True)
    report = _load_labelability_evaluation_portably(evaluation_path)
    anchors, generation, scoring, handoff = build_candidate_labeling_comparison(
        bundle=bundle,
        baseline=baseline,
        evaluation_report=report,
        tokenizer=tokenizer,
    )
    # Re-open all deep sources and the tracked revision to close the publication
    # time-of-check/time-of-use window.
    final_bundle = load_candidate_cluster_input_bundle(input_root)
    final_baseline = load_candidate_clustering_baseline(
        baseline_root, verify_source=True
    )
    final_report = _load_labelability_evaluation_portably(evaluation_path)
    if (
        dict(final_bundle.manifest) != dict(bundle.manifest)
        or dict(final_baseline.manifest) != dict(baseline.manifest)
        or final_report != report
        or collect_candidate_labeling_revision(repo_root) != revision
    ):
        raise ValueError("labeling comparison inputs changed during preparation")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        payloads = {
            ANCHORS_FILE: _json_bytes(anchors),
            GENERATION_FILE: _jsonl_bytes(generation),
            SCORING_FILE: _jsonl_bytes(scoring),
            HANDOFF_FILE: _jsonl_bytes(handoff),
        }
        for name, payload in payloads.items():
            _write_bytes(temporary / name, payload)
        files = [
            {
                "path": name,
                "sha256": file_sha256(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
                "row_count": (
                    1
                    if name == ANCHORS_FILE
                    else (
                        len(generation)
                        if name == GENERATION_FILE
                        else len(scoring) if name == SCORING_FILE else len(handoff)
                    )
                ),
            }
            for name in sorted(payloads)
        ]
        manifest: dict[str, Any] = {
            "schema_version": CANDIDATE_LABELING_COMPARISON_SCHEMA,
            "purpose": "provider_neutral_pre_model_call_evidence_only_labeling_comparison",
            "source_input_bundle": _artifact_binding(
                bundle.root / MANIFEST_FILE, bundle.manifest
            ),
            "source_clustering_baseline": _artifact_binding(
                baseline.root / MANIFEST_FILE, baseline.manifest
            ),
            "source_labelability_evaluation": {
                "path": str(evaluation_path.resolve()),
                "manifest_sha256": str(report["manifest_sha256"]),
                "file_sha256": file_sha256(evaluation_path),
                "schema_version": str(report["schema_version"]),
            },
            "code_revision": revision,
            "tokenizer": _validated_tokenizer_identity(tokenizer),
            "eligible_arms": EXPECTED_ELIGIBLE_ARMS,
            "anchor_cluster_ids_in_target_point_order": list(EXPECTED_W_ANCHORS),
            "firewall": EXPECTED_FIREWALL,
            "prompt_contract": EXPECTED_PROMPT_CONTRACT,
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _write_bytes(temporary / MANIFEST_FILE, _json_bytes(manifest))
        candidate_clustering_execution_module._publish_directory_no_replace(
            temporary, output_root
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"JSONL row {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSONL: {path}") from error
    return tuple(rows)


def load_candidate_labeling_comparison(
    root: Path, *, verify_sources: bool = True, tokenizer: Any | None = None
) -> LoadedCandidateLabelingComparison:
    """Content-validate one frozen preparation artifact and its deep sources."""

    root = root.resolve()
    manifest = load_json_object(root / MANIFEST_FILE)
    core = dict(manifest)
    recorded = core.pop("manifest_sha256", None)
    if recorded != canonical_sha256(core):
        raise ValueError("candidate labeling comparison self-hash mismatch")
    if manifest.get("schema_version") != CANDIDATE_LABELING_COMPARISON_SCHEMA:
        raise ValueError("unsupported candidate labeling comparison schema")
    if manifest.get("purpose") != (
        "provider_neutral_pre_model_call_evidence_only_labeling_comparison"
    ):
        raise ValueError("candidate labeling comparison purpose drift")
    if manifest.get("eligible_arms") != EXPECTED_ELIGIBLE_ARMS:
        raise ValueError("candidate labeling eligible-arm contract drift")
    if manifest.get("anchor_cluster_ids_in_target_point_order") != list(
        EXPECTED_W_ANCHORS
    ):
        raise ValueError("candidate labeling manifest anchor drift")
    if manifest.get("firewall") != EXPECTED_FIREWALL:
        raise ValueError("candidate labeling manifest firewall drift")
    if manifest.get("prompt_contract") != EXPECTED_PROMPT_CONTRACT:
        raise ValueError("candidate labeling prompt contract drift")
    tokenizer_record = manifest.get("tokenizer")
    if not isinstance(tokenizer_record, Mapping):
        raise TypeError("candidate labeling tokenizer binding is invalid")
    expected_tokenizer_fields = {
        "model_id": TOKENIZER_ID,
        "revision": TOKENIZER_REVISION,
        "name_or_path": TOKENIZER_ID,
        "chat_template_sha256": TOKENIZER_CHAT_TEMPLATE_SHA256,
        "local_files_only": True,
        "reconstruction": "tokenize_teacher_forced_response",
    }
    if any(
        tokenizer_record.get(field) != value
        for field, value in expected_tokenizer_fields.items()
    ):
        raise ValueError("candidate labeling tokenizer binding drift")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise TypeError("candidate labeling file inventory is invalid")
    if any(not isinstance(item, Mapping) for item in files):
        raise TypeError("candidate labeling file inventory entry is invalid")
    paths = [str(item["path"]) for item in files]
    if len(paths) != len(set(paths)):
        raise ValueError("candidate labeling file inventory repeats a path")
    inventory = {str(item["path"]): item for item in files}
    required = {ANCHORS_FILE, GENERATION_FILE, SCORING_FILE, HANDOFF_FILE}
    if set(inventory) != required:
        raise ValueError("candidate labeling file inventory drift")
    for name, item in inventory.items():
        path = root / name
        if not path.is_file() or file_sha256(path) != item.get("sha256"):
            raise ValueError(f"candidate labeling file drift: {name}")
        if path.stat().st_size != item.get("size_bytes"):
            raise ValueError(f"candidate labeling file size drift: {name}")
    anchors = load_json_object(root / ANCHORS_FILE)
    generation = _load_jsonl(root / GENERATION_FILE)
    scoring = _load_jsonl(root / SCORING_FILE)
    handoff = _load_jsonl(root / HANDOFF_FILE)
    if anchors.get("schema_version") != ANCHOR_SCHEMA:
        raise ValueError("anchor schema drift")
    if inventory[ANCHORS_FILE].get("row_count") != 1:
        raise ValueError("anchor file row-count drift")
    selected_anchor_ids = anchors.get("selection", {}).get(
        "anchors_in_target_point_order"
    )
    if selected_anchor_ids != list(EXPECTED_W_ANCHORS):
        raise ValueError("anchor payload selection drift")
    anchor_rows = anchors.get("anchors")
    if not isinstance(anchor_rows, list) or [
        int(item["cluster_id"]) for item in anchor_rows
    ] != list(EXPECTED_W_ANCHORS):
        raise ValueError("anchor payload order drift")
    for rows, schema, name in (
        (generation, GENERATION_EVIDENCE_SCHEMA, GENERATION_FILE),
        (scoring, SCORING_EVIDENCE_SCHEMA, SCORING_FILE),
        (handoff, ARM_HANDOFF_SCHEMA, HANDOFF_FILE),
    ):
        if len(rows) != inventory[name].get("row_count") or any(
            row.get("schema_version") != schema for row in rows
        ):
            raise ValueError(f"candidate labeling row inventory drift: {name}")
    if any(row.get("family_partition") != "generation" for row in generation):
        raise ValueError("generation evidence contains held-out partitions")
    if any(row.get("prompt_eligible") is not True for row in generation):
        raise ValueError("generation evidence prompt firewall drift")
    if any(
        row.get("family_partition") not in {"selection_scoring", "audit"}
        or row.get("prompt_eligible") is not False
        for row in scoring
    ):
        raise ValueError("scoring evidence prompt firewall drift")
    if verify_sources:
        revision = manifest.get("code_revision")
        if not isinstance(revision, Mapping):
            raise TypeError("candidate labeling code revision is invalid")
        repo_root = Path(str(revision["repo_root"]))
        if collect_candidate_labeling_revision(repo_root) != revision:
            raise ValueError("candidate labeling code revision drift")
        input_record = manifest["source_input_bundle"]
        baseline_record = manifest["source_clustering_baseline"]
        evaluation_record = manifest["source_labelability_evaluation"]
        bundle = load_candidate_cluster_input_bundle(
            Path(str(input_record["manifest_path"])).parent
        )
        baseline = load_candidate_clustering_baseline(
            Path(str(baseline_record["manifest_path"])).parent,
            verify_source=True,
        )
        report_path = Path(str(evaluation_record["path"]))
        report = _load_labelability_evaluation_portably(report_path)
        if (
            bundle.manifest["manifest_sha256"] != input_record["manifest_sha256"]
            or baseline.manifest["manifest_sha256"]
            != baseline_record["manifest_sha256"]
            or report["manifest_sha256"] != evaluation_record["manifest_sha256"]
            or bundle.manifest["schema_version"] != input_record["schema_version"]
            or baseline.manifest["schema_version"] != baseline_record["schema_version"]
            or file_sha256(Path(str(input_record["manifest_path"])))
            != input_record["manifest_file_sha256"]
            or file_sha256(Path(str(baseline_record["manifest_path"])))
            != baseline_record["manifest_file_sha256"]
            or report["schema_version"] != evaluation_record["schema_version"]
            or file_sha256(report_path) != evaluation_record["file_sha256"]
        ):
            raise ValueError("candidate labeling deep source binding drift")
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_ID,
                revision=TOKENIZER_REVISION,
                local_files_only=True,
            )
        actual_tokenizer = _validated_tokenizer_identity(tokenizer)
        if dict(tokenizer_record) != actual_tokenizer:
            raise ValueError("candidate labeling persisted tokenizer identity drift")
        recomputed = build_candidate_labeling_comparison(
            bundle=bundle,
            baseline=baseline,
            evaluation_report=report,
            tokenizer=tokenizer,
        )
        persisted = (
            anchors,
            list(generation),
            list(scoring),
            list(handoff),
        )
        if recomputed != persisted:
            raise ValueError("candidate labeling derived evidence recomputation drift")
    return LoadedCandidateLabelingComparison(
        root=root,
        manifest=manifest,
        anchors=anchors,
        generation_evidence=generation,
        scoring_evidence=scoring,
        arm_handoff=handoff,
    )
