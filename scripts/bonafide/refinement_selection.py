"""Select a frozen BonaFide corpus and build second-stage refinement probes.

The selector treats screening as a measurement, not as ground truth.  It
validates reviewed prompt membership and its exact semantic/workload balances;
feature signatures remain diagnostics rather than selection evidence.  Dense
prompts probe every response position, while broad prompts probe at most 64
annotation-, answer-, source-, boundary-, and phase-centered candidates.  One
resident-model wave holds every independent probe.  Final broad trace targets
remain unfrozen until these refinement probes are analyzed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from circuits.tracing.probe_artifact import load_probe_artifact
from scripts.bonafide.manifest import SCHEMA_VERSION as TRACE_MANIFEST_SCHEMA
from scripts.bonafide.screening_manifest import (
    DEFAULT_SELECTION_PATH,
    _runner_example,
    _runtime_response_ids,
    _validate_candidate_contract,
    _validate_tokenizer_provenance,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase


SCHEMA_VERSION = "bonafide-refinement-selection/v1"
DEFAULT_DENSE_COUNT = 11
DEFAULT_BROAD_COUNT = 32
DEFAULT_EXPECTED_DENSE_PROBE_COUNT = 2_083
EXPECTED_SCREENING_TARGETS = 16
DEFAULT_DENSE_AUGMENTATION_IDS = ("bf-d8f174d2963759f617ca",)
DEFAULT_BROAD_HOLDOUT_IDS = (
    "bf-e2bba4914227c5830e59",
    "bf-1b8802d78871354168cb",
    "bf-294d5f48dfbed4c92866",
    "bf-3542009c51ab7938bd49",
    "bf-dad1d72a7b3cc093de01",
    "bf-e4395d5647d0b8dd68b1",
    "bf-cf7828e52206a450c3ca",
    "bf-25e42b77af628f80d8e5",
)
DEFAULT_BROAD_DISCOVERY_IDS = (
    "bf-0ab256ceb38d6f79758e",
    "bf-0b1309e47c3d3c24df11",
    "bf-1367b1431d97e196ad11",
    "bf-1827ee6d68003baef76d",
    "bf-404ecf1756e799781214",
    "bf-46f86d93e0e79429ff06",
    "bf-4f3bab852b0bea33fe6d",
    "bf-702c6535d2bc1a329992",
    "bf-72c58caa775145db5022",
    "bf-79aac6260ceb0428e611",
    "bf-84336ff6834b479e2e24",
    "bf-90973487c4682297bcde",
    "bf-991a9a660802c0c22726",
    "bf-a77f6cef3645c4d92542",
    "bf-ab53638e0a4ae6268a7a",
    "bf-b2c2e6b6bf99608ed220",
    "bf-b8b1a50caa4a2585fa78",
    "bf-cbc6049df97456b88ce8",
    "bf-ce557b2d625d29790786",
    "bf-d6cb87f188c46b0e1871",
    "bf-daae96f5c8653ab73b43",
    "bf-e56c46688fe719ee09c4",
    "bf-e7ff98220aca12d84dbe",
    "bf-f1532030de2282bb3f28",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_hash(path: Path, expected: str | None, label: str) -> str:
    actual = _file_sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"{label} SHA-256 changed: expected {expected}, found {actual}")
    return actual


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _jaccard(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class ScreenedTarget:
    source_item: Mapping[str, Any]
    summary: Mapping[str, Any]
    feature_ids: frozenset[tuple[int, int]]

    @property
    def position(self) -> int:
        return int(self.source_item["target_selection"]["response_token_positions"][0])


@dataclass(frozen=True)
class PromptAggregate:
    example: Mapping[str, Any]
    inventory: str
    targets: tuple[ScreenedTarget, ...]
    workload: Mapping[str, Any]
    feature_ids: frozenset[tuple[int, int]]
    feature_stability: float
    target_sensitivity: float
    workload_bin: str = "unassigned"

    @property
    def example_id(self) -> str:
        return str(self.example["example_id"])


def _read_summary(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank screening summary line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid screening summary JSON on line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"screening summary line {line_number} is not an object")
            records.append(value)
    return records


def _validate_screening_sources(
    *,
    candidate_selection: Mapping[str, Any],
    candidate_sha256: str,
    screening_manifest: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    _validate_candidate_contract(candidate_selection)
    if screening_manifest.get("artifact_kind") != "bonafide_prompt_screening_manifest":
        raise ValueError("input is not a BonaFide prompt-screening manifest")
    contract = screening_manifest.get("screening_contract")
    if not isinstance(contract, Mapping) or any(
        contract.get(field) is not expected
        for field, expected in {
            "probe_targets_frozen_for_this_estimation": True,
            "final_trace_prompt_membership_frozen": False,
            "final_trace_target_membership_frozen": False,
            "may_not_be_interpreted_as_final_trace_selection": True,
        }.items()
    ):
        raise ValueError("screening contract is missing or no longer provisional")
    source = screening_manifest.get("source_selection")
    if not isinstance(source, Mapping) or source.get("sha256") != candidate_sha256:
        raise ValueError("screening manifest is not bound to the candidate selection")
    waves = screening_manifest.get("waves")
    if not isinstance(waves, list) or len(waves) != 1:
        raise ValueError("screening manifest must contain exactly one wave")
    wave = waves[0]
    design = wave.get("screening_design") if isinstance(wave, Mapping) else None
    items = wave.get("items") if isinstance(wave, Mapping) else None
    if (
        not isinstance(design, Mapping)
        or design.get("targets_per_example") != EXPECTED_SCREENING_TARGETS
        or not isinstance(items, list)
        or not items
    ):
        raise ValueError("screening wave is missing its 16-target design")
    return items


def aggregate_screening(
    *,
    candidate_selection: Mapping[str, Any],
    candidate_sha256: str,
    screening_manifest: Mapping[str, Any],
    summary_records: Sequence[Mapping[str, Any]],
    artifact_root: Path,
) -> list[PromptAggregate]:
    """Validate every probe artifact and aggregate exactly 16 targets per prompt."""

    items = _validate_screening_sources(
        candidate_selection=candidate_selection,
        candidate_sha256=candidate_sha256,
        screening_manifest=screening_manifest,
    )
    candidate_examples = candidate_selection.get("examples")
    if not isinstance(candidate_examples, list):
        raise ValueError("candidate selection requires examples")
    candidate_by_id = {
        str(example["example_id"]): example
        for example in candidate_examples
        if isinstance(example, Mapping) and isinstance(example.get("example_id"), str)
    }
    if len(candidate_by_id) != len(candidate_examples):
        raise ValueError("candidate examples contain duplicate or invalid IDs")
    expected: dict[str, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("screening work item must be an object")
        source_id = item.get("artifact_id")
        if not isinstance(source_id, str) or source_id in expected:
            raise ValueError(f"duplicate or invalid screening source artifact: {source_id!r}")
        expected[source_id] = item

    by_source: dict[str, Mapping[str, Any]] = {}
    for record in summary_records:
        source_id = record.get("source_artifact_id")
        if not isinstance(source_id, str) or source_id in by_source:
            raise ValueError(f"duplicate or invalid summary source artifact: {source_id!r}")
        if source_id not in expected:
            raise ValueError(f"summary contains an unknown source artifact: {source_id}")
        by_source[source_id] = record
    missing = sorted(set(expected) - set(by_source))
    if missing:
        raise ValueError(f"screening summary is incomplete; missing {len(missing)} records")

    root = artifact_root.resolve()
    grouped: dict[str, list[ScreenedTarget]] = defaultdict(list)
    seen_runtime_ids: set[str] = set()
    for source_id, item in expected.items():
        record = by_source[source_id]
        if record.get("status") != "complete":
            raise ValueError(f"screening probe is not complete: {source_id}")
        runtime_id = record.get("artifact_id")
        if not isinstance(runtime_id, str) or runtime_id in seen_runtime_ids:
            raise ValueError(f"duplicate or invalid runtime artifact: {runtime_id!r}")
        seen_runtime_ids.add(runtime_id)
        artifact_path_raw = record.get("artifact_path")
        if not isinstance(artifact_path_raw, str):
            raise ValueError(f"summary record lacks artifact_path: {source_id}")
        artifact_path = Path(artifact_path_raw).resolve()
        if artifact_path != root and root not in artifact_path.parents:
            raise ValueError(f"probe artifact is outside the declared root: {artifact_path}")
        loaded = load_probe_artifact(artifact_path)
        position = int(item["target_selection"]["response_token_positions"][0])
        example_id = str(item["example"]["example_id"])
        expected_fields = {
            "source_artifact_id": source_id,
            "artifact_id": runtime_id,
            "example_id": example_id,
            "response_token_count": int(item["response_token_count"]),
        }
        for field, value in expected_fields.items():
            if record.get(field) != value:
                raise ValueError(f"summary {field} disagrees for {source_id}")
        if record.get("target_response_positions") != [position]:
            raise ValueError(f"summary target position disagrees for {source_id}")
        if loaded.manifest.get("artifact_id") != runtime_id:
            raise ValueError(f"artifact identity disagrees for {source_id}")
        if loaded.manifest.get("source_artifact_id") != source_id:
            raise ValueError(f"artifact source identity disagrees for {source_id}")
        if loaded.manifest.get("bonafide_example", {}).get("example_id") != example_id:
            raise ValueError(f"artifact example disagrees for {source_id}")
        provenance = loaded.probe["target_provenance"]
        if provenance.get("response_token_position") != position:
            raise ValueError(f"probe target position disagrees for {source_id}")
        metrics = loaded.metrics
        for field in (
            "candidate_mlp_edge_count",
            "probe_selected_occurrence_count",
        ):
            counter = metrics.get("instrumentation", {}).get("counters", {}).get(field)
            if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
                raise ValueError(f"probe metric {field} is invalid for {source_id}")
        feature_ids = frozenset(
            (int(layer), int(neuron))
            for layer, neuron in loaded.probe["feature_basis_signature"]["feature_ids"]
        )
        grouped[example_id].append(
            ScreenedTarget(source_item=item, summary=record, feature_ids=feature_ids)
        )

    aggregates: list[PromptAggregate] = []
    for example_id, targets in sorted(grouped.items()):
        if len(targets) != EXPECTED_SCREENING_TARGETS:
            raise ValueError(
                f"example {example_id} has {len(targets)} probes; expected "
                f"{EXPECTED_SCREENING_TARGETS}"
            )
        targets.sort(key=lambda target: target.position)
        positions = [target.position for target in targets]
        if len(set(positions)) != EXPECTED_SCREENING_TARGETS:
            raise ValueError(f"example {example_id} has duplicate screened positions")
        first = targets[0].source_item
        screening_example = first["example"]
        example = candidate_by_id.get(example_id)
        if example is None:
            raise ValueError(f"screened example is absent from candidate selection: {example_id}")
        for field, value in screening_example.items():
            if example.get(field) != value:
                raise ValueError(
                    f"screening example field {field} disagrees with candidate {example_id}"
                )
        inventory = first["target_selection"]["screening_selection"][
            "candidate_inventory"
        ]
        if any(target.source_item["example"] != screening_example for target in targets):
            raise ValueError(f"screening examples disagree within {example_id}")
        if any(
            target.source_item["target_selection"]["screening_selection"][
                "candidate_inventory"
            ]
            != inventory
            for target in targets
        ):
            raise ValueError(f"candidate inventory disagrees within {example_id}")

        edges = [
            float(target.summary["instrumentation"]["counters"]["candidate_mlp_edge_count"])
            for target in targets
        ]
        occurrences = [float(target.summary["selected_occurrence_count"]) for target in targets]
        memory = [float(target.summary["cuda_peak_reserved_bytes"]) for target in targets]
        wall = [float(target.summary["probe_wall_seconds"]) for target in targets]
        union = frozenset().union(*(target.feature_ids for target in targets))
        pairwise = [
            _jaccard(set(left.feature_ids), set(right.feature_ids))
            for index, left in enumerate(targets)
            for right in targets[index + 1 :]
        ]
        sensitivity = (
            _coefficient_of_variation(edges) + _coefficient_of_variation(occurrences)
        ) / 2
        aggregates.append(
            PromptAggregate(
                example=example,
                inventory=str(inventory),
                targets=tuple(targets),
                workload={
                    "candidate_mlp_edge_count": {
                        "min": min(edges),
                        "p50": _quantile(edges, 0.5),
                        "p90": _quantile(edges, 0.9),
                        "max": max(edges),
                        "mean": statistics.fmean(edges),
                        "coefficient_of_variation": _coefficient_of_variation(edges),
                    },
                    "selected_occurrence_count": {
                        "min": min(occurrences),
                        "p50": _quantile(occurrences, 0.5),
                        "p90": _quantile(occurrences, 0.9),
                        "max": max(occurrences),
                        "mean": statistics.fmean(occurrences),
                        "coefficient_of_variation": _coefficient_of_variation(occurrences),
                    },
                    "cuda_peak_reserved_bytes": {
                        "p50": _quantile(memory, 0.5),
                        "p90": _quantile(memory, 0.9),
                        "max": max(memory),
                    },
                    "probe_wall_seconds": {
                        "p50": _quantile(wall, 0.5),
                        "p90": _quantile(wall, 0.9),
                        "max": max(wall),
                    },
                },
                feature_ids=union,
                feature_stability=statistics.fmean(pairwise),
                target_sensitivity=sensitivity,
            )
        )
    expected_example_count = len(items) // EXPECTED_SCREENING_TARGETS
    if len(aggregates) != expected_example_count:
        raise ValueError("screening prompt aggregation is incomplete")
    return _assign_workload_bins(aggregates)


def _assign_workload_bins(aggregates: Sequence[PromptAggregate]) -> list[PromptAggregate]:
    result: list[PromptAggregate] = []
    for inventory in ("dense_inventory", "broad_eligible_inventory"):
        subset = [item for item in aggregates if item.inventory == inventory]
        edges = [float(item.workload["candidate_mlp_edge_count"]["p90"]) for item in subset]
        lower, upper = _quantile(edges, 1 / 3), _quantile(edges, 2 / 3)
        for item in subset:
            value = float(item.workload["candidate_mlp_edge_count"]["p90"])
            label = "low" if value <= lower else "middle" if value <= upper else "high"
            result.append(
                PromptAggregate(
                    example=item.example,
                    inventory=item.inventory,
                    targets=item.targets,
                    workload=item.workload,
                    feature_ids=item.feature_ids,
                    feature_stability=item.feature_stability,
                    target_sensitivity=item.target_sensitivity,
                    workload_bin=label,
                )
            )
    return sorted(result, key=lambda item: item.example_id)


def _coverage_tokens(example: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    diversity = example.get("diversity", {})
    axes = (
        "cot_phenotype",
        "answer_relation",
        "annotation_position_bin",
        "response_length_bin",
        "total_length_bin",
        "question_novelty_control_family_marker",
    )
    for axis in axes:
        value = diversity.get(axis)
        if isinstance(value, str):
            tokens.add(f"{axis}:{value}")
    for axis in ("label_types", "hint_types", "hint_datasets", "src_types"):
        values = example.get(axis, [])
        if isinstance(values, list):
            tokens.update(f"{axis}:{value}" for value in values)
    return tokens


def select_prompts(
    aggregates: Sequence[PromptAggregate],
    *,
    inventory: str,
    count: int,
    anchor_reasons: Mapping[str, str] | None = None,
) -> list[tuple[PromptAggregate, Mapping[str, Any]]]:
    """Greedily select prompts with explicit marginal-selection provenance."""

    candidates = [item for item in aggregates if item.inventory == inventory]
    if not 1 <= count <= len(candidates):
        raise ValueError(f"cannot select {count} prompts from {len(candidates)} {inventory}")
    selected: list[tuple[PromptAggregate, Mapping[str, Any]]] = []
    covered: set[str] = set()
    features: set[tuple[int, int]] = set()
    workload_bins: set[str] = set()
    questions: set[str] = set()
    anchors = dict(anchor_reasons or {})
    unknown_anchors = sorted(set(anchors) - {item.example_id for item in candidates})
    if unknown_anchors:
        raise ValueError(f"selection anchors are outside {inventory}: {unknown_anchors}")
    if len(anchors) > count:
        raise ValueError("selection has more anchors than requested prompts")
    while len(selected) < count:
        ranked: list[tuple[tuple[float, ...], str, PromptAggregate, dict[str, Any]]] = []
        for candidate in candidates:
            if any(candidate.example_id == item.example_id for item, _ in selected):
                continue
            semantic_new = sorted(_coverage_tokens(candidate.example) - covered)
            feature_new = set(candidate.feature_ids) - features
            feature_gain = len(feature_new) / max(1, len(candidate.feature_ids))
            workload_gain = candidate.workload_bin not in workload_bins
            question_id = str(candidate.example.get("base_question_id", ""))
            question_gain = question_id not in questions
            membership = candidate.example.get("selection_membership", {})
            prior_bonus = (
                1.0
                if membership.get("recommended_dense_core") is True
                else 0.5
                if membership.get("broad_role") == "primary"
                else 0.0
            )
            is_anchor = candidate.example_id in anchors
            # The tuple keeps semantic coverage dominant while ensuring all
            # workload regimes and genuinely new feature bases are represented.
            score = (
                float(is_anchor),
                float(len(semantic_new)),
                float(workload_gain),
                float(question_gain if inventory == "broad_eligible_inventory" else 0),
                feature_gain,
                min(candidate.target_sensitivity, 3.0),
                candidate.feature_stability,
                prior_bonus,
                -float(candidate.workload["candidate_mlp_edge_count"]["p90"]),
            )
            ranked.append(
                (
                    score,
                    candidate.example_id,
                    candidate,
                    {
                        "selection_iteration": len(selected),
                        "anchor_reason": anchors.get(candidate.example_id),
                        "new_semantic_coverage": semantic_new,
                        "new_workload_bin": candidate.workload_bin if workload_gain else None,
                        "new_feature_count": len(feature_new),
                        "screened_feature_count": len(candidate.feature_ids),
                        "new_feature_fraction": feature_gain,
                        "new_base_question": question_gain,
                        "prior_candidate_role": (
                            "recommended_dense_core"
                            if membership.get("recommended_dense_core") is True
                            else membership.get("broad_role")
                        ),
                        "ranking_tuple": list(score),
                    },
                )
            )
        ranked.sort(key=lambda row: (tuple(-value for value in row[0]), row[1]))
        _, _, winner, reason = ranked[0]
        selected.append((winner, reason))
        covered.update(_coverage_tokens(winner.example))
        features.update(winner.feature_ids)
        workload_bins.add(winner.workload_bin)
        questions.add(str(winner.example.get("base_question_id", "")))
    return selected


def _single_value(example: Mapping[str, Any], field: str) -> str:
    values = example.get(field)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
        raise ValueError(f"broad constraint field {field} must contain exactly one value")
    return values[0]


def validate_frozen_broad_prompts(
    aggregates: Sequence[PromptAggregate],
    *,
    holdout_ids: Sequence[str] = DEFAULT_BROAD_HOLDOUT_IDS,
    discovery_ids: Sequence[str] = DEFAULT_BROAD_DISCOVERY_IDS,
) -> list[tuple[PromptAggregate, Mapping[str, Any]]]:
    """Validate and return the reviewed 8-holdout/24-discovery membership."""

    if len(holdout_ids) != 8 or len(discovery_ids) != 24:
        raise ValueError("frozen broad selection requires 8 holdout and 24 discovery IDs")
    if len(set(holdout_ids) | set(discovery_ids)) != 32:
        raise ValueError("frozen broad prompt IDs must be unique and disjoint")
    broad = {
        item.example_id: item
        for item in aggregates
        if item.inventory == "broad_eligible_inventory"
    }
    missing = sorted((set(holdout_ids) | set(discovery_ids)) - set(broad))
    if missing:
        raise ValueError(f"frozen broad prompt IDs are absent: {missing}")
    holdout = [broad[example_id] for example_id in holdout_ids]
    discovery = [broad[example_id] for example_id in discovery_ids]
    holdout_families = {str(item.example["base_question_id"]) for item in holdout}
    discovery_families = [str(item.example["base_question_id"]) for item in discovery]
    if holdout_families.intersection(discovery_families):
        raise ValueError("holdout base-question families leaked into discovery")
    if len(set(discovery_families)) != 24:
        raise ValueError("discovery prompts must have unique base-question families")

    def counts(items: Sequence[PromptAggregate], getter) -> dict[str, int]:
        result: dict[str, int] = defaultdict(int)
        for item in items:
            result[str(getter(item))] += 1
        return dict(sorted(result.items()))

    holdout_phenotypes = counts(
        holdout, lambda item: item.example["diversity"]["cot_phenotype"]
    )
    expected_holdout = {"both": 2, "commission": 2, "faithful": 2, "omission": 2}
    if holdout_phenotypes != expected_holdout:
        raise ValueError(f"holdout phenotype balance changed: {holdout_phenotypes}")
    exact_discovery = {
        "phenotype": (
            counts(discovery, lambda item: item.example["diversity"]["cot_phenotype"]),
            {"both": 6, "commission": 6, "faithful": 6, "omission": 6},
        ),
        "hint_type": (
            counts(discovery, lambda item: _single_value(item.example, "hint_types")),
            {
                "error_message": 4,
                "metadata": 4,
                "security_audit": 4,
                "sycophancy": 4,
                "unauthorized_access": 4,
                "validator": 4,
            },
        ),
        "response_length_bin": (
            counts(discovery, lambda item: item.example["diversity"]["response_length_bin"]),
            {"225-384": 8, "385-512": 8, "513-768": 8},
        ),
        "workload_bin": (
            counts(discovery, lambda item: item.workload_bin),
            {"high": 8, "low": 8, "middle": 8},
        ),
        "hint_dataset": (
            counts(discovery, lambda item: _single_value(item.example, "hint_datasets")),
            {
                "aai530-group6_ddxplus": 7,
                "cais_hle": 5,
                "google_simpleqa-verified": 12,
            },
        ),
    }
    for axis, (actual, expected) in exact_discovery.items():
        if actual != expected:
            raise ValueError(f"frozen broad discovery {axis} balance changed: {actual}")

    selected: list[tuple[PromptAggregate, Mapping[str, Any]]] = []
    for index, item in enumerate(holdout):
        selected.append(
            (
                item,
                {
                    "selection_partition": "confirmatory_holdout",
                    "frozen_membership_index": index,
                    "holdout_family_locked_out_of_discovery": True,
                    "feature_signatures_used_for_selection": False,
                },
            )
        )
    for index, item in enumerate(discovery):
        selected.append(
            (
                item,
                {
                    "selection_partition": "discovery",
                    "frozen_membership_index": index,
                    "validated_exact_balances": sorted(exact_discovery),
                    "unique_base_question_family": True,
                    "feature_signatures_used_only_as_reviewed_tie_break": True,
                },
            )
        )
    return selected


def _evenly_spaced(values: Sequence[int], count: int) -> list[int]:
    ordered = sorted(set(values))
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    return [ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count)]


def _broad_refinement_targets(
    *,
    tokenizer: PreTrainedTokenizerBase,
    response: str,
    response_ids: Sequence[int],
    example: Mapping[str, Any],
    cap: int = 64,
    micro_window_radius: int = 2,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Choose deterministic semantic/boundary refinement candidates.

    Offsets must exactly reproduce teacher-forced response IDs; this prevents a
    tokenizer-boundary approximation from silently shifting annotation anchors.
    """

    if cap < 16:
        raise ValueError("broad refinement cap must leave room for 16 phase controls")
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    standalone_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if standalone_ids != list(response_ids) or len(offsets) != len(response_ids):
        raise ValueError("standalone response offsets disagree with teacher-forced token IDs")
    reasons: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def token_for_char(character: int) -> int:
        character = min(max(character, 0), max(0, len(response) - 1))
        for index, (start, end) in enumerate(offsets):
            if start <= character < end:
                return index
        return min(
            range(len(offsets)),
            key=lambda index: (abs(offsets[index][0] - character), index),
        )

    def add(position: int, reason: Mapping[str, Any]) -> None:
        if 0 <= position < len(response_ids):
            canonical = deepcopy(dict(reason))
            if canonical not in reasons[position]:
                reasons[position].append(canonical)

    def add_window(center: int, reason: Mapping[str, Any]) -> None:
        for position in range(center - micro_window_radius, center + micro_window_radius + 1):
            add(
                position,
                {
                    **dict(reason),
                    "micro_window_center": center,
                    "micro_window_offset": position - center,
                },
            )

    phase_positions = _evenly_spaced(list(range(len(response_ids))), 16)
    for phase_index, position in enumerate(phase_positions):
        add(
            position,
            {
                "reason_type": "phase_control",
                "phase_index": phase_index,
                "phase_count": len(phase_positions),
            },
        )

    annotation_spans = example.get("annotation_spans", [])
    if isinstance(annotation_spans, list):
        for annotation in annotation_spans:
            if not isinstance(annotation, Mapping):
                continue
            for kind in ("sentence", "extract"):
                if annotation.get(f"{kind}_span_valid") is not True:
                    continue
                start = int(annotation[f"{kind}_span_start"])
                end = int(annotation[f"{kind}_span_end"])
                for boundary, character in (
                    ("start", start),
                    ("center", (start + end - 1) // 2),
                    ("end", end - 1),
                ):
                    add_window(
                        token_for_char(character),
                        {
                            "reason_type": "bonafide_annotation_anchor",
                            "annotation_row_id": annotation.get("annotation_row_id"),
                            "label_type": annotation.get("label_type"),
                            "span_kind": kind,
                            "span_boundary": boundary,
                            "character_span": [start, end],
                        },
                    )

    lower_response = response.casefold()
    phrases: list[tuple[str, str]] = []
    answer_records = example.get("answer_records", [])
    if isinstance(answer_records, list):
        for record in answer_records:
            if isinstance(record, Mapping):
                for field in ("model_answer", "hinted_answer", "correct_answer"):
                    phrase = record.get(field)
                    if isinstance(phrase, str) and phrase.strip():
                        phrases.append((field, phrase.strip()))
    source_markers = (
        "hint",
        "provided",
        "given",
        "according",
        "source",
        "professor",
        "indicates",
        "states",
        "suggests",
        "clue",
        "told",
        "information",
    )
    phrases.extend(("source_marker", marker) for marker in source_markers)
    for phrase_type, phrase in sorted(set(phrases)):
        needle = phrase.casefold()
        start = 0
        occurrence_index = 0
        while True:
            found = lower_response.find(needle, start)
            if found < 0:
                break
            for boundary, character in (("start", found), ("end", found + len(phrase) - 1)):
                add_window(
                    token_for_char(character),
                    {
                        "reason_type": "answer_or_source_anchor",
                        "phrase_type": phrase_type,
                        "phrase": phrase,
                        "occurrence_index": occurrence_index,
                        "phrase_boundary": boundary,
                        "character_span": [found, found + len(phrase)],
                    },
                )
            occurrence_index += 1
            start = found + max(1, len(needle))

    boundary_positions = []
    for index, (start, end) in enumerate(offsets):
        fragment = response[start:end]
        if any(character in fragment for character in ("\n", ".", "?", "!", ":", ";")):
            boundary_positions.append(index)
    for position in _evenly_spaced(boundary_positions, 16):
        add_window(
            position,
            {
                "reason_type": "semantic_boundary_control",
                "token_character_span": list(offsets[position]),
            },
        )

    priority = {
        "bonafide_annotation_anchor": 0,
        "answer_or_source_anchor": 1,
        "phase_control": 2,
        "semantic_boundary_control": 3,
    }
    ranked = sorted(
        reasons,
        key=lambda position: (
            min(priority[reason["reason_type"]] for reason in reasons[position]),
            position,
        ),
    )
    selected = set(ranked[:cap])
    # Phase controls are non-negotiable; replace the lowest-priority positions
    # if unusually many annotation/source windows consume the cap.
    missing_phase = [position for position in phase_positions if position not in selected]
    for position in missing_phase:
        removable = sorted(
            (candidate for candidate in selected if candidate not in phase_positions),
            key=lambda candidate: (
                min(priority[reason["reason_type"]] for reason in reasons[candidate]),
                candidate,
            ),
            reverse=True,
        )
        if not removable:
            raise ValueError("cannot preserve phase controls within broad refinement cap")
        selected.remove(removable[0])
        selected.add(position)
    return [
        (
            position,
            {
                "policy": "semantic_boundary_refinement_candidates",
                "candidate_cap": cap,
                "micro_window_radius": micro_window_radius,
                "reasons": reasons[position],
            },
        )
        for position in sorted(selected)
    ]


def _trace_item(
    *,
    wave_id: str,
    aggregate: PromptAggregate,
    position: int,
    token_id: int,
    prompt_reason: Mapping[str, Any],
    target_reason: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "wave_id": wave_id,
        "example_id": aggregate.example_id,
        "response_token_position": position,
        "token_id": token_id,
    }
    return {
        "artifact_id": "probe-source-" + _sha256_bytes(_canonical_json(identity))[:24],
        "example": _runner_example(aggregate.example),
        "response_token_count": int(aggregate.example["token_counts"]["response"]),
        "target_selection": {
            "kind": "explicit_response_positions",
            "response_token_positions": [position],
            "width": 1,
            "final_target_token_id": token_id,
            "refinement_selection": {
                "corpus_role": (
                    "dense_full_response_refinement"
                    if aggregate.inventory == "dense_inventory"
                    else "broad_semantic_boundary_refinement"
                ),
                "prompt_selection_reason": deepcopy(dict(prompt_reason)),
                "target_selection_reason": deepcopy(dict(target_reason)),
                "screening_workload_bin": aggregate.workload_bin,
            },
        },
        "objective": {
            "name": "single_selected_logit",
            "benchmark_only_multi_target": False,
        },
    }


def build_refinement_probe_manifest(
    *,
    candidate_selection: Mapping[str, Any],
    candidate_path: Path,
    candidate_sha256: str,
    screening_manifest: Mapping[str, Any],
    screening_manifest_path: Path,
    screening_manifest_sha256: str,
    summary_path: Path,
    summary_sha256: str,
    aggregates: Sequence[PromptAggregate],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_path: Path,
    dense_count: int = DEFAULT_DENSE_COUNT,
    broad_count: int = DEFAULT_BROAD_COUNT,
    dense_augmentation_ids: Sequence[str] = DEFAULT_DENSE_AUGMENTATION_IDS,
    broad_holdout_ids: Sequence[str] = DEFAULT_BROAD_HOLDOUT_IDS,
    broad_discovery_ids: Sequence[str] = DEFAULT_BROAD_DISCOVERY_IDS,
    broad_candidate_cap: int = 64,
    expected_dense_probe_count: int | None = DEFAULT_EXPECTED_DENSE_PROBE_COUNT,
) -> dict[str, Any]:
    """Freeze prompts and emit the second-stage graph-free refinement probes."""

    tokenizer_meta = screening_manifest["tokenizer"]
    _validate_tokenizer_provenance(
        selection=candidate_selection,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        model_id=str(tokenizer_meta["model_id"]),
        model_revision=str(tokenizer_meta["revision"]),
    )
    recommended_dense_ids = {
        item.example_id
        for item in aggregates
        if item.inventory == "dense_inventory"
        and item.example.get("selection_membership", {}).get("recommended_dense_core")
        is True
    }
    dense_anchor_reasons = {
        example_id: "pre-screening recommended dense core"
        for example_id in recommended_dense_ids
    }
    for example_id in dense_augmentation_ids:
        dense_anchor_reasons[example_id] = (
            "same-prompt phenotype validator contrast: commission versus both"
        )
    dense = select_prompts(
        aggregates,
        inventory="dense_inventory",
        count=dense_count,
        anchor_reasons=dense_anchor_reasons,
    )
    if broad_count != 32:
        raise ValueError("reviewed frozen broad selection contains exactly 32 prompts")
    broad = validate_frozen_broad_prompts(
        aggregates,
        holdout_ids=broad_holdout_ids,
        discovery_ids=broad_discovery_ids,
    )
    dense_items: list[dict[str, Any]] = []
    broad_items: list[dict[str, Any]] = []
    prompt_analysis: list[dict[str, Any]] = []

    for corpus_role, selected in (("dense", dense), ("broad", broad)):
        for aggregate, reason in selected:
            response_ids = _runtime_response_ids(tokenizer, aggregate.example)
            if corpus_role == "dense":
                targets = [
                    (
                        position,
                        int(token_id),
                        {
                            "policy": "all_teacher_forced_response_positions",
                            "full_response_coverage": True,
                        },
                    )
                    for position, token_id in enumerate(response_ids)
                ]
                destination = dense_items
                wave_id = "prompt-refinement-probes"
            else:
                targets = [
                    (position, int(response_ids[position]), target_reason)
                    for position, target_reason in _broad_refinement_targets(
                        tokenizer=tokenizer,
                        response=str(aggregate.example["response"]),
                        response_ids=response_ids,
                        example=aggregate.example,
                        cap=broad_candidate_cap,
                    )
                ]
                destination = broad_items
                wave_id = "prompt-refinement-probes"
            for position, token_id, target_reason in targets:
                destination.append(
                    _trace_item(
                        wave_id=wave_id,
                        aggregate=aggregate,
                        position=position,
                        token_id=token_id,
                        prompt_reason=reason,
                        target_reason=target_reason,
                    )
                )
            prompt_analysis.append(
                {
                    "example_id": aggregate.example_id,
                    "corpus_role": corpus_role,
                    "example_metadata": deepcopy(dict(aggregate.example)),
                    "selection_reason": deepcopy(dict(reason)),
                    "screening_workload_bin": aggregate.workload_bin,
                    "screening_workload": deepcopy(dict(aggregate.workload)),
                    "screened_feature_union_count": len(aggregate.feature_ids),
                    "cross_target_feature_jaccard_mean": aggregate.feature_stability,
                    "target_workload_sensitivity": aggregate.target_sensitivity,
                    "refinement_target_count": len(targets),
                }
            )

    if (
        expected_dense_probe_count is not None
        and len(dense_items) != expected_dense_probe_count
    ):
        raise ValueError(
            "dense all-position probe count changed: expected "
            f"{expected_dense_probe_count}, found {len(dense_items)}"
        )

    return {
        "schema_version": TRACE_MANIFEST_SCHEMA,
        "artifact_kind": "bonafide_refinement_probe_manifest",
        "selection_schema_version": SCHEMA_VERSION,
        "selection_contract": {
            "prompt_membership_frozen": True,
            "refinement_probe_membership_frozen": True,
            "final_trace_target_membership_frozen": False,
            "dense_refinement_policy": "all_teacher_forced_response_positions",
            "broad_refinement_policy": "up_to_64_semantic_boundary_candidates",
            "post_refinement_broad_policy": "freeze_16_event_centered_targets",
            "trace_units_are_independent": True,
            "merge_graphs": False,
            "selection_is_probe_informed_not_causal_evidence": True,
        },
        "source_artifacts": {
            "candidate_selection": {
                "path": str(candidate_path),
                "sha256": candidate_sha256,
            },
            "screening_manifest": {
                "path": str(screening_manifest_path),
                "sha256": screening_manifest_sha256,
            },
            "screening_summary": {
                "path": str(summary_path),
                "sha256": summary_sha256,
            },
        },
        "dataset": deepcopy(candidate_selection["dataset"]),
        "tokenizer": deepcopy(screening_manifest["tokenizer"]),
        "execution_contract": {
            "batch_size": 1,
            "trace_units_are_independent": True,
            "merge_graphs": False,
        },
        "selection_policy": {
            "version": "bonafide-two-stage-refinement-v1",
            "dense_prompt_count": dense_count,
            "dense_anchor_reasons": dense_anchor_reasons,
            "broad_prompt_count": broad_count,
            "broad_holdout_ids": list(broad_holdout_ids),
            "broad_discovery_ids": list(broad_discovery_ids),
            "broad_refinement_candidate_cap": broad_candidate_cap,
            "semantic_axes": sorted({
                token.split(":", 1)[0]
                for aggregate in aggregates
                for token in _coverage_tokens(aggregate.example)
            }),
            "workload_axis": "within-inventory tertiles of screened p90 candidate MLP edges",
            "feature_axis": "diagnostic only; reviewed membership is frozen explicitly",
            "uncertainty_axis": "mean CV of target edge and selected-occurrence counts",
            "tie_break": "ascending example_id",
        },
        "prompt_analysis": sorted(prompt_analysis, key=lambda row: row["example_id"]),
        "waves": [
            {
                "wave_id": "prompt-refinement-probes",
                "purpose": (
                    "one resident-model graph-free wave: all dense positions plus "
                    "broad semantic/boundary candidates before the final 16-target freeze"
                ),
                "refinement_design": {
                    "dense_probe_count": len(dense_items),
                    "broad_probe_count": len(broad_items),
                    "model_load_scope": "selected_wave",
                    "resident_model_across_wave": True,
                },
                "items": [*dense_items, *broad_items],
            }
        ],
    }


def write_manifest(manifest: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--expected-screening-manifest-sha256")
    parser.add_argument("--screening-summary", type=Path, required=True)
    parser.add_argument("--expected-screening-summary-sha256")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--dense-count", type=int, default=DEFAULT_DENSE_COUNT)
    parser.add_argument("--broad-count", type=int, default=DEFAULT_BROAD_COUNT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    candidate_sha = _require_hash(
        args.candidate_selection, args.expected_candidate_sha256, "candidate selection"
    )
    manifest_sha = _require_hash(
        args.screening_manifest,
        args.expected_screening_manifest_sha256,
        "screening manifest",
    )
    summary_sha = _require_hash(
        args.screening_summary,
        args.expected_screening_summary_sha256,
        "screening summary",
    )
    candidate = _load_object(args.candidate_selection)
    screening = _load_object(args.screening_manifest)
    summary = _read_summary(args.screening_summary)
    aggregates = aggregate_screening(
        candidate_selection=candidate,
        candidate_sha256=candidate_sha,
        screening_manifest=screening,
        summary_records=summary,
        artifact_root=args.artifact_root,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    refinement = build_refinement_probe_manifest(
        candidate_selection=candidate,
        candidate_path=args.candidate_selection,
        candidate_sha256=candidate_sha,
        screening_manifest=screening,
        screening_manifest_path=args.screening_manifest,
        screening_manifest_sha256=manifest_sha,
        summary_path=args.screening_summary,
        summary_sha256=summary_sha,
        aggregates=aggregates,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer_path,
        dense_count=args.dense_count,
        broad_count=args.broad_count,
    )
    write_manifest(refinement, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _file_sha256(args.output),
                "dense_prompt_count": args.dense_count,
                "broad_prompt_count": args.broad_count,
                "dense_probe_count": refinement["waves"][0]["refinement_design"][
                    "dense_probe_count"
                ],
                "broad_probe_count": refinement["waves"][0]["refinement_design"][
                    "broad_probe_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
