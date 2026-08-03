"""Freeze final dense and broad BonaFide trace targets after refinement probes.

Probe artifacts are authoritative.  The append-only summary is audited but may
contain repeated ``complete``/``skipped_complete`` records from safe resume.
Conflicting artifacts for one frozen refinement item fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from circuits.tracing.probe_artifact import ProbeArtifact, load_probe_artifact

from scripts.bonafide.manifest import SCHEMA_VERSION as TRACE_MANIFEST_SCHEMA
from scripts.bonafide.runner import _sha256 as stable_object_sha256
from scripts.bonafide.runner import validate_target_selection

SCHEMA_VERSION = "bonafide-final-trace-selection/v1"
EXPECTED_BROAD_TARGETS = 16
EXPECTED_PHASE_INDICES = (0, 5, 10, 15)
EXTREME_EDGE_ISOLATION_THRESHOLD = 500_000


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


def _read_summary(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank refinement summary line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid refinement summary JSON on line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"refinement summary line {line_number} is not an object"
                )
            records.append(value)
    return records


@dataclass(frozen=True)
class RefinementProbe:
    item: Mapping[str, Any]
    artifact_path: Path
    artifact_id: str
    token_text: str
    logit: float
    probability: float
    candidate_edge_count: int
    selected_occurrence_count: int
    feature_ids: frozenset[tuple[int, int]]
    probe_sha256: str
    metrics_sha256: str
    cohort_identity: Mapping[str, Any]
    cohort_identity_sha256: str

    @property
    def position(self) -> int:
        return int(self.item["target_selection"]["response_token_positions"][0])

    @property
    def token_id(self) -> int:
        return int(self.item["target_selection"]["final_target_token_id"])


@dataclass(frozen=True)
class BroadCandidate:
    probe: RefinementProbe
    candidate_reasons: tuple[Mapping[str, Any], ...]

    @property
    def position(self) -> int:
        return self.probe.position


def _validate_refinement_manifest(
    manifest: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if manifest.get("artifact_kind") != "bonafide_refinement_probe_manifest":
        raise ValueError("input is not a BonaFide refinement-probe manifest")
    contract = manifest.get("selection_contract")
    if not isinstance(contract, Mapping) or any(
        contract.get(field) is not expected
        for field, expected in {
            "prompt_membership_frozen": True,
            "refinement_probe_membership_frozen": True,
            "final_trace_target_membership_frozen": False,
        }.items()
    ):
        raise ValueError("refinement manifest contract is missing or already final")
    waves = manifest.get("waves")
    if not isinstance(waves, list) or len(waves) != 1:
        raise ValueError("refinement manifest must contain one resident-model wave")
    items = waves[0].get("items") if isinstance(waves[0], Mapping) else None
    if not isinstance(items, list) or not items:
        raise ValueError("refinement manifest requires non-empty items")
    source_ids: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("refinement work item must be an object")
        validate_target_selection(item)
        source_id = item.get("artifact_id")
        if not isinstance(source_id, str) or source_id in source_ids:
            raise ValueError(
                f"duplicate or invalid refinement source ID: {source_id!r}"
            )
        source_ids.add(source_id)
    return items


def audit_append_only_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_source_ids: set[str],
    expected_wave_id: str,
) -> dict[str, Any]:
    """Audit an append-only run log without treating it as completion state."""

    counts: dict[str, int] = defaultdict(int)
    per_source: dict[str, list[str]] = defaultdict(list)
    complete_runtime_ids: dict[str, set[str]] = defaultdict(set)
    accepted = {"complete", "skipped_complete", "planned", "error", "oom"}
    for record in records:
        wave_id = record.get("wave_id")
        if wave_id != expected_wave_id:
            continue
        source_id = record.get("source_artifact_id")
        if source_id not in expected_source_ids:
            raise ValueError(
                f"summary contains unknown refinement source: {source_id!r}"
            )
        status = record.get("status")
        if status not in accepted:
            raise ValueError(f"unsupported refinement summary status: {status!r}")
        counts[str(status)] += 1
        per_source[str(source_id)].append(str(status))
        if status in {"complete", "skipped_complete"}:
            artifact_id = record.get("artifact_id")
            if not isinstance(artifact_id, str):
                raise ValueError("completed summary record requires artifact_id")
            complete_runtime_ids[str(source_id)].add(artifact_id)
    return {
        "record_count": sum(counts.values()),
        "status_counts": dict(sorted(counts.items())),
        "sources_with_records": len(per_source),
        "sources_with_repeated_records": sum(
            len(statuses) > 1 for statuses in per_source.values()
        ),
        "completed_runtime_ids": {
            source_id: sorted(ids)
            for source_id, ids in sorted(complete_runtime_ids.items())
        },
    }


def _probe_from_artifact(
    *, item: Mapping[str, Any], loaded: ProbeArtifact
) -> RefinementProbe:
    manifest = loaded.manifest
    source_id = str(item["artifact_id"])
    if manifest.get("source_artifact_id") != source_id:
        raise ValueError(f"probe artifact source disagrees for {source_id}")
    identity = manifest.get("artifact_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"probe artifact lacks identity for {source_id}")
    if identity.get("source_work_item_sha256") != stable_object_sha256(item):
        raise ValueError(f"probe artifact work-item hash disagrees for {source_id}")
    if identity.get("source_target_selection") != item["target_selection"]:
        raise ValueError(f"probe artifact target selection disagrees for {source_id}")
    cohort_identity = {
        field: deepcopy(identity.get(field))
        for field in ("mode", "batch_size", "model", "adag_config", "code_revision")
    }
    if any(value is None for value in cohort_identity.values()):
        raise ValueError(
            f"probe artifact cohort identity is incomplete for {source_id}"
        )
    cohort_model = cast(Mapping[str, Any], cohort_identity["model"])
    if manifest.get("model_revision") != cohort_model.get("revision"):
        raise ValueError(f"probe artifact model revision disagrees for {source_id}")
    if manifest.get("code_revision") != cohort_identity["code_revision"]:
        raise ValueError(f"probe artifact code revision disagrees for {source_id}")
    cohort_identity_sha256 = stable_object_sha256(cohort_identity)
    position = int(item["target_selection"]["response_token_positions"][0])
    token_id = int(item["target_selection"]["final_target_token_id"])
    provenance = cast(Mapping[str, Any], loaded.probe["target_provenance"])
    if provenance.get("response_token_position") != position:
        raise ValueError(f"probe position disagrees for {source_id}")
    if provenance.get("token_id") != token_id:
        raise ValueError(f"probe token ID disagrees for {source_id}")
    probability = provenance.get("probability")
    logit = provenance.get("logit")
    token_text = provenance.get("token_text")
    for value, label in ((probability, "probability"), (logit, "logit")):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"probe {label} is invalid for {source_id}")
    if not isinstance(token_text, str):
        raise ValueError(f"probe token text is invalid for {source_id}")
    instrumentation = cast(Mapping[str, Any], loaded.metrics.get("instrumentation", {}))
    counters = cast(Mapping[str, Any], instrumentation.get("counters", {}))
    edge_count = counters.get("candidate_mlp_edge_count")
    occurrences = loaded.metrics.get("selected_occurrence_count")
    for value, label in ((edge_count, "candidate edge"), (occurrences, "occurrence")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"probe {label} count is invalid for {source_id}")
    feature_basis = cast(Mapping[str, Any], loaded.probe["feature_basis_signature"])
    features = frozenset(
        (int(layer), int(neuron)) for layer, neuron in feature_basis["feature_ids"]
    )
    return RefinementProbe(
        item=item,
        artifact_path=loaded.path,
        artifact_id=str(manifest["artifact_id"]),
        token_text=token_text,
        logit=float(cast(int | float, logit)),
        probability=float(cast(int | float, probability)),
        candidate_edge_count=int(cast(int, edge_count)),
        selected_occurrence_count=cast(int, occurrences),
        feature_ids=features,
        probe_sha256=str(manifest["probe_sha256"]),
        metrics_sha256=str(manifest["metrics_sha256"]),
        cohort_identity=cohort_identity,
        cohort_identity_sha256=cohort_identity_sha256,
    )


def load_authoritative_refinement_probes(
    *,
    manifest: Mapping[str, Any],
    artifact_root: Path,
    summary_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, RefinementProbe], dict[str, Any]]:
    """Resolve every frozen item from validated artifacts, not summary order."""

    items = _validate_refinement_manifest(manifest)
    by_source = {str(item["artifact_id"]): item for item in items}
    wave = manifest["waves"][0]
    wave_id = str(wave["wave_id"])
    audit = audit_append_only_summary(
        summary_records,
        expected_source_ids=set(by_source),
        expected_wave_id=wave_id,
    )
    wave_root = artifact_root / wave_id
    if not wave_root.is_dir():
        raise ValueError(f"refinement artifact wave directory is absent: {wave_root}")
    candidates: dict[str, list[RefinementProbe]] = defaultdict(list)
    for manifest_path in sorted(wave_root.rglob("manifest.json")):
        try:
            raw = _load_object(manifest_path)
        except ValueError:
            raise
        source_id = raw.get("source_artifact_id")
        if source_id not in by_source:
            continue
        loaded = load_probe_artifact(manifest_path.parent)
        candidates[str(source_id)].append(
            _probe_from_artifact(item=by_source[str(source_id)], loaded=loaded)
        )
    missing = sorted(set(by_source) - set(candidates))
    if missing:
        raise ValueError(
            f"refinement artifacts are incomplete; missing {len(missing)} items"
        )
    resolved: dict[str, RefinementProbe] = {}
    duplicate_sources = 0
    for source_id, probes in sorted(candidates.items()):
        fingerprints = {
            (probe.probe_sha256, probe.metrics_sha256, probe.artifact_id)
            for probe in probes
        }
        if len(fingerprints) != 1:
            raise ValueError(f"conflicting authoritative artifacts for {source_id}")
        if len(probes) > 1:
            duplicate_sources += 1
        resolved[source_id] = min(probes, key=lambda probe: str(probe.artifact_path))
    cohort_audit = _audit_probe_cohort(resolved.values())
    completed_ids = audit.pop("completed_runtime_ids")
    for source_id, artifact_ids in completed_ids.items():
        if any(
            artifact_id != resolved[source_id].artifact_id
            for artifact_id in artifact_ids
        ):
            raise ValueError(
                f"summary runtime identity conflicts with artifact for {source_id}"
            )
    audit["authoritative_artifact_count"] = len(resolved)
    audit["duplicate_identical_artifact_sources"] = duplicate_sources
    audit.update(cohort_audit)
    return resolved, audit


def _audit_probe_cohort(probes: Iterable[RefinementProbe]) -> dict[str, Any]:
    probes = list(probes)
    if not probes:
        raise ValueError("cannot audit an empty refinement probe cohort")
    cohort_hashes = {probe.cohort_identity_sha256 for probe in probes}
    if len(cohort_hashes) != 1:
        raise ValueError(
            "refinement artifacts mix model, ADAG config, or code-revision identities"
        )
    cohort_hash = next(iter(cohort_hashes))
    cohort_identity = probes[0].cohort_identity
    if stable_object_sha256(cohort_identity) != cohort_hash:
        raise ValueError("refinement probe cohort identity hash is inconsistent")
    return {
        "homogeneous_probe_cohort": True,
        "probe_cohort_identity_sha256": cohort_hash,
        "probe_cohort_identity": deepcopy(dict(cohort_identity)),
    }


def _jaccard_distance(
    left: frozenset[tuple[int, int]], right: frozenset[tuple[int, int]]
) -> float:
    union = left | right
    return 1.0 - (len(left & right) / len(union) if union else 1.0)


def _candidate_reasons(item: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    selection = item["target_selection"].get("refinement_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("refinement item lacks refinement_selection")
    target_reason = selection.get("target_selection_reason")
    if not isinstance(target_reason, Mapping):
        raise ValueError("refinement item lacks target selection reason")
    reasons = target_reason.get("reasons", [])
    if not isinstance(reasons, list) or any(
        not isinstance(reason, Mapping) for reason in reasons
    ):
        raise ValueError("refinement candidate reasons must be a list of objects")
    return tuple(reasons)


def _reason_matches(candidate: BroadCandidate, predicate) -> list[Mapping[str, Any]]:
    return [reason for reason in candidate.candidate_reasons if predicate(reason)]


def _semantic_candidates(candidates: Sequence[BroadCandidate]) -> list[BroadCandidate]:
    return [
        candidate
        for candidate in candidates
        if any(
            reason.get("reason_type") != "phase_control"
            for reason in candidate.candidate_reasons
        )
    ]


def _select_window(
    candidates: Sequence[BroadCandidate], center: int, width: int = 3
) -> list[BroadCandidate]:
    nearby = sorted(
        candidates,
        key=lambda candidate: (abs(candidate.position - center), candidate.position),
    )
    chosen = sorted(nearby[:width], key=lambda candidate: candidate.position)
    if len(chosen) != width:
        raise ValueError("not enough candidates for semantic micro-window")
    return chosen


def _answer_anchor_center(
    candidates: Sequence[BroadCandidate], *, latest: bool
) -> tuple[int | None, Mapping[str, Any] | None]:
    matches: list[tuple[int, int, int, Mapping[str, Any]]] = []
    phrase_priority = {"hinted_answer": 0, "model_answer": 1}
    for candidate in candidates:
        for reason in _reason_matches(
            candidate,
            lambda value: (
                value.get("reason_type") == "answer_or_source_anchor"
                and value.get("phrase_type") in phrase_priority
                and _answer_phrase_is_reliable(value.get("phrase"))
            ),
        ):
            span = reason.get(
                "character_span", [candidate.position, candidate.position + 1]
            )
            boundary_priority = 0 if reason.get("phrase_boundary") == "end" else 1
            matches.append(
                (
                    int(span[0]),
                    phrase_priority[str(reason["phrase_type"])],
                    boundary_priority,
                    reason,
                )
            )
    if not matches:
        return None, None
    occurrence_starts = sorted({match[0] for match in matches})
    wanted = occurrence_starts[-1] if latest else occurrence_starts[0]
    selected = min(
        (match for match in matches if match[0] == wanted), key=lambda match: match[1:3]
    )
    reason = selected[3]
    return int(reason.get("micro_window_center", wanted)), reason


def _answer_phrase_is_reliable(value: Any) -> bool:
    """Reject answer strings likely to match incidental text (for example ``3``)."""

    if not isinstance(value, str):
        return False
    normalized = " ".join(value.split()).strip()
    if not normalized or re.fullmatch(r"[\d\W_]+", normalized):
        return False
    alphanumeric = "".join(character for character in normalized if character.isalnum())
    return len(alphanumeric) >= 4


STRONG_SOURCE_MARKERS = frozenset(
    {
        "hint",
        "according",
        "source",
        "professor",
        "indicates",
        "suggests",
        "clue",
        "told",
    }
)


def _is_strong_source_marker(reason: Mapping[str, Any]) -> bool:
    return (
        reason.get("reason_type") == "answer_or_source_anchor"
        and reason.get("phrase_type") == "source_marker"
        and isinstance(reason.get("phrase"), str)
        and str(reason["phrase"]).casefold().strip() in STRONG_SOURCE_MARKERS
    )


def _source_anchor_center(
    candidates: Sequence[BroadCandidate], *, phenotype: str
) -> tuple[int, Mapping[str, Any]]:
    annotation: list[tuple[int, Mapping[str, Any]]] = []
    source: list[tuple[int, Mapping[str, Any]]] = []
    for candidate in candidates:
        for reason in candidate.candidate_reasons:
            center = int(reason.get("micro_window_center", candidate.position))
            if reason.get("reason_type") == "bonafide_annotation_anchor":
                annotation.append((center, reason))
            elif _is_strong_source_marker(reason):
                source.append((center, reason))
    # Curated BonaFide spans are the strongest available evidence for both
    # faithful source acknowledgement and fabricated attribution.  Generic
    # lexical source markers are only a fallback when no valid span survived.
    preferred = annotation
    fallback_anchors = source
    if preferred or fallback_anchors:
        return min(preferred or fallback_anchors, key=lambda value: value[0])
    semantic = _semantic_candidates(candidates)
    fallback = min(
        semantic or list(candidates),
        key=lambda candidate: (candidate.probe.probability, candidate.position),
    )
    return fallback.position, {
        "reason_type": "unsupported_evidence_fallback",
        "phenotype": phenotype,
        "fallback_probability": fallback.probe.probability,
    }


def select_broad_targets(
    probes: Sequence[RefinementProbe], *, phenotype: str
) -> list[tuple[RefinementProbe, list[Mapping[str, Any]]]]:
    """Freeze exactly 16 targets using reviewed semantic and diagnostic buckets."""

    candidates = sorted(
        (
            BroadCandidate(
                probe=probe, candidate_reasons=_candidate_reasons(probe.item)
            )
            for probe in probes
        ),
        key=lambda candidate: candidate.position,
    )
    if len(candidates) < EXPECTED_BROAD_TARGETS:
        raise ValueError("broad refinement requires at least 16 candidates")
    selected: dict[int, tuple[RefinementProbe, list[Mapping[str, Any]]]] = {}

    def add(candidate: BroadCandidate, reason: Mapping[str, Any]) -> None:
        entry = selected.setdefault(candidate.position, (candidate.probe, []))
        canonical = deepcopy(dict(reason))
        if canonical not in entry[1]:
            entry[1].append(canonical)

    source_center, source_anchor = _source_anchor_center(
        candidates, phenotype=phenotype
    )
    first_center, first_anchor = _answer_anchor_center(candidates, latest=False)
    if first_center is None:
        first_center = source_center
        first_anchor = {
            "reason_type": "unreliable_or_missing_answer_anchor_fallback",
            "fallback": "curated_source_or_annotation_center",
            "source_anchor": deepcopy(dict(source_anchor)),
        }
    for candidate in _select_window(candidates, first_center):
        add(
            candidate,
            {
                "bucket": "first_hinted_answer_commitment_window",
                "window_center": first_center,
                "anchor": deepcopy(dict(first_anchor)) if first_anchor else None,
            },
        )

    for candidate in _select_window(candidates, source_center):
        add(
            candidate,
            {
                "bucket": "source_or_unsupported_evidence_window",
                "window_center": source_center,
                "anchor": deepcopy(dict(source_anchor)),
            },
        )

    final_center, final_anchor = _answer_anchor_center(candidates, latest=True)
    if final_center is None:
        final_phase_candidates = [
            (candidate, reason)
            for candidate in candidates
            for reason in candidate.candidate_reasons
            if reason.get("reason_type") == "phase_control"
        ]
        if final_phase_candidates:
            final_candidate, phase_reason = max(
                final_phase_candidates,
                key=lambda value: (
                    int(value[1].get("phase_index", -1)),
                    value[0].position,
                ),
            )
            final_center = final_candidate.position
            final_anchor = {
                "reason_type": "unreliable_or_missing_answer_anchor_fallback",
                "fallback": "final_phase_control",
                "phase_reason": deepcopy(dict(phase_reason)),
            }
        else:
            final_center = candidates[-1].position
            final_anchor = {
                "reason_type": "unreliable_or_missing_answer_anchor_fallback",
                "fallback": "response_tail",
            }
    for candidate in _select_window(candidates, final_center):
        add(
            candidate,
            {
                "bucket": "final_answer_commitment_window",
                "window_center": final_center,
                "anchor": deepcopy(dict(final_anchor)) if final_anchor else None,
            },
        )

    phase_candidates = [
        (candidate, reason)
        for candidate in candidates
        for reason in candidate.candidate_reasons
        if reason.get("reason_type") == "phase_control"
    ]
    for wanted in EXPECTED_PHASE_INDICES:
        if not phase_candidates:
            break
        candidate, reason = min(
            phase_candidates,
            key=lambda value: (
                abs(int(value[1].get("phase_index", -1)) - wanted),
                value[0].position,
            ),
        )
        add(
            candidate,
            {
                "bucket": "phase_control",
                "requested_phase_index": wanted,
                "source_reason": deepcopy(dict(reason)),
            },
        )

    semantic = _semantic_candidates(candidates) or list(candidates)
    low_probability = min(
        semantic,
        key=lambda candidate: (candidate.probe.probability, candidate.position),
    )
    add(
        low_probability,
        {
            "bucket": "low_probability_semantic",
            "probability": low_probability.probe.probability,
        },
    )

    adjacent_changes: list[tuple[float, BroadCandidate, BroadCandidate]] = []
    for left, right in pairwise(candidates):
        adjacent_changes.append(
            (
                _jaccard_distance(left.probe.feature_ids, right.probe.feature_ids),
                left,
                right,
            )
        )
    change, left, right = max(
        adjacent_changes,
        key=lambda value: (value[0], -value[1].position, -value[2].position),
    )
    add(
        right,
        {
            "bucket": "large_adjacent_feature_change",
            "left_position": left.position,
            "right_position": right.position,
            "feature_jaccard_distance": change,
        },
    )

    median_edges = statistics.median(
        candidate.probe.candidate_edge_count for candidate in candidates
    )
    median_workload = min(
        candidates,
        key=lambda candidate: (
            abs(candidate.probe.candidate_edge_count - median_edges),
            candidate.position,
        ),
    )
    add(
        median_workload,
        {
            "bucket": "median_workload_control",
            "candidate_edge_count": median_workload.probe.candidate_edge_count,
            "candidate_edge_median": median_edges,
        },
    )

    local_change: dict[int, float] = defaultdict(float)
    for distance, left_candidate, right_candidate in adjacent_changes:
        local_change[left_candidate.position] = max(
            local_change[left_candidate.position], distance
        )
        local_change[right_candidate.position] = max(
            local_change[right_candidate.position], distance
        )
    remaining = sorted(
        (candidate for candidate in candidates if candidate.position not in selected),
        key=lambda candidate: (
            -sum(
                reason.get("reason_type") != "phase_control"
                for reason in candidate.candidate_reasons
            ),
            -local_change[candidate.position],
            candidate.probe.probability,
            abs(candidate.probe.candidate_edge_count - median_edges),
            candidate.position,
        ),
    )
    for candidate in remaining:
        if len(selected) == EXPECTED_BROAD_TARGETS:
            break
        add(
            candidate,
            {
                "bucket": "deterministic_refill_after_deduplication",
                "semantic_reason_count": sum(
                    reason.get("reason_type") != "phase_control"
                    for reason in candidate.candidate_reasons
                ),
                "local_feature_change": local_change[candidate.position],
                "probability": candidate.probe.probability,
            },
        )
    if len(selected) != EXPECTED_BROAD_TARGETS:
        raise ValueError(
            f"broad target freeze produced {len(selected)} targets, expected 16"
        )
    return [selected[position] for position in sorted(selected)]


def _final_item(
    *, probe: RefinementProbe, reasons: Sequence[Mapping[str, Any]], role: str
) -> dict[str, Any]:
    item = probe.item
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_refinement_artifact_id": item["artifact_id"],
        "example_id": item["example"]["example_id"],
        "response_token_position": probe.position,
        "token_id": probe.token_id,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "artifact_id": f"trace-source-{digest[:24]}",
        "example": deepcopy(dict(item["example"])),
        "response_token_count": int(item["response_token_count"]),
        "target_selection": {
            "kind": "explicit_response_positions",
            "response_token_positions": [probe.position],
            "width": 1,
            "final_target_token_id": probe.token_id,
            "final_selection": {
                "corpus_role": role,
                "selection_reasons": deepcopy(list(reasons)),
                "source_refinement_item_id": item["artifact_id"],
                "source_refinement_probe_id": probe.artifact_id,
                "source_probe_sha256": probe.probe_sha256,
                "source_metrics_sha256": probe.metrics_sha256,
                "refinement_diagnostics": {
                    "token_text": probe.token_text,
                    "logit": probe.logit,
                    "probability": probe.probability,
                    "candidate_mlp_edge_count": probe.candidate_edge_count,
                    "selected_occurrence_count": probe.selected_occurrence_count,
                    "feature_count": len(probe.feature_ids),
                },
            },
        },
        "objective": {
            "name": "single_selected_logit",
            "benchmark_only_multi_target": False,
        },
    }


def build_final_trace_manifest(
    *,
    refinement_manifest: Mapping[str, Any],
    refinement_manifest_path: Path,
    refinement_manifest_sha256: str,
    refinement_summary_path: Path,
    refinement_summary_sha256: str,
    refinement_artifact_root: Path,
    probes_by_source: Mapping[str, RefinementProbe],
    summary_audit: Mapping[str, Any],
) -> dict[str, Any]:
    items = _validate_refinement_manifest(refinement_manifest)
    grouped: dict[str, list[RefinementProbe]] = defaultdict(list)
    for item in items:
        source_id = str(item["artifact_id"])
        probe = probes_by_source.get(source_id)
        if probe is None:
            raise ValueError(f"authoritative refinement probe is missing: {source_id}")
        grouped[str(item["example"]["example_id"])].append(probe)
    analysis_by_id = {
        str(row["example_id"]): row
        for row in refinement_manifest.get("prompt_analysis", [])
    }
    waves: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = defaultdict(int)
    for example_id, probes in sorted(grouped.items()):
        probes.sort(key=lambda probe: probe.position)
        selection = probes[0].item["target_selection"]["refinement_selection"]
        refinement_role = str(selection["corpus_role"])
        example = probes[0].item["example"]
        if refinement_role == "dense_full_response_refinement":
            response_count = int(probes[0].item["response_token_count"])
            if [probe.position for probe in probes] != list(range(response_count)):
                raise ValueError(
                    f"dense refinement positions are incomplete for {example_id}"
                )
            role = "dense_discovery"
            final = [
                _final_item(
                    probe=probe,
                    reasons=[{"bucket": "all_dense_response_positions"}],
                    role=role,
                )
                for probe in probes
            ]
            cluster_fit_eligible = True
        elif refinement_role == "broad_semantic_boundary_refinement":
            prompt_analysis = analysis_by_id.get(example_id)
            if not isinstance(prompt_analysis, Mapping):
                raise ValueError(f"prompt analysis is absent for {example_id}")
            prompt_reason = prompt_analysis.get("selection_reason")
            partition = (
                prompt_reason.get("selection_partition")
                if isinstance(prompt_reason, Mapping)
                else None
            )
            if partition not in {"discovery", "confirmatory_holdout"}:
                raise ValueError(f"broad prompt partition is invalid for {example_id}")
            role = f"broad_{partition}"
            chosen = select_broad_targets(
                probes,
                phenotype=str(example["diversity"]["cot_phenotype"]),
            )
            final = [
                _final_item(probe=probe, reasons=reasons, role=role)
                for probe, reasons in chosen
            ]
            cluster_fit_eligible = partition == "discovery"
        else:
            raise ValueError(f"unsupported refinement corpus role: {refinement_role}")
        for item in final:
            validate_target_selection(item)
        selected_counts[role] += len(final)
        wave_base = {
            "purpose": "independent full ADAG traces for one frozen BonaFide response",
            "example_id": example_id,
            "corpus_role": role,
            "cluster_fit_eligible": cluster_fit_eligible,
            "holdout_excluded_from_cluster_fitting": role
            == "broad_confirmatory_holdout",
        }
        routine = [
            item
            for item in final
            if item["target_selection"]["final_selection"]["refinement_diagnostics"][
                "candidate_mlp_edge_count"
            ]
            < EXTREME_EDGE_ISOLATION_THRESHOLD
        ]
        extreme = [item for item in final if item not in routine]
        if routine:
            waves.append(
                {
                    **wave_base,
                    "wave_id": f"final-trace-{role}-{example_id}",
                    "extreme_workload_isolation": False,
                    "items": routine,
                }
            )
        for item in extreme:
            position = int(item["target_selection"]["response_token_positions"][0])
            edge_count = int(
                item["target_selection"]["final_selection"]["refinement_diagnostics"][
                    "candidate_mlp_edge_count"
                ]
            )
            waves.append(
                {
                    **wave_base,
                    "wave_id": f"final-trace-{role}-{example_id}-extreme-p{position}",
                    "purpose": (
                        "isolated extreme-workload target; run a full-trace preflight "
                        "before the routine corpus"
                    ),
                    "extreme_workload_isolation": True,
                    "full_trace_preflight_required": True,
                    "screening_candidate_mlp_edge_count": edge_count,
                    "items": [item],
                }
            )
    return {
        "schema_version": TRACE_MANIFEST_SCHEMA,
        "artifact_kind": "bonafide_final_trace_manifest",
        "selection_schema_version": SCHEMA_VERSION,
        "selection_contract": {
            "prompt_membership_frozen": True,
            "final_trace_target_membership_frozen": True,
            "dense_policy": "every_teacher_forced_response_position",
            "broad_target_count_per_prompt": EXPECTED_BROAD_TARGETS,
            "trace_units_are_independent": True,
            "merge_graphs": False,
            "confirmatory_holdouts_excluded_from_cluster_fitting": True,
            "extreme_workload_targets_remain_selected_but_are_isolated": True,
        },
        "source_artifacts": {
            "refinement_manifest": {
                "path": str(refinement_manifest_path),
                "sha256": refinement_manifest_sha256,
            },
            "refinement_summary": {
                "path": str(refinement_summary_path),
                "sha256": refinement_summary_sha256,
            },
            "refinement_probe_artifact_root": str(refinement_artifact_root),
            "refinement_artifact_audit": deepcopy(dict(summary_audit)),
        },
        "dataset": deepcopy(refinement_manifest["dataset"]),
        "tokenizer": deepcopy(refinement_manifest["tokenizer"]),
        "execution_contract": {
            "batch_size": 1,
            "trace_units_are_independent": True,
            "merge_graphs": False,
            "wave_parallelism_unit": "prompt_or_isolated_extreme_target",
        },
        "selection_policy": {
            "version": "bonafide-post-refinement-freeze-v1",
            "semantic_window_width": 3,
            "phase_indices": list(EXPECTED_PHASE_INDICES),
            "deduplication": "one target position with all bucket reasons preserved",
            "refill": "semantic count, local feature change, probability, median workload, position",
            "extreme_edge_isolation_threshold": EXTREME_EDGE_ISOLATION_THRESHOLD,
            "extreme_target_policy": "retain selection, isolate wave, require full-trace preflight",
        },
        "selected_trace_counts": dict(sorted(selected_counts.items())),
        "waves": waves,
    }


def write_manifest(manifest: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-manifest", type=Path, required=True)
    parser.add_argument("--expected-refinement-manifest-sha256")
    parser.add_argument("--refinement-summary", type=Path, required=True)
    parser.add_argument("--expected-refinement-summary-sha256")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_sha = _file_sha256(args.refinement_manifest)
    if (
        args.expected_refinement_manifest_sha256 is not None
        and manifest_sha != args.expected_refinement_manifest_sha256
    ):
        raise ValueError("refinement manifest SHA-256 changed")
    summary_sha = _file_sha256(args.refinement_summary)
    if (
        args.expected_refinement_summary_sha256 is not None
        and summary_sha != args.expected_refinement_summary_sha256
    ):
        raise ValueError("refinement summary SHA-256 changed")
    refinement = _load_object(args.refinement_manifest)
    summary = _read_summary(args.refinement_summary)
    probes, audit = load_authoritative_refinement_probes(
        manifest=refinement,
        artifact_root=args.artifact_root,
        summary_records=summary,
    )
    final = build_final_trace_manifest(
        refinement_manifest=refinement,
        refinement_manifest_path=args.refinement_manifest,
        refinement_manifest_sha256=manifest_sha,
        refinement_summary_path=args.refinement_summary,
        refinement_summary_sha256=summary_sha,
        refinement_artifact_root=args.artifact_root,
        probes_by_source=probes,
        summary_audit=audit,
    )
    write_manifest(final, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _file_sha256(args.output),
                "wave_count": len(final["waves"]),
                "trace_count": sum(len(wave["items"]) for wave in final["waves"]),
                "selected_trace_counts": final["selected_trace_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
