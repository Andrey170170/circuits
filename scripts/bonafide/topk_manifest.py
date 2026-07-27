"""Validation for contribution-aware same-position candidate trace manifests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from circuits.tracing.candidates import (
    CANDIDATE_POLICY_VERSION,
    JOINT_OBJECTIVE_VERSION,
)
from scripts.bonafide.runner import validate_target_selection

SCHEMA_VERSION = "bonafide-topk-trace-manifest/v1"
DISCOVERY_ONLY_PHASES = {
    "observed_k1_parity",
    "c0_candidate_reference",
    "c1_policy_resource",
    "c2_scientific_utility",
}
Phase = Literal[
    "observed_k1_parity",
    "c0_candidate_reference",
    "c1_policy_resource",
    "c2_scientific_utility",
    "matched_corpus",
]


def validate_trace_family(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("top-k manifest requires a trace_family object")
    result = dict(value)
    trace_family_id = result.get("trace_family_id")
    if not isinstance(trace_family_id, str) or not trace_family_id:
        raise ValueError("top-k trace_family_id must be a non-empty string")
    policy_id = result.get("candidate_policy_id")
    candidate_count = result.get("candidate_count")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise ValueError("top-k candidate_count must be an integer")
    if result.get("candidate_policy_version") != CANDIDATE_POLICY_VERSION:
        raise ValueError("top-k candidate policy version is unsupported")
    if result.get("joint_objective_version") != JOINT_OBJECTIVE_VERSION:
        raise ValueError("top-k joint objective version is unsupported")
    objective_id = result.get("joint_objective_id")

    expected_counts = {
        "observed_token": 1,
        "model_top5": 5,
        "observed_plus_top4_alternatives": 5,
    }
    if policy_id not in expected_counts:
        raise ValueError(f"unsupported top-k candidate policy: {policy_id!r}")
    if candidate_count != expected_counts[policy_id]:
        raise ValueError("top-k candidate count disagrees with candidate policy")
    if objective_id not in {"raw_logit_sum", "observed_vs_alternatives"}:
        raise ValueError(f"unsupported top-k joint objective: {objective_id!r}")
    if policy_id in {"observed_token", "model_top5"} and objective_id != "raw_logit_sum":
        raise ValueError(
            f"{policy_id} supports only raw_logit_sum in the v1 trace contract"
        )
    return result


def validate_topk_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate one immutable policy/objective trace-family manifest."""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported top-k manifest schema: {manifest.get('schema_version')!r}"
        )
    phase = manifest.get("phase")
    allowed_phases = {*DISCOVERY_ONLY_PHASES, "matched_corpus"}
    if phase not in allowed_phases:
        raise ValueError(f"unsupported top-k manifest phase: {phase!r}")
    validate_trace_family(manifest.get("trace_family"))

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("top-k manifest requires width-one source provenance")
    for field in (
        "width1_manifest_sha256",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "chat_template_sha256",
    ):
        value = source.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"top-k source.{field} must be a non-empty string")
    if len(source["width1_manifest_sha256"]) != 64:
        raise ValueError("top-k source width-one manifest hash is invalid")

    waves = manifest.get("waves")
    if not isinstance(waves, list) or not waves:
        raise ValueError("top-k manifest requires non-empty waves")
    seen_wave_ids: set[str] = set()
    seen_source_artifact_ids: set[str] = set()
    for wave in waves:
        if not isinstance(wave, Mapping):
            raise ValueError("top-k wave must be an object")
        wave_id = wave.get("wave_id")
        if (
            not isinstance(wave_id, str)
            or not wave_id
            or wave_id in seen_wave_ids
        ):
            raise ValueError(f"invalid or duplicate top-k wave_id: {wave_id!r}")
        seen_wave_ids.add(wave_id)
        corpus_role = wave.get("corpus_role")
        if phase in DISCOVERY_ONLY_PHASES and corpus_role == "confirmatory_holdout":
            raise ValueError(f"{phase} cannot include confirmatory holdout targets")
        items = wave.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("top-k wave requires non-empty items")
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("top-k work item must be an object")
            validate_target_selection(item)
            positions = item["target_selection"]["response_token_positions"]
            if len(positions) != 1:
                raise ValueError("top-k work items require one response target position")
            artifact_id = item.get("artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("top-k source artifact_id must be non-empty")
            if artifact_id in seen_source_artifact_ids:
                raise ValueError(f"duplicate top-k source artifact_id: {artifact_id}")
            seen_source_artifact_ids.add(artifact_id)
            example = item.get("example")
            if not isinstance(example, Mapping):
                raise ValueError("top-k work item requires example provenance")
            for field in ("example_id", "prompt", "response"):
                if not isinstance(example.get(field), str) or not example[field]:
                    raise ValueError(f"top-k work item example.{field} is required")
