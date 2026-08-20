"""Validation for contribution-aware same-position candidate trace manifests."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from circuits.tracing.candidates import (
    CANDIDATE_POLICY_VERSION,
    JOINT_OBJECTIVE_VERSION,
)

from scripts.bonafide.runner import validate_target_selection

SCHEMA_VERSION = "bonafide-topk-trace-manifest/v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
DISCOVERY_ONLY_PHASES = {
    "observed_k1_parity",
    "c0_candidate_reference",
    "c1_policy_resource",
    "c2_scientific_utility",
    "step0_t5_smoke",
    "process_witness_resource_calibration_v1",
}
MODEL_TOP5_PLUS_OBSERVED_COUNT_RULE = "5_if_observed_in_model_top5_else_6"
STEP0_T5_SMOKE_PHASE = "step0_t5_smoke"
PROCESS_WITNESS_RESOURCE_CALIBRATION_PHASE = "process_witness_resource_calibration_v1"
STRICT_T5_PHASES = {
    STEP0_T5_SMOKE_PHASE,
    PROCESS_WITNESS_RESOURCE_CALIBRATION_PHASE,
}
HISTORICAL_THINKING_SERIALIZATION_MODE = "historical_thinking_continuation"


def candidate_count_bounds(
    trace_family: Mapping[str, Any],
) -> tuple[int, int]:
    """Return the valid realized candidate-count range for a trace family."""

    if trace_family.get("candidate_policy_id") == "model_top5_plus_observed":
        return (
            int(trace_family["candidate_count_min"]),
            int(trace_family["candidate_count_max"]),
        )
    count = int(trace_family["candidate_count"])
    return count, count


def candidate_selection_limit(trace_family: Mapping[str, Any]) -> int:
    """Return the requested selection width, including a variable policy ceiling."""

    return candidate_count_bounds(trace_family)[1]


def validate_trace_family(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("top-k manifest requires a trace_family object")
    result = dict(value)
    trace_family_id = result.get("trace_family_id")
    if not isinstance(trace_family_id, str) or not SAFE_ID.fullmatch(trace_family_id):
        raise ValueError("top-k trace_family_id must be a filesystem-safe identifier")
    policy_id = result.get("candidate_policy_id")
    if result.get("candidate_policy_version") != CANDIDATE_POLICY_VERSION:
        raise ValueError("top-k candidate policy version is unsupported")
    if result.get("joint_objective_version") != JOINT_OBJECTIVE_VERSION:
        raise ValueError("top-k joint objective version is unsupported")
    objective_id = result.get("joint_objective_id")

    expected_counts = {
        "observed_token": 1,
        "specified_token": 1,
        "model_top5": 5,
        "observed_plus_top4_alternatives": 5,
    }
    if policy_id == "model_top5_plus_observed":
        if result.get("candidate_count") is not None:
            raise ValueError(
                "model_top5_plus_observed uses candidate_count_min/max, not "
                "a fixed candidate_count"
            )
        if (
            result.get("candidate_count_min") != 5
            or result.get("candidate_count_max") != 6
            or result.get("candidate_count_rule") != MODEL_TOP5_PLUS_OBSERVED_COUNT_RULE
        ):
            raise ValueError(
                "model_top5_plus_observed requires the frozen 5-or-6 "
                "candidate-count contract"
            )
    elif policy_id not in expected_counts:
        raise ValueError(f"unsupported top-k candidate policy: {policy_id!r}")
    else:
        candidate_count = result.get("candidate_count")
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
            raise ValueError("top-k candidate_count must be an integer")
        if candidate_count != expected_counts[policy_id]:
            raise ValueError("top-k candidate count disagrees with candidate policy")
        if any(
            field in result
            for field in (
                "candidate_count_min",
                "candidate_count_max",
                "candidate_count_rule",
            )
        ):
            raise ValueError(
                "fixed-width policies cannot declare variable candidate counts"
            )
    if objective_id not in {"raw_logit_sum", "observed_vs_alternatives"}:
        raise ValueError(f"unsupported top-k joint objective: {objective_id!r}")
    if (
        policy_id
        in {
            "observed_token",
            "specified_token",
            "model_top5",
        }
        and objective_id != "raw_logit_sum"
    ):
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
    trace_family = validate_trace_family(manifest.get("trace_family"))
    if phase in STRICT_T5_PHASES:
        expected = {
            "candidate_policy_id": "model_top5",
            "candidate_count": 5,
            "joint_objective_id": "raw_logit_sum",
        }
        for field, expected_value in expected.items():
            if trace_family.get(field) != expected_value:
                raise ValueError(
                    "strict T5 phases require upstream T5 semantics: "
                    f"trace_family.{field}={expected_value!r}"
                )
        teacher_forcing_contract = manifest.get("teacher_forcing_contract")
        if not isinstance(teacher_forcing_contract, Mapping):
            raise ValueError(
                "strict T5 phases require a teacher_forcing_contract object"
            )
        if (
            teacher_forcing_contract.get("serialization_mode")
            != HISTORICAL_THINKING_SERIALIZATION_MODE
        ):
            raise ValueError(
                "strict T5 phases require historical thinking continuation serialization"
            )
        if (
            teacher_forcing_contract.get("token_identity_schema_version")
            != "adag.teacher-forced-token-identity.v1"
        ):
            raise ValueError("strict T5 token identity schema version is unsupported")
        if (
            teacher_forcing_contract.get("hash_encoding")
            != "sha256_utf8_canonical_json_integer_array_v1"
        ):
            raise ValueError("strict T5 token hash encoding is unsupported")
        system_hash = teacher_forcing_contract.get("system_prompt_sha256")
        if not isinstance(system_hash, str) or not SHA256.fullmatch(system_hash):
            raise ValueError("strict T5 system_prompt_sha256 must be a SHA-256 digest")

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("top-k manifest requires width-one source provenance")
    for field in (
        "width1_manifest_path",
        "width1_manifest_sha256",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "chat_template_sha256",
    ):
        value = source.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"top-k source.{field} must be a non-empty string")
    if not SHA256.fullmatch(source["width1_manifest_sha256"]):
        raise ValueError("top-k source width-one manifest hash is invalid")
    if not SHA256.fullmatch(source["chat_template_sha256"]):
        raise ValueError("top-k source chat-template hash is invalid")

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
            or not SAFE_ID.fullmatch(wave_id)
            or wave_id in seen_wave_ids
        ):
            raise ValueError(f"invalid or duplicate top-k wave_id: {wave_id!r}")
        seen_wave_ids.add(wave_id)
        corpus_role = wave.get("corpus_role")
        if not isinstance(corpus_role, str) or not corpus_role:
            raise ValueError("top-k wave corpus_role must be a non-empty string")
        if phase in DISCOVERY_ONLY_PHASES and corpus_role.endswith(
            "confirmatory_holdout"
        ):
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
                raise ValueError(
                    "top-k work items require one response target position"
                )
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
            if phase in STRICT_T5_PHASES:
                system_prompt = example.get("system_prompt")
                if not isinstance(system_prompt, str) or not system_prompt:
                    raise ValueError(
                        "strict T5 example.system_prompt must be non-empty"
                    )
                if (
                    hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
                    != manifest["teacher_forcing_contract"]["system_prompt_sha256"]
                ):
                    raise ValueError(
                        "strict T5 example system prompt hash disagrees with "
                        "teacher_forcing_contract"
                    )
                token_identity = example.get("token_identity")
                if not isinstance(token_identity, Mapping):
                    raise ValueError(
                        "strict T5 example.token_identity must be an object"
                    )
                for field in (
                    "assistant_prefix_ids_sha256",
                    "response_ids_sha256",
                ):
                    value = token_identity.get(field)
                    if not isinstance(value, str) or not SHA256.fullmatch(value):
                        raise ValueError(
                            f"example.token_identity.{field} must be a SHA-256 digest"
                        )
                if token_identity.get("response_token_count") != item.get(
                    "response_token_count"
                ):
                    raise ValueError(
                        "example token identity response count disagrees with item"
                    )
            specified_token_id = item.get("specified_candidate_token_id")
            if manifest["trace_family"]["candidate_policy_id"] == "specified_token":
                if (
                    isinstance(specified_token_id, bool)
                    or not isinstance(specified_token_id, int)
                    or specified_token_id < 0
                ):
                    raise ValueError(
                        "specified_token work items require "
                        "specified_candidate_token_id"
                    )
            elif specified_token_id is not None:
                raise ValueError(
                    "specified_candidate_token_id is valid only for "
                    "specified_token manifests"
                )
