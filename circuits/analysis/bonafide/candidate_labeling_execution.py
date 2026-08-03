"""Deterministic, non-billable execution of a candidate-labeling cohort.

This adapter exercises the complete generation -> rewrite -> paired-summary
artifact graph without resolving a provider endpoint.  Fake outputs are
deliberately synthetic plumbing fixtures and are never scientific labels.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from circuits.analysis.bonafide.candidate_labeling_renderer import (
    HELDOUT_FORBIDDEN_INPUTS,
    STATUS_ENUM,
    TYPED_OUTPUT_FIELDS,
)
from circuits.analysis.bonafide.candidate_labeling_runtime import (
    EXPECTED_REWRITE_DEPENDENCIES,
    EXPECTED_TOTAL_PLANNED_REQUESTS,
    PreparedCandidateLabelingRequest,
    RewriteDependency,
    load_candidate_labeling_execution_cohort,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from circuits.labeling.schema import ChatMessage, StrictModel

RUN_SCHEMA = "adag.bonafide.candidate-labeling-fake-run.v1"
REWRITE_REQUEST_SCHEMA = "adag.bonafide.candidate-labeling-rewrite-request.v1"
EVENT_SCHEMA = "adag.bonafide.candidate-labeling-fake-event.v1"
SUMMARY_SCHEMA = "adag.bonafide.candidate-labeling-paired-summary.v1"
COMPLETION_SCHEMA = "adag.bonafide.candidate-labeling-fake-completion.v1"

RUN_MANIFEST_FILE = "run-manifest.json"
REWRITE_REQUESTS_FILE = "rewrite-requests.jsonl"
PAIRED_SUMMARY_FILE = "paired-summary.json"
COMPLETION_MANIFEST_FILE = "completion-manifest.json"

_SOURCE_BINDINGS = {
    "candidate_labeling_execution": (
        "circuits/analysis/bonafide/candidate_labeling_execution.py"
    ),
    "candidate_labeling_execution_cli": (
        "scripts/bonafide/candidate_labeling_execute.py"
    ),
}


class CandidateLabelingRewriteRequest(StrictModel):
    """A rewrite request unlocked by exactly five validated semantic outputs."""

    schema_version: Literal["adag.bonafide.candidate-labeling-rewrite-request.v1"] = (
        REWRITE_REQUEST_SCHEMA
    )
    request_id: str
    stage_id: Literal["semantic_rewrite"] = "semantic_rewrite"
    model_role: Literal["semantic_rewriter"] = "semantic_rewriter"
    logical_prompt_id: str
    arm_id: str
    arm_sha256: str
    anchor_index: int
    cluster_id: int
    family_partition: Literal["generation"] = "generation"
    generation_only: Literal[True] = True
    selection_audit_visible: Literal[False] = False
    forbidden_input_fields: list[str]
    source_cohort_manifest_sha256: str
    source_dependency_sha256: str
    source_prompt_sha256: str
    source_message_payload_sha256: str
    required_semantic_request_ids: list[str]
    validated_semantic_output_sha256_in_order: list[str]
    provider: str
    model: str
    transport: Literal["native_batch"] = "native_batch"
    endpoint: None = None
    endpoints_resolved: Literal[False] = False
    calls_made: Literal[False] = False
    max_output_tokens: int
    temperature: float | None
    reasoning: dict[str, Any]
    provider_parameters: dict[str, Any]
    role_config_sha256: str
    messages: list[ChatMessage]
    expected_output_json_schema: dict[str, Any]
    typed_output_fields: list[str]
    status_enum: list[str]
    request_sha256: str


class CandidateLabelingFakeEvent(StrictModel):
    """One validated fake output together with its zero-cost telemetry."""

    schema_version: Literal["adag.bonafide.candidate-labeling-fake-event.v1"] = (
        EVENT_SCHEMA
    )
    request_id: str
    request_sha256: str
    request_kind: Literal["prepared", "rewrite"]
    stage_id: Literal["semantic_generation", "semantic_rewrite", "conservative_control"]
    logical_prompt_id: str
    arm_id: str
    anchor_index: int
    cluster_id: int
    sample_index: int | None
    requested_provider: str
    requested_model: str
    executor: Literal["deterministic_fake"] = "deterministic_fake"
    endpoint_identity: Literal["none"] = "none"
    endpoints_resolved: Literal[False] = False
    network_call_count: Literal[0] = 0
    api_call_count: Literal[0] = 0
    billable: Literal[False] = False
    cost_usd: Literal[0.0] = 0.0
    parse_status: Literal["success"] = "success"
    parsed: dict[str, Any]
    parsed_sha256: str
    output_character_count: int = Field(ge=1)
    event_sha256: str


@dataclass(frozen=True)
class LoadedCandidateLabelingFakeEvaluation:
    root: Path
    run_manifest: Mapping[str, Any]
    rewrite_requests: tuple[CandidateLabelingRewriteRequest, ...]
    events: tuple[CandidateLabelingFakeEvent, ...]
    paired_summary: Mapping[str, Any]
    completion_manifest: Mapping[str, Any]


def _self_hashed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(payload)
    if field in value:
        raise ValueError(f"payload already contains {field}")
    value[field] = canonical_sha256(value)
    return value


def _verify_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    value = dict(payload)
    recorded = value.pop(field, None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(value):
        raise ValueError(f"{label} self-hash drift")


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
            f"unable to bind fake evaluation revision: {message}"
        ) from error
    return completed.stdout.strip()


def _repo_root() -> Path:
    inferred = Path(__file__).resolve().parents[3]
    actual = Path(_git(inferred, "rev-parse", "--show-toplevel")).resolve()
    if inferred != actual:
        raise ValueError("candidate labeling execution repository root drift")
    return actual


def collect_candidate_labeling_execution_revision() -> dict[str, Any]:
    """Bind fake artifacts to clean, tracked adapter sources at HEAD."""

    root = _repo_root()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError("candidate labeling fake execution requires a clean worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for role, relative in _SOURCE_BINDINGS.items():
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(
                f"candidate labeling execution source is untracked: {relative}"
            )
        path = root / relative
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        content = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        sha256 = hashlib.sha256(content).hexdigest()
        if file_sha256(path) != sha256:
            raise ValueError(
                f"candidate labeling execution source is not at HEAD: {relative}"
            )
        files.append(
            {"role": role, "path": relative, "git_blob": blob, "sha256": sha256}
        )
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "files": files,
    }


def _validate_execution_revision(value: Any, *, verify_sources: bool) -> None:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("git_commit"), str)
        or len(value["git_commit"]) != 40
        or not isinstance(value.get("git_tree"), str)
        or len(value["git_tree"]) != 40
        or value.get("tracked_worktree_clean") is not True
        or not isinstance(value.get("files"), list)
    ):
        raise ValueError("fake execution revision binding is malformed")
    expected_paths = set(_SOURCE_BINDINGS.values())
    observed_paths = {
        item.get("path") for item in value["files"] if isinstance(item, Mapping)
    }
    if observed_paths != expected_paths:
        raise ValueError("fake execution revision source inventory drift")
    if (
        verify_sources
        and dict(value) != collect_candidate_labeling_execution_revision()
    ):
        raise ValueError("fake execution revision no longer matches source HEAD")


def _validate_output(
    value: Any,
    *,
    expected_schema: Mapping[str, Any],
    typed_fields: Sequence[str],
    status_enum: Sequence[str],
) -> dict[str, Any]:
    """Validate the exact frozen typed object without permissive coercion."""

    if list(typed_fields) != list(TYPED_OUTPUT_FIELDS):
        raise ValueError("typed output field inventory drift")
    if list(status_enum) != list(STATUS_ENUM):
        raise ValueError("typed status enum drift")
    properties = expected_schema.get("properties")
    if (
        expected_schema.get("type") != "object"
        or expected_schema.get("additionalProperties") is not False
        or expected_schema.get("required") != list(TYPED_OUTPUT_FIELDS)
        or not isinstance(properties, Mapping)
        or set(properties) != set(TYPED_OUTPUT_FIELDS)
    ):
        raise ValueError("expected output JSON schema drift")
    if not isinstance(value, Mapping) or set(value) != set(TYPED_OUTPUT_FIELDS):
        raise ValueError("output must contain exactly the five typed fields")
    parsed = dict(value)
    for field in TYPED_OUTPUT_FIELDS:
        observed = parsed[field]
        rule = properties[field]
        if not isinstance(rule, Mapping) or not isinstance(observed, str):
            raise TypeError(f"typed output field is not a string: {field}")
        if "const" in rule:
            if observed != rule["const"]:
                raise ValueError(f"typed output const mismatch: {field}")
        elif rule.get("type") != "string" or (
            rule.get("minLength", 0) > 0 and not observed
        ):
            raise ValueError(f"typed output schema/value mismatch: {field}")
        if "enum" in rule and observed not in rule["enum"]:
            raise ValueError(f"typed output enum mismatch: {field}")
    if properties["status"].get("enum") != list(STATUS_ENUM):
        raise ValueError("status schema enum drift")
    return parsed


def _fake_output(
    *,
    request_sha256: str,
    stage_id: str,
    expected_schema: Mapping[str, Any],
) -> dict[str, Any]:
    digest = hashlib.sha256(f"fake-v1:{stage_id}:{request_sha256}".encode()).hexdigest()
    candidate_rule = expected_schema["properties"]["exploratory_candidate_description"]
    candidate = candidate_rule.get(
        "const", f"Synthetic candidate-direction fixture {digest[12:24]}."
    )
    value = {
        "input_localization_hypothesis": (
            f"Synthetic local-input fixture {digest[:12]}."
        ),
        "exploratory_candidate_description": candidate,
        "background_or_confound": (
            f"Synthetic background-confound fixture {digest[24:36]}."
        ),
        "limitations": (
            "Deterministic fake output for execution validation only; it is not a "
            "scientific label or evidence assessment."
        ),
        "status": STATUS_ENUM[int(digest[-1], 16) % len(STATUS_ENUM)],
    }
    return _validate_output(
        value,
        expected_schema=expected_schema,
        typed_fields=TYPED_OUTPUT_FIELDS,
        status_enum=STATUS_ENUM,
    )


def _event_for_request(
    request: PreparedCandidateLabelingRequest | CandidateLabelingRewriteRequest,
) -> CandidateLabelingFakeEvent:
    parsed = _fake_output(
        request_sha256=request.request_sha256,
        stage_id=request.stage_id,
        expected_schema=request.expected_output_json_schema,
    )
    serialized = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    payload = {
        "schema_version": EVENT_SCHEMA,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "request_kind": (
            "rewrite"
            if isinstance(request, CandidateLabelingRewriteRequest)
            else "prepared"
        ),
        "stage_id": request.stage_id,
        "logical_prompt_id": request.logical_prompt_id,
        "arm_id": request.arm_id,
        "anchor_index": request.anchor_index,
        "cluster_id": request.cluster_id,
        "sample_index": getattr(request, "sample_index", None),
        "requested_provider": request.provider,
        "requested_model": request.model,
        "executor": "deterministic_fake",
        "endpoint_identity": "none",
        "endpoints_resolved": False,
        "network_call_count": 0,
        "api_call_count": 0,
        "billable": False,
        "cost_usd": 0.0,
        "parse_status": "success",
        "parsed": parsed,
        "parsed_sha256": canonical_sha256(parsed),
        "output_character_count": len(serialized),
    }
    return CandidateLabelingFakeEvent.model_validate(
        _self_hashed(payload, "event_sha256")
    )


def _event_path(root: Path, event: CandidateLabelingFakeEvent) -> Path:
    return root / "events" / event.stage_id / f"{event.request_sha256}.json"


def _persist_or_validate_event(
    root: Path,
    request: PreparedCandidateLabelingRequest | CandidateLabelingRewriteRequest,
) -> CandidateLabelingFakeEvent:
    expected = _event_for_request(request)
    path = _event_path(root, expected)
    if path.exists():
        observed = CandidateLabelingFakeEvent.model_validate(load_json_object(path))
        _verify_self_hash(
            observed.model_dump(mode="json"), "event_sha256", "fake event"
        )
        if observed != expected:
            raise ValueError(f"persisted fake event drift: {request.request_id}")
        return observed
    atomic_write_json(path, expected.model_dump(mode="json"))
    return expected


def _rewrite_message(outputs: Sequence[Mapping[str, Any]]) -> ChatMessage:
    if len(outputs) != 5:
        raise ValueError("rewrite construction requires exactly five semantic outputs")
    content = "\n\n".join(
        [
            "SEMANTIC_REWRITE_INPUT_V1",
            (
                "Rewrite the five generation-only candidate objects below into one "
                "conservative object using the original prompt's exact JSON schema. "
                "Do not use any selection, audit, automatic-score, or held-out data."
            ),
            json.dumps(list(outputs), sort_keys=True, ensure_ascii=False),
        ]
    )
    if any(field in content for field in HELDOUT_FORBIDDEN_INPUTS):
        raise ValueError("rewrite message contains a forbidden input field")
    return ChatMessage(role="user", content=content)


def construct_candidate_labeling_rewrite_request(
    *,
    cohort_manifest_sha256: str,
    dependency: RewriteDependency,
    semantic_requests: Sequence[PreparedCandidateLabelingRequest],
    semantic_outputs: Sequence[Mapping[str, Any]],
    semantic_output_sha256_in_order: Sequence[str],
    max_output_tokens: int,
    temperature: float | None,
    reasoning: Mapping[str, Any],
    provider_parameters: Mapping[str, Any],
) -> CandidateLabelingRewriteRequest:
    """Construct one rewrite request from five validated, ordered outputs."""

    if (
        len(semantic_requests) != 5
        or len(semantic_outputs) != 5
        or len(semantic_output_sha256_in_order) != 5
    ):
        raise ValueError("rewrite dependency is not satisfied by exactly five outputs")
    request_ids = [request.request_id for request in semantic_requests]
    if request_ids != dependency.required_semantic_request_ids:
        raise ValueError("rewrite semantic request ordering or identity drift")
    source = semantic_requests[0]
    if any(
        request.logical_prompt_id != dependency.logical_prompt_id
        or request.sample_index != index
        or request.expected_output_json_schema != source.expected_output_json_schema
        or request.messages != source.messages
        for index, request in enumerate(semantic_requests)
    ):
        raise ValueError("rewrite semantic prompt/sample graph drift")
    parsed = [
        _validate_output(
            output,
            expected_schema=source.expected_output_json_schema,
            typed_fields=source.typed_output_fields,
            status_enum=source.status_enum,
        )
        for output in semantic_outputs
    ]
    observed_output_hashes = [canonical_sha256(output) for output in parsed]
    if observed_output_hashes != list(semantic_output_sha256_in_order):
        raise ValueError("rewrite semantic output hash binding drift")
    messages = [*source.messages, _rewrite_message(parsed)]
    payload = {
        "schema_version": REWRITE_REQUEST_SCHEMA,
        "request_id": dependency.planned_request_id,
        "stage_id": "semantic_rewrite",
        "model_role": "semantic_rewriter",
        "logical_prompt_id": dependency.logical_prompt_id,
        "arm_id": dependency.arm_id,
        "arm_sha256": dependency.arm_sha256,
        "anchor_index": source.anchor_index,
        "cluster_id": source.cluster_id,
        "family_partition": "generation",
        "generation_only": True,
        "selection_audit_visible": False,
        "forbidden_input_fields": list(HELDOUT_FORBIDDEN_INPUTS),
        "source_cohort_manifest_sha256": cohort_manifest_sha256,
        "source_dependency_sha256": dependency.dependency_sha256,
        "source_prompt_sha256": dependency.source_prompt_sha256,
        "source_message_payload_sha256": dependency.source_message_payload_sha256,
        "required_semantic_request_ids": request_ids,
        "validated_semantic_output_sha256_in_order": list(
            semantic_output_sha256_in_order
        ),
        "provider": dependency.provider,
        "model": dependency.model,
        "transport": dependency.transport,
        "endpoint": None,
        "endpoints_resolved": False,
        "calls_made": False,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "reasoning": dict(reasoning),
        "provider_parameters": dict(provider_parameters),
        "role_config_sha256": dependency.role_config_sha256,
        "messages": [message.model_dump(mode="json") for message in messages],
        "expected_output_json_schema": source.expected_output_json_schema,
        "typed_output_fields": list(TYPED_OUTPUT_FIELDS),
        "status_enum": list(STATUS_ENUM),
    }
    request = CandidateLabelingRewriteRequest.model_validate(
        _self_hashed(payload, "request_sha256")
    )
    serialized_messages = json.dumps(
        [message.model_dump(mode="json") for message in request.messages],
        sort_keys=True,
    )
    if any(field in serialized_messages for field in HELDOUT_FORBIDDEN_INPUTS):
        raise ValueError("rewrite request violates the held-out input firewall")
    return request


def _construct_rewrite_request(
    *,
    cohort_manifest_sha256: str,
    dependency: RewriteDependency,
    semantic_requests: Sequence[PreparedCandidateLabelingRequest],
    semantic_events: Sequence[CandidateLabelingFakeEvent],
    max_output_tokens: int,
    temperature: float | None,
    reasoning: Mapping[str, Any],
    provider_parameters: Mapping[str, Any],
) -> CandidateLabelingRewriteRequest:
    """Compatibility wrapper for deterministic fake execution."""

    request_ids = [request.request_id for request in semantic_requests]
    if [event.request_id for event in semantic_events] != request_ids:
        raise ValueError("rewrite semantic request ordering or identity drift")
    return construct_candidate_labeling_rewrite_request(
        cohort_manifest_sha256=cohort_manifest_sha256,
        dependency=dependency,
        semantic_requests=semantic_requests,
        semantic_outputs=[event.parsed for event in semantic_events],
        semantic_output_sha256_in_order=[
            event.parsed_sha256 for event in semantic_events
        ],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        reasoning=reasoning,
        provider_parameters=provider_parameters,
    )


def _construct_rewrites(
    cohort: Any, events: Sequence[CandidateLabelingFakeEvent]
) -> list[CandidateLabelingRewriteRequest]:
    semantic_requests = {
        request.request_id: request
        for request in cohort.initial_requests
        if request.stage_id == "semantic_generation"
    }
    semantic_events = {
        event.request_id: event
        for event in events
        if event.stage_id == "semantic_generation"
    }
    role = cohort.config.semantic_rewriter
    result = []
    for dependency in cohort.rewrite_dependencies:
        requests = [
            semantic_requests[item] for item in dependency.required_semantic_request_ids
        ]
        outputs = [
            semantic_events[item] for item in dependency.required_semantic_request_ids
        ]
        result.append(
            _construct_rewrite_request(
                cohort_manifest_sha256=cohort.manifest["manifest_sha256"],
                dependency=dependency,
                semantic_requests=requests,
                semantic_events=outputs,
                max_output_tokens=role.max_output_tokens,
                temperature=role.temperature,
                reasoning=role.reasoning,
                provider_parameters=role.provider_parameters,
            )
        )
    return result


def _summary(
    *,
    run_manifest: Mapping[str, Any],
    requests: Sequence[PreparedCandidateLabelingRequest],
    rewrites: Sequence[CandidateLabelingRewriteRequest],
    events: Sequence[CandidateLabelingFakeEvent],
) -> dict[str, Any]:
    events_by_id = {event.request_id: event for event in events}
    rewrites_by_prompt = {request.logical_prompt_id: request for request in rewrites}
    by_prompt: dict[str, list[PreparedCandidateLabelingRequest]] = {}
    for request in requests:
        by_prompt.setdefault(request.logical_prompt_id, []).append(request)
    prompt_rows = []
    for prompt_id, prompt_requests in by_prompt.items():
        semantic = sorted(
            (
                item
                for item in prompt_requests
                if item.stage_id == "semantic_generation"
            ),
            key=lambda item: int(item.sample_index),
        )
        controls = [
            item for item in prompt_requests if item.stage_id == "conservative_control"
        ]
        if len(semantic) != 5 or len(controls) != 1:
            raise ValueError("paired summary prompt request cardinality drift")
        rewrite = rewrites_by_prompt[prompt_id]
        row = {
            "logical_prompt_id": prompt_id,
            "arm_id": semantic[0].arm_id,
            "anchor_index": semantic[0].anchor_index,
            "cluster_id": semantic[0].cluster_id,
            "semantic_event_sha256_in_order": [
                events_by_id[item.request_id].event_sha256 for item in semantic
            ],
            "semantic_statuses_in_order": [
                events_by_id[item.request_id].parsed["status"] for item in semantic
            ],
            "rewrite_event_sha256": events_by_id[rewrite.request_id].event_sha256,
            "rewrite_status": events_by_id[rewrite.request_id].parsed["status"],
            "control_event_sha256": events_by_id[controls[0].request_id].event_sha256,
            "control_status": events_by_id[controls[0].request_id].parsed["status"],
        }
        prompt_rows.append(_self_hashed(row, "prompt_summary_sha256"))
    prompt_rows.sort(key=lambda row: (row["anchor_index"], row["arm_id"]))
    pairs = []
    for anchor_index in range(12):
        arms = [row for row in prompt_rows if row["anchor_index"] == anchor_index]
        if len(arms) != 2 or len({row["cluster_id"] for row in arms}) != 1:
            raise ValueError("paired summary arm/cluster alignment drift")
        pair = {
            "anchor_index": anchor_index,
            "cluster_id": arms[0]["cluster_id"],
            "arm_summary_sha256_in_order": [
                row["prompt_summary_sha256"] for row in arms
            ],
            "arm_ids_in_order": [row["arm_id"] for row in arms],
        }
        pairs.append(_self_hashed(pair, "pair_sha256"))
    stage_counts = Counter(event.stage_id for event in events)
    status_counts = Counter(str(event.parsed["status"]) for event in events)
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "purpose": "evaluation_plumbing_only_not_scientific_labels",
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "source_cohort_manifest_sha256": run_manifest["source_cohort_manifest_sha256"],
        "family_partition": "generation",
        "generation_only": True,
        "selection_audit_visible": False,
        "heldout_scoring_inputs_used": [],
        "prompt_summaries": prompt_rows,
        "paired_arm_summaries": pairs,
        "counts": {
            "logical_prompts": len(prompt_rows),
            "paired_anchors": len(pairs),
            "events": len(events),
            "by_stage": dict(sorted(stage_counts.items())),
            "by_status": dict(sorted(status_counts.items())),
        },
        "telemetry": {
            "executor": "deterministic_fake",
            "network_call_count": 0,
            "api_call_count": 0,
            "billable_event_count": 0,
            "known_cost_usd": 0.0,
        },
    }
    return _self_hashed(payload, "summary_sha256")


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted((root / "events").glob("*/*.json")):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    for name in (REWRITE_REQUESTS_FILE, PAIRED_SUMMARY_FILE):
        path = root / name
        files.append(
            {
                "path": name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return files


def _completion(root: Path, run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": COMPLETION_SCHEMA,
        "run_manifest_sha256": run_manifest["run_manifest_sha256"],
        "source_cohort_manifest_sha256": run_manifest["source_cohort_manifest_sha256"],
        "complete": True,
        "planned_request_count": EXPECTED_TOTAL_PLANNED_REQUESTS,
        "successful_event_count": EXPECTED_TOTAL_PLANNED_REQUESTS,
        "network_call_count": 0,
        "api_call_count": 0,
        "billable_event_count": 0,
        "known_cost_usd": 0.0,
        "files": _file_inventory(root),
    }
    return _self_hashed(payload, "completion_sha256")


def _initialize_run(root: Path, cohort: Any) -> dict[str, Any]:
    expected = _self_hashed(
        {
            "schema_version": RUN_SCHEMA,
            "run_id": (
                "candidate-labeling-fake:"
                + str(cohort.manifest["manifest_sha256"])[:32]
            ),
            "purpose": "deterministic_non_billable_execution_validation",
            "source_cohort_root": str(cohort.root),
            "source_cohort_manifest_sha256": cohort.manifest["manifest_sha256"],
            "source_cohort_manifest_file_sha256": file_sha256(
                cohort.root / "manifest.json"
            ),
            "execution_revision": collect_candidate_labeling_execution_revision(),
            "family_partition": "generation",
            "generation_only": True,
            "selection_audit_visible": False,
            "heldout_scoring_inputs_forbidden": list(HELDOUT_FORBIDDEN_INPUTS),
            "executor": "deterministic_fake",
            "provider_model_endpoints_resolved": False,
            "network_calls_made": False,
            "api_calls_made": False,
            "billable": False,
        },
        "run_manifest_sha256",
    )
    path = root / RUN_MANIFEST_FILE
    if path.exists():
        observed = load_json_object(path)
        _verify_self_hash(observed, "run_manifest_sha256", "fake run manifest")
        if observed != expected:
            raise ValueError("fake run manifest or source revision drift")
        return observed
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("fake run root is non-empty without a run manifest")
    atomic_write_json(path, expected)
    return expected


def execute_candidate_labeling_fake_evaluation(
    *, cohort_root: Path, output_root: Path, verify_sources: bool = True
) -> LoadedCandidateLabelingFakeEvaluation:
    """Execute or resume the zero-cost deterministic candidate-labeling graph."""

    cohort = load_candidate_labeling_execution_cohort(
        cohort_root, verify_sources=verify_sources
    )
    root = output_root.resolve()
    run_manifest = _initialize_run(root, cohort)

    initial_events = [
        _persist_or_validate_event(root, request) for request in cohort.initial_requests
    ]
    rewrites = _construct_rewrites(cohort, initial_events)
    rewrite_rows = [request.model_dump(mode="json") for request in rewrites]
    rewrite_path = root / REWRITE_REQUESTS_FILE
    if rewrite_path.exists():
        if read_jsonl(rewrite_path) != rewrite_rows:
            raise ValueError("persisted rewrite request graph drift")
    else:
        atomic_write_jsonl(rewrite_path, rewrite_rows)

    rewrite_events = [_persist_or_validate_event(root, request) for request in rewrites]
    events = [*initial_events, *rewrite_events]
    expected_summary = _summary(
        run_manifest=run_manifest,
        requests=cohort.initial_requests,
        rewrites=rewrites,
        events=events,
    )
    summary_path = root / PAIRED_SUMMARY_FILE
    if summary_path.exists():
        if load_json_object(summary_path) != expected_summary:
            raise ValueError("persisted paired summary drift")
    else:
        atomic_write_json(summary_path, expected_summary)

    expected_completion = _completion(root, run_manifest)
    completion_path = root / COMPLETION_MANIFEST_FILE
    if completion_path.exists():
        if load_json_object(completion_path) != expected_completion:
            raise ValueError("persisted fake completion drift")
    else:
        atomic_write_json(completion_path, expected_completion)
    return load_candidate_labeling_fake_evaluation(
        root, cohort_root=cohort_root, verify_sources=verify_sources
    )


def load_candidate_labeling_fake_evaluation(
    root: Path,
    *,
    cohort_root: Path | None = None,
    verify_sources: bool = True,
) -> LoadedCandidateLabelingFakeEvaluation:
    """Deep-validate a complete fake evaluation and its source cohort."""

    root = root.resolve()
    run_manifest = load_json_object(root / RUN_MANIFEST_FILE)
    _verify_self_hash(run_manifest, "run_manifest_sha256", "fake run manifest")
    if (
        run_manifest.get("schema_version") != RUN_SCHEMA
        or run_manifest.get("generation_only") is not True
        or run_manifest.get("selection_audit_visible") is not False
        or run_manifest.get("heldout_scoring_inputs_forbidden")
        != list(HELDOUT_FORBIDDEN_INPUTS)
        or run_manifest.get("provider_model_endpoints_resolved") is not False
        or run_manifest.get("network_calls_made") is not False
        or run_manifest.get("api_calls_made") is not False
        or run_manifest.get("billable") is not False
    ):
        raise ValueError("fake run contract drift")
    _validate_execution_revision(
        run_manifest.get("execution_revision"), verify_sources=verify_sources
    )
    source_root = (
        Path(str(run_manifest["source_cohort_root"]))
        if cohort_root is None
        else cohort_root
    )
    cohort = load_candidate_labeling_execution_cohort(
        source_root, verify_sources=verify_sources
    )
    if (
        cohort.manifest["manifest_sha256"]
        != run_manifest["source_cohort_manifest_sha256"]
        or file_sha256(cohort.root / "manifest.json")
        != run_manifest["source_cohort_manifest_file_sha256"]
    ):
        raise ValueError("fake evaluation source cohort drift")

    rewrite_rows = read_jsonl(root / REWRITE_REQUESTS_FILE)
    rewrites = tuple(
        CandidateLabelingRewriteRequest.model_validate(row) for row in rewrite_rows
    )
    for request in rewrites:
        _verify_self_hash(
            request.model_dump(mode="json"), "request_sha256", "rewrite request"
        )

    event_paths = sorted((root / "events").glob("*/*.json"))
    events = tuple(
        CandidateLabelingFakeEvent.model_validate(load_json_object(path))
        for path in event_paths
    )
    for event in events:
        _verify_self_hash(event.model_dump(mode="json"), "event_sha256", "fake event")
    if (
        len(rewrites) != EXPECTED_REWRITE_DEPENDENCIES
        or len(events) != EXPECTED_TOTAL_PLANNED_REQUESTS
        or len({event.request_id for event in events}) != len(events)
    ):
        raise ValueError("fake evaluation event cardinality drift")

    initial_events = [event for event in events if event.request_kind == "prepared"]
    expected_rewrites = _construct_rewrites(cohort, initial_events)
    if list(rewrites) != expected_rewrites:
        raise ValueError("fake evaluation rewrite construction drift")
    expected_by_id = {
        request.request_id: _event_for_request(request)
        for request in [*cohort.initial_requests, *rewrites]
    }
    if {event.request_id: event for event in events} != expected_by_id:
        raise ValueError("fake evaluation output or telemetry drift")

    summary = load_json_object(root / PAIRED_SUMMARY_FILE)
    _verify_self_hash(summary, "summary_sha256", "paired summary")
    expected_summary = _summary(
        run_manifest=run_manifest,
        requests=cohort.initial_requests,
        rewrites=rewrites,
        events=events,
    )
    if summary != expected_summary:
        raise ValueError("fake evaluation paired summary drift")
    completion = load_json_object(root / COMPLETION_MANIFEST_FILE)
    _verify_self_hash(completion, "completion_sha256", "fake completion")
    if completion != _completion(root, run_manifest):
        raise ValueError("fake evaluation completion inventory drift")
    return LoadedCandidateLabelingFakeEvaluation(
        root=root,
        run_manifest=run_manifest,
        rewrite_requests=rewrites,
        events=events,
        paired_summary=summary,
        completion_manifest=completion,
    )
