"""Prepare immutable, non-billable execution cohorts for C2 labeling prompts.

The frozen renderer is provider neutral.  This module binds it to a typed model
recipe and a dated price snapshot, but deliberately stops before endpoint
resolution, provider batch construction, or any API call.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from circuits.analysis.bonafide import candidate_labeling_renderer as renderer_module
from circuits.analysis.bonafide.candidate_clustering_execution import (
    _publish_directory_no_replace,
)
from circuits.analysis.bonafide.candidate_labeling_renderer import (
    HELDOUT_FORBIDDEN_INPUTS,
    STATUS_ENUM,
    TYPED_OUTPUT_FIELDS,
    LoadedCandidateLabelingRenderer,
    load_candidate_labeling_renderer,
)
from circuits.analysis.bonafide.candidate_labeling_renderer import (
    MANIFEST_FILE as RENDERER_MANIFEST_FILE,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.labeling.config import ModelRoleConfig
from circuits.labeling.pricing import load_price_snapshot
from circuits.labeling.schema import ChatMessage, StrictModel

CONFIG_SCHEMA = "adag.bonafide.candidate-labeling-runtime-config.v1"
COHORT_SCHEMA = "adag.bonafide.candidate-labeling-execution-cohort.v1"
REQUEST_SCHEMA = "adag.bonafide.candidate-labeling-prepared-request.v1"
REWRITE_DEPENDENCY_SCHEMA = "adag.bonafide.candidate-labeling-rewrite-dependency.v1"

MANIFEST_FILE = "manifest.json"
CONFIG_SNAPSHOT_FILE = "execution-config.json"
INITIAL_REQUESTS_FILE = "initial-requests.jsonl"
REWRITE_DEPENDENCIES_FILE = "rewrite-dependencies.jsonl"

GENERIC_STAGE_IDS = (
    "semantic_generation",
    "semantic_rewrite",
    "conservative_control",
)
INITIAL_STAGE_IDS = ("semantic_generation", "conservative_control")
ROLE_NAMES = (
    "semantic_generator",
    "semantic_rewriter",
    "conservative_control",
)
EXPECTED_LOGICAL_PROMPTS = 24
SEMANTIC_SAMPLES_PER_PROMPT = 5
EXPECTED_SEMANTIC_REQUESTS = 120
EXPECTED_CONTROL_REQUESTS = 24
EXPECTED_REWRITE_DEPENDENCIES = 24
EXPECTED_TOTAL_PLANNED_REQUESTS = 168
_RUNTIME_SOURCE_BINDINGS = {
    "candidate_labeling_runtime": (
        "circuits/analysis/bonafide/candidate_labeling_runtime.py"
    ),
    "candidate_labeling_runtime_cli": (
        "scripts/bonafide/candidate_labeling_runtime.py"
    ),
}


class CandidateLabelingRuntimeConfig(StrictModel):
    """Strict provider/model recipe for the frozen candidate-labeling cohort."""

    schema_version: Literal["adag.bonafide.candidate-labeling-runtime-config.v1"] = (
        CONFIG_SCHEMA
    )
    recipe_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evaluation_phase: Literal["iteration", "deferred_comparison"]
    deferred: bool
    semantic_samples_per_prompt: Literal[5] = SEMANTIC_SAMPLES_PER_PROMPT
    semantic_generator: ModelRoleConfig
    semantic_rewriter: ModelRoleConfig
    conservative_control: ModelRoleConfig
    price_snapshot: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_phase_and_transport(self) -> CandidateLabelingRuntimeConfig:
        if self.deferred != (self.evaluation_phase == "deferred_comparison"):
            raise ValueError("deferred must match the deferred_comparison phase")
        for role_name in ROLE_NAMES:
            role = getattr(self, role_name)
            if role.transport != "native_batch":
                raise ValueError(f"{role_name} must use native_batch")
        return self


class PreparedCandidateLabelingRequest(StrictModel):
    """One generation-only request, still unresolved and unsubmitted."""

    schema_version: Literal["adag.bonafide.candidate-labeling-prepared-request.v1"] = (
        REQUEST_SCHEMA
    )
    request_id: str
    stage_id: Literal["semantic_generation", "conservative_control"]
    model_role: Literal["semantic_generator", "conservative_control"]
    logical_prompt_id: str
    arm_id: str
    arm_sha256: str
    anchor_index: int
    cluster_id: int
    sample_index: int | None
    family_partition: Literal["generation"] = "generation"
    generation_only: Literal[True] = True
    selection_audit_visible: Literal[False] = False
    forbidden_input_fields: list[str]
    source_renderer_manifest_sha256: str
    source_prompt_sha256: str
    source_message_payload_sha256: str
    width_evidence_sha256: str
    rendered_candidate_witness_sha256_in_order: list[str] | None
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


class RewriteDependency(StrictModel):
    """A deferred rewrite dependency, not a constructed provider request."""

    schema_version: Literal[
        "adag.bonafide.candidate-labeling-rewrite-dependency.v1"
    ] = REWRITE_DEPENDENCY_SCHEMA
    dependency_id: str
    planned_request_id: str
    stage_id: Literal["semantic_rewrite"] = "semantic_rewrite"
    model_role: Literal["semantic_rewriter"] = "semantic_rewriter"
    logical_prompt_id: str
    arm_id: str
    arm_sha256: str
    source_prompt_sha256: str
    source_message_payload_sha256: str
    required_semantic_request_ids: list[str]
    required_validated_output_count: Literal[5] = SEMANTIC_SAMPLES_PER_PROMPT
    input_policy: Literal[
        "original_generation_prompt_plus_exactly_five_validated_semantic_outputs"
    ] = "original_generation_prompt_plus_exactly_five_validated_semantic_outputs"
    construction_status: Literal["blocked_pending_validated_outputs"] = (
        "blocked_pending_validated_outputs"
    )
    request_constructed: Literal[False] = False
    generation_only: Literal[True] = True
    selection_audit_visible: Literal[False] = False
    forbidden_input_fields: list[str]
    provider: str
    model: str
    transport: Literal["native_batch"] = "native_batch"
    endpoint: None = None
    endpoints_resolved: Literal[False] = False
    calls_made: Literal[False] = False
    role_config_sha256: str
    dependency_sha256: str


@dataclass(frozen=True)
class LoadedCandidateLabelingExecutionCohort:
    root: Path
    manifest: Mapping[str, Any]
    config: CandidateLabelingRuntimeConfig
    initial_requests: tuple[PreparedCandidateLabelingRequest, ...]
    rewrite_dependencies: tuple[RewriteDependency, ...]


def _self_hashed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    if field in result:
        raise ValueError(f"payload already contains {field}")
    result[field] = canonical_sha256(result)
    return result


def _verify_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    recorded = payload.get(field)
    unhashed = dict(payload)
    unhashed.pop(field, None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(unhashed):
        raise ValueError(f"{label} self-hash drift")


def load_candidate_labeling_runtime_config(
    path: Path,
) -> CandidateLabelingRuntimeConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"unreadable candidate labeling runtime config: {path}"
        ) from error
    return CandidateLabelingRuntimeConfig.model_validate(raw)


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
            f"unable to bind execution adapter revision: {message}"
        ) from error
    return completed.stdout.strip()


def _git_blob_bytes(repo_root: Path, blob: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode(errors="replace").strip() or str(error)
        raise ValueError(f"unable to read execution adapter blob: {message}") from error
    return completed.stdout


def _repository_root() -> Path:
    inferred = Path(__file__).resolve().parents[3]
    actual = Path(_git(inferred, "rev-parse", "--show-toplevel")).resolve()
    if inferred != actual:
        raise ValueError("candidate labeling runtime repository root drift")
    return actual


def collect_candidate_labeling_runtime_revision(repo_root: Path) -> dict[str, Any]:
    """Bind a clean producing commit and exact adapter module/CLI blobs."""

    repo_root = repo_root.resolve()
    if Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve() != repo_root:
        raise ValueError("candidate labeling runtime must run from repository root")
    expected_module = repo_root / _RUNTIME_SOURCE_BINDINGS["candidate_labeling_runtime"]
    if Path(__file__).resolve() != expected_module.resolve():
        raise ValueError(
            "candidate labeling runtime was imported from another worktree"
        )
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError("candidate labeling runtime requires a clean tracked worktree")
    commit = _git(repo_root, "rev-parse", "HEAD")
    files: list[dict[str, str]] = []
    for role, relative in _RUNTIME_SOURCE_BINDINGS.items():
        if _git(repo_root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(
                f"candidate labeling runtime source is not tracked: {relative}"
            )
        path = repo_root / relative
        blob = _git(repo_root, "rev-parse", f"{commit}:{relative}")
        blob_sha256 = hashlib.sha256(_git_blob_bytes(repo_root, blob)).hexdigest()
        if not path.is_file() or file_sha256(path) != blob_sha256:
            raise ValueError(
                f"candidate labeling runtime source is not at HEAD: {relative}"
            )
        files.append(
            {
                "role": role,
                "path": relative,
                "git_blob": blob,
                "sha256": blob_sha256,
            }
        )
    return {
        "repo_root": str(repo_root),
        "git_commit": commit,
        "git_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "files": files,
    }


def _validate_runtime_revision_portably(revision: Any) -> None:
    """Validate producing adapter sources from Git objects, not an old worktree."""

    expected_fields = {
        "repo_root",
        "git_commit",
        "git_tree",
        "tracked_worktree_clean",
        "tracked_status_sha256",
        "files",
    }
    if not isinstance(revision, Mapping) or set(revision) != expected_fields:
        raise TypeError("execution adapter recorded revision shape is invalid")
    repo_root = _repository_root()
    commit = revision["git_commit"]
    tree = revision["git_tree"]
    if (
        not isinstance(commit, str)
        or not isinstance(tree, str)
        or revision["tracked_worktree_clean"] is not True
        or revision["tracked_status_sha256"] != hashlib.sha256(b"").hexdigest()
        or _git(repo_root, "cat-file", "-t", commit) != "commit"
        or _git(repo_root, "rev-parse", f"{commit}^{{tree}}") != tree
    ):
        raise ValueError("execution adapter recorded commit/tree drift")
    records = revision["files"]
    if not isinstance(records, list) or len(records) != len(_RUNTIME_SOURCE_BINDINGS):
        raise ValueError("execution adapter recorded source inventory drift")
    expected_inventory = list(_RUNTIME_SOURCE_BINDINGS.items())
    if [
        (record.get("role"), record.get("path"))
        for record in records
        if isinstance(record, Mapping)
    ] != expected_inventory:
        raise ValueError("execution adapter recorded role/path inventory drift")
    for record, (role, relative) in zip(records, expected_inventory, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "role",
            "path",
            "git_blob",
            "sha256",
        }:
            raise TypeError("execution adapter recorded source entry is invalid")
        blob = _git(repo_root, "rev-parse", f"{commit}:{relative}")
        if (
            blob != record["git_blob"]
            or _git(repo_root, "cat-file", "-t", blob) != "blob"
        ):
            raise ValueError(f"execution adapter recorded blob drift: {role}")
        if (
            hashlib.sha256(_git_blob_bytes(repo_root, blob)).hexdigest()
            != record["sha256"]
        ):
            raise ValueError(f"execution adapter recorded file hash drift: {role}")


def _load_candidate_labeling_renderer_portably(
    root: Path,
) -> LoadedCandidateLabelingRenderer:
    """Deep-validate a frozen renderer using its recorded Git-object contract."""

    renderer = load_candidate_labeling_renderer(root, verify_sources=False)
    revision = renderer.manifest.get("code_revision")
    runtime_paths = {
        "candidate_labeling_renderer": Path(renderer_module.__file__),
        **{
            role: Path(str(module.__file__))
            for role, module in renderer_module._RUNTIME_MODULE_BINDINGS.items()
        },
    }
    repo_root = _repository_root()
    renderer_module._validate_recorded_revision_portably(
        revision,
        source_bindings=renderer_module._SOURCE_BINDINGS,
        runtime_paths=runtime_paths,
        current_repo_root=repo_root,
        label="candidate labeling renderer",
    )
    source = renderer.manifest.get("source_labeling_comparison")
    if not isinstance(source, Mapping):
        raise TypeError("candidate labeling renderer source binding is invalid")
    comparison = renderer_module._load_candidate_labeling_comparison_portably(
        Path(str(source["path"])), repo_root=repo_root
    )
    source_manifest_path = Path(str(source["manifest_path"]))
    if (
        comparison.manifest["manifest_sha256"] != source.get("manifest_sha256")
        or comparison.manifest["schema_version"] != source.get("schema_version")
        or source_manifest_path != comparison.root / RENDERER_MANIFEST_FILE
        or file_sha256(source_manifest_path) != source.get("manifest_file_sha256")
        or renderer_module.build_candidate_labeling_renderer(comparison)
        != (
            dict(renderer.witness_selection),
            list(renderer.generation_prompts),
            dict(renderer.stage_plan),
        )
    ):
        raise ValueError("candidate labeling renderer portable recomputation drift")
    renderer_module._validate_recorded_revision_portably(
        revision,
        source_bindings=renderer_module._SOURCE_BINDINGS,
        runtime_paths=runtime_paths,
        current_repo_root=repo_root,
        label="candidate labeling renderer",
    )
    if load_candidate_labeling_renderer(root, verify_sources=False) != renderer:
        raise ValueError("candidate labeling renderer changed during deep validation")
    return renderer


def _role(config: CandidateLabelingRuntimeConfig, name: str) -> ModelRoleConfig:
    if name not in ROLE_NAMES:
        raise ValueError(f"unknown model role: {name}")
    value = getattr(config, name)
    if not isinstance(value, ModelRoleConfig):
        raise TypeError(f"invalid model role: {name}")
    return value


def _role_sha256(config: CandidateLabelingRuntimeConfig, name: str) -> str:
    return canonical_sha256(_role(config, name).model_dump(mode="json"))


def _deterministic_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}:{canonical_sha256(payload)[:32]}"


def _arm_bindings(
    prompts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for prompt in prompts:
        grouped.setdefault(str(prompt["arm_id"]), []).append(prompt)
    bindings: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for arm_id, arm_prompts in grouped.items():
        rows = [
            {
                "logical_prompt_id": str(prompt["logical_prompt_id"]),
                "prompt_sha256": str(prompt["prompt_sha256"]),
                "message_payload_sha256": str(prompt["message_payload_sha256"]),
                "width_evidence_sha256": str(prompt["width_evidence_sha256"]),
                "candidate_evidence_included": bool(
                    prompt["candidate_evidence_included"]
                ),
            }
            for prompt in arm_prompts
        ]
        payload = {
            "arm_id": arm_id,
            "candidate_evidence_included": bool(
                arm_prompts[0]["candidate_evidence_included"]
            ),
            "logical_prompt_count": len(rows),
            "prompt_bindings_in_order": rows,
        }
        binding = _self_hashed(payload, "arm_sha256")
        bindings[arm_id] = binding
        ordered.append(binding)
    return bindings, ordered


def _prompt_bindings(prompts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "logical_prompt_id": str(prompt["logical_prompt_id"]),
            "arm_id": str(prompt["arm_id"]),
            "anchor_index": int(prompt["anchor_index"]),
            "cluster_id": int(prompt["cluster_id"]),
            "prompt_sha256": str(prompt["prompt_sha256"]),
            "message_payload_sha256": str(prompt["message_payload_sha256"]),
            "width_evidence_sha256": str(prompt["width_evidence_sha256"]),
            "rendered_candidate_witness_sha256_in_order": prompt[
                "rendered_candidate_witness_sha256_in_order"
            ],
        }
        for prompt in prompts
    ]


def _request_for_prompt(
    *,
    prompt: Mapping[str, Any],
    renderer_manifest_sha256: str,
    arm_sha256: str,
    config: CandidateLabelingRuntimeConfig,
    config_sha256: str,
    stage_id: Literal["semantic_generation", "conservative_control"],
    role_name: Literal["semantic_generator", "conservative_control"],
    sample_index: int | None,
) -> PreparedCandidateLabelingRequest:
    role = _role(config, role_name)
    identity = {
        "config_sha256": config_sha256,
        "renderer_manifest_sha256": renderer_manifest_sha256,
        "source_prompt_sha256": prompt["prompt_sha256"],
        "stage_id": stage_id,
        "sample_index": sample_index,
    }
    request_id = _deterministic_id(stage_id, identity)
    payload = prompt["message_payload"]
    request = {
        "schema_version": REQUEST_SCHEMA,
        "request_id": request_id,
        "stage_id": stage_id,
        "model_role": role_name,
        "logical_prompt_id": str(prompt["logical_prompt_id"]),
        "arm_id": str(prompt["arm_id"]),
        "arm_sha256": arm_sha256,
        "anchor_index": int(prompt["anchor_index"]),
        "cluster_id": int(prompt["cluster_id"]),
        "sample_index": sample_index,
        "family_partition": "generation",
        "generation_only": True,
        "selection_audit_visible": False,
        "forbidden_input_fields": list(HELDOUT_FORBIDDEN_INPUTS),
        "source_renderer_manifest_sha256": renderer_manifest_sha256,
        "source_prompt_sha256": str(prompt["prompt_sha256"]),
        "source_message_payload_sha256": str(prompt["message_payload_sha256"]),
        "width_evidence_sha256": str(prompt["width_evidence_sha256"]),
        "rendered_candidate_witness_sha256_in_order": prompt[
            "rendered_candidate_witness_sha256_in_order"
        ],
        "provider": role.provider,
        "model": role.model,
        "transport": role.transport,
        "endpoint": None,
        "endpoints_resolved": False,
        "calls_made": False,
        "max_output_tokens": role.max_output_tokens,
        "temperature": role.temperature,
        "reasoning": role.reasoning,
        "provider_parameters": role.provider_parameters,
        "role_config_sha256": _role_sha256(config, role_name),
        "messages": payload["messages"],
        "expected_output_json_schema": payload["expected_output_json_schema"],
        "typed_output_fields": list(TYPED_OUTPUT_FIELDS),
        "status_enum": list(STATUS_ENUM),
    }
    return PreparedCandidateLabelingRequest.model_validate(
        _self_hashed(request, "request_sha256")
    )


def build_candidate_labeling_execution_cohort(
    renderer: LoadedCandidateLabelingRenderer,
    config: CandidateLabelingRuntimeConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build initial requests and rewrite dependencies without external effects."""

    prompts = list(renderer.generation_prompts)
    if len(prompts) != EXPECTED_LOGICAL_PROMPTS:
        raise ValueError("execution cohort requires exactly 24 frozen prompts")
    if config.semantic_samples_per_prompt != SEMANTIC_SAMPLES_PER_PROMPT:
        raise ValueError("execution cohort requires exactly five semantic samples")
    config_payload = config.model_dump(mode="json")
    config_sha256 = canonical_sha256(config_payload)
    renderer_hash = str(renderer.manifest["manifest_sha256"])
    arms, ordered_arms = _arm_bindings(prompts)

    semantic: list[PreparedCandidateLabelingRequest] = []
    controls: list[PreparedCandidateLabelingRequest] = []
    semantic_by_prompt: dict[str, list[str]] = {}
    for prompt in prompts:
        prompt_id = str(prompt["logical_prompt_id"])
        arm_sha256 = str(arms[str(prompt["arm_id"])]["arm_sha256"])
        prompt_requests = [
            _request_for_prompt(
                prompt=prompt,
                renderer_manifest_sha256=renderer_hash,
                arm_sha256=arm_sha256,
                config=config,
                config_sha256=config_sha256,
                stage_id="semantic_generation",
                role_name="semantic_generator",
                sample_index=sample_index,
            )
            for sample_index in range(SEMANTIC_SAMPLES_PER_PROMPT)
        ]
        semantic.extend(prompt_requests)
        semantic_by_prompt[prompt_id] = [
            request.request_id for request in prompt_requests
        ]
        controls.append(
            _request_for_prompt(
                prompt=prompt,
                renderer_manifest_sha256=renderer_hash,
                arm_sha256=arm_sha256,
                config=config,
                config_sha256=config_sha256,
                stage_id="conservative_control",
                role_name="conservative_control",
                sample_index=None,
            )
        )

    dependencies: list[RewriteDependency] = []
    rewrite_role = _role(config, "semantic_rewriter")
    for prompt in prompts:
        identity = {
            "config_sha256": config_sha256,
            "renderer_manifest_sha256": renderer_hash,
            "source_prompt_sha256": prompt["prompt_sha256"],
            "stage_id": "semantic_rewrite",
        }
        dependency = {
            "schema_version": REWRITE_DEPENDENCY_SCHEMA,
            "dependency_id": _deterministic_id("rewrite_dependency", identity),
            "planned_request_id": _deterministic_id("semantic_rewrite", identity),
            "stage_id": "semantic_rewrite",
            "model_role": "semantic_rewriter",
            "logical_prompt_id": str(prompt["logical_prompt_id"]),
            "arm_id": str(prompt["arm_id"]),
            "arm_sha256": str(arms[str(prompt["arm_id"])]["arm_sha256"]),
            "source_prompt_sha256": str(prompt["prompt_sha256"]),
            "source_message_payload_sha256": str(prompt["message_payload_sha256"]),
            "required_semantic_request_ids": semantic_by_prompt[
                str(prompt["logical_prompt_id"])
            ],
            "required_validated_output_count": SEMANTIC_SAMPLES_PER_PROMPT,
            "input_policy": (
                "original_generation_prompt_plus_exactly_five_validated_"
                "semantic_outputs"
            ),
            "construction_status": "blocked_pending_validated_outputs",
            "request_constructed": False,
            "generation_only": True,
            "selection_audit_visible": False,
            "forbidden_input_fields": list(HELDOUT_FORBIDDEN_INPUTS),
            "provider": rewrite_role.provider,
            "model": rewrite_role.model,
            "transport": rewrite_role.transport,
            "endpoint": None,
            "endpoints_resolved": False,
            "calls_made": False,
            "role_config_sha256": _role_sha256(config, "semantic_rewriter"),
        }
        dependencies.append(
            RewriteDependency.model_validate(
                _self_hashed(dependency, "dependency_sha256")
            )
        )

    if (
        len(semantic) != EXPECTED_SEMANTIC_REQUESTS
        or len(controls) != EXPECTED_CONTROL_REQUESTS
        or len(dependencies) != EXPECTED_REWRITE_DEPENDENCIES
    ):
        raise AssertionError("candidate labeling execution cohort cardinality drift")
    requests = [request.model_dump(mode="json") for request in [*semantic, *controls]]
    dependency_rows = [item.model_dump(mode="json") for item in dependencies]
    derived = {
        "config_sha256": config_sha256,
        "prompt_bindings_in_order": _prompt_bindings(prompts),
        "arm_bindings": ordered_arms,
    }
    return requests, dependency_rows, derived


def _price_binding(
    config: CandidateLabelingRuntimeConfig, price_path: Path
) -> dict[str, Any]:
    snapshot = load_price_snapshot(price_path)
    role_rates: list[dict[str, Any]] = []
    for stage_id, role_name in zip(GENERIC_STAGE_IDS, ROLE_NAMES, strict=True):
        role = _role(config, role_name)
        try:
            rates = snapshot["rates"][role.provider][role.model][role.transport]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"price snapshot has no rate for {role_name}: "
                f"{role.provider}/{role.model}/{role.transport}"
            ) from error
        role_rates.append(
            {
                "stage_id": stage_id,
                "model_role": role_name,
                "provider": role.provider,
                "model": role.model,
                "transport": role.transport,
                "rates": rates,
                "rates_sha256": canonical_sha256(rates),
            }
        )
    return {
        "path": str(price_path.resolve()),
        "file_sha256": file_sha256(price_path),
        "snapshot_id": snapshot["snapshot_id"],
        "effective_date": snapshot["effective_date"],
        "currency": snapshot["currency"],
        "unit": snapshot["unit"],
        "role_rates": role_rates,
    }


def _renderer_binding(
    renderer: LoadedCandidateLabelingRenderer, derived: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = renderer.root / RENDERER_MANIFEST_FILE
    return {
        "root": str(renderer.root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "schema_version": renderer.manifest["schema_version"],
        "manifest_sha256": renderer.manifest["manifest_sha256"],
        "manifest_file_sha256": file_sha256(manifest_path),
        "prompt_bindings_in_order": derived["prompt_bindings_in_order"],
        "arm_bindings": derived["arm_bindings"],
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
        for value in values
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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


def prepare_candidate_labeling_execution_cohort(
    *, renderer_root: Path, config_path: Path, output_root: Path
) -> dict[str, Any]:
    """Atomically publish an immutable cohort without resolving or calling APIs."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace execution cohort: {output_root}")
    config_path = config_path.resolve()
    revision = collect_candidate_labeling_runtime_revision(_repository_root())
    renderer = _load_candidate_labeling_renderer_portably(renderer_root)
    config = load_candidate_labeling_runtime_config(config_path)
    price_path = (config_path.parent / config.price_snapshot).resolve()
    price_binding = _price_binding(config, price_path)
    requests, dependencies, derived = build_candidate_labeling_execution_cohort(
        renderer, config
    )
    config_payload = config.model_dump(mode="json")
    config_binding = {
        "path": str(config_path),
        "schema_version": config.schema_version,
        "recipe_id": config.recipe_id,
        "file_sha256": file_sha256(config_path),
        "config_sha256": derived["config_sha256"],
    }

    # Re-read every provenance source before publication to fail closed on drift.
    final_renderer = _load_candidate_labeling_renderer_portably(renderer_root)
    final_config = load_candidate_labeling_runtime_config(config_path)
    if (
        final_renderer != renderer
        or final_config != config
        or file_sha256(config_path) != config_binding["file_sha256"]
        or _price_binding(final_config, price_path) != price_binding
        or collect_candidate_labeling_runtime_revision(_repository_root()) != revision
        or build_candidate_labeling_execution_cohort(final_renderer, final_config)
        != (requests, dependencies, derived)
    ):
        raise ValueError(
            "renderer, runtime config, or price snapshot changed during preparation"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        payloads = {
            CONFIG_SNAPSHOT_FILE: _json_bytes(config_payload),
            INITIAL_REQUESTS_FILE: _jsonl_bytes(requests),
            REWRITE_DEPENDENCIES_FILE: _jsonl_bytes(dependencies),
        }
        for name, payload in payloads.items():
            _write_exclusive(temporary / name, payload)
        files = [
            {
                "path": name,
                "sha256": file_sha256(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
                "row_count": (
                    len(requests)
                    if name == INITIAL_REQUESTS_FILE
                    else len(dependencies)
                    if name == REWRITE_DEPENDENCIES_FILE
                    else 1
                ),
            }
            for name in sorted(payloads)
        ]
        manifest = {
            "schema_version": COHORT_SCHEMA,
            "purpose": "non_billable_provider_configured_execution_cohort",
            "source_renderer": _renderer_binding(renderer, derived),
            "code_revision": revision,
            "runtime_config": config_binding,
            "price_binding": price_binding,
            "generic_stage_ids": list(GENERIC_STAGE_IDS),
            "stage_counts": [
                {
                    "stage_id": "semantic_generation",
                    "planned_request_count": EXPECTED_SEMANTIC_REQUESTS,
                    "constructed_request_count": EXPECTED_SEMANTIC_REQUESTS,
                },
                {
                    "stage_id": "semantic_rewrite",
                    "planned_request_count": EXPECTED_REWRITE_DEPENDENCIES,
                    "constructed_request_count": 0,
                    "blocked_dependency_count": EXPECTED_REWRITE_DEPENDENCIES,
                },
                {
                    "stage_id": "conservative_control",
                    "planned_request_count": EXPECTED_CONTROL_REQUESTS,
                    "constructed_request_count": EXPECTED_CONTROL_REQUESTS,
                },
            ],
            "logical_prompt_count": EXPECTED_LOGICAL_PROMPTS,
            "initial_request_count": len(requests),
            "rewrite_dependency_count": len(dependencies),
            "total_planned_request_count": EXPECTED_TOTAL_PLANNED_REQUESTS,
            "typed_output_fields": list(TYPED_OUTPUT_FIELDS),
            "status_enum": list(STATUS_ENUM),
            "family_partition": "generation",
            "generation_only": True,
            "heldout_scoring_inputs_forbidden": list(HELDOUT_FORBIDDEN_INPUTS),
            "provider_model_endpoints_resolved": False,
            "calls_made": False,
            "files": files,
        }
        manifest = _self_hashed(manifest, "manifest_sha256")
        _write_exclusive(temporary / MANIFEST_FILE, _json_bytes(manifest))
        _publish_directory_no_replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _validate_source_bindings(
    manifest: Mapping[str, Any], *, verify_sources: bool
) -> tuple[LoadedCandidateLabelingRenderer | None, CandidateLabelingRuntimeConfig]:
    _validate_runtime_revision_portably(manifest.get("code_revision"))
    config_binding = manifest.get("runtime_config")
    if not isinstance(config_binding, Mapping):
        raise TypeError("execution cohort runtime config binding is invalid")
    config_path = Path(str(config_binding["path"]))
    config = load_candidate_labeling_runtime_config(config_path)
    if (
        config.schema_version != config_binding.get("schema_version")
        or config.recipe_id != config_binding.get("recipe_id")
        or canonical_sha256(config.model_dump(mode="json"))
        != config_binding.get("config_sha256")
        or file_sha256(config_path) != config_binding.get("file_sha256")
    ):
        raise ValueError("execution cohort runtime config drift")

    price_binding = manifest.get("price_binding")
    if not isinstance(price_binding, Mapping):
        raise TypeError("execution cohort price binding is invalid")
    price_path = Path(str(price_binding["path"]))
    if _price_binding(config, price_path) != dict(price_binding):
        raise ValueError("execution cohort price snapshot drift")

    if not verify_sources:
        return None, config
    renderer_binding = manifest.get("source_renderer")
    if not isinstance(renderer_binding, Mapping):
        raise TypeError("execution cohort renderer binding is invalid")
    renderer = _load_candidate_labeling_renderer_portably(
        Path(str(renderer_binding["root"]))
    )
    derived = build_candidate_labeling_execution_cohort(renderer, config)[2]
    if _renderer_binding(renderer, derived) != dict(renderer_binding):
        raise ValueError("execution cohort renderer drift")
    return renderer, config


def _validate_internal_execution_graph(
    *,
    manifest: Mapping[str, Any],
    config: CandidateLabelingRuntimeConfig,
    requests: Sequence[PreparedCandidateLabelingRequest],
    dependencies: Sequence[RewriteDependency],
) -> None:
    """Validate the complete cohort graph without consulting renderer sources."""

    renderer_binding = manifest.get("source_renderer")
    if not isinstance(renderer_binding, Mapping):
        raise TypeError("execution cohort renderer binding is invalid")
    renderer_hash = renderer_binding.get("manifest_sha256")
    config_binding = manifest.get("runtime_config")
    if not isinstance(config_binding, Mapping):
        raise TypeError("execution cohort runtime config binding is invalid")
    config_hash = config_binding.get("config_sha256")
    prompt_rows = renderer_binding.get("prompt_bindings_in_order")
    arm_rows = renderer_binding.get("arm_bindings")
    if (
        not isinstance(prompt_rows, list)
        or len(prompt_rows) != EXPECTED_LOGICAL_PROMPTS
        or not isinstance(arm_rows, list)
        or len(arm_rows) != 2
    ):
        raise ValueError("execution cohort prompt/arm binding cardinality drift")
    prompt_by_id: dict[str, Mapping[str, Any]] = {}
    for prompt in prompt_rows:
        if not isinstance(prompt, Mapping) or set(prompt) != {
            "logical_prompt_id",
            "arm_id",
            "anchor_index",
            "cluster_id",
            "prompt_sha256",
            "message_payload_sha256",
            "width_evidence_sha256",
            "rendered_candidate_witness_sha256_in_order",
        }:
            raise TypeError("execution cohort prompt binding is invalid")
        prompt_id = prompt.get("logical_prompt_id")
        if not isinstance(prompt_id, str) or prompt_id in prompt_by_id:
            raise ValueError("execution cohort prompt binding identity drift")
        prompt_by_id[prompt_id] = prompt

    arm_by_id: dict[str, Mapping[str, Any]] = {}
    for arm in arm_rows:
        if not isinstance(arm, Mapping):
            raise TypeError("execution cohort arm binding is invalid")
        _verify_self_hash(arm, "arm_sha256", "execution cohort arm binding")
        arm_id = arm.get("arm_id")
        if not isinstance(arm_id, str) or arm_id in arm_by_id:
            raise ValueError("execution cohort arm binding identity drift")
        expected_prompts = [
            {
                "logical_prompt_id": prompt["logical_prompt_id"],
                "prompt_sha256": prompt["prompt_sha256"],
                "message_payload_sha256": prompt["message_payload_sha256"],
                "width_evidence_sha256": prompt["width_evidence_sha256"],
                "candidate_evidence_included": prompt[
                    "rendered_candidate_witness_sha256_in_order"
                ]
                is not None,
            }
            for prompt in prompt_rows
            if prompt["arm_id"] == arm_id
        ]
        if (
            arm.get("logical_prompt_count") != 12
            or arm.get("prompt_bindings_in_order") != expected_prompts
            or arm.get("candidate_evidence_included")
            is not expected_prompts[0]["candidate_evidence_included"]
        ):
            raise ValueError("execution cohort arm/prompt binding drift")
        arm_by_id[arm_id] = arm
    if {prompt["arm_id"] for prompt in prompt_rows} != set(arm_by_id):
        raise ValueError("execution cohort prompt arm inventory drift")

    request_ids = [request.request_id for request in requests]
    request_hashes = [request.request_sha256 for request in requests]
    if len(set(request_ids)) != len(request_ids) or len(set(request_hashes)) != len(
        request_hashes
    ):
        raise ValueError("execution cohort request identity/hash collision")
    semantic_by_prompt: dict[str, dict[int, PreparedCandidateLabelingRequest]] = {}
    controls_by_prompt: dict[str, list[PreparedCandidateLabelingRequest]] = {}
    for request in requests:
        prompt = prompt_by_id.get(request.logical_prompt_id)
        if prompt is None:
            raise ValueError("prepared request references an unknown prompt")
        arm = arm_by_id.get(request.arm_id)
        role_name = request.model_role
        role = _role(config, role_name)
        expected_identity = {
            "config_sha256": config_hash,
            "renderer_manifest_sha256": renderer_hash,
            "source_prompt_sha256": prompt["prompt_sha256"],
            "stage_id": request.stage_id,
            "sample_index": request.sample_index,
        }
        message_payload = {
            "messages": [
                message.model_dump(mode="json") for message in request.messages
            ],
            "expected_output_json_schema": request.expected_output_json_schema,
        }
        status = request.expected_output_json_schema.get("properties", {}).get(
            "status", {}
        )
        if (
            arm is None
            or request.arm_sha256 != arm.get("arm_sha256")
            or request.arm_id != prompt["arm_id"]
            or request.anchor_index != prompt["anchor_index"]
            or request.cluster_id != prompt["cluster_id"]
            or request.source_renderer_manifest_sha256 != renderer_hash
            or request.source_prompt_sha256 != prompt["prompt_sha256"]
            or request.source_message_payload_sha256 != prompt["message_payload_sha256"]
            or request.width_evidence_sha256 != prompt["width_evidence_sha256"]
            or request.rendered_candidate_witness_sha256_in_order
            != prompt["rendered_candidate_witness_sha256_in_order"]
            or canonical_sha256(message_payload) != prompt["message_payload_sha256"]
            or request.request_id
            != _deterministic_id(request.stage_id, expected_identity)
            or request.provider != role.provider
            or request.model != role.model
            or request.transport != role.transport
            or request.max_output_tokens != role.max_output_tokens
            or request.temperature != role.temperature
            or request.reasoning != role.reasoning
            or request.provider_parameters != role.provider_parameters
            or request.role_config_sha256 != _role_sha256(config, role_name)
            or request.expected_output_json_schema.get("required")
            != list(TYPED_OUTPUT_FIELDS)
            or status.get("enum") != list(STATUS_ENUM)
        ):
            raise ValueError("prepared request prompt, arm, role, or schema drift")
        if request.stage_id == "semantic_generation":
            if (
                role_name != "semantic_generator"
                or type(request.sample_index) is not int
            ):
                raise ValueError("semantic request role or sample-index drift")
            by_index = semantic_by_prompt.setdefault(request.logical_prompt_id, {})
            if request.sample_index in by_index:
                raise ValueError("semantic request repeats a prompt sample index")
            by_index[request.sample_index] = request
        else:
            if role_name != "conservative_control" or request.sample_index is not None:
                raise ValueError("control request role or sample-index drift")
            controls_by_prompt.setdefault(request.logical_prompt_id, []).append(request)
    if set(semantic_by_prompt) != set(prompt_by_id) or any(
        set(by_index) != set(range(SEMANTIC_SAMPLES_PER_PROMPT))
        for by_index in semantic_by_prompt.values()
    ):
        raise ValueError("semantic request prompt/sample graph drift")
    if set(controls_by_prompt) != set(prompt_by_id) or any(
        len(values) != 1 for values in controls_by_prompt.values()
    ):
        raise ValueError("control request prompt graph drift")

    dependency_ids = [dependency.dependency_id for dependency in dependencies]
    planned_ids = [dependency.planned_request_id for dependency in dependencies]
    if (
        len(set(dependency_ids)) != len(dependency_ids)
        or len(set(planned_ids)) != len(planned_ids)
        or len(dependencies) != len(prompt_by_id)
    ):
        raise ValueError("rewrite dependency identity graph drift")
    dependency_prompts: set[str] = set()
    referenced_semantic_ids: list[str] = []
    rewrite_role = _role(config, "semantic_rewriter")
    for dependency in dependencies:
        prompt = prompt_by_id.get(dependency.logical_prompt_id)
        arm = arm_by_id.get(dependency.arm_id)
        if prompt is None or dependency.logical_prompt_id in dependency_prompts:
            raise ValueError("rewrite dependency prompt graph drift")
        identity = {
            "config_sha256": config_hash,
            "renderer_manifest_sha256": renderer_hash,
            "source_prompt_sha256": prompt["prompt_sha256"],
            "stage_id": "semantic_rewrite",
        }
        expected_semantic_ids = [
            semantic_by_prompt[dependency.logical_prompt_id][index].request_id
            for index in range(SEMANTIC_SAMPLES_PER_PROMPT)
        ]
        if (
            arm is None
            or dependency.arm_id != prompt["arm_id"]
            or dependency.arm_sha256 != arm.get("arm_sha256")
            or dependency.source_prompt_sha256 != prompt["prompt_sha256"]
            or dependency.source_message_payload_sha256
            != prompt["message_payload_sha256"]
            or dependency.required_semantic_request_ids != expected_semantic_ids
            or dependency.dependency_id
            != _deterministic_id("rewrite_dependency", identity)
            or dependency.planned_request_id
            != _deterministic_id("semantic_rewrite", identity)
            or dependency.provider != rewrite_role.provider
            or dependency.model != rewrite_role.model
            or dependency.transport != rewrite_role.transport
            or dependency.role_config_sha256
            != _role_sha256(config, "semantic_rewriter")
        ):
            raise ValueError(
                "rewrite dependency prompt, arm, role, or input graph drift"
            )
        dependency_prompts.add(dependency.logical_prompt_id)
        referenced_semantic_ids.extend(dependency.required_semantic_request_ids)
    if (
        dependency_prompts != set(prompt_by_id)
        or set(referenced_semantic_ids)
        != {
            request.request_id
            for request in requests
            if request.stage_id == "semantic_generation"
        }
        or len(referenced_semantic_ids) != EXPECTED_SEMANTIC_REQUESTS
    ):
        raise ValueError("rewrite dependency semantic-reference graph drift")


def load_candidate_labeling_execution_cohort(
    root: Path, *, verify_sources: bool = True
) -> LoadedCandidateLabelingExecutionCohort:
    """Load and deeply validate one immutable prepared execution cohort."""

    root = root.resolve()
    manifest = load_json_object(root / MANIFEST_FILE)
    _verify_self_hash(manifest, "manifest_sha256", "execution cohort manifest")
    if (
        manifest.get("schema_version") != COHORT_SCHEMA
        or manifest.get("purpose")
        != "non_billable_provider_configured_execution_cohort"
        or manifest.get("generic_stage_ids") != list(GENERIC_STAGE_IDS)
        or manifest.get("logical_prompt_count") != EXPECTED_LOGICAL_PROMPTS
        or manifest.get("initial_request_count")
        != EXPECTED_SEMANTIC_REQUESTS + EXPECTED_CONTROL_REQUESTS
        or manifest.get("rewrite_dependency_count") != EXPECTED_REWRITE_DEPENDENCIES
        or manifest.get("total_planned_request_count")
        != EXPECTED_TOTAL_PLANNED_REQUESTS
        or manifest.get("family_partition") != "generation"
        or manifest.get("generation_only") is not True
        or manifest.get("provider_model_endpoints_resolved") is not False
        or manifest.get("calls_made") is not False
        or manifest.get("typed_output_fields") != list(TYPED_OUTPUT_FIELDS)
        or manifest.get("status_enum") != list(STATUS_ENUM)
        or manifest.get("heldout_scoring_inputs_forbidden")
        != list(HELDOUT_FORBIDDEN_INPUTS)
    ):
        raise ValueError("candidate labeling execution cohort contract drift")
    expected_stage_counts = [
        {
            "stage_id": "semantic_generation",
            "planned_request_count": 120,
            "constructed_request_count": 120,
        },
        {
            "stage_id": "semantic_rewrite",
            "planned_request_count": 24,
            "constructed_request_count": 0,
            "blocked_dependency_count": 24,
        },
        {
            "stage_id": "conservative_control",
            "planned_request_count": 24,
            "constructed_request_count": 24,
        },
    ]
    if manifest.get("stage_counts") != expected_stage_counts:
        raise ValueError("execution cohort stage-count drift")

    files = manifest.get("files")
    if not isinstance(files, list) or any(
        not isinstance(item, Mapping) for item in files
    ):
        raise TypeError("execution cohort file inventory is invalid")
    inventory = {str(item["path"]): item for item in files}
    if len(inventory) != len(files) or set(inventory) != {
        CONFIG_SNAPSHOT_FILE,
        INITIAL_REQUESTS_FILE,
        REWRITE_DEPENDENCIES_FILE,
    }:
        raise ValueError("execution cohort file inventory drift")
    for name, item in inventory.items():
        path = root / name
        if (
            not path.is_file()
            or file_sha256(path) != item.get("sha256")
            or path.stat().st_size != item.get("size_bytes")
        ):
            raise ValueError(f"execution cohort file drift: {name}")

    config_snapshot = CandidateLabelingRuntimeConfig.model_validate(
        load_json_object(root / CONFIG_SNAPSHOT_FILE)
    )
    renderer, source_config = _validate_source_bindings(
        manifest, verify_sources=verify_sources
    )
    if config_snapshot != source_config:
        raise ValueError("execution cohort config snapshot drift")

    request_rows = _load_jsonl(root / INITIAL_REQUESTS_FILE)
    dependency_rows = _load_jsonl(root / REWRITE_DEPENDENCIES_FILE)
    requests = tuple(
        PreparedCandidateLabelingRequest.model_validate(row) for row in request_rows
    )
    dependencies = tuple(
        RewriteDependency.model_validate(row) for row in dependency_rows
    )
    if (
        len(requests) != EXPECTED_SEMANTIC_REQUESTS + EXPECTED_CONTROL_REQUESTS
        or len(dependencies) != EXPECTED_REWRITE_DEPENDENCIES
        or inventory[INITIAL_REQUESTS_FILE].get("row_count") != len(requests)
        or inventory[REWRITE_DEPENDENCIES_FILE].get("row_count") != len(dependencies)
        or inventory[CONFIG_SNAPSHOT_FILE].get("row_count") != 1
    ):
        raise ValueError("execution cohort persisted row-count drift")
    for request in requests:
        payload = request.model_dump(mode="json")
        _verify_self_hash(payload, "request_sha256", "prepared request")
        if (
            request.forbidden_input_fields != list(HELDOUT_FORBIDDEN_INPUTS)
            or request.typed_output_fields != list(TYPED_OUTPUT_FIELDS)
            or request.status_enum != list(STATUS_ENUM)
        ):
            raise ValueError("prepared request generation-only fence drift")
    for dependency in dependencies:
        payload = dependency.model_dump(mode="json")
        _verify_self_hash(payload, "dependency_sha256", "rewrite dependency")
        if (
            dependency.forbidden_input_fields != list(HELDOUT_FORBIDDEN_INPUTS)
            or len(dependency.required_semantic_request_ids)
            != SEMANTIC_SAMPLES_PER_PROMPT
            or len(set(dependency.required_semantic_request_ids))
            != SEMANTIC_SAMPLES_PER_PROMPT
        ):
            raise ValueError("rewrite dependency fence or cardinality drift")

    _validate_internal_execution_graph(
        manifest=manifest,
        config=source_config,
        requests=requests,
        dependencies=dependencies,
    )

    if renderer is not None:
        expected_requests, expected_dependencies, _ = (
            build_candidate_labeling_execution_cohort(renderer, source_config)
        )
        if (
            list(request_rows) != expected_requests
            or list(dependency_rows) != expected_dependencies
        ):
            raise ValueError("execution cohort request derivation drift")
    return LoadedCandidateLabelingExecutionCohort(
        root=root,
        manifest=manifest,
        config=config_snapshot,
        initial_requests=requests,
        rewrite_dependencies=dependencies,
    )
