"""Resumable, receipt-bound OpenAI Batch lifecycle for candidate labeling."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from circuits.analysis.bonafide.candidate_labeling_execution import (
    CandidateLabelingRewriteRequest,
    construct_candidate_labeling_rewrite_request,
)
from circuits.analysis.bonafide.candidate_labeling_openai_batch import (
    CandidateBatchRequest,
    CandidateOpenAIBatchResult,
    openai_candidate_batch_line,
    parse_openai_candidate_batch_row,
    prepare_openai_candidate_batch_input,
)
from circuits.analysis.bonafide.candidate_labeling_runtime import (
    LoadedCandidateLabelingExecutionCohort,
    PreparedCandidateLabelingRequest,
    RewriteDependency,
    load_candidate_labeling_execution_cohort,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
    load_json_object,
)
from circuits.labeling.batch import (
    download_openai_batch_files,
    download_openai_file_bytes,
    retrieve_batch,
    submit_openai_batch,
)
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
)
from circuits.labeling.pricing import estimate_cost, load_price_snapshot
from circuits.labeling.schema import CostEstimate, StrictModel

RUN_SCHEMA = "adag.bonafide.candidate-labeling-openai-run.v1"
COST_PLAN_SCHEMA = "adag.bonafide.candidate-labeling-openai-cost-plan.v1"
COLLECTION_SCHEMA = "adag.bonafide.candidate-labeling-openai-collection.v1"
EVENT_SCHEMA = "adag.bonafide.candidate-labeling-openai-event.v1"
SELECTION_SCHEMA = "adag.bonafide.candidate-labeling-openai-selection.v1"
SUBMISSION_INTENT_SCHEMA = (
    "adag.bonafide.candidate-labeling-openai-submission-intent.v1"
)
SUBMISSION_SCHEMA = "adag.bonafide.candidate-labeling-openai-submission.v1"
STATUS_SCHEMA = "adag.bonafide.candidate-labeling-openai-status.v1"
DOWNLOAD_SCHEMA = "adag.bonafide.candidate-labeling-openai-download.v1"

RUN_MANIFEST_FILE = "run-manifest.json"
COST_PLAN_FILE = "cost-plan.json"
REWRITE_REQUESTS_FILE = "rewrite-requests.jsonl"
STAGES = ("semantic_generation", "conservative_control", "semantic_rewrite")
_SOURCE_BINDINGS = {
    "candidate_labeling_openai_run": (
        "circuits/analysis/bonafide/candidate_labeling_openai_run.py"
    ),
    "candidate_labeling_openai_run_cli": (
        "scripts/bonafide/candidate_labeling_openai_run.py"
    ),
    "candidate_labeling_openai_batch": (
        "circuits/analysis/bonafide/candidate_labeling_openai_batch.py"
    ),
    "candidate_labeling_execution": (
        "circuits/analysis/bonafide/candidate_labeling_execution.py"
    ),
    "candidate_labeling_runtime": (
        "circuits/analysis/bonafide/candidate_labeling_runtime.py"
    ),
    "bonafide_canonical": "circuits/analysis/bonafide/canonical.py",
    "labeling_api": "circuits/labeling/api.py",
    "labeling_batch": "circuits/labeling/batch.py",
    "labeling_io": "circuits/labeling/io.py",
    "labeling_pricing": "circuits/labeling/pricing.py",
    "labeling_schema": "circuits/labeling/schema.py",
}


class CandidateOpenAIBatchEvent(StrictModel):
    """One exact parsed result plus usage-priced telemetry."""

    schema_version: Literal["adag.bonafide.candidate-labeling-openai-event.v1"] = (
        EVENT_SCHEMA
    )
    request_id: str
    request_sha256: str
    stage_id: str
    result: CandidateOpenAIBatchResult
    result_sha256: str
    price_snapshot_id: str
    price_snapshot_file_sha256: str
    cost: CostEstimate
    event_sha256: str


@dataclass(frozen=True)
class LoadedCandidateOpenAIRun:
    root: Path
    manifest: Mapping[str, Any]
    cohort: LoadedCandidateLabelingExecutionCohort
    initial_requests: tuple[PreparedCandidateLabelingRequest, ...]
    dependencies: tuple[RewriteDependency, ...]
    rewrite_requests: tuple[CandidateLabelingRewriteRequest, ...]
    events: tuple[CandidateOpenAIBatchEvent, ...]


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
        raise ValueError(f"unable to bind OpenAI run revision: {message}") from error
    return completed.stdout.strip()


def collect_candidate_openai_execution_revision() -> dict[str, Any]:
    """Bind provider-facing artifacts to committed lifecycle sources at HEAD."""

    inferred = Path(__file__).resolve().parents[3]
    root = Path(_git(inferred, "rev-parse", "--show-toplevel")).resolve()
    if root != inferred:
        raise ValueError("candidate OpenAI run repository root drift")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise ValueError("candidate OpenAI run requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for role, relative in _SOURCE_BINDINGS.items():
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"candidate OpenAI run source is untracked: {relative}")
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
            raise ValueError(f"candidate OpenAI run source is not at HEAD: {relative}")
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
    def valid_hex(item: Any, length: int) -> bool:
        return (
            isinstance(item, str)
            and len(item) == length
            and all(character in "0123456789abcdef" for character in item)
        )

    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("repo_root"), str)
        or not value["repo_root"]
        or not valid_hex(value.get("git_commit"), 40)
        or not valid_hex(value.get("git_tree"), 40)
        or value.get("tracked_worktree_clean") is not True
        or not valid_hex(value.get("tracked_status_sha256"), 64)
        or not isinstance(value.get("files"), list)
    ):
        raise ValueError("candidate OpenAI execution revision binding is malformed")
    observed: dict[str, str] = {}
    paths: set[str] = set()
    for item in value["files"]:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"role", "path", "git_blob", "sha256"}
            or not isinstance(item.get("role"), str)
            or not isinstance(item.get("path"), str)
            or not valid_hex(item.get("git_blob"), 40)
            or not valid_hex(item.get("sha256"), 64)
            or item["role"] in observed
            or item["path"] in paths
        ):
            raise ValueError(
                "candidate OpenAI execution revision file binding is malformed"
            )
        observed[item["role"]] = item["path"]
        paths.add(item["path"])
    if observed != _SOURCE_BINDINGS:
        raise ValueError("candidate OpenAI execution source inventory drift")
    if verify_sources and dict(value) != collect_candidate_openai_execution_revision():
        raise ValueError("candidate OpenAI execution revision no longer matches HEAD")


def _selection(
    cohort: LoadedCandidateLabelingExecutionCohort,
    *,
    selection_kind: Literal["full", "paired_anchor_smoke"],
    anchor_index: int | None,
) -> tuple[tuple[PreparedCandidateLabelingRequest, ...], tuple[RewriteDependency, ...]]:
    if selection_kind == "full":
        if anchor_index is not None:
            raise ValueError("full selection cannot specify anchor_index")
        requests = cohort.initial_requests
        dependencies = cohort.rewrite_dependencies
    else:
        if anchor_index is None:
            raise ValueError("paired_anchor_smoke requires anchor_index")
        requests = tuple(
            request
            for request in cohort.initial_requests
            if request.anchor_index == anchor_index
        )
        prompt_ids = {request.logical_prompt_id for request in requests}
        dependencies = tuple(
            dependency
            for dependency in cohort.rewrite_dependencies
            if dependency.logical_prompt_id in prompt_ids
        )
        arms = {request.arm_id for request in requests}
        counts = Counter(request.stage_id for request in requests)
        if (
            len(arms) != 2
            or len(prompt_ids) != 2
            or counts != {"semantic_generation": 10, "conservative_control": 2}
            or len(dependencies) != 2
        ):
            raise ValueError("paired smoke must contain both complete arms")
    expected_initial = 144 if selection_kind == "full" else 12
    expected_dependencies = 24 if selection_kind == "full" else 2
    if len(requests) != expected_initial or len(dependencies) != expected_dependencies:
        raise ValueError("candidate-labeling run selection cardinality drift")
    selected_ids = {request.request_id for request in requests}
    if any(
        set(dependency.required_semantic_request_ids) - selected_ids
        for dependency in dependencies
    ):
        raise ValueError("selection leaves a partial rewrite dependency")
    return requests, dependencies


def initialize_candidate_openai_run(
    *,
    cohort_root: Path,
    output_root: Path,
    selection_kind: Literal["full", "paired_anchor_smoke"],
    anchor_index: int | None = None,
    verify_sources: bool = True,
) -> LoadedCandidateOpenAIRun:
    """Initialize, or exactly resume, a run without making provider calls."""

    cohort = load_candidate_labeling_execution_cohort(
        cohort_root, verify_sources=verify_sources
    )
    if any(request.provider != "openai" for request in cohort.initial_requests):
        raise ValueError("candidate OpenAI run requires an OpenAI cohort")
    requests, dependencies = _selection(
        cohort, selection_kind=selection_kind, anchor_index=anchor_index
    )
    root = output_root.resolve()
    selection = _self_hashed(
        {
            "schema_version": SELECTION_SCHEMA,
            "kind": selection_kind,
            "anchor_index": anchor_index,
            "paired_arms_required": True,
            "request_bindings_in_order": [
                {"request_id": item.request_id, "request_sha256": item.request_sha256}
                for item in requests
            ],
            "dependency_bindings_in_order": [
                {
                    "dependency_id": item.dependency_id,
                    "dependency_sha256": item.dependency_sha256,
                }
                for item in dependencies
            ],
        },
        "selection_sha256",
    )
    expected = _self_hashed(
        {
            "schema_version": RUN_SCHEMA,
            "purpose": "receipt_bound_resumable_openai_native_batch_evaluation",
            "source_cohort_root": str(cohort.root),
            "source_cohort_manifest_sha256": cohort.manifest["manifest_sha256"],
            "source_cohort_manifest_file_sha256": file_sha256(
                cohort.root / "manifest.json"
            ),
            "source_price_binding": cohort.manifest["price_binding"],
            "execution_revision": collect_candidate_openai_execution_revision(),
            "selection": selection,
            "initial_request_count": len(requests),
            "rewrite_request_count_planned": len(dependencies),
            "total_request_count_planned": len(requests) + len(dependencies),
            "generation_only": True,
            "selection_audit_visible": False,
            "provider_submission_implemented": True,
            "provider_calls_supported_by_runtime": True,
            "provider_output_collection_supported": True,
            "provider_receipt_required_for_collection": True,
        },
        "run_manifest_sha256",
    )
    path = root / RUN_MANIFEST_FILE
    if path.exists():
        observed = load_json_object(path)
        _verify_self_hash(observed, "run_manifest_sha256", "OpenAI run manifest")
        if observed != expected:
            raise ValueError("OpenAI run manifest, selection, or cohort drift")
    else:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError("OpenAI run root is non-empty without a manifest")
        atomic_write_json(path, expected)
    return load_candidate_openai_run(
        root, cohort_root=cohort_root, verify_sources=verify_sources
    )


def _stage_requests(
    run: LoadedCandidateOpenAIRun, stage_id: str
) -> tuple[CandidateBatchRequest, ...]:
    if stage_id == "semantic_rewrite":
        return run.rewrite_requests
    return tuple(
        request for request in run.initial_requests if request.stage_id == stage_id
    )


def _batch_paths(root: Path, stage_id: str) -> tuple[Path, Path]:
    input_path = root / "batches" / stage_id / "input.jsonl"
    return input_path, input_path.with_name("input.jsonl.manifest.json")


def _validate_batch_input(
    root: Path, stage_id: str, requests: Sequence[CandidateBatchRequest]
) -> Mapping[str, Any] | None:
    input_path, manifest_path = _batch_paths(root, stage_id)
    if not input_path.exists() and not manifest_path.exists():
        return None
    if not input_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"partial persisted batch input for {stage_id}")
    observed = load_json_object(manifest_path)
    _verify_self_hash(observed, "manifest_sha256", f"batch input {stage_id}")
    expected_bindings = [
        {"request_id": item.request_id, "request_sha256": item.request_sha256}
        for item in requests
    ]
    expected_model = requests[0].model if requests else None
    if (
        observed.get("stage_id") != stage_id
        or observed.get("model") != expected_model
        or observed.get("request_bindings_in_order") != expected_bindings
        or observed.get("input_file") != input_path.name
        or observed.get("input_file_sha256") != file_sha256(input_path)
        or observed.get("request_count") != len(requests)
    ):
        raise ValueError(f"persisted batch input or manifest drift: {stage_id}")
    expected_rows = [openai_candidate_batch_line(request) for request in requests]
    if read_jsonl(input_path) != expected_rows:
        raise ValueError(f"persisted batch request body drift: {stage_id}")
    return observed


def prepare_candidate_openai_batch(
    *, run_root: Path, stage_id: str, verify_sources: bool = True
) -> Mapping[str, Any]:
    """Prepare or validate one local Batch input; never submit it."""

    if stage_id not in STAGES:
        raise ValueError(f"unsupported stage: {stage_id}")
    run = load_candidate_openai_run(run_root, verify_sources=verify_sources)
    requests = _stage_requests(run, stage_id)
    if not requests:
        if stage_id == "semantic_rewrite":
            raise ValueError("semantic rewrites have not been constructed")
        raise ValueError(f"run has no requests for {stage_id}")
    input_path, manifest_path = _batch_paths(run.root, stage_id)
    observed = _validate_batch_input(run.root, stage_id, requests)
    if observed is not None:
        return observed
    batches_root = input_path.parent.parent
    batches_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{stage_id}.", dir=batches_root))
    try:
        prepare_openai_candidate_batch_input(requests, temporary / input_path.name)
        if input_path.parent.exists():
            raise FileExistsError(f"batch stage appeared concurrently: {stage_id}")
        temporary.replace(input_path.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load_json_object(manifest_path)


def _prompt_token_proxy(request: CandidateBatchRequest) -> int:
    """Conservative body-byte proxy: one UTF-8 byte is charged as one token."""

    body = openai_candidate_batch_line(request)["body"]
    return max(1, len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + 256)


def _rate_for_stage(run: LoadedCandidateOpenAIRun, stage_id: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in run.manifest["source_price_binding"]["role_rates"]
        if item["stage_id"] == stage_id
    ]
    if len(matches) != 1:
        raise ValueError(f"price binding drift for {stage_id}")
    return matches[0]["rates"]


def _candidate_openai_cost_plan(
    run: LoadedCandidateOpenAIRun, max_cumulative_cost_usd: float
) -> dict[str, Any]:
    """Reconstruct the deterministic planning proxy for one selected run."""

    if not math.isfinite(max_cumulative_cost_usd) or max_cumulative_cost_usd < 0:
        raise ValueError("max cumulative cost must be finite and non-negative")
    semantic = [
        item for item in run.initial_requests if item.stage_id == "semantic_generation"
    ]
    source_by_prompt = {item.logical_prompt_id: item for item in semantic}
    stage_rows = []
    cumulative = 0.0
    for stage_id in STAGES:
        if stage_id == "semantic_rewrite":
            count = len(run.dependencies)
            output_ceiling = run.cohort.config.semantic_rewriter.max_output_tokens
            input_proxies = [
                _prompt_token_proxy(source_by_prompt[item.logical_prompt_id])
                + 5 * run.cohort.config.semantic_generator.max_output_tokens
                + 512
                for item in run.dependencies
            ]
        else:
            requests = [
                item for item in run.initial_requests if item.stage_id == stage_id
            ]
            count = len(requests)
            output_ceiling = requests[0].max_output_tokens
            input_proxies = [_prompt_token_proxy(item) for item in requests]
        rates = _rate_for_stage(run, stage_id)
        projected = (
            sum(input_proxies) * float(rates["input_per_million"])
            + count * output_ceiling * float(rates["output_per_million"])
        ) / 1_000_000
        cumulative += projected
        stage_rows.append(
            {
                "stage_id": stage_id,
                "request_count": count,
                "input_token_proxy_total": sum(input_proxies),
                "max_output_tokens_per_request": output_ceiling,
                "max_output_tokens_total": count * output_ceiling,
                "projected_cost_proxy_usd": projected,
                "cumulative_projected_cost_proxy_usd": cumulative,
            }
        )
    if max_cumulative_cost_usd + 1e-12 < cumulative:
        raise ValueError(
            f"cost guard ${max_cumulative_cost_usd:.6f} is below projected "
            f"cost proxy ${cumulative:.6f}"
        )
    return _self_hashed(
        {
            "schema_version": COST_PLAN_SCHEMA,
            "run_manifest_sha256": run.manifest["run_manifest_sha256"],
            "method": (
                "utf8_request_body_byte_input_proxy_no_cache_plus_configured_"
                "output_ceiling"
            ),
            "price_snapshot_id": run.manifest["source_price_binding"]["snapshot_id"],
            "price_snapshot_file_sha256": run.manifest["source_price_binding"][
                "file_sha256"
            ],
            "stages": stage_rows,
            "request_count": sum(item["request_count"] for item in stage_rows),
            "projected_cost_proxy_usd": cumulative,
            "caller_max_cumulative_cost_usd": max_cumulative_cost_usd,
            "planning_guard_passed": True,
            "submission_authorized": False,
        },
        "cost_plan_sha256",
    )


def build_candidate_openai_cost_plan(
    *,
    run_root: Path,
    max_cumulative_cost_usd: float,
    verify_sources: bool = True,
) -> Mapping[str, Any]:
    """Persist a conservative planning proxy; this does not authorize spend."""

    run = load_candidate_openai_run(run_root, verify_sources=verify_sources)
    expected = _candidate_openai_cost_plan(run, max_cumulative_cost_usd)
    path = run.root / COST_PLAN_FILE
    if path.exists():
        observed = load_json_object(path)
        _verify_self_hash(observed, "cost_plan_sha256", "cost plan")
        if observed != expected:
            raise ValueError("persisted cost plan or caller guard drift")
    else:
        atomic_write_json(path, expected)
    return expected


def _required_cost_plan(run: LoadedCandidateOpenAIRun) -> Mapping[str, Any]:
    path = run.root / COST_PLAN_FILE
    if not path.is_file():
        raise ValueError("provider submission requires a persisted cost plan")
    _validate_cost_plan(run)
    return load_json_object(path)


def _role_for_stage(run: LoadedCandidateOpenAIRun, stage_id: str) -> Any:
    role_name = {
        "semantic_generation": "semantic_generator",
        "conservative_control": "conservative_control",
        "semantic_rewrite": "semantic_rewriter",
    }.get(stage_id)
    if role_name is None:
        raise ValueError(f"unsupported stage: {stage_id}")
    return getattr(run.cohort.config, role_name)


def _provider_stage_root(root: Path, stage_id: str) -> Path:
    return root / "provider" / stage_id


def _submission_intent_payload(
    run: LoadedCandidateOpenAIRun,
    stage_id: str,
    requests: Sequence[CandidateBatchRequest],
    cost_plan: Mapping[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    input_path, input_manifest_path = _batch_paths(run.root, stage_id)
    input_manifest = load_json_object(input_manifest_path)
    return {
        "schema_version": SUBMISSION_INTENT_SCHEMA,
        "run_manifest_sha256": run.manifest["run_manifest_sha256"],
        "cost_plan_sha256": cost_plan["cost_plan_sha256"],
        "stage_id": stage_id,
        "provider": "openai",
        "endpoint": "/v1/responses",
        "completion_window": "24h",
        "model": requests[0].model,
        "request_count": len(requests),
        "request_bindings_in_order": [
            {"request_id": item.request_id, "request_sha256": item.request_sha256}
            for item in requests
        ],
        "batch_input_file": input_path.relative_to(run.root).as_posix(),
        "batch_input_file_sha256": file_sha256(input_path),
        "batch_input_manifest_file_sha256": file_sha256(input_manifest_path),
        "batch_input_manifest_sha256": input_manifest["manifest_sha256"],
        "created_at": created_at,
    }


def _load_submission_intent(
    run: LoadedCandidateOpenAIRun, stage_id: str
) -> Mapping[str, Any] | None:
    path = _provider_stage_root(run.root, stage_id) / "submission-intent.json"
    if not path.exists():
        return None
    value = load_json_object(path)
    _verify_self_hash(value, "intent_sha256", f"submission intent {stage_id}")
    created_at = value.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError(f"submission intent timestamp is malformed: {stage_id}")
    requests = _stage_requests(run, stage_id)
    if not requests:
        raise ValueError(f"submission intent exists without requests: {stage_id}")
    cost_plan = _required_cost_plan(run)
    expected = _self_hashed(
        _submission_intent_payload(
            run, stage_id, requests, cost_plan, created_at=created_at
        ),
        "intent_sha256",
    )
    if value != expected:
        raise ValueError(f"submission intent binding drift: {stage_id}")
    return value


def _validate_provider_batch_payload(value: Any, *, batch_id: str | None) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != "adag.labeling.provider-batch.v1"
        or value.get("provider") != "openai"
        or not isinstance(value.get("batch_id"), str)
        or not value["batch_id"]
        or (batch_id is not None and value["batch_id"] != batch_id)
        or not isinstance(value.get("status"), str)
        or not value["status"]
    ):
        raise ValueError("OpenAI provider batch payload is malformed")
    for field in ("input_file_id", "output_file_id", "error_file_id"):
        item = value.get(field)
        if item is not None and (not isinstance(item, str) or not item):
            raise ValueError(f"OpenAI provider batch {field} is malformed")
    if value.get("endpoint") not in (None, "/v1/responses"):
        raise ValueError("OpenAI provider batch endpoint drift")
    if value.get("completion_window") not in (None, "24h"):
        raise ValueError("OpenAI provider batch completion window drift")
    if not isinstance(value.get("metadata"), Mapping):
        raise TypeError("OpenAI provider batch metadata is malformed")


def _expected_provider_metadata(
    run: LoadedCandidateOpenAIRun, stage_id: str
) -> dict[str, str]:
    return {
        "run_id": str(run.manifest["run_manifest_sha256"]),
        "stage": stage_id,
    }


def _load_submission(
    run: LoadedCandidateOpenAIRun, stage_id: str
) -> Mapping[str, Any] | None:
    stage_root = _provider_stage_root(run.root, stage_id)
    path = stage_root / "submission.json"
    intent = _load_submission_intent(run, stage_id)
    if not path.exists():
        return None
    if intent is None:
        raise ValueError(f"submission exists without intent: {stage_id}")
    value = load_json_object(path)
    _verify_self_hash(value, "submission_sha256", f"submission {stage_id}")
    provider = value.get("provider_response")
    _validate_provider_batch_payload(provider, batch_id=None)
    input_path, input_manifest_path = _batch_paths(run.root, stage_id)
    if (
        value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("run_manifest_sha256") != run.manifest["run_manifest_sha256"]
        or value.get("cost_plan_sha256") != intent["cost_plan_sha256"]
        or value.get("intent_sha256") != intent["intent_sha256"]
        or value.get("stage_id") != stage_id
        or value.get("provider") != "openai"
        or value.get("endpoint") != "/v1/responses"
        or value.get("completion_window") != "24h"
        or value.get("model") != intent["model"]
        or value.get("batch_input_file_sha256") != file_sha256(input_path)
        or value.get("batch_input_manifest_file_sha256")
        != file_sha256(input_manifest_path)
        or value.get("batch_id") != provider["batch_id"]
        or value.get("input_file_id") != provider["input_file_id"]
        or value.get("status_at_submission") != provider["status"]
        or provider.get("endpoint") != "/v1/responses"
        or provider.get("completion_window") != "24h"
        or provider.get("metadata") != _expected_provider_metadata(run, stage_id)
        or value.get("receipt_mode") not in ("direct", "recovered")
        or not isinstance(value.get("receipt_created_at"), str)
    ):
        raise ValueError(f"submission receipt binding drift: {stage_id}")
    if not isinstance(value.get("input_file_id"), str) or not value["input_file_id"]:
        raise ValueError(f"submission input file id is missing: {stage_id}")
    return value


def _persist_submission_receipt(
    run: LoadedCandidateOpenAIRun,
    stage_id: str,
    intent: Mapping[str, Any],
    provider: Mapping[str, Any],
    *,
    receipt_mode: Literal["direct", "recovered"],
) -> Mapping[str, Any]:
    _validate_provider_batch_payload(provider, batch_id=None)
    if (
        not isinstance(provider.get("input_file_id"), str)
        or not provider["input_file_id"]
    ):
        raise ValueError("OpenAI submission did not return an input file id")
    if (
        provider.get("endpoint") != "/v1/responses"
        or provider.get("completion_window") != "24h"
        or provider.get("metadata") != _expected_provider_metadata(run, stage_id)
    ):
        raise ValueError("OpenAI submission metadata, endpoint, or window drift")
    requests = _stage_requests(run, stage_id)
    input_path, input_manifest_path = _batch_paths(run.root, stage_id)
    submission = _self_hashed(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "run_manifest_sha256": run.manifest["run_manifest_sha256"],
            "cost_plan_sha256": intent["cost_plan_sha256"],
            "intent_sha256": intent["intent_sha256"],
            "stage_id": stage_id,
            "provider": "openai",
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "model": requests[0].model,
            "batch_input_file_sha256": file_sha256(input_path),
            "batch_input_manifest_file_sha256": file_sha256(input_manifest_path),
            "batch_id": provider["batch_id"],
            "input_file_id": provider["input_file_id"],
            "status_at_submission": provider["status"],
            "provider_response": dict(provider),
            "receipt_mode": receipt_mode,
            "receipt_created_at": datetime.now(UTC).isoformat(),
        },
        "submission_sha256",
    )
    atomic_write_json(
        _provider_stage_root(run.root, stage_id) / "submission.json", submission
    )
    return submission


def submit_candidate_openai_batch(
    *, run_root: Path, stage_id: str, verify_sources: bool = True
) -> Mapping[str, Any]:
    """Submit one prepared stage exactly once, after persisting an intent guard."""

    run = load_candidate_openai_run(run_root, verify_sources=verify_sources)
    requests = _stage_requests(run, stage_id)
    if not requests:
        raise ValueError(f"run has no constructed requests for {stage_id}")
    cost_plan = _required_cost_plan(run)
    prepare_candidate_openai_batch(
        run_root=run.root, stage_id=stage_id, verify_sources=verify_sources
    )
    stage_root = _provider_stage_root(run.root, stage_id)
    stage_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        # mkdir is the cross-process exclusive claim. No provider call may occur
        # before this contender owns the stage directory and persists its intent.
        stage_root.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"provider stage has already been submitted or has an indeterminate "
            f"submission intent: {stage_id}"
        ) from error
    intent_path = stage_root / "submission-intent.json"
    created_at = datetime.now(UTC).isoformat()
    intent = _self_hashed(
        _submission_intent_payload(
            run, stage_id, requests, cost_plan, created_at=created_at
        ),
        "intent_sha256",
    )
    # The durable intent is deliberately written before the external call. If the
    # process dies, a retry fails closed instead of creating a duplicate batch.
    atomic_write_json(intent_path, intent)
    role = _role_for_stage(run, stage_id)
    key_env = role.api_key_env or "OPENAI_API_KEY"
    input_path, _ = _batch_paths(run.root, stage_id)
    provider = submit_openai_batch(
        input_path,
        run_id=str(run.manifest["run_manifest_sha256"]),
        stage=stage_id,
        key_env=key_env,
    )
    return _persist_submission_receipt(
        run, stage_id, intent, provider, receipt_mode="direct"
    )


def recover_candidate_openai_submission(
    *,
    run_root: Path,
    stage_id: str,
    batch_id: str,
    verify_sources: bool = True,
) -> Mapping[str, Any]:
    """Recover an intent-only stage by proving the supplied provider batch identity."""

    if not batch_id:
        raise ValueError("recovery requires a non-empty OpenAI batch id")
    run = load_candidate_openai_run(run_root, verify_sources=verify_sources)
    intent = _load_submission_intent(run, stage_id)
    if intent is None:
        raise ValueError(f"recovery requires a persisted submission intent: {stage_id}")
    if _load_submission(run, stage_id) is not None:
        raise FileExistsError(f"submission receipt already exists: {stage_id}")
    role = _role_for_stage(run, stage_id)
    provider = retrieve_batch("openai", batch_id, role)
    _validate_provider_batch_payload(provider, batch_id=batch_id)
    input_file_id = provider.get("input_file_id")
    if not isinstance(input_file_id, str) or not input_file_id:
        raise ValueError("recovery batch lacks an input file id")
    if (
        provider.get("endpoint") != "/v1/responses"
        or provider.get("completion_window") != "24h"
        or provider.get("metadata") != _expected_provider_metadata(run, stage_id)
    ):
        raise ValueError("recovery batch metadata, endpoint, or window drift")
    key_env = role.api_key_env or "OPENAI_API_KEY"
    provider_input = download_openai_file_bytes(input_file_id, key_env=key_env)
    input_path, _ = _batch_paths(run.root, stage_id)
    if provider_input != input_path.read_bytes():
        raise ValueError("recovery batch input file content does not match local input")
    return _persist_submission_receipt(
        run, stage_id, intent, provider, receipt_mode="recovered"
    )


def _load_status_receipts(
    run: LoadedCandidateOpenAIRun, stage_id: str
) -> tuple[Mapping[str, Any], ...]:
    status_root = _provider_stage_root(run.root, stage_id) / "status"
    if not status_root.exists():
        return ()
    submission = _load_submission(run, stage_id)
    if submission is None:
        raise ValueError(f"status exists without submission: {stage_id}")
    paths = sorted(status_root.glob("receipt-*.json"))
    if {path.name for path in paths} != {path.name for path in status_root.iterdir()}:
        raise ValueError(f"status receipt inventory drift: {stage_id}")
    values = []
    previous_sha256: str | None = None
    for sequence, path in enumerate(paths):
        if path.name != f"receipt-{sequence:04d}.json":
            raise ValueError(f"status receipt sequence drift: {stage_id}")
        value = load_json_object(path)
        _verify_self_hash(value, "status_sha256", f"status receipt {stage_id}")
        provider = value.get("provider_response")
        _validate_provider_batch_payload(provider, batch_id=submission["batch_id"])
        if (
            value.get("schema_version") != STATUS_SCHEMA
            or value.get("run_manifest_sha256") != run.manifest["run_manifest_sha256"]
            or value.get("stage_id") != stage_id
            or value.get("submission_sha256") != submission["submission_sha256"]
            or value.get("sequence") != sequence
            or value.get("previous_status_sha256") != previous_sha256
            or value.get("batch_id") != submission["batch_id"]
            or value.get("input_file_id") != submission["input_file_id"]
            or value.get("provider_status") != provider["status"]
            or value.get("output_file_id") != provider.get("output_file_id")
            or value.get("error_file_id") != provider.get("error_file_id")
            or provider.get("endpoint") != "/v1/responses"
            or provider.get("completion_window") != "24h"
            or provider.get("metadata") != _expected_provider_metadata(run, stage_id)
            or not isinstance(value.get("checked_at"), str)
        ):
            raise ValueError(f"status receipt binding drift: {stage_id}")
        provider_input = provider.get("input_file_id")
        if provider_input is not None and provider_input != submission["input_file_id"]:
            raise ValueError(f"status input file binding drift: {stage_id}")
        values.append(value)
        previous_sha256 = value["status_sha256"]
    return tuple(values)


def check_candidate_openai_batch(
    *, run_root: Path, stage_id: str, verify_sources: bool = True
) -> Mapping[str, Any]:
    """Retrieve and persist a hash-chained provider status receipt."""

    run = load_candidate_openai_run(run_root, verify_sources=verify_sources)
    submission = _load_submission(run, stage_id)
    if submission is None:
        raise ValueError(f"provider stage has not been submitted: {stage_id}")
    previous = _load_status_receipts(run, stage_id)
    role = _role_for_stage(run, stage_id)
    provider = retrieve_batch("openai", submission["batch_id"], role)
    _validate_provider_batch_payload(provider, batch_id=submission["batch_id"])
    if (
        provider.get("endpoint") != "/v1/responses"
        or provider.get("completion_window") != "24h"
        or provider.get("metadata") != _expected_provider_metadata(run, stage_id)
    ):
        raise ValueError("OpenAI status endpoint or completion window drift")
    provider_input = provider.get("input_file_id")
    if provider_input is not None and provider_input != submission["input_file_id"]:
        raise ValueError("OpenAI status returned a different input file id")
    sequence = len(previous)
    receipt = _self_hashed(
        {
            "schema_version": STATUS_SCHEMA,
            "run_manifest_sha256": run.manifest["run_manifest_sha256"],
            "stage_id": stage_id,
            "submission_sha256": submission["submission_sha256"],
            "sequence": sequence,
            "previous_status_sha256": (
                previous[-1]["status_sha256"] if previous else None
            ),
            "batch_id": submission["batch_id"],
            "input_file_id": submission["input_file_id"],
            "provider_status": provider["status"],
            "output_file_id": provider.get("output_file_id"),
            "error_file_id": provider.get("error_file_id"),
            "provider_response": dict(provider),
            "checked_at": datetime.now(UTC).isoformat(),
        },
        "status_sha256",
    )
    atomic_write_json(
        _provider_stage_root(run.root, stage_id)
        / "status"
        / f"receipt-{sequence:04d}.json",
        receipt,
    )
    return receipt


def _load_download_receipt(
    run: LoadedCandidateOpenAIRun, stage_id: str
) -> Mapping[str, Any] | None:
    download_root = _provider_stage_root(run.root, stage_id) / "download"
    if not download_root.exists():
        return None
    receipt_path = download_root / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"partial provider download: {stage_id}")
    submission = _load_submission(run, stage_id)
    statuses = _load_status_receipts(run, stage_id)
    if submission is None:
        raise ValueError(f"download exists without submission: {stage_id}")
    value = load_json_object(receipt_path)
    _verify_self_hash(value, "download_sha256", f"provider download {stage_id}")
    provider = value.get("provider_response")
    _validate_provider_batch_payload(provider, batch_id=submission["batch_id"])
    status_by_sha = {item["status_sha256"]: item for item in statuses}
    status = status_by_sha.get(value.get("status_sha256"))
    if status is None or status["provider_status"] != "completed":
        raise ValueError(f"download lacks a completed status receipt: {stage_id}")
    if (
        value.get("schema_version") != DOWNLOAD_SCHEMA
        or value.get("run_manifest_sha256") != run.manifest["run_manifest_sha256"]
        or value.get("stage_id") != stage_id
        or value.get("submission_sha256") != submission["submission_sha256"]
        or value.get("batch_id") != submission["batch_id"]
        or value.get("input_file_id") != submission["input_file_id"]
        or provider.get("status") != "completed"
        or provider.get("endpoint") != "/v1/responses"
        or provider.get("completion_window") != "24h"
        or provider.get("metadata") != _expected_provider_metadata(run, stage_id)
        or provider.get("output_file_id") != status["output_file_id"]
        or provider.get("error_file_id") != status["error_file_id"]
        or provider.get("input_file_id") not in (None, submission["input_file_id"])
        or not isinstance(value.get("downloaded_at"), str)
    ):
        raise ValueError(f"provider download receipt binding drift: {stage_id}")
    files = value.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"provider download file receipt is malformed: {stage_id}")
    expected_kinds = {
        kind
        for kind, file_id in (
            ("output", status["output_file_id"]),
            ("error", status["error_file_id"]),
        )
        if file_id is not None
    }
    if set(files) != expected_kinds:
        raise ValueError(f"provider download file set drift: {stage_id}")
    expected_paths = {receipt_path.resolve()}
    for kind, record in files.items():
        if not isinstance(record, Mapping):
            raise TypeError(f"provider download file receipt is malformed: {stage_id}")
        relative = f"provider/{stage_id}/download/{kind}.jsonl"
        path = run.root / relative
        if (
            record.get("file_id") != status[f"{kind}_file_id"]
            or record.get("path") != relative
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"provider {kind} download file drift: {stage_id}")
        expected_paths.add(path.resolve())
    actual_paths = {path.resolve() for path in download_root.iterdir()}
    if actual_paths != expected_paths:
        raise ValueError(f"provider download inventory drift: {stage_id}")
    return value


def collect_candidate_openai_batch(
    *, run_root: Path, stage_id: str, verify_sources: bool = True
) -> tuple[CandidateOpenAIBatchEvent, ...]:
    """Download, receipt, and scientifically collect one completed provider stage."""

    run = load_candidate_openai_run(run_root, verify_sources=verify_sources)
    submission = _load_submission(run, stage_id)
    statuses = _load_status_receipts(run, stage_id)
    if submission is None:
        raise ValueError(f"provider stage has not been submitted: {stage_id}")
    completed = [item for item in statuses if item["provider_status"] == "completed"]
    if not completed:
        raise ValueError(f"provider stage lacks a completed status receipt: {stage_id}")
    status = completed[-1]
    download = _load_download_receipt(run, stage_id)
    if download is None:
        role = _role_for_stage(run, stage_id)
        key_env = role.api_key_env or "OPENAI_API_KEY"
        provider, raw_files = download_openai_batch_files(
            submission["batch_id"], key_env=key_env
        )
        _validate_provider_batch_payload(provider, batch_id=submission["batch_id"])
        if (
            provider.get("status") != "completed"
            or provider.get("endpoint") != "/v1/responses"
            or provider.get("completion_window") != "24h"
            or provider.get("metadata") != _expected_provider_metadata(run, stage_id)
            or provider.get("input_file_id") not in (None, submission["input_file_id"])
            or provider.get("output_file_id") != status["output_file_id"]
            or provider.get("error_file_id") != status["error_file_id"]
        ):
            raise ValueError(
                "OpenAI download state does not match completed status receipt"
            )
        expected_kinds = {
            kind
            for kind, file_id in (
                ("output", status["output_file_id"]),
                ("error", status["error_file_id"]),
            )
            if file_id is not None
        }
        if set(raw_files) != expected_kinds:
            raise ValueError("OpenAI download returned a different provider file set")
        provider_stage_root = _provider_stage_root(run.root, stage_id)
        provider_stage_root.mkdir(parents=True, exist_ok=True)
        download_root = provider_stage_root / "download"
        temporary = Path(tempfile.mkdtemp(prefix=".download.", dir=provider_stage_root))
        try:
            records = {}
            for kind in ("output", "error"):
                if kind not in raw_files:
                    continue
                item = raw_files[kind]
                content = item.get("content")
                if not isinstance(content, bytes):
                    raise TypeError(f"OpenAI {kind} file content must be bytes")
                if item.get("file_id") != status[f"{kind}_file_id"]:
                    raise ValueError(f"OpenAI {kind} file id drift")
                raw_path = temporary / f"{kind}.jsonl"
                atomic_write_bytes(raw_path, content)
                records[kind] = {
                    "file_id": item["file_id"],
                    "path": f"provider/{stage_id}/download/{kind}.jsonl",
                    "sha256": file_sha256(raw_path),
                    "size_bytes": len(content),
                }
            receipt = _self_hashed(
                {
                    "schema_version": DOWNLOAD_SCHEMA,
                    "run_manifest_sha256": run.manifest["run_manifest_sha256"],
                    "stage_id": stage_id,
                    "submission_sha256": submission["submission_sha256"],
                    "status_sha256": status["status_sha256"],
                    "batch_id": submission["batch_id"],
                    "input_file_id": submission["input_file_id"],
                    "provider_response": dict(provider),
                    "files": records,
                    "downloaded_at": datetime.now(UTC).isoformat(),
                },
                "download_sha256",
            )
            atomic_write_json(temporary / "receipt.json", receipt)
            if download_root.exists():
                raise FileExistsError(
                    f"provider download appeared concurrently: {stage_id}"
                )
            temporary.replace(download_root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        download = _load_download_receipt(run, stage_id)
        if download is None:
            raise AssertionError("provider download receipt was not persisted")
    provider_files = [
        run.root / download["files"][kind]["path"]
        for kind in ("output", "error")
        if kind in download["files"]
    ]
    return _collect_candidate_openai_batch_files(
        run_root=run.root,
        stage_id=stage_id,
        provider_files=provider_files,
        download_receipt=download,
        verify_sources=verify_sources,
    )


def _read_provider_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _collect_candidate_openai_batch_files(
    *,
    run_root: Path,
    stage_id: str,
    provider_files: Sequence[Path],
    download_receipt: Mapping[str, Any],
    verify_sources: bool = True,
) -> tuple[CandidateOpenAIBatchEvent, ...]:
    """Parse already archived provider files behind a validated receipt."""

    if not provider_files:
        raise ValueError("at least one provider output/error file is required")
    run = load_candidate_openai_run(run_root, verify_sources=verify_sources)
    persisted_download = _load_download_receipt(run, stage_id)
    if persisted_download is None or dict(download_receipt) != persisted_download:
        raise ValueError(f"provider download receipt is required: {stage_id}")
    prepare_candidate_openai_batch(
        run_root=run.root, stage_id=stage_id, verify_sources=verify_sources
    )
    requests = _stage_requests(run, stage_id)
    request_by_id = {request.request_id: request for request in requests}
    rows = _read_provider_rows(provider_files)
    row_ids = [row.get("custom_id") for row in rows]
    duplicates = sorted(
        str(item) for item, count in Counter(row_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError("duplicate provider custom_id: " + ", ".join(duplicates))
    missing = sorted(set(request_by_id) - set(row_ids))
    unknown = sorted(str(item) for item in set(row_ids) - set(request_by_id))
    if missing or unknown or len(rows) != len(requests):
        raise ValueError(
            f"provider result identity mismatch; missing={missing}, unknown={unknown}"
        )
    price_path = Path(str(run.manifest["source_price_binding"]["path"]))
    if file_sha256(price_path) != run.manifest["source_price_binding"]["file_sha256"]:
        raise ValueError("bound price snapshot drift")
    snapshot = load_price_snapshot(price_path)
    events = []
    for row in rows:
        request = request_by_id[str(row["custom_id"])]
        result = parse_openai_candidate_batch_row(row, request)
        result_payload = result.model_dump(mode="json")
        cost = estimate_cost(
            snapshot,
            provider=request.provider,
            model=request.model,
            transport=request.transport,
            usage=result.usage,
        )
        if result.validation_status == "success" and not cost.complete:
            raise ValueError(
                f"successful result has incomplete cost telemetry: {request.request_id}"
            )
        payload = {
            "schema_version": EVENT_SCHEMA,
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "stage_id": request.stage_id,
            "result": result_payload,
            "result_sha256": canonical_sha256(result_payload),
            "price_snapshot_id": snapshot["snapshot_id"],
            "price_snapshot_file_sha256": file_sha256(price_path),
            "cost": cost.model_dump(mode="json"),
        }
        events.append(
            CandidateOpenAIBatchEvent.model_validate(
                _self_hashed(payload, "event_sha256")
            )
        )
    # Store events in deterministic request order, irrespective of provider row order.
    event_by_id = {event.request_id: event for event in events}
    ordered = tuple(event_by_id[request.request_id] for request in requests)
    stage_root = run.root / "collections" / stage_id
    new_collection = not stage_root.exists()
    temporary: Path | None = None
    write_root = stage_root
    if new_collection:
        stage_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{stage_id}.", dir=stage_root.parent)
        )
        write_root = temporary
    result_path = write_root / "events.jsonl"
    collection_path = write_root / "collection-manifest.json"
    raw_bindings = []
    provider_file_records = download_receipt["files"]
    provider_order = [
        item for item in ("output", "error") if item in provider_file_records
    ]
    if len(provider_files) != len(provider_order):
        raise ValueError(f"provider download file inventory drift: {stage_id}")
    for index, (provider_kind, source) in enumerate(
        zip(provider_order, provider_files, strict=True)
    ):
        source = source.resolve()
        provider_record = provider_file_records[provider_kind]
        if (
            source != (run.root / provider_record["path"]).resolve()
            or file_sha256(source) != provider_record["sha256"]
        ):
            raise ValueError(f"provider download file binding drift: {stage_id}")
        raw_path = write_root / "raw" / f"provider-{index:02d}.jsonl"
        content = source.read_bytes()
        if raw_path.exists():
            if raw_path.read_bytes() != content:
                raise ValueError(f"persisted raw provider file drift: {stage_id}")
        else:
            atomic_write_bytes(raw_path, content)
        raw_bindings.append(
            {
                "source_path": str(source),
                "provider_file_kind": provider_kind,
                "provider_file_id": provider_record["file_id"],
                "preserved_path": (
                    Path("collections")
                    / stage_id
                    / "raw"
                    / f"provider-{index:02d}.jsonl"
                ).as_posix(),
                "sha256": file_sha256(raw_path),
                "size_bytes": raw_path.stat().st_size,
            }
        )
    event_rows = [event.model_dump(mode="json") for event in ordered]
    if result_path.exists():
        if read_jsonl(result_path) != event_rows:
            raise ValueError(f"persisted collected events drift: {stage_id}")
    else:
        atomic_write_jsonl(result_path, event_rows)
    input_path, input_manifest_path = _batch_paths(run.root, stage_id)
    expected_manifest = _self_hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "run_manifest_sha256": run.manifest["run_manifest_sha256"],
            "stage_id": stage_id,
            "request_count": len(requests),
            "success_count": sum(
                event.result.validation_status == "success" for event in ordered
            ),
            "request_bindings_in_order": [
                {"request_id": item.request_id, "request_sha256": item.request_sha256}
                for item in requests
            ],
            "batch_input_file_sha256": file_sha256(input_path),
            "batch_input_manifest_sha256": file_sha256(input_manifest_path),
            "provider_submission_sha256": download_receipt["submission_sha256"],
            "provider_status_sha256": download_receipt["status_sha256"],
            "provider_download_sha256": download_receipt["download_sha256"],
            "raw_provider_files": raw_bindings,
            "events_file": (Path("collections") / stage_id / "events.jsonl").as_posix(),
            "events_file_sha256": file_sha256(result_path),
            "known_cost_usd": sum(
                float(event.cost.total_cost or 0.0) for event in ordered
            ),
            "complete_cost_count": sum(event.cost.complete for event in ordered),
            "incomplete_cost_count": sum(not event.cost.complete for event in ordered),
        },
        "collection_sha256",
    )
    try:
        if collection_path.exists():
            observed = load_json_object(collection_path)
            _verify_self_hash(observed, "collection_sha256", "collection manifest")
            if observed != expected_manifest:
                raise ValueError(f"persisted collection manifest drift: {stage_id}")
        else:
            atomic_write_json(collection_path, expected_manifest)
        if new_collection:
            if stage_root.exists():
                raise FileExistsError(
                    f"collection stage appeared concurrently: {stage_id}"
                )
            write_root.replace(stage_root)
        return ordered
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def construct_candidate_openai_rewrites(
    *, run_root: Path, verify_sources: bool = True
) -> tuple[CandidateLabelingRewriteRequest, ...]:
    """Unlock rewrites only after every selected five-sample group succeeds."""

    run = load_candidate_openai_run(run_root, verify_sources=verify_sources)
    rewrites = _derive_candidate_openai_rewrites(run)
    rows = [request.model_dump(mode="json") for request in rewrites]
    path = run.root / REWRITE_REQUESTS_FILE
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError("persisted rewrite request graph drift")
    else:
        atomic_write_jsonl(path, rows)
    return rewrites


def _derive_candidate_openai_rewrites(
    run: LoadedCandidateOpenAIRun,
) -> tuple[CandidateLabelingRewriteRequest, ...]:
    events = _load_stage_events(run.root, "semantic_generation")
    event_by_id = {event.request_id: event for event in events}
    request_by_id = {request.request_id: request for request in run.initial_requests}
    role = run.cohort.config.semantic_rewriter
    rewrites = []
    for dependency in run.dependencies:
        requests = [
            request_by_id[item] for item in dependency.required_semantic_request_ids
        ]
        dependency_events = [
            event_by_id.get(item) for item in dependency.required_semantic_request_ids
        ]
        if any(event is None for event in dependency_events):
            raise ValueError("rewrite dependency has missing semantic results")
        concrete_events = [event for event in dependency_events if event is not None]
        if any(
            event.result.validation_status != "success"
            or event.result.parsed_output is None
            for event in concrete_events
        ):
            raise ValueError(
                "rewrite dependency requires exactly five successful outputs"
            )
        rewrites.append(
            construct_candidate_labeling_rewrite_request(
                cohort_manifest_sha256=run.cohort.manifest["manifest_sha256"],
                dependency=dependency,
                semantic_requests=requests,
                semantic_outputs=[
                    event.result.parsed_output for event in concrete_events
                ],
                semantic_output_sha256_in_order=[
                    canonical_sha256(event.result.parsed_output)
                    for event in concrete_events
                ],
                max_output_tokens=role.max_output_tokens,
                temperature=role.temperature,
                reasoning=role.reasoning,
                provider_parameters=role.provider_parameters,
            )
        )
    return tuple(rewrites)


def _load_stage_events(
    root: Path, stage_id: str
) -> tuple[CandidateOpenAIBatchEvent, ...]:
    path = root / "collections" / stage_id / "events.jsonl"
    if not path.exists():
        return ()
    values = tuple(
        CandidateOpenAIBatchEvent.model_validate(row) for row in read_jsonl(path)
    )
    for event in values:
        _verify_self_hash(event.model_dump(mode="json"), "event_sha256", "OpenAI event")
        if event.stage_id != stage_id:
            raise ValueError("collected event stage drift")
    return values


def _validate_collection_stage(
    run: LoadedCandidateOpenAIRun,
    stage_id: str,
    requests: Sequence[CandidateBatchRequest],
) -> tuple[CandidateOpenAIBatchEvent, ...]:
    stage_root = run.root / "collections" / stage_id
    result_path = stage_root / "events.jsonl"
    manifest_path = stage_root / "collection-manifest.json"
    has_any = stage_root.exists() and any(stage_root.iterdir())
    if not has_any:
        return ()
    if not result_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"partial persisted collection for {stage_id}")
    if not requests:
        raise ValueError(f"collection exists without requests for {stage_id}")
    input_manifest = _validate_batch_input(run.root, stage_id, requests)
    if input_manifest is None:
        raise ValueError(f"collection exists without batch input for {stage_id}")

    events = _load_stage_events(run.root, stage_id)
    expected_ids = [request.request_id for request in requests]
    observed_ids = [event.request_id for event in events]
    if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise ValueError(f"collected event identity/order drift: {stage_id}")
    price_path = Path(str(run.manifest["source_price_binding"]["path"]))
    if file_sha256(price_path) != run.manifest["source_price_binding"]["file_sha256"]:
        raise ValueError("bound price snapshot drift")
    snapshot = load_price_snapshot(price_path)
    request_by_id = {request.request_id: request for request in requests}
    for event in events:
        request = request_by_id[event.request_id]
        result_payload = event.result.model_dump(mode="json")
        expected_cost = estimate_cost(
            snapshot,
            provider=request.provider,
            model=request.model,
            transport=request.transport,
            usage=event.result.usage,
        )
        if (
            event.request_sha256 != request.request_sha256
            or event.stage_id != request.stage_id
            or event.result.request_id != request.request_id
            or event.result.request_sha256 != request.request_sha256
            or event.result.stage_id != request.stage_id
            or event.result.logical_prompt_id != request.logical_prompt_id
            or event.result_sha256 != canonical_sha256(result_payload)
            or event.price_snapshot_id != snapshot["snapshot_id"]
            or event.price_snapshot_file_sha256 != file_sha256(price_path)
            or event.cost != expected_cost
            or (event.result.validation_status == "success" and not event.cost.complete)
        ):
            raise ValueError(
                f"collected event request/result/telemetry drift: {stage_id}"
            )

    manifest = load_json_object(manifest_path)
    _verify_self_hash(manifest, "collection_sha256", f"collection {stage_id}")
    download = _load_download_receipt(run, stage_id)
    if download is None:
        raise ValueError(f"collection lacks provider download receipt: {stage_id}")
    raw_bindings = manifest.get("raw_provider_files")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise ValueError(f"collection raw provider binding is malformed: {stage_id}")
    raw_rows: list[dict[str, Any]] = []
    expected_raw_paths: set[Path] = set()
    for index, binding in enumerate(raw_bindings):
        if not isinstance(binding, Mapping):
            raise TypeError(f"collection raw provider binding is malformed: {stage_id}")
        expected_relative = f"collections/{stage_id}/raw/provider-{index:02d}.jsonl"
        provider_order = [
            kind for kind in ("output", "error") if kind in download["files"]
        ]
        if index >= len(provider_order):
            raise ValueError(f"collection provider file inventory drift: {stage_id}")
        provider_kind = provider_order[index]
        provider_record = download["files"][provider_kind]
        if binding.get("preserved_path") != expected_relative:
            raise ValueError(f"collection raw provider path drift: {stage_id}")
        if (
            binding.get("source_path")
            != str((run.root / provider_record["path"]).resolve())
            or binding.get("provider_file_kind") != provider_kind
            or binding.get("provider_file_id") != provider_record["file_id"]
        ):
            raise ValueError(f"collection provider receipt binding drift: {stage_id}")
        raw_path = run.root / expected_relative
        if (
            not raw_path.is_file()
            or binding.get("sha256") != file_sha256(raw_path)
            or binding.get("size_bytes") != raw_path.stat().st_size
        ):
            raise ValueError(f"collection raw provider file drift: {stage_id}")
        expected_raw_paths.add(raw_path.resolve())
        raw_rows.extend(read_jsonl(raw_path))
    actual_raw_paths = {path.resolve() for path in (stage_root / "raw").glob("*.jsonl")}
    if actual_raw_paths != expected_raw_paths:
        raise ValueError(f"collection raw provider inventory drift: {stage_id}")
    row_ids = [row.get("custom_id") for row in raw_rows]
    if (
        len(row_ids) != len(set(row_ids))
        or set(row_ids) != set(expected_ids)
        or len(row_ids) != len(expected_ids)
    ):
        raise ValueError(f"collection raw result identity drift: {stage_id}")
    raw_by_id = {str(row["custom_id"]): row for row in raw_rows}
    event_by_id = {event.request_id: event for event in events}
    for request in requests:
        reparsed = parse_openai_candidate_batch_row(
            raw_by_id[request.request_id], request
        )
        if reparsed != event_by_id[request.request_id].result:
            raise ValueError(f"collection raw/result binding drift: {stage_id}")

    input_path, input_manifest_path = _batch_paths(run.root, stage_id)
    expected_manifest = _self_hashed(
        {
            "schema_version": COLLECTION_SCHEMA,
            "run_manifest_sha256": run.manifest["run_manifest_sha256"],
            "stage_id": stage_id,
            "request_count": len(requests),
            "success_count": sum(
                event.result.validation_status == "success" for event in events
            ),
            "request_bindings_in_order": [
                {"request_id": item.request_id, "request_sha256": item.request_sha256}
                for item in requests
            ],
            "batch_input_file_sha256": file_sha256(input_path),
            "batch_input_manifest_sha256": file_sha256(input_manifest_path),
            "provider_submission_sha256": download["submission_sha256"],
            "provider_status_sha256": download["status_sha256"],
            "provider_download_sha256": download["download_sha256"],
            "raw_provider_files": [dict(item) for item in raw_bindings],
            "events_file": result_path.relative_to(run.root).as_posix(),
            "events_file_sha256": file_sha256(result_path),
            "known_cost_usd": sum(
                float(event.cost.total_cost or 0.0) for event in events
            ),
            "complete_cost_count": sum(event.cost.complete for event in events),
            "incomplete_cost_count": sum(not event.cost.complete for event in events),
        },
        "collection_sha256",
    )
    if manifest != expected_manifest:
        raise ValueError(f"persisted collection manifest drift: {stage_id}")
    return events


def _validate_cost_plan(run: LoadedCandidateOpenAIRun) -> None:
    path = run.root / COST_PLAN_FILE
    if not path.exists():
        return
    value = load_json_object(path)
    _verify_self_hash(value, "cost_plan_sha256", "cost plan")
    caller_guard = value.get("caller_max_cumulative_cost_usd")
    if not isinstance(caller_guard, int | float) or isinstance(caller_guard, bool):
        raise TypeError("cost plan caller guard is malformed")
    if value != _candidate_openai_cost_plan(run, float(caller_guard)):
        raise ValueError("cost plan contract drift")


def load_candidate_openai_run(
    root: Path,
    *,
    cohort_root: Path | None = None,
    verify_sources: bool = True,
) -> LoadedCandidateOpenAIRun:
    """Deep-load the immutable selection and every currently persisted stage."""

    root = root.resolve()
    manifest = load_json_object(root / RUN_MANIFEST_FILE)
    _verify_self_hash(manifest, "run_manifest_sha256", "OpenAI run manifest")
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or manifest.get("generation_only") is not True
        or manifest.get("selection_audit_visible") is not False
        or manifest.get("provider_submission_implemented") is not True
        or manifest.get("provider_calls_supported_by_runtime") is not True
        or manifest.get("provider_output_collection_supported") is not True
        or manifest.get("provider_receipt_required_for_collection") is not True
    ):
        raise ValueError("OpenAI run contract drift")
    _validate_execution_revision(
        manifest.get("execution_revision"), verify_sources=verify_sources
    )
    source_root = (
        Path(str(manifest["source_cohort_root"]))
        if cohort_root is None
        else cohort_root
    )
    cohort = load_candidate_labeling_execution_cohort(
        source_root, verify_sources=verify_sources
    )
    if (
        cohort.manifest["manifest_sha256"] != manifest["source_cohort_manifest_sha256"]
        or file_sha256(cohort.root / "manifest.json")
        != manifest["source_cohort_manifest_file_sha256"]
        or cohort.manifest["price_binding"] != manifest["source_price_binding"]
    ):
        raise ValueError("OpenAI run source cohort drift")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise TypeError("OpenAI run selection is malformed")
    _verify_self_hash(selection, "selection_sha256", "OpenAI run selection")
    requests, dependencies = _selection(
        cohort,
        selection_kind=selection["kind"],
        anchor_index=selection["anchor_index"],
    )
    if selection["request_bindings_in_order"] != [
        {"request_id": item.request_id, "request_sha256": item.request_sha256}
        for item in requests
    ] or selection["dependency_bindings_in_order"] != [
        {
            "dependency_id": item.dependency_id,
            "dependency_sha256": item.dependency_sha256,
        }
        for item in dependencies
    ]:
        raise ValueError("OpenAI run selection binding drift")
    rewrite_path = root / REWRITE_REQUESTS_FILE
    rewrite_requests = (
        tuple(
            CandidateLabelingRewriteRequest.model_validate(row)
            for row in read_jsonl(rewrite_path)
        )
        if rewrite_path.exists()
        else ()
    )
    for request in rewrite_requests:
        _verify_self_hash(
            request.model_dump(mode="json"),
            "request_sha256",
            "candidate-labeling rewrite request",
        )
    run = LoadedCandidateOpenAIRun(
        root=root,
        manifest=manifest,
        cohort=cohort,
        initial_requests=requests,
        dependencies=dependencies,
        rewrite_requests=rewrite_requests,
        events=(),
    )
    for stage in STAGES:
        _load_submission_intent(run, stage)
        _load_submission(run, stage)
        _load_status_receipts(run, stage)
        _load_download_receipt(run, stage)
    collected: list[CandidateOpenAIBatchEvent] = []
    for stage in ("semantic_generation", "conservative_control"):
        stage_requests = _stage_requests(run, stage)
        _validate_batch_input(root, stage, stage_requests)
        collected.extend(_validate_collection_stage(run, stage, stage_requests))
    if rewrite_requests:
        if len(rewrite_requests) != len(dependencies):
            raise ValueError("partial persisted rewrite request graph")
        expected_rewrites = _derive_candidate_openai_rewrites(run)
        if rewrite_requests != expected_rewrites:
            raise ValueError("persisted rewrite request graph drift")
    rewrite_stage_requests = _stage_requests(run, "semantic_rewrite")
    _validate_batch_input(root, "semantic_rewrite", rewrite_stage_requests)
    collected.extend(
        _validate_collection_stage(run, "semantic_rewrite", rewrite_stage_requests)
    )
    completed = LoadedCandidateOpenAIRun(
        root=root,
        manifest=manifest,
        cohort=cohort,
        initial_requests=requests,
        dependencies=dependencies,
        rewrite_requests=rewrite_requests,
        events=tuple(collected),
    )
    _validate_cost_plan(completed)
    return completed
