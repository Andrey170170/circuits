"""Preparation and execution machinery for provenance-bound labeling runs."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.api import create_backend
from circuits.labeling.config import LabelingRecipe, ModelRoleConfig, load_recipe
from circuits.labeling.evidence import (
    CANDIDATE_PROMPT_VERSION,
    SUMMARY_PROMPT_VERSION,
    evidence_identity,
    candidate_messages,
    load_frozen_bundle,
    select_cluster_ids,
    summary_messages,
)
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from circuits.labeling.pricing import estimate_cost, load_price_snapshot
from circuits.labeling.profiles import (
    build_partition_profiles,
    load_cluster_members,
    render_highlighted_profile,
)
from circuits.labeling.schema import GenerationRequest, GenerationResult, TelemetryRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCHEMA = "adag.labeling.run.v1"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def collect_code_revision() -> dict[str, Any]:
    scoped = (
        "circuits/labeling",
        "scripts/bonafide/labeling_pipeline.py",
        "scripts/bonafide/labeling_score.sbatch",
        "scripts/bonafide/configs/labeling",
        "pyproject.toml",
        "uv.lock",
    )
    status = _git("status", "--porcelain=v1", "--untracked-files=all", "--", *scoped)
    paths: list[Path] = []
    for relative in scoped:
        path = REPO_ROOT / relative
        if path.is_dir():
            paths.extend(
                item
                for item in path.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and not item.name.endswith(".pyc")
            )
        elif path.is_file():
            paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(REPO_ROOT).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "source_tree_sha256": digest.hexdigest(),
    }


def collect_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("anthropic", "circuits", "openai", "pandas", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def resolve_local_snapshot(model_id: str, revision: str) -> Path:
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache:
        raise ValueError("HF_HUB_CACHE is required to resolve the frozen tokenizer")
    path = (
        Path(cache)
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not path.is_dir():
        raise ValueError(f"frozen tokenizer snapshot is not cached: {path}")
    return path


def _role(recipe: LabelingRecipe, stage: str) -> ModelRoleConfig:
    if stage == "candidate_generation":
        return recipe.candidate_generator
    if stage == "cluster_summary":
        return recipe.cluster_summarizer
    raise ValueError(f"unsupported request stage: {stage}")


def _profile_payload(profile: Any) -> dict[str, Any]:
    return {
        "trace_unit_id": profile.trace_unit_id,
        "family_partition": profile.family_partition,
        "source_manifest_sha256": profile.source_manifest_sha256,
        "matched_signed_basis_count": profile.matched_signed_basis_count,
        "record": profile.record.model_dump(mode="json"),
    }


def prepare_candidate_run(
    *,
    frozen_root: Path,
    recipe_path: Path,
    output_root: Path,
    states: Iterable[Literal["primary", "alternative"]],
    cluster_limit: int | None = None,
    explicit_clusters: dict[str, list[int]] | None = None,
    run_id: str | None = None,
    transport_override: Literal["live", "native_batch"] | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    bundle = load_frozen_bundle(frozen_root)
    recipe = load_recipe(recipe_path)
    code_revision = collect_code_revision()
    if code_revision["git_dirty"] and not allow_dirty:
        raise ValueError("labeling source tree is dirty; commit it or pass --allow-dirty")
    chosen_states = list(dict.fromkeys(states))
    if not chosen_states:
        raise ValueError("at least one state is required")
    selected: dict[str, list[int]] = {}
    for name in chosen_states:
        state = bundle.states[name]
        selected[name] = select_cluster_ids(
            state,
            explicit=(explicit_clusters or {}).get(name),
            limit=cluster_limit,
        )
    identity = {
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": file_sha256(recipe_path),
        "source_manifest_sha256": bundle.manifest["manifest_sha256"],
        "selected_clusters": selected,
        "code_revision": code_revision,
        "transport_override": transport_override,
    }
    resolved_run_id = run_id or f"labeling-{canonical_sha256(identity)[:16]}"

    snapshot_path = resolve_local_snapshot(
        recipe.scorer.source_tokenizer, recipe.scorer.source_tokenizer_revision
    )
    from transformers import AutoTokenizer

    source_tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path, local_files_only=True
    )
    role = recipe.candidate_generator
    transport = transport_override or role.transport
    staging = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    requests: list[GenerationRequest] = []
    profile_files: list[dict[str, Any]] = []
    try:
        for name in chosen_states:
            state = bundle.states[name]
            members = load_cluster_members(state.assignments_path)
            for cluster_id in selected[name]:
                row = state.evidence[cluster_id]
                partition_profiles: dict[str, list[Any]] = {}
                for partition in ("generation", "selection_scoring", "audit"):
                    partition_profiles[partition] = build_partition_profiles(
                        row,
                        partition=partition,
                        members=members,
                        source_tokenizer=source_tokenizer,
                    )
                profile_payload = {
                    "schema_version": "adag.labeling.cluster-profiles.v1",
                    "state": name,
                    "cluster_id": cluster_id,
                    "aggregate": recipe.scorer.aggregate,
                    "source_tokenizer": recipe.scorer.source_tokenizer,
                    "source_tokenizer_revision": recipe.scorer.source_tokenizer_revision,
                    "evidence_sha256": evidence_identity(row),
                    "partitions": {
                        partition: [
                            _profile_payload(profile)
                            for profile in partition_profiles[partition]
                        ]
                        for partition in partition_profiles
                    },
                }
                profile_relative = Path("profiles") / name / f"cluster-{cluster_id:04d}.json"
                atomic_write_json(staging / profile_relative, profile_payload)
                profile_files.append(
                    {
                        "path": profile_relative.as_posix(),
                        "sha256": file_sha256(staging / profile_relative),
                    }
                )
                highlighted = {
                    profile.trace_unit_id: render_highlighted_profile(profile)
                    for profile in partition_profiles["generation"]
                }
                messages, prompt_sha256 = candidate_messages(
                    row, highlighted_sequences=highlighted
                )
                for sample_index in range(recipe.candidate_samples):
                    request_identity = {
                        "run_id": resolved_run_id,
                        "recipe_id": recipe.recipe_id,
                        "stage": "candidate_generation",
                        "state": name,
                        "cluster_id": cluster_id,
                        "sample_index": sample_index,
                        "prompt_sha256": prompt_sha256,
                    }
                    request_id = f"req-{canonical_sha256(request_identity)[:24]}"
                    requests.append(
                        GenerationRequest(
                            request_id=request_id,
                            run_id=resolved_run_id,
                            recipe_id=recipe.recipe_id,
                            stage="candidate_generation",
                            state=name,  # type: ignore[arg-type]
                            cluster_id=cluster_id,
                            sample_index=sample_index,
                            evidence_partition_id="generation",
                            provider=role.provider,
                            model=role.model,
                            transport=transport,
                            messages=messages,
                            max_output_tokens=role.max_output_tokens,
                            temperature=role.temperature,
                            reasoning=role.reasoning,
                            provider_parameters=role.provider_parameters,
                            prompt_template_version=CANDIDATE_PROMPT_VERSION,
                            prompt_sha256=prompt_sha256,
                            evidence_sha256=evidence_identity(row),
                            source_manifest_sha256=bundle.manifest["manifest_sha256"],
                        )
                    )
        request_path = Path("requests") / "candidate_generation.jsonl"
        atomic_write_jsonl(
            staging / request_path,
            (request.model_dump(mode="json") for request in requests),
        )
        manifest: dict[str, Any] = {
            "schema_version": RUN_SCHEMA,
            "run_id": resolved_run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "planned",
            "recipe": recipe.model_dump(mode="json"),
            "recipe_path": str(recipe_path.resolve()),
            "recipe_sha256": file_sha256(recipe_path),
            "price_snapshot_path": str((recipe_path.parent / recipe.price_snapshot).resolve()),
            "price_snapshot_sha256": file_sha256(
                recipe_path.parent / recipe.price_snapshot
            ),
            "source_bundle_path": str(frozen_root.resolve()),
            "source_manifest_sha256": bundle.manifest["manifest_sha256"],
            "selected_clusters": selected,
            "code_revision": code_revision,
            "environment": collect_environment(),
            "request_files": [
                {
                    "stage": "candidate_generation",
                    "path": request_path.as_posix(),
                    "sha256": file_sha256(staging / request_path),
                    "request_count": len(requests),
                    "transport": transport,
                }
            ],
            "profile_files": profile_files,
            "holdout_opened": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        atomic_write_json(staging / "manifest.json", manifest)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_run_manifest(run_root: Path) -> dict[str, Any]:
    value = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    expected = value.pop("manifest_sha256", None)
    actual = canonical_sha256(value)
    value["manifest_sha256"] = expected
    if expected != actual:
        raise ValueError("labeling run manifest hash mismatch")
    if value.get("schema_version") != RUN_SCHEMA:
        raise ValueError("unsupported labeling run manifest")
    return value


def load_stage_requests(run_root: Path, stage: str) -> list[GenerationRequest]:
    manifest = load_run_manifest(run_root)
    if stage == "candidate_generation":
        request_entry = next(
            item for item in manifest["request_files"] if item["stage"] == stage
        )
    else:
        stage_manifest_path = run_root / "stages" / stage / "manifest.json"
        stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
        expected = stage_manifest.pop("manifest_sha256", None)
        if expected != canonical_sha256(stage_manifest):
            raise ValueError(f"{stage} stage manifest hash mismatch")
        if stage_manifest.get("source_run_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"{stage} stage refers to a different run manifest")
        request_entry = stage_manifest["request_file"]
    request_path = run_root / request_entry["path"]
    if file_sha256(request_path) != request_entry["sha256"]:
        raise ValueError(f"{stage} request file hash mismatch")
    return [
        GenerationRequest.model_validate(value) for value in read_jsonl(request_path)
    ]


def prepare_summary_stage(
    *,
    run_root: Path,
    transport_override: Literal["live", "native_batch"] | None = None,
) -> dict[str, Any]:
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    bundle = load_frozen_bundle(Path(manifest["source_bundle_path"]))
    role = recipe.cluster_summarizer
    requests: list[GenerationRequest] = []
    for state, cluster_ids in manifest["selected_clusters"].items():
        for raw_cluster_id in cluster_ids:
            cluster_id = int(raw_cluster_id)
            score_path = (
                run_root
                / "scores"
                / "candidate_selection"
                / state
                / f"cluster-{cluster_id:04d}.json"
            )
            if not score_path.is_file():
                raise ValueError(f"candidate score is missing: {score_path}")
            score_value = json.loads(score_path.read_text(encoding="utf-8"))
            scored_candidates = score_value["scores"]
            row = bundle.states[state].evidence[cluster_id]
            messages, prompt_sha256 = summary_messages(
                row, scored_candidates=scored_candidates
            )
            request_identity = {
                "run_id": manifest["run_id"],
                "recipe_id": recipe.recipe_id,
                "stage": "cluster_summary",
                "state": state,
                "cluster_id": cluster_id,
                "prompt_sha256": prompt_sha256,
            }
            requests.append(
                GenerationRequest(
                    request_id=f"req-{canonical_sha256(request_identity)[:24]}",
                    run_id=manifest["run_id"],
                    recipe_id=recipe.recipe_id,
                    stage="cluster_summary",
                    state=state,
                    cluster_id=cluster_id,
                    evidence_partition_id="generation+selection_scoring",
                    provider=role.provider,
                    model=role.model,
                    transport=transport_override or role.transport,
                    messages=messages,
                    max_output_tokens=role.max_output_tokens,
                    temperature=role.temperature,
                    reasoning=role.reasoning,
                    provider_parameters=role.provider_parameters,
                    prompt_template_version=SUMMARY_PROMPT_VERSION,
                    prompt_sha256=prompt_sha256,
                    evidence_sha256=evidence_identity(row),
                    source_manifest_sha256=manifest["source_manifest_sha256"],
                )
            )
    request_relative = Path("requests") / "cluster_summary.jsonl"
    request_path = run_root / request_relative
    atomic_write_jsonl(
        request_path, (request.model_dump(mode="json") for request in requests)
    )
    stage_manifest = {
        "schema_version": "adag.labeling.stage.v1",
        "stage": "cluster_summary",
        "source_run_manifest_sha256": manifest["manifest_sha256"],
        "source_score_phase": "candidate_selection",
        "request_file": {
            "stage": "cluster_summary",
            "path": request_relative.as_posix(),
            "sha256": file_sha256(request_path),
            "request_count": len(requests),
            "transport": transport_override or role.transport,
        },
    }
    stage_manifest["manifest_sha256"] = canonical_sha256(stage_manifest)
    atomic_write_json(
        run_root / "stages" / "cluster_summary" / "manifest.json",
        stage_manifest,
    )
    return stage_manifest


async def execute_live(
    *,
    run_root: Path,
    stage: str = "candidate_generation",
    request_ids: set[str] | None = None,
) -> dict[str, int]:
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    requests = load_stage_requests(run_root, stage)
    if request_ids is not None:
        requests = [request for request in requests if request.request_id in request_ids]
    if any(request.transport != "live" for request in requests):
        raise ValueError("execute-live accepts only requests prepared with live transport")
    config = _role(recipe, stage)
    backend = create_backend(config)
    price_path = Path(manifest["price_snapshot_path"])
    if file_sha256(price_path) != manifest["price_snapshot_sha256"]:
        raise ValueError("price snapshot hash mismatch")
    prices = load_price_snapshot(price_path)
    semaphore = asyncio.Semaphore(config.concurrency)
    counts = {"planned": len(requests), "completed": 0, "skipped": 0, "failed": 0}

    async def run_one(request: GenerationRequest) -> None:
        result_relative = (
            Path("results") / stage / f"{request.request_id}.json"
        )
        telemetry_relative = (
            Path("telemetry") / stage / f"{request.request_id}.json"
        )
        result_path = run_root / result_relative
        telemetry_path = run_root / telemetry_relative
        if result_path.exists() and telemetry_path.exists():
            counts["skipped"] += 1
            return
        if result_path.exists() != telemetry_path.exists():
            raise ValueError(f"partial request output exists for {request.request_id}")
        async with semaphore:
            result = await backend.generate(request)
        persist_generation_result(
            run_root=run_root,
            manifest=manifest,
            request=request,
            result=result,
            endpoint_identity=backend.endpoint_identity,
            prices=prices,
        )
        counts["completed"] += 1
        if result.parse_status != "success":
            counts["failed"] += 1

    await asyncio.gather(*(run_one(request) for request in requests))
    return counts


def persist_generation_result(
    *,
    run_root: Path,
    manifest: dict[str, Any],
    request: GenerationRequest,
    result: GenerationResult,
    endpoint_identity: str,
    prices: dict[str, Any] | None = None,
) -> None:
    result_relative = Path("results") / request.stage / f"{request.request_id}.json"
    telemetry_relative = (
        Path("telemetry") / request.stage / f"{request.request_id}.json"
    )
    result_path = run_root / result_relative
    telemetry_path = run_root / telemetry_relative
    if result_path.exists() or telemetry_path.exists():
        raise FileExistsError(f"request output already exists: {request.request_id}")
    if prices is None:
        price_path = Path(manifest["price_snapshot_path"])
        if file_sha256(price_path) != manifest["price_snapshot_sha256"]:
            raise ValueError("price snapshot hash mismatch")
        prices = load_price_snapshot(price_path)
    cost = estimate_cost(
        prices,
        provider=request.provider,
        model=request.model,
        transport=request.transport,
        usage=result.usage,
    )
    telemetry = TelemetryRecord.from_request_result(
        request,
        result,
        endpoint_identity=endpoint_identity,
        result_artifact=result_relative.as_posix(),
        cost=cost,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        host=socket.gethostname(),
    )
    atomic_write_json(result_path, result.model_dump(mode="json"))
    try:
        atomic_write_json(telemetry_path, telemetry.model_dump(mode="json"))
    except BaseException:
        # Leave a conspicuous partial result rather than hiding telemetry loss.
        raise
