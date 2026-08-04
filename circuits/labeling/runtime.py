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
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.api import create_backend
from circuits.labeling.config import (
    HYBRID_CANDIDATE_RECIPE_ID,
    HYBRID_CANDIDATE_RECIPE_PATH,
    LabelingRecipe,
    ModelRoleConfig,
    load_recipe,
)
from circuits.labeling.evidence import (
    candidate_messages,
    evidence_identity,
    load_frozen_bundle,
    prompt_versions,
    render_persisted_partition_witnesses,
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
from circuits.labeling.provenance import validate_local_score_artifact
from circuits.labeling.schema import (
    GenerationRequest,
    GenerationResult,
    TelemetryRecord,
)

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
        Path(cache) / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
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


def allocate_cluster_limit(
    states: Sequence[str], total_limit: int | None
) -> dict[str, int | None]:
    if total_limit is None:
        return {state: None for state in states}
    if total_limit < 1:
        raise ValueError("cluster limit must be positive")
    base, remainder = divmod(total_limit, len(states))
    return {
        state: base + (1 if index < remainder else 0)
        for index, state in enumerate(states)
    }


def validate_explicit_cluster_selection(
    states: Iterable[str],
    explicit_clusters: dict[str, list[int]] | None,
    cluster_limit: int | None,
) -> None:
    chosen = set(states)
    explicit = set(explicit_clusters or {})
    unselected = sorted(explicit - chosen)
    if unselected:
        raise ValueError(
            "explicit clusters were provided for unselected states: "
            + ", ".join(unselected)
        )
    if explicit and cluster_limit is not None:
        raise ValueError(
            "cluster limit cannot be combined with explicit cluster selections"
        )
    missing = sorted(chosen - explicit) if explicit else []
    if missing:
        raise ValueError(
            "explicit cluster selection must cover every selected state; missing: "
            + ", ".join(missing)
        )


def hybrid_cluster_limits(
    bundle: Any,
    states: Sequence[str],
    explicit_clusters: dict[str, list[int]] | None,
    cluster_limit: int | None,
) -> dict[str, int]:
    """Enforce the frozen 12-per-passing-state hybrid launch contract."""

    passing = [
        role
        for role in ("primary", "alternative")
        if bundle.states[role].manifest.get("exploratory_labeling_authorized") is True
    ]
    if not passing:
        raise ValueError("no hybrid state is authorized for exploratory labeling")
    if len(states) != len(passing) or set(states) != set(passing):
        raise ValueError(
            "hybrid requested states must be exactly all authorized roles: "
            + ", ".join(passing)
        )
    if explicit_clusters:
        raise ValueError("hybrid cluster overrides are forbidden; use frozen anchors")
    if cluster_limit != 12:
        raise ValueError("hybrid cluster limit must be exactly 12 per state")
    return {state: 12 for state in states}


def validate_hybrid_recipe_binding(
    bundle: Any, recipe: LabelingRecipe, recipe_path: Path
) -> None:
    """Require the exact frozen OpenAI recipe recorded by the bridge."""

    expected = bundle.manifest.get("labeling_recipe")
    if not isinstance(expected, dict) or expected != {
        "recipe_id": HYBRID_CANDIDATE_RECIPE_ID,
        "path": HYBRID_CANDIDATE_RECIPE_PATH,
        "sha256": expected.get("sha256") if isinstance(expected, dict) else None,
    }:
        raise ValueError("hybrid labeling recipe binding is missing or malformed")
    if recipe.recipe_id != HYBRID_CANDIDATE_RECIPE_ID:
        raise ValueError("hybrid labeling requires the frozen OpenAI recipe ID")
    if file_sha256(recipe_path) != expected["sha256"]:
        raise ValueError("hybrid labeling recipe file hash differs from frozen binding")


def deep_validate_hybrid_bundle_for_prepare(
    frozen_root: Path, recipe: LabelingRecipe
) -> None:
    """Make deep bridge recomputation a mandatory hybrid prepare step."""

    if recipe.prompt_policy != "hybrid_candidate_v1":
        return
    from circuits.analysis.bonafide.hybrid_candidate_labeling import (
        load_hybrid_labeling_bundle,
    )

    load_hybrid_labeling_bundle(frozen_root)


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
    bundle_policy = bundle.manifest.get("prompt_policy")
    if bundle_policy is not None and recipe.prompt_policy != bundle_policy:
        raise ValueError(
            f"recipe prompt policy {recipe.prompt_policy!r} differs from "
            f"frozen bundle policy {bundle_policy!r}"
        )
    if recipe.prompt_policy == "hybrid_candidate_v1":
        deep_validate_hybrid_bundle_for_prepare(frozen_root, recipe)
        validate_hybrid_recipe_binding(bundle, recipe, recipe_path)
    code_revision = collect_code_revision()
    if code_revision["git_dirty"] and not allow_dirty:
        raise ValueError(
            "labeling source tree is dirty; commit it or pass --allow-dirty"
        )
    chosen_states = list(dict.fromkeys(states))
    if not chosen_states:
        raise ValueError("at least one state is required")
    if recipe.prompt_policy == "hybrid_candidate_v1":
        limits = hybrid_cluster_limits(
            bundle, chosen_states, explicit_clusters, cluster_limit
        )
    else:
        validate_explicit_cluster_selection(
            chosen_states, explicit_clusters, cluster_limit
        )
        limits = allocate_cluster_limit(chosen_states, cluster_limit)
    selected: dict[str, list[int]] = {}
    for name in chosen_states:
        state = bundle.states[name]
        explicit = (explicit_clusters or {}).get(name)
        state_limit = limits[name]
        if recipe.prompt_policy == "hybrid_candidate_v1":
            recommendation = state.manifest.get("recommended_cluster_selection")
            if not isinstance(recommendation, dict) or recommendation.get("method") != (
                "fixed_3x4_midrank_hungarian_lexicographic_v1"
            ):
                raise ValueError(
                    f"{name} lacks the frozen hybrid 12-cluster recommendation"
                )
            recommended = list(recommendation.get("anchors_in_target_point_order", []))
            if len(recommended) != 12 or not set(recommended).issubset(
                state.ready_cluster_ids
            ):
                raise ValueError(f"{name} hybrid cluster recommendation is invalid")
            selected[name] = recommended
        elif explicit is None and state_limit == 0:
            selected[name] = []
        else:
            selected[name] = select_cluster_ids(
                state,
                explicit=explicit,
                limit=state_limit,
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
                profile_relative = (
                    Path("profiles") / name / f"cluster-{cluster_id:04d}.json"
                )
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
                    row,
                    highlighted_sequences=highlighted,
                    prompt_policy=recipe.prompt_policy,
                )
                candidate_prompt_version, _ = prompt_versions(recipe.prompt_policy)
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
                            prompt_template_version=candidate_prompt_version,
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
            "price_snapshot_path": str(
                (recipe_path.parent / recipe.price_snapshot).resolve()
            ),
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
        if recipe.prompt_policy == "width_one_v2":
            manifest["evidence_limitations"] = {
                "trace_scope": "single_target_width_one",
                "contribution_evidence": "shallow",
                "non_degenerate_contribution_comparison": False,
                "top_k_target_comparison": False,
            }
        elif recipe.prompt_policy == "hybrid_candidate_v1":
            manifest["evidence_limitations"] = {
                "trace_scope": "single_target_candidate_union",
                "source_highlights": "exact_width_one_input_attribution",
                "candidate_topology": "candidate_union_fixed_union_refinement",
                "candidate_width": "five_or_six_target_local",
                "signed_cancellation_preserved": True,
                "non_degenerate_contribution_comparison": True,
                "top_k_target_comparison": True,
                "cross_target_candidate_rank_semantics": False,
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
        if (
            stage_manifest.get("source_run_manifest_sha256")
            != manifest["manifest_sha256"]
        ):
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
    v2_inputs: list[dict[str, Any]] = []
    candidate_requests = load_stage_requests(run_root, "candidate_generation")
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
            parsed_candidates: dict[str, dict[str, Any]] = {}
            for candidate_request in candidate_requests:
                if (
                    candidate_request.state != state
                    or candidate_request.cluster_id != cluster_id
                ):
                    continue
                result_path = (
                    run_root
                    / "results"
                    / "candidate_generation"
                    / f"{candidate_request.request_id}.json"
                )
                if not result_path.is_file():
                    raise ValueError(f"candidate result is missing: {result_path}")
                candidate_result = GenerationResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
                if candidate_result.request_id != candidate_request.request_id:
                    raise ValueError(
                        f"candidate result request ID mismatch: {result_path}"
                    )
                if (
                    candidate_result.parse_status == "success"
                    and isinstance(candidate_result.parsed, dict)
                    and isinstance(candidate_result.parsed.get("description"), str)
                    and candidate_result.parsed["description"].strip()
                ):
                    parsed_candidates[candidate_request.request_id] = dict(
                        candidate_result.parsed
                    )
            skipped_candidate_ids = {
                request_id
                for request_id, parsed in parsed_candidates.items()
                if recipe.prompt_policy in {"width_one_v2", "hybrid_candidate_v1"}
                and parsed["description"].strip() == "insufficient_evidence"
            }
            scored_candidates = validate_local_score_artifact(
                score_value,
                recipe=recipe,
                run_id=manifest["run_id"],
                phase="candidate_selection",
                state=state,
                cluster_id=cluster_id,
                expected_request_ids=set(parsed_candidates) - skipped_candidate_ids,
                expected_skipped_request_ids=skipped_candidate_ids,
            )
            if recipe.prompt_policy in {"width_one_v2", "hybrid_candidate_v1"}:
                for scored in scored_candidates:
                    request_id = scored["request_id"]
                    if scored.get("candidate") != parsed_candidates[request_id]:
                        raise ValueError(
                            f"scored candidate payload mismatch: {state} cluster "
                            f"{cluster_id} request {request_id}"
                        )
                skipped_by_request = {
                    item["request_id"]: item for item in score_value.get("skipped", [])
                }
                for request_id in sorted(skipped_candidate_ids):
                    skipped = skipped_by_request[request_id]
                    if (
                        skipped.get("reason")
                        != "candidate_reported_insufficient_evidence"
                        or skipped.get("text") != "insufficient_evidence"
                        or skipped.get("candidate") != parsed_candidates[request_id]
                    ):
                        raise ValueError(
                            f"skipped candidate payload mismatch: {state} cluster "
                            f"{cluster_id} request {request_id}"
                        )
                    scored_candidates.append(
                        {
                            **skipped,
                            "correlation": None,
                            "rsquared": None,
                            "score_status": "not_scored_control_flow",
                        }
                    )
            row = bundle.states[state].evidence[cluster_id]
            highlighted_witnesses: dict[str, str] | None = None
            if recipe.prompt_policy in {"width_one_v2", "hybrid_candidate_v1"}:
                profile_relative = (
                    Path("profiles") / state / f"cluster-{cluster_id:04d}.json"
                )
                profile_path = run_root / profile_relative
                profile_entry = next(
                    (
                        item
                        for item in manifest["profile_files"]
                        if item["path"] == profile_relative.as_posix()
                    ),
                    None,
                )
                if (
                    profile_entry is None
                    or file_sha256(profile_path) != profile_entry["sha256"]
                ):
                    raise ValueError(f"profile hash mismatch: {profile_path}")
                profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
                if profile_payload.get("evidence_sha256") != evidence_identity(row):
                    raise ValueError(
                        f"profile evidence identity mismatch: {profile_path}"
                    )
                highlighted_witnesses = {
                    partition: render_persisted_partition_witnesses(
                        row, profile_payload, partition=partition
                    )
                    for partition in ("generation", "selection_scoring")
                }
                v2_inputs.append(
                    {
                        "state": state,
                        "cluster_id": cluster_id,
                        "profile_path": profile_relative.as_posix(),
                        "profile_sha256": profile_entry["sha256"],
                        "candidate_score_path": score_path.relative_to(
                            run_root
                        ).as_posix(),
                        "candidate_score_sha256": file_sha256(score_path),
                        "witness_partitions": ["generation", "selection_scoring"],
                    }
                )
            messages, prompt_sha256 = summary_messages(
                row,
                scored_candidates=scored_candidates,
                prompt_policy=recipe.prompt_policy,
                highlighted_witnesses=highlighted_witnesses,
            )
            _, summary_prompt_version = prompt_versions(recipe.prompt_policy)
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
                    prompt_template_version=summary_prompt_version,
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
    if recipe.prompt_policy in {"width_one_v2", "hybrid_candidate_v1"}:
        stage_manifest["evidence_inputs"] = v2_inputs
        stage_manifest["evidence_limitations"] = (
            {
                "trace_scope": "single_target_width_one",
                "contribution_evidence": "shallow",
                "non_degenerate_contribution_comparison": False,
                "top_k_target_comparison": False,
                "excluded_witness_partitions": ["audit"],
            }
            if recipe.prompt_policy == "width_one_v2"
            else {
                "trace_scope": "single_target_candidate_union",
                "source_highlights": "exact_width_one_input_attribution",
                "candidate_width": "five_or_six_target_local",
                "signed_cancellation_preserved": True,
                "top_k_target_comparison": True,
                "cross_target_candidate_rank_semantics": False,
                "excluded_witness_partitions": ["audit"],
            }
        )
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
    if recipe.prompt_policy == "hybrid_candidate_v1":
        raise ValueError(
            "hybrid_candidate_v1 forbids execute-live; use native batch with its cost plan"
        )
    requests = load_stage_requests(run_root, stage)
    if request_ids is not None:
        requests = [
            request for request in requests if request.request_id in request_ids
        ]
    if any(request.transport != "live" for request in requests):
        raise ValueError(
            "execute-live accepts only requests prepared with live transport"
        )
    config = _role(recipe, stage)
    backend = create_backend(config)
    price_path = Path(manifest["price_snapshot_path"])
    if file_sha256(price_path) != manifest["price_snapshot_sha256"]:
        raise ValueError("price snapshot hash mismatch")
    prices = load_price_snapshot(price_path)
    semaphore = asyncio.Semaphore(config.concurrency)
    counts = {"planned": len(requests), "completed": 0, "skipped": 0, "failed": 0}

    async def run_one(request: GenerationRequest) -> None:
        result_relative = Path("results") / stage / f"{request.request_id}.json"
        telemetry_relative = Path("telemetry") / stage / f"{request.request_id}.json"
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


def _generation_telemetry(
    *,
    request: GenerationRequest,
    result: GenerationResult,
    endpoint_identity: str,
    result_artifact: str,
    prices: dict[str, Any],
) -> TelemetryRecord:
    cost = estimate_cost(
        prices,
        provider=request.provider,
        model=request.model,
        transport=request.transport,
        usage=result.usage,
    )
    return TelemetryRecord.from_request_result(
        request,
        result,
        endpoint_identity=endpoint_identity,
        result_artifact=result_artifact,
        cost=cost,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        host=socket.gethostname(),
    )


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
    telemetry = _generation_telemetry(
        request=request,
        result=result,
        endpoint_identity=endpoint_identity,
        result_artifact=result_relative.as_posix(),
        prices=prices,
    )
    atomic_write_json(result_path, result.model_dump(mode="json"))
    try:
        atomic_write_json(telemetry_path, telemetry.model_dump(mode="json"))
    except BaseException:
        # Leave a conspicuous partial result rather than hiding telemetry loss.
        raise


def _retry_manifest_payload(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def _write_retry_manifest(
    path: Path, value: dict[str, Any], *, overwrite: bool
) -> None:
    atomic_write_json(
        path,
        _retry_manifest_payload(value),
        overwrite=overwrite,
    )


def _validate_failed_output_pair(
    *,
    run_root: Path,
    request: GenerationRequest,
) -> dict[str, Any]:
    result_relative = Path("results") / request.stage / f"{request.request_id}.json"
    telemetry_relative = (
        Path("telemetry") / request.stage / f"{request.request_id}.json"
    )
    result_path = run_root / result_relative
    telemetry_path = run_root / telemetry_relative
    if result_path.exists() != telemetry_path.exists():
        raise ValueError(f"partial request output exists for {request.request_id}")
    if not result_path.is_file():
        raise ValueError(f"request output is missing for {request.request_id}")
    try:
        result = GenerationResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        telemetry = TelemetryRecord.model_validate_json(
            telemetry_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid request output pair for {request.request_id}"
        ) from error
    expected_logical_hash = canonical_sha256(request.logical_payload())
    expected_artifact = result_relative.as_posix()
    if (
        result.request_id != request.request_id
        or telemetry.request_id != request.request_id
    ):
        raise ValueError(f"request identity mismatch for {request.request_id}")
    if telemetry.stage != request.stage:
        raise ValueError(f"request stage mismatch for {request.request_id}")
    if result.provider != request.provider or telemetry.backend != request.provider:
        raise ValueError(f"request provider mismatch for {request.request_id}")
    if (
        result.model_requested != request.model
        or telemetry.model_requested != request.model
    ):
        raise ValueError(f"request model mismatch for {request.request_id}")
    if telemetry.transport != request.transport:
        raise ValueError(f"request transport mismatch for {request.request_id}")
    if telemetry.logical_request_sha256 != expected_logical_hash:
        raise ValueError(f"logical request hash mismatch for {request.request_id}")
    if telemetry.result_artifact != expected_artifact:
        raise ValueError(f"result artifact mismatch for {request.request_id}")
    if telemetry.parse_status != result.parse_status:
        raise ValueError(f"parse status mismatch for {request.request_id}")
    if telemetry.provider_request_id != result.provider_request_id:
        raise ValueError(f"provider request identity mismatch for {request.request_id}")
    if telemetry.response_sha256 != result.raw_response_sha256:
        raise ValueError(f"response hash mismatch for {request.request_id}")
    if result.parse_status == "success":
        raise ValueError(f"refusing to retry successful request {request.request_id}")
    return {
        "result": result,
        "telemetry": telemetry,
        "result_path": result_path,
        "telemetry_path": telemetry_path,
        "result_relative": result_relative,
        "telemetry_relative": telemetry_relative,
        "result_sha256": file_sha256(result_path),
        "telemetry_sha256": file_sha256(telemetry_path),
        "logical_request_sha256": expected_logical_hash,
    }


def _archive_failed_output(
    *,
    run_root: Path,
    request: GenerationRequest,
    retry_request: GenerationRequest,
    validated: dict[str, Any],
    code_revision: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    retry_parent = run_root / "provider_batches" / request.stage / "retries"
    request_retry_root = retry_parent / request.request_id
    request_retry_root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(request_retry_root.glob("attempt-[0-9][0-9][0-9][0-9]"))
    attempt_number = len(attempts) + 1
    retry_dir = request_retry_root / f"attempt-{attempt_number:04d}"
    if retry_dir.exists():
        raise FileExistsError(f"retry attempt already exists: {retry_dir}")
    staging = (
        request_retry_root / f".attempt-{attempt_number:04d}.tmp-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    try:
        original_result_path = staging / "original-result.json"
        original_telemetry_path = staging / "original-telemetry.json"
        shutil.copy2(validated["result_path"], original_result_path)
        shutil.copy2(validated["telemetry_path"], original_telemetry_path)
        if file_sha256(original_result_path) != validated["result_sha256"]:
            raise ValueError(f"result changed while archiving {request.request_id}")
        if file_sha256(original_telemetry_path) != validated["telemetry_sha256"]:
            raise ValueError(f"telemetry changed while archiving {request.request_id}")
        atomic_write_json(
            staging / "original-request.json",
            request.model_dump(mode="json"),
        )
        atomic_write_json(
            staging / "retry-request.json",
            retry_request.model_dump(mode="json"),
        )
        retry_manifest: dict[str, Any] = {
            "schema_version": "adag.labeling.retry.v2",
            "request_id": request.request_id,
            "attempt_number": attempt_number,
            "run_id": request.run_id,
            "stage": request.stage,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "original_archived",
            "code_revision": code_revision,
            "override": {
                "max_output_tokens": retry_request.max_output_tokens,
                "transport": "live",
            },
            "original": {
                "logical_request_sha256": validated["logical_request_sha256"],
                "transport": request.transport,
                "provider_request_id": validated["result"].provider_request_id,
                "parse_status": validated["result"].parse_status,
                "result_artifact": "original-result.json",
                "result_sha256": validated["result_sha256"],
                "telemetry_artifact": "original-telemetry.json",
                "telemetry_sha256": validated["telemetry_sha256"],
                "request_artifact": "original-request.json",
                "request_sha256": file_sha256(staging / "original-request.json"),
            },
            "retry": {
                "logical_request_sha256": canonical_sha256(
                    retry_request.logical_payload()
                ),
                "transport": "live",
                "request_artifact": "retry-request.json",
                "request_sha256": file_sha256(staging / "retry-request.json"),
            },
        }
        _write_retry_manifest(
            staging / "manifest.json",
            retry_manifest,
            overwrite=False,
        )
        os.replace(staging, retry_dir)
        return retry_dir, retry_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


async def retry_failed_generation(
    *,
    run_root: Path,
    stage: str,
    request_ids: set[str],
    max_output_tokens: int,
) -> dict[str, int]:
    """Retry selected failed results live without mutating their frozen requests."""

    if max_output_tokens < 1:
        raise ValueError("max output tokens must be positive")
    if not request_ids:
        raise ValueError("at least one request ID is required")
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    if recipe.prompt_policy == "hybrid_candidate_v1":
        raise ValueError(
            "hybrid_candidate_v1 forbids live retries under the fixed $10 batch cost plan"
        )
    requests_by_id = {
        request.request_id: request for request in load_stage_requests(run_root, stage)
    }
    unknown = sorted(request_ids - requests_by_id.keys())
    if unknown:
        raise ValueError(f"unknown request IDs for {stage}: {', '.join(unknown)}")
    requests = [requests_by_id[request_id] for request_id in sorted(request_ids)]
    config = _role(recipe, stage).model_copy(
        update={"transport": "live", "max_output_tokens": max_output_tokens}
    )
    price_path = Path(manifest["price_snapshot_path"])
    if file_sha256(price_path) != manifest["price_snapshot_sha256"]:
        raise ValueError("price snapshot hash mismatch")
    prices = load_price_snapshot(price_path)
    code_revision = collect_code_revision()

    validated_by_id: dict[str, dict[str, Any]] = {}
    retry_requests: dict[str, GenerationRequest] = {}
    for request in requests:
        validated_by_id[request.request_id] = _validate_failed_output_pair(
            run_root=run_root,
            request=request,
        )
        retry_requests[request.request_id] = request.model_copy(
            update={
                "transport": "live",
                "max_output_tokens": max_output_tokens,
            }
        )

    backend = create_backend(config)
    archives: dict[str, tuple[Path, dict[str, Any]]] = {}
    for request in requests:
        archives[request.request_id] = _archive_failed_output(
            run_root=run_root,
            request=request,
            retry_request=retry_requests[request.request_id],
            validated=validated_by_id[request.request_id],
            code_revision=code_revision,
        )

    semaphore = asyncio.Semaphore(config.concurrency)
    counts = {"planned": len(requests), "completed": 0, "failed": 0}

    async def run_one(request: GenerationRequest) -> None:
        retry_request = retry_requests[request.request_id]
        validated = validated_by_id[request.request_id]
        retry_dir, retry_manifest = archives[request.request_id]
        async with semaphore:
            result = await backend.generate(retry_request)
        telemetry = _generation_telemetry(
            request=retry_request,
            result=result,
            endpoint_identity=backend.endpoint_identity,
            result_artifact=validated["result_relative"].as_posix(),
            prices=prices,
        )
        retry_result_path = retry_dir / "retry-result.json"
        retry_telemetry_path = retry_dir / "retry-telemetry.json"
        atomic_write_json(retry_result_path, result.model_dump(mode="json"))
        atomic_write_json(retry_telemetry_path, telemetry.model_dump(mode="json"))
        response_obtained = result.provider_request_id is not None
        retry_manifest["retry"].update(
            {
                "provider_request_id": result.provider_request_id,
                "parse_status": result.parse_status,
                "result_artifact": "retry-result.json",
                "result_sha256": file_sha256(retry_result_path),
                "telemetry_artifact": "retry-telemetry.json",
                "telemetry_sha256": file_sha256(retry_telemetry_path),
            }
        )
        retry_manifest["status"] = (
            "provider_response_archived"
            if response_obtained
            else "no_provider_response"
        )
        _write_retry_manifest(
            retry_dir / "manifest.json",
            retry_manifest,
            overwrite=True,
        )
        if not response_obtained or result.parse_status != "success":
            retry_manifest["status"] = (
                "response_invalid" if response_obtained else "no_provider_response"
            )
            _write_retry_manifest(
                retry_dir / "manifest.json",
                retry_manifest,
                overwrite=True,
            )
            counts["failed"] += 1
            return

        # Detect concurrent mutation before replacing either canonical artifact.
        if file_sha256(validated["result_path"]) != validated["result_sha256"]:
            raise ValueError(
                f"canonical result changed during retry {request.request_id}"
            )
        if file_sha256(validated["telemetry_path"]) != validated["telemetry_sha256"]:
            raise ValueError(
                f"canonical telemetry changed during retry {request.request_id}"
            )
        atomic_write_json(
            validated["result_path"],
            result.model_dump(mode="json"),
            overwrite=True,
        )
        atomic_write_json(
            validated["telemetry_path"],
            telemetry.model_dump(mode="json"),
            overwrite=True,
        )
        if file_sha256(validated["result_path"]) != file_sha256(retry_result_path):
            raise ValueError(
                f"canonical retry result hash mismatch: {request.request_id}"
            )
        if file_sha256(validated["telemetry_path"]) != file_sha256(
            retry_telemetry_path
        ):
            raise ValueError(
                f"canonical retry telemetry hash mismatch: {request.request_id}"
            )
        retry_manifest["status"] = "committed"
        retry_manifest["committed_at"] = datetime.now(timezone.utc).isoformat()
        _write_retry_manifest(
            retry_dir / "manifest.json",
            retry_manifest,
            overwrite=True,
        )
        counts["completed"] += 1

    await asyncio.gather(*(run_one(request) for request in requests))
    return counts
