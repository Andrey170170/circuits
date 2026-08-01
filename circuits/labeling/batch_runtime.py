"""Run-state integration for hosted providers' asynchronous native batches."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.batch import (
    collect_anthropic_batch,
    collect_openai_batch,
    prepare_batch_input,
    retrieve_batch,
    submit_anthropic_batch,
    submit_openai_batch,
)
from circuits.labeling.config import LabelingRecipe
from circuits.labeling.io import atomic_write_json
from circuits.labeling.runtime import (
    load_run_manifest,
    load_stage_requests,
    persist_generation_result,
)


def _role(recipe: LabelingRecipe, stage: str) -> Any:
    return (
        recipe.candidate_generator
        if stage == "candidate_generation"
        else recipe.cluster_summarizer
    )


def prepare_native_batch(run_root: Path, stage: str) -> dict[str, Any]:
    manifest = load_run_manifest(run_root)
    requests = load_stage_requests(run_root, stage)
    if not requests:
        raise ValueError("cannot prepare an empty provider batch")
    provider = requests[0].provider
    if provider not in ("openai", "anthropic"):
        raise ValueError(f"native batch is unsupported for provider {provider!r}")
    if any(
        request.provider != provider or request.transport != "native_batch"
        for request in requests
    ):
        raise ValueError("native batch requests must share provider and transport")
    extension = "jsonl" if provider == "openai" else "json"
    relative = Path("provider_batches") / stage / f"input.{extension}"
    path = run_root / relative
    prepare_batch_input(requests, path, provider)
    value = {
        "schema_version": "adag.labeling.prepared-batch.v1",
        "run_id": manifest["run_id"],
        "stage": stage,
        "provider": provider,
        "model": requests[0].model,
        "request_count": len(requests),
        "input_path": relative.as_posix(),
        "input_sha256": file_sha256(path),
        "source_run_manifest_sha256": manifest["manifest_sha256"],
    }
    value["manifest_sha256"] = canonical_sha256(value)
    atomic_write_json(run_root / "provider_batches" / stage / "prepared.json", value)
    return value


def submit_native_batch(run_root: Path, stage: str) -> dict[str, Any]:
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    prepared_path = run_root / "provider_batches" / stage / "prepared.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    expected = prepared.pop("manifest_sha256", None)
    if expected != canonical_sha256(prepared):
        raise ValueError("prepared provider-batch manifest hash mismatch")
    input_path = run_root / prepared["input_path"]
    if file_sha256(input_path) != prepared["input_sha256"]:
        raise ValueError("prepared provider-batch input hash mismatch")
    role = _role(recipe, stage)
    key_env = role.api_key_env or (
        "OPENAI_API_KEY" if prepared["provider"] == "openai" else "ANTHROPIC_API_KEY"
    )
    if prepared["provider"] == "openai":
        submission = submit_openai_batch(
            input_path,
            run_id=manifest["run_id"],
            stage=stage,
            key_env=key_env,
        )
    else:
        submission = submit_anthropic_batch(input_path, key_env=key_env)
    submission.update(
        {
            "run_id": manifest["run_id"],
            "stage": stage,
            "prepared_manifest_sha256": expected,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    submission["manifest_sha256"] = canonical_sha256(submission)
    atomic_write_json(
        run_root / "provider_batches" / stage / "submission.json", submission
    )
    return submission


def native_batch_status(run_root: Path, stage: str) -> dict[str, Any]:
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    submission = json.loads(
        (run_root / "provider_batches" / stage / "submission.json").read_text(
            encoding="utf-8"
        )
    )
    expected = submission.pop("manifest_sha256", None)
    if expected != canonical_sha256(submission):
        raise ValueError("provider-batch submission manifest hash mismatch")
    role = _role(recipe, stage)
    status = retrieve_batch(submission["provider"], submission["batch_id"], role)
    status.update(
        {
            "run_id": manifest["run_id"],
            "stage": stage,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "submission_manifest_sha256": expected,
        }
    )
    return status


def collect_native_batch(run_root: Path, stage: str) -> dict[str, int]:
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    requests = {
        request.request_id: request for request in load_stage_requests(run_root, stage)
    }
    submission = json.loads(
        (run_root / "provider_batches" / stage / "submission.json").read_text(
            encoding="utf-8"
        )
    )
    expected = submission.pop("manifest_sha256", None)
    if expected != canonical_sha256(submission):
        raise ValueError("provider-batch submission manifest hash mismatch")
    role = _role(recipe, stage)
    key_env = role.api_key_env or (
        "OPENAI_API_KEY"
        if submission["provider"] == "openai"
        else "ANTHROPIC_API_KEY"
    )
    if submission["provider"] == "openai":
        results, raw = collect_openai_batch(
            submission["batch_id"], requests, key_env=key_env
        )
        raw_path = run_root / "provider_batches" / stage / "raw-output.jsonl"
        # Preserve the provider output exactly while still using an atomic JSON
        # result per logical request downstream.
        from circuits.labeling.io import atomic_write_jsonl

        atomic_write_jsonl(
            raw_path,
            (json.loads(line) for line in raw.splitlines() if line.strip()),
        )
        endpoint = "https://api.openai.com/v1/responses"
    else:
        results = collect_anthropic_batch(
            submission["batch_id"], requests, key_env=key_env
        )
        endpoint = "https://api.anthropic.com/v1/messages"
    missing = sorted(set(requests) - set(results))
    for request_id in missing:
        raise ValueError(f"provider batch omitted request result: {request_id}")
    counts = {"collected": 0, "failed": 0}
    for request_id, result in results.items():
        persist_generation_result(
            run_root=run_root,
            manifest=manifest,
            request=requests[request_id],
            result=result,
            endpoint_identity=endpoint,
        )
        counts["collected"] += 1
        if result.parse_status != "success":
            counts["failed"] += 1
    return counts
