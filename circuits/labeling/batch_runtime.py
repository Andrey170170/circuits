"""Run-state integration for hosted providers' asynchronous native batches."""

from __future__ import annotations

import json
from datetime import UTC, datetime
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
from circuits.labeling.cost_guard import load_pre_submit_cost_plan
from circuits.labeling.io import atomic_write_bytes, atomic_write_json
from circuits.labeling.runtime import (
    load_run_manifest,
    load_stage_requests,
    persist_generation_result,
)
from circuits.labeling.schema import (
    GenerationRequest,
    GenerationResult,
    TelemetryRecord,
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
    submission_path = run_root / "provider_batches" / stage / "submission.json"
    if submission_path.exists():
        raise FileExistsError(
            f"provider batch has already been submitted: {submission_path}"
        )
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    cost_plan = (
        load_pre_submit_cost_plan(run_root=run_root, stage=stage)
        if recipe.prompt_policy == "hybrid_candidate_v1"
        else None
    )
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
            "submitted_at": datetime.now(UTC).isoformat(),
            "cost_plan_sha256": (
                None if cost_plan is None else cost_plan["plan_sha256"]
            ),
        }
    )
    submission["manifest_sha256"] = canonical_sha256(submission)
    atomic_write_json(submission_path, submission)
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
            "checked_at": datetime.now(UTC).isoformat(),
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
        results, raw_files = collect_openai_batch(
            submission["batch_id"], requests, key_env=key_env
        )
        _archive_openai_batch_files(
            run_root,
            stage,
            batch_id=submission["batch_id"],
            submission_manifest_sha256=expected,
            raw_files=raw_files,
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
    counts = {"collected": 0, "skipped": 0, "failed": 0}
    for request_id, result in results.items():
        request = requests[request_id]
        if _validate_or_absent_result_pair(
            run_root=run_root,
            request=request,
            expected_result=result,
            endpoint_identity=endpoint,
        ):
            counts["skipped"] += 1
            if result.parse_status != "success":
                counts["failed"] += 1
            continue
        persist_generation_result(
            run_root=run_root,
            manifest=manifest,
            request=request,
            result=result,
            endpoint_identity=endpoint,
        )
        counts["collected"] += 1
        if result.parse_status != "success":
            counts["failed"] += 1
    return counts


def _archive_openai_batch_files(
    run_root: Path,
    stage: str,
    *,
    batch_id: str,
    submission_manifest_sha256: str,
    raw_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Archive exact OpenAI result files and a hash-bound collection manifest."""

    batch_root = run_root / "provider_batches" / stage
    collection_path = batch_root / "collection.json"
    if collection_path.exists():
        return _validate_openai_batch_archive(
            run_root,
            collection_path=collection_path,
            batch_id=batch_id,
            submission_manifest_sha256=submission_manifest_sha256,
            raw_files=raw_files,
        )
    archived: dict[str, Any] = {}
    for source in ("output", "error"):
        item = raw_files.get(source)
        if item is None:
            continue
        content = item["content"]
        if not isinstance(content, bytes):
            raise TypeError(f"raw OpenAI {source} file content must be bytes")
        relative = Path("provider_batches") / stage / f"raw-{source}.jsonl"
        path = run_root / relative
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"existing raw OpenAI {source} file hash mismatch")
        else:
            atomic_write_bytes(path, content)
        archived[source] = {
            "file_id": item["file_id"],
            "path": relative.as_posix(),
            "sha256": file_sha256(path),
            "byte_count": len(content),
        }
    value: dict[str, Any] = {
        "schema_version": "adag.labeling.openai-batch-collection.v1",
        "provider": "openai",
        "batch_id": batch_id,
        "submission_manifest_sha256": submission_manifest_sha256,
        "files": archived,
        "collected_at": datetime.now(UTC).isoformat(),
    }
    value["manifest_sha256"] = canonical_sha256(value)
    atomic_write_json(collection_path, value)
    return value


def _validate_openai_batch_archive(
    run_root: Path,
    *,
    collection_path: Path,
    batch_id: str,
    submission_manifest_sha256: str,
    raw_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        value = json.loads(collection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid OpenAI batch collection manifest") from error
    expected = value.pop("manifest_sha256", None)
    if expected != canonical_sha256(value):
        raise ValueError("OpenAI batch collection manifest hash mismatch")
    value["manifest_sha256"] = expected
    if (
        value.get("batch_id") != batch_id
        or value.get("submission_manifest_sha256")
        != submission_manifest_sha256
    ):
        raise ValueError("OpenAI batch collection identity mismatch")
    archived = value.get("files")
    if not isinstance(archived, dict) or set(archived) != set(raw_files):
        raise ValueError("OpenAI batch collection file set mismatch")
    for source, item in raw_files.items():
        record = archived[source]
        content = item.get("content")
        if not isinstance(content, bytes):
            raise TypeError(f"raw OpenAI {source} file content must be bytes")
        path = run_root / record["path"]
        if (
            item.get("file_id") != record.get("file_id")
            or not path.is_file()
            or file_sha256(path) != record.get("sha256")
            or path.read_bytes() != content
            or record.get("byte_count") != len(content)
        ):
            raise ValueError(f"OpenAI batch {source} archive mismatch")
    return value


def _validate_or_absent_result_pair(
    *,
    run_root: Path,
    request: GenerationRequest,
    expected_result: GenerationResult,
    endpoint_identity: str,
) -> bool:
    """Return true for a complete matching pair, false when neither file exists."""

    result_relative = Path("results") / request.stage / f"{request.request_id}.json"
    telemetry_relative = (
        Path("telemetry") / request.stage / f"{request.request_id}.json"
    )
    result_path = run_root / result_relative
    telemetry_path = run_root / telemetry_relative
    if result_path.exists() != telemetry_path.exists():
        raise ValueError(f"partial request output exists for {request.request_id}")
    if not result_path.exists():
        return False
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
    actual_payload = result.model_dump(mode="json", exclude={"created_at"})
    expected_payload = expected_result.model_dump(mode="json", exclude={"created_at"})
    if actual_payload != expected_payload:
        raise ValueError(f"provider result mismatch for {request.request_id}")
    if (
        telemetry.request_id != request.request_id
        or telemetry.run_id != request.run_id
        or telemetry.recipe_id != request.recipe_id
        or telemetry.stage != request.stage
        or telemetry.backend != request.provider
        or telemetry.model_requested != request.model
        or telemetry.transport != request.transport
        or telemetry.prompt_sha256 != request.prompt_sha256
        or telemetry.evidence_sha256 != request.evidence_sha256
        or telemetry.source_manifest_sha256 != request.source_manifest_sha256
        or telemetry.logical_request_sha256
        != canonical_sha256(request.logical_payload())
        or telemetry.endpoint_identity != endpoint_identity
        or telemetry.result_artifact != result_relative.as_posix()
        or telemetry.provider_request_id != result.provider_request_id
        or telemetry.parse_status != result.parse_status
        or telemetry.response_sha256 != result.raw_response_sha256
        or telemetry.stop_reason != result.stop_reason
        or telemetry.error_type != result.error_type
    ):
        raise ValueError(f"telemetry mismatch for {request.request_id}")
    return True
