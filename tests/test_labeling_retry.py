from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl
from circuits.labeling.runtime import retry_failed_generation
from circuits.labeling.schema import (
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    TelemetryRecord,
    Usage,
)


def _raw_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_run(
    run_root: Path,
    *,
    parse_status: str = "invalid_json",
    omit_telemetry: bool = False,
) -> GenerationRequest:
    price_path = run_root / "prices.json"
    atomic_write_json(
        price_path,
        {
            "schema_version": "adag.labeling.prices.v1",
            "snapshot_id": "retry-test-prices",
            "rates": {},
        },
    )
    recipe = {
        "schema_version": "adag.labeling.recipe.v1",
        "recipe_id": "retry-test-v1",
        "description": "retry test",
        "candidate_samples": 1,
        "candidate_generator": {
            "provider": "fake",
            "model": "candidate-model",
            "transport": "native_batch",
            "max_output_tokens": 100,
        },
        "scorer": {},
        "cluster_summarizer": {
            "provider": "fake",
            "model": "summary-model",
            "transport": "native_batch",
            "max_output_tokens": 100,
            "concurrency": 2,
        },
        "price_snapshot": "prices.json",
    }
    request = GenerationRequest(
        request_id="req-retry-test",
        run_id="run-retry-test",
        recipe_id="retry-test-v1",
        stage="cluster_summary",
        state="alternative",
        cluster_id=7,
        evidence_partition_id="generation+selection_scoring",
        provider="fake",
        model="summary-model",
        transport="native_batch",
        messages=[
            ChatMessage(role="system", content="Return JSON."),
            ChatMessage(role="user", content="Summarize this cluster."),
        ],
        max_output_tokens=100,
        temperature=0.2,
        prompt_template_version="summary-test-v1",
        prompt_sha256="a" * 64,
        evidence_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )
    request_path = run_root / "requests" / "cluster_summary.jsonl"
    atomic_write_jsonl(request_path, [request.model_dump(mode="json")])
    manifest: dict[str, Any] = {
        "schema_version": "adag.labeling.run.v1",
        "run_id": request.run_id,
        "recipe": recipe,
        "price_snapshot_path": str(price_path),
        "price_snapshot_sha256": file_sha256(price_path),
        "request_files": [],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(run_root / "manifest.json", manifest)
    stage_manifest: dict[str, Any] = {
        "schema_version": "adag.labeling.stage.v1",
        "stage": "cluster_summary",
        "source_run_manifest_sha256": manifest["manifest_sha256"],
        "request_file": {
            "stage": "cluster_summary",
            "path": "requests/cluster_summary.jsonl",
            "sha256": file_sha256(request_path),
            "request_count": 1,
            "transport": "native_batch",
        },
    }
    stage_manifest["manifest_sha256"] = canonical_sha256(stage_manifest)
    atomic_write_json(
        run_root / "stages" / "cluster_summary" / "manifest.json",
        stage_manifest,
    )

    raw = (
        '{"label":"complete","rationale":"complete","confidence":1}'
        if parse_status == "success"
        else '{"label":"truncated"'
    )
    result = GenerationResult(
        request_id=request.request_id,
        provider_request_id="batch-response-original",
        provider="fake",
        model_requested=request.model,
        model_resolved=request.model,
        raw_text=raw,
        raw_response_sha256=_raw_sha256(raw),
        parsed=(
            {"label": "complete", "rationale": "complete", "confidence": 1}
            if parse_status == "success"
            else None
        ),
        parse_status=parse_status,  # type: ignore[arg-type]
        usage=Usage(input_tokens=50, output_tokens=100),
        stop_reason="max_tokens",
    )
    result_relative = Path("results/cluster_summary") / f"{request.request_id}.json"
    telemetry_relative = (
        Path("telemetry/cluster_summary") / f"{request.request_id}.json"
    )
    telemetry = TelemetryRecord.from_request_result(
        request,
        result,
        endpoint_identity="batch://fake",
        result_artifact=result_relative.as_posix(),
        cost=None,
        slurm_job_id=None,
        slurm_array_task_id=None,
        host="test-host",
    )
    atomic_write_json(run_root / result_relative, result.model_dump(mode="json"))
    if not omit_telemetry:
        atomic_write_json(
            run_root / telemetry_relative,
            telemetry.model_dump(mode="json"),
        )
    return request


class _SuccessBackend:
    endpoint_identity = "fake://live-retry"

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        raw = '{"label":"retry","rationale":"valid now","confidence":0.9}'
        return GenerationResult(
            request_id=request.request_id,
            provider_request_id="live-response-retry",
            provider=request.provider,
            model_requested=request.model,
            model_resolved=request.model,
            raw_text=raw,
            raw_response_sha256=_raw_sha256(raw),
            parsed={
                "label": "retry",
                "rationale": "valid now",
                "confidence": 0.9,
            },
            parse_status="success",
            usage=Usage(input_tokens=60, output_tokens=40),
            stop_reason="end_turn",
        )


def test_retry_failed_archives_original_and_commits_live_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _build_run(tmp_path)
    result_path = (
        tmp_path / "results" / "cluster_summary" / f"{request.request_id}.json"
    )
    telemetry_path = (
        tmp_path / "telemetry" / "cluster_summary" / f"{request.request_id}.json"
    )
    original_result_sha256 = file_sha256(result_path)
    original_telemetry_sha256 = file_sha256(telemetry_path)
    backend = _SuccessBackend()
    monkeypatch.setattr(
        "circuits.labeling.runtime.collect_code_revision",
        lambda: {"git_commit": "test-commit", "git_dirty": False},
    )
    monkeypatch.setattr(
        "circuits.labeling.runtime.create_backend",
        lambda config: backend,
    )

    counts = asyncio.run(
        retry_failed_generation(
            run_root=tmp_path,
            stage="cluster_summary",
            request_ids={request.request_id},
            max_output_tokens=600,
        )
    )

    assert counts == {"planned": 1, "completed": 1, "failed": 0}
    assert len(backend.requests) == 1
    retry_request = backend.requests[0]
    assert retry_request.provider == request.provider
    assert retry_request.model == request.model
    assert retry_request.messages == request.messages
    assert retry_request.transport == "live"
    assert retry_request.max_output_tokens == 600

    archive = (
        tmp_path
        / "provider_batches"
        / "cluster_summary"
        / "retries"
        / request.request_id
    )
    assert file_sha256(archive / "original-result.json") == original_result_sha256
    assert file_sha256(archive / "original-telemetry.json") == original_telemetry_sha256
    retry_manifest = json.loads((archive / "manifest.json").read_text())
    expected_manifest_sha256 = retry_manifest.pop("manifest_sha256")
    assert expected_manifest_sha256 == canonical_sha256(retry_manifest)
    assert retry_manifest["status"] == "committed"
    assert retry_manifest["original"]["parse_status"] == "invalid_json"
    assert retry_manifest["original"]["transport"] == "native_batch"
    assert retry_manifest["retry"]["parse_status"] == "success"
    assert retry_manifest["retry"]["transport"] == "live"
    assert (
        retry_manifest["original"]["logical_request_sha256"]
        != retry_manifest["retry"]["logical_request_sha256"]
    )
    assert retry_manifest["override"]["max_output_tokens"] == 600
    assert retry_manifest["code_revision"]["git_commit"] == "test-commit"

    canonical_result = json.loads(result_path.read_text())
    canonical_telemetry = json.loads(telemetry_path.read_text())
    assert canonical_result["provider_request_id"] == "live-response-retry"
    assert canonical_result["parse_status"] == "success"
    assert canonical_telemetry["transport"] == "live"
    assert canonical_telemetry["generation_parameters"]["max_output_tokens"] == 600
    assert (
        canonical_telemetry["logical_request_sha256"]
        == retry_manifest["retry"]["logical_request_sha256"]
    )


def test_retry_failed_refuses_success_without_creating_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _build_run(tmp_path, parse_status="success")
    monkeypatch.setattr(
        "circuits.labeling.runtime.collect_code_revision",
        lambda: {"git_commit": "test-commit", "git_dirty": False},
    )

    def unexpected_backend(config: object) -> object:
        raise AssertionError("backend must not be created for a successful result")

    monkeypatch.setattr(
        "circuits.labeling.runtime.create_backend",
        unexpected_backend,
    )
    with pytest.raises(ValueError, match="refusing to retry successful request"):
        asyncio.run(
            retry_failed_generation(
                run_root=tmp_path,
                stage="cluster_summary",
                request_ids={request.request_id},
                max_output_tokens=600,
            )
        )
    assert not (tmp_path / "provider_batches" / "cluster_summary" / "retries").exists()


def test_retry_failed_refuses_partial_canonical_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _build_run(tmp_path, omit_telemetry=True)
    monkeypatch.setattr(
        "circuits.labeling.runtime.collect_code_revision",
        lambda: {"git_commit": "test-commit", "git_dirty": False},
    )
    with pytest.raises(ValueError, match="partial request output"):
        asyncio.run(
            retry_failed_generation(
                run_root=tmp_path,
                stage="cluster_summary",
                request_ids={request.request_id},
                max_output_tokens=600,
            )
        )
    assert not (tmp_path / "provider_batches" / "cluster_summary" / "retries").exists()
