from __future__ import annotations

from pathlib import Path

import pytest

from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.labeling.io import atomic_write_json
from circuits.labeling.schema import (
    ChatMessage,
    CostEstimate,
    GenerationRequest,
    GenerationResult,
    TelemetryRecord,
    Usage,
)
from circuits.labeling.telemetry import summarize_telemetry


def _run(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "adag.labeling.run.v1",
        "run_id": "run-telemetry",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(tmp_path / "manifest.json", manifest)


def _telemetry(
    *,
    request_id: str,
    stage: str,
    provider_request_id: str,
    cost: float,
) -> TelemetryRecord:
    request = GenerationRequest(
        request_id=request_id,
        run_id="run-telemetry",
        recipe_id="recipe-v2",
        stage=stage,  # type: ignore[arg-type]
        state="primary",
        cluster_id=1,
        sample_index=0 if stage == "candidate_generation" else None,
        evidence_partition_id="generation",
        provider="openai",
        model="model",
        transport="native_batch",
        messages=[ChatMessage(role="user", content="prompt")],
        max_output_tokens=100,
        prompt_template_version="v2",
        prompt_sha256="a" * 64,
        evidence_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )
    result = GenerationResult(
        request_id=request_id,
        provider_request_id=provider_request_id,
        provider="openai",
        model_requested="model",
        model_resolved="model",
        raw_text="{}",
        raw_response_sha256=provider_request_id.ljust(64, "0")[:64],
        parsed={"description": "x"},
        parse_status="success",
        usage=Usage(
            input_tokens=10,
            uncached_input_tokens=8,
            cache_read_tokens=2,
            cache_write_tokens=0,
            output_tokens=3,
            reasoning_tokens=1,
        ),
        stop_reason="completed",
    )
    return TelemetryRecord.from_request_result(
        request,
        result,
        endpoint_identity="https://api.openai.com/v1/responses",
        result_artifact=f"results/{stage}/{request_id}.json",
        cost=CostEstimate(
            price_snapshot_id="prices",
            input_cost=cost / 2,
            output_cost=cost / 2,
            total_cost=cost,
            complete=True,
        ),
        slurm_job_id=None,
        slurm_array_task_id=None,
        host="test",
    )


def _write(path: Path, telemetry: TelemetryRecord) -> None:
    atomic_write_json(path, telemetry.model_dump(mode="json"))


def test_telemetry_summary_without_retries(tmp_path: Path) -> None:
    _run(tmp_path)
    value = _telemetry(
        request_id="candidate-1",
        stage="candidate_generation",
        provider_request_id="provider-original",
        cost=0.1,
    )
    _write(tmp_path / "telemetry/candidate_generation/candidate-1.json", value)

    summary = summarize_telemetry(run_root=tmp_path)["provider_api"]

    assert summary["event_count"] == 1
    assert summary["known_cost_usd"] == pytest.approx(0.1)
    assert summary["usage"]["input_tokens"] == 10
    assert summary["duplicate_file_count"] == 0


def test_successful_retry_counts_original_and_replacement_once(tmp_path: Path) -> None:
    _run(tmp_path)
    original = _telemetry(
        request_id="summary-1",
        stage="cluster_summary",
        provider_request_id="provider-original",
        cost=0.2,
    )
    replacement = _telemetry(
        request_id="summary-1",
        stage="cluster_summary",
        provider_request_id="provider-retry",
        cost=0.3,
    )
    _write(tmp_path / "telemetry/cluster_summary/summary-1.json", replacement)
    attempt = (
        tmp_path / "provider_batches/cluster_summary/retries/summary-1/attempt-0001"
    )
    _write(attempt / "original-telemetry.json", original)
    _write(attempt / "retry-telemetry.json", replacement)

    summary = summarize_telemetry(run_root=tmp_path)["provider_api"]

    assert summary["event_count"] == 2
    assert summary["known_cost_usd"] == pytest.approx(0.5)
    assert summary["duplicate_file_count"] == 1
    assert summary["included_source_counts"] == {
        "archived_original": 1,
        "canonical": 1,
    }


def test_failed_retry_attempt_is_counted_without_recounting_original(
    tmp_path: Path,
) -> None:
    _run(tmp_path)
    original = _telemetry(
        request_id="candidate-1",
        stage="candidate_generation",
        provider_request_id="provider-original",
        cost=0.1,
    )
    failed_attempt = _telemetry(
        request_id="candidate-1",
        stage="candidate_generation",
        provider_request_id="provider-failed-retry",
        cost=0.15,
    )
    _write(tmp_path / "telemetry/candidate_generation/candidate-1.json", original)
    retry_root = tmp_path / "provider_batches/candidate_generation/retries/candidate-1"
    _write(retry_root / "original-telemetry.json", original)
    _write(retry_root / "attempt-1-telemetry.json", failed_attempt)

    summary = summarize_telemetry(run_root=tmp_path)["provider_api"]

    assert summary["event_count"] == 2
    assert summary["known_cost_usd"] == pytest.approx(0.25)
    assert summary["included_source_counts"] == {
        "canonical": 1,
        "retry_attempt": 1,
    }


def test_telemetry_summary_aggregates_multiple_stages(tmp_path: Path) -> None:
    _run(tmp_path)
    candidate = _telemetry(
        request_id="candidate-1",
        stage="candidate_generation",
        provider_request_id="provider-candidate",
        cost=0.1,
    )
    summary_value = _telemetry(
        request_id="summary-1",
        stage="cluster_summary",
        provider_request_id="provider-summary",
        cost=0.4,
    )
    _write(tmp_path / "telemetry/candidate_generation/candidate-1.json", candidate)
    _write(tmp_path / "telemetry/cluster_summary/summary-1.json", summary_value)

    summary = summarize_telemetry(run_root=tmp_path)["provider_api"]

    assert summary["known_cost_usd"] == pytest.approx(0.5)
    assert summary["by_stage"]["candidate_generation"]["event_count"] == 1
    assert summary["by_stage"]["cluster_summary"]["event_count"] == 1


def test_local_scoring_gpu_telemetry_is_aggregated(tmp_path: Path) -> None:
    _run(tmp_path)
    for phase, gpu_hours, completed in (
        ("candidate_selection", 0.25, 4),
        ("summary_audit", 0.5, 2),
    ):
        atomic_write_json(
            tmp_path / f"telemetry/local_scoring/{phase}-job-0.json",
            {
                "schema_version": "adag.labeling.local-scoring-telemetry.v1",
                "run_id": "run-telemetry",
                "phase": phase,
                "elapsed_seconds": gpu_hours * 3600,
                "gpu_hours": gpu_hours,
                "peak_hbm_bytes": 100 if phase == "candidate_selection" else 200,
                "peak_host_rss_kib": 50,
                "counts": {"planned": completed, "completed": completed, "skipped": 0},
            },
        )

    summary = summarize_telemetry(run_root=tmp_path)["local_scoring"]

    assert summary["record_count"] == 2
    assert summary["gpu_hours"] == pytest.approx(0.75)
    assert summary["completed_cluster_count"] == 6
    assert summary["peak_hbm_bytes_max"] == 200
