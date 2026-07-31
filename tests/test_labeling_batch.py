from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from circuits.analysis.bonafide.canonical import file_sha256
from circuits.labeling.batch import collect_openai_batch
from circuits.labeling.batch_runtime import (
    _archive_openai_batch_files,
    _validate_or_absent_result_pair,
    submit_native_batch,
)
from circuits.labeling.io import atomic_write_json
from circuits.labeling.schema import (
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    TelemetryRecord,
)


def _request(request_id: str) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        run_id="run-1",
        recipe_id="recipe-1",
        stage="cluster_summary",
        state="primary",
        cluster_id=1,
        evidence_partition_id="selection_scoring",
        provider="openai",
        model="model-1",
        transport="native_batch",
        messages=[ChatMessage(role="user", content="label")],
        max_output_tokens=100,
        prompt_template_version="v1",
        prompt_sha256="a" * 64,
        evidence_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )


def _success_row(request_id: str) -> dict[str, object]:
    return {
        "custom_id": request_id,
        "response": {
            "status_code": 200,
            "request_id": f"provider-{request_id}",
            "body": {
                "id": f"response-{request_id}",
                "model": "model-1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    '{"label":"feature","rationale":"evidence",'
                                    '"confidence":0.5}'
                                ),
                            }
                        ],
                    }
                ],
            },
        },
        "error": None,
    }


def _error_row(request_id: str) -> dict[str, object]:
    return {
        "custom_id": request_id,
        "response": None,
        "error": {"code": "rate_limit_exceeded", "message": "try again"},
    }


def _install_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output: bytes | None,
    error: bytes | None,
) -> list[str]:
    payloads = {"file-output": output, "file-error": error}
    fetched: list[str] = []

    class FakeFiles:
        def content(self, file_id: str) -> SimpleNamespace:
            fetched.append(file_id)
            return SimpleNamespace(content=payloads[file_id])

    client = SimpleNamespace(
        batches=SimpleNamespace(
            retrieve=lambda _batch_id: SimpleNamespace(
                status="completed",
                output_file_id="file-output" if output is not None else None,
                error_file_id="file-error" if error is not None else None,
            )
        ),
        files=FakeFiles(),
    )
    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=lambda api_key: client)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return fetched


def _jsonl(*rows: dict[str, object]) -> bytes:
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()


def test_openai_collection_unions_output_and_error_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _jsonl(_success_row("request-ok"))
    error = _jsonl(_error_row("request-error"))
    fetched = _install_openai(monkeypatch, output=output, error=error)
    requests = {
        request_id: _request(request_id)
        for request_id in ("request-ok", "request-error")
    }

    results, raw_files = collect_openai_batch("batch-1", requests)

    assert fetched == ["file-output", "file-error"]
    assert results["request-ok"].parse_status == "success"
    assert results["request-error"].parse_status == "provider_error"
    assert results["request-error"].error_type == (
        "batch_request_error:rate_limit_exceeded"
    )
    assert raw_files["output"]["content"] == output
    assert raw_files["error"]["content"] == error


def test_openai_collection_preserves_incomplete_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _success_row("request-1")
    body = row["response"]["body"]  # type: ignore[index]
    body["status"] = "incomplete"  # type: ignore[index]
    body["incomplete_details"] = {"reason": "max_output_tokens"}  # type: ignore[index]
    _install_openai(monkeypatch, output=_jsonl(row), error=None)

    results, _ = collect_openai_batch(
        "batch-1", {"request-1": _request("request-1")}
    )

    assert results["request-1"].stop_reason == "max_output_tokens"


@pytest.mark.parametrize(
    ("output", "error", "message"),
    [
        (
            _jsonl(_success_row("request-1")),
            _jsonl(_error_row("request-1")),
            "repeat custom_id across their union",
        ),
        (_jsonl(_success_row("request-1")), None, "omitted request results"),
    ],
)
def test_openai_collection_requires_unique_complete_union(
    monkeypatch: pytest.MonkeyPatch,
    output: bytes,
    error: bytes | None,
    message: str,
) -> None:
    _install_openai(monkeypatch, output=output, error=error)
    requests = {
        request_id: _request(request_id)
        for request_id in ("request-1", "request-2")
    }

    with pytest.raises(ValueError, match=message):
        collect_openai_batch("batch-1", requests)


def test_openai_raw_files_are_preserved_with_hash_manifest(tmp_path: Path) -> None:
    output = b'{"custom_id":"request-1", "spacing": "preserved"}\n'
    error = b'{"custom_id":"request-2","error":{"code":"bad"}}\n'

    manifest = _archive_openai_batch_files(
        tmp_path,
        "cluster_summary",
        batch_id="batch-1",
        submission_manifest_sha256="a" * 64,
        raw_files={
            "output": {"file_id": "file-output", "content": output},
            "error": {"file_id": "file-error", "content": error},
        },
    )

    output_path = tmp_path / manifest["files"]["output"]["path"]
    error_path = tmp_path / manifest["files"]["error"]["path"]
    assert output_path.read_bytes() == output
    assert error_path.read_bytes() == error
    assert manifest["files"]["output"]["sha256"] == file_sha256(output_path)
    assert manifest["files"]["error"]["sha256"] == file_sha256(error_path)
    persisted = json.loads(
        (tmp_path / "provider_batches/cluster_summary/collection.json").read_text()
    )
    assert persisted == manifest

    repeated = _archive_openai_batch_files(
        tmp_path,
        "cluster_summary",
        batch_id="batch-1",
        submission_manifest_sha256="a" * 64,
        raw_files={
            "output": {"file_id": "file-output", "content": output},
            "error": {"file_id": "file-error", "content": error},
        },
    )
    assert repeated == manifest


def test_openai_raw_archive_refuses_changed_provider_bytes(tmp_path: Path) -> None:
    raw_files = {
        "output": {"file_id": "file-output", "content": b"original\n"},
    }
    _archive_openai_batch_files(
        tmp_path,
        "cluster_summary",
        batch_id="batch-1",
        submission_manifest_sha256="a" * 64,
        raw_files=raw_files,
    )
    raw_files["output"]["content"] = b"changed\n"

    with pytest.raises(ValueError, match="output archive mismatch"):
        _archive_openai_batch_files(
            tmp_path,
            "cluster_summary",
            batch_id="batch-1",
            submission_manifest_sha256="a" * 64,
            raw_files=raw_files,
        )


def test_submit_refuses_existing_manifest_before_provider_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    submission_path = (
        tmp_path / "provider_batches/candidate_generation/submission.json"
    )
    atomic_write_json(submission_path, {"batch_id": "already-submitted"})
    monkeypatch.setattr(
        "circuits.labeling.batch_runtime.load_run_manifest",
        lambda _path: pytest.fail("submission guard must run before provider work"),
    )

    with pytest.raises(FileExistsError, match="already been submitted"):
        submit_native_batch(tmp_path, "candidate_generation")


def test_existing_batch_result_pair_is_validated(tmp_path: Path) -> None:
    request = _request("request-1")
    result = GenerationResult(
        request_id=request.request_id,
        provider="openai",
        provider_request_id="provider-request-1",
        model_requested=request.model,
        model_resolved=request.model,
        raw_text="raw",
        raw_response_sha256="d" * 64,
        parsed={"label": "feature", "rationale": "evidence", "confidence": 0.5},
        parse_status="success",
        stop_reason="completed",
    )
    result_relative = Path("results/cluster_summary/request-1.json")
    telemetry_relative = Path("telemetry/cluster_summary/request-1.json")
    telemetry = TelemetryRecord.from_request_result(
        request,
        result,
        endpoint_identity="https://api.openai.com/v1/responses",
        result_artifact=result_relative.as_posix(),
        cost=None,
        slurm_job_id=None,
        slurm_array_task_id=None,
        host="test-host",
    )
    atomic_write_json(tmp_path / result_relative, result.model_dump(mode="json"))
    atomic_write_json(
        tmp_path / telemetry_relative, telemetry.model_dump(mode="json")
    )

    assert _validate_or_absent_result_pair(
        run_root=tmp_path,
        request=request,
        expected_result=result.model_copy(update={"created_at": "later"}),
        endpoint_identity="https://api.openai.com/v1/responses",
    )

    telemetry_value = json.loads((tmp_path / telemetry_relative).read_text())
    telemetry_value["provider_request_id"] = "different"
    atomic_write_json(
        tmp_path / telemetry_relative,
        telemetry_value,
        overwrite=True,
    )
    with pytest.raises(ValueError, match="telemetry mismatch"):
        _validate_or_absent_result_pair(
            run_root=tmp_path,
            request=request,
            expected_result=result,
            endpoint_identity="https://api.openai.com/v1/responses",
        )
