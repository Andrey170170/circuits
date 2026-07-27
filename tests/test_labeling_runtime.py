from __future__ import annotations

import asyncio
import json
from pathlib import Path

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl
from circuits.labeling.runtime import (
    allocate_cluster_limit,
    execute_live,
    resolve_local_snapshot,
)
from circuits.labeling.schema import ChatMessage, GenerationRequest


def test_resolve_local_snapshot_uses_exact_revision(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = (
        tmp_path
        / "models--Qwen--Example"
        / "snapshots"
        / "012345"
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert resolve_local_snapshot("Qwen/Example", "012345") == snapshot


def test_cluster_limit_is_total_across_states() -> None:
    assert allocate_cluster_limit(["primary", "alternative"], 12) == {
        "primary": 6,
        "alternative": 6,
    }
    assert allocate_cluster_limit(["primary", "alternative"], 3) == {
        "primary": 2,
        "alternative": 1,
    }


def test_fake_live_execution_writes_atomic_result_and_telemetry(
    tmp_path: Path,
) -> None:
    price_path = tmp_path / "prices.json"
    atomic_write_json(
        price_path,
        {
            "schema_version": "adag.labeling.prices.v1",
            "snapshot_id": "test-prices",
            "rates": {},
        },
    )
    recipe = {
        "schema_version": "adag.labeling.recipe.v1",
        "recipe_id": "fake-v1",
        "description": "test",
        "candidate_samples": 1,
        "candidate_generator": {
            "provider": "fake",
            "model": "fake",
            "transport": "live",
            "max_output_tokens": 20,
        },
        "scorer": {},
        "cluster_summarizer": {
            "provider": "fake",
            "model": "fake",
            "transport": "live",
            "max_output_tokens": 20,
        },
        "price_snapshot": "prices.json",
    }
    request = GenerationRequest(
        request_id="req-test",
        run_id="run-test",
        recipe_id="fake-v1",
        stage="candidate_generation",
        state="primary",
        cluster_id=1,
        sample_index=0,
        evidence_partition_id="generation",
        provider="fake",
        model="fake",
        transport="live",
        messages=[ChatMessage(role="user", content="describe")],
        max_output_tokens=20,
        prompt_template_version="test-v1",
        prompt_sha256="a" * 64,
        evidence_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )
    request_path = tmp_path / "requests" / "candidate_generation.jsonl"
    atomic_write_jsonl(request_path, [request.model_dump(mode="json")])
    manifest = {
        "schema_version": "adag.labeling.run.v1",
        "run_id": "run-test",
        "recipe": recipe,
        "price_snapshot_path": str(price_path),
        "price_snapshot_sha256": file_sha256(price_path),
        "request_files": [
            {
                "stage": "candidate_generation",
                "path": "requests/candidate_generation.jsonl",
                "sha256": file_sha256(request_path),
            }
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(tmp_path / "manifest.json", manifest)

    first = asyncio.run(execute_live(run_root=tmp_path))
    assert first == {"planned": 1, "completed": 1, "skipped": 0, "failed": 0}
    result = json.loads(
        (tmp_path / "results" / "candidate_generation" / "req-test.json").read_text()
    )
    telemetry = json.loads(
        (tmp_path / "telemetry" / "candidate_generation" / "req-test.json").read_text()
    )
    assert result["parsed"]["description"].startswith("Deterministic")
    assert telemetry["parse_status"] == "success"
    assert telemetry["cost"]["complete"] is False

    resumed = asyncio.run(execute_live(run_root=tmp_path))
    assert resumed["skipped"] == 1
