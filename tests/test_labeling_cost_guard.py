from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.batch_runtime import submit_native_batch
from circuits.labeling.config import load_recipe
from circuits.labeling.cost_guard import (
    build_pre_submit_cost_plan,
    load_pre_submit_cost_plan,
)
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl
from circuits.labeling.runtime import execute_live, retry_failed_generation
from circuits.labeling.schema import ChatMessage, GenerationRequest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = (
    REPO_ROOT
    / "scripts"
    / "bonafide"
    / "configs"
    / "labeling"
    / "openai-hybrid-candidate-v1.json"
)


def _write_run(root: Path) -> None:
    recipe = load_recipe(RECIPE_PATH)
    request = GenerationRequest(
        request_id="request-1",
        run_id="run-1",
        recipe_id=recipe.recipe_id,
        stage="candidate_generation",
        state="primary",
        cluster_id=1,
        sample_index=0,
        evidence_partition_id="generation",
        provider="openai",
        model=recipe.candidate_generator.model,
        transport="native_batch",
        messages=[ChatMessage(role="user", content="frozen hybrid evidence")],
        max_output_tokens=recipe.candidate_generator.max_output_tokens,
        prompt_template_version="hybrid-v1",
        prompt_sha256="a" * 64,
        evidence_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )
    request_relative = Path("requests/candidate_generation.jsonl")
    atomic_write_jsonl(root / request_relative, [request.model_dump(mode="json")])
    price_path = RECIPE_PATH.parent / recipe.price_snapshot
    manifest = {
        "schema_version": "adag.labeling.run.v1",
        "run_id": "run-1",
        "recipe": recipe.model_dump(mode="json"),
        "price_snapshot_path": str(price_path),
        "price_snapshot_sha256": file_sha256(price_path),
        "request_files": [
            {
                "stage": "candidate_generation",
                "path": request_relative.as_posix(),
                "sha256": file_sha256(root / request_relative),
                "request_count": 1,
                "transport": "native_batch",
            }
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(root / "manifest.json", manifest)


def test_cost_plan_loader_rejects_missing_and_tampered_plan(tmp_path: Path) -> None:
    _write_run(tmp_path)
    with pytest.raises(ValueError, match="requires a persisted cost plan"):
        load_pre_submit_cost_plan(run_root=tmp_path, stage="candidate_generation")

    build_pre_submit_cost_plan(run_root=tmp_path, max_cumulative_cost_usd=10.0)
    plan_path = tmp_path / "pre-submit-cost-plan.json"
    tampered = json.loads(plan_path.read_text())
    tampered["projected_upper_bound_usd"] += 1.0
    plan_path.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_pre_submit_cost_plan(run_root=tmp_path, stage="candidate_generation")


def test_cost_plan_loader_rejects_persisted_over_cap_plan(tmp_path: Path) -> None:
    _write_run(tmp_path)
    with pytest.raises(ValueError, match="exceeds"):
        build_pre_submit_cost_plan(run_root=tmp_path, max_cumulative_cost_usd=1e-12)
    with pytest.raises(ValueError, match="does not authorize spend"):
        load_pre_submit_cost_plan(run_root=tmp_path, stage="candidate_generation")


def test_hybrid_submit_refuses_missing_plan_before_provider_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_run(tmp_path)
    monkeypatch.setattr(
        "circuits.labeling.batch_runtime.submit_openai_batch",
        lambda *args, **kwargs: pytest.fail("provider must not be called"),
    )
    with pytest.raises(ValueError, match="requires a persisted cost plan"):
        submit_native_batch(tmp_path, "candidate_generation")


def test_hybrid_run_rejects_live_execution_and_retries(tmp_path: Path) -> None:
    _write_run(tmp_path)
    with pytest.raises(ValueError, match="forbids execute-live"):
        asyncio.run(execute_live(run_root=tmp_path))
    with pytest.raises(ValueError, match="forbids live retries"):
        asyncio.run(
            retry_failed_generation(
                run_root=tmp_path,
                stage="candidate_generation",
                request_ids={"request-1"},
                max_output_tokens=100,
            )
        )
