from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import circuits.analysis.bonafide.candidate_labeling_openai_run as run_module
from circuits.analysis.bonafide.candidate_labeling_openai_run import (
    build_candidate_openai_cost_plan,
    check_candidate_openai_batch,
    collect_candidate_openai_batch,
    construct_candidate_openai_rewrites,
    initialize_candidate_openai_run,
    load_candidate_openai_run,
    prepare_candidate_openai_batch,
    recover_candidate_openai_submission,
    submit_candidate_openai_batch,
)
from circuits.analysis.bonafide.candidate_labeling_renderer import (
    STATUS_ENUM,
    TYPED_OUTPUT_FIELDS,
    LoadedCandidateLabelingRenderer,
)
from circuits.analysis.bonafide.candidate_labeling_runtime import (
    LoadedCandidateLabelingExecutionCohort,
    PreparedCandidateLabelingRequest,
    RewriteDependency,
    _price_binding,
    build_candidate_labeling_execution_cohort,
    load_candidate_labeling_runtime_config,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256

CONFIG_ROOT = Path("scripts/bonafide/configs/labeling")


def _schema(include_candidate: bool) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(TYPED_OUTPUT_FIELDS),
        "properties": {
            "input_localization_hypothesis": {"type": "string", "minLength": 1},
            "exploratory_candidate_description": (
                {"type": "string", "minLength": 1}
                if include_candidate
                else {"const": "not_available"}
            ),
            "background_or_confound": {"type": "string", "minLength": 1},
            "limitations": {"type": "string", "minLength": 1},
            "status": {"type": "string", "enum": list(STATUS_ENUM)},
        },
    }


def _cohort(tmp_path: Path) -> LoadedCandidateLabelingExecutionCohort:
    renderer_root = tmp_path / "renderer"
    renderer_root.mkdir()
    prompts = []
    for arm_id, include_candidate in (
        ("arm_1_width_only", False),
        ("arm_2_width_plus_candidate", True),
    ):
        for index in range(12):
            payload = {
                "messages": [
                    {"role": "system", "content": "bounded generation evidence"},
                    {"role": "user", "content": f"arm {arm_id} cluster {index}"},
                ],
                "expected_output_json_schema": _schema(include_candidate),
            }
            prompt = {
                "logical_prompt_id": f"{arm_id}:w64:{index:02d}",
                "arm_id": arm_id,
                "anchor_index": index,
                "cluster_id": index,
                "candidate_evidence_included": include_candidate,
                "message_payload": payload,
                "message_payload_sha256": canonical_sha256(payload),
                "width_evidence_sha256": f"{index:064x}",
                "rendered_candidate_witness_sha256_in_order": (
                    [f"{index + offset + 1:064x}" for offset in range(8)]
                    if include_candidate
                    else None
                ),
            }
            prompt["prompt_sha256"] = canonical_sha256(prompt)
            prompts.append(prompt)
    renderer = LoadedCandidateLabelingRenderer(
        root=renderer_root,
        manifest={"manifest_sha256": "a" * 64},
        witness_selection={},
        generation_prompts=tuple(prompts),
        stage_plan={},
    )
    config = load_candidate_labeling_runtime_config(
        CONFIG_ROOT / "openai-c2-evidence-v1.json"
    )
    rows, dependencies, _ = build_candidate_labeling_execution_cohort(renderer, config)
    price_path = tmp_path / "prices.json"
    shutil.copyfile(CONFIG_ROOT / "prices-2026-07-30.json", price_path)
    root = tmp_path / "cohort"
    root.mkdir()
    manifest = {
        "manifest_sha256": "b" * 64,
        "price_binding": _price_binding(config, price_path),
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return LoadedCandidateLabelingExecutionCohort(
        root=root,
        manifest=manifest,
        config=config,
        initial_requests=tuple(
            PreparedCandidateLabelingRequest.model_validate(row) for row in rows
        ),
        rewrite_dependencies=tuple(
            RewriteDependency.model_validate(row) for row in dependencies
        ),
    )


def _patch_cohort(
    monkeypatch: pytest.MonkeyPatch, cohort: LoadedCandidateLabelingExecutionCohort
) -> None:
    monkeypatch.setattr(
        run_module,
        "load_candidate_labeling_execution_cohort",
        lambda *args, **kwargs: cohort,
    )
    monkeypatch.setattr(
        run_module,
        "collect_candidate_openai_execution_revision",
        lambda: {
            "repo_root": str(cohort.root.parent),
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "tracked_worktree_clean": True,
            "tracked_status_sha256": "3" * 64,
            "files": [
                {
                    "role": role,
                    "path": path,
                    "git_blob": "4" * 40,
                    "sha256": "5" * 64,
                }
                for role, path in run_module._SOURCE_BINDINGS.items()
            ],
        },
    )


def _valid_output(request: object) -> dict[str, str]:
    schema = request.expected_output_json_schema  # type: ignore[attr-defined]
    candidate_rule = schema["properties"]["exploratory_candidate_description"]
    return {
        "input_localization_hypothesis": "Repeated local punctuation evidence.",
        "exploratory_candidate_description": candidate_rule.get(
            "const", "Exploratory candidate direction."
        ),
        "background_or_confound": "The prompt template may confound this pattern.",
        "limitations": "Only local single-target evidence is displayed.",
        "status": "provisional_description",
    }


def _provider_row(request: object, *, text: str | None = None) -> dict:
    if text is None:
        text = json.dumps(_valid_output(request))
    return {
        "custom_id": request.request_id,  # type: ignore[attr-defined]
        "response": {
            "status_code": 200,
            "request_id": f"provider-{request.request_id}",  # type: ignore[attr-defined]
            "body": {
                "id": f"response-{request.request_id}",  # type: ignore[attr-defined]
                "model": request.model,  # type: ignore[attr-defined]
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
                "usage": {
                    "input_tokens": 101,
                    "input_tokens_details": {"cached_tokens": 11},
                    "output_tokens": 29,
                    "output_tokens_details": {"reasoning_tokens": 7},
                },
            },
        },
        "error": None,
    }


def _provider_error_row(request: object) -> dict:
    return {
        "custom_id": request.request_id,  # type: ignore[attr-defined]
        "response": None,
        "error": {"code": "server_error", "message": "retryable fixture"},
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _initialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[LoadedCandidateLabelingExecutionCohort, Path]:
    cohort = _cohort(tmp_path)
    _patch_cohort(monkeypatch, cohort)
    root = tmp_path / "run"
    initialize_candidate_openai_run(
        cohort_root=cohort.root,
        output_root=root,
        selection_kind="paired_anchor_smoke",
        anchor_index=0,
    )
    return cohort, root


def _provider_payload(
    root: Path, stage: str, *, status: str, output_file_id: str | None
) -> dict[str, object]:
    run_id = json.loads((root / "run-manifest.json").read_text())["run_manifest_sha256"]
    return {
        "schema_version": "adag.labeling.provider-batch.v1",
        "provider": "openai",
        "batch_id": f"batch-{stage}",
        "input_file_id": f"input-{stage}",
        "endpoint": "/v1/responses",
        "completion_window": "24h",
        "metadata": {"run_id": run_id, "stage": stage},
        "status": status,
        "output_file_id": output_file_id,
        "error_file_id": None,
        "request_counts": None,
    }


def _provider_ready(
    root: Path,
    stage: str,
    provider_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (root / "cost-plan.json").exists():
        build_candidate_openai_cost_plan(run_root=root, max_cumulative_cost_usd=10.0)
    prepare_candidate_openai_batch(run_root=root, stage_id=stage)
    monkeypatch.setattr(
        run_module,
        "submit_openai_batch",
        lambda *args, **kwargs: _provider_payload(
            root, stage, status="validating", output_file_id=None
        ),
    )
    submit_candidate_openai_batch(run_root=root, stage_id=stage)
    monkeypatch.setattr(
        run_module,
        "retrieve_batch",
        lambda *args, **kwargs: _provider_payload(
            root, stage, status="completed", output_file_id=f"output-{stage}"
        ),
    )
    check_candidate_openai_batch(run_root=root, stage_id=stage)
    monkeypatch.setattr(
        run_module,
        "download_openai_batch_files",
        lambda *args, **kwargs: (
            _provider_payload(
                root, stage, status="completed", output_file_id=f"output-{stage}"
            ),
            {
                "output": {
                    "file_id": f"output-{stage}",
                    "content": provider_path.read_bytes(),
                }
            },
        ),
    )


def _collect_stage(
    root: Path,
    stage: str,
    provider_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = load_candidate_openai_run(root)
    requests = (
        run.rewrite_requests
        if stage == "semantic_rewrite"
        else tuple(item for item in run.initial_requests if item.stage_id == stage)
    )
    _write_rows(provider_path, [_provider_row(item) for item in reversed(requests)])
    _provider_ready(root, stage, provider_path, monkeypatch)
    collect_candidate_openai_batch(run_root=root, stage_id=stage)


def test_paired_smoke_full_synthetic_lifecycle_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    run = load_candidate_openai_run(root)
    assert len(run.initial_requests) == 12
    assert {item.stage_id for item in run.initial_requests} == {
        "semantic_generation",
        "conservative_control",
    }
    with pytest.raises(ValueError, match="cost guard"):
        build_candidate_openai_cost_plan(run_root=root, max_cumulative_cost_usd=0.0)
    plan = build_candidate_openai_cost_plan(run_root=root, max_cumulative_cost_usd=1.0)
    assert plan["request_count"] == 14

    _collect_stage(
        root, "semantic_generation", tmp_path / "semantic.jsonl", monkeypatch
    )
    _collect_stage(
        root, "conservative_control", tmp_path / "control.jsonl", monkeypatch
    )
    rewrites = construct_candidate_openai_rewrites(run_root=root)
    assert len(rewrites) == 2
    assert all(len(item.required_semantic_request_ids) == 5 for item in rewrites)
    _collect_stage(root, "semantic_rewrite", tmp_path / "rewrite.jsonl", monkeypatch)

    completed = load_candidate_openai_run(root)
    resumed = initialize_candidate_openai_run(
        cohort_root=completed.cohort.root,
        output_root=root,
        selection_kind="paired_anchor_smoke",
        anchor_index=0,
    )
    assert resumed == completed
    assert len(completed.events) == 14
    assert all(event.cost.complete for event in completed.events)
    assert sum(float(event.cost.total_cost or 0) for event in completed.events) > 0
    for stage in (
        "semantic_generation",
        "conservative_control",
        "semantic_rewrite",
    ):
        assert prepare_candidate_openai_batch(run_root=root, stage_id=stage)[
            "request_count"
        ] in {2, 10}


def test_submit_requires_cost_plan_and_refuses_duplicate_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    called = 0

    def submit(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called += 1
        return _provider_payload(
            root,
            "conservative_control",
            status="validating",
            output_file_id=None,
        )

    monkeypatch.setattr(run_module, "submit_openai_batch", submit)
    with pytest.raises(ValueError, match="persisted cost plan"):
        submit_candidate_openai_batch(run_root=root, stage_id="conservative_control")
    assert called == 0

    build_candidate_openai_cost_plan(run_root=root, max_cumulative_cost_usd=10.0)
    submit_candidate_openai_batch(run_root=root, stage_id="conservative_control")
    assert called == 1
    with pytest.raises(FileExistsError, match="already been submitted"):
        submit_candidate_openai_batch(run_root=root, stage_id="conservative_control")
    assert called == 1


def test_same_stage_submission_claim_allows_only_one_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    build_candidate_openai_cost_plan(run_root=root, max_cumulative_cost_usd=10.0)
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def submit(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=10)
        return _provider_payload(
            root,
            "conservative_control",
            status="validating",
            output_file_id=None,
        )

    monkeypatch.setattr(run_module, "submit_openai_batch", submit)
    with ThreadPoolExecutor(max_workers=2) as executor:
        winner = executor.submit(
            submit_candidate_openai_batch,
            run_root=root,
            stage_id="conservative_control",
        )
        assert entered.wait(timeout=10)
        loser = executor.submit(
            submit_candidate_openai_batch,
            run_root=root,
            stage_id="conservative_control",
        )
        with pytest.raises(FileExistsError, match="already been submitted"):
            loser.result(timeout=10)
        release.set()
        assert winner.result(timeout=10)["receipt_mode"] == "direct"
    assert calls == 1


def test_intent_only_submission_recovers_only_verified_provider_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    build_candidate_openai_cost_plan(run_root=root, max_cumulative_cost_usd=10.0)

    def interrupted(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("lost provider response")

    monkeypatch.setattr(run_module, "submit_openai_batch", interrupted)
    with pytest.raises(RuntimeError, match="lost provider response"):
        submit_candidate_openai_batch(run_root=root, stage_id="conservative_control")
    with pytest.raises(FileExistsError, match="indeterminate submission intent"):
        submit_candidate_openai_batch(run_root=root, stage_id="conservative_control")

    wrong = _provider_payload(
        root,
        "semantic_generation",
        status="validating",
        output_file_id=None,
    )
    wrong["batch_id"] = "batch-conservative_control"
    wrong["input_file_id"] = "input-conservative_control"
    monkeypatch.setattr(run_module, "retrieve_batch", lambda *args, **kwargs: wrong)
    with pytest.raises(ValueError, match="metadata, endpoint, or window drift"):
        recover_candidate_openai_submission(
            run_root=root,
            stage_id="conservative_control",
            batch_id="batch-conservative_control",
        )

    provider = _provider_payload(
        root,
        "conservative_control",
        status="validating",
        output_file_id=None,
    )
    monkeypatch.setattr(run_module, "retrieve_batch", lambda *args, **kwargs: provider)
    input_bytes = (root / "batches/conservative_control/input.jsonl").read_bytes()
    monkeypatch.setattr(
        run_module,
        "download_openai_file_bytes",
        lambda *args, **kwargs: input_bytes,
    )
    receipt = recover_candidate_openai_submission(
        run_root=root,
        stage_id="conservative_control",
        batch_id="batch-conservative_control",
    )
    assert receipt["receipt_mode"] == "recovered"
    assert receipt["provider_response"]["metadata"]["stage"] == ("conservative_control")


def test_rehashed_cross_receipt_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    run = load_candidate_openai_run(root)
    requests = tuple(
        item for item in run.initial_requests if item.stage_id == "conservative_control"
    )
    provider = tmp_path / "control.jsonl"
    _write_rows(provider, [_provider_row(item) for item in requests])
    _provider_ready(root, "conservative_control", provider, monkeypatch)
    path = root / "provider/conservative_control/status/receipt-0000.json"
    receipt = json.loads(path.read_text())
    receipt["submission_sha256"] = "0" * 64
    receipt.pop("status_sha256")
    receipt["status_sha256"] = canonical_sha256(receipt)
    path.write_text(json.dumps(receipt) + "\n")

    with pytest.raises(ValueError, match="status receipt binding drift"):
        load_candidate_openai_run(root)


@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_collect_rejects_missing_or_duplicate_custom_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    requests = tuple(
        item
        for item in load_candidate_openai_run(root).initial_requests
        if item.stage_id == "conservative_control"
    )
    rows = [_provider_row(item) for item in requests]
    rows = rows[:1] if failure == "missing" else [rows[0], rows[0]]
    provider = tmp_path / f"{failure}.jsonl"
    _write_rows(provider, rows)
    with pytest.raises(ValueError, match="identity mismatch|duplicate provider"):
        _provider_ready(root, "conservative_control", provider, monkeypatch)
        collect_candidate_openai_batch(run_root=root, stage_id="conservative_control")


def test_partial_events_and_rehashed_cost_telemetry_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    _collect_stage(
        root, "conservative_control", tmp_path / "control.jsonl", monkeypatch
    )
    events_path = root / "collections/conservative_control/events.jsonl"
    original = events_path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    _write_rows(events_path, rows[:1])
    with pytest.raises(ValueError, match="identity/order|manifest drift"):
        load_candidate_openai_run(root)

    events_path.write_text(original)
    rows[0]["cost"]["total_cost"] += 1.0
    unhashed = dict(rows[0])
    unhashed.pop("event_sha256")
    rows[0]["event_sha256"] = canonical_sha256(unhashed)
    _write_rows(events_path, rows)
    manifest_path = root / "collections/conservative_control/collection-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["events_file_sha256"] = file_sha256(events_path)
    manifest.pop("collection_sha256")
    manifest["collection_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="telemetry drift"):
        load_candidate_openai_run(root)


def test_rehashed_lowered_cost_plan_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    build_candidate_openai_cost_plan(run_root=root, max_cumulative_cost_usd=1.0)
    path = root / "cost-plan.json"
    plan = json.loads(path.read_text())
    plan["projected_cost_proxy_usd"] = 0.0
    plan["stages"][0]["projected_cost_proxy_usd"] = 0.0
    plan.pop("cost_plan_sha256")
    plan["cost_plan_sha256"] = canonical_sha256(plan)
    path.write_text(json.dumps(plan) + "\n")
    with pytest.raises(ValueError, match="cost plan contract drift"):
        load_candidate_openai_run(root)


def test_provider_error_is_preserved_with_incomplete_cost_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    requests = tuple(
        item
        for item in load_candidate_openai_run(root).initial_requests
        if item.stage_id == "conservative_control"
    )
    provider = tmp_path / "provider-error.jsonl"
    _write_rows(
        provider,
        [_provider_error_row(requests[0]), _provider_row(requests[1])],
    )
    _provider_ready(root, "conservative_control", provider, monkeypatch)
    events = collect_candidate_openai_batch(
        run_root=root, stage_id="conservative_control"
    )
    assert events[0].result.validation_status == "provider_error"
    assert events[0].cost.complete is False
    assert load_candidate_openai_run(root).events == events


def test_interrupted_stage_writes_do_not_publish_partial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    original_prepare = run_module.prepare_openai_candidate_batch_input

    def interrupted_prepare(requests: object, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}\n")
        raise RuntimeError("fixture interruption")

    monkeypatch.setattr(
        run_module,
        "prepare_openai_candidate_batch_input",
        interrupted_prepare,
    )
    with pytest.raises(RuntimeError, match="fixture interruption"):
        prepare_candidate_openai_batch(run_root=root, stage_id="conservative_control")
    assert not (root / "batches/conservative_control").exists()
    assert len(load_candidate_openai_run(root).initial_requests) == 12

    monkeypatch.setattr(
        run_module,
        "prepare_openai_candidate_batch_input",
        original_prepare,
    )
    prepare_candidate_openai_batch(run_root=root, stage_id="conservative_control")
    requests = tuple(
        item
        for item in load_candidate_openai_run(root).initial_requests
        if item.stage_id == "conservative_control"
    )
    provider = tmp_path / "interrupted-collection.jsonl"
    _write_rows(provider, [_provider_row(item) for item in requests])
    _provider_ready(root, "conservative_control", provider, monkeypatch)

    def interrupted_manifest(*args: object, **kwargs: object) -> None:
        raise RuntimeError("fixture collection interruption")

    monkeypatch.setattr(run_module, "atomic_write_json", interrupted_manifest)
    with pytest.raises(RuntimeError, match="fixture collection interruption"):
        collect_candidate_openai_batch(
            run_root=root,
            stage_id="conservative_control",
        )
    assert not (root / "collections/conservative_control").exists()
    assert load_candidate_openai_run(root).events == ()


def test_malformed_semantic_output_blocks_rewrite_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    requests = tuple(
        item
        for item in load_candidate_openai_run(root).initial_requests
        if item.stage_id == "semantic_generation"
    )
    rows = [_provider_row(item) for item in requests]
    rows[0] = _provider_row(requests[0], text="not-json")
    provider = tmp_path / "malformed.jsonl"
    _write_rows(provider, rows)
    _provider_ready(root, "semantic_generation", provider, monkeypatch)
    collect_candidate_openai_batch(run_root=root, stage_id="semantic_generation")
    with pytest.raises(ValueError, match="five successful outputs"):
        construct_candidate_openai_rewrites(run_root=root)


def test_loader_reconstructs_rewrite_hash_and_raw_result_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    _collect_stage(
        root, "semantic_generation", tmp_path / "semantic.jsonl", monkeypatch
    )
    construct_candidate_openai_rewrites(run_root=root)
    rewrite_path = root / "rewrite-requests.jsonl"
    rows = [json.loads(line) for line in rewrite_path.read_text().splitlines()]
    rows[0]["validated_semantic_output_sha256_in_order"][0] = "0" * 64
    unhashed = dict(rows[0])
    unhashed.pop("request_sha256")
    rows[0]["request_sha256"] = canonical_sha256(unhashed)
    _write_rows(rewrite_path, rows)
    with pytest.raises(ValueError, match="hash binding|rewrite request graph drift"):
        load_candidate_openai_run(root)


def test_archival_loader_rejects_rehashed_execution_inventory_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _initialized(tmp_path, monkeypatch)
    manifest_path = root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["execution_revision"]["files"][0]["path"] = manifest["execution_revision"][
        "files"
    ][1]["path"]
    manifest.pop("run_manifest_sha256")
    manifest["run_manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="file binding|source inventory"):
        load_candidate_openai_run(root, verify_sources=False)
