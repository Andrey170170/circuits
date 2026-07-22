"""Focused tests for final-trace execution planning and compound execution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from circuits.tracing.clja import ADAGConfig
from circuits.tracing.trace import CircuitData
from scripts.bonafide.execution_plan import (
    build_execution_plan,
    canonical_json,
    sha256_file,
    validate_execution_plan,
    write_execution_plan,
)
from scripts.bonafide.manifest import SCHEMA_VERSION
from scripts.bonafide.runner import RUN_CONFIG_SCHEMA, run_compound_shard


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _instrumentation(edges: int, chunks: int) -> dict:
    return {
        "counters": {
            "candidate_mlp_edge_count": edges,
            "planned_jacobian_target_chunk_executions": chunks,
        }
    }


def _source_item(index: int, probe_id: str, edges: int) -> dict:
    return {
        "artifact_id": f"source-{index}",
        "response_token_count": 1,
        "target_selection": {
            "width": 1,
            "kind": "explicit_response_positions",
            "response_token_positions": [0],
            "final_target_token_id": 77,
            "final_selection": {
                "corpus_role": "dense_discovery",
                "source_refinement_probe_id": probe_id,
                "refinement_diagnostics": {
                    "candidate_mlp_edge_count": edges,
                    "token_text": f" token-{index}",
                },
                "selection_reasons": [{"bucket": "phase_control"}],
            },
        },
        "objective": {"benchmark_only_multi_target": False, "name": "single_selected_logit"},
        "example": {
            "example_id": f"example-{index}",
            "annotation_row_ids": [f"row-{index}"],
            "question_ids": [f"question-{index}"],
            "label_types": ["FAITHFUL_STEP"],
            "prompt": f"prompt {index}",
            "response": f"response {index}",
        },
    }


def _fixture(tmp_path: Path, *, omit_probe: int | None = None) -> tuple[dict, dict, Path, Path]:
    config = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "batch_size": 1,
        "model": {
            "model_id": "fake/model",
            "revision": "exact-revision",
            "device": "cpu",
            "dtype": "float32",
        },
        "adag_config": {},
    }
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    calibration = [
        (10, 1, 10),
        (20, 2, 15),
        (30, 4, 12),
        (40, 3, 20),
        (50, 6, 18),
        (60, 5, 25),
    ]
    historical = [
        {
            "status": "complete",
            "source_artifact_id": f"historical-{index}",
            "input_token_count": inputs,
            "trace_wall_seconds": 5.0 + 0.7 * edges + 1.2 * chunks + 0.3 * inputs,
            "instrumentation": _instrumentation(edges, chunks),
        }
        for index, (edges, chunks, inputs) in enumerate(calibration)
    ]
    historical_path = tmp_path / "historical.jsonl"
    _write_jsonl(historical_path, historical)
    workloads = [(100, 7, 30), (200, 8, 40), (600_000, 12, 50), (700_000, 13, 55), (800_000, 14, 60), (20_000_000, 20, 70)]
    probes = [
        {
            "status": "complete",
            "artifact_id": f"probe-{index}",
            "input_token_count": inputs,
            "instrumentation": _instrumentation(edges, chunks),
        }
        for index, (edges, chunks, inputs) in enumerate(workloads)
        if index != omit_probe
    ]
    refinement_path = tmp_path / "refinement.jsonl"
    _write_jsonl(refinement_path, probes)
    waves = []
    for index, (edges, _chunks, _inputs) in enumerate(workloads):
        extreme = index >= 2
        waves.append(
            {
                "wave_id": f"wave-{index}",
                "corpus_role": "dense_discovery",
                "cluster_fit_eligible": True,
                "extreme_workload_isolation": extreme,
                "items": [_source_item(index, f"probe-{index}", edges)],
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tokenizer": {"model_id": "fake/model", "revision": "exact-revision"},
        "source_artifacts": {
            "refinement_summary": {"sha256": sha256_file(refinement_path)}
        },
        "waves": waves,
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    plan = build_execution_plan(
        manifest_path=manifest_path,
        config_path=config_path,
        historical_summary_path=historical_path,
        refinement_summary_path=refinement_path,
        shard_count=1,
        expected_historical_sha256=sha256_file(historical_path),
    )
    plan_path = tmp_path / "plan.json"
    write_execution_plan(plan_path, plan)
    return config, manifest, plan_path, manifest_path


def _rehash(plan: dict) -> None:
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = hashlib.sha256(canonical_json(plan)).hexdigest()


def _trace() -> CircuitData:
    return CircuitData(
        df_node=pd.DataFrame({"attribution": [0.1], "activation": [0.2], "layer": [0]}),
        df_edge=pd.DataFrame({"attribution": [0.3], "weight": [0.4], "layer": ["0->1"]}),
        cis=[[1, 2]],
        attention_masks=[[1, 1]],
        labels=["example"],
        target_logits=[[77]],
        target_logit_probs=[[0.5]],
        target_logit_values=[[1.0]],
        target_provenance=[{"response_token_position": 0, "token_id": 77}],
        trace_metadata={"response_token_count": 1},
        benchmark_only=False,
        k=1,
        config=ADAGConfig(device="cpu"),
        model_id="fake/model",
    )


def test_plan_is_deterministic_balanced_and_separates_extremes(tmp_path: Path) -> None:
    _, manifest, plan_path, _ = _fixture(tmp_path)
    first = json.loads(plan_path.read_text())
    validate_execution_plan(first, manifest=manifest)
    assert first["cost_model"]["training_record_count"] == 6
    assert first["sharding"]["routine_target_count"] == 2
    assert len(first["extremes"]["preflight"]) == 3
    assert first["extremes"]["manual_pathological"][0]["workload"]["candidate_mlp_edge_count"] == 20_000_000
    assert [task["task_index"] for task in first["tasks"]] == list(range(5))
    assert first["tasks"][-1]["requires_explicit_manual_opt_in"] is True


def test_plan_rejects_duplicate_assignments_and_source_hash_drift(tmp_path: Path) -> None:
    _, manifest, plan_path, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    duplicate = copy.deepcopy(plan["sharding"]["shards"][0]["items"][0])
    plan["sharding"]["shards"][0]["items"].append(duplicate)
    seconds = duplicate["estimated_seconds"]
    plan["sharding"]["shards"][0]["item_count"] += 1
    plan["sharding"]["shards"][0]["estimated_seconds"] += seconds
    plan["sharding"]["routine_target_count"] += 1
    plan["sharding"]["routine_total_estimated_seconds"] += seconds
    plan["tasks"][0]["item_count"] += 1
    _rehash(plan)
    with pytest.raises(ValueError, match="duplicate target assignments"):
        validate_execution_plan(plan, manifest=manifest)

    original = json.loads(plan_path.read_text())
    Path(original["sources"]["refinement_probe_summary"]["path"]).write_text("{}\n")
    with pytest.raises(ValueError, match="source hash drift"):
        validate_execution_plan(original, manifest=manifest)


def test_plan_fails_closed_on_missing_final_probe_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing refinement metrics"):
        _fixture(tmp_path, omit_probe=0)


def test_compound_dry_run_does_not_load_model(tmp_path: Path, monkeypatch) -> None:
    import scripts.bonafide.runner as runner_module

    config, manifest, plan_path, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    monkeypatch.setattr(
        runner_module,
        "_load_model_and_tokenizer",
        lambda _config: pytest.fail("dry run loaded model"),
    )
    records = run_compound_shard(
        config=config,
        manifest=manifest,
        execution_plan=plan,
        task_index=0,
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
        dry_run=True,
    )
    assert len(records) == 2
    assert {record["wave_id"] for record in records} == {"wave-0", "wave-1"}
    assert not (tmp_path / "summary.jsonl").exists()


def test_compound_loads_once_and_preserves_wave_identity_and_path(tmp_path: Path, monkeypatch) -> None:
    import scripts.bonafide.runner as runner_module

    config, manifest, plan_path, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    loads = []
    monkeypatch.setattr(runner_module, "collect_code_revision", lambda _root: {"revision": "same"})
    monkeypatch.setattr(runner_module, "collect_runtime_environment", lambda: {"runtime": "same"})
    monkeypatch.setattr(
        runner_module,
        "_load_model_and_tokenizer",
        lambda _config: (loads.append(1) or object(), object()),
    )
    monkeypatch.setattr(runner_module, "trace_teacher_forced_response", lambda **_kwargs: _trace())
    records = run_compound_shard(
        config=config,
        manifest=manifest,
        execution_plan=plan,
        task_index=0,
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
    )
    complete = [record for record in records if record.get("status") == "complete"]
    assert loads == [1]
    assert {record["wave_id"] for record in complete} == {"wave-0", "wave-1"}
    assert all(Path(record["artifact_path"]).parent.name == record["wave_id"] for record in complete)
    assert records[-1]["status"] == "task_complete"
    assert records[-1]["completed_item_count"] == 2


def test_compound_rejects_warmup_applicable_source_wave(tmp_path: Path) -> None:
    config, manifest, plan_path, _ = _fixture(tmp_path)
    config["trace_warmup"] = {
        "enabled": True,
        "mode": "first_wave_item_full_trace_discard",
        "wave_id_prefixes": ["wave-"],
    }
    plan = json.loads(plan_path.read_text())
    # Keep the plan's config identity in sync so the warmup rejection is the first failure.
    config_path = Path(plan["sources"]["trace_run_config"]["path"])
    _write_json(config_path, config)
    plan["sources"]["trace_run_config"]["sha256"] = sha256_file(config_path)
    plan["sources"]["trace_run_config"]["canonical_sha256"] = hashlib.sha256(
        canonical_json(config)
    ).hexdigest()
    _rehash(plan)
    with pytest.raises(ValueError, match="warmup-applicable"):
        run_compound_shard(
            config=config,
            manifest=manifest,
            execution_plan=plan,
            task_index=0,
            artifact_root=tmp_path / "artifacts",
            summary_jsonl=tmp_path / "summary.jsonl",
            dry_run=True,
        )


def test_compound_generic_failure_records_terminal_task_event(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.bonafide.runner as runner_module

    config, manifest, plan_path, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text())
    monkeypatch.setattr(runner_module, "collect_code_revision", lambda _root: {"revision": "same"})
    monkeypatch.setattr(runner_module, "collect_runtime_environment", lambda: {"runtime": "same"})
    monkeypatch.setattr(
        runner_module,
        "_load_model_and_tokenizer",
        lambda _config: (object(), object()),
    )

    def fail_trace(**_kwargs):
        raise RuntimeError("synthetic compound failure")

    monkeypatch.setattr(runner_module, "trace_teacher_forced_response", fail_trace)
    summary = tmp_path / "summary.jsonl"
    with pytest.raises(RuntimeError, match="synthetic compound failure"):
        run_compound_shard(
            config=config,
            manifest=manifest,
            execution_plan=plan,
            task_index=0,
            artifact_root=tmp_path / "artifacts",
            summary_jsonl=summary,
        )

    records = [json.loads(line) for line in summary.read_text().splitlines()]
    assert [record["status"] for record in records] == [
        "task_started",
        "error",
        "task_stopped",
    ]
    assert records[-1]["stop_reason"] == "trace_error"
    assert records[-1]["remaining_item_count"] == 2


def test_compound_gate_writes_task_stop_and_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    import scripts.bonafide.runner as runner_module

    config, manifest, plan_path, _ = _fixture(tmp_path)
    config["wave_limits"] = {"max_trace_seconds": 0.0}
    plan = json.loads(plan_path.read_text())
    config_path = Path(plan["sources"]["trace_run_config"]["path"])
    _write_json(config_path, config)
    plan["sources"]["trace_run_config"]["sha256"] = sha256_file(config_path)
    plan["sources"]["trace_run_config"]["canonical_sha256"] = hashlib.sha256(
        canonical_json(config)
    ).hexdigest()
    _rehash(plan)
    monkeypatch.setattr(runner_module, "collect_code_revision", lambda _root: {"revision": "same"})
    monkeypatch.setattr(runner_module, "collect_runtime_environment", lambda: {"runtime": "same"})
    monkeypatch.setattr(
        runner_module, "_load_model_and_tokenizer", lambda _config: (object(), object())
    )
    monkeypatch.setattr(runner_module, "trace_teacher_forced_response", lambda **_kwargs: _trace())
    summary = tmp_path / "summary.jsonl"
    with pytest.raises(RuntimeError, match="compound task 0 stopped"):
        run_compound_shard(
            config=config,
            manifest=manifest,
            execution_plan=plan,
            task_index=0,
            artifact_root=tmp_path / "artifacts",
            summary_jsonl=summary,
        )
    records = [json.loads(line) for line in summary.read_text().splitlines()]
    stop = records[-1]
    assert stop["status"] == "task_stopped"
    assert stop["stop_reason"] == "max_trace_seconds_exceeded"
    assert stop["remaining_item_count"] == 1
