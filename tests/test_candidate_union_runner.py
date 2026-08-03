from __future__ import annotations

from pathlib import Path

import pytest
from circuits.tracing.artifact import (
    save_topk_compact_trace,
    validate_topk_compact_trace_integrity,
)
from circuits.tracing.candidate_union import frozen_union_topologies
from scripts.bonafide.candidate_union_runner import (
    _assert_refinement_topology,
    run_candidate_union_wave,
    validate_candidate_union_plan,
)
from scripts.bonafide.runner import _sha256
from tests.test_bonafide_benchmark import _config
from tests.test_topk_topology_comparison import _joint_and_references


def _saved_references(tmp_path: Path):
    _joint, traces, _candidate_ids = _joint_and_references()
    records = []
    for index, trace in enumerate(traces):
        path = tmp_path / f"reference-{index}"
        save_topk_compact_trace(
            path,
            trace,
            metrics={"status": "complete"},
            manifest={
                "artifact_id": f"reference-{index}",
                "source_width1_artifact_id": "source-1",
            },
        )
        manifest = validate_topk_compact_trace_integrity(path)
        records.append(
            {
                "candidate_index": index,
                "token_id": trace.candidate_selection.candidates[0].token_id,
                "artifact_id": manifest["artifact_id"],
                "path": str(path),
                "payload_sha256": manifest["data_sha256"],
            }
        )
    return traces, records


def _plan(records) -> dict:
    return {
        "schema_version": "bonafide-candidate-union-plan/v1",
        "source": {
            "model_id": "fake/model",
            "model_revision": "exact-revision",
            "tokenizer_revision": "exact-revision",
            "chat_template_sha256": "a" * 64,
        },
        "waves": [
            {
                "wave_id": "candidate-union-c0-test",
                "cases": [
                    {
                        "case_id": "case-1",
                        "source_width1_artifact_id": "source-1",
                        "source_item": {"artifact_id": "source-1"},
                        "reference_artifacts": records,
                    }
                ],
            }
        ],
    }


def _code_revision() -> dict:
    return {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "git_status_sha256": "b" * 64,
        "source_tree_sha256": "c" * 64,
    }


def test_candidate_union_wave_dry_run_binds_reference_topology(
    tmp_path: Path,
) -> None:
    _traces, records = _saved_references(tmp_path)
    config = _config()
    config["model"]["device"] = "cpu"
    config["model"]["dtype"] = "float32"

    results = run_candidate_union_wave(
        config=config,
        plan=_plan(records),
        wave_id="candidate-union-c0-test",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
        dry_run=True,
        _code_revision=_code_revision(),
        _runtime_environment={"python": "test"},
    )

    assert len(results) == 1
    assert results[0]["status"] == "planned"
    assert results[0]["candidate_count"] == 5
    assert len(results[0]["topology_sha256"]) == 64
    assert not (tmp_path / "summary.jsonl").exists()


def test_candidate_union_plan_rejects_reference_order_drift(tmp_path: Path) -> None:
    _traces, records = _saved_references(tmp_path)
    records[1]["candidate_index"] = 4

    with pytest.raises(ValueError, match="indices are not ordered"):
        validate_candidate_union_plan(_plan(records))


def test_candidate_union_execution_contract_rejects_config_or_code_drift(
    tmp_path: Path,
) -> None:
    _traces, records = _saved_references(tmp_path)
    config = _config()
    config["model"]["device"] = "cpu"
    config["model"]["dtype"] = "float32"
    plan = _plan(records)
    plan["execution"] = {
        "config_canonical_sha256": _sha256(config),
        "required_clean_worktree": True,
    }
    changed = {**config, "test_drift": True}
    with pytest.raises(ValueError, match="config hash drift"):
        run_candidate_union_wave(
            config=changed,
            plan=plan,
            wave_id="candidate-union-c0-test",
            artifact_root=tmp_path / "artifacts",
            summary_jsonl=tmp_path / "summary.jsonl",
            dry_run=True,
            _code_revision=_code_revision(),
            _runtime_environment={"python": "test"},
        )
    dirty = {**_code_revision(), "git_dirty": True}
    with pytest.raises(ValueError, match="clean frozen worktree"):
        run_candidate_union_wave(
            config=config,
            plan=plan,
            wave_id="candidate-union-c0-test",
            artifact_root=tmp_path / "artifacts",
            summary_jsonl=tmp_path / "summary.jsonl",
            dry_run=True,
            _code_revision=dirty,
            _runtime_environment={"python": "test"},
        )


def test_refinement_topology_requires_exact_nodes_and_edges(
    tmp_path: Path,
) -> None:
    traces, records = _saved_references(tmp_path)
    from circuits.tracing.artifact import load_topk_compact_trace

    artifacts = [load_topk_compact_trace(record["path"]) for record in records]
    _topology_sha256, topologies = frozen_union_topologies(artifacts)
    _assert_refinement_topology(
        traces[0],
        topologies[0],
        traces[0].candidate_selection.candidates[0].token_id,
    )

    traces[0].circuit_data.df_edge = traces[0].circuit_data.df_edge.iloc[1:]
    with pytest.raises(ValueError, match="fixed-topology edge mismatch"):
        _assert_refinement_topology(
            traces[0],
            topologies[0],
            traces[0].candidate_selection.candidates[0].token_id,
        )
