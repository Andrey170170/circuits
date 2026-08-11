from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from scripts.bonafide.build_t5_corpus_bundle import BUNDLE_SCHEMA
from scripts.bonafide.build_t5_pass1_salvage import build_salvage_plan
from scripts.bonafide.run_t5_pass1_salvage import (
    _forward_usr1,
    _validate_file,
    execute_salvage_task,
    load_salvage_plan,
)
from scripts.bonafide.runner import _sha256, normalized_trace_warmup
from scripts.bonafide.topk_runner import topk_runtime_artifact_identity
from tests.test_topk_runner import (
    _code_revision,
    _cpu_config,
    _runner_manifest,
    _runtime_environment,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _planner_fixture(tmp_path: Path) -> dict[str, object]:
    frozen = tmp_path / "frozen"
    orchestration = tmp_path / "orchestration"
    artifacts = tmp_path / "artifacts"
    frozen.mkdir()
    orchestration.mkdir()
    artifacts.mkdir()
    config = _cpu_config()
    config_path = frozen / "scripts" / "bonafide" / "configs" / "config.json"
    _write_json(config_path, config)
    python_bin = frozen / "env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("test", encoding="utf-8")
    python_bin.chmod(0o755)

    common_sources = {}
    for name in ("selection", "source_manifest", "rank_screen"):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"kind": name})
        common_sources[f"{name}_path"] = str(path)
        common_sources[f"{name}_sha256"] = file_sha256(path)

    manifests = []
    tasks = []
    selected_manifest = None
    for candidate_index in range(6):
        manifest = _runner_manifest()
        manifest["trace_family"]["trace_family_id"] = (
            f"bonafide.t5-test.independent-candidate-{candidate_index}.v1"
        )
        manifest["waves"][0]["wave_id"] = f"t5-test-c{candidate_index}-000"
        if candidate_index == 0:
            second = deepcopy(manifest["waves"][0]["items"][0])
            second["artifact_id"] = "source-width1-b"
            second["example"]["example_id"] = "example-b"
            manifest["waves"][0]["items"].append(second)
            selected_manifest = manifest
        path = tmp_path / f"candidate-{candidate_index}.json"
        _write_json(path, manifest)
        manifests.append(
            {
                "candidate_index": candidate_index,
                "label": f"independent-candidate-{candidate_index}",
                "path": str(path),
                "sha256": file_sha256(path),
                "canonical_sha256": _sha256(manifest),
                "trace_family_id": manifest["trace_family"]["trace_family_id"],
                "wave_count": 1,
                "work_item_count": len(manifest["waves"][0]["items"]),
            }
        )
        tasks.append(
            {
                "task_index": candidate_index,
                "candidate_index": candidate_index,
                "manifest_path": str(path),
                "manifest_sha256": file_sha256(path),
                "wave_id": manifest["waves"][0]["wave_id"],
                "corpus_role": manifest["waves"][0]["corpus_role"],
                "work_item_count": len(manifest["waves"][0]["items"]),
            }
        )
    assert selected_manifest is not None
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "cohort_id": "test-cohort",
        **common_sources,
        "counts": {},
        "execution_profile": {},
        "manifests": manifests,
        "tasks": tasks,
    }
    bundle_path = tmp_path / "bundle.json"
    _write_json(bundle_path, bundle)
    return {
        "frozen": frozen,
        "orchestration": orchestration,
        "artifacts": artifacts,
        "config": config,
        "config_path": config_path,
        "python_bin": python_bin,
        "bundle_path": bundle_path,
        "manifest": selected_manifest,
    }


def test_salvage_planner_lists_only_missing_and_binds_prior_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.bonafide.build_t5_pass1_salvage as planner

    fixture = _planner_fixture(tmp_path)
    code_revision = _code_revision()
    runtime = _runtime_environment()
    monkeypatch.setattr(
        planner, "validate_frozen_source_tree", lambda _tree, _commit: code_revision
    )
    monkeypatch.setattr(
        planner,
        "discover_identity_execution_contract",
        lambda _root, _manifests, _revision, _waves: {
            "contract": {
                "model": deepcopy(fixture["config"]["model"]),
                "adag_config": deepcopy(fixture["config"]["adag_config"]),
                "trace_warmup": normalized_trace_warmup(fixture["config"]),
                "batch_size": 1,
                "code_revision": code_revision,
                "runtime_environment": runtime,
            },
            "metadata_artifact_count": 1,
            "full_integrity_reference_artifacts": [
                {"path": "test", "manifest_sha256": "a" * 64}
            ],
        },
    )
    manifest = fixture["manifest"]
    assert isinstance(manifest, dict)
    first, second = manifest["waves"][0]["items"]
    artifact_id, identity = topk_runtime_artifact_identity(
        first,
        config=fixture["config"],
        trace_family=manifest["trace_family"],
        code_revision=code_revision,
        runtime_environment=runtime,
        source_manifest_sha256=manifest["source"]["width1_manifest_sha256"],
        topk_manifest_sha256=_sha256(manifest),
        wave_id=manifest["waves"][0]["wave_id"],
    )
    complete_path = (
        fixture["artifacts"]
        / manifest["trace_family"]["trace_family_id"]
        / manifest["waves"][0]["wave_id"]
        / artifact_id
    )
    complete_path.mkdir(parents=True)

    def validate_complete(path):
        if path != complete_path:
            raise AssertionError(path)
        return {
            "artifact_id": artifact_id,
            "artifact_identity": identity,
            "data_sha256": "f" * 64,
        }

    monkeypatch.setattr(
        planner, "validate_topk_compact_trace_integrity", validate_complete
    )
    summary = (
        fixture["artifacts"]
        / "execution-summaries"
        / manifest["waves"][0]["wave_id"]
        / "failed.jsonl"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "status": "error",
                "wave_id": manifest["waves"][0]["wave_id"],
                "source_width1_artifact_id": second["artifact_id"],
                "error_type": "ValueError",
                "error": "non-finite attribution",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # An unrelated array task may still be appending a partial JSON line. The
    # scoped planner must never read or hash that live file.
    unrelated = (
        fixture["artifacts"] / "execution-summaries" / "other-wave" / "live.jsonl"
    )
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text('{"status": "error"', encoding="utf-8")

    kwargs = {
        "bundle_path": fixture["bundle_path"],
        "bundle_sha256": file_sha256(fixture["bundle_path"]),
        "artifact_root": fixture["artifacts"],
        "frozen_source_tree": fixture["frozen"],
        "frozen_git_commit": "a" * 40,
        "orchestration_source_tree": fixture["orchestration"],
        "orchestration_git_commit": "d" * 40,
        "config_path": fixture["config_path"],
        "python_bin": fixture["python_bin"],
        "selected_task_indices": [0],
    }
    plan = build_salvage_plan(**kwargs)
    repeated = build_salvage_plan(**kwargs)

    assert plan == repeated
    assert plan["counts"] == {
        "scoped_original_tasks": 1,
        "scoped_expected_artifacts": 2,
        "completed_artifacts": 1,
        "missing_artifacts": 1,
        "missing_with_prior_failures": 1,
        "known_failure_events": 1,
        "salvage_tasks": 1,
    }
    assert plan["items"][0]["source_width1_artifact_id"] == second["artifact_id"]
    assert plan["items"][0]["prior_failures"][0]["error_type"] == "ValueError"
    assert plan["execution"]["one_frozen_runner_subprocess_per_artifact"] is True
    assert plan["scan"]["mode"] == "terminal_failure_repair"
    assert plan["scan"]["terminal_failure_wave_ids"] == [
        manifest["waves"][0]["wave_id"]
    ]
    assert plan["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in plan.items() if key != "manifest_sha256"}
    )

    monkeypatch.setattr(
        planner,
        "validate_topk_compact_trace_integrity",
        lambda _path: (_ for _ in ()).throw(ValueError("payload checksum drift")),
    )
    with pytest.raises(ValueError, match="payload checksum drift"):
        build_salvage_plan(**kwargs)
    monkeypatch.setattr(
        planner, "validate_topk_compact_trace_integrity", validate_complete
    )

    wrong_config = deepcopy(fixture["config"])
    wrong_config["adag_config"]["edge_threshold"] = 0.02
    _write_json(fixture["config_path"], wrong_config)
    with pytest.raises(ValueError, match="supplied trace config does not match"):
        build_salvage_plan(**kwargs)

    _write_json(fixture["config_path"], fixture["config"])
    summary.unlink()
    with pytest.raises(ValueError, match="terminal error/oom evidence"):
        build_salvage_plan(**kwargs)
    quiescent = build_salvage_plan(
        **kwargs,
        allow_quiescent_missing_scan=True,
    )
    assert quiescent["scan"]["mode"] == "quiescent_missing_scan"
    assert quiescent["scan"]["allow_quiescent_missing_scan"] is True

    all_task_kwargs = {**kwargs, "selected_task_indices": None}
    with pytest.raises(ValueError, match="globally quiescent"):
        build_salvage_plan(**all_task_kwargs)


def test_salvage_executor_continues_after_one_runner_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.bonafide.run_t5_pass1_salvage as executor

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    items = [
        {
            "salvage_item_index": index,
            "original_task_index": 37,
            "manifest_path": str(manifest_path),
            "manifest_sha256": "a" * 64,
            "wave_id": "wave",
            "source_width1_artifact_id": f"source-{index}",
            "expected_artifact_id": f"expected-{index}",
            "expected_artifact_identity_sha256": str(index) * 64,
            "expected_artifact_identity": {"sha256": str(index) * 64},
            "expected_artifact_path": str(artifacts / f"expected-{index}"),
        }
        for index in range(2)
    ]
    plan = {
        "manifest_sha256": "f" * 64,
        "execution": {"runner_module": "scripts.bonafide.topk_runner"},
        "items": items,
        "tasks": [{"task_index": 0, "item_count": 2, "salvage_item_indices": [0, 1]}],
    }
    contract = {
        "frozen_tree": tmp_path,
        "config_path": tmp_path / "config.json",
        "python_bin": tmp_path / "python",
        "artifact_root": artifacts,
    }
    monkeypatch.setattr(executor, "validate_execution_contract", lambda _plan: contract)
    monkeypatch.setattr(
        executor, "_validate_file", lambda path, _digest, _label: Path(str(path))
    )
    calls = []

    def invoke(command, **_kwargs):
        calls.append(command)
        expected = command[command.index("--only-artifact-id") + 1]
        if expected == "source-0":
            return subprocess.CompletedProcess(command, 1, "", "trace failed")
        stdout = json.dumps({"status": "complete", "artifact_id": "expected-1"})
        return subprocess.CompletedProcess(command, 0, stdout + "\n", "")

    monkeypatch.setattr(executor, "_invoke_runner", invoke)
    monkeypatch.setattr(executor, "_validate_expected_artifact", lambda _item: None)

    failures, receipt_path = execute_salvage_task(
        plan=plan, task_index=0, attempt_id="test-attempt"
    )

    assert failures == 1
    assert len(calls) == 2
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    assert [record["status"] for record in receipts] == ["failed", "complete"]
    assert all(
        record["event_sha256"]
        == canonical_sha256(
            {key: value for key, value in record.items() if key != "event_sha256"}
        )
        for record in receipts
    )


def test_salvage_plan_loader_rejects_self_hash_drift(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    value = {
        "schema_version": "bonafide-t5-pass1-salvage-plan/v1",
        "counts": {},
    }
    value["manifest_sha256"] = canonical_sha256(value)
    _write_json(path, value)
    valid_file_hash = file_sha256(path)
    load_salvage_plan(path, valid_file_hash)

    with pytest.raises(ValueError, match="absolute file"):
        load_salvage_plan(Path("plan.json"), valid_file_hash)
    with pytest.raises(ValueError, match="absolute file"):
        _validate_file(Path("plan.json"), valid_file_hash, "test input")

    value["counts"] = {"missing": 1}
    _write_json(path, value)
    with pytest.raises(ValueError, match="self-hash drift"):
        load_salvage_plan(path, file_sha256(path))


def test_salvage_usr1_forwarding_tolerates_child_exit_race() -> None:
    class ExitedDuringSignal:
        @staticmethod
        def poll():
            return None

        @staticmethod
        def send_signal(_signum):
            raise ProcessLookupError

    _forward_usr1(ExitedDuringSignal())


def test_salvage_runner_forwards_usr1_requested_before_child_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.bonafide.run_t5_pass1_salvage as executor

    class Child:
        returncode = 0

        def __init__(self) -> None:
            self.signals = []

        def poll(self):
            return None

        def send_signal(self, signum):
            self.signals.append(signum)

        def communicate(self):
            return "", ""

    child = Child()
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *_args, **_kwargs: child)
    state = {"requested": True, "process": None}

    executor._invoke_runner(
        ["python", "runner.py"],
        cwd=tmp_path,
        environment={},
        signal_state=state,
    )

    assert child.signals == [executor.signal.SIGUSR1]
    assert state["process"] is None
