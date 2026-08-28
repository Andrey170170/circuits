from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from circuits.tracing.trace import TOPK_TRACE_FAMILY_ID
from scripts.bonafide.execution_plan import sha256_file
from scripts.bonafide.topk_runner import (
    _trace_cuda_peak_bytes,
    run_topk_wave,
    topk_runtime_artifact_identity,
)
from tests.test_bonafide_benchmark import _config, _single_item_manifest
from tests.test_teacher_forced_trace import _topk_trace
from tests.test_topk_manifest import _manifest


def _runner_manifest() -> dict:
    manifest = _manifest()
    manifest["phase"] = "c0_candidate_reference"
    manifest["trace_family"] = {
        "trace_family_id": TOPK_TRACE_FAMILY_ID,
        "candidate_policy_id": "observed_plus_top4_alternatives",
        "candidate_policy_version": "1",
        "candidate_count": 5,
        "joint_objective_id": "raw_logit_sum",
        "joint_objective_version": "1",
    }
    item = manifest["waves"][0]["items"][0]
    item["target_selection"]["response_token_positions"] = [0]
    item["target_selection"]["width"] = 1
    item["target_selection"]["final_target_token_id"] = 40
    item["target_selection"].pop("sampling", None)
    return manifest


def _cpu_config() -> dict:
    config = _config()
    config["model"]["device"] = "cpu"
    config["model"]["dtype"] = "float32"
    return config


def _code_revision() -> dict:
    return {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "git_status_sha256": "b" * 64,
        "source_tree_sha256": "c" * 64,
    }


def _runtime_environment() -> dict:
    return {"python": "3.12.12", "packages": {"torch": "test"}}


def test_topk_runtime_identity_changes_with_candidate_policy() -> None:
    manifest = _runner_manifest()
    item = manifest["waves"][0]["items"][0]
    common = {
        "item": item,
        "config": _cpu_config(),
        "code_revision": _code_revision(),
        "runtime_environment": _runtime_environment(),
        "source_manifest_sha256": "d" * 64,
        "topk_manifest_sha256": "e" * 64,
        "wave_id": "parity-01",
    }

    observed_id, observed_identity = topk_runtime_artifact_identity(
        trace_family=manifest["trace_family"], **common
    )
    model_top5 = {
        **manifest["trace_family"],
        "trace_family_id": "bonafide.model-top5.v1",
        "candidate_policy_id": "model_top5",
    }
    model_id, model_identity = topk_runtime_artifact_identity(
        trace_family=model_top5, **common
    )

    assert observed_id != model_id
    assert observed_identity["sha256"] != model_identity["sha256"]


def test_topk_runtime_identity_binds_explicit_instrumentation_policy() -> None:
    manifest = _runner_manifest()
    item = manifest["waves"][0]["items"][0]
    common = {
        "item": item,
        "trace_family": manifest["trace_family"],
        "code_revision": _code_revision(),
        "runtime_environment": _runtime_environment(),
        "source_manifest_sha256": "d" * 64,
        "topk_manifest_sha256": "e" * 64,
        "wave_id": "parity-01",
    }
    base_config = _cpu_config()
    profiling_config = deepcopy(base_config)
    profiling_config["instrumentation"] = {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": True,
    }

    base_id, base_identity = topk_runtime_artifact_identity(
        config=base_config, **common
    )
    profiling_id, profiling_identity = topk_runtime_artifact_identity(
        config=profiling_config, **common
    )

    assert profiling_id != base_id
    assert "instrumentation" not in base_identity
    assert profiling_identity["instrumentation"] == {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": True,
    }


def test_topk_runtime_identity_binds_cuda_headroom_policy_and_threshold() -> None:
    manifest = _runner_manifest()
    item = manifest["waves"][0]["items"][0]
    common = {
        "item": item,
        "trace_family": manifest["trace_family"],
        "code_revision": _code_revision(),
        "runtime_environment": _runtime_environment(),
        "source_manifest_sha256": "d" * 64,
        "topk_manifest_sha256": "e" * 64,
        "wave_id": "parity-01",
    }
    legacy_config = _cpu_config()
    threshold_config = deepcopy(legacy_config)
    threshold_config["wave_limits"] = {"min_cuda_headroom_bytes": 8}
    allocator_config = deepcopy(threshold_config)
    allocator_config["instrumentation"] = {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": True,
        "cuda_dense_joint_pressure_telemetry": True,
    }
    allocator_config["wave_limits"]["cuda_headroom_policy"] = "allocator_dense_joint_v1"
    allocator_config["wave_limits"]["cuda_headroom_action"] = "warn"

    legacy_id, legacy_identity = topk_runtime_artifact_identity(
        config=legacy_config, **common
    )
    threshold_id, threshold_identity = topk_runtime_artifact_identity(
        config=threshold_config, **common
    )
    allocator_id, allocator_identity = topk_runtime_artifact_identity(
        config=allocator_config, **common
    )

    assert legacy_id == threshold_id
    assert allocator_id != legacy_id
    assert "cuda_headroom_gate" not in legacy_identity
    assert "cuda_headroom_gate" not in threshold_identity
    assert allocator_identity["cuda_headroom_gate"] == {
        "policy": "allocator_dense_joint_v1",
        "min_cuda_headroom_bytes": 8,
        "action": "warn",
        "sampling_version": "boundary_cuda_metrics_v1",
    }


def test_topk_runtime_identity_binds_explicit_cuda_allocator_policy() -> None:
    manifest = _runner_manifest()
    item = manifest["waves"][0]["items"][0]
    config = _cpu_config()
    config["cuda_allocator_policy"] = "expandable_segments_v1"
    _, identity = topk_runtime_artifact_identity(
        item,
        config=config,
        trace_family=manifest["trace_family"],
        code_revision=_code_revision(),
        runtime_environment={
            **_runtime_environment(),
            "cuda_allocator_policy": {"intended_policy_id": "expandable_segments_v1"},
        },
        source_manifest_sha256="d" * 64,
        topk_manifest_sha256="e" * 64,
        wave_id="parity-01",
    )

    assert identity["cuda_allocator_policy"] == "expandable_segments_v1"
    assert identity["runtime_environment"]["cuda_allocator_policy"] == {
        "intended_policy_id": "expandable_segments_v1"
    }


def test_topk_metrics_use_reset_safe_instrumentation_peaks(monkeypatch) -> None:
    monkeypatch.setattr(
        "torch.cuda.max_memory_allocated",
        lambda: (_ for _ in ()).throw(AssertionError("used reset CUDA peak")),
    )
    monkeypatch.setattr(
        "torch.cuda.max_memory_reserved",
        lambda: (_ for _ in ()).throw(AssertionError("used reset CUDA peak")),
    )
    assert _trace_cuda_peak_bytes(
        {
            "cuda_memory": {
                "overall": {
                    "peak": {
                        "peak_allocated_bytes": 123,
                        "peak_reserved_bytes": 456,
                    }
                }
            }
        },
        cuda_memory_telemetry=True,
        uses_cuda=True,
    ) == (123, 456)

    with pytest.raises(ValueError, match="peak_reserved_bytes is invalid"):
        _trace_cuda_peak_bytes(
            {
                "cuda_memory": {
                    "overall": {
                        "peak": {
                            "peak_allocated_bytes": 123,
                            "peak_reserved_bytes": "456",
                        }
                    }
                }
            },
            cuda_memory_telemetry=True,
            uses_cuda=True,
        )


def test_topk_wave_dry_run_does_not_load_model_or_write(monkeypatch, tmp_path) -> None:
    import scripts.bonafide.topk_runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "_load_model_and_tokenizer",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("dry-run loaded the model")
        ),
    )
    records = run_topk_wave(
        config=_cpu_config(),
        manifest=_runner_manifest(),
        wave_id="parity-01",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
        dry_run=True,
        verify_source=False,
        _code_revision=_code_revision(),
        _runtime_environment=_runtime_environment(),
    )

    assert len(records) == 1
    assert records[0]["status"] == "planned"
    assert TOPK_TRACE_FAMILY_ID in records[0]["artifact_path"]
    assert not (tmp_path / "summary.jsonl").exists()


def test_topk_wave_dry_run_accepts_cuda_memory_instrumentation(
    monkeypatch, tmp_path
) -> None:
    import scripts.bonafide.topk_runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "_load_model_and_tokenizer",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("dry-run loaded the model")
        ),
    )
    config = _cpu_config()
    config["instrumentation"] = {"cuda_memory_telemetry": True}

    records = run_topk_wave(
        config=config,
        manifest=_runner_manifest(),
        wave_id="parity-01",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
        dry_run=True,
        verify_source=False,
        _code_revision=_code_revision(),
        _runtime_environment=_runtime_environment(),
    )

    assert [record["status"] for record in records] == ["planned"]
    assert not (tmp_path / "summary.jsonl").exists()


def test_topk_wave_dry_run_records_variable_candidate_bounds(
    monkeypatch, tmp_path
) -> None:
    import scripts.bonafide.topk_runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "_load_model_and_tokenizer",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("dry-run loaded the model")
        ),
    )
    manifest = _runner_manifest()
    manifest["phase"] = "c1_policy_resource"
    manifest["trace_family"] = {
        "trace_family_id": "bonafide.model-top5-plus-observed.c1-smoke.v1",
        "candidate_policy_id": "model_top5_plus_observed",
        "candidate_policy_version": "1",
        "candidate_count_min": 5,
        "candidate_count_max": 6,
        "candidate_count_rule": "5_if_observed_in_model_top5_else_6",
        "joint_objective_id": "raw_logit_sum",
        "joint_objective_version": "1",
    }

    records = run_topk_wave(
        config=_cpu_config(),
        manifest=manifest,
        wave_id="parity-01",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
        dry_run=True,
        verify_source=False,
        _code_revision=_code_revision(),
        _runtime_environment=_runtime_environment(),
    )

    assert records[0]["candidate_count_min"] == 5
    assert records[0]["candidate_count_max"] == 6
    assert "candidate_count" not in records[0]


def test_topk_wave_rejects_implicit_logit_centering(tmp_path) -> None:
    config = _cpu_config()
    config["adag_config"]["center_logits"] = True

    with pytest.raises(ValueError, match="center_logits=false"):
        run_topk_wave(
            config=config,
            manifest=_runner_manifest(),
            wave_id="parity-01",
            artifact_root=tmp_path / "artifacts",
            summary_jsonl=tmp_path / "summary.jsonl",
            dry_run=True,
            verify_source=False,
            _code_revision=_code_revision(),
            _runtime_environment=_runtime_environment(),
        )


def test_topk_wave_verifies_items_against_hashed_width1_source(tmp_path) -> None:
    source_manifest = _single_item_manifest()
    source_manifest["waves"][0]["corpus_role"] = "dense_discovery"
    source_manifest["tokenizer"]["chat_template_sha256"] = "b" * 64
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    manifest = _manifest()
    manifest["source"]["width1_manifest_path"] = str(source_path)
    manifest["source"]["width1_manifest_sha256"] = sha256_file(source_path)

    records = run_topk_wave(
        config=_cpu_config(),
        manifest=manifest,
        wave_id="parity-01",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
        dry_run=True,
        _code_revision=_code_revision(),
        _runtime_environment=_runtime_environment(),
    )

    assert [record["status"] for record in records] == ["planned"]
    manifest["waves"][0]["items"][0]["example"]["prompt"] = "drifted"
    with pytest.raises(ValueError, match="drifted from width-one source"):
        run_topk_wave(
            config=_cpu_config(),
            manifest=manifest,
            wave_id="parity-01",
            artifact_root=tmp_path / "artifacts",
            summary_jsonl=tmp_path / "summary.jsonl",
            dry_run=True,
            _code_revision=_code_revision(),
            _runtime_environment=_runtime_environment(),
        )


def test_topk_wave_saves_and_checksum_validates_resume(monkeypatch, tmp_path) -> None:
    import scripts.bonafide.topk_runner as runner_module

    manifest = _runner_manifest()
    item = manifest["waves"][0]["items"][0]
    trace = _topk_trace()
    trace.circuit_data.trace_metadata["response_token_count"] = item[
        "response_token_count"
    ]
    trace.circuit_data.trace_metadata["chat_template_sha256"] = "b" * 64
    calls = []

    def fake_trace(*args, **kwargs):
        calls.append((args, kwargs))
        return deepcopy(trace)

    monkeypatch.setattr(runner_module, "trace_teacher_forced_candidates", fake_trace)
    model = SimpleNamespace(
        config=SimpleNamespace(to_dict=lambda: {"model_type": "fake"})
    )
    common = {
        "config": _cpu_config(),
        "manifest": manifest,
        "wave_id": "parity-01",
        "artifact_root": tmp_path / "artifacts",
        "summary_jsonl": tmp_path / "summary.jsonl",
        "verify_source": False,
        "_model_bundle": (model, object()),
        "_code_revision": _code_revision(),
        "_runtime_environment": _runtime_environment(),
    }

    completed = run_topk_wave(**common)
    resumed = run_topk_wave(**common)

    assert [record["status"] for record in completed] == ["complete"]
    assert [record["status"] for record in resumed] == ["skipped_complete"]
    assert len(calls) == 1
    summary = [
        json.loads(line)
        for line in (tmp_path / "summary.jsonl").read_text().splitlines()
    ]
    assert [record["status"] for record in summary] == [
        "complete",
        "skipped_complete",
    ]
