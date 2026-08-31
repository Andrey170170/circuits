from __future__ import annotations

import json
from pathlib import Path

import pytest
from circuits.analysis.bonafide import (
    process_witness_near_10k_qualification_v1 as near10k,
)
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.process_witness_resource_calibration_v1 import (
    EXECUTION_SOURCE_PATHS as HISTORICAL_EXECUTION_SOURCE_PATHS,
)


class FakeThinkingTokenizer:
    chat_template = "fake-thinking-template"


def _candidate(
    target_id: str,
    context: int,
    *,
    response_id: str = "response-a",
    policy: str = "balanced",
    budget: int = 40_000,
) -> dict:
    return {
        "target_id": target_id,
        "response_id": response_id,
        "psu_id": f"psu-{target_id}",
        "unit_id": f"unit-{target_id}",
        "token_index": context - 3,
        "rendered_total_context_token_count": context,
        "balanced_40k_first_owner_mechanism": "uncertainty_missing",
        "policy_memberships": [{"policy": policy, "budget": budget}],
    }


def test_selects_nearest_balanced_40k_exact_response_strictly_above() -> None:
    candidates = [
        _candidate("below", 10_000),
        _candidate("inexact-closer", 10_001, response_id="response-inexact"),
        _candidate("wrong-policy", 10_002, policy="uncertainty_weighted"),
        _candidate("chosen", 10_006),
        _candidate("farther", 10_007),
    ]

    selected = near10k.select_nearest_strictly_above(
        candidates,
        exact_tokenizations={"response-a": ([1, 2], list(range(10_010)))},
    )

    assert selected["target_id"] == "chosen"
    assert selected["rendered_total_context_token_count"] == 10_006


def test_selection_tie_breaks_by_target_id() -> None:
    selected = near10k.select_nearest_strictly_above(
        [_candidate("target-b", 10_006), _candidate("target-a", 10_006)],
        exact_tokenizations={"response-a": ([1, 2], list(range(10_010)))},
    )

    assert selected["target_id"] == "target-a"


def _patch_synthetic_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    source = tmp_path / "sampling-v2"
    source.mkdir()
    prefix_ids = [1, 2]
    response_ids = list(range(10))
    selected = _candidate("target-near", 12)
    selected["token_index"] = 9
    state = {
        "source": source.resolve(),
        "source_manifest": {
            "manifest_sha256": near10k.EXPECTED_SAMPLING_MANIFEST_SHA256,
            "inventory_sha256": near10k.EXPECTED_SAMPLING_INVENTORY_SHA256,
        },
        "documents": {
            "response-a": {
                "response_id": "response-a",
                "task_context": {"prompt": "prompt-a"},
                "text": "response-a-text",
            }
        },
        "exact": {"response-a": (prefix_ids, response_ids)},
        "excluded": [],
        "non_generation": [],
        "selected": selected,
    }
    monkeypatch.setattr(near10k, "CONTEXT_THRESHOLD_EXCLUSIVE", 10)
    monkeypatch.setattr(near10k, "EXPECTED_SELECTED_CONTEXT", 12)
    monkeypatch.setattr(near10k, "EXPECTED_SELECTED_TARGET_ID", "target-near")
    monkeypatch.setattr(near10k, "_load_source_state", lambda **_kwargs: state)
    monkeypatch.setattr(
        near10k,
        "_execution_source_revision",
        lambda _root: {
            "repo_root": str(tmp_path),
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "clean_scope": "execution_source_paths",
            "binding_scope": "near_10k_selection_validation_and_trace_execution",
            "files": [],
        },
    )
    return state


def test_full_build_and_strict_load(tmp_path: Path, monkeypatch) -> None:
    _patch_synthetic_source(monkeypatch, tmp_path)
    destination = tmp_path / "near-10k-v1"

    manifest = near10k.build_near_10k_qualification_v1(
        sampling_v2_root=tmp_path / "sampling-v2",
        destination=destination,
        tokenizer=FakeThinkingTokenizer(),
        system_prompt="system",
    )
    loaded = near10k.load_frozen_near_10k_qualification_v1(
        destination,
        tokenizer=FakeThinkingTokenizer(),
        system_prompt="system",
    )

    assert loaded["manifest"] == manifest
    assert manifest["selected_target"]["target_id"] == "target-near"
    assert manifest["selected_target"]["rendered_total_context_token_count"] == 12
    assert manifest["selected_target"]["prompt_utf8_sha256"]
    assert manifest["selected_target"]["response_utf8_sha256"]
    assert manifest["selected_target"]["target_identity_sha256"]
    trace = json.loads((destination / "trace-manifest.json").read_text())
    assert trace["phase"] == "process_witness_resource_calibration_v1"
    assert trace["waves"][0]["wave_id"] == "context-gt-10000"
    assert len(trace["waves"][0]["items"]) == 1
    assert (destination.stat().st_mode & 0o777) == 0o555
    assert all((path.stat().st_mode & 0o777) == 0o444 for path in destination.iterdir())


def test_loader_rejects_rerun_selection_drift(tmp_path: Path, monkeypatch) -> None:
    state = _patch_synthetic_source(monkeypatch, tmp_path)
    destination = tmp_path / "near-10k-v1"
    near10k.build_near_10k_qualification_v1(
        sampling_v2_root=tmp_path / "sampling-v2",
        destination=destination,
        tokenizer=FakeThinkingTokenizer(),
        system_prompt="system",
    )
    state["selected"] = {**state["selected"], "target_id": "target-source-drift"}

    with pytest.raises(ValueError, match="selected target drift"):
        near10k.load_frozen_near_10k_qualification_v1(
            destination,
            tokenizer=FakeThinkingTokenizer(),
            system_prompt="system",
        )


def _rewrite_readonly_json(path: Path, value: dict) -> None:
    path.chmod(0o644)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)


def test_loader_rejects_rehashed_extra_inventory_field(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_source(monkeypatch, tmp_path)
    destination = tmp_path / "near-10k-v1"
    near10k.build_near_10k_qualification_v1(
        sampling_v2_root=tmp_path / "sampling-v2",
        destination=destination,
        tokenizer=FakeThinkingTokenizer(),
        system_prompt="system",
    )
    inventory_path = destination / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["extra"] = "self-consistent-but-noncanonical"
    inventory_core = dict(inventory)
    inventory_core.pop("inventory_sha256")
    inventory["inventory_sha256"] = canonical_sha256(inventory_core)
    _rewrite_readonly_json(inventory_path, inventory)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inventory_sha256"] = inventory["inventory_sha256"]
    manifest_core = dict(manifest)
    manifest_core.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest_core)
    _rewrite_readonly_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="canonical inventory drift"):
        near10k.load_frozen_near_10k_qualification_v1(
            destination,
            tokenizer=FakeThinkingTokenizer(),
            system_prompt="system",
        )


def test_loader_rejects_rehashed_extra_manifest_field(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_synthetic_source(monkeypatch, tmp_path)
    destination = tmp_path / "near-10k-v1"
    near10k.build_near_10k_qualification_v1(
        sampling_v2_root=tmp_path / "sampling-v2",
        destination=destination,
        tokenizer=FakeThinkingTokenizer(),
        system_prompt="system",
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extra"] = "self-consistent-but-noncanonical"
    manifest_core = dict(manifest)
    manifest_core.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest_core)
    _rewrite_readonly_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="canonical manifest drift"):
        near10k.load_frozen_near_10k_qualification_v1(
            destination,
            tokenizer=FakeThinkingTokenizer(),
            system_prompt="system",
        )


def test_historical_calibration_execution_source_scope_is_unchanged() -> None:
    assert (
        "circuits/analysis/bonafide/process_witness_near_10k_qualification_v1.py"
        not in HISTORICAL_EXECUTION_SOURCE_PATHS
    )
    assert (
        "scripts/bonafide/build_process_witness_near_10k_qualification_v1.py"
        not in HISTORICAL_EXECUTION_SOURCE_PATHS
    )


def test_run_config_clones_qualified_expandable_lane_except_artifact_root() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "scripts/bonafide/configs/qwen3_4b_thinking_post_selection_state_storage_compact_cpu_allocator_qualification_expandable_segments_v1.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expected = near10k._run_config()

    assert source.pop("artifact_root") != expected.pop("artifact_root")
    assert source == expected
