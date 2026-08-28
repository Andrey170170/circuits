from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.bonafide.cuda_allocator_policy import (
    ALLOCATOR_ENVIRONMENT_VARIABLE,
    apply_cuda_allocator_policy_to_environment,
    bind_cuda_allocator_runtime_receipt,
    declared_cuda_allocator_policy,
    validate_cuda_allocator_environment,
)

ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "scripts/bonafide/configs"
SOURCE_CONFIG = (
    CONFIG_ROOT
    / "qwen3_4b_thinking_contribution_source_leaf_profiling_flash_sdpa_causal_v1.json"
)
COMPACT_DEFAULT_CONFIG = (
    CONFIG_ROOT
    / "qwen3_4b_thinking_post_selection_state_storage_qualification_compact_cpu_v1.json"
)
COMPACT_EXPANDABLE_CONFIG = (
    CONFIG_ROOT
    / "qwen3_4b_thinking_post_selection_state_storage_compact_cpu_allocator_qualification_expandable_segments_v1.json"
)


def _config(policy: str) -> dict:
    path = CONFIG_ROOT / f"qwen3_4b_thinking_allocator_qualification_{policy}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_allocator_configs_clone_source_leaf_except_for_explicit_policy() -> None:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    for policy in ("default_v1", "expandable_segments_v1"):
        config = _config(policy)
        assert config.pop("cuda_allocator_policy") == policy
        assert config == source


def test_compact_allocator_configs_differ_only_by_explicit_policy() -> None:
    default = json.loads(COMPACT_DEFAULT_CONFIG.read_text(encoding="utf-8"))
    expandable = json.loads(COMPACT_EXPANDABLE_CONFIG.read_text(encoding="utf-8"))

    assert default.pop("cuda_allocator_policy") == "default_v1"
    assert expandable.pop("cuda_allocator_policy") == "expandable_segments_v1"
    assert expandable == default


def test_policy_environment_and_runtime_receipt_are_exact() -> None:
    environment = {ALLOCATOR_ENVIRONMENT_VARIABLE: "inherited:bad"}
    apply_cuda_allocator_policy_to_environment("default_v1", environment)
    assert ALLOCATOR_ENVIRONMENT_VARIABLE not in environment
    default = bind_cuda_allocator_runtime_receipt(
        {"cuda_allocator_policy": "default_v1"},
        {"python": "3.12.12"},
        environ=environment,
        allocator_backend=lambda: "native",
    )
    assert default["cuda_allocator_policy"] == {
        "intended_policy_id": "default_v1",
        "observed_environment": {
            "name": ALLOCATOR_ENVIRONMENT_VARIABLE,
            "value": None,
            "is_set": False,
        },
        "observed_allocator_backend": "native",
    }

    apply_cuda_allocator_policy_to_environment("expandable_segments_v1", environment)
    assert environment[ALLOCATOR_ENVIRONMENT_VARIABLE] == "expandable_segments:True"
    expandable = bind_cuda_allocator_runtime_receipt(
        {"cuda_allocator_policy": "expandable_segments_v1"},
        {"python": "3.12.12"},
        environ=environment,
        allocator_backend=lambda: "native",
    )
    assert expandable["cuda_allocator_policy"] == {
        "intended_policy_id": "expandable_segments_v1",
        "observed_environment": {
            "name": ALLOCATOR_ENVIRONMENT_VARIABLE,
            "value": "expandable_segments:True",
            "is_set": True,
        },
        "observed_allocator_backend": "native",
    }


def test_policy_fails_closed_on_unknown_or_mismatched_environment() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        declared_cuda_allocator_policy({"cuda_allocator_policy": "unknown"})
    with pytest.raises(RuntimeError, match="disagrees"):
        validate_cuda_allocator_environment(
            {"cuda_allocator_policy": "expandable_segments_v1"},
            environ={},
        )
    with pytest.raises(RuntimeError, match="conflicting CUDA allocator receipt"):
        bind_cuda_allocator_runtime_receipt(
            {"cuda_allocator_policy": "default_v1"},
            {"cuda_allocator_policy": {"intended_policy_id": "mismatched"}},
            environ={},
            allocator_backend=lambda: "native",
        )
    with pytest.raises(RuntimeError, match="conflicting CUDA allocator receipt"):
        bind_cuda_allocator_runtime_receipt(
            {"cuda_allocator_policy": "default_v1"},
            {"cuda_allocator_policy": None},
            environ={},
            allocator_backend=lambda: "native",
        )


def test_legacy_config_preserves_runtime_environment_without_receipt() -> None:
    runtime = {"python": "3.12.12"}
    assert bind_cuda_allocator_runtime_receipt({}, runtime) == runtime
