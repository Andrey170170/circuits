from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from circuits.tracing.trace import TOPK_TRACE_FAMILY_ID
from scripts.bonafide.t5_runner import (
    run_step0_t5_wave,
    validate_step0_t5_manifest,
)
from tests.test_teacher_forced_trace import _topk_trace
from tests.test_topk_runner import (
    _code_revision,
    _cpu_config,
    _runner_manifest,
    _runtime_environment,
)

SERIALIZATION_MODE = "historical_thinking_continuation"
TOKEN_IDENTITY_SCHEMA = "adag.teacher-forced-token-identity.v1"
TOKEN_HASH_ENCODING = "sha256_utf8_canonical_json_integer_array_v1"


def _t5_manifest() -> dict:
    manifest = _runner_manifest()
    manifest["phase"] = "step0_t5_smoke"
    manifest["trace_family"] = {
        "trace_family_id": TOPK_TRACE_FAMILY_ID,
        "candidate_policy_id": "model_top5",
        "candidate_policy_version": "1",
        "candidate_count": 5,
        "joint_objective_id": "raw_logit_sum",
        "joint_objective_version": "1",
    }
    system_prompt = "historical system prompt"
    manifest["teacher_forcing_contract"] = {
        "serialization_mode": SERIALIZATION_MODE,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "token_identity_schema_version": TOKEN_IDENTITY_SCHEMA,
        "hash_encoding": TOKEN_HASH_ENCODING,
    }
    item = manifest["waves"][0]["items"][0]
    item["example"]["system_prompt"] = system_prompt
    item["example"]["token_identity"] = {
        "assistant_prefix_ids_sha256": "c" * 64,
        "response_ids_sha256": "d" * 64,
        "response_token_count": item["response_token_count"],
    }
    return manifest


def _t5_trace():
    trace = _topk_trace()
    selection = replace(trace.candidate_selection, policy_id="model_top5")
    trace = replace(trace, candidate_selection=selection)
    trace.circuit_data.trace_metadata.update(
        {
            "candidate_trace_contract": trace.contract_dict(),
            "response_token_count": 8,
            "chat_template_sha256": "b" * 64,
            "system_prompt": "historical system prompt",
            "system_prompt_sha256": hashlib.sha256(
                b"historical system prompt"
            ).hexdigest(),
            "teacher_forced_serialization_mode": SERIALIZATION_MODE,
            "teacher_forced_token_identity": {
                "schema_version": TOKEN_IDENTITY_SCHEMA,
                "hash_encoding": TOKEN_HASH_ENCODING,
                "assistant_prefix_ids_sha256": "c" * 64,
                "response_ids_sha256": "d" * 64,
            },
        }
    )
    return trace


def test_step0_t5_manifest_accepts_only_strict_upstream_semantics() -> None:
    manifest = _t5_manifest()
    validate_step0_t5_manifest(manifest)

    manifest["trace_family"]["candidate_policy_id"] = "observed_plus_top4_alternatives"
    with pytest.raises(ValueError, match="strict upstream T5 semantics"):
        validate_step0_t5_manifest(manifest)


def test_step0_t5_runner_passes_and_validates_historical_contract(
    monkeypatch, tmp_path
) -> None:
    import scripts.bonafide.topk_runner as runner_module

    calls = []

    def fake_trace(*args, **kwargs):
        calls.append((args, kwargs))
        return deepcopy(_t5_trace())

    monkeypatch.setattr(runner_module, "trace_teacher_forced_candidates", fake_trace)
    model = SimpleNamespace(
        config=SimpleNamespace(to_dict=lambda: {"model_type": "fake"})
    )
    records = run_step0_t5_wave(
        config=_cpu_config(),
        manifest=_t5_manifest(),
        wave_id="parity-01",
        artifact_root=tmp_path / "artifacts",
        summary_jsonl=tmp_path / "summary.jsonl",
        verify_source=False,
        _model_bundle=(model, object()),
        _code_revision=_code_revision(),
        _runtime_environment=_runtime_environment(),
    )

    assert [record["status"] for record in records] == ["complete"]
    assert calls[0][1]["candidate_policy_id"] == "model_top5"
    assert calls[0][1]["candidate_count"] == 5
    assert calls[0][1]["joint_objective_id"] == "raw_logit_sum"
    assert calls[0][1]["system_prompt"] == "historical system prompt"
    assert calls[0][1]["serialization_mode"] == SERIALIZATION_MODE


def test_step0_t5_runner_rejects_live_token_identity_drift(
    monkeypatch, tmp_path
) -> None:
    import scripts.bonafide.topk_runner as runner_module

    drifted = _t5_trace()
    drifted.circuit_data.trace_metadata["teacher_forced_token_identity"][
        "response_ids_sha256"
    ] = "e" * 64
    monkeypatch.setattr(
        runner_module,
        "trace_teacher_forced_candidates",
        lambda *_args, **_kwargs: deepcopy(drifted),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(to_dict=lambda: {"model_type": "fake"})
    )

    with pytest.raises(ValueError, match="response_ids_sha256"):
        run_step0_t5_wave(
            config=_cpu_config(),
            manifest=_t5_manifest(),
            wave_id="parity-01",
            artifact_root=tmp_path / "artifacts",
            summary_jsonl=tmp_path / "summary.jsonl",
            verify_source=False,
            _model_bundle=(model, object()),
            _code_revision=_code_revision(),
            _runtime_environment=_runtime_environment(),
        )
