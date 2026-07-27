from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.bonafide.topk_manifest import validate_topk_manifest
from scripts.bonafide.runner import validate_runtime_topk_trace_against_item
from tests.test_bonafide_benchmark import _single_item_manifest
from tests.test_teacher_forced_trace import _topk_trace


def _manifest() -> dict:
    source_manifest = _single_item_manifest()
    item = deepcopy(source_manifest["waves"][0]["items"][0])
    item["target_selection"]["response_token_positions"] = [
        item["target_selection"]["response_token_positions"][0]
    ]
    item["target_selection"]["width"] = 1
    item["target_selection"].pop("sampling", None)
    item["objective"] = {
        "name": "single_selected_logit",
        "benchmark_only_multi_target": False,
    }
    return {
        "schema_version": "bonafide-topk-trace-manifest/v1",
        "phase": "observed_k1_parity",
        "trace_family": {
            "trace_family_id": "bonafide.observed-k1-parity.v1",
            "candidate_policy_id": "observed_token",
            "candidate_policy_version": "1",
            "candidate_count": 1,
            "joint_objective_id": "raw_logit_sum",
            "joint_objective_version": "1",
        },
        "source": {
            "width1_manifest_sha256": "a" * 64,
            "model_id": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": "revision",
            "tokenizer_revision": "revision",
            "chat_template_sha256": "b" * 64,
        },
        "waves": [
            {
                "wave_id": "parity-01",
                "corpus_role": "dense_discovery",
                "items": [item],
            }
        ],
    }


def test_topk_manifest_accepts_one_immutable_trace_family() -> None:
    validate_topk_manifest(_manifest())


def test_topk_manifest_rejects_policy_width_mismatch() -> None:
    manifest = _manifest()
    manifest["trace_family"]["candidate_count"] = 5

    with pytest.raises(ValueError, match="count disagrees"):
        validate_topk_manifest(manifest)


def test_topk_manifest_rejects_multi_position_work_item() -> None:
    manifest = _manifest()
    item = manifest["waves"][0]["items"][0]
    item["target_selection"]["response_token_positions"] = [0, 1]
    item["target_selection"]["width"] = 2

    with pytest.raises(ValueError, match="one response target"):
        validate_topk_manifest(manifest)


def test_topk_discovery_phases_reject_holdout() -> None:
    manifest = _manifest()
    manifest["waves"][0]["corpus_role"] = "confirmatory_holdout"

    with pytest.raises(ValueError, match="cannot include confirmatory"):
        validate_topk_manifest(manifest)


def test_model_top5_rejects_observed_contrastive_objective() -> None:
    manifest = _manifest()
    manifest["trace_family"].update(
        {
            "candidate_policy_id": "model_top5",
            "candidate_count": 5,
            "joint_objective_id": "observed_vs_alternatives",
        }
    )

    with pytest.raises(ValueError, match="only raw_logit_sum"):
        validate_topk_manifest(manifest)


def test_runtime_topk_validation_binds_source_target_and_trace_family() -> None:
    manifest = _manifest()
    item = manifest["waves"][0]["items"][0]
    item["target_selection"]["response_token_positions"] = [0]
    item["target_selection"]["final_target_token_id"] = 40
    trace = _topk_trace()
    trace.circuit_data.trace_metadata["response_token_count"] = item[
        "response_token_count"
    ]

    validate_runtime_topk_trace_against_item(
        trace,
        item,
        {
            **manifest["trace_family"],
            "trace_family_id": trace.trace_family_id,
            "candidate_policy_id": "observed_plus_top4_alternatives",
            "candidate_count": 5,
        },
    )


def test_runtime_topk_validation_rejects_policy_drift() -> None:
    manifest = _manifest()
    item = manifest["waves"][0]["items"][0]
    item["target_selection"]["response_token_positions"] = [0]
    item["target_selection"]["final_target_token_id"] = 40
    trace = _topk_trace()
    trace.circuit_data.trace_metadata["response_token_count"] = item[
        "response_token_count"
    ]
    trace_family = {
        **manifest["trace_family"],
        "trace_family_id": trace.trace_family_id,
        "candidate_policy_id": "model_top5",
        "candidate_count": 5,
    }

    with pytest.raises(ValueError, match="candidate_policy_id"):
        validate_runtime_topk_trace_against_item(trace, item, trace_family)
