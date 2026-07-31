from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from scripts.bonafide.topk_manifest import validate_topk_manifest
from scripts.bonafide.runner import validate_runtime_topk_trace_against_item
from tests.test_bonafide_benchmark import _single_item_manifest
from tests.test_teacher_forced_trace import _topk_trace


def _manifest() -> dict:
    source_manifest = _single_item_manifest()
    item = deepcopy(source_manifest["waves"][0]["items"][0])
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
            "width1_manifest_path": "scripts/bonafide/manifests/source.json",
            "width1_manifest_sha256": "a" * 64,
            "model_id": "fake/model",
            "model_revision": "exact-revision",
            "tokenizer_revision": "exact-revision",
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
    item["target_selection"].pop("sampling", None)

    with pytest.raises(ValueError, match="one response target"):
        validate_topk_manifest(manifest)


def test_topk_discovery_phases_reject_holdout() -> None:
    manifest = _manifest()
    manifest["waves"][0]["corpus_role"] = "broad_confirmatory_holdout"

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


def test_top5_plus_observed_accepts_frozen_variable_width_contract() -> None:
    manifest = _manifest()
    manifest["trace_family"] = {
        "trace_family_id": "bonafide.model-top5-plus-observed.v1",
        "candidate_policy_id": "model_top5_plus_observed",
        "candidate_policy_version": "1",
        "candidate_count_min": 5,
        "candidate_count_max": 6,
        "candidate_count_rule": "5_if_observed_in_model_top5_else_6",
        "joint_objective_id": "raw_logit_sum",
        "joint_objective_version": "1",
    }

    validate_topk_manifest(manifest)


def test_top5_plus_observed_rejects_fixed_width_manifest() -> None:
    manifest = _manifest()
    manifest["trace_family"].update(
        {
            "candidate_policy_id": "model_top5_plus_observed",
            "candidate_count": 6,
        }
    )

    with pytest.raises(ValueError, match="candidate_count_min/max"):
        validate_topk_manifest(manifest)


def test_specified_token_manifest_requires_fixed_candidate_id() -> None:
    manifest = _manifest()
    manifest["trace_family"].update(
        {
            "candidate_policy_id": "specified_token",
            "candidate_count": 1,
        }
    )

    with pytest.raises(ValueError, match="specified_candidate_token_id"):
        validate_topk_manifest(manifest)
    manifest["waves"][0]["items"][0]["specified_candidate_token_id"] = 123
    validate_topk_manifest(manifest)


def test_runtime_topk_validation_binds_source_target_and_trace_family() -> None:
    manifest = _manifest()
    item = manifest["waves"][0]["items"][0]
    item["target_selection"]["response_token_positions"] = [0]
    item["target_selection"]["final_target_token_id"] = 40
    item["target_selection"].pop("sampling", None)
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
    item["target_selection"].pop("sampling", None)
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


def test_runtime_topk_validation_accepts_realized_width_five() -> None:
    manifest = _manifest()
    item = manifest["waves"][0]["items"][0]
    item["target_selection"]["response_token_positions"] = [0]
    item["target_selection"]["final_target_token_id"] = 40
    item["target_selection"].pop("sampling", None)
    trace = _topk_trace()
    selection = replace(
        trace.candidate_selection,
        policy_id="model_top5_plus_observed",
        ordering_rule=(
            "observed_first_then_model_top5_descending_logit_then_ascending_token_id"
        ),
    )
    trace = replace(trace, candidate_selection=selection)
    trace.circuit_data.trace_metadata["candidate_trace_contract"] = (
        trace.contract_dict()
    )
    trace.circuit_data.trace_metadata["response_token_count"] = item[
        "response_token_count"
    ]
    trace_family = {
        "trace_family_id": trace.trace_family_id,
        "candidate_policy_id": "model_top5_plus_observed",
        "candidate_policy_version": "1",
        "candidate_count_min": 5,
        "candidate_count_max": 6,
        "candidate_count_rule": "5_if_observed_in_model_top5_else_6",
        "joint_objective_id": "raw_logit_sum",
        "joint_objective_version": "1",
    }

    validate_runtime_topk_trace_against_item(trace, item, trace_family)
