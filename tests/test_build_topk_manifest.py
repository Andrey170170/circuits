from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.bonafide.build_topk_manifest import build_topk_manifest, save_manifest
from tests.test_bonafide_benchmark import _single_item_manifest


def _source_manifest() -> dict:
    source = _single_item_manifest()
    source["tokenizer"]["chat_template_sha256"] = "b" * 64
    source["waves"][0]["corpus_role"] = "dense_discovery"
    return source


def test_builder_copies_exact_source_item_and_provenance(tmp_path: Path) -> None:
    source = _source_manifest()
    item = source["waves"][0]["items"][0]

    manifest = build_topk_manifest(
        source,
        source_manifest_path=tmp_path / "source.json",
        source_manifest_sha256="a" * 64,
        source_artifact_ids=[item["artifact_id"]],
        phase="observed_k1_parity",
        trace_family_id="bonafide.observed-k1-smoke.v1",
        candidate_policy_id="observed_token",
        joint_objective_id="raw_logit_sum",
        wave_id="parity-smoke",
    )

    assert manifest["waves"][0]["items"][0] == item
    assert manifest["source"]["model_id"] == "fake/model"
    assert manifest["source"]["tokenizer_revision"] == "exact-revision"


def test_builder_rejects_mixed_corpus_roles(tmp_path: Path) -> None:
    source = _source_manifest()
    second = deepcopy(source["waves"][0])
    second["corpus_role"] = "broad_discovery"
    second["items"][0]["artifact_id"] = "source-trace-2"
    source["waves"].append(second)

    with pytest.raises(ValueError, match="cannot mix"):
        build_topk_manifest(
            source,
            source_manifest_path=tmp_path / "source.json",
            source_manifest_sha256="a" * 64,
            source_artifact_ids=["source-trace-1", "source-trace-2"],
            phase="observed_k1_parity",
            trace_family_id="bonafide.observed-k1-smoke.v1",
            candidate_policy_id="observed_token",
            joint_objective_id="raw_logit_sum",
            wave_id="parity-smoke",
        )


def test_builder_records_top5_plus_observed_variable_width(tmp_path: Path) -> None:
    source = _source_manifest()
    item = source["waves"][0]["items"][0]

    manifest = build_topk_manifest(
        source,
        source_manifest_path=tmp_path / "source.json",
        source_manifest_sha256="a" * 64,
        source_artifact_ids=[item["artifact_id"]],
        phase="c1_policy_resource",
        trace_family_id="bonafide.model-top5-plus-observed.c1-smoke.v1",
        candidate_policy_id="model_top5_plus_observed",
        joint_objective_id="raw_logit_sum",
        wave_id="c1-smoke",
    )

    family = manifest["trace_family"]
    assert "candidate_count" not in family
    assert family["candidate_count_min"] == 5
    assert family["candidate_count_max"] == 6
    assert family["candidate_count_rule"] == ("5_if_observed_in_model_top5_else_6")


def test_save_manifest_is_atomic_and_does_not_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "manifests" / "topk.json"
    value = {"hello": "world"}

    save_manifest(output, value)

    assert output.is_file()
    assert not list(output.parent.glob(".topk.json.tmp-*"))
    with pytest.raises(FileExistsError):
        save_manifest(output, value)
