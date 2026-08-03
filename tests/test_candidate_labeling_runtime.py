from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import circuits.analysis.bonafide.candidate_labeling_runtime as runtime_module
from circuits.analysis.bonafide.candidate_labeling_renderer import (
    HELDOUT_FORBIDDEN_INPUTS,
    STATUS_ENUM,
    TYPED_OUTPUT_FIELDS,
    LoadedCandidateLabelingRenderer,
)
from circuits.analysis.bonafide.candidate_labeling_runtime import (
    GENERIC_STAGE_IDS,
    build_candidate_labeling_execution_cohort,
    load_candidate_labeling_execution_cohort,
    load_candidate_labeling_runtime_config,
    prepare_candidate_labeling_execution_cohort,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256

CONFIG_ROOT = Path("scripts/bonafide/configs/labeling")


def _output_schema(include_candidate: bool) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(TYPED_OUTPUT_FIELDS),
        "properties": {
            "input_localization_hypothesis": {"type": "string", "minLength": 1},
            "exploratory_candidate_description": (
                {"type": "string", "minLength": 1}
                if include_candidate
                else {"const": "not_available"}
            ),
            "background_or_confound": {"type": "string", "minLength": 1},
            "limitations": {"type": "string", "minLength": 1},
            "status": {"type": "string", "enum": list(STATUS_ENUM)},
        },
    }


def _renderer(tmp_path: Path) -> LoadedCandidateLabelingRenderer:
    root = tmp_path / "renderer"
    root.mkdir()
    manifest = {
        "schema_version": "adag.bonafide.candidate-labeling-renderer.v1",
        "manifest_sha256": "a" * 64,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    prompts = []
    for arm_id, include_candidate in (
        ("arm_1_width_only", False),
        ("arm_2_width_plus_candidate", True),
    ):
        for index in range(12):
            payload = {
                "messages": [
                    {"role": "system", "content": "bounded generation evidence"},
                    {"role": "user", "content": f"arm {arm_id} cluster {index}"},
                ],
                "expected_output_json_schema": _output_schema(include_candidate),
            }
            prompt = {
                "logical_prompt_id": f"{arm_id}:w64:{index:02d}",
                "arm_id": arm_id,
                "anchor_index": index,
                "cluster_id": index,
                "candidate_evidence_included": include_candidate,
                "message_payload": payload,
                "message_payload_sha256": canonical_sha256(payload),
                "width_evidence_sha256": f"{index:064x}",
                "rendered_candidate_witness_sha256_in_order": (
                    [f"{index + offset + 1:064x}" for offset in range(8)]
                    if include_candidate
                    else None
                ),
            }
            prompt["prompt_sha256"] = canonical_sha256(prompt)
            prompts.append(prompt)
    return LoadedCandidateLabelingRenderer(
        root=root,
        manifest=manifest,
        witness_selection={},
        generation_prompts=tuple(prompts),
        stage_plan={},
    )


def _copy_config_and_price(tmp_path: Path, config_name: str) -> Path:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    shutil.copyfile(CONFIG_ROOT / config_name, config_root / config_name)
    shutil.copyfile(
        CONFIG_ROOT / "prices-2026-07-30.json",
        config_root / "prices-2026-07-30.json",
    )
    return config_root / config_name


def _patch_provenance(
    monkeypatch: pytest.MonkeyPatch, renderer: LoadedCandidateLabelingRenderer
) -> dict:
    revision = {
        "repo_root": "/recorded/worktree",
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
        "tracked_worktree_clean": True,
        "tracked_status_sha256": "3" * 64,
        "files": [],
    }
    monkeypatch.setattr(
        runtime_module,
        "_load_candidate_labeling_renderer_portably",
        lambda *args, **kwargs: renderer,
    )
    monkeypatch.setattr(
        runtime_module,
        "collect_candidate_labeling_runtime_revision",
        lambda *args, **kwargs: revision,
    )
    monkeypatch.setattr(
        runtime_module, "_validate_runtime_revision_portably", lambda value: None
    )
    return revision


def test_openai_config_is_iteration_default_with_luna_and_terra() -> None:
    config = load_candidate_labeling_runtime_config(
        CONFIG_ROOT / "openai-c2-evidence-v1.json"
    )

    assert config.evaluation_phase == "iteration"
    assert config.deferred is False
    assert config.semantic_samples_per_prompt == 5
    assert config.semantic_generator.provider == "openai"
    assert config.semantic_generator.model == "gpt-5.6-luna"
    assert config.semantic_rewriter.model == "gpt-5.6-terra"
    assert config.conservative_control.model == "gpt-5.6-terra"
    assert {
        config.semantic_generator.transport,
        config.semantic_rewriter.transport,
        config.conservative_control.transport,
    } == {"native_batch"}
    assert config.price_snapshot == "prices-2026-07-30.json"


def test_anthropic_config_is_valid_all_anthropic_deferred_comparison() -> None:
    config = load_candidate_labeling_runtime_config(
        CONFIG_ROOT / "anthropic-c2-evidence-v1.json"
    )

    assert config.evaluation_phase == "deferred_comparison"
    assert config.deferred is True
    assert config.semantic_generator.provider == "anthropic"
    assert config.semantic_generator.model == "claude-haiku-4-5-20251001"
    assert config.semantic_rewriter.provider == "anthropic"
    assert config.semantic_rewriter.model == "claude-opus-5"
    assert config.conservative_control.provider == "anthropic"
    assert config.conservative_control.model == "claude-opus-5"


def test_build_emits_120_semantic_and_24_control_requests_only(
    tmp_path: Path,
) -> None:
    renderer = _renderer(tmp_path)
    config = load_candidate_labeling_runtime_config(
        CONFIG_ROOT / "openai-c2-evidence-v1.json"
    )
    requests, dependencies, derived = build_candidate_labeling_execution_cohort(
        renderer, config
    )

    assert len(requests) == 144
    assert len(dependencies) == 24
    assert [request["stage_id"] for request in requests].count(
        "semantic_generation"
    ) == 120
    assert [request["stage_id"] for request in requests].count(
        "conservative_control"
    ) == 24
    assert {dependency["stage_id"] for dependency in dependencies} == {
        "semantic_rewrite"
    }
    assert all(
        dependency["request_constructed"] is False for dependency in dependencies
    )
    assert all(
        len(dependency["required_semantic_request_ids"]) == 5
        for dependency in dependencies
    )
    assert all("messages" not in dependency for dependency in dependencies)
    assert tuple(GENERIC_STAGE_IDS) == (
        "semantic_generation",
        "semantic_rewrite",
        "conservative_control",
    )
    assert all(
        provider_name not in stage_id
        for stage_id in GENERIC_STAGE_IDS
        for provider_name in ("openai", "anthropic", "luna", "terra", "opus", "haiku")
    )
    assert len(derived["arm_bindings"]) == 2


def test_requests_preserve_exact_prompts_schema_and_generation_fence(
    tmp_path: Path,
) -> None:
    renderer = _renderer(tmp_path)
    config = load_candidate_labeling_runtime_config(
        CONFIG_ROOT / "openai-c2-evidence-v1.json"
    )
    requests, dependencies, derived = build_candidate_labeling_execution_cohort(
        renderer, config
    )
    source_by_id = {
        prompt["logical_prompt_id"]: prompt for prompt in renderer.generation_prompts
    }
    arm_hashes = {arm["arm_id"]: arm["arm_sha256"] for arm in derived["arm_bindings"]}

    for request in requests:
        prompt = source_by_id[request["logical_prompt_id"]]
        assert request["source_prompt_sha256"] == prompt["prompt_sha256"]
        assert (
            request["source_message_payload_sha256"] == prompt["message_payload_sha256"]
        )
        assert request["arm_sha256"] == arm_hashes[prompt["arm_id"]]
        assert request["messages"] == prompt["message_payload"]["messages"]
        assert (
            request["expected_output_json_schema"]
            == prompt["message_payload"]["expected_output_json_schema"]
        )
        assert request["typed_output_fields"] == list(TYPED_OUTPUT_FIELDS)
        assert request["status_enum"] == list(STATUS_ENUM)
        assert request["family_partition"] == "generation"
        assert request["generation_only"] is True
        assert request["selection_audit_visible"] is False
        assert request["forbidden_input_fields"] == list(HELDOUT_FORBIDDEN_INPUTS)
        serialized_messages = json.dumps(request["messages"], sort_keys=True)
        assert not any(
            field in serialized_messages for field in HELDOUT_FORBIDDEN_INPUTS
        )
    assert all(
        dependency["forbidden_input_fields"] == list(HELDOUT_FORBIDDEN_INPUTS)
        for dependency in dependencies
    )


def test_preparation_is_immutable_and_loader_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = _renderer(tmp_path)
    config_path = _copy_config_and_price(tmp_path, "openai-c2-evidence-v1.json")
    revision = _patch_provenance(monkeypatch, renderer)
    output = tmp_path / "cohort"

    manifest = prepare_candidate_labeling_execution_cohort(
        renderer_root=renderer.root,
        config_path=config_path,
        output_root=output,
    )
    loaded = load_candidate_labeling_execution_cohort(output)

    assert manifest["calls_made"] is False
    assert manifest["provider_model_endpoints_resolved"] is False
    assert manifest["initial_request_count"] == 144
    assert manifest["code_revision"] == revision
    assert [row["constructed_request_count"] for row in manifest["stage_counts"]] == [
        120,
        0,
        24,
    ]
    assert loaded.manifest == manifest
    assert len(loaded.initial_requests) == 144
    with pytest.raises(FileExistsError, match="refusing to replace"):
        prepare_candidate_labeling_execution_cohort(
            renderer_root=renderer.root,
            config_path=config_path,
            output_root=output,
        )

    requests_path = output / "initial-requests.jsonl"
    requests_path.write_text(
        requests_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="file drift"):
        load_candidate_labeling_execution_cohort(output)


@pytest.mark.parametrize("source", ["renderer", "recipe"])
def test_loader_fails_on_renderer_or_recipe_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    renderer = _renderer(tmp_path)
    config_path = _copy_config_and_price(tmp_path, "openai-c2-evidence-v1.json")
    _patch_provenance(monkeypatch, renderer)
    output = tmp_path / "cohort"
    prepare_candidate_labeling_execution_cohort(
        renderer_root=renderer.root,
        config_path=config_path,
        output_root=output,
    )

    if source == "renderer":
        (renderer.root / "manifest.json").write_text("{}\n", encoding="utf-8")
        match = "renderer drift"
    else:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        raw["description"] += " drift"
        config_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
        match = "runtime config drift"
    with pytest.raises(ValueError, match=match):
        load_candidate_labeling_execution_cohort(output)


def test_prepare_cannot_bypass_deep_portable_renderer_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = _renderer(tmp_path)
    config_path = _copy_config_and_price(tmp_path, "openai-c2-evidence-v1.json")
    revision = _patch_provenance(monkeypatch, renderer)
    monkeypatch.setattr(
        runtime_module,
        "collect_candidate_labeling_runtime_revision",
        lambda *args, **kwargs: revision,
    )
    monkeypatch.setattr(
        runtime_module,
        "load_candidate_labeling_renderer",
        lambda *args, **kwargs: renderer,
    )
    monkeypatch.setattr(
        runtime_module,
        "_load_candidate_labeling_renderer_portably",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("deep renderer source rejection")
        ),
    )

    with pytest.raises(ValueError, match="deep renderer source rejection"):
        prepare_candidate_labeling_execution_cohort(
            renderer_root=renderer.root,
            config_path=config_path,
            output_root=tmp_path / "cohort",
        )


def test_archival_loader_rejects_coherently_rehashed_malformed_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = _renderer(tmp_path)
    config_path = _copy_config_and_price(tmp_path, "openai-c2-evidence-v1.json")
    _patch_provenance(monkeypatch, renderer)
    output = tmp_path / "cohort"
    prepare_candidate_labeling_execution_cohort(
        renderer_root=renderer.root,
        config_path=config_path,
        output_root=output,
    )

    request_path = output / "initial-requests.jsonl"
    rows = [json.loads(line) for line in request_path.read_text().splitlines()]
    rows[0]["sample_index"] = 1
    unhashed = dict(rows[0])
    unhashed.pop("request_sha256")
    rows[0]["request_sha256"] = canonical_sha256(unhashed)
    request_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        if item["path"] == "initial-requests.jsonl":
            item["sha256"] = file_sha256(request_path)
            item["size_bytes"] = request_path.stat().st_size
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match="prompt, arm|sample-index|prompt/sample graph"
    ):
        load_candidate_labeling_execution_cohort(output, verify_sources=False)
