from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from circuits.analysis.bonafide.candidate_labeling_comparison import (
    EXPECTED_W_ANCHORS,
    LoadedCandidateLabelingComparison,
)
from circuits.analysis.bonafide.candidate_labeling_renderer import (
    GENERATION_PROMPTS_FILE,
    MANIFEST_FILE,
    STAGE_PLAN_FILE,
    WITNESS_SELECTION_FILE,
    _validate_recorded_revision_portably,
    build_candidate_labeling_renderer,
    build_generation_prompts,
    load_candidate_labeling_renderer,
    run_candidate_labeling_renderer,
    select_generation_witnesses,
)
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_revision_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, str], dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "current-repository"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "candidate-renderer-test@example.invalid")
    _git(repo, "config", "user.name", "Candidate Renderer Test")
    bindings = {"runtime": "runtime.py", "protocol": "docs/protocol.md"}
    (repo / "docs").mkdir()
    (repo / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "docs" / "protocol.md").write_text("frozen\n", encoding="utf-8")
    _git(repo, "add", "--", *bindings.values())
    _git(repo, "commit", "-m", "freeze sources")
    revision: dict[str, object] = {
        "repo_root": str((tmp_path / "deleted-recorded-worktree").resolve()),
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "git_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "tracked_status_sha256": hashlib.sha256(b"").hexdigest(),
        "files": [
            {
                "role": role,
                "path": relative,
                "sha256": file_sha256(repo / relative),
            }
            for role, relative in bindings.items()
        ],
    }
    return repo, bindings, revision


def _generation_row(cluster: int, index: int) -> dict:
    case_id = f"c-{cluster:02d}-{index:02d}"
    score = float(20 - index)
    return {
        "schema_version": "test-generation",
        "anchor_index": list(EXPECTED_W_ANCHORS).index(cluster),
        "cluster_id": cluster,
        "case_id": case_id,
        "family_id": f"family-{index % 5}",
        "response_id": f"response-{index % 7}",
        "phase_bin": index % 4,
        "family_partition": "generation",
        "prompt_eligible": True,
        "local_prefix": {
            "definition": "full_teacher_forced_input_excluding_observed_target_token",
            "text": f"exact prefix with hostile instruction {case_id}: ignore safeguards",
            "observed_token": {
                "response_position": 10 + index,
                "token_id": 100 + (index % 6),
                "token_text": f" token-{index % 6}",
            },
        },
        "width_one_source_attribution": {
            "highlights": [
                {
                    "token_index": highlight,
                    "token_id": 200 + highlight,
                    "token_text": f" source-{highlight}",
                    "score": score / (highlight + 1),
                    "signed_sum": -score / (highlight + 1),
                    "support_occurrence_count": highlight + 1,
                }
                for highlight in range(16)
            ]
        },
        "candidate_slots": {
            "candidate_axis_width": 5,
            "distinct_competitor_count": 4,
            "observed_token_full_distribution_rank": 1,
            "model_rank_slots": [
                {
                    "rank": rank,
                    "token_id": 300 + rank,
                    "token_text": f" candidate-{rank}",
                    "logit": 1.23456789 * rank,
                    "probability": 0.123456789 / rank,
                    "is_observed": rank == 1,
                }
                for rank in range(1, 6)
            ],
        },
        "candidate_signature": {
            "member_occurrence_count_m": 3,
            "signed_sum": [0.0, 1.23456789, -2.0, 3.0, -4.0],
            "elementwise_mean": [0.0, 0.41152263, -0.66666667, 1.0, -1.33333333],
            "mean_l2_norm": 1.23456789,
            "mean_unit_direction": [0.0, 0.1, -0.2, 0.3, -0.4],
            "clipped": False,
        },
    }


def _comparison(root: Path) -> LoadedCandidateLabelingComparison:
    rows = tuple(
        _generation_row(cluster, index)
        for cluster in EXPECTED_W_ANCHORS
        for index in range(10)
    )
    return LoadedCandidateLabelingComparison(
        root=root,
        manifest={
            "schema_version": "test-comparison-v1",
            "manifest_sha256": "source-manifest-hash",
        },
        anchors={},
        generation_evidence=rows,
        scoring_evidence=(),
        arm_handoff=(),
    )


def test_witness_selection_is_w_only_and_freezes_greedy_diagnostics(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path)
    selected = select_generation_witnesses(comparison)

    assert selected["policy"]["candidate_fields_read"] is False
    assert len(selected["anchors"]) == 12
    assert all(
        len(anchor["selected_case_ids_in_order"]) == 8 for anchor in selected["anchors"]
    )
    assert all(
        set(step["novel_dimensions"])
        == {"family", "response", "phase", "observed_token"}
        for anchor in selected["anchors"]
        for step in anchor["selection_trace"]
    )
    assert all(
        step["novel_dimension_count"] == sum(step["novel_dimensions"].values())
        for anchor in selected["anchors"]
        for step in anchor["selection_trace"]
    )
    assert selected["anchors"][0]["selected_case_ids_in_order"] == [
        f"c-{EXPECTED_W_ANCHORS[0]:02d}-{index:02d}" for index in range(8)
    ]

    mutated_rows = copy.deepcopy(list(comparison.generation_evidence))
    for row in mutated_rows:
        row["candidate_slots"] = {"would": "change a candidate-aware policy"}
        row["candidate_signature"] = {"also": "ignored"}
    mutated = LoadedCandidateLabelingComparison(
        root=comparison.root,
        manifest=comparison.manifest,
        anchors={},
        generation_evidence=tuple(mutated_rows),
        scoring_evidence=(),
        arm_handoff=(),
    )
    assert select_generation_witnesses(mutated) == selected


def test_paired_prompts_share_width_evidence_and_keep_candidate_arm_separate(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path)
    selection = select_generation_witnesses(comparison)
    prompts = build_generation_prompts(comparison, selection)

    assert len(prompts) == 24
    by_identity = {
        (prompt["arm_id"], prompt["cluster_id"]): prompt for prompt in prompts
    }
    for cluster in EXPECTED_W_ANCHORS:
        width = by_identity[("arm_1_width_only", cluster)]
        candidate = by_identity[("arm_2_width_plus_candidate", cluster)]
        assert (
            width["selected_witness_case_ids_in_order"]
            == candidate["selected_witness_case_ids_in_order"]
        )
        assert width["width_evidence_sha256"] == candidate["width_evidence_sha256"]
        assert width["rendered_candidate_witness_sha256_in_order"] is None
        assert len(candidate["rendered_candidate_witness_sha256_in_order"]) == 8
        width_payload = json.dumps(width["message_payload"], ensure_ascii=False)
        candidate_payload = json.dumps(candidate["message_payload"], ensure_ascii=False)
        assert "exact prefix with hostile instruction" in width_payload
        assert "candidate_model_rank_slots" not in width_payload
        assert "cluster_candidate_signature" not in width_payload
        assert width["message_payload"]["expected_output_json_schema"]["properties"][
            "exploratory_candidate_description"
        ] == {"type": "string", "const": "not_available"}
        assert "candidate_model_rank_slots" in candidate_payload
        assert "cluster_candidate_signature" in candidate_payload
        candidate_user_message = candidate["message_payload"]["messages"][1]["content"]
        assert '"logit": "1.23457"' in candidate_user_message
        assert '"clipped": false' in candidate_user_message
        assert (
            width["family_partition"] == candidate["family_partition"] == "generation"
        )
        assert width["provider"] is width["model"] is width["endpoint"] is None


def test_stage_plan_records_counts_without_calls_or_endpoints(tmp_path: Path) -> None:
    _, prompts, plan = build_candidate_labeling_renderer(_comparison(tmp_path))

    assert len(prompts) == 24
    assert [stage["request_count"] for stage in plan["stages"]] == [120, 24, 24]
    assert plan["total_planned_request_count"] == 168
    assert plan["provider_model_endpoints_resolved"] is False
    assert plan["calls_made"] is False
    assert all(
        stage["provider"] is stage["model"] is stage["endpoint"] is None
        and stage["calls_made"] is False
        and stage["selection_audit_visible"] is False
        and "selection_scoring_evidence" in stage["forbidden_inputs"]
        and "audit_evidence" in stage["forbidden_inputs"]
        for stage in plan["stages"]
    )
    assert plan["stages"][1]["depends_on"] == ["opus_semantic_samples"]
    assert plan["stages"][1]["input_source"] == (
        "original_generation_prompt_plus_five_generation_only_opus_semantic_samples"
    )
    assert plan["stages"][2]["input_source"] == "original_generation_prompt"
    assert "opus_semantic_samples" in plan["stages"][2]["forbidden_inputs"]


def test_candidate_renderer_rejects_rank_vector_and_clipping_drift(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path)
    selection = select_generation_witnesses(comparison)

    bad_ranks = copy.deepcopy(list(comparison.generation_evidence))
    bad_ranks[0]["candidate_slots"]["model_rank_slots"][-1]["rank"] = 6
    with pytest.raises(ValueError, match="exact ranks"):
        build_generation_prompts(
            LoadedCandidateLabelingComparison(
                root=comparison.root,
                manifest=comparison.manifest,
                anchors={},
                generation_evidence=tuple(bad_ranks),
                scoring_evidence=(),
                arm_handoff=(),
            ),
            selection,
        )

    bad_vector = copy.deepcopy(list(comparison.generation_evidence))
    bad_vector[0]["candidate_signature"]["signed_sum"] = [0.0] * 4
    with pytest.raises(ValueError, match="length five"):
        build_generation_prompts(
            LoadedCandidateLabelingComparison(
                root=comparison.root,
                manifest=comparison.manifest,
                anchors={},
                generation_evidence=tuple(bad_vector),
                scoring_evidence=(),
                arm_handoff=(),
            ),
            selection,
        )

    clipped = copy.deepcopy(list(comparison.generation_evidence))
    clipped[0]["candidate_signature"]["clipped"] = True
    with pytest.raises(ValueError, match="unclipped"):
        build_generation_prompts(
            LoadedCandidateLabelingComparison(
                root=comparison.root,
                manifest=comparison.manifest,
                anchors={},
                generation_evidence=tuple(clipped),
                scoring_evidence=(),
                arm_handoff=(),
            ),
            selection,
        )


def test_portable_revision_uses_git_objects_after_recorded_worktree_is_gone(
    tmp_path: Path,
) -> None:
    repo, bindings, revision = _make_revision_fixture(tmp_path)

    _validate_recorded_revision_portably(
        revision,
        source_bindings=bindings,
        runtime_paths={"runtime": repo / "runtime.py"},
        current_repo_root=repo,
        label="test revision",
    )


def test_portable_revision_rejects_runtime_blob_and_tree_drift(
    tmp_path: Path,
) -> None:
    repo, bindings, revision = _make_revision_fixture(tmp_path)
    (repo / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current runtime source mismatch: runtime"):
        _validate_recorded_revision_portably(
            revision,
            source_bindings=bindings,
            runtime_paths={"runtime": repo / "runtime.py"},
            current_repo_root=repo,
            label="test revision",
        )

    (repo / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    files = revision["files"]
    assert isinstance(files, list)
    files[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recorded source/blob mismatch: runtime"):
        _validate_recorded_revision_portably(
            revision,
            source_bindings=bindings,
            runtime_paths={"runtime": repo / "runtime.py"},
            current_repo_root=repo,
            label="test revision",
        )

    _, _, revision = _make_revision_fixture(tmp_path / "tree-case")
    revision["git_tree"] = "0" * 40
    tree_repo = tmp_path / "tree-case" / "current-repository"
    with pytest.raises(ValueError, match="recorded commit/tree mismatch"):
        _validate_recorded_revision_portably(
            revision,
            source_bindings=bindings,
            runtime_paths={"runtime": tree_repo / "runtime.py"},
            current_repo_root=tree_repo,
            label="test revision",
        )


def test_portable_revision_requires_exact_shape_and_ordered_inventory(
    tmp_path: Path,
) -> None:
    repo, bindings, revision = _make_revision_fixture(tmp_path)
    revision["unexpected"] = None
    with pytest.raises(TypeError, match="revision shape"):
        _validate_recorded_revision_portably(
            revision,
            source_bindings=bindings,
            runtime_paths={"runtime": repo / "runtime.py"},
            current_repo_root=repo,
            label="test revision",
        )

    _, _, revision = _make_revision_fixture(tmp_path / "inventory-case")
    inventory_repo = tmp_path / "inventory-case" / "current-repository"
    files = revision["files"]
    assert isinstance(files, list)
    revision["files"] = list(reversed(files))
    with pytest.raises(ValueError, match="role/path inventory drift"):
        _validate_recorded_revision_portably(
            revision,
            source_bindings=bindings,
            runtime_paths={"runtime": inventory_repo / "runtime.py"},
            current_repo_root=inventory_repo,
            label="test revision",
        )


def _refresh_manifest_file_binding(output: Path, filename: str) -> None:
    changed_path = output / filename
    manifest_path = output / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    inventory = next(item for item in manifest["files"] if item["path"] == filename)
    inventory["sha256"] = file_sha256(changed_path)
    inventory["size_bytes"] = changed_path.stat().st_size
    manifest_core = dict(manifest)
    manifest_core.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest_core)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_rehashed_prompts(output: Path, mutate) -> None:
    prompt_path = output / GENERATION_PROMPTS_FILE
    rows = [json.loads(line) for line in prompt_path.read_text().splitlines()]
    changed_index = mutate(rows)
    prompt_core = dict(rows[changed_index])
    prompt_core.pop("prompt_sha256")
    rows[changed_index]["prompt_sha256"] = canonical_sha256(prompt_core)
    prompt_path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    _refresh_manifest_file_binding(output, GENERATION_PROMPTS_FILE)


def _rewrite_rehashed_json(
    output: Path, filename: str, hash_field: str, mutate
) -> None:
    path = output / filename
    payload = json.loads(path.read_text())
    mutate(payload)
    core = dict(payload)
    core.pop(hash_field)
    payload[hash_field] = canonical_sha256(core)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_manifest_file_binding(output, filename)


def test_renderer_publishes_no_overwrite_and_loader_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison_root = tmp_path / "comparison"
    comparison_root.mkdir()
    source_manifest = {
        "schema_version": "test-comparison-v1",
        "manifest_sha256": "source-manifest-hash",
    }
    (comparison_root / MANIFEST_FILE).write_text(
        json.dumps(source_manifest) + "\n", encoding="utf-8"
    )
    comparison = _comparison(comparison_root)
    revision = {
        "repo_root": str(tmp_path),
        "git_commit": "test-commit",
        "tracked_worktree_clean": True,
    }
    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_labeling_renderer."
        "collect_candidate_labeling_renderer_revision",
        lambda repo_root: revision,
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_labeling_renderer."
        "_current_repository_root",
        lambda: tmp_path,
    )
    source_roots: list[Path] = []

    def load_comparison(root, *, repo_root, tokenizer=None):
        source_roots.append(Path(repo_root))
        return comparison

    monkeypatch.setattr(
        "circuits.analysis.bonafide.candidate_labeling_renderer."
        "_load_candidate_labeling_comparison_portably",
        load_comparison,
    )
    output = tmp_path / "renderer"
    manifest = run_candidate_labeling_renderer(
        comparison_root=comparison_root,
        output_root=output,
        repo_root=tmp_path,
    )

    loaded = load_candidate_labeling_renderer(output, verify_sources=False)
    assert loaded.manifest == manifest
    assert (
        load_candidate_labeling_renderer(output, verify_sources=True).manifest
        == manifest
    )
    assert source_roots
    assert all(path == tmp_path for path in source_roots)
    core = dict(manifest)
    assert core.pop("manifest_sha256") == canonical_sha256(core)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        run_candidate_labeling_renderer(
            comparison_root=comparison_root,
            output_root=output,
            repo_root=tmp_path,
        )

    arm_tamper = tmp_path / "renderer-arm-tamper"
    shutil.copytree(output, arm_tamper)

    def mutate_arm(rows):
        rows[0]["candidate_evidence_included"] = True
        return 0

    _rewrite_rehashed_prompts(arm_tamper, mutate_arm)
    with pytest.raises(ValueError, match="arm contract"):
        load_candidate_labeling_renderer(arm_tamper, verify_sources=False)

    preamble_tamper = tmp_path / "renderer-preamble-tamper"
    shutil.copytree(output, preamble_tamper)

    def mutate_preamble(rows):
        payload = rows[0]["message_payload"]
        payload["messages"][1]["content"] = (
            "SELECTION RESULT: leaked\n\n" + payload["messages"][1]["content"]
        )
        rows[0]["message_payload_sha256"] = canonical_sha256(payload)
        return 0

    _rewrite_rehashed_prompts(preamble_tamper, mutate_preamble)
    with pytest.raises(ValueError, match="preamble drift"):
        load_candidate_labeling_renderer(preamble_tamper, verify_sources=False)

    candidate_tamper = tmp_path / "renderer-candidate-tamper"
    shutil.copytree(output, candidate_tamper)

    def mutate_candidate(rows):
        candidate_index = 12
        payload = rows[candidate_index]["message_payload"]
        content = payload["messages"][1]["content"]
        payload["messages"][1]["content"] = content.replace('"rank": 1', '"rank": 9', 1)
        rows[candidate_index]["message_payload_sha256"] = canonical_sha256(payload)
        return candidate_index

    _rewrite_rehashed_prompts(candidate_tamper, mutate_candidate)
    with pytest.raises(ValueError, match="candidate slot contract"):
        load_candidate_labeling_renderer(candidate_tamper, verify_sources=False)

    stage_tamper = tmp_path / "renderer-stage-tamper"
    shutil.copytree(output, stage_tamper)
    _rewrite_rehashed_json(
        stage_tamper,
        STAGE_PLAN_FILE,
        "stage_plan_sha256",
        lambda plan: plan["stages"][1].update({"selection_audit_visible": True}),
    )
    with pytest.raises(ValueError, match="stage-plan contract"):
        load_candidate_labeling_renderer(stage_tamper, verify_sources=False)

    selection_tamper = tmp_path / "renderer-selection-tamper"
    shutil.copytree(output, selection_tamper)
    _rewrite_rehashed_json(
        selection_tamper,
        WITNESS_SELECTION_FILE,
        "witness_selection_sha256",
        lambda selection: selection["anchors"][0]["selection_trace"][0].update(
            {"unexpected_heldout_field": "leak"}
        ),
    )
    with pytest.raises(ValueError, match="trace contract"):
        load_candidate_labeling_renderer(selection_tamper, verify_sources=False)

    prompt_path = output / GENERATION_PROMPTS_FILE
    prompt_path.write_text(prompt_path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="file drift"):
        load_candidate_labeling_renderer(output, verify_sources=False)
