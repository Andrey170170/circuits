"""Focused tests for deterministic frozen-corpus inventory construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
)
from circuits.analysis.bonafide.inventory import (
    INVENTORY_SCHEMA,
    build_inventory,
    write_inventory,
)
from circuits.tracing.artifact import DATA_FILENAME, SCHEMA_VERSION


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selection_item(
    source_id: str,
    *,
    example_id: str,
    base_question_id: str,
    role: str,
    position: int,
    token_id: int,
) -> dict:
    return {
        "artifact_id": source_id,
        "response_token_count": 3,
        "objective": {
            "benchmark_only_multi_target": False,
            "name": "single_selected_logit",
        },
        "example": {
            "example_id": example_id,
            "base_question_id": base_question_id,
        },
        "target_selection": {
            "width": 1,
            "kind": "explicit_response_positions",
            "response_token_positions": [position],
            "final_target_token_id": token_id,
            "final_selection": {
                "corpus_role": role,
                "selection_reasons": [{"bucket": "fixture"}],
            },
        },
    }


def _write_compact_artifact(
    root: Path,
    *,
    wave_id: str,
    item: dict,
    runtime_id: str,
    model_id: str = "fake/model",
    model_revision: str = "exact-revision",
) -> Path:
    target = root / wave_id / runtime_id
    target.mkdir(parents=True)
    payload = b"trusted-fixture-payload"
    (target / DATA_FILENAME).write_bytes(payload)
    example = item["example"]
    selection = item["target_selection"]
    position = selection["response_token_positions"][0]
    token_id = selection["final_target_token_id"]
    source_tree_hash = "1" * 64
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_file": DATA_FILENAME,
        "data_size_bytes": len(payload),
        "data_sha256": hashlib.sha256(payload).hexdigest(),
        "artifact_id": runtime_id,
        "source_artifact_id": item["artifact_id"],
        "benchmark_wave_id": wave_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "target_count": 1,
        "scientifically_reusable": True,
        "numerically_valid": True,
        "benchmark_only": False,
        "source_target_selection": selection,
        "bonafide_example": example,
        "artifact_identity": {
            "sha256": "2" * 64,
            "source_work_item_sha256": canonical_sha256(item),
        },
        "code_revision": {
            "git_commit": "fixture",
            "git_dirty": False,
            "source_tree_sha256": source_tree_hash,
        },
        "target_provenance": [
            {
                "response_token_position": position,
                "prediction_token_position": position + 10,
                "token_id": token_id,
                "token_text": f" token-{token_id}",
                "logit": 3.0,
                "probability": 0.5,
            }
        ],
    }
    _write_json(target / "manifest.json", manifest)
    _write_json(target / "metrics.json", {})
    return target


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    specifications = [
        ("dense", "dense_discovery", True, 0, 10, "family-a"),
        (
            "holdout",
            "broad_confirmatory_holdout",
            False,
            1,
            11,
            "family-b",
        ),
        ("pathological", "broad_discovery", True, 2, 12, "family-c"),
    ]
    waves = []
    items = []
    for name, role, eligible, position, token_id, family_id in specifications:
        source_id = f"trace-source-{name}"
        example_id = f"example-{name}"
        item = _selection_item(
            source_id,
            example_id=example_id,
            base_question_id=family_id,
            role=role,
            position=position,
            token_id=token_id,
        )
        items.append(item)
        waves.append(
            {
                "wave_id": f"wave-{name}",
                "example_id": example_id,
                "corpus_role": role,
                "cluster_fit_eligible": eligible,
                "items": [item],
            }
        )
    selection = {
        "schema_version": "bonafide-trace-benchmark/v1",
        "tokenizer": {
            "model_id": "fake/model",
            "revision": "exact-revision",
            "chat_template_sha256": "3" * 64,
            "file_manifest": {"aggregate_sha256": "4" * 64},
        },
        "waves": waves,
    }
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)

    plan = {
        "schema_version": "bonafide-trace-execution-plan/v1",
        "sources": {
            "final_trace_manifest": {
                "path": str(selection_path),
                "sha256": file_sha256(selection_path),
                "canonical_sha256": canonical_sha256(selection),
            }
        },
        "extremes": {
            "manual_pathological": [
                {
                    "source_artifact_id": "trace-source-pathological",
                }
            ]
        },
        "tasks": [],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    _write_compact_artifact(
        artifact_root,
        wave_id="wave-dense",
        item=items[0],
        runtime_id="trace-runtime-dense",
    )
    _write_compact_artifact(
        artifact_root,
        wave_id="wave-holdout",
        item=items[1],
        runtime_id="trace-runtime-holdout",
    )
    return selection_path, plan_path, artifact_root, items


def test_inventory_is_deterministic_partitioned_and_self_hashed(
    tmp_path: Path,
) -> None:
    selection_path, plan_path, artifact_root, _ = _fixture(tmp_path)
    first = build_inventory(
        selection_path=selection_path,
        execution_plan_path=plan_path,
        artifact_root=artifact_root,
        validation_level="integrity",
    )
    second = build_inventory(
        selection_path=selection_path,
        execution_plan_path=plan_path,
        artifact_root=artifact_root,
        validation_level="integrity",
    )
    assert first == second
    assert first["schema_version"] == INVENTORY_SCHEMA
    assert first["summary"] == {
        "planned": 3,
        "completed": 2,
        "discovery_planned": 2,
        "discovery_completed": 1,
        "holdout_planned": 1,
        "holdout_completed": 1,
        "excluded_pathological": 1,
        "missing": 0,
        "corrupt": 0,
        "unexpected": 0,
    }
    assert [record["status"] for record in first["records"]] == [
        "discovery",
        "holdout",
        "excluded_pathological",
    ]
    unhashed = dict(first)
    recorded_hash = unhashed.pop("inventory_sha256")
    assert recorded_hash == canonical_sha256(unhashed)

    output = tmp_path / "inventory.json"
    write_inventory(output, first)
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_inventory_reports_unknown_physical_artifact(tmp_path: Path) -> None:
    selection_path, plan_path, artifact_root, items = _fixture(tmp_path)
    unknown = {
        **items[0],
        "artifact_id": "trace-source-unknown",
    }
    _write_compact_artifact(
        artifact_root,
        wave_id="wave-unknown",
        item=unknown,
        runtime_id="trace-runtime-unknown",
    )
    inventory = build_inventory(
        selection_path=selection_path,
        execution_plan_path=plan_path,
        artifact_root=artifact_root,
        validation_level="integrity",
    )
    assert inventory["summary"]["unexpected"] == 1
    assert (
        inventory["unexpected_artifacts"][0]["source_artifact_id"]
        == "trace-source-unknown"
    )


def test_inventory_classifies_payload_checksum_failure_as_corrupt(
    tmp_path: Path,
) -> None:
    selection_path, plan_path, artifact_root, _ = _fixture(tmp_path)
    payload = artifact_root / "wave-dense" / "trace-runtime-dense" / DATA_FILENAME
    payload.write_bytes(b"same-size-corruption!!!")
    inventory = build_inventory(
        selection_path=selection_path,
        execution_plan_path=plan_path,
        artifact_root=artifact_root,
        validation_level="integrity",
    )
    dense = next(
        record
        for record in inventory["records"]
        if record["source_artifact_id"] == "trace-source-dense"
    )
    assert dense["status"] == "corrupt"
    assert "checksum mismatch" in dense["error"]
    assert inventory["summary"]["corrupt"] == 1
    assert inventory["summary"]["completed"] == 1


def test_inventory_fails_closed_on_partition_contract_drift(tmp_path: Path) -> None:
    selection_path, plan_path, artifact_root, _ = _fixture(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["waves"][1]["cluster_fit_eligible"] = True
    _write_json(selection_path, selection)
    try:
        build_inventory(
            selection_path=selection_path,
            execution_plan_path=plan_path,
            artifact_root=artifact_root,
            validation_level="integrity",
        )
    except ValueError as error:
        assert "partition contract mismatch" in str(error)
    else:
        raise AssertionError("partition drift should fail closed")
