"""Focused tests for immutable response shards and virtual compaction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from circuits.analysis.bonafide import streaming as streaming_module
from circuits.analysis.bonafide.build_plan import (
    build_downstream_plan,
    collect_downstream_code_revision,
    collect_downstream_environment,
)
from circuits.analysis.bonafide.canonical import (
    canonical_sha256,
    file_sha256,
)
from circuits.analysis.bonafide.compaction import compact_downstream_lane
from circuits.analysis.bonafide.streaming import (
    build_joint_response_shards,
    build_response_shard,
)
from circuits.tracing.artifact import save_compact_trace
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.trace import CircuitData

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _trace(
    *,
    response_id: str,
    response_position: int,
    token_id: int,
) -> CircuitData:
    label = f"{response_id}___0"
    return CircuitData(
        df_node=pd.DataFrame(
            [
                {
                    "layer": 0,
                    "token": 0,
                    "neuron": 10,
                    "attribution": 0.2,
                    "activation": 1.0,
                    "attr_map": [0.1, None],
                    "contrib_map": [0.3],
                    "label": label,
                },
                {
                    "layer": 1,
                    "token": 1,
                    "neuron": 20,
                    "attribution": -0.4,
                    "activation": -1.0,
                    "attr_map": [0.0, 0.2],
                    "contrib_map": [-0.1],
                    "label": label,
                },
                {
                    "layer": 2,
                    "token": response_position + 2,
                    "neuron": token_id,
                    "attribution": 1.0,
                    "activation": 3.0,
                    "attr_map": [0.2, 0.4],
                    "contrib_map": [3.0],
                    "label": label,
                },
            ]
        ),
        df_edge=pd.DataFrame(
            [
                {
                    "layer": "0->1",
                    "token": "0->1",
                    "neuron": "10->20",
                    "attribution": 0.2,
                    "weight": 0.3,
                    "label": label,
                },
                {
                    "layer": "1->2",
                    "token": f"1->{response_position + 2}",
                    "neuron": f"20->{token_id}",
                    "attribution": -0.1,
                    "weight": 0.2,
                    "label": label,
                },
            ]
        ),
        cis=[[1, 2]],
        attention_masks=[[1, 1]],
        labels=[response_id],
        target_logits=[[token_id]],
        target_logit_probs=[[0.5]],
        target_logit_values=[[3.0]],
        target_provenance=[
            {
                "response_token_position": response_position,
                "prediction_token_position": response_position + 1,
                "token_id": token_id,
                "token_text": f" token-{token_id}",
                "logit": 3.0,
                "probability": 0.5,
            }
        ],
        trace_metadata={"response_token_count": 2},
        benchmark_only=False,
        k=1,
        config=ADAGConfig(device="cpu"),
        model_id="fake/model",
    )


def _inventory_fixture(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "artifacts"
    records = []
    atlas_index = 0
    for response_index, response_id in enumerate(("response-a", "response-b")):
        for response_position in range(2):
            source_id = f"source-{response_id}-{response_position}"
            runtime_id = f"trace-{response_id}-{response_position}"
            artifact_path = artifact_root / response_id / runtime_id
            save_compact_trace(
                artifact_path,
                _trace(
                    response_id=response_id,
                    response_position=response_position,
                    token_id=100 + response_index * 10 + response_position,
                ),
                manifest={
                    "artifact_id": runtime_id,
                    "source_artifact_id": source_id,
                    "model_revision": "exact-revision",
                },
            )
            manifest = json.loads(
                (artifact_path / "manifest.json").read_text(encoding="utf-8")
            )
            records.append(
                {
                    "atlas_trace_index": atlas_index,
                    "source_artifact_id": source_id,
                    "source_wave_id": f"wave-{response_id}",
                    "base_question_id": f"family-{response_id}",
                    "example_id": response_id,
                    "response_id": response_id,
                    "condition": {"fixture": True},
                    "corpus_role": "dense_discovery",
                    "cluster_fit_eligible": True,
                    "response_position": response_position,
                    "target_token_id": (100 + response_index * 10 + response_position),
                    "selection_reasons": [{"bucket": "fixture"}],
                    "source_selection_manifest_sha256": "1" * 64,
                    "source_execution_plan_sha256": "2" * 64,
                    "trace_unit_id": runtime_id,
                    "artifact_id": runtime_id,
                    "artifact_path": str(artifact_path),
                    "artifact_manifest_sha256": file_sha256(
                        artifact_path / "manifest.json"
                    ),
                    "artifact_payload_sha256": manifest["data_sha256"],
                    "prediction_position": response_position + 1,
                    "target_token_text": f" token-{100 + response_index * 10 + response_position}",
                    "target_logit": 3.0,
                    "target_probability": 0.5,
                    "model_id": "fake/model",
                    "model_revision": "exact-revision",
                    "trace_configuration_identity": "3" * 64,
                    "code_revision": {"git_commit": "fixture"},
                    "tracing_source_tree_sha256": "4" * 64,
                    "status": "discovery",
                    "error": None,
                }
            )
            atlas_index += 1
    inventory = {
        "schema_version": "adag.bonafide.atlas-inventory.v1",
        "validation_level": "full",
        "sources": {},
        "tokenizer_identity": {
            "model_id": "fake/model",
            "revision": "exact-revision",
        },
        "summary": {
            "planned": 4,
            "completed": 4,
            "discovery_planned": 4,
            "discovery_completed": 4,
            "holdout_planned": 0,
            "holdout_completed": 0,
            "excluded_pathological": 0,
            "missing": 0,
            "corrupt": 0,
            "unexpected": 0,
        },
        "records": records,
        "unexpected_artifacts": [],
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    path = tmp_path / "inventory.json"
    _write_json(path, inventory)
    return path


def _plan_from_inventory(
    tmp_path: Path,
    lane: str,
    inventory_path: Path,
) -> dict:
    return build_downstream_plan(
        inventory_path=inventory_path,
        output_root=tmp_path / f"{lane}-output",
        lane=lane,  # type: ignore[arg-type]
        repo_root=REPO_ROOT,
        allow_dirty_development=True,
        code_revision=collect_downstream_code_revision(REPO_ROOT),
        runtime_environment=collect_downstream_environment(),
    )


def _plan(tmp_path: Path, lane: str) -> dict:
    inventory_path = _inventory_fixture(tmp_path / lane)
    return _plan_from_inventory(tmp_path, lane, inventory_path)


def test_joint_builder_loads_each_trace_once_and_buffers_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inventory_path = _inventory_fixture(tmp_path / "joint")
    feature_plan = _plan_from_inventory(
        tmp_path,
        "dense_features",
        inventory_path,
    )
    multiplex_plan = _plan_from_inventory(
        tmp_path,
        "dense_multiplex",
        inventory_path,
    )
    original_load = streaming_module.load_compact_trace
    loaded_paths: list[str] = []

    def counted_load(path: str):
        loaded_paths.append(path)
        return original_load(path)

    monkeypatch.setattr(streaming_module, "load_compact_trace", counted_load)
    for task_index in range(2):
        result = build_joint_response_shards(
            feature_plan=feature_plan,
            multiplex_plan=multiplex_plan,
            task_index=task_index,
        )
        assert result["status"] == "complete"
        for lane_result in result["lanes"].values():
            manifest = lane_result["manifest"]
            assert manifest["build_mode"] == "joint_one_pass"
            assert (
                manifest["stage_timings"]["artifact_load_validate"]["call_count"] == 2
            )
            assert manifest["parquet_sinks"]["targets.parquet"]["row_count"] == 2
            assert manifest["parquet_sinks"]["targets.parquet"]["flush_count"] == 1
    assert len(loaded_paths) == 4
    assert compact_downstream_lane(feature_plan)["manifest"]["target_count"] == 4
    assert compact_downstream_lane(multiplex_plan)["manifest"]["occurrence_count"] == 12

    resumed = build_joint_response_shards(
        feature_plan=feature_plan,
        multiplex_plan=multiplex_plan,
        task_index=0,
    )
    assert resumed["status"] == "skipped_complete"
    assert len(loaded_paths) == 4


def test_feature_shards_resume_and_compact_without_dense_materialization(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, "dense_features")
    first = build_response_shard(plan=plan, task_index=0)
    second = build_response_shard(plan=plan, task_index=1)
    assert first["status"] == second["status"] == "complete"
    resumed = build_response_shard(plan=plan, task_index=0)
    assert resumed["status"] == "skipped_complete"

    compacted = compact_downstream_lane(plan)
    manifest = compacted["manifest"]
    assert manifest["target_count"] == 4
    assert manifest["response_count"] == 2
    assert manifest["signed_basis_count"] == 6
    assert manifest["resource_estimate"]["profile_column_count"] == 8
    assert manifest["resource_estimate"]["supported_profile_cell_count"] == 20
    assert manifest["resource_estimate"]["nonzero_profile_cell_count"] == 16
    assert not (
        Path(plan["output_root"]) / "compacted" / "occurrence-index.parquet"
    ).exists()
    assert compact_downstream_lane(plan)["status"] == "skipped_complete"


def test_multiplex_shards_preserve_occurrences_support_and_correspondence(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, "dense_multiplex")
    for task_index in range(2):
        build_response_shard(plan=plan, task_index=task_index)
    compacted = compact_downstream_lane(plan)
    manifest = compacted["manifest"]
    assert manifest["target_count"] == 4
    assert manifest["occurrence_count"] == 12

    shard = Path(plan["output_root"]) / "shards" / "task-000-response-a"
    correspondences = pq.read_table(shard / "longitudinal-correspondence.parquet")
    assert correspondences.num_rows == 2
    assert set(correspondences.column("explicitly_noncausal").to_pylist()) == {True}
    edge_support = pq.read_table(shard / "aggregated-edge-support.parquet")
    assert edge_support.num_rows == 3
    assert set(edge_support.column("support_target_count").to_pylist()) == {1, 2}

    occurrence_index = pq.read_table(
        Path(plan["output_root"]) / "compacted" / "occurrence-index.parquet"
    )
    assert occurrence_index.num_rows == 12
