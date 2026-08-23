from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from circuits.observatory import CATALOG_SCHEMA, TRACE_GRAPH_SCHEMA
from circuits.observatory.bundle import discover_trace_directories, sync_bundle
from circuits.tracing.artifact import save_compact_trace
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.trace import CircuitData


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_trace(root: Path, *, artifact_id: str, response_position: int) -> Path:
    target = root / artifact_id
    label = "generic-teacher-forced-label"
    nodes = pd.DataFrame(
        [
            {
                "layer": -1,
                "token": 0,
                "neuron": 10,
                "attribution": -0.25,
                "activation": 1.0,
                "attr_map": [-0.25],
                "contrib_map": [0.125],
                "label": label,
            },
            {
                "layer": 0,
                "token": 1,
                "neuron": 7,
                "attribution": 0.75,
                "activation": -2.0,
                "attr_map": [0.5],
                "contrib_map": [-0.375],
                "label": label,
            },
            {
                "layer": 2,
                "token": 1,
                "neuron": 13,
                "attribution": 1.0,
                "activation": 3.5,
                "attr_map": [1.0],
                "contrib_map": [3.5],
                "label": label,
            },
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "layer": "-1->0",
                "token": "0->1",
                "neuron": "10->7",
                "attribution": -0.125,
                "weight": 4.0,
                "label": label,
            },
            {
                "layer": "0->2",
                "token": "1->1",
                "neuron": "7->13",
                "attribution": 0.25,
                "weight": -8.0,
                "label": label,
            },
        ]
    )
    data = CircuitData(
        df_node=nodes,
        df_edge=edges,
        cis=[[10, 13]],
        attention_masks=[[1, 1]],
        labels=[label],
        target_logits=[[13]],
        target_logit_probs=[[0.625]],
        target_logit_values=[[3.5]],
        k=1,
        config=ADAGConfig(),
        model_id="example/model",
        target_provenance=[
            {
                "response_token_position": response_position,
                "absolute_token_position": 1,
                "prediction_token_position": 0,
                "token_id": 13,
                "token_text": "answer",
                "logit": 3.5,
                "probability": 0.625,
            }
        ],
        trace_metadata={
            "prompt": "Compute it.",
            "response": "answer",
            "system_prompt": "Be exact.",
            "assistant_prefix_token_count": 1,
            "input_token_count": 2,
        },
    )
    return save_compact_trace(
        target,
        data,
        metrics={"status": "success"},
        manifest={
            "artifact_id": artifact_id,
            "model_revision": "revision-abc",
            "objective": {"name": "selected_logit"},
            "bonafide_example": {
                "example_id": f"response-{response_position}",
                "prompt": "Compute it.",
                "response": "answer",
            },
            "source_target_selection": {
                "human_selection": {"comment": f"position {response_position}"}
            },
        },
    )


def _sync(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    _compact_trace(source, artifact_id="trace-later", response_position=9)
    _compact_trace(source, artifact_id="trace-earlier", response_position=3)
    site = tmp_path / "site"
    state = tmp_path / "state"
    sync_bundle(
        trace_root=source,
        site_root=site,
        state_root=state,
        allow_numeric_tokens=True,
    )
    return source, site, state


def test_sync_preserves_signed_evidence_and_trace_boundaries(tmp_path: Path) -> None:
    source, site, _state = _sync(tmp_path)
    catalog = json.loads((site / "catalog.json").read_text())
    assert catalog["schema_version"] == CATALOG_SCHEMA
    assert [item["response_position"] for item in catalog["traces"]] == [3, 9]
    assert {item["model_id"] for item in catalog["traces"]} == {"example/model"}
    assert all(
        item["data_sha256"] == item["source_payload_sha256"]
        for item in catalog["traces"]
    )

    trace_documents = [
        json.loads((site / "traces" / f"{item['artifact_id']}.json").read_text())
        for item in catalog["traces"]
    ]
    all_node_ids: list[set[str]] = []
    for trace in trace_documents:
        assert trace["schema_version"] == TRACE_GRAPH_SCHEMA
        artifact_id = trace["artifact"]["artifact_id"]
        assert trace["target"]["prediction_position"] == 0
        assert trace["target"]["observed_absolute_position"] == 1
        assert trace["context"]["tokens"] == [
            {
                "absolute_position": 0,
                "response_position": None,
                "role": "prefix",
                "token_id": 10,
                "text": "[10]",
            },
            {
                "absolute_position": 1,
                "response_position": 0,
                "role": "assistant",
                "token_id": 13,
                "text": "[13]",
            },
        ]
        kinds = {node["kind"] for node in trace["nodes"]}
        assert kinds == {"input_token", "raw_mlp_neuron", "target_logit"}
        mlp = next(node for node in trace["nodes"] if node["kind"] == "raw_mlp_neuron")
        assert mlp["activation"] == -2.0
        assert mlp["activation_polarity"] == "-"
        assert mlp["attribution"] == 0.75
        assert mlp["attribution_sign"] == "+"
        edge = next(edge for edge in trace["edges"] if edge["attribution"] < 0)
        assert edge["attribution"] == -0.125
        assert edge["weight"] == 4.0
        assert all(edge["trace_unit_id"] == artifact_id for edge in trace["edges"])
        assert all(
            node["occurrence"]["trace_unit_id"] == artifact_id
            for node in trace["nodes"]
        )
        all_node_ids.append({node["id"] for node in trace["nodes"]})
    assert all_node_ids[0].isdisjoint(all_node_ids[1])
    assert str(source) not in json.dumps(trace_documents)


def test_sync_never_changes_source_and_replacement_keeps_backup(tmp_path: Path) -> None:
    source, site, state = _sync(tmp_path)
    before = {
        str(path.relative_to(source)): _sha256(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError, match="--replace"):
        sync_bundle(
            trace_root=source,
            site_root=site,
            state_root=state,
            allow_numeric_tokens=True,
        )
    result = sync_bundle(
        trace_root=source,
        site_root=site,
        state_root=state,
        allow_numeric_tokens=True,
        replace=True,
    )
    assert result["backup_root"] is not None
    assert Path(result["backup_root"]).is_dir()
    after = {
        str(path.relative_to(source)): _sha256(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_sync_fails_closed_without_tokenizer_or_explicit_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _compact_trace(source, artifact_id="trace-one", response_position=1)
    with pytest.raises(ValueError, match="--allow-numeric-tokens"):
        sync_bundle(
            trace_root=source,
            site_root=tmp_path / "site",
            state_root=tmp_path / "state",
        )
    assert not (tmp_path / "site").exists()


def test_discovery_rejects_other_manifest_schemas(tmp_path: Path) -> None:
    artifact = tmp_path / "source" / "other"
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text(
        json.dumps({"schema_version": "not-a-compact-trace"})
    )
    with pytest.raises(ValueError, match="unsupported compact trace schema"):
        discover_trace_directories(tmp_path / "source")


def test_two_synthetic_label_sets_are_versioned_and_separate(tmp_path: Path) -> None:
    _source, site, _state = _sync(tmp_path)
    index = json.loads((site / "label-sets" / "index.json").read_text())
    assert len(index["label_sets"]) == 2
    assert all(item["synthetic"] is True for item in index["label_sets"])
    for item in index["label_sets"]:
        overlay = json.loads(
            (site / "label-sets" / f"{item['label_set_id']}.json").read_text()
        )
        assert overlay["synthetic"] is True
        assert "no semantic meaning" in overlay["warning"]
        assert overlay["model_id"] == "example/model"
        assert overlay["model_revision"] == "revision-abc"
        assert overlay["basis_schema"] == "adag.bonafide.signed-basis-key.v1"
        assert (
            overlay["polarity_derivation"]
            == "activation-sign-nonnegative-positive.v1"
        )
        assert set(overlay["labels_by_trace"]) == {"trace-earlier", "trace-later"}
