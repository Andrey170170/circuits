from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.graph_labeling.evidence import (
    allowed_evidence_ids,
    reads_from_evidence_ids,
)
from circuits.graph_labeling.openai_batch import (
    _parse_output,
    collect_openai_batch,
    openai_batch_status,
    prepare_openai_batch,
    recover_openai_batch,
    recover_openai_upload,
    submit_openai_batch,
)
from circuits.graph_labeling.runtime import (
    execute,
    export_overlay,
    ingest_results,
    label_set_identity,
    normalize_structured_label,
    prepare,
    status,
)
from circuits.graph_labeling.schema import (
    EvidencePacket,
    ExecutionSpec,
    GraphLabelingSpec,
    PromptRequest,
)
from circuits.observatory import (
    CATALOG_SCHEMA,
    LABEL_SET_SCHEMA,
    MANIFEST_SCHEMA,
    TRACE_GRAPH_SCHEMA,
)


def _bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value) + b"\n")


def _node(
    occurrence_id: str,
    basis_id: str,
    *,
    kind: str,
    layer: int,
    neuron: int,
    position: int,
    attr: float,
    profile: list[float],
) -> dict[str, object]:
    return {
        "id": occurrence_id,
        "basis_id": basis_id,
        "kind": kind,
        "activation": 2.0,
        "attribution": attr,
        "attribution_map": profile,
        "contribution_map": [attr * 2],
        "occurrence": {
            "schema_version": "adag.bonafide.occurrence-key.v1",
            "trace_unit_id": "trace-test",
            "layer": layer,
            "neuron_index": neuron,
            "polarity": "+",
            "token_position": position,
        },
        "basis": {
            "schema_version": "adag.bonafide.signed-basis-key.v1",
            "model_id": "example/model",
            "model_revision": "revision-1",
            "layer": layer,
            "neuron_index": neuron,
            "polarity": "+",
        },
    }


def _site(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    site = tmp_path / "site"
    source_hash = "a" * 64
    nodes = [
        _node(
            "occ-selected",
            "basis-selected",
            kind="raw_mlp_neuron",
            layer=2,
            neuron=7,
            position=2,
            attr=0.7,
            profile=[1.0, -2.0, 0.5],
        ),
        _node(
            "occ-unselected",
            "basis-unselected",
            kind="raw_mlp_neuron",
            layer=1,
            neuron=8,
            position=1,
            attr=0.1,
            profile=[0.1, 0.2, -0.1],
        ),
        _node(
            "occ-target",
            "basis-target",
            kind="target_logit",
            layer=3,
            neuron=13,
            position=2,
            attr=1.0,
            profile=[0.2, 0.3, 0.4],
        ),
    ]
    edges = [
        {
            "id": "edge-in",
            "trace_unit_id": "trace-test",
            "source": "occ-unselected",
            "target": "occ-selected",
            "attribution": -0.25,
            "weight": 1.5,
        },
        {
            "id": "edge-direct",
            "trace_unit_id": "trace-test",
            "source": "occ-selected",
            "target": "occ-target",
            "attribution": 0.75,
            "weight": 2.0,
        },
    ]
    trace = {
        "schema_version": TRACE_GRAPH_SCHEMA,
        "claim_boundary": "local pruned graph only",
        "model": {"model_id": "example/model", "model_revision": "revision-1"},
        "artifact": {
            "artifact_id": "trace-test",
            "trace_unit_id": "trace-test",
            "source_hash": source_hash,
        },
        "target": {
            "response_position": 3,
            "observed_absolute_position": 3,
            "prediction_position": 2,
            "token_id": 13,
            "token_text": "answer",
        },
        "context": {
            "system_prompt": "system",
            "prompt": "question",
            "response": "work answer FUTURE_SENTINEL",
            "tokens": [
                {"absolute_position": 0, "token_id": 10, "text": "q", "role": "prefix"},
                {
                    "absolute_position": 1,
                    "token_id": 11,
                    "text": "work",
                    "role": "assistant",
                },
                {
                    "absolute_position": 2,
                    "token_id": 12,
                    "text": " ",
                    "role": "assistant",
                },
                {
                    "absolute_position": 3,
                    "token_id": 13,
                    "text": "answer",
                    "role": "assistant",
                },
                {
                    "absolute_position": 4,
                    "token_id": 14,
                    "text": "FUTURE_SENTINEL",
                    "role": "assistant",
                },
            ],
        },
        "nodes": nodes,
        "edges": edges,
        "diagnostics": {"node_count": len(nodes), "edge_count": len(edges)},
    }
    catalog = {
        "schema_version": CATALOG_SCHEMA,
        "claim_boundary": "local pruned graph only",
        "model": {"model_id": "example/model", "model_revision": "revision-1"},
        "traces": [
            {
                "artifact_id": "trace-test",
                "response_position": 3,
                "source_hash": source_hash,
            }
        ],
    }
    index = {
        "schema_version": "adag.observatory.label-set-index.v1",
        "label_sets": [],
    }
    _write(site / "catalog.json", catalog)
    _write(site / "traces" / "trace-test.json", trace)
    _write(site / "label-sets" / "index.json", index)
    files = {
        "catalog.json": canonical_sha256(catalog),
        "traces/trace-test.json": canonical_sha256(trace),
        "label-sets/index.json": canonical_sha256(index),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": "2026-01-01T00:00:00+00:00",
        "file_hash_algorithm": "sha256-canonical-json-v1",
        "catalog_schema": CATALOG_SCHEMA,
        "trace_schema": TRACE_GRAPH_SCHEMA,
        "trace_count": 1,
        "artifact_ids": ["trace-test"],
        "files": files,
    }
    manifest["content_hash"] = canonical_sha256(manifest)
    _write(site / "viewer-manifest.json", manifest)
    return site, {
        "viewer": file_sha256(site / "viewer-manifest.json"),
        "catalog": file_sha256(site / "catalog.json"),
        "trace": file_sha256(site / "traces" / "trace-test.json"),
        "source": source_hash,
    }


def _spec(site: Path, hashes: dict[str, str]) -> GraphLabelingSpec:
    return GraphLabelingSpec.model_validate(
        {
            "run_name": "test-run",
            "study": {
                "study_name": "test-study",
                "source": {
                    "site_root": str(site),
                    "viewer_manifest_sha256": hashes["viewer"],
                    "catalog_sha256": hashes["catalog"],
                },
                "trace": {
                    "artifact_id": "trace-test",
                    "response_position": 3,
                    "artifact_source_sha256": hashes["source"],
                    "trace_file_sha256": hashes["trace"],
                },
                "selection": {"groups": {"direct_target_parent": ["occ-selected"]}},
                "methods": [
                    {
                        "method_id": "deterministic-evidence-summary-v1",
                        "kind": "deterministic_evidence_summary_v1",
                        "prompt_version": "deterministic-v1",
                    },
                    {
                        "method_id": "structured-llm-graph-role-v1",
                        "kind": "structured_llm_graph_role_v1",
                        "prompt_version": "structured-v1",
                        "labeler": {
                            "provider": "openai",
                            "model": "test-model",
                            "max_output_tokens": 100,
                        },
                    },
                ],
            },
        }
    )


def test_prepare_preserves_causal_profile_boundary_and_materializes_requests(
    tmp_path: Path,
) -> None:
    site, hashes = _site(tmp_path)
    spec = _spec(site, hashes)
    receipt = prepare(spec, tmp_path / "run")
    assert receipt.occurrence_count == 1
    assert receipt.request_count == 1
    packet = EvidencePacket.model_validate(
        json.loads((tmp_path / "run/evidence/occ-selected.json").read_text())
    )
    assert packet.context["observed_target_token"]["text"] == "answer"
    assert len(packet.context["observed_tokens"]) == 4
    assert len(packet.context["causal_profile_tokens"]) == 3
    assert packet.coverage["observed_target_excluded_from_causal_profile"] is True
    assert packet.coverage["direct_target_edge_count"] == 1
    assert packet.direct_target_edges[0]["edge_id"] == "edge-direct"
    assert "response" not in packet.context
    assert "FUTURE_SENTINEL" not in json.dumps(packet.model_dump(mode="json"))

    label_set_id = status(tmp_path / "run")["methods"]["structured-llm-graph-role-v1"][
        "label_set_id"
    ]
    request = json.loads(
        (tmp_path / f"run/requests/{label_set_id}/occ-selected.json").read_text()
    )
    assert request["evidence_sha256"] == packet.evidence_sha256
    assert "not a global neuron meaning" in request["messages"][0]["content"]
    manifest = json.loads((tmp_path / "run/run-manifest.json").read_text())
    assert len(manifest["code_revision"]["source_tree_sha256"]) == 64
    assert "FUTURE_SENTINEL" not in json.dumps(request)


def test_scientific_and_execution_identities_are_separate(tmp_path: Path) -> None:
    site, hashes = _site(tmp_path)
    first = _spec(site, hashes)
    payload = first.model_dump(mode="json")
    payload["study"]["source"]["site_root"] = str(tmp_path / "another-mount")
    second = GraphLabelingSpec.model_validate(payload)
    assert first.study.identity_sha256 == second.study.identity_sha256
    alias_payload = first.model_dump(mode="json")
    alias_payload["study"]["study_name"] = "renamed-study"
    alias_payload["study"]["methods"][0]["method_id"] = "renamed-method"
    aliased = GraphLabelingSpec.model_validate(alias_payload)
    assert first.study.identity_sha256 == aliased.study.identity_sha256
    assert (
        first.study.methods[0].identity_sha256
        == aliased.study.methods[0].identity_sha256
    )
    assert label_set_identity(
        first.study.identity_sha256, first.study.methods[0].identity_sha256
    ) == label_set_identity(
        aliased.study.identity_sha256, aliased.study.methods[0].identity_sha256
    )
    ordered_payload = first.model_dump(mode="json")
    ordered_payload["study"]["selection"]["groups"] = {
        "z-group": ["occ-z", "occ-a"],
        "a-group": ["occ-m"],
    }
    reordered_payload = first.model_dump(mode="json")
    reordered_payload["study"]["selection"]["groups"] = {
        "a-group": ["occ-m"],
        "z-group": ["occ-a", "occ-z"],
    }
    assert (
        GraphLabelingSpec.model_validate(ordered_payload).study.identity_sha256
        == GraphLabelingSpec.model_validate(reordered_payload).study.identity_sha256
    )
    assert (
        ExecutionSpec(mode="local").identity_sha256
        != ExecutionSpec(mode="materialize_only").identity_sha256
    )

    duplicate_method_payload = first.model_dump(mode="json")
    duplicate_method = dict(duplicate_method_payload["study"]["methods"][0])
    duplicate_method["method_id"] = "deterministic-alias"
    duplicate_method_payload["study"]["methods"].append(duplicate_method)
    with pytest.raises(ValueError, match="unique semantic identities"):
        GraphLabelingSpec.model_validate(duplicate_method_payload)


def test_deterministic_execute_and_complete_overlay_export(tmp_path: Path) -> None:
    site, hashes = _site(tmp_path)
    run = tmp_path / "run"
    prepare(_spec(site, hashes), run)
    receipt = execute(
        run,
        "deterministic-evidence-summary-v1",
        ExecutionSpec(mode="local"),
    )
    assert receipt.label_count == 1
    assert (
        status(run)["methods"]["deterministic-evidence-summary-v1"]["label_count"] == 1
    )
    destination = tmp_path / "overlay.json"
    exported = export_overlay(
        run,
        receipt.label_set_id,
        site,
        destination,
    )
    assert exported.selected_count == 1
    assert exported.unselected_count == 2
    overlay = json.loads(destination.read_text())
    assert overlay["schema_version"] == LABEL_SET_SCHEMA
    records = overlay["labels_by_trace"]["trace-test"]
    assert len(records) == 3
    by_id = {record["occurrence_id"]: record for record in records}
    assert by_id["occ-selected"]["status"] == "provisional_label"
    assert by_id["occ-unselected"]["status"] == "not_selected"
    assert overlay["content_hash"] == canonical_sha256(
        {key: value for key, value in overlay.items() if key != "content_hash"}
    )
    manifest = json.loads((site / "viewer-manifest.json").read_text())
    (site / "viewer-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        export_overlay(run, receipt.label_set_id, site, tmp_path / "drifted.json")


def test_structured_execution_is_materialize_only_and_rejects_unknown_citations(
    tmp_path: Path,
) -> None:
    site, hashes = _site(tmp_path)
    run = tmp_path / "run"
    spec = _spec(site, hashes)
    prepare(spec, run)
    materialized = execute(
        run,
        "structured-llm-graph-role-v1",
        ExecutionSpec(mode="materialize_only"),
    )
    assert materialized.state == "materialized"
    with pytest.raises(ValueError, match="intentionally unavailable"):
        execute(
            run,
            "structured-llm-graph-role-v1",
            ExecutionSpec(mode="local"),
        )

    packet = EvidencePacket.model_validate(
        json.loads((run / "evidence/occ-selected.json").read_text())
    )
    method = next(
        item
        for item in spec.study.methods
        if item.method_id == "structured-llm-graph-role-v1"
    )
    assert allowed_evidence_ids(packet)
    with pytest.raises(ValueError, match="unknown evidence"):
        normalize_structured_label(
            {
                "status": "provisional_label",
                "label": "invented role",
                "reads_from": ["invented source"],
                "cited_evidence_ids": ["ev-not-real"],
                "claim_citations": {
                    "label": ["ev-not-real"],
                    "reads_from": ["ev-not-real"],
                    "apparent_role": ["ev-not-real"],
                    "target_effect": ["ev-not-real"],
                    "rationale": ["ev-not-real"],
                },
                "apparent_role": "invented",
                "target_effect": "supports",
                "rationale": "invented",
                "alternative_hypothesis": None,
                "limitations": [],
                "confidence": 0.9,
            },
            packet=packet,
            method=method,
            logical_request_sha256="b" * 64,
            result_sha256=hashlib.sha256(b"result").hexdigest(),
        )
    with pytest.raises(ValueError, match="require cited evidence"):
        normalize_structured_label(
            {
                "status": "provisional_label",
                "label": "uncited role",
                "reads_from": ["some text"],
                "cited_evidence_ids": [],
                "claim_citations": {},
                "apparent_role": "uncited",
                "target_effect": "unclear",
                "rationale": "uncited",
                "alternative_hypothesis": None,
                "limitations": [],
                "confidence": 0.2,
            },
            packet=packet,
            method=method,
            logical_request_sha256="b" * 64,
            result_sha256=hashlib.sha256(b"result").hexdigest(),
        )

    source_evidence_id = sorted(reads_from_evidence_ids(packet))[0]
    direct_target_evidence_id = packet.direct_target_edges[0]["evidence_id"]
    payload = {
        "status": "provisional_label",
        "label": "bounded role",
        "reads_from": ["retained graph input"],
        "cited_evidence_ids": [source_evidence_id, direct_target_evidence_id],
        "claim_citations": {
            "label": [direct_target_evidence_id],
            "reads_from": [direct_target_evidence_id],
            "apparent_role": [direct_target_evidence_id],
            "target_effect": [direct_target_evidence_id],
            "rationale": [direct_target_evidence_id],
            "alternative_hypothesis": [direct_target_evidence_id],
        },
        "apparent_role": "bounded apparent role",
        "target_effect": "supports",
        "rationale": "retained evidence",
        "alternative_hypothesis": "omitted graph structure may matter",
        "limitations": [],
        "confidence": 0.3,
    }
    with pytest.raises(ValueError, match="reads_from citations must reference"):
        normalize_structured_label(
            payload,
            packet=packet,
            method=method,
            logical_request_sha256="b" * 64,
            result_sha256=hashlib.sha256(b"result").hexdigest(),
        )

    payload["claim_citations"]["reads_from"] = [source_evidence_id]
    del payload["claim_citations"]["alternative_hypothesis"]
    with pytest.raises(ValueError, match="alternative_hypothesis"):
        normalize_structured_label(
            payload,
            packet=packet,
            method=method,
            logical_request_sha256="b" * 64,
            result_sha256=hashlib.sha256(b"result").hexdigest(),
        )


def _external_row(run: Path) -> dict[str, object]:
    summary = status(run)
    label_set_id = summary["methods"]["structured-llm-graph-role-v1"]["label_set_id"]
    request = json.loads(
        (run / f"requests/{label_set_id}/occ-selected.json").read_text()
    )
    packet = EvidencePacket.model_validate(
        json.loads((run / "evidence/occ-selected.json").read_text())
    )
    evidence_id = sorted(reads_from_evidence_ids(packet))[0]
    payload = {
        "status": "provisional_label",
        "label": "routes selected local evidence",
        "reads_from": ["the strongest retained source-token evidence"],
        "cited_evidence_ids": [evidence_id],
        "claim_citations": {
            "label": [evidence_id],
            "reads_from": [evidence_id],
            "apparent_role": [evidence_id],
            "target_effect": [evidence_id],
            "rationale": [evidence_id],
            "alternative_hypothesis": [evidence_id],
        },
        "apparent_role": "appears to route local evidence in this trace",
        "target_effect": "unclear",
        "rationale": "bounded packet evidence",
        "alternative_hypothesis": "the retained graph may omit relevant structure",
        "limitations": ["local pruned graph only"],
        "confidence": 0.4,
    }
    return {
        "schema_version": "adag.graph-labeling.external-result.v1",
        "request_id": request["request_id"],
        "logical_request_sha256": request["logical_request_sha256"],
        "evidence_sha256": request["evidence_sha256"],
        "method_sha256": request["method_sha256"],
        "raw_payload": payload,
        "raw_response_sha256": canonical_sha256(payload),
    }


def _rewrite_label_set(
    run: Path,
    label_set_id: str,
    *,
    label_result_sha256: str | None = None,
    binding_result_sha256: str | None = None,
) -> None:
    result_root = run / "label-sets" / label_set_id
    labels_path = result_root / "labels.jsonl"
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    manifest_path = result_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if label_result_sha256 is not None:
        labels[0]["result_sha256"] = label_result_sha256
        labels_path.write_bytes(b"".join(_bytes(label) + b"\n" for label in labels))
        manifest["labels_file_sha256"] = file_sha256(labels_path)
        manifest["labels_content_sha256"] = canonical_sha256(labels)
    if binding_result_sha256 is not None:
        manifest["request_bindings"][0]["raw_response_sha256"] = binding_result_sha256
    manifest["content_hash"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "content_hash"}
    )
    _write(manifest_path, manifest)


def test_external_result_ingestion_is_complete_bound_and_immutable(
    tmp_path: Path,
) -> None:
    site, hashes = _site(tmp_path)
    run = tmp_path / "run"
    prepare(_spec(site, hashes), run)
    row = _external_row(run)
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(json.dumps(row) + "\n")
    receipt = ingest_results(run, "structured-llm-graph-role-v1", result_path)
    assert receipt.label_count == 1
    assert receipt.label_set_id.startswith("occ-role-")
    result_manifest = json.loads(
        (run / f"label-sets/{receipt.label_set_id}/manifest.json").read_text()
    )
    assert result_manifest["content_hash"] == canonical_sha256(
        {key: value for key, value in result_manifest.items() if key != "content_hash"}
    )
    assert status(run)["methods"]["structured-llm-graph-role-v1"]["label_count"] == 1
    labels_path = run / f"label-sets/{receipt.label_set_id}/labels.jsonl"
    labels_path.write_bytes(labels_path.read_bytes() + b" \n")
    with pytest.raises(ValueError, match="labels file hash mismatch"):
        status(run)

    duplicate_run = tmp_path / "duplicate-run"
    prepare(_spec(site, hashes), duplicate_run)
    duplicate_row = _external_row(duplicate_run)
    duplicate_path = tmp_path / "duplicates.jsonl"
    duplicate_path.write_text(
        json.dumps(duplicate_row) + "\n" + json.dumps(duplicate_row) + "\n"
    )
    with pytest.raises(ValueError, match="repeat request_id"):
        ingest_results(duplicate_run, "structured-llm-graph-role-v1", duplicate_path)

    tampered_run = tmp_path / "tampered-run"
    prepare(_spec(site, hashes), tampered_run)
    tampered_row = _external_row(tampered_run)
    tampered_row["raw_response_sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered.jsonl"
    tampered_path.write_text(json.dumps(tampered_row) + "\n")
    with pytest.raises(ValueError, match="raw response hash mismatch"):
        ingest_results(tampered_run, "structured-llm-graph-role-v1", tampered_path)

    label_binding_run = tmp_path / "label-binding-run"
    prepare(_spec(site, hashes), label_binding_run)
    label_binding_path = tmp_path / "label-binding.jsonl"
    label_binding_path.write_text(json.dumps(_external_row(label_binding_run)) + "\n")
    label_binding_receipt = ingest_results(
        label_binding_run,
        "structured-llm-graph-role-v1",
        label_binding_path,
    )
    _rewrite_label_set(
        label_binding_run,
        label_binding_receipt.label_set_id,
        label_result_sha256="1" * 64,
    )
    with pytest.raises(
        ValueError, match=r"result/request binding mismatch|finalization receipt drift"
    ):
        status(label_binding_run)

    manifest_binding_run = tmp_path / "manifest-binding-run"
    prepare(_spec(site, hashes), manifest_binding_run)
    manifest_binding_path = tmp_path / "manifest-binding.jsonl"
    manifest_binding_path.write_text(
        json.dumps(_external_row(manifest_binding_run)) + "\n"
    )
    manifest_binding_receipt = ingest_results(
        manifest_binding_run,
        "structured-llm-graph-role-v1",
        manifest_binding_path,
    )
    _rewrite_label_set(
        manifest_binding_run,
        manifest_binding_receipt.label_set_id,
        binding_result_sha256="2" * 64,
    )
    with pytest.raises(
        ValueError, match=r"result/request binding mismatch|finalization receipt drift"
    ):
        status(manifest_binding_run)


def test_file_bearing_ids_reject_traversal(tmp_path: Path) -> None:
    site, hashes = _site(tmp_path)
    payload = _spec(site, hashes).model_dump(mode="json")
    payload["study"]["methods"][0]["method_id"] = "../escape"
    with pytest.raises(ValueError, match="safe identifier"):
        GraphLabelingSpec.model_validate(payload)
    payload = _spec(site, hashes).model_dump(mode="json")
    payload["study"]["selection"]["groups"] = {"group": ["../escape"]}
    with pytest.raises(ValueError, match="safe identifier"):
        GraphLabelingSpec.model_validate(payload)

    payload = _spec(site, hashes).model_dump(mode="json")
    payload["study"]["methods"][1]["labeler"]["provider_parameters"] = {
        "api_key": "must-not-enter-a-run-artifact"
    }
    with pytest.raises(ValueError, match="secret provider parameter"):
        GraphLabelingSpec.model_validate(payload)


class _FakeOpenAIBatchTransport:
    def __init__(
        self,
        output: bytes = b"",
        *,
        fail_create: bool = False,
        fail_upload: bool = False,
    ) -> None:
        self.output = output
        self.fail_create = fail_create
        self.fail_upload = fail_upload
        self.create_calls = 0
        self.metadata: dict[str, str] = {}
        self.input_bytes = b""
        self.aggregate_output_tokens = 40

    def upload_batch_input(self, path: Path) -> dict[str, object]:
        self.input_bytes = path.read_bytes()
        assert self.input_bytes
        if self.fail_upload:
            raise RuntimeError("simulated ambiguous upload")
        return {"id": "file-input", "purpose": "batch", "bytes": path.stat().st_size}

    def retrieve_file(self, file_id: str) -> dict[str, object]:
        assert file_id == "file-input"
        return {"id": file_id, "purpose": "batch", "bytes": len(self.input_bytes)}

    def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str],
    ) -> dict[str, object]:
        self.create_calls += 1
        self.metadata = metadata
        if self.fail_create:
            raise RuntimeError("simulated ambiguous create")
        return {
            "id": "batch-test",
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
            "metadata": metadata,
            "status": "validating",
            "output_file_id": None,
            "error_file_id": None,
        }

    def retrieve_batch(self, batch_id: str) -> dict[str, object]:
        assert batch_id == "batch-test"
        return {
            "id": batch_id,
            "input_file_id": "file-input",
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": self.metadata,
            "status": "completed",
            "output_file_id": "file-output",
            "error_file_id": None,
            "request_counts": {"total": 1, "completed": 1, "failed": 0},
            "usage": {
                "input_tokens": 100,
                "output_tokens": self.aggregate_output_tokens,
                "total_tokens": 100 + self.aggregate_output_tokens,
            },
        }

    def download_file(self, file_id: str) -> bytes:
        if file_id == "file-input":
            return self.input_bytes
        if file_id == "file-output":
            return self.output
        raise AssertionError(file_id)


def _hold_flock(path: str, ready: Any, release: Any) -> None:
    with Path(path).open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait(10)


def _openai_batch_spec(site: Path, hashes: dict[str, str]) -> GraphLabelingSpec:
    payload = _spec(site, hashes).model_dump(mode="json")
    payload["study"]["methods"][1]["labeler"]["model"] = "gpt-5.6-terra"
    payload["study"]["methods"][1]["labeler"]["reasoning"] = {"effort": "medium"}
    payload["study"]["methods"][1]["prompt_version"] = "structured-llm-graph-role-v2"
    return GraphLabelingSpec.model_validate(payload)


def _batch_output(run: Path) -> bytes:
    external = _external_row(run)
    item = {
        "custom_id": external["request_id"],
        "error": None,
        "response": {
            "status_code": 200,
            "request_id": "req-remote",
            "body": {
                "id": "resp-remote",
                "status": "completed",
                "error": None,
                "model": "gpt-5.6-terra-2026-08-01",
                "service_tier": "default",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(external["raw_payload"]),
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 40,
                    "total_tokens": 140,
                },
            },
        },
    }
    return _bytes(item) + b"\n"


def test_openai_batch_adapter_is_cost_guarded_idempotent_and_collects(
    tmp_path: Path,
) -> None:
    site, hashes = _site(tmp_path)
    run = tmp_path / "batch-run"
    prepare(_openai_batch_spec(site, hashes), run)
    with pytest.raises(ValueError, match="cost guard"):
        prepare_openai_batch(run, "structured-llm-graph-role-v1", max_cost_usd=0.000001)
    plan = prepare_openai_batch(run, "structured-llm-graph-role-v1", max_cost_usd=10.0)
    batch_root = run / "openai-batches" / plan["label_set_id"]
    input_rows = [
        json.loads(line)
        for line in (batch_root / "input.jsonl").read_text().splitlines()
    ]
    assert input_rows[0]["url"] == "/v1/responses"
    assert input_rows[0]["body"]["store"] is False
    assert input_rows[0]["body"]["text"]["format"]["strict"] is True
    serialized_input = json.dumps(input_rows[0]["body"]["input"])
    assert "selection_group" not in serialized_input
    assert "direct_target_parent" not in serialized_input
    assert '"coverage"' not in serialized_input
    assert '"path_search"' not in serialized_input
    assert "coverage_and_truncation" not in serialized_input
    assert "target_path_search" not in serialized_input
    assert "subject_identity" not in serialized_input
    assert "trace_unit_id" not in serialized_input
    assert "source_trace_sha256" not in serialized_input
    assert plan["price_snapshot_id"] == "official-api-prices-2026-08-25-graph-labeling"

    fake = _FakeOpenAIBatchTransport()
    first = submit_openai_batch(
        run,
        "structured-llm-graph-role-v1",
        max_cost_usd=10.0,
        transport=fake,
    )
    second = submit_openai_batch(
        run,
        "structured-llm-graph-role-v1",
        max_cost_usd=10.0,
        transport=fake,
    )
    assert first == second
    assert fake.create_calls == 1
    assert (batch_root / "upload.json").is_file()
    assert (batch_root / "upload-intent.json").is_file()
    assert (batch_root / "create-intent.json").is_file()

    status_receipt = openai_batch_status(
        run, "structured-llm-graph-role-v1", transport=fake
    )
    assert status_receipt["usage"]["total_tokens"] == 140
    fake.output = _batch_output(run)
    collection = collect_openai_batch(
        run,
        "structured-llm-graph-role-v1",
        transport=fake,
        finalize=True,
    )
    assert collection["request_count"] == 1
    assert collection["per_request_remote_metadata"]
    assert collection["actual_cost_usd"] <= collection["authorized_cost_guard_usd"]
    attempts = list((batch_root / "collection-attempts").iterdir())
    assert len(attempts) == 1
    assert (attempts[0] / "intent.json").is_file()
    assert (attempts[0] / "snapshot.json").is_file()
    assert (attempts[0] / "download.json").is_file()
    assert (batch_root / "collection/provider-output.jsonl").is_file()
    assert (batch_root / "collection/finalization.json").is_file()
    result_manifest = json.loads(
        (run / f"label-sets/{plan['label_set_id']}/manifest.json").read_text()
    )
    assert result_manifest["result_source"]["kind"] == "openai_batch_collection_v1"
    assert result_manifest["result_source"]["provider_exact_models"] == [
        "gpt-5.6-terra-2026-08-01"
    ]
    finalization = json.loads((batch_root / "collection/finalization.json").read_text())
    assert finalization["label_set_manifest_sha256"] == result_manifest["content_hash"]
    assert finalization["output_file_sha256"] == collection["output_file_sha256"]
    fake.output = b"provider output must not be fetched again"
    repeated = collect_openai_batch(
        run,
        "structured-llm-graph-role-v1",
        transport=fake,
        finalize=True,
    )
    assert repeated == collection
    assert len(list((batch_root / "collection-attempts").iterdir())) == 1
    assert status(run)["methods"]["structured-llm-graph-role-v1"]["label_count"] == 1


def test_openai_batch_adapter_rejects_missing_custom_id(tmp_path: Path) -> None:
    site, hashes = _site(tmp_path)
    run = tmp_path / "missing-run"
    prepare(_openai_batch_spec(site, hashes), run)
    fake = _FakeOpenAIBatchTransport()
    submit_openai_batch(
        run,
        "structured-llm-graph-role-v1",
        max_cost_usd=10.0,
        transport=fake,
    )
    foreign = json.loads(_batch_output(run))
    foreign["custom_id"] = "req-foreign"
    fake.output = _bytes(foreign) + b"\n"
    with pytest.raises(ValueError, match="unknown custom_id"):
        collect_openai_batch(run, "structured-llm-graph-role-v1", transport=fake)

    summary = status(run)
    label_set_id = summary["methods"]["structured-llm-graph-role-v1"]["label_set_id"]
    request = PromptRequest.model_validate(
        json.loads((run / f"requests/{label_set_id}/occ-selected.json").read_text())
    )
    valid = _batch_output(run)
    with pytest.raises(ValueError, match="repeats custom_id"):
        _parse_output(valid + valid, {request.request_id: request})
    with pytest.raises(ValueError, match="omitted custom_ids"):
        _parse_output(b"", {request.request_id: request})
    malformed = json.loads(valid)
    malformed["response"]["body"]["output"][0]["content"][0]["text"] = "not-json"
    with pytest.raises(ValueError, match="malformed structured output"):
        _parse_output(_bytes(malformed) + b"\n", {request.request_id: request})
    failed = json.loads(valid)
    failed["response"]["status_code"] = 500
    with pytest.raises(ValueError, match="request failed"):
        _parse_output(_bytes(failed) + b"\n", {request.request_id: request})


def test_position_120_v2_has_separate_luna_and_terra_batches() -> None:
    spec = GraphLabelingSpec.model_validate(
        json.loads(
            Path(
                "scripts/bonafide/configs/graph_labeling/"
                "qwen-position-120-occurrence-role-v2.json"
            ).read_text()
        )
    )
    methods = {method.method_id: method for method in spec.study.methods}
    assert set(methods) == {
        "structured-llm-graph-role-luna-medium-v2",
        "structured-llm-graph-role-terra-medium-v2",
    }
    assert {method.labeler.model for method in methods.values() if method.labeler} == {
        "gpt-5.6-luna",
        "gpt-5.6-terra",
    }
    assert all(
        method.labeler is not None
        and method.prompt_version == "structured-llm-graph-role-v2"
        and method.labeler.reasoning == {"effort": "medium"}
        and method.labeler.max_output_tokens == 4000
        for method in methods.values()
    )


def test_labeler_rejects_reasoning_with_temperature(tmp_path: Path) -> None:
    site, hashes = _site(tmp_path)
    payload = _spec(site, hashes).model_dump(mode="json")
    payload["study"]["methods"][1]["labeler"]["reasoning"] = {"effort": "medium"}
    payload["study"]["methods"][1]["labeler"]["temperature"] = 0.2
    with pytest.raises(ValueError, match="reasoning and temperature"):
        GraphLabelingSpec.model_validate(payload)


def test_openai_batch_recovery_proves_remote_input(tmp_path: Path) -> None:
    site, hashes = _site(tmp_path)
    run = tmp_path / "recovery-run"
    prepare(_openai_batch_spec(site, hashes), run)
    fake = _FakeOpenAIBatchTransport(fail_create=True)
    with pytest.raises(RuntimeError, match="ambiguous create"):
        submit_openai_batch(
            run,
            "structured-llm-graph-role-v1",
            max_cost_usd=10.0,
            transport=fake,
        )
    fake.fail_create = False
    recovered = recover_openai_batch(
        run,
        "structured-llm-graph-role-v1",
        "batch-test",
        transport=fake,
    )
    assert recovered["receipt_mode"] == "recovered"
    assert recovered["batch_id"] == "batch-test"
    status_receipt = openai_batch_status(
        run, "structured-llm-graph-role-v1", transport=fake
    )
    assert status_receipt["batch_id"] == "batch-test"


def test_openai_batch_upload_recovery_is_explicit_and_byte_bound(
    tmp_path: Path,
) -> None:
    site, hashes = _site(tmp_path)
    run = tmp_path / "upload-recovery-run"
    prepare(_openai_batch_spec(site, hashes), run)
    fake = _FakeOpenAIBatchTransport(fail_upload=True)
    with pytest.raises(RuntimeError, match="ambiguous upload"):
        submit_openai_batch(
            run,
            "structured-llm-graph-role-v1",
            max_cost_usd=10.0,
            transport=fake,
        )
    fake.fail_upload = False
    recovered = recover_openai_upload(
        run,
        "structured-llm-graph-role-v1",
        "file-input",
        transport=fake,
    )
    assert recovered["receipt_mode"] == "recovered"
    submitted = submit_openai_batch(
        run,
        "structured-llm-graph-role-v1",
        max_cost_usd=10.0,
        transport=fake,
    )
    assert submitted["attempt_id"] == recovered["attempt_id"]


def test_openai_batch_gate_and_usage_reconciliation_fail_closed(tmp_path: Path) -> None:
    site, hashes = _site(tmp_path)
    run = tmp_path / "gate-run"
    prepare(_openai_batch_spec(site, hashes), run)
    fake = _FakeOpenAIBatchTransport()
    submission = submit_openai_batch(
        run,
        "structured-llm-graph-role-v1",
        max_cost_usd=10.0,
        transport=fake,
    )
    lock = (
        run
        / "openai-batches/.locks"
        / f"{submission['remote']['metadata']['graph_label_set']}.lock"
    )
    process_context = multiprocessing.get_context("spawn")
    ready = process_context.Event()
    release = process_context.Event()
    holder = process_context.Process(
        target=_hold_flock, args=(str(lock), ready, release)
    )
    holder.start()
    assert ready.wait(5)
    try:
        with pytest.raises(RuntimeError, match="lifecycle gate"):
            openai_batch_status(run, "structured-llm-graph-role-v1", transport=fake)
    finally:
        release.set()
        holder.join(5)
    assert holder.exitcode == 0
    fake.output = _batch_output(run)
    fake.aggregate_output_tokens = 41
    with pytest.raises(ValueError, match="aggregate usage mismatch"):
        collect_openai_batch(run, "structured-llm-graph-role-v1", transport=fake)
    attempts_root = (
        run
        / "openai-batches"
        / submission["remote"]["metadata"]["graph_label_set"]
        / "collection-attempts"
    )
    [attempt] = list(attempts_root.iterdir())
    assert (attempt / "intent.json").is_file()
    assert (attempt / "snapshot.json").is_file()
    assert (attempt / "output.jsonl").is_file()
    assert not (attempt.parent.parent / "collection").exists()
