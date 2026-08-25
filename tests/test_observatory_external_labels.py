from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_json, canonical_sha256
from circuits.analysis.bonafide.identity import BASIS_KEY_SCHEMA, POLARITY_DERIVATION
from circuits.observatory import (
    CATALOG_SCHEMA,
    LABEL_SET_SCHEMA,
    MANIFEST_SCHEMA,
    TRACE_GRAPH_SCHEMA,
)
from circuits.observatory.external_labels import install_label_set
from circuits.observatory.server import validate_site_bundle


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")
    return canonical_sha256(value)


def _source_site(root: Path) -> Path:
    site = root / "source-site"
    trace = {
        "schema_version": TRACE_GRAPH_SCHEMA,
        "artifact": {"artifact_id": "trace-one", "source_hash": "trace-hash"},
        "model": {"model_id": "example/model", "model_revision": "revision-1"},
        "nodes": [
            {
                "id": "occ-one",
                "basis_id": "basis-one",
                "basis": {"schema_version": BASIS_KEY_SCHEMA},
            },
            {
                "id": "occ-two",
                "basis_id": "basis-two",
                "basis": {"schema_version": BASIS_KEY_SCHEMA},
            },
        ],
    }
    catalog = {
        "schema_version": CATALOG_SCHEMA,
        "claim_boundary": "test",
        "model": {"model_id": "example/model", "model_revision": "revision-1"},
        "traces": [{"artifact_id": "trace-one", "source_hash": "trace-hash"}],
    }
    index = {"schema_version": "adag.observatory.label-set-index.v1", "label_sets": []}
    files = {
        "catalog.json": _write_json(site / "catalog.json", catalog),
        "traces/trace-one.json": _write_json(site / "traces/trace-one.json", trace),
        "label-sets/index.json": _write_json(site / "label-sets/index.json", index),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": "2026-01-01T00:00:00+00:00",
        "file_hash_algorithm": "sha256-canonical-json-v1",
        "catalog_schema": CATALOG_SCHEMA,
        "trace_schema": TRACE_GRAPH_SCHEMA,
        "trace_count": 1,
        "artifact_ids": ["trace-one"],
        "files": files,
    }
    manifest["content_hash"] = canonical_sha256(manifest)
    _write_json(site / "viewer-manifest.json", manifest)
    (site / "assets").mkdir()
    (site / "assets" / "local.css").write_text("body {}", encoding="utf-8")
    validate_site_bundle(site)
    return site


def _label_set(path: Path, *, label: str = "routes arithmetic") -> Path:
    value = {
        "schema_version": LABEL_SET_SCHEMA,
        "label_set_id": "external-role-v1",
        "name": "External graph roles",
        "synthetic": False,
        "warning": "Exploratory graph-local hypotheses.",
        "method": "graph-role-v1",
        "model_id": "example/model",
        "model_revision": "revision-1",
        "basis_schema": BASIS_KEY_SCHEMA,
        "polarity_derivation": POLARITY_DERIVATION,
        "source_trace_hashes": {"trace-one": "trace-hash"},
        "labels_by_trace": {
            "trace-one": [
                {
                    "occurrence_id": "occ-one",
                    "basis_id": "basis-one",
                    "label": label,
                    "status": "provisional_label",
                    "confidence": 0.6,
                },
                {
                    "occurrence_id": "occ-two",
                    "basis_id": "basis-two",
                    "label": "insufficient_evidence",
                    "status": "insufficient_evidence",
                    "confidence": None,
                },
            ]
        },
    }
    value["content_hash"] = canonical_sha256(value)
    _write_json(path, value)
    return path


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_install_builds_valid_derived_site_without_changing_source(
    tmp_path: Path,
) -> None:
    source = _source_site(tmp_path)
    before = _tree_hashes(source)
    label_set = _label_set(tmp_path / "external.json")
    destination = tmp_path / "derived" / "site"

    receipt = install_label_set(source, label_set, destination)

    assert receipt["idempotent"] is False
    assert receipt["label_set_id"] == "external-role-v1"
    assert destination.is_dir()
    assert (destination / "assets" / "local.css").read_text() == "body {}"
    assert before == _tree_hashes(source)
    validate_site_bundle(destination)
    index = json.loads((destination / "label-sets" / "index.json").read_text())
    assert [item["label_set_id"] for item in index["label_sets"]] == [
        "external-role-v1"
    ]


def test_install_rejects_mismatched_source_binding(tmp_path: Path) -> None:
    source = _source_site(tmp_path)
    label_path = _label_set(tmp_path / "external.json")
    value = json.loads(label_path.read_text())
    value["source_trace_hashes"]["trace-one"] = "wrong-hash"
    value.pop("content_hash")
    value["content_hash"] = canonical_sha256(value)
    _write_json(label_path, value)
    destination = tmp_path / "derived"

    with pytest.raises(ValueError, match="source binding mismatch"):
        install_label_set(source, label_path, destination)

    assert not destination.exists()


def test_install_rejects_label_set_id_collision(tmp_path: Path) -> None:
    source = _source_site(tmp_path)
    first = _label_set(tmp_path / "first.json")
    first_destination = tmp_path / "first-derived"
    install_label_set(source, first, first_destination)
    conflicting = _label_set(tmp_path / "conflicting.json", label="formats a number")

    with pytest.raises(ValueError, match="collision with different content"):
        install_label_set(first_destination, conflicting, tmp_path / "second-derived")

    assert not (tmp_path / "second-derived").exists()


def test_install_is_idempotent_for_identical_existing_destination(
    tmp_path: Path,
) -> None:
    source = _source_site(tmp_path)
    label_set = _label_set(tmp_path / "external.json")
    destination = tmp_path / "derived"
    first = install_label_set(source, label_set, destination)
    before = _tree_hashes(destination)

    second = install_label_set(source, label_set, destination)

    assert second["idempotent"] is True
    assert second["destination_bundle_hash"] == first["destination_bundle_hash"]
    assert second["receipt_hash"] != first["receipt_hash"]
    assert before == _tree_hashes(destination)
