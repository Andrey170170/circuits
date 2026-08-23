from __future__ import annotations

import hashlib
import http.client
import json
import threading
from pathlib import Path

import pytest
from circuits.observatory import CATALOG_SCHEMA, MANIFEST_SCHEMA
from circuits.observatory.server import DEFAULT_HOST, make_server, validate_site_bundle


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _site(root: Path) -> Path:
    site = root / "site"
    (site / "traces").mkdir(parents=True)
    (site / "label-sets").mkdir()
    catalog = {
        "schema_version": CATALOG_SCHEMA,
        "claim_boundary": "test",
        "model": {"model_id": "m", "model_revision": "r"},
        "traces": [],
    }
    label_index = {
        "schema_version": "adag.observatory.label-set-index.v1",
        "label_sets": [],
    }
    _write_json(site / "catalog.json", catalog)
    _write_json(site / "label-sets" / "index.json", label_index)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": "2026-01-01T00:00:00+00:00",
        "file_hash_algorithm": "sha256-canonical-json-v1",
        "catalog_schema": CATALOG_SCHEMA,
        "trace_schema": "adag.observatory.trace-graph.v1",
        "trace_count": 0,
        "artifact_ids": [],
        "files": {
            "catalog.json": _canonical_hash(catalog),
            "label-sets/index.json": _canonical_hash(label_index),
        },
    }
    manifest["content_hash"] = _canonical_hash(manifest)
    _write_json(site / "viewer-manifest.json", manifest)
    return site


def test_site_validation_rejects_canonical_json_tampering(tmp_path: Path) -> None:
    site = _site(tmp_path)
    validate_site_bundle(site)
    catalog = json.loads((site / "catalog.json").read_text())
    catalog["claim_boundary"] = "tampered"
    _write_json(site / "catalog.json", catalog)
    with pytest.raises(ValueError, match="canonical JSON hash mismatch"):
        validate_site_bundle(site)


def _request(
    server, method: str, path: str, body: object | None = None
) -> tuple[int, dict[str, object], dict[str, str]]:
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    payload = None if body is None else json.dumps(body)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    data = json.loads(response.read())
    response_headers = dict(response.getheaders())
    status = response.status
    connection.close()
    return status, data, response_headers


def test_server_defaults_loopback_and_rejects_traversal(tmp_path: Path) -> None:
    site = _site(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index.html").write_text("viewer")
    (assets / "styles.css").write_text("body {}")
    server = make_server(
        site_root=site,
        state_root=tmp_path / "state",
        port=0,
        assets_root=assets,
    )
    assert server.server_address[0] == DEFAULT_HOST
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _body, _headers = _request(server, "GET", "/%2e%2e/secret.json")
        assert status == 400
        status, catalog, headers = _request(server, "GET", "/api/v1/catalog")
        assert status == 200
        assert catalog["schema_version"] == CATALOG_SCHEMA
        assert "ETag" in headers
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request("GET", "/assets/styles.css")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b"body {}"
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_workspace_updates_are_atomic_and_revision_checked(tmp_path: Path) -> None:
    server = make_server(
        site_root=_site(tmp_path), state_root=tmp_path / "state", port=0
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, empty, _headers = _request(server, "GET", "/api/v1/workspaces/review")
        assert status == 200
        assert empty["revision"] == 0
        update = {"expected_revision": 0, "workspace": {"comments": ["first"]}}
        status, saved, _headers = _request(
            server, "PUT", "/api/v1/workspaces/review", update
        )
        assert status == 200
        assert saved["revision"] == 1
        status, conflict, _headers = _request(
            server, "PUT", "/api/v1/workspaces/review", update
        )
        assert status == 409
        assert conflict["current"]["workspace"] == {"comments": ["first"]}
        assert not list((tmp_path / "state" / "workspaces").glob(".*.tmp-*"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_vendor_mapping_is_narrow_and_read_only(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    vendor = tmp_path / "vendor"
    assets.mkdir()
    vendor.mkdir()
    (assets / "index.html").write_text("viewer")
    (vendor / "d3.js").write_text("window.d3 = {};")
    server = make_server(
        site_root=_site(tmp_path),
        state_root=tmp_path / "state",
        port=0,
        assets_root=assets,
        vendor_root=vendor,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request("GET", "/vendor/d3.js")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b"window.d3 = {};"
        connection.close()
        status, _body, _headers = _request(server, "GET", "/vendor/secret.js")
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_unknown_trace_and_bad_ids_do_not_escape_site_root(tmp_path: Path) -> None:
    server = make_server(
        site_root=_site(tmp_path), state_root=tmp_path / "state", port=0
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _body, _headers = _request(server, "GET", "/api/v1/traces/missing")
        assert status == 404
        status, _body, _headers = _request(server, "GET", "/api/v1/workspaces/%2e%2e")
        assert status in {400, 404}
        assert not (tmp_path / "state" / "..json").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
