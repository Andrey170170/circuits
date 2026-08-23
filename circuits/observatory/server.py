"""CPU-only loopback-oriented HTTP server for safe observatory JSON and assets."""

from __future__ import annotations

import gzip
import hashlib
import json
import mimetypes
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from circuits.observatory import (
    CATALOG_SCHEMA,
    LABEL_SET_SCHEMA,
    MANIFEST_SCHEMA,
    TRACE_GRAPH_SCHEMA,
    WORKSPACE_SCHEMA,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8032
MAX_WORKSPACE_BYTES = 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")


def _safe_id(value: str) -> bool:
    return bool(_SAFE_ID.fullmatch(value))


def _safe_static_path(raw_path: str) -> PurePosixPath | None:
    try:
        decoded = unquote(raw_path, errors="strict")
    except UnicodeError:
        return None
    if "\\" in decoded or "\x00" in decoded:
        return None
    relative = decoded.lstrip("/") or "index.html"
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _read_json_object(path: Path, expected_schema: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable server JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"server JSON must be an object: {path.name}")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError(f"unsupported server JSON schema: {path.name}")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)[:-1]).hexdigest()


def validate_site_bundle(site_root: Path) -> None:
    """Fail closed if immutable JSON or its scientific identity bindings drift."""

    site = site_root.resolve()
    manifest = _read_json_object(site / "viewer-manifest.json", MANIFEST_SCHEMA)
    manifest_core = dict(manifest)
    recorded_manifest_hash = manifest_core.pop("content_hash", None)
    if recorded_manifest_hash != _canonical_sha256(manifest_core):
        raise ValueError("viewer manifest content hash mismatch")
    if manifest.get("file_hash_algorithm") != "sha256-canonical-json-v1":
        raise ValueError("unsupported viewer file hash algorithm")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("viewer manifest files must be a nonempty object")

    loaded: dict[str, dict[str, Any]] = {}
    for relative_name, recorded_hash in files.items():
        relative = _safe_static_path(f"/{relative_name}") if isinstance(relative_name, str) else None
        if relative is None or not isinstance(recorded_hash, str):
            raise ValueError("viewer manifest contains an invalid file entry")
        candidate = site.joinpath(*relative.parts).resolve()
        try:
            candidate.relative_to(site)
        except ValueError as error:
            raise ValueError("viewer manifest file escapes site root") from error
        value = _read_json_object(candidate)
        if _canonical_sha256(value) != recorded_hash:
            raise ValueError(f"viewer canonical JSON hash mismatch: {relative_name}")
        loaded[relative_name] = value

    catalog = loaded.get("catalog.json")
    if catalog is None or catalog.get("schema_version") != CATALOG_SCHEMA:
        raise ValueError("viewer catalog is absent or invalid")
    model = catalog.get("model")
    if not isinstance(model, dict):
        raise ValueError("viewer catalog lacks model identity")
    traces: dict[str, dict[str, Any]] = {}
    trace_identities: dict[str, dict[str, str]] = {}
    source_hashes: dict[str, str] = {}
    for item in catalog.get("traces", []):
        artifact_id = item.get("artifact_id")
        if not isinstance(artifact_id, str) or not _safe_id(artifact_id):
            raise ValueError("viewer catalog contains an invalid artifact id")
        trace = loaded.get(f"traces/{artifact_id}.json")
        if trace is None or trace.get("schema_version") != TRACE_GRAPH_SCHEMA:
            raise ValueError("viewer catalog trace is absent or invalid")
        if trace.get("model", {}).get("model_id") != model.get("model_id") or trace.get(
            "model", {}
        ).get("model_revision") != model.get("model_revision"):
            raise ValueError("viewer trace model identity mismatch")
        source_hash = trace.get("artifact", {}).get("source_hash")
        if source_hash != item.get("source_hash"):
            raise ValueError("viewer trace source hash mismatch")
        traces[artifact_id] = trace
        source_hashes[artifact_id] = source_hash
        trace_identities[artifact_id] = {
            node["id"]: node["basis_id"]
            for node in trace.get("nodes", [])
            if isinstance(node, dict) and "id" in node and "basis_id" in node
        }

    label_index = loaded.get("label-sets/index.json")
    if label_index is None or label_index.get("schema_version") != "adag.observatory.label-set-index.v1":
        raise ValueError("viewer label-set index is absent or invalid")
    for descriptor in label_index.get("label_sets", []):
        label_set_id = descriptor.get("label_set_id")
        if not isinstance(label_set_id, str) or not _safe_id(label_set_id):
            raise ValueError("viewer label-set id is invalid")
        label_set = loaded.get(f"label-sets/{label_set_id}.json")
        if label_set is None or label_set.get("schema_version") != LABEL_SET_SCHEMA:
            raise ValueError("viewer label set is absent or invalid")
        core = dict(label_set)
        recorded_content_hash = core.pop("content_hash", None)
        if recorded_content_hash != _canonical_sha256(core) or recorded_content_hash != descriptor.get(
            "content_hash"
        ):
            raise ValueError("viewer label-set content hash mismatch")
        if label_set.get("model_id") != model.get("model_id") or label_set.get(
            "model_revision"
        ) != model.get("model_revision"):
            raise ValueError("viewer label-set model identity mismatch")
        basis_schemas = {
            node.get("basis", {}).get("schema_version")
            for trace in traces.values()
            for node in trace.get("nodes", [])
            if isinstance(node, dict) and isinstance(node.get("basis"), dict)
        }
        if basis_schemas != {label_set.get("basis_schema")}:
            raise ValueError("viewer label-set basis schema mismatch")
        if (
            label_set.get("polarity_derivation")
            != "activation-sign-nonnegative-positive.v1"
        ):
            raise ValueError("viewer label-set polarity derivation mismatch")
        labels_by_trace = label_set.get("labels_by_trace")
        if not isinstance(labels_by_trace, dict) or set(labels_by_trace) != set(traces):
            raise ValueError("viewer label-set trace coverage mismatch")
        if label_set.get("source_trace_hashes") != source_hashes:
            raise ValueError("viewer label-set source binding mismatch")
        for artifact_id, labels in labels_by_trace.items():
            identities = trace_identities[artifact_id]
            if not isinstance(labels, list) or len(labels) != len(identities):
                raise ValueError("viewer label-set occurrence coverage mismatch")
            for label in labels:
                if not isinstance(label, dict) or identities.get(
                    label.get("occurrence_id")
                ) != label.get("basis_id"):
                    raise ValueError("viewer label-set occurrence/basis binding mismatch")


class ObservatoryServer(ThreadingHTTPServer):
    """HTTP server carrying only prevalidated roots and a workspace lock."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        site_root: Path,
        state_root: Path,
        assets_root: Path | None = None,
        vendor_root: Path | None = None,
    ) -> None:
        self.site_root = site_root.resolve()
        self.state_root = state_root.resolve()
        self.assets_root = (assets_root or Path(__file__).with_name("assets")).resolve()
        self.vendor_root = (
            vendor_root
            or Path(__file__).resolve().parents[1] / "frontend" / "assets" / "lib"
        ).resolve()
        self.workspace_lock = threading.Lock()
        super().__init__(address, ObservatoryRequestHandler)


class ObservatoryRequestHandler(BaseHTTPRequestHandler):
    server: ObservatoryServer

    def log_message(self, format: str, *args: object) -> None:
        # Keep stdlib's useful request log, but send no URL-derived text to formatting.
        super().log_message(format, *args)

    def _send_bytes(
        self,
        payload: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        etag: str | None = None,
    ) -> None:
        tag = etag or hashlib.sha256(payload).hexdigest()
        quoted_tag = f'"{tag}"'
        if self.headers.get("If-None-Match") == quoted_tag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", quoted_tag)
            self.end_headers()
            return
        encoded = payload
        use_gzip = (
            len(payload) >= 1024
            and "gzip" in self.headers.get("Accept-Encoding", "").lower()
        )
        if use_gzip:
            encoded = gzip.compress(payload, compresslevel=5, mtime=0)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("ETag", quoted_tag)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            _json_bytes(value),
            content_type="application/json; charset=utf-8",
            status=status,
        )

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message, "status": int(status)}, status)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/v1/catalog":
            return self._serve_json_file(
                self.server.site_root / "catalog.json", CATALOG_SCHEMA
            )
        if path == "/api/v1/label-sets":
            return self._serve_json_file(
                self.server.site_root / "label-sets" / "index.json",
                "adag.observatory.label-set-index.v1",
            )

        parts = [part for part in path.split("/") if part]
        if parts[:3] == ["api", "v1", "traces"] and len(parts) in {4, 6}:
            artifact_id = parts[3]
            if not _safe_id(artifact_id):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid artifact id")
            trace_path = self.server.site_root / "traces" / f"{artifact_id}.json"
            if len(parts) == 4:
                return self._serve_json_file(trace_path, TRACE_GRAPH_SCHEMA)
            if parts[4] != "nodes" or not _safe_id(parts[5]):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid node route")
            return self._serve_node(trace_path, parts[5])
        if parts[:3] == ["api", "v1", "label-sets"] and len(parts) == 6:
            label_set_id, marker, artifact_id = parts[3:]
            if (
                marker != "traces"
                or not _safe_id(label_set_id)
                or not _safe_id(artifact_id)
            ):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid label-set route")
            return self._serve_label_trace(label_set_id, artifact_id)
        if parts[:3] == ["api", "v1", "workspaces"] and len(parts) == 4:
            return self._get_workspace(parts[3])
        if path.startswith("/api/"):
            return self._error(HTTPStatus.NOT_FOUND, "API route not found")
        return self._serve_static(path)

    def do_PUT(self) -> None:
        path = urlsplit(self.path).path
        parts = [part for part in path.split("/") if part]
        if parts[:3] != ["api", "v1", "workspaces"] or len(parts) != 4:
            return self._error(HTTPStatus.NOT_FOUND, "API route not found")
        return self._put_workspace(parts[3])

    def _serve_json_file(self, path: Path, schema: str) -> None:
        try:
            value = _read_json_object(path, schema)
        except FileNotFoundError:
            return self._error(HTTPStatus.NOT_FOUND, "resource not found")
        except ValueError:
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid site bundle")
        self._send_json(value)
        return None

    def _serve_node(self, trace_path: Path, node_id: str) -> None:
        try:
            trace = _read_json_object(trace_path, TRACE_GRAPH_SCHEMA)
        except FileNotFoundError:
            return self._error(HTTPStatus.NOT_FOUND, "trace not found")
        except ValueError:
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid site bundle")
        nodes = trace.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, dict) and node.get("id") == node_id:
                    incoming = [
                        e for e in trace.get("edges", []) if e.get("target") == node_id
                    ]
                    outgoing = [
                        e for e in trace.get("edges", []) if e.get("source") == node_id
                    ]
                    return self._send_json(
                        {
                            "node": node,
                            "incoming_edges": incoming,
                            "outgoing_edges": outgoing,
                        }
                    )
        return self._error(HTTPStatus.NOT_FOUND, "node not found")

    def _serve_label_trace(self, label_set_id: str, artifact_id: str) -> None:
        path = self.server.site_root / "label-sets" / f"{label_set_id}.json"
        try:
            label_set = _read_json_object(path, LABEL_SET_SCHEMA)
        except FileNotFoundError:
            return self._error(HTTPStatus.NOT_FOUND, "label set not found")
        except ValueError:
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid site bundle")
        labels = label_set.get("labels_by_trace", {}).get(artifact_id)
        if labels is None:
            return self._error(HTTPStatus.NOT_FOUND, "trace labels not found")
        self._send_json(
            {
                "schema_version": LABEL_SET_SCHEMA,
                "label_set_id": label_set_id,
                "artifact_id": artifact_id,
                "content_hash": label_set.get("content_hash"),
                "synthetic": label_set.get("synthetic"),
                "warning": label_set.get("warning"),
                "labels": labels,
            }
        )
        return None

    def _serve_static(self, raw_path: str) -> None:
        relative = _safe_static_path(raw_path)
        if relative is None:
            return self._error(HTTPStatus.BAD_REQUEST, "invalid static path")
        root = self.server.assets_root
        if relative.parts[0] == "assets":
            if len(relative.parts) < 2:
                return self._error(HTTPStatus.NOT_FOUND, "static asset not found")
            relative = PurePosixPath(*relative.parts[1:])
        elif relative.parts[0] == "vendor":
            if len(relative.parts) != 2 or relative.parts[1] not in {
                "d3.js",
                "dagre.min.js",
            }:
                return self._error(HTTPStatus.NOT_FOUND, "static asset not found")
            root = self.server.vendor_root
            relative = PurePosixPath(relative.parts[1])
        candidate = root.joinpath(*relative.parts).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "invalid static path")
        if not candidate.is_file():
            # Client-side routes fall back to the packaged index, but paths that look
            # like missing files remain honest 404s.
            if "." not in relative.name:
                candidate = self.server.assets_root / "index.html"
            if not candidate.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "static asset not found")
        content_type = (
            mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        )
        self._send_bytes(candidate.read_bytes(), content_type=content_type)
        return None

    def _workspace_path(self, workspace_id: str) -> Path | None:
        if not _safe_id(workspace_id):
            return None
        return self.server.state_root / "workspaces" / f"{workspace_id}.json"

    def _empty_workspace(self, workspace_id: str) -> dict[str, Any]:
        return {
            "schema_version": WORKSPACE_SCHEMA,
            "workspace_id": workspace_id,
            "revision": 0,
            "updated_at": None,
            "workspace": {},
        }

    def _get_workspace(self, workspace_id: str) -> None:
        path = self._workspace_path(workspace_id)
        if path is None:
            return self._error(HTTPStatus.BAD_REQUEST, "invalid workspace id")
        try:
            value = _read_json_object(path, WORKSPACE_SCHEMA)
        except FileNotFoundError:
            value = self._empty_workspace(workspace_id)
        except ValueError:
            return self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR, "invalid workspace state"
            )
        self._send_json(value)
        return None

    def _put_workspace(self, workspace_id: str) -> None:
        path = self._workspace_path(workspace_id)
        if path is None:
            return self._error(HTTPStatus.BAD_REQUEST, "invalid workspace id")
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header or "")
        except ValueError:
            return self._error(
                HTTPStatus.LENGTH_REQUIRED, "valid Content-Length required"
            )
        if length < 0 or length > MAX_WORKSPACE_BYTES:
            return self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "workspace request too large"
            )
        try:
            body = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._error(HTTPStatus.BAD_REQUEST, "workspace body must be JSON")
        if not isinstance(body, dict):
            return self._error(
                HTTPStatus.BAD_REQUEST, "workspace body must be an object"
            )
        expected = body.get("expected_revision")
        workspace = body.get("workspace")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            return self._error(
                HTTPStatus.BAD_REQUEST, "expected_revision must be nonnegative"
            )
        if not isinstance(workspace, dict):
            return self._error(HTTPStatus.BAD_REQUEST, "workspace must be an object")
        # Ensure the payload is finite and bounded before entering the write lock.
        try:
            _json_bytes(workspace)
        except (TypeError, ValueError):
            return self._error(HTTPStatus.BAD_REQUEST, "workspace is not finite JSON")

        with self.server.workspace_lock:
            try:
                current = _read_json_object(path, WORKSPACE_SCHEMA)
            except FileNotFoundError:
                current = self._empty_workspace(workspace_id)
            except ValueError:
                return self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR, "invalid workspace state"
                )
            if current.get("revision") != expected:
                return self._send_json(
                    {
                        "error": "workspace revision conflict",
                        "status": int(HTTPStatus.CONFLICT),
                        "current": current,
                    },
                    HTTPStatus.CONFLICT,
                )
            updated = {
                "schema_version": WORKSPACE_SCHEMA,
                "workspace_id": workspace_id,
                "revision": expected + 1,
                "updated_at": datetime.now(UTC).isoformat(),
                "workspace": workspace,
            }
            payload = _json_bytes(updated)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
            try:
                with temporary.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            except OSError:
                temporary.unlink(missing_ok=True)
                return self._error(
                    HTTPStatus.INSUFFICIENT_STORAGE, "workspace state is not writable"
                )
        self._send_json(updated)
        return None


def make_server(
    *,
    site_root: str | os.PathLike[str],
    state_root: str | os.PathLike[str],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    assets_root: str | os.PathLike[str] | None = None,
    vendor_root: str | os.PathLike[str] | None = None,
) -> ObservatoryServer:
    site = Path(site_root).expanduser().resolve()
    state = Path(state_root).expanduser().resolve()
    if not site.is_dir():
        raise ValueError(f"site root is not a directory: {site}")
    validate_site_bundle(site)
    state.mkdir(parents=True, exist_ok=True)
    (state / "workspaces").mkdir(exist_ok=True)
    assets = Path(assets_root).expanduser() if assets_root is not None else None
    vendor = Path(vendor_root).expanduser() if vendor_root is not None else None
    return ObservatoryServer(
        (host, port),
        site_root=site,
        state_root=state,
        assets_root=assets,
        vendor_root=vendor,
    )


def serve(
    *,
    site_root: str | os.PathLike[str],
    state_root: str | os.PathLike[str],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    server = make_server(
        site_root=site_root, state_root=state_root, host=host, port=port
    )
    print(f"Trace Observatory serving on http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "make_server", "serve", "validate_site_bundle"]
