"""Build a safe, immutable JSON observatory bundle from trusted compact traces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_json, canonical_sha256
from circuits.analysis.bonafide.identity import BASIS_KEY_SCHEMA, POLARITY_DERIVATION
from circuits.analysis.bonafide.multiplex import (
    TargetSlice,
    build_target_slice,
    validate_target_slice_round_trip,
)
from circuits.observatory import (
    CATALOG_SCHEMA,
    CLAIM_BOUNDARY,
    LABEL_SET_SCHEMA,
    MANIFEST_SCHEMA,
    TRACE_GRAPH_SCHEMA,
)
from circuits.tracing.artifact import (
    DATA_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    CompactTraceArtifact,
    load_compact_trace,
)
from circuits.tracing.artifact import (
    SCHEMA_VERSION as COMPACT_TRACE_SCHEMA,
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
_SOURCE_FILES = (MANIFEST_FILENAME, METRICS_FILENAME, DATA_FILENAME)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(path: Path) -> dict[str, str]:
    return {name: _file_sha256(path / name) for name in _SOURCE_FILES}


def _safe_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} is not a safe identifier: {value!r}")
    return value


def _write_json(path: Path, value: object) -> str:
    payload = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"compact manifest is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"compact manifest must be an object: {path}")
    return value


def discover_trace_directories(trace_root: str | os.PathLike[str]) -> list[Path]:
    """Find compact manifests recursively, rejecting an empty or unsafe cohort."""

    root = Path(trace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"trace root is not a directory: {root}")
    manifests = sorted(root.rglob(MANIFEST_FILENAME))
    if not manifests:
        raise ValueError(f"no compact trace manifests found below: {root}")
    directories: list[Path] = []
    for manifest_path in manifests:
        manifest = _read_manifest(manifest_path)
        if manifest.get("schema_version") != COMPACT_TRACE_SCHEMA:
            raise ValueError(
                f"unsupported compact trace schema in {manifest_path}: "
                f"{manifest.get('schema_version')!r}"
            )
        directories.append(manifest_path.parent)
    return directories


def _validated_artifact(path: Path) -> tuple[CompactTraceArtifact, dict[str, str]]:
    before = _source_hashes(path)
    artifact = load_compact_trace(path)
    manifest = artifact.manifest
    if manifest.get("schema_version") != COMPACT_TRACE_SCHEMA:
        raise ValueError("only adag.compact-trace.v1 artifacts are supported")
    if manifest.get("scientifically_reusable") is not True:
        raise ValueError(f"trace is not scientifically reusable: {path.name}")
    if manifest.get("target_count") != 1:
        raise ValueError(f"trace must have exactly one target: {path.name}")
    if (
        manifest.get("benchmark_only") is not False
        or artifact.circuit_data.benchmark_only
    ):
        raise ValueError(f"benchmark-only trace is not viewable: {path.name}")
    after = _source_hashes(path)
    if before != after:
        raise RuntimeError(f"source trace changed while it was being read: {path.name}")
    return artifact, before


def _resolve_declared_snapshot(manifest: Mapping[str, Any]) -> Path | None:
    identity = manifest.get("artifact_identity")
    if not isinstance(identity, Mapping):
        return None
    model = identity.get("model")
    if not isinstance(model, Mapping):
        return None
    declared = model.get("local_snapshot_path")
    if not isinstance(declared, str) or not declared:
        return None
    expanded = os.path.expandvars(declared)
    if "$" in expanded:
        return None
    path = Path(expanded).expanduser()
    return path.resolve() if path.exists() else None


def _load_decoder(
    artifacts: Sequence[CompactTraceArtifact],
    *,
    tokenizer_path: str | os.PathLike[str] | None,
    allow_numeric_tokens: bool,
) -> tuple[Callable[[int], str], dict[str, object]]:
    """Load one offline tokenizer for sync, or require an explicit numeric fallback."""

    model_ids = {str(artifact.manifest.get("model_id")) for artifact in artifacts}
    revisions = {str(artifact.manifest.get("model_revision")) for artifact in artifacts}
    if len(model_ids) != 1 or len(revisions) != 1:
        raise ValueError("one observatory bundle cannot mix model identity spaces")
    model_id = next(iter(model_ids))
    revision = next(iter(revisions))
    declared = {_resolve_declared_snapshot(artifact.manifest) for artifact in artifacts}
    declared.discard(None)
    if len(declared) > 1:
        raise ValueError("trace cohort declares multiple tokenizer snapshots")
    declared_path = next(iter(declared), None)

    requested = Path(tokenizer_path).expanduser().resolve() if tokenizer_path else None
    selected = requested or declared_path
    if requested is not None:
        if not requested.is_dir():
            raise ValueError(
                f"requested tokenizer path is not a directory: {requested}"
            )
        same_declared = declared_path is not None and requested == declared_path
        revision_named = revision in requested.parts or requested.name == revision
        if not same_declared and not revision_named:
            raise ValueError(
                "requested tokenizer path does not match the trace model revision"
            )
    if selected is None:
        if not allow_numeric_tokens:
            raise ValueError(
                "no matching offline tokenizer snapshot is available; pass "
                "--allow-numeric-tokens to use an explicit numeric fallback"
            )
        return (lambda token_id: f"[{token_id}]"), {
            "mode": "numeric_fallback",
            "model_id": model_id,
            "model_revision": revision,
        }

    # Deliberately lazy: this module is imported only by the sync command.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(selected, local_files_only=True)
    loaded_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if loaded_revision is not None and loaded_revision != revision:
        raise ValueError(
            "loaded tokenizer revision does not match compact trace revision"
        )

    def decode(token_id: int) -> str:
        return tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    return decode, {
        "mode": "offline_tokenizer",
        "model_id": model_id,
        "model_revision": revision,
        "snapshot_revision": selected.name,
    }


def _stable_id(prefix: str, record: object) -> str:
    return f"{prefix}-{canonical_sha256(record)[:24]}"


def _target_slice(artifact: CompactTraceArtifact) -> TargetSlice:
    manifest = artifact.manifest
    data = artifact.circuit_data
    artifact_id = _safe_id(manifest.get("artifact_id"), "artifact_id")
    provenance = data.target_provenance[0]
    response_position = provenance.get("response_token_position")
    if isinstance(response_position, bool) or not isinstance(response_position, int):
        raise ValueError("target response position must be an integer")
    example = manifest.get("bonafide_example")
    response_id = (
        str(example.get("example_id", "")) if isinstance(example, Mapping) else ""
    )
    if not response_id:
        response_id = data.labels[0] if len(data.labels) == 1 else ""
    if not response_id:
        raise ValueError("trace lacks a stable response identity")
    model_revision = manifest.get("model_revision")
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("trace lacks model_revision")
    node_rows = data.df_node.to_dict(orient="records")
    edge_rows = data.df_edge.to_dict(orient="records")
    target_slice = build_target_slice(
        response_id=response_id,
        target_response_position=response_position,
        trace_unit_id=artifact_id,
        model_id=data.model_id,
        model_revision=model_revision,
        node_rows=node_rows,
        edge_rows=edge_rows,
    )
    validate_target_slice_round_trip(
        target_slice, source_node_rows=node_rows, source_edge_rows=edge_rows
    )
    return target_slice


def _human_comment(manifest: Mapping[str, Any]) -> str | None:
    selection = manifest.get("source_target_selection")
    if not isinstance(selection, Mapping):
        return None
    human = selection.get("human_selection")
    if not isinstance(human, Mapping):
        return None
    comment = human.get("comment")
    return comment if isinstance(comment, str) and comment else None


def _projection_document(
    artifact: CompactTraceArtifact,
    target_slice: TargetSlice,
    hashes: Mapping[str, str],
    decode: Callable[[int], str],
    tokenizer_identity: Mapping[str, object],
) -> dict[str, Any]:
    manifest = artifact.manifest
    data = artifact.circuit_data
    artifact_id = target_slice.trace_unit_id
    provenance = dict(data.target_provenance[0])
    trace_metadata = data.trace_metadata
    input_ids = [int(value) for value in data.cis[0]]
    prefix_count = trace_metadata.get("assistant_prefix_token_count")
    if isinstance(prefix_count, bool) or not isinstance(prefix_count, int):
        raise ValueError("trace lacks assistant_prefix_token_count")
    if not 0 <= prefix_count <= len(input_ids):
        raise ValueError("assistant_prefix_token_count is outside the input")

    token_records = []
    for absolute_position, token_id in enumerate(input_ids):
        response_position = absolute_position - prefix_count
        token_records.append(
            {
                "absolute_position": absolute_position,
                "response_position": response_position
                if response_position >= 0
                else None,
                "role": "assistant" if response_position >= 0 else "prefix",
                "token_id": token_id,
                "text": decode(token_id),
            }
        )

    nodes: list[dict[str, Any]] = []
    occurrence_ids: dict[object, str] = {}
    output_layer = max(
        (node.occurrence.layer for node in target_slice.nodes), default=None
    )
    for node in target_slice.nodes:
        occurrence = node.occurrence.to_record()
        basis = node.basis.to_record()
        occurrence_id = _stable_id("occ", occurrence)
        occurrence_ids[node.occurrence] = occurrence_id
        nodes.append(
            {
                "id": occurrence_id,
                "occurrence": occurrence,
                "basis_id": _stable_id("basis", basis),
                "basis": basis,
                "kind": (
                    "input_token"
                    if node.occurrence.layer == -1
                    else (
                        "target_logit"
                        if node.occurrence.layer == output_layer
                        else "raw_mlp_neuron"
                    )
                ),
                "attribution": node.attribution,
                "activation": node.activation,
                "activation_polarity": node.occurrence.polarity,
                "attribution_sign": "+" if node.attribution >= 0 else "-",
                "attribution_map": list(node.attribution_map),
                "contribution_map": list(node.contribution_map),
                "local_label": node.local_label,
            }
        )

    duplicate_ordinals: defaultdict[tuple[str, str], int] = defaultdict(int)
    edges: list[dict[str, Any]] = []
    for edge in target_slice.edges:
        source_id = occurrence_ids[edge.source]
        target_id = occurrence_ids[edge.target]
        pair = (source_id, target_id)
        ordinal = duplicate_ordinals[pair]
        duplicate_ordinals[pair] += 1
        identity = {
            "trace_unit_id": artifact_id,
            "source": source_id,
            "target": target_id,
            "duplicate_ordinal": ordinal,
            "attribution": edge.attribution,
            "weight": edge.weight,
            "local_label": edge.local_label,
        }
        edges.append(
            {
                "id": _stable_id("edge", identity),
                **identity,
                "attribution_sign": "+" if edge.attribution >= 0 else "-",
            }
        )

    source_hash = canonical_sha256(dict(hashes))
    example = manifest.get("bonafide_example")
    example = example if isinstance(example, Mapping) else {}
    objective = manifest.get("objective")
    document: dict[str, Any] = {
        "schema_version": TRACE_GRAPH_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "artifact": {
            "artifact_id": artifact_id,
            "response_id": target_slice.response_id,
            "trace_unit_id": target_slice.trace_unit_id,
            "source_schema": manifest.get("schema_version"),
            "source_hash": source_hash,
            "source_hashes": dict(hashes),
            "source_artifact_id": manifest.get("source_artifact_id"),
            "created_at": manifest.get("created_at"),
            "code_revision": manifest.get("code_revision"),
        },
        "model": {
            "model_id": manifest.get("model_id"),
            "model_revision": manifest.get("model_revision"),
            "tokenizer": dict(tokenizer_identity),
        },
        "target": {
            **provenance,
            "prediction_position": provenance.get("prediction_token_position"),
            "observed_absolute_position": provenance.get("absolute_token_position"),
            "response_position": provenance.get("response_token_position"),
            "comment": _human_comment(manifest),
            "objective": objective,
        },
        "context": {
            "system_prompt": trace_metadata.get("system_prompt"),
            "prompt": trace_metadata.get("prompt", example.get("prompt")),
            "response": trace_metadata.get("response", example.get("response")),
            "assistant_prefix_token_count": prefix_count,
            "input_token_ids": input_ids,
            "tokens": token_records,
        },
        "nodes": nodes,
        "edges": edges,
        "diagnostics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source_node_count": manifest.get("node_count"),
            "source_edge_count": manifest.get("edge_count"),
            "input_token_count": len(input_ids),
            "trace_configuration": (
                manifest.get("artifact_identity", {}).get("adag_config")
                if isinstance(manifest.get("artifact_identity"), Mapping)
                else None
            ),
            "metrics": artifact.metrics,
        },
    }
    _validate_projection(document)
    return document


def _validate_projection(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != TRACE_GRAPH_SCHEMA:
        raise ValueError("invalid observatory trace schema")
    artifact = document.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("trace document lacks artifact identity")
    artifact_id = _safe_id(artifact.get("artifact_id"), "artifact_id")
    nodes = document.get("nodes")
    edges = document.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("trace document nodes and edges must be lists")
    node_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("trace node must be an object")
        node_id = _safe_id(node.get("id"), "node id")
        if node_id in node_ids:
            raise ValueError(f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        occurrence = node.get("occurrence")
        if (
            not isinstance(occurrence, Mapping)
            or occurrence.get("trace_unit_id") != artifact_id
        ):
            raise ValueError("trace node belongs to another trace unit")
    edge_ids: set[str] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ValueError("trace edge must be an object")
        edge_id = _safe_id(edge.get("id"), "edge id")
        if edge_id in edge_ids:
            raise ValueError(f"duplicate edge id: {edge_id}")
        edge_ids.add(edge_id)
        if edge.get("trace_unit_id") != artifact_id:
            raise ValueError("trace edge belongs to another trace unit")
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            raise ValueError("trace edge endpoint is absent")
    # Enforce finite JSON and deterministic serializability.
    canonical_json(document)


def _catalog_item(document: Mapping[str, Any]) -> dict[str, Any]:
    artifact = document["artifact"]
    target = document["target"]
    diagnostics = document["diagnostics"]
    return {
        "artifact_id": artifact["artifact_id"],
        "target_token": target.get("token_text"),
        "target_token_id": target.get("token_id"),
        "comment": target.get("comment"),
        "probability": target.get("probability"),
        "logit": target.get("logit"),
        "response_position": target.get("response_position"),
        "prediction_position": target.get("prediction_position"),
        "observed_absolute_position": target.get("observed_absolute_position"),
        "node_count": diagnostics["node_count"],
        "edge_count": diagnostics["edge_count"],
        "source_hash": artifact["source_hash"],
        "data_sha256": artifact["source_hashes"][DATA_FILENAME],
        "source_payload_sha256": artifact["source_hashes"][DATA_FILENAME],
        "model_id": document["model"]["model_id"],
        "model_revision": document["model"]["model_revision"],
    }


def _synthetic_label_sets(
    documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    specs = (
        (
            "synthetic-layer-band-v1",
            "Synthetic layer bands",
            "synthetic_debug_only_layer_thirds_and_polarity",
            lambda node: (
                f"synthetic:{node['occurrence']['polarity']}:layer-{int(node['occurrence']['layer']) // 12}"
            ),
        ),
        (
            "synthetic-position-bucket-v1",
            "Synthetic position buckets",
            "synthetic_debug_only_position_modulo_and_polarity",
            lambda node: (
                f"synthetic:{node['occurrence']['polarity']}:position-{int(node['occurrence']['token_position']) % 5}"
            ),
        ),
    )
    results: list[dict[str, Any]] = []
    model = documents[0]["model"]
    for label_set_id, name, method, labeler in specs:
        labels_by_trace: dict[str, list[dict[str, Any]]] = {}
        source_hashes: dict[str, str] = {}
        for document in documents:
            artifact_id = document["artifact"]["artifact_id"]
            source_hashes[artifact_id] = document["artifact"]["source_hash"]
            labels_by_trace[artifact_id] = [
                {
                    "occurrence_id": node["id"],
                    "basis_id": node["basis_id"],
                    "label": labeler(node),
                    "status": "synthetic",
                    "confidence": None,
                }
                for node in document["nodes"]
            ]
        core = {
            "schema_version": LABEL_SET_SCHEMA,
            "label_set_id": label_set_id,
            "name": name,
            "synthetic": True,
            "warning": "Debug-only synthetic labels; they have no semantic meaning.",
            "method": method,
            "model_id": model["model_id"],
            "model_revision": model["model_revision"],
            "basis_schema": BASIS_KEY_SCHEMA,
            "polarity_derivation": POLARITY_DERIVATION,
            "source_trace_hashes": source_hashes,
            "labels_by_trace": labels_by_trace,
        }
        results.append({**core, "content_hash": canonical_sha256(core)})
    return results


def _validate_bundle(root: Path) -> None:
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    if catalog.get("schema_version") != CATALOG_SCHEMA:
        raise ValueError("generated catalog schema is invalid")
    seen: set[str] = set()
    documents: dict[str, dict[str, Any]] = {}
    for item in catalog.get("traces", []):
        artifact_id = _safe_id(item.get("artifact_id"), "artifact_id")
        if artifact_id in seen:
            raise ValueError(f"duplicate artifact id in catalog: {artifact_id}")
        seen.add(artifact_id)
        document = json.loads(
            (root / "traces" / f"{artifact_id}.json").read_text(encoding="utf-8")
        )
        _validate_projection(document)
        documents[artifact_id] = document
        if document["artifact"]["source_hash"] != item["source_hash"]:
            raise ValueError("catalog source hash does not match trace document")

    label_index = json.loads(
        (root / "label-sets" / "index.json").read_text(encoding="utf-8")
    )
    for descriptor in label_index.get("label_sets", []):
        label_set_id = _safe_id(descriptor.get("label_set_id"), "label_set_id")
        label_set = json.loads(
            (root / "label-sets" / f"{label_set_id}.json").read_text(
                encoding="utf-8"
            )
        )
        if label_set.get("schema_version") != LABEL_SET_SCHEMA:
            raise ValueError("generated label-set schema is invalid")
        core = dict(label_set)
        recorded_content_hash = core.pop("content_hash", None)
        if recorded_content_hash != canonical_sha256(core):
            raise ValueError("generated label-set content hash is invalid")
        if (
            label_set.get("model_id") != catalog["model"]["model_id"]
            or label_set.get("model_revision")
            != catalog["model"]["model_revision"]
            or label_set.get("basis_schema") != BASIS_KEY_SCHEMA
            or label_set.get("polarity_derivation") != POLARITY_DERIVATION
        ):
            raise ValueError("generated label-set identity space is invalid")
        if set(label_set.get("labels_by_trace", {})) != set(documents):
            raise ValueError("generated label-set trace coverage is invalid")
        for artifact_id, labels in label_set["labels_by_trace"].items():
            document = documents[artifact_id]
            occurrence_to_basis = {
                node["id"]: node["basis_id"] for node in document["nodes"]
            }
            if label_set["source_trace_hashes"].get(artifact_id) != document[
                "artifact"
            ]["source_hash"]:
                raise ValueError("generated label-set source binding is invalid")
            if len(labels) != len(occurrence_to_basis):
                raise ValueError("generated label-set occurrence coverage is invalid")
            for label in labels:
                occurrence_id = label.get("occurrence_id")
                if occurrence_to_basis.get(occurrence_id) != label.get("basis_id"):
                    raise ValueError("generated label-set basis binding is invalid")


def sync_bundle(
    *,
    trace_root: str | os.PathLike[str],
    site_root: str | os.PathLike[str],
    state_root: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str] | None = None,
    allow_numeric_tokens: bool = False,
    replace: bool = False,
) -> dict[str, Any]:
    """Validate, project, and atomically publish one immutable site bundle."""

    source_directories = discover_trace_directories(trace_root)
    loaded = [_validated_artifact(path) for path in source_directories]
    artifacts = [item[0] for item in loaded]
    decode, tokenizer_identity = _load_decoder(
        artifacts,
        tokenizer_path=tokenizer_path,
        allow_numeric_tokens=allow_numeric_tokens,
    )
    documents = [
        _projection_document(
            artifact,
            _target_slice(artifact),
            hashes,
            decode,
            tokenizer_identity,
        )
        for artifact, hashes in loaded
    ]
    documents.sort(
        key=lambda doc: (
            doc["target"]["response_position"],
            doc["artifact"]["artifact_id"],
        )
    )
    artifact_ids = [doc["artifact"]["artifact_id"] for doc in documents]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("duplicate artifact IDs in trace cohort")
    model = documents[0]["model"]
    catalog = {
        "schema_version": CATALOG_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "model": {
            "model_id": model["model_id"],
            "model_revision": model["model_revision"],
        },
        "traces": [_catalog_item(document) for document in documents],
    }
    label_sets = _synthetic_label_sets(documents)

    destination = Path(site_root).expanduser().resolve()
    state = Path(state_root).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    (state / "workspaces").mkdir(exist_ok=True)
    if destination.exists() and not replace:
        raise FileExistsError(
            f"site destination already exists (pass --replace): {destination}"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    backup: Path | None = None
    try:
        files: dict[str, str] = {}
        files["catalog.json"] = _write_json(staging / "catalog.json", catalog)
        for document in documents:
            relative = f"traces/{document['artifact']['artifact_id']}.json"
            files[relative] = _write_json(staging / relative, document)
        label_index = {
            "schema_version": "adag.observatory.label-set-index.v1",
            "label_sets": [
                {
                    "label_set_id": label_set["label_set_id"],
                    "name": label_set["name"],
                    "synthetic": label_set["synthetic"],
                    "warning": label_set["warning"],
                    "method": label_set["method"],
                    "content_hash": label_set["content_hash"],
                }
                for label_set in label_sets
            ],
        }
        files["label-sets/index.json"] = _write_json(
            staging / "label-sets" / "index.json", label_index
        )
        for label_set in label_sets:
            relative = f"label-sets/{label_set['label_set_id']}.json"
            files[relative] = _write_json(staging / relative, label_set)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
            "file_hash_algorithm": "sha256-canonical-json-v1",
            "catalog_schema": CATALOG_SCHEMA,
            "trace_schema": TRACE_GRAPH_SCHEMA,
            "trace_count": len(documents),
            "artifact_ids": artifact_ids,
            "files": files,
        }
        manifest["content_hash"] = canonical_sha256(manifest)
        _write_json(staging / "viewer-manifest.json", manifest)
        _validate_bundle(staging)

        if destination.exists():
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = destination.with_name(
                f"{destination.name}.backup-{timestamp}-{manifest['content_hash'][:8]}"
            )
            if backup.exists():
                raise FileExistsError(f"replacement backup already exists: {backup}")
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "site_root": str(destination),
        "state_root": str(state),
        "trace_count": len(documents),
        "artifact_ids": artifact_ids,
        "bundle_hash": manifest["content_hash"],
        "backup_root": str(backup) if backup else None,
    }


__all__ = ["discover_trace_directories", "sync_bundle"]
