"""Install a provenance-bound external label set into a derived viewer site."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_json, canonical_sha256
from circuits.observatory import LABEL_SET_SCHEMA, MANIFEST_SCHEMA
from circuits.observatory.server import validate_site_bundle

LABEL_INDEX_SCHEMA = "adag.observatory.label-set-index.v1"
RECEIPT_SCHEMA = "adag.observatory.external-label-install-receipt.v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> str:
    payload = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.urandom(8).hex()}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return canonical_sha256(value)


def _load_self_hashed(path: Path, schema: str) -> dict[str, Any]:
    value = _load_json_object(path)
    if value.get("schema_version") != schema:
        raise ValueError(f"unsupported JSON schema: {path.name}")
    core = dict(value)
    recorded = core.pop("content_hash", None)
    if recorded != canonical_sha256(core):
        raise ValueError(f"content hash mismatch: {path.name}")
    return value


def _label_descriptor(label_set: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_set_id": label_set["label_set_id"],
        "name": label_set.get("name"),
        "synthetic": label_set.get("synthetic"),
        "warning": label_set.get("warning"),
        "method": label_set.get("method"),
        "content_hash": label_set["content_hash"],
    }


def _validate_source_tree(source: Path) -> None:
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"viewer source contains a symbolic link: {candidate}")
        if not candidate.is_dir() and not candidate.is_file():
            raise ValueError(f"viewer source contains a special file: {candidate}")


def _derivation_identity(
    *, source_bundle_hash: str, label_set_id: str, label_set_hash: str
) -> dict[str, str]:
    core = {
        "source_bundle_hash": source_bundle_hash,
        "label_set_id": label_set_id,
        "label_set_hash": label_set_hash,
    }
    return {**core, "derivation_hash": canonical_sha256(core)}


def _receipt(
    *,
    source: Path,
    destination: Path,
    manifest: dict[str, Any],
    derivation: dict[str, str],
    idempotent: bool,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "source_site": str(source),
        "destination_site": str(destination),
        **derivation,
        "destination_bundle_hash": manifest["content_hash"],
        "destination_created_at": manifest.get("created_at"),
        "idempotent": idempotent,
    }
    value["receipt_hash"] = canonical_sha256(value)
    return value


def _existing_destination_receipt(
    *,
    source: Path,
    destination: Path,
    derivation: dict[str, str],
) -> dict[str, Any]:
    validate_site_bundle(destination)
    manifest = _load_self_hashed(destination / "viewer-manifest.json", MANIFEST_SCHEMA)
    if manifest.get("external_label_install") != derivation:
        raise FileExistsError(
            "destination exists and is not the identical derived viewer bundle: "
            f"{destination}"
        )
    label_path = destination / "label-sets" / f"{derivation['label_set_id']}.json"
    label_set = _load_self_hashed(label_path, LABEL_SET_SCHEMA)
    if label_set["content_hash"] != derivation["label_set_hash"]:
        raise FileExistsError(
            "destination label-set identity differs from the requested installation"
        )
    return _receipt(
        source=source,
        destination=destination,
        manifest=manifest,
        derivation=derivation,
        idempotent=True,
    )


def install_label_set(
    source_site: str | os.PathLike[str],
    label_set_path: str | os.PathLike[str],
    destination_site: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build a new immutable viewer site containing one external label set.

    The source is validated and copied but never modified. Existing destinations
    are accepted only when they are the exact derivation requested here.
    """

    source = Path(source_site).expanduser().resolve()
    label_path = Path(label_set_path).expanduser().resolve()
    destination = Path(destination_site).expanduser().resolve()
    if source == destination or source in destination.parents:
        raise ValueError("destination must not be the source or lie below it")

    validate_site_bundle(source)
    _validate_source_tree(source)
    source_manifest = _load_self_hashed(
        source / "viewer-manifest.json", MANIFEST_SCHEMA
    )
    label_set = _load_self_hashed(label_path, LABEL_SET_SCHEMA)
    label_set_id = label_set.get("label_set_id")
    if not isinstance(label_set_id, str) or not _SAFE_ID.fullmatch(label_set_id):
        raise ValueError("external label-set id is invalid")
    label_set_hash = label_set["content_hash"]
    derivation = _derivation_identity(
        source_bundle_hash=source_manifest["content_hash"],
        label_set_id=label_set_id,
        label_set_hash=label_set_hash,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return _existing_destination_receipt(
            source=source,
            destination=destination,
            derivation=derivation,
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, copy_function=shutil.copy2)
        index_path = staging / "label-sets" / "index.json"
        label_index = _load_json_object(index_path)
        if label_index.get("schema_version") != LABEL_INDEX_SCHEMA:
            raise ValueError("viewer label-set index is absent or invalid")
        descriptors = label_index.get("label_sets")
        if not isinstance(descriptors, list):
            raise ValueError("viewer label-set index entries must be a list")

        matches = [
            descriptor
            for descriptor in descriptors
            if isinstance(descriptor, dict)
            and descriptor.get("label_set_id") == label_set_id
        ]
        if any(
            descriptor.get("content_hash") != label_set_hash for descriptor in matches
        ):
            raise ValueError(
                f"label-set id collision with different content: {label_set_id}"
            )

        installed_path = staging / "label-sets" / f"{label_set_id}.json"
        if installed_path.exists():
            installed = _load_self_hashed(installed_path, LABEL_SET_SCHEMA)
            if installed.get("content_hash") != label_set_hash:
                raise ValueError(
                    f"label-set file collision with different content: {label_set_id}"
                )

        descriptor = _label_descriptor(label_set)
        if matches:
            descriptors = [
                descriptor
                if isinstance(item, dict) and item.get("label_set_id") == label_set_id
                else item
                for item in descriptors
            ]
        else:
            descriptors = [*descriptors, descriptor]
        label_index = {**label_index, "label_sets": descriptors}

        installed_hash = _write_json(installed_path, label_set)
        index_hash = _write_json(index_path, label_index)

        files = source_manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("viewer manifest files must be an object")
        updated_files = dict(files)
        updated_files[f"label-sets/{label_set_id}.json"] = installed_hash
        updated_files["label-sets/index.json"] = index_hash
        manifest = dict(source_manifest)
        manifest.pop("content_hash", None)
        manifest.update(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "files": updated_files,
                "external_label_install": derivation,
            }
        )
        manifest["content_hash"] = canonical_sha256(manifest)
        _write_json(staging / "viewer-manifest.json", manifest)

        # This performs the same complete binding checks used by the server:
        # model/revision, basis/polarity, trace hashes, and occurrence coverage.
        validate_site_bundle(staging)
        if destination.exists():
            raise FileExistsError(f"destination appeared during install: {destination}")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return _receipt(
        source=source,
        destination=destination,
        manifest=manifest,
        derivation=derivation,
        idempotent=False,
    )


__all__ = ["install_label_set"]
