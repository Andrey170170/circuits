"""Build an immutable top-k manifest from exact width-one source items."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.bonafide.topk_manifest import (
    MODEL_TOP5_PLUS_OBSERVED_COUNT_RULE,
    SCHEMA_VERSION,
    validate_topk_manifest,
)

POLICY_COUNTS = {
    "observed_token": 1,
    "specified_token": 1,
    "model_top5": 5,
    "observed_plus_top4_alternatives": 5,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_topk_manifest(
    source_manifest: Mapping[str, Any],
    *,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    source_artifact_ids: Sequence[str],
    phase: str,
    trace_family_id: str,
    candidate_policy_id: str,
    joint_objective_id: str,
    wave_id: str,
) -> dict[str, Any]:
    """Select exact source items and retain their original corpus role."""

    if not source_artifact_ids:
        raise ValueError("at least one source artifact ID is required")
    if len(set(source_artifact_ids)) != len(source_artifact_ids):
        raise ValueError("source artifact IDs must be unique")
    tokenizer = source_manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("width-one source manifest lacks tokenizer provenance")
    for field in ("model_id", "revision", "chat_template_sha256"):
        if not isinstance(tokenizer.get(field), str) or not tokenizer[field]:
            raise ValueError(f"width-one tokenizer.{field} is required")

    source_items: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for source_wave in source_manifest.get("waves", []):
        corpus_role = source_wave.get("corpus_role")
        for item in source_wave.get("items", []):
            artifact_id = item.get("artifact_id")
            if artifact_id in source_items:
                raise ValueError(
                    f"duplicate width-one source artifact ID: {artifact_id!r}"
                )
            source_items[artifact_id] = (corpus_role, item)

    selected: list[Mapping[str, Any]] = []
    corpus_roles: set[str] = set()
    for artifact_id in source_artifact_ids:
        pair = source_items.get(artifact_id)
        if pair is None:
            raise ValueError(
                f"source artifact ID is absent from width-one manifest: {artifact_id}"
            )
        corpus_role, item = pair
        if not isinstance(corpus_role, str) or not corpus_role:
            raise ValueError(f"source item {artifact_id} lacks a corpus role")
        corpus_roles.add(corpus_role)
        selected.append(item)
    if len(corpus_roles) != 1:
        raise ValueError("one top-k wave cannot mix source corpus roles")

    variable_candidate_count = candidate_policy_id == "model_top5_plus_observed"
    if not variable_candidate_count:
        try:
            candidate_count = POLICY_COUNTS[candidate_policy_id]
        except KeyError as error:
            raise ValueError(
                f"unsupported top-k candidate policy: {candidate_policy_id!r}"
            ) from error
    else:
        candidate_count = None
    trace_family = {
        "trace_family_id": trace_family_id,
        "candidate_policy_id": candidate_policy_id,
        "candidate_policy_version": "1",
        "joint_objective_id": joint_objective_id,
        "joint_objective_version": "1",
    }
    if variable_candidate_count:
        trace_family.update(
            {
                "candidate_count_min": 5,
                "candidate_count_max": 6,
                "candidate_count_rule": MODEL_TOP5_PLUS_OBSERVED_COUNT_RULE,
            }
        )
    else:
        trace_family["candidate_count"] = candidate_count
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "trace_family": trace_family,
        "source": {
            "width1_manifest_path": str(source_manifest_path.resolve()),
            "width1_manifest_sha256": source_manifest_sha256,
            "model_id": tokenizer["model_id"],
            "model_revision": tokenizer["revision"],
            "tokenizer_revision": tokenizer["revision"],
            "chat_template_sha256": tokenizer["chat_template_sha256"],
        },
        "waves": [
            {
                "wave_id": wave_id,
                "corpus_role": next(iter(corpus_roles)),
                "items": selected,
            }
        ],
    }
    validate_topk_manifest(manifest)
    return manifest


def save_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Create a manifest atomically without replacing prior evidence."""

    if path.exists():
        raise FileExistsError(f"top-k manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-artifact-id", action="append", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--trace-family-id", required=True)
    parser.add_argument("--candidate-policy-id", required=True)
    parser.add_argument("--joint-objective-id", required=True)
    parser.add_argument("--wave-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.source_manifest.open(encoding="utf-8") as handle:
        source = json.load(handle)
    manifest = build_topk_manifest(
        source,
        source_manifest_path=args.source_manifest,
        source_manifest_sha256=_sha256_file(args.source_manifest),
        source_artifact_ids=args.source_artifact_id,
        phase=args.phase,
        trace_family_id=args.trace_family_id,
        candidate_policy_id=args.candidate_policy_id,
        joint_objective_id=args.joint_objective_id,
        wave_id=args.wave_id,
    )
    save_manifest(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
