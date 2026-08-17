#!/usr/bin/env python3
"""Build the globally sealed blind-review UI before v4 Batch reveal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation_v4 import (
    load_v4_qualification,
)
from circuits.analysis.bonafide.coarse_sampling_review_v4 import (
    PACKET_SCHEMA,
    build_review_payload,
    render_review_html,
)
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def build(
    *,
    qualification_root: Path,
    workstation_bundle_path: Path,
    destination: Path,
) -> dict:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    qualification = load_v4_qualification(qualification_root)
    workstation = json.loads(workstation_bundle_path.read_text(encoding="utf-8"))
    if (
        file_sha256(workstation_bundle_path)
        != qualification["manifest"]["source_workstation_bundle_sha256"]
    ):
        raise ValueError("coarse v4 review workstation binding drift")
    payload = build_review_payload(
        qualification=qualification,
        workstation_bundle=workstation,
    )
    html = render_review_html(payload).encode("utf-8")
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        atomic_write_json(temporary / "packet.json", payload["packet"])
        atomic_write_jsonl(temporary / "documents.jsonl", payload["documents"])
        atomic_write_jsonl(temporary / "items.jsonl", payload["items"])
        atomic_write_bytes(temporary / "review.html", html)
        files = [
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir())
        ]
        manifest = {
            "schema_version": PACKET_SCHEMA,
            "status": "frozen_blind_global_seal_review_packet",
            "packet_id": payload["packet"]["packet_id"],
            "qualification_manifest_sha256": qualification["manifest"][
                "manifest_sha256"
            ],
            "counts": {"response_blocks": 15, "items": 24},
            "model_reveal_policy": "all 24 human decisions globally sealed first",
            "files": files,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        _readonly_tree(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--workstation-bundle", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        qualification_root=args.qualification_root.resolve(),
        workstation_bundle_path=args.workstation_bundle.resolve(),
        destination=args.destination.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
