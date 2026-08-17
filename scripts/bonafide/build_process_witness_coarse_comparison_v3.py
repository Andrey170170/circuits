#!/usr/bin/env python3
"""Build the v3 three-vote comparison and optional sealed-human gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_comparison_v3 import (
    apply_human_gate,
    build_comparison,
    load_completed_v3_inputs,
)
from circuits.analysis.bonafide.coarse_sampling_review_v3 import load_review_packet
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl, read_jsonl


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def build(
    *,
    qualification_root: Path,
    run_root: Path,
    destination: Path,
    human_decisions_path: Path,
    review_packet_root: Path,
) -> dict:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    inputs = load_completed_v3_inputs(
        qualification_root=qualification_root, run_root=run_root
    )
    report = build_comparison(inputs)
    human = read_jsonl(human_decisions_path)
    review_packet = load_review_packet(review_packet_root)
    report = apply_human_gate(report, human, review_packet)
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        vote_rows = report.pop("vote_rows")
        report["report_sha256"] = canonical_sha256(report)
        atomic_write_json(temporary / "report.json", report)
        atomic_write_jsonl(temporary / "vote-rows.jsonl", vote_rows)
        atomic_write_jsonl(temporary / "human-decisions.jsonl", human)
        atomic_write_json(
            temporary / "review-packet-binding.json",
            {
                "packet_id": review_packet["packet"]["packet_id"],
                "packet_binding_sha256": review_packet["packet"][
                    "packet_binding_sha256"
                ],
                "packet_manifest_sha256": review_packet["manifest"]["manifest_sha256"],
            },
        )
        files = [
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir())
        ]
        manifest = {
            "schema_version": "adag.process-witness.coarse-comparison-bundle.v3",
            "status": report["status"],
            "qualification_manifest_sha256": report["qualification_manifest_sha256"],
            "collection_manifest_sha256": report["collection_manifest_sha256"],
            "human_decisions_source_sha256": (file_sha256(human_decisions_path)),
            "review_packet_manifest_sha256": review_packet["manifest"][
                "manifest_sha256"
            ],
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--human-decisions", type=Path, required=True)
    parser.add_argument("--review-packet-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        qualification_root=args.qualification_root.resolve(),
        run_root=args.run_root.resolve(),
        destination=args.destination.resolve(),
        human_decisions_path=args.human_decisions.resolve(),
        review_packet_root=args.review_packet_root.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
