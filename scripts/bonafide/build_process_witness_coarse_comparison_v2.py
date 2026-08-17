#!/usr/bin/env python3
"""Build an immutable offline comparison of completed coarse v1/v2 decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_comparison_v2 import (
    MANIFEST_SCHEMA,
    build_comparison_report,
    load_completed_comparison_inputs,
)
from circuits.labeling.io import atomic_write_json, atomic_write_jsonl

_BOUND_SOURCE_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_comparison_v2.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v2.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/io.py",
    "scripts/bonafide/build_process_witness_coarse_comparison_v2.py",
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("coarse comparison repository root drift")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("coarse comparison build requires a clean tracked worktree")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_SOURCE_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse comparison source is untracked: {relative}")
        path = root / relative
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        committed = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        expected = hashlib.sha256(committed).hexdigest()
        if file_sha256(path) != expected:
            raise ValueError(f"coarse comparison source differs from HEAD: {relative}")
        files.append({"path": relative, "git_blob": blob, "sha256": expected})
    return {
        "repo_root": str(root),
        "git_commit": commit,
        "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": True,
        "files": files,
    }


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def build(
    *, run_root: Path, qualification_root: Path, destination: Path
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_revision = _source_revision()
    inputs = load_completed_comparison_inputs(
        run_root=run_root, qualification_root=qualification_root
    )
    report, examples = build_comparison_report(inputs)
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        atomic_write_json(temporary / "comparison-report.json", report)
        atomic_write_jsonl(temporary / "examples-disagreements.jsonl", examples)
        files = [
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir())
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete_offline_immutable_comparison",
            "network_calls_made": 0,
            "claim_boundary": report["claim_boundary"],
            "source_run_root": str(run_root.resolve()),
            "source_qualification_root": str(qualification_root.resolve()),
            "source_bindings": report["source_bindings"],
            "comparison_plan_sha256": report["comparison_plan_sha256"],
            "report_sha256": report["report_sha256"],
            "example_row_count": len(examples),
            "source_revision": source_revision,
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        run_root=args.run_root.resolve(),
        qualification_root=args.qualification_root.resolve(),
        destination=args.destination.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
