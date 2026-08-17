#!/usr/bin/env python3
"""Build the immutable v4 compatibility report after global human seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_compatibility_v4 import (
    build_compatibility_report,
    load_completed_v4_inputs,
)
from circuits.labeling.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
)

_BOUND_SOURCE_FILES = (
    "circuits/analysis/bonafide/coarse_sampling_compatibility_v4.py",
    "circuits/analysis/bonafide/coarse_sampling_annotation_v4.py",
    "circuits/analysis/bonafide/coarse_sampling_review_v4.py",
    "circuits/analysis/bonafide/canonical.py",
    "circuits/labeling/io.py",
    "scripts/bonafide/build_process_witness_coarse_compatibility_v4.py",
    "scripts/bonafide/configs/process_witness_coarse_openai_v4.json",
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_revision() -> dict:
    root = Path(__file__).resolve().parents[2]
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("coarse v4 compatibility build requires clean tracked source")
    commit = _git(root, "rev-parse", "HEAD")
    files = []
    for relative in _BOUND_SOURCE_FILES:
        if _git(root, "ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"coarse v4 compatibility source is untracked: {relative}")
        blob = _git(root, "rev-parse", f"{commit}:{relative}")
        committed = subprocess.run(
            ["git", "cat-file", "blob", blob],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        expected = hashlib.sha256(committed).hexdigest()
        if file_sha256(root / relative) != expected:
            raise ValueError(
                f"coarse v4 compatibility source differs from HEAD: {relative}"
            )
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
    *,
    qualification_root: Path,
    run_root: Path,
    review_root: Path,
    human_ledger_path: Path,
    destination: Path,
) -> dict:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_revision = _source_revision()
    inputs = load_completed_v4_inputs(
        qualification_root=qualification_root,
        run_root=run_root,
        review_root=review_root,
        human_ledger_path=human_ledger_path,
    )
    report = build_compatibility_report(inputs)
    temporary = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        rows = report.pop("rows")
        report["rows_relative_path"] = "rows.jsonl"
        report["rows_sha256"] = canonical_sha256(rows)
        report["report_sha256"] = canonical_sha256(report)
        atomic_write_json(temporary / "report.json", report)
        atomic_write_jsonl(temporary / "rows.jsonl", rows)
        atomic_write_bytes(
            temporary / "sealed-human-ledger.jsonl", human_ledger_path.read_bytes()
        )
        if (
            file_sha256(temporary / "sealed-human-ledger.jsonl")
            != report["human_ledger_sha256"]
        ):
            raise ValueError("coarse v4 copied human ledger hash drift")
        manifest = {
            "schema_version": "adag.process-witness.coarse-compatibility-artifact.v4",
            "status": report["status"],
            "report_sha256": report["report_sha256"],
            "source_revision": source_revision,
            "qualification_root": str(qualification_root.resolve()),
            "run_root": str(run_root.resolve()),
            "review_root": str(review_root.resolve()),
            "human_ledger_source_path": str(human_ledger_path.resolve()),
            "human_ledger_sha256": report["human_ledger_sha256"],
            "files": [
                {
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(temporary.iterdir())
            ],
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
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--human-ledger", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                qualification_root=args.qualification_root.resolve(),
                run_root=args.run_root.resolve(),
                review_root=args.review_root.resolve(),
                human_ledger_path=args.human_ledger.resolve(),
                destination=args.destination.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
