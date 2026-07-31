"""Build immutable C2 candidate-aware clustering inputs and family partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.candidate_profiles import (
    build_candidate_cluster_input_bundle,
)
from circuits.analysis.bonafide.canonical import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPED_PATHS = (
    "circuits/analysis/bonafide/canonical.py",
    "circuits/analysis/bonafide/candidate_profiles.py",
    "circuits/analysis/bonafide/clustering_evaluation.py",
    "circuits/analysis/bonafide/identity.py",
    "circuits/tracing/artifact.py",
    "circuits/tracing/candidates.py",
    "circuits/tracing/candidate_union.py",
    "docs/CANDIDATE_AWARE_CLUSTERING_LABELABILITY_PROTOCOL.md",
    "scripts/bonafide/build_candidate_cluster_inputs.py",
    "pyproject.toml",
    "uv.lock",
)


def collect_candidate_input_code_revision(repo_root: Path) -> dict[str, Any]:
    """Bind the small executable surface and refuse source drift at execution."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    tracked_status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for relative in SCOPED_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"candidate input source is missing: {path}")
        if git("ls-files", "--error-unmatch", "--", relative) != relative:
            raise ValueError(f"candidate input source is not tracked: {relative}")
        encoded = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        files.append({"path": relative, "sha256": file_sha256(path)})
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "git_dirty": bool(tracked_status),
        "git_status_sha256": hashlib.sha256(tracked_status.encode("utf-8")).hexdigest(),
        "source_tree_sha256": digest.hexdigest(),
        "files": files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate-union-plan", type=Path, required=True)
    parser.add_argument("--c2-report", type=Path, required=True)
    parser.add_argument("--salvage-report", type=Path, required=True)
    parser.add_argument("--width1-root", type=Path, required=True)
    parser.add_argument("--candidate-union-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    revision = collect_candidate_input_code_revision(REPO_ROOT)
    if revision["git_dirty"]:
        raise ValueError(
            "refuse to freeze candidate clustering inputs from dirty source"
        )
    manifest = build_candidate_cluster_input_bundle(
        selection_path=args.selection,
        plan_path=args.candidate_union_plan,
        c2_report_path=args.c2_report,
        salvage_report_path=args.salvage_report,
        width1_root=args.width1_root,
        candidate_union_root=args.candidate_union_root,
        output_root=args.output_root,
        code_revision=revision,
        protocol_path=args.protocol,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": manifest["manifest_sha256"],
                "target_count": manifest["cohort"]["target_count"],
                "width_profile_row_count": manifest["cohort"][
                    "width_profile_row_count"
                ],
                "candidate_profile_row_count": manifest["cohort"][
                    "candidate_profile_row_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
