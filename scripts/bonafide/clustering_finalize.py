"""Freeze the primary and alternative label-free cluster states."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.analysis.bonafide.cluster_execution import (
    collect_clustering_code_revision,
    collect_clustering_environment,
)
from circuits.analysis.bonafide.clustering_selection import (
    build_selected_states,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--structural-report", type=Path, required=True)
    parser.add_argument("--projection-root", type=Path, required=True)
    parser.add_argument("--resample-plan", type=Path, required=True)
    parser.add_argument("--resample-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    revision = collect_clustering_code_revision(REPO_ROOT)
    if revision["git_dirty"]:
        raise ValueError("refuse to freeze selected states from dirty source")
    manifest = build_selected_states(
        source_plan_path=args.source_plan,
        structural_report_path=args.structural_report,
        projection_root=args.projection_root,
        resample_plan_path=args.resample_plan,
        resample_report_path=args.resample_report,
        output_root=args.output_root,
        code_revision=revision,
        environment=collect_clustering_environment(),
    )
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "primary_source_task_index": (manifest["primary_source_task_index"]),
                "alternative_source_task_index": (
                    manifest["alternative_source_task_index"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
