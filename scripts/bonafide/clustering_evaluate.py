"""Build the label-free structural report for a completed clustering sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from circuits.analysis.bonafide.clustering_evaluation import (
    build_structural_report,
    write_structural_report,
)
from circuits.analysis.bonafide.cluster_execution import (
    collect_clustering_code_revision,
    collect_clustering_environment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    revision = collect_clustering_code_revision(REPO_ROOT)
    if revision["git_dirty"]:
        raise ValueError("refuse to persist structural report from dirty source")
    report = build_structural_report(
        args.source_plan,
        code_revision=revision,
        environment=collect_clustering_environment(),
    )
    write_structural_report(args.output, report)
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "candidate_resolutions": [
                    {
                        "n_clusters": candidate["n_clusters"],
                        "medoid_task_index": candidate["medoid_task_index"],
                        "passes_preliminary_gates": candidate[
                            "passes_preliminary_gates"
                        ],
                    }
                    for candidate in report["candidate_resolutions"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
