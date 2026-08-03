"""Freeze family-blocked and checkpoint refits for cluster candidates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.analysis.bonafide.clustering_resampling import (
    build_resample_plan,
    write_resample_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--structural-report", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_resample_plan(
        repo_root=REPO_ROOT,
        source_plan_path=args.source_plan,
        structural_report_path=args.structural_report,
        projection_manifest_path=args.projection_manifest,
        output_root=args.output_root,
    )
    write_resample_plan(args.plan, plan)
    print(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "evidence_task_count": len(plan["evidence_tasks"]),
                "fit_task_count": len(plan["fit_tasks"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
