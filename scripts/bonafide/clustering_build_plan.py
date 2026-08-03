"""Freeze the initial exact sparse clustering sweep."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.analysis.bonafide.cluster_execution import (
    build_clustering_plan,
    write_clustering_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_clustering_plan(
        repo_root=REPO_ROOT,
        feature_store_root=args.feature_store,
        output_root=args.output_root,
    )
    write_clustering_plan(args.plan, plan)
    print(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "pair_evidence_task_count": len(plan["pair_evidence"]),
                "clustering_task_count": len(plan["configurations"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
