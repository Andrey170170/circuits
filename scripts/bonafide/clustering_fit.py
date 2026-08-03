"""Fit one provisional sparse cluster state from a frozen sweep plan."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from circuits.analysis.bonafide.canonical import load_json_object
from circuits.analysis.bonafide.cluster_execution import fit_clustering_task

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_index = (
        args.task_index
        if args.task_index is not None
        else int(os.environ["SLURM_ARRAY_TASK_ID"])
    )
    state = fit_clustering_task(
        load_json_object(args.plan),
        repo_root=REPO_ROOT,
        task_index=task_index,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "task_index": task_index,
                "manifest_sha256": state["manifest_sha256"],
                "assigned_basis_count": state["assigned_basis_count"],
                "cluster_count": len(state["cluster_sizes"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
