"""Build one immutable exact pair-evidence state from a frozen clustering plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from circuits.analysis.bonafide.canonical import load_json_object
from circuits.analysis.bonafide.cluster_execution import (
    validate_clustering_plan,
)
from circuits.analysis.bonafide.clustering_store import (
    build_pair_evidence_from_feature_store,
    write_pair_evidence_build,
)

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
    plan = validate_clustering_plan(
        load_json_object(args.plan),
        repo_root=REPO_ROOT,
        verify_code=True,
    )
    evidence_tasks = plan["pair_evidence"]
    if task_index < 0 or task_index >= len(evidence_tasks):
        raise ValueError("pair-evidence task index is out of range")
    task = evidence_tasks[task_index]
    build = build_pair_evidence_from_feature_store(
        Path(str(plan["feature_store"]["path"])),
        weighting=str(task["weighting"]),  # type: ignore[arg-type]
    )
    manifest = write_pair_evidence_build(
        Path(str(task["output_path"])),
        build,
        code_revision=plan["code_revision"],
        environment=plan["environment"],
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "task_index": task_index,
                "weighting": task["weighting"],
                "manifest_sha256": manifest["manifest_sha256"],
                "pair_count": manifest["matrix_records"]["overlap_count"]["nnz"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
