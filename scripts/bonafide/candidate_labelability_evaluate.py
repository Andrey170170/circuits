#!/usr/bin/env python3
"""Run the frozen label-free candidate labelability evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.candidate_labelability_evaluation import (
    run_candidate_labelability_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    report = run_candidate_labelability_evaluation(
        input_root=args.input_root,
        baseline_root=args.baseline_root,
        output_path=args.output_path,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "manifest_sha256": report["manifest_sha256"],
                "output_path": str(args.output_path.resolve()),
                "chosen_cluster_count": report["evaluation"]["chosen_cluster_count"],
                "gate_status": report["evaluation"]["pre_null_pre_jackknife_gates"][
                    "status"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
