#!/usr/bin/env python3
"""Run and atomically persist the candidate-aware clustering baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.candidate_clustering_execution import (
    run_candidate_clustering_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest = run_candidate_clustering_baseline(
        input_root=args.input_root,
        output_root=args.output_root,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "output_root": str(args.output_root.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "basis_count": manifest["basis_count"],
                "common_eligible_basis_count": manifest["common_eligible_basis_count"],
                "chosen_cluster_count": manifest["chosen_cluster_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
