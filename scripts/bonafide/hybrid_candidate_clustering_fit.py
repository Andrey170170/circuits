#!/usr/bin/env python3
"""Fit and atomically persist fresh hybrid candidate cluster states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.hybrid_candidate_clustering_execution import (
    run_hybrid_candidate_clustering,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    manifest = run_hybrid_candidate_clustering(
        input_root=args.input_root,
        output_root=args.output_root,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": manifest["manifest_sha256"],
                "fit_count": len(manifest["fits"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
