"""Validate resampled states and build the family/checkpoint stability report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from circuits.analysis.bonafide.clustering_resampling import (
    build_resample_report,
    write_resample_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_resample_report(args.plan, repo_root=REPO_ROOT)
    write_resample_report(args.output, report)
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "candidates": [
                    {
                        "source_task_index": candidate["source_task_index"],
                        "n_clusters": candidate["n_clusters"],
                        "family_jackknife_median_ari": candidate[
                            "family_jackknife_median_ari"
                        ],
                        "family_jackknife_p10_ari": candidate[
                            "family_jackknife_p10_ari"
                        ],
                        "passes_family_jackknife_gate": candidate[
                            "passes_family_jackknife_gate"
                        ],
                    }
                    for candidate in report["candidates"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
