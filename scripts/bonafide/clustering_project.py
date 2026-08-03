"""Project structurally credible cluster states onto the dense multiplex."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.analysis.bonafide.cluster_execution import (
    collect_clustering_code_revision,
    collect_clustering_environment,
)
from circuits.analysis.bonafide.clustering_projection import (
    build_multiplex_projection,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--structural-report", type=Path, required=True)
    parser.add_argument("--multiplex-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    revision = collect_clustering_code_revision(REPO_ROOT)
    if revision["git_dirty"]:
        raise ValueError("refuse to persist projection from dirty source")
    manifest = build_multiplex_projection(
        source_plan_path=args.source_plan,
        structural_report_path=args.structural_report,
        multiplex_root=args.multiplex_root,
        output_root=args.output_root,
        code_revision=revision,
        environment=collect_clustering_environment(),
    )
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "candidates": [
                    {
                        "task_index": candidate["task_index"],
                        "n_clusters": candidate["n_clusters"],
                        "metrics": candidate["metrics"],
                    }
                    for candidate in manifest["candidates"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
