#!/usr/bin/env python3
"""Build the label-free C2 candidate/width multiplex assessment artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.candidate_multiplex_assessment import (
    build_candidate_multiplex_assessment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-input-root", type=Path, required=True)
    parser.add_argument("--c2-baseline-root", type=Path, required=True)
    parser.add_argument("--dense-multiplex-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--without-occurrence-projection",
        action="store_true",
        help="omit exact dense occurrence pointer rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_candidate_multiplex_assessment(
        c2_input_root=args.c2_input_root,
        c2_baseline_root=args.c2_baseline_root,
        dense_multiplex_root=args.dense_multiplex_root,
        output_root=args.output_root,
        repo_root=args.repo_root,
        include_occurrence_projection=not args.without_occurrence_projection,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "overlap": manifest["overlap"],
                "counts": manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
