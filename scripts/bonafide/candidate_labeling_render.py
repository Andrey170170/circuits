#!/usr/bin/env python3
"""Freeze bounded generation prompts for the C2 W64 labeling comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.candidate_labeling_renderer import (
    run_candidate_labeling_renderer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest = run_candidate_labeling_renderer(
        comparison_root=args.comparison_root,
        output_root=args.output_root,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output_root": str(args.output_root.resolve()),
                "logical_prompt_count": manifest["logical_prompt_count"],
                "calls_made": manifest["calls_made"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
