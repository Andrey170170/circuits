#!/usr/bin/env python3
"""Run the frozen hybrid candidate-cluster labelability evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.hybrid_candidate_labelability import (
    run_hybrid_candidate_labelability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_hybrid_candidate_labelability(
        input_root=args.input_root,
        fit_root=args.fit_root,
        output_root=args.output_root,
        repo_root=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "manifest_sha256": report["manifest_sha256"],
                "output_root": str(args.output_root.resolve()),
                "exploratory_labeling_authorized": report[
                    "exploratory_labeling_authorized"
                ],
                "states": {
                    role: state["exploratory_labeling_authorized"]
                    for role, state in report["states"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
