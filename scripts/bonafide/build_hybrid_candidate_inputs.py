#!/usr/bin/env python3
"""Build immutable candidate-union inputs for paper-style clustering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.hybrid_candidate_inputs import build_hybrid_input_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    manifest = build_hybrid_input_bundle(
        source_root=args.source_root,
        output_root=args.output_root,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                **manifest["counts"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
