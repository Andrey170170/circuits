#!/usr/bin/env python3
"""Publish the metrics-free executable-only candidate-identity source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.candidate_identity_source import (
    build_candidate_identity_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = build_candidate_identity_source(
        output_root=args.output_root,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "counts": manifest["counts"],
                "exposure_contract": manifest["exposure_contract"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
