#!/usr/bin/env python3
"""Build label-free C2-W64 candidate multiplex diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.candidate_multiplex_diagnostics import (
    build_candidate_multiplex_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = build_candidate_multiplex_diagnostics(
        assessment_root=args.assessment_root,
        output_root=args.output_root,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "decision": manifest["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
