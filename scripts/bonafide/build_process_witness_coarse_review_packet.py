#!/usr/bin/env python3
"""Build a frozen offline review packet from a completed coarse qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.coarse_sampling_review import build_review_packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_review_packet(
        qualification_root=args.qualification_root.resolve(),
        run_root=args.run_root.resolve(),
        destination=args.output.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
