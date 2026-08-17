#!/usr/bin/env python3
"""Build the immutable full-response two-arm blind review packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.coarse_sampling_review_v2 import (
    build_review_packet_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_review_packet_v2(
        qualification_root=args.qualification_root.resolve(),
        run_root=args.run_root.resolve(),
        comparison_root=args.comparison_root.resolve(),
        destination=args.output.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
