#!/usr/bin/env python3
"""Build or validate the frozen coarse post-campaign analysis artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.coarse_sampling_post_campaign_v1 import (
    build_post_campaign_analysis,
    load_frozen_post_campaign_analysis,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--bundle-root", type=Path, required=True)
    build.add_argument("--run-root", type=Path, required=True)
    build.add_argument("--destination", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_post_campaign_analysis(
            bundle_root=args.bundle_root,
            run_root=args.run_root,
            destination=args.destination,
        )
    else:
        result = load_frozen_post_campaign_analysis(args.root)["manifest"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
