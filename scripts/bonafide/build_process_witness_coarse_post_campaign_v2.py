#!/usr/bin/env python3
"""Build or validate the additive coarse post-campaign sampling v2 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.coarse_sampling_post_campaign_v2 import (
    DEFAULT_COHORT_ROOT,
    build_post_campaign_sampling_v2,
    load_frozen_post_campaign_sampling_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--parent-v1-root", type=Path, required=True)
    build.add_argument("--destination", type=Path, required=True)
    build.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--parent-v1-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_post_campaign_sampling_v2(
            parent_v1_root=args.parent_v1_root,
            destination=args.destination,
            cohort_root=args.cohort_root,
        )
    else:
        result = load_frozen_post_campaign_sampling_v2(
            args.root, parent_v1_root=args.parent_v1_root
        )["manifest"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
