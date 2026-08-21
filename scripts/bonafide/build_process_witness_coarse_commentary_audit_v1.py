#!/usr/bin/env python3
"""Build or validate the frozen non-blind coarse commentary packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.coarse_sampling_commentary_audit_v1 import (
    build_commentary_audit_packet,
    load_commentary_audit_packet,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--analysis-root", type=Path, required=True)
    build.add_argument("--sampling-root", type=Path, required=True)
    build.add_argument("--destination", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_commentary_audit_packet(
            analysis_root=args.analysis_root,
            sampling_root=args.sampling_root,
            destination=args.destination,
        )
    else:
        result = load_commentary_audit_packet(args.root)["manifest"]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
