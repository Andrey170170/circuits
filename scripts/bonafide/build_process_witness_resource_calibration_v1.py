#!/usr/bin/env python3
"""Build or strictly validate the process-witness resource calibration v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.process_witness_resource_calibration_v1 import (
    _load_default_tokenizer,
    _run_config,
    build_resource_calibration_v1,
    load_frozen_resource_calibration_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--sampling-v2-root", type=Path, required=True)
    build.add_argument("--destination", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    tokenizer = _load_default_tokenizer(_run_config())
    if args.command == "build":
        result = build_resource_calibration_v1(
            sampling_v2_root=args.sampling_v2_root,
            destination=args.destination,
            tokenizer=tokenizer,
        )
    else:
        result = load_frozen_resource_calibration_v1(args.root, tokenizer=tokenizer)[
            "manifest"
        ]
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
