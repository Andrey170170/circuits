"""Build hybrid labeling evidence or enforce its conservative API cost ceiling."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.analysis.bonafide.hybrid_candidate_labeling import (
    build_hybrid_labeling_bundle,
    load_hybrid_labeling_bundle,
)
from circuits.labeling.cost_guard import build_pre_submit_cost_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("build-bundle")
    bundle.add_argument("--evaluation-root", type=Path, required=True)
    bundle.add_argument("--input-root", type=Path, required=True)
    bundle.add_argument("--fit-root", type=Path, required=True)
    bundle.add_argument("--output-root", type=Path, required=True)
    validate = commands.add_parser("validate-bundle")
    validate.add_argument("--root", type=Path, required=True)
    cost = commands.add_parser("cost-plan")
    cost.add_argument("--run-root", type=Path, required=True)
    cost.add_argument("--max-cumulative-cost-usd", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-bundle":
        value = build_hybrid_labeling_bundle(
            evaluation_root=args.evaluation_root,
            input_root=args.input_root,
            fit_root=args.fit_root,
            output_root=args.output_root,
        )
    elif args.command == "validate-bundle":
        value = load_hybrid_labeling_bundle(args.root)
    else:
        value = build_pre_submit_cost_plan(
            run_root=args.run_root,
            max_cumulative_cost_usd=args.max_cumulative_cost_usd,
        )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
