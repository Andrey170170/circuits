"""Create or validate an immutable dense downstream build plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.analysis.bonafide.build_plan import (
    BUILD_LANES,
    build_downstream_plan,
    validate_downstream_plan,
    write_downstream_plan,
)
from circuits.analysis.bonafide.canonical import load_json_object

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--inventory", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--lane", choices=BUILD_LANES, required=True)
    create.add_argument("--plan-output", type=Path, required=True)
    create.add_argument("--allow-dirty-development", action="store_true")
    create.add_argument("--require-frozen-dense", action="store_true")
    create.add_argument("--development-targets-per-response", type=int)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--verify-code", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create":
        plan = build_downstream_plan(
            inventory_path=args.inventory,
            output_root=args.output_root,
            lane=args.lane,
            repo_root=REPO_ROOT,
            allow_dirty_development=args.allow_dirty_development,
            require_frozen_dense=args.require_frozen_dense,
            development_targets_per_response=(args.development_targets_per_response),
        )
        write_downstream_plan(args.plan_output, plan)
    else:
        plan = validate_downstream_plan(
            load_json_object(args.plan),
            verify_inputs=True,
            verify_code=args.verify_code,
        )
    print(
        json.dumps(
            {
                "plan_sha256": plan["plan_sha256"],
                "lane": plan["lane"],
                "development": plan["development"],
                "dense_summary": plan["dense_summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
