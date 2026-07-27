"""Execute one immutable dense downstream response-shard task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from circuits.analysis.bonafide.build_plan import validate_downstream_plan
from circuits.analysis.bonafide.canonical import load_json_object
from circuits.analysis.bonafide.streaming import build_response_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--allow-development-plan",
        action="store_true",
        help="allow a dirty-source plan for bounded development smoke only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = validate_downstream_plan(
        load_json_object(args.plan),
        verify_inputs=True,
        verify_code=True,
    )
    if plan["development"] and not args.allow_development_plan:
        raise ValueError("production downstream builder refuses a development plan")
    result = build_response_shard(plan=plan, task_index=args.task_index)
    print(
        json.dumps(
            {
                "status": result["status"],
                "plan_sha256": plan["plan_sha256"],
                "lane": plan["lane"],
                "task_index": args.task_index,
                "response_id": result["manifest"]["response_id"],
                "manifest_sha256": result["manifest"]["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
