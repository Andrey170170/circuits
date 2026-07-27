"""Execute one joint feature/multiplex dense response-shard task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from circuits.analysis.bonafide.build_plan import validate_downstream_plan
from circuits.analysis.bonafide.canonical import load_json_object
from circuits.analysis.bonafide.streaming import build_joint_response_shards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--multiplex-plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--allow-development-plan",
        action="store_true",
        help="allow dirty-source plans for bounded development smoke only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    feature_plan = validate_downstream_plan(
        load_json_object(args.feature_plan),
        verify_inputs=True,
        verify_code=True,
    )
    multiplex_plan = validate_downstream_plan(
        load_json_object(args.multiplex_plan),
        verify_inputs=True,
        verify_code=True,
    )
    if (
        feature_plan["development"] or multiplex_plan["development"]
    ) and not args.allow_development_plan:
        raise ValueError("production joint builder refuses a development plan")
    result = build_joint_response_shards(
        feature_plan=feature_plan,
        multiplex_plan=multiplex_plan,
        task_index=args.task_index,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "task_index": args.task_index,
                "lanes": {
                    lane: {
                        "status": lane_result["status"],
                        "plan_sha256": lane_result["manifest"]["plan_sha256"],
                        "response_id": lane_result["manifest"]["response_id"],
                        "manifest_sha256": lane_result["manifest"]["manifest_sha256"],
                    }
                    for lane, lane_result in sorted(result["lanes"].items())
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
