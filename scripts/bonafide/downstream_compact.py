"""Validate and compact all response shards for one dense downstream lane."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.analysis.bonafide.build_plan import validate_downstream_plan
from circuits.analysis.bonafide.canonical import load_json_object
from circuits.analysis.bonafide.compaction import compact_downstream_lane


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--allow-development-plan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = validate_downstream_plan(
        load_json_object(args.plan),
        verify_inputs=True,
        verify_code=True,
    )
    if plan["development"] and not args.allow_development_plan:
        raise ValueError("production downstream compactor refuses a development plan")
    result = compact_downstream_lane(plan)
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "plan_sha256": plan["plan_sha256"],
                "lane": plan["lane"],
                "manifest_sha256": manifest["manifest_sha256"],
                "response_count": manifest["response_count"],
                "target_count": manifest["target_count"],
                "signed_basis_count": manifest["signed_basis_count"],
                "resource_estimate": manifest["resource_estimate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
