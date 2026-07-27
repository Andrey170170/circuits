"""Prepare and execute the frozen, provider-neutral BonaFide labeling comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from circuits.labeling.runtime import execute_live, prepare_candidate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-candidates")
    prepare.add_argument("--frozen-root", type=Path, required=True)
    prepare.add_argument("--recipe", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument(
        "--states",
        nargs="+",
        choices=("primary", "alternative"),
        default=("primary", "alternative"),
    )
    prepare.add_argument("--cluster-limit", type=int)
    prepare.add_argument("--run-id")
    prepare.add_argument("--transport-override", choices=("live", "native_batch"))
    prepare.add_argument("--allow-dirty", action="store_true")

    live = subparsers.add_parser("execute-live")
    live.add_argument("--run-root", type=Path, required=True)
    live.add_argument("--stage", default="candidate_generation")
    live.add_argument("--request-id", action="append", dest="request_ids")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-candidates":
        manifest = prepare_candidate_run(
            frozen_root=args.frozen_root,
            recipe_path=args.recipe,
            output_root=args.output_root,
            states=args.states,
            cluster_limit=args.cluster_limit,
            run_id=args.run_id,
            transport_override=args.transport_override,
            allow_dirty=args.allow_dirty,
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "run_id": manifest["run_id"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "selected_clusters": manifest["selected_clusters"],
                    "request_count": manifest["request_files"][0]["request_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    counts = asyncio.run(
        execute_live(
            run_root=args.run_root,
            stage=args.stage,
            request_ids=set(args.request_ids) if args.request_ids else None,
        )
    )
    print(json.dumps(counts, sort_keys=True))
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
