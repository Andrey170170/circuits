#!/usr/bin/env python3
"""Manage a local, non-submitting candidate-labeling OpenAI Batch run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.candidate_labeling_openai_run import (
    build_candidate_openai_cost_plan,
    collect_candidate_openai_batch,
    construct_candidate_openai_rewrites,
    initialize_candidate_openai_run,
    load_candidate_openai_run,
    prepare_candidate_openai_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archival-validation",
        action="store_true",
        help="Validate persisted provenance without reopening upstream sources.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("initialize")
    initialize.add_argument("--cohort-root", type=Path, required=True)
    initialize.add_argument("--output-root", type=Path, required=True)
    initialize.add_argument(
        "--selection", choices=("full", "paired_anchor_smoke"), required=True
    )
    initialize.add_argument("--anchor-index", type=int)

    cost = commands.add_parser("cost-plan")
    cost.add_argument("--run-root", type=Path, required=True)
    cost.add_argument("--max-cumulative-cost-usd", type=float, required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument(
        "--stage",
        choices=("semantic_generation", "conservative_control", "semantic_rewrite"),
        required=True,
    )

    collect = commands.add_parser("collect")
    collect.add_argument("--run-root", type=Path, required=True)
    collect.add_argument(
        "--stage",
        choices=("semantic_generation", "conservative_control", "semantic_rewrite"),
        required=True,
    )
    collect.add_argument("--provider-file", type=Path, action="append", required=True)

    rewrites = commands.add_parser("construct-rewrites")
    rewrites.add_argument("--run-root", type=Path, required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--cohort-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_sources = not args.archival_validation
    if args.command == "initialize":
        run = initialize_candidate_openai_run(
            cohort_root=args.cohort_root,
            output_root=args.output_root,
            selection_kind=args.selection,
            anchor_index=args.anchor_index,
            verify_sources=verify_sources,
        )
        result = {
            "run_root": str(run.root),
            "run_manifest_sha256": run.manifest["run_manifest_sha256"],
            "initial_request_count": len(run.initial_requests),
            "planned_total_request_count": run.manifest["total_request_count_planned"],
        }
    elif args.command == "cost-plan":
        result = build_candidate_openai_cost_plan(
            run_root=args.run_root,
            max_cumulative_cost_usd=args.max_cumulative_cost_usd,
            verify_sources=verify_sources,
        )
    elif args.command == "prepare":
        result = prepare_candidate_openai_batch(
            run_root=args.run_root,
            stage_id=args.stage,
            verify_sources=verify_sources,
        )
    elif args.command == "collect":
        events = collect_candidate_openai_batch(
            run_root=args.run_root,
            stage_id=args.stage,
            provider_files=args.provider_file,
            verify_sources=verify_sources,
        )
        result = {
            "stage_id": args.stage,
            "event_count": len(events),
            "success_count": sum(
                event.result.validation_status == "success" for event in events
            ),
            "known_cost_usd": sum(
                float(event.cost.total_cost or 0.0) for event in events
            ),
        }
    elif args.command == "construct-rewrites":
        rewrites = construct_candidate_openai_rewrites(
            run_root=args.run_root, verify_sources=verify_sources
        )
        result = {"rewrite_request_count": len(rewrites)}
    else:
        run = load_candidate_openai_run(
            args.run_root,
            cohort_root=args.cohort_root,
            verify_sources=verify_sources,
        )
        result = {
            "run_root": str(run.root),
            "initial_request_count": len(run.initial_requests),
            "rewrite_request_count": len(run.rewrite_requests),
            "collected_event_count": len(run.events),
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
