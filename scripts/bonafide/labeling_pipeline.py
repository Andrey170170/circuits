"""Prepare and execute the frozen, provider-neutral BonaFide labeling comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from circuits.labeling.runtime import (
    execute_live,
    prepare_candidate_run,
    prepare_summary_stage,
    retry_failed_generation,
    validate_explicit_cluster_selection,
)


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
    prepare.add_argument("--primary-clusters", nargs="+", type=int)
    prepare.add_argument("--alternative-clusters", nargs="+", type=int)
    prepare.add_argument("--run-id")
    prepare.add_argument("--transport-override", choices=("live", "native_batch"))
    prepare.add_argument("--allow-dirty", action="store_true")

    live = subparsers.add_parser("execute-live")
    live.add_argument("--run-root", type=Path, required=True)
    live.add_argument("--stage", default="candidate_generation")
    live.add_argument("--request-id", action="append", dest="request_ids")

    retry = subparsers.add_parser("retry-failed")
    retry.add_argument("--run-root", type=Path, required=True)
    retry.add_argument(
        "--stage",
        choices=("candidate_generation", "cluster_summary"),
        required=True,
    )
    retry.add_argument(
        "--request-id",
        action="append",
        dest="request_ids",
        required=True,
    )
    retry.add_argument("--max-output-tokens", type=int, required=True)

    scoring = subparsers.add_parser("score-local")
    scoring.add_argument("--run-root", type=Path, required=True)
    scoring.add_argument(
        "--phase",
        required=True,
        choices=("candidate_selection", "summary_selection", "summary_audit"),
    )
    scoring.add_argument("--states", nargs="+", choices=("primary", "alternative"))
    scoring.add_argument("--cluster-id", action="append", type=int, dest="cluster_ids")

    summary = subparsers.add_parser("prepare-summaries")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--transport-override", choices=("live", "native_batch"))

    quality = subparsers.add_parser("assess-quality")
    quality.add_argument("--run-root", type=Path, required=True)

    telemetry = subparsers.add_parser("summarize-telemetry")
    telemetry.add_argument("--run-root", type=Path, required=True)

    for command in ("prepare-batch", "submit-batch", "batch-status", "collect-batch"):
        batch = subparsers.add_parser(command)
        batch.add_argument("--run-root", type=Path, required=True)
        batch.add_argument(
            "--stage",
            choices=("candidate_generation", "cluster_summary"),
            required=True,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-candidates":
        explicit_clusters = {
            state: values
            for state, values in (
                ("primary", args.primary_clusters),
                ("alternative", args.alternative_clusters),
            )
            if values is not None
        }
        validate_explicit_cluster_selection(
            args.states, explicit_clusters or None, args.cluster_limit
        )
        manifest = prepare_candidate_run(
            frozen_root=args.frozen_root,
            recipe_path=args.recipe,
            output_root=args.output_root,
            states=args.states,
            cluster_limit=args.cluster_limit,
            explicit_clusters=explicit_clusters or None,
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
    if args.command == "execute-live":
        counts = asyncio.run(
            execute_live(
                run_root=args.run_root,
                stage=args.stage,
                request_ids=set(args.request_ids) if args.request_ids else None,
            )
        )
        print(json.dumps(counts, sort_keys=True))
        return 0 if counts["failed"] == 0 else 1
    if args.command == "retry-failed":
        counts = asyncio.run(
            retry_failed_generation(
                run_root=args.run_root,
                stage=args.stage,
                request_ids=set(args.request_ids),
                max_output_tokens=args.max_output_tokens,
            )
        )
        print(json.dumps(counts, sort_keys=True))
        return 0 if counts["failed"] == 0 else 1
    if args.command == "score-local":
        from circuits.labeling.scoring import score_run

        counts = score_run(
            run_root=args.run_root,
            phase=args.phase,
            states=set(args.states) if args.states else None,
            cluster_ids=set(args.cluster_ids) if args.cluster_ids else None,
        )
        print(json.dumps(counts, sort_keys=True))
        return 0
    if args.command in (
        "prepare-batch",
        "submit-batch",
        "batch-status",
        "collect-batch",
    ):
        from circuits.labeling.batch_runtime import (
            collect_native_batch,
            native_batch_status,
            prepare_native_batch,
            submit_native_batch,
        )

        if args.command == "prepare-batch":
            value = prepare_native_batch(args.run_root, args.stage)
        elif args.command == "submit-batch":
            value = submit_native_batch(args.run_root, args.stage)
        elif args.command == "batch-status":
            value = native_batch_status(args.run_root, args.stage)
        else:
            value = collect_native_batch(args.run_root, args.stage)
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.command == "assess-quality":
        from circuits.labeling.quality import assess_width_one_quality

        value = assess_width_one_quality(run_root=args.run_root)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "counts": value["counts"],
                    "manifest_sha256": value["manifest_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "summarize-telemetry":
        from circuits.labeling.telemetry import summarize_telemetry

        print(json.dumps(summarize_telemetry(run_root=args.run_root), sort_keys=True))
        return 0
    stage_manifest = prepare_summary_stage(
        run_root=args.run_root,
        transport_override=args.transport_override,
    )
    print(
        json.dumps(
            {
                "status": "planned",
                "stage": stage_manifest["stage"],
                "request_count": stage_manifest["request_file"]["request_count"],
                "manifest_sha256": stage_manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
