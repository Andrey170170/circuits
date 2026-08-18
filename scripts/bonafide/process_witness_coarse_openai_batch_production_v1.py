#!/usr/bin/env python3
"""Initialize, submit, inspect, collect, and finalize coarse production v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 import (
    check_recovery_shard,
    check_shard,
    collect_recovery_shard,
    collect_shard,
    finalize_campaign,
    initialize_campaign_run,
    prepare_failed_only_recovery,
    recover_recovery_submission,
    recover_shard_submission,
    submit_recovery_shard,
    submit_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--bundle-root", type=Path, required=True)
    initialize.add_argument("--run-root", type=Path, required=True)
    initialize.add_argument("--maximum-authorized-cost-usd", type=float, required=True)
    initialize.add_argument("--authorization-note", required=True)
    initialize.add_argument(
        "--provider-queued-input-token-limit", type=int, required=True
    )
    initialize.add_argument("--maximum-concurrent-shards", type=int, default=1)
    for name in (
        "submit-shard",
        "status-shard",
        "collect-shard",
        "recover-shard-submission",
        "submit-recovery-shard",
        "status-recovery-shard",
        "collect-recovery-shard",
        "recover-recovery-submission",
    ):
        command = commands.add_parser(name)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--shard-id", required=True)
    recovery = commands.add_parser("prepare-recovery")
    recovery.add_argument("--run-root", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        result = initialize_campaign_run(
            bundle_root=args.bundle_root.resolve(),
            run_root=args.run_root.resolve(),
            maximum_authorized_cost_usd=args.maximum_authorized_cost_usd,
            authorization_note=args.authorization_note,
            provider_queued_input_token_limit=args.provider_queued_input_token_limit,
            maximum_concurrent_shards=args.maximum_concurrent_shards,
        )
    elif args.command == "submit-shard":
        result = submit_shard(run_root=args.run_root.resolve(), shard_id=args.shard_id)
    elif args.command == "status-shard":
        result = check_shard(run_root=args.run_root.resolve(), shard_id=args.shard_id)
    elif args.command == "collect-shard":
        result = collect_shard(run_root=args.run_root.resolve(), shard_id=args.shard_id)
    elif args.command == "recover-shard-submission":
        result = recover_shard_submission(
            run_root=args.run_root.resolve(), shard_id=args.shard_id
        )
    elif args.command == "submit-recovery-shard":
        result = submit_recovery_shard(
            run_root=args.run_root.resolve(), shard_id=args.shard_id
        )
    elif args.command == "status-recovery-shard":
        result = check_recovery_shard(
            run_root=args.run_root.resolve(), shard_id=args.shard_id
        )
    elif args.command == "collect-recovery-shard":
        result = collect_recovery_shard(
            run_root=args.run_root.resolve(), shard_id=args.shard_id
        )
    elif args.command == "recover-recovery-submission":
        result = recover_recovery_submission(
            run_root=args.run_root.resolve(), shard_id=args.shard_id
        )
    elif args.command == "prepare-recovery":
        result = prepare_failed_only_recovery(run_root=args.run_root.resolve())
    else:
        result = finalize_campaign(
            run_root=args.run_root.resolve(), destination=args.destination.resolve()
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
