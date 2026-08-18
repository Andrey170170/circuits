#!/usr/bin/env python3
"""Prepare and operate the additive coarse-production continuation v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.coarse_sampling_openai_batch_continuation_v1 import (
    collect_attempt,
    finalize_continuation,
    prepare_continuation,
    prepare_failed_only_recovery,
    recover_attempt_submission,
    status_attempt,
    submit_attempt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--bundle-root", type=Path, required=True)
    prepare.add_argument("--calibration-run-root", type=Path, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--provider-queued-input-token-limit", type=int, required=True)
    prepare.add_argument("--tranche-empirical-queue-cap", type=int, required=True)
    prepare.add_argument("--maximum-concurrent-attempts", type=int, default=1)
    prepare.add_argument("--authorized-forecast-budget-usd", type=float, required=True)
    prepare.add_argument("--warning-spend-threshold-usd", type=float, required=True)
    prepare.add_argument("--hard-campaign-stop-usd", type=float, required=True)
    prepare.add_argument("--authorization-note", required=True)
    prepare.add_argument("--calibration-observed-input-tokens", type=int, required=True)
    prepare.add_argument("--calibration-forecast-input-tokens", type=int, required=True)
    for name in (
        "submit-attempt",
        "recover-attempt-submission",
        "status-attempt",
        "collect-attempt",
    ):
        command = commands.add_parser(name)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--attempt-id", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--destination", type=Path, required=True)
    recovery = commands.add_parser("prepare-failed-only-recovery")
    recovery.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_continuation(
            bundle_root=args.bundle_root.resolve(),
            calibration_run_root=args.calibration_run_root.resolve(),
            run_root=args.run_root.resolve(),
            provider_queued_input_token_limit=args.provider_queued_input_token_limit,
            tranche_empirical_queue_cap=args.tranche_empirical_queue_cap,
            maximum_concurrent_attempts=args.maximum_concurrent_attempts,
            authorized_forecast_budget_usd=args.authorized_forecast_budget_usd,
            warning_spend_threshold_usd=args.warning_spend_threshold_usd,
            hard_campaign_stop_usd=args.hard_campaign_stop_usd,
            authorization_note=args.authorization_note,
            calibration_observed_input_tokens=args.calibration_observed_input_tokens,
            calibration_forecast_input_tokens=args.calibration_forecast_input_tokens,
        )
    elif args.command == "submit-attempt":
        result = submit_attempt(
            run_root=args.run_root.resolve(), attempt_id=args.attempt_id
        )
    elif args.command == "recover-attempt-submission":
        result = recover_attempt_submission(
            run_root=args.run_root.resolve(), attempt_id=args.attempt_id
        )
    elif args.command == "status-attempt":
        result = status_attempt(
            run_root=args.run_root.resolve(), attempt_id=args.attempt_id
        )
    elif args.command == "collect-attempt":
        result = collect_attempt(
            run_root=args.run_root.resolve(), attempt_id=args.attempt_id
        )
    elif args.command == "prepare-failed-only-recovery":
        result = prepare_failed_only_recovery(run_root=args.run_root.resolve())
    else:
        result = finalize_continuation(
            run_root=args.run_root.resolve(), destination=args.destination.resolve()
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
