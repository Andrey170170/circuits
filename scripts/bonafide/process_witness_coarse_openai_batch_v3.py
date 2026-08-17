#!/usr/bin/env python3
"""Submit, inspect, or collect the frozen coarse qualification v3 Batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.coarse_sampling_openai_batch_v3 import (
    cancel_v3_batch,
    check_v3_batch,
    collect_v3_batch,
    recover_v3_submission,
    submit_v3_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--qualification-root", type=Path, required=True)
    submit.add_argument("--run-root", type=Path, required=True)
    submit.add_argument("--maximum-authorized-cost-usd", type=float, required=True)
    submit.add_argument("--authorization-note", required=True)
    status = commands.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--run-root", type=Path, required=True)
    recover = commands.add_parser("recover-submission")
    recover.add_argument("--run-root", type=Path, required=True)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("--run-root", type=Path, required=True)
    cancel.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command == "submit":
        result = submit_v3_batch(
            qualification_root=args.qualification_root.resolve(),
            run_root=args.run_root.resolve(),
            maximum_authorized_cost_usd=args.maximum_authorized_cost_usd,
            authorization_note=args.authorization_note,
        )
    elif args.command == "status":
        result = check_v3_batch(run_root=args.run_root.resolve())
    elif args.command == "collect":
        result = collect_v3_batch(run_root=args.run_root.resolve())
    elif args.command == "recover-submission":
        result = recover_v3_submission(args.run_root.resolve())
    else:
        result = cancel_v3_batch(run_root=args.run_root.resolve(), reason=args.reason)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
