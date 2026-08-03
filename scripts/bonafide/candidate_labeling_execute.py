#!/usr/bin/env python3
"""Execute or validate deterministic candidate-labeling evaluation plumbing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.candidate_labeling_execution import (
    execute_candidate_labeling_fake_evaluation,
    load_candidate_labeling_fake_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("fake", description="Run the zero-cost adapter.")
    execute.add_argument("--cohort-root", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate", description="Deep-validate a completed fake evaluation."
    )
    validate.add_argument("--evaluation-root", type=Path, required=True)
    validate.add_argument("--cohort-root", type=Path)
    for command in (execute, validate):
        command.add_argument(
            "--archival-validation",
            action="store_true",
            help="Validate persisted provenance without reopening upstream sources.",
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fake":
        evaluation = execute_candidate_labeling_fake_evaluation(
            cohort_root=args.cohort_root,
            output_root=args.output_root,
            verify_sources=not args.archival_validation,
        )
    else:
        evaluation = load_candidate_labeling_fake_evaluation(
            args.evaluation_root,
            cohort_root=args.cohort_root,
            verify_sources=not args.archival_validation,
        )
    print(
        json.dumps(
            {
                "schema_version": evaluation.completion_manifest["schema_version"],
                "completion_sha256": evaluation.completion_manifest[
                    "completion_sha256"
                ],
                "output_root": str(evaluation.root),
                "successful_event_count": evaluation.completion_manifest[
                    "successful_event_count"
                ],
                "api_call_count": 0,
                "known_cost_usd": 0.0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
