#!/usr/bin/env python3
"""Prepare or validate a non-billable C2 candidate-labeling execution cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.analysis.bonafide.candidate_labeling_runtime import (
    load_candidate_labeling_execution_cohort,
    prepare_candidate_labeling_execution_cohort,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", description="Freeze requests and dependencies without API calls."
    )
    prepare.add_argument("--renderer-root", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate", description="Deep-validate a prepared execution cohort."
    )
    validate.add_argument("--cohort-root", type=Path, required=True)
    validate.add_argument(
        "--archival-validation",
        action="store_true",
        help=(
            "Validate the complete persisted execution graph and Git-bound adapter "
            "revision without reopening the renderer's upstream artifact graph."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        manifest = prepare_candidate_labeling_execution_cohort(
            renderer_root=args.renderer_root,
            config_path=args.config,
            output_root=args.output_root,
        )
        root = args.output_root.resolve()
    else:
        cohort = load_candidate_labeling_execution_cohort(
            args.cohort_root,
            verify_sources=not args.archival_validation,
        )
        manifest = cohort.manifest
        root = cohort.root
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output_root": str(root),
                "recipe_id": manifest["runtime_config"]["recipe_id"],
                "initial_request_count": manifest["initial_request_count"],
                "rewrite_dependency_count": manifest["rewrite_dependency_count"],
                "provider_model_endpoints_resolved": manifest[
                    "provider_model_endpoints_resolved"
                ],
                "calls_made": manifest["calls_made"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
