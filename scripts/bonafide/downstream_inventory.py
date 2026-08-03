"""Build the frozen width-one BonaFide downstream inventory."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from circuits.analysis.bonafide.inventory import build_inventory, write_inventory

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTION = (
    REPO_ROOT / "scripts/bonafide/manifests/qwen3_4b_instruct_final_traces.json"
)
DEFAULT_EXECUTION_PLAN = (
    REPO_ROOT / "scripts/bonafide/manifests/qwen3_4b_instruct_final_execution_plan.json"
)


def _default_artifact_root() -> Path:
    results_root = os.environ.get("CIRCUITS_RESULTS_DIR")
    if not results_root:
        raise ValueError(
            "CIRCUITS_RESULTS_DIR is unset; pass --artifact-root explicitly"
        )
    return Path(results_root) / "bonafide/final-traces"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--execution-plan", type=Path, default=DEFAULT_EXECUTION_PLAN)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-level",
        choices=("integrity", "full"),
        default="full",
        help="'full' also unpickles and numerically validates every compact trace",
    )
    parser.add_argument(
        "--require-frozen-baseline",
        action="store_true",
        help="exit nonzero unless the expected 2594/2595 baseline is exact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_root = args.artifact_root or _default_artifact_root()
    inventory = build_inventory(
        selection_path=args.selection,
        execution_plan_path=args.execution_plan,
        artifact_root=artifact_root,
        validation_level=args.validation_level,
    )
    write_inventory(args.output, inventory)
    summary = inventory["summary"]
    print(json.dumps(summary, sort_keys=True))
    if args.require_frozen_baseline:
        expected = {
            "planned": 2595,
            "completed": 2594,
            "discovery_planned": 2467,
            "discovery_completed": 2466,
            "holdout_planned": 128,
            "holdout_completed": 128,
            "excluded_pathological": 1,
            "missing": 0,
            "corrupt": 0,
            "unexpected": 0,
        }
        if summary != expected:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
