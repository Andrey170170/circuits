#!/usr/bin/env python3
"""Prepare the frozen W64 width-only versus candidate-evidence comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from circuits.analysis.bonafide.candidate_labeling_comparison import (
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    run_candidate_labeling_preparation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--evaluation-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
        local_files_only=True,
    )
    manifest = run_candidate_labeling_preparation(
        input_root=args.input_root,
        baseline_root=args.baseline_root,
        evaluation_path=args.evaluation_path,
        output_root=args.output_root,
        repo_root=repo_root,
        tokenizer=tokenizer,
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output_root": str(args.output_root.resolve()),
                "anchor_cluster_ids": manifest[
                    "anchor_cluster_ids_in_target_point_order"
                ],
                "eligible_arm_ids": [
                    arm["arm_id"] for arm in manifest["eligible_arms"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
