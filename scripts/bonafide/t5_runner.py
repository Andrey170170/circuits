"""Run the strict upstream-style top-five Step-0 tracing smoke.

This is a deliberately narrow facade over ``topk_runner``.  It prevents the
new process-witness lane from accidentally using the separate CU5 candidate-
union semantics while retaining the mature artifact, resume, and resource
gates of the existing runner.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.bonafide.runner import load_json
from scripts.bonafide.topk_manifest import (
    HISTORICAL_THINKING_SERIALIZATION_MODE,
    STEP0_T5_SMOKE_PHASE,
    validate_topk_manifest,
)
from scripts.bonafide.topk_runner import run_topk_wave


def validate_step0_t5_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject every manifest except the frozen historical strict-T5 smoke."""

    validate_topk_manifest(manifest)
    if manifest.get("phase") != STEP0_T5_SMOKE_PHASE:
        raise ValueError(
            f"T5 runner requires phase {STEP0_T5_SMOKE_PHASE!r}; got "
            f"{manifest.get('phase')!r}"
        )
    trace_family = manifest["trace_family"]
    expected = {
        "candidate_policy_id": "model_top5",
        "candidate_count": 5,
        "joint_objective_id": "raw_logit_sum",
    }
    for field, value in expected.items():
        if trace_family.get(field) != value:
            raise ValueError(f"T5 runner requires trace_family.{field}={value!r}")
    mode = manifest["teacher_forcing_contract"]["serialization_mode"]
    if mode != HISTORICAL_THINKING_SERIALIZATION_MODE:
        raise ValueError(
            "T5 runner requires historical thinking continuation serialization"
        )


def run_step0_t5_wave(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    wave_id: str,
    artifact_root: Path,
    summary_jsonl: Path,
    only_artifact_id: str | None = None,
    dry_run: bool = False,
    **runner_kwargs: Any,
) -> list[dict[str, Any]]:
    """Execute one Step-0 smoke wave under its frozen serialization contract."""

    validate_step0_t5_manifest(manifest)
    return run_topk_wave(
        config=config,
        manifest=manifest,
        wave_id=wave_id,
        artifact_root=artifact_root,
        summary_jsonl=summary_jsonl,
        only_artifact_id=only_artifact_id,
        dry_run=dry_run,
        teacher_forced_serialization_mode=manifest["teacher_forcing_contract"][
            "serialization_mode"
        ],
        **runner_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--summary-jsonl", type=Path, required=True)
    parser.add_argument("--only-artifact-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-records", action="store_true")
    args = parser.parse_args()

    records = run_step0_t5_wave(
        config=load_json(args.config),
        manifest=load_json(args.manifest),
        wave_id=args.wave,
        artifact_root=args.artifact_root,
        summary_jsonl=args.summary_jsonl,
        only_artifact_id=args.only_artifact_id,
        dry_run=args.dry_run,
    )
    if args.print_records:
        for record in records:
            print(json.dumps(record, sort_keys=True, allow_nan=False))
        return
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    print(
        json.dumps(
            {
                "wave_id": args.wave,
                "record_count": len(records),
                "status_counts": counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
