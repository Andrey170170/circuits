"""Compare saved observed-token k=1 traces with frozen width-one artifacts."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuits.tracing.artifact import (
    load_compact_trace,
    load_topk_compact_trace,
)
from circuits.tracing.parity import compare_observed_token_k1

PARITY_PLAN_SCHEMA = "bonafide-topk-k1-parity-plan/v1"
PARITY_REPORT_SCHEMA = "bonafide-topk-k1-parity-report/v1"


def validate_parity_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PARITY_PLAN_SCHEMA:
        raise ValueError(f"unsupported parity plan: {plan.get('schema_version')!r}")
    pairs = plan.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("parity plan requires non-empty pairs")
    seen: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("parity pair must be an object")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise ValueError(f"invalid or duplicate parity pair_id: {pair_id!r}")
        seen.add(pair_id)
        for field in ("width1_artifact_path", "topk_artifact_path"):
            value = pair.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"parity pair {field} must be a non-empty string")


def run_parity_plan(
    plan: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Load, canonicalize, and compare every frozen parity pair."""

    validate_parity_plan(plan)
    results: list[dict[str, Any]] = []
    for pair in plan["pairs"]:
        width1_path = Path(os.path.expandvars(pair["width1_artifact_path"]))
        topk_path = Path(os.path.expandvars(pair["topk_artifact_path"]))
        width1 = load_compact_trace(width1_path)
        topk = load_topk_compact_trace(topk_path)
        source_id = width1.manifest.get("artifact_id")
        paired_source_id = topk.manifest.get("source_width1_artifact_id")
        if (
            not isinstance(source_id, str)
            or not source_id
            or paired_source_id != source_id
        ):
            raise ValueError(
                f"parity pair {pair['pair_id']} source artifact identity mismatch"
            )
        report = compare_observed_token_k1(
            width1.circuit_data,
            topk.topk_trace,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        results.append(
            {
                "pair_id": pair["pair_id"],
                "passed": report.passed,
                "mismatches": list(report.mismatches),
                "node_count": report.node_count,
                "edge_count": report.edge_count,
                "source_width1_artifact_id": source_id,
                "topk_artifact_id": topk.manifest.get("artifact_id"),
                "width1_payload_sha256": width1.manifest["data_sha256"],
                "topk_payload_sha256": topk.manifest["data_sha256"],
                "locations": {
                    "width1_artifact_path": str(width1_path),
                    "topk_artifact_path": str(topk_path),
                },
            }
        )
    return {
        "schema_version": PARITY_REPORT_SCHEMA,
        "all_passed": all(result["passed"] for result in results),
        "pair_count": len(results),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "results": results,
    }


def save_parity_report(path: Path, report: Mapping[str, Any]) -> Path:
    """Create a report atomically without overwriting prior evidence."""

    if path.exists():
        raise FileExistsError(f"parity report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    with args.plan.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    report = run_parity_plan(
        plan,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    save_parity_report(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
