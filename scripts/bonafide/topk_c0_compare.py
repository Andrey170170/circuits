"""Build C0 joint-versus-independent candidate topology reports."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuits.tracing.artifact import load_topk_compact_trace
from circuits.tracing.topology_comparison import (
    compare_joint_to_independent_candidates,
)

C0_PLAN_SCHEMA = "bonafide-topk-c0-comparison-plan/v1"
C0_REPORT_SCHEMA = "bonafide-topk-c0-comparison-report/v1"


def validate_c0_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != C0_PLAN_SCHEMA:
        raise ValueError(f"unsupported C0 plan: {plan.get('schema_version')!r}")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("C0 comparison plan requires non-empty cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("C0 comparison case must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"invalid or duplicate C0 case_id: {case_id!r}")
        seen.add(case_id)
        if not isinstance(case.get("joint_artifact_path"), str):
            raise ValueError("C0 joint_artifact_path must be a string")
        independent_paths = case.get("independent_artifact_paths")
        if not isinstance(independent_paths, list) or not independent_paths:
            raise ValueError("C0 case requires independent_artifact_paths")
        if any(not isinstance(path, str) or not path for path in independent_paths):
            raise ValueError("C0 independent artifact paths must be non-empty strings")


def run_c0_plan(
    plan: Mapping[str, Any],
    *,
    max_omitted_path_witnesses_per_candidate: int = 100,
) -> dict[str, Any]:
    validate_c0_plan(plan)
    results: list[dict[str, Any]] = []
    for case in plan["cases"]:
        joint = load_topk_compact_trace(
            Path(os.path.expandvars(case["joint_artifact_path"]))
        )
        independent = [
            load_topk_compact_trace(Path(os.path.expandvars(path)))
            for path in case["independent_artifact_paths"]
        ]
        comparison = compare_joint_to_independent_candidates(
            joint.topk_trace,
            [artifact.topk_trace for artifact in independent],
            max_omitted_path_witnesses_per_candidate=(
                max_omitted_path_witnesses_per_candidate
            ),
        )
        results.append(
            {
                "case_id": case["case_id"],
                "comparison": comparison,
                "joint_artifact": {
                    "artifact_id": joint.manifest.get("artifact_id"),
                    "payload_sha256": joint.manifest["data_sha256"],
                    "location": str(joint.path),
                },
                "independent_artifacts": [
                    {
                        "artifact_id": artifact.manifest.get("artifact_id"),
                        "payload_sha256": artifact.manifest["data_sha256"],
                        "location": str(artifact.path),
                    }
                    for artifact in independent
                ],
            }
        )
    return {
        "schema_version": C0_REPORT_SCHEMA,
        "case_count": len(results),
        "max_omitted_path_witnesses_per_candidate": (
            max_omitted_path_witnesses_per_candidate
        ),
        "results": results,
    }


def save_c0_report(path: Path, report: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"C0 comparison report already exists: {path}")
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
    parser.add_argument(
        "--max-omitted-path-witnesses-per-candidate", type=int, default=100
    )
    args = parser.parse_args()
    with args.plan.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    report = run_c0_plan(
        plan,
        max_omitted_path_witnesses_per_candidate=(
            args.max_omitted_path_witnesses_per_candidate
        ),
    )
    save_c0_report(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
