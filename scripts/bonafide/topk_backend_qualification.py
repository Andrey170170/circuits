"""CLI for qualifying numerical drift between top-k execution strategies."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuits.tracing.backend_qualification import (
    CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
    TOLERANCE_GROUPS,
    NumericTolerance,
    compare_execution_artifacts,
)


def save_qualification_report(path: Path, report: Mapping[str, Any]) -> Path:
    """Publish a JSON qualification report atomically without overwriting evidence."""

    if path.exists():
        raise FileExistsError(f"qualification report already exists: {path}")
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


def _tolerance_arguments(
    values: Mapping[str, tuple[float | None, float | None]],
) -> dict[str, NumericTolerance]:
    tolerances: dict[str, NumericTolerance] = {}
    for group, (absolute, relative) in values.items():
        if absolute is None and relative is None:
            continue
        tolerances[group] = NumericTolerance(
            absolute=0.0 if absolute is None else absolute,
            relative=0.0 if relative is None else relative,
        )
    return tolerances


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two compact top-k traces from different execution strategies. "
            "No numerical threshold is assumed: provide tolerances to create "
            "numerical pass/fail gates."
        )
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cuda-allocator-ab",
        action="store_true",
        help=(
            "Use the fail-closed allocator A/B profile: same GPU model, exact "
            "topology, zero numerical tolerances, and only the exact allocator "
            "policy and receipt fields may differ."
        ),
    )
    parser.add_argument(
        "--allow-identity-difference",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Explicit artifact_identity path allowed to differ. Suffix .* allows "
            "a subtree. Only named execution strategies, the CUDA allocator "
            "policy, code_revision, and runtime_environment paths are accepted. "
            "Repeat as needed."
        ),
    )
    parser.add_argument("--require-same-gpu-family", action="store_true")
    parser.add_argument("--require-same-gpu-model", action="store_true")
    parser.add_argument("--require-exact-node-topology", action="store_true")
    parser.add_argument("--require-exact-edge-topology", action="store_true")
    parser.add_argument(
        "--require-exact-topology",
        action="store_true",
        help="Require both node and edge topology to match exactly.",
    )
    for group in TOLERANCE_GROUPS:
        option = group.replace("_", "-")
        parser.add_argument(f"--{option}-atol", type=float)
        parser.add_argument(f"--{option}-rtol", type=float)
    return parser


def comparison_options(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve generic or fail-closed allocator qualification arguments."""

    supplied_tolerances = {
        group: (getattr(args, f"{group}_atol"), getattr(args, f"{group}_rtol"))
        for group in TOLERANCE_GROUPS
    }
    if args.cuda_allocator_ab:
        if args.allow_identity_difference:
            raise ValueError(
                "--cuda-allocator-ab cannot be combined with "
                "--allow-identity-difference"
            )
        if any(
            absolute is not None or relative is not None
            for absolute, relative in supplied_tolerances.values()
        ):
            raise ValueError(
                "--cuda-allocator-ab fixes every numerical tolerance at zero"
            )
        return {
            "allowed_identity_difference_paths": CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
            "tolerances": {
                group: NumericTolerance(absolute=0.0, relative=0.0)
                for group in TOLERANCE_GROUPS
            },
            "require_same_gpu_model": True,
            "require_same_gpu_family": True,
            "require_exact_node_topology": True,
            "require_exact_edge_topology": True,
            "require_canonical_cuda_allocator_ab": True,
        }

    return {
        "allowed_identity_difference_paths": args.allow_identity_difference,
        "tolerances": _tolerance_arguments(supplied_tolerances),
        "require_same_gpu_model": args.require_same_gpu_model,
        "require_same_gpu_family": args.require_same_gpu_family,
        "require_exact_node_topology": (
            args.require_exact_topology or args.require_exact_node_topology
        ),
        "require_exact_edge_topology": (
            args.require_exact_topology or args.require_exact_edge_topology
        ),
        "require_canonical_cuda_allocator_ab": False,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        options = comparison_options(args)
    except ValueError as error:
        parser.error(str(error))
    report = compare_execution_artifacts(
        args.reference,
        args.candidate,
        **options,
    )
    save_qualification_report(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if not report["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
