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
    CROSS_LAYER_JACOBIAN_AB_IDENTITY_PATHS,
    CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
    EMBEDDING_EDGE_AB_IDENTITY_PATHS,
    SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_IDENTITY_PATHS,
    SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
    SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
    STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
    STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_IDENTITY_PATHS,
    TOLERANCE_GROUPS,
    NumericTolerance,
    compare_execution_artifacts,
)

SELECTED_EMBED_WIDTH_ONE_BF16_TOLERANCES = {
    "target": NumericTolerance(absolute=0.0, relative=0.0),
    "node": NumericTolerance(absolute=0.125, relative=1e-2),
    "edge": NumericTolerance(absolute=5e-4, relative=1e-2),
    "candidate_profile": NumericTolerance(absolute=0.125, relative=1e-2),
}


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
        "--embedding-edge-ab",
        action="store_true",
        help=(
            "Use the fail-closed embedding-edge materialization A/B profile: "
            "canonical scalar reference and vectorized candidate, same GPU "
            "model, exact topology, zero numerical tolerances, and only the "
            "exact materialization strategy field may differ."
        ),
    )
    parser.add_argument(
        "--cross-layer-jacobian-ab",
        action="store_true",
        help=(
            "Use the fail-closed cross-layer Jacobian A/B profile: canonical "
            "full-model reference and cached-range candidate, same GPU model, "
            "exact topology, zero numerical tolerances, and only the exact "
            "Jacobian execution strategy field may differ."
        ),
    )
    parser.add_argument(
        "--selected-attribution-neuron-lane-chunk-ab",
        action="store_true",
        help=(
            "Use the fail-closed ordinary selected-attribution neuron-lane "
            "chunk A/B profile: explicit None control and width-one candidate, "
            "same GPU model, exact topology and runtime width receipts, zero "
            "numerical tolerances, and only the exact chunk-width field may differ."
        ),
    )
    parser.add_argument(
        "--stop-gradient-selected-attribution-forward-ab",
        action="store_true",
        help=(
            "Use the fail-closed stop-gradient selected-attribution forward A/B "
            "profile: explicit full-model reference and prefix-stop candidate, "
            "same GPU model, exact topology and ordered execution/materialization "
            "receipts, zero numerical tolerances, and only the exact forward "
            "execution strategy field may differ."
        ),
    )
    parser.add_argument(
        "--selected-neuron-contribution-target-lane-chunk-ab",
        action="store_true",
        help=(
            "Use the fail-closed ordinary selected-neuron contribution target-lane "
            "chunk A/B profile: explicit None control and width-one candidate, "
            "same GPU model, exact topology and per-layer projected-VJP receipts, "
            "zero numerical tolerances, and only the exact chunk-width field may "
            "differ."
        ),
    )
    parser.add_argument(
        "--selected-embed-contribution-target-lane-full-width-ab",
        action="store_true",
        help=(
            "Use the fail-closed ordinary embedding contribution full-width "
            "adapter profile: explicit None control and width-five candidate, "
            "same GPU model, exact topology and projected-VJP receipts, and "
            "zero numerical tolerances."
        ),
    )
    parser.add_argument(
        "--selected-embed-contribution-target-lane-width-one-bf16-ab",
        action="store_true",
        help=(
            "Use the declared BF16 ordinary embedding contribution profile: "
            "width-five reference and width-one candidate, same GPU model, "
            "exact topology and ordered receipt presence, with fixed "
            "dtype-scale numerical tolerances. Preserve a separate zero-"
            "tolerance diagnostic report before applying this gate."
        ),
    )
    parser.add_argument(
        "--stop-gradient-embed-contribution-target-lane-full-width-ab",
        action="store_true",
        help=(
            "Use the fail-closed stop-gradient embedding contribution full-width "
            "adapter profile: explicit None control and width-five candidate, "
            "same GPU model, exact topology and projected-VJP receipts, and "
            "zero numerical tolerances."
        ),
    )
    parser.add_argument(
        "--stop-gradient-embed-contribution-target-lane-width-one-exact-ab",
        action="store_true",
        help=(
            "Use the fail-closed stop-gradient embedding contribution width-one "
            "profile: explicit width-five reference and width-one candidate, "
            "same GPU model, exact topology and projected-VJP receipts, and "
            "zero numerical tolerances."
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
    """Resolve generic or fail-closed execution qualification arguments."""

    supplied_tolerances = {
        group: (getattr(args, f"{group}_atol"), getattr(args, f"{group}_rtol"))
        for group in TOLERANCE_GROUPS
    }
    selected_profiles = [
        option
        for enabled, option in (
            (args.cuda_allocator_ab, "--cuda-allocator-ab"),
            (args.embedding_edge_ab, "--embedding-edge-ab"),
            (args.cross_layer_jacobian_ab, "--cross-layer-jacobian-ab"),
            (
                args.selected_attribution_neuron_lane_chunk_ab,
                "--selected-attribution-neuron-lane-chunk-ab",
            ),
            (
                args.stop_gradient_selected_attribution_forward_ab,
                "--stop-gradient-selected-attribution-forward-ab",
            ),
            (
                args.selected_neuron_contribution_target_lane_chunk_ab,
                "--selected-neuron-contribution-target-lane-chunk-ab",
            ),
            (
                args.selected_embed_contribution_target_lane_full_width_ab,
                "--selected-embed-contribution-target-lane-full-width-ab",
            ),
            (
                args.selected_embed_contribution_target_lane_width_one_bf16_ab,
                "--selected-embed-contribution-target-lane-width-one-bf16-ab",
            ),
            (
                args.stop_gradient_embed_contribution_target_lane_full_width_ab,
                "--stop-gradient-embed-contribution-target-lane-full-width-ab",
            ),
            (
                args.stop_gradient_embed_contribution_target_lane_width_one_exact_ab,
                "--stop-gradient-embed-contribution-target-lane-width-one-exact-ab",
            ),
        )
        if enabled
    ]
    if len(selected_profiles) > 1:
        raise ValueError(f"{', '.join(selected_profiles)} are mutually exclusive")
    strict_profile = selected_profiles[0] if selected_profiles else None
    if strict_profile is not None:
        if args.allow_identity_difference:
            raise ValueError(
                f"{strict_profile} cannot be combined with --allow-identity-difference"
            )
        if any(
            absolute is not None or relative is not None
            for absolute, relative in supplied_tolerances.values()
        ):
            raise ValueError(
                f"{strict_profile} fixes every numerical tolerance at zero "
                "or its declared BF16 value"
            )
        if args.cuda_allocator_ab:
            identity_paths = CUDA_ALLOCATOR_AB_IDENTITY_PATHS
        elif args.embedding_edge_ab:
            identity_paths = EMBEDDING_EDGE_AB_IDENTITY_PATHS
        elif args.cross_layer_jacobian_ab:
            identity_paths = CROSS_LAYER_JACOBIAN_AB_IDENTITY_PATHS
        elif args.selected_neuron_contribution_target_lane_chunk_ab:
            identity_paths = (
                SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
            )
        elif args.selected_attribution_neuron_lane_chunk_ab:
            identity_paths = SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_IDENTITY_PATHS
        elif args.stop_gradient_selected_attribution_forward_ab:
            identity_paths = (
                STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_IDENTITY_PATHS
            )
        elif (
            args.selected_embed_contribution_target_lane_full_width_ab
            or args.selected_embed_contribution_target_lane_width_one_bf16_ab
        ):
            identity_paths = (
                SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
            )
        else:
            identity_paths = (
                STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
            )
        selected_embed_profile = None
        if args.selected_embed_contribution_target_lane_full_width_ab:
            selected_embed_profile = "full_width_exact_v1"
        elif args.selected_embed_contribution_target_lane_width_one_bf16_ab:
            selected_embed_profile = "width_one_bf16_v1"
        if args.stop_gradient_embed_contribution_target_lane_full_width_ab:
            stop_gradient_embed_profile = "full_width_exact_v1"
        elif args.stop_gradient_embed_contribution_target_lane_width_one_exact_ab:
            stop_gradient_embed_profile = "width_one_exact_v1"
        else:
            stop_gradient_embed_profile = None
        fixed_tolerances = (
            SELECTED_EMBED_WIDTH_ONE_BF16_TOLERANCES
            if selected_embed_profile == "width_one_bf16_v1"
            else {
                group: NumericTolerance(absolute=0.0, relative=0.0)
                for group in TOLERANCE_GROUPS
            }
        )
        return {
            "allowed_identity_difference_paths": identity_paths,
            "tolerances": fixed_tolerances,
            "require_same_gpu_model": True,
            "require_same_gpu_family": True,
            "require_exact_node_topology": True,
            "require_exact_edge_topology": True,
            "require_canonical_cuda_allocator_ab": args.cuda_allocator_ab,
            "require_canonical_embedding_edge_ab": args.embedding_edge_ab,
            "require_canonical_cross_layer_jacobian_ab": (args.cross_layer_jacobian_ab),
            "require_canonical_selected_attribution_neuron_lane_chunk_ab": (
                args.selected_attribution_neuron_lane_chunk_ab
            ),
            "require_canonical_stop_gradient_selected_attribution_forward_ab": (
                args.stop_gradient_selected_attribution_forward_ab
            ),
            "require_canonical_selected_neuron_contribution_target_lane_chunk_ab": (
                args.selected_neuron_contribution_target_lane_chunk_ab
            ),
            "selected_embed_contribution_target_lane_chunk_ab_profile": (
                selected_embed_profile
            ),
            "stop_gradient_embed_contribution_target_lane_chunk_ab_profile": (
                stop_gradient_embed_profile
            ),
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
        "require_canonical_embedding_edge_ab": False,
        "require_canonical_cross_layer_jacobian_ab": False,
        "require_canonical_selected_attribution_neuron_lane_chunk_ab": False,
        "require_canonical_stop_gradient_selected_attribution_forward_ab": False,
        "require_canonical_selected_neuron_contribution_target_lane_chunk_ab": False,
        "selected_embed_contribution_target_lane_chunk_ab_profile": None,
        "stop_gradient_embed_contribution_target_lane_chunk_ab_profile": None,
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
