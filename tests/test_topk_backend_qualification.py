from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pandas as pd
import pytest
from circuits.tracing.artifact import save_topk_compact_trace
from circuits.tracing.backend_qualification import (
    CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
    CROSS_LAYER_JACOBIAN_AB_IDENTITY_PATHS,
    CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
    EMBEDDING_EDGE_AB_IDENTITY_PATHS,
    SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
    SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
    TOLERANCE_GROUPS,
    NumericTolerance,
    compare_attention_backend_artifacts,
    compare_execution_artifacts,
)
from scripts.bonafide.topk_backend_qualification import (
    build_parser,
    comparison_options,
    save_qualification_report,
)
from tests.test_teacher_forced_trace import _topk_trace

_UNSET = object()


def _manifest(
    backend: str,
    *,
    dtype: str = "bfloat16",
    source_id: str = "source-width1",
    contribution_execution: str | None = "full_graph_v1",
    allocator_policy: str | None = None,
    embedding_edge_materialization: str | None = None,
    cross_layer_jacobian_execution: str | None = None,
    contribution_target_lane_chunk_size: object = _UNSET,
    selected_neuron_contribution_target_lane_chunk_size: object = _UNSET,
    selected_embed_contribution_target_lane_chunk_size: object = _UNSET,
    code_revision: str | None = None,
) -> dict:
    manifest = {
        "source_width1_artifact_id": source_id,
        "source_width1_manifest_sha256": "a" * 64,
        "source_target_selection": {"response_token_positions": [0]},
        "bonafide_example": {
            "example_id": "row-1",
            "prompt": "prompt",
            "response": "response",
        },
        "model_revision": "revision-1",
        "gpu": {
            "name": "NVIDIA A100 80GB PCIe",
            "total_memory_bytes": 80_000_000_000,
            "compute_capability": [8, 0],
        },
        "runtime_environment": {
            "python": "3.12.12",
            "gpu_runtime": {"devices": [{"name": "NVIDIA A100 80GB PCIe"}]},
        },
        "artifact_identity": {
            "source_width1_artifact_id": source_id,
            "source_width1_manifest_sha256": "a" * 64,
            "source_target_selection": {"response_token_positions": [0]},
            "trace_family": {"trace_family_id": "bonafide.topk-position.v1"},
            "model": {
                "model_id": "fake/model",
                "revision": "revision-1",
                "device": "cuda:0",
                "dtype": dtype,
            },
            "adag_config": {
                "percentage_threshold": 0.05,
                "stop_gradient_attention_backend": backend,
                "disable_stop_grad": False,
                "use_stop_grad_on_mlps": True,
            },
            "trace_warmup": {"enabled": False},
            "batch_size": 1,
            "wave_id": "wave-1",
            "code_revision": {
                "git_commit": code_revision
                or ("commit-eager" if backend == "eager" else "commit-sdpa"),
                "source_tree_sha256": "b" * 64 if backend == "eager" else "c" * 64,
            },
            "runtime_environment": {
                "python": "3.12.12",
                "packages": {"torch": "2.5.1"},
            },
        },
    }
    if contribution_execution is not None:
        manifest["artifact_identity"]["adag_config"][
            "stop_gradient_contribution_execution"
        ] = contribution_execution
    if contribution_target_lane_chunk_size is not _UNSET:
        manifest["artifact_identity"]["adag_config"][
            "stop_gradient_contribution_target_lane_chunk_size"
        ] = contribution_target_lane_chunk_size
    if selected_neuron_contribution_target_lane_chunk_size is not _UNSET:
        manifest["artifact_identity"]["adag_config"][
            "selected_neuron_contribution_target_lane_chunk_size"
        ] = selected_neuron_contribution_target_lane_chunk_size
    if selected_embed_contribution_target_lane_chunk_size is not _UNSET:
        manifest["artifact_identity"]["adag_config"][
            "selected_embed_contribution_target_lane_chunk_size"
        ] = selected_embed_contribution_target_lane_chunk_size
    if embedding_edge_materialization is not None:
        manifest["artifact_identity"]["adag_config"][
            "embedding_edge_materialization"
        ] = embedding_edge_materialization
    if cross_layer_jacobian_execution is not None:
        manifest["artifact_identity"]["adag_config"][
            "cross_layer_jacobian_execution"
        ] = cross_layer_jacobian_execution
    if allocator_policy is not None:
        allocator_value = (
            None if allocator_policy == "default_v1" else "expandable_segments:True"
        )
        allocator_receipt = {
            "intended_policy_id": allocator_policy,
            "observed_environment": {
                "name": "PYTORCH_CUDA_ALLOC_CONF",
                "value": allocator_value,
                "is_set": allocator_value is not None,
            },
            "observed_allocator_backend": "native",
        }
        manifest["artifact_identity"]["cuda_allocator_policy"] = allocator_policy
        manifest["artifact_identity"]["runtime_environment"][
            "cuda_allocator_policy"
        ] = allocator_receipt
        manifest["runtime_environment"]["cuda_allocator_policy"] = allocator_receipt
    identity = manifest["artifact_identity"]
    identity_sha256 = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    identity["sha256"] = identity_sha256
    manifest["artifact_id"] = f"topk-trace-{identity_sha256[:24]}"
    return manifest


def _complete_trace_metadata(trace) -> None:
    trace.circuit_data.trace_metadata.update(
        {
            "prompt": "prompt",
            "prompt_sha256": "1" * 64,
            "response": "response",
            "response_sha256": "2" * 64,
            "system_prompt": None,
            "system_prompt_sha256": None,
            "teacher_forced_serialization_mode": "assistant_turn",
            "teacher_forced_token_identity": {
                "assistant_prefix_ids_sha256": "3" * 64,
                "response_ids_sha256": "4" * 64,
            },
            "assistant_prefix_token_count": 3,
            "response_token_count": 1,
            "included_response_token_count": 1,
            "input_token_count": 3,
            "chat_template_sha256": "5" * 64,
        }
    )


def _selected_neuron_receipt_instrumentation(layers: list[dict]) -> dict:
    return {
        "layers": layers,
        "early_predictors": {
            "selected_neuron_counts_by_layer": [
                {
                    "layer": layer["layer"],
                    "count": layer.get("selected_neuron_count", 0),
                }
                for layer in layers
            ]
        },
    }


def _rehash_manifest(manifest: dict) -> dict:
    identity = manifest["artifact_identity"]
    identity_without_hash = {
        key: value for key, value in identity.items() if key != "sha256"
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(
            identity_without_hash,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    identity["sha256"] = identity_sha256
    manifest["artifact_id"] = f"topk-trace-{identity_sha256[:24]}"
    return manifest


def _save_pair(
    tmp_path,
    *,
    reference_backend: str = "eager",
    candidate_backend: str = "sdpa_ov_only",
    reference_execution: str | None = "full_graph_v1",
    candidate_execution: str | None = "full_graph_v1",
    reference_chunk_size: object = _UNSET,
    candidate_chunk_size: object = _UNSET,
    reference_selected_neuron_chunk_size: object = _UNSET,
    candidate_selected_neuron_chunk_size: object = _UNSET,
    reference_selected_neuron_receipts: list[dict] | None = None,
    candidate_selected_neuron_receipts: list[dict] | None = None,
    reference_selected_embed_chunk_size: object = _UNSET,
    candidate_selected_embed_chunk_size: object = _UNSET,
    reference_selected_embed_receipts: list[dict] | None = None,
    candidate_selected_embed_receipts: list[dict] | None = None,
    reference_dtype: str = "bfloat16",
    candidate_dtype: str = "bfloat16",
    candidate_mlp_profile_delta: float = 0.0,
    candidate_unaffected_edge_delta: float = 0.0,
):
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    if (
        reference_selected_embed_receipts is not None
        or candidate_selected_embed_receipts is not None
    ):
        for trace in (reference, candidate):
            embedding_row = trace.circuit_data.df_node.iloc[[0]].copy()
            embedding_row.loc[:, "layer"] = -1
            embedding_row.loc[:, "token"] = 2
            embedding_row.loc[:, "neuron"] = 42
            trace.circuit_data.df_node = pd.concat(
                [trace.circuit_data.df_node, embedding_row], ignore_index=True
            )
        if candidate_mlp_profile_delta:
            profile = list(candidate.circuit_data.df_node.iloc[0].contrib_map)
            profile[0] += candidate_mlp_profile_delta
            candidate.circuit_data.df_node.at[0, "contrib_map"] = profile
        if candidate_unaffected_edge_delta:
            candidate.circuit_data.df_edge.loc[0, "attribution"] += (
                candidate_unaffected_edge_delta
            )
    if reference_selected_neuron_receipts is not None:
        reference.circuit_data.trace_metadata["instrumentation"] = (
            _selected_neuron_receipt_instrumentation(reference_selected_neuron_receipts)
        )
    if candidate_selected_neuron_receipts is not None:
        candidate.circuit_data.trace_metadata["instrumentation"] = (
            _selected_neuron_receipt_instrumentation(candidate_selected_neuron_receipts)
        )
    if reference_selected_embed_receipts is not None:
        instrumentation = reference.circuit_data.trace_metadata.setdefault(
            "instrumentation", {}
        )
        instrumentation.setdefault("execution_records", {})[
            "selected_embed_contribution_vjp"
        ] = reference_selected_embed_receipts
    if candidate_selected_embed_receipts is not None:
        instrumentation = candidate.circuit_data.trace_metadata.setdefault(
            "instrumentation", {}
        )
        instrumentation.setdefault("execution_records", {})[
            "selected_embed_contribution_vjp"
        ] = candidate_selected_embed_receipts
    reference_path = tmp_path / "reference"
    candidate_path = tmp_path / "candidate"
    save_topk_compact_trace(
        reference_path,
        reference,
        metrics={
            "trace_wall_seconds": 10.0,
            "cuda_peak_allocated_bytes": 100,
            "cuda_peak_reserved_bytes": 120,
            "rss_peak_after_bytes": 200,
        },
        manifest=_manifest(
            reference_backend,
            dtype=reference_dtype,
            contribution_execution=reference_execution,
            contribution_target_lane_chunk_size=reference_chunk_size,
            selected_neuron_contribution_target_lane_chunk_size=(
                reference_selected_neuron_chunk_size
            ),
            selected_embed_contribution_target_lane_chunk_size=(
                reference_selected_embed_chunk_size
            ),
        ),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        metrics={
            "trace_wall_seconds": 6.0,
            "cuda_peak_allocated_bytes": 70,
            "cuda_peak_reserved_bytes": 80,
            "rss_peak_after_bytes": 190,
        },
        manifest=_manifest(
            candidate_backend,
            dtype=candidate_dtype,
            contribution_execution=candidate_execution,
            contribution_target_lane_chunk_size=candidate_chunk_size,
            selected_neuron_contribution_target_lane_chunk_size=(
                candidate_selected_neuron_chunk_size
            ),
            selected_embed_contribution_target_lane_chunk_size=(
                candidate_selected_embed_chunk_size
            ),
        ),
    )
    return reference_path, candidate_path


def _allowed_paths() -> list[str]:
    return [
        "artifact_identity.adag_config.stop_gradient_attention_backend",
        "artifact_identity.code_revision.*",
    ]


def test_backend_qualification_passes_explicit_gates_and_reports_resources(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(tmp_path)

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=_allowed_paths(),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in ("target", "node", "edge", "candidate_profile")
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
    )

    assert report["qualification_passed"] is True
    assert report["validation_passed"] is True
    assert report["diagnostic_only"] is False
    assert report["scientific_parity_claimed"] is False
    assert report["hardware"]["reference"]["family"] == "A100"
    assert report["topology"]["nodes"]["jaccard"] == 1.0
    assert report["target_values"]["candidate_logits"]["max_absolute_error"] == 0.0
    assert report["resources"]["trace_wall_seconds"]["candidate_over_reference"] == 0.6
    assert (
        report["resources"]["cuda_peak_reserved_bytes"]["candidate_minus_reference"]
        == -40
    )


def test_backend_qualification_can_allow_contribution_execution_difference(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_execution=None,
        candidate_execution="source_leaf_v1",
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=[
            *_allowed_paths(),
            "artifact_identity.adag_config.stop_gradient_contribution_execution",
        ],
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
    )

    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} >= {
        "artifact_identity.adag_config.stop_gradient_attention_backend",
        "artifact_identity.adag_config.stop_gradient_contribution_execution",
    }


def test_backend_qualification_can_allow_target_lane_chunk_size_difference(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_chunk_size=None,
        candidate_chunk_size=1,
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=[
            *_allowed_paths(),
            *CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
        ],
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
    )

    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} >= {
        "artifact_identity.adag_config.stop_gradient_attention_backend",
        "artifact_identity.adag_config."
        "stop_gradient_contribution_target_lane_chunk_size",
    }

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config."
                "stop_gradient_contribution_target_lane_chunk_size.*"
            ],
        )


def test_backend_qualification_rejects_wildcard_for_scalar_strategy(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(tmp_path)

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config.stop_gradient_contribution_execution.*"
            ],
        )


def test_backend_qualification_can_allow_selected_neuron_chunk_size_difference(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_selected_neuron_chunk_size=None,
        candidate_selected_neuron_chunk_size=1,
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=[
            *_allowed_paths(),
            *SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
        ],
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
    )

    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} >= {
        "artifact_identity.adag_config.stop_gradient_attention_backend",
        "artifact_identity.adag_config."
        "selected_neuron_contribution_target_lane_chunk_size",
    }

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config."
                "selected_neuron_contribution_target_lane_chunk_size.*"
            ],
        )


def _selected_neuron_receipts() -> list[dict]:
    selected = [
        {
            "layer": layer,
            "selected_neuron_count": layer + 1,
            "selected_neuron_contribution_projected_vjp_shape": [layer + 1, 1, 5],
            "selected_neuron_contribution_target_lane_count": 5,
            "selected_neuron_contribution_projected_vjp_sha256": character * 64,
        }
        for layer, character in ((0, "1"), (3, "a"))
    ]
    selected.insert(
        1,
        {
            "layer": 1,
            "selected_neuron_count": 0,
            "unrelated_layer_telemetry": True,
        },
    )
    return selected


def test_selected_neuron_chunk_ab_requires_exact_projected_receipts(tmp_path) -> None:
    receipts = _selected_neuron_receipts()
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_neuron_chunk_size=None,
        candidate_selected_neuron_chunk_size=1,
        reference_selected_neuron_receipts=deepcopy(receipts),
        candidate_selected_neuron_receipts=deepcopy(receipts),
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        require_canonical_selected_neuron_contribution_target_lane_chunk_ab=True,
    )

    assert report["qualification_passed"] is True
    contract = report["selected_neuron_contribution_target_lane_chunk_ab_contract"]
    assert contract["passed"] is True
    assert contract["exact_receipts"]["checks"] == {
        "reference_presence_and_order": True,
        "candidate_presence_and_order": True,
        "layer_order_equal": True,
        "receipt_hashes_exact": True,
    }


def test_selected_neuron_chunk_ab_cli_selects_fail_closed_profile(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--reference",
            str(tmp_path / "reference"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "report.json"),
            "--selected-neuron-contribution-target-lane-chunk-ab",
        ]
    )

    options = comparison_options(args)

    assert options["allowed_identity_difference_paths"] == (
        SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert (
        options["require_canonical_selected_neuron_contribution_target_lane_chunk_ab"]
        is True
    )
    assert all(
        tolerance == NumericTolerance(absolute=0.0, relative=0.0)
        for tolerance in options["tolerances"].values()
    )


def _selected_embed_receipt(character: str, width: int | None) -> list[dict]:
    target_count = 5
    resolved_width = min(width or target_count, target_count)
    chunk_widths = [
        min(resolved_width, target_count - start)
        for start in range(0, target_count, resolved_width)
    ]
    raw_shapes = [[chunk_width, 1, 7, 11] for chunk_width in chunk_widths]
    grad_shapes = [[chunk_width, chunk_width] for chunk_width in chunk_widths]
    return [
        {
            "execution_index": None,
            "receipt_mode": "singular",
            "return_gradient_only": False,
            "canonical_result_order": "source_batch_target",
            "source_tokens": [4, 1, 4],
            "raw_vjp_shape": raw_shapes[0] if len(raw_shapes) == 1 else None,
            "raw_vjp_chunk_shapes": raw_shapes,
            "grad_outputs_shape": grad_shapes[0] if len(grad_shapes) == 1 else None,
            "grad_outputs_chunk_shapes": grad_shapes,
            "max_grad_outputs_shape": [resolved_width, resolved_width],
            "projected_vjp_shape": [3, 1, 5],
            "target_lane_count": target_count,
            "projected_vjp_sha256": character * 64,
            "target_lane_chunk_size_requested": width,
            "target_lane_chunk_size_resolved": resolved_width,
            "target_lane_chunk_count": len(chunk_widths),
            "max_materialized_target_lanes": resolved_width,
            "max_materialized_autograd_lanes": resolved_width,
            "dense_vjp_result_materialized": True,
            "retain_graph": True,
        }
    ]


@pytest.mark.parametrize(
    ("profile", "reference_width", "candidate_width", "candidate_hash"),
    [
        ("full_width_exact_v1", None, 5, "1"),
        ("width_one_bf16_v1", 5, 1, "a"),
    ],
)
def test_selected_embed_chunk_profiles_validate_strategy_and_receipts(
    tmp_path,
    profile: str,
    reference_width: int | None,
    candidate_width: int,
    candidate_hash: str,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_embed_chunk_size=reference_width,
        candidate_selected_embed_chunk_size=candidate_width,
        reference_selected_embed_receipts=_selected_embed_receipt("1", reference_width),
        candidate_selected_embed_receipts=_selected_embed_receipt(
            candidate_hash, candidate_width
        ),
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        selected_embed_contribution_target_lane_chunk_ab_profile=profile,
    )

    assert report["qualification_passed"] is True
    contract = report["selected_embed_contribution_target_lane_chunk_ab_contract"]
    assert contract["profile"] == profile
    assert contract["passed"] is True
    assert contract["projected_receipts"]["checks"]["receipt_hashes_exact"] is (
        profile == "full_width_exact_v1"
    )
    if profile == "width_one_bf16_v1":
        assert contract["bf16_scope"]["passed"] is True


@pytest.mark.parametrize(
    "defect",
    ["claimed_width_one_unbounded", "missing_chunks", "reordered_sources", "bad_shape"],
)
def test_selected_embed_chunk_profile_rejects_unproven_runtime_width(
    tmp_path, defect: str
) -> None:
    reference_receipts = _selected_embed_receipt("1", 5)
    candidate_receipts = _selected_embed_receipt("a", 1)
    candidate_record = candidate_receipts[0]
    if defect == "claimed_width_one_unbounded":
        candidate_record.update(_selected_embed_receipt("a", 5)[0])
        candidate_record["target_lane_chunk_size_requested"] = 1
    elif defect == "missing_chunks":
        candidate_record.pop("raw_vjp_chunk_shapes")
    elif defect == "reordered_sources":
        candidate_record["source_tokens"] = [1, 4, 4]
    else:
        candidate_record["grad_outputs_chunk_shapes"][2] = [2, 2]
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_embed_chunk_size=5,
        candidate_selected_embed_chunk_size=1,
        reference_selected_embed_receipts=reference_receipts,
        candidate_selected_embed_receipts=candidate_receipts,
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        tolerances={
            "target": NumericTolerance(absolute=0.0, relative=0.0),
            "node": NumericTolerance(absolute=0.0, relative=0.0),
            "edge": NumericTolerance(absolute=5e-4, relative=1e-2),
            "candidate_profile": NumericTolerance(absolute=0.125, relative=1e-2),
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        selected_embed_contribution_target_lane_chunk_ab_profile="width_one_bf16_v1",
    )

    assert report["validation_passed"] is False
    contract = report["selected_embed_contribution_target_lane_chunk_ab_contract"]
    assert contract["projected_receipts"]["passed"] is False


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("candidate_mlp_profile_delta", 0.01),
        ("candidate_unaffected_edge_delta", 0.0001),
        ("candidate_dtype", "float32"),
    ],
)
def test_selected_embed_width_one_bf16_scope_rejects_unrelated_drift(
    tmp_path, mutation: str, value: float | str
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_embed_chunk_size=5,
        candidate_selected_embed_chunk_size=1,
        reference_selected_embed_receipts=_selected_embed_receipt("1", 5),
        candidate_selected_embed_receipts=_selected_embed_receipt("a", 1),
        **{mutation: value},
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        tolerances={
            "target": NumericTolerance(absolute=0.0, relative=0.0),
            "node": NumericTolerance(absolute=0.0, relative=0.0),
            "edge": NumericTolerance(absolute=5e-4, relative=1e-2),
            "candidate_profile": NumericTolerance(absolute=0.125, relative=1e-2),
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        selected_embed_contribution_target_lane_chunk_ab_profile="width_one_bf16_v1",
    )

    assert report["validation_passed"] is False
    scope = report["selected_embed_contribution_target_lane_chunk_ab_contract"][
        "bf16_scope"
    ]
    assert scope["passed"] is False
    if mutation == "candidate_mlp_profile_delta":
        assert scope["checks"]["non_embedding_profiles_exact"] is False
    elif mutation == "candidate_unaffected_edge_delta":
        assert scope["checks"]["unaffected_edges_exact"] is False
    else:
        assert scope["checks"]["exact_bf16_dtype_identity"] is False


@pytest.mark.parametrize(
    ("flag", "profile", "expected_tolerances"),
    [
        (
            "--selected-embed-contribution-target-lane-full-width-ab",
            "full_width_exact_v1",
            {
                group: NumericTolerance(absolute=0.0, relative=0.0)
                for group in TOLERANCE_GROUPS
            },
        ),
        (
            "--selected-embed-contribution-target-lane-width-one-bf16-ab",
            "width_one_bf16_v1",
            {
                "target": NumericTolerance(absolute=0.0, relative=0.0),
                "node": NumericTolerance(absolute=0.0, relative=0.0),
                "edge": NumericTolerance(absolute=5e-4, relative=1e-2),
                "candidate_profile": NumericTolerance(absolute=0.125, relative=1e-2),
            },
        ),
    ],
)
def test_selected_embed_chunk_cli_profiles_are_canonical(
    tmp_path,
    flag: str,
    profile: str,
    expected_tolerances: dict[str, NumericTolerance],
) -> None:
    args = build_parser().parse_args(
        [
            "--reference",
            str(tmp_path / "reference"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "report.json"),
            flag,
        ]
    )
    options = comparison_options(args)
    assert options["allowed_identity_difference_paths"] == (
        SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
    )
    assert (
        options["selected_embed_contribution_target_lane_chunk_ab_profile"] == profile
    )
    assert options["tolerances"] == expected_tolerances
    assert options["require_same_gpu_model"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True


@pytest.mark.parametrize(
    "defect", ["missing", "reordered", "mismatch", "both_complete_missing"]
)
def test_selected_neuron_chunk_ab_fails_closed_on_receipt_defects(
    tmp_path, defect: str
) -> None:
    reference_receipts = _selected_neuron_receipts()
    candidate_receipts = deepcopy(reference_receipts)
    receipt_fields = (
        "selected_neuron_contribution_projected_vjp_shape",
        "selected_neuron_contribution_target_lane_count",
        "selected_neuron_contribution_projected_vjp_sha256",
    )
    if defect == "missing":
        candidate_receipts[0].pop("selected_neuron_contribution_projected_vjp_sha256")
    elif defect == "reordered":
        candidate_receipts.reverse()
    elif defect == "mismatch":
        candidate_receipts[2]["selected_neuron_contribution_projected_vjp_sha256"] = (
            "b" * 64
        )
    else:
        for receipts in (reference_receipts, candidate_receipts):
            for field in receipt_fields:
                receipts[2].pop(field)
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_neuron_chunk_size=None,
        candidate_selected_neuron_chunk_size=1,
        reference_selected_neuron_receipts=reference_receipts,
        candidate_selected_neuron_receipts=candidate_receipts,
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        require_canonical_selected_neuron_contribution_target_lane_chunk_ab=True,
    )

    assert report["validation_passed"] is False
    assert (
        report["selected_neuron_contribution_target_lane_chunk_ab_contract"]["passed"]
        is False
    )


def test_execution_qualification_allows_exact_allocator_policy_only(
    tmp_path,
) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_path = tmp_path / "allocator-default"
    candidate_path = tmp_path / "allocator-expandable"
    save_topk_compact_trace(
        reference_path,
        reference,
        manifest=_manifest("eager", allocator_policy="default_v1"),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        manifest=_manifest("eager", allocator_policy="expandable_segments_v1"),
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        allowed_identity_difference_paths=CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        require_canonical_cuda_allocator_ab=True,
    )
    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    assert {gate["gate"] for gate in report["gates"]} >= {
        "same_gpu_model",
        "canonical_cuda_allocator_ab_pair",
        "exact_node_topology",
        "exact_edge_topology",
        *(f"{group}_numeric_tolerance" for group in TOLERANCE_GROUPS),
    }
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} == set(
        CUDA_ALLOCATOR_AB_IDENTITY_PATHS
    )

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference_path,
            candidate_path,
            allowed_identity_difference_paths=[
                "artifact_identity.cuda_allocator_policy.*"
            ],
        )


def test_cuda_allocator_ab_cli_resolves_strict_non_overridable_profile() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference",
            "reference",
            "--candidate",
            "candidate",
            "--output",
            "report.json",
            "--cuda-allocator-ab",
        ]
    )

    options = comparison_options(args)

    assert options["allowed_identity_difference_paths"] == (
        CUDA_ALLOCATOR_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_same_gpu_family"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert options["require_canonical_cuda_allocator_ab"] is True
    assert options["tolerances"] == {
        group: NumericTolerance(absolute=0.0, relative=0.0)
        for group in TOLERANCE_GROUPS
    }

    args.allow_identity_difference = ["artifact_identity.code_revision.*"]
    with pytest.raises(ValueError, match="cannot be combined"):
        comparison_options(args)


def test_cuda_allocator_ab_rejects_same_target_code_revision_drift(tmp_path) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_path = tmp_path / "allocator-default"
    candidate_path = tmp_path / "allocator-expandable-code-drift"
    save_topk_compact_trace(
        reference_path,
        reference,
        manifest=_manifest(
            "eager", allocator_policy="default_v1", code_revision="same-commit"
        ),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        manifest=_manifest(
            "eager",
            allocator_policy="expandable_segments_v1",
            code_revision="different-commit",
        ),
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        **comparison_options(
            build_parser().parse_args(
                [
                    "--reference",
                    str(reference_path),
                    "--candidate",
                    str(candidate_path),
                    "--output",
                    str(tmp_path / "unused.json"),
                    "--cuda-allocator-ab",
                ]
            )
        ),
    )

    assert report["validation_passed"] is False
    disallowed = report["identity"]["artifact_identity"]["unallowed_differences"]
    assert {difference["path"] for difference in disallowed} == {
        "artifact_identity.code_revision.git_commit"
    }


@pytest.mark.parametrize(
    ("reference_policy", "candidate_policy"),
    [
        ("default_v1", "default_v1"),
        ("expandable_segments_v1", "default_v1"),
    ],
)
def test_cuda_allocator_ab_rejects_equal_or_reversed_lanes(
    tmp_path, reference_policy, candidate_policy
) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_path = tmp_path / "allocator-reference"
    candidate_path = tmp_path / "allocator-candidate"
    save_topk_compact_trace(
        reference_path,
        reference,
        manifest=_manifest("eager", allocator_policy=reference_policy),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        manifest=_manifest("eager", allocator_policy=candidate_policy),
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        allowed_identity_difference_paths=CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
        require_canonical_cuda_allocator_ab=True,
    )

    assert report["validation_passed"] is False
    assert report["cuda_allocator_ab_contract"]["passed"] is False


def test_cuda_allocator_ab_rejects_equally_malformed_receipts(tmp_path) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_path = tmp_path / "allocator-default-malformed"
    candidate_path = tmp_path / "allocator-expandable-malformed"
    manifests = [
        _manifest("eager", allocator_policy="default_v1"),
        _manifest("eager", allocator_policy="expandable_segments_v1"),
    ]
    for manifest in manifests:
        for runtime in (
            manifest["runtime_environment"],
            manifest["artifact_identity"]["runtime_environment"],
        ):
            receipt = runtime["cuda_allocator_policy"]
            receipt["observed_environment"]["name"] = "WRONG_ALLOCATOR_VARIABLE"
            receipt["observed_allocator_backend"] = "wrong-backend"
        _rehash_manifest(manifest)
    save_topk_compact_trace(reference_path, reference, manifest=manifests[0])
    save_topk_compact_trace(candidate_path, candidate, manifest=manifests[1])

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        allowed_identity_difference_paths=CUDA_ALLOCATOR_AB_IDENTITY_PATHS,
        require_canonical_cuda_allocator_ab=True,
    )

    assert report["validation_passed"] is False
    assert report["identity"]["artifact_identity"]["passed"] is True
    assert report["cuda_allocator_ab_contract"]["passed"] is False


def _save_embedding_edge_pair(
    tmp_path,
    *,
    reference_strategy: str = "scalar_v1",
    candidate_strategy: str = "vectorized_v1",
    candidate_code_revision: str = "same-commit",
):
    reference_trace = _topk_trace()
    candidate_trace = deepcopy(reference_trace)
    _complete_trace_metadata(reference_trace)
    _complete_trace_metadata(candidate_trace)
    reference_path = tmp_path / "embedding-edge-scalar"
    candidate_path = tmp_path / "embedding-edge-vectorized"
    save_topk_compact_trace(
        reference_path,
        reference_trace,
        manifest=_manifest(
            "eager",
            embedding_edge_materialization=reference_strategy,
            code_revision="same-commit",
        ),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate_trace,
        manifest=_manifest(
            "eager",
            embedding_edge_materialization=candidate_strategy,
            code_revision=candidate_code_revision,
        ),
    )
    return reference_path, candidate_path


def test_embedding_edge_ab_passes_strict_canonical_profile(tmp_path) -> None:
    reference, candidate = _save_embedding_edge_pair(tmp_path)

    report = compare_execution_artifacts(
        reference,
        candidate,
        **comparison_options(
            build_parser().parse_args(
                [
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(tmp_path / "unused.json"),
                    "--embedding-edge-ab",
                ]
            )
        ),
    )

    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    assert report["embedding_edge_ab_contract"]["passed"] is True
    assert {gate["gate"] for gate in report["gates"]} >= {
        "same_gpu_family",
        "same_gpu_model",
        "canonical_embedding_edge_ab_pair",
        "exact_node_topology",
        "exact_edge_topology",
        *(f"{group}_numeric_tolerance" for group in TOLERANCE_GROUPS),
    }
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} == set(
        EMBEDDING_EDGE_AB_IDENTITY_PATHS
    )


def test_embedding_edge_ab_cli_resolves_strict_non_overridable_profile() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference",
            "reference",
            "--candidate",
            "candidate",
            "--output",
            "report.json",
            "--embedding-edge-ab",
        ]
    )

    options = comparison_options(args)

    assert options["allowed_identity_difference_paths"] == (
        EMBEDDING_EDGE_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_same_gpu_family"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert options["require_canonical_cuda_allocator_ab"] is False
    assert options["require_canonical_embedding_edge_ab"] is True
    assert options["tolerances"] == {
        group: NumericTolerance(absolute=0.0, relative=0.0)
        for group in TOLERANCE_GROUPS
    }

    args.allow_identity_difference = ["artifact_identity.code_revision.*"]
    with pytest.raises(ValueError, match="cannot be combined"):
        comparison_options(args)

    args.allow_identity_difference = []
    args.target_atol = 1e-6
    with pytest.raises(ValueError, match="fixes every numerical tolerance at zero"):
        comparison_options(args)

    args.target_atol = None
    args.cuda_allocator_ab = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        comparison_options(args)


def test_embedding_edge_ab_rejects_code_revision_drift(tmp_path) -> None:
    reference, candidate = _save_embedding_edge_pair(
        tmp_path, candidate_code_revision="different-commit"
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        **comparison_options(
            build_parser().parse_args(
                [
                    "--reference",
                    str(reference),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(tmp_path / "unused.json"),
                    "--embedding-edge-ab",
                ]
            )
        ),
    )

    assert report["validation_passed"] is False
    disallowed = report["identity"]["artifact_identity"]["unallowed_differences"]
    assert {difference["path"] for difference in disallowed} == {
        "artifact_identity.code_revision.git_commit"
    }


@pytest.mark.parametrize(
    ("reference_strategy", "candidate_strategy"),
    [
        ("scalar_v1", "scalar_v1"),
        ("vectorized_v1", "scalar_v1"),
    ],
)
def test_embedding_edge_ab_rejects_equal_or_reversed_lanes(
    tmp_path, reference_strategy, candidate_strategy
) -> None:
    reference, candidate = _save_embedding_edge_pair(
        tmp_path,
        reference_strategy=reference_strategy,
        candidate_strategy=candidate_strategy,
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=EMBEDDING_EDGE_AB_IDENTITY_PATHS,
        require_canonical_embedding_edge_ab=True,
    )

    assert report["validation_passed"] is False
    assert report["embedding_edge_ab_contract"]["passed"] is False


def test_embedding_edge_materialization_requires_exact_allowlist_path(
    tmp_path,
) -> None:
    reference, candidate = _save_embedding_edge_pair(tmp_path)

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config.embedding_edge_materialization.*"
            ],
        )


def _save_cross_layer_jacobian_pair(
    tmp_path,
    *,
    reference_strategy: str = "full_model_v1",
    candidate_strategy: str = "cached_range_v1",
    candidate_code_revision: str = "same-commit",
    reference_receipts: list[dict[str, str]] | None = None,
    candidate_receipts: list[dict[str, str]] | None = None,
    include_candidate_pair_coordinates: bool = True,
):
    reference_trace = _topk_trace()
    candidate_trace = deepcopy(reference_trace)
    _complete_trace_metadata(reference_trace)
    _complete_trace_metadata(candidate_trace)
    canonical_receipts = [
        {"name": "selected_source_activations", "sha256": "1" * 64},
        {"name": "selected_target_activations", "sha256": "2" * 64},
        {"name": "selected_raw_jacobian", "sha256": "3" * 64},
    ]
    if reference_receipts is None:
        reference_receipts = deepcopy(canonical_receipts)
    if candidate_receipts is None:
        candidate_receipts = deepcopy(canonical_receipts)
    reference_trace.circuit_data.trace_metadata["instrumentation"] = {
        "layer_pairs": [
            {
                "src_layer": 1,
                "tgt_layer": 3,
                "exact_receipts": reference_receipts,
            }
        ]
    }
    candidate_pair = {"exact_receipts": candidate_receipts}
    if include_candidate_pair_coordinates:
        candidate_pair.update({"src_layer": 1, "tgt_layer": 3})
    candidate_trace.circuit_data.trace_metadata["instrumentation"] = {
        "layer_pairs": [candidate_pair]
    }
    reference_path = tmp_path / "cross-layer-jacobian-full-model"
    candidate_path = tmp_path / "cross-layer-jacobian-cached-range"
    save_topk_compact_trace(
        reference_path,
        reference_trace,
        manifest=_manifest(
            "eager",
            cross_layer_jacobian_execution=reference_strategy,
            code_revision="same-commit",
        ),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate_trace,
        manifest=_manifest(
            "eager",
            cross_layer_jacobian_execution=candidate_strategy,
            code_revision=candidate_code_revision,
        ),
    )
    return reference_path, candidate_path


def _cross_layer_jacobian_options(reference, candidate, output):
    return comparison_options(
        build_parser().parse_args(
            [
                "--reference",
                str(reference),
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                "--cross-layer-jacobian-ab",
            ]
        )
    )


def test_cross_layer_jacobian_ab_passes_strict_canonical_profile(tmp_path) -> None:
    reference, candidate = _save_cross_layer_jacobian_pair(tmp_path)

    report = compare_execution_artifacts(
        reference,
        candidate,
        **_cross_layer_jacobian_options(reference, candidate, tmp_path / "unused.json"),
    )

    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    assert report["cross_layer_jacobian_ab_contract"]["passed"] is True
    assert report["cross_layer_jacobian_ab_contract"]["exact_receipts"]["passed"]
    assert {gate["gate"] for gate in report["gates"]} >= {
        "same_gpu_family",
        "same_gpu_model",
        "canonical_cross_layer_jacobian_ab_pair",
        "exact_node_topology",
        "exact_edge_topology",
        *(f"{group}_numeric_tolerance" for group in TOLERANCE_GROUPS),
    }
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} == set(
        CROSS_LAYER_JACOBIAN_AB_IDENTITY_PATHS
    )


@pytest.mark.parametrize("malformation", ["missing", "mismatched", "reordered"])
def test_cross_layer_jacobian_ab_fails_closed_on_receipts(
    tmp_path, malformation
) -> None:
    canonical = [
        {"name": "selected_source_activations", "sha256": "1" * 64},
        {"name": "selected_target_activations", "sha256": "2" * 64},
        {"name": "selected_raw_jacobian", "sha256": "3" * 64},
    ]
    candidate = deepcopy(canonical)
    if malformation == "missing":
        candidate.pop()
    elif malformation == "mismatched":
        candidate[2]["sha256"] = "4" * 64
    else:
        candidate.reverse()
    reference_path, candidate_path = _save_cross_layer_jacobian_pair(
        tmp_path,
        reference_receipts=canonical,
        candidate_receipts=candidate,
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        **_cross_layer_jacobian_options(
            reference_path, candidate_path, tmp_path / "unused.json"
        ),
    )

    receipt_contract = report["cross_layer_jacobian_ab_contract"]["exact_receipts"]
    assert report["validation_passed"] is False
    assert receipt_contract["passed"] is False


def test_cross_layer_jacobian_ab_fails_closed_without_pair_coordinates(
    tmp_path,
) -> None:
    reference_path, candidate_path = _save_cross_layer_jacobian_pair(
        tmp_path,
        include_candidate_pair_coordinates=False,
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        **_cross_layer_jacobian_options(
            reference_path, candidate_path, tmp_path / "unused.json"
        ),
    )

    assert report["validation_passed"] is False
    assert not report["cross_layer_jacobian_ab_contract"]["exact_receipts"]["passed"]


def test_cross_layer_jacobian_ab_cli_is_strict_and_non_overridable() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference",
            "reference",
            "--candidate",
            "candidate",
            "--output",
            "report.json",
            "--cross-layer-jacobian-ab",
        ]
    )

    options = comparison_options(args)

    assert options["allowed_identity_difference_paths"] == (
        CROSS_LAYER_JACOBIAN_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_same_gpu_family"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert options["require_canonical_cuda_allocator_ab"] is False
    assert options["require_canonical_embedding_edge_ab"] is False
    assert options["require_canonical_cross_layer_jacobian_ab"] is True
    assert options["tolerances"] == {
        group: NumericTolerance(absolute=0.0, relative=0.0)
        for group in TOLERANCE_GROUPS
    }

    args.allow_identity_difference = ["artifact_identity.code_revision.*"]
    with pytest.raises(ValueError, match="cannot be combined"):
        comparison_options(args)

    args.allow_identity_difference = []
    args.edge_rtol = 1e-6
    with pytest.raises(ValueError, match="fixes every numerical tolerance at zero"):
        comparison_options(args)

    args.edge_rtol = None
    args.embedding_edge_ab = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        comparison_options(args)


def test_cross_layer_jacobian_ab_rejects_code_revision_drift(tmp_path) -> None:
    reference, candidate = _save_cross_layer_jacobian_pair(
        tmp_path, candidate_code_revision="different-commit"
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        **_cross_layer_jacobian_options(reference, candidate, tmp_path / "unused.json"),
    )

    assert report["validation_passed"] is False
    disallowed = report["identity"]["artifact_identity"]["unallowed_differences"]
    assert {difference["path"] for difference in disallowed} == {
        "artifact_identity.code_revision.git_commit"
    }


@pytest.mark.parametrize(
    ("reference_strategy", "candidate_strategy"),
    [
        ("full_model_v1", "full_model_v1"),
        ("cached_range_v1", "full_model_v1"),
    ],
)
def test_cross_layer_jacobian_ab_rejects_equal_or_reversed_lanes(
    tmp_path, reference_strategy, candidate_strategy
) -> None:
    reference, candidate = _save_cross_layer_jacobian_pair(
        tmp_path,
        reference_strategy=reference_strategy,
        candidate_strategy=candidate_strategy,
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=CROSS_LAYER_JACOBIAN_AB_IDENTITY_PATHS,
        require_canonical_cross_layer_jacobian_ab=True,
    )

    assert report["validation_passed"] is False
    assert report["cross_layer_jacobian_ab_contract"]["passed"] is False


def test_cross_layer_jacobian_execution_requires_exact_allowlist_path(
    tmp_path,
) -> None:
    reference, candidate = _save_cross_layer_jacobian_pair(tmp_path)

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config.cross_layer_jacobian_execution.*"
            ],
        )


def test_backend_qualification_fails_hard_source_identity_mismatch(tmp_path) -> None:
    reference, candidate = _save_pair(tmp_path)
    candidate_manifest_path = candidate / "manifest.json"
    # Re-save is unnecessary: source identity is present in both the top-level
    # manifest and the hashed payload identity, so create a second valid pair.
    _ = candidate_manifest_path
    mismatched = tmp_path / "candidate-mismatched"
    trace = _topk_trace()
    _complete_trace_metadata(trace)
    save_topk_compact_trace(
        mismatched,
        trace,
        manifest=_manifest("sdpa_ov_only", source_id="wrong-source"),
    )

    report = compare_attention_backend_artifacts(
        reference,
        mismatched,
        allowed_identity_difference_paths=_allowed_paths(),
    )

    assert report["validation_passed"] is False
    assert report["qualification_passed"] is None
    assert report["diagnostic_only"] is True
    hard = {item["field"]: item for item in report["identity"]["hard_checks"]}
    assert hard["manifest.source_width1_artifact_id"]["reason"] == "mismatch"
    assert report["identity"]["artifact_identity"]["passed"] is False


def test_backend_qualification_reports_topology_and_numeric_drift(tmp_path) -> None:
    reference_trace = _topk_trace()
    candidate_trace = deepcopy(reference_trace)
    _complete_trace_metadata(reference_trace)
    _complete_trace_metadata(candidate_trace)
    candidate_trace.circuit_data.df_node.loc[0, "attribution"] += 0.25
    candidate_trace.circuit_data.df_node.at[0, "contrib_map"] = [
        0.4,
        0.1,
        0.2,
        0.3,
        0.4,
    ]
    candidate_trace.circuit_data.df_edge = candidate_trace.circuit_data.df_edge.iloc[
        0:0
    ].copy()
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    save_topk_compact_trace(reference, reference_trace, manifest=_manifest("eager"))
    save_topk_compact_trace(
        candidate, candidate_trace, manifest=_manifest("sdpa_ov_only")
    )

    report = compare_attention_backend_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=_allowed_paths(),
        tolerances={
            "node": NumericTolerance(absolute=0.01, relative=0.0),
            "candidate_profile": NumericTolerance(absolute=0.01, relative=0.0),
        },
        require_exact_edge_topology=True,
    )

    assert report["qualification_passed"] is False
    assert report["topology"]["edges"]["jaccard"] == 0.0
    assert report["node_values_on_intersection"]["attribution"][
        "max_absolute_error"
    ] == pytest.approx(0.25)
    assert report["candidate_profiles_on_node_intersection"]["overall"][
        "max_absolute_error"
    ] == pytest.approx(0.4)
    failed_gates = {gate["gate"] for gate in report["gates"] if not gate["passed"]}
    assert "exact_edge_topology" in failed_gates
    assert "node_numeric_tolerance" in failed_gates
    assert "candidate_profile_numeric_tolerance" in failed_gates


def test_backend_qualification_rejects_allowlisting_scientific_identity(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(tmp_path)

    with pytest.raises(ValueError, match="may only allow"):
        compare_attention_backend_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=["artifact_identity.model.revision"],
        )

    with pytest.raises(ValueError, match="stop-gradient attention backend"):
        compare_attention_backend_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config.percentage_threshold"
            ],
        )


def test_backend_qualification_rejects_invalid_identity_hash(tmp_path) -> None:
    reference, candidate = _save_pair(tmp_path)
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_identity"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_attention_backend_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=_allowed_paths(),
    )

    assert report["validation_passed"] is False
    assert report["identity"]["candidate_integrity"]["passed"] is False


def test_qualification_report_is_atomic_and_never_overwritten(tmp_path) -> None:
    reference, candidate = _save_pair(tmp_path)
    report = compare_attention_backend_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=_allowed_paths(),
    )
    output = tmp_path / "reports" / "qualification.json"

    save_qualification_report(output, report)

    assert output.is_file()
    assert not list(output.parent.glob(".qualification.json.tmp-*"))
    with pytest.raises(FileExistsError):
        save_qualification_report(output, report)
