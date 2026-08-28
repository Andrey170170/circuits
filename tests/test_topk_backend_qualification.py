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
    CUDA_ALLOCATOR_SNAPSHOT_TELEMETRY_IDENTITY_PATH,
    EMBEDDING_EDGE_AB_IDENTITY_PATHS,
    POST_SELECTION_STATE_STORAGE_AB_IDENTITY_PATHS,
    SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_IDENTITY_PATHS,
    SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
    SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
    SELECTED_TARGET_LOGIT_EXECUTION_AB_IDENTITY_PATHS,
    STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS,
    STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_IDENTITY_PATHS,
    STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_IDENTITY_PATHS,
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
_OMIT = object()


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
    stop_gradient_embed_contribution_target_lane_chunk_size: object = _UNSET,
    selected_attribution_neuron_lane_chunk_size: object = _UNSET,
    stop_gradient_selected_attribution_forward_execution: object = _UNSET,
    stop_gradient_selected_attribution_storage: object = _UNSET,
    selected_target_logit_execution: object = _UNSET,
    post_selection_state_storage: object = _UNSET,
    selected_target_logit_ig_steps: object = _UNSET,
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
    if stop_gradient_embed_contribution_target_lane_chunk_size is not _UNSET:
        manifest["artifact_identity"]["adag_config"][
            "stop_gradient_embed_contribution_target_lane_chunk_size"
        ] = stop_gradient_embed_contribution_target_lane_chunk_size
    if selected_attribution_neuron_lane_chunk_size is not _UNSET:
        manifest["artifact_identity"]["adag_config"][
            "selected_attribution_neuron_lane_chunk_size"
        ] = selected_attribution_neuron_lane_chunk_size
    if stop_gradient_selected_attribution_forward_execution is not _UNSET:
        manifest["artifact_identity"]["adag_config"][
            "stop_gradient_selected_attribution_forward_execution"
        ] = stop_gradient_selected_attribution_forward_execution
    if stop_gradient_selected_attribution_storage is not _UNSET:
        manifest["artifact_identity"]["adag_config"][
            "stop_gradient_selected_attribution_storage"
        ] = stop_gradient_selected_attribution_storage
    if selected_target_logit_execution is not _UNSET:
        manifest["artifact_identity"]["adag_config"][
            "selected_target_logit_execution"
        ] = selected_target_logit_execution
        if selected_target_logit_ig_steps is not _OMIT:
            manifest["artifact_identity"]["adag_config"]["ig_steps"] = (
                None
                if selected_target_logit_ig_steps is _UNSET
                else selected_target_logit_ig_steps
            )
        manifest["artifact_identity"]["adag_config"]["center_logits"] = False
    if post_selection_state_storage is not _UNSET:
        manifest["artifact_identity"]["adag_config"]["post_selection_state_storage"] = (
            post_selection_state_storage
        )
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
    reference_stop_gradient_embed_chunk_size: object = _UNSET,
    candidate_stop_gradient_embed_chunk_size: object = _UNSET,
    reference_selected_attribution_chunk_size: object = _UNSET,
    candidate_selected_attribution_chunk_size: object = _UNSET,
    reference_selected_attribution_instrumentation: dict | None = None,
    candidate_selected_attribution_instrumentation: dict | None = None,
    reference_selected_attribution_forward_execution: object = _UNSET,
    candidate_selected_attribution_forward_execution: object = _UNSET,
    reference_selected_attribution_forward_instrumentation: dict | None = None,
    candidate_selected_attribution_forward_instrumentation: dict | None = None,
    reference_selected_attribution_storage: object = _UNSET,
    candidate_selected_attribution_storage: object = _UNSET,
    reference_selected_attribution_storage_instrumentation: dict | None = None,
    candidate_selected_attribution_storage_instrumentation: dict | None = None,
    reference_selected_target_logit_execution: object = _UNSET,
    candidate_selected_target_logit_execution: object = _UNSET,
    reference_selected_target_logit_ig_steps: object = _UNSET,
    candidate_selected_target_logit_ig_steps: object = _UNSET,
    reference_selected_target_logit_instrumentation: dict | None = None,
    candidate_selected_target_logit_instrumentation: dict | None = None,
    reference_post_selection_state_storage: object = _UNSET,
    candidate_post_selection_state_storage: object = _UNSET,
    reference_post_selection_state_instrumentation: dict | None = None,
    candidate_post_selection_state_instrumentation: dict | None = None,
    reference_stop_gradient_embed_receipts: list[dict] | None = None,
    candidate_stop_gradient_embed_receipts: list[dict] | None = None,
    reference_dtype: str = "bfloat16",
    candidate_dtype: str = "bfloat16",
    candidate_mlp_profile_delta: float = 0.0,
    candidate_non_logit_attr_map_delta: float = 0.0,
    candidate_logit_attr_map_delta: float = 0.0,
    candidate_unaffected_edge_delta: float = 0.0,
    extra_logit_node: bool = False,
):
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    if (
        reference_selected_embed_receipts is not None
        or candidate_selected_embed_receipts is not None
        or reference_stop_gradient_embed_receipts is not None
        or candidate_stop_gradient_embed_receipts is not None
    ):
        for trace in (reference, candidate):
            logit_rows = pd.concat(
                [trace.circuit_data.df_node.iloc[[0]].copy() for _ in range(5)],
                ignore_index=True,
            )
            logit_rows.loc[:, "layer"] = 1
            logit_rows.loc[:, "neuron"] = [40, 101, 102, 103, 104]
            if extra_logit_node:
                extra_logit = logit_rows.iloc[[0]].copy()
                extra_logit.loc[:, "neuron"] = 105
                logit_rows = pd.concat([logit_rows, extra_logit], ignore_index=True)
            embedding_row = trace.circuit_data.df_node.iloc[[0]].copy()
            embedding_row.loc[:, "layer"] = -1
            embedding_row.loc[:, "token"] = 2
            embedding_row.loc[:, "neuron"] = 42
            trace.circuit_data.df_node = pd.concat(
                [trace.circuit_data.df_node, logit_rows, embedding_row],
                ignore_index=True,
            )
        if candidate_mlp_profile_delta:
            profile = list(candidate.circuit_data.df_node.iloc[0].contrib_map)
            profile[0] += candidate_mlp_profile_delta
            candidate.circuit_data.df_node.at[0, "contrib_map"] = profile
        if candidate_non_logit_attr_map_delta:
            profile = list(candidate.circuit_data.df_node.iloc[0].attr_map)
            profile[0] += candidate_non_logit_attr_map_delta
            candidate.circuit_data.df_node.at[0, "attr_map"] = profile
        if candidate_logit_attr_map_delta:
            row_index = candidate.circuit_data.df_node.index[
                candidate.circuit_data.df_node["layer"] == 1
            ][0]
            profile = list(candidate.circuit_data.df_node.at[row_index, "attr_map"])
            profile[0] += candidate_logit_attr_map_delta
            candidate.circuit_data.df_node.at[row_index, "attr_map"] = profile
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
    if reference_stop_gradient_embed_receipts is not None:
        instrumentation = reference.circuit_data.trace_metadata.setdefault(
            "instrumentation", {}
        )
        instrumentation.setdefault("execution_records", {})[
            "stop_gradient_embed_contribution_vjp"
        ] = reference_stop_gradient_embed_receipts
    if candidate_stop_gradient_embed_receipts is not None:
        instrumentation = candidate.circuit_data.trace_metadata.setdefault(
            "instrumentation", {}
        )
        instrumentation.setdefault("execution_records", {})[
            "stop_gradient_embed_contribution_vjp"
        ] = candidate_stop_gradient_embed_receipts
    if reference_selected_attribution_instrumentation is not None:
        reference.circuit_data.trace_metadata["instrumentation"] = (
            reference_selected_attribution_instrumentation
        )
    if candidate_selected_attribution_instrumentation is not None:
        candidate.circuit_data.trace_metadata["instrumentation"] = (
            candidate_selected_attribution_instrumentation
        )
    if reference_selected_attribution_forward_instrumentation is not None:
        reference.circuit_data.trace_metadata["instrumentation"] = (
            reference_selected_attribution_forward_instrumentation
        )
    if candidate_selected_attribution_forward_instrumentation is not None:
        candidate.circuit_data.trace_metadata["instrumentation"] = (
            candidate_selected_attribution_forward_instrumentation
        )
    if reference_selected_attribution_storage_instrumentation is not None:
        reference.circuit_data.trace_metadata["instrumentation"] = (
            reference_selected_attribution_storage_instrumentation
        )
    if candidate_selected_attribution_storage_instrumentation is not None:
        candidate.circuit_data.trace_metadata["instrumentation"] = (
            candidate_selected_attribution_storage_instrumentation
        )
    if reference_selected_target_logit_instrumentation is not None:
        reference.circuit_data.trace_metadata["instrumentation"] = (
            reference_selected_target_logit_instrumentation
        )
    if candidate_selected_target_logit_instrumentation is not None:
        candidate.circuit_data.trace_metadata["instrumentation"] = (
            candidate_selected_target_logit_instrumentation
        )
    if reference_post_selection_state_instrumentation is not None:
        reference.circuit_data.trace_metadata["instrumentation"] = (
            reference_post_selection_state_instrumentation
        )
    if candidate_post_selection_state_instrumentation is not None:
        candidate.circuit_data.trace_metadata["instrumentation"] = (
            candidate_post_selection_state_instrumentation
        )
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
            stop_gradient_embed_contribution_target_lane_chunk_size=(
                reference_stop_gradient_embed_chunk_size
            ),
            selected_attribution_neuron_lane_chunk_size=(
                reference_selected_attribution_chunk_size
            ),
            stop_gradient_selected_attribution_forward_execution=(
                reference_selected_attribution_forward_execution
            ),
            stop_gradient_selected_attribution_storage=(
                reference_selected_attribution_storage
            ),
            selected_target_logit_execution=(reference_selected_target_logit_execution),
            selected_target_logit_ig_steps=reference_selected_target_logit_ig_steps,
            post_selection_state_storage=reference_post_selection_state_storage,
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
            stop_gradient_embed_contribution_target_lane_chunk_size=(
                candidate_stop_gradient_embed_chunk_size
            ),
            selected_attribution_neuron_lane_chunk_size=(
                candidate_selected_attribution_chunk_size
            ),
            stop_gradient_selected_attribution_forward_execution=(
                candidate_selected_attribution_forward_execution
            ),
            stop_gradient_selected_attribution_storage=(
                candidate_selected_attribution_storage
            ),
            selected_target_logit_execution=(candidate_selected_target_logit_execution),
            selected_target_logit_ig_steps=candidate_selected_target_logit_ig_steps,
            post_selection_state_storage=candidate_post_selection_state_storage,
        ),
    )
    return reference_path, candidate_path


def _allowed_paths() -> list[str]:
    return [
        "artifact_identity.adag_config.stop_gradient_attention_backend",
        "artifact_identity.code_revision.*",
    ]


def _post_selection_state_instrumentation(strategy: str) -> dict:
    dense = strategy == "dense_v1"
    selected_values_bytes = 4
    selected_coordinates_bytes = 48
    retained_bytes = 1000 if dense else 4
    released_bytes = 0 if dense else 996
    record = {
        "strategy": strategy,
        "selected_occurrence_count": 2,
        "active_layers": [0, 1],
        "selected_values_shape": [2, 1, 1],
        "selected_values_dtype": "torch.bfloat16",
        "selected_values_bytes": selected_values_bytes,
        "selected_values_raw_sha256": "a" * 64,
        "selected_coordinates_shape": [2, 3],
        "selected_coordinates_dtype": "torch.int64",
        "selected_coordinates_bytes": selected_coordinates_bytes,
        "selected_coordinates_raw_sha256": "b" * 64,
        "logical_input_bytes": 1000,
        "logical_retained_bytes": retained_bytes,
        "logical_released_bytes": released_bytes,
        "retains_dense_mlp_final_attributions": dense,
        "retains_dense_important_neuron_mask": dense,
        "retains_unused_mlp_final_acts": dense,
        "retains_unused_embed_final_acts": dense,
        "state_values_device": "cuda:0" if dense else "cpu",
    }
    return {
        "counters": {
            "post_selection_state_storage": strategy,
            "post_selection_state_storage_execution_count": 1,
            "post_selection_state_selected_occurrence_count": 2,
        },
        "execution_records": {"post_selection_state_storage": [record]},
        "cuda_allocator_snapshots": {
            "captures": [
                {
                    "capture_index": 0,
                    "point": "before_post_selection_state_storage",
                    "metadata": {
                        "strategy": strategy,
                        "logical_input_bytes": 1000,
                    },
                    "current_allocator_stats": {"active_bytes": 2000},
                    "block_states": {"active_allocated": {"bytes": 2000}},
                },
                {
                    "capture_index": 1,
                    "point": "after_post_selection_state_storage_release",
                    "metadata": {
                        "strategy": strategy,
                        "logical_input_bytes": 1000,
                        "logical_retained_bytes": retained_bytes,
                        "logical_released_bytes": released_bytes,
                    },
                    "current_allocator_stats": {
                        "active_bytes": 2000 if dense else 1000
                    },
                    "block_states": {
                        "active_allocated": {"bytes": 2000 if dense else 1000}
                    },
                },
            ]
        },
    }


def test_post_selection_state_storage_ab_is_strict_and_receipt_bound(tmp_path) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="flash_sdpa_causal_v1",
        candidate_backend="flash_sdpa_causal_v1",
        reference_post_selection_state_storage="dense_v1",
        candidate_post_selection_state_storage="compact_cpu_v1",
        reference_post_selection_state_instrumentation=(
            _post_selection_state_instrumentation("dense_v1")
        ),
        candidate_post_selection_state_instrumentation=(
            _post_selection_state_instrumentation("compact_cpu_v1")
        ),
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            POST_SELECTION_STATE_STORAGE_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        require_canonical_post_selection_state_storage_ab=True,
    )

    assert report["validation_passed"] is True
    assert report["post_selection_state_storage_ab_contract"]["passed"] is True

    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output",
            str(tmp_path / "report.json"),
            "--post-selection-state-storage-ab",
        ]
    )
    options = comparison_options(args)
    assert options["allowed_identity_difference_paths"] == (
        POST_SELECTION_STATE_STORAGE_AB_IDENTITY_PATHS
    )
    assert options["require_canonical_post_selection_state_storage_ab"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_snapshots",
        "compact_cuda_placement",
        "bad_values_rank",
        "bad_batch_width",
        "bad_target_width",
        "bad_identity_dtype",
        "bad_coordinate_dtype",
        "bad_value_bytes",
        "zero_release",
        "unequal_input_bytes",
        "reversed_snapshots",
        "insufficient_active_allocated_drop",
        "nondecreasing_compact_active_bytes",
    ],
)
def test_post_selection_state_storage_ab_rejects_malformed_evidence(
    tmp_path, mutation: str
) -> None:
    reference_instrumentation = _post_selection_state_instrumentation("dense_v1")
    candidate_instrumentation = _post_selection_state_instrumentation("compact_cpu_v1")
    candidate_record = candidate_instrumentation["execution_records"][
        "post_selection_state_storage"
    ][0]
    candidate_captures = candidate_instrumentation["cuda_allocator_snapshots"][
        "captures"
    ]
    if mutation == "missing_snapshots":
        del candidate_instrumentation["cuda_allocator_snapshots"]
    elif mutation == "compact_cuda_placement":
        candidate_record["state_values_device"] = "cuda:0"
    elif mutation == "bad_values_rank":
        candidate_record["selected_values_shape"] = [2, 1]
    elif mutation in {"bad_batch_width", "bad_target_width"}:
        candidate_record["selected_values_shape"] = (
            [2, 2, 1] if mutation == "bad_batch_width" else [2, 1, 2]
        )
        candidate_record["selected_values_bytes"] = 8
        candidate_record["logical_retained_bytes"] = 8
        candidate_record["logical_released_bytes"] = 992
        candidate_captures[1]["metadata"]["logical_retained_bytes"] = 8
        candidate_captures[1]["metadata"]["logical_released_bytes"] = 992
    elif mutation == "bad_identity_dtype":
        candidate_record["selected_values_dtype"] = "torch.float16"
    elif mutation == "bad_coordinate_dtype":
        candidate_record["selected_coordinates_dtype"] = "torch.int32"
    elif mutation == "bad_value_bytes":
        candidate_record["selected_values_bytes"] = 5
    elif mutation == "zero_release":
        candidate_record["logical_retained_bytes"] = 1000
        candidate_record["logical_released_bytes"] = 0
        candidate_captures[1]["metadata"]["logical_retained_bytes"] = 1000
        candidate_captures[1]["metadata"]["logical_released_bytes"] = 0
    elif mutation == "unequal_input_bytes":
        candidate_record["logical_input_bytes"] = 1001
        candidate_record["logical_released_bytes"] = 949
        candidate_captures[0]["metadata"]["logical_input_bytes"] = 1001
        candidate_captures[1]["metadata"]["logical_input_bytes"] = 1001
        candidate_captures[1]["metadata"]["logical_released_bytes"] = 949
    elif mutation == "reversed_snapshots":
        candidate_captures.reverse()
    elif mutation == "insufficient_active_allocated_drop":
        candidate_captures[1]["block_states"]["active_allocated"]["bytes"] = 1100
    elif mutation == "nondecreasing_compact_active_bytes":
        candidate_captures[1]["current_allocator_stats"]["active_bytes"] = 2000
    else:  # pragma: no cover - guards the test table
        raise AssertionError(mutation)

    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="flash_sdpa_causal_v1",
        candidate_backend="flash_sdpa_causal_v1",
        reference_post_selection_state_storage="dense_v1",
        candidate_post_selection_state_storage="compact_cpu_v1",
        reference_post_selection_state_instrumentation=reference_instrumentation,
        candidate_post_selection_state_instrumentation=candidate_instrumentation,
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            POST_SELECTION_STATE_STORAGE_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        require_canonical_post_selection_state_storage_ab=True,
    )

    assert report["validation_passed"] is False
    assert report["post_selection_state_storage_ab_contract"]["passed"] is False


def test_post_selection_state_storage_allowlist_is_scalar_only(tmp_path) -> None:
    reference, candidate = _save_pair(tmp_path)
    with pytest.raises(ValueError, match="scalar execution-strategy"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config.post_selection_state_storage.*"
            ],
        )


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


def test_backend_qualification_can_allow_selected_attribution_width_difference(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_selected_attribution_chunk_size=None,
        candidate_selected_attribution_chunk_size=1,
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=[
            *_allowed_paths(),
            *SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_IDENTITY_PATHS,
        ],
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
    )

    assert report["validation_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} >= {
        "artifact_identity.adag_config.stop_gradient_attention_backend",
        "artifact_identity.adag_config.selected_attribution_neuron_lane_chunk_size",
    }

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config."
                "selected_attribution_neuron_lane_chunk_size.*"
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


def _selected_attribution_runtime_instrumentation(
    requested_width: int | None,
) -> dict:
    resolved_width = 50 if requested_width is None else requested_width
    selected_counts = [{"layer": 0, "count": 2}, {"layer": 3, "count": 3}]
    vjp_calls = []
    projection_calls = []
    for selected in selected_counts:
        layer = selected["layer"]
        count = selected["count"]
        chunk_count = (count + resolved_width - 1) // resolved_width
        for chunk_index, chunk_start in enumerate(range(0, count, resolved_width)):
            chunk_neuron_count = min(resolved_width, count - chunk_start)
            lanes = chunk_neuron_count
            raw_shape = [lanes, 1, 7, 11]
            common = {
                "layer": layer,
                "chunk_start": chunk_start,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "neuron_lane_chunk_size_resolved": resolved_width,
                "chunk_neuron_count": chunk_neuron_count,
            }
            call_index = len(vjp_calls)
            vjp_calls.append(
                {
                    "call_index": call_index,
                    "failed": False,
                    "wall_seconds": 0.1,
                    "metadata": {
                        **common,
                        "operation_kind": "batched_vjp",
                        "lane_count": lanes,
                        "differentiated_output_shape": [chunk_neuron_count, 1],
                        "differentiated_input_shape": [1, 7, 11],
                        "grad_outputs_shape": [lanes, lanes],
                        "vjp_result_shape": raw_shape,
                    },
                    "cuda_memory": {},
                }
            )
            projection_calls.append(
                {
                    "call_index": call_index,
                    "failed": False,
                    "wall_seconds": 0.01,
                    "metadata": {
                        **common,
                        "operation_kind": "terminal_projection",
                        "raw_vjp_result_shape": raw_shape,
                        "source_token_count": 3,
                        "return_gradient_only": False,
                        "terminal_projection_detached": True,
                        "retained_chunk_count_before": chunk_index,
                        "retained_chunk_count_after": chunk_index + 1,
                        "projected_shape": [chunk_neuron_count, 1, 3],
                        "projected_requires_grad": False,
                    },
                    "cuda_memory": {},
                }
            )
    chunk_executions = len(vjp_calls)
    chunk_counters = {
        "selected_attribution_chunk_size": resolved_width,
        "selected_attribution_chunks_per_pass": chunk_executions,
        "selected_attribution_pass_count": 1,
        "selected_attribution_chunk_executions": chunk_executions,
    }
    return {
        "counters": {
            "selected_attribution_neuron_lane_chunk_size_requested": (requested_width),
            "selected_attribution_neuron_lane_chunk_size_resolved": resolved_width,
            **chunk_counters,
        },
        "early_predictors": {
            "selected_neuron_counts_by_layer": selected_counts,
            "ig_steps": None,
            "ig_execution_count": 1,
            **chunk_counters,
        },
        "stages": {
            "selected_attribution_vjp": {
                "calls": chunk_executions,
                "failed_calls": 0,
                "call_measurements": vjp_calls,
            },
            "selected_attribution_chunk_projection": {
                "calls": chunk_executions,
                "failed_calls": 0,
                "call_measurements": projection_calls,
            },
        },
    }


def test_selected_attribution_neuron_lane_ab_requires_canonical_runtime_widths(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_attribution_chunk_size=None,
        candidate_selected_attribution_chunk_size=1,
        reference_selected_attribution_instrumentation=(
            _selected_attribution_runtime_instrumentation(None)
        ),
        candidate_selected_attribution_instrumentation=(
            _selected_attribution_runtime_instrumentation(1)
        ),
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        require_canonical_selected_attribution_neuron_lane_chunk_ab=True,
    )

    assert report["qualification_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    contract = report["selected_attribution_neuron_lane_chunk_ab_contract"]
    assert contract["passed"] is True
    assert contract["reference_strategy"] == {
        "expected": None,
        "observed": None,
        "field_present": True,
        "passed": True,
    }
    assert contract["candidate_strategy"] == {
        "expected": 1,
        "observed": 1,
        "field_present": True,
        "passed": True,
    }
    runtime = contract["runtime_width_receipts"]
    assert runtime["checks"] == {
        "reference_runtime_width_proven": True,
        "candidate_runtime_width_proven": True,
        "cross_side_workload_equal": True,
    }
    assert len(runtime["reference_runtime"]["calls"]) == 2
    assert len(runtime["candidate_runtime"]["calls"]) == 5


@pytest.mark.parametrize(
    ("reference_width", "candidate_width", "failed_side"),
    [
        (_UNSET, 1, "reference_strategy"),
        (None, _UNSET, "candidate_strategy"),
        (50, 1, "reference_strategy"),
        (None, 2, "candidate_strategy"),
        (_UNSET, _UNSET, "both"),
    ],
)
def test_selected_attribution_neuron_lane_ab_rejects_wrong_or_missing_identity(
    tmp_path,
    reference_width: object,
    candidate_width: object,
    failed_side: str,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_attribution_chunk_size=reference_width,
        candidate_selected_attribution_chunk_size=candidate_width,
        reference_selected_attribution_instrumentation=(
            _selected_attribution_runtime_instrumentation(None)
        ),
        candidate_selected_attribution_instrumentation=(
            _selected_attribution_runtime_instrumentation(1)
        ),
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        require_canonical_selected_attribution_neuron_lane_chunk_ab=True,
    )

    assert report["validation_passed"] is False
    contract = report["selected_attribution_neuron_lane_chunk_ab_contract"]
    assert contract["passed"] is False
    if failed_side == "both":
        assert contract["reference_strategy"]["passed"] is False
        assert contract["candidate_strategy"]["passed"] is False
    else:
        assert contract[failed_side]["passed"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "missing_instrumentation",
        "missing_vjp_stage",
        "missing_projection_stage",
        "wrong_requested_counter",
        "wrong_duplicated_counter",
        "wrong_predictor_width",
        "wrong_lane_count",
        "wrong_projection_detach",
        "wrong_projected_requires_grad",
        "both_wrong_chunk_start",
    ],
)
def test_selected_attribution_neuron_lane_ab_fails_closed_on_runtime_receipts(
    tmp_path,
    defect: str,
) -> None:
    reference_runtime = _selected_attribution_runtime_instrumentation(None)
    candidate_runtime = _selected_attribution_runtime_instrumentation(1)
    if defect == "missing_instrumentation":
        candidate_runtime = None
    elif defect == "missing_vjp_stage":
        candidate_runtime["stages"].pop("selected_attribution_vjp")
    elif defect == "missing_projection_stage":
        candidate_runtime["stages"].pop("selected_attribution_chunk_projection")
    elif defect == "wrong_requested_counter":
        candidate_runtime["counters"][
            "selected_attribution_neuron_lane_chunk_size_requested"
        ] = 2
    elif defect == "wrong_duplicated_counter":
        candidate_runtime["counters"]["selected_attribution_chunk_executions"] = 4
    elif defect == "wrong_predictor_width":
        candidate_runtime["early_predictors"]["selected_attribution_chunk_size"] = 2
    elif defect == "wrong_lane_count":
        candidate_runtime["stages"]["selected_attribution_vjp"]["call_measurements"][0][
            "metadata"
        ]["lane_count"] = 2
    elif defect == "wrong_projection_detach":
        candidate_runtime["stages"]["selected_attribution_chunk_projection"][
            "call_measurements"
        ][0]["metadata"]["terminal_projection_detached"] = False
    elif defect == "wrong_projected_requires_grad":
        candidate_runtime["stages"]["selected_attribution_chunk_projection"][
            "call_measurements"
        ][0]["metadata"]["projected_requires_grad"] = True
    else:
        for runtime in (reference_runtime, candidate_runtime):
            runtime["stages"]["selected_attribution_vjp"]["call_measurements"][0][
                "metadata"
            ]["chunk_start"] = 99
            runtime["stages"]["selected_attribution_chunk_projection"][
                "call_measurements"
            ][0]["metadata"]["chunk_start"] = 99

    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_attribution_chunk_size=None,
        candidate_selected_attribution_chunk_size=1,
        reference_selected_attribution_instrumentation=reference_runtime,
        candidate_selected_attribution_instrumentation=candidate_runtime,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        require_canonical_selected_attribution_neuron_lane_chunk_ab=True,
    )

    assert report["validation_passed"] is False
    contract = report["selected_attribution_neuron_lane_chunk_ab_contract"]
    assert contract["passed"] is False
    assert contract["runtime_width_receipts"]["passed"] is False


def test_selected_attribution_neuron_lane_ab_cli_is_strict_and_non_overridable(
    tmp_path,
) -> None:
    args = build_parser().parse_args(
        [
            "--reference",
            str(tmp_path / "reference"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "report.json"),
            "--selected-attribution-neuron-lane-chunk-ab",
        ]
    )

    options = comparison_options(args)
    assert options["allowed_identity_difference_paths"] == (
        SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_same_gpu_family"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert options["require_canonical_selected_attribution_neuron_lane_chunk_ab"]
    assert options["tolerances"] == {
        group: NumericTolerance(absolute=0.0, relative=0.0)
        for group in TOLERANCE_GROUPS
    }

    args.allow_identity_difference = ["artifact_identity.code_revision.*"]
    with pytest.raises(ValueError, match="cannot be combined"):
        comparison_options(args)
    args.allow_identity_difference = []
    args.node_atol = 1e-6
    with pytest.raises(ValueError, match="fixes every numerical tolerance at zero"):
        comparison_options(args)
    args.node_atol = None
    args.cross_layer_jacobian_ab = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        comparison_options(args)


def _selected_attribution_forward_instrumentation(execution: str) -> dict:
    records = []
    full_entries = list(range(5))
    for layer in (0, 3):
        is_full = execution == "full_model_v1"
        record = {
            "execution": execution,
            "layer": layer,
            "decoder_layer_entries": (
                full_entries if is_full else list(range(layer + 1))
            ),
            "selected_down_projection_completed": is_full,
            "lm_head_completed": is_full,
            "logits_completed": is_full,
            "down_projection_materialized": is_full,
            "decoder_suffix_materialized": is_full and layer + 1 < len(full_entries),
            "logits_materialized": is_full,
        }
        records.append(record)
    calls = [
        {
            "call_index": index,
            "failed": False,
            "wall_seconds": 0.1,
            "metadata": {**record, "activation_shape": [1, 7, 11]},
            "cuda_memory": {},
        }
        for index, record in enumerate(records)
    ]
    return {
        "execution_records": {
            "stop_gradient_selected_attribution_forward": records,
        },
        "counters": {
            "stop_gradient_selected_attribution_forward_execution": execution,
            "stop_gradient_selected_attribution_forward_execution_count": len(records),
            f"stop_gradient_selected_attribution_{execution}_execution_count": len(
                records
            ),
            "stop_gradient_selected_attribution_down_projection_materialized_count": sum(
                int(record["down_projection_materialized"]) for record in records
            ),
            "stop_gradient_selected_attribution_decoder_suffix_materialized_count": sum(
                int(record["decoder_suffix_materialized"]) for record in records
            ),
            "stop_gradient_selected_attribution_logits_materialized_count": sum(
                int(record["logits_materialized"]) for record in records
            ),
            "stop_gradient_selected_attribution_decoder_layer_entry_count": sum(
                len(record["decoder_layer_entries"]) for record in records
            ),
            "stop_gradient_selected_attribution_selected_down_projection_completed_count": sum(
                int(record["selected_down_projection_completed"]) for record in records
            ),
            "stop_gradient_selected_attribution_lm_head_completed_count": sum(
                int(record["lm_head_completed"]) for record in records
            ),
            "stop_gradient_selected_attribution_logits_completed_count": sum(
                int(record["logits_completed"]) for record in records
            ),
        },
        "stages": {
            "stop_grad_selected_layer_forward": {
                "calls": len(calls),
                "failed_calls": 0,
                "call_measurements": calls,
            }
        },
    }


def _selected_attribution_forward_pair(tmp_path, *, reference=None, candidate=None):
    return _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_attribution_forward_execution="full_model_v1",
        candidate_selected_attribution_forward_execution="prefix_stop_v1",
        reference_selected_attribution_forward_instrumentation=(
            reference
            if reference is not None
            else _selected_attribution_forward_instrumentation("full_model_v1")
        ),
        candidate_selected_attribution_forward_instrumentation=(
            candidate
            if candidate is not None
            else _selected_attribution_forward_instrumentation("prefix_stop_v1")
        ),
    )


def test_selected_attribution_forward_ab_requires_ordered_execution_receipts(
    tmp_path,
) -> None:
    reference, candidate = _selected_attribution_forward_pair(tmp_path)

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        require_canonical_stop_gradient_selected_attribution_forward_ab=True,
    )

    assert report["qualification_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    contract = report["stop_gradient_selected_attribution_forward_ab_contract"]
    assert contract["passed"] is True
    assert contract["checks"] == {
        "canonical_identity_strategies": True,
        "reference_full_model_receipts": True,
        "candidate_prefix_stop_receipts": True,
        "cross_side_workload_equal": True,
    }
    assert contract["reference_runtime"]["full_decoder_layer_entries"] == list(range(5))
    assert contract["candidate_runtime"]["records"][1]["decoder_layer_entries"] == list(
        range(4)
    )


@pytest.mark.parametrize(
    "defect",
    [
        "missing_execution_records",
        "malformed_record",
        "reference_truncated_decoder_entries",
        "reference_lm_head_not_completed",
        "candidate_selected_down_projection_completed",
        "candidate_noncanonical_decoder_entries",
        "stage_receipt_mismatch",
        "aggregate_counter_mismatch",
        "cross_side_layer_mismatch",
    ],
)
def test_selected_attribution_forward_ab_fails_closed_on_receipt_defects(
    tmp_path,
    defect: str,
) -> None:
    reference_runtime = _selected_attribution_forward_instrumentation("full_model_v1")
    candidate_runtime = _selected_attribution_forward_instrumentation("prefix_stop_v1")
    reference_records = reference_runtime["execution_records"][
        "stop_gradient_selected_attribution_forward"
    ]
    candidate_records = candidate_runtime["execution_records"][
        "stop_gradient_selected_attribution_forward"
    ]
    if defect == "missing_execution_records":
        candidate_runtime.pop("execution_records")
    elif defect == "malformed_record":
        candidate_records[0] = None
    elif defect == "reference_truncated_decoder_entries":
        reference_records[0]["decoder_layer_entries"] = list(range(4))
        reference_runtime["stages"]["stop_grad_selected_layer_forward"][
            "call_measurements"
        ][0]["metadata"]["decoder_layer_entries"] = list(range(4))
    elif defect == "reference_lm_head_not_completed":
        reference_records[0]["lm_head_completed"] = False
        reference_runtime["stages"]["stop_grad_selected_layer_forward"][
            "call_measurements"
        ][0]["metadata"]["lm_head_completed"] = False
    elif defect == "candidate_selected_down_projection_completed":
        candidate_records[0]["selected_down_projection_completed"] = True
        candidate_runtime["stages"]["stop_grad_selected_layer_forward"][
            "call_measurements"
        ][0]["metadata"]["selected_down_projection_completed"] = True
    elif defect == "candidate_noncanonical_decoder_entries":
        candidate_records[1]["decoder_layer_entries"] = [0, 1, 3]
        candidate_runtime["stages"]["stop_grad_selected_layer_forward"][
            "call_measurements"
        ][1]["metadata"]["decoder_layer_entries"] = [0, 1, 3]
    elif defect == "stage_receipt_mismatch":
        candidate_runtime["stages"]["stop_grad_selected_layer_forward"][
            "call_measurements"
        ][0]["metadata"]["logits_completed"] = True
    elif defect == "aggregate_counter_mismatch":
        candidate_runtime["counters"][
            "stop_gradient_selected_attribution_forward_execution_count"
        ] = 1
    else:
        candidate_records[1]["layer"] = 2
        candidate_records[1]["decoder_layer_entries"] = [0, 1, 2]
        candidate_metadata = candidate_runtime["stages"][
            "stop_grad_selected_layer_forward"
        ]["call_measurements"][1]["metadata"]
        candidate_metadata["layer"] = 2
        candidate_metadata["decoder_layer_entries"] = [0, 1, 2]

    reference, candidate = _selected_attribution_forward_pair(
        tmp_path,
        reference=reference_runtime,
        candidate=candidate_runtime,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_IDENTITY_PATHS
        ),
        require_canonical_stop_gradient_selected_attribution_forward_ab=True,
    )

    assert report["validation_passed"] is False
    assert (
        report["stop_gradient_selected_attribution_forward_ab_contract"]["passed"]
        is False
    )


@pytest.mark.parametrize("candidate_execution", [_UNSET, "full_model_v1"])
def test_selected_attribution_forward_ab_rejects_missing_or_wrong_identity_strategy(
    tmp_path,
    candidate_execution: object,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_attribution_forward_execution="full_model_v1",
        candidate_selected_attribution_forward_execution=candidate_execution,
        reference_selected_attribution_forward_instrumentation=(
            _selected_attribution_forward_instrumentation("full_model_v1")
        ),
        candidate_selected_attribution_forward_instrumentation=(
            _selected_attribution_forward_instrumentation("prefix_stop_v1")
        ),
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_IDENTITY_PATHS
        ),
        require_canonical_stop_gradient_selected_attribution_forward_ab=True,
    )

    assert report["validation_passed"] is False
    contract = report["stop_gradient_selected_attribution_forward_ab_contract"]
    assert contract["checks"]["canonical_identity_strategies"] is False


def test_selected_attribution_forward_strategy_is_scalar_allowlist_only(
    tmp_path,
) -> None:
    reference, candidate = _selected_attribution_forward_pair(tmp_path)

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config."
                "stop_gradient_selected_attribution_forward_execution.*"
            ],
        )


def test_selected_attribution_forward_ab_cli_is_strict_and_non_overridable(
    tmp_path,
) -> None:
    args = build_parser().parse_args(
        [
            "--reference",
            str(tmp_path / "reference"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "report.json"),
            "--stop-gradient-selected-attribution-forward-ab",
        ]
    )

    options = comparison_options(args)
    assert options["allowed_identity_difference_paths"] == (
        STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_same_gpu_family"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert options["require_canonical_stop_gradient_selected_attribution_forward_ab"]
    assert options["tolerances"] == {
        group: NumericTolerance(absolute=0.0, relative=0.0)
        for group in TOLERANCE_GROUPS
    }

    args.allow_identity_difference = ["artifact_identity.code_revision.*"]
    with pytest.raises(ValueError, match="cannot be combined"):
        comparison_options(args)
    args.allow_identity_difference = []
    args.edge_atol = 1e-6
    with pytest.raises(ValueError, match="fixes every numerical tolerance at zero"):
        comparison_options(args)
    args.edge_atol = None
    args.cross_layer_jacobian_ab = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        comparison_options(args)


def _selected_attribution_storage_instrumentation(strategy: str) -> dict:
    graph_retaining = strategy == "graph_retaining_v1"
    chunk_specs = [(0, 0, 2, 0), (0, 2, 1, 1), (3, 0, 3, 0)]
    records = []
    calls = []
    for index, (layer, chunk_start, chunk_count, retained_before) in enumerate(
        chunk_specs
    ):
        record = {
            "layer": layer,
            "chunk_start": chunk_start,
            "strategy": strategy,
            "input_requires_grad": True,
            "input_grad_fn_retained": True,
            "stored_requires_grad": graph_retaining,
            "stored_grad_fn_retained": graph_retaining,
            "terminal_detached": not graph_retaining,
            "shares_projection_storage": True,
        }
        records.append(record)
        calls.append(
            {
                "call_index": index,
                "failed": False,
                "wall_seconds": 0.1,
                "metadata": {
                    "operation_kind": "vjp_projection",
                    "layer": layer,
                    "chunk_start": chunk_start,
                    "chunk_neuron_count": chunk_count,
                    "source_token_count": 3,
                    "raw_vjp_result_shape": [chunk_count, 1, 7, 11],
                    "projected_shape": [chunk_count, 1, 3],
                    "retained_chunk_count_before": retained_before,
                    "retained_chunk_count_after": retained_before + 1,
                    "selected_attribution_storage": strategy,
                    **{
                        field: value
                        for field, value in record.items()
                        if field
                        not in {
                            "layer",
                            "chunk_start",
                            "strategy",
                        }
                    },
                },
                "cuda_memory": {},
            }
        )
    return {
        "execution_records": {
            "stop_gradient_selected_attribution_storage": records,
        },
        "counters": {
            "stop_gradient_selected_attribution_storage": strategy,
            "stop_gradient_selected_attribution_storage_execution_count": len(records),
            f"stop_gradient_selected_attribution_{strategy}_storage_count": len(
                records
            ),
            "stop_gradient_selected_attribution_projection_graph_retained_count": len(
                records
            ),
            "stop_gradient_selected_attribution_stored_graph_retained_count": (
                len(records) if graph_retaining else 0
            ),
            "stop_gradient_selected_attribution_terminal_detached_count": (
                0 if graph_retaining else len(records)
            ),
        },
        "stages": {
            "stop_grad_selected_chunk_projection": {
                "calls": len(calls),
                "failed_calls": 0,
                "call_measurements": calls,
            }
        },
    }


def _selected_attribution_storage_pair(tmp_path, *, reference=None, candidate=None):
    return _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_attribution_storage="graph_retaining_v1",
        candidate_selected_attribution_storage="terminal_detached_v1",
        reference_selected_attribution_storage_instrumentation=(
            reference
            if reference is not None
            else _selected_attribution_storage_instrumentation("graph_retaining_v1")
        ),
        candidate_selected_attribution_storage_instrumentation=(
            candidate
            if candidate is not None
            else _selected_attribution_storage_instrumentation("terminal_detached_v1")
        ),
    )


def test_selected_attribution_storage_ab_requires_graph_lifetime_receipts(
    tmp_path,
) -> None:
    reference, candidate = _selected_attribution_storage_pair(tmp_path)

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        require_canonical_stop_gradient_selected_attribution_storage_ab=True,
    )

    assert report["qualification_passed"] is True
    assert report["schema_version"] == "bonafide-execution-qualification/v1"
    contract = report["stop_gradient_selected_attribution_storage_ab_contract"]
    assert contract["passed"] is True
    assert contract["checks"] == {
        "canonical_identity_strategies": True,
        "reference_graph_retaining_receipts": True,
        "candidate_terminal_detached_receipts": True,
        "cross_side_workload_equal": True,
    }
    assert contract["reference_runtime"]["coordinates"] == [
        [0, 0],
        [0, 2],
        [3, 0],
    ]


@pytest.mark.parametrize(
    "defect",
    [
        "missing_execution_records",
        "malformed_record",
        "input_graph_missing",
        "reference_stored_graph_missing",
        "candidate_stored_graph_retained",
        "candidate_not_terminal_detached",
        "storage_not_shared",
        "stage_receipt_mismatch",
        "aggregate_counter_mismatch",
        "noncanonical_order",
        "cross_side_workload_mismatch",
    ],
)
def test_selected_attribution_storage_ab_fails_closed_on_receipt_defects(
    tmp_path,
    defect: str,
) -> None:
    reference_runtime = _selected_attribution_storage_instrumentation(
        "graph_retaining_v1"
    )
    candidate_runtime = _selected_attribution_storage_instrumentation(
        "terminal_detached_v1"
    )
    reference_records = reference_runtime["execution_records"][
        "stop_gradient_selected_attribution_storage"
    ]
    candidate_records = candidate_runtime["execution_records"][
        "stop_gradient_selected_attribution_storage"
    ]
    reference_calls = reference_runtime["stages"][
        "stop_grad_selected_chunk_projection"
    ]["call_measurements"]
    candidate_calls = candidate_runtime["stages"][
        "stop_grad_selected_chunk_projection"
    ]["call_measurements"]
    if defect == "missing_execution_records":
        candidate_runtime.pop("execution_records")
    elif defect == "malformed_record":
        candidate_records[0] = None
    elif defect == "input_graph_missing":
        candidate_records[0]["input_grad_fn_retained"] = False
        candidate_calls[0]["metadata"]["input_grad_fn_retained"] = False
    elif defect == "reference_stored_graph_missing":
        reference_records[0]["stored_grad_fn_retained"] = False
        reference_calls[0]["metadata"]["stored_grad_fn_retained"] = False
    elif defect == "candidate_stored_graph_retained":
        candidate_records[0]["stored_requires_grad"] = True
        candidate_calls[0]["metadata"]["stored_requires_grad"] = True
    elif defect == "candidate_not_terminal_detached":
        candidate_records[0]["terminal_detached"] = False
        candidate_calls[0]["metadata"]["terminal_detached"] = False
    elif defect == "storage_not_shared":
        candidate_records[0]["shares_projection_storage"] = False
        candidate_calls[0]["metadata"]["shares_projection_storage"] = False
    elif defect == "stage_receipt_mismatch":
        candidate_calls[0]["metadata"]["stored_grad_fn_retained"] = True
    elif defect == "aggregate_counter_mismatch":
        candidate_runtime["counters"][
            "stop_gradient_selected_attribution_storage_execution_count"
        ] = 2
    elif defect == "noncanonical_order":
        candidate_records[0], candidate_records[1] = (
            candidate_records[1],
            candidate_records[0],
        )
        candidate_calls[0]["metadata"], candidate_calls[1]["metadata"] = (
            candidate_calls[1]["metadata"],
            candidate_calls[0]["metadata"],
        )
    else:
        candidate_calls[0]["metadata"]["source_token_count"] = 4
        candidate_calls[0]["metadata"]["projected_shape"] = [2, 1, 4]

    reference, candidate = _selected_attribution_storage_pair(
        tmp_path,
        reference=reference_runtime,
        candidate=candidate_runtime,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_IDENTITY_PATHS
        ),
        require_canonical_stop_gradient_selected_attribution_storage_ab=True,
    )

    assert report["validation_passed"] is False
    assert (
        report["stop_gradient_selected_attribution_storage_ab_contract"]["passed"]
        is False
    )


@pytest.mark.parametrize("candidate_strategy", [_UNSET, "graph_retaining_v1"])
def test_selected_attribution_storage_ab_rejects_missing_or_wrong_identity_strategy(
    tmp_path,
    candidate_strategy: object,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_attribution_storage="graph_retaining_v1",
        candidate_selected_attribution_storage=candidate_strategy,
        reference_selected_attribution_storage_instrumentation=(
            _selected_attribution_storage_instrumentation("graph_retaining_v1")
        ),
        candidate_selected_attribution_storage_instrumentation=(
            _selected_attribution_storage_instrumentation("terminal_detached_v1")
        ),
    )

    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_IDENTITY_PATHS
        ),
        require_canonical_stop_gradient_selected_attribution_storage_ab=True,
    )

    assert report["validation_passed"] is False
    contract = report["stop_gradient_selected_attribution_storage_ab_contract"]
    assert contract["checks"]["canonical_identity_strategies"] is False


def test_selected_attribution_storage_strategy_is_scalar_allowlist_only(
    tmp_path,
) -> None:
    reference, candidate = _selected_attribution_storage_pair(tmp_path)

    with pytest.raises(ValueError, match="exact path"):
        compare_execution_artifacts(
            reference,
            candidate,
            allowed_identity_difference_paths=[
                "artifact_identity.adag_config."
                "stop_gradient_selected_attribution_storage.*"
            ],
        )


def test_selected_attribution_storage_ab_cli_is_strict_and_non_overridable(
    tmp_path,
) -> None:
    args = build_parser().parse_args(
        [
            "--reference",
            str(tmp_path / "reference"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "report.json"),
            "--stop-gradient-selected-attribution-storage-ab",
        ]
    )

    options = comparison_options(args)
    assert options["allowed_identity_difference_paths"] == (
        STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_same_gpu_family"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert options["require_canonical_stop_gradient_selected_attribution_storage_ab"]
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
    args.stop_gradient_selected_attribution_forward_ab = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        comparison_options(args)


def _selected_target_logit_instrumentation(
    strategy: str,
    *,
    execution_indexes: tuple[int | None, ...] = (None,),
) -> dict:
    full = strategy == "full_logits_v1"
    head_positions = 9 if full else 3
    records = [
        {
            "execution": strategy,
            "execution_index": execution_index,
            "batch_size": 1,
            "sequence_position_count": 9,
            "selected_position_count": 3,
            "unique_selected_position_count": 2,
            "vocab_size": 11,
            "lm_head_input_shape": [1, head_positions, 5],
            "lm_head_output_shape": [1, head_positions, 11],
            "selected_position_logit_shape": [1, 3, 11],
            "target_logit_shape": [3, 1],
            "causal_lm_forward_completed": True,
            "selected_position_request_forwarded": not full,
            "full_sequence_logits_materialized": full,
            "selected_position_logits_materialized": True,
            "center_logits": False,
        }
        for execution_index in execution_indexes
    ]
    execution_count = len(records)
    return {
        "execution_records": {"selected_target_logit_execution": records},
        "counters": {
            "selected_target_logit_execution": strategy,
            "selected_target_logit_execution_count": execution_count,
            f"selected_target_logit_{strategy}_execution_count": execution_count,
            "selected_target_logit_full_sequence_logits_materialized_count": (
                execution_count if full else 0
            ),
            "selected_target_logit_selected_position_logits_materialized_count": (
                execution_count
            ),
            "selected_target_logit_lm_head_position_rows": (
                head_positions * execution_count
            ),
        },
    }


def _selected_target_logit_pair(
    tmp_path,
    *,
    candidate_runtime=None,
    ig_steps: object = None,
    reference_runtime=None,
):
    expected_indexes = (
        tuple(range(ig_steps + 1))
        if type(ig_steps) is int and ig_steps > 0
        else (None,)
    )
    return _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_target_logit_execution="full_logits_v1",
        candidate_selected_target_logit_execution="selected_position_logits_v1",
        reference_selected_target_logit_ig_steps=ig_steps,
        candidate_selected_target_logit_ig_steps=ig_steps,
        reference_selected_target_logit_instrumentation=(
            reference_runtime
            if reference_runtime is not None
            else _selected_target_logit_instrumentation(
                "full_logits_v1",
                execution_indexes=expected_indexes,
            )
        ),
        candidate_selected_target_logit_instrumentation=(
            candidate_runtime
            if candidate_runtime is not None
            else _selected_target_logit_instrumentation(
                "selected_position_logits_v1",
                execution_indexes=expected_indexes,
            )
        ),
    )


def test_selected_target_logit_execution_ab_requires_exact_row_receipts(
    tmp_path,
) -> None:
    reference, candidate = _selected_target_logit_pair(tmp_path)
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_TARGET_LOGIT_EXECUTION_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        require_canonical_selected_target_logit_execution_ab=True,
    )

    assert report["qualification_passed"] is True
    contract = report["selected_target_logit_execution_ab_contract"]
    assert contract["passed"] is True
    assert contract["checks"]["aggregate_lm_head_row_reduction"] is True
    assert contract["reference_runtime"]["lm_head_position_rows"] == 9
    assert contract["candidate_runtime"]["lm_head_position_rows"] == 3


@pytest.mark.parametrize(
    "defect",
    ["full_materialized", "head_rows", "workload", "counter", "no_row_reduction"],
)
def test_selected_target_logit_execution_ab_fails_closed_on_receipt_defects(
    tmp_path,
    defect: str,
) -> None:
    runtime = _selected_target_logit_instrumentation("selected_position_logits_v1")
    record = runtime["execution_records"]["selected_target_logit_execution"][0]
    if defect == "full_materialized":
        record["full_sequence_logits_materialized"] = True
    elif defect == "head_rows":
        record["lm_head_input_shape"] = [1, 9, 5]
        record["lm_head_output_shape"] = [1, 9, 11]
    elif defect == "workload":
        record["selected_position_count"] = 2
    elif defect == "counter":
        runtime["counters"]["selected_target_logit_lm_head_position_rows"] = 9
    else:
        record["selected_position_count"] = 9
        record["lm_head_input_shape"] = [1, 9, 5]
        record["lm_head_output_shape"] = [1, 9, 11]
        record["selected_position_logit_shape"] = [1, 9, 11]
        record["target_logit_shape"] = [9, 1]
        runtime["counters"]["selected_target_logit_lm_head_position_rows"] = 9
    reference, candidate = _selected_target_logit_pair(
        tmp_path, candidate_runtime=runtime
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_TARGET_LOGIT_EXECUTION_AB_IDENTITY_PATHS
        ),
        require_canonical_selected_target_logit_execution_ab=True,
    )
    assert report["validation_passed"] is False
    contract = report["selected_target_logit_execution_ab_contract"]
    assert contract["passed"] is False
    if defect in {"counter", "no_row_reduction"}:
        assert contract["checks"]["aggregate_lm_head_row_reduction"] is False


def test_selected_target_logit_execution_ab_rejects_extra_null_ig_record(
    tmp_path,
) -> None:
    candidate_runtime = _selected_target_logit_instrumentation(
        "selected_position_logits_v1",
        execution_indexes=(None, None),
    )
    reference, candidate = _selected_target_logit_pair(
        tmp_path,
        candidate_runtime=candidate_runtime,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_TARGET_LOGIT_EXECUTION_AB_IDENTITY_PATHS
        ),
        require_canonical_selected_target_logit_execution_ab=True,
    )
    contract = report["selected_target_logit_execution_ab_contract"]
    assert report["validation_passed"] is False
    assert contract["candidate_runtime"]["expected_execution_indexes"] == [None]
    assert contract["candidate_runtime"]["observed_execution_indexes"] == [None, None]


@pytest.mark.parametrize("execution_indexes", [(0, 0, 2), (0, 1)])
def test_selected_target_logit_execution_ab_rejects_wrong_ig_indexes_or_count(
    tmp_path,
    execution_indexes: tuple[int, ...],
) -> None:
    candidate_runtime = _selected_target_logit_instrumentation(
        "selected_position_logits_v1",
        execution_indexes=execution_indexes,
    )
    reference, candidate = _selected_target_logit_pair(
        tmp_path,
        ig_steps=2,
        candidate_runtime=candidate_runtime,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_TARGET_LOGIT_EXECUTION_AB_IDENTITY_PATHS
        ),
        require_canonical_selected_target_logit_execution_ab=True,
    )
    contract = report["selected_target_logit_execution_ab_contract"]
    assert report["validation_passed"] is False
    assert contract["candidate_runtime"]["expected_execution_indexes"] == [0, 1, 2]
    assert contract["candidate_runtime"]["observed_execution_indexes"] == list(
        execution_indexes
    )


@pytest.mark.parametrize("ig_steps", [_OMIT, 0, True, "2"])
def test_selected_target_logit_execution_ab_rejects_missing_or_malformed_ig_steps(
    tmp_path,
    ig_steps: object,
) -> None:
    reference, candidate = _selected_target_logit_pair(
        tmp_path,
        ig_steps=ig_steps,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_TARGET_LOGIT_EXECUTION_AB_IDENTITY_PATHS
        ),
        require_canonical_selected_target_logit_execution_ab=True,
    )
    contract = report["selected_target_logit_execution_ab_contract"]
    assert report["validation_passed"] is False
    assert contract["reference_runtime"]["config_valid"] is False
    assert contract["candidate_runtime"]["config_valid"] is False


def test_selected_target_logit_execution_ab_cli_is_strict_and_exact(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--reference",
            str(tmp_path / "reference"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "report.json"),
            "--selected-target-logit-execution-ab",
        ]
    )
    options = comparison_options(args)
    assert options["allowed_identity_difference_paths"] == (
        SELECTED_TARGET_LOGIT_EXECUTION_AB_IDENTITY_PATHS
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True
    assert options["require_canonical_selected_target_logit_execution_ab"] is True
    assert options["tolerances"] == {
        group: NumericTolerance(absolute=0.0, relative=0.0)
        for group in TOLERANCE_GROUPS
    }


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


def _stop_gradient_embed_receipt(character: str, width: int | None) -> list[dict]:
    records = _selected_embed_receipt(character, width)
    record = records[0]
    for field in ("execution_index", "receipt_mode", "return_gradient_only"):
        del record[field]
    del record["retain_graph"]
    record["retain_graph_after_execution"] = False
    return records


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
    ("profile", "reference_width", "candidate_width"),
    [("full_width_exact_v1", None, 5), ("width_one_exact_v1", 5, 1)],
)
def test_stop_gradient_embed_profiles_are_exact_and_namespace_bound(
    tmp_path, profile: str, reference_width: int | None, candidate_width: int
) -> None:
    receipts = _stop_gradient_embed_receipt("1", reference_width)
    candidate_receipts = _stop_gradient_embed_receipt("1", candidate_width)
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_stop_gradient_embed_chunk_size=reference_width,
        candidate_stop_gradient_embed_chunk_size=candidate_width,
        reference_stop_gradient_embed_receipts=receipts,
        candidate_stop_gradient_embed_receipts=candidate_receipts,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        stop_gradient_embed_contribution_target_lane_chunk_ab_profile=(profile),
    )

    assert report["qualification_passed"] is True
    contract = report["stop_gradient_embed_contribution_target_lane_chunk_ab_contract"]
    assert contract["passed"] is True
    assert contract["projected_receipts"]["execution_record_namespace"] == (
        "stop_gradient_embed_contribution_vjp"
    )
    assert contract["projected_receipts"]["execution_contract"] == (
        "stop_gradient_direct_v1"
    )
    assert contract["projected_receipts"]["checks"]["receipt_hashes_exact"] is True


@pytest.mark.parametrize(
    "defect", ["wrong_namespace", "hash_mismatch", "retains_final_graph"]
)
def test_stop_gradient_embed_full_width_profile_fails_closed_on_receipts(
    tmp_path, defect: str
) -> None:
    reference_receipts = _stop_gradient_embed_receipt("1", None)
    candidate_receipts = _stop_gradient_embed_receipt(
        "a" if defect == "hash_mismatch" else "1", 5
    )
    pair_args = {
        "reference_stop_gradient_embed_chunk_size": None,
        "candidate_stop_gradient_embed_chunk_size": 5,
        "reference_stop_gradient_embed_receipts": reference_receipts,
        "candidate_stop_gradient_embed_receipts": candidate_receipts,
    }
    if defect == "wrong_namespace":
        pair_args.pop("candidate_stop_gradient_embed_receipts")
        pair_args["candidate_selected_embed_receipts"] = candidate_receipts
    elif defect == "retains_final_graph":
        candidate_receipts[0]["retain_graph_after_execution"] = True
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        **pair_args,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        stop_gradient_embed_contribution_target_lane_chunk_ab_profile=(
            "full_width_exact_v1"
        ),
    )

    assert report["validation_passed"] is False
    contract = report["stop_gradient_embed_contribution_target_lane_chunk_ab_contract"]
    assert contract["projected_receipts"]["passed"] is False


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
        ("candidate_non_logit_attr_map_delta", 0.01),
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
            "node": NumericTolerance(absolute=0.125, relative=1e-2),
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
    elif mutation == "candidate_non_logit_attr_map_delta":
        assert scope["checks"]["non_logit_source_attribution_profiles_exact"] is False
    elif mutation == "candidate_unaffected_edge_delta":
        assert scope["checks"]["unaffected_edges_exact"] is False
    else:
        assert scope["checks"]["exact_bf16_dtype_identity"] is False


def test_selected_embed_width_one_allows_only_logit_attr_map_bf16_drift(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_embed_chunk_size=5,
        candidate_selected_embed_chunk_size=1,
        reference_selected_embed_receipts=_selected_embed_receipt("1", 5),
        candidate_selected_embed_receipts=_selected_embed_receipt("a", 1),
        candidate_logit_attr_map_delta=0.01,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        tolerances={
            "target": NumericTolerance(absolute=0.0, relative=0.0),
            "node": NumericTolerance(absolute=0.125, relative=1e-2),
            "edge": NumericTolerance(absolute=5e-4, relative=1e-2),
            "candidate_profile": NumericTolerance(absolute=0.125, relative=1e-2),
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        selected_embed_contribution_target_lane_chunk_ab_profile="width_one_bf16_v1",
    )

    assert report["validation_passed"] is True
    scope = report["selected_embed_contribution_target_lane_chunk_ab_contract"][
        "bf16_scope"
    ]
    assert scope["passed"] is True
    assert scope["logit_layer"] == 1
    assert scope["source_attribution_profiles"]["target_logit"][
        "max_absolute_error"
    ] == pytest.approx(0.01)


def test_selected_embed_width_one_rejects_unclassified_max_layer_nodes(
    tmp_path,
) -> None:
    reference, candidate = _save_pair(
        tmp_path,
        reference_backend="eager",
        candidate_backend="eager",
        reference_selected_embed_chunk_size=5,
        candidate_selected_embed_chunk_size=1,
        reference_selected_embed_receipts=_selected_embed_receipt("1", 5),
        candidate_selected_embed_receipts=_selected_embed_receipt("a", 1),
        extra_logit_node=True,
    )
    report = compare_execution_artifacts(
        reference,
        candidate,
        allowed_identity_difference_paths=(
            SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
        ),
        tolerances={
            "target": NumericTolerance(absolute=0.0, relative=0.0),
            "node": NumericTolerance(absolute=0.125, relative=1e-2),
            "edge": NumericTolerance(absolute=5e-4, relative=1e-2),
            "candidate_profile": NumericTolerance(absolute=0.125, relative=1e-2),
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
        selected_embed_contribution_target_lane_chunk_ab_profile="width_one_bf16_v1",
    )

    scope = report["selected_embed_contribution_target_lane_chunk_ab_contract"][
        "bf16_scope"
    ]
    assert report["validation_passed"] is False
    assert scope["checks"]["logit_node_scope_classified"] is False
    assert (
        scope["source_attribution_profiles"]["non_logit"]["max_absolute_error"] == 0.0
    )


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
                "node": NumericTolerance(absolute=0.125, relative=1e-2),
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


def test_stop_gradient_embed_full_width_cli_profile_is_canonical(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--reference",
            str(tmp_path / "reference"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "report.json"),
            "--stop-gradient-embed-contribution-target-lane-full-width-ab",
        ]
    )
    options = comparison_options(args)
    assert options["allowed_identity_difference_paths"] == (
        STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
    )
    assert (
        options["stop_gradient_embed_contribution_target_lane_chunk_ab_profile"]
        == "full_width_exact_v1"
    )
    assert options["selected_embed_contribution_target_lane_chunk_ab_profile"] is None
    assert all(
        tolerance == NumericTolerance(absolute=0.0, relative=0.0)
        for tolerance in options["tolerances"].values()
    )
    assert options["require_same_gpu_model"] is True
    assert options["require_exact_node_topology"] is True
    assert options["require_exact_edge_topology"] is True


def test_stop_gradient_embed_width_one_exact_cli_profile_is_canonical(
    tmp_path,
) -> None:
    args = build_parser().parse_args(
        [
            "--reference",
            str(tmp_path / "reference"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "report.json"),
            "--stop-gradient-embed-contribution-target-lane-width-one-exact-ab",
        ]
    )
    options = comparison_options(args)
    assert options["allowed_identity_difference_paths"] == (
        STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_AB_IDENTITY_PATHS
    )
    assert (
        options["stop_gradient_embed_contribution_target_lane_chunk_ab_profile"]
        == "width_one_exact_v1"
    )
    assert all(
        tolerance == NumericTolerance(absolute=0.0, relative=0.0)
        for tolerance in options["tolerances"].values()
    )
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


def test_execution_qualification_allows_only_snapshot_telemetry_and_code_revision(
    tmp_path,
) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_manifest = _manifest("eager", code_revision="reference-commit")
    candidate_manifest = _manifest("eager", code_revision="candidate-commit")
    reference_manifest["artifact_identity"]["instrumentation"] = {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": False,
    }
    candidate_manifest["artifact_identity"]["instrumentation"] = {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": True,
    }
    reference_path = tmp_path / "snapshot-telemetry-disabled"
    candidate_path = tmp_path / "snapshot-telemetry-enabled"
    save_topk_compact_trace(
        reference_path,
        reference,
        manifest=_rehash_manifest(reference_manifest),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        manifest=_rehash_manifest(candidate_manifest),
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        allowed_identity_difference_paths=(
            CUDA_ALLOCATOR_SNAPSHOT_TELEMETRY_IDENTITY_PATH,
            "artifact_identity.code_revision.*",
        ),
        tolerances={
            group: NumericTolerance(absolute=0.0, relative=0.0)
            for group in TOLERANCE_GROUPS
        },
        require_same_gpu_model=True,
        require_exact_node_topology=True,
        require_exact_edge_topology=True,
    )

    assert report["validation_passed"] is True
    assert {gate["gate"] for gate in report["gates"] if gate["required"]} >= {
        "same_gpu_model",
        "exact_node_topology",
        "exact_edge_topology",
        *(f"{group}_numeric_tolerance" for group in TOLERANCE_GROUPS),
    }
    allowed = report["identity"]["artifact_identity"]["allowed_differences"]
    assert {difference["path"] for difference in allowed} == {
        CUDA_ALLOCATOR_SNAPSHOT_TELEMETRY_IDENTITY_PATH,
        "artifact_identity.code_revision.git_commit",
    }


def test_execution_qualification_rejects_arbitrary_instrumentation_difference(
    tmp_path,
) -> None:
    reference = _topk_trace()
    candidate = deepcopy(reference)
    _complete_trace_metadata(reference)
    _complete_trace_metadata(candidate)
    reference_manifest = _manifest("eager", code_revision="same-commit")
    candidate_manifest = _manifest("eager", code_revision="same-commit")
    reference_manifest["artifact_identity"]["instrumentation"] = {
        "cuda_memory_telemetry": False,
        "cuda_allocator_snapshot_telemetry": False,
    }
    candidate_manifest["artifact_identity"]["instrumentation"] = {
        "cuda_memory_telemetry": True,
        "cuda_allocator_snapshot_telemetry": True,
    }
    reference_path = tmp_path / "instrumentation-reference"
    candidate_path = tmp_path / "instrumentation-candidate"
    save_topk_compact_trace(
        reference_path,
        reference,
        manifest=_rehash_manifest(reference_manifest),
    )
    save_topk_compact_trace(
        candidate_path,
        candidate,
        manifest=_rehash_manifest(candidate_manifest),
    )

    report = compare_execution_artifacts(
        reference_path,
        candidate_path,
        allowed_identity_difference_paths=(
            CUDA_ALLOCATOR_SNAPSHOT_TELEMETRY_IDENTITY_PATH,
        ),
    )

    assert report["validation_passed"] is False
    disallowed = report["identity"]["artifact_identity"]["unallowed_differences"]
    assert {difference["path"] for difference in disallowed} == {
        "artifact_identity.instrumentation.cuda_memory_telemetry"
    }
    with pytest.raises(ValueError, match="may only allow"):
        compare_execution_artifacts(
            reference_path,
            candidate_path,
            allowed_identity_difference_paths=["artifact_identity.instrumentation.*"],
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
