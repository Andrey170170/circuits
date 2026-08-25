from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER = (
    ROOT / "scripts/bonafide/process_witness_attention_backend_qualification_v1.sbatch"
)
CONFIG_ROOT = ROOT / "scripts/bonafide/configs"
BACKENDS = (
    "legacy_eager_unmasked_v1",
    "eager_causal_v1",
    "flash_sdpa_causal_v1",
)
EMBEDDING_EDGE_MATERIALIZATIONS = ("scalar_v1", "vectorized_v1")
CROSS_LAYER_JACOBIAN_EXECUTIONS = ("full_model_v1", "cached_range_v1")
CONTRIBUTION_EXECUTIONS = (
    "full_graph_v1",
    "source_leaf_v1",
    "sparse_source_leaf_v1",
)


def _config(backend: str) -> dict:
    path = (
        CONFIG_ROOT
        / f"qwen3_4b_thinking_attention_backend_qualification_{backend}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_launcher_has_valid_bash_syntax_and_bounded_a100_resources() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    launcher = LAUNCHER.read_text(encoding="utf-8")

    required = {
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --time=02:00:00",
        "#SBATCH --clusters=notchpeak",
        "#SBATCH --partition=marasovic-gpu-np",
        "#SBATCH --qos=marasovic-gpu-np",
        "#SBATCH --account=marasovic-gpu-np",
        "#SBATCH --gres=gpu:a100:1",
        "#SBATCH --signal=B:USR1@300",
        "#SBATCH --no-requeue",
    }
    assert required <= set(launcher.splitlines())


def test_launcher_is_single_item_fail_closed_and_provenance_bound() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "EXPECTED_GIT_COMMIT",
        "CONFIG_FILE_SHA256",
        "MANIFEST_FILE_SHA256",
        "WIDTH1_SOURCE_MANIFEST_FILE_SHA256",
        "EXPECTED_ATTENTION_BACKEND",
        "EXPECTED_STOP_GRADIENT_CONTRIBUTION_EXECUTION",
        "EXPECTED_STOP_GRADIENT_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE",
        "EXPECTED_SELECTED_NEURON_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE",
        "EXPECTED_CUDA_ALLOCATOR_POLICY",
        "EXPECTED_EMBEDDING_EDGE_MATERIALIZATION",
        "EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION",
        "unset PYTORCH_CUDA_ALLOC_CONF",
        'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
        "run config allocator policy disagrees",
        "run config contribution execution disagrees",
        "run config target-lane chunk size disagrees",
        "run config selected-neuron target-lane chunk size disagrees",
        "saved artifact allocator identity disagrees",
        "saved artifact contribution execution identity disagrees",
        "saved artifact target-lane chunk size identity disagrees",
        "saved artifact selected-neuron target-lane chunk size identity disagrees",
        "chunk_size_field not in artifact_adag",
        "saved artifact lacks the exact requested allocator runtime receipt",
        "run config embedding-edge materialization disagrees",
        "saved artifact embedding-edge materialization identity disagrees",
        "run config cross-layer Jacobian execution disagrees",
        "saved artifact cross-layer Jacobian execution identity disagrees",
        "validate_cuda_allocator_environment(config)",
        '"observed_allocator_backend": "native"',
        "EXPECTED_GPU_NAME",
        "ONLY_ARTIFACT_ID",
        '--only-artifact-id "$ONLY_ARTIFACT_ID"',
        "process_witness_resource_calibration_v1",
        "Refusing existing qualification output or summary target",
        "qualification summary must contain exactly one record",
        'row.get("status") != "complete"',
        "Forwarding Slurm USR1 warning",
        "min_cuda_headroom_bytes",
        "max_trace_seconds",
    ):
        assert required in launcher
    assert "skipped_complete" not in launcher
    for execution in CONTRIBUTION_EXECUTIONS:
        assert execution in launcher
    assert '!= "none"' in launcher
    assert "^[1-9][0-9]*$" in launcher


def test_embedding_edge_materialization_configs_clone_allocator_default() -> None:
    allocator_default = json.loads(
        (
            CONFIG_ROOT / "qwen3_4b_thinking_allocator_qualification_default_v1.json"
        ).read_text(encoding="utf-8")
    )
    materialization_configs = []
    for materialization in EMBEDDING_EDGE_MATERIALIZATIONS:
        path = (
            CONFIG_ROOT
            / "qwen3_4b_thinking_embedding_edge_materialization_qualification_"
            f"{materialization}.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert (
            config["adag_config"]["embedding_edge_materialization"] == materialization
        )
        normalized = json.loads(json.dumps(config))
        del normalized["adag_config"]["embedding_edge_materialization"]
        assert normalized == allocator_default
        materialization_configs.append(config)

    scalar = json.loads(json.dumps(materialization_configs[0]))
    vectorized = json.loads(json.dumps(materialization_configs[1]))
    del scalar["adag_config"]["embedding_edge_materialization"]
    del vectorized["adag_config"]["embedding_edge_materialization"]
    assert scalar == vectorized


def test_cross_layer_jacobian_configs_clone_accepted_vectorized_config() -> None:
    vectorized = json.loads(
        (
            CONFIG_ROOT
            / "qwen3_4b_thinking_embedding_edge_materialization_qualification_"
            "vectorized_v1.json"
        ).read_text(encoding="utf-8")
    )
    execution_configs = []
    for execution in CROSS_LAYER_JACOBIAN_EXECUTIONS:
        path = (
            CONFIG_ROOT / "qwen3_4b_thinking_cross_layer_jacobian_qualification_"
            f"{execution}.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["adag_config"]["cross_layer_jacobian_execution"] == execution
        assert config["cuda_allocator_policy"] == "default_v1"
        assert (
            config["adag_config"]["stop_gradient_attention_backend"]
            == "flash_sdpa_causal_v1"
        )
        assert (
            config["adag_config"]["stop_gradient_contribution_execution"]
            == "source_leaf_v1"
        )
        assert (
            config["adag_config"]["embedding_edge_materialization"] == "vectorized_v1"
        )
        normalized = json.loads(json.dumps(config))
        del normalized["adag_config"]["cross_layer_jacobian_execution"]
        assert normalized == vectorized
        execution_configs.append(config)

    full_model = json.loads(json.dumps(execution_configs[0]))
    cached_range = json.loads(json.dumps(execution_configs[1]))
    del full_model["adag_config"]["cross_layer_jacobian_execution"]
    del cached_range["adag_config"]["cross_layer_jacobian_execution"]
    assert full_model == cached_range


def test_sparse_contribution_config_clones_cached_range_control_exactly() -> None:
    control = json.loads(
        (
            CONFIG_ROOT / "qwen3_4b_thinking_cross_layer_jacobian_qualification_"
            "cached_range_v1.json"
        ).read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (
            CONFIG_ROOT / "qwen3_4b_thinking_stop_gradient_contribution_qualification_"
            "sparse_source_leaf_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        control["adag_config"]["stop_gradient_contribution_execution"]
        == "source_leaf_v1"
    )
    assert (
        candidate["adag_config"]["stop_gradient_contribution_execution"]
        == "sparse_source_leaf_v1"
    )
    normalized_control = json.loads(json.dumps(control))
    normalized_candidate = json.loads(json.dumps(candidate))
    del normalized_control["adag_config"]["stop_gradient_contribution_execution"]
    del normalized_candidate["adag_config"]["stop_gradient_contribution_execution"]
    assert normalized_candidate == normalized_control


def test_target_lane_chunk_configs_differ_only_by_explicit_chunk_size() -> None:
    configs = []
    for suffix, expected_chunk_size in (("none", None), ("1", 1)):
        path = (
            CONFIG_ROOT
            / "qwen3_4b_thinking_stop_gradient_target_lane_chunk_qualification_"
            f"{suffix}_v1.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert (
            config["adag_config"]["stop_gradient_contribution_execution"]
            == "source_leaf_v1"
        )
        assert (
            config["adag_config"]["stop_gradient_contribution_target_lane_chunk_size"]
            == expected_chunk_size
        )
        assert (
            config["artifact_root"] == "results/bonafide/"
            "process-witness-target-lane-chunk-qualification-v1"
        )
        configs.append(config)

    normalized = []
    for config in configs:
        copied = json.loads(json.dumps(config))
        del copied["adag_config"]["stop_gradient_contribution_target_lane_chunk_size"]
        normalized.append(copied)
    assert normalized[0] == normalized[1]


def test_selected_neuron_target_lane_chunk_configs_differ_only_by_explicit_chunk_size() -> (
    None
):
    configs = []
    field = "selected_neuron_contribution_target_lane_chunk_size"
    for suffix, expected_chunk_size in (("none", None), ("1", 1)):
        path = (
            CONFIG_ROOT
            / "qwen3_4b_thinking_selected_neuron_target_lane_chunk_qualification_"
            f"{suffix}_v1.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["adag_config"][field] == expected_chunk_size
        assert (
            config["adag_config"]["stop_gradient_contribution_target_lane_chunk_size"]
            == 1
        )
        assert (
            config["artifact_root"] == "results/bonafide/"
            "process-witness-selected-neuron-target-lane-chunk-qualification-v1"
        )
        configs.append(config)

    normalized = []
    for config in configs:
        copied = json.loads(json.dumps(config))
        del copied["adag_config"][field]
        normalized.append(copied)
    assert normalized[0] == normalized[1]


def test_backend_configs_clone_frozen_v4_except_for_explicit_backend() -> None:
    configs = [_config(backend) for backend in BACKENDS]
    normalized = []
    for backend, config in zip(BACKENDS, configs, strict=True):
        assert config["adag_config"]["stop_gradient_attention_backend"] == backend
        assert config["trace_warmup"] == {
            "enabled": False,
            "mode": "first_wave_item_full_trace_discard",
            "wave_id_prefixes": [],
        }
        assert config["wave_limits"] == {
            "max_trace_seconds": 1800,
            "min_cuda_headroom_bytes": 8589934592,
            "stop_on_oom": True,
        }
        copied = json.loads(json.dumps(config))
        del copied["adag_config"]["stop_gradient_attention_backend"]
        normalized.append(copied)

    assert normalized[1:] == normalized[:-1]
    assert normalized[0] == {
        "adag_config": {
            "ablation_mode": "zero",
            "apply_blacklist": True,
            "batch_aggregation": "any",
            "center_logits": False,
            "disable_half_rule": False,
            "disable_stop_grad": False,
            "edge_threshold": 0.01,
            "focus_last_residual": False,
            "ig_mode": "ig-inputs",
            "ig_steps": None,
            "node_attribution_threshold": None,
            "parent_threshold": None,
            "percentage_threshold": 0.005,
            "return_nodes_only": False,
            "return_only_important_neurons": False,
            "skip_attr_contrib": False,
            "topk": None,
            "topk_neurons": None,
            "use_relp_grad": True,
            "use_stop_grad_on_mlps": True,
            "verbose": False,
        },
        "artifact_root": "results/bonafide/process-witness-resource-calibration-v1",
        "batch_size": 1,
        "continue_on_error": False,
        "model": {
            "device": "cuda:0",
            "dtype": "bfloat16",
            "from_pretrained_kwargs": {},
            "local_files_only": True,
            "local_snapshot_path": "${HF_HUB_CACHE}/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea",
            "model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "revision": "768f209d9ea81521153ed38c47d515654e938aea",
        },
        "schema_version": "bonafide-trace-run-config/v1",
        "trace_warmup": {
            "enabled": False,
            "mode": "first_wave_item_full_trace_discard",
            "wave_id_prefixes": [],
        },
        "wave_limits": {
            "max_trace_seconds": 1800,
            "min_cuda_headroom_bytes": 8589934592,
            "stop_on_oom": True,
        },
    }
