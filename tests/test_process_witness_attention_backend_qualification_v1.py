from __future__ import annotations

import json
import subprocess
import sys
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


def _selected_target_logit_record(
    *,
    sequence_count: int = 4,
    selected_count: int = 1,
) -> dict:
    batch_size = 2
    vocab_size = 7
    return {
        "execution": "selected_position_logits_v1",
        "execution_index": None,
        "batch_size": batch_size,
        "sequence_position_count": sequence_count,
        "selected_position_count": selected_count,
        "unique_selected_position_count": min(selected_count, sequence_count),
        "vocab_size": vocab_size,
        "lm_head_input_shape": [batch_size, selected_count, 3],
        "lm_head_output_shape": [batch_size, selected_count, vocab_size],
        "selected_position_logit_shape": [batch_size, selected_count, vocab_size],
        "target_logit_shape": [selected_count, batch_size],
        "causal_lm_forward_completed": True,
        "selected_position_request_forwarded": True,
        "full_sequence_logits_materialized": False,
        "selected_position_logits_materialized": True,
        "center_logits": False,
    }


def _run_launcher_postflight(
    tmp_path: Path,
    record: dict,
    *,
    lm_head_position_rows: int | None = None,
) -> subprocess.CompletedProcess[str]:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    start_marker = "    'import json,pathlib,sys\n"
    start = launcher.rindex(start_marker) + len("    '")
    end = launcher.index('\' \\\n    "$SUMMARY_JSONL"', start)
    postflight = launcher[start:end]

    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    runtime_environment = {
        "gpu_runtime": {
            "devices": [{"name": "NVIDIA A100 80GB PCIe"}],
            "driver_versions": {"cuda_driver": "qualification-test"},
        }
    }
    strategy = "selected_position_logits_v1"
    adag_config = {
        "stop_gradient_attention_backend": "flash_sdpa_causal_v1",
        "stop_gradient_contribution_execution": "source_leaf_v1",
        "stop_gradient_contribution_target_lane_chunk_size": None,
        "selected_target_logit_execution": strategy,
        "center_logits": False,
        "ig_steps": None,
    }
    manifest = {
        "artifact_identity": {
            "adag_config": adag_config,
            "runtime_environment": runtime_environment,
        },
        "runtime_environment": runtime_environment,
    }
    (artifact_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    observed_rows = (
        record["batch_size"] * record["lm_head_input_shape"][1]
        if lm_head_position_rows is None
        else lm_head_position_rows
    )
    instrumentation = {
        "execution_records": {"selected_target_logit_execution": [record]},
        "counters": {
            "selected_target_logit_execution_count": 1,
            f"selected_target_logit_{strategy}_execution_count": 1,
            "selected_target_logit_full_sequence_logits_materialized_count": 0,
            "selected_target_logit_selected_position_logits_materialized_count": 1,
            "selected_target_logit_lm_head_position_rows": observed_rows,
        },
    }
    summary_path = tmp_path / "summary.jsonl"
    summary_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "source_width1_artifact_id": "source-artifact",
                "artifact_path": str(artifact_root),
                "runtime_environment": runtime_environment,
                "instrumentation": instrumentation,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            postflight,
            str(summary_path),
            "source-artifact",
            str(artifact_root),
            "flash_sdpa_causal_v1",
            "source_leaf_v1",
            "none",
            "NVIDIA A100 80GB PCIe",
            "",
            "",
            "",
            "legacy_unbound",
            "legacy_unbound",
            "legacy_unbound",
            "legacy_unbound",
            "legacy_unbound",
            "legacy_unbound",
            strategy,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
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
        "EXPECTED_SELECTED_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE",
        "EXPECTED_STOP_GRADIENT_EMBED_CONTRIBUTION_TARGET_LANE_CHUNK_SIZE",
        "EXPECTED_SELECTED_ATTRIBUTION_NEURON_LANE_CHUNK_SIZE",
        "EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_FORWARD_EXECUTION",
        "EXPECTED_STOP_GRADIENT_SELECTED_ATTRIBUTION_STORAGE",
        "EXPECTED_SELECTED_TARGET_LOGIT_EXECUTION",
        "EXPECTED_CUDA_ALLOCATOR_POLICY",
        "EXPECTED_EMBEDDING_EDGE_MATERIALIZATION",
        "EXPECTED_CROSS_LAYER_JACOBIAN_EXECUTION",
        "unset PYTORCH_CUDA_ALLOC_CONF",
        'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
        "run config allocator policy disagrees",
        "run config contribution execution disagrees",
        "run config target-lane chunk size disagrees",
        "run config selected-neuron target-lane chunk size disagrees",
        "run config selected-embed target-lane chunk size disagrees",
        "run config stop-gradient-embed target-lane chunk size disagrees",
        "run config selected-attribution neuron-lane chunk size disagrees",
        "legacy-unbound qualification config unexpectedly declares selected-attribution neuron-lane chunk size",
        "run config stop-gradient selected-attribution forward execution disagrees",
        "legacy-unbound qualification config unexpectedly declares stop-gradient selected-attribution forward execution",
        "run config stop-gradient selected-attribution storage disagrees",
        "legacy-unbound qualification config unexpectedly declares stop-gradient selected-attribution storage",
        "run config selected target-logit execution disagrees",
        "legacy-unbound qualification config unexpectedly declares selected target-logit execution",
        "saved artifact allocator identity disagrees",
        "saved artifact contribution execution identity disagrees",
        "saved artifact target-lane chunk size identity disagrees",
        "saved artifact selected-neuron target-lane chunk size identity disagrees",
        "saved artifact selected-embed target-lane chunk size identity disagrees",
        "saved artifact stop-gradient-embed target-lane chunk size identity disagrees",
        "saved artifact selected-attribution neuron-lane chunk size identity disagrees",
        "legacy-unbound artifact unexpectedly declares selected-attribution neuron-lane chunk size",
        "saved artifact stop-gradient selected-attribution forward execution identity disagrees",
        "saved artifact lacks exact stop-gradient selected-attribution forward execution receipts",
        "saved artifact stop-gradient selected-attribution storage identity disagrees",
        "saved artifact lacks exact stop-gradient selected-attribution storage receipts",
        "saved artifact graph-lifetime receipts disagree",
        "saved artifact selected target-logit execution identity disagrees",
        "saved artifact selected target-logit qualification requires center_logits=false",
        "saved artifact lacks exact selected target-logit execution receipts",
        "saved artifact selected target-logit execution lacks explicit ig_steps",
        "saved artifact selected target-logit execution has invalid ig_steps",
        "saved artifact selected target-logit execution indexes disagree with ig_steps",
        "saved artifact materialization receipts disagree with requested selected target-logit execution",
        "saved artifact selected target-logit execution has malformed shape receipts",
        "saved artifact selected target-logit execution record disagrees with requested strategy",
        "saved artifact selected target-logit LM-head row counter disagrees with records",
        "saved artifact selected target-logit execution did not reduce LM-head rows",
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


def test_launcher_postflight_rejects_malformed_target_logit_shape(
    tmp_path: Path,
) -> None:
    record = _selected_target_logit_record()
    record["lm_head_output_shape"] = [2, 1, 8]

    completed = _run_launcher_postflight(tmp_path, record)

    assert completed.returncode != 0
    assert (
        "saved artifact selected target-logit execution has malformed shape receipts"
        in completed.stderr
    )


def test_launcher_postflight_rejects_candidate_without_row_reduction(
    tmp_path: Path,
) -> None:
    record = _selected_target_logit_record(sequence_count=4, selected_count=4)

    completed = _run_launcher_postflight(tmp_path, record)

    assert completed.returncode != 0
    assert (
        "saved artifact selected target-logit execution did not reduce LM-head rows"
        in completed.stderr
    )


def test_launcher_postflight_cross_checks_target_logit_row_counter(
    tmp_path: Path,
) -> None:
    completed = _run_launcher_postflight(
        tmp_path,
        _selected_target_logit_record(),
        lm_head_position_rows=99,
    )

    assert completed.returncode != 0
    assert (
        "saved artifact selected target-logit LM-head row counter disagrees with records"
        in completed.stderr
    )


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


def test_selected_embed_target_lane_configs_differ_only_by_width() -> None:
    field = "selected_embed_contribution_target_lane_chunk_size"
    configs = []
    for suffix, expected_width in (("none", None), ("5", 5), ("1", 1)):
        path = (
            CONFIG_ROOT
            / "qwen3_4b_thinking_selected_embed_target_lane_chunk_qualification_"
            f"{suffix}_v1.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["adag_config"][field] == expected_width
        assert (
            config["adag_config"]["selected_neuron_contribution_target_lane_chunk_size"]
            == 1
        )
        assert (
            config["artifact_root"] == "results/bonafide/"
            "process-witness-selected-embed-target-lane-chunk-qualification-v1"
        )
        configs.append(config)

    normalized = []
    for config in configs:
        copied = json.loads(json.dumps(config))
        del copied["adag_config"][field]
        normalized.append(copied)
    assert normalized[0] == normalized[1] == normalized[2]


def test_stop_gradient_embed_target_lane_configs_differ_only_by_width() -> None:
    field = "stop_gradient_embed_contribution_target_lane_chunk_size"
    configs = []
    for suffix, expected_width in (("none", None), ("5", 5), ("1", 1)):
        path = (
            CONFIG_ROOT / "qwen3_4b_thinking_stop_gradient_embed_target_lane_chunk_"
            f"qualification_{suffix}_v1.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["adag_config"][field] == expected_width
        assert (
            config["adag_config"]["selected_embed_contribution_target_lane_chunk_size"]
            == 1
        )
        assert (
            config["adag_config"]["selected_neuron_contribution_target_lane_chunk_size"]
            == 1
        )
        assert (
            config["artifact_root"]
            == "results/bonafide/process-witness-stop-gradient-embed-"
            "target-lane-chunk-qualification-v1"
        )
        configs.append(config)

    normalized = []
    for config in configs:
        copied = json.loads(json.dumps(config))
        del copied["adag_config"][field]
        normalized.append(copied)
    assert normalized[0] == normalized[1] == normalized[2]


def test_selected_attribution_neuron_lane_configs_differ_only_by_width() -> None:
    field = "selected_attribution_neuron_lane_chunk_size"
    source = json.loads(
        (
            CONFIG_ROOT / "qwen3_4b_thinking_stop_gradient_embed_target_lane_chunk_"
            "qualification_1_v1.json"
        ).read_text(encoding="utf-8")
    )
    configs = []
    for suffix, expected_width in (("none", None), ("1", 1)):
        path = (
            CONFIG_ROOT / "qwen3_4b_thinking_selected_attribution_neuron_lane_chunk_"
            f"qualification_{suffix}_v1.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["adag_config"][field] == expected_width
        assert (
            config["adag_config"][
                "stop_gradient_embed_contribution_target_lane_chunk_size"
            ]
            == 1
        )
        assert (
            config["artifact_root"]
            == "results/bonafide/process-witness-selected-attribution-"
            "neuron-lane-chunk-qualification-v1"
        )
        normalized_source_clone = json.loads(json.dumps(config))
        del normalized_source_clone["adag_config"][field]
        normalized_source_clone["artifact_root"] = source["artifact_root"]
        assert normalized_source_clone == source
        configs.append(config)

    normalized = []
    for config in configs:
        copied = json.loads(json.dumps(config))
        del copied["adag_config"][field]
        normalized.append(copied)
    assert normalized[0] == normalized[1]


def test_stop_gradient_selected_attribution_forward_configs_are_exact_pair() -> None:
    field = "stop_gradient_selected_attribution_forward_execution"
    source = json.loads(
        (
            CONFIG_ROOT / "qwen3_4b_thinking_selected_attribution_neuron_lane_chunk_"
            "qualification_1_v1.json"
        ).read_text(encoding="utf-8")
    )
    configs = []
    for execution in ("full_model_v1", "prefix_stop_v1"):
        path = (
            CONFIG_ROOT
            / "qwen3_4b_thinking_stop_gradient_selected_attribution_forward_"
            f"qualification_{execution}.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["adag_config"][field] == execution
        assert config["adag_config"]["selected_attribution_neuron_lane_chunk_size"] == 1
        assert (
            config["artifact_root"]
            == "results/bonafide/process-witness-stop-gradient-selected-"
            "attribution-forward-qualification-v1"
        )
        normalized_source_clone = json.loads(json.dumps(config))
        del normalized_source_clone["adag_config"][field]
        normalized_source_clone["artifact_root"] = source["artifact_root"]
        assert normalized_source_clone == source
        configs.append(config)

    normalized = []
    for config in configs:
        copied = json.loads(json.dumps(config))
        del copied["adag_config"][field]
        normalized.append(copied)
    assert normalized[0] == normalized[1]


def test_stop_gradient_selected_attribution_storage_configs_are_exact_pair() -> None:
    field = "stop_gradient_selected_attribution_storage"
    source = json.loads(
        (
            CONFIG_ROOT
            / "qwen3_4b_thinking_stop_gradient_selected_attribution_forward_"
            "qualification_prefix_stop_v1.json"
        ).read_text(encoding="utf-8")
    )
    configs = []
    for strategy in ("graph_retaining_v1", "terminal_detached_v1"):
        path = (
            CONFIG_ROOT
            / "qwen3_4b_thinking_stop_gradient_selected_attribution_storage_"
            f"qualification_{strategy}.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["adag_config"][field] == strategy
        assert (
            config["adag_config"][
                "stop_gradient_selected_attribution_forward_execution"
            ]
            == "prefix_stop_v1"
        )
        assert config["adag_config"]["selected_attribution_neuron_lane_chunk_size"] == 1
        assert (
            config["artifact_root"]
            == "results/bonafide/process-witness-stop-gradient-selected-"
            "attribution-storage-qualification-v1"
        )
        normalized_source_clone = json.loads(json.dumps(config))
        del normalized_source_clone["adag_config"][field]
        normalized_source_clone["artifact_root"] = source["artifact_root"]
        assert normalized_source_clone == source
        configs.append(config)

    normalized = []
    for config in configs:
        copied = json.loads(json.dumps(config))
        del copied["adag_config"][field]
        normalized.append(copied)
    assert normalized[0] == normalized[1]


def test_selected_target_logit_execution_configs_are_exact_optimized_pair() -> None:
    field = "selected_target_logit_execution"
    source = json.loads(
        (
            CONFIG_ROOT
            / "qwen3_4b_thinking_stop_gradient_selected_attribution_storage_"
            "qualification_terminal_detached_v1.json"
        ).read_text(encoding="utf-8")
    )
    configs = []
    for strategy in ("full_logits_v1", "selected_position_logits_v1"):
        path = (
            CONFIG_ROOT
            / "qwen3_4b_thinking_selected_target_logit_execution_qualification_"
            f"{strategy}.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["adag_config"][field] == strategy
        assert config["adag_config"]["stop_gradient_selected_attribution_storage"] == (
            "terminal_detached_v1"
        )
        assert (
            config["adag_config"][
                "stop_gradient_selected_attribution_forward_execution"
            ]
            == "prefix_stop_v1"
        )
        assert (
            config["artifact_root"]
            == "results/bonafide/process-witness-selected-target-logit-"
            "execution-qualification-v1"
        )
        normalized_source_clone = json.loads(json.dumps(config))
        del normalized_source_clone["adag_config"][field]
        normalized_source_clone["artifact_root"] = source["artifact_root"]
        assert normalized_source_clone == source
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
