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
        "EXPECTED_CUDA_ALLOCATOR_POLICY",
        "unset PYTORCH_CUDA_ALLOC_CONF",
        'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
        "run config allocator policy disagrees",
        "saved artifact allocator identity disagrees",
        "saved artifact lacks the exact requested allocator runtime receipt",
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
