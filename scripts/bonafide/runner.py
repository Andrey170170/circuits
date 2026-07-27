"""Run one selectable wave of the staged BonaFide tracing benchmark.

Each work item is traced at batch size one and saved as its own compact
artifact.  This runner never combines graphs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from circuits.tracing.artifact import (
    save_compact_trace,
    validate_topk_trace_data,
    validate_compact_trace_integrity,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.trace import (
    CircuitData,
    TopKPositionTrace,
    trace_teacher_forced_response,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.bonafide.manifest import SCHEMA_VERSION, resolve_pretrained_source
from scripts.bonafide.execution_plan import (
    sha256_file,
    validate_execution_plan,
)


RUN_CONFIG_SCHEMA = "bonafide-trace-run-config/v1"
WARMUP_MODE = "first_wave_item_full_trace_discard"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def normalized_trace_warmup(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the identity-bound discarded-trace warm-up policy."""

    raw = config.get(
        "trace_warmup",
        {"enabled": False, "mode": WARMUP_MODE, "wave_id_prefixes": []},
    )
    if not isinstance(raw, Mapping):
        raise ValueError("run config trace_warmup must be an object")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("run config trace_warmup.enabled must be boolean")
    mode = raw.get("mode", WARMUP_MODE)
    if mode != WARMUP_MODE:
        raise ValueError(f"run config trace_warmup.mode must be {WARMUP_MODE!r}")
    prefixes = raw.get("wave_id_prefixes", [])
    if not isinstance(prefixes, list) or any(
        not isinstance(prefix, str) or not prefix for prefix in prefixes
    ):
        raise ValueError("run config trace_warmup.wave_id_prefixes must be non-empty strings")
    if enabled and not prefixes:
        raise ValueError("enabled trace_warmup requires at least one wave_id_prefix")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("run config trace_warmup.wave_id_prefixes must be unique")
    return {"enabled": enabled, "mode": mode, "wave_id_prefixes": list(prefixes)}


def trace_warmup_applies(policy: Mapping[str, Any], wave_id: str) -> bool:
    return bool(policy["enabled"]) and any(
        wave_id.startswith(prefix) for prefix in policy["wave_id_prefixes"]
    )


def validate_run_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != RUN_CONFIG_SCHEMA:
        raise ValueError(f"Unsupported run config schema: {config.get('schema_version')!r}")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("run config requires a model object")
    for field in ("model_id", "revision", "device", "dtype"):
        if not model.get(field):
            raise ValueError(f"run config model.{field} is required")
    if not isinstance(config.get("adag_config"), Mapping):
        raise ValueError("run config requires an adag_config object")
    if int(config.get("batch_size", 1)) != 1:
        raise ValueError("BonaFide performance runs require batch_size=1")
    normalized_trace_warmup(config)


def _require_manifest_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"target sampling {field} must be an integer")
    return value


def validate_target_selection(item: Mapping[str, Any]) -> None:
    """Fail closed if statistical sampling provenance is internally inconsistent."""

    selection = item.get("target_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("work item requires target_selection")
    positions = selection.get("response_token_positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("target_selection.response_token_positions must be non-empty")
    width = _require_manifest_int(selection.get("width"), "width")
    if width != len(positions):
        raise ValueError("target selection width does not match response positions")
    response_length = _require_manifest_int(item.get("response_token_count"), "response length")
    if response_length < 1:
        raise ValueError("target sampling response length must be positive")
    for position in positions:
        position = _require_manifest_int(position, "response token position")
        if not 0 <= position < response_length:
            raise ValueError("target response position is outside the response")

    sampling = selection.get("sampling")
    if sampling is None:
        return
    if not isinstance(sampling, Mapping):
        raise ValueError("target_selection.sampling must be an object")
    if width != 1:
        raise ValueError("sampled work items must have exactly one target")

    position = positions[0]
    sampled_position = _require_manifest_int(
        sampling.get("response_token_position"), "response_token_position"
    )
    if sampled_position != position:
        raise ValueError("target sampling position does not match selected position")
    stratum_index = _require_manifest_int(sampling.get("stratum_index"), "stratum_index")
    stratum_count = _require_manifest_int(sampling.get("stratum_count"), "stratum_count")
    start = _require_manifest_int(sampling.get("stratum_start"), "stratum_start")
    end = _require_manifest_int(
        sampling.get("stratum_end_exclusive"), "stratum_end_exclusive"
    )
    size = _require_manifest_int(sampling.get("stratum_size"), "stratum_size")
    if not 1 <= stratum_count <= response_length:
        raise ValueError("target sampling stratum_count is invalid")
    if not 0 <= stratum_index < stratum_count:
        raise ValueError("target sampling stratum_index is invalid")
    expected_start = (stratum_index * response_length) // stratum_count
    expected_end = ((stratum_index + 1) * response_length) // stratum_count
    if (start, end) != (expected_start, expected_end):
        raise ValueError("target sampling stratum bounds are inconsistent")
    if size != end - start or size < 1:
        raise ValueError("target sampling stratum_size is inconsistent")
    if not start <= position < end:
        raise ValueError("sampled target is outside its stratum")

    probability = sampling.get("selection_probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("target sampling selection_probability must be numeric")
    if not math.isfinite(float(probability)) or not math.isclose(
        float(probability), 1 / size, rel_tol=1e-12, abs_tol=0.0
    ):
        raise ValueError("target sampling selection_probability is inconsistent")
    weight = sampling.get("projection_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("target sampling projection_weight must be numeric")
    if not math.isfinite(float(weight)) or float(weight) != size:
        raise ValueError("target sampling projection_weight is inconsistent")


def validate_wave_sampling_design(
    wave: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Validate the complete probability sample before filtering/resume."""

    items = wave.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("benchmark wave requires non-empty items")
    design = wave.get("sampling_design")
    sampled_items = [
        item
        for item in items
        if isinstance(item.get("target_selection"), Mapping)
        and item["target_selection"].get("sampling") is not None
    ]
    if design is None:
        if sampled_items:
            raise ValueError("sampled wave requires wave-level sampling_design")
        return
    if not isinstance(design, Mapping):
        raise ValueError("wave sampling_design must be an object")
    if len(sampled_items) != len(items):
        raise ValueError("sampling_design wave cannot mix sampled and unsampled items")

    stratum_count = _require_manifest_int(design.get("stratum_count"), "stratum_count")
    population_size = _require_manifest_int(
        design.get("response_token_population_size"), "population size"
    )
    if wave.get("wave_id", "").startswith("wave2c-") and stratum_count != 8:
        raise ValueError("Wave 2c sampling_design must contain exactly 8 strata")
    if len(items) != stratum_count:
        raise ValueError("sampled wave must contain one item for every stratum")

    reference_id = design.get("excluded_reference_example_id")
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError("sampling_design excluded reference must be non-empty")
    wave2_matches = [
        candidate
        for candidate in manifest.get("waves", [])
        if candidate.get("wave_id") == "wave2-progressive-target-window"
    ]
    if wave2_matches:
        wave2_items = wave2_matches[0].get("items", [])
        if not wave2_items or wave2_items[0]["example"]["example_id"] != reference_id:
            raise ValueError("sampling_design excluded reference disagrees with Wave 2")

    expected_common = {
        field: design.get(field) for field in ("design", "seed", "sampler")
    }
    if any(not isinstance(value, str) or not value for value in expected_common.values()):
        raise ValueError("sampling_design design, seed, and sampler must be non-empty")
    seen_strata: set[int] = set()
    sampled_example_ids: set[str] = set()
    for item in items:
        validate_target_selection(item)
        if item["response_token_count"] != population_size:
            raise ValueError("sampled item population size disagrees with sampling_design")
        example_id = item["example"]["example_id"]
        if example_id == reference_id:
            raise ValueError("sampled item cannot be the excluded reference example")
        sampled_example_ids.add(example_id)
        sampling = item["target_selection"]["sampling"]
        for field, expected in expected_common.items():
            if sampling.get(field) != expected:
                raise ValueError(f"sampled item {field} disagrees with sampling_design")
        if sampling.get("stratum_count") != stratum_count:
            raise ValueError("sampled item stratum_count disagrees with sampling_design")
        seen_strata.add(sampling["stratum_index"])
    if len(sampled_example_ids) != 1:
        raise ValueError("sampled wave must contain exactly one response example")
    if seen_strata != set(range(stratum_count)):
        raise ValueError("sampled wave strata must be unique and complete")


def collect_code_revision(repo_root: Path) -> dict[str, Any]:
    """Fingerprint the actual tracing/benchmark source, including dirty files."""

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    # Dataset files, paper checkouts, logs, and downstream results do not change
    # the executable tracing source. Keep the dirty fingerprint scoped to the
    # code/config tree that this runner actually uses.
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "circuits",
        "scripts/bonafide",
    )
    source_paths = sorted(
        [*repo_root.glob("circuits/**/*.py"), *repo_root.glob("scripts/bonafide/**/*.py")]
    )
    digest = hashlib.sha256()
    for path in source_paths:
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_tree_sha256": digest.hexdigest(),
    }


def collect_runtime_environment() -> dict[str, Any]:
    """Record the execution stack versions that can affect numerical output."""

    distributions = (
        "circuits",
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "pandas",
        "safetensors",
        "huggingface-hub",
    )
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    gpu_runtime: dict[str, Any] | None = None
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
        driver_versions: list[str] = []
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            driver_versions = sorted(
                {line.strip() for line in completed.stdout.splitlines() if line.strip()}
            )
        except (OSError, subprocess.CalledProcessError):
            driver_versions = []
        gpu_runtime = {
            "visible_device_count": len(devices),
            "devices": devices,
            "driver_versions": driver_versions,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_cuda_version": torch.version.cuda,
        "gpu_runtime": gpu_runtime,
    }


def select_wave(manifest: Mapping[str, Any], wave_id: str) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported benchmark manifest: {manifest.get('schema_version')!r}")
    matches = [wave for wave in manifest.get("waves", []) if wave.get("wave_id") == wave_id]
    if len(matches) != 1:
        available = [wave.get("wave_id") for wave in manifest.get("waves", [])]
        raise ValueError(f"Wave {wave_id!r} not found exactly once; available: {available}")
    return matches[0]


def runtime_artifact_identity(
    item: Mapping[str, Any],
    config: Mapping[str, Any],
    code_revision: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    *,
    wave_id: str,
    warmup_source_item: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Bind dataset work-unit identity to exact model and tracing configuration."""

    validate_target_selection(item)
    warmup_policy = normalized_trace_warmup(config)
    warmup_identity: dict[str, Any] = {
        **warmup_policy,
        "applies_to_wave": trace_warmup_applies(warmup_policy, wave_id),
    }
    if warmup_identity["applies_to_wave"]:
        if warmup_source_item is None:
            raise ValueError("warm-up source item is required for an enabled wave")
        validate_target_selection(warmup_source_item)
        warmup_identity.update(
            {
                "source_artifact_id": warmup_source_item["artifact_id"],
                "source_work_item_sha256": _sha256(warmup_source_item),
                "source_target_selection": dict(
                    warmup_source_item["target_selection"]
                ),
            }
        )
    identity = {
        "source_artifact_id": item["artifact_id"],
        "source_work_item_sha256": _sha256(item),
        "source_target_selection": dict(item["target_selection"]),
        "model": dict(config["model"]),
        "adag_config": dict(config["adag_config"]),
        "trace_warmup": warmup_identity,
        "batch_size": 1,
        "code_revision": dict(code_revision),
        "runtime_environment": dict(runtime_environment),
    }
    digest = _sha256(identity)
    identity["sha256"] = digest
    return f"trace-{digest[:24]}", identity


def _completed_artifact_matches(path: Path, identity: Mapping[str, Any]) -> bool:
    # Existing artifacts are never trusted based on identity JSON alone.  A
    # damaged payload must block resume rather than being silently overwritten.
    manifest = validate_compact_trace_integrity(path)
    return manifest.get("artifact_identity") == identity


def validate_runtime_trace_against_item(
    trace: CircuitData, item: Mapping[str, Any]
) -> None:
    """Ensure live tokenization still matches the frozen benchmark manifest."""

    selection = item["target_selection"]
    expected_positions = [int(value) for value in selection["response_token_positions"]]
    expected_response_count = int(item["response_token_count"])
    actual_response_count = trace.trace_metadata.get("response_token_count")
    if actual_response_count != expected_response_count:
        raise ValueError(
            "runtime response token count does not match manifest: "
            f"{actual_response_count!r} != {expected_response_count}"
        )
    actual_positions = [
        provenance.get("response_token_position")
        for provenance in trace.target_provenance
    ]
    if actual_positions != expected_positions:
        raise ValueError(
            "runtime target response positions do not match manifest: "
            f"{actual_positions!r} != {expected_positions!r}"
        )
    if len(trace.target_logits) != 1:
        raise ValueError("runtime trace must contain exactly one target-logit batch item")
    actual_token_ids = [int(value) for value in trace.target_logits[0]]
    provenance_token_ids = [
        provenance.get("token_id") for provenance in trace.target_provenance
    ]
    if provenance_token_ids != actual_token_ids:
        raise ValueError(
            "runtime target token IDs disagree between trace payload and provenance"
        )
    if len(actual_token_ids) != len(expected_positions):
        raise ValueError(
            "runtime target token count does not match manifest target positions"
        )
    expected_final_token_id = int(selection["final_target_token_id"])
    if not actual_token_ids or actual_token_ids[-1] != expected_final_token_id:
        actual_final = actual_token_ids[-1] if actual_token_ids else None
        raise ValueError(
            "runtime final target token ID does not match manifest: "
            f"{actual_final!r} != {expected_final_token_id}"
        )


def validate_runtime_topk_trace_against_item(
    trace: TopKPositionTrace,
    item: Mapping[str, Any],
    trace_family: Mapping[str, Any],
) -> None:
    """Ensure a live candidate-axis trace matches its frozen work item."""

    validate_topk_trace_data(trace)
    validate_target_selection(item)
    positions = item["target_selection"]["response_token_positions"]
    if len(positions) != 1:
        raise ValueError("runtime top-k trace requires one response target position")
    expected_position = int(positions[0])
    if trace.shared_response_position != expected_position:
        raise ValueError("runtime top-k response position does not match manifest")

    data = trace.circuit_data
    expected_response_count = int(item["response_token_count"])
    if data.trace_metadata.get("response_token_count") != expected_response_count:
        raise ValueError(
            "runtime top-k response token count does not match manifest"
        )
    expected_observed_token_id = int(
        item["target_selection"]["final_target_token_id"]
    )
    if data.target_logits != [[expected_observed_token_id]]:
        raise ValueError("runtime top-k observed token does not match manifest")
    if trace.candidate_selection.observed_token_id != expected_observed_token_id:
        raise ValueError(
            "runtime candidate selection observed token does not match manifest"
        )

    expected_fields = {
        "trace_family_id": trace.trace_family_id,
        "candidate_policy_id": trace.candidate_selection.policy_id,
        "candidate_policy_version": trace.candidate_selection.policy_version,
        "candidate_count": trace.candidate_count,
        "joint_objective_id": trace.joint_objective.objective_id,
        "joint_objective_version": trace.joint_objective.objective_version,
    }
    for field, actual in expected_fields.items():
        if trace_family.get(field) != actual:
            raise ValueError(
                f"runtime top-k {field} does not match frozen trace family"
            )


def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _rss_peak_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _gpu_info(device: str) -> dict[str, Any] | None:
    if not device.startswith("cuda"):
        return None
    properties = torch.cuda.get_device_properties(torch.device(device))
    return {
        "device": device,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
    }


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def wave_stop_reason(
    record: Mapping[str, Any],
    *,
    uses_cuda: bool,
    max_trace_seconds: float | None,
    min_cuda_headroom_bytes: int,
    stop_on_oom: bool,
) -> str | None:
    """Apply monotonic Wave 2 safety gates after each completed work unit."""

    if record["status"] == "oom" and stop_on_oom:
        return "cuda_oom"
    if (
        record["status"] == "complete"
        and max_trace_seconds is not None
        and record["trace_wall_seconds"] > max_trace_seconds
    ):
        return "max_trace_seconds_exceeded"
    if (
        record["status"] == "complete"
        and uses_cuda
        and record["cuda_headroom_after_peak_bytes"] < min_cuda_headroom_bytes
    ):
        return "min_cuda_headroom_not_met"
    return None


def _torch_dtype(name: str) -> torch.dtype:
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return choices[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(choices)}") from error


def _load_model_and_tokenizer(config: Mapping[str, Any]):
    model_config = config["model"]
    model_id = model_config["model_id"]
    revision = model_config["revision"]
    local_files_only = bool(model_config.get("local_files_only", True))
    explicit_path = model_config.get("local_snapshot_path")
    pretrained_source = resolve_pretrained_source(
        model_id=model_id,
        revision=revision,
        local_files_only=local_files_only,
        explicit_path=Path(os.path.expandvars(explicit_path)) if explicit_path else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_source,
        revision=None if pretrained_source != model_id else revision,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = model_config["device"]
    kwargs = dict(model_config.get("from_pretrained_kwargs", {}))
    kwargs.update(
        {
            "revision": revision,
            "local_files_only": local_files_only,
            "torch_dtype": _torch_dtype(model_config["dtype"]),
        }
    )
    if device.startswith("cuda"):
        kwargs.setdefault("device_map", {"": device})
    if pretrained_source != model_id:
        kwargs["revision"] = None
    model = AutoModelForCausalLM.from_pretrained(pretrained_source, **kwargs)
    model.config._name_or_path = model_id
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return model, tokenizer


def _base_record(
    *,
    wave_id: str,
    item: Mapping[str, Any],
    runtime_artifact_id: str,
    identity: Mapping[str, Any],
    model_load_seconds: float,
    code_revision: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    gpu_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selection = item["target_selection"]
    example = item["example"]
    return {
        "wave_id": wave_id,
        "source_artifact_id": item["artifact_id"],
        "artifact_id": runtime_artifact_id,
        "artifact_identity_sha256": identity["sha256"],
        "example_id": example["example_id"],
        "annotation_row_ids": example["annotation_row_ids"],
        "label_types": example["label_types"],
        "response_token_count": item["response_token_count"],
        "target_count": selection["width"],
        "target_response_positions": selection["response_token_positions"],
        "target_sampling": dict(selection["sampling"])
        if "sampling" in selection
        else None,
        "batch_size": 1,
        "objective": item["objective"],
        "model_load_seconds": model_load_seconds,
        "code_revision": dict(code_revision),
        "runtime_environment": dict(runtime_environment),
        "gpu": dict(gpu_info) if gpu_info is not None else None,
    }


def _discarded_trace_warmup(
    *,
    item: Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    adag_config: ADAGConfig,
    device: str,
    uses_cuda: bool,
) -> tuple[dict[str, Any], dict[str, Any], Exception | None]:
    """Run and discard one complete trace, then release all trace-owned state."""

    example = item["example"]
    positions = item["target_selection"]["response_token_positions"]
    instrumentation = TraceInstrumentation(device=device, synchronize_cuda=uses_cuda)
    trace: CircuitData | None = None
    started = time.perf_counter()
    outcome = "complete"
    error_type = None
    error_message = None
    node_count = None
    edge_count = None
    failure: Exception | None = None
    cleanup_failure: Exception | None = None
    try:
        trace = trace_teacher_forced_response(
            model=model,
            tokenizer=tokenizer,
            prompt=example["prompt"],
            response=example["response"],
            target_response_positions=positions,
            config=adag_config,
            label=example["example_id"],
            benchmark_only=bool(item["objective"]["benchmark_only_multi_target"]),
            instrumentation=instrumentation,
        )
        validate_runtime_trace_against_item(trace, item)
        if uses_cuda:
            torch.cuda.synchronize()
        node_count = len(trace.df_node)
        edge_count = len(trace.df_edge)
    except torch.cuda.OutOfMemoryError as error:
        outcome = "oom"
        failure = error
        error_type = type(error).__name__
        error_message = str(error)
    except Exception as error:
        outcome = "error"
        failure = error
        error_type = type(error).__name__
        error_message = str(error)
    elapsed = time.perf_counter() - started
    instrumentation_snapshot = instrumentation.snapshot()

    # The measured loop must not inherit graph tensors, allocator blocks, or
    # Python objects from the discarded trace.
    try:
        trace = None
        gc.collect()
        if uses_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as error:
        cleanup_failure = error
        if failure is None:
            outcome = "error"
            failure = error
            error_type = type(error).__name__
            error_message = str(error)

    provenance = {
        "enabled": True,
        "mode": WARMUP_MODE,
        "source_artifact_id": item["artifact_id"],
        "source_example_id": example["example_id"],
        "source_target_selection": item["target_selection"],
        "status": outcome,
        "wall_seconds": elapsed,
    }
    record = {
        "record_type": "discarded_trace_warmup",
        "status": f"warmup_{outcome}",
        "warmup": provenance,
        "instrumentation": instrumentation_snapshot,
        "node_count": node_count,
        "edge_count": edge_count,
    }
    if error_type is not None:
        record["error_type"] = error_type
        record["error"] = error_message
    if cleanup_failure is not None:
        record["cleanup_error_type"] = type(cleanup_failure).__name__
        record["cleanup_error"] = str(cleanup_failure)
    return provenance, record, failure


def run_wave(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    wave_id: str,
    artifact_root: Path,
    summary_jsonl: Path,
    only_artifact_id: str | None = None,
    dry_run: bool = False,
    _model_bundle: tuple[Any, Any] | None = None,
    _model_load_seconds: float = 0.0,
    _code_revision: Mapping[str, Any] | None = None,
    _runtime_environment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run exactly one explicitly selected wave, never subsequent waves."""

    validate_run_config(config)
    wave = select_wave(manifest, wave_id)
    validate_wave_sampling_design(wave, manifest)
    wave_items = list(wave["items"])
    warmup_policy = normalized_trace_warmup(config)
    warmup_source_item = (
        wave_items[0] if trace_warmup_applies(warmup_policy, wave_id) else None
    )
    model_config = config["model"]
    repo_root = Path(__file__).resolve().parents[2]
    code_revision = dict(_code_revision or collect_code_revision(repo_root))
    runtime_environment = dict(_runtime_environment or collect_runtime_environment())
    if manifest["tokenizer"]["model_id"] != model_config["model_id"]:
        raise ValueError("manifest tokenizer model_id does not match run config model_id")
    if manifest["tokenizer"]["revision"] != model_config["revision"]:
        raise ValueError("manifest tokenizer revision does not match run config revision")

    items = wave_items
    if only_artifact_id:
        items = [item for item in items if item["artifact_id"] == only_artifact_id]
        if not items:
            raise ValueError(f"No item {only_artifact_id!r} in wave {wave_id!r}")

    planned: list[tuple[dict[str, Any], str, dict[str, Any], Path]] = []
    results: list[dict[str, Any]] = []
    for item in items:
        artifact_id, identity = runtime_artifact_identity(
            item,
            config,
            code_revision,
            runtime_environment,
            wave_id=wave_id,
            warmup_source_item=warmup_source_item,
        )
        artifact_path = artifact_root / wave_id / artifact_id
        base = _base_record(
            wave_id=wave_id,
            item=item,
            runtime_artifact_id=artifact_id,
            identity=identity,
            model_load_seconds=0.0,
            code_revision=code_revision,
            runtime_environment=runtime_environment,
            gpu_info=None,
        )
        if artifact_path.exists():
            if not _completed_artifact_matches(artifact_path, identity):
                raise FileExistsError(
                    f"artifact path exists but identity/completion does not match: {artifact_path}"
                )
            record = {
                **base,
                "status": "skipped_complete",
                "artifact_path": str(artifact_path),
                "artifact_bytes": _directory_size(artifact_path),
            }
            results.append(record)
            if not dry_run:
                _append_jsonl(summary_jsonl, record)
            continue
        planned.append((item, artifact_id, identity, artifact_path))

    if dry_run:
        results.extend(
            {
                **_base_record(
                    wave_id=wave_id,
                    item=item,
                    runtime_artifact_id=artifact_id,
                    identity=identity,
                    model_load_seconds=0.0,
                    code_revision=code_revision,
                    runtime_environment=runtime_environment,
                    gpu_info=None,
                ),
                "status": "planned",
                "artifact_path": str(artifact_path),
            }
            for item, artifact_id, identity, artifact_path in planned
        )
        return results
    if not planned:
        return results

    if _model_bundle is None:
        load_started = time.perf_counter()
        model, tokenizer = _load_model_and_tokenizer(config)
        model_load_seconds = time.perf_counter() - load_started
    else:
        model, tokenizer = _model_bundle
        model_load_seconds = _model_load_seconds
    device = model_config["device"]
    uses_cuda = device.startswith("cuda")
    gpu_info = _gpu_info(device)
    adag_config = ADAGConfig(**{**config["adag_config"], "device": device})

    warmup_provenance: dict[str, Any] = {
        "enabled": False,
        "mode": WARMUP_MODE,
        "status": "disabled",
    }
    if warmup_source_item is not None:
        warmup_provenance, warmup_record, warmup_failure = _discarded_trace_warmup(
            item=warmup_source_item,
            model=model,
            tokenizer=tokenizer,
            adag_config=adag_config,
            device=device,
            uses_cuda=uses_cuda,
        )
        warmup_record.update(
            {
                "wave_id": wave_id,
                "model_load_seconds": model_load_seconds,
                "code_revision": code_revision,
                "runtime_environment": runtime_environment,
                "gpu": gpu_info,
            }
        )
        _append_jsonl(summary_jsonl, warmup_record)
        results.append(warmup_record)
        if warmup_failure is not None:
            raise warmup_failure

    limits = config.get("wave_limits", {})
    max_trace_seconds = limits.get("max_trace_seconds")
    min_cuda_headroom_bytes = int(limits.get("min_cuda_headroom_bytes", 0))
    stop_on_oom = bool(limits.get("stop_on_oom", True))
    for planned_index, (item, artifact_id, identity, artifact_path) in enumerate(planned):
        base = _base_record(
            wave_id=wave_id,
            item=item,
            runtime_artifact_id=artifact_id,
            identity=identity,
            model_load_seconds=model_load_seconds,
            code_revision=code_revision,
            runtime_environment=runtime_environment,
            gpu_info=gpu_info,
        )
        base["trace_warmup"] = warmup_provenance
        example = item["example"]
        positions = item["target_selection"]["response_token_positions"]
        benchmark_only = bool(item["objective"]["benchmark_only_multi_target"])
        rss_before = _rss_peak_bytes()
        if uses_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated_before = torch.cuda.memory_allocated()
            reserved_before = torch.cuda.memory_reserved()
        else:
            allocated_before = reserved_before = 0
        instrumentation = TraceInstrumentation(
            device=device,
            synchronize_cuda=uses_cuda,
        )
        started = time.perf_counter()
        try:
            trace = trace_teacher_forced_response(
                model=model,
                tokenizer=tokenizer,
                prompt=example["prompt"],
                response=example["response"],
                target_response_positions=positions,
                config=adag_config,
                label=example["example_id"],
                benchmark_only=benchmark_only,
                instrumentation=instrumentation,
            )
            validate_runtime_trace_against_item(trace, item)
            if uses_cuda:
                torch.cuda.synchronize()
            trace_elapsed = time.perf_counter() - started
            instrumentation_snapshot = instrumentation.snapshot()
            trace.trace_metadata["instrumentation"] = instrumentation_snapshot
            metrics = {
                "status": "complete",
                "trace_wall_seconds": trace_elapsed,
                "cuda_allocated_before_bytes": allocated_before,
                "cuda_reserved_before_bytes": reserved_before,
                "cuda_allocated_after_trace_bytes": torch.cuda.memory_allocated()
                if uses_cuda
                else 0,
                "cuda_reserved_after_trace_bytes": torch.cuda.memory_reserved()
                if uses_cuda
                else 0,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if uses_cuda
                else 0,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()
                if uses_cuda
                else 0,
                "cuda_headroom_after_peak_bytes": (
                    gpu_info["total_memory_bytes"] - torch.cuda.max_memory_reserved()
                    if uses_cuda and gpu_info is not None
                    else 0
                ),
                "rss_peak_before_bytes": rss_before,
                "rss_peak_after_bytes": _rss_peak_bytes(),
                "node_count": len(trace.df_node),
                "edge_count": len(trace.df_edge),
                "input_token_count": len(trace.cis[0]),
                "response_token_count": item["response_token_count"],
                "target_count": len(positions),
                "instrumentation": instrumentation_snapshot,
            }
            serialization_started = time.perf_counter()
            save_compact_trace(
                artifact_path,
                trace,
                metrics=metrics,
                manifest={
                    "artifact_id": artifact_id,
                    "artifact_identity": identity,
                    "benchmark_wave_id": wave_id,
                    "source_artifact_id": item["artifact_id"],
                    "source_target_selection": item["target_selection"],
                    "trace_warmup": warmup_provenance,
                    "bonafide_example": example,
                    "objective": item["objective"],
                    "model_revision": model_config["revision"],
                    "code_revision": code_revision,
                    "runtime_environment": runtime_environment,
                    "gpu": gpu_info,
                },
            )
            serialization_elapsed = time.perf_counter() - serialization_started
            artifact_bytes = _directory_size(artifact_path)
            total_unit_elapsed = time.perf_counter() - started
            record = {
                **base,
                **metrics,
                "serialization_wall_seconds": serialization_elapsed,
                "total_unit_wall_seconds": total_unit_elapsed,
                "artifact_path": str(artifact_path),
                "artifact_bytes": artifact_bytes,
            }
        except torch.cuda.OutOfMemoryError as error:
            elapsed = time.perf_counter() - started
            record = {
                **base,
                "status": "oom",
                "trace_wall_seconds": elapsed,
                "serialization_wall_seconds": 0.0,
                "total_unit_wall_seconds": elapsed,
                "cuda_allocated_before_bytes": allocated_before,
                "cuda_reserved_before_bytes": reserved_before,
                "cuda_allocated_after_trace_bytes": torch.cuda.memory_allocated()
                if uses_cuda
                else 0,
                "cuda_reserved_after_trace_bytes": torch.cuda.memory_reserved()
                if uses_cuda
                else 0,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if uses_cuda
                else 0,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()
                if uses_cuda
                else 0,
                "cuda_headroom_after_peak_bytes": (
                    gpu_info["total_memory_bytes"] - torch.cuda.max_memory_reserved()
                    if uses_cuda and gpu_info is not None
                    else 0
                ),
                "rss_peak_before_bytes": rss_before,
                "rss_peak_after_bytes": _rss_peak_bytes(),
                "error_type": type(error).__name__,
                "error": str(error),
                "instrumentation": instrumentation.snapshot(),
            }
            if uses_cuda:
                torch.cuda.empty_cache()
        except Exception as error:
            elapsed = time.perf_counter() - started
            record = {
                **base,
                "status": "error",
                "trace_wall_seconds": elapsed,
                "serialization_wall_seconds": 0.0,
                "total_unit_wall_seconds": elapsed,
                "rss_peak_before_bytes": rss_before,
                "rss_peak_after_bytes": _rss_peak_bytes(),
                "error_type": type(error).__name__,
                "error": str(error),
                "instrumentation": instrumentation.snapshot(),
            }
            _append_jsonl(summary_jsonl, record)
            results.append(record)
            if not config.get("continue_on_error", False):
                raise
            continue
        _append_jsonl(summary_jsonl, record)
        results.append(record)
        stop_reason = wave_stop_reason(
            record,
            uses_cuda=uses_cuda,
            max_trace_seconds=float(max_trace_seconds)
            if max_trace_seconds is not None
            else None,
            min_cuda_headroom_bytes=min_cuda_headroom_bytes,
            stop_on_oom=stop_on_oom,
        )
        if stop_reason is not None:
            stop_record = {
                "status": "wave_stopped",
                "wave_id": wave_id,
                "stop_reason": stop_reason,
                "after_artifact_id": artifact_id,
                "remaining_item_count": len(planned) - planned_index - 1,
                "code_revision": code_revision,
                "gpu": gpu_info,
            }
            _append_jsonl(summary_jsonl, stop_record)
            results.append(stop_record)
            break
    return results


def _ensure_execution_cohort(
    *,
    artifact_root: Path,
    plan_sha256: str,
    config: Mapping[str, Any],
    code_revision: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> Path:
    """Atomically establish one numerical/software cohort for a plan."""

    cohort = {
        "schema_version": "bonafide-execution-cohort/v1",
        "plan_sha256": plan_sha256,
        "config_sha256": _sha256(config),
        "code_revision": dict(code_revision),
        "runtime_environment": dict(runtime_environment),
    }
    cohort_dir = artifact_root / "execution-cohorts"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    path = cohort_dir / f"{plan_sha256}.json"
    encoded = _canonical_json(cohort) + b"\n"
    temporary = cohort_dir / f".{plan_sha256}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()
    if not path.is_file():
        raise RuntimeError(f"failed to establish execution cohort lock: {path}")
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid execution cohort lock: {path}") from error
    if existing != cohort:
        raise ValueError(
            "execution cohort mismatch; refusing to mix config, code, or runtime environments"
        )
    return path


def _compound_item_lookup(
    manifest: Mapping[str, Any], refs: list[Mapping[str, Any]]
) -> list[tuple[str, str]]:
    available: dict[tuple[str, str], Mapping[str, Any]] = {}
    for wave in manifest.get("waves", []):
        wave_id = wave.get("wave_id")
        for item in wave.get("items", []):
            key = (wave_id, item.get("artifact_id"))
            if key in available:
                raise ValueError(f"duplicate item in source manifest: {key}")
            available[key] = item
    selected: list[tuple[str, str]] = []
    for ref in refs:
        key = (ref.get("source_wave_id"), ref.get("source_artifact_id"))
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError(f"invalid compound item reference: {key}")
        if key in selected:
            raise ValueError(f"duplicate compound assignment: {key}")
        if key not in available:
            raise ValueError(f"compound item is absent from source manifest: {key}")
        selected.append(key)
    if not selected:
        raise ValueError("compound shard contains no work items")
    return selected


def run_compound_shard(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    execution_plan: dict[str, Any],
    task_index: int,
    artifact_root: Path,
    summary_jsonl: Path,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run one explicit cross-wave routine shard with one resident model."""

    validate_run_config(config)
    validate_execution_plan(execution_plan, manifest=manifest, verify_sources=True)
    source_manifest = execution_plan["sources"]["final_trace_manifest"]
    if sha256_file(Path(source_manifest["path"])) != source_manifest["sha256"]:
        raise ValueError("execution plan source manifest hash drift")
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise ValueError("compound task index must be an integer")
    tasks = execution_plan["tasks"]
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"compound task index {task_index} is out of range")
    task = tasks[task_index]
    if task.get("task_index") != task_index:
        raise ValueError("compound task index disagrees with execution plan")
    task_kind = task["task_kind"]
    source_index = task["source_index"]
    if task_kind == "routine":
        refs = execution_plan["sharding"]["shards"][source_index]["items"]
    elif task_kind == "extreme_preflight":
        refs = [execution_plan["extremes"]["preflight"][source_index]]
    elif task_kind == "pathological_manual":
        refs = [execution_plan["extremes"]["manual_pathological"][source_index]]
    else:
        raise ValueError(f"unsupported compound task kind {task_kind!r}")
    selected = _compound_item_lookup(manifest, refs)
    warmup_policy = normalized_trace_warmup(config)
    warmup_waves = sorted(
        {wave_id for wave_id, _ in selected if trace_warmup_applies(warmup_policy, wave_id)}
    )
    if warmup_waves:
        raise ValueError(
            "compound shards cannot contain warmup-applicable source waves: "
            + ", ".join(warmup_waves)
        )

    # Complete the entire identity/resume preflight before loading the model.
    repo_root = Path(__file__).resolve().parents[2]
    code_revision = collect_code_revision(repo_root)
    runtime_environment = collect_runtime_environment()
    preflight: list[dict[str, Any]] = []
    for wave_id, artifact_id in selected:
        preflight.extend(
            run_wave(
                config=config,
                manifest=manifest,
                wave_id=wave_id,
                artifact_root=artifact_root,
                summary_jsonl=summary_jsonl,
                only_artifact_id=artifact_id,
                dry_run=True,
                _code_revision=code_revision,
                _runtime_environment=runtime_environment,
            )
        )
    if dry_run:
        return preflight
    source_config = execution_plan["sources"]["trace_run_config"]
    if _sha256(config) != source_config.get("canonical_sha256"):
        raise ValueError("execution plan tracing-config identity disagrees with loaded config")
    if _sha256(manifest) != source_manifest.get("canonical_sha256"):
        raise ValueError("execution plan manifest identity disagrees with loaded manifest")
    _ensure_execution_cohort(
        artifact_root=artifact_root,
        plan_sha256=execution_plan["plan_sha256"],
        config=config,
        code_revision=code_revision,
        runtime_environment=runtime_environment,
    )
    expected_count = len(selected)
    task_base = {
        "task_index": task_index,
        "task_kind": task_kind,
        "plan_sha256": execution_plan["plan_sha256"],
        "expected_item_count": expected_count,
    }
    _append_jsonl(
        summary_jsonl,
        {"record_type": "compound_task", "status": "task_started", **task_base},
    )
    has_planned = any(record.get("status") == "planned" for record in preflight)
    if has_planned:
        load_started = time.perf_counter()
        model_bundle = _load_model_and_tokenizer(config)
        model_load_seconds = time.perf_counter() - load_started
    else:
        skipped_count = sum(
            record.get("status") == "skipped_complete" for record in preflight
        )
        complete_record = {
            "record_type": "compound_task",
            "status": "task_complete",
            **task_base,
            "completed_item_count": 0,
            "skipped_item_count": skipped_count,
            "remaining_item_count": 0,
        }
        _append_jsonl(summary_jsonl, complete_record)
        return [*preflight, complete_record]
    results: list[dict[str, Any]] = []
    completed_count = 0
    skipped_count = 0
    signal_state = {"requested": False}
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def request_stop(_signum, _frame) -> None:
        signal_state["requested"] = True

    signal.signal(signal.SIGUSR1, request_stop)
    for selected_index, (wave_id, artifact_id) in enumerate(selected):
        if signal_state["requested"]:
            stop_record = {
                "record_type": "compound_task",
                "status": "task_stopped",
                **task_base,
                "stop_reason": "slurm_time_limit_signal",
                "completed_item_count": completed_count,
                "skipped_item_count": skipped_count,
                "remaining_item_count": expected_count - selected_index,
            }
            _append_jsonl(summary_jsonl, stop_record)
            signal.signal(signal.SIGUSR1, previous_handler)
            raise RuntimeError(f"compound task {task_index} stopped after SIGUSR1")
        try:
            records = run_wave(
                config=config,
                manifest=manifest,
                wave_id=wave_id,
                artifact_root=artifact_root,
                summary_jsonl=summary_jsonl,
                only_artifact_id=artifact_id,
                dry_run=False,
                _model_bundle=model_bundle,
                _model_load_seconds=model_load_seconds,
                _code_revision=code_revision,
                _runtime_environment=runtime_environment,
            )
        except Exception as error:
            stop_record = {
                "record_type": "compound_task",
                "status": "task_stopped",
                **task_base,
                "source_wave_id": wave_id,
                "source_artifact_id": artifact_id,
                "stop_reason": "trace_error",
                "completed_item_count": completed_count,
                "skipped_item_count": skipped_count,
                "remaining_item_count": expected_count - selected_index,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _append_jsonl(summary_jsonl, stop_record)
            signal.signal(signal.SIGUSR1, previous_handler)
            raise
        results.extend(records)
        completed_count += sum(record.get("status") == "complete" for record in records)
        skipped_count += sum(record.get("status") == "skipped_complete" for record in records)
        stop = next((record for record in records if record.get("status") == "wave_stopped"), None)
        if stop is not None:
            shard_stop = {
                "record_type": "compound_task",
                "status": "task_stopped",
                **task_base,
                "source_wave_id": wave_id,
                "source_artifact_id": artifact_id,
                "stop_reason": stop["stop_reason"],
                "completed_item_count": completed_count,
                "skipped_item_count": skipped_count,
                "remaining_item_count": expected_count - selected_index - 1,
            }
            _append_jsonl(summary_jsonl, shard_stop)
            results.append(shard_stop)
            signal.signal(signal.SIGUSR1, previous_handler)
            raise RuntimeError(
                f"compound task {task_index} stopped: {stop['stop_reason']} at {artifact_id}"
            )
        failed = next(
            (record for record in records if record.get("status") in {"error", "oom"}),
            None,
        )
        if failed is not None:
            stop_record = {
                "record_type": "compound_task",
                "status": "task_stopped",
                **task_base,
                "source_wave_id": wave_id,
                "source_artifact_id": artifact_id,
                "stop_reason": "failed_item_without_stop_gate",
                "completed_item_count": completed_count,
                "skipped_item_count": skipped_count,
                "remaining_item_count": expected_count - completed_count - skipped_count,
                "error_type": failed.get("error_type"),
                "error": failed.get("error"),
            }
            _append_jsonl(summary_jsonl, stop_record)
            results.append(stop_record)
            signal.signal(signal.SIGUSR1, previous_handler)
            raise RuntimeError(
                f"compound task {task_index} contains failed item {artifact_id}"
            )
    signal.signal(signal.SIGUSR1, previous_handler)
    if completed_count + skipped_count != expected_count:
        stop_record = {
            "record_type": "compound_task",
            "status": "task_stopped",
            **task_base,
            "stop_reason": "incomplete_item_accounting",
            "completed_item_count": completed_count,
            "skipped_item_count": skipped_count,
            "remaining_item_count": expected_count - completed_count - skipped_count,
        }
        _append_jsonl(summary_jsonl, stop_record)
        results.append(stop_record)
        raise RuntimeError(f"compound task {task_index} has incomplete item accounting")
    complete_record = {
        "record_type": "compound_task",
        "status": "task_complete",
        **task_base,
        "completed_item_count": completed_count,
        "skipped_item_count": skipped_count,
        "remaining_item_count": 0,
    }
    _append_jsonl(summary_jsonl, complete_record)
    results.append(complete_record)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--wave", help="One exact wave_id; no later wave is implied")
    selector.add_argument("--execution-plan", type=Path, help="Validated compound execution plan")
    parser.add_argument("--execution-task-index", type=int)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--summary-jsonl", type=Path)
    parser.add_argument("--only-artifact-id")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list work without a model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    artifact_root = args.artifact_root or Path(config.get("artifact_root", "results/bonafide"))
    if args.execution_plan is not None:
        if args.execution_task_index is None:
            raise ValueError("--execution-plan requires --execution-task-index")
        if args.only_artifact_id is not None:
            raise ValueError("--only-artifact-id is only valid with legacy --wave")
        execution_plan = load_json(args.execution_plan)
        summary_jsonl = args.summary_jsonl or (
            artifact_root
            / "execution-summaries"
            / execution_plan["plan_sha256"]
            / f"task-{args.execution_task_index:02d}.jsonl"
        )
        results = run_compound_shard(
            config=config,
            manifest=manifest,
            execution_plan=execution_plan,
            task_index=args.execution_task_index,
            artifact_root=artifact_root,
            summary_jsonl=summary_jsonl,
            dry_run=args.dry_run,
        )
    else:
        if args.execution_task_index is not None:
            raise ValueError("--execution-task-index requires --execution-plan")
        summary_jsonl = args.summary_jsonl or artifact_root / "benchmark-summary.jsonl"
        results = run_wave(
            config=config,
            manifest=manifest,
            wave_id=args.wave,
            artifact_root=artifact_root,
            summary_jsonl=summary_jsonl,
            only_artifact_id=args.only_artifact_id,
            dry_run=args.dry_run,
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
