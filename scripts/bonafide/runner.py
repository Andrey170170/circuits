"""Run one selectable wave of the staged BonaFide tracing benchmark.

Each work item is traced at batch size one and saved as its own compact
artifact.  This runner never combines graphs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from circuits.tracing.artifact import (
    save_compact_trace,
    validate_compact_trace_integrity,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.trace import CircuitData, trace_teacher_forced_response
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.bonafide.manifest import SCHEMA_VERSION, resolve_pretrained_source


RUN_CONFIG_SCHEMA = "bonafide-trace-run-config/v1"


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
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_cuda_version": torch.version.cuda,
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
) -> tuple[str, dict[str, Any]]:
    """Bind dataset work-unit identity to exact model and tracing configuration."""

    identity = {
        "source_artifact_id": item["artifact_id"],
        "source_work_item_sha256": _sha256(item),
        "model": dict(config["model"]),
        "adag_config": dict(config["adag_config"]),
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
        "batch_size": 1,
        "objective": item["objective"],
        "model_load_seconds": model_load_seconds,
        "code_revision": dict(code_revision),
        "runtime_environment": dict(runtime_environment),
        "gpu": dict(gpu_info) if gpu_info is not None else None,
    }


def run_wave(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    wave_id: str,
    artifact_root: Path,
    summary_jsonl: Path,
    only_artifact_id: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run exactly one explicitly selected wave, never subsequent waves."""

    validate_run_config(config)
    wave = select_wave(manifest, wave_id)
    model_config = config["model"]
    repo_root = Path(__file__).resolve().parents[2]
    code_revision = collect_code_revision(repo_root)
    runtime_environment = collect_runtime_environment()
    if manifest["tokenizer"]["model_id"] != model_config["model_id"]:
        raise ValueError("manifest tokenizer model_id does not match run config model_id")
    if manifest["tokenizer"]["revision"] != model_config["revision"]:
        raise ValueError("manifest tokenizer revision does not match run config revision")

    items = list(wave["items"])
    if only_artifact_id:
        items = [item for item in items if item["artifact_id"] == only_artifact_id]
        if not items:
            raise ValueError(f"No item {only_artifact_id!r} in wave {wave_id!r}")

    planned: list[tuple[dict[str, Any], str, dict[str, Any], Path]] = []
    results: list[dict[str, Any]] = []
    for item in items:
        artifact_id, identity = runtime_artifact_identity(
            item, config, code_revision, runtime_environment
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

    load_started = time.perf_counter()
    model, tokenizer = _load_model_and_tokenizer(config)
    model_load_seconds = time.perf_counter() - load_started
    device = model_config["device"]
    uses_cuda = device.startswith("cuda")
    gpu_info = _gpu_info(device)
    adag_config = ADAGConfig(**{**config["adag_config"], "device": device})

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wave", required=True, help="One exact wave_id; no later wave is implied")
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
