"""Execute one contribution-aware same-position candidate trace wave."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from circuits.tracing.artifact import (
    save_topk_compact_trace,
    validate_topk_compact_trace_integrity,
)
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.trace import trace_teacher_forced_candidates

from scripts.bonafide.cuda_allocator_policy import (
    bind_cuda_allocator_runtime_receipt,
    declared_cuda_allocator_policy,
    validate_cuda_allocator_environment,
)
from scripts.bonafide.execution_plan import sha256_file
from scripts.bonafide.runner import (
    _append_jsonl,
    _directory_size,
    _gpu_info,
    _load_model_and_tokenizer,
    _rss_peak_bytes,
    _sha256,
    collect_code_revision,
    collect_runtime_environment,
    load_json,
    normalized_instrumentation,
    normalized_trace_warmup,
    trace_warmup_applies,
    validate_run_config,
    validate_runtime_topk_trace_against_item,
    wave_stop_reason,
)
from scripts.bonafide.topk_manifest import (
    STEP0_T5_SMOKE_PHASE,
    candidate_count_bounds,
    candidate_selection_limit,
    validate_topk_manifest,
)


def select_topk_wave(manifest: Mapping[str, Any], wave_id: str) -> dict[str, Any]:
    validate_topk_manifest(manifest)
    matches = [wave for wave in manifest["waves"] if wave.get("wave_id") == wave_id]
    if len(matches) != 1:
        available = [wave.get("wave_id") for wave in manifest["waves"]]
        raise ValueError(
            f"Top-k wave {wave_id!r} not found exactly once; available: {available}"
        )
    return dict(matches[0])


def topk_runtime_artifact_identity(
    item: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    trace_family: Mapping[str, Any],
    code_revision: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    source_manifest_sha256: str,
    topk_manifest_sha256: str,
    wave_id: str,
    teacher_forced_serialization_mode: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Bind a top-k unit to its source target and complete executable contract."""

    identity = {
        "source_width1_artifact_id": item["artifact_id"],
        "source_width1_work_item_sha256": _sha256(item),
        "source_width1_manifest_sha256": source_manifest_sha256,
        "topk_manifest_sha256": topk_manifest_sha256,
        "source_target_selection": dict(item["target_selection"]),
        "trace_family": dict(trace_family),
        "model": dict(config["model"]),
        "adag_config": dict(config["adag_config"]),
        "trace_warmup": normalized_trace_warmup(config),
        "batch_size": 1,
        "wave_id": wave_id,
        "code_revision": dict(code_revision),
        "runtime_environment": dict(runtime_environment),
    }
    if "instrumentation" in config:
        identity["instrumentation"] = normalized_instrumentation(config)
    allocator_policy = declared_cuda_allocator_policy(config)
    if allocator_policy is not None:
        identity["cuda_allocator_policy"] = allocator_policy
    if teacher_forced_serialization_mode is not None:
        identity["teacher_forced_serialization_mode"] = (
            teacher_forced_serialization_mode
        )
    digest = _sha256(identity)
    identity["sha256"] = digest
    return f"topk-trace-{digest[:24]}", identity


def _completed_artifact_matches(path: Path, identity: Mapping[str, Any]) -> bool:
    manifest = validate_topk_compact_trace_integrity(path)
    return manifest.get("artifact_identity") == identity


def _model_config_sha256(model) -> str:
    to_dict = getattr(model.config, "to_dict", None)
    value = to_dict() if callable(to_dict) else dict(vars(model.config))
    return _sha256(value)


def _candidate_profile_diagnostics(frame, candidate_count: int) -> dict[str, Any]:
    rows = [
        [float(value) for value in contribution]
        for contribution in frame.get("contrib_map", [])
        if contribution is not None
    ]
    if not rows:
        return {
            "candidate_profile_row_count": 0,
            "candidate_vector_l2_norms": [0.0] * candidate_count,
            "candidate_profile_matrix_rank": 0,
            "candidate_sign_counts": [
                {"positive": 0, "negative": 0, "zero": 0}
                for _ in range(candidate_count)
            ],
        }
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != candidate_count:
        raise ValueError("runtime contribution profile width is inconsistent")
    if not np.isfinite(matrix).all():
        raise ValueError("runtime contribution profiles contain non-finite values")
    return {
        "candidate_profile_row_count": int(matrix.shape[0]),
        "candidate_vector_l2_norms": [
            float(value) for value in np.linalg.norm(matrix, axis=0)
        ],
        "candidate_profile_matrix_rank": int(np.linalg.matrix_rank(matrix)),
        "candidate_sign_counts": [
            {
                "positive": int((matrix[:, index] > 0).sum()),
                "negative": int((matrix[:, index] < 0).sum()),
                "zero": int((matrix[:, index] == 0).sum()),
            }
            for index in range(candidate_count)
        ],
    }


def _trace_cuda_peak_bytes(
    instrumentation: Mapping[str, Any],
    *,
    cuda_memory_telemetry: bool,
    uses_cuda: bool,
) -> tuple[int, int]:
    """Read reset-safe trace peaks or the legacy process-global CUDA peaks."""

    if not cuda_memory_telemetry:
        return (
            int(torch.cuda.max_memory_allocated()) if uses_cuda else 0,
            int(torch.cuda.max_memory_reserved()) if uses_cuda else 0,
        )
    cuda_memory = instrumentation.get("cuda_memory")
    if not isinstance(cuda_memory, Mapping):
        raise ValueError("instrumentation lacks CUDA memory telemetry")
    overall = cuda_memory.get("overall")
    peak = overall.get("peak") if isinstance(overall, Mapping) else None
    if not isinstance(peak, Mapping):
        raise ValueError("instrumentation lacks overall CUDA memory peaks")
    values = []
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        value = peak.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"instrumentation CUDA peak {field} is invalid")
        values.append(value)
    return values[0], values[1]


def _stable_text_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _serialization_mode_for_example(
    example: Mapping[str, Any], explicit_mode: str | None
) -> str:
    if explicit_mode is not None:
        return explicit_mode
    value = example.get("teacher_forced_serialization_mode", "assistant_turn")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "example.teacher_forced_serialization_mode must be a non-empty string"
        )
    return value


def _teacher_forcing_contract(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    value = manifest.get("teacher_forcing_contract")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("teacher_forcing_contract must be an object")
    return value


def validate_runtime_serialization_contract(
    trace,
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    serialization_mode: str,
) -> None:
    """Bind a live trace to the frozen historical prompt/tokenization contract."""

    metadata = trace.circuit_data.trace_metadata
    example = item["example"]
    system_prompt = example.get("system_prompt")
    should_validate = (
        manifest.get("phase") == STEP0_T5_SMOKE_PHASE
        or system_prompt is not None
        or "teacher_forced_serialization_mode" in example
        or _teacher_forcing_contract(manifest) is not None
    )
    if not should_validate:
        return
    if metadata.get("system_prompt") != system_prompt:
        raise ValueError("runtime top-k system prompt disagrees with frozen example")
    if metadata.get("system_prompt_sha256") != _stable_text_sha256(system_prompt):
        raise ValueError(
            "runtime top-k system prompt hash disagrees with frozen example"
        )
    if metadata.get("teacher_forced_serialization_mode") != serialization_mode:
        raise ValueError(
            "runtime top-k teacher-forced serialization mode disagrees with "
            "frozen example"
        )

    contract = _teacher_forcing_contract(manifest)
    if contract is None:
        return
    token_identity = example.get("token_identity")
    if not isinstance(token_identity, Mapping):
        raise ValueError("frozen example lacks token_identity")
    runtime_identity = metadata.get("teacher_forced_token_identity")
    if not isinstance(runtime_identity, Mapping):
        raise ValueError("runtime top-k trace lacks teacher-forced token identity")
    expected_identity = {
        "schema_version": contract.get("token_identity_schema_version"),
        "hash_encoding": contract.get("hash_encoding"),
        "assistant_prefix_ids_sha256": token_identity.get(
            "assistant_prefix_ids_sha256"
        ),
        "response_ids_sha256": token_identity.get("response_ids_sha256"),
    }
    for field, expected in expected_identity.items():
        if runtime_identity.get(field) != expected:
            raise ValueError(
                f"runtime top-k {field} disagrees with frozen tokenization contract"
            )


def _verify_source_manifest(manifest: Mapping[str, Any], repo_root: Path) -> str:
    source = manifest["source"]
    source_path = Path(os.path.expandvars(source["width1_manifest_path"]))
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    expected = source["width1_manifest_sha256"]
    if not source_path.is_file() or sha256_file(source_path) != expected:
        raise ValueError("top-k source width-one manifest hash drift")
    with source_path.open(encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    tokenizer = source_manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("width-one source manifest lacks tokenizer provenance")
    expected_tokenizer = {
        "model_id": source["model_id"],
        "revision": source["tokenizer_revision"],
        "chat_template_sha256": source["chat_template_sha256"],
    }
    for field, expected_value in expected_tokenizer.items():
        if tokenizer.get(field) != expected_value:
            raise ValueError(
                f"top-k source.{field} disagrees with width-one source manifest"
            )
    source_items: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for source_wave in source_manifest.get("waves", []):
        for source_item in source_wave.get("items", []):
            artifact_id = source_item.get("artifact_id")
            if artifact_id in source_items:
                raise ValueError(
                    f"duplicate width-one source artifact ID: {artifact_id!r}"
                )
            source_items[artifact_id] = (source_wave, source_item)
    for topk_wave in manifest["waves"]:
        for topk_item in topk_wave["items"]:
            artifact_id = topk_item["artifact_id"]
            source_pair = source_items.get(artifact_id)
            if source_pair is None:
                raise ValueError(
                    f"top-k item is absent from width-one source: {artifact_id}"
                )
            source_wave, source_item = source_pair
            comparable = dict(topk_item)
            comparable.pop("specified_candidate_token_id", None)
            if comparable != source_item:
                raise ValueError(
                    f"top-k item drifted from width-one source: {artifact_id}"
                )
            if topk_wave.get("corpus_role") is not None and topk_wave.get(
                "corpus_role"
            ) != source_wave.get("corpus_role"):
                raise ValueError(
                    f"top-k corpus role drifted from width-one source: {artifact_id}"
                )
    return expected


def run_topk_wave(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    wave_id: str,
    artifact_root: Path,
    summary_jsonl: Path,
    only_artifact_id: str | None = None,
    dry_run: bool = False,
    verify_source: bool = True,
    _model_bundle=None,
    _code_revision: Mapping[str, Any] | None = None,
    _runtime_environment: Mapping[str, Any] | None = None,
    teacher_forced_serialization_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Run or dry-run one immutable top-k trace-family wave."""

    validate_run_config(config, allow_instrumentation=True)
    if config["adag_config"].get("center_logits", False):
        raise ValueError(
            "top-k tracing requires center_logits=false; centering belongs in "
            "the named candidate objective"
        )
    wave = select_topk_wave(manifest, wave_id)
    repo_root = Path(__file__).resolve().parents[2]
    source_manifest_sha256 = (
        _verify_source_manifest(manifest, repo_root)
        if verify_source
        else manifest["source"]["width1_manifest_sha256"]
    )
    code_revision = dict(_code_revision or collect_code_revision(repo_root))
    validate_cuda_allocator_environment(config)
    base_runtime_environment = _runtime_environment or collect_runtime_environment()
    runtime_environment = bind_cuda_allocator_runtime_receipt(
        config, base_runtime_environment
    )
    trace_family = dict(manifest["trace_family"])
    instrumentation_policy = normalized_instrumentation(config)
    if manifest["source"]["model_id"] != config["model"]["model_id"]:
        raise ValueError("top-k source model ID disagrees with run config")
    if manifest["source"]["model_revision"] != config["model"]["revision"]:
        raise ValueError("top-k source model revision disagrees with run config")
    if manifest["source"]["tokenizer_revision"] != config["model"]["revision"]:
        raise ValueError("top-k source tokenizer revision disagrees with run config")
    topk_manifest_sha256 = _sha256(manifest)
    family_root = artifact_root / trace_family["trace_family_id"]
    candidate_count_min, candidate_count_max = candidate_count_bounds(trace_family)
    selection_limit = candidate_selection_limit(trace_family)

    items = list(wave["items"])
    if only_artifact_id is not None:
        items = [item for item in items if item["artifact_id"] == only_artifact_id]
        if not items:
            raise ValueError(
                f"No source item {only_artifact_id!r} in top-k wave {wave_id!r}"
            )

    planned: list[tuple[dict[str, Any], str, dict[str, Any], Path]] = []
    results: list[dict[str, Any]] = []
    for item in items:
        artifact_id, identity = topk_runtime_artifact_identity(
            item,
            config=config,
            trace_family=trace_family,
            code_revision=code_revision,
            runtime_environment=runtime_environment,
            source_manifest_sha256=source_manifest_sha256,
            topk_manifest_sha256=topk_manifest_sha256,
            wave_id=wave_id,
            teacher_forced_serialization_mode=(teacher_forced_serialization_mode),
        )
        path = family_root / wave_id / artifact_id
        base = {
            "wave_id": wave_id,
            "trace_family_id": trace_family["trace_family_id"],
            "source_width1_artifact_id": item["artifact_id"],
            "artifact_id": artifact_id,
            "artifact_identity_sha256": identity["sha256"],
            "example_id": item["example"]["example_id"],
            "target_response_position": item["target_selection"][
                "response_token_positions"
            ][0],
            "candidate_policy_id": trace_family["candidate_policy_id"],
            "joint_objective_id": trace_family["joint_objective_id"],
            "candidate_count_min": candidate_count_min,
            "candidate_count_max": candidate_count_max,
            "code_revision": code_revision,
            "runtime_environment": runtime_environment,
        }
        if candidate_count_min == candidate_count_max:
            base["candidate_count"] = candidate_count_min
        if path.exists():
            if not _completed_artifact_matches(path, identity):
                raise FileExistsError(
                    "top-k artifact path exists but identity/completion does not "
                    f"match: {path}"
                )
            skipped = {
                **base,
                "status": "skipped_complete",
                "artifact_path": str(path),
                "artifact_bytes": _directory_size(path),
            }
            results.append(skipped)
            if not dry_run:
                _append_jsonl(summary_jsonl, skipped)
            continue
        if dry_run:
            results.append({**base, "status": "planned", "artifact_path": str(path)})
        else:
            planned.append((item, artifact_id, identity, path))
    if dry_run or not planned:
        return results

    if _model_bundle is None:
        load_started = time.perf_counter()
        model, tokenizer = _load_model_and_tokenizer(config)
        model_load_seconds = time.perf_counter() - load_started
    else:
        model, tokenizer = _model_bundle
        model_load_seconds = 0.0
    device = config["model"]["device"]
    uses_cuda = device.startswith("cuda")
    gpu_info = _gpu_info(device)
    adag_config = ADAGConfig(**{**config["adag_config"], "device": device})
    warmup_policy = normalized_trace_warmup(config)

    if trace_warmup_applies(warmup_policy, wave_id):
        warmup_item = planned[0][0]
        warmup_position = warmup_item["target_selection"]["response_token_positions"][0]
        warmup_config_before = _model_config_sha256(model)
        warmup_trace = trace_teacher_forced_candidates(
            model,
            tokenizer,
            warmup_item["example"]["prompt"],
            warmup_item["example"]["response"],
            warmup_position,
            adag_config,
            candidate_policy_id=trace_family["candidate_policy_id"],
            candidate_count=selection_limit,
            specified_candidate_token_id=warmup_item.get(
                "specified_candidate_token_id"
            ),
            joint_objective_id=trace_family["joint_objective_id"],
            trace_family_id=trace_family["trace_family_id"],
            label=warmup_item["example"]["example_id"],
            system_prompt=warmup_item["example"].get("system_prompt"),
            serialization_mode=_serialization_mode_for_example(
                warmup_item["example"], teacher_forced_serialization_mode
            ),
        )
        validate_runtime_serialization_contract(
            warmup_trace,
            warmup_item,
            manifest,
            serialization_mode=_serialization_mode_for_example(
                warmup_item["example"], teacher_forced_serialization_mode
            ),
        )
        if _model_config_sha256(model) != warmup_config_before:
            raise RuntimeError(
                "top-k warm-up leaked model configuration state; resident "
                "model reuse is unsafe"
            )
        del warmup_trace
        gc.collect()
        if uses_cuda:
            torch.cuda.empty_cache()

    limits = config.get("wave_limits", {})
    max_trace_seconds = limits.get("max_trace_seconds")
    min_cuda_headroom_bytes = int(limits.get("min_cuda_headroom_bytes", 0))
    stop_on_oom = bool(limits.get("stop_on_oom", True))
    signal_state = {"requested": False}
    previous_handler = signal.getsignal(signal.SIGUSR1)

    def request_stop(_signum, _frame) -> None:
        signal_state["requested"] = True

    signal.signal(signal.SIGUSR1, request_stop)
    try:
        for planned_index, (item, artifact_id, identity, path) in enumerate(planned):
            if signal_state["requested"]:
                stop_record = {
                    "status": "wave_stopped",
                    "wave_id": wave_id,
                    "stop_reason": "slurm_time_limit_signal",
                    "remaining_item_count": len(planned) - planned_index,
                }
                _append_jsonl(summary_jsonl, stop_record)
                results.append(stop_record)
                raise RuntimeError("top-k wave stopped after Slurm SIGUSR1")
            position = item["target_selection"]["response_token_positions"][0]
            example = item["example"]
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
                cuda_memory_telemetry=(instrumentation_policy["cuda_memory_telemetry"]),
                cuda_allocator_snapshot_telemetry=(
                    instrumentation_policy["cuda_allocator_snapshot_telemetry"]
                ),
            )
            config_before = _model_config_sha256(model)
            started = time.perf_counter()
            try:
                trace = trace_teacher_forced_candidates(
                    model,
                    tokenizer,
                    example["prompt"],
                    example["response"],
                    position,
                    adag_config,
                    candidate_policy_id=trace_family["candidate_policy_id"],
                    candidate_count=selection_limit,
                    specified_candidate_token_id=item.get(
                        "specified_candidate_token_id"
                    ),
                    joint_objective_id=trace_family["joint_objective_id"],
                    trace_family_id=trace_family["trace_family_id"],
                    label=example["example_id"],
                    system_prompt=example.get("system_prompt"),
                    serialization_mode=_serialization_mode_for_example(
                        example, teacher_forced_serialization_mode
                    ),
                    instrumentation=instrumentation,
                )
                if _model_config_sha256(model) != config_before:
                    raise RuntimeError(
                        "top-k tracing leaked model configuration state; "
                        "resident model reuse is unsafe"
                    )
                validate_runtime_topk_trace_against_item(trace, item, trace_family)
                validate_runtime_serialization_contract(
                    trace,
                    item,
                    manifest,
                    serialization_mode=_serialization_mode_for_example(
                        example, teacher_forced_serialization_mode
                    ),
                )
                if trace.circuit_data.model_id != config["model"]["model_id"]:
                    raise ValueError("runtime top-k model ID disagrees with run config")
                if (
                    trace.circuit_data.trace_metadata.get("chat_template_sha256")
                    != manifest["source"]["chat_template_sha256"]
                ):
                    raise ValueError(
                        "runtime top-k chat template disagrees with frozen source"
                    )
                if uses_cuda:
                    torch.cuda.synchronize()
                trace_elapsed = time.perf_counter() - started
                snapshot = instrumentation.snapshot()
                trace.circuit_data.trace_metadata["instrumentation"] = snapshot
                peak_allocated, peak_reserved = _trace_cuda_peak_bytes(
                    snapshot,
                    cuda_memory_telemetry=(
                        instrumentation_policy["cuda_memory_telemetry"]
                    ),
                    uses_cuda=uses_cuda,
                )
                profile_diagnostics = _candidate_profile_diagnostics(
                    trace.circuit_data.df_node, trace.candidate_count
                )
                metrics = {
                    "status": "complete",
                    "trace_wall_seconds": trace_elapsed,
                    "cuda_allocated_before_bytes": allocated_before,
                    "cuda_reserved_before_bytes": reserved_before,
                    "cuda_allocated_after_trace_bytes": (
                        torch.cuda.memory_allocated() if uses_cuda else 0
                    ),
                    "cuda_reserved_after_trace_bytes": (
                        torch.cuda.memory_reserved() if uses_cuda else 0
                    ),
                    "cuda_peak_allocated_bytes": peak_allocated,
                    "cuda_peak_reserved_bytes": peak_reserved,
                    "cuda_headroom_after_peak_bytes": (
                        gpu_info["total_memory_bytes"] - peak_reserved
                        if uses_cuda and gpu_info is not None
                        else 0
                    ),
                    "rss_peak_before_bytes": rss_before,
                    "rss_peak_after_bytes": _rss_peak_bytes(),
                    "node_count": len(trace.circuit_data.df_node),
                    "edge_count": len(trace.circuit_data.df_edge),
                    "input_token_count": len(trace.circuit_data.cis[0]),
                    "response_token_count": item["response_token_count"],
                    "target_count": 1,
                    "candidate_count": trace.candidate_count,
                    "observed_token_rank": (
                        trace.candidate_selection.observed_token_rank
                    ),
                    "instrumentation": snapshot,
                    **profile_diagnostics,
                }
                serialization_started = time.perf_counter()
                save_topk_compact_trace(
                    path,
                    trace,
                    metrics=metrics,
                    manifest={
                        "artifact_id": artifact_id,
                        "artifact_identity": identity,
                        "source_width1_artifact_id": item["artifact_id"],
                        "source_target_selection": item["target_selection"],
                        "bonafide_example": example,
                        "topk_manifest_sha256": topk_manifest_sha256,
                        "source_width1_manifest_sha256": (source_manifest_sha256),
                        "model_revision": config["model"]["revision"],
                        "code_revision": code_revision,
                        "runtime_environment": runtime_environment,
                        "gpu": gpu_info,
                    },
                )
                record = {
                    "wave_id": wave_id,
                    "trace_family_id": trace_family["trace_family_id"],
                    "source_width1_artifact_id": item["artifact_id"],
                    "artifact_id": artifact_id,
                    "artifact_identity_sha256": identity["sha256"],
                    "example_id": example["example_id"],
                    "target_response_position": position,
                    "candidate_policy_id": trace_family["candidate_policy_id"],
                    "joint_objective_id": trace_family["joint_objective_id"],
                    "model_load_seconds": model_load_seconds,
                    **metrics,
                    "serialization_wall_seconds": (
                        time.perf_counter() - serialization_started
                    ),
                    "total_unit_wall_seconds": time.perf_counter() - started,
                    "artifact_path": str(path),
                    "artifact_bytes": _directory_size(path),
                    "code_revision": code_revision,
                    "runtime_environment": runtime_environment,
                    "gpu": gpu_info,
                }
            except torch.cuda.OutOfMemoryError as error:
                record = {
                    "status": "oom",
                    "wave_id": wave_id,
                    "artifact_id": artifact_id,
                    "source_width1_artifact_id": item["artifact_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "trace_wall_seconds": time.perf_counter() - started,
                    "rss_peak_after_bytes": _rss_peak_bytes(),
                    "instrumentation": instrumentation.snapshot(),
                }
                _append_jsonl(summary_jsonl, record)
                results.append(record)
                if uses_cuda:
                    torch.cuda.empty_cache()
                if stop_on_oom:
                    raise RuntimeError(
                        f"top-k wave stopped after OOM at {artifact_id}"
                    ) from error
                raise
            except BaseException as error:
                record = {
                    "status": "error",
                    "wave_id": wave_id,
                    "artifact_id": artifact_id,
                    "source_width1_artifact_id": item["artifact_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "trace_wall_seconds": time.perf_counter() - started,
                    "rss_peak_after_bytes": _rss_peak_bytes(),
                    "instrumentation": instrumentation.snapshot(),
                }
                _append_jsonl(summary_jsonl, record)
                results.append(record)
                raise
            _append_jsonl(summary_jsonl, record)
            results.append(record)
            stop_reason = wave_stop_reason(
                record,
                uses_cuda=uses_cuda,
                max_trace_seconds=(
                    float(max_trace_seconds) if max_trace_seconds is not None else None
                ),
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
                }
                _append_jsonl(summary_jsonl, stop_record)
                results.append(stop_record)
                raise RuntimeError(
                    f"top-k wave stopped after resource gate: {stop_reason}"
                )
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--summary-jsonl", type=Path, required=True)
    parser.add_argument("--only-artifact-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-records", action="store_true")
    args = parser.parse_args()

    records = run_topk_wave(
        config=load_json(args.config),
        manifest=load_json(args.manifest),
        wave_id=args.wave,
        artifact_root=args.artifact_root,
        summary_jsonl=args.summary_jsonl,
        only_artifact_id=args.only_artifact_id,
        dry_run=args.dry_run,
    )
    if args.print_records:
        for record in records:
            print(json.dumps(record, sort_keys=True, allow_nan=False))
    else:
        counts: dict[str, int] = {}
        for record in records:
            status = str(record["status"])
            counts[status] = counts.get(status, 0) + 1
        print(
            json.dumps(
                {
                    "wave_id": args.wave,
                    "record_count": len(records),
                    "status_counts": counts,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
