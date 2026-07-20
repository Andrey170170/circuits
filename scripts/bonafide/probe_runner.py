"""Run graph-free ADAG probes for one BonaFide manifest wave.

The model is loaded once and remains resident while every selected one-target
work item is probed and atomically persisted as plain JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from circuits.tracing.clja import ADAGConfig
from circuits.tracing.instrumentation import TraceInstrumentation
from circuits.tracing.probe_artifact import (
    save_probe_artifact,
    validate_probe_artifact_integrity,
)
from circuits.tracing.trace import probe_teacher_forced_response
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
    select_wave,
    validate_run_config,
    validate_target_selection,
    validate_wave_sampling_design,
)


def probe_artifact_identity(
    item: Mapping[str, Any],
    config: Mapping[str, Any],
    code_revision: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    validate_target_selection(item)
    positions = item["target_selection"]["response_token_positions"]
    if len(positions) != 1:
        raise ValueError("probe wave items must select exactly one target")
    identity = {
        "mode": "teacher_forced_probe",
        "source_artifact_id": item["artifact_id"],
        "source_work_item_sha256": _sha256(item),
        "source_target_selection": dict(item["target_selection"]),
        "model": dict(config["model"]),
        "adag_config": dict(config["adag_config"]),
        "batch_size": 1,
        "code_revision": dict(code_revision),
        "runtime_environment": dict(runtime_environment),
    }
    digest = _sha256(identity)
    identity["sha256"] = digest
    return f"probe-{digest[:24]}", identity


def _validate_probe_against_item(
    probe: Mapping[str, Any], item: Mapping[str, Any]
) -> None:
    provenance = probe["target_provenance"]
    selection = item["target_selection"]
    expected_position = int(selection["response_token_positions"][0])
    if provenance.get("response_token_position") != expected_position:
        raise ValueError("runtime probe target position does not match manifest")
    expected_token = int(selection["final_target_token_id"])
    if provenance.get("token_id") != expected_token:
        raise ValueError("runtime probe target token ID does not match manifest")
    expected_count = int(item["response_token_count"])
    if probe["trace_metadata"].get("response_token_count") != expected_count:
        raise ValueError("runtime probe response token count does not match manifest")


def run_probe_wave(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    wave_id: str,
    artifact_root: Path,
    summary_jsonl: Path,
    only_artifact_id: str | None = None,
    dry_run: bool = False,
    progress_every: int = 0,
) -> list[dict[str, Any]]:
    if isinstance(progress_every, bool) or progress_every < 0:
        raise ValueError("progress_every must be a non-negative integer")
    validate_run_config(config)
    wave = select_wave(manifest, wave_id)
    validate_wave_sampling_design(wave, manifest)
    items = list(wave["items"])
    if only_artifact_id is not None:
        items = [item for item in items if item["artifact_id"] == only_artifact_id]
        if not items:
            raise ValueError(f"No item {only_artifact_id!r} in wave {wave_id!r}")
    if manifest["tokenizer"]["model_id"] != config["model"]["model_id"]:
        raise ValueError(
            "manifest tokenizer model_id does not match run config model_id"
        )
    if manifest["tokenizer"]["revision"] != config["model"]["revision"]:
        raise ValueError(
            "manifest tokenizer revision does not match run config revision"
        )

    repo_root = Path(__file__).resolve().parents[2]
    code_revision = collect_code_revision(repo_root)
    runtime_environment = collect_runtime_environment()
    planned: list[tuple[Mapping[str, Any], str, dict[str, Any], Path]] = []
    results: list[dict[str, Any]] = []
    progress_status_counts: Counter[str] = Counter()

    def record_progress(record: Mapping[str, Any]) -> None:
        progress_status_counts[str(record["status"])] += 1
        processed = len(results)
        if progress_every and (
            processed % progress_every == 0 or processed == len(items)
        ):
            print(
                json.dumps(
                    {
                        "event": "probe_wave_progress",
                        "wave_id": wave_id,
                        "processed_items": processed,
                        "total_items": len(items),
                        "status_counts": dict(sorted(progress_status_counts.items())),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )

    for item in items:
        artifact_id, identity = probe_artifact_identity(
            item, config, code_revision, runtime_environment
        )
        artifact_path = artifact_root / wave_id / artifact_id
        base = {
            "mode": "teacher_forced_probe",
            "wave_id": wave_id,
            "source_artifact_id": item["artifact_id"],
            "artifact_id": artifact_id,
            "artifact_identity_sha256": identity["sha256"],
            "example_id": item["example"]["example_id"],
            "target_response_positions": item["target_selection"][
                "response_token_positions"
            ],
        }
        if artifact_path.exists():
            stored = validate_probe_artifact_integrity(artifact_path)
            if stored.get("artifact_identity") != identity:
                raise FileExistsError(
                    f"probe artifact identity mismatch at {artifact_path}"
                )
            record = {
                **base,
                "status": "skipped_complete",
                "artifact_path": str(artifact_path),
                "artifact_bytes": _directory_size(artifact_path),
            }
            results.append(record)
            record_progress(record)
            if not dry_run:
                _append_jsonl(summary_jsonl, record)
            continue
        planned.append((item, artifact_id, identity, artifact_path))
        if dry_run:
            record = {
                **base,
                "status": "planned",
                "artifact_path": str(artifact_path),
            }
            results.append(record)
            record_progress(record)
    if dry_run or not planned:
        return results

    load_started = time.perf_counter()
    model, tokenizer = _load_model_and_tokenizer(config)
    model_load_seconds = time.perf_counter() - load_started
    device = config["model"]["device"]
    uses_cuda = str(device).startswith("cuda")
    gpu_info = _gpu_info(device)
    adag_config = ADAGConfig(**{**config["adag_config"], "device": device})
    for item, artifact_id, identity, artifact_path in planned:
        runtime_base = {
            "mode": "teacher_forced_probe",
            "wave_id": wave_id,
            "source_artifact_id": item["artifact_id"],
            "artifact_id": artifact_id,
            "artifact_identity_sha256": identity["sha256"],
            "artifact_identity": identity,
            "example_id": item["example"]["example_id"],
            "annotation_row_ids": item["example"].get("annotation_row_ids", []),
            "label_types": item["example"].get("label_types", []),
            "source_target_selection": item["target_selection"],
            "target_response_positions": item["target_selection"][
                "response_token_positions"
            ],
            "model_revision": config["model"]["revision"],
            "model_load_seconds": model_load_seconds,
            "code_revision": code_revision,
            "runtime_environment": runtime_environment,
            "gpu": gpu_info,
        }
        if uses_cuda:
            torch.cuda.synchronize()
            # Probes share one resident model, but allocator cache from an earlier
            # target must not inflate this target's peak-reserved measurement or
            # accumulate across a long candidate wave.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        recorder = TraceInstrumentation(device=device, synchronize_cuda=uses_cuda)
        started = time.perf_counter()
        rss_before = _rss_peak_bytes()
        try:
            example = item["example"]
            probe = probe_teacher_forced_response(
                model=model,
                tokenizer=tokenizer,
                prompt=example["prompt"],
                response=example["response"],
                target_response_positions=item["target_selection"][
                    "response_token_positions"
                ],
                config=adag_config,
                instrumentation=recorder,
                model_revision=config["model"]["revision"],
            )
            if uses_cuda:
                torch.cuda.synchronize()
            probe_dict = probe.to_dict()
            _validate_probe_against_item(probe_dict, item)
            probe_seconds = time.perf_counter() - started
            metrics = {
                "status": "complete",
                "probe_wall_seconds": probe_seconds,
                "input_token_count": probe.trace_metadata["input_token_count"],
                "response_token_count": probe.trace_metadata["response_token_count"],
                "selected_occurrence_count": len(probe.selected_occurrences),
                "rss_peak_before_bytes": rss_before,
                "rss_peak_after_bytes": _rss_peak_bytes(),
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if uses_cuda else 0
                ),
                "cuda_peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved() if uses_cuda else 0
                ),
                "instrumentation": probe.instrumentation,
            }
            serialization_started = time.perf_counter()
            save_probe_artifact(
                artifact_path,
                probe,
                metrics=metrics,
                manifest={
                    "artifact_id": artifact_id,
                    "artifact_identity": identity,
                    "benchmark_wave_id": wave_id,
                    "source_artifact_id": item["artifact_id"],
                    "source_target_selection": item["target_selection"],
                    "bonafide_example": {
                        key: value
                        for key, value in example.items()
                        if key not in {"prompt", "response"}
                    },
                    "model_revision": config["model"]["revision"],
                    "code_revision": code_revision,
                    "runtime_environment": runtime_environment,
                    "gpu": gpu_info,
                },
            )
            record = {
                **runtime_base,
                **metrics,
                "serialization_wall_seconds": time.perf_counter()
                - serialization_started,
                "total_unit_wall_seconds": time.perf_counter() - started,
                "artifact_path": str(artifact_path),
                "artifact_bytes": _directory_size(artifact_path),
            }
        except Exception as error:
            if isinstance(error, torch.cuda.OutOfMemoryError) and uses_cuda:
                torch.cuda.empty_cache()
            instrumentation = recorder.snapshot()
            resident_model_reuse_forbidden = (
                instrumentation.get("counters", {}).get(
                    "probe_model_config_leak_during_failed_clja"
                )
                is True
            )
            record = {
                **runtime_base,
                "status": (
                    "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error"
                ),
                "probe_wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "instrumentation": instrumentation,
                "resident_model_reuse_forbidden": resident_model_reuse_forbidden,
            }
            _append_jsonl(summary_jsonl, record)
            results.append(record)
            record_progress(record)
            # A failed probe may leave the shared resident model mutated. The
            # public probe records that condition without masking its active
            # exception; fail closed here even when ordinary per-item failures
            # are configured to continue.
            if resident_model_reuse_forbidden:
                raise
            if not config.get("continue_on_error", False):
                raise
            continue
        _append_jsonl(summary_jsonl, record)
        results.append(record)
        record_progress(record)
    return results


def summarize_probe_wave(
    results: list[Mapping[str, Any]],
    *,
    wave_id: str,
    artifact_root: Path,
    summary_jsonl: Path,
) -> dict[str, Any]:
    """Return the compact, scheduler-log-friendly CLI result."""

    status_counts = Counter(str(record.get("status", "unknown")) for record in results)
    return {
        "mode": "teacher_forced_probe",
        "wave_id": wave_id,
        "item_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "artifact_root": str(artifact_root),
        "wave_artifact_root": str(artifact_root / wave_id),
        "summary_jsonl": str(summary_jsonl),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--summary-jsonl", type=Path)
    parser.add_argument("--only-artifact-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        metavar="N",
        help="emit one compact stderr progress record every N items (default: off)",
    )
    parser.add_argument(
        "--print-records",
        action="store_true",
        help="print full per-target records instead of the compact aggregate summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    artifact_root = args.artifact_root or Path(
        config.get("probe_artifact_root", "results/bonafide/probes")
    )
    summary = args.summary_jsonl or artifact_root / "probe-summary.jsonl"
    results = run_probe_wave(
        config=config,
        manifest=manifest,
        wave_id=args.wave,
        artifact_root=artifact_root,
        summary_jsonl=summary,
        only_artifact_id=args.only_artifact_id,
        dry_run=args.dry_run,
        progress_every=args.progress_every,
    )
    output: Any = (
        results
        if args.print_records
        else summarize_probe_wave(
            results,
            wave_id=args.wave,
            artifact_root=artifact_root,
            summary_jsonl=summary,
        )
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
