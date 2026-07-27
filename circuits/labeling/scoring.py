"""Fixed local Transluce scoring for candidates and final-label audit."""

from __future__ import annotations

import json
import os
import resource
import socket
import time
from pathlib import Path
from typing import Any, Literal

from circuits.descriptions.types import ActivationRecord
from circuits.descriptions.vllm_backend import FinetunedSimulator, score_attr_explanations
from circuits.labeling.config import LabelingRecipe
from circuits.labeling.io import atomic_write_json
from circuits.labeling.profiles import retokenize_for_simulator
from circuits.labeling.runtime import load_run_manifest, load_stage_requests
from circuits.labeling.schema import GenerationResult


def _profile_path(run_root: Path, state: str, cluster_id: int) -> Path:
    return run_root / "profiles" / state / f"cluster-{cluster_id:04d}.json"


def _load_records(
    run_root: Path,
    *,
    state: str,
    cluster_id: int,
    partition: str,
    simulator_tokenizer: Any,
) -> tuple[list[ActivationRecord], list[dict[str, Any]]]:
    value = json.loads(
        _profile_path(run_root, state, cluster_id).read_text(encoding="utf-8")
    )
    records: list[ActivationRecord] = []
    diagnostics: list[dict[str, Any]] = []
    for profile in value["partitions"][partition]:
        source = ActivationRecord.model_validate(profile["record"])
        mapped, alignment = retokenize_for_simulator(source, simulator_tokenizer)
        records.append(mapped)
        diagnostics.append(
            {
                "trace_unit_id": profile["trace_unit_id"],
                "matched_signed_basis_count": profile["matched_signed_basis_count"],
                **alignment,
            }
        )
    return records, diagnostics


def _candidate_texts(
    run_root: Path, state: str, cluster_id: int
) -> list[tuple[str, str]]:
    requests = load_stage_requests(run_root, "candidate_generation")
    values: list[tuple[str, str]] = []
    for request in requests:
        if request.state != state or request.cluster_id != cluster_id:
            continue
        path = (
            run_root
            / "results"
            / "candidate_generation"
            / f"{request.request_id}.json"
        )
        if not path.is_file():
            raise ValueError(f"candidate result is missing: {path}")
        result = GenerationResult.model_validate_json(path.read_text(encoding="utf-8"))
        if result.parse_status != "success" or result.parsed is None:
            continue
        description = result.parsed.get("description")
        if isinstance(description, str) and description.strip():
            values.append((request.request_id, description.strip()))
    return values


def _summary_texts(
    run_root: Path, state: str, cluster_id: int
) -> list[tuple[str, str]]:
    requests = load_stage_requests(run_root, "cluster_summary")
    values: list[tuple[str, str]] = []
    for request in requests:
        if request.state != state or request.cluster_id != cluster_id:
            continue
        path = run_root / "results" / "cluster_summary" / f"{request.request_id}.json"
        if not path.is_file():
            raise ValueError(f"summary result is missing: {path}")
        result = GenerationResult.model_validate_json(path.read_text(encoding="utf-8"))
        if result.parse_status != "success" or result.parsed is None:
            continue
        label = result.parsed.get("label")
        if isinstance(label, str) and label.strip():
            values.append((request.request_id, label.strip()))
    return values


def score_run(
    *,
    run_root: Path,
    phase: Literal["candidate_selection", "summary_audit"],
    states: set[str] | None = None,
    cluster_ids: set[int] | None = None,
) -> dict[str, int]:
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    selected = {
        state: [
            int(cluster_id)
            for cluster_id in values
            if cluster_ids is None or int(cluster_id) in cluster_ids
        ]
        for state, values in manifest["selected_clusters"].items()
        if states is None or state in states
    }
    partition = "selection_scoring" if phase == "candidate_selection" else "audit"
    output_name = "candidate_selection" if phase == "candidate_selection" else "summary_audit"
    started = time.monotonic()
    counts = {"planned": sum(len(values) for values in selected.values()), "completed": 0, "skipped": 0}

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(recipe.scorer.gpu_index)
    simulator = FinetunedSimulator(
        model_name=recipe.scorer.model,
        gpu_idx=recipe.scorer.gpu_index,
    )
    try:
        for state, ids in selected.items():
            for cluster_id in ids:
                output = (
                    run_root
                    / "scores"
                    / output_name
                    / state
                    / f"cluster-{cluster_id:04d}.json"
                )
                if output.exists():
                    counts["skipped"] += 1
                    continue
                texts = (
                    _candidate_texts(run_root, state, cluster_id)
                    if phase == "candidate_selection"
                    else _summary_texts(run_root, state, cluster_id)
                )
                if not texts:
                    raise ValueError(
                        f"no parsed texts to score for {state} cluster {cluster_id}"
                    )
                records, alignment = _load_records(
                    run_root,
                    state=state,
                    cluster_id=cluster_id,
                    partition=partition,
                    simulator_tokenizer=simulator.tokenizer,
                )
                explanations = [text for _, text in texts]
                scored = score_attr_explanations(
                    simulator,
                    explanations,
                    records,
                    "pos",
                    use_raw_activations=True,
                    keep_only_top_predictions=False,
                )
                values = [
                    {
                        "request_id": request_id,
                        "text": text,
                        "correlation": score.score,
                        "rsquared": score.rsquared,
                    }
                    for (request_id, text), score in zip(texts, scored, strict=True)
                ]
                values.sort(
                    key=lambda item: (
                        item["correlation"]
                        if item["correlation"] is not None
                        else float("-inf"),
                        item["request_id"],
                    ),
                    reverse=True,
                )
                atomic_write_json(
                    output,
                    {
                        "schema_version": "adag.labeling.local-scores.v1",
                        "run_id": manifest["run_id"],
                        "phase": phase,
                        "state": state,
                        "cluster_id": cluster_id,
                        "partition": partition,
                        "simulator": recipe.scorer.model_dump(mode="json"),
                        "alignment_diagnostics": alignment,
                        "scores": values,
                    },
                )
                counts["completed"] += 1
    finally:
        simulator.cleanup()

    elapsed = time.monotonic() - started
    allocated_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
    peak_hbm = (
        int(torch.cuda.max_memory_allocated(recipe.scorer.gpu_index))
        if torch.cuda.is_available()
        else None
    )
    telemetry_path = (
        run_root
        / "telemetry"
        / "local_scoring"
        / f"{phase}-{os.environ.get('SLURM_JOB_ID', 'local')}-"
        f"{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}.json"
    )
    atomic_write_json(
        telemetry_path,
        {
            "schema_version": "adag.labeling.local-scoring-telemetry.v1",
            "run_id": manifest["run_id"],
            "phase": phase,
            "backend": recipe.scorer.backend,
            "model": recipe.scorer.model,
            "elapsed_seconds": elapsed,
            "allocated_gpu_count": allocated_gpus,
            "gpu_hours": allocated_gpus * elapsed / 3600,
            "peak_hbm_bytes": peak_hbm,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "counts": counts,
        },
    )
    return counts
